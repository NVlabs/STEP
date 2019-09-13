
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import numpy as np
from collections import OrderedDict
import time
from datetime import datetime
#from tensorboardX import SummaryWriter
import glob

from config import parse_config
from models import BaseNet, ROINet, TwoBranchNet, ContextNet
from external.maskrcnn_benchmark.roi_layers import nms
from utils.utils import inference, train_select, AverageMeter, get_gpu_memory
from utils.tube_utils import flatten_tubes, valid_tubes, compute_box_iou
from utils.vis_utils import overlay_image
from data.customize import CustomizedDataset, detection_collate, WIDTH, HEIGHT
from data.augmentations import BaseTransform



def main():

    ################## Customize your configuratons here ###################

    checkpoint_path = 'pretrained/ava_step.pth'
    if os.path.isfile(checkpoint_path):
        print ("Loading pretrain model from %s" % checkpoint_path)
        map_location = 'cuda:0'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        args = checkpoint['cfg']
    else:
        raise ValueError("Pretrain model not found!", checkpoint_path)

    # TODO: Set data_root to the customized input dataset
    args.data_root = '/datasets/demo/frames/'
    args.save_root = os.path.join(os.path.dirname(args.data_root), 'results/')
    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    # TODO: modify this setting according to the actual frame rate and file name
    source_fps = 30
    im_format = 'frame%04d.jpg'
    conf_thresh = 0.4
    global_thresh = 0.8    # used for cross-class NMS
    
    ################ Define models #################

    gpu_count = torch.cuda.device_count()
    nets = OrderedDict()
    # backbone network
    nets['base_net'] = BaseNet(args)
    # ROI pooling
    nets['roi_net'] = ROINet(args.pool_mode, args.pool_size)

    # detection network
    for i in range(args.max_iter):
        if args.det_net == "two_branch":
            nets['det_net%d' % i] = TwoBranchNet(args)
        else:
            raise NotImplementedError
    if not args.no_context:
        # context branch
        nets['context_net'] = ContextNet(args)

    for key in nets:
        nets[key] = nets[key].cuda()

    nets['base_net'] = torch.nn.DataParallel(nets['base_net'])
    if not args.no_context:
        nets['context_net'] = torch.nn.DataParallel(nets['context_net'])
    for i in range(args.max_iter):
        nets['det_net%d' % i].to('cuda:%d' % ((i+1)%gpu_count))
        nets['det_net%d' % i].set_device('cuda:%d' % ((i+1)%gpu_count))

    # load pretrained model 
    nets['base_net'].load_state_dict(checkpoint['base_net'])
    if not args.no_context and 'context_net' in checkpoint:
        nets['context_net'].load_state_dict(checkpoint['context_net'])
    for i in range(args.max_iter):
        pretrained_dict = checkpoint['det_net%d' % i]
        nets['det_net%d' % i].load_state_dict(pretrained_dict)

    
    ################ DataLoader setup #################

    dataset = CustomizedDataset(args.data_root, args.T, args.NUM_CHUNKS[args.max_iter], source_fps, args.fps, BaseTransform(args.image_size, args.means, args.stds,args.scale_norm), anchor_mode=args.anchor_mode, im_format=im_format)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)

    ################ Inference #################

    for _, net in nets.items():
        net.eval()

    fout = open(os.path.join(args.save_root, 'results.txt'), 'w')
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _, (images, tubes, infos) in enumerate(dataloader):

            _, _, channels, height, width = images.size()
            images = images.cuda()

            # get conv features
            conv_feat = nets['base_net'](images)
            context_feat = None
            if not args.no_context:
                context_feat = nets['context_net'](conv_feat)

            history, _ = inference(args, conv_feat, context_feat, nets, args.max_iter, tubes)

            # collect result of the last step
            pred_prob = history[-1]['pred_prob'].cpu()
            pred_prob = pred_prob[:,int(pred_prob.shape[1]/2)]
            pred_tubes = history[-1]['pred_loc'].cpu()
            pred_tubes = pred_tubes[:,int(pred_tubes.shape[1]/2)]
            tubes_nums = history[-1]['tubes_nums']

            # loop for each batch
            tubes_count = 0
            for b in range(len(tubes_nums)):
                info = infos[b]
                seq_start = tubes_count
                tubes_count = tubes_count + tubes_nums[b]

                cur_pred_prob = pred_prob[seq_start:seq_start+tubes_nums[b]]
                cur_pred_tubes = pred_tubes[seq_start:seq_start+tubes_nums[b]]

                # do NMS first
                all_scores = []
                all_boxes = []
                all_idx = []
                for cl_ind in range(args.num_classes):
                    scores = cur_pred_prob[:, cl_ind].squeeze()
                    c_mask = scores.gt(conf_thresh) # greater than a threshold
                    scores = scores[c_mask]
                    idx = np.where(c_mask.numpy())[0]
                    if len(scores) == 0:
                        all_scores.append([])
                        all_boxes.append([])
                        continue
                    boxes = cur_pred_tubes.clone()
                    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    boxes = boxes[l_mask].view(-1, 4)

                    boxes = valid_tubes(boxes.view(-1,1,4)).view(-1,4)
                    keep = nms(boxes, scores, args.nms_thresh)
                    boxes = boxes[keep].numpy()
                    scores = scores[keep].numpy()
                    idx = idx[keep]

                    boxes[:, ::2] /= width
                    boxes[:, 1::2] /= height
                    all_scores.append(scores)
                    all_boxes.append(boxes)
                    all_idx.append(idx)

                # get the top scores
                scores_list = [(s,cl_ind,j) for cl_ind,scores in enumerate(all_scores) for j,s in enumerate(scores)]
                if args.evaluate_topk > 0:
                    scores_list.sort(key=lambda x: x[0])
                    scores_list = scores_list[::-1]
                    scores_list = scores_list[:args.topk]
                
                # merge high overlapping boxes (a simple greedy method)
                merged_result = {}
                flag = [1 for _ in range(len(scores_list))]
                for i in range(len(scores_list)):
                    if flag[i]:
                        s, cl_ind, j = scores_list[i]
                        box = all_boxes[cl_ind][j]
                        temp = ([box], [args.label_dict[cl_ind]], [s])

                        # find all high IoU boxes
                        for ii in range(i+1, len(scores_list)):
                            if flag[ii]:
                                s2, cl_ind2, j2 = scores_list[ii]
                                box2 = all_boxes[cl_ind2][j2]
                                if compute_box_iou(box, box2) > global_thresh:
                                    flag[ii] = 0
                                    temp[0].append(box2)
                                    temp[1].append(args.label_dict[cl_ind2])
                                    temp[2].append(s2)
                        
                        merged_box = np.mean(np.concatenate(temp[0], axis=0).reshape(-1,4), axis=0)
                        key = ','.join(merged_box.astype(str).tolist())
                        merged_result[key] = [(l, s) for l,s in zip(temp[1], temp[2])]

                # visualize results
                if not os.path.isdir(os.path.join(args.save_root, info['video_name'])):
                    os.makedirs(os.path.join(args.save_root, info['video_name']))
                print (info)
                overlay_image(os.path.join(args.data_root, info['video_name'], im_format % info['fid']),
                              os.path.join(args.save_root, info['video_name'], im_format % info['fid']),
                              pred_boxes = merged_result,
                              id2class = args.id2class)

                # write to files
                for key in merged_result:
                    box = np.asarray(key.split(','), dtype=np.float32)
                    for l, s in merged_result[key]:
                        fout.write('{0},{1:04},{2:.4},{3:.4},{4:.4},{5:.4},{6},{7:.4}\n'.format(
                                                info['video_name'],
                                                info['fid'],
                                                box[0],box[1],box[2],box[3],
                                                l, s))
            torch.cuda.synchronize()
            t1 = time.time()
            print ("Batch time: ", t1-t0)

            torch.cuda.synchronize()
            t0 = time.time()
                    
    fout.close()

if __name__ == "__main__":
    main()
