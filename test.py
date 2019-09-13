
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
from utils.utils import inference
from utils.tube_utils import valid_tubes, compute_box_iou
from data.ava import AVADataset, detection_collate, WIDTH, HEIGHT
from data.augmentations import BaseTransform
from utils.eval_utils import ava_evaluation
from external.ActivityNet.Evaluation.get_ava_performance import read_labelmap


def main():

    ################## Load pretrained model and configurations ###################

    checkpoint_path = 'pretrained/ava_step.pth'
    if os.path.isfile(checkpoint_path):
        print ("Loading pretrain model from %s" % checkpoint_path)
        map_location = 'cuda:0'
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        args = checkpoint['cfg']
    else:
        raise ValueError("Pretrain model not found!", checkpoint_path)

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)
    
    label_dict = {}
    if args.num_classes == 60:
        label_map = os.path.join(args.data_root, 'label/ava_action_list_v2.1_for_activitynet_2018.pbtxt')
        categories, class_whitelist = read_labelmap(open(label_map, 'r'))
        classes = [(val['id'], val['name']) for val in categories]
        id2class = {c[0]: c[1] for c in classes}    # gt class id (1~80) --> class name
        for i, c in enumerate(sorted(list(class_whitelist))):
            label_dict[i] = c
    else:
        for i in range(80):
            label_dict[i] = i+1

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

    # load pretrained weights
    nets['base_net'].load_state_dict(checkpoint['base_net'])
    if not args.no_context and 'context_net' in checkpoint:
        nets['context_net'].load_state_dict(checkpoint['context_net'])
    for i in range(args.max_iter):
        pretrained_dict = checkpoint['det_net%d' % i]
        nets['det_net%d' % i].load_state_dict(pretrained_dict)

    
    ################ DataLoader setup #################

    dataset = AVADataset(args.data_root, 'test', args.input_type, args.T, args.NUM_CHUNKS[args.max_iter], args.fps, BaseTransform(args.image_size, args.means, args.stds,args.scale_norm), proposal_path=args.proposal_path_val, stride=1, anchor_mode=args.anchor_mode, num_classes=args.num_classes, foreground_only=False)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)

    ################ Inference #################

    for _, net in nets.items():
        net.eval()

    # write results to files for evaluation
    output_files = []
    fouts = []
    for i in range(args.max_iter):
        output_file = args.save_root+'testing_result-iter'+str(i+1)+'.csv'
        output_files.append(output_file)
        f = open(output_file, 'w')
        fouts.append(f)

    gt_file = args.save_root+'testing_gt.csv'
    fout = open(gt_file, 'w')

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():    # for evaluation
        for num, (images, targets, tubes, infos) in enumerate(dataloader):

            if (num+1) % 100 == 0:
                print ("%d / %d" % (num+1, len(dataloader.dataset)/args.batch_size))

            for b in range(len(infos)):
                for n in range(len(infos[b]['boxes'])):
                    mid = int(len(infos[b]['boxes'][n])/2)
                    box = infos[b]['boxes'][n][mid]
                    labels = infos[b]['labels'][n][mid]
                    for label in labels:
                        fout.write('{0},{1:04},{2:.4},{3:.4},{4:.4},{5:.4},{6}\n'.format(
                                    infos[b]['video_name'],
                                    infos[b]['fid'],
                                    box[0], box[1], box[2], box[3],
                                    label))

            _, _, channels, height, width = images.size()
            images = images.cuda()

            # get conv features
            conv_feat = nets['base_net'](images)
            context_feat = None
            if not args.no_context:
                context_feat = nets['context_net'](conv_feat)

            ############## Inference ##############

            history, _ = inference(args, conv_feat, context_feat, nets, args.max_iter, tubes)

            #################### Evaluation #################

            # loop for each  iteration
            for i in range(len(history)):
                pred_prob = history[i]['pred_prob'].cpu()
                pred_prob = pred_prob[:,int(pred_prob.shape[1]/2)]
                pred_tubes = history[i]['pred_loc'].cpu()
                pred_tubes = pred_tubes[:,int(pred_tubes.shape[1]/2)]
                tubes_nums = history[i]['tubes_nums']

                # loop for each sample in a batch
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
                        scores = cur_pred_prob[:, cl_ind].squeeze().reshape(-1)
                        c_mask = scores.gt(args.conf_thresh) # greater than minmum threshold
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

                    for s,cl_ind,j in scores_list:
                        # write to files
                        box = all_boxes[cl_ind][j]
                        fouts[i].write('{0},{1:04},{2:.4},{3:.4},{4:.4},{5:.4},{6},{7:.4}\n'.format(
                                                    info['video_name'],
                                                    info['fid'],
                                                    box[0],box[1],box[2],box[3],
                                                    label_dict[cl_ind],
                                                    s))
    fout.close()

    all_metrics = []
    for i in range(args.max_iter):
        fouts[i].close()

        metrics = ava_evaluation(os.path.join(args.data_root, 'label/'), output_files[i], gt_file)
        all_metrics.append(metrics)

    # Logging
    log_name = args.save_root+"testing_results.log"
    log_file = open(log_name, "w", 1)
    prt_str = ''
    for i in range(args.max_iter):
        prt_str += 'Iter '+str(i+1)+': MEANAP =>'+str(all_metrics[i]['PascalBoxes_Precision/mAP@0.5IOU'])+'\n'
    log_file.write(prt_str)
    
    for i in class_whitelist:
        log_file.write("({}) {}: {}\n".format(i,id2class[i], 
            all_metrics[-1]["PascalBoxes_PerformanceByCategory/AP@0.5IOU/{}".format(id2class[i])]))

    log_file.close()

if __name__ == "__main__":
    main()
