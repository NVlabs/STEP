"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import pickle
import subprocess
import random
import torch
import torch.distributed as dist

from utils.tube_utils import extrapolate_tubes, valid_tubes, decode_coef, flatten_tubes, compute_tube_iou

def inference(args, conv_feat, context_feat, nets, exec_iter, tubes):
    """
    Inference on two-branch networks of different steps.
    In training, it is used to collect all candidate tubes.
    In testing, it is used to get detection results of each step

    Arguments:
        conv_feat: conv features from the backbone network
        context_feat: context features from the context network (None if the context network is not used)
        nets: a list of two-branch networks
        exec_iter: the number of iterations to execute
        tubes: initial proposal tubes

    return:
        history: collecting output results for each iteration
        trajectory: collecting input for each iteration
    """

    # flatten list of tubes
    flat_tubes, tubes_nums = flatten_tubes(tubes, batch_idx=True)    # add batch_idx for ROI pooling
    flat_tubes = torch.FloatTensor(flat_tubes).to(conv_feat)

    history = []
    trajectory = []
    for i in range(1, exec_iter+1):    # index from 1
        # adaptively get the start chunk
        chunks = args.NUM_CHUNKS[i]
        T_start = int((args.NUM_CHUNKS[args.max_iter] - chunks) / 2) * args.T
        T_length = chunks * args.T
        chunk_idx = [j*args.T + int(args.T/2) for j in range(chunks)]    # used to index the middel frame of each chunk
        half_T = int(args.T/2)
    
        # ROI Pooling
        pooled_feat = nets['roi_net'](conv_feat[:, T_start:T_start+T_length].contiguous(), flat_tubes)
        _,C,W,H = pooled_feat.size()
        pooled_feat = pooled_feat.view(-1, T_length,C,W,H)
                        
        # detection head
        temp_context_feat = None
        if not args.no_context:
            temp_context_feat = torch.zeros((pooled_feat.size(0),context_feat.size(1),T_length,1,1)).to(context_feat)
            for p in range(pooled_feat.size(0)):
                temp_context_feat[p] = context_feat[int(flat_tubes[p,0,0].item()/T_length),:,T_start:T_start+T_length].contiguous().clone()

        global_prob, local_loc, first_loc, last_loc, _,_,_ = nets['det_net%d' % (i-1)](pooled_feat, context_feat=temp_context_feat, tubes=None, targets=None)


        ########## prepare data for next iteration ###########
    
        pred_prob = global_prob.view(-1,1,args.num_classes).expand(-1,T_length,-1)
    
        # decode regression results to output tubes
        flat_tubes = flat_tubes.to(local_loc)
        pred_loc = decode_coef(flat_tubes.view(-1,5)[:, 1:],
                                     local_loc.view(-1, 4))
        pred_loc = pred_loc.view(local_loc.size())

        if args.temporal_mode == "predict":
            pred_first_loc = decode_coef(flat_tubes[:, chunk_idx[0]-half_T:chunk_idx[0]+half_T+1].contiguous().view(-1,5)[:, 1:],
                                     first_loc.view(-1, 4))
            pred_first_loc = pred_first_loc.view(first_loc.size())    # [N*T, 4*C] --> [N, T, 4*C]

            pred_last_loc = decode_coef(flat_tubes[:, chunk_idx[-1]-half_T:chunk_idx[-1]+half_T+1].contiguous().view(-1,5)[:, 1:],
                                     last_loc.view(-1, 4))
            pred_last_loc = pred_last_loc.view(last_loc.size())    # [N*T, 4*C] --> [N, T, 4*C]
    
        history.append({'pred_prob': pred_prob.data, 
                        'pred_loc': pred_loc.data, 
                        'pred_first_loc': pred_first_loc.data if args.temporal_mode=="predict" else None, 
                        'pred_last_loc': pred_last_loc.data if args.temporal_mode=="predict" else None, 
                        'tubes_nums': tubes_nums})

        # loop for each batch
        cur_trajectory = []
        selected_tubes = []
        tubes_count = 0
        for b in range(len(tubes_nums)):
            seq_start = tubes_count
            tubes_count = tubes_count + tubes_nums[b]

            cur_pred_prob = pred_prob[seq_start:seq_start+tubes_nums[b]]
            cur_pred_tubes = pred_loc[seq_start:seq_start+tubes_nums[b]]
            cur_pred_class = torch.argmax(cur_pred_prob, dim=-1)

            # check whether extending tubes is needed
            if i < args.max_iter and args.NUM_CHUNKS[i+1] == args.NUM_CHUNKS[i]+2:
                # check which method to extend tubes
                if args.temporal_mode == "predict":
                    cur_first_tubes = pred_first_loc[seq_start:seq_start+tubes_nums[b]]
                    cur_last_tubes = pred_last_loc[seq_start:seq_start+tubes_nums[b]]

                    cur_proposals = torch.cat([cur_first_tubes, cur_pred_tubes, cur_last_tubes], dim=1)    # concatenate along time axis
                    cur_proposals = cur_proposals.cpu().numpy()

                elif args.temporal_mode == "extrapolate":
                    # expand tubes along temporal axis with extrapolation
                    cur_proposals = cur_pred_tubes.cpu().numpy()
                    cur_proposals = extrapolate_tubes(cur_proposals, args.T)

                else:    # mean tubes
                    cur_proposals = cur_pred_tubes.cpu().numpy()
                    mean_tubes = np.mean(cur_proposals, axis=1, keepdims=True)
                    mean_tubes = np.tile(mean_tubes, (1,args.T,1))
                    cur_proposals = np.concatenate((mean_tubes, cur_proposals, mean_tubes), axis=1)
            else:
                cur_proposals = cur_pred_tubes.cpu().numpy()
            cur_proposals = valid_tubes(cur_proposals, width=args.image_size[0], height=args.image_size[1])
            cur_trajectory.append((cur_proposals, cur_pred_class))

            selected_tubes.append(cur_proposals)
        trajectory.append(cur_trajectory)

        # flatten list of tubes
        flat_tubes, tubes_nums = flatten_tubes(selected_tubes, batch_idx=True)    # add batch_idx for ROI pooling
        flat_tubes = torch.FloatTensor(flat_tubes).to(conv_feat)

    return history, trajectory



def train_select(step, history, targets, tubes, args):
    """
    Select candidate samples for model training
    Arguments:
        step: int, the current step
        history: dict, inference output
        targets: list, ground truths
        tubes: np.array, initial proposals
        args: configs
    """

    # adaptively get the start chunk
    chunks = args.NUM_CHUNKS[step]
    max_chunks = args.NUM_CHUNKS[args.max_iter]
    T_start = int((args.NUM_CHUNKS[args.max_iter] - chunks) / 2) * args.T
    T_length = chunks * args.T

    cls_thresh = args.cls_thresh[step-1]
    reg_thresh = args.reg_thresh[step-1]

    ######### Collect candidates for training ########

    candidates = []
    if step > 1:    # for step > 1
        pred_prob = history['pred_prob'].cpu()
        pred_tubes = history['pred_loc'].cpu()
        tubes_nums = history['tubes_nums']
        tubes_count = 0
        
        if args.temporal_mode == "predict":
            pred_first_loc = history['pred_first_loc'].cpu()
            pred_last_loc = history['pred_last_loc'].cpu()

    for b in range(len(targets)):
        if step == 1:    # for 1st step
            candidates.append((tubes[b], None))

        else:    # for step > 1
            seq_start = tubes_count
            tubes_count = tubes_count + tubes_nums[b]
            cur_pred_prob = pred_prob[seq_start:seq_start+tubes_nums[b]]
            # get averaged score for each tube
            cur_pred_prob = torch.mean(cur_pred_prob, dim=1)
            cur_pred_tubes = pred_tubes[seq_start:seq_start+tubes_nums[b]]

            # select top-scoring boxes from each class
            all_scores = []
            all_idx = []
            for cl_ind in range(args.num_classes):
                scores = cur_pred_prob[:, cl_ind].squeeze()

                # sort according to the scores
                scores = scores.numpy().reshape(-1)
                ids = np.argsort(scores)[::-1]
                scores = scores[ids]
                idx = ids

                if args.topk > 0:
                    scores = scores[:int(args.topk/args.num_classes)*2]
                    idx = idx[:int(args.topk/args.num_classes)*2]
                all_scores.append(scores)
                all_idx.append(idx)

            # get the top scores
            scores_list = [(s,cl_ind,j) for cl_ind,scores in enumerate(all_scores) for j,s in enumerate(scores)]
            scores_list.sort(key=lambda x: x[0])
            scores_list = scores_list[::-1]
            temp_list = []
            temp = set()
            for s,cl_ind,j in scores_list:
                if not all_idx[cl_ind][j] in temp:
                    temp.add(all_idx[cl_ind][j])
                    temp_list.append((s,cl_ind,j))
                
            if args.topk > 0:
                scores_list = temp_list[:args.topk]
            else:
                scores_list = temp_list

            cur_tubes = []
            cur_scores = []
            for s,cl_ind,j in scores_list:
                cur_tubes.append(cur_pred_tubes[all_idx[cl_ind][j], :].numpy())
                cur_scores.append(s)
            try:
                cur_tubes = np.stack(cur_tubes, axis=0)
            except:
                pdb.set_trace()
            cur_tubes = valid_tubes(cur_tubes, args.image_size[0], args.image_size[1])
            cur_scores = np.asarray(cur_scores)

            
            if args.temporal_mode == "predict":
                cur_pred_first_loc = pred_first_loc[seq_start:seq_start+tubes_nums[b]]
                cur_pred_last_loc = pred_last_loc[seq_start:seq_start+tubes_nums[b]]

                cur_first_tubes = []
                cur_last_tubes = []
                for s,cl_ind,j in scores_list:
                    cur_first_tubes.append(cur_pred_first_loc[all_idx[cl_ind][j], :])
                    cur_last_tubes.append(cur_pred_last_loc[all_idx[cl_ind][j], :])
                cur_first_tubes = np.stack(cur_first_tubes, axis=0)
                cur_first_tubes = valid_tubes(cur_first_tubes, args.image_size[0], args.image_size[1])
                cur_last_tubes = np.stack(cur_last_tubes, axis=0)
                cur_last_tubes = valid_tubes(cur_last_tubes, args.image_size[0], args.image_size[1])
            else:
                cur_first_tubes, cur_last_tubes = None, None

            candidates.append((cur_tubes, cur_scores, cur_first_tubes, cur_last_tubes))


    ######### Select training samples ########

    selected_tubes = []
    target_tubes = []
    for b in range(len(targets)):
        cur_tubes = candidates[b][0]
        cur_scores = candidates[b][1]
        selected_pos, selected_neg, ious = select_proposals(
                targets[b][:,int(max_chunks/2)].reshape(targets[b].shape[0],1,-1), 
                cur_tubes[:,int(cur_tubes.shape[1]/2)].reshape(cur_tubes.shape[0],1,-1),
                cur_scores,
                cls_thresh, args.max_pos_num, args.selection_sampling, args.neg_ratio)

        cur_selected_tubes = np.zeros((len(selected_pos)+len(selected_neg), cur_tubes.shape[1], 4), dtype=np.float32)
        cur_target_tubes = np.zeros((len(selected_pos)+len(selected_neg), 1, 6+args.num_classes), dtype=np.float32)    # only one frame for loss
        row = 0
        for ii,jj in selected_pos:
            cur_selected_tubes[row] = cur_tubes[jj]
            cur_target_tubes[row,:,:4] = targets[b][ii,int(max_chunks/2),:4]
            cur_target_tubes[row,:,6:] = targets[b][ii,int(max_chunks/2),4:]
            cur_target_tubes[row,:,5] = 1    # flag for regression
            cur_target_tubes[row,:,4] = 1    # flag for classification
            row += 1

        for ii,jj in selected_neg:
            cur_selected_tubes[row] = cur_tubes[jj]
            # for regreesion only samples
            if ious[ii, jj] >= reg_thresh:
                cur_target_tubes[row,:,:4] = targets[b][ii,int(max_chunks/2),:4]
                cur_target_tubes[row,:,6:] = targets[b][ii,int(max_chunks/2),4:]
                cur_target_tubes[row,:,5] = 1    # for regression
            # FIXME: cur_target_tubes[row,:,4] = 1     # flag for classification
            row += 1


        ###### check whether extend tube is needed ######

        if step-1 in args.NUM_CHUNKS and args.NUM_CHUNKS[step] == args.NUM_CHUNKS[step-1]+2:

            if args.temporal_mode == "predict":
                cur_first_tubes = candidates[b][2]
                cur_last_tubes = candidates[b][3]

                cur_selected_first = np.zeros((len(selected_pos)+len(selected_neg), args.T, 4), dtype=np.float32)
                cur_selected_last = np.zeros((len(selected_pos)+len(selected_neg), args.T, 4), dtype=np.float32)
                row = 0
                for ii,jj in selected_pos:
                    cur_selected_first[row] = cur_first_tubes[jj]
                    cur_selected_last[row] = cur_last_tubes[jj]
                    row += 1

                for ii,jj in selected_neg:
                    cur_selected_first[row] = cur_first_tubes[jj]
                    cur_selected_last[row] = cur_last_tubes[jj]
                    row += 1
                
                cur_selected_tubes = np.concatenate([cur_selected_first,
                                                    cur_selected_tubes,
                                                    cur_selected_last], axis=1)
                
            elif args.temporal_mode == "extrapolate":    # linear extrapolation
                cur_selected_tubes = extrapolate_tubes(cur_selected_tubes, args.T)

            else:    # mean tubes
                mean_tubes = np.mean(cur_selected_tubes, axis=1, keepdims=True)
                mean_tubes = np.tile(mean_tubes, (1,args.T,1))
                cur_selected_tubes = np.concatenate((mean_tubes, cur_selected_tubes, mean_tubes), axis=1)

        ###### check whether predicting neighbor is needed ######

        cur_target_first = np.zeros((len(selected_pos)+len(selected_neg), 1, 6+args.num_classes), dtype=np.float32)
        cur_target_last = np.zeros((len(selected_pos)+len(selected_neg), 1, 6+args.num_classes), dtype=np.float32)

        if args.temporal_mode == "predict" and step < args.max_iter and args.NUM_CHUNKS[step+1] == args.NUM_CHUNKS[step]+2:
            row = 0
            for ii, jj in selected_pos:
                cur_target_first[row,:,:4] = targets[b][ii,int((T_start-args.T)/args.T),:4]
                if cur_target_first[row,:,:4].sum() > 0:    # valid box
                    cur_target_first[row,:,5] = 1
                cur_target_first[row,:,6:] = targets[b][ii,int((T_start-args.T)/args.T),4:]

                cur_target_last[row,:,:4] = targets[b][ii,int((T_start+T_length)/args.T),:4]
                if cur_target_last[row,:,:4].sum() > 0:    # valid box
                    cur_target_last[row,:,5] = 1
                cur_target_last[row,:,6:] = targets[b][ii,int((T_start+T_length)/args.T),4:]
                row += 1

        cur_target_tubes = np.concatenate([cur_target_first,
                                          cur_target_tubes,
                                          cur_target_last], axis=1)

        selected_tubes.append(cur_selected_tubes)
        target_tubes.append(cur_target_tubes)

    return selected_tubes, target_tubes

def select_proposals(gt_tubes, anchors, scores=None, cls_thresh=0.2, max_pos_num=5, sampling="random", neg_ratio=2):

    """
    if scores is None, use iou as scores
    """

    ious = compute_tube_iou(gt_tubes[:,:,:4], anchors)
    if scores is None:
        scores = np.max(ious, axis=0)
#    print (ious)

    selected_pos = []
    selected_neg = []
    occupied = set()

    ####### select positive samples #########

    # assign at least one foreground for each gt
    temp = ious.copy()
    for i in range(ious.shape[0]):
        idx = np.argmax(np.max(temp, axis=1))
        for j in np.argsort(ious[idx,:])[::-1]:
            if int(j) not in occupied:
                occupied.add(int(j))
                selected_pos.append((idx, int(j)))
                temp[idx, :] = -1    # set it small
                break
    if len(selected_pos) > max_pos_num:
        random.shuffle(selected_pos)
        selected_pos = selected_pos[:max_pos_num]

    pos_cand_idx = list(np.where(np.sum(ious>cls_thresh, axis=0))[0])
    for idx in occupied:
        if idx in pos_cand_idx:
            pos_cand_idx.remove(idx)
    
    if len(pos_cand_idx) > 0 and len(selected_pos) < max_pos_num:
        pos_candidate = list(zip(np.argmax(ious[:,pos_cand_idx], axis=0), pos_cand_idx))

        pos_scores = np.ones((len(pos_cand_idx),)) / len(pos_cand_idx)    # random selection for positive samples
        selected_idx = np.random.choice(len(pos_cand_idx),
                                        min(len(pos_cand_idx), max_pos_num-len(selected_pos)),
                                        p=pos_scores, replace=False)
        for idx in selected_idx:
            occupied.add(pos_candidate[idx][1])
            selected_pos.append(pos_candidate[idx])
            if len(selected_pos) == max_pos_num:
                break
    selected_pos = selected_pos[:max_pos_num]

    # pos_cand_idx are all occupied since they have high IOU with at least one gt
    for idx in pos_cand_idx:
        occupied.add(idx)

    ####### select negative samples #########

    neg_cand_idx = [i for i in range(anchors.shape[0]) if i not in occupied]

    if len(neg_cand_idx):
        # sampling hard negatives
        neg_scores = scores[neg_cand_idx]
        if sampling == "uniform":
            neg_scores = (neg_scores+1e-6) / np.sum(neg_scores+1e-6)
        elif sampling == "random":
            neg_scores = np.ones((len(neg_cand_idx),)) / len(neg_cand_idx)
        elif sampling == "softmax":
            neg_scores = np.exp(neg_scores) / np.sum(np.exp(neg_scores))
        else:
            raise NotImplementedError
        selected_idx = np.random.choice(len(neg_cand_idx), 
                                        min(len(selected_pos)*neg_ratio, len(neg_cand_idx)), 
                                        p=neg_scores, replace=False)

        for idx in selected_idx:
            # stiil assign the maximum iou box as gt box
            selected_neg.append((np.argmax(ious[:,neg_cand_idx[idx]]), neg_cand_idx[idx]))

    # keep the ratio
    if neg_ratio > 0:
        selected_pos = selected_pos[:max(max_pos_num, int(len(selected_neg)/neg_ratio))]

    return selected_pos, selected_neg, ious


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, T):
        self.T = T
        self.val = 0
        self.avg = 0
        self.reset()

    def reset(self):
        self.arr = [0]*self.T
        self.val = 0
        self.avg = 0
        self.ptr = 0
        self.flag = False    # to keep track whether arr has been once filled

    def update(self, val):
        self.val = val
        if self.flag:
            self.avg = self.avg + (val - self.arr[self.ptr]) / self.T
        else:
            self.avg = (self.avg * self.ptr + val) / (self.ptr + 1)
        self.arr[self.ptr] = val
        self.ptr = (self.ptr+1) % self.T
        if self.ptr == 0:
            self.flag = True

def get_gpu_memory():
    """
    Reference: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    """

    result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in result.decode().strip().split('\n')]
    return gpu_memory

