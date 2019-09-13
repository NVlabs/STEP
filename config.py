"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
def str2none(v):
    return None if v.lower() == "none" else v


def parse_config():
    parser = argparse.ArgumentParser()
    
    ############ General configs ##############

    parser.add_argument('--name', default='Debug', help='Experiment Name')
    parser.add_argument('--data_root', default='/vulcan/scratch/xyang35/datasets/AVA/', help='Location of dataset root directory')
    parser.add_argument('--save_root', default='/vulcan/scratch/xyang35/datasets/AVA/cache', help='Location to save checkpoint models')
    parser.add_argument('--base_net', default='i3d', help='Architecture used for backbone network')
    parser.add_argument('--det_net', default='two_branch', help='Architecture used for backbone network')
    parser.add_argument('--no_context', action='store_true', help='If true, context branch is no used.')
    parser.set_defaults(no_context=False)
    parser.add_argument('--fp16', action='store_true', help='If true, fp16 is used.')
    parser.set_defaults(fp16=False)
    
    parser.add_argument('--kinetics_pretrain', default=None, type=str2none, help='Path to the Kinetics pretrained model')
    parser.add_argument('--proposal_path_train', default=None, type=str2none, help='Path to the extracted proposals')
    parser.add_argument('--proposal_path_val', default=None, type=str2none, help='Path to the extracted proposals')
    parser.add_argument('--input_type', default='rgb', type=str, help='Input type for model: rgb | flow | stack')
    parser.add_argument('--dataset', default='ava', help='dataset name')
    parser.add_argument('--num_classes', default=60, type=int, help='Number of classes')
    parser.add_argument('--T', default=3, type=int, help='Sequence length for a tube')
    parser.add_argument('--fps', default=12, type=int, help='FPS for sequence')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
    parser.add_argument('--cuda', default=True, type=str2bool, help='whether to use GPU')
    parser.add_argument('--mGPUs', default=True, type=str2bool, help='whether to use multiple GPUs')
    parser.add_argument('--max_iter', default=1, type=int, help='the number of iterative updates, start from 1')

    ############ Trianing configs ##############

    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
    parser.add_argument('--max_epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--start_epochs', default=0, type=int, help='starting number of training epochs')
    parser.add_argument('--start_iteration', default=0, type=int, help='starting number of training steps')
    parser.add_argument('--save_step', default=9999999, type=int, help='Inteval of iterations to save model')
    parser.add_argument('--print_step', default=100, type=int, help='Inteval of iterations to print losses')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Which optimizer to use: sgd | adam')
    parser.add_argument('--base_lr', default=1e-3, type=float, help='initial learning rate for base_net')
    parser.add_argument('--det_lr', default=1e-3, type=float, help='initial learning rate for det_net')
    parser.add_argument('--det_lr0', default=1e-3, type=float, help='initial learning rate for det_net (i3d part)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--milestones', default='-1', type=str, help='epoch numbers where learing rate to be dropped')
    parser.add_argument('--weight_decay', default=1e-7, type=float, help='Weight decay for SGD, default 5e-5')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')


    parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
    parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
    parser.add_argument('--nms_thresh', default=0.4, type=float, help='NMS threshold')
    parser.add_argument('--topk', default=-1, type=int, help='topk scores for selection and evaluation (default -1 for all scores)')
    parser.add_argument('--evaluate_topk', default=-1, type=int, help='topk scores for evaluation')
    
    parser.add_argument('--pool_mode', default='pool', type=str, help='Mode for aggregate ROI features: pool | align ')
    parser.add_argument('--pool_size', default=7, type=int, help='size for ROI pooling')
    parser.add_argument('--lambda_reg', default=10, type=float, help='lambda for tube regression')
    parser.add_argument('--lambda_cls', default=0.1, type=float, help='lambda for classification')
    parser.add_argument('--lambda_neighbor', default=0.1, type=float, help='lambda for frame-level classification')
    parser.add_argument('--cls_thresh', default="0.2", type=str, help='threshold for selecting positive example for action classification')
    parser.add_argument('--reg_thresh', default="0.2", type=str, help='threshold for selecting positive example for regression')
    parser.add_argument('--max_pos_num', default=5, type=int, help='the number of maximum positive tube for a batch')
    parser.add_argument('--neg_ratio', default=1, type=int, help='the ratio of negative to positive')
    parser.add_argument('--selection_sampling', default="softmax", type=str, help='choice of sampling method for choosing proposals: uniform | random | softmax')
    parser.add_argument('--selection_score', default="score", type=str, help='choice of sampling method for choosing proposals: score | iou')
    parser.add_argument('--selection_nms', default="False", type=str2bool, help='whether to use nms before selection')
    parser.add_argument('--NUM_SAMPLE', default=-1, type=int, help='')
    
    parser.add_argument('--do_flip', default=False, type=str2bool, help='Use random horizontal flipping for data augmentation')
    parser.add_argument('--do_crop', default=False, type=str2bool, help='Use random cropping for data augmentation')
    parser.add_argument('--do_photometric', default=False, type=str2bool, help='Use photometric distortion for data augmentation')
    parser.add_argument('--do_erase', default=False, type=str2bool, help='Use random erasing for data augmentation')
    parser.add_argument('--do_proposal_augment', default=False, type=str2bool, help='Use proposal augmentation')
    
    parser.add_argument('--pretrain_path', default=None, type=str2none, help='pretrained model for initialization')
    parser.add_argument('--resume_path', default=None, type=str2none, help='resume the model and optimization stage')
    parser.add_argument('--sampling', default="uniform", type=str, help='choice of sampling method for choosing proposals: uniform | random | softmax')
    parser.add_argument('--fc_dim', default=4096, type=int, help='dimensionality of fc layer')
    parser.add_argument('--iterative_mode', default="temporal", type=str, help='choice of iterative update mode: spatial | temporal')
    parser.add_argument('--temporal_mode', default="extrapolate", type=str, help='choice of temporal update mode: extrapolate | predict')
    parser.add_argument('--anchor_mode', default="1", type=str, help='choice of anchor mode: 1 | 2 | 3 | 4')
    parser.add_argument('--scale_norm', default=1, type=int, help='whether to normalize the scale: 0 for [0,255] | 1 for [0,1] | 2 for [-1,1]')
    parser.add_argument('--dropout', default=0., type=float, help='probability of an element to be zeroed.')
    parser.add_argument('--freeze_stats', default=True, type=str2bool, help='whether to use agnostic regression')
    parser.add_argument('--freeze_affine', default=True, type=str2bool, help='whether to use agnostic regression')
    
    parser.add_argument('--scheduler', default='cosine', type=str)
    parser.add_argument('--cycle_decay', default=1.0, type=float, help='Reduce the base_lr after the completion of each cycle.')
    parser.add_argument('--min_ratio', default=0., type=float, help='Minimum ratio decay for base_lr.')
    parser.add_argument('--warmup_iters', default=1000, type=int, help='The number of iterations for linear warmup.')
    
    
    ## Parse arguments
    args = parser.parse_args()

    args.milestones = [int(val) for val in args.milestones.split(',')]
    if args.iterative_mode == "spatial":
        args.NUM_CHUNKS = {1: 1, 2: 1, 3: 1}    # the number of chunks for temporal iterative updates
    elif args.iterative_mode == "temporal":
        args.NUM_CHUNKS = {1: 1, 2: 1, 3: 3, 4: 3}    # the number of chunks for temporal iterative updates
    elif args.iterative_mode == "temporal2":
        args.NUM_CHUNKS = {1: 1, 2: 3, 3: 3, 4: 5}    # the number of chunks for temporal iterative updates

    # parse a list of cls_thresh or reg_thresh, do padding automatically
    args.cls_thresh = [float(val) for val in args.cls_thresh.split(',')]
    if len(args.cls_thresh) < args.max_iter:
        for i in range(args.max_iter-len(args.cls_thresh)):
            args.cls_thresh.append(args.cls_thresh[-1])
    args.reg_thresh = [float(val) for val in args.reg_thresh.split(',')]
    if len(args.reg_thresh) < args.max_iter:
        for i in range(args.max_iter-len(args.reg_thresh)):
            args.reg_thresh.append(args.reg_thresh[-1])

    if args.base_net == "i3d":    # confirm settings for I3D backbone
        args.scale_norm = 2
        args.means = (0,0,0)
        args.stds = (1,1,1)

    return args
