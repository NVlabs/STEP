"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
from numpy import random

def extrapolate_tubes(tubes, T=6, height=400, width=400):
    """
    Linear extrapolation
    """

    new_tubes = np.zeros((tubes.shape[0], tubes.shape[1]+2*T, tubes.shape[2]), dtype=np.float32)
    new_tubes[:,T:-T] = tubes

    for i in range(T):
        new_tubes[:,-T+i] = (T/(T-1))*new_tubes[:,-T+i-1] - (1/(T-1))*new_tubes[:,-T+i-T]
        new_tubes[:,T-i-1] = (T/(T-1))*new_tubes[:,T-i] - (1/(T-1))*new_tubes[:,T-i+T-1]

    new_tubes[:,:, 0] = np.maximum(0, new_tubes[:,:, 0])
    new_tubes[:,:, 1] = np.maximum(0, new_tubes[:,:, 1])
    new_tubes[:,:, 2] = np.minimum(width-1, new_tubes[:,:, 2])
    new_tubes[:,:, 3] = np.minimum(height-1, new_tubes[:,:, 3])

    return new_tubes

def augment_tubes(tubes, width=400, height=400):
    """
    Augment the tubes with random jittering and scaling
    tubes : shape [num_tubes, T, 4]
    """

    jitter_step = width / 20
    scale_step = 0.2

    for i in range(tubes.shape[0]):

        x, y, w, h = get_center_size(tubes[i])

        # random jittering
        if random.randint(2):
            x += random.uniform(-jitter_step, jitter_step)
            y += random.uniform(-jitter_step, jitter_step)

        # random scaling
        if random.randint(2):
            w += random.uniform(-scale_step, scale_step) * w
            h += random.uniform(-scale_step, scale_step) * h

        tubes[i,:,0] = x - 0.5 * w
        tubes[i,:,1] = y - 0.5 * h
        tubes[i,:,2] = x + 0.5 * w - 1
        tubes[i,:,3] = y + 0.5 * h - 1

    return valid_tubes(tubes, width, height)

def valid_tubes(tubes, width=400, height=400):
    """
    Return valid tubes that have size >= 0 and inside the image
    tubes : shape [num_tubes, T, 4]

    Return:
    tubes: valid tubes
    mask_idx: the index of valid tubes
    """

    num_tubes, T, _ = tubes.shape
    boxes = tubes.reshape(-1, 4)

    if isinstance(boxes, np.ndarray):
        boxes[:, 0] = np.maximum(0, boxes[:, 0])
        boxes[:, 1] = np.maximum(0, boxes[:, 1])
        boxes[:, 2] = np.minimum(width, boxes[:, 2])
        boxes[:, 3] = np.minimum(height, boxes[:, 3])
    else:
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0)
        boxes[:, 2] = torch.clamp(boxes[:, 2], max=width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], max=height)

    # replace invalid boxes to whole image boxes
    for i in range(boxes.shape[0]):
        if not (boxes[i,0]<boxes[i,2]-2 and boxes[i,1]<boxes[i,3]-2):
            boxes[i, :2] = 0
            boxes[i, 2] = width
            boxes[i, 3] = height

    tubes = boxes.reshape(num_tubes, T, 4)

    return tubes


def valid_tubes_old(tubes, width=400, height=400):
    """
    Return valid tubes that have size >= 0 and inside the image
    tubes : shape [num_tubes, T, 4]

    Return:
    tubes: valid tubes
    mask_idx: the index of valid tubes
    """

    num_tubes, T, _ = tubes.shape
    tubes = tubes.reshape(-1, 4)

    tubes[:, 0] = np.maximum(0, tubes[:, 0])
    tubes[:, 1] = np.maximum(0, tubes[:, 1])
    tubes[:, 2] = np.minimum(width-1, tubes[:, 2])
    tubes[:, 3] = np.minimum(height-1, tubes[:, 3])

    valid_idx = np.where(np.logical_and(tubes[:,0]<tubes[:,2]-2, tubes[:,1]<tubes[:,3]-2))[0]

    indicator = np.zeros(tubes.shape[0])
    indicator[valid_idx] = 1
    indicator = indicator.reshape(num_tubes, T)

    mask_idx = np.where(indicator.sum(axis=-1)==T)[0]

    tubes = tubes.reshape(num_tubes, T, 4)
    tubes = tubes[mask_idx]

    return tubes, mask_idx


def get_center_size(boxes):
    """
    Compute the box center (x, y) and width and height
    Args:
        boxes: FloatTensor [num_boxes, 4]

    Reference: https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/rpn/bbox_transform.py
    """

    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    x = boxes[:, 0] + 0.5 * w
    y = boxes[:, 1] + 0.5 * h

    return x, y, w, h

def encode_coef(gt_tubes, tubes):
    """
    Transform x,y to new parameterization for regresssion

    Args:
        gt_tubes: FloatTensor [num_tubes, 4], ground truth tubes. 
        tubes: FloatTensor [num_tubes, 4], proposal tubes. 

    Return:
        coef: FloatTensor [num_tubes, 4], regression target
    """

    gt_x, gt_y, gt_w, gt_h = get_center_size(gt_tubes)
    x, y, w, h = get_center_size(tubes)

    tx = (gt_x - x) / w
    ty = (gt_y - y) / h
    tw = torch.log(gt_w / w)
    th = torch.log(gt_h / h)

    return torch.stack((tx, ty, tw, th), dim=1)

def decode_coef(anchors, deltas):
    """
    Inverse Transform from regression parameterization to (xmin, ymin, xmax, ymax)

    Args:
        anchors: FloatTensor [num_tubes, 4], proposal tubes. 
        deltas: FloatTensor [num_tubes, 4], refinement measure. 

    Return:
        pred_tubes: FloatTensor [num_tubes, 4], regression result with absolution position
    """
    x, y, w, h = get_center_size(anchors)

    pred_x = w * deltas[:, 0] + x
    pred_y = h * deltas[:, 1] + y
    pred_w = w * torch.exp(deltas[:, 2])
    pred_h = h * torch.exp(deltas[:, 3])

    pred_tubes = deltas.clone()
    pred_tubes[:, 0] = pred_x - 0.5 * pred_w
    pred_tubes[:, 1] = pred_y - 0.5 * pred_h
    pred_tubes[:, 2] = pred_x + 0.5 * pred_w - 1    # adjusted for +1 in get_center_size()
    pred_tubes[:, 3] = pred_y + 0.5 * pred_h - 1

    return pred_tubes

def scale_tubes(tubes, width, height):
    """
    scale x/y to [0,1]
    """
    tubes = np.maximum(tubes, 0.0)
    for i in range(4):
        scale = float(width) if i % 2 == 0 else float(height)
        tubes[:,:,i] = np.minimum(tubes[:,:,i], scale)
        tubes[:,:,i] /= scale
    return tubes

def scale_tubes_abs(tubes, width, height):
    """
    scale x/y from [0,1] to absolute position
    """
    tubes = np.maximum(tubes, 0.0)
    tubes = np.minimum(tubes, 1.0)
    for i in range(4):
        scale = float(width) if i % 2 == 0 else float(height)
        tubes[:,:,i] *= scale

    return tubes

def flatten_tubes(tubes, batch_idx=False):
    """
    Flatten the tubes and add batch_idx for ROI pooling
    Empty element is available for the one in batch with no valid tubes
    Args:
        tubes: list of Tensor with shape [num_tubes, T, dim]
        batch_idx: bool, whether to add batch_idx, true for proposal tubes, false for gt tubes
    Return:
        flat_tubes: Tensor with shape [-1, T, dim(+1 if batch_idx)]  (dim 0 is the batch_idx)
        tubes_nums: a list storing num_tubes of each element
    """


    _, T, dim = tubes[0].shape

    flat_tubes = []
    tubes_nums = []
    for i in range(len(tubes)):
        num_tubes = tubes[i].shape[0]
        tubes_nums.append(num_tubes)
        if num_tubes == 0:
            continue

        if batch_idx:
            temp = np.arange(T) + i*T
            temp = np.tile(np.reshape(temp, (1, -1)), (num_tubes, 1))
            flat_tubes.append(np.concatenate(( 
                temp.reshape(num_tubes,T,1).astype(tubes[i].dtype), tubes[i].copy()), axis=2))
        else:
            flat_tubes.append(tubes[i].copy())

    flat_tubes = np.concatenate(flat_tubes, axis=0)
    return flat_tubes, tubes_nums

def extend_tubes(tubes, ratio=1.2, width=400, height=400):
    """
    Extend the tubes by a ratio (default is 1.2)

    Args:
        tubes: Tensor with shape [-1, T, 5]  (dim 0 is the batch_idx)
    """

    x, y, w, h = get_center_size(tubes.view(-1,5)[:,-4:])

    w *= ratio
    h *= ratio
    ext_tubes = tubes.view(-1,5).clone()
    ext_tubes[:, 1] = torch.max(x - 0.5 * w, torch.zeros_like(x))
    ext_tubes[:, 2] = torch.max(y - 0.5 * h, torch.zeros_like(y))
    ext_tubes[:, 3] = torch.min(x + 0.5 * w - 1, torch.ones_like(x).fill_(width-1))
    ext_tubes[:, 4] = torch.min(y + 0.5 * h - 1, torch.ones_like(y).fill_(height-1))

    return ext_tubes.view(tubes.size())


def compute_box_iou(box_arr1, box_arr2):
    """
    Compute IOU between boxes
    Args:
        box_arr1: Float array [num_box1, 4], with content [xmin, ymin, xmax, ymax]
        box_arr2: Float array [num_box2, 4], with content [xmin, ymin, xmax, ymax]

    Return:
        IOU_arr: Float array [num_box1, num_box2]
    """

    if len(box_arr1.shape) < 2:
        box_arr1 = box_arr1.reshape(1, 4)
    if len(box_arr2.shape) < 2:
        box_arr2 = box_arr2.reshape(1, 4)

    IOU_arr = np.zeros((box_arr1.shape[0], box_arr2.shape[0]), dtype=np.float32)

    for i in range(box_arr1.shape[0]):
        for j in range(box_arr2.shape[0]):
            box1 = box_arr1[i]
            box2 = box_arr2[j]

            xmin = max(box1[0], box2[0])
            ymin = max(box1[1], box2[1])
            xmax = min(box1[2], box2[2])
            ymax = min(box1[3], box2[3])
            iw = np.maximum(xmax - xmin, 0.)
            ih = np.maximum(ymax - ymin, 0.)
            if iw>0 and ih>0:
                intsc = iw * ih
            else:
                intsc = 0.

            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + \
                    (box2[2]-box2[0])*(box2[3]-box2[1]) - intsc

            IOU_arr[i, j] = intsc / union

    return IOU_arr

def compute_tube_iou(tube_arr1, tube_arr2):
    """
    Compute IOU between tubes
    For two tube, the IOU is obtained by averaging the IOU between boxes within the sequence (IOU = 0 for invalid boxes)
    Two sequence with partial overlapping can be evaluated by padding zero boxes (before calling this function)

    Args:
        tube_arr1: Float array [num_tube1, T, 4], with content [xmin, ymin, xmax, ymax]
        tube_arr2: Float array [num_tube2, T, 4], with content [xmin, ymin, xmax, ymax]

    Return:
        IOU_arr: Float array [num_tube1, num_tube2]
    """

    if len(tube_arr1.shape) < 3:
        tube_arr1 = tube_arr1.reshape(1, -1, 4)
    if len(tube_arr2.shape) < 3:
        tube_arr2 = tube_arr2.reshape(1, -1, 4)
    assert tube_arr1.shape[1] == tube_arr2.shape[1], "Tube with different length!"
    T = tube_arr1.shape[1]

    IOU_arr = np.zeros((tube_arr1.shape[0], tube_arr2.shape[0]), dtype=np.float32)

    for i in range(tube_arr1.shape[0]):
        for j in range(tube_arr2.shape[0]):
            tube1 = tube_arr1[i]
            tube2 = tube_arr2[j]

            box_ious = 0
            count = 0
            for t in range(T):
                # check valid boxes
                if np.sum(tube1) and np.sum(tube2):
                    box_ious += float(compute_box_iou(tube1[t], tube2[t]))

                # otherwise just ious += 0
                count += 1
            if count > 0:
                box_ious /= count
            IOU_arr[i, j] = box_ious

    return IOU_arr
