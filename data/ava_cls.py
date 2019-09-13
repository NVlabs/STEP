"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import os.path
import torch
import torch.utils.data as data
import pickle
import numpy as np
import cv2
import glob
from utils.tube_utils import scale_tubes, scale_tubes_abs
from external.ActivityNet.Evaluation.get_ava_performance import read_labelmap
from .data_utils import generate_anchors
from .augmentations import jaccard_numpy
import random


WIDTH, HEIGHT = 400, 400
TEM_REDUCE = 4    # 4 for I3D backbone
NUM_CLASSES = 60


def make_list(label_path, chunks=1, stride=1, foreground_only=True):
    """
    Loop for each video and frame (with "step")
    Return a list of selected tubes, gt_boxes
    """

    # load annotaion file
    with open(os.path.join(label_path),'rb') as fin:
        annots = pickle.load(fin)

    data_list = []
    videoname_list = []

    # loop through each video
    vid = 0
    for videoname in sorted(annots.keys()):
        videoname_list.append(videoname)
        numf = len(annots[videoname])
        frames = sorted(annots[videoname].keys())

        # loop through each frame
        for fid in np.arange(902, 1799, stride):    # AVA v2.1 annotations at timestamps 902:1798 inclusive

            # no foreground label
            if foreground_only and (not fid in frames):
                continue

            # with foreground labels
            if fid in frames:
                annotation = annots[videoname][fid]

                boxes = []
                labels = []
                persons = []
                # loop through each person
                for pid, val in annotation.items():
                    temp_boxes = [[] for _ in range(chunks)]
                    temp_labels = [[] for _ in range(chunks)]
                    temp_persons = [[] for _ in range(chunks)]
                    # center frame
                    mid = int(chunks/2)
                    temp_boxes[mid] = val['box']
                    temp_labels[mid] = val['label']
                    temp_persons[mid] = pid

                    # previous frames
                    for t in range(1, mid+1):
                        if fid-t in frames and pid in annots[videoname][fid-t]:    # valid frame with ground truth
                            temp_boxes[mid-t] = annots[videoname][fid-t][pid]['box']
                            temp_labels[mid-t] = annots[videoname][fid-t][pid]['label']
                            temp_persons[mid-t] = pid
                    # future frames
                    for t in range(1, mid+1):
                        if fid+t in frames and pid in annots[videoname][fid+t]:    # valid frame with ground truth
                            temp_boxes[mid+t] = annots[videoname][fid+t][pid]['box']
                            temp_labels[mid+t] = annots[videoname][fid+t][pid]['label']
                            temp_persons[mid+t] = pid

                    boxes.append(temp_boxes)
                    labels.append(temp_labels)
                    persons.append(temp_persons)
            else:
                boxes = [[[] for _ in range(chunks)]]
                labels = [[[] for _ in range(chunks)]]
                persons = [[[] for _ in range(chunks)]]

            data_list.append([vid, fid, boxes, labels, persons])

        vid += 1

    return data_list, videoname_list


def get_target_tubes(root, boxes, labels, num_classes=60):
    """
    Input:
        boxes: list of tubes (list of boxes (list))
        labels: list of list of labels (list)
    Output
        Shape of gt_tubes: [num_tubes, chunks, 4+num_classes]
    """

    chunks = len(boxes[0])

    # background frame
    if chunks == 0:
        return np.zeros((1, chunks, 4+num_classes), dtype=np.float32)

    label_map = os.path.join(root, 'label/ava_action_list_v2.1_for_activitynet_2018.pbtxt')
    categories, class_whitelist = read_labelmap(open(label_map, 'r'))
    classes = np.array(list(class_whitelist)) - 1

    gt_tubes = np.zeros((len(boxes), chunks, 4), dtype=np.float32)
    gt_classes = np.zeros((len(boxes), chunks, 80), dtype=np.float32)
    for i in range(len(boxes)):
        for t in range(chunks):
            if boxes[i][t]:
                gt_tubes[i,t] = boxes[i][t]
                for l in labels[i][t]:
                    gt_classes[i,t,l-1] = 1    # foreground labels in annotation start from 1

    if num_classes == 60:
        gt_classes = gt_classes[:,:,classes]
    gt = np.concatenate((gt_tubes, gt_classes), axis=2)

    return gt


def read_images(path, videoname, fid, num=36, fps=12):
    """
    Load images from disk for middel frame fid with given num and fps

    return:
        a list of array with shape (num, H,W,C)
    """


    images = []
    
    # left of middel frame
    num_left = int(num/2)
    i = 1
    while num_left > 0:
        img_path = os.path.join(path, videoname+'/{:05d}/'.format(fid-i))
        images.extend(_load_images(img_path, num=min(num_left, fps), fps=fps, direction='backward'))

        num_left -= fps
        i += 1
    # reverse list
    images = images[::-1]

    # right of middel frame
    num_right = int(np.ceil(num/2))
    i = 0
    while num_right > 0:
        img_path = os.path.join(path, videoname+'/{:05d}/'.format(fid+i))
        images.extend(_load_images(img_path, num=min(num_right, fps), fps=fps, direction='forward'))

        num_right -= fps
        i += 1

    return np.stack(images, axis=0)


def _load_images(path, num, fps=12, direction='forward'):
    """
    Load images in a folder wiht given num and fps, direction can be either 'forward' or 'backward'
    """

    img_names = glob.glob(os.path.join(path, '*.jpg'))
    if len(img_names) == 0:
        raise ValueError("Image path {} not Found".format(path))
    img_names = sorted(img_names)

    # resampling according to fps
    index = np.linspace(0, len(img_names), fps, endpoint=False, dtype=np.int)
    if direction == 'forward':
        index = index[:num]
    elif direction == 'backward':
        index = index[-num:][::-1]
    else:
        raise ValueError("Not recognized direction", direction)

    images = []
    for idx in index:
        img_name = img_names[idx]
        if os.path.isfile(img_name):
            img = cv2.imread(img_name)
            images.append(img)
        else:
            raise ValueError("Image not found!", img_name)

    return images

def sample_anchors(anchors, pos_num=1, neg_ratio=1, pos_thresh=0.75, neg_thresh=0.2, mode='train'):
    """ sampling positive and negative boxes near anchors"""

    neg_num = pos_num * neg_ratio

    new_anchors = []
    for i in range(anchors.shape[0]):
        anchor = anchors[i]
        w = anchor[2] - anchor[0]
        h = anchor[3] - anchor[1]
        x = anchor[0] + 0.5*w
        y = anchor[1] + 0.5*h

        pos_anchors = []
        neg_anchors = []
        if mode == 'train':
            for j in range(50):    # maximum trail 50
                # sample a box near anchor
                new_w = random.uniform(0.8*w, min(1,1.2*w))
                new_h = random.uniform(0.8*h, min(1,1.2*h))
                new_x = random.uniform(max(0.5*new_w, x-0.2*w), min(1-0.5*new_w, x+0.2*w))
                new_y = random.uniform(max(0.5*new_h, y-0.2*h), min(1-0.5*new_h, y+0.2*h))
    
                new_anchor = np.array([new_x-0.5*new_w, new_y-0.5*new_h,
                                       new_x+0.5*new_w, new_y+0.5*new_h], dtype=np.float)
                ious = jaccard_numpy(anchors, new_anchor)
                if len(pos_anchors)<pos_num and ious[i] > pos_thresh and (ious < neg_thresh).sum() == (ious.shape[0]-1):
                    pos_anchors.append(new_anchor)
                elif len(neg_anchors)<neg_num and (ious < neg_thresh).sum() == ious.shape[0]:
                    neg_anchors.append(new_anchor)
    
                if len(pos_anchors)==pos_num:
                    break
        else:
            pos_anchors.append(anchor)

        for j in range(50):    # maximum trail 50
            # sample a box near anchor
            new_w = random.choice((random.uniform(0.3*w, 0.7*w), min(1,random.uniform(1.5*w,2*w))))
            new_h = random.choice((random.uniform(0.3*h, 0.7*h), min(1,random.uniform(1.5*h,2*h))))
            new_x = random.choice((random.uniform(max(0.5*new_w, x-w), max(0.5*new_w, x-0.3*w)), \
                                   random.uniform(min(1-0.5*new_w, x+0.3*w), min(1-0.5*new_w, x+w))))
            new_y = random.choice((random.uniform(max(0.5*new_h, x-h), max(0.5*new_h, y-0.3*h)), \
                                   random.uniform(min(1-0.5*new_h, y+0.3*h), min(1-0.5*new_h, y+h))))

            new_anchor = np.array([new_x-0.5*new_w, new_y-0.5*new_h,
                                   new_x+0.5*new_w, new_y+0.5*new_h], dtype=np.float)
            ious = jaccard_numpy(anchors, new_anchor)
            if len(neg_anchors)<neg_num and (ious < neg_thresh).sum() == ious.shape[0]:
                neg_anchors.append(new_anchor)

            if len(neg_anchors)==neg_num:
                break

        if len(pos_anchors):
            new_anchors.extend(pos_anchors)
        else:
            new_anchors.append(anchor)
        new_anchors.extend(neg_anchors)

    new_anchors = np.stack(new_anchors, axis=0)
    return new_anchors


class AVADataset(data.Dataset):
    """AVA Action Detection Dataset
    to access input sequence, GT tubes and proposal tubes
    """

    def __init__(self, root, mode, input_type, T=3, chunks=1, fps=12, transform=None, stride=1, num_classes=60, foreground_only=False):
        """
        Args:
            root: str, root path of the dataset
            input_type: str, 'rgb' | 'flow' | 'stack' | 'label'
            mode: str, 'train', 'val' or 'test'
            T: int, tube length
            chunks: int, number of chunks
            fps: int, frame rate (default 12)
            transform: list of class, data augmentation / preprocessing
            stride: int, used for scan through whole video (usually set it to T/2 for training)
            anchor_mode: str, what anchor to use, 'gt' | 'cache' | '1' | '2' | '3' | '4' 
            num_class: int, number of action classes, 60 | 80
            foreground_only: bool, whether include frames with no foreground actions (usually False for val and test)
        """


        self.name = 'ava'
        self.root = root
        self.mode = mode
        self.input_type = input_type
        self.T = T
        self.chunks = chunks
        self.fps = fps
        self.transform = transform
        self.stride = stride
        self.num_classes = num_classes
        self.foreground_only = foreground_only


        self.imgpath_rgb = os.path.join(root, 'frames/')
        if self.mode == 'train':
            self.label_path = os.path.join(root, 'label/train.pkl')
        elif self.mode == 'val':
            self.label_path = os.path.join(root, 'label/val.pkl')
        else:
            self.stride = 1
            self.label_path = os.path.join(root, 'label/val.pkl')
            self.foreground_only = False
           
        data_list, videoname_list = make_list(self.label_path, self.chunks, self.stride, self.foreground_only)
        self.data = data_list

        self.video_name = videoname_list
        print(self.mode+' set | Datalist len: ', len(data_list))

    def __getitem__(self, index):
        """
        Return:
            images: FloatTensor, shape [T, C, H, W]
            target_tubes: FloatTensor, shape [num_selected, 4+num_classes]    (including labels)
            selected_anchors: FloatTensor, shape [num_selected, T, 4]
            info: dict ['vid', 'sf', 'label', 'num_selected']
        """

        # pull an example sequence
        data = self.data[index]
        vid, fid, boxes, labels, persons = data
        videoname = self.video_name[vid]

        gt_tubes = get_target_tubes(self.root, boxes, labels, self.num_classes)    # gt boxes scaled to [0, 1]

        # load data
        if self.input_type == "rgb":
            images = read_images(self.imgpath_rgb, videoname, fid, num=TEM_REDUCE*self.T*self.chunks, fps=self.fps)    # for i3d backbone
        else:
            images = None

        if self.input_type != 'label':
            # data augmentation
            if self.transform is not None:
                images, gt_tubes, _ = self.transform(images, gt_tubes, None)

            if self.input_type == 'rgb':
                # BGR to RGB (for opencv)
                images = images[:, :, :, (2,1,0)]

            # swap dimensions to [T, C, W, H]
            images = torch.from_numpy(images).permute(0,3,1,2)


        # use (sampled) ground truths as anchor tubes
        anchors = gt_tubes[:, int(self.chunks/2), :4]
        anchors = sample_anchors(anchors, neg_ratio=3, mode=self.mode)
        anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))

        # rescale tubes to absolute position
        gt_tubes[:,:,:4] = scale_tubes_abs(gt_tubes[:,:,:4], WIDTH, HEIGHT)
        anchor_tubes = scale_tubes_abs(anchor_tubes, WIDTH, HEIGHT)

        # collect useful information
        info = {'vid': vid, 'video_name': videoname, 'labels': labels, 'T': self.T,
                'boxes': boxes, 'chunks': self.chunks, 'fid': fid}

        return images, gt_tubes, anchor_tubes, info

    def __len__(self):
        return len(self.data)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images, tubes and anchors
       We use a list of tensors for tubes and anchors since they may have different sizes for each sequence
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) gt tubes for a given image are stacked on 0 dim
            3) (list of tensors) proposal tubes for a given image are stacked on 0 dim
            4) (list of dict) informations
    """

    imgs = []
    gt = []
    tubes = []
    infos = []
    for sample in batch:
        imgs.append(sample[0])
        gt.append(sample[1])
        tubes.append(sample[2])
        infos.append(sample[3])

    if imgs[0] is not None:
        imgs = torch.stack(imgs, 0)

    return imgs, gt, tubes, infos


