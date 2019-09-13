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
from .data_utils import generate_anchors
import random


WIDTH, HEIGHT = 400, 400
TEM_REDUCE = 4    # 4 for I3D backbone

class CustomizedDataset(data.Dataset):
    """
    Customized Dataset to generate input sequences from videos
    Dataset should be constructed as follows:
        data_root/
            video1/
                frame0000.jpg
                frame0001.jpg
                ...
            video2/
                ...
            ...
    """

    def __init__(self, data_root, T=3, chunks=3, source_fps=30, target_fps=12, transform=None, stride=1, anchor_mode="1", im_format='frame%04d.jpg'):
        """
        Arguments:
            data_root: path to videos
            T: input sequence length
            source_fps: frame rate of the original videos
            target_fps: target frame rate for our model
            transform: input transformation
            stride: stride
            anchor_mode: anchor_mode used
            im_format: format used to load frames
        """

        self.data_root = data_root
        self.T = T
        self.chunks = chunks
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.transform = transform
        self.stride = stride
        self.anchor_mode = anchor_mode
        self.im_format = im_format

        self.make_list()
        print('Datalist len: ', len(self.data))
    
    def make_list(self):

        self.data = []
        video_paths = glob.glob(self.data_root + '/*')
        for video_path in video_paths:
            video_name = os.path.basename(video_path)

            frames = glob.glob(video_path + '/*.jpg')
            numf = len(frames)
            for i in range(numf):
                self.data.append((video_name, i, numf))


    def read_images(self, videoname, fid, numf):
        """
        Load images from disk for middel frame fid

        return:
            an array with shape (T, H,W,C)
        """

        # set stride according to source fps and target fps
        T = self.T * self.chunks * TEM_REDUCE
        stride = self.source_fps / self.target_fps
        images = []
    
        # left of middel frame
        num_left = int(T/2)
        p = fid
        for _ in range(num_left):
            img_name = os.path.join(self.data_root, videoname, self.im_format % max(0, int(p)))
            images.append(cv2.imread(img_name))
            p -= stride
        images = images[::-1]

        # right of middel frame
        num_right = T - num_left
        p = fid
        for _ in range(num_right):
            p += stride
            img_name = os.path.join(self.data_root, videoname, self.im_format % min(numf-1, int(p)))
            images.append(cv2.imread(img_name))

        return np.stack(images, axis=0)


    def __getitem__(self, index):
        """
        Return:
            images: FloatTensor, shape [T, C, H, W]
            anchors: FloatTensor, shape [num, self.T, 4]
        """

        # pull an example sequence
        data = self.data[index]
        videoname, fid, numf = data

        # load data
        images = self.read_images(videoname, fid, numf)
        # data augmentation
        images, _,_ = self.transform(images)

        # BGR to RGB (for opencv)
        images = images[:, :, :, (2,1,0)]
        # swap dimensions to [T, C, W, H]
        images = torch.from_numpy(images).permute(0,3,1,2)

        # get anchor tubes
        if self.anchor_mode == "1":
            anchors = generate_anchors([4/3, 2], [5/6, 3/4])
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        elif self.anchor_mode == "2":
            anchors = generate_anchors([4/3,2,3], [5/6,3/4,1/2]) 
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        elif self.anchor_mode == "3":
            anchors = generate_anchors([4/3,2,3,4], [5/6,3/4,1/2,1/4]) 
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        elif self.anchor_mode == "4":
            anchors = generate_anchors([4/3,2,3,4,5], [5/6,3/4,1/2,1/4,0]) 
            anchor_tubes = np.tile(np.expand_dims(anchors, axis=1), (1,self.T,1))
        else:    # void anchor
            anchor_tubes = np.zeros([1,self.T,4])

        # rescale tubes to absolute position
        anchor_tubes = scale_tubes_abs(anchor_tubes, WIDTH, HEIGHT)

        # collect useful information
        info = {'video_name': videoname, 'fid': fid}

        return images, anchor_tubes, info

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
            2) (list of tensors) proposal tubes for a given image are stacked on 0 dim
            3) (list of dict) informations
    """

    imgs = []
    tubes = []
    infos = []
    for sample in batch:
        imgs.append(sample[0])
        tubes.append(sample[1])
        infos.append(sample[2])

    if imgs[0] is not None:
        imgs = torch.stack(imgs, 0)

    return imgs, tubes, infos

