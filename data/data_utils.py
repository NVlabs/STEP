"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import numpy as np
import torch
import cv2

def readsplitfile(splitfile):
    if not os.path.isfile(splitfile):
        raise ValueError('Split file not exists.')

    with open(splitfile, 'r') as f:
        videos = f.read().strip().split('\n')
    return videos

def generate_anchors(scales, overlaps):
    """
    Generate the initial anchors according to scales and overlaps
    Reference: Grid-CNN
    Arguments:
        scales -- list of float (>1)
        overlaps -- list of float
    Return:
        anchors -- array of shape [num_anchors, 4]
    """

    assert len(scales) == len(overlaps)

    anchors = []
    for scale, overlap in zip(scales, overlaps):
        size = 1./scale
        stride = size * (1-overlap)

        i = 0
        while i+size <= 1:
            j = 0
            while j+size <= 1:
                anchors.append([i,j,i+size,j+size])
                j += stride
            i += stride

    return np.asarray(anchors, dtype=np.float32)
