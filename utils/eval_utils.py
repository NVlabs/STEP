"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch

from external.ActivityNet.Evaluation.get_ava_performance import run_evaluation


def ava_evaluation(root, result_file, gt_file=None, iou=0.5):
    label_map = root + 'ava_action_list_v2.1_for_activitynet_2018.pbtxt'
    exclude_file = root + 'ava_val_excluded_timestamps_v2.1.csv'

    if gt_file is None:
        gt_file= root + 'ava_val_v2.1.csv'
    metrics = run_evaluation(open(label_map, 'r'), 
                         open(gt_file, 'r'), 
                         open(result_file, 'r'), 
                         open(exclude_file, 'r'))

    return metrics
