"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import subprocess
import cv2
import glob
import random
import pickle

videodir = "<Path_to_AVA_dataset>"    # TODO: put the path to your AVA dataset here
root = os.path.dirname(videodir)
outdir_clips = os.path.join(root, "frames/")

clip_length = 1    # seconds
clip_time_padding = 1.0    # seconds


# utils
def hou_min_sec(millis):
    millis = int(millis)
    seconds = int((millis/1000) % 60)
    minutes = int((millis/(60*1000)) % 60)
    hours = int((millis/(60*60*1000)) % 60)
    return "%d:%d:%d" % (hours,minutes,seconds)

videonames = glob.glob(videodir + '*')
videonames = [os.path.basename(v).split('.')[0] for v in videonames]

for video_id in videonames:
    videofile = glob.glob(os.path.join(videodir, video_id+"*"))[0]

    for mid in range(900, 1800):

        # Extract clips
        clips_dir = os.path.join(outdir_clips, video_id)
        if not os.path.isdir(clips_dir):
            os.makedirs(clips_dir)
    
        if not os.path.isdir(os.path.join(clips_dir, "{:05d}".format(mid))):
            os.makedirs(os.path.join(clips_dir, "{:05d}".format(mid)))
    
            print ("Working on", os.path.join(clips_dir, "%d" % mid))
            ffmpeg_command = 'ffmpeg -ss {} -i {} -qscale:v 4 -vf scale=-1:360 -t {} {}/%05d.jpg'.format(mid, videofile, clip_length, os.path.join(clips_dir, "{:05d}".format(mid)))
            subprocess.call(ffmpeg_command, shell=True)
