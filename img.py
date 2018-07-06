#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# This script is used for testing computer vision algorithms in the
# Lego Task Assistance project. It does processing for one image.
# Usage: python img.py <image-path>
#

'''
This script loads a single image from file, and try to generate relevant information of Pingpong playing.
It is primarily used as a quick test tool for the computer vision algorithm.
'''

import argparse
import cv2
import sys
import time

import config
import pingpong_cv as pc
sys.path.insert(0, "..")
import zhuocv as zc

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help = "The image to process",
                       )
    parser.add_argument("-p", "--previous_file",
                        help = "The image before the current one",
                        default = "None",
                       )
    args = parser.parse_args()
    return (args.input_file, args.previous_file)


# set configs...
config.setup(is_streaming = False)
display_list = config.DISPLAY_LIST

input_file, prev_file = parse_arguments()
if prev_file == "None":
    prev_file = input_file
img = cv2.imread(input_file)
img_prev = cv2.imread(prev_file)

## preprocessing of input images
if max(img.shape) != config.IMAGE_MAX_WH:
    resize_ratio = float(config.IMAGE_MAX_WH) / max(img.shape)
    img = cv2.resize(img, (0, 0), fx = resize_ratio, fy = resize_ratio, interpolation = cv2.INTER_AREA)
    img_prev = cv2.resize(img_prev, (0, 0), fx = resize_ratio, fy = resize_ratio, interpolation = cv2.INTER_AREA)
zc.check_and_display('input', img, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

####################### Start ###############################
#pc.check_image(img, display_list)
rtn_msg, objects = pc.find_table(img, display_list)
if objects is not None:
    img_rotated, mask_table, M = objects
print rtn_msg
if rtn_msg['status'] == 'success':
    rtn_msg, objects = pc.find_table(img_prev, display_list)
    if objects is not None:
        img_prev_rotated, mask_table_prev, M = objects
    print rtn_msg
if rtn_msg['status'] == 'success':
    rtn_msg, opponent_x = pc.find_opponent(img_rotated, img_prev_rotated, display_list)
    print rtn_msg
#if rtn_msg['status'] == 'success':
#    rtn_msg, objects = pc.find_pingpong(img_prev, None, mask_table_prev, None, M, display_list)
#    if objects is not None:
#        mask_ball_prev = objects
#    else:
#        mask_ball_prev = None
#    print rtn_msg
#if rtn_msg['status'] == 'success':
#    rtn_msg, objects = pc.find_pingpong(img, img_prev, mask_table, mask_ball_prev, display_list)
#    if objects is not None:
#        mask_ball = objects
#    print rtn_msg

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.stdout.write("user exits\n")
