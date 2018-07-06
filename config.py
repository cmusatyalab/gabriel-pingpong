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

# If True, configurations are set to process video stream in real-time (use with lego_server.py)
# If False, configurations are set to process one independent image (use with img.py)
IS_STREAMING = True

RECOGNIZE_ONLY = False

# Port for communication between proxy and task server
TASK_SERVER_PORT = 7080

# Whether or not to save the displayed image in a temporary directory
SAVE_IMAGE = False

# Whether to play verbal guidance on a laptop at the same time
PLAY_SOUND = False

# Convert all incoming frames to a fixed size to ease processing
IMAGE_MAX_WH = 640

BLUR_KERNEL_SIZE = IMAGE_MAX_WH / 16 + 1

# Display
DISPLAY_MAX_PIXEL = 640
DISPLAY_SCALE = 5
DISPLAY_LIST_ALL = ['input', 'DoB', 'mask_white', 'mask_white_raw', 'table_purple', 'table_purple_fixed', 'table',
                    'ball',
                    'rotated', 'rotated_prev', 'denseflow', 'denseflow_cleaned', 'dense_hist', 'LKflow', 'LKflow_cleaned', 'LK_hist', 'mask_white_wall', 'wall_hist',
                    'opponent', 'state']
DISPLAY_LIST_TEST = ['input', 'table_purple', 'table_purple_fixed', 'table']
DISPLAY_LIST_STREAM = ['input', 'state']
DISPLAY_LIST_TASK = []
if not IS_STREAMING:
    DISPLAY_LIST = DISPLAY_LIST_TEST
else:
    if RECOGNIZE_ONLY:
        DISPLAY_LIST = DISPLAY_LIST_STREAM
    else:
        DISPLAY_LIST = DISPLAY_LIST_TASK
DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500

# Opponent
O_IMG_WIDTH = 300
O_IMG_HEIGHT = 300

## Color detection
# H: hue, S: saturation, V: value (which means brightness)
# L: lower_bound, U: upper_bound, TH: threshold
# TODO:
BLUE = {'H' : 110, 'S_L' : 100, 'B_TH' : 110} # H: 108
YELLOW = {'H' : 30, 'S_L' : 100, 'B_TH' : 170} # H: 25 B_TH: 180
GREEN = {'H' : 70, 'S_L' : 100, 'B_TH' : 60} # H: 80 B_TH: 75
RED = {'H' : 0, 'S_L' : 100, 'B_TH' : 130}
BLACK = {'S_U' : 70, 'V_U' : 60}
#WHITE = {'S_U' : 60, 'B_L' : 101, 'B_TH' : 160} # this includes side white, too
WHITE = {'S_U' : 60, 'V_L' : 150}
BLACK_DOB_MIN_V = 15
BD_DOB_MIN_V = 30
# If using a labels to represent color, this is the right color: 0 means nothing (background) and 7 means unsure
COLOR_ORDER = ['nothing', 'white', 'green', 'yellow', 'red', 'blue', 'black', 'unsure']

## Optimizations
##

GOOD_WORDS = ["Excellent. ", "Great. ", "Good job. ", "Wonderful. "]

def setup(is_streaming):
    global IS_STREAMING, DISPLAY_LIST, DISPLAY_WAIT_TIME, SAVE_IMAGE
    IS_STREAMING = is_streaming
    if not IS_STREAMING:
        DISPLAY_LIST = DISPLAY_LIST_TEST
    else:
        if RECOGNIZE_ONLY:
            DISPLAY_LIST = DISPLAY_LIST_STREAM
        else:
            DISPLAY_LIST = DISPLAY_LIST_TASK
    DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500
    SAVE_IMAGE = not IS_STREAMING

