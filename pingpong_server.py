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

import cv2
import json
import numpy as np
import os
import select
import socket
import struct
import sys
import threading
import time
import traceback

if os.path.isdir("../../gabriel/server"):
    sys.path.insert(0, "../../gabriel/server")
import gabriel
LOG = gabriel.logging.getLogger(__name__)

import config
import pingpong_cv as pc
sys.path.insert(0, "..")
import zhuocv as zc

config.setup(is_streaming = True)
display_list = config.DISPLAY_LIST

LOG_TAG = "Pingpong Server: "

current_milli_time = lambda: int(round(time.time() * 1000))


class Trace:
    def __init__(self, n):
        self.trace = []
        self.max_len = n
    def insert(self, item):
        # item is a tuple of (timestamp, data)
        if item[1] is None: # item is None
            return
        self.trace.append(item)
        if len(self.trace) > self.max_len:
            del self.trace[0]
        while self.trace[-1][0] - self.trace[0][0] > 2000:
            del self.trace[0]
    def is_playing(self, t):
        counter = 0
        for item in self.trace:
            if t - item[0] < 2000:
                counter += 1
        return counter >= 4
    def leftOrRight(self):
        # TODO: there should be a better way of doing this
        area_min = 10000
        loc_min = None
        for item in self.trace:
            area, loc = item[1]
            if area < area_min:
                area_min = area
                loc_min = loc
        if loc_min is None:
            return "unknown"
        loc_x = loc_min[0]
        if loc_x < config.O_IMG_WIDTH / 2:
            return "left"
        else:
            return "right"


class PingpongHandler(gabriel.network.CommonHandler):
    def setup(self):
        LOG.info(LOG_TAG + "proxy connected to Pingpong server")
        super(PingpongHandler, self).setup()

        self.stop = threading.Event()

        self.prev_frame_info = None
        self.ball_trace = Trace(20)
        self.opponent_x = config.O_IMG_WIDTH / 2
        self.state = {'is_playing' : False,
                      'ball_position' : "unknown",
                      'opponent_position' : "unknown",
                     }

        self.last_played_t = time.time()
        self.last_played = "nothing"

        self.seen_opponent = False

        if config.PLAY_SOUND:
            ## for playing sound
            sound_server_addr = ("128.2.208.111", 4299)
            try:
                self.sound_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sound_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sound_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sound_sock.connect(sound_server_addr)
            except socket.error as e:
                LOG.warning(LOG_TAG + "Failed to connect to sound server at %s" % str(sound_server_addr))

    def __repr__(self):
        return "Pingpong Handler"

    def _handle_input_data(self):
        img_size = struct.unpack("!I", self._recv_all(4))[0]
        img = self._recv_all(img_size)
        cv_img = zc.raw2cv_image(img)
        return_data = self._handle_img(cv_img)

        packet = struct.pack("!I%ds" % len(return_data), len(return_data), return_data)
        self.request.sendall(packet)
        self.wfile.flush()

    def _handle_img(self, img):
        result = {'status' : "nothing"} # default
        frame_time = current_milli_time()
        self.state['is_playing'] = self.ball_trace.is_playing(frame_time) and self.seen_opponent
        if 'state' in display_list:
            pc.display_state(self.state)

        ## preprocessing of input image
        if max(img.shape) != config.IMAGE_MAX_WH:
            resize_ratio = float(config.IMAGE_MAX_WH) / max(img.shape)
            img = cv2.resize(img, (0, 0), fx = resize_ratio, fy = resize_ratio, interpolation = cv2.INTER_AREA)
        zc.check_and_display('input', img, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

        #pc.check_image(img, display_list)

        ## check if two frames are too close
        if self.prev_frame_info is not None and frame_time - self.prev_frame_info['time'] < 80:
            LOG.info(LOG_TAG + "two frames too close!")
            if gabriel.Debug.TIME_MEASUREMENT:
                result[gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = time.time()
            return json.dumps(result)

        ## find table
        rtn_msg, objects = pc.find_table(img, display_list)
        if rtn_msg['status'] != 'success':
            LOG.info(LOG_TAG + rtn_msg['message'])
            if gabriel.Debug.TIME_MEASUREMENT:
                result[gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = time.time()
            return json.dumps(result)

        img_rotated, mask_table, rotation_matrix = objects

        current_frame_info = {'time'        : frame_time,
                              'img'         : img,
                              'img_rotated' : img_rotated,
                              'mask_ball'   : None}

        ## in case we don't have good "previous" frame, process the current one and return
        mask_ball = None
        ball_stat = None
        if self.prev_frame_info is None or frame_time - self.prev_frame_info['time'] > 300:
            LOG.info(LOG_TAG +  "previous frame not good")
            rtn_msg, objects = pc.find_pingpong(img, None, mask_table, None, rotation_matrix, display_list)
            if rtn_msg['status'] != 'success':
                LOG.info(LOG_TAG + rtn_msg['message'])
            else:
                mask_ball, ball_stat = objects
            self.ball_trace.insert((frame_time, ball_stat))
            current_frame_info['mask_ball'] = mask_ball
            self.prev_frame_info = current_frame_info
            if gabriel.Debug.TIME_MEASUREMENT:
                result[gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = time.time()
            return json.dumps(result)

        ## now we do have an okay previous frame
        rtn_msg, objects = pc.find_pingpong(img, self.prev_frame_info['img'], mask_table, self.prev_frame_info['mask_ball'], rotation_matrix, display_list)
        if rtn_msg['status'] != 'success':
            LOG.info(LOG_TAG + rtn_msg['message'])
        else:
            mask_ball, ball_stat = objects
        self.ball_trace.insert((frame_time, ball_stat))
        current_frame_info['mask_ball'] = mask_ball

        ## determine where the wall was hit to
        self.state['ball_position'] = self.ball_trace.leftOrRight()
        if 'state' in display_list:
            pc.display_state(self.state)

        ## find position (relatively, left or right) of your opponent
        rtn_msg, objects = pc.find_opponent(img_rotated, self.prev_frame_info['img_rotated'], display_list)
        if rtn_msg['status'] != 'success':
            self.seen_opponent = False
            if 'state' in display_list:
                pc.display_state(self.state)
            print rtn_msg['message']
            self.prev_frame_info = current_frame_info
            result[gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = time.time()
            return json.dumps(result)
        self.seen_opponent = True
        opponent_x = objects
        # a simple averaging over history
        self.opponent_x = self.opponent_x * 0.7 + opponent_x * 0.3
        self.state['opponent_position'] = "left" if self.opponent_x < config.O_IMG_WIDTH * 0.58 else "right"
        if 'state' in display_list:
            pc.display_state(self.state)

        ## mode for not proving feedback
        if config.RECOGNIZE_ONLY:
            self.prev_frame_info = current_frame_info
            return json.dumps(result)

        ## now user has done something, provide some feedback
        result[gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = time.time()

        t = time.time()
        result['status'] = "success"
        if self.state['is_playing']:
            #if self.state['ball_position'] == "left" and self.state['opponent_position'] == "left":
            if self.state['opponent_position'] == "left":
                if (t - self.last_played_t < 3 and self.last_played == "right") or (t - self.last_played_t < 1):
                    result['status'] = "nothing"
                    return json.dumps(result)
                result['speech'] = "right"
                print "\n\n\n\nright\n\n\n\n"
                self.last_played_t = t
                self.last_played = "right"
                if config.PLAY_SOUND:
                    self.sound_sock.sendall("right")
            #elif self.state['ball_position'] == "right" and self.state['opponent_position'] == "right":
            elif self.state['opponent_position'] == "right":
                if (t - self.last_played_t < 3 and self.last_played == "left") or (t - self.last_played_t < 1):
                    result['status'] = "nothing"
                    return json.dumps(result)
                result['speech'] = "left"
                print "\n\n\n\nleft\n\n\n\n"
                self.last_played_t = t
                self.last_played = "left"
                if config.PLAY_SOUND:
                    self.sound_sock.sendall("leftt")
            else:
                result['status'] = "nothing"
        else:
            result['status'] = "nothing"

        return json.dumps(result)

    def terminate(self):
        if config.PLAY_SOUND:
            if self.sound_sock is not None:
                self.sound_sock.close()
        super(PingpongHandler, self).terminate()

class PingpongServer(gabriel.network.CommonServer):
    def __init__(self, port, handler):
        gabriel.network.CommonServer.__init__(self, port, handler) # cannot use super because it's old style class
        LOG.info(LOG_TAG + "* Pingpong server configuration")
        LOG.info(LOG_TAG + " - Open TCP Server at %s" % (str(self.server_address)))
        LOG.info(LOG_TAG + " - Disable nagle (No TCP delay)  : %s" %
                str(self.socket.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)))
        LOG.info(LOG_TAG + "-" * 50)

    def terminate(self):
        gabriel.network.CommonServer.terminate(self)

if __name__ == "__main__":
    pingpong_server = PingpongServer(config.TASK_SERVER_PORT, PingpongHandler)
    pingpong_thread = threading.Thread(target = pingpong_server.serve_forever)
    pingpong_thread.daemon = True

    try:
        pingpong_thread.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt as e:
        LOG.info(LOG_TAG + "Exit by user\n")
        sys.exit(1)
    except Exception as e:
        LOG.error(str(e))
        sys.exit(1)
    finally:
        if pingpong_server is not None:
            pingpong_server.terminate()
