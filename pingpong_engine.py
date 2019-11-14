#
# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#           Roger Iyengar <iyengar@cmu.edu>
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

from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import numpy as np
import pingpong_cv
import cv2
import instruction_pb2
import logging
import time


ENGINE_NAME = "instruction"

# Convert all incoming frames to a fixed size to ease processing
IMAGE_MAX_WH = 640

# Opponent
O_IMG_WIDTH = 300
O_IMG_HEIGHT = 300


logger = logging.getLogger(__name__)

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


def complete_result_wrapper(result_wrapper, engine_fields):
    result_wrapper.engine_fields.Pack(engine_fields)

    return result_wrapper


def result_with_update(result_wrapper, engine_fields, speech):
    engine_fields.update_count += 1

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.TEXT
    result.engine_name = ENGINE_NAME
    result.payload = speech.encode(encoding="utf-8")

    result_wrapper.results.append(result)

    return complete_result_wrapper(result_wrapper, engine_fields)


class PingpongEngine(cognitive_engine.Engine):
    def __init__(self):
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

    def handle(self, from_client):
        if from_client.payload_type != gabriel_pb2.PayloadType.IMAGE:
            return cognitive_engine.wrong_input_format_error(
                from_client.frame_id)

        engine_fields = cognitive_engine.unpack_engine_fields(
            instruction_pb2.EngineFields, from_client)

        result_wrapper = gabriel_pb2.ResultWrapper()
        result_wrapper.frame_id = from_client.frame_id
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS

        img_array = np.asarray(bytearray(from_client.payload), dtype=np.int8)
        img = cv2.imdecode(img_array, -1)

        if max(img.shape) != IMAGE_MAX_WH:
            resize_ratio = float(IMAGE_MAX_WH) / max(img.shape[0], img.shape[1])
            img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio,
                             interpolation=cv2.INTER_AREA)

        ## check if two frames are too close
        if (self.prev_frame_info is not None and
            frame_time - self.prev_frame_info['time'] < 80):
            logger.info("two frames too close!")
            return complete_result_wrapper(result_wrapper, engine_fields)

        ## find table
        rtn_msg, objects = pingpong_cv.find_table(img)
        if rtn_msg['status'] != 'success':
            logger.info(rtn_msg['message'])
            return complete_result_wrapper(result_wrapper, engine_fields)

        img_rotated, mask_table, rotation_matrix = objects

        current_frame_info = {'time'        : frame_time,
                              'img'         : img,
                              'img_rotated' : img_rotated,
                              'mask_ball'   : None}

        ## in case we don't have good "previous" frame, process the current one
        # and return
        mask_ball = None
        ball_stat = None
        if (self.prev_frame_info is None or
            frame_time - self.prev_frame_info['time'] > 300):
            logger.info("previous frame not good")
            rtn_msg, objects = pingpong_cv.find_pingpong(
                img, None, mask_table, None, rotation_matrix)
            if rtn_msg['status'] != 'success':
                logger.info(rtn_msg['message'])
            else:
                mask_ball, ball_stat = objects
            self.ball_trace.insert((frame_time, ball_stat))
            current_frame_info['mask_ball'] = mask_ball
            self.prev_frame_info = current_frame_info
            return complete_result_wrapper(result_wrapper, engine_fields)

        ## now we do have an okay previous frame
        rtn_msg, objects = pingpong_cv.find_pingpong(
            img, self.prev_frame_info['img'], mask_table,
            self.prev_frame_info['mask_ball'], rotation_matrix)
        if rtn_msg['status'] != 'success':
            logger.info(rtn_msg['message'])
        else:
            mask_ball, ball_stat = objects
        self.ball_trace.insert((frame_time, ball_stat))
        current_frame_info['mask_ball'] = mask_ball

        ## determine where the wall was hit to
        self.state['ball_position'] = self.ball_trace.leftOrRight()

        ## find position (relatively, left or right) of your opponent
        rtn_msg, objects = pingpong_cv.find_opponent(
            img_rotated, self.prev_frame_info['img_rotated'])
        if rtn_msg['status'] != 'success':
            self.seen_opponent = False
            logger.info(rtn_msg['message'])
            self.prev_frame_info = current_frame_info
            return complete_result_wrapper(result_wrapper, engine_fields)
        self.seen_opponent = True
        opponent_x = objects
        # a simple averaging over history
        self.opponent_x = self.opponent_x * 0.7 + opponent_x * 0.3
        self.state['opponent_position'] = (
            "left" if self.opponent_x < O_IMG_WIDTH * 0.58 else "right")

        t = time.time()
        result['status'] = "success"
        if self.state['is_playing']:
            if self.state['opponent_position'] == "left":
                if ((t - self.last_played_t < 3 and self.last_played == "right")
                    or (t - self.last_played_t < 1)):
                    return complete_result_wrapper(
                        result_wrapper, engine_fields)

                speech = "right"
                self.last_played_t = t
                self.last_played = speech
                result_with_update(result_wrapper, engine_fields, speech)

            elif self.state['opponent_position'] == "right":
                if ((t - self.last_played_t < 3 and self.last_played == "left")
                    or (t - self.last_played_t < 1)):
                    return complete_result_wrapper(
                        result_wrapper, engine_fields)

                speech = "left"
                self.last_played_t = t
                self.last_played = speech
                result_with_update(result_wrapper, engine_fields, speech)


    return complete_result_wrapper(result_wrapper, engine_fields)
