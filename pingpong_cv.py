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
import math
import numpy as np
import os
import sys

sys.path.insert(0, "..")
import zhuocv as zc


def check_image(img):
    zc.checkBlurByGradient(img)

def find_table(img, o_img_height, o_img_width):
    ## find white border
    DoB = zc.get_DoB(img, 1, 31, method = 'Average')
    mask_white = zc.color_inrange(DoB, 'HSV', V_L = 10)

    ## find purple table (roughly)
    mask_table = zc.color_inrange(img, 'HSV', H_L = 130, H_U = 160, S_L = 50, V_L = 50, V_U = 220)
    mask_table, _ = zc.get_big_blobs(mask_table, min_area = 50)
    mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_CLOSE, zc.generate_kernel(7, 'circular'), iterations = 1)
    mask_table, _ = zc.find_largest_CC(mask_table)
    if mask_table is None:
        rtn_msg = {'status': 'fail', 'message' : 'Cannot find table'}
        return (rtn_msg, None)
    mask_table_convex, _ = zc.make_convex(mask_table.copy(), app_ratio = 0.005)
    mask_table = np.bitwise_or(mask_table, mask_table_convex)
    mask_table_raw = mask_table.copy()

    ## fine tune the purple table based on white border
    mask_white = np.bitwise_and(np.bitwise_not(mask_table), mask_white)
    for i in range(15):
        mask_table = zc.expand(mask_table, 3)
        mask_table = np.bitwise_and(np.bitwise_not(mask_white), mask_table)
        if i % 4 == 3:
            mask_table, _ = zc.make_convex(mask_table, app_ratio = 0.01)
            mask_table = np.bitwise_and(np.bitwise_not(mask_white), mask_table)
    mask_table, _ = zc.find_largest_CC(mask_table)
    mask_table, hull_table = zc.make_convex(mask_table, app_ratio = 0.01)

    ## check if table is big enough
    table_area = cv2.contourArea(hull_table)
    table_area_percentage = float(table_area) / img.shape[0] / img.shape[1]
    if table_area_percentage < 0.06:
        rtn_msg = {'status' : 'fail', 'message' : "Detected table too small: %f" % table_area_percentage}
        return (rtn_msg, None)

    ## find top line of table
    hull_table = np.array(zc.sort_pts(hull_table[:,0,:], order_first = 'y'))
    ul = hull_table[0]
    ur = hull_table[1]
    if ul[0] > ur[0]:
        t = ul; ul = ur; ur = t
    i = 2
    # the top two points in the hull are probably on the top line, but may not be the corners
    while i < hull_table.shape[0] and hull_table[i, 1] - hull_table[0, 1] < 80:
        pt_tmp = hull_table[i]
        if pt_tmp[0] < ul[0] or pt_tmp[0] > ur[0]:
            # computing the area of the part of triangle that lies inside the table
            triangle = np.vstack([pt_tmp, ul, ur]).astype(np.int32)
            mask_triangle = np.zeros_like(mask_table)
            cv2.drawContours(mask_triangle, [triangle], 0, 255, -1)
            pts = mask_table_raw[mask_triangle.astype(bool)]
            if np.sum(pts == 255) > 10:
                break
            if pt_tmp[0] < ul[0]:
                ul = pt_tmp
            else:
                ur = pt_tmp
            i += 1
        else:
            break
    ul = [int(x) for x in ul]
    ur = [int(x) for x in ur]

    ## sanity checks about table top line detection
    if zc.euc_dist(ul, ur) ** 2 * 2.5 < table_area:
        rtn_msg = {'status' : 'fail', 'message' : "Table top line too short"}
        return (rtn_msg, None)
    if abs(zc.line_angle(ul, ur)) > 0.4:
        rtn_msg = {'status' : 'fail', 'message' : "Table top line tilted too much"}
        return (rtn_msg, None)
    # check if two table sides form a reasonable angle
    mask_table_bottom = mask_table.copy()
    mask_table_bottom[:-30] = 0
    p_left_most = zc.get_edge_point(mask_table_bottom, (-1, 0))
    p_right_most = zc.get_edge_point(mask_table_bottom, (1, 0))
    if p_left_most is None or p_right_most is None:
        rtn_msg = {'status' : 'fail', 'message' : "Table doesn't occupy bottom part of image"}
        return (rtn_msg, None)
    left_side_angle = zc.line_angle(ul, p_left_most)
    right_side_angle = zc.line_angle(ur, p_right_most)
    angle_diff = zc.angle_dist(left_side_angle, right_side_angle, angle_range = math.pi * 2)
    if abs(angle_diff) > 1.8:
        rtn_msg = {'status' : 'fail', 'message' : "Angle between two side edge not right"}
        return (rtn_msg, None)

    ## rotate to make opponent upright, use table edge as reference
    pts1 = np.float32([ul,ur,[ul[0] + (ur[1] - ul[1]), ul[1] - (ur[0] - ul[0])]])
    pts2 = np.float32([[0, o_img_height], [o_img_width, o_img_height], [0, 0]])
    M = cv2.getAffineTransform(pts1, pts2)
    img[np.bitwise_not(zc.get_mask(img, rtn_type = "bool", th = 3)), :] = [3,3,3]
    img_rotated = cv2.warpAffine(img, M, (o_img_width, o_img_height))

    ## sanity checks about rotated opponent image
    bool_img_rotated_valid = zc.get_mask(img_rotated, rtn_type = "bool")
    if float(bool_img_rotated_valid.sum()) / o_img_width / o_img_height < 0.7:
        rtn_msg = {'status' : 'fail', 'message' : "Valid area too small after rotation"}
        return (rtn_msg, None)

    rtn_msg = {'status' : 'success'}
    return (rtn_msg, (img_rotated, mask_table, M))

def find_pingpong(img, img_prev, mask_table, mask_ball_prev, rotation_matrix):
    def get_ball_stat(mask_ball):
        cnt_ball = zc.mask2cnt(mask_ball)
        area = cv2.contourArea(cnt_ball)
        center = cnt_ball.mean(axis = 0)[0]
        center_homo = np.hstack((center, 1)).reshape(3, 1)
        center_rotated = np.dot(rotation_matrix, center_homo)
        return (area, center_rotated)

    rtn_msg = {'status' : 'success'}

    mask_ball = zc.color_inrange(img, 'HSV', H_L = 165, H_U = 25, S_L = 60, V_L = 90, V_U = 240)
    mask_ball, _ = zc.get_small_blobs(mask_ball, max_area = 2300)
    mask_ball, _ = zc.get_big_blobs(mask_ball, min_area = 8)
    mask_ball, counter = zc.get_square_blobs(mask_ball, th_diff = 0.2, th_area = 0.2)

    if counter == 0:
        rtn_msg = {'status' : 'fail', 'message' : "No good color candidate"}
        return (rtn_msg, None)

    cnt_table = zc.mask2cnt(mask_table)
    loc_table_center = zc.get_contour_center(cnt_table)[::-1]
    mask_ball_ontable = np.bitwise_and(mask_ball, mask_table)
    mask_ball_ontable = zc.get_closest_blob(mask_ball_ontable, loc_table_center)

    if mask_ball_ontable is not None: # if any ball on the table, we don't have to rely on previous ball positions
        mask_ball = mask_ball_ontable
        return (rtn_msg, (mask_ball, get_ball_stat(mask_ball)))

    if mask_ball_prev is None: # mask_ball_ontable is already None
        rtn_msg = {'status' : 'fail', 'message' : "Cannot initialize a location of ball"}
        return (rtn_msg, None)

    cnt_ball_prev = zc.mask2cnt(mask_ball_prev)
    loc_ball_prev = zc.get_contour_center(cnt_ball_prev)[::-1]
    mask_ball = zc.get_closest_blob(mask_ball, loc_ball_prev)
    cnt_ball = zc.mask2cnt(mask_ball)
    loc_ball = zc.get_contour_center(cnt_ball)[::-1]
    ball_moved_dist = zc.euc_dist(loc_ball_prev, loc_ball)
    if ball_moved_dist > 110:
        rtn_msg = {'status' : 'fail', 'message' : "Lost track of ball: %d" % ball_moved_dist}
        return (rtn_msg, None)

    return (rtn_msg, (mask_ball, get_ball_stat(mask_ball)))

def find_opponent(img, img_prev, o_img_height):
    def draw_flow(img, flow, step = 16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis
    def draw_rects(img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    ## General preparations
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)

    # valid part of img_prev
    mask_img_prev_valid = zc.get_mask(img_prev, rtn_type = "mask")
    bool_img_prev_valid = zc.shrink(mask_img_prev_valid, 15, iterations = 3).astype(bool)
    bool_img_prev_invalid = np.bitwise_not(bool_img_prev_valid)
    mask_white_prev = zc.color_inrange(img_prev, 'HSV', S_U = 50, V_L = 130)
    bool_white_prev = zc.shrink(mask_white_prev, 13, iterations = 3, method = 'circular').astype(bool)
    # valid part of img
    mask_img_valid = zc.get_mask(img, rtn_type = "mask")
    bool_img_valid = zc.shrink(mask_img_valid, 15, iterations = 3).astype(bool)
    bool_img_invalid = np.bitwise_not(bool_img_valid)
    mask_white = zc.color_inrange(img, 'HSV', S_U = 50, V_L = 130)
    bool_white = zc.shrink(mask_white, 13, iterations = 3, method = 'circular').astype(bool)

    # prior score according to height
    row_score, col_score = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    row_score = img.shape[0] * 1.2 - row_score.astype(np.float32)

    ## method 1: optical flow - dense
    opt_flow = np.zeros((bw.shape[0], bw.shape[1], 2), dtype=np.float32)
    opt_flow[::2, ::2, :] = cv2.calcOpticalFlowFarneback(
        bw_prev[::2, ::2], bw[::2, ::2], None, pyr_scale=0.5, levels=1, winsize=15, iterations=3, poly_n=7,
        poly_sigma=1.5, flags=0)
    # clean optical flow
    mag_flow = np.sqrt(np.sum(np.square(opt_flow), axis = 2))
    bool_flow_valid = mag_flow > 2
    bool_flow_valid = np.bitwise_and(bool_flow_valid, bool_img_prev_valid)
    bool_flow_valid = np.bitwise_and(bool_flow_valid, np.bitwise_not(bool_white_prev))
    bool_flow_invalid = np.bitwise_not(bool_flow_valid)
    # substract all the flow by flow average
    x_ave = np.mean(opt_flow[bool_flow_valid, 0])
    y_ave = np.mean(opt_flow[bool_flow_valid, 1])
    opt_flow[:, :, 0] -= x_ave
    opt_flow[:, :, 1] -= y_ave
    opt_flow[bool_flow_invalid, :] = 0

    # give the flow a "score"
    score_flow = np.sqrt(np.sum(np.square(opt_flow), axis = 2))
    score_flow = score_flow * row_score
    score_horizonal = np.sum(score_flow, axis = 0)
    low_pass_h = np.ones(120)
    low_pass_h /= low_pass_h.sum()
    score_horizonal_filtered_dense = np.convolve(score_horizonal, low_pass_h, mode = 'same')

    ## method 2: optical flow - LK
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.03,
                          minDistance = 5,
                          blockSize = 3 )
    lk_params = dict(winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = cv2.goodFeaturesToTrack(bw_prev, mask = mask_img_prev_valid, useHarrisDetector = False, **feature_params)
    if p0 is None:
        # TODO: this is also a possible indication that the rally is not on
        rtn_msg = {'status': 'fail', 'message' : 'No good featuresToTrack at all, probably no one in the scene'}
        return (rtn_msg, None)
    p1, st, err = cv2.calcOpticalFlowPyrLK(bw_prev, bw, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    bool_flow_valid= np.bitwise_and(bool_img_valid, np.bitwise_not(bool_white))
    bool_flow_invalid= np.bitwise_not(bool_flow_valid)
    bool_flow_valid_prev = np.bitwise_and(bool_img_prev_valid, np.bitwise_not(bool_white_prev))
    bool_flow_invalid_prev = np.bitwise_not(bool_flow_valid_prev)
    is_reallygood = np.zeros((good_new.shape[0]), dtype = bool)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        if (bool_flow_invalid_prev[int(d), int(c)] or max(a, b) > o_img_height or
            min(a, b) < 0 or bool_flow_invalid[int(b), int(a)]):
            continue
        is_reallygood[i] = True
    reallygood_new = good_new[is_reallygood]
    reallygood_old = good_old[is_reallygood]
    motion = reallygood_new - reallygood_old
    motion_real = motion - np.mean(motion, axis = 0)
    score_flow = np.zeros(bw.shape, dtype = np.float32)
    score_flow[reallygood_old[:, 1].astype(np.int), reallygood_old[:, 0].astype(np.int)] = np.sqrt(np.sum(np.square(motion_real), axis = 1))
    score_flow = score_flow * row_score
    score_horizonal = np.sum(score_flow, axis = 0)
    low_pass_h = np.ones(120)
    low_pass_h /= low_pass_h.sum()
    score_horizonal_filtered_LK = np.convolve(score_horizonal, low_pass_h, mode = 'same')
    # if motion too small, probably no one is there...
    if np.max(score_horizonal_filtered_LK) < 300:
        # TODO: this is also a possible indication that the rally is not on
        rtn_msg = {'status': 'fail', 'message' : 'Motion too small, probably no one in the scene'}
        return (rtn_msg, None)

    ## method 3: remove white wall
    mask_white = zc.color_inrange(img_prev, 'HSV', S_U = 50, V_L = 130)

    score = row_score
    score[bool_img_invalid] = 0
    score[bool_white] = 0
    score_horizonal = np.sum(score, axis = 0)
    low_pass_h = np.ones(120)
    low_pass_h /= low_pass_h.sum()
    score_horizonal_filtered_wall = np.convolve(score_horizonal, low_pass_h, mode = 'same')

    ## combining results of three methods
    score_horizonal_filtered = score_horizonal_filtered_dense / 10 + score_horizonal_filtered_LK * 10
    opponent_x = np.argmax(score_horizonal_filtered)

    rtn_msg = {'status' : 'success'}
    return (rtn_msg, opponent_x)
