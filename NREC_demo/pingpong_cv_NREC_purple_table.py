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
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

sys.path.insert(0, "..")
import zhuocv as zc
import config

current_milli_time = lambda: int(round(time.time() * 1000))


#############################################################
def plot_bar(bar_data, name = 'unknown', h = 300, w = 300, wait_time = None, is_resize = False, resize_method = "max", resize_max = None, resize_scale = None, save_image = None):
    if wait_time is None:
        wait_time = config.DISPLAY_WAIT_TIME
    if resize_max is None:
        resize_max = config.DISPLAY_MAX_PIXEL
    if resize_scale is None:
        resize_scale = config.DISPLAY_SCALE
    if save_image is None:
        save_image = config.SAVE_IMAGE
    zc.plot_bar(bar_data, name, h, w, wait_time, is_resize, resize_method, resize_max, resize_scale, save_image)

def display_state(state):
    img_display = np.ones((200, 400, 3), dtype = np.uint8) * 100
    if state['is_playing']:
        cv2.putText(img_display, "Playing", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0])
        cv2.putText(img_display, "The ball was last hit to %s" % state['ball_position'], (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 100, 100])
        cv2.putText(img_display, "Opponent is on the %s" % state['opponent_position'], (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 100, 100])
    else:
        cv2.putText(img_display, "Stopped", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255])

    zc.display_image('state', img_display, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

#double angle(  Point pt1,  Point pt2,  Point pt0 ) {
#    double dx1 = pt1.x - pt0.x;
#    double dy1 = pt1.y - pt0.y;
#    double dx2 = pt2.x - pt0.x;
#    double dy2 = pt2.y - pt0.y;
#    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
#}


#############################################################

def check_image(img, display_list):
    zc.checkBlurByGradient(img)

def find_screen(img, display_list):
    # ground
    mask_ground = zc.color_inrange(img, 'HSV', H_L = 15, H_U = 30, S_L = 55, S_U = 150, V_L = 100, V_U = 255)
    zc.check_and_display_mask('ground', img, mask_ground, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

    # red car
    mask_red = zc.color_inrange(img, 'HSV', H_L = 170, H_U = 10, S_L = 150)
    mask_ground = np.bitwise_or(mask_ground, mask_red)

    # ceiling
    #mask_ceiling = np.zeros((360,640), dtype = np.uint8)
    #mask_ceiling[:40,:] = 255
    #mask_ground = np.bitwise_or(mask_ground, mask_ceiling)

    # screen
    mask_screen1 = zc.color_inrange(img, 'HSV', H_L = 15, H_U = 45, S_L = 25, S_U = 150, V_L = 40, V_U = 150)
    mask_screen2 = zc.color_inrange(img, 'HSV', H_L = 0, H_U = 180, S_U = 25, V_L = 40, V_U = 150)
    mask_screen3 = ((img[:,:,2] + 5) > img[:,:,0]).astype(np.uint8) * 255
    mask_screen = np.bitwise_or(mask_screen1, np.bitwise_and(mask_screen2, mask_screen3))
    mask_screen = np.bitwise_and(np.bitwise_not(mask_ground), mask_screen)
    mask_screen = cv2.morphologyEx(mask_screen, cv2.MORPH_CLOSE, zc.generate_kernel(7, 'circular'), iterations = 1)
    #kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype = np.uint8)
    #mask_screen = cv2.dilate(mask_screen, kernel, iterations = 80)
    #mask_screen = zc.shrink(mask_screen, 5, iterations = 1)
    zc.check_and_display_mask('screen', img, mask_screen, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

    # backgournd
    #mask_background = zc.super_bitwise_or((mask_ground, mask_ceiling, mask_red, mask_screen))
    #mask_background = zc.shrink(mask_background, 5, iterations = 2)
    #zc.check_and_display_mask('background', img, mask_background, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

    return mask_screen


def find_table(img, display_list):
    #mask_screen = find_screen(img, display_list)
    ## find white border
    DoB = zc.get_DoB(img, 1, 31, method = 'Average')
    zc.check_and_display('DoB', DoB, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)
    mask_white = zc.color_inrange(DoB, 'HSV', V_L = 10)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, zc.generate_kernel(7, 'circular'), iterations = 1)
    zc.check_and_display_mask('mask_white_raw', img, mask_white, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

    ## find purple table (roughly)
    mask_table = zc.color_inrange(img, 'HSV', H_L = 130, H_U = 160, S_L = 60, V_L = 50, V_U = 240)
    mask_table, _ = zc.get_big_blobs(mask_table, min_area = 50)
    mask_table = cv2.morphologyEx(mask_table, cv2.MORPH_CLOSE, zc.generate_kernel(7, 'circular'), iterations = 1)
    mask_table, _ = zc.find_largest_CC(mask_table)
    if mask_table is None:
        rtn_msg = {'status': 'fail', 'message' : 'Cannot find table'}
        return (rtn_msg, None)
    mask_table_convex, _ = zc.make_convex(mask_table.copy(), app_ratio = 0.005)
    mask_table = np.bitwise_or(mask_table, mask_table_convex)
    mask_table_raw = mask_table.copy()
    zc.check_and_display_mask('table_purple', img, mask_table, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

    ## fine tune the purple table based on white border
    #mask_white = np.bitwise_and(np.bitwise_not(mask_table), mask_white)
    #mask_white = np.bitwise_and(np.bitwise_not(zc.shrink(mask_table, 3, method = "circular")), mask_white)
    mask_white = np.bitwise_and(np.bitwise_not(mask_table), mask_white)
    if 'mask_white' in display_list:
        gray = np.float32(mask_white)
        dst = cv2.cornerHarris(gray, 10, 3, 0.04)
        dst = cv2.dilate(dst, None)
        img_white = img.copy()
        img_white[mask_white > 0, :] = [0, 255, 0]
        #img_white[dst > 2.4e7] = [0, 0, 255]
        zc.check_and_display('mask_white', img_white, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)
    #mask_table, _ = zc.make_convex(mask_table, app_ratio = 0.005)
    for i in xrange(15):
        mask_table = zc.expand(mask_table, 3)
        mask_table = np.bitwise_and(np.bitwise_not(mask_white), mask_table)
        if i % 4 == 3:
            mask_table, _ = zc.make_convex(mask_table, app_ratio = 0.01)
            #img_display = img.copy()
            #img_display[mask_table > 0, :] = [0, 0, 255]
            #zc.display_image('table%d-b' % i, img_display, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)
            #mask_white = np.bitwise_and(np.bitwise_not(mask_table), mask_white)
            mask_table = np.bitwise_and(np.bitwise_not(mask_white), mask_table)
    mask_table, _ = zc.find_largest_CC(mask_table)
    if mask_table is None:
        rtn_msg = {'status': 'fail', 'message' : 'Cannot find table'}
        return (rtn_msg, None)
    mask_table, hull_table = zc.make_convex(mask_table, app_ratio = 0.01)
    zc.check_and_display_mask('table_purple_fixed', img, mask_table, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

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

    if 'table' in display_list:
        img_table = img.copy()
        img_table[mask_table.astype(bool), :] = [255, 0, 255]
        cv2.line(img_table, tuple(ul), tuple(ur), [0, 255, 0], 3)
        zc.check_and_display('table', img_table, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

    ## rotate to make opponent upright, use table edge as reference
    pts1 = np.float32([ul,ur,[ul[0] + (ur[1] - ul[1]), ul[1] - (ur[0] - ul[0])]])
    pts2 = np.float32([[0, config.O_IMG_HEIGHT], [config.O_IMG_WIDTH, config.O_IMG_HEIGHT], [0, 0]])
    M = cv2.getAffineTransform(pts1, pts2)
    img[np.bitwise_not(zc.get_mask(img, rtn_type = "bool", th = 3)), :] = [3,3,3]
    img_rotated = cv2.warpAffine(img, M, (config.O_IMG_WIDTH, config.O_IMG_HEIGHT))

    ## sanity checks about rotated opponent image
    bool_img_rotated_valid = zc.get_mask(img_rotated, rtn_type = "bool")
    if float(bool_img_rotated_valid.sum()) / config.O_IMG_WIDTH / config.O_IMG_HEIGHT < 0.6:
        rtn_msg = {'status' : 'fail', 'message' : "Valid area too small after rotation"}
        return (rtn_msg, None)

    rtn_msg = {'status' : 'success'}
    return (rtn_msg, (img_rotated, mask_table, M))

def find_pingpong(img, img_prev, mask_table, mask_ball_prev, rotation_matrix, display_list):
    def get_ball_stat(mask_ball):
        cnt_ball = zc.mask2cnt(mask_ball)
        area = cv2.contourArea(cnt_ball)
        center = cnt_ball.mean(axis = 0)[0]
        center_homo = np.hstack((center, 1)).reshape(3, 1)
        center_rotated = np.dot(rotation_matrix, center_homo)
        return (area, center_rotated)

    rtn_msg = {'status' : 'success'}


    mask_ball = zc.color_inrange(img, 'HSV', H_L = 165, H_U = 25, S_L = 65, V_L = 150, V_U = 255)

    mask_screen = find_screen(img, display_list)
    mask_screen, _ = zc.find_largest_CC(mask_screen)
    mask_possible = np.bitwise_or(mask_screen, zc.expand(mask_table, 3))
    mask_ball = np.bitwise_and(mask_ball, mask_possible)

    mask_ball, _ = zc.get_small_blobs(mask_ball, max_area = 2300)
    mask_ball, _ = zc.get_big_blobs(mask_ball, min_area = 8)
    mask_ball, counter = zc.get_square_blobs(mask_ball, th_diff = 0.2, th_area = 0.2)
    zc.check_and_display_mask('ball_raw', img, mask_ball, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

    if counter == 0:
        rtn_msg = {'status' : 'fail', 'message' : "No good color candidate"}
        return (rtn_msg, None)

    cnt_table = zc.mask2cnt(mask_table)
    loc_table_center = zc.get_contour_center(cnt_table)[::-1]
    mask_ball_ontable = np.bitwise_and(mask_ball, mask_table)
    mask_ball_ontable = zc.get_closest_blob(mask_ball_ontable, loc_table_center)

    if mask_ball_ontable is not None: # if any ball on the table, we don't have to rely on previous ball positions
        mask_ball = mask_ball_ontable
        zc.check_and_display_mask('ball', img, mask_ball, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)
        return (rtn_msg, (mask_ball, get_ball_stat(mask_ball)))

    if mask_ball_prev is None: # mask_ball_ontable is already None
        rtn_msg = {'status' : 'fail', 'message' : "Cannot initialize a location of ball"}
        return (rtn_msg, None)

    cnt_ball_prev = zc.mask2cnt(mask_ball_prev)
    loc_ball_prev = zc.get_contour_center(cnt_ball_prev)[::-1]
    mask_ball = zc.get_closest_blob(mask_ball, loc_ball_prev)
    zc.check_and_display_mask('ball', img, mask_ball, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)
    cnt_ball = zc.mask2cnt(mask_ball)
    loc_ball = zc.get_contour_center(cnt_ball)[::-1]
    ball_moved_dist = zc.euc_dist(loc_ball_prev, loc_ball)
    if ball_moved_dist > 110:
        rtn_msg = {'status' : 'fail', 'message' : "Lost track of ball: %d" % ball_moved_dist}
        return (rtn_msg, None)

    return (rtn_msg, (mask_ball, get_ball_stat(mask_ball)))

def find_opponent(img, img_prev, display_list):
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

    #start_time = current_milli_time()

    ## General preparations
    if 'opponent' in display_list:
        img_opponent = img_prev.copy()
    zc.check_and_display('rotated', img, display_list, is_resize = False, wait_time = config.DISPLAY_WAIT_TIME)
    zc.check_and_display('rotated_prev', img_prev, display_list, is_resize = False, wait_time = config.DISPLAY_WAIT_TIME)
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)

    # valid part of img_prev
    mask_img_prev_valid = zc.get_mask(img_prev, rtn_type = "mask")
    bool_img_prev_valid = zc.shrink(mask_img_prev_valid, 15, iterations = 3).astype(bool)
    bool_img_prev_invalid = np.bitwise_not(bool_img_prev_valid)
    mask_screen_prev = zc.shrink(find_screen(img_prev, []), 5)
    mask_white_prev = zc.color_inrange(img_prev, 'HSV', S_U = 50, V_L = 130)
    mask_white_prev = mask_screen_prev
    bool_white_prev = zc.shrink(mask_white_prev, 13, iterations = 3, method = 'circular').astype(bool)
    # valid part of img
    mask_img_valid = zc.get_mask(img, rtn_type = "mask")
    bool_img_valid = zc.shrink(mask_img_valid, 15, iterations = 3).astype(bool)
    bool_img_invalid = np.bitwise_not(bool_img_valid)
    mask_screen = zc.shrink(find_screen(img, []), 5)
    mask_white = zc.color_inrange(img, 'HSV', S_U = 50, V_L = 130)
    mask_white = mask_screen
    bool_white = zc.shrink(mask_white, 13, iterations = 3, method = 'circular').astype(bool)

    # prior score according to height
    row_score, col_score = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    #row_score = img.shape[0] * 1.2 - row_score.astype(np.float32)
    row_score = row_score.astype(np.float32) + 30

    #print "time0: %f" % (current_milli_time() - start_time)
    ## method 1: optical flow - dense
    opt_flow = np.zeros((bw.shape[0], bw.shape[1], 2), dtype=np.float32)
    opt_flow[::2, ::2, :] = cv2.calcOpticalFlowFarneback(bw_prev[::2, ::2], bw[::2, ::2], pyr_scale = 0.5, levels = 1, winsize = 15, iterations = 3, poly_n = 7, poly_sigma = 1.5, flags = 0)
    if 'denseflow' in display_list:
        zc.display_image('denseflow', draw_flow(bw, opt_flow, step = 16), is_resize = False, wait_time = config.DISPLAY_WAIT_TIME)
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
    if 'denseflow_cleaned' in display_list:
        zc.display_image('denseflow_cleaned', draw_flow(bw, opt_flow, step = 16), is_resize = False, wait_time = config.DISPLAY_WAIT_TIME)

    # give the flow a "score"
    score_flow = np.sqrt(np.sum(np.square(opt_flow), axis = 2))
    score_flow = score_flow * row_score
    score_horizonal = np.sum(score_flow, axis = 0)
    low_pass_h = np.ones(120)
    low_pass_h /= low_pass_h.sum()
    score_horizonal_filtered_dense = np.convolve(score_horizonal, low_pass_h, mode = 'same')
    if 'dense_hist' in display_list:
        plot_bar(score_horizonal_filtered_dense, name = 'dense_hist')
        print np.argmax(score_horizonal_filtered_dense)
    if np.max(score_horizonal_filtered_dense) < 20000:
        # TODO: this is also a possible indication that the rally is not on
        rtn_msg = {'status': 'fail', 'message' : 'Motion too small, probably no one in the scene'}
        return (rtn_msg, None)
    if 'opponent' in display_list:
        cv2.circle(img_opponent, (np.argmax(score_horizonal_filtered_dense), 220), 20, (0, 255, 0), -1)
    #print "time1: %f" % (current_milli_time() - start_time)

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
    if 'LKflow' in display_list:
        img_LK = img_prev.copy()
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(img_LK, (a, b), (c, d), (0, 255, 0), 2)
            cv2.circle(img_LK, (c, d), 5, (0, 255, 0), -1)
        zc.display_image('LKflow', img_LK, is_resize = False, wait_time = config.DISPLAY_WAIT_TIME)
    bool_flow_valid= np.bitwise_and(bool_img_valid, np.bitwise_not(bool_white))
    bool_flow_invalid= np.bitwise_not(bool_flow_valid)
    bool_flow_valid_prev = np.bitwise_and(bool_img_prev_valid, np.bitwise_not(bool_white_prev))
    bool_flow_invalid_prev = np.bitwise_not(bool_flow_valid_prev)
    is_reallygood = np.zeros((good_new.shape[0]), dtype = bool)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        if bool_flow_invalid_prev[d, c] or max(a, b) > config.O_IMG_HEIGHT or min(a, b) < 0 or bool_flow_invalid[b, a]:
            continue
        is_reallygood[i] = True
    reallygood_new = good_new[is_reallygood]
    reallygood_old = good_old[is_reallygood]
    motion = reallygood_new - reallygood_old
    motion_real = motion - np.mean(motion, axis = 0)
    if 'LKflow_cleaned' in display_list:
        img_LK_cleaned = img_prev.copy()
        img_LK_cleaned[bool_flow_invalid_prev, :] = [0, 0, 255]
        for i, (new, old) in enumerate(zip(reallygood_new, reallygood_old)):
            c, d = old.ravel()
            cv2.line(img_LK_cleaned, (c, d), (c + motion_real[i, 0], d + motion_real[i, 1]), (0, 255, 0), 2)
            cv2.circle(img_LK_cleaned, (c, d), 5, (0, 255, 0), -1)
        zc.display_image('LKflow_cleaned', img_LK_cleaned, is_resize = False, wait_time = config.DISPLAY_WAIT_TIME)
    score_flow = np.zeros(bw.shape, dtype = np.float32)
    score_flow[reallygood_old[:, 1].astype(np.int), reallygood_old[:, 0].astype(np.int)] = np.sqrt(np.sum(np.square(motion_real), axis = 1))
    score_flow = score_flow * row_score
    score_horizonal = np.sum(score_flow, axis = 0)
    low_pass_h = np.ones(120)
    low_pass_h /= low_pass_h.sum()
    score_horizonal_filtered_LK = np.convolve(score_horizonal, low_pass_h, mode = 'same')
    if 'LK_hist' in display_list:
        plot_bar(score_horizonal_filtered_LK, name = 'LK_hist')
        print np.argmax(score_horizonal_filtered_LK)
    # if motion too small, probably no one is there...
    if np.max(score_horizonal_filtered_LK) < 900:
        # TODO: this is also a possible indication that the rally is not on
        rtn_msg = {'status': 'fail', 'message' : 'Motion too small, probably no one in the scene'}
        return (rtn_msg, None)
    if 'opponent' in display_list:
        cv2.circle(img_opponent, (np.argmax(score_horizonal_filtered_LK), 220), 20, (0, 0, 255), -1)
    #print "time2: %f" % (current_milli_time() - start_time)

    ## method 3: remove white wall
    mask_screen = zc.shrink(find_screen(img_prev, []), 5)
    mask_white = zc.color_inrange(img_prev, 'HSV', S_U = 50, V_L = 130)
    mask_white = mask_screen
    zc.check_and_display('mask_white_wall', mask_white, display_list, is_resize = False, wait_time = config.DISPLAY_WAIT_TIME)
    score = row_score
    score[bool_img_invalid] = 0
    score[bool_white] = 0
    score_horizonal = np.sum(score, axis = 0)
    low_pass_h = np.ones(120)
    low_pass_h /= low_pass_h.sum()
    score_horizonal_filtered_wall = np.convolve(score_horizonal, low_pass_h, mode = 'same')
    if 'wall_hist' in display_list:
        plot_bar(score_horizonal_filtered_wall, name = 'wall_hist')
        print np.argmax(score_horizonal_filtered_wall)
    if 'opponent' in display_list:
        cv2.circle(img_opponent, (np.argmax(score_horizonal_filtered_wall), 220), 20, (255, 0, 0), -1)
    #print "time3: %f" % (current_milli_time() - start_time)

    ## combining results of three methods
    #score_horizonal_filtered = score_horizonal_filtered_dense * score_horizonal_filtered_LK * score_horizonal_filtered_wall
    score_horizonal_filtered = score_horizonal_filtered_wall / 20 + score_horizonal_filtered_dense / 10 + score_horizonal_filtered_LK * 2
    opponent_x = np.argmax(score_horizonal_filtered)
    if 'opponent' in display_list:
        cv2.circle(img_opponent, (opponent_x, 220), 20, (200, 200, 200), -1)
        zc.check_and_display('opponent', img_opponent, display_list, is_resize = False, wait_time = config.DISPLAY_WAIT_TIME)

    rtn_msg = {'status' : 'success'}
    return (rtn_msg, opponent_x)
