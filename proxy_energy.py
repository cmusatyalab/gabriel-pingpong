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

import json
import multiprocessing
import numpy as np
import os
import pickle
import pprint
import Queue
from scikits.talkbox.features import mfcc
from scipy.io import wavfile
from sklearn.svm import SVC
import socket
import struct
import sys
import threading
import time
import wave

if os.path.isdir("../../gabriel/server"):
    sys.path.insert(0, "../../gabriel/server")
import gabriel
import gabriel.proxy
LOG = gabriel.logging.getLogger(__name__)

import config

LOG_TAG = "PingPong Proxy: "
APP_PATH = "./pingpong_server.py"

camera_state = True
camera_state_change_time = -1

class PingpongProxy(gabriel.proxy.CognitiveProcessThread):
    def __init__(self, image_queue, output_queue, task_server_addr, engine_id, log_flag = True):
        super(PingpongProxy, self).__init__(image_queue, output_queue, engine_id)
        self.log_flag = log_flag
        try:
            self.task_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.task_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.task_server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.task_server_sock.connect(task_server_addr)
        except socket.error as e:
            LOG.warning(LOG_TAG + "Failed to connect to task server at %s" % str(task_server_addr))

    def __repr__(self):
        return "Pingpong Proxy"

    def terminate(self):
        if self.task_server_sock is not None:
            self.task_server_sock.close()
        super(PingpongProxy, self).terminate()

    def _recv_all(self, sock, recv_size):
        data = ''
        while len(data) < recv_size:
            tmp_data = sock.recv(recv_size - len(data))
            if tmp_data == None:
                raise gabriel.network.TCPNetworkError("Cannot recv data at %s" % str(self))
            if len(tmp_data) == 0:
                raise gabriel.network.TCPZeroBytesError("Recv 0 bytes.")
            data += tmp_data
        return data

    def handle(self, header, data):
        # receive data from control VM
        #LOG.info("received new image")

        # feed data to the task assistance app
        packet = struct.pack("!I%ds" % len(data), len(data), data)
        self.task_server_sock.sendall(packet)
        try:
            result_size = struct.unpack("!I", self._recv_all(self.task_server_sock, 4))[0]
            result_data = self._recv_all(self.task_server_sock, result_size)
        except gabriel.network.TCPZeroBytesError as e:
            LOG.warning("Pingpong server disconnedted")
            result_data = json.dumps({'status' : "nothing"})
            self.terminate()
        #LOG.info("result : %s" % result_data)

        result_json = json.loads(result_data)
        header['status'] = result_json.pop('status')
        header[gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = result_json.pop(gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME, -1)
        result_data = json.dumps(result_json)

        return result_data

class ProxyAudio(gabriel.proxy.CognitiveProcessThread):
    def __init__(self, audio_queue, output_queue, engine_id):
        super(ProxyAudio, self).__init__(audio_queue, output_queue, engine_id)
        self.wav_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_audio.wav")

        N_SAMPLES = 16000 # sample per second
        self.BIT_PER_SAMPLE = 16
        self.step_size = N_SAMPLES / 1000 * 10 # 10 ms
        self.STE_past10 = []

        self.sound_past10 = []

        with open('classifier.pkl', 'rb') as f:
            self.clf = pickle.load(f)

    def handle(self, header, data):
        header['status'] = "success"

        # write data into file
        self.wav_file = wave.open(self.wav_file_path, "wb")
        self.wav_file.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        self.wav_file.writeframes(data)
        self.wav_file.close()

        # then load from file
        fs, data = wavfile.read(self.wav_file_path)
        #print "the number of samples: %d" % len(data)
        data = (data / 2.0 ** self.BIT_PER_SAMPLE) * 2 # 16 bit track, now normalized on [-1, 1)

        i = 0
        while i + self.step_size < len(data):
            data_curr = data[i : i + self.step_size]

            # MFCC
            ceps, mspec, spec = mfcc(data_curr, nwin = 160)
            ceps = ceps[:, 1:]

            # Short-time Energy (STE)
            s = 0.0
            for j in xrange(self.step_size):
                s += data_curr[j] * data_curr[j]
            s /= self.step_size
            self.STE_past10.append(s)
            if len(self.STE_past10) > 10:
                del self.STE_past10[0]
            STE_ave = sum(self.STE_past10) / 10.0
            peak_index = s - STE_ave
            r = np.zeros(1)
            if peak_index > 0.0001:
                r = self.clf.predict((ceps))

            if r.any():
                self.sound_past10.append(True)
            else:
                self.sound_past10.append(False)
            if len(self.sound_past10) > 200:
                del self.sound_past10[0]

            i += self.step_size

        #print self.sound_past10
        global camera_state, camera_state_change_time
        if not camera_state and time.time() - camera_state_change_time > 3:
            if sum(self.sound_past10) >= 2:
                header[gabriel.Protocol_client.JSON_KEY_CONTROL_MESSAGE] = json.dumps({gabriel.Protocol_control.JSON_KEY_SENSOR_TYPE_IMAGE : True})
                camera_state = True
                camera_state_change_time = time.time()
        if camera_state and time.time() - camera_state_change_time > 3:
            if len(self.sound_past10) == 200 and sum(self.sound_past10) < 2:
                header[gabriel.Protocol_client.JSON_KEY_CONTROL_MESSAGE] = json.dumps({gabriel.Protocol_control.JSON_KEY_SENSOR_TYPE_IMAGE : False})
                camera_state = False
                camera_state_change_time = time.time()

        return json.dumps({})

if __name__ == "__main__":
    settings = gabriel.util.process_command_line(sys.argv[1:])

    ip_addr, port = gabriel.network.get_registry_server_address(settings.address)
    service_list = gabriel.network.get_service_list(ip_addr, port)
    LOG.info("Gabriel Server :")
    LOG.info(pprint.pformat(service_list))

    video_ip = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_IP)
    video_port = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_PORT)
    audio_ip = service_list.get(gabriel.ServiceMeta.AUDIO_TCP_STREAMING_IP)
    audio_port = service_list.get(gabriel.ServiceMeta.AUDIO_TCP_STREAMING_PORT)
    ucomm_ip = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_IP)
    ucomm_port = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_PORT)

    # task assistance app thread
    app_thread = gabriel.proxy.AppLauncher(APP_PATH, is_print = True)
    app_thread.start()
    app_thread.isDaemon = True
    time.sleep(2)


    result_queue = multiprocessing.Queue()

    # image receiving thread
    image_queue = Queue.Queue(gabriel.Const.APP_LEVEL_TOKEN_SIZE)
    print "TOKEN SIZE OF OFFLOADING ENGINE: %d" % gabriel.Const.APP_LEVEL_TOKEN_SIZE
    video_streaming = gabriel.proxy.SensorReceiveClient((video_ip, video_port), image_queue)
    video_streaming.start()
    video_streaming.isDaemon = True

    # audio receiving and processing
    audio_queue = Queue.Queue(gabriel.Const.APP_LEVEL_TOKEN_SIZE)
    audio_streaming = gabriel.proxy.SensorReceiveClient((audio_ip, audio_port), audio_queue)
    audio_streaming.start()
    audio_streaming.isDaemon = True

    audio_app = ProxyAudio(audio_queue, result_queue, engine_id = "pingpong_audio")
    audio_app.start()
    audio_app.isDaemon = True

    # app proxy
    task_server_ip = gabriel.network.get_ip()
    task_server_port = config.TASK_SERVER_PORT
    app_proxy = PingpongProxy(image_queue, result_queue, (task_server_ip, task_server_port), engine_id = "Pingpong")
    app_proxy.start()
    app_proxy.isDaemon = True

    # result pub/sub
    result_pub = gabriel.proxy.ResultPublishClient((ucomm_ip, ucomm_port), result_queue)
    result_pub.start()
    result_pub.isDaemon = True

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        pass
    except KeyboardInterrupt as e:
        LOG.info("user exits\n")
    finally:
        if video_streaming is not None:
            video_streaming.terminate()
        if app_proxy is not None:
            app_proxy.terminate()
        if audio_streaming is not None:
            audio_streaming.terminate()
        if audio_app is not None:
            audio_app.terminate()
        result_pub.terminate()
        if app_thread is not None:
            app_thread.terminate()

