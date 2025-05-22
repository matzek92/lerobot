# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import threading
import time
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image
import imagezmq

from lerobot.common.robot_devices.cameras.configs import ImageZMQCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc


# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data = None
        self._data_ready = threading.Event()
        self._data_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        with self._data_lock:
            return self._data

    def datacopy(self):
        with self._data_lock:
            return copy.copy(self._data)

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        while not self._stop:
            temp_data = receiver.recv_image()
            with self._data_lock:
                self._data = temp_data
            self._data_ready.set()
        receiver.close()

    def close(self):
        self._stop = True


class ImgZMQCamera:
    """ """
    def __init__(self, config: ImageZMQCameraConfig):
        self.config = config
        self.hostname = config.hostname
        self.port = config.port
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.channels = config.channels

        # Linux uses ports for connecting to cameras
        self.camera = None
        self.is_connected = False
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"ImgZMQCamera(tcp://{self.hostname}:{self.port}) is already connected.")

        self.camera = VideoStreamSubscriber(self.hostname, self.port)
        self.is_connected = True
   
    def async_read(self):

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ImageZMQCamera({self.urls}) is not connected. Try running `camera.connect()` first."
            )
        """ # log the number of seconds it took to read the image
        

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        datetime.now(timezone.utc)"""
        num_tries = 0
        while True:
            
            start_time = time.perf_counter()
            data = self.camera.datacopy()
            if data is None:
                time.sleep(1 / 30)
                num_tries += 1
                if num_tries > 180:
                    raise TimeoutError("Timed out waiting for async_read() to start.")
                continue
            _, image = data
            if self.channels == 1 and len(image.shape) == 2:
                image = image.reshape((image.shape[0], image.shape[1], 1))
            elif self.channels == 3 and len(image.shape) == 2:
                image = image.reshape((image.shape[0], image.shape[1], 1))
                image = np.repeat(image, 3, axis=2)

            self.logs["delta_timestamp_s"] =  - start_time
            return image

       
    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ImgZMQCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        self.camera.close()
        self.is_connected = False
        self.camera = None

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
