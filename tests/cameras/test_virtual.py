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

# Example of running these tests:
# ```bash
# pytest tests/cameras/test_virtual.py
# ```

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.virtual import VirtualCamera, VirtualCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

RealVideoCapture = cv2.VideoCapture

TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
SOURCE_IMAGE_PATH = TEST_ARTIFACTS_DIR / "image_160x120.png"


class MockLoopingVideoCapture:
    """Wraps the real OpenCV VideoCapture — reads file once, caches frame for subsequent reads."""

    def __init__(self, *args, **kwargs):
        args_clean = [str(a) if isinstance(a, Path) else a for a in args]
        self._real_vc = RealVideoCapture(*args_clean, **kwargs)
        self._cached_frame = None

    def read(self):
        ret, frame = self._real_vc.read()
        if ret:
            self._cached_frame = frame
            return ret, frame
        if not ret and self._cached_frame is not None:
            return True, self._cached_frame.copy()
        return ret, frame

    def __getattr__(self, name):
        return getattr(self._real_vc, name)


@pytest.fixture(autouse=True)
def patch_opencv_videocapture():
    module_path = OpenCVCamera.__module__
    target = f"{module_path}.cv2.VideoCapture"
    with patch(target, new=MockLoopingVideoCapture):
        yield


def make_source_camera(path: Path = SOURCE_IMAGE_PATH) -> OpenCVCamera:
    config = OpenCVCameraConfig(index_or_path=path, warmup_s=0)
    return OpenCVCamera(config)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


def test_virtual_config_defaults():
    cfg = VirtualCameraConfig(camera_key="top")
    assert cfg.camera_key == "top"
    assert cfg.fps is None
    assert cfg.width is None
    assert cfg.height is None
    assert cfg.type == "virtual"


def test_virtual_config_empty_camera_key_raises():
    with pytest.raises(ValueError, match="camera_key"):
        VirtualCameraConfig(camera_key="")


def test_virtual_config_with_dims():
    cfg = VirtualCameraConfig(camera_key="wrist", fps=10, width=80, height=60)
    assert cfg.fps == 10
    assert cfg.width == 80
    assert cfg.height == 60


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


def test_connect_and_disconnect():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=False)

    assert cam.is_connected

    cam.disconnect()
    assert not cam.is_connected

    src.disconnect()


def test_connect_requires_source_connected():
    src = make_source_camera()  # NOT connected
    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)

    with pytest.raises(RuntimeError, match="source camera"):
        cam.connect()


def test_connect_already_connected():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=False)

    with pytest.raises(DeviceAlreadyConnectedError):
        cam.connect()

    cam.disconnect()
    src.disconnect()


def test_disconnect_before_connect():
    src = make_source_camera()
    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)

    with pytest.raises(DeviceNotConnectedError):
        cam.disconnect()


# ---------------------------------------------------------------------------
# Frame reading tests
# ---------------------------------------------------------------------------


def test_read_returns_ndarray():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    frame = cam.read()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3

    cam.disconnect()
    src.disconnect()


def test_read_before_connect():
    src = make_source_camera()
    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)

    with pytest.raises(DeviceNotConnectedError):
        cam.read()


def test_async_read_returns_ndarray():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    frame = cam.async_read(timeout_ms=2000)
    assert isinstance(frame, np.ndarray)

    cam.disconnect()
    src.disconnect()


def test_async_read_before_connect():
    src = make_source_camera()
    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)

    with pytest.raises(DeviceNotConnectedError):
        cam.async_read()


def test_async_read_timeout():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    # Consume the available frame, then request another immediately with 0 ms timeout
    cam.async_read(timeout_ms=2000)
    with pytest.raises(TimeoutError):
        cam.async_read(timeout_ms=0)

    cam.disconnect()
    src.disconnect()


def test_read_latest_returns_ndarray():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    # Prime the buffer
    _ = cam.read()
    frame = cam.read_latest()
    assert isinstance(frame, np.ndarray)

    cam.disconnect()
    src.disconnect()


def test_read_latest_before_connect():
    src = make_source_camera()
    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)

    with pytest.raises(DeviceNotConnectedError):
        cam.read_latest()


def test_read_latest_too_old():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    _ = cam.read()  # prime
    with pytest.raises(TimeoutError):
        cam.read_latest(max_age_ms=0)

    cam.disconnect()
    src.disconnect()


# ---------------------------------------------------------------------------
# Resize / dimensions tests
# ---------------------------------------------------------------------------


def test_inherits_source_dimensions():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")  # no width/height specified
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    assert cam.width == src.width
    assert cam.height == src.height

    frame = cam.read()
    assert frame.shape[1] == src.width
    assert frame.shape[0] == src.height

    cam.disconnect()
    src.disconnect()


@pytest.mark.parametrize("out_w,out_h", [(80, 60), (320, 240), (64, 64)])
def test_resize_output(out_w, out_h):
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top", width=out_w, height=out_h)
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    frame = cam.read()
    assert cam.width == out_w
    assert cam.height == out_h
    assert frame.shape[:2] == (out_h, out_w)

    cam.disconnect()
    src.disconnect()


# ---------------------------------------------------------------------------
# FPS throttling test
# ---------------------------------------------------------------------------


def test_inherits_source_fps():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top")  # no fps specified
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    assert cam.fps == src.fps

    cam.disconnect()
    src.disconnect()


def test_custom_fps_stored():
    src = make_source_camera()
    src.connect(warmup=False)

    cfg = VirtualCameraConfig(camera_key="top", fps=5)
    cam = VirtualCamera(cfg, src)
    cam.connect(warmup=True)

    assert cam.fps == 5

    cam.disconnect()
    src.disconnect()


# ---------------------------------------------------------------------------
# make_cameras_from_configs integration test
# ---------------------------------------------------------------------------


def test_make_cameras_from_configs_with_virtual():
    camera_configs = {
        "top": OpenCVCameraConfig(index_or_path=SOURCE_IMAGE_PATH, warmup_s=0),
        "top_small": VirtualCameraConfig(camera_key="top", width=80, height=60),
    }

    cameras = make_cameras_from_configs(camera_configs)

    assert "top" in cameras
    assert "top_small" in cameras
    assert isinstance(cameras["top"], OpenCVCamera)
    assert isinstance(cameras["top_small"], VirtualCamera)

    # Verify the virtual camera references the correct source
    assert cameras["top_small"].source_camera is cameras["top"]


def test_make_cameras_from_configs_virtual_missing_source():
    camera_configs = {
        "top_small": VirtualCameraConfig(camera_key="nonexistent"),
    }

    with pytest.raises(ValueError, match="nonexistent"):
        make_cameras_from_configs(camera_configs)
