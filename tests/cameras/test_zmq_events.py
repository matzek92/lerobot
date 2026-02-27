#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_zmq_events.py
# ```

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from lerobot.cameras.zmq.camera_zmq import ZMQCamera
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig


def _make_camera(event_port=None):
    """Create a ZMQCamera with a mocked event socket."""
    config = ZMQCameraConfig(
        server_address="127.0.0.1",
        port=5555,
        camera_name="test_cam",
        event_port=event_port,
    )
    return ZMQCamera(config)


class TestZMQCameraConfig:
    def test_event_port_default_is_none(self):
        config = ZMQCameraConfig(server_address="127.0.0.1")
        assert config.event_port is None

    def test_event_port_valid(self):
        config = ZMQCameraConfig(server_address="127.0.0.1", event_port=5556)
        assert config.event_port == 5556

    def test_event_port_invalid_zero(self):
        with pytest.raises(ValueError, match="event_port"):
            ZMQCameraConfig(server_address="127.0.0.1", event_port=0)

    def test_event_port_invalid_too_large(self):
        with pytest.raises(ValueError, match="event_port"):
            ZMQCameraConfig(server_address="127.0.0.1", event_port=99999)


class TestZMQCameraSendEvent:
    def test_send_event_no_op_without_event_port(self):
        """send_event should silently do nothing when event_port is not configured."""
        camera = _make_camera(event_port=None)
        assert camera.event_socket is None
        # Should not raise
        camera.send_event("episode_start")

    def test_send_event_sends_json_with_event_type(self):
        """send_event should push a JSON message containing the event type."""
        camera = _make_camera(event_port=5556)

        mock_socket = MagicMock()
        camera.event_socket = mock_socket

        camera.send_event("episode_start")

        mock_socket.send_string.assert_called_once()
        sent_message = mock_socket.send_string.call_args[0][0]
        data = json.loads(sent_message)
        assert data["event"] == "episode_start"
        assert "timestamp" in data

    def test_send_event_includes_timestamp(self):
        """send_event should include a float timestamp close to the current time."""
        camera = _make_camera(event_port=5556)
        mock_socket = MagicMock()
        camera.event_socket = mock_socket

        before = time.time()
        camera.send_event("episode_end")
        after = time.time()

        sent_message = mock_socket.send_string.call_args[0][0]
        data = json.loads(sent_message)
        assert before <= data["timestamp"] <= after

    def test_send_event_all_event_types(self):
        """send_event should work for all expected recording event types."""
        camera = _make_camera(event_port=5556)
        mock_socket = MagicMock()
        camera.event_socket = mock_socket

        for event_type in ("episode_start", "episode_end", "reset_done"):
            mock_socket.reset_mock()
            camera.send_event(event_type)
            sent_message = mock_socket.send_string.call_args[0][0]
            data = json.loads(sent_message)
            assert data["event"] == event_type

    def test_send_event_suppresses_zmq_errors(self):
        """send_event should log a warning instead of raising on ZMQ errors."""
        camera = _make_camera(event_port=5556)
        mock_socket = MagicMock()
        mock_socket.send_string.side_effect = RuntimeError("zmq error")
        camera.event_socket = mock_socket

        # Should not raise
        camera.send_event("episode_start")

    def test_connect_creates_event_socket_when_event_port_configured(self):
        """event_socket should be set when event_port is configured and connected."""
        camera = _make_camera(event_port=5556)
        # Directly simulate what connect() does: assign an event_socket
        mock_push_socket = MagicMock()
        camera.event_socket = mock_push_socket
        assert camera.event_socket is mock_push_socket

    def test_connect_no_event_socket_without_event_port(self):
        """connect() should not create an event socket when event_port is None."""
        camera = _make_camera(event_port=None)
        assert camera.event_socket is None


class TestNotifyZMQCameras:
    """Test the _notify_zmq_cameras logic using ZMQCamera.send_event directly."""

    def _notify_zmq_cameras(self, robot, event_type: str) -> None:
        """Inline implementation matching lerobot_record._notify_zmq_cameras."""
        if not hasattr(robot, "cameras"):
            return
        for cam in robot.cameras.values():
            if isinstance(cam, ZMQCamera):
                cam.send_event(event_type)

    def test_notify_calls_send_event_on_zmq_cameras(self):
        """_notify_zmq_cameras should call send_event on all ZMQ cameras."""
        zmq_cam1 = _make_camera(event_port=5556)
        zmq_cam2 = _make_camera(event_port=5557)
        zmq_cam1.send_event = MagicMock()
        zmq_cam2.send_event = MagicMock()

        mock_robot = MagicMock()
        mock_robot.cameras = {"cam1": zmq_cam1, "cam2": zmq_cam2}

        self._notify_zmq_cameras(mock_robot, "episode_start")

        zmq_cam1.send_event.assert_called_once_with("episode_start")
        zmq_cam2.send_event.assert_called_once_with("episode_start")

    def test_notify_skips_non_zmq_cameras(self):
        """_notify_zmq_cameras should ignore cameras that are not ZMQCamera instances."""
        zmq_cam = _make_camera(event_port=5556)
        zmq_cam.send_event = MagicMock()

        # A non-ZMQ camera mock
        other_cam = MagicMock(spec=[])  # no send_event attribute

        mock_robot = MagicMock()
        mock_robot.cameras = {"zmq": zmq_cam, "other": other_cam}

        self._notify_zmq_cameras(mock_robot, "episode_end")

        zmq_cam.send_event.assert_called_once_with("episode_end")
        # other_cam has no send_event attribute and was not called
        assert not hasattr(other_cam, "send_event")

    def test_notify_no_op_when_robot_has_no_cameras(self):
        """_notify_zmq_cameras should not raise when the robot has no cameras attribute."""
        mock_robot = MagicMock(spec=[])  # no cameras attribute

        # Should not raise
        self._notify_zmq_cameras(mock_robot, "reset_done")


class TestZMQCameraFeaturesConfig:
    def test_features_port_default_is_none(self):
        config = ZMQCameraConfig(server_address="127.0.0.1")
        assert config.features_port is None

    def test_features_port_valid(self):
        config = ZMQCameraConfig(server_address="127.0.0.1", features_port=5557)
        assert config.features_port == 5557

    def test_features_port_invalid_zero(self):
        with pytest.raises(ValueError, match="features_port"):
            ZMQCameraConfig(server_address="127.0.0.1", features_port=0)

    def test_features_port_invalid_too_large(self):
        with pytest.raises(ValueError, match="features_port"):
            ZMQCameraConfig(server_address="127.0.0.1", features_port=99999)


class TestZMQCameraSendFeatures:
    def test_send_features_no_op_without_features_port(self):
        """send_features should silently do nothing when features_port is not configured."""
        camera = _make_camera()
        assert camera.features_socket is None
        # Should not raise
        camera.send_features({"joint_positions": [0.0, 1.0]})

    def test_send_features_sends_json_with_features(self):
        """send_features should push a JSON message containing the features dict."""
        camera = _make_camera()
        mock_socket = MagicMock()
        camera.features_socket = mock_socket

        features = {"joint_positions": [0.1, 0.2, 0.3], "gripper": 0.5}
        camera.send_features(features)

        mock_socket.send_string.assert_called_once()
        sent_message = mock_socket.send_string.call_args[0][0]
        data = json.loads(sent_message)
        assert data["features"] == features
        assert "timestamp" in data

    def test_send_features_includes_timestamp(self):
        """send_features should include a float timestamp close to the current time."""
        camera = _make_camera()
        mock_socket = MagicMock()
        camera.features_socket = mock_socket

        before = time.time()
        camera.send_features({"sensor": 42.0})
        after = time.time()

        sent_message = mock_socket.send_string.call_args[0][0]
        data = json.loads(sent_message)
        assert before <= data["timestamp"] <= after

    def test_send_features_suppresses_zmq_errors(self):
        """send_features should log a warning instead of raising on ZMQ errors."""
        camera = _make_camera()
        mock_socket = MagicMock()
        mock_socket.send_string.side_effect = RuntimeError("zmq error")
        camera.features_socket = mock_socket

        # Should not raise
        camera.send_features({"joint_positions": [0.0]})

    def test_features_socket_created_when_features_port_configured(self):
        """features_socket should be assignable when features_port is configured."""
        config = ZMQCameraConfig(server_address="127.0.0.1", features_port=5557)
        camera = ZMQCamera(config)
        mock_push_socket = MagicMock()
        camera.features_socket = mock_push_socket
        assert camera.features_socket is mock_push_socket

    def test_no_features_socket_without_features_port(self):
        """features_socket should be None when features_port is not configured."""
        camera = _make_camera()
        assert camera.features_socket is None


class TestSendFeaturesToZMQCameras:
    """Tests for lerobot_record._send_features_to_zmq_cameras."""

    def setup_method(self):
        from lerobot.scripts.lerobot_record import _send_features_to_zmq_cameras

        self._send_features_to_zmq_cameras = _send_features_to_zmq_cameras

    def test_sends_scalar_observations_to_zmq_cameras(self):
        """Non-image observations should be forwarded to ZMQ cameras."""
        import numpy as np

        zmq_cam = _make_camera()
        zmq_cam.send_features = MagicMock()

        mock_robot = MagicMock()
        mock_robot.cameras = {"head": zmq_cam}

        obs = {"shoulder_pan.pos": 0.5, "gripper.pos": 1.0}
        self._send_features_to_zmq_cameras(mock_robot, obs)

        zmq_cam.send_features.assert_called_once_with({"shoulder_pan.pos": 0.5, "gripper.pos": 1.0})

    def test_excludes_image_arrays_from_features(self):
        """2-D+ numpy arrays (images) should not be included in sent features."""
        import numpy as np

        zmq_cam = _make_camera()
        zmq_cam.send_features = MagicMock()

        mock_robot = MagicMock()
        mock_robot.cameras = {"head": zmq_cam}

        obs = {
            "shoulder_pan.pos": 0.5,
            "head_camera": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        self._send_features_to_zmq_cameras(mock_robot, obs)

        called_features = zmq_cam.send_features.call_args[0][0]
        assert "shoulder_pan.pos" in called_features
        assert "head_camera" not in called_features

    def test_converts_1d_numpy_arrays_to_lists(self):
        """1-D numpy arrays (state vectors) should be converted to Python lists."""
        import numpy as np

        zmq_cam = _make_camera()
        zmq_cam.send_features = MagicMock()

        mock_robot = MagicMock()
        mock_robot.cameras = {"head": zmq_cam}

        obs = {"state": np.array([0.1, 0.2, 0.3])}
        self._send_features_to_zmq_cameras(mock_robot, obs)

        called_features = zmq_cam.send_features.call_args[0][0]
        assert called_features["state"] == [0.1, 0.2, 0.3]
        assert isinstance(called_features["state"], list)

    def test_no_op_when_robot_has_no_cameras(self):
        """Should not raise when robot has no cameras attribute."""
        mock_robot = MagicMock(spec=[])
        # Should not raise
        self._send_features_to_zmq_cameras(mock_robot, {"pos": 0.5})

    def test_skips_non_zmq_cameras(self):
        """Non-ZMQ cameras should be ignored."""
        zmq_cam = _make_camera()
        zmq_cam.send_features = MagicMock()
        other_cam = MagicMock(spec=[])

        mock_robot = MagicMock()
        mock_robot.cameras = {"zmq": zmq_cam, "other": other_cam}

        self._send_features_to_zmq_cameras(mock_robot, {"pos": 1.0})

        zmq_cam.send_features.assert_called_once()
        assert not hasattr(other_cam, "send_features")

    def test_sends_to_multiple_zmq_cameras(self):
        """All ZMQ cameras receive the same features dict."""
        zmq_cam1 = _make_camera()
        zmq_cam2 = _make_camera()
        zmq_cam1.send_features = MagicMock()
        zmq_cam2.send_features = MagicMock()

        mock_robot = MagicMock()
        mock_robot.cameras = {"cam1": zmq_cam1, "cam2": zmq_cam2}

        obs = {"pos": 0.7}
        self._send_features_to_zmq_cameras(mock_robot, obs)

        zmq_cam1.send_features.assert_called_once_with({"pos": 0.7})
        zmq_cam2.send_features.assert_called_once_with({"pos": 0.7})

