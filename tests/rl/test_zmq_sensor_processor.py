#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for ZmqObservationProcessorStep and ZmqSensorClient."""

import json
import time
from unittest.mock import MagicMock, patch

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.rl.joint_observations_processor import (
    ZmqObservationProcessorStep,
    ZmqSensorClient,
    ZmqSensorConfig,
)
from lerobot.utils.constants import OBS_STATE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zmq_pub_socket(port: int):
    """Create a ZMQ PUB socket bound to localhost:port."""
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://127.0.0.1:{port}")
    return ctx, sock


def _publish(sock, features: list[float]) -> None:
    msg = json.dumps({"timestamp": time.time(), "features": features})
    sock.send_string(msg)


# ---------------------------------------------------------------------------
# ZmqSensorConfig
# ---------------------------------------------------------------------------


def test_zmq_sensor_config_defaults():
    cfg = ZmqSensorConfig()
    assert cfg.address == "127.0.0.1"
    assert cfg.port == 5600
    assert cfg.feature_dim == 1
    assert cfg.timeout_ms == 100
    assert cfg.optional is True
    assert cfg.warmup_s == 0.0


def test_zmq_sensor_config_custom():
    cfg = ZmqSensorConfig(address="192.168.0.1", port=9000, feature_dim=5, timeout_ms=50)
    assert cfg.address == "192.168.0.1"
    assert cfg.port == 9000
    assert cfg.feature_dim == 5
    assert cfg.timeout_ms == 50


def test_zmq_sensor_config_optional_false():
    cfg = ZmqSensorConfig(optional=False, warmup_s=0.5)
    assert cfg.optional is False
    assert cfg.warmup_s == 0.5


# ---------------------------------------------------------------------------
# ZmqSensorClient (mocked socket)
# ---------------------------------------------------------------------------


def test_zmq_sensor_client_read_latest_default():
    """Before any message arrives, read_latest returns zeros."""
    import zmq

    cfg = ZmqSensorConfig(feature_dim=3, timeout_ms=10)
    client = ZmqSensorClient(cfg)

    with patch("zmq.Context") as mock_ctx_cls:
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx_cls.return_value = mock_ctx
        mock_ctx.socket.return_value = mock_socket
        # recv_string raises zmq.Again to simulate a receive timeout
        mock_socket.recv_string.side_effect = zmq.Again()
        client.connect()
        time.sleep(0.05)
        result = client.read_latest()
        client.disconnect()

    assert result == [0.0, 0.0, 0.0]


def test_zmq_sensor_client_read_latest_with_message():
    """After a message is set internally, read_latest returns it."""
    cfg = ZmqSensorConfig(feature_dim=3, timeout_ms=10)
    client = ZmqSensorClient(cfg)
    # Directly set internal state (bypassing ZMQ)
    with client._lock:
        client._latest_features = [1.0, 2.0, 3.0]

    result = client.read_latest()
    assert result == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# ZmqObservationProcessorStep (unit tests with mocked clients)
# ---------------------------------------------------------------------------


def _make_mock_client(features: list[float]) -> ZmqSensorClient:
    mock = MagicMock(spec=ZmqSensorClient)
    mock.read_latest.return_value = features
    return mock


def test_zmq_processor_no_clients():
    """With no ZMQ configs the processor should be a no-op."""
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = []
    step._clients = []

    state = torch.tensor([[1.0, 2.0, 3.0]])
    observation = {OBS_STATE: state}
    result = step.observation(observation)
    torch.testing.assert_close(result[OBS_STATE], state)


def test_zmq_processor_appends_to_obs_state():
    """Processor appends ZMQ features to OBS_STATE."""
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = [ZmqSensorConfig(feature_dim=2)]
    step._clients = [_make_mock_client([4.0, 5.0])]

    state = torch.tensor([[1.0, 2.0, 3.0]])
    observation = {OBS_STATE: state}
    result = step.observation(observation)

    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    torch.testing.assert_close(result[OBS_STATE], expected)


def test_zmq_processor_creates_obs_state_if_missing():
    """When OBS_STATE is missing the processor creates it from ZMQ features."""
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = [ZmqSensorConfig(feature_dim=2)]
    step._clients = [_make_mock_client([7.0, 8.0])]

    observation = {}
    result = step.observation(observation)

    expected = torch.tensor([[7.0, 8.0]])
    torch.testing.assert_close(result[OBS_STATE], expected)


def test_zmq_processor_multiple_clients():
    """Features from multiple clients are concatenated in order."""
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = [ZmqSensorConfig(feature_dim=2), ZmqSensorConfig(feature_dim=1)]
    step._clients = [_make_mock_client([1.0, 2.0]), _make_mock_client([9.0])]

    state = torch.tensor([[0.0]])
    observation = {OBS_STATE: state}
    result = step.observation(observation)

    expected = torch.tensor([[0.0, 1.0, 2.0, 9.0]])
    torch.testing.assert_close(result[OBS_STATE], expected)


def test_zmq_processor_full_pipeline():
    """Processor integrates correctly in a DataProcessorPipeline."""
    from lerobot.processor.converters import identity_transition
    from lerobot.processor.pipeline import DataProcessorPipeline

    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = [ZmqSensorConfig(feature_dim=2)]
    step._clients = [_make_mock_client([10.0, 20.0])]

    pipeline = DataProcessorPipeline(
        steps=[step],
        to_transition=identity_transition,
        to_output=identity_transition,
    )

    state = torch.tensor([[0.5, 1.5]])
    transition = create_transition(observation={OBS_STATE: state})
    result = pipeline(transition)

    expected = torch.tensor([[0.5, 1.5, 10.0, 20.0]])
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], expected)


# ---------------------------------------------------------------------------
# transform_features
# ---------------------------------------------------------------------------


def test_zmq_transform_features_extends_obs_state():
    """transform_features must increase OBS_STATE shape by total ZMQ dims."""
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = [ZmqSensorConfig(feature_dim=3), ZmqSensorConfig(feature_dim=2)]
    step._clients = []

    original = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    features = {PipelineFeatureType.OBSERVATION: {OBS_STATE: original}}
    updated = step.transform_features(features)

    new_feature = updated[PipelineFeatureType.OBSERVATION][OBS_STATE]
    assert new_feature.shape == (11,)  # 6 + 3 + 2
    assert new_feature.type == FeatureType.STATE


def test_zmq_transform_features_no_configs():
    """With no configs transform_features is a no-op."""
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = []
    step._clients = []

    original = PolicyFeature(type=FeatureType.STATE, shape=(4,))
    features = {PipelineFeatureType.OBSERVATION: {OBS_STATE: original}}
    updated = step.transform_features(features)

    assert updated[PipelineFeatureType.OBSERVATION][OBS_STATE] == original


def test_zmq_transform_features_missing_obs_state():
    """If OBS_STATE is absent transform_features does not crash."""
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = [ZmqSensorConfig(feature_dim=2)]
    step._clients = []

    features = {PipelineFeatureType.OBSERVATION: {}}
    updated = step.transform_features(features)

    assert OBS_STATE not in updated[PipelineFeatureType.OBSERVATION]


# ---------------------------------------------------------------------------
# get_config / reset
# ---------------------------------------------------------------------------


def test_zmq_processor_get_config():
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    cfg = ZmqSensorConfig(address="10.0.0.1", port=6000, feature_dim=4, timeout_ms=200)
    step.zmq_configs = [cfg]
    step._clients = []

    config = step.get_config()
    assert config["zmq_configs"][0]["address"] == "10.0.0.1"
    assert config["zmq_configs"][0]["port"] == 6000
    assert config["zmq_configs"][0]["feature_dim"] == 4
    assert config["zmq_configs"][0]["timeout_ms"] == 200


def test_zmq_processor_get_config_includes_optional_and_warmup():
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    cfg = ZmqSensorConfig(address="10.0.0.1", port=6000, feature_dim=4, timeout_ms=200, optional=False, warmup_s=1.5)
    step.zmq_configs = [cfg]
    step._clients = []

    config = step.get_config()
    assert config["zmq_configs"][0]["optional"] is False
    assert config["zmq_configs"][0]["warmup_s"] == 1.5


def test_zmq_processor_reset_does_not_raise():
    step = ZmqObservationProcessorStep.__new__(ZmqObservationProcessorStep)
    step.zmq_configs = []
    step._clients = []
    step.reset()  # Should not raise


# ---------------------------------------------------------------------------
# optional / warmup behaviour (mocked socket)
# ---------------------------------------------------------------------------


def _make_timeout_socket():
    """Return a mock ZMQ socket whose recv_string always times out."""
    import zmq

    mock_socket = MagicMock()
    mock_socket.recv_string.side_effect = zmq.Again()
    return mock_socket


def _patched_connect(client, mock_socket):
    """Call client.connect() with a fully mocked ZMQ context/socket."""
    with patch("zmq.Context") as mock_ctx_cls:
        mock_ctx = MagicMock()
        mock_ctx_cls.return_value = mock_ctx
        mock_ctx.socket.return_value = mock_socket
        client.connect()


def test_connect_no_warmup_does_not_raise_when_publisher_absent():
    """warmup_s=0 (default) — connect never blocks, even with no publisher."""
    import zmq

    cfg = ZmqSensorConfig(feature_dim=2, timeout_ms=10, optional=False, warmup_s=0.0)
    client = ZmqSensorClient(cfg)
    mock_socket = MagicMock()
    mock_socket.recv_string.side_effect = zmq.Again()

    _patched_connect(client, mock_socket)
    client.disconnect()  # Should not raise


def test_connect_optional_warmup_warns_when_no_publisher(caplog):
    """optional=True + warmup_s>0 — logs a warning but does not raise."""
    import logging

    import zmq

    cfg = ZmqSensorConfig(feature_dim=2, timeout_ms=10, optional=True, warmup_s=0.1)
    client = ZmqSensorClient(cfg)
    mock_socket = MagicMock()
    mock_socket.recv_string.side_effect = zmq.Again()

    with caplog.at_level(logging.WARNING, logger="lerobot.rl.joint_observations_processor"):
        _patched_connect(client, mock_socket)

    client.disconnect()

    assert any("optional" in record.message for record in caplog.records)


def test_connect_required_warmup_raises_when_no_publisher():
    """optional=False + warmup_s>0 — raises ConnectionError if no publisher."""
    import zmq

    import pytest

    cfg = ZmqSensorConfig(feature_dim=2, timeout_ms=10, optional=False, warmup_s=0.1)
    client = ZmqSensorClient(cfg)
    mock_socket = MagicMock()
    mock_socket.recv_string.side_effect = zmq.Again()

    with pytest.raises(ConnectionError, match="no data received"):
        _patched_connect(client, mock_socket)


def test_connect_required_warmup_succeeds_when_publisher_responds():
    """optional=False + warmup_s>0 — succeeds when publisher sends data quickly."""
    import zmq

    cfg = ZmqSensorConfig(feature_dim=2, timeout_ms=10, optional=False, warmup_s=0.5)
    client = ZmqSensorClient(cfg)

    # Simulate publisher: first call times out, second delivers data
    call_count = {"n": 0}

    def recv_side_effect():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise zmq.Again()
        import json as _json
        return _json.dumps({"timestamp": 0.0, "features": [1.0, 2.0]})

    mock_socket = MagicMock()
    mock_socket.recv_string.side_effect = recv_side_effect

    _patched_connect(client, mock_socket)

    assert client.read_latest() == [1.0, 2.0]
    client.disconnect()
