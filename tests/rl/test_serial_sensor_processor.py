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

"""Tests for SerialSensorConfig, SerialSensorClient and SerialObservationProcessorStep."""

import json
import logging
import time
from contextlib import contextmanager
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.rl.joint_observations_processor import (
    SerialObservationProcessorStep,
    SerialSensorClient,
    SerialSensorConfig,
    _parse_serial_line,
)
from lerobot.utils.constants import OBS_STATE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_serial(lines: list[bytes]) -> MagicMock:
    """Return a mock serial.Serial that delivers lines one at a time."""
    buf = BytesIO(b"\n".join(lines) + b"\n")
    mock_ser = MagicMock()
    mock_ser.is_open = True
    mock_ser.in_waiting = 0  # trigger blocking readline path

    def readline():
        return buf.readline()

    mock_ser.readline.side_effect = readline
    return mock_ser


@contextmanager
def _patched_serial(mock_ser):
    """Context manager that replaces serial.Serial with a mock."""
    with patch("serial.Serial", return_value=mock_ser):
        yield


# ---------------------------------------------------------------------------
# _parse_serial_line unit tests
# ---------------------------------------------------------------------------


def test_parse_csv_line():
    assert _parse_serial_line(b"1.0,2.0,3.0\n", 3) == [1.0, 2.0, 3.0]


def test_parse_space_separated_line():
    assert _parse_serial_line(b"1.0 2.0 3.0\n", 3) == [1.0, 2.0, 3.0]


def test_parse_json_line():
    msg = json.dumps({"timestamp": 1.0, "features": [4.0, 5.0]}) + "\n"
    assert _parse_serial_line(msg.encode(), 2) == [4.0, 5.0]


def test_parse_json_line_truncates_to_feature_dim():
    msg = json.dumps({"timestamp": 0.0, "features": [1.0, 2.0, 3.0, 4.0]}) + "\n"
    assert _parse_serial_line(msg.encode(), 2) == [1.0, 2.0]


def test_parse_csv_truncates_to_feature_dim():
    assert _parse_serial_line(b"1.0,2.0,3.0,4.0\n", 2) == [1.0, 2.0]


def test_parse_empty_line_returns_none():
    assert _parse_serial_line(b"\n", 3) is None
    assert _parse_serial_line(b"  \n", 3) is None


def test_parse_too_few_values_returns_none():
    assert _parse_serial_line(b"1.0,2.0\n", 3) is None


def test_parse_malformed_json_returns_none():
    assert _parse_serial_line(b"{bad json}\n", 2) is None


def test_parse_non_numeric_csv_returns_none():
    assert _parse_serial_line(b"a,b,c\n", 3) is None


def test_parse_json_missing_features_key_returns_none():
    msg = json.dumps({"timestamp": 0.0, "data": [1.0, 2.0]}) + "\n"
    assert _parse_serial_line(msg.encode(), 2) is None


# ---------------------------------------------------------------------------
# SerialSensorConfig
# ---------------------------------------------------------------------------


def test_serial_sensor_config_defaults():
    cfg = SerialSensorConfig()
    assert cfg.port == "/dev/ttyUSB0"
    assert cfg.baudrate == 115200
    assert cfg.feature_dim == 1
    assert cfg.optional is True
    assert cfg.warmup_s == 0.0


def test_serial_sensor_config_custom():
    cfg = SerialSensorConfig(port="/dev/ttyACM0", baudrate=9600, feature_dim=6, optional=False, warmup_s=1.0)
    assert cfg.port == "/dev/ttyACM0"
    assert cfg.baudrate == 9600
    assert cfg.feature_dim == 6
    assert cfg.optional is False
    assert cfg.warmup_s == 1.0


# ---------------------------------------------------------------------------
# SerialSensorClient
# ---------------------------------------------------------------------------


def test_client_read_latest_default_zeros():
    """Before any data is received, read_latest returns zeros."""
    cfg = SerialSensorConfig(feature_dim=3)
    client = SerialSensorClient(cfg)
    assert client.read_latest() == [0.0, 0.0, 0.0]


def test_client_read_latest_after_inject():
    """Directly setting _latest_features is returned by read_latest."""
    cfg = SerialSensorConfig(feature_dim=2)
    client = SerialSensorClient(cfg)
    with client._lock:
        client._latest_features = [7.0, 8.0]
    assert client.read_latest() == [7.0, 8.0]


def test_client_connect_and_read_csv(tmp_path):
    """Client connects, reads a CSV line and returns the values."""
    cfg = SerialSensorConfig(port="/dev/ttyUSB0", feature_dim=3)
    client = SerialSensorClient(cfg)

    mock_ser = _make_mock_serial([b"1.0,2.0,3.0"])
    with _patched_serial(mock_ser):
        client.connect()
        # Give the background thread a moment to read
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if client.read_latest() != [0.0, 0.0, 0.0]:
                break
            time.sleep(0.01)
        client.disconnect()

    assert client.read_latest() == [1.0, 2.0, 3.0]


def test_client_connect_and_read_json():
    """Client connects and parses a JSON line."""
    cfg = SerialSensorConfig(port="/dev/ttyUSB0", feature_dim=2)
    client = SerialSensorClient(cfg)

    msg = json.dumps({"timestamp": 0.0, "features": [9.5, 0.5]}).encode() + b"\n"
    mock_ser = _make_mock_serial([msg.rstrip(b"\n")])
    with _patched_serial(mock_ser):
        client.connect()
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if client.read_latest() != [0.0, 0.0]:
                break
            time.sleep(0.01)
        client.disconnect()

    assert client.read_latest() == [9.5, 0.5]


def test_client_optional_connect_continues_when_port_unavailable():
    """optional=True — SerialException during open logs a warning and continues."""
    import serial

    cfg = SerialSensorConfig(port="/dev/ttyUSB99", optional=True)
    client = SerialSensorClient(cfg)

    with patch("serial.Serial", side_effect=serial.SerialException("no such port")):
        client.connect()  # Should NOT raise

    # No thread started — read_latest must still return zeros
    assert client.read_latest() == [0.0]


def test_client_required_connect_raises_when_port_unavailable():
    """optional=False — SerialException during open raises ConnectionError."""
    import serial

    cfg = SerialSensorConfig(port="/dev/ttyUSB99", optional=False)
    client = SerialSensorClient(cfg)

    with patch("serial.Serial", side_effect=serial.SerialException("no such port")):
        with pytest.raises(ConnectionError, match="could not open"):
            client.connect()


def test_client_optional_warmup_warns_when_no_data(caplog):
    """optional=True + warmup_s>0 — warns but does not raise when device is silent."""
    cfg = SerialSensorConfig(port="/dev/ttyUSB0", feature_dim=2, optional=True, warmup_s=0.1)
    client = SerialSensorClient(cfg)

    # Device is open but never sends data (readline returns empty bytes)
    mock_ser = MagicMock()
    mock_ser.is_open = True
    mock_ser.in_waiting = 0
    mock_ser.readline.return_value = b""

    with _patched_serial(mock_ser):
        with caplog.at_level(logging.WARNING, logger="lerobot.rl.joint_observations_processor"):
            client.connect()

    client.disconnect()
    assert any("optional" in r.message for r in caplog.records)
    assert client.read_latest() == [0.0, 0.0]


def test_client_required_warmup_raises_when_no_data():
    """optional=False + warmup_s>0 — raises ConnectionError when device is silent."""
    cfg = SerialSensorConfig(port="/dev/ttyUSB0", feature_dim=2, optional=False, warmup_s=0.1)
    client = SerialSensorClient(cfg)

    mock_ser = MagicMock()
    mock_ser.is_open = True
    mock_ser.in_waiting = 0
    mock_ser.readline.return_value = b""

    with _patched_serial(mock_ser):
        with pytest.raises(ConnectionError, match="no valid data"):
            client.connect()


def test_client_required_warmup_succeeds_when_data_arrives():
    """optional=False + warmup_s>0 — succeeds when device sends data quickly."""
    cfg = SerialSensorConfig(port="/dev/ttyUSB0", feature_dim=2, optional=False, warmup_s=0.5)
    client = SerialSensorClient(cfg)

    call_count = {"n": 0}

    def readline():
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return b""
        return b"3.0,4.0\n"

    mock_ser = MagicMock()
    mock_ser.is_open = True
    mock_ser.in_waiting = 0
    mock_ser.readline.side_effect = readline

    with _patched_serial(mock_ser):
        client.connect()  # Should NOT raise

    assert client.read_latest() == [3.0, 4.0]
    client.disconnect()


# ---------------------------------------------------------------------------
# SerialObservationProcessorStep
# ---------------------------------------------------------------------------


def _make_mock_client(features: list[float]) -> SerialSensorClient:
    mock = MagicMock(spec=SerialSensorClient)
    mock.read_latest.return_value = features
    return mock


def test_serial_processor_no_clients():
    """With no configs the processor is a no-op."""
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = []
    step._clients = []

    state = torch.tensor([[1.0, 2.0]])
    obs = {OBS_STATE: state}
    torch.testing.assert_close(step.observation(obs)[OBS_STATE], state)


def test_serial_processor_appends_to_obs_state():
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = [SerialSensorConfig(feature_dim=2)]
    step._clients = [_make_mock_client([5.0, 6.0])]

    state = torch.tensor([[1.0, 2.0, 3.0]])
    result = step.observation({OBS_STATE: state})

    torch.testing.assert_close(result[OBS_STATE], torch.tensor([[1.0, 2.0, 3.0, 5.0, 6.0]]))


def test_serial_processor_creates_obs_state_if_missing():
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = [SerialSensorConfig(feature_dim=2)]
    step._clients = [_make_mock_client([7.0, 8.0])]

    result = step.observation({})
    torch.testing.assert_close(result[OBS_STATE], torch.tensor([[7.0, 8.0]]))


def test_serial_processor_multiple_clients():
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = [SerialSensorConfig(feature_dim=2), SerialSensorConfig(feature_dim=1)]
    step._clients = [_make_mock_client([1.0, 2.0]), _make_mock_client([9.0])]

    state = torch.tensor([[0.0]])
    result = step.observation({OBS_STATE: state})
    torch.testing.assert_close(result[OBS_STATE], torch.tensor([[0.0, 1.0, 2.0, 9.0]]))


def test_serial_processor_full_pipeline():
    from lerobot.processor.converters import identity_transition
    from lerobot.processor.pipeline import DataProcessorPipeline

    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = [SerialSensorConfig(feature_dim=2)]
    step._clients = [_make_mock_client([10.0, 20.0])]

    pipeline = DataProcessorPipeline(
        steps=[step],
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    state = torch.tensor([[0.5, 1.5]])
    transition = create_transition(observation={OBS_STATE: state})
    result = pipeline(transition)

    torch.testing.assert_close(
        result[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[0.5, 1.5, 10.0, 20.0]])
    )


# ---------------------------------------------------------------------------
# transform_features
# ---------------------------------------------------------------------------


def test_serial_transform_features_extends_obs_state():
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = [SerialSensorConfig(feature_dim=3), SerialSensorConfig(feature_dim=2)]
    step._clients = []

    original = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    features = {PipelineFeatureType.OBSERVATION: {OBS_STATE: original}}
    updated = step.transform_features(features)

    new_feat = updated[PipelineFeatureType.OBSERVATION][OBS_STATE]
    assert new_feat.shape == (11,)  # 6 + 3 + 2
    assert new_feat.type == FeatureType.STATE


def test_serial_transform_features_no_configs():
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = []
    step._clients = []

    original = PolicyFeature(type=FeatureType.STATE, shape=(4,))
    features = {PipelineFeatureType.OBSERVATION: {OBS_STATE: original}}
    assert step.transform_features(features)[PipelineFeatureType.OBSERVATION][OBS_STATE] == original


def test_serial_transform_features_missing_obs_state():
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = [SerialSensorConfig(feature_dim=2)]
    step._clients = []

    features = {PipelineFeatureType.OBSERVATION: {}}
    updated = step.transform_features(features)
    assert OBS_STATE not in updated[PipelineFeatureType.OBSERVATION]


# ---------------------------------------------------------------------------
# get_config / reset
# ---------------------------------------------------------------------------


def test_serial_processor_get_config():
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    cfg = SerialSensorConfig(port="/dev/ttyACM0", baudrate=9600, feature_dim=4, optional=False, warmup_s=2.0)
    step.serial_configs = [cfg]
    step._clients = []

    config = step.get_config()
    entry = config["serial_configs"][0]
    assert entry["port"] == "/dev/ttyACM0"
    assert entry["baudrate"] == 9600
    assert entry["feature_dim"] == 4
    assert entry["optional"] is False
    assert entry["warmup_s"] == 2.0


def test_serial_processor_reset_does_not_raise():
    step = SerialObservationProcessorStep.__new__(SerialObservationProcessorStep)
    step.serial_configs = []
    step._clients = []
    step.reset()
