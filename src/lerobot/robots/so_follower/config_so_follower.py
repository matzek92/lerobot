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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True

    # Optional external analog sensor streaming over serial (e.g. Arduino on A0).
    sensor_enabled: bool = False
    sensor_port: str | None = None
    sensor_baud_rate: int = 115200
    sensor_timeout_s: float = 0.01
    # Which value to extract from the incoming serial line.
    # Supported: "live" (ADC_live), "mean10" (ADC_mean10), "dmean" (dMean).
    sensor_value_mode: str = "mean10"
    # Name of the extra observation scalar feature in the dataset/state vector.
    sensor_feature_name: str = "sensor.a0_mean"
    # Used when no valid serial sample is available yet.
    sensor_default_value: float = 0.0
    # If True, fail on serial connection/read setup issues. If False, continue without sensor.
    sensor_strict: bool = False


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass


SO100FollowerConfig = SOFollowerRobotConfig
SO101FollowerConfig = SOFollowerRobotConfig
