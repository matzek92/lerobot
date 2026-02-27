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

from dataclasses import dataclass

from ..configs import SensorConfig

__all__ = ["SerialSensorConfig"]


@SensorConfig.register_subclass("serial")
@dataclass
class SerialSensorConfig(SensorConfig):
    """Configuration for a serial-port sensor.

    The device must emit newline-terminated UTF-8 text at **at least 30 Hz**.
    Two line formats are accepted:

    * **CSV / space-separated floats**::

          1.0,2.0,3.0\\n
          1.0 2.0 3.0\\n

    * **JSON** (same schema as the ZMQ sensor)::

          {"timestamp": 1234.5, "features": [1.0, 2.0, 3.0]}\\n

    Inherits ``feature_dim``, ``optional``, and ``warmup_s`` from
    :py:class:`~lerobot.sensors.configs.SensorConfig`.

    Attributes:
        port: Serial device path, e.g. ``"/dev/ttyUSB0"`` or ``"COM3"``.
        baudrate: Communication speed.  Must match the device's setting.
        read_timeout_s: Per-line read timeout passed to ``serial.Serial``.
            Keep it at or below 1/30 s so the background thread stays
            responsive.
    """

    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    read_timeout_s: float = 0.033  # ≈ 1 frame at 30 Hz
