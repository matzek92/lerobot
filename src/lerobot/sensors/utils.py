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

from .configs import SensorConfig
from .sensor import Sensor


def make_sensors_from_configs(sensor_configs: dict[str, SensorConfig]) -> dict[str, Sensor]:
    """Instantiate :py:class:`Sensor` objects from a mapping of name → config.

    This is the sensor-side analogue of
    :py:func:`lerobot.cameras.utils.make_cameras_from_configs`.

    Args:
        sensor_configs: Mapping from sensor name to a
            :py:class:`~lerobot.sensors.configs.SensorConfig` subclass instance.

    Returns:
        Mapping from sensor name to a concrete :py:class:`Sensor` instance.

    Raises:
        ValueError: If the config type is unknown and no generic factory can
            handle it.
    """
    sensors: dict[str, Sensor] = {}

    for key, cfg in sensor_configs.items():
        if cfg.type == "serial":
            from .serial.sensor_serial import SerialSensor

            sensors[key] = SerialSensor(cfg)
        else:
            raise ValueError(
                f"Unknown sensor type {cfg.type!r} for sensor {key!r}. "
                "Registered types: 'serial'."
            )

    return sensors
