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

import abc

from .configs import SensorConfig


class Sensor(abc.ABC):
    """Abstract base class for sensor implementations.

    Defines a standard interface for sensor operations across different
    backends (serial, network, etc.).  The pattern mirrors
    :py:class:`lerobot.cameras.camera.Camera` so robots treat sensors and
    cameras symmetrically.

    Core lifecycle::

        sensor.connect()      # open device, optionally wait for first data
        values = sensor.read_latest()  # non-blocking, returns zeros if no data yet
        sensor.disconnect()   # release resources

    Sensors can also be used as context managers::

        with SerialSensor(config) as sensor:
            values = sensor.read_latest()

    Attributes:
        feature_dim: Number of scalar values produced per reading.
    """

    def __init__(self, config: SensorConfig) -> None:
        self.feature_dim: int = config.feature_dim

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Sensor":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def __del__(self) -> None:
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """``True`` if the sensor is open and ready to produce readings."""

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """Open the sensor device and start producing readings.

        Args:
            warmup: If ``True`` (default) and :py:attr:`SensorConfig.warmup_s`
                is greater than zero, waits up to ``warmup_s`` seconds for the
                first valid reading before returning.  The behaviour when no
                reading arrives is controlled by
                :py:attr:`SensorConfig.optional`.
                If ``False``, the warmup check is skipped regardless of the
                configured ``warmup_s``.
        """

    @abc.abstractmethod
    def read_latest(self) -> list[float]:
        """Return the most recent reading as a list of floats.

        This is non-blocking; when no reading has been received yet (or the
        device has not connected) it returns a list of zeros of length
        :py:attr:`feature_dim`.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the device and release all resources."""
