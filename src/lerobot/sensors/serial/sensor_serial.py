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

import json
import logging
import time
from threading import Event, Lock, Thread

from ..sensor import Sensor
from .configuration_serial import SerialSensorConfig

logger = logging.getLogger(__name__)

__all__ = ["SerialSensor", "parse_serial_line"]


def parse_serial_line(line: bytes, feature_dim: int) -> list[float] | None:
    """Parse a single serial line into a list of floats.

    Accepts two formats:

    * **JSON**: ``{"features": [1.0, 2.0, 3.0]}\\n``
    * **CSV / whitespace-separated**: ``1.0,2.0,3.0\\n`` or ``1.0 2.0 3.0\\n``

    Returns ``None`` when the line is empty, malformed, or contains fewer
    than ``feature_dim`` values.
    """
    text = line.decode("utf-8", errors="ignore").strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            data = json.loads(text)
            features = data.get("features")
            if features is not None and len(features) >= feature_dim:
                return [float(v) for v in features[:feature_dim]]
        except json.JSONDecodeError:
            pass
        return None

    sep = "," if "," in text else None
    parts = text.split(sep)
    try:
        values = [float(p.strip()) for p in parts if p.strip()]
    except ValueError:
        return None
    if len(values) < feature_dim:
        return None
    return values[:feature_dim]


class SerialSensor(Sensor):
    """Reads feature vectors from a serial-port device in a background thread.

    The device must send newline-terminated lines at **at least 30 Hz**.  The
    background thread drains any backlogged bytes on every iteration so that
    :py:meth:`read_latest` always returns the freshest sample without blocking.

    Accepted line formats:

    * CSV / space-separated floats: ``1.0,2.0,3.0\\n``
    * JSON: ``{"timestamp": 0.0, "features": [1.0, 2.0, 3.0]}\\n``

    This class follows the same connection semantics as
    :py:class:`lerobot.cameras.camera.Camera`:

    * :py:meth:`connect` opens the device and optionally waits for the first
      valid reading (controlled by :py:attr:`SerialSensorConfig.warmup_s`).
    * :py:meth:`read_latest` is non-blocking and returns zeros until data
      arrives.
    * :py:meth:`disconnect` stops the background thread and closes the port.

    When :py:attr:`SerialSensorConfig.optional` is ``True`` (default), a
    missing or unresponsive device only emits a warning — the robot/pipeline
    keeps running with zeros.  Set ``optional=False`` to raise
    :exc:`ConnectionError` instead.

    Args:
        config: :py:class:`SerialSensorConfig` describing the connection.
    """

    def __init__(self, config: SerialSensorConfig) -> None:
        super().__init__(config)
        self.config = config
        self._lock: Lock = Lock()
        self._latest_features: list[float] | None = None
        self._stop_event: Event = Event()
        self._thread: Thread | None = None
        self._ser = None  # serial.Serial instance; None when not connected
        self._connected: bool = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        """Open the serial port and start the background read thread.

        Args:
            warmup: If ``True`` (default) and ``config.warmup_s > 0``, waits
                up to ``warmup_s`` seconds for the first valid line.
                If ``False``, skips the warmup check even if ``warmup_s > 0``.

        Raises:
            ConnectionError: When ``optional=False`` and the device is
                unavailable or does not respond within the warmup window.
        """
        import serial

        try:
            self._ser = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.read_timeout_s,
            )
            self._ser.reset_input_buffer()
        except serial.SerialException as exc:
            if self.config.optional:
                logger.warning(
                    f"SerialSensor: could not open {self.config.port!r}: {exc}. "
                    "Sensor is optional — continuing with zeros."
                )
                return
            raise ConnectionError(
                f"SerialSensor: could not open {self.config.port!r}: {exc}"
            ) from exc

        self._stop_event.clear()
        self._thread = Thread(
            target=self._read_loop,
            daemon=True,
            name=f"SerialSensor_{self.config.port}",
        )
        self._thread.start()
        self._connected = True
        logger.info(
            f"SerialSensor connected to {self.config.port!r} @ {self.config.baudrate} baud"
        )

        if warmup and self.config.warmup_s > 0:
            deadline = time.time() + self.config.warmup_s
            while time.time() < deadline:
                with self._lock:
                    if self._latest_features is not None:
                        break
                time.sleep(0.02)
            else:
                if self.config.optional:
                    logger.warning(
                        f"SerialSensor: no valid data from {self.config.port!r} within "
                        f"{self.config.warmup_s}s warmup. Sensor is optional — "
                        "continuing with zeros."
                    )
                else:
                    self.disconnect()
                    raise ConnectionError(
                        f"SerialSensor: no valid data from {self.config.port!r} within "
                        f"{self.config.warmup_s}s warmup. Check that the device is "
                        "connected and sending data. Set optional=True to continue "
                        "with zeros instead."
                    )

    def _read_loop(self) -> None:
        """Background thread: continuously drain and parse the serial buffer."""
        while not self._stop_event.is_set():
            try:
                if self._ser is None or not self._ser.is_open:
                    break
                # Drain buffer — keep only the newest complete line
                latest_line: bytes | None = None
                while self._ser.in_waiting > 0:
                    line = self._ser.readline()
                    if line:
                        latest_line = line

                if latest_line is None:
                    # Buffer empty — blocking readline with configured timeout
                    latest_line = self._ser.readline()

                if latest_line:
                    features = parse_serial_line(latest_line, self.config.feature_dim)
                    if features is not None:
                        with self._lock:
                            self._latest_features = features
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"SerialSensor read error: {exc}")

    def read_latest(self) -> list[float]:
        """Return the most recent feature vector.

        Returns zeros of length :py:attr:`~Sensor.feature_dim` until the
        first valid line has been received.
        """
        with self._lock:
            if self._latest_features is not None:
                return list(self._latest_features)
        return [0.0] * self.config.feature_dim

    def disconnect(self) -> None:
        """Stop the background thread and close the serial port."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
            self._ser = None
        self._connected = False
        logger.info(f"SerialSensor disconnected from {self.config.port!r}")
