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

"""
VirtualCamera - A camera that reads frames from another camera and can output
them at a different resolution and/or a lower frame rate.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_virtual import VirtualCameraConfig

logger = logging.getLogger(__name__)

# Maximum number of consecutive frame read failures before raising an exception.
_MAX_CONSECUTIVE_FAILURES = 10

# Fallback FPS used when neither the virtual camera nor the source camera has an FPS set.
_DEFAULT_FALLBACK_FPS = 30

# Multiplier (in ms) applied to the source period to compute the maximum acceptable
# frame age when calling read_latest() on the source camera.
_SOURCE_FRAME_AGE_MULTIPLIER_MS = 2000.0

# Minimum acceptable frame age (ms) regardless of source FPS.
_MIN_FRAME_AGE_MS = 500

# Short delay (s) after spawning the background thread to let it start.
_THREAD_START_DELAY_S = 0.05


class VirtualCamera(Camera):
    """A camera that reads frames from another camera (the *source* camera).

    The virtual camera can optionally:

    * **Resize** output frames to a different ``(width, height)``.
    * **Throttle** the output frame rate to a lower ``fps`` value.

    It does **not** own the source camera's lifecycle — the caller is responsible
    for connecting and disconnecting the source camera.

    Example usage:
        ```python
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
        from lerobot.cameras.virtual import VirtualCamera, VirtualCameraConfig

        src_cfg = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
        src_cam = OpenCVCamera(src_cfg)

        virt_cfg = VirtualCameraConfig(camera_key="top", fps=10, width=320, height=240)
        virt_cam = VirtualCamera(virt_cfg, src_cam)

        src_cam.connect()
        virt_cam.connect()

        frame = virt_cam.read()   # 240×320 image at ≤10 FPS

        virt_cam.disconnect()
        src_cam.disconnect()
        ```
    """

    def __init__(self, config: VirtualCameraConfig, source_camera: Camera):
        """Initialise the VirtualCamera.

        Args:
            config: Virtual camera configuration.
            source_camera: The already-instantiated source camera whose frames
                           this virtual camera will consume.
        """
        super().__init__(config)

        self.config = config
        self.source_camera = source_camera

        # Threading resources
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self._connected: bool = False

    def __str__(self) -> str:
        return f"VirtualCamera({self.config.camera_key})"

    @property
    def is_connected(self) -> bool:
        """True when the background read thread is running."""
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Not applicable — virtual cameras are defined in configuration, not discovered."""
        raise NotImplementedError("Camera detection is not implemented for virtual cameras.")

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Start the virtual camera's background thread.

        The source camera **must** already be connected before calling this.

        Args:
            warmup: If ``True`` (default), waits until at least one transformed
                    frame has been captured before returning.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            RuntimeError: If the source camera is not connected.
        """
        if not self.source_camera.is_connected:
            raise RuntimeError(
                f"{self}: source camera '{self.config.camera_key}' must be connected before "
                "connecting the virtual camera."
            )

        # Inherit dimensions from the source camera when not specified
        if self.width is None:
            self.width = self.source_camera.width
        if self.height is None:
            self.height = self.source_camera.height

        # Inherit FPS from source when not specified
        if self.fps is None:
            self.fps = self.source_camera.fps

        self._start_read_thread()
        self._connected = True

        if warmup:
            # Wait for the first frame to be available
            timeout_ms = 5000.0
            if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
                self._connected = False
                self._stop_read_thread()
                raise RuntimeError(
                    f"{self}: timed out waiting for the first frame during warmup "
                    f"({timeout_ms:.0f} ms)."
                )

        logger.info(f"{self} connected.")

    def _transform_frame(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Resize *frame* to the configured output dimensions if necessary."""
        src_h, src_w = frame.shape[:2]
        if self.height is not None and self.width is not None:
            if src_h != self.height or src_w != self.width:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _read_loop(self) -> None:
        """Background thread: read from source, transform, throttle to target FPS."""
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialised.")

        period_s = (1.0 / self.fps) if self.fps else 0.0
        failure_count = 0

        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                # Use read_latest for non-blocking access to the most recent source frame.
                # Allow frames up to 2× the source period to be considered fresh.
                source_fps = self.source_camera.fps or self.fps or _DEFAULT_FALLBACK_FPS
                max_age_ms = max(int(_SOURCE_FRAME_AGE_MULTIPLIER_MS / source_fps), _MIN_FRAME_AGE_MS)
                raw_frame = self.source_camera.read_latest(max_age_ms=max_age_ms)
                frame = self._transform_frame(raw_frame)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= _MAX_CONSECUTIVE_FAILURES:
                    failure_count += 1
                    logger.warning(f"{self}: error reading source frame: {e}")
                else:
                    raise RuntimeError(f"{self}: exceeded maximum consecutive read failures.") from e

            # Throttle to the desired output FPS
            elapsed = time.perf_counter() - loop_start
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _start_read_thread(self) -> None:
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop,
            daemon=True,
            name=f"{self}_read_loop",
        )
        self.thread.start()
        time.sleep(_THREAD_START_DELAY_S)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """Capture and return one frame synchronously.

        Clears the *new-frame* flag and waits up to 10 s for a fresh frame.

        Returns:
            np.ndarray: Transformed frame ``(height, width, channels)``.

        Raises:
            DeviceNotConnectedError: If not connected.
            RuntimeError: If the background thread is not running.
        """
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self}: read thread is not running.")

        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the most recently captured transformed frame.

        Blocks up to *timeout_ms* if no new frame is available yet.

        Args:
            timeout_ms: Maximum wait time in milliseconds. Defaults to 200 ms.

        Returns:
            np.ndarray: Transformed frame ``(height, width, channels)``.

        Raises:
            DeviceNotConnectedError: If not connected.
            TimeoutError: If no frame is available within *timeout_ms*.
            RuntimeError: If the background thread is not running.
        """
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self}: read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"{self}: timed out waiting for a frame after {timeout_ms} ms."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self}: event was set but no frame is available.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent frame immediately (non-blocking peek).

        Args:
            max_age_ms: Maximum acceptable age of the frame in milliseconds.

        Returns:
            NDArray[Any]: The latest transformed frame.

        Raises:
            TimeoutError: If the latest frame is older than *max_age_ms*.
            DeviceNotConnectedError: If not connected.
            RuntimeError: If no frame has been captured yet.
        """
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self}: read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self}: no frame has been captured yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self}: latest frame is too old: {age_ms:.1f} ms "
                f"(max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        """Stop the background thread and release resources.

        Note: the source camera is **not** disconnected here — its lifecycle is
        managed externally.

        Raises:
            DeviceNotConnectedError: If not connected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self}: not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        self._connected = False

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
