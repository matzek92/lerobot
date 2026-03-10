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

"""
Streams camera images over ZMQ using a multipart binary protocol.
Two image streams are published:
  - **live camera** – real-time feed from an OpenCV camera.
  - **guide image** – a frozen snapshot of the scene captured at the start of
    each episode (triggered by a ``recording_start`` event).  Before the first
    event the guide image is a black placeholder.
Protocol (version 2):
  - Part 0: UTF-8 JSON metadata with keys:
      ``timestamps``        – mapping of camera_name -> float
      ``cameras``           – ordered list of camera names (matches binary parts below)
      ``encoding``          – ``"jpeg"``
      ``protocol_version``  – ``2``
  - Parts 1..N: raw JPEG bytes for each camera in the order given by ``cameras``.

The server uses ``socket.send_multipart([meta_bytes, jpeg1, jpeg2, ...])``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from collections import deque

import cv2
import numpy as np
import zmq

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

logger = logging.getLogger(__name__)

GUIDE_CAMERA_SUFFIX = "_guide"


def encode_image(image: np.ndarray, quality: int = 80) -> bytes:
    """Encode RGB image to raw JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes()


class ImageServer:
    def __init__(
        self,
        config: dict,
        host: str = "*",
        port: int = 5555,
        event_port: int | None = None,
        features_port: int | None = None,
        segmenter_type: str = "none",
        segmenter_model: str = "facebook/sam2.1-hiera-small",
    ):
        self.fps = config.get("fps", 30)
        self.cameras: dict[str, OpenCVCamera] = {}

        for name, cfg in config.get("cameras", {}).items():
            shape = cfg.get("shape", [480, 640])
            cam_config = OpenCVCameraConfig(
                index_or_path=cfg.get("device_id", 0),
                fps=self.fps,
                width=shape[1],
                height=shape[0],
                color_mode=ColorMode.RGB,
            )
            camera = OpenCVCamera(cam_config)
            camera.connect()
            self.cameras[name] = camera
            logger.info(f"Camera {name}: {shape[1]}x{shape[0]}")

        # --- Segmenter (optional, runs locally on the server) -------------
        self.segmenter = None
        if segmenter_type == "sam2":
            from lerobot.cameras.zmq.segment import SAM2Segmenter

            self.segmenter = SAM2Segmenter(model_id=segmenter_model)
            logger.info("SAM2 segmenter loaded on server.")

        # Guide frames stored as raw JPEG bytes.  Initialised to a black image.
        # Updated either by a plain ``recording_start`` (server captures its own
        # snapshot) or by a client-supplied highlighted frame (e.g. SAM2).
        self.guide_jpegs: dict[str, bytes] = {}
        for name, cfg in config.get("cameras", {}).items():
            shape = cfg.get("shape", [480, 640])
            black = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
            self.guide_jpegs[name] = encode_image(black)
        self._pending_snapshot = False

        bind_host = host

        # ZMQ PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        # self.socket.setsockopt(zmq.CONFLATE, 1)  # Keep only latest message
        self.socket.bind(f"tcp://{bind_host}:{port}")

        # Optional ZMQ PULL socket for receiving recording event notifications
        self.event_socket: zmq.Socket | None = None
        if event_port is not None:
            self.event_socket = self.context.socket(zmq.PULL)
            self.event_socket.setsockopt(zmq.LINGER, 0)
            self.event_socket.bind(f"tcp://{bind_host}:{event_port}")
            logger.info(f"ImageServer event listener on port {event_port}")

        # Optional ZMQ PULL socket for receiving robot features and sensor readings
        self.features_socket: zmq.Socket | None = None
        self.latest_features: dict | None = None
        if features_port is not None:
            self.features_socket = self.context.socket(zmq.PULL)
            self.features_socket.setsockopt(zmq.LINGER, 0)
            self.features_socket.bind(f"tcp://{bind_host}:{features_port}")
            logger.info(f"ImageServer features listener on port {features_port}")

        logger.info(f"ImageServer running on {bind_host}:{port}")

    def _reset_guide_jpegs(self) -> None:
        """Reset all guide frames to black placeholders."""
        for name, cam in self.cameras.items():
            h = cam.config.height
            w = cam.config.width
            black = np.zeros((h, w, 3), dtype=np.uint8)
            self.guide_jpegs[name] = encode_image(black)

    def _handle_events(self, live_frames: dict[str, np.ndarray] | None = None) -> None:
        """Process pending event messages (non-blocking).

        On ``recording_start``:
          - If a segmenter is configured, the server opens an interactive
            OpenCV window so the user can select the target object.  The
            highlighted result becomes the guide frame.
          - Otherwise a plain snapshot of the current live frame is used.

        On ``recording_stop`` / ``episode_end`` the guide frames are reset
        to black.
        """
        if self.event_socket is None:
            return
        while True:
            try:
                parts = self.event_socket.recv_multipart(zmq.NOBLOCK)
                data = json.loads(parts[0].decode("utf-8"))
                event_type = data.get("event", "unknown")
                logger.info(f"Received recording event: {event_type}")

                if event_type == "recording_start":
                    if live_frames and self.segmenter is not None:
                        self._interactive_guide(live_frames)
                    elif live_frames:
                        # Plain snapshot
                        for name, frame in live_frames.items():
                            self.guide_jpegs[name] = encode_image(frame)
                        logger.info("Guide-image snapshot captured.")
                    else:
                        self._pending_snapshot = True
                elif event_type in ("recording_stop", "episode_end"):
                    self._reset_guide_jpegs()
                    logger.info("Guide images reset to black.")
            except zmq.Again:
                break
            except Exception as e:
                logger.warning(f"Error handling recording event: {e}")
                break

    def _interactive_guide(self, live_frames: dict[str, np.ndarray]) -> None:
        """Open an interactive segmentation window for each camera.

        Blocks the main loop until the user confirms or cancels.
        """
        from lerobot.cameras.zmq.segment import highlight_object, interactive_select

        for name, frame_rgb in live_frames.items():
            # OpenCV windows expect BGR
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            logger.info(f"Opening object selection for camera '{name}' …")

            highlighted_bgr, mask = interactive_select(
                frame_bgr, self.segmenter,
                window_name=f"Select Object – {name}",
            )

            if highlighted_bgr is not None:
                # Convert back to RGB for encoding
                highlighted_rgb = cv2.cvtColor(highlighted_bgr, cv2.COLOR_BGR2RGB)
                self.guide_jpegs[name] = encode_image(highlighted_rgb, quality=90)
                logger.info(f"Guide frame for '{name}' set (segmented).")
            else:
                # Cancelled – use plain snapshot
                self.guide_jpegs[name] = encode_image(frame_rgb)
                logger.info(f"Selection cancelled for '{name}', using plain snapshot.")

    def _handle_features(self) -> None:
        """Process any pending robot features and sensor readings (non-blocking)."""
        if self.features_socket is None:
            return
        while True:
            try:
                message = self.features_socket.recv_string(zmq.NOBLOCK)
                data = json.loads(message)
                self.latest_features = data.get("features")
                logger.debug("Received robot features")
            except zmq.Again:
                break
            except Exception as e:
                logger.warning(f"Error handling robot features: {e}")
                break

    def run(self):
        frame_count = 0
        frame_times = deque(maxlen=60)

        try:
            while True:
                t0 = time.time()

                # --- Capture live frames from all cameras ------------------
                live_frames: dict[str, np.ndarray] = {}
                for name, cam in self.cameras.items():
                    live_frames[name] = cam.read()  # Returns RGB

                # --- Handle incoming events (may open interactive window) --
                self._handle_events(live_frames)
                self._handle_features()

                # --- Capture plain guide-image snapshot if requested -------
                if self._pending_snapshot:
                    for name, frame in live_frames.items():
                        self.guide_jpegs[name] = encode_image(frame)
                    self._pending_snapshot = False
                    logger.info("Guide-image snapshot captured.")

                # --- Build multipart message -------------------------------
                # Order: live cameras first, then guide cameras.
                camera_names: list[str] = []
                timestamps: dict[str, float] = {}
                jpeg_parts: list[bytes] = []

                now = time.time()
                for name in self.cameras:
                    camera_names.append(name)
                    timestamps[name] = now
                    jpeg_parts.append(encode_image(live_frames[name]))

                for name in self.cameras:
                    guide_name = name + GUIDE_CAMERA_SUFFIX
                    camera_names.append(guide_name)
                    timestamps[guide_name] = now
                    jpeg_parts.append(self.guide_jpegs[name])

                meta = {
                    "timestamps": timestamps,
                    "cameras": camera_names,
                    "encoding": "jpeg",
                    "protocol_version": 2,
                }
                parts = [json.dumps(meta).encode("utf-8")] + jpeg_parts

                # Send as multipart (suppress if buffer full)
                with contextlib.suppress(zmq.Again):
                    self.socket.send_multipart(parts, zmq.NOBLOCK)

                frame_count += 1
                frame_times.append(time.time() - t0)

                if frame_count % 60 == 0:
                    logger.debug(f"FPS: {len(frame_times) / sum(frame_times):.1f}")

                sleep = (1.0 / self.fps) - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            pass
        finally:
            for cam in self.cameras.values():
                cam.disconnect()
            if self.features_socket is not None:
                self.features_socket.close()
            if self.event_socket is not None:
                self.event_socket.close()
            self.socket.close()
            self.context.term()


def main():
    parser = argparse.ArgumentParser(
        description="Stream camera images over ZMQ using a multipart binary protocol.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index or path (default: 0)",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="head_camera",
        help="Name to assign to the camera stream (default: head_camera)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height in pixels (default: 480)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="*",
        help="IP address or hostname to bind to (default: * for all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ PUB socket port for publishing frames (default: 5555)",
    )
    parser.add_argument(
        "--event-port",
        type=int,
        default=None,
        help="ZMQ PULL socket port for receiving recording event notifications (optional)",
    )
    parser.add_argument(
        "--features-port",
        type=int,
        default=None,
        help="ZMQ PULL socket port for receiving robot features/sensor readings (optional)",
    )
    parser.add_argument(        "--segmenter",
        type=str,
        default="none",
        choices=["none", "sam2"],
        help=(
            "Segmentation variant for guide-image highlighting. "
            "'none' = plain snapshot, 'sam2' = interactive SAM2 selection "
            "(default: none)"
        ),
    )
    parser.add_argument(
        "--segmenter-model",
        type=str,
        default="facebook/sam2.1-hiera-small",
        help=(
            "HuggingFace model ID for SAM2 segmenter. Only used when "
            "--segmenter=sam2 (default: facebook/sam2.1-hiera-small)"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    config = {
        "fps": args.fps,
        "cameras": {
            args.camera_name: {
                "device_id": args.camera_index,
                "shape": [args.height, args.width],
            }
        },
    }
    ImageServer(
        config,
        host=args.host,
        port=args.port,
        event_port=args.event_port,
        features_port=args.features_port,
        segmenter_type=args.segmenter,
        segmenter_model=args.segmenter_model,
    ).run()


if __name__ == "__main__":
    main()
