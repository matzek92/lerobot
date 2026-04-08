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
Streams Intel RealSense camera images over ZMQ using a multipart binary protocol.

Four image streams are published per connected RealSense camera:

  - **{name}_rgb**      – color (RGB) stream, JPEG encoded.
  - **{name}_ir_left**  – left infrared stream (grayscale→BGR), JPEG encoded.
  - **{name}_ir_right** – right infrared stream (grayscale→BGR), JPEG encoded.
  - **{name}_depth**    – depth stream with 16-bit depth values encoded losslessly
                          in two color channels (PNG): B = low byte, G = high byte,
                          R = 0.  Recover depth via ``depth = B + G.astype(uint16) * 256``.

Protocol (version 2):
  - Part 0: UTF-8 JSON metadata::

        {
          "timestamps": {"<stream_name>": float, ...},
          "cameras":    ["<stream_name>", ...],
          "encoding":   {"<stream_name>": "jpeg" | "png", ...},
          "protocol_version": 2
        }

  - Parts 1..N: raw JPEG or PNG bytes for each stream listed in ``cameras``.

The server uses ``socket.send_multipart([meta_bytes, img1, img2, ...])``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque

import cv2
import numpy as np
import zmq

try:
    import pyrealsense2 as rs  # type: ignore
except Exception as e:
    logging.warning(f"Could not import pyrealsense2: {e}")

logger = logging.getLogger(__name__)


# ── Encoding helpers ──────────────────────────────────────────────────────────


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR/grayscale image to raw JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes()


def encode_png(image: np.ndarray) -> bytes:
    """Encode an image to PNG bytes (lossless)."""
    _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


def depth_to_color_channels(depth: np.ndarray) -> np.ndarray:
    """Encode a 16-bit depth array into a 3-channel BGR image.

    Encoding::

        B channel = depth & 0xFF          (low byte)
        G channel = (depth >> 8) & 0xFF   (high byte)
        R channel = 0                     (unused)

    Recovery::

        depth = B.astype(np.uint16) + G.astype(np.uint16) * 256

    Args:
        depth: 2-D ``np.uint16`` array of depth values (millimetres for most
               RealSense presets).

    Returns:
        3-channel ``np.uint8`` BGR image with the depth split across B and G.
    """
    low_byte = (depth & 0xFF).astype(np.uint8)
    high_byte = ((depth >> 8) & 0xFF).astype(np.uint8)
    zeros = np.zeros_like(low_byte)
    # Stack as BGR: B=low, G=high, R=zeros
    return np.stack([low_byte, high_byte, zeros], axis=-1)


def grayscale_to_bgr(gray: np.ndarray) -> np.ndarray:
    """Convert a single-channel grayscale image to 3-channel BGR."""
    if gray.dtype != np.uint8:
        # Normalise to 8-bit for JPEG compatibility
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray_norm = gray
    return cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR)


# ── Server class ──────────────────────────────────────────────────────────────


class RealSenseImageServer:
    """ZMQ publisher that streams four image feeds per Intel RealSense camera.

    The four streams per camera are:

    * ``{name}_rgb``      – color frame (JPEG)
    * ``{name}_ir_left``  – left IR frame converted to BGR (JPEG)
    * ``{name}_ir_right`` – right IR frame converted to BGR (JPEG)
    * ``{name}_depth``    – depth frame with bytes split across B/G channels (PNG)

    Args:
        serial_number: RealSense serial number.  Pass ``""`` to use the first
            available device.
        name: Human-readable prefix used in the stream names (default
            ``"realsense"``).
        fps: Capture frame rate (default 30).
        width: Stream width in pixels (default 640).
        height: Stream height in pixels (default 480).
        host: ZMQ bind address (default ``"*"`` for all interfaces).
        port: ZMQ PUB port (default 5555).
        jpeg_quality: JPEG quality for colour/IR streams (1–100, default 80).
        show_preview: Open a local OpenCV preview window (default ``False``).
    """

    def __init__(
        self,
        serial_number: str = "",
        name: str = "realsense",
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        host: str = "*",
        port: int = 5555,
        jpeg_quality: int = 80,
        show_preview: bool = False,
    ) -> None:
        self.name = name
        self.fps = fps
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.show_preview = show_preview

        # ── RealSense pipeline ────────────────────────────────────────────────
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        if serial_number:
            rs.config.enable_device(rs_config, serial_number)

        rs_config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        rs_config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        rs_config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = self.pipeline.start(rs_config)
        logger.info(f"RealSense pipeline started ({width}x{height} @ {fps} FPS)")

        # Retrieve actual resolution from the active color profile
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.actual_width = color_profile.width()
        self.actual_height = color_profile.height()
        logger.info(f"Active resolution: {self.actual_width}x{self.actual_height}")

        # Stream names published over ZMQ
        self.stream_rgb = f"{name}_rgb"
        self.stream_ir_left = f"{name}_ir_left"
        self.stream_ir_right = f"{name}_ir_right"
        self.stream_depth = f"{name}_depth"

        # ── ZMQ PUB socket ────────────────────────────────────────────────────
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
        self.socket.setsockopt(zmq.LINGER, 0)
        bind_host = host
        self.socket.bind(f"tcp://{bind_host}:{port}")

        server_addr = "localhost" if host == "*" else host
        logger.info(f"RealSenseImageServer publishing on tcp://{server_addr}:{port}")

        # Give the socket time to bind properly
        time.sleep(0.5)

        self._print_zmq_config(server_addr, port)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _print_zmq_config(self, server_addr: str, port: int) -> None:
        """Print stream names and a sample lerobot-record config snippet."""
        streams = [
            self.stream_rgb,
            self.stream_ir_left,
            self.stream_ir_right,
            self.stream_depth,
        ]
        cam_cfgs = []
        for s in streams:
            cam_cfgs.append(
                f"{s}:{{type: zmq, server_address: {server_addr}, port: {port}, camera_name: {s}}}"
            )
        full_cfg = "{ " + ", ".join(cam_cfgs) + " }"

        print("\n" + "=" * 80)
        print("RealSense ZMQ Image Server – available streams:")
        for s in streams:
            print(f"  • {s}")
        print("\nTo use with lerobot-record, add this to your command:")
        print(f'  --robot.cameras="{full_cfg}"')
        print("=" * 80 + "\n")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Capture frames from the RealSense device and publish over ZMQ."""
        frame_count = 0
        frame_times: deque[float] = deque(maxlen=60)

        if self.show_preview:
            cv2.namedWindow("RealSense Preview", cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                t0 = time.time()

                # ── Capture frameset ──────────────────────────────────────────
                success, frameset = self.pipeline.try_wait_for_frames(timeout_ms=5000)
                if not success or frameset is None:
                    logger.warning("Failed to get frameset, skipping tick")
                    continue

                # ── Decode individual streams ─────────────────────────────────
                color_frame = frameset.get_color_frame()
                ir_left_frame = frameset.get_infrared_frame(1)
                ir_right_frame = frameset.get_infrared_frame(2)
                depth_frame = frameset.get_depth_frame()

                if not color_frame or not ir_left_frame or not ir_right_frame or not depth_frame:
                    logger.warning("Incomplete frameset, skipping tick")
                    continue

                # RGB: RealSense gives RGB8, convert to BGR for OpenCV/JPEG
                rgb_np = np.asanyarray(color_frame.get_data())  # H×W×3 RGB uint8
                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

                ir_left_np = np.asanyarray(ir_left_frame.get_data())   # H×W uint8
                ir_right_np = np.asanyarray(ir_right_frame.get_data())  # H×W uint8

                depth_np = np.asanyarray(depth_frame.get_data())        # H×W uint16

                # ── Encode streams ────────────────────────────────────────────
                jpeg_rgb = encode_jpeg(bgr_np, self.jpeg_quality)
                jpeg_ir_left = encode_jpeg(grayscale_to_bgr(ir_left_np), self.jpeg_quality)
                jpeg_ir_right = encode_jpeg(grayscale_to_bgr(ir_right_np), self.jpeg_quality)
                png_depth = encode_png(depth_to_color_channels(depth_np))

                # ── Build multipart ZMQ message ───────────────────────────────
                now = time.time()
                camera_names = [
                    self.stream_rgb,
                    self.stream_ir_left,
                    self.stream_ir_right,
                    self.stream_depth,
                ]
                timestamps = {n: now for n in camera_names}
                encoding = {
                    self.stream_rgb: "jpeg",
                    self.stream_ir_left: "jpeg",
                    self.stream_ir_right: "jpeg",
                    self.stream_depth: "png",
                }
                meta = {
                    "timestamps": timestamps,
                    "cameras": camera_names,
                    "encoding": encoding,
                    "protocol_version": 2,
                }
                parts = [
                    json.dumps(meta).encode("utf-8"),
                    jpeg_rgb,
                    jpeg_ir_left,
                    jpeg_ir_right,
                    png_depth,
                ]

                try:
                    self.socket.send_multipart(parts, zmq.NOBLOCK)
                except zmq.Again:
                    logger.warning("Send buffer full, skipping frame")
                except Exception as e:
                    logger.error(f"Error sending frame: {e}")

                frame_count += 1
                frame_times.append(time.time() - t0)

                if frame_count % 60 == 0:
                    logger.debug(f"FPS: {len(frame_times) / sum(frame_times):.1f}")

                # ── Optional live preview ─────────────────────────────────────
                if self.show_preview:
                    fps_text = (
                        f"{len(frame_times) / max(sum(frame_times), 1e-6):.1f} FPS"
                    )
                    h, w = bgr_np.shape[:2]

                    # Resize helper
                    def _thumb(img: np.ndarray) -> np.ndarray:
                        return cv2.resize(img, (w // 2, h // 2))

                    depth_vis = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_np, alpha=0.03),
                        cv2.COLORMAP_JET,
                    )
                    top = np.hstack([_thumb(bgr_np), _thumb(grayscale_to_bgr(ir_left_np))])
                    bot = np.hstack([_thumb(grayscale_to_bgr(ir_right_np)), _thumb(depth_vis)])
                    canvas = np.vstack([top, bot])
                    cv2.putText(
                        canvas, fps_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                    cv2.imshow("RealSense Preview", canvas)

                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        logger.info("Preview window closed by user.")
                        break

                elapsed = time.time() - t0
                sleep = (1.0 / self.fps) - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            pass
        finally:
            if self.show_preview:
                cv2.destroyAllWindows()
            self.pipeline.stop()
            self.socket.close()
            self.context.term()
            logger.info("RealSenseImageServer stopped.")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream Intel RealSense camera images over ZMQ. "
            "Publishes four streams: RGB, IR left, IR right and depth "
            "(depth values encoded in color channels)."
        )
    )
    parser.add_argument(
        "--serial-number",
        type=str,
        default="",
        help="RealSense device serial number. Leave empty to use the first device found.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="realsense",
        help="Stream name prefix (default: realsense). "
        "Streams will be {name}_rgb, {name}_ir_left, {name}_ir_right, {name}_depth.",
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
        help="ZMQ PUB socket port (default: 5555)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for RGB and IR streams, 1-100 (default: 80)",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show a live 2×2 preview window on the server (press 'q' to quit)",
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

    RealSenseImageServer(
        serial_number=args.serial_number,
        name=args.name,
        fps=args.fps,
        width=args.width,
        height=args.height,
        host=args.host,
        port=args.port,
        jpeg_quality=args.jpeg_quality,
        show_preview=args.show_preview,
    ).run()


if __name__ == "__main__":
    main()
