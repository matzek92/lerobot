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
Simple ZMQ image server client for verifying that the ImageServer video stream
is working correctly.

Connects to a running :class:`ImageServer` over ZMQ, receives JPEG frames and
displays them in an OpenCV window using ``cv2.imshow``.  Press **q** or
**Escape** to quit.

Usage example::

    # On the robot / server machine:
    python -m lerobot.cameras.zmq.image_server --host 0.0.0.0 --port 5555

    # On the client / viewer machine:
    python -m lerobot.cameras.zmq.image_server_client --host 192.168.1.100 --port 5555

    # Or via the installed script:
    lerobot-image-server-client --host 192.168.1.100 --port 5555
"""

import argparse
import json
import logging
import time

import cv2
import numpy as np
import zmq

logger = logging.getLogger(__name__)


def run_client(
    host: str = "localhost",
    port: int = 5555,
    camera_name: str | None = None,
    timeout_ms: int = 5000,
    window_title: str = "ZMQ Image Server",
) -> None:
    """Connect to an :class:`ImageServer` and display frames with ``cv2.imshow``.

    Args:
        host: Hostname or IP address of the ImageServer (default: ``"localhost"``).
        port: ZMQ PUB port of the ImageServer (default: ``5555``).
        camera_name: Name of the camera stream to display.  If *None* or not
            present in the metadata, the first available camera is used.
        timeout_ms: Receive timeout in milliseconds before printing a warning
            (default: ``5000``).
        window_title: Title of the OpenCV display window (default:
            ``"ZMQ Image Server"``).
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.CONFLATE, True)
    socket.connect(f"tcp://{host}:{port}")

    logger.info(f"Connected to tcp://{host}:{port}. Press 'q' or Escape to quit.")

    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            try:
                parts = socket.recv_multipart()
            except zmq.Again:
                logger.warning(f"No frame received within {timeout_ms} ms, retrying…")
                continue

            # Decode frame --------------------------------------------------
            if len(parts) > 1:
                # Protocol v2: multipart binary JPEG
                meta = json.loads(parts[0].decode("utf-8"))
                cameras = meta.get("cameras", [])

                if camera_name is not None and camera_name in cameras:
                    idx = cameras.index(camera_name)
                elif cameras:
                    idx = 0
                else:
                    logger.warning("No cameras listed in server metadata, skipping frame.")
                    continue

                jpeg_bytes = parts[idx + 1]
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                active_camera = cameras[idx] if cameras else "unknown"
            else:
                # Legacy protocol v1: single-part JSON / base64
                import base64

                data = json.loads(parts[0].decode("utf-8"))
                images = data.get("images", {})

                if not images:
                    logger.warning("No images in legacy message, skipping frame.")
                    continue

                if camera_name is not None and camera_name in images:
                    img_b64 = images[camera_name]
                    active_camera = camera_name
                else:
                    active_camera, img_b64 = next(iter(images.items()))

                img_bytes = base64.b64decode(img_b64)
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                logger.warning("Failed to decode image, skipping frame.")
                continue

            # FPS overlay ---------------------------------------------------
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
                logger.debug(f"Camera '{active_camera}' – FPS: {fps:.1f}")
            else:
                fps = frame_count / max(elapsed, 1e-6)

            cv2.putText(
                frame,
                f"{active_camera}  {fps:.1f} FPS",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Escape
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        socket.close()
        context.term()
        logger.info("Client stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simple ZMQ image server client – connects to a running ImageServer "
            "and displays the video stream using cv2.imshow. Press 'q' or Escape to quit."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Hostname or IP address of the ImageServer (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ PUB port of the ImageServer (default: 5555)",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=None,
        help=(
            "Name of the camera stream to display. "
            "If omitted, the first available camera in the server metadata is used."
        ),
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="Receive timeout in milliseconds before printing a warning (default: 5000)",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default="ZMQ Image Server",
        help="Title of the OpenCV display window (default: 'ZMQ Image Server')",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_client(
        host=args.host,
        port=args.port,
        camera_name=args.camera_name,
        timeout_ms=args.timeout_ms,
        window_title=args.window_title,
    )


if __name__ == "__main__":
    main()
