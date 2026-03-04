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

Keyboard controls:
  - **q** / **Escape** – quit
  - **r** – send ``recording_start`` event
  - **s** – send ``recording_stop`` event
  - **e** – send ``episode_end`` event

Usage example::

    # On the robot / server machine:
    python -m lerobot.cameras.zmq.image_server --host 0.0.0.0 --port 5555 --event-port 5556

    # On the client / viewer machine:
    python -m lerobot.cameras.zmq.image_server_client --host 192.168.1.100 --port 5555 --event-port 5556
"""

import argparse
import json
import logging
import time

import cv2
import numpy as np
import zmq

logger = logging.getLogger(__name__)

# Key bindings for sending events
EVENT_KEY_MAP = {
    ord("r"): "recording_start",
    ord("s"): "recording_stop",
    ord("e"): "episode_end",
}


def _send_event(event_socket: zmq.Socket | None, event_type: str) -> None:
    """Send a JSON event message to the server via ZMQ PUSH socket."""
    if event_socket is None:
        logger.warning("No event socket configured – ignoring key press.")
        return
    payload = json.dumps({"event": event_type, "timestamp": time.time()})
    try:
        event_socket.send_string(payload, zmq.NOBLOCK)
        logger.info(f"Sent event: {event_type}")
    except zmq.Again:
        logger.warning(f"Event socket buffer full, could not send: {event_type}")


def run_client(
    host: str = "localhost",
    port: int = 5555,
    camera_name: str | None = None,
    timeout_ms: int = 5000,
    window_title: str = "ZMQ Image Server",
    event_port: int | None = None,
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
        event_port: ZMQ PUSH port for sending recording events to the server
            (default: *None* – no event sending).
    """
    context = zmq.Context()

    # SUB socket for receiving frames
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.connect(f"tcp://{host}:{port}")

    # Optional PUSH socket for sending events
    event_socket: zmq.Socket | None = None
    if event_port is not None:
        event_socket = context.socket(zmq.PUSH)
        event_socket.setsockopt(zmq.LINGER, 0)
        event_socket.setsockopt(zmq.SNDHWM, 10)
        event_socket.connect(f"tcp://{host}:{event_port}")
        logger.info(f"Event socket connected to tcp://{host}:{event_port}")
        logger.info("Keys: [r] recording_start  [s] recording_stop  [e] episode_end")

    logger.info(f"Connected to tcp://{host}:{port}. Press 'q' or Escape to quit.")

    frame_count = 0
    fps_start = time.time()
    recording = False

    try:
        while True:
            try:
                parts = socket.recv_multipart()
            except zmq.Again:
                logger.warning(f"No frame received within {timeout_ms} ms, retrying…")
                continue

            # Drain buffer: keep only the latest message
            while True:
                try:
                    newer = socket.recv_multipart(zmq.NOBLOCK)
                    parts = newer
                except zmq.Again:
                    break

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

            # Status overlay
            status_text = f"{active_camera}  {fps:.1f} FPS"
            if recording:
                status_text += "  [REC]"
            status_color = (0, 0, 255) if recording else (0, 255, 0)

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # Key hints when event port is configured
            if event_socket is not None:
                cv2.putText(
                    frame,
                    "[r] rec start  [s] rec stop  [e] episode end",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Escape
                break
            elif key in EVENT_KEY_MAP:
                event_type = EVENT_KEY_MAP[key]
                _send_event(event_socket, event_type)
                # Track recording state for overlay
                if event_type == "recording_start":
                    recording = True
                elif event_type == "recording_stop":
                    recording = False

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if event_socket is not None:
            event_socket.close()
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
        "--event-port",
        type=int,
        default=None,
        help=(
            "ZMQ PUSH port for sending recording events to the server. "
            "Keys: [r] recording_start, [s] recording_stop, [e] episode_end (optional)"
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
        event_port=args.event_port,
    )


if __name__ == "__main__":
    main()
