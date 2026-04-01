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
import threading
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
        show_preview: bool = False,
    ):
        self.fps = config.get("fps", 30)
        self.show_preview = show_preview
        self.cameras: dict[str, OpenCVCamera] = {}
        self.capture_threads: dict[str, CameraCaptureThread] = {}

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
        # NOTE: CONFLATE does NOT work with multipart messages - it can split them!
        # self.socket.setsockopt(zmq.CONFLATE, 1)
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
        
        # Print JSON configuration for lerobot-record CLI
        self._print_record_cli_config(host, port, event_port, features_port)
        
        # Give the socket time to bind properly before starting
        time.sleep(0.5)

    def _print_record_cli_config(self, host: str, port: int, event_port: int | None, features_port: int | None) -> None:
        """Print the JSON configuration string for use with lerobot-record CLI."""
        
        # Determine the server address to use (convert "*" to localhost)
        server_addr = "localhost" if host == "*" else host
        
        # Build camera configurations
        camera_configs = []
        for name, cam in self.cameras.items():
            # Live camera config
            cam_config_str = f"{name}:{{"
            cam_config_str += f"type: zmq, "
            cam_config_str += f"server_address: {server_addr}, "
            cam_config_str += f"port: {port}, "
            cam_config_str += f"camera_name: {name}, "
            cam_config_str += f"width: {cam.config.width}, "
            cam_config_str += f"height: {cam.config.height}, "
            cam_config_str += f"fps: {self.fps}"
            if event_port is not None:
                cam_config_str += f", event_port: {event_port}"
            if features_port is not None:
                cam_config_str += f", features_port: {features_port}"
            cam_config_str += "}"
            camera_configs.append(cam_config_str)
            
            # Guide camera config (if segmenter is active)
            if self.segmenter is not None:
                guide_name = name + GUIDE_CAMERA_SUFFIX
                guide_config_str = f"{guide_name}:{{"
                guide_config_str += f"type: zmq, "
                guide_config_str += f"server_address: {server_addr}, "
                guide_config_str += f"port: {port}, "
                guide_config_str += f"camera_name: {guide_name}, "
                guide_config_str += f"width: {cam.config.width}, "
                guide_config_str += f"height: {cam.config.height}, "
                guide_config_str += f"fps: {self.fps}"
                guide_config_str += "}"
                camera_configs.append(guide_config_str)
        
        # Combine all cameras
        full_config = "{ " + ", ".join(camera_configs) + " }"
        
        print("\n" + "="*80)
        print("To use this camera server with lerobot-record, add this to your command:")
        print("="*80)
        print(f'--robot.cameras="{full_config}"')
        if self.segmenter is not None:
            print("\nNote: Guide camera streams (*_guide) are included for SAM2 segmentation")
        print("="*80 + "\n")

    def _reset_guide_jpegs(self) -> None:
        """Reset all guide frames to black placeholders."""
        for name, cam in self.cameras.items():
            h = cam.config.height
            w = cam.config.width
            black = np.zeros((h, w, 3), dtype=np.uint8)
            self.guide_jpegs[name] = encode_image(black)

    def _handle_events(self, live_frames: dict[str, np.ndarray] | None = None) -> None:
        """Process pending event messages (non-blocking).

        On ``episode_end``:
          - If a segmenter is configured, the server opens an interactive
            OpenCV window so the user can select the target object for the
            next episode. The highlighted result becomes the guide frame.
          - Otherwise the guide frames are reset to black.

        On ``recording_start``:
          - If no guide frame has been set yet, capture a plain snapshot.

        On ``recording_stop`` the guide frames are reset to black.
        """
        if self.event_socket is None:
            return
        while True:
            try:
                parts = self.event_socket.recv_multipart(zmq.NOBLOCK)
                data = json.loads(parts[0].decode("utf-8"))
                event_type = data.get("event", "unknown")
                print(f"[EVENT] Received: {event_type}")
                print(f"[EVENT] Full data: {data}")
                logger.info(f"Received recording event: {event_type}")

                if event_type == "episode_end":
                    print(f"[EVENT] Processing episode_end - Episode finished")
                    logger.info("Episode finished, waiting for reset...")
                elif event_type == "reset_done":
                    print(f"[EVENT] Processing reset_done - segmenter={self.segmenter is not None}, live_frames={len(live_frames) if live_frames else 0}")
                    if live_frames and self.segmenter is not None:
                        # Open interactive segmentation after reset, before next episode
                        logger.info("Reset complete, starting interactive segmentation...")
                        self._interactive_guide(live_frames)
                    else:
                        # No segmenter - reset to black
                        self._reset_guide_jpegs()
                        logger.info("Guide images reset to black.")
                elif event_type == "recording_start":
                    print(f"[EVENT] Processing recording_start - segmenter={self.segmenter is not None}")
                    # Only capture snapshot if guide is not already set
                    if live_frames and not self.segmenter:
                        for name, frame in live_frames.items():
                            self.guide_jpegs[name] = encode_image(frame)
                        logger.info("Guide-image snapshot captured.")
                elif event_type == "recording_stop":
                    print(f"[EVENT] Processing recording_stop")
                    self._reset_guide_jpegs()
                    logger.info("Guide images reset to black.")
                else:
                    print(f"[EVENT] Unknown event type: {event_type}")
            except zmq.Again:
                break
            except Exception as e:
                print(f"[EVENT] Error handling recording event: {e}")
                logger.warning(f"Error handling recording event: {e}")
                break

    def _interactive_guide(self, live_frames: dict[str, np.ndarray]) -> None:
        """Open an interactive segmentation window for each camera.

        Blocks the main loop until the user confirms or cancels.
        """
        from lerobot.cameras.zmq.segment import interactive_select

        # Temporarily close preview window if it's open
        if self.show_preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        for name, frame_rgb in live_frames.items():
            # OpenCV windows expect BGR
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            logger.info(f"Opening object selection for camera '{name}' …")

            try:
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
            except Exception as e:
                logger.error(f"Error during interactive segmentation for '{name}': {e}")
                # Fallback: use plain snapshot
                self.guide_jpegs[name] = encode_image(frame_rgb)
                logger.info(f"Using plain snapshot for '{name}' due to error.")
        
        # Clean up segmentation windows
        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(1)
        time.sleep(0.5)
        logger.info("Segmentation complete.")

    def _handle_features(self):
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
        window_title = "ImageServer Preview" if self.show_preview else None
        
        # Pre-create preview window if needed
        if self.show_preview:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

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

                # Send as multipart
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

                # --- Live preview window (optional) -----------------------
                if self.show_preview and live_frames:
                    # Show all cameras with their guide frames
                    preview_frames = []
                    for name in sorted(self.cameras.keys()):
                        # Live frame
                        frame_rgb = live_frames[name]
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Add "LIVE" label and FPS overlay
                        fps_text = f"LIVE: {name}  {len(frame_times) / max(sum(frame_times), 1e-6):.1f} FPS"
                        cv2.putText(
                            frame_bgr, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                        )
                        preview_frames.append(frame_bgr)
                        
                        # Guide frame
                        guide_jpeg = self.guide_jpegs[name]
                        guide_img = cv2.imdecode(np.frombuffer(guide_jpeg, np.uint8), cv2.IMREAD_COLOR)
                        
                        if guide_img is not None:
                            # Resize guide to match live frame height if needed
                            h, w = frame_bgr.shape[:2]
                            gh, gw = guide_img.shape[:2]
                            if gh != h:
                                scale = h / gh
                                guide_img = cv2.resize(guide_img, (int(gw * scale), h))
                            
                            # Add "GUIDE" label and border
                            cv2.putText(
                                guide_img, f"GUIDE: {name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
                            )
                            cv2.rectangle(
                                guide_img, (0, 0),
                                (guide_img.shape[1] - 1, guide_img.shape[0] - 1),
                                (0, 200, 255), 3,
                            )
                            preview_frames.append(guide_img)
                    
                    # Combine all frames horizontally with separators
                    if len(preview_frames) == 1:
                        canvas = preview_frames[0]
                    else:
                        # Add separators between frames
                        h = preview_frames[0].shape[0]
                        sep = np.full((h, 2, 3), 255, dtype=np.uint8)
                        canvas = preview_frames[0]
                        for frame in preview_frames[1:]:
                            canvas = np.hstack([canvas, sep, frame])
                    
                    cv2.imshow(window_title, canvas)
                    
                    # Check for 'q' or Escape to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        logger.info("Preview window closed by user.")
                        break

                sleep = (1.0 / self.fps) - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            pass
        finally:
            if self.show_preview:
                cv2.destroyAllWindows()
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
        type=str,
        default="0",
        help="Camera device index (e.g., 0) or path (e.g., /dev/video0) (default: 0)",
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
        "--show-preview",
        action="store_true",
        help="Show live preview window of camera feed on the server (press 'q' to quit)",
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

    # Parse camera_index: convert to int if numeric, otherwise keep as path string
    try:
        camera_device = int(args.camera_index)
    except ValueError:
        camera_device = args.camera_index

    config = {
        "fps": args.fps,
        "cameras": {
            args.camera_name: {
                "device_id": camera_device,
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
        show_preview=args.show_preview,
    ).run()


if __name__ == "__main__":
    main()
