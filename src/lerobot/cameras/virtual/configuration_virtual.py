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

from dataclasses import dataclass

from ..configs import CameraConfig

__all__ = ["VirtualCameraConfig"]


@CameraConfig.register_subclass("virtual")
@dataclass
class VirtualCameraConfig(CameraConfig):
    """Configuration class for a virtual camera that wraps an existing camera.

    A virtual camera reads frames from another (source) camera identified by
    ``camera_key``.  It can optionally down-sample the frame rate and/or resize
    the output frames to different dimensions, making it useful for producing
    lower-resolution or lower-FPS views of the same physical camera stream.

    Example configurations:
    ```python
    # Half-resolution view at half the source FPS
    VirtualCameraConfig(camera_key="top_camera", fps=15, width=320, height=240)

    # Same resolution, but throttled to 10 FPS
    VirtualCameraConfig(camera_key="wrist_camera", fps=10)
    ```

    Attributes:
        camera_key: The key of the source camera in the cameras dict.
        fps: Desired output frames per second.  Must be ``<= source fps``.
             If ``None``, the source camera's FPS is used.
        width: Desired output frame width in pixels.  If ``None``, the source
               camera's width is used.
        height: Desired output frame height in pixels.  If ``None``, the source
                camera's height is used.
    """

    camera_key: str = ""

    def __post_init__(self) -> None:
        if not self.camera_key:
            raise ValueError(
                "`camera_key` must be set to the key of an existing camera in the cameras dict."
            )
