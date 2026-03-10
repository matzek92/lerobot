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

"""Object segmentation for guide-image highlighting.

Supported variants:
  - **sam2** – Segment Anything 2, interactive point-based segmentation.

The main entry point for interactive use is :func:`interactive_select`, which
opens an OpenCV window and lets the user click foreground / background points
while seeing a live mask preview.

Usage example::

    from lerobot.cameras.zmq.segment import SAM2Segmenter, interactive_select

    segmenter = SAM2Segmenter()                    # loads SAM2 model
    highlighted, mask = interactive_select(frame_bgr, segmenter)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Highlighting utilities
# ---------------------------------------------------------------------------

def highlight_object(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    dim_factor: float = 0.3,
    outline_color: tuple[int, int, int] = (0, 255, 0),
    outline_thickness: int = 2,
) -> np.ndarray:
    """Create a highlighted copy of *image* using a binary *mask*.

    The object region (``mask == 1``) stays at full brightness while the
    background is dimmed by *dim_factor*.  A coloured contour is drawn around
    the object boundary.

    Args:
        image: Input image (H, W, 3), any colour space (BGR recommended for
            direct use with OpenCV display / encoding).
        mask: Binary mask (H, W) with dtype ``uint8`` – ``1`` for the object,
            ``0`` for background.
        dim_factor: Multiplier applied to background pixels (0 = black,
            1 = unchanged).
        outline_color: BGR colour for the contour outline.
        outline_thickness: Pixel width of the contour outline.

    Returns:
        A copy of *image* with the highlighting applied.
    """
    result = image.copy()

    # Dim background
    bg = ~mask.astype(bool)
    result[bg] = (result[bg].astype(np.float32) * dim_factor).astype(np.uint8)

    # Draw object contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(result, contours, -1, outline_color, outline_thickness)

    return result


# ---------------------------------------------------------------------------
# Abstract segmenter
# ---------------------------------------------------------------------------

class BaseSegmenter(ABC):
    """Abstract base class for object segmenters."""

    @abstractmethod
    def segment(
        self,
        image_bgr: np.ndarray,
        points: list[list[int]],
        labels: list[int],
    ) -> np.ndarray:
        """Return a binary mask (H, W, uint8) for the selected object.

        Args:
            image_bgr: BGR image as numpy array (H, W, 3).
            points: List of ``[x, y]`` pixel coordinates.
            labels: Per-point label – ``1`` = foreground, ``0`` = background.

        Returns:
            Binary mask of shape (H, W) with dtype ``uint8``.
        """
        ...


# ---------------------------------------------------------------------------
# Variant 1 – Segment Anything 2
# ---------------------------------------------------------------------------

class SAM2Segmenter(BaseSegmenter):
    """Segment Anything 2 (SAM2) segmenter using point prompts.

    The model is loaded from the HuggingFace Hub on first instantiation.

    Args:
        model_id: HuggingFace model identifier
            (default: ``"facebook/sam2.1-hiera-small"``).
        device: Torch device string.  Defaults to ``"cuda"`` when available,
            otherwise ``"cpu"``.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-small",
        device: str | None = None,
    ):
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SAM2 is required for Variant 1 segmentation.  "
                "Install it with:  pip install sam-2"
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading SAM2 model '%s' on %s …", model_id, device)
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id)
        self.predictor.model.to(device)
        self._current_image_id: int | None = None
        logger.info("SAM2 model loaded.")

    # -- BaseSegmenter interface -------------------------------------------

    def segment(
        self,
        image_bgr: np.ndarray,
        points: list[list[int]],
        labels: list[int],
    ) -> np.ndarray:
        """Segment using SAM2 with point prompts.

        Internally converts BGR → RGB for the model and caches the image
        embedding so that subsequent calls with different points on the same
        image are fast.
        """
        # Re-encode the image only when the underlying array changes.
        img_id = id(image_bgr)
        if img_id != self._current_image_id:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
            self._current_image_id = img_id

        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(points, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            multimask_output=True,
        )

        # Return the mask with the highest confidence score.
        best_idx = int(scores.argmax())
        return masks[best_idx].astype(np.uint8)


# ---------------------------------------------------------------------------
# Interactive point selection
# ---------------------------------------------------------------------------

def interactive_select(
    image_bgr: np.ndarray,
    segmenter: BaseSegmenter,
    window_name: str = "Select Object",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Open an OpenCV window for interactive point selection with live preview.

    Controls:
      - **Left click** – add foreground point (green dot)
      - **Right click** – add background point (red dot)
      - **Z** – undo last point
      - **Enter** – confirm selection
      - **Escape** – cancel

    Args:
        image_bgr: BGR image (H, W, 3) – typically the latest live frame.
        segmenter: A :class:`BaseSegmenter` instance (e.g. :class:`SAM2Segmenter`).
        window_name: Title of the OpenCV window.

    Returns:
        ``(highlighted_bgr, mask)`` on success, or ``(None, None)`` if the
        user cancelled.
    """
    points: list[list[int]] = []
    labels: list[int] = []
    current_mask: np.ndarray | None = None

    def _update_mask() -> None:
        nonlocal current_mask
        if points:
            try:
                current_mask = segmenter.segment(image_bgr, points, labels)
            except Exception:
                logger.exception("Segmentation failed")
                current_mask = None
        else:
            current_mask = None

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            labels.append(1)
            _update_mask()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y])
            labels.append(0)
            _update_mask()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _on_mouse)

    instructions = (
        "L-click: foreground | R-click: background | "
        "Z: undo | Enter: confirm | Esc: cancel"
    )

    result: tuple[np.ndarray | None, np.ndarray | None] = (None, None)

    try:
        while True:
            # Build display frame
            if current_mask is not None:
                display = highlight_object(image_bgr, current_mask)
            else:
                display = image_bgr.copy()

            # Draw points
            for pt, lbl in zip(points, labels):
                colour = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                cv2.circle(display, (pt[0], pt[1]), 6, colour, -1)
                cv2.circle(display, (pt[0], pt[1]), 7, (255, 255, 255), 1)

            # Instructions bar
            cv2.putText(
                display, instructions,
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # Enter – confirm
                if current_mask is not None:
                    highlighted = highlight_object(image_bgr, current_mask)
                    result = (highlighted, current_mask)
                break
            elif key == 27:  # Escape – cancel
                break
            elif key in (ord("z"), ord("Z")):
                if points:
                    points.pop()
                    labels.pop()
                    _update_mask()
    finally:
        cv2.destroyWindow(window_name)

    return result
