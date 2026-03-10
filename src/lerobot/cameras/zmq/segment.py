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
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

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


def highlight_object_fadeout(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    fade_pixels: int = 16,
    min_brightness: float = 0.0,
) -> np.ndarray:
    """Create a copy of *image* with a gradient fade-to-black around the segmented region.

    The segmented region (``mask == 1``) is left completely unchanged.
    Starting at the mask boundary, pixel brightness fades linearly to
    *min_brightness* over *fade_pixels* distance.  Pixels beyond the fade
    zone are multiplied by *min_brightness* (0 = pure black).

    Args:
        image: Input image (H, W, 3), BGR recommended.
        mask: Binary mask (H, W) with dtype ``uint8`` – ``1`` for the object,
            ``0`` for background.
        fade_pixels: Number of pixels over which the brightness fades from
            1.0 (at the mask boundary) to *min_brightness*.  Larger values
            give a softer, wider gradient.
        min_brightness: Brightness multiplier for pixels at or beyond the
            fade distance (0.0 = black, 1.0 = unchanged).

    Returns:
        A copy of *image* with the fade-out applied.
    """
    mask_u8 = mask.astype(np.uint8)

    # Compute distance from each background pixel to the nearest mask pixel.
    # cv2.distanceTransform only works on the *zero* pixels, so we pass the
    # inverted mask (background = 0 inside distanceTransform's input means
    # "compute distance for these").  We invert: bg=255 → 0, fg=0 → 255,
    # but distanceTransform measures distance of 0-pixels to nearest non-zero.
    # So: pass (1 - mask) → fg pixels become 0, bg pixels become 1.
    # distanceTransform measures distance of 0-pixels — that's the fg, not
    # what we want.  Instead: pass mask itself as src; then distance is
    # computed for the 0-valued (background) pixels.
    dist = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, cv2.DIST_MASK_5)
    # dist[fg] == 0, dist[bg] == euclidean distance to nearest fg pixel

    # Build per-pixel brightness multiplier
    if fade_pixels > 0:
        # Linear fade: 1.0 at distance 0 → min_brightness at distance >= fade_pixels
        alpha = 1.0 - (1.0 - min_brightness) * np.clip(dist / fade_pixels, 0.0, 1.0)
    else:
        # No fade: foreground = 1.0, background = min_brightness
        alpha = np.where(mask_u8, 1.0, min_brightness)

    # Foreground pixels must stay fully unchanged
    alpha[mask_u8 == 1] = 1.0

    result = (image.astype(np.float32) * alpha[:, :, np.newaxis]).astype(np.uint8)
    return result


def highlight_object_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 180, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a copy of *image* with a semi-transparent colour overlay on the mask.

    Unlike :func:`highlight_object_fadeout` (intended for episode video streams),
    this variant is designed for **interactive selection**: the segmented region
    is tinted with a translucent colour and no contours are drawn, giving a
    clean visual cue without obscuring detail.

    Args:
        image: Input image (H, W, 3), BGR recommended.
        mask: Binary mask (H, W) with dtype ``uint8`` – ``1`` for the object,
            ``0`` for background.
        color: BGR colour of the overlay.
        alpha: Opacity of the overlay (0.0 = invisible, 1.0 = fully opaque).

    Returns:
        A copy of *image* with the overlay applied.
    """
    result = image.copy()
    overlay = result.copy()
    overlay[mask.astype(bool)] = color
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
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
# Variant 2 – SAM2 Video Predictor (temporal propagation)
# ---------------------------------------------------------------------------

class SAM2VideoSegmenter:
    """SAM2 video predictor for propagating a segmentation mask across frames.

    Given a list of BGR frames and initial point prompts on the first frame,
    this class uses the SAM2 video predictor to track the object through
    every frame.

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
            from sam2.sam2_video_predictor import SAM2VideoPredictor  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SAM2 is required for video segmentation.  "
                "Install it with:  pip install sam-2"
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading SAM2 video predictor '%s' on %s …", model_id, device)
        self.predictor = SAM2VideoPredictor.from_pretrained(model_id)
        self.predictor.to(device)
        self.device = device
        logger.info("SAM2 video predictor loaded.")

    def propagate(
        self,
        frames_bgr: list[np.ndarray],
        points: list[list[int]],
        labels: list[int],
        obj_id: int = 1,
    ) -> list[np.ndarray]:
        """Propagate a segmentation mask across all frames.

        The points/labels are applied to the first frame and then tracked
        through every subsequent frame using SAM2's temporal propagation.

        Args:
            frames_bgr: List of BGR images (H, W, 3), one per frame.
            points: List of ``[x, y]`` pixel coordinates on the first frame.
            labels: Per-point label – ``1`` = foreground, ``0`` = background.
            obj_id: Object identifier (default: ``1``).

        Returns:
            List of binary masks ``(H, W, uint8)`` – one per frame.
        """
        import torch

        if not frames_bgr:
            return []

        # SAM2 video predictor expects a directory of JPEG frames.
        # We write frames to a temporary directory, run propagation,
        # then clean up.
        with tempfile.TemporaryDirectory(prefix="sam2_video_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for i, bgr in enumerate(frames_bgr):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(tmp_path / f"{i:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                state = self.predictor.init_state(video_path=str(tmp_path))

                # Add point prompts on first frame
                pts = np.array(points, dtype=np.float32)
                lbls = np.array(labels, dtype=np.int32)
                self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbls,
                    normalize_coords=False,
                )

                # Propagate across all frames
                masks_by_frame: dict[int, np.ndarray] = {}
                for frame_idx, _obj_ids, video_res_masks in self.predictor.propagate_in_video(state):
                    # video_res_masks: (num_objects, H, W) float tensor
                    mask = (video_res_masks[0] > 0.5).cpu().numpy().astype(np.uint8)
                    masks_by_frame[frame_idx] = mask

        # Return masks in frame order
        return [masks_by_frame.get(i, np.zeros_like(masks_by_frame[0])) for i in range(len(frames_bgr))]


# ---------------------------------------------------------------------------
# Interactive point selection
# ---------------------------------------------------------------------------

def interactive_select(
    image_bgr: np.ndarray,
    segmenter: BaseSegmenter,
    window_name: str = "Select Object",
    fade_pixels: int = 16,
    min_brightness: float = 0.0,
    overlay_color: tuple[int, int, int] = (0, 180, 0),
    overlay_alpha: float = 0.4,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Open an OpenCV window for interactive point selection with live preview.

    During selection the mask is shown as a semi-transparent colour overlay
    (via :func:`highlight_object_overlay`).  The final confirmed image uses
    :func:`highlight_object_fadeout` for the episode video stream.

    Controls:
      - **Left click** – add foreground point (green dot)
      - **Right click** – add background point (red dot)
      - **Mouse wheel** – zoom in / out (centred on cursor)
      - **Middle mouse drag** – pan the view
      - **R** – reset zoom & pan to default
      - **Z** – undo last point
      - **Enter** – confirm selection
      - **Escape** – cancel

    Args:
        image_bgr: BGR image (H, W, 3) – typically the latest live frame.
        segmenter: A :class:`BaseSegmenter` instance (e.g. :class:`SAM2Segmenter`).
        window_name: Title of the OpenCV window.
        fade_pixels: Number of pixels over which brightness fades from full
            (at the mask boundary) to *min_brightness* in the **final** image.
        min_brightness: Brightness multiplier for pixels beyond the fade zone
            in the **final** image (0.0 = black, 1.0 = unchanged).
        overlay_color: BGR colour of the semi-transparent mask overlay shown
            during interactive selection.
        overlay_alpha: Opacity of the overlay during selection.

    Returns:
        ``(highlighted_bgr, mask)`` on success, or ``(None, None)`` if the
        user cancelled.
    """
    points: list[list[int]] = []
    labels: list[int] = []
    current_mask: np.ndarray | None = None

    # -- View state (zoom & pan) -------------------------------------------
    # ``scale`` = display pixels per image pixel (1.0 = 1:1).
    # ``offset_x/y`` = image coordinate mapped to the top-left display corner.
    img_h, img_w = image_bgr.shape[:2]
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    zoom_min: float = 0.25
    zoom_max: float = 16.0
    zoom_step: float = 1.15  # multiplicative factor per scroll tick

    # Pan state (middle-button drag)
    panning: bool = False
    pan_start_x: int = 0
    pan_start_y: int = 0
    pan_start_offset_x: float = 0.0
    pan_start_offset_y: float = 0.0

    # -- coordinate helpers ------------------------------------------------

    def _display_to_image(dx: int, dy: int) -> tuple[int, int]:
        """Convert display (window) pixel to image pixel."""
        ix = int(round(dx / scale + offset_x))
        iy = int(round(dy / scale + offset_y))
        return ix, iy

    def _image_to_display(ix: float, iy: float) -> tuple[int, int]:
        """Convert image pixel to display (window) pixel."""
        dx = int(round((ix - offset_x) * scale))
        dy = int(round((iy - offset_y) * scale))
        return dx, dy

    def _clamp_offset() -> None:
        """Keep the view within reasonable bounds of the image."""
        nonlocal offset_x, offset_y
        visible_w = img_w / scale if scale > 0 else img_w
        visible_h = img_h / scale if scale > 0 else img_h
        # Allow panning up to half the visible area beyond each edge
        offset_x = max(-visible_w * 0.5, min(offset_x, img_w - visible_w * 0.5))
        offset_y = max(-visible_h * 0.5, min(offset_y, img_h - visible_h * 0.5))

    # -- mask update -------------------------------------------------------

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

    # -- mouse callback ----------------------------------------------------

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal scale, offset_x, offset_y
        nonlocal panning, pan_start_x, pan_start_y
        nonlocal pan_start_offset_x, pan_start_offset_y

        # --- Zoom (mouse wheel) ---
        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom centred on cursor position
            img_cx, img_cy = _display_to_image(x, y)
            if flags > 0:
                scale = min(scale * zoom_step, zoom_max)
            else:
                scale = max(scale / zoom_step, zoom_min)
            # Adjust offset so cursor stays on the same image pixel
            offset_x = img_cx - x / scale
            offset_y = img_cy - y / scale
            _clamp_offset()
            return

        # --- Pan (middle button drag) ---
        if event == cv2.EVENT_MBUTTONDOWN:
            panning = True
            pan_start_x = x
            pan_start_y = y
            pan_start_offset_x = offset_x
            pan_start_offset_y = offset_y
            return

        if event == cv2.EVENT_MBUTTONUP:
            panning = False
            return

        if event == cv2.EVENT_MOUSEMOVE and panning:
            dx = x - pan_start_x
            dy = y - pan_start_y
            offset_x = pan_start_offset_x - dx / scale
            offset_y = pan_start_offset_y - dy / scale
            _clamp_offset()
            return

        # --- Point placement (left / right click) ---
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = _display_to_image(x, y)
            if 0 <= ix < img_w and 0 <= iy < img_h:
                points.append([ix, iy])
                labels.append(1)
                _update_mask()
        elif event == cv2.EVENT_RBUTTONDOWN:
            ix, iy = _display_to_image(x, y)
            if 0 <= ix < img_w and 0 <= iy < img_h:
                points.append([ix, iy])
                labels.append(0)
                _update_mask()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _on_mouse)

    instructions = (
        "L-click: fg | R-click: bg | Wheel: zoom | Mid-drag: pan | "
        "R: reset view | Z: undo | Enter: confirm | Esc: cancel"
    )

    result: tuple[np.ndarray | None, np.ndarray | None] = (None, None)

    try:
        while True:
            # Build full-resolution annotated frame (interactive overlay)
            if current_mask is not None:
                base = highlight_object_overlay(
                    image_bgr, current_mask,
                    color=overlay_color, alpha=overlay_alpha,
                )
            else:
                base = image_bgr.copy()

            # Draw points (in image space)
            for pt, lbl in zip(points, labels):
                colour = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                radius = max(1, int(6 / scale)) if scale > 1 else 6
                thickness = max(1, int(1 / scale)) if scale > 1 else 1
                cv2.circle(base, (pt[0], pt[1]), radius, colour, -1)
                cv2.circle(base, (pt[0], pt[1]), radius + 1, (255, 255, 255), thickness)

            # Apply zoom & pan via affine warp
            display_h, display_w = img_h, img_w
            M = np.array(
                [[scale, 0, -offset_x * scale],
                 [0, scale, -offset_y * scale]],
                dtype=np.float64,
            )
            display = cv2.warpAffine(
                base, M, (display_w, display_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(40, 40, 40),
            )

            # Instructions bar (always in display space)
            cv2.putText(
                display, instructions,
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Zoom indicator
            zoom_text = f"Zoom: {scale:.1f}x"
            cv2.putText(
                display, zoom_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # Enter – confirm
                if current_mask is not None:
                    highlighted = highlight_object_fadeout(
                        image_bgr, current_mask,
                        fade_pixels=fade_pixels, min_brightness=min_brightness,
                    )
                    result = (highlighted, current_mask)
                break
            elif key == 27:  # Escape – cancel
                break
            elif key in (ord("z"), ord("Z")):
                if points:
                    points.pop()
                    labels.pop()
                    _update_mask()
            elif key in (ord("r"), ord("R")):
                # Reset zoom & pan
                scale = 1.0
                offset_x = 0.0
                offset_y = 0.0
    finally:
        cv2.destroyWindow(window_name)

    return result
