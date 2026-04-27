#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team.
# All rights reserved.
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

"""Utilities for episode-level motion analysis.

This module identifies motor dead-times at the beginning and end of an episode,
computes per-frame motion scores for motor states and camera streams, and flags
camera freeze candidates when camera motion is near-zero while motors move.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from lerobot.utils.constants import OBS_STATE


def _suppress_torchvision_video_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"torchvision\.io\._video_deprecation_warning",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
    )


def _worker_init_suppress_torchvision_warnings(_: int) -> None:
    _suppress_torchvision_video_warnings()


_suppress_torchvision_video_warnings()


def _compute_threshold(values: np.ndarray, min_threshold: float) -> float:
    if values.size == 0:
        return min_threshold
    robust_peak = float(np.quantile(values, 0.95))
    return max(min_threshold, robust_peak * 0.05)


def _find_consecutive_true_positions(mask: np.ndarray, min_run_length: int = 2) -> list[int]:
    positions: list[int] = []
    run_start: int | None = None
    for idx, is_true in enumerate(mask):
        if is_true and run_start is None:
            run_start = idx
        elif not is_true and run_start is not None:
            if idx - run_start >= min_run_length:
                positions.extend(range(run_start, idx))
            run_start = None

    if run_start is not None and len(mask) - run_start >= min_run_length:
        positions.extend(range(run_start, len(mask)))
    return positions


def _find_longest_movement_region(
    moving_transitions: np.ndarray,
    min_idle_frames: int,
) -> tuple[int, int] | None:
    """Find the longest movement region, splitting only on sufficiently long idle gaps.

    Returns transition-index bounds `(start_transition, end_transition)`.
    Short idle gaps inside the main activity are ignored; only idle runs with at least
    `min_idle_frames` frames split movement into separate regions.
    """
    moving_positions = np.flatnonzero(moving_transitions)
    if moving_positions.size == 0:
        return None

    min_idle_frames = max(1, int(min_idle_frames))
    regions: list[tuple[int, int]] = []
    region_start = int(moving_positions[0])
    region_end = int(moving_positions[0])

    for pos in moving_positions[1:]:
        pos = int(pos)
        idle_gap = pos - region_end - 1
        if idle_gap >= min_idle_frames:
            regions.append((region_start, region_end))
            region_start = pos
            region_end = pos
        else:
            region_end = pos

    regions.append((region_start, region_end))
    return max(regions, key=lambda bounds: bounds[1] - bounds[0])


def _smooth_scores(scores: list[float], window_size: int = 3) -> list[float]:
    """Apply moving average smoothing to reduce noise in motion scores.

    Args:
        scores: Sequence of motion scores.
        window_size: Window size for moving average (must be odd, >= 1).

    Returns:
        Smoothed scores list.
    """
    if window_size < 1 or len(scores) < window_size:
        return scores

    # Ensure window_size is odd
    window_size = max(1, window_size if window_size % 2 == 1 else window_size + 1)
    half = window_size // 2
    smoothed = []

    for i in range(len(scores)):
        start = max(0, i - half)
        end = min(len(scores), i + half + 1)
        smoothed.append(np.mean(scores[start:end]))

    return smoothed


def _detect_binary_motor_state_changes(
    frame_indices: list[int],
    motor_signal: np.ndarray,
) -> dict:
    """Detect frame indices where a near-binary motor signal flips state.

    The lower-value cluster is treated as "open" and the higher-value cluster
    as "closed".
    """
    if motor_signal.ndim != 1:
        raise ValueError(f"Expected 1D motor_signal, got shape {motor_signal.shape}")

    if motor_signal.size != len(frame_indices):
        raise ValueError(
            f"motor_signal length ({motor_signal.size}) does not match frame_indices length ({len(frame_indices)})"
        )

    if motor_signal.size == 0:
        return {
            "open_state_is_low_value": True,
            "threshold": None,
            "signal_min": None,
            "signal_max": None,
            "change_frames": [],
            "open_to_closed_frames": [],
            "closed_to_open_frames": [],
        }

    q10 = float(np.quantile(motor_signal, 0.10))
    q90 = float(np.quantile(motor_signal, 0.90))
    threshold = float((q10 + q90) / 2.0)

    # If there is no useful separation between low/high clusters, report no flips.
    if abs(q90 - q10) <= 1e-6:
        return {
            "open_state_is_low_value": True,
            "threshold": threshold,
            "signal_min": float(np.min(motor_signal)),
            "signal_max": float(np.max(motor_signal)),
            "change_frames": [],
            "open_to_closed_frames": [],
            "closed_to_open_frames": [],
        }

    is_closed = motor_signal >= threshold
    transition_positions = np.flatnonzero(np.diff(is_closed.astype(np.int8)) != 0) + 1

    change_frames = [int(frame_indices[pos]) for pos in transition_positions]
    open_to_closed_frames = [int(frame_indices[pos]) for pos in transition_positions if is_closed[pos]]
    closed_to_open_frames = [int(frame_indices[pos]) for pos in transition_positions if not is_closed[pos]]

    return {
        "open_state_is_low_value": True,
        "threshold": threshold,
        "signal_min": float(np.min(motor_signal)),
        "signal_max": float(np.max(motor_signal)),
        "change_frames": change_frames,
        "open_to_closed_frames": open_to_closed_frames,
        "closed_to_open_frames": closed_to_open_frames,
    }


def analyze_episode_motion_arrays(
    frame_indices: list[int],
    observation_state: np.ndarray,
    camera_motion_scores: dict[str, list[float]],
    filter_initial_motion: bool = False,
    filter_final_motion: bool = False,
    min_idle_frames: int = 5,
    frame_padding: int = 0,
    motor_score_aggregation: str = "max",
    gripper_motor_index: int | None = 5,
) -> dict:
    """Analyze one episode given state and per-frame camera motion scores.

    Args:
        frame_indices: Episode-local frame indices.
        observation_state: Array with shape (num_frames, num_motors).
        camera_motion_scores: Per-camera per-frame motion scores.

    Returns:
        A JSON-serializable dictionary containing activity window, scores and
        camera freeze candidates.
    """
    if len(frame_indices) == 0:
        raise ValueError("Empty episode: no frames found")

    if observation_state.ndim != 2:
        raise ValueError(
            f"Expected observation_state with shape (num_frames, num_motors), got {observation_state.shape}"
        )

    num_frames, num_motors = observation_state.shape
    if num_frames != len(frame_indices):
        raise ValueError(
            f"frame_indices length ({len(frame_indices)}) does not match state frames ({num_frames})"
        )

    if motor_score_aggregation not in {"max", "sum"}:
        raise ValueError(f"Unsupported motor_score_aggregation '{motor_score_aggregation}'. Use 'max' or 'sum'.")

    state_transition_scores = np.abs(np.diff(observation_state, axis=0))

    # Aggregate motor transitions: max or sum across motors
    if motor_score_aggregation == "sum":
        aggregated_motor_transition_score = (
            np.sum(state_transition_scores, axis=1) if state_transition_scores.size > 0 else np.zeros(0, dtype=np.float32)
        )
    else:  # default "max"
        aggregated_motor_transition_score = (
            np.max(state_transition_scores, axis=1) if state_transition_scores.size > 0 else np.zeros(0, dtype=np.float32)
        )

    motor_motion_threshold = _compute_threshold(aggregated_motor_transition_score, min_threshold=1e-6)
    moving_transitions = aggregated_motor_transition_score > motor_motion_threshold

    start_frame_index: int | None
    end_frame_index: int | None
    start_pos: int | None
    end_pos: int | None
    if moving_transitions.any():
        first_transition = int(np.flatnonzero(moving_transitions)[0])
        last_transition = int(np.flatnonzero(moving_transitions)[-1])

        if filter_initial_motion or filter_final_motion:
            longest_region = _find_longest_movement_region(moving_transitions, min_idle_frames=min_idle_frames)
            if longest_region is not None:
                first_transition, last_transition = longest_region

        start_pos = first_transition + 1
        end_pos = last_transition + 1

        # Be robust to filtering corner-cases where first/last transitions cross.
        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos

        # Apply frame padding to extend the activity window
        if frame_padding > 0:
            start_pos = max(0, start_pos - frame_padding)
            end_pos = min(len(frame_indices) - 1, end_pos + frame_padding)

        start_frame_index = int(frame_indices[start_pos])
        end_frame_index = int(frame_indices[end_pos])
    else:
        start_pos = None
        end_pos = None
        start_frame_index = None
        end_frame_index = None

    motor_frame_scores = np.zeros((num_frames, num_motors), dtype=np.float32)
    if num_frames > 1:
        motor_frame_scores[1:] = state_transition_scores

    if start_pos is not None and end_pos is not None:
        active_slice = slice(start_pos, end_pos + 1)
        active_frame_indices = frame_indices[active_slice]
    else:
        active_slice = slice(0, 0)
        active_frame_indices = []

    motor_global_frame_score = np.zeros(num_frames, dtype=np.float32)
    if num_frames > 1:
        motor_global_frame_score[1:] = aggregated_motor_transition_score

    camera_stall_candidates: dict[str, list[int]] = {}
    camera_thresholds: dict[str, float] = {}
    for camera_key, scores in camera_motion_scores.items():
        if len(scores) != num_frames:
            raise ValueError(
                f"Camera '{camera_key}' score length ({len(scores)}) does not match number of frames ({num_frames})"
            )

        camera_score_arr = np.asarray(scores, dtype=np.float32)

        camera_transition_scores = camera_score_arr[1:] if num_frames > 1 else np.zeros(0, dtype=np.float32)
        camera_threshold = _compute_threshold(camera_transition_scores, min_threshold=1e-8)
        camera_thresholds[camera_key] = camera_threshold

        potential_stall_mask = (
            (motor_global_frame_score > motor_motion_threshold) & (camera_score_arr <= camera_threshold)
        )

        if start_pos is None or end_pos is None:
            potential_stall_mask[:] = False
        else:
            active_mask = np.zeros(num_frames, dtype=bool)
            active_mask[start_pos : end_pos + 1] = True
            potential_stall_mask &= active_mask

        stall_positions = _find_consecutive_true_positions(potential_stall_mask, min_run_length=2)
        camera_stall_candidates[camera_key] = [int(frame_indices[pos]) for pos in stall_positions]

    motor_scores = {
        f"motor_{i}": [float(v) for v in motor_frame_scores[active_slice, i].tolist()]
        for i in range(num_motors)
    }

    gripper_transitions: dict
    if gripper_motor_index is None:
        gripper_transitions = {
            "motor_index": None,
            "available": False,
            "reason": "disabled",
            "open_state_is_low_value": True,
            "threshold": None,
            "signal_min": None,
            "signal_max": None,
            "change_frames": [],
            "open_to_closed_frames": [],
            "closed_to_open_frames": [],
        }
    elif gripper_motor_index < 0 or gripper_motor_index >= num_motors:
        gripper_transitions = {
            "motor_index": int(gripper_motor_index),
            "available": False,
            "reason": f"out_of_bounds_for_num_motors_{num_motors}",
            "open_state_is_low_value": True,
            "threshold": None,
            "signal_min": None,
            "signal_max": None,
            "change_frames": [],
            "open_to_closed_frames": [],
            "closed_to_open_frames": [],
        }
    else:
        gripper_transitions = {
            "motor_index": int(gripper_motor_index),
            "available": True,
            **_detect_binary_motor_state_changes(frame_indices, observation_state[:, gripper_motor_index]),
        }

    return {
        "num_frames": num_frames,
        "num_motors": num_motors,
        "activity_window": {
            "motor_motion_threshold": float(motor_motion_threshold),
            "start_frame_index": start_frame_index,
            "end_frame_index": end_frame_index,
            "active_frame_indices": [int(i) for i in active_frame_indices],
        },
        "motor_scores": motor_scores,
        "camera_freeze_candidates": camera_stall_candidates,
        "camera_motion_thresholds": camera_thresholds,
        "gripper_transitions": gripper_transitions,
    }


def analyze_episode_motion(
    dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    filter_initial_motion: bool = False,
    filter_final_motion: bool = False,
    min_idle_frames: int = 5,
    frame_padding: int = 0,
    motor_score_aggregation: str = "max",
    gripper_motor_index: int | None = 5,
    smoothing_window: int = 1,
    show_progress: bool = True,
) -> dict:
    """Analyze a single-episode LeRobotDataset instance.

    The dataset is expected to contain exactly one episode (as loaded by the
    caller with `episodes=[episode_index]`).
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        worker_init_fn=_worker_init_suppress_torchvision_warnings if num_workers > 0 else None,
    )

    frame_indices: list[int] = []
    state_batches: list[np.ndarray] = []

    previous_image_by_camera: dict[str, torch.Tensor] = {}
    camera_motion_scores: dict[str, list[float]] = defaultdict(list)

    dataloader_iter = tqdm(dataloader, desc="Analyzing episode", unit="batch") if show_progress else dataloader

    for batch in dataloader_iter:
        frame_indices.extend([int(v) for v in batch["frame_index"].tolist()])

        if OBS_STATE in batch:
            state_batches.append(batch[OBS_STATE].cpu().numpy())

        for camera_key in dataset.meta.camera_keys:
            for image in batch[camera_key]:
                if camera_key not in previous_image_by_camera:
                    camera_motion_scores[camera_key].append(0.0)
                else:
                    score = torch.mean(torch.abs(image - previous_image_by_camera[camera_key])).item()
                    camera_motion_scores[camera_key].append(float(score))
                previous_image_by_camera[camera_key] = image

    if OBS_STATE not in dataset.meta.features:
        raise ValueError(
            f"Dataset '{dataset.repo_id}' does not contain '{OBS_STATE}'. Motor dead-time analysis requires it."
        )

    if len(state_batches) == 0:
        raise ValueError(f"Dataset '{dataset.repo_id}' has no '{OBS_STATE}' values in the selected episode")

    # Apply smoothing to camera motion scores
    if smoothing_window > 1:
        for camera_key in camera_motion_scores:
            camera_motion_scores[camera_key] = _smooth_scores(
                camera_motion_scores[camera_key],
                window_size=smoothing_window,
            )

    observation_state = np.concatenate(state_batches, axis=0)
    result = analyze_episode_motion_arrays(
        frame_indices,
        observation_state,
        camera_motion_scores,
        filter_initial_motion=filter_initial_motion,
        filter_final_motion=filter_final_motion,
        min_idle_frames=min_idle_frames,
        frame_padding=frame_padding,
        motor_score_aggregation=motor_score_aggregation,
        gripper_motor_index=gripper_motor_index,
    )
    # Attach raw data for optional verbose plotting
    result["_raw"] = {
        "frame_indices": frame_indices,
        "observation_state": observation_state,
        "camera_motion_scores": {k: list(v) for k, v in camera_motion_scores.items()},
        "motor_score_aggregation": motor_score_aggregation,
        "smoothing_window": smoothing_window,
    }
    return result


def plot_episode_motion(
    result: dict,
    output_html_path: str | Path | None = None,
    show: bool = True,
) -> None:  # pragma: no cover
    """Display interactive Plotly charts for motor state, derivative and camera scores."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError("plotly is required for verbose mode. Install it with: pip install plotly") from e

    raw = result["_raw"]
    frame_indices: list[int] = raw["frame_indices"]
    observation_state: np.ndarray = raw["observation_state"]
    camera_motion_scores: dict[str, list[float]] = raw["camera_motion_scores"]
    motor_score_aggregation: str = raw.get("motor_score_aggregation", "max")
    smoothing_window: int = int(raw.get("smoothing_window", 1))

    num_motors = observation_state.shape[1]

    def _camera_priority(item: tuple[str, list[float]]) -> tuple[int, str]:
        key = item[0].lower()
        if "wrist" in key:
            return (0, key)
        if "color" in key:
            return (1, key)
        return (2, key)

    camera_items = sorted(camera_motion_scores.items(), key=_camera_priority)
    num_cameras = len(camera_items)
    num_rows = 5 + num_cameras  # aggregated scores, derivative, raw motor state, gripper plots, then cameras

    aggregation_label = "sum" if motor_score_aggregation == "sum" else "max"
    gripper_title_suffix = "auto"
    gripper_transitions = result.get("gripper_transitions", {})
    gripper_available = bool(gripper_transitions.get("available", False))
    gripper_motor_index = gripper_transitions.get("motor_index", None)
    if isinstance(gripper_motor_index, int):
        gripper_state_title = f"Gripper Motor State (motor_{gripper_motor_index})"
        gripper_derivative_title = f"Gripper Motor Derivative (motor_{gripper_motor_index}, |Δstate|)"
    else:
        gripper_state_title = f"Gripper Motor State ({gripper_title_suffix})"
        gripper_derivative_title = f"Gripper Motor Derivative ({gripper_title_suffix}, |Δstate|)"

    subplot_titles = [
        f"Motor Scores ({aggregation_label} |Δstate| per frame)",
        "Motor State Derivative (|Δstate|)",
        "Motor State (raw)",
        gripper_state_title,
        gripper_derivative_title,
        *[f"Camera Motion Score: {k}" for k, _ in camera_items],
    ]
    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, subplot_titles=subplot_titles)

    motor_names = [f"motor_{i}" for i in range(num_motors)]
    frame_to_pos = {int(frame): idx for idx, frame in enumerate(frame_indices)}

    row_aggregated = 1
    row_motor_derivative = 2
    row_motor_state = 3
    row_gripper_state = 4
    row_gripper_derivative = 5
    row_camera_start = 6
    motor_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    open_to_closed_frames = [
        int(f) for f in gripper_transitions.get("open_to_closed_frames", []) if int(f) in frame_to_pos
    ]
    closed_to_open_frames = [
        int(f) for f in gripper_transitions.get("closed_to_open_frames", []) if int(f) in frame_to_pos
    ]

    # Row 3: raw motor state
    for i, name in enumerate(motor_names):
        motor_color = motor_palette[i % len(motor_palette)]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=observation_state[:, i].tolist(),
                name=name,
                legendgroup=name,
                line=dict(color=motor_color),
            ),
            row=row_motor_state, col=1,
        )

    # Row 2: derivative (|Δstate| per motor)
    derivatives = np.abs(np.diff(observation_state, axis=0))
    deriv_indices = frame_indices[1:]
    deriv_frame_to_pos = {int(frame): idx for idx, frame in enumerate(deriv_indices)}
    for i, name in enumerate(motor_names):
        motor_color = motor_palette[i % len(motor_palette)]
        fig.add_trace(
            go.Scatter(
                x=deriv_indices,
                y=derivatives[:, i].tolist(),
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(color=motor_color),
            ),
            row=row_motor_derivative, col=1,
        )

    if gripper_available and isinstance(gripper_motor_index, int) and 0 <= gripper_motor_index < num_motors:
        gripper_color = motor_palette[gripper_motor_index % len(motor_palette)]
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=observation_state[:, gripper_motor_index].tolist(),
                name=f"motor_{gripper_motor_index} state",
                legendgroup=f"motor_{gripper_motor_index}",
                showlegend=True,
                line=dict(color=gripper_color),
            ),
            row=row_gripper_state,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=deriv_indices,
                y=derivatives[:, gripper_motor_index].tolist(),
                name=f"motor_{gripper_motor_index} |Δstate|",
                legendgroup=f"motor_{gripper_motor_index}",
                showlegend=True,
                line=dict(color=gripper_color),
            ),
            row=row_gripper_derivative,
            col=1,
        )

        if open_to_closed_frames:
            open_to_closed_y = [observation_state[frame_to_pos[f], gripper_motor_index] for f in open_to_closed_frames]
            fig.add_trace(
                go.Scatter(
                    x=open_to_closed_frames,
                    y=open_to_closed_y,
                    mode="markers",
                    marker=dict(color="#D81B60", size=9, symbol="triangle-up"),
                    name=f"motor_{gripper_motor_index} open->closed",
                    showlegend=True,
                ),
                row=row_gripper_state,
                col=1,
            )

            open_to_closed_dy = [
                derivatives[deriv_frame_to_pos[f], gripper_motor_index]
                for f in open_to_closed_frames
                if f in deriv_frame_to_pos
            ]
            open_to_closed_dx = [f for f in open_to_closed_frames if f in deriv_frame_to_pos]
            if open_to_closed_dx:
                fig.add_trace(
                    go.Scatter(
                        x=open_to_closed_dx,
                        y=open_to_closed_dy,
                        mode="markers",
                        marker=dict(color="#D81B60", size=9, symbol="triangle-up"),
                        name=f"motor_{gripper_motor_index} open->closed (|Δ|)",
                        showlegend=True,
                    ),
                    row=row_gripper_derivative,
                    col=1,
                )

        if closed_to_open_frames:
            closed_to_open_y = [observation_state[frame_to_pos[f], gripper_motor_index] for f in closed_to_open_frames]
            fig.add_trace(
                go.Scatter(
                    x=closed_to_open_frames,
                    y=closed_to_open_y,
                    mode="markers",
                    marker=dict(color="#1E88E5", size=9, symbol="triangle-down"),
                    name=f"motor_{gripper_motor_index} closed->open",
                    showlegend=True,
                ),
                row=row_gripper_state,
                col=1,
            )

            closed_to_open_dy = [
                derivatives[deriv_frame_to_pos[f], gripper_motor_index]
                for f in closed_to_open_frames
                if f in deriv_frame_to_pos
            ]
            closed_to_open_dx = [f for f in closed_to_open_frames if f in deriv_frame_to_pos]
            if closed_to_open_dx:
                fig.add_trace(
                    go.Scatter(
                        x=closed_to_open_dx,
                        y=closed_to_open_dy,
                        mode="markers",
                        marker=dict(color="#1E88E5", size=9, symbol="triangle-down"),
                        name=f"motor_{gripper_motor_index} closed->open (|Δ|)",
                        showlegend=True,
                    ),
                    row=row_gripper_derivative,
                    col=1,
                )

    # Row 1: aggregated motor score per frame
    aggregated_score = np.zeros(len(frame_indices), dtype=np.float32)
    if derivatives.size > 0:
        if motor_score_aggregation == "sum":
            aggregated_score[1:] = np.sum(derivatives, axis=1)
        else:  # max
            aggregated_score[1:] = np.max(derivatives, axis=1)
    activity = result["activity_window"]
    threshold = activity["motor_motion_threshold"]

    active_frame_indices = [int(f) for f in activity.get("active_frame_indices", [])]
    if active_frame_indices:
        plotted_start_frame = int(min(active_frame_indices))
        plotted_end_frame = int(max(active_frame_indices))
    else:
        plotted_start_frame = activity.get("start_frame_index")
        plotted_end_frame = activity.get("end_frame_index")

    if plotted_start_frame is not None and plotted_end_frame is not None and plotted_start_frame > plotted_end_frame:
        plotted_start_frame, plotted_end_frame = plotted_end_frame, plotted_start_frame
    score_label = "sum motor score" if motor_score_aggregation == "sum" else "max motor score"
    fig.add_trace(
        go.Scatter(x=frame_indices, y=aggregated_score.tolist(), name=score_label, showlegend=True),
        row=row_aggregated, col=1,
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text="motion threshold", row=row_aggregated, col=1)
    if plotted_start_frame is not None and plotted_end_frame is not None:
        fig.add_vline(x=plotted_start_frame, line_dash="dot", line_color="green",
                      annotation_text="start", row=row_aggregated, col=1)
        fig.add_vline(x=plotted_end_frame, line_dash="dot", line_color="orange",
                      annotation_text="end", row=row_aggregated, col=1)

    if gripper_available:
        for frame in open_to_closed_frames:
            fig.add_vline(x=frame, line_dash="dot", line_color="#D81B60", row=row_gripper_state, col=1)
            fig.add_vline(x=frame, line_dash="dot", line_color="#D81B60", row=row_gripper_derivative, col=1)
        for frame in closed_to_open_frames:
            fig.add_vline(x=frame, line_dash="dot", line_color="#1E88E5", row=row_gripper_state, col=1)
            fig.add_vline(x=frame, line_dash="dot", line_color="#1E88E5", row=row_gripper_derivative, col=1)

    # Remaining rows: camera motion scores, with wrist/color keys shown first.
    for row_offset, (camera_key, scores) in enumerate(camera_items, start=row_camera_start):
        cam_threshold = result["camera_motion_thresholds"].get(camera_key, None)
        fig.add_trace(
            go.Scatter(x=frame_indices, y=scores, name=camera_key, showlegend=True),
            row=row_offset, col=1,
        )
        if cam_threshold is not None:
            fig.add_hline(y=cam_threshold, line_dash="dash", line_color="red",
                          annotation_text="freeze threshold", row=row_offset, col=1)
        freeze_frames = result["camera_freeze_candidates"].get(camera_key, [])
        if freeze_frames:
            freeze_y = [scores[frame_indices.index(f)] for f in freeze_frames if f in frame_indices]
            fig.add_trace(
                go.Scatter(x=freeze_frames, y=freeze_y, mode="markers",
                           marker=dict(color="red", size=6, symbol="x"),
                           name=f"{camera_key} freeze", showlegend=True),
                row=row_offset, col=1,
            )

    episode_index = result.get("episode_index", "?")
    repo_id = result.get("repo_id", "")
    smoothing_suffix = "" if smoothing_window <= 1 else f" | smoothing={smoothing_window}"
    fig.update_layout(
        title=f"Episode Analysis — {repo_id} episode {episode_index}{smoothing_suffix}",
        height=300 * num_rows,
    )

    if output_html_path is not None:
        output_html_path = Path(output_html_path)
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html_path), include_plotlyjs="cdn")

    if show:
        fig.show()
