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

import numpy as np
import pytest

from lerobot.datasets.episode_analysis import analyze_episode_motion_arrays


def test_analyze_episode_motion_arrays_finds_motor_activity_window_and_scores():
    frame_indices = list(range(8))
    observation_state = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.2, 0.0],
            [0.2, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float32,
    )

    camera_motion_scores = {
        "observation.images.top": [0.0, 0.0, 0.0, 0.05, 0.04, 0.01, 0.0, 0.0],
    }

    result = analyze_episode_motion_arrays(frame_indices, observation_state, camera_motion_scores)

    assert result["activity_window"]["start_frame_index"] == 3
    assert result["activity_window"]["end_frame_index"] == 4
    assert result["activity_window"]["active_frame_indices"] == [3, 4]

    assert result["motor_scores"]["motor_0"] == pytest.approx([0.1, 0.1])
    assert result["motor_scores"]["motor_1"] == [0.0, 0.0]


def test_analyze_episode_motion_arrays_flags_camera_freeze_candidates():
    frame_indices = list(range(8))
    observation_state = np.asarray(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [3.0],
            [3.0],
        ],
        dtype=np.float32,
    )

    camera_motion_scores = {
        "observation.images.top": [0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.2, 0.0],
    }

    result = analyze_episode_motion_arrays(frame_indices, observation_state, camera_motion_scores)

    assert result["activity_window"]["start_frame_index"] == 3
    assert result["activity_window"]["end_frame_index"] == 5
    assert result["camera_freeze_candidates"]["observation.images.top"] == [4, 5]


def test_analyze_episode_motion_arrays_detects_gripper_motor_change_frames():
    frame_indices = list(range(8))

    # Motor 5 flips low->high at frame 2, high->low at frame 5, low->high at frame 7.
    gripper_signal = np.asarray([0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9], dtype=np.float32)
    observation_state = np.zeros((8, 6), dtype=np.float32)
    observation_state[:, 5] = gripper_signal

    camera_motion_scores = {
        "observation.images.top": [0.0] * 8,
    }

    result = analyze_episode_motion_arrays(frame_indices, observation_state, camera_motion_scores)
    gripper = result["gripper_transitions"]

    assert gripper["available"] is True
    assert gripper["motor_index"] == 5
    assert gripper["change_frames"] == [2, 5, 7]
    assert gripper["open_to_closed_frames"] == [2, 7]
    assert gripper["closed_to_open_frames"] == [5]


def test_initial_motion_filter_skips_early_blip_and_keeps_main_activity():
    frame_indices = list(range(12))
    observation_state = np.asarray(
        [
            [0.0],
            [0.3],
            [0.3],
            [0.3],
            [0.3],
            [0.35],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [4.0],
            [4.0],
        ],
        dtype=np.float32,
    )

    result = analyze_episode_motion_arrays(
        frame_indices,
        observation_state,
        {"observation.images.top": [0.0] * len(frame_indices)},
        filter_initial_motion=True,
        min_idle_frames=3,
    )

    assert result["activity_window"]["start_frame_index"] == 6
    assert result["activity_window"]["end_frame_index"] == 9


def test_motion_filter_uses_longest_movement_region():
    frame_indices = list(range(18))
    observation_state = np.asarray(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [3.0],
            [3.0],
            [3.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0],
            [10.0],
            [10.0],
            [10.0],
        ],
        dtype=np.float32,
    )

    result = analyze_episode_motion_arrays(
        frame_indices,
        observation_state,
        {"observation.images.top": [0.0] * len(frame_indices)},
        filter_initial_motion=True,
        filter_final_motion=True,
        min_idle_frames=3,
    )

    assert result["activity_window"]["start_frame_index"] == 8
    assert result["activity_window"]["end_frame_index"] == 14
    assert result["activity_window"]["active_frame_indices"] == [8, 9, 10, 11, 12, 13, 14]
