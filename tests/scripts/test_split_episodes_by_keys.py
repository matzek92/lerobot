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

from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_split_episodes_by_keys import (  # noqa: E402
    _build_episode_split_specs,
    _find_split_frames_in_episode,
    _parse_marker_keys,
)


class _DummyDataset:
    def __init__(self, marker_values, total_episodes, episodes, features):
        self.hf_dataset = {"observation.episode_key_markers": marker_values}
        self.meta = SimpleNamespace(total_episodes=total_episodes, episodes=episodes)
        self.features = features

    def _ensure_hf_dataset_loaded(self):
        return None


def test_parse_marker_keys_from_string() -> None:
    assert _parse_marker_keys("a|b|c") == {"a", "b", "c"}


def test_find_split_frames_in_episode() -> None:
    markers = ["", "x", "s|x", ""]
    assert _find_split_frames_in_episode(markers, {"s"}) == [2]


def test_build_episode_split_specs() -> None:
    dataset = _DummyDataset(
        marker_values=["", "s", "", "", "", "", "k", ""],
        total_episodes=2,
        episodes={"dataset_from_index": [0, 4], "dataset_to_index": [4, 8]},
        features={"observation.episode_key_markers": {"dtype": "string"}},
    )

    split_specs = _build_episode_split_specs(dataset, "observation.episode_key_markers", {"s", "k"})

    assert split_specs == {0: 1, 1: 2}


def test_build_episode_split_specs_raises_for_multiple_markers_per_episode() -> None:
    dataset = _DummyDataset(
        marker_values=["", "s", "s", "", "", "", "", ""],
        total_episodes=2,
        episodes={"dataset_from_index": [0, 4], "dataset_to_index": [4, 8]},
        features={"observation.episode_key_markers": {"dtype": "string"}},
    )

    with pytest.raises(ValueError, match="multiple split key markers"):
        _build_episode_split_specs(dataset, "observation.episode_key_markers", {"s"})


def test_build_episode_split_specs_raises_when_no_valid_markers() -> None:
    dataset = _DummyDataset(
        marker_values=["", "", "", ""],
        total_episodes=1,
        episodes={"dataset_from_index": [0], "dataset_to_index": [4]},
        features={"observation.episode_key_markers": {"dtype": "string"}},
    )

    with pytest.raises(ValueError, match="No valid split markers found"):
        _build_episode_split_specs(dataset, "observation.episode_key_markers", {"s"})


def test_build_episode_split_specs_raises_when_marker_feature_missing() -> None:
    dataset = _DummyDataset(
        marker_values=["", "s", "", ""],
        total_episodes=1,
        episodes={"dataset_from_index": [0], "dataset_to_index": [4]},
        features={},
    )

    with pytest.raises(ValueError, match="Marker feature"):
        _build_episode_split_specs(dataset, "observation.episode_key_markers", {"s"})
