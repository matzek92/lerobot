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

"""Split dataset episodes at frames marked by recorded key presses.

Requires: pip install 'lerobot[dataset]'

The recording script stores per-frame key markers in
``observation.episode_key_markers``. This script finds split frames from those
markers and applies ``split_episodes``.

Example:
    lerobot-split-episodes-by-keys \
        --repo-id my_user/my_dataset \
        --split-keys s \
        --new-repo-id my_user/my_dataset_split
"""

import argparse
import logging
from pathlib import Path
from typing import Iterable

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import split_episodes
from lerobot.utils.utils import init_logging


def _parse_marker_keys(marker_value: object) -> set[str]:
    if marker_value is None:
        return set()
    if isinstance(marker_value, str):
        raw_parts = marker_value.split("|")
    elif isinstance(marker_value, (list, tuple)):
        raw_parts = [str(part) for part in marker_value]
    else:
        raw_parts = [str(marker_value)]
    return {part for part in raw_parts if part}


def _find_split_frames_in_episode(marker_values: Iterable[object], split_keys: set[str]) -> list[int]:
    split_frames: list[int] = []
    for rel_frame_idx, marker_value in enumerate(marker_values):
        marker_keys = _parse_marker_keys(marker_value)
        if marker_keys & split_keys:
            split_frames.append(rel_frame_idx)
    return split_frames


def _build_episode_split_specs(
    dataset: LeRobotDataset,
    marker_feature: str,
    split_keys: set[str],
) -> dict[int, int]:
    if marker_feature not in dataset.features:
        raise ValueError(
            f"Marker feature '{marker_feature}' not found in dataset. "
            "Record with lerobot-record to include observation.episode_key_markers."
        )

    dataset._ensure_hf_dataset_loaded()
    marker_values: list[object] = dataset.hf_dataset[marker_feature]

    episode_split_specs: dict[int, int] = {}
    episodes_with_multiple_markers: dict[int, list[int]] = {}

    for ep_idx in range(dataset.meta.total_episodes):
        from_idx = int(dataset.meta.episodes["dataset_from_index"][ep_idx])
        to_idx = int(dataset.meta.episodes["dataset_to_index"][ep_idx])
        ep_length = to_idx - from_idx

        matches = _find_split_frames_in_episode(marker_values[from_idx:to_idx], split_keys)
        valid_matches = [frame_idx for frame_idx in matches if 0 < frame_idx < ep_length]

        if len(valid_matches) == 1:
            episode_split_specs[ep_idx] = valid_matches[0]
        elif len(valid_matches) > 1:
            episodes_with_multiple_markers[ep_idx] = valid_matches

    if episodes_with_multiple_markers:
        details = ", ".join(f"{ep}: {frames}" for ep, frames in episodes_with_multiple_markers.items())
        raise ValueError(
            "Found multiple split key markers in one or more episodes. "
            "Use one split marker per episode for this script. "
            f"Conflicts: {details}"
        )

    if not episode_split_specs:
        raise ValueError(
            "No valid split markers found. "
            "Ensure the selected key(s) were pressed after frame 0 and before the last frame."
        )

    return episode_split_specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id (e.g. user/dataset).")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root. Defaults to the LeRobot cache path for repo-id.",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        default=None,
        help="Output dataset repo id. Defaults to '<repo-id>_split'.",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=None,
        help="Output dataset path. Defaults to LeRobot cache path for output repo id.",
    )
    parser.add_argument(
        "--marker-feature",
        type=str,
        default="observation.episode_key_markers",
        help="Feature key containing per-frame key markers.",
    )
    parser.add_argument(
        "--split-keys",
        nargs="+",
        default=["s"],
        help="Keys that should trigger an episode split (default: s).",
    )

    args = parser.parse_args()

    init_logging()

    split_keys = {key for key in args.split_keys if key}
    if not split_keys:
        raise ValueError("--split-keys must contain at least one non-empty key.")

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    episode_split_specs = _build_episode_split_specs(dataset, args.marker_feature, split_keys)

    logging.info(f"Splitting {len(episode_split_specs)} episodes using keys {sorted(split_keys)}")
    logging.info(f"Split specs: {episode_split_specs}")

    split_repo_id = args.new_repo_id or f"{args.repo_id}_split"
    split_output_dir = args.new_root

    result_dataset = split_episodes(
        dataset,
        episode_split_specs=episode_split_specs,
        output_dir=split_output_dir,
        repo_id=split_repo_id,
    )

    logging.info(f"Dataset saved to {result_dataset.root}")
    logging.info(
        f"Episodes: {result_dataset.meta.total_episodes}, Frames: {result_dataset.meta.total_frames}"
    )


if __name__ == "__main__":
    main()
