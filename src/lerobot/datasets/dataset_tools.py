#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Dataset tools utilities for LeRobotDataset.

This module provides utilities for:
- Deleting episodes from datasets
- Splitting datasets into multiple smaller datasets
- Adding/removing features from datasets
- Merging datasets (wrapper around aggregate functionality)
"""

import logging
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_FEATURES,
    get_parquet_file_size_in_mb,
    load_episodes,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import (
    _get_codec_options,
    decode_video_frames,
    encode_video_frames,
    get_video_info,
    resolve_vcodec,
)
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_IMAGE


def _load_episode_with_stats(src_dataset: LeRobotDataset, episode_idx: int) -> dict:
    """Load a single episode's metadata including stats from parquet file.

    Args:
        src_dataset: Source dataset
        episode_idx: Episode index to load

    Returns:
        dict containing episode metadata and stats
    """
    ep_meta = src_dataset.meta.episodes[episode_idx]
    chunk_idx = ep_meta["meta/episodes/chunk_index"]
    file_idx = ep_meta["meta/episodes/file_index"]

    parquet_path = src_dataset.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    df = pd.read_parquet(parquet_path)

    episode_row = df[df["episode_index"] == episode_idx].iloc[0]

    return episode_row.to_dict()


def delete_episodes(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Delete episodes from a LeRobotDataset and create a new dataset.

    Args:
        dataset: The source LeRobotDataset.
        episode_indices: List of episode indices to delete.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.
    """
    if not episode_indices:
        raise ValueError("No episodes to delete")

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_indices) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    logging.info(f"Deleting {len(episode_indices)} episodes from dataset")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    episodes_to_keep = [i for i in range(dataset.meta.total_episodes) if i not in episode_indices]
    if not episodes_to_keep:
        raise ValueError("Cannot delete all episodes from dataset")

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(episodes_to_keep)}

    video_metadata = None
    if dataset.meta.video_keys:
        video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

    _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    logging.info(f"Created new dataset with {len(episodes_to_keep)} episodes")
    return new_dataset


def trim_episodes(
    dataset: LeRobotDataset,
    episode_trim_specs: dict[int, tuple[int, int]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    append_to_dataset: LeRobotDataset | None = None,
) -> LeRobotDataset:
    """Trim frames from the beginning and/or end of episodes in a dataset.

    Creates a new dataset where specified episodes have frames removed from their
    start and/or end. This function uses the same ``add_frame`` / ``save_episode``
    / ``finalize`` mechanism as ``lerobot record``, so the v3 format's multi-episode
    video/parquet containers are managed automatically by the framework.

    Episodes not listed in *episode_trim_specs* are copied unchanged.

    When *append_to_dataset* is provided the trimmed episodes are **appended to that
    existing dataset** rather than written to a new one.  This enables an incremental
    workflow where you process multiple source datasets in separate runs and accumulate
    all results in a single target dataset.

    Args:
        dataset: The source LeRobotDataset.
        episode_trim_specs: Dict mapping episode index to (trim_start_frames, trim_end_frames).
            ``trim_start_frames``: Number of frames to remove from the beginning of the episode.
            ``trim_end_frames``: Number of frames to remove from the end of the episode.
            Episodes not in this dict are kept as-is.
        output_dir: Directory to save the new dataset. Ignored when *append_to_dataset* is given.
            If None and no append target, uses default location based on *repo_id*.
        repo_id: Repository ID for the new dataset. Ignored when *append_to_dataset* is given.
            If None and no append target, appends "_trimmed" to the source repo_id.
        append_to_dataset: When provided, trimmed episodes are appended to this existing dataset
            rather than written to a new one.  *output_dir* and *repo_id* must be ``None`` when
            this argument is used.  The source and target datasets must have identical user-defined
            features and the same FPS.

    Returns:
        :class:`LeRobotDataset` containing the trimmed episodes — either the newly created dataset
        or the updated *append_to_dataset* target.

    Example::

        # Trim 5 frames from start and 3 from end of episode 0,
        # and 2 frames from start of episode 2
        new_dataset = trim_episodes(
            dataset,
            episode_trim_specs={0: (5, 3), 2: (2, 0)},
            output_dir="./output",
        )

        # Append trimmed episodes to an existing dataset instead of creating a new one
        target = LeRobotDataset("my_repo", root="./target")
        trim_episodes(
            dataset,
            episode_trim_specs={0: (5, 3)},
            append_to_dataset=target,
        )
    """
    if not episode_trim_specs:
        raise ValueError("No trim specifications provided")

    if append_to_dataset is not None and (output_dir is not None or repo_id is not None):
        raise ValueError(
            "Cannot specify 'output_dir' or 'repo_id' together with 'append_to_dataset'. "
            "When appending to an existing dataset, the output directory is determined by that dataset."
        )

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_trim_specs.keys()) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.meta.root)

    for ep_idx, (trim_start, trim_end) in episode_trim_specs.items():
        if trim_start < 0 or trim_end < 0:
            raise ValueError(f"Trim values must be non-negative for episode {ep_idx}")
        ep_length = dataset.meta.episodes[ep_idx]["length"]
        if trim_start + trim_end >= ep_length:
            raise ValueError(
                f"Cannot trim {trim_start + trim_end} frames from episode {ep_idx} "
                f"with length {ep_length}. Result would have no frames."
            )

    logging.info(f"Trimming {len(episode_trim_specs)} episodes in dataset")

    # Feature dict without the default system features (they are re-added by create()).
    user_features = {k: v for k, v in dataset.meta.features.items() if k not in DEFAULT_FEATURES}

    if append_to_dataset is not None:
        # Validate feature compatibility between source and target.
        target_user_features = {
            k: v for k, v in append_to_dataset.meta.features.items() if k not in DEFAULT_FEATURES
        }
        if set(user_features.keys()) != set(target_user_features.keys()):
            raise ValueError(
                f"Feature mismatch between source dataset and append target. "
                f"Source has: {set(user_features.keys())}, "
                f"Target has: {set(target_user_features.keys())}"
            )
        if dataset.meta.fps != append_to_dataset.meta.fps:
            raise ValueError(
                f"FPS mismatch: source dataset has {dataset.meta.fps} fps "
                f"but target dataset has {append_to_dataset.meta.fps} fps"
            )
        # Enable write mode on the existing target dataset.
        # The LeRobotDataset write machinery (add_frame / save_episode / finalize) already
        # handles the "resume" case: when `latest_episode` is None but `meta.episodes` is
        # non-empty, it picks up chunk/file indices from the last existing episode and starts
        # a new parquet/video file without touching existing data.
        #
        # Safety note: if `latest_episode` is already set (e.g. this function was called before
        # on the same Python object in the same process), the parquet writer machinery would
        # re-open the last parquet file and overwrite its content.  We prevent that by setting
        # `_writer_closed_for_reading = True` whenever a previous write session was detected
        # (i.e. `latest_episode is not None`).  This flag makes `_save_episode_data` always
        # move to a fresh parquet file for the first episode written in this new session.
        if append_to_dataset.latest_episode is not None:
            append_to_dataset._writer_closed_for_reading = True
        append_to_dataset.episode_buffer = append_to_dataset.create_episode_buffer()
        write_target = append_to_dataset
    else:
        if repo_id is None:
            repo_id = f"{dataset.repo_id}_trimmed"
        output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

        # Create the output dataset using the same factory as lerobot record.
        # The framework then handles v3 container packaging automatically.
        write_target = LeRobotDataset.create(
            repo_id=repo_id,
            fps=dataset.meta.fps,
            features=user_features,
            robot_type=dataset.meta.robot_type,
            root=output_dir,
            use_videos=len(dataset.meta.video_keys) > 0,
        )

    # Make sure the source HF dataset is loaded before iterating.
    dataset._ensure_hf_dataset_loaded()

    has_video = len(dataset.meta.video_keys) > 0

    for ep_idx in tqdm(range(dataset.meta.total_episodes), desc="Trimming episodes"):
        ep_meta = dataset.meta.episodes[ep_idx]
        trim_start, trim_end = episode_trim_specs.get(ep_idx, (0, 0))

        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        kept_from_idx = from_idx + trim_start
        kept_to_idx = to_idx - trim_end
        kept_length = kept_to_idx - kept_from_idx

        # Batch-decode all video frames for this episode at once.
        # decode_video_frames accepts absolute timestamps within the video file.
        video_frames: dict[str, np.ndarray] = {}
        if has_video:
            for vid_key in dataset.meta.video_keys:
                from_ts = ep_meta[f"videos/{vid_key}/from_timestamp"]
                # Absolute timestamps in the source video for each kept frame.
                kept_timestamps = [
                    from_ts + (trim_start + i) / dataset.fps for i in range(kept_length)
                ]
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
                # Returns (N, C, H, W) float32 tensor with values in [0, 1].
                frames_tensor = decode_video_frames(
                    video_path, kept_timestamps, dataset.tolerance_s, dataset.video_backend
                )
                video_frames[vid_key] = frames_tensor.numpy()  # (N, C, H, W) float32

        # Feed frames one by one via the record-style add_frame API.
        for i in range(kept_length):
            abs_idx = kept_from_idx + i
            # hf_dataset[idx] applies hf_transform_to_torch → returns torch tensors.
            item = dataset.hf_dataset[abs_idx]

            frame: dict = {}

            # Required task string.
            frame["task"] = dataset.meta.tasks.iloc[item["task_index"].item()].name

            # Video features: CHW float32 [0, 1] numpy array.
            for vid_key in dataset.meta.video_keys:
                frame[vid_key] = video_frames[vid_key][i]

            # Image and numerical features.
            for key, feat in user_features.items():
                if feat["dtype"] == "video":
                    continue
                elif feat["dtype"] == "image":
                    # hf_transform_to_torch converts PIL → CHW float32 [0, 1].
                    # add_frame / write_image expects HWC uint8 [0, 255].
                    frame[key] = (item[key] * 255).byte().permute(1, 2, 0).numpy()
                else:
                    # Numerical: tensor → numpy (dtype is preserved by from_numpy).
                    frame[key] = item[key].numpy()

            write_target.add_frame(frame)

        write_target.save_episode()

    write_target.finalize()
    # Reload the HF dataset from disk so the returned object is fully readable.
    write_target._ensure_hf_dataset_loaded()

    total_trimmed = sum(ts + te for ts, te in episode_trim_specs.values())
    action = "Appended trimmed episodes to" if append_to_dataset is not None else "Created"
    logging.info(
        f"{action} trimmed dataset: {write_target.meta.total_episodes} episodes, "
        f"{write_target.meta.total_frames} frames "
        f"(removed {total_trimmed} frames from specified episodes)"
    )
    return write_target


def split_episodes(
    dataset: LeRobotDataset,
    episode_split_specs: dict[int, int],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Split episodes at specific frame positions into two separate episodes each.

    Creates a new dataset where each episode listed in *episode_split_specs* is split
    into two consecutive episodes at the given frame index.  Episodes not listed in
    *episode_split_specs* are copied unchanged.

    The frame at *split_frame* becomes the **first** frame of the second sub-episode
    (i.e. the split point is exclusive for the first part and inclusive for the second).

    Args:
        dataset: The source LeRobotDataset.
        episode_split_specs: Dict mapping episode index to the zero-based frame index at
            which to split the episode.  For an episode of length ``N``, valid split
            positions are ``1 … N-1`` (you need at least one frame in each half).
        output_dir: Directory to save the new dataset. If None, uses default location
            based on *repo_id*.
        repo_id: Repository ID for the new dataset. If None, appends ``"_split"`` to the
            source repo_id.

    Returns:
        :class:`LeRobotDataset` containing the resulting episodes.

    Example::

        # Split episode 2 at frame 15 (frames 0-14 → ep 2, frames 15-end → ep 3)
        new_dataset = split_episodes(
            dataset,
            episode_split_specs={2: 15},
            output_dir="./output",
        )
    """
    if not episode_split_specs:
        raise ValueError("No split specifications provided")

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = set(episode_split_specs.keys()) - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.meta.root)

    for ep_idx, split_frame in episode_split_specs.items():
        if split_frame <= 0:
            raise ValueError(
                f"split_frame must be >= 1 for episode {ep_idx}, got {split_frame}. "
                "The first part must contain at least one frame."
            )
        ep_length = dataset.meta.episodes[ep_idx]["length"]
        if split_frame >= ep_length:
            raise ValueError(
                f"split_frame {split_frame} is out of range for episode {ep_idx} "
                f"with length {ep_length}. "
                "The second part must contain at least one frame."
            )

    logging.info(f"Splitting {len(episode_split_specs)} episodes in dataset")

    user_features = {k: v for k, v in dataset.meta.features.items() if k not in DEFAULT_FEATURES}

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_split"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    write_target = LeRobotDataset.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=user_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    dataset._ensure_hf_dataset_loaded()

    has_video = len(dataset.meta.video_keys) > 0

    for ep_idx in tqdm(range(dataset.meta.total_episodes), desc="Splitting episodes"):
        ep_meta = dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        ep_length = to_idx - from_idx

        split_frame = episode_split_specs.get(ep_idx, None)

        # Determine the segments: a list of (start, end) relative frame offsets.
        if split_frame is not None:
            segments = [(0, split_frame), (split_frame, ep_length)]
        else:
            segments = [(0, ep_length)]

        video_frames: dict[str, np.ndarray] = {}
        if has_video:
            for vid_key in dataset.meta.video_keys:
                from_ts = ep_meta[f"videos/{vid_key}/from_timestamp"]
                kept_timestamps = [from_ts + i / dataset.fps for i in range(ep_length)]
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
                frames_tensor = decode_video_frames(
                    video_path, kept_timestamps, dataset.tolerance_s, dataset.video_backend
                )
                video_frames[vid_key] = frames_tensor.numpy()

        for seg_start, seg_end in segments:
            for i in range(seg_start, seg_end):
                abs_idx = from_idx + i
                item = dataset.hf_dataset[abs_idx]

                frame: dict = {}
                frame["task"] = dataset.meta.tasks.iloc[item["task_index"].item()].name

                for vid_key in dataset.meta.video_keys:
                    frame[vid_key] = video_frames[vid_key][i]

                for key, feat in user_features.items():
                    if feat["dtype"] == "video":
                        continue
                    elif feat["dtype"] == "image":
                        frame[key] = (item[key] * 255).byte().permute(1, 2, 0).numpy()
                    else:
                        frame[key] = item[key].numpy()

                write_target.add_frame(frame)

            write_target.save_episode()

    write_target.finalize()
    write_target._ensure_hf_dataset_loaded()

    n_new_episodes = write_target.meta.total_episodes
    logging.info(
        f"Created split dataset: {n_new_episodes} episodes, {write_target.meta.total_frames} frames "
        f"(split {len(episode_split_specs)} episode(s) into two)"
    )
    return write_target


def split_dataset(
    dataset: LeRobotDataset,
    splits: dict[str, float | list[int]],
    output_dir: str | Path | None = None,
) -> dict[str, LeRobotDataset]:
    """Split a LeRobotDataset into multiple smaller datasets.

    Args:
        dataset: The source LeRobotDataset to split.
        splits: Either a dict mapping split names to episode indices, or a dict mapping
                split names to fractions (must sum to <= 1.0).
        output_dir: Base directory for output datasets. If None, uses default location.

    Examples:
      Split by specific episodes
        splits = {"train": [0, 1, 2], "val": [3, 4]}
        datasets = split_dataset(dataset, splits)

      Split by fractions
        splits = {"train": 0.8, "val": 0.2}
        datasets = split_dataset(dataset, splits)
    """
    if not splits:
        raise ValueError("No splits provided")

    if all(isinstance(v, float) for v in splits.values()):
        splits = _fractions_to_episode_indices(dataset.meta.total_episodes, splits)

    all_episodes = set()
    for split_name, episodes in splits.items():
        if not episodes:
            raise ValueError(f"Split '{split_name}' has no episodes")
        episode_set = set(episodes)
        if episode_set & all_episodes:
            raise ValueError("Episodes cannot appear in multiple splits")
        all_episodes.update(episode_set)

    valid_indices = set(range(dataset.meta.total_episodes))
    invalid = all_episodes - valid_indices
    if invalid:
        raise ValueError(f"Invalid episode indices: {invalid}")

    if output_dir is not None:
        output_dir = Path(output_dir)

    result_datasets = {}

    for split_name, episodes in splits.items():
        logging.info(f"Creating split '{split_name}' with {len(episodes)} episodes")

        split_repo_id = f"{dataset.repo_id}_{split_name}"

        split_output_dir = (
            output_dir / split_name if output_dir is not None else HF_LEROBOT_HOME / split_repo_id
        )

        episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episodes))}

        new_meta = LeRobotDatasetMetadata.create(
            repo_id=split_repo_id,
            fps=dataset.meta.fps,
            features=dataset.meta.features,
            robot_type=dataset.meta.robot_type,
            root=split_output_dir,
            use_videos=len(dataset.meta.video_keys) > 0,
            chunks_size=dataset.meta.chunks_size,
            data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
            video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
        )

        video_metadata = None
        if dataset.meta.video_keys:
            video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping)

        data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)

        _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

        new_dataset = LeRobotDataset(
            repo_id=split_repo_id,
            root=split_output_dir,
            image_transforms=dataset.image_transforms,
            delta_timestamps=dataset.delta_timestamps,
            tolerance_s=dataset.tolerance_s,
        )

        result_datasets[split_name] = new_dataset

    return result_datasets


def merge_datasets(
    datasets: list[LeRobotDataset],
    output_repo_id: str,
    output_dir: str | Path | None = None,
) -> LeRobotDataset:
    """Merge multiple LeRobotDatasets into a single dataset.

    This is a wrapper around the aggregate_datasets functionality with a cleaner API.

    Args:
        datasets: List of LeRobotDatasets to merge.
        output_repo_id: Repository ID for the merged dataset.
        output_dir: Directory to save the merged dataset. If None, uses default location.
    """
    if not datasets:
        raise ValueError("No datasets to merge")

    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / output_repo_id

    repo_ids = [ds.repo_id for ds in datasets]
    roots = [ds.root for ds in datasets]

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_dir,
    )

    merged_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
        image_transforms=datasets[0].image_transforms,
        delta_timestamps=datasets[0].delta_timestamps,
        tolerance_s=datasets[0].tolerance_s,
    )

    return merged_dataset


def modify_features(
    dataset: LeRobotDataset,
    add_features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]] | None = None,
    remove_features: str | list[str] | None = None,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Modify a LeRobotDataset by adding and/or removing features in a single pass.

    This is the most efficient way to modify features, as it only copies the dataset once
    regardless of how many features are being added or removed.

    Args:
        dataset: The source LeRobotDataset.
        add_features: Optional dict mapping feature names to (feature_values, feature_info) tuples.
        remove_features: Optional feature name(s) to remove. Can be a single string or list.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        New dataset with features modified.

    Example:
        new_dataset = modify_features(
            dataset,
            add_features={
                "reward": (reward_array, {"dtype": "float32", "shape": [1], "names": None}),
            },
            remove_features=["old_feature"],
            output_dir="./output",
        )
    """
    if add_features is None and remove_features is None:
        raise ValueError("Must specify at least one of add_features or remove_features")

    remove_features_list: list[str] = []
    if remove_features is not None:
        remove_features_list = [remove_features] if isinstance(remove_features, str) else remove_features

    if add_features:
        required_keys = {"dtype", "shape"}
        for feature_name, (_, feature_info) in add_features.items():
            if feature_name in dataset.meta.features:
                raise ValueError(f"Feature '{feature_name}' already exists in dataset")

            if not required_keys.issubset(feature_info.keys()):
                raise ValueError(f"feature_info for '{feature_name}' must contain keys: {required_keys}")

    if remove_features_list:
        for name in remove_features_list:
            if name not in dataset.meta.features:
                raise ValueError(f"Feature '{name}' not found in dataset")

        required_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if any(name in required_features for name in remove_features_list):
            raise ValueError(f"Cannot remove required features: {required_features}")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    new_features = dataset.meta.features.copy()

    if remove_features_list:
        for name in remove_features_list:
            new_features.pop(name, None)

    if add_features:
        for feature_name, (_, feature_info) in add_features.items():
            new_features[feature_name] = feature_info

    video_keys_to_remove = [name for name in remove_features_list if name in dataset.meta.video_keys]
    remaining_video_keys = [k for k in dataset.meta.video_keys if k not in video_keys_to_remove]

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(remaining_video_keys) > 0,
    )

    _copy_data_with_feature_changes(
        dataset=dataset,
        new_meta=new_meta,
        add_features=add_features,
        remove_features=remove_features_list if remove_features_list else None,
    )

    if new_meta.video_keys:
        _copy_videos(dataset, new_meta, exclude_keys=video_keys_to_remove if video_keys_to_remove else None)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    return new_dataset


def add_features(
    dataset: LeRobotDataset,
    features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Add multiple features to a LeRobotDataset in a single pass.

    This is more efficient than calling add_feature() multiple times, as it only
    copies the dataset once regardless of how many features are being added.

    Args:
        dataset: The source LeRobotDataset.
        features: Dictionary mapping feature names to (feature_values, feature_info) tuples.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        New dataset with all features added.

    Example:
        features = {
            "task_embedding": (task_emb_array, {"dtype": "float32", "shape": [384], "names": None}),
            "cam1_embedding": (cam1_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
            "cam2_embedding": (cam2_emb_array, {"dtype": "float32", "shape": [768], "names": None}),
        }
        new_dataset = add_features(dataset, features, output_dir="./output", repo_id="my_dataset")
    """
    if not features:
        raise ValueError("No features provided")

    return modify_features(
        dataset=dataset,
        add_features=features,
        remove_features=None,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def remove_feature(
    dataset: LeRobotDataset,
    feature_names: str | list[str],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Remove features from a LeRobotDataset.

    Args:
        dataset: The source LeRobotDataset.
        feature_names: Name(s) of features to remove. Can be a single string or list.
        output_dir: Directory to save the new dataset. If None, uses default location.
        repo_id: Repository ID for the new dataset. If None, appends "_modified" to original.

    Returns:
        New dataset with features removed.
    """
    return modify_features(
        dataset=dataset,
        add_features=None,
        remove_features=feature_names,
        output_dir=output_dir,
        repo_id=repo_id,
    )


def _fractions_to_episode_indices(
    total_episodes: int,
    splits: dict[str, float],
) -> dict[str, list[int]]:
    """Convert split fractions to episode indices."""
    if sum(splits.values()) > 1.0:
        raise ValueError("Split fractions must sum to <= 1.0")

    indices = list(range(total_episodes))
    result = {}
    start_idx = 0

    for split_name, fraction in splits.items():
        num_episodes = int(total_episodes * fraction)
        if num_episodes == 0:
            logging.warning(f"Split '{split_name}' has no episodes, skipping...")
            continue
        end_idx = start_idx + num_episodes
        if split_name == list(splits.keys())[-1]:
            end_idx = total_episodes
        result[split_name] = indices[start_idx:end_idx]
        start_idx = end_idx

    return result


def _copy_and_reindex_data(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict]:
    """Copy and filter data files, only modifying files with deleted episodes.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices

    Returns:
        dict mapping episode index to its data file metadata (chunk_index, file_index, etc.)
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    file_to_episodes: dict[Path, set[int]] = {}
    for old_idx in episode_mapping:
        file_path = src_dataset.meta.get_data_file_path(old_idx)
        if file_path not in file_to_episodes:
            file_to_episodes[file_path] = set()
        file_to_episodes[file_path].add(old_idx)

    global_index = 0
    episode_data_metadata: dict[int, dict] = {}

    if dst_meta.tasks is None:
        all_task_indices = set()
        for src_path in file_to_episodes:
            df = pd.read_parquet(src_dataset.root / src_path)
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            task_series: pd.Series = df[mask]["task_index"]
            all_task_indices.update(task_series.unique().tolist())
        tasks = [src_dataset.meta.tasks.iloc[idx].name for idx in all_task_indices]
        dst_meta.save_episode_tasks(list(set(tasks)))

    task_mapping = {}
    for old_task_idx in range(len(src_dataset.meta.tasks)):
        task_name = src_dataset.meta.tasks.iloc[old_task_idx].name
        new_task_idx = dst_meta.get_task_index(task_name)
        if new_task_idx is not None:
            task_mapping[old_task_idx] = new_task_idx

    for src_path in tqdm(sorted(file_to_episodes.keys()), desc="Processing data files"):
        df = pd.read_parquet(src_dataset.root / src_path)

        all_episodes_in_file = set(df["episode_index"].unique())
        episodes_to_keep = file_to_episodes[src_path]

        if all_episodes_in_file == episodes_to_keep:
            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]
        else:
            mask = df["episode_index"].isin(list(episode_mapping.keys()))
            df = df[mask].copy().reset_index(drop=True)

            if len(df) == 0:
                continue

            df["episode_index"] = df["episode_index"].replace(episode_mapping)
            df["index"] = range(global_index, global_index + len(df))
            df["task_index"] = df["task_index"].replace(task_mapping)

            first_ep_old_idx = min(episodes_to_keep)
            src_ep = src_dataset.meta.episodes[first_ep_old_idx]
            chunk_idx = src_ep["data/chunk_index"]
            file_idx = src_ep["data/file_index"]

        dst_path = dst_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, dst_meta)

        for ep_old_idx in episodes_to_keep:
            ep_new_idx = episode_mapping[ep_old_idx]
            ep_df = df[df["episode_index"] == ep_new_idx]
            episode_data_metadata[ep_new_idx] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": int(ep_df["index"].min()),
                "dataset_to_index": int(ep_df["index"].max() + 1),
            }

        global_index += len(df)

    return episode_data_metadata


def _keep_episodes_from_video_with_av(
    input_path: Path,
    output_path: Path,
    episodes_to_keep: list[tuple[int, int]],
    fps: float,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    """Keep only specified episodes from a video file using PyAV.

    This function decodes frames from specified frame ranges and re-encodes them with
    properly reset timestamps to ensure monotonic progression.

    Args:
        input_path: Source video file path.
        output_path: Destination video file path.
        episodes_to_keep: List of (start_frame, end_frame) tuples for episodes to keep.
            Ranges are half-open intervals: [start_frame, end_frame), where start_frame
            is inclusive and end_frame is exclusive.
        fps: Frame rate of the video.
        vcodec: Video codec to use for encoding.
        pix_fmt: Pixel format for output video.
    """
    from fractions import Fraction

    import av

    if not episodes_to_keep:
        raise ValueError("No episodes to keep")

    in_container = av.open(str(input_path))

    # Check if video stream exists.
    if not in_container.streams.video:
        raise ValueError(
            f"No video streams found in {input_path}. "
            "The video file may be corrupted or empty. "
            "Try re-downloading the dataset or checking the video file."
        )

    v_in = in_container.streams.video[0]

    out = av.open(str(output_path), mode="w")

    # Convert fps to Fraction for PyAV compatibility.
    fps_fraction = Fraction(fps).limit_denominator(1000)
    v_out = out.add_stream(vcodec, rate=fps_fraction)

    # PyAV type stubs don't distinguish video streams from audio/subtitle streams.
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = pix_fmt

    # Set time_base to match the frame rate for proper timestamp handling.
    v_out.time_base = Fraction(1, int(fps))

    out.start_encoding()

    # Create set of (start, end) ranges for fast lookup.
    # Convert to a sorted list for efficient checking.
    frame_ranges = sorted(episodes_to_keep)

    # Track frame index for setting PTS and current range being processed.
    src_frame_count = 0
    frame_count = 0
    range_idx = 0

    # Read through entire video once and filter frames.
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue

            # Check if frame is in any of our desired frame ranges.
            # Skip ranges that have already passed.
            while range_idx < len(frame_ranges) and src_frame_count >= frame_ranges[range_idx][1]:
                range_idx += 1

            # If we've passed all ranges, stop processing.
            if range_idx >= len(frame_ranges):
                break

            # Check if frame is in current range.
            start_frame = frame_ranges[range_idx][0]

            if src_frame_count < start_frame:
                src_frame_count += 1
                continue

            # Frame is in range - create a new frame with reset timestamps.
            # We need to create a copy to avoid modifying the original.
            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = frame_count
            new_frame.time_base = Fraction(1, int(fps))

            # Encode and mux the frame.
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)

            src_frame_count += 1
            frame_count += 1

    # Flush encoder.
    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


def _copy_and_reindex_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> dict[int, dict]:
    """Copy and filter video files, only re-encoding files with deleted episodes.

    For video files that only contain kept episodes, we copy them directly.
    For files with mixed kept/deleted episodes, we use PyAV filters to efficiently
    re-encode only the desired segments.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices

    Returns:
        dict mapping episode index to its video metadata (chunk_index, file_index, timestamps)
    """
    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    episodes_video_metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}

    for video_key in src_dataset.meta.video_keys:
        logging.info(f"Processing videos for {video_key}")

        if dst_meta.video_path is None:
            raise ValueError("Destination metadata has no video_path defined")

        file_to_episodes: dict[tuple[int, int], list[int]] = {}
        for old_idx in episode_mapping:
            src_ep = src_dataset.meta.episodes[old_idx]
            chunk_idx = src_ep[f"videos/{video_key}/chunk_index"]
            file_idx = src_ep[f"videos/{video_key}/file_index"]
            file_key = (chunk_idx, file_idx)
            if file_key not in file_to_episodes:
                file_to_episodes[file_key] = []
            file_to_episodes[file_key].append(old_idx)

        for (src_chunk_idx, src_file_idx), episodes_in_file in tqdm(
            sorted(file_to_episodes.items()), desc=f"Processing {video_key} video files"
        ):
            all_episodes_in_file = [
                ep_idx
                for ep_idx in range(src_dataset.meta.total_episodes)
                if src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/chunk_index") == src_chunk_idx
                and src_dataset.meta.episodes[ep_idx].get(f"videos/{video_key}/file_index") == src_file_idx
            ]

            episodes_to_keep_set = set(episodes_in_file)
            all_in_file_set = set(all_episodes_in_file)

            if all_in_file_set == episodes_to_keep_set:
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_video_path, dst_video_path)

                for old_idx in episodes_in_file:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = src_ep[
                        f"videos/{video_key}/from_timestamp"
                    ]
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = src_ep[
                        f"videos/{video_key}/to_timestamp"
                    ]
            else:
                # Build list of frame ranges to keep, in sorted order.
                sorted_keep_episodes = sorted(episodes_in_file, key=lambda x: episode_mapping[x])
                episodes_to_keep_ranges: list[tuple[int, int]] = []
                for old_idx in sorted_keep_episodes:
                    src_ep = src_dataset.meta.episodes[old_idx]
                    from_frame = round(src_ep[f"videos/{video_key}/from_timestamp"] * src_dataset.meta.fps)
                    to_frame = round(src_ep[f"videos/{video_key}/to_timestamp"] * src_dataset.meta.fps)
                    assert src_ep["length"] == to_frame - from_frame, (
                        f"Episode length mismatch: {src_ep['length']} vs {to_frame - from_frame}"
                    )
                    episodes_to_keep_ranges.append((from_frame, to_frame))

                # Use PyAV filters to efficiently re-encode only the desired segments.
                assert src_dataset.meta.video_path is not None
                src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path = dst_meta.root / dst_meta.video_path.format(
                    video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
                )
                dst_video_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(
                    f"Re-encoding {video_key} (chunk {src_chunk_idx}, file {src_file_idx}) "
                    f"with {len(episodes_to_keep_ranges)} episodes"
                )
                _keep_episodes_from_video_with_av(
                    src_video_path,
                    dst_video_path,
                    episodes_to_keep_ranges,
                    src_dataset.meta.fps,
                    vcodec,
                    pix_fmt,
                )

                cumulative_ts = 0.0
                for old_idx in sorted_keep_episodes:
                    new_idx = episode_mapping[old_idx]
                    src_ep = src_dataset.meta.episodes[old_idx]
                    ep_length = src_ep["length"]
                    ep_duration = ep_length / src_dataset.meta.fps

                    episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                    episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = cumulative_ts
                    episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = (
                        cumulative_ts + ep_duration
                    )

                    cumulative_ts += ep_duration

    return episodes_video_metadata


def _copy_and_reindex_episodes_metadata(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    data_metadata: dict[int, dict],
    video_metadata: dict[int, dict] | None = None,
) -> None:
    """Copy and reindex episodes metadata using provided data and video metadata.

    Args:
        src_dataset: Source dataset to copy from
        dst_meta: Destination metadata object
        episode_mapping: Mapping from old episode indices to new indices
        data_metadata: Dict mapping new episode index to its data file metadata
        video_metadata: Optional dict mapping new episode index to its video metadata
    """
    from lerobot.datasets.utils import flatten_dict

    if src_dataset.meta.episodes is None:
        src_dataset.meta.episodes = load_episodes(src_dataset.meta.root)

    all_stats = []
    total_frames = 0

    for old_idx, new_idx in tqdm(
        sorted(episode_mapping.items(), key=lambda x: x[1]), desc="Processing episodes metadata"
    ):
        src_episode_full = _load_episode_with_stats(src_dataset, old_idx)

        src_episode = src_dataset.meta.episodes[old_idx]

        episode_meta = data_metadata[new_idx].copy()

        if video_metadata and new_idx in video_metadata:
            episode_meta.update(video_metadata[new_idx])

        # Extract episode statistics from parquet metadata.
        # Note (maractingi): When pandas/pyarrow serializes numpy arrays with shape (3, 1, 1) to parquet,
        # they are being deserialized as nested object arrays like:
        #   array([array([array([0.])]), array([array([0.])]), array([array([0.])])])
        # This happens particularly with image/video statistics. We need to detect and flatten
        # these nested structures back to proper (3, 1, 1) arrays so aggregate_stats can process them.
        episode_stats = {}
        for key in src_episode_full:
            if key.startswith("stats/"):
                stat_key = key.replace("stats/", "")
                parts = stat_key.split("/")
                if len(parts) == 2:
                    feature_name, stat_name = parts
                    if feature_name not in episode_stats:
                        episode_stats[feature_name] = {}

                    value = src_episode_full[key]

                    if feature_name in src_dataset.meta.features:
                        feature_dtype = src_dataset.meta.features[feature_name]["dtype"]
                        if feature_dtype in ["image", "video"] and stat_name != "count":
                            if isinstance(value, np.ndarray) and value.dtype == object:
                                flat_values = []
                                for item in value:
                                    while isinstance(item, np.ndarray):
                                        item = item.flatten()[0]
                                    flat_values.append(item)
                                value = np.array(flat_values, dtype=np.float64).reshape(3, 1, 1)
                            elif isinstance(value, np.ndarray) and value.shape == (3,):
                                value = value.reshape(3, 1, 1)

                    episode_stats[feature_name][stat_name] = value

        all_stats.append(episode_stats)

        episode_dict = {
            "episode_index": new_idx,
            "tasks": src_episode["tasks"],
            "length": src_episode["length"],
        }
        episode_dict.update(episode_meta)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        dst_meta._save_episode_metadata(episode_dict)

        total_frames += src_episode["length"]

    dst_meta._close_writer()

    dst_meta.info.update(
        {
            "total_episodes": len(episode_mapping),
            "total_frames": total_frames,
            "total_tasks": len(dst_meta.tasks) if dst_meta.tasks is not None else 0,
            "splits": {"train": f"0:{len(episode_mapping)}"},
        }
    )
    write_info(dst_meta.info, dst_meta.root)

    if not all_stats:
        logging.warning("No statistics found to aggregate")
        return

    logging.info(f"Aggregating statistics for {len(all_stats)} episodes")
    aggregated_stats = aggregate_stats(all_stats)
    filtered_stats = {k: v for k, v in aggregated_stats.items() if k in dst_meta.features}
    write_stats(filtered_stats, dst_meta.root)


def _write_parquet(df: pd.DataFrame, path: Path, meta: LeRobotDatasetMetadata) -> None:
    """Write DataFrame to parquet

    This ensures images are properly embedded and the file can be loaded correctly by HF datasets.
    """
    from lerobot.datasets.utils import embed_images, get_hf_features_from_features

    hf_features = get_hf_features_from_features(meta.features)
    ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")

    if len(meta.image_keys) > 0:
        ep_dataset = embed_images(ep_dataset)

    table = ep_dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _save_data_chunk(
    df: pd.DataFrame,
    meta: LeRobotDatasetMetadata,
    chunk_idx: int = 0,
    file_idx: int = 0,
) -> tuple[int, int, dict[int, dict]]:
    """Save a data chunk and return updated indices and episode metadata.

    Returns:
        tuple: (next_chunk_idx, next_file_idx, episode_metadata_dict)
            where episode_metadata_dict maps episode_index to its data file metadata
    """
    path = meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    _write_parquet(df, path, meta)

    episode_metadata = {}
    for ep_idx in df["episode_index"].unique():
        ep_df = df[df["episode_index"] == ep_idx]
        episode_metadata[ep_idx] = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": int(ep_df["index"].min()),
            "dataset_to_index": int(ep_df["index"].max() + 1),
        }

    file_size = get_parquet_file_size_in_mb(path)
    if file_size >= DEFAULT_DATA_FILE_SIZE_IN_MB * 0.9:
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    return chunk_idx, file_idx, episode_metadata


def _copy_data_with_feature_changes(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    add_features: dict[str, tuple] | None = None,
    remove_features: list[str] | None = None,
) -> None:
    """Copy data while adding or removing features."""
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    frame_idx = 0

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        relative_path = src_path.relative_to(dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]

        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        if remove_features:
            df = df.drop(columns=remove_features, errors="ignore")

        if add_features:
            end_idx = frame_idx + len(df)
            for feature_name, (values, _) in add_features.items():
                if callable(values):
                    feature_values = []
                    for _, row in df.iterrows():
                        ep_idx = row["episode_index"]
                        frame_in_ep = row["frame_index"]
                        value = values(row.to_dict(), ep_idx, frame_in_ep)
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        feature_values.append(value)
                    df[feature_name] = feature_values
                else:
                    feature_slice = values[frame_idx:end_idx]
                    if len(feature_slice.shape) > 1 and feature_slice.shape[1] == 1:
                        df[feature_name] = feature_slice.flatten()
                    else:
                        df[feature_name] = feature_slice
            frame_idx = end_idx

        # Write using the same chunk/file structure as source
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, new_meta)

    _copy_episodes_metadata_and_stats(dataset, new_meta)


def _copy_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    exclude_keys: list[str] | None = None,
) -> None:
    """Copy video files, optionally excluding certain keys."""
    if exclude_keys is None:
        exclude_keys = []

    for video_key in src_dataset.meta.video_keys:
        if video_key in exclude_keys:
            continue

        video_files = set()
        for ep_idx in range(len(src_dataset.meta.episodes)):
            try:
                video_files.add(src_dataset.meta.get_video_file_path(ep_idx, video_key))
            except KeyError:
                continue

        for src_path in tqdm(sorted(video_files), desc=f"Copying {video_key} videos"):
            dst_path = dst_meta.root / src_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dataset.root / src_path, dst_path)


def _copy_episodes_metadata_and_stats(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
) -> None:
    """Copy episodes metadata and recalculate stats."""
    if src_dataset.meta.tasks is not None:
        write_tasks(src_dataset.meta.tasks, dst_meta.root)
        dst_meta.tasks = src_dataset.meta.tasks.copy()

    episodes_dir = src_dataset.root / "meta/episodes"
    dst_episodes_dir = dst_meta.root / "meta/episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, dst_episodes_dir, dirs_exist_ok=True)

    dst_meta.info.update(
        {
            "total_episodes": src_dataset.meta.total_episodes,
            "total_frames": src_dataset.meta.total_frames,
            "total_tasks": src_dataset.meta.total_tasks,
            "splits": src_dataset.meta.info.get("splits", {"train": f"0:{src_dataset.meta.total_episodes}"}),
        }
    )

    if dst_meta.video_keys and src_dataset.meta.video_keys:
        for key in dst_meta.video_keys:
            if key in src_dataset.meta.features:
                dst_meta.info["features"][key]["info"] = src_dataset.meta.info["features"][key].get(
                    "info", {}
                )

    write_info(dst_meta.info, dst_meta.root)

    if set(dst_meta.features.keys()) != set(src_dataset.meta.features.keys()):
        logging.info("Recalculating dataset statistics...")
        if src_dataset.meta.stats:
            new_stats = {}
            for key in dst_meta.features:
                if key in src_dataset.meta.stats:
                    new_stats[key] = src_dataset.meta.stats[key]
            write_stats(new_stats, dst_meta.root)
    else:
        if src_dataset.meta.stats:
            write_stats(src_dataset.meta.stats, dst_meta.root)


def _save_episode_images_for_video(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    img_key: str,
    episode_index: int,
    num_workers: int = 4,
) -> None:
    """Save images from a specific episode and camera to disk for video encoding.

    Args:
        dataset: The LeRobot dataset to extract images from
        imgs_dir: Directory to save images to
        img_key: The image key (camera) to extract
        episode_index: Index of the episode to save
        num_workers: Number of threads for parallel image saving
    """
    # Create directory
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset without torch format for PIL image access
    hf_dataset = dataset.hf_dataset.with_format(None)

    # Select only this camera's images
    imgs_dataset = hf_dataset.select_columns(img_key)

    # Get episode start and end indices
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    # Get all items for this episode
    episode_dataset = imgs_dataset.select(range(from_idx, to_idx))

    # Define function to save a single image
    def save_single_image(i_item_tuple):
        i, item = i_item_tuple
        img = item[img_key]
        # Use frame-XXXXXX.png format to match encode_video_frames expectations
        img.save(str(imgs_dir / f"frame-{i:06d}.png"), quality=100)
        return i

    # Save images with proper naming convention for encode_video_frames (frame-XXXXXX.png)
    items = list(enumerate(episode_dataset))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_single_image, item) for item in items]
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred


def _save_batch_episodes_images(
    dataset: LeRobotDataset,
    imgs_dir: Path,
    img_key: str,
    episode_indices: list[int],
    num_workers: int = 4,
) -> list[float]:
    """Save images from multiple episodes to disk for batch video encoding.

    Args:
        dataset: The LeRobot dataset to extract images from
        imgs_dir: Directory to save images to
        img_key: The image key (camera) to extract
        episode_indices: List of episode indices to save
        num_workers: Number of threads for parallel image saving

    Returns:
        List of episode durations in seconds
    """
    imgs_dir.mkdir(parents=True, exist_ok=True)
    hf_dataset = dataset.hf_dataset.with_format(None)
    imgs_dataset = hf_dataset.select_columns(img_key)

    # Define function to save a single image with global frame index
    # Defined once outside the loop to avoid repeated closure creation
    def save_single_image(i_item_tuple, base_frame_idx, img_key_param):
        i, item = i_item_tuple
        img = item[img_key_param]
        # Use global frame index for naming
        img.save(str(imgs_dir / f"frame-{base_frame_idx + i:06d}.png"), quality=100)
        return i

    episode_durations = []
    frame_idx = 0

    for ep_idx in episode_indices:
        # Get episode range
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        episode_length = to_idx - from_idx
        episode_durations.append(episode_length / dataset.fps)

        # Get episode images
        episode_dataset = imgs_dataset.select(range(from_idx, to_idx))

        # Save images
        items = list(enumerate(episode_dataset))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(save_single_image, item, frame_idx, img_key) for item in items]
            for future in as_completed(futures):
                future.result()

        frame_idx += episode_length

    return episode_durations


def _iter_episode_batches(
    episode_indices: list[int],
    episode_lengths: dict[int, int],
    size_per_frame_mb: float,
    video_file_size_limit: float,
    max_episodes: int | None,
    max_frames: int | None,
):
    """Generator that yields batches of episode indices for video encoding.

    Groups episodes into batches that respect size and memory constraints:
    - Stays under video file size limit
    - Respects maximum episodes per batch (if specified)
    - Respects maximum frames per batch (if specified)

    Args:
        episode_indices: List of episode indices to batch
        episode_lengths: Dictionary mapping episode index to episode length
        size_per_frame_mb: Estimated size per frame in MB
        video_file_size_limit: Maximum video file size in MB
        max_episodes: Maximum number of episodes per batch (None = no limit)
        max_frames: Maximum number of frames per batch (None = no limit)

    Yields:
        List of episode indices for each batch
    """
    batch_episodes = []
    estimated_size = 0.0
    total_frames = 0

    for ep_idx in episode_indices:
        ep_length = episode_lengths[ep_idx]
        ep_estimated_size = ep_length * size_per_frame_mb

        # we check if adding this episode would exceed any constraint
        would_exceed_size = estimated_size > 0 and estimated_size + ep_estimated_size >= video_file_size_limit
        would_exceed_episodes = max_episodes is not None and len(batch_episodes) >= max_episodes
        would_exceed_frames = max_frames is not None and total_frames + ep_length > max_frames

        if batch_episodes and (would_exceed_size or would_exceed_episodes or would_exceed_frames):
            # yield current batch before adding this episode
            yield batch_episodes
            # start a new batch with current episode
            batch_episodes = [ep_idx]
            estimated_size = ep_estimated_size
            total_frames = ep_length
        else:
            # add to current batch
            batch_episodes.append(ep_idx)
            estimated_size += ep_estimated_size
            total_frames += ep_length

    # yield final batch if not empty
    if batch_episodes:
        yield batch_episodes


def _estimate_frame_size_via_calibration(
    dataset: LeRobotDataset,
    img_key: str,
    episode_indices: list[int],
    temp_dir: Path,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    g: int,
    crf: int,
    fast_decode: int,
    num_calibration_frames: int = 30,
) -> float:
    """Estimate MB per frame by encoding a small calibration sample.

    Encodes a representative sample of frames using the exact codec parameters
    to measure actual compression ratio, which is more accurate than heuristics.

    Args:
        dataset: Source dataset with images.
        img_key: Image key to calibrate (e.g., "observation.images.top").
        episode_indices: List of episode indices being processed.
        temp_dir: Temporary directory for calibration files.
        fps: Frames per second for video encoding.
        vcodec: Video codec (libsvtav1, h264, hevc).
        pix_fmt: Pixel format (yuv420p, etc.).
        g: GOP size (group of pictures).
        crf: Constant Rate Factor (quality).
        fast_decode: Fast decode tuning parameter.
        num_calibration_frames: Number of frames to use for calibration (default: 30).

    Returns:
        Estimated size in MB per frame based on actual encoding.
    """
    calibration_dir = temp_dir / "calibration" / img_key
    calibration_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Select a representative episode (prefer middle episode if available)
        calibration_ep_idx = episode_indices[len(episode_indices) // 2]

        # Get episode range
        from_idx = dataset.meta.episodes["dataset_from_index"][calibration_ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][calibration_ep_idx]
        episode_length = to_idx - from_idx

        # Use up to num_calibration_frames from this episode
        num_frames = min(num_calibration_frames, episode_length)

        # Get frames from dataset
        hf_dataset = dataset.hf_dataset.with_format(None)
        sample_indices = range(from_idx, from_idx + num_frames)

        # Save calibration frames
        for i, idx in enumerate(sample_indices):
            img = hf_dataset[idx][img_key]
            img.save(str(calibration_dir / f"frame-{i:06d}.png"), quality=100)

        # Encode calibration video
        calibration_video_path = calibration_dir / "calibration.mp4"
        encode_video_frames(
            imgs_dir=calibration_dir,
            video_path=calibration_video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
            fast_decode=fast_decode,
            overwrite=True,
        )

        # Measure actual compressed size
        video_size_bytes = calibration_video_path.stat().st_size
        video_size_mb = video_size_bytes / BYTES_PER_MIB
        size_per_frame_mb = video_size_mb / num_frames

        logging.info(
            f"  Calibration: {num_frames} frames -> {video_size_mb:.2f} MB "
            f"= {size_per_frame_mb:.4f} MB/frame for {img_key}"
        )

        return size_per_frame_mb

    finally:
        # Clean up calibration files
        if calibration_dir.exists():
            shutil.rmtree(calibration_dir)


def _copy_data_without_images(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_indices: list[int],
    img_keys: list[str],
) -> None:
    """Copy data files without image columns.

    Args:
        src_dataset: Source dataset
        dst_meta: Destination metadata
        episode_indices: Episodes to include
        img_keys: Image keys to remove
    """
    from lerobot.datasets.utils import DATA_DIR

    data_dir = src_dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    episode_set = set(episode_indices)

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        # Filter to only include selected episodes
        df = df[df["episode_index"].isin(episode_set)].copy()

        if len(df) == 0:
            continue

        # Remove image columns
        columns_to_drop = [col for col in img_keys if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        # Get chunk and file indices from path
        relative_path = src_path.relative_to(src_dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]
        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        # Write to destination without pandas index
        dst_path = dst_meta.root / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_path, index=False)


# Video conversion constants
BYTES_PER_KIB = 1024
BYTES_PER_MIB = BYTES_PER_KIB * BYTES_PER_KIB


def modify_tasks(
    dataset: LeRobotDataset,
    new_task: str | None = None,
    episode_tasks: dict[int, str] | None = None,
) -> LeRobotDataset:
    """Modify tasks in a LeRobotDataset.

    This function allows you to either:
    1. Set a single task for the entire dataset (using `new_task`)
    2. Set specific tasks for specific episodes (using `episode_tasks`)

    You can combine both: `new_task` sets the default, and `episode_tasks` overrides
    specific episodes.

    The dataset is modified in-place, updating only the task-related files:
    - meta/tasks.parquet
    - data/**/*.parquet (task_index column)
    - meta/episodes/**/*.parquet (tasks column)
    - meta/info.json (total_tasks)

    Args:
        dataset: The source LeRobotDataset to modify.
        new_task: A single task string to apply to all episodes. If None and episode_tasks
            is also None, raises an error.
        episode_tasks: Optional dict mapping episode indices to their task strings.
            Overrides `new_task` for specific episodes.


    Examples:
        Set a single task for all episodes:
            dataset = modify_tasks(dataset, new_task="Pick up the cube")

        Set different tasks for specific episodes:
            dataset = modify_tasks(
                dataset,
                episode_tasks={0: "Task A", 1: "Task B", 2: "Task A"}
            )

        Set a default task with overrides:
            dataset = modify_tasks(
                dataset,
                new_task="Default task",
                episode_tasks={5: "Special task for episode 5"}
            )
    """
    if new_task is None and episode_tasks is None:
        raise ValueError("Must specify at least one of new_task or episode_tasks")

    if episode_tasks is not None:
        valid_indices = set(range(dataset.meta.total_episodes))
        invalid = set(episode_tasks.keys()) - valid_indices
        if invalid:
            raise ValueError(f"Invalid episode indices: {invalid}")

    # Ensure episodes metadata is loaded
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    # Build the mapping from episode index to task string
    episode_to_task: dict[int, str] = {}
    for ep_idx in range(dataset.meta.total_episodes):
        if episode_tasks and ep_idx in episode_tasks:
            episode_to_task[ep_idx] = episode_tasks[ep_idx]
        elif new_task is not None:
            episode_to_task[ep_idx] = new_task
        else:
            # Keep original task if not overridden and no default provided
            original_tasks = dataset.meta.episodes[ep_idx]["tasks"]
            if not original_tasks:
                raise ValueError(f"Episode {ep_idx} has no tasks and no default task was provided")
            episode_to_task[ep_idx] = original_tasks[0]

    # Collect all unique tasks and create new task mapping
    unique_tasks = sorted(set(episode_to_task.values()))
    new_task_df = pd.DataFrame({"task_index": list(range(len(unique_tasks)))}, index=unique_tasks)
    task_to_index = {task: idx for idx, task in enumerate(unique_tasks)}

    logging.info(f"Modifying tasks in {dataset.repo_id}")
    logging.info(f"New tasks: {unique_tasks}")

    root = dataset.root

    # Update data files - modify task_index column
    logging.info("Updating data files...")
    data_dir = root / DATA_DIR

    for parquet_path in tqdm(sorted(data_dir.rglob("*.parquet")), desc="Updating data"):
        df = pd.read_parquet(parquet_path)

        # Build a mapping from episode_index to new task_index for rows in this file
        episode_indices_in_file = df["episode_index"].unique()
        ep_to_new_task_idx = {
            ep_idx: task_to_index[episode_to_task[ep_idx]] for ep_idx in episode_indices_in_file
        }

        # Update task_index column
        df["task_index"] = df["episode_index"].map(ep_to_new_task_idx)
        df.to_parquet(parquet_path, index=False)

    # Update episodes metadata - modify tasks column
    logging.info("Updating episodes metadata...")
    episodes_dir = root / "meta" / "episodes"

    for parquet_path in tqdm(sorted(episodes_dir.rglob("*.parquet")), desc="Updating episodes"):
        df = pd.read_parquet(parquet_path)

        # Update tasks column
        df["tasks"] = df["episode_index"].apply(lambda ep_idx: [episode_to_task[ep_idx]])
        df.to_parquet(parquet_path, index=False)

    # Write new tasks.parquet
    write_tasks(new_task_df, root)

    # Update info.json
    dataset.meta.info["total_tasks"] = len(unique_tasks)
    write_info(dataset.meta.info, root)

    # Reload metadata to reflect changes
    dataset.meta.tasks = new_task_df
    dataset.meta.episodes = load_episodes(root)

    logging.info(f"Tasks: {unique_tasks}")

    return dataset


def convert_image_to_video_dataset(
    dataset: LeRobotDataset,
    output_dir: Path,
    repo_id: str | None = None,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    fast_decode: int = 0,
    episode_indices: list[int] | None = None,
    num_workers: int = 4,
    max_episodes_per_batch: int | None = None,
    max_frames_per_batch: int | None = None,
) -> LeRobotDataset:
    """Convert image-to-video dataset.

    Creates a new LeRobotDataset with images encoded as videos, following the proper
    LeRobot dataset structure with videos stored in chunked MP4 files.

    Args:
        dataset: The source LeRobot dataset with images
        output_dir: Directory to save the new video dataset
        repo_id: Repository ID for the new dataset (default: original_id + "_video")
        vcodec: Video codec (default: libsvtav1)
        pix_fmt: Pixel format (default: yuv420p)
        g: Group of pictures size (default: 2)
        crf: Constant rate factor (default: 30)
        fast_decode: Fast decode tuning (default: 0)
        episode_indices: List of episode indices to convert (None = all episodes)
        num_workers: Number of threads for parallel processing (default: 4)
        max_episodes_per_batch: Maximum episodes per video batch to avoid memory issues (None = no limit)
        max_frames_per_batch: Maximum frames per video batch to avoid memory issues (None = no limit)

    Returns:
        New LeRobotDataset with images encoded as videos
    """
    # Check that it's an image dataset
    if len(dataset.meta.video_keys) > 0:
        raise ValueError(
            f"This operation is for image datasets only. Video dataset provided: {dataset.repo_id}"
        )

    # Get all image keys
    hf_dataset = dataset.hf_dataset.with_format(None)
    img_keys = [key for key in hf_dataset.features if key.startswith(OBS_IMAGE)]

    if len(img_keys) == 0:
        raise ValueError(f"No image keys found in dataset {dataset.repo_id}")

    # Determine which episodes to process
    if episode_indices is None:
        episode_indices = list(range(dataset.meta.total_episodes))

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_video"

    logging.info(
        f"Converting {len(episode_indices)} episodes with {len(img_keys)} cameras from {dataset.repo_id}"
    )
    logging.info(f"Video codec: {vcodec}, pixel format: {pix_fmt}, GOP: {g}, CRF: {crf}")

    # Create new features dict, converting image features to video features
    new_features = {}
    for key, value in dataset.meta.features.items():
        if key not in img_keys:
            new_features[key] = value
        else:
            # Convert image key to video format
            new_features[key] = value.copy()
            new_features[key]["dtype"] = "video"  # Change dtype from "image" to "video"
            # Video info will be updated after episodes are encoded

    # Create new metadata for video dataset
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Create temporary directory for image extraction
    temp_dir = output_dir / "temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process all episodes and batch encode videos
    # Use dictionary for O(1) episode metadata lookups instead of O(n) linear search
    all_episode_metadata = {}
    fps = int(dataset.fps)

    try:
        # Build episode metadata entries first
        logging.info("Building episode metadata...")
        cumulative_frame_idx = 0
        for ep_idx in episode_indices:
            src_episode = dataset.meta.episodes[ep_idx]
            ep_length = src_episode["length"]
            ep_meta = {
                "episode_index": ep_idx,
                "length": ep_length,
                "dataset_from_index": cumulative_frame_idx,
                "dataset_to_index": cumulative_frame_idx + ep_length,
            }
            if "data/chunk_index" in src_episode:
                ep_meta["data/chunk_index"] = src_episode["data/chunk_index"]
                ep_meta["data/file_index"] = src_episode["data/file_index"]
            all_episode_metadata[ep_idx] = ep_meta
            cumulative_frame_idx += ep_length

        # Process each camera and batch encode multiple episodes together
        video_file_size_limit = new_meta.video_files_size_in_mb

        # Pre-compute episode lengths for batching
        episode_lengths = {ep_idx: dataset.meta.episodes["length"][ep_idx] for ep_idx in episode_indices}

        for img_key in tqdm(img_keys, desc="Processing cameras"):
            # Estimate size per frame by encoding a small calibration sample
            # This provides accurate compression ratio for the specific codec parameters
            size_per_frame_mb = _estimate_frame_size_via_calibration(
                dataset=dataset,
                img_key=img_key,
                episode_indices=episode_indices,
                temp_dir=temp_dir,
                fps=fps,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
                g=g,
                crf=crf,
                fast_decode=fast_decode,
            )

            logging.info(f"Processing camera: {img_key}")
            chunk_idx, file_idx = 0, 0
            cumulative_timestamp = 0.0

            # Process episodes in batches to stay under size limit
            for batch_episodes in _iter_episode_batches(
                episode_indices=episode_indices,
                episode_lengths=episode_lengths,
                size_per_frame_mb=size_per_frame_mb,
                video_file_size_limit=video_file_size_limit,
                max_episodes=max_episodes_per_batch,
                max_frames=max_frames_per_batch,
            ):
                total_frames_in_batch = sum(episode_lengths[idx] for idx in batch_episodes)
                logging.info(
                    f"  Encoding batch of {len(batch_episodes)} episodes "
                    f"({batch_episodes[0]}-{batch_episodes[-1]}) = {total_frames_in_batch} frames"
                )

                # Save images for all episodes in this batch
                imgs_dir = temp_dir / f"batch_{chunk_idx}_{file_idx}" / img_key
                episode_durations = _save_batch_episodes_images(
                    dataset=dataset,
                    imgs_dir=imgs_dir,
                    img_key=img_key,
                    episode_indices=batch_episodes,
                    num_workers=num_workers,
                )

                # Encode all batched episodes into single video
                video_path = new_meta.root / new_meta.video_path.format(
                    video_key=img_key, chunk_index=chunk_idx, file_index=file_idx
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)

                encode_video_frames(
                    imgs_dir=imgs_dir,
                    video_path=video_path,
                    fps=fps,
                    vcodec=vcodec,
                    pix_fmt=pix_fmt,
                    g=g,
                    crf=crf,
                    fast_decode=fast_decode,
                    overwrite=True,
                )

                # Clean up temporary images
                shutil.rmtree(imgs_dir)

                # Update metadata for each episode in the batch
                for ep_idx, duration in zip(batch_episodes, episode_durations, strict=True):
                    from_timestamp = cumulative_timestamp
                    to_timestamp = cumulative_timestamp + duration
                    cumulative_timestamp = to_timestamp

                    # Find episode metadata entry and add video metadata (O(1) dictionary lookup)
                    ep_meta = all_episode_metadata[ep_idx]
                    ep_meta[f"videos/{img_key}/chunk_index"] = chunk_idx
                    ep_meta[f"videos/{img_key}/file_index"] = file_idx
                    ep_meta[f"videos/{img_key}/from_timestamp"] = from_timestamp
                    ep_meta[f"videos/{img_key}/to_timestamp"] = to_timestamp

                # Move to next video file for next batch
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, new_meta.chunks_size)
                cumulative_timestamp = 0.0

        # Copy and transform data files (removing image columns)
        _copy_data_without_images(dataset, new_meta, episode_indices, img_keys)

        # Save episode metadata
        episodes_df = pd.DataFrame(list(all_episode_metadata.values()))
        episodes_path = new_meta.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_df.to_parquet(episodes_path, index=False)

        # Update metadata info
        new_meta.info["total_episodes"] = len(episode_indices)
        new_meta.info["total_frames"] = sum(ep["length"] for ep in all_episode_metadata.values())
        new_meta.info["total_tasks"] = dataset.meta.total_tasks
        new_meta.info["splits"] = {"train": f"0:{len(episode_indices)}"}

        # Update video info for all image keys (now videos)
        # We need to manually set video info since update_video_info() checks video_keys first
        for img_key in img_keys:
            if not new_meta.features[img_key].get("info", None):
                video_path = new_meta.root / new_meta.video_path.format(
                    video_key=img_key, chunk_index=0, file_index=0
                )
                new_meta.info["features"][img_key]["info"] = get_video_info(video_path)

        write_info(new_meta.info, new_meta.root)

        # Copy stats and tasks
        if dataset.meta.stats is not None:
            # Remove image stats
            new_stats = {k: v for k, v in dataset.meta.stats.items() if k not in img_keys}
            write_stats(new_stats, new_meta.root)

        if dataset.meta.tasks is not None:
            write_tasks(dataset.meta.tasks, new_meta.root)

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    logging.info(f"Completed converting {dataset.repo_id} to video format")
    logging.info(f"New dataset saved to: {output_dir}")

    # Return new dataset
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


def _encode_guide_video(
    src_dataset: LeRobotDataset,
    source_key: str,
    episodes: list[int],
    output_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
) -> list[tuple[float, float]]:
    """Create a guide video by repeating the first frame of each episode.

    For each episode, decodes the first frame from ``source_key`` in the source
    dataset and encodes it ``episode_length`` times into ``output_path``.
    Multiple episodes are concatenated into the same video file, matching the
    chunked storage layout used by other video streams.

    Args:
        src_dataset: Source dataset to read frames from.
        source_key: Video key to use as the source for the guide frames.
        episodes: Ordered list of episode indices to include in this video file.
        output_path: Destination video file path.
        fps: Frames per second for the output video.
        vcodec: Video codec (default: libsvtav1).
        pix_fmt: Pixel format (default: yuv420p).
        g: GOP size (default: 2).
        crf: Constant rate factor for quality (default: 30).

    Returns:
        List of ``(from_timestamp, to_timestamp)`` tuples (one per episode).
    """
    from fractions import Fraction

    import av

    vcodec = resolve_vcodec(vcodec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps_fraction = Fraction(fps).limit_denominator(1000)
    codec_options = _get_codec_options(vcodec, g, crf)

    episode_timestamps: list[tuple[float, float]] = []
    cumulative_ts = 0.0
    frame_count = 0

    with av.open(str(output_path), mode="w") as out_container:
        out_stream = None

        for ep_idx in episodes:
            ep = src_dataset.meta.episodes[ep_idx]
            ep_length = ep["length"]
            src_from_ts = ep[f"videos/{source_key}/from_timestamp"]

            # Locate and open source video file for this episode
            src_video_rel_path = src_dataset.meta.get_video_file_path(ep_idx, source_key)
            src_video_path = src_dataset.root / src_video_rel_path

            # Decode the first frame of this episode
            first_frame_image = None
            with av.open(str(src_video_path)) as in_container:
                in_stream = in_container.streams.video[0]
                # Seek to episode start (in microseconds, the global AV time base)
                seek_us = int(src_from_ts * 1_000_000)
                in_container.seek(seek_us, backward=True, stream=in_stream)

                for packet in in_container.demux(in_stream):
                    for frame in packet.decode():
                        if frame is not None:
                            first_frame_image = frame.to_image().convert("RGB")
                            break
                    if first_frame_image is not None:
                        break

            if first_frame_image is None:
                raise ValueError(
                    f"Could not decode first frame for episode {ep_idx}, "
                    f"source key '{source_key}'"
                )

            # Initialize output stream from first decoded frame
            if out_stream is None:
                out_stream = out_container.add_stream(vcodec, rate=fps_fraction, options=codec_options)
                out_stream.width = first_frame_image.width
                out_stream.height = first_frame_image.height
                out_stream.pix_fmt = pix_fmt

            # Convert PIL image to av.VideoFrame in the target pixel format.
            # Convert *once* to a numpy array (outside the repetition loop) so that
            # av.VideoFrame.from_ndarray() — which is cheap — can create distinct
            # frame objects per repetition.  av.VideoFrame has no copy() method, and
            # reusing the same object with mutated PTS is unsafe with some encoders.
            guide_av_frame = av.VideoFrame.from_image(first_frame_image)
            guide_av_frame = guide_av_frame.reformat(
                width=out_stream.width,
                height=out_stream.height,
                format=pix_fmt,
            )
            guide_array = guide_av_frame.to_ndarray(format=pix_fmt)

            # Encode the guide frame ep_length times (once per episode frame)
            for _ in range(ep_length):
                frame_to_encode = av.VideoFrame.from_ndarray(guide_array, format=pix_fmt)
                frame_to_encode.pts = frame_count
                for pkt in out_stream.encode(frame_to_encode):
                    out_container.mux(pkt)
                frame_count += 1

            from_ts = cumulative_ts
            to_ts = cumulative_ts + ep_length / fps
            episode_timestamps.append((from_ts, to_ts))
            cumulative_ts = to_ts

        # Flush encoder
        if out_stream is not None:
            for pkt in out_stream.encode(None):
                out_container.mux(pkt)

    return episode_timestamps


def _update_episodes_metadata_with_video(
    new_meta: LeRobotDatasetMetadata,
    video_metadata: dict[int, dict],
) -> None:
    """Append new video-key columns to existing episodes metadata parquet files.

    After copying episodes metadata from the source dataset, call this function
    to add the chunk/file/timestamp columns for a newly created video stream.

    Args:
        new_meta: Metadata of the destination dataset (used to locate parquet files).
        video_metadata: Mapping from episode index to a dict of column_name → value.
            All episode entries must have the same set of keys.
    """
    if not video_metadata:
        return

    video_cols = list(next(iter(video_metadata.values())).keys())
    episodes_dir = new_meta.root / "meta" / "episodes"

    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)

        # Build one mapping dict per column and use pandas .map() for efficiency.
        for col in video_cols:
            col_map = {ep_idx: ep_data[col] for ep_idx, ep_data in video_metadata.items()}
            df[col] = df["episode_index"].map(col_map)

        df.to_parquet(parquet_path, index=False)


def add_guide_stream(
    dataset: LeRobotDataset,
    source_key: str,
    new_key: str,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
) -> LeRobotDataset:
    """Add a guide video stream to a LeRobotDataset.

    The guide stream repeats the *first frame* of a source camera throughout
    each episode.  It serves as a static reference image that shows where a
    relevant object is at the start of the episode — useful for downstream
    policies or visualisation.

    The new video files follow the same chunk/file layout as ``source_key``:
    episodes that share a source video file are grouped into the same guide
    video file.

    Args:
        dataset: The source LeRobotDataset (must contain at least one video key).
        source_key: Existing video key whose first episode frame is used as the
            guide (e.g. ``"observation.images.laptop"``).
        new_key: Name for the new guide video stream
            (e.g. ``"observation.images.guide_laptop"``).
        output_dir: Directory for the new dataset.  If ``None``, defaults to
            ``HF_LEROBOT_HOME / repo_id``.
        repo_id: Repository ID for the new dataset.  If ``None``, ``"_with_guide"``
            is appended to the original repo ID.
        vcodec: Video codec for the guide stream (default: ``"libsvtav1"``).
        pix_fmt: Pixel format for the guide stream (default: ``"yuv420p"``).
        g: GOP size for encoding (default: ``2``).
        crf: Constant rate factor / quality setting (default: ``30``).

    Returns:
        New :class:`LeRobotDataset` with the guide stream added.

    Example::

        dataset = LeRobotDataset("my_user/my_dataset")
        new_dataset = add_guide_stream(
            dataset,
            source_key="observation.images.laptop",
            new_key="observation.images.guide_laptop",
        )
    """
    if source_key not in dataset.meta.video_keys:
        raise ValueError(
            f"source_key '{source_key}' must be a video stream. "
            f"Video keys in dataset: {dataset.meta.video_keys}. "
            "If source_key is stored as images, first convert to video using "
            "convert_image_to_video_dataset()."
        )

    if new_key in dataset.meta.features:
        raise ValueError(f"Key '{new_key}' already exists in dataset features.")

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_with_guide"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    # Build new features dict: copy source feature spec (shape/dtype) for the guide
    source_feature = {
        "dtype": "video",
        "shape": dataset.meta.features[source_key]["shape"],
        "names": dataset.meta.features[source_key].get("names"),
    }
    new_features = dataset.meta.features.copy()
    new_features[new_key] = source_feature

    # Create destination metadata
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Copy parquet data files (no column changes needed for a video-only addition)
    logging.info("Copying data files…")
    data_dir_src = dataset.root / DATA_DIR
    data_dir_dst = new_meta.root / DATA_DIR
    shutil.copytree(data_dir_src, data_dir_dst)

    # Copy existing video files
    logging.info("Copying existing video files…")
    _copy_videos(dataset, new_meta)

    # Copy episodes metadata (will be extended with guide-stream columns below)
    logging.info("Copying episodes metadata…")
    episodes_src = dataset.root / "meta" / "episodes"
    episodes_dst = new_meta.root / "meta" / "episodes"
    if episodes_src.exists():
        shutil.copytree(episodes_src, episodes_dst, dirs_exist_ok=True)

    # Copy tasks and stats
    if dataset.meta.tasks is not None:
        write_tasks(dataset.meta.tasks, new_meta.root)
    if dataset.meta.stats is not None:
        write_stats(dataset.meta.stats, new_meta.root)

    # Group episodes by their source video file (same chunk/file layout)
    file_to_episodes: dict[tuple[int, int], list[int]] = {}
    for ep_idx in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        chunk_idx = ep[f"videos/{source_key}/chunk_index"]
        file_idx = ep[f"videos/{source_key}/file_index"]
        file_key = (chunk_idx, file_idx)
        file_to_episodes.setdefault(file_key, []).append(ep_idx)

    # Encode guide stream videos and collect per-episode metadata
    fps = dataset.meta.fps
    guide_video_metadata: dict[int, dict] = {}

    for (src_chunk_idx, src_file_idx), eps_in_file in tqdm(
        sorted(file_to_episodes.items()), desc="Encoding guide stream videos"
    ):
        assert new_meta.video_path is not None
        out_video_path = new_meta.root / new_meta.video_path.format(
            video_key=new_key, chunk_index=src_chunk_idx, file_index=src_file_idx
        )

        episode_timestamps = _encode_guide_video(
            src_dataset=dataset,
            source_key=source_key,
            episodes=eps_in_file,
            output_path=out_video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
        )

        for ep_idx, (from_ts, to_ts) in zip(eps_in_file, episode_timestamps, strict=True):
            guide_video_metadata[ep_idx] = {
                f"videos/{new_key}/chunk_index": src_chunk_idx,
                f"videos/{new_key}/file_index": src_file_idx,
                f"videos/{new_key}/from_timestamp": from_ts,
                f"videos/{new_key}/to_timestamp": to_ts,
            }

    # Extend episodes metadata parquet files with the new video columns
    logging.info("Updating episodes metadata with guide stream info…")
    _update_episodes_metadata_with_video(new_meta, guide_video_metadata)

    # Populate video codec/resolution info for the new key
    first_guide_path = new_meta.root / new_meta.video_path.format(
        video_key=new_key, chunk_index=0, file_index=0
    )
    if first_guide_path.exists():
        new_meta.info["features"][new_key]["info"] = get_video_info(first_guide_path)

    # Propagate existing video feature info (codec metadata) from source dataset
    for key in dataset.meta.video_keys:
        if key in dataset.meta.features and dataset.meta.info["features"][key].get("info"):
            new_meta.info["features"][key]["info"] = dataset.meta.info["features"][key]["info"]

    # Update dataset-level counters
    new_meta.info.update(
        {
            "total_episodes": dataset.meta.total_episodes,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "splits": dataset.meta.info.get("splits", {"train": f"0:{dataset.meta.total_episodes}"}),
        }
    )
    write_info(new_meta.info, new_meta.root)

    logging.info(f"Guide stream '{new_key}' added. Dataset saved to: {output_dir}")
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


# ---------------------------------------------------------------------------
# Helper: encode a video containing only black frames
# ---------------------------------------------------------------------------


def _encode_black_video(
    episodes: list[int],
    episode_lengths: dict[int, int],
    width: int,
    height: int,
    output_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
) -> list[tuple[float, float]]:
    """Create a video file containing black frames for multiple episodes.

    All frames are pure black (zero-valued).  The spatial dimensions are given
    by ``width`` and ``height``.  Multiple episodes are concatenated into the
    same video file, matching the chunked storage layout used by other video
    streams.

    Args:
        episodes: Ordered list of episode indices to include in this video file.
        episode_lengths: Mapping from episode index to number of frames.
        width: Frame width in pixels.
        height: Frame height in pixels.
        output_path: Destination video file path.
        fps: Frames per second for the output video.
        vcodec: Video codec (default: libsvtav1).
        pix_fmt: Pixel format (default: yuv420p).
        g: GOP size (default: 2).
        crf: Constant rate factor for quality (default: 30).

    Returns:
        List of ``(from_timestamp, to_timestamp)`` tuples (one per episode).
    """
    from fractions import Fraction

    import av
    from PIL import Image

    vcodec = resolve_vcodec(vcodec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps_fraction = Fraction(fps).limit_denominator(1000)
    codec_options = _get_codec_options(vcodec, g, crf)

    episode_timestamps: list[tuple[float, float]] = []
    cumulative_ts = 0.0
    frame_count = 0

    with av.open(str(output_path), mode="w") as out_container:
        out_stream = out_container.add_stream(vcodec, rate=fps_fraction, options=codec_options)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = pix_fmt

        # Build the black frame once in the target pixel format.
        # av.VideoFrame.from_ndarray() is cheap so we create a fresh frame
        # object per encoded frame (avoids PTS mutation issues with some encoders).
        black_pil = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
        black_av_frame = av.VideoFrame.from_image(black_pil)
        black_av_frame = black_av_frame.reformat(width=width, height=height, format=pix_fmt)
        black_array = black_av_frame.to_ndarray(format=pix_fmt)

        for ep_idx in episodes:
            ep_length = episode_lengths[ep_idx]
            from_ts = cumulative_ts

            for _ in range(ep_length):
                frame_to_encode = av.VideoFrame.from_ndarray(black_array, format=pix_fmt)
                frame_to_encode.pts = frame_count
                for pkt in out_stream.encode(frame_to_encode):
                    out_container.mux(pkt)
                frame_count += 1

            to_ts = cumulative_ts + ep_length / fps
            episode_timestamps.append((from_ts, to_ts))
            cumulative_ts = to_ts

        # Flush encoder
        for pkt in out_stream.encode(None):
            out_container.mux(pkt)

    return episode_timestamps


# ---------------------------------------------------------------------------
# Public API: add_black_stream
# ---------------------------------------------------------------------------


def add_black_stream(
    dataset: LeRobotDataset,
    source_key: str,
    new_key: str,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
) -> LeRobotDataset:
    """Add an all-black video stream to a LeRobotDataset.

    Every frame of every episode in the new stream is a pure black image.
    The spatial dimensions (width × height) are taken from ``source_key``
    so the new stream is compatible with any downstream code that expects
    the same resolution as the source camera.

    This is useful as a placeholder stream when a second camera view is not
    yet available, or to provide a neutral reference channel that does not
    carry any visual information.

    The new video files follow the same chunk/file layout as ``source_key``.

    Args:
        dataset: The source LeRobotDataset (must contain at least one video key).
        source_key: Existing video key whose resolution is used for the black
            stream (e.g. ``"observation.images.top"``).
        new_key: Name for the new black video stream
            (e.g. ``"observation.images.black_top"``).
        output_dir: Directory for the new dataset.  If ``None``, defaults to
            ``HF_LEROBOT_HOME / repo_id``.
        repo_id: Repository ID for the new dataset.  If ``None``,
            ``"_with_black"`` is appended to the original repo ID.
        vcodec: Video codec for the black stream (default: ``"libsvtav1"``).
        pix_fmt: Pixel format for the black stream (default: ``"yuv420p"``).
        g: GOP size for encoding (default: ``2``).
        crf: Constant rate factor / quality setting (default: ``30``).

    Returns:
        New :class:`LeRobotDataset` with the black stream added.

    Example::

        dataset = LeRobotDataset("my_user/my_dataset")
        new_dataset = add_black_stream(
            dataset,
            source_key="observation.images.top",
            new_key="observation.images.black_top",
        )
    """
    if source_key not in dataset.meta.video_keys:
        raise ValueError(
            f"source_key '{source_key}' must be a video stream. "
            f"Video keys in dataset: {dataset.meta.video_keys}. "
            "If source_key is stored as images, first convert to video using "
            "convert_image_to_video_dataset()."
        )

    if new_key in dataset.meta.features:
        raise ValueError(f"Key '{new_key}' already exists in dataset features.")

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_with_black"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    # Build new features dict: copy source feature spec (shape/dtype) for the black stream
    source_feature = {
        "dtype": "video",
        "shape": dataset.meta.features[source_key]["shape"],
        "names": dataset.meta.features[source_key].get("names"),
    }
    new_features = dataset.meta.features.copy()
    new_features[new_key] = source_feature

    # Create destination metadata
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Copy parquet data files (no column changes needed for a video-only addition)
    logging.info("Copying data files…")
    data_dir_src = dataset.root / DATA_DIR
    data_dir_dst = new_meta.root / DATA_DIR
    shutil.copytree(data_dir_src, data_dir_dst)

    # Copy existing video files
    logging.info("Copying existing video files…")
    _copy_videos(dataset, new_meta)

    # Copy episodes metadata (will be extended with black-stream columns below)
    logging.info("Copying episodes metadata…")
    episodes_src = dataset.root / "meta" / "episodes"
    episodes_dst = new_meta.root / "meta" / "episodes"
    if episodes_src.exists():
        shutil.copytree(episodes_src, episodes_dst, dirs_exist_ok=True)

    # Copy tasks and stats
    if dataset.meta.tasks is not None:
        write_tasks(dataset.meta.tasks, new_meta.root)
    if dataset.meta.stats is not None:
        write_stats(dataset.meta.stats, new_meta.root)

    # Determine black frame dimensions from source_key feature shape (H, W, C)
    source_shape = dataset.meta.features[source_key]["shape"]
    height, width = source_shape[0], source_shape[1]

    # Group episodes by their source video file (same chunk/file layout)
    file_to_episodes: dict[tuple[int, int], list[int]] = {}
    episode_lengths: dict[int, int] = {}
    for ep_idx in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        episode_lengths[ep_idx] = ep["length"]
        chunk_idx = ep[f"videos/{source_key}/chunk_index"]
        file_idx = ep[f"videos/{source_key}/file_index"]
        file_to_episodes.setdefault((chunk_idx, file_idx), []).append(ep_idx)

    fps = dataset.meta.fps
    black_video_metadata: dict[int, dict] = {}

    for (src_chunk_idx, src_file_idx), eps_in_file in tqdm(
        sorted(file_to_episodes.items()), desc="Encoding black stream videos"
    ):
        assert new_meta.video_path is not None
        out_video_path = new_meta.root / new_meta.video_path.format(
            video_key=new_key, chunk_index=src_chunk_idx, file_index=src_file_idx
        )

        episode_timestamps = _encode_black_video(
            episodes=eps_in_file,
            episode_lengths=episode_lengths,
            width=width,
            height=height,
            output_path=out_video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
        )

        for ep_idx, (from_ts, to_ts) in zip(eps_in_file, episode_timestamps, strict=True):
            black_video_metadata[ep_idx] = {
                f"videos/{new_key}/chunk_index": src_chunk_idx,
                f"videos/{new_key}/file_index": src_file_idx,
                f"videos/{new_key}/from_timestamp": from_ts,
                f"videos/{new_key}/to_timestamp": to_ts,
            }

    # Extend episodes metadata parquet files with the new video columns
    logging.info("Updating episodes metadata with black stream info…")
    _update_episodes_metadata_with_video(new_meta, black_video_metadata)

    # Populate video codec/resolution info for the new key
    first_black_path = new_meta.root / new_meta.video_path.format(
        video_key=new_key, chunk_index=0, file_index=0
    )
    if first_black_path.exists():
        new_meta.info["features"][new_key]["info"] = get_video_info(first_black_path)

    # Propagate existing video feature info (codec metadata) from source dataset
    for key in dataset.meta.video_keys:
        if key in dataset.meta.features and dataset.meta.info["features"][key].get("info"):
            new_meta.info["features"][key]["info"] = dataset.meta.info["features"][key]["info"]

    # Update dataset-level counters
    new_meta.info.update(
        {
            "total_episodes": dataset.meta.total_episodes,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "splits": dataset.meta.info.get("splits", {"train": f"0:{dataset.meta.total_episodes}"}),
        }
    )
    write_info(new_meta.info, new_meta.root)

    logging.info(f"Black stream '{new_key}' added. Dataset saved to: {output_dir}")
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


# ---------------------------------------------------------------------------
# Helper: decode the first frame of an episode from a video stream
# ---------------------------------------------------------------------------

def _decode_first_frame_bgr(
    dataset: LeRobotDataset,
    episode_idx: int,
    source_key: str,
) -> np.ndarray:
    """Decode the first frame of an episode's video stream and return it as BGR numpy array.

    Args:
        dataset: Source dataset.
        episode_idx: Episode index.
        source_key: Video key to read from.

    Returns:
        BGR image as numpy array (H, W, 3), dtype uint8.
    """
    import av
    import cv2

    ep = dataset.meta.episodes[episode_idx]
    src_from_ts = ep[f"videos/{source_key}/from_timestamp"]
    src_video_rel_path = dataset.meta.get_video_file_path(episode_idx, source_key)
    src_video_path = dataset.root / src_video_rel_path

    first_frame_image = None
    with av.open(str(src_video_path)) as container:
        stream = container.streams.video[0]
        seek_us = int(src_from_ts * 1_000_000)
        container.seek(seek_us, backward=True, stream=stream)
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame is not None:
                    first_frame_image = frame.to_image().convert("RGB")
                    break
            if first_frame_image is not None:
                break

    if first_frame_image is None:
        raise ValueError(
            f"Could not decode first frame for episode {episode_idx}, "
            f"source key '{source_key}'"
        )

    frame_rgb = np.array(first_frame_image)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Helper: encode pre-computed frames as a static (repeated) video
# ---------------------------------------------------------------------------

def _encode_precomputed_guide_video(
    precomputed_frames: dict[int, np.ndarray],
    episode_lengths: dict[int, int],
    episodes: list[int],
    output_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
) -> list[tuple[float, float]]:
    """Encode a video by repeating pre-computed BGR frames for each episode.

    Behaves like :func:`_encode_guide_video` but uses frames that have already
    been processed (e.g. segmented / highlighted) instead of decoding them from
    a source video.

    Args:
        precomputed_frames: Mapping of episode index → BGR image (H, W, 3).
        episode_lengths: Mapping of episode index → number of frames.
        episodes: Ordered list of episode indices for this video file.
        output_path: Destination video file path.
        fps: Frames per second.
        vcodec, pix_fmt, g, crf: Encoding parameters.

    Returns:
        List of ``(from_timestamp, to_timestamp)`` tuples, one per episode.
    """
    from fractions import Fraction

    import av
    import cv2

    vcodec = resolve_vcodec(vcodec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps_fraction = Fraction(fps).limit_denominator(1000)
    codec_options = _get_codec_options(vcodec, g, crf)

    episode_timestamps: list[tuple[float, float]] = []
    cumulative_ts = 0.0
    frame_count = 0

    with av.open(str(output_path), mode="w") as out_container:
        out_stream = None

        for ep_idx in episodes:
            bgr_frame = precomputed_frames[ep_idx]
            ep_length = episode_lengths[ep_idx]

            # Convert BGR → RGB PIL → av.VideoFrame
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(rgb_frame)

            if out_stream is None:
                out_stream = out_container.add_stream(
                    vcodec, rate=fps_fraction, options=codec_options
                )
                out_stream.width = pil_image.width
                out_stream.height = pil_image.height
                out_stream.pix_fmt = pix_fmt

            av_frame = av.VideoFrame.from_image(pil_image)
            av_frame = av_frame.reformat(
                width=out_stream.width, height=out_stream.height, format=pix_fmt
            )
            frame_array = av_frame.to_ndarray(format=pix_fmt)

            for _ in range(ep_length):
                frame_to_encode = av.VideoFrame.from_ndarray(frame_array, format=pix_fmt)
                frame_to_encode.pts = frame_count
                for pkt in out_stream.encode(frame_to_encode):
                    out_container.mux(pkt)
                frame_count += 1

            from_ts = cumulative_ts
            to_ts = cumulative_ts + ep_length / fps
            episode_timestamps.append((from_ts, to_ts))
            cumulative_ts = to_ts

        if out_stream is not None:
            for pkt in out_stream.encode(None):
                out_container.mux(pkt)

    return episode_timestamps


# ---------------------------------------------------------------------------
# Public API: add_initial_scene_segmentation_stream
# ---------------------------------------------------------------------------

def add_initial_scene_segmentation_stream(
    dataset: LeRobotDataset,
    source_key: str,
    new_key: str,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    fade_pixels: int = 16,
    min_brightness: float = 0.0,
) -> LeRobotDataset:
    """Add a segmented scene stream by interactively segmenting the first frame of each episode.

    For every episode the first frame of ``source_key`` is shown in an
    interactive OpenCV window.  The user selects an object using SAM2
    point-prompt segmentation (left click = foreground, right click =
    background, Enter = confirm).  The resulting highlighted image —
    with the selected object in full colour and a gradient fade-to-black
    around it — is then repeated for the full duration of that episode,
    exactly like :func:`add_guide_stream`.

    If the user cancels segmentation for an episode (presses Escape),
    the raw first frame is used instead (no highlighting).

    Args:
        dataset: Source LeRobotDataset (must contain at least one video key).
        source_key: Existing video key whose first frame is segmented
            (e.g. ``"observation.images.laptop"``).
        new_key: Name for the new segmented stream
            (e.g. ``"observation.images.segmented_laptop"``).
        output_dir: Directory for the new dataset.  Defaults to
            ``HF_LEROBOT_HOME / repo_id``.
        repo_id: Repository ID for the new dataset.  Defaults to
            ``"<original>_with_seg"``.
        vcodec: Video codec (default: ``"libsvtav1"``).
        pix_fmt: Pixel format (default: ``"yuv420p"``).
        g: GOP size (default: ``2``).
        crf: Constant rate factor (default: ``30``).
        fade_pixels: Number of pixels over which brightness fades from full
            (at the mask edge) to *min_brightness* (default: ``80``).
        min_brightness: Brightness multiplier beyond the fade zone
            (``0.0`` = black, ``1.0`` = unchanged; default: ``0.0``).

    Returns:
        New :class:`LeRobotDataset` with the segmented stream added.
    """
    from lerobot.cameras.zmq.segment import SAM2Segmenter, interactive_select

    if source_key not in dataset.meta.video_keys:
        raise ValueError(
            f"source_key '{source_key}' must be a video stream. "
            f"Video keys in dataset: {dataset.meta.video_keys}."
        )

    if new_key in dataset.meta.features:
        raise ValueError(f"Key '{new_key}' already exists in dataset features.")

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_with_seg"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    # -- Phase 1: interactive segmentation of the first frame per episode ----
    logging.info("Loading SAM2 segmenter …")
    segmenter = SAM2Segmenter()

    precomputed_frames: dict[int, np.ndarray] = {}
    episode_lengths: dict[int, int] = {}
    total_episodes = dataset.meta.total_episodes

    for ep_idx in range(total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        episode_lengths[ep_idx] = ep["length"]

        first_frame_bgr = _decode_first_frame_bgr(dataset, ep_idx, source_key)

        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            "Select object to segment (Enter=confirm, Esc=skip)"
        )
        highlighted, mask = interactive_select(
            first_frame_bgr,
            segmenter,
            window_name=f"Segment Episode {ep_idx} ({source_key})",
            fade_pixels=fade_pixels,
            min_brightness=min_brightness,
        )

        if highlighted is not None:
            precomputed_frames[ep_idx] = highlighted
        else:
            # User cancelled – fall back to the raw first frame
            logging.info(f"Episode {ep_idx}: segmentation skipped, using raw frame.")
            precomputed_frames[ep_idx] = first_frame_bgr

    # -- Phase 2: build new dataset (same structure as add_guide_stream) -----
    source_feature = {
        "dtype": "video",
        "shape": dataset.meta.features[source_key]["shape"],
        "names": dataset.meta.features[source_key].get("names"),
    }
    new_features = dataset.meta.features.copy()
    new_features[new_key] = source_feature

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Copy parquet data files
    logging.info("Copying data files …")
    data_dir_src = dataset.root / DATA_DIR
    data_dir_dst = new_meta.root / DATA_DIR
    shutil.copytree(data_dir_src, data_dir_dst)

    # Copy existing video files
    logging.info("Copying existing video files …")
    _copy_videos(dataset, new_meta)

    # Copy episodes metadata
    logging.info("Copying episodes metadata …")
    episodes_src = dataset.root / "meta" / "episodes"
    episodes_dst = new_meta.root / "meta" / "episodes"
    if episodes_src.exists():
        shutil.copytree(episodes_src, episodes_dst, dirs_exist_ok=True)

    # Copy tasks and stats
    if dataset.meta.tasks is not None:
        write_tasks(dataset.meta.tasks, new_meta.root)
    if dataset.meta.stats is not None:
        write_stats(dataset.meta.stats, new_meta.root)

    # Group episodes by source video file (same chunk/file layout)
    file_to_episodes: dict[tuple[int, int], list[int]] = {}
    for ep_idx in range(total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        chunk_idx = ep[f"videos/{source_key}/chunk_index"]
        file_idx = ep[f"videos/{source_key}/file_index"]
        file_to_episodes.setdefault((chunk_idx, file_idx), []).append(ep_idx)

    # -- Phase 3: encode segmented stream videos -----------------------------
    fps = dataset.meta.fps
    seg_video_metadata: dict[int, dict] = {}

    for (src_chunk_idx, src_file_idx), eps_in_file in tqdm(
        sorted(file_to_episodes.items()), desc="Encoding segmented stream videos"
    ):
        assert new_meta.video_path is not None
        out_video_path = new_meta.root / new_meta.video_path.format(
            video_key=new_key, chunk_index=src_chunk_idx, file_index=src_file_idx
        )

        episode_timestamps = _encode_precomputed_guide_video(
            precomputed_frames=precomputed_frames,
            episode_lengths=episode_lengths,
            episodes=eps_in_file,
            output_path=out_video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
        )

        for ep_idx, (from_ts, to_ts) in zip(eps_in_file, episode_timestamps, strict=True):
            seg_video_metadata[ep_idx] = {
                f"videos/{new_key}/chunk_index": src_chunk_idx,
                f"videos/{new_key}/file_index": src_file_idx,
                f"videos/{new_key}/from_timestamp": from_ts,
                f"videos/{new_key}/to_timestamp": to_ts,
            }

    # Update episodes metadata with new video columns
    logging.info("Updating episodes metadata with segmented stream info …")
    _update_episodes_metadata_with_video(new_meta, seg_video_metadata)

    # Populate video codec/resolution info
    first_seg_path = new_meta.root / new_meta.video_path.format(
        video_key=new_key, chunk_index=0, file_index=0
    )
    if first_seg_path.exists():
        new_meta.info["features"][new_key]["info"] = get_video_info(first_seg_path)

    for key in dataset.meta.video_keys:
        if key in dataset.meta.features and dataset.meta.info["features"][key].get("info"):
            new_meta.info["features"][key]["info"] = dataset.meta.info["features"][key]["info"]

    new_meta.info.update(
        {
            "total_episodes": dataset.meta.total_episodes,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "splits": dataset.meta.info.get("splits", {"train": f"0:{dataset.meta.total_episodes}"}),
        }
    )
    write_info(new_meta.info, new_meta.root)

    logging.info(f"Segmented stream '{new_key}' added. Dataset saved to: {output_dir}")
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


# ---------------------------------------------------------------------------
# Helper: decode ALL frames of an episode from a video stream as BGR
# ---------------------------------------------------------------------------

def _decode_all_frames_bgr(
    dataset: LeRobotDataset,
    episode_idx: int,
    source_key: str,
) -> list[np.ndarray]:
    """Decode every frame of an episode's video stream and return as BGR numpy arrays.

    Args:
        dataset: Source dataset.
        episode_idx: Episode index.
        source_key: Video key to read from.

    Returns:
        List of BGR images (H, W, 3), dtype uint8, one per frame.
    """
    import av
    import cv2

    ep = dataset.meta.episodes[episode_idx]
    src_from_ts = ep[f"videos/{source_key}/from_timestamp"]
    src_to_ts = ep[f"videos/{source_key}/to_timestamp"]
    ep_length = ep["length"]
    src_video_rel_path = dataset.meta.get_video_file_path(episode_idx, source_key)
    src_video_path = dataset.root / src_video_rel_path

    frames: list[np.ndarray] = []
    with av.open(str(src_video_path)) as container:
        stream = container.streams.video[0]
        seek_us = int(src_from_ts * 1_000_000)
        container.seek(seek_us, backward=True, stream=stream)

        end_us = int(src_to_ts * 1_000_000)
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame is None:
                    continue
                frame_us = int(frame.time * 1_000_000) if frame.time is not None else 0
                if frame_us < seek_us:
                    continue
                if frame_us > end_us and len(frames) >= ep_length:
                    break
                rgb = np.array(frame.to_image().convert("RGB"))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                frames.append(bgr)
                if len(frames) >= ep_length:
                    break
            if len(frames) >= ep_length:
                break

    return frames[:ep_length]


# ---------------------------------------------------------------------------
# Helper: encode a list of per-frame BGR images as a video
# ---------------------------------------------------------------------------

def _encode_frame_list_video(
    frames_per_episode: dict[int, list[np.ndarray]],
    episode_lengths: dict[int, int],
    episodes: list[int],
    output_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
) -> list[tuple[float, float]]:
    """Encode per-frame BGR images for multiple episodes into one video file.

    Unlike ``_encode_precomputed_guide_video`` (which repeats a single frame),
    this function encodes a different image per frame.

    Args:
        frames_per_episode: Mapping of episode index → list of BGR images.
        episode_lengths: Mapping of episode index → expected number of frames.
        episodes: Ordered list of episode indices to encode.
        output_path: Destination video file path.
        fps: Frames per second.
        vcodec, pix_fmt, g, crf: Encoding parameters.

    Returns:
        List of ``(from_timestamp, to_timestamp)`` tuples, one per episode.
    """
    from fractions import Fraction

    import av
    import cv2

    vcodec = resolve_vcodec(vcodec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps_fraction = Fraction(fps).limit_denominator(1000)
    codec_options = _get_codec_options(vcodec, g, crf)

    episode_timestamps: list[tuple[float, float]] = []
    cumulative_ts = 0.0
    frame_count = 0

    with av.open(str(output_path), mode="w") as out_container:
        out_stream = None

        for ep_idx in episodes:
            bgr_frames = frames_per_episode[ep_idx]
            ep_length = episode_lengths[ep_idx]

            for i in range(min(len(bgr_frames), ep_length)):
                rgb = cv2.cvtColor(bgr_frames[i], cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_image = Image.fromarray(rgb)

                if out_stream is None:
                    out_stream = out_container.add_stream(
                        vcodec, rate=fps_fraction, options=codec_options
                    )
                    out_stream.width = pil_image.width
                    out_stream.height = pil_image.height
                    out_stream.pix_fmt = pix_fmt

                av_frame = av.VideoFrame.from_image(pil_image)
                av_frame = av_frame.reformat(
                    width=out_stream.width, height=out_stream.height, format=pix_fmt
                )
                frame_to_encode = av.VideoFrame.from_ndarray(
                    av_frame.to_ndarray(format=pix_fmt), format=pix_fmt
                )
                frame_to_encode.pts = frame_count
                for pkt in out_stream.encode(frame_to_encode):
                    out_container.mux(pkt)
                frame_count += 1

            from_ts = cumulative_ts
            to_ts = cumulative_ts + ep_length / fps
            episode_timestamps.append((from_ts, to_ts))
            cumulative_ts = to_ts

        if out_stream is not None:
            for pkt in out_stream.encode(None):
                out_container.mux(pkt)

    return episode_timestamps


# ---------------------------------------------------------------------------
# Public API: add_sam2_stream
# ---------------------------------------------------------------------------

def add_sam2_stream(
    dataset: LeRobotDataset,
    source_key: str,
    new_key: str,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    fade_pixels: int = 16,
    min_brightness: float = 0.0,
) -> LeRobotDataset:
    """Add a SAM2 video-tracked segmentation stream to the dataset.

    For every episode:

    1. The first frame of ``source_key`` is shown in an interactive OpenCV
       window where the user selects an object using SAM2 point prompts.
    2. The SAM2 **video predictor** propagates the mask across all frames
       of the episode.
    3. Each frame is highlighted (foreground preserved, background faded).
    4. The resulting stream is previewed in **Rerun** so the user can
       inspect the tracking quality.
    5. The user confirms (``y``) or rejects (``n``) the result.
       If rejected, the raw source frames are used for that episode.
    6. The confirmed frames are encoded as a new video stream.

    Args:
        dataset: Source LeRobotDataset (must contain at least one video key).
        source_key: Existing video key to segment
            (e.g. ``"observation.images.top_back"``).
        new_key: Name for the new segmented stream
            (e.g. ``"observation.images.sam2_top_back"``).
        output_dir: Directory for the new dataset.
        repo_id: Repository ID for the new dataset.
        vcodec: Video codec (default: ``"libsvtav1"``).
        pix_fmt: Pixel format (default: ``"yuv420p"``).
        g: GOP size (default: ``2``).
        crf: Constant rate factor (default: ``30``).
        fade_pixels: Number of pixels for the fade-to-black gradient.
        min_brightness: Background brightness (0.0 = black).

    Returns:
        New :class:`LeRobotDataset` with the SAM2-tracked stream added.
    """
    import gc
    import time

    import rerun as rr

    from lerobot.cameras.zmq.segment import (
        SAM2Segmenter,
        SAM2VideoSegmenter,
        highlight_object_overlay,
        interactive_select,
    )

    if source_key not in dataset.meta.video_keys:
        raise ValueError(
            f"source_key '{source_key}' must be a video stream. "
            f"Video keys in dataset: {dataset.meta.video_keys}."
        )

    if new_key in dataset.meta.features:
        raise ValueError(f"Key '{new_key}' already exists in dataset features.")

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_sam2"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    # -- Load models --------------------------------------------------------
    logging.info("Loading SAM2 image segmenter (for interactive selection) …")
    image_segmenter = SAM2Segmenter()
    logging.info("Loading SAM2 video segmenter (for temporal propagation) …")
    video_segmenter = SAM2VideoSegmenter()

    total_episodes = dataset.meta.total_episodes
    fps = dataset.meta.fps
    processed_frames: dict[int, list[np.ndarray]] = {}
    episode_lengths: dict[int, int] = {}

    for ep_idx in range(total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        ep_length = ep["length"]
        episode_lengths[ep_idx] = ep_length

        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            f"Decoding {ep_length} frames from '{source_key}' …"
        )
        frames_bgr = _decode_all_frames_bgr(dataset, ep_idx, source_key)

        if not frames_bgr:
            logging.warning(f"Episode {ep_idx}: no frames decoded, skipping.")
            processed_frames[ep_idx] = []
            continue

        # -- Step 1: interactive selection on first frame -------------------
        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            "Select object to track (Enter=confirm, Esc=skip)"
        )
        _highlighted, mask = interactive_select(
            frames_bgr[0],
            image_segmenter,
            window_name=f"SAM2 Track – Episode {ep_idx} ({source_key})",
            fade_pixels=fade_pixels,
            min_brightness=min_brightness,
        )

        if mask is None:
            logging.info(f"Episode {ep_idx}: selection cancelled, using raw frames.")
            processed_frames[ep_idx] = frames_bgr
            continue

        # Recover the points/labels from the interactive session
        # We re-do image segmentation to extract the points that were used.
        # However, interactive_select doesn't return points. So instead,
        # we use the mask directly with the video predictor via add_new_mask.
        # Let's propagate using the mask from frame 0.
        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            f"Propagating mask across {ep_length} frames with SAM2 video predictor …"
        )
        masks = _propagate_mask_across_frames(video_segmenter, frames_bgr, mask)
        del mask  # free initial mask

        # Apply highlighting to each frame (in-place to save memory)
        highlighted_frames: list[np.ndarray] = []
        for i in range(len(frames_bgr)):
            highlighted = highlight_object_overlay(frames_bgr[i], masks[i])
            highlighted_frames.append(highlighted)
            # Free source frame and mask immediately after use
            frames_bgr[i] = None  # type: ignore[assignment]
            masks[i] = None  # type: ignore[assignment]

        # Re-decode source frames (lightweight) only for the Rerun preview
        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            "Starting Rerun preview …"
        )
        preview_source = _decode_all_frames_bgr(dataset, ep_idx, source_key)
        accepted = _preview_and_confirm_rerun(
            episode_idx=ep_idx,
            source_frames=preview_source,
            segmented_frames=highlighted_frames,
            masks=[],  # masks already freed
            fps=fps,
            source_key=source_key,
            new_key=new_key,
        )
        del preview_source

        if accepted:
            logging.info(f"Episode {ep_idx}: confirmed ✓")
            processed_frames[ep_idx] = highlighted_frames
        else:
            logging.info(f"Episode {ep_idx}: rejected, using raw frames.")
            del highlighted_frames
            processed_frames[ep_idx] = _decode_all_frames_bgr(
                dataset, ep_idx, source_key
            )

    # -- Build new dataset structure ----------------------------------------
    source_feature = {
        "dtype": "video",
        "shape": dataset.meta.features[source_key]["shape"],
        "names": dataset.meta.features[source_key].get("names"),
    }
    new_features = dataset.meta.features.copy()
    new_features[new_key] = source_feature

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Copy parquet data files
    logging.info("Copying data files …")
    data_dir_src = dataset.root / DATA_DIR
    data_dir_dst = new_meta.root / DATA_DIR
    shutil.copytree(data_dir_src, data_dir_dst)

    # Copy existing video files
    logging.info("Copying existing video files …")
    _copy_videos(dataset, new_meta)

    # Copy episodes metadata
    logging.info("Copying episodes metadata …")
    episodes_src = dataset.root / "meta" / "episodes"
    episodes_dst = new_meta.root / "meta" / "episodes"
    if episodes_src.exists():
        shutil.copytree(episodes_src, episodes_dst, dirs_exist_ok=True)

    # Copy tasks and stats
    if dataset.meta.tasks is not None:
        write_tasks(dataset.meta.tasks, new_meta.root)
    if dataset.meta.stats is not None:
        write_stats(dataset.meta.stats, new_meta.root)

    # Group episodes by source video file (same chunk/file layout)
    file_to_episodes: dict[tuple[int, int], list[int]] = {}
    for ep_idx in range(total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        chunk_idx = ep[f"videos/{source_key}/chunk_index"]
        file_idx = ep[f"videos/{source_key}/file_index"]
        file_to_episodes.setdefault((chunk_idx, file_idx), []).append(ep_idx)

    # -- Encode SAM2-tracked stream videos ----------------------------------
    sam2_video_metadata: dict[int, dict] = {}

    for (src_chunk_idx, src_file_idx), eps_in_file in tqdm(
        sorted(file_to_episodes.items()), desc="Encoding SAM2 stream videos"
    ):
        assert new_meta.video_path is not None
        out_video_path = new_meta.root / new_meta.video_path.format(
            video_key=new_key, chunk_index=src_chunk_idx, file_index=src_file_idx
        )

        episode_timestamps = _encode_frame_list_video(
            frames_per_episode=processed_frames,
            episode_lengths=episode_lengths,
            episodes=eps_in_file,
            output_path=out_video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
        )

        for ep_idx, (from_ts, to_ts) in zip(eps_in_file, episode_timestamps, strict=True):
            sam2_video_metadata[ep_idx] = {
                f"videos/{new_key}/chunk_index": src_chunk_idx,
                f"videos/{new_key}/file_index": src_file_idx,
                f"videos/{new_key}/from_timestamp": from_ts,
                f"videos/{new_key}/to_timestamp": to_ts,
            }

    # Update episodes metadata with new video columns
    logging.info("Updating episodes metadata with SAM2 stream info …")
    _update_episodes_metadata_with_video(new_meta, sam2_video_metadata)

    # Populate video codec/resolution info
    first_sam2_path = new_meta.root / new_meta.video_path.format(
        video_key=new_key, chunk_index=0, file_index=0
    )
    if first_sam2_path.exists():
        new_meta.info["features"][new_key]["info"] = get_video_info(first_sam2_path)

    for key in dataset.meta.video_keys:
        if key in dataset.meta.features and dataset.meta.info["features"][key].get("info"):
            new_meta.info["features"][key]["info"] = dataset.meta.info["features"][key]["info"]

    new_meta.info.update(
        {
            "total_episodes": dataset.meta.total_episodes,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "splits": dataset.meta.info.get("splits", {"train": f"0:{dataset.meta.total_episodes}"}),
        }
    )
    write_info(new_meta.info, new_meta.root)

    logging.info(f"SAM2 stream '{new_key}' added. Dataset saved to: {output_dir}")
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


def _propagate_mask_across_frames(
    video_segmenter,
    frames_bgr: list[np.ndarray],
    initial_mask: np.ndarray,
    obj_id: int = 1,
) -> list[np.ndarray]:
    """Use SAM2 video predictor to propagate a mask from the first frame.

    Instead of using point prompts (which ``interactive_select`` does not
    return), we supply the confirmed binary mask directly via
    ``add_new_mask``.

    Args:
        video_segmenter: A :class:`SAM2VideoSegmenter` instance.
        frames_bgr: All BGR frames of the episode.
        initial_mask: Binary mask (H, W, uint8) for frame 0.
        obj_id: Object identifier for tracking.

    Returns:
        List of binary masks (H, W, uint8), one per frame.
    """
    import tempfile

    import cv2
    import torch

    if not frames_bgr:
        return []

    with tempfile.TemporaryDirectory(prefix="sam2_video_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for i, bgr in enumerate(frames_bgr):
            # SAM2 video predictor reads JPEG files from a directory
            cv2.imwrite(str(tmp_path / f"{i:06d}.jpg"), bgr)

        with torch.inference_mode(), torch.autocast(
            video_segmenter.device, dtype=torch.bfloat16
        ):
            state = video_segmenter.predictor.init_state(video_path=str(tmp_path))

            # Supply the initial mask on frame 0
            mask_tensor = torch.from_numpy(initial_mask.astype(np.float32)).to(
                video_segmenter.device
            )
            video_segmenter.predictor.add_new_mask(
                state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask_tensor,
            )

            # Propagate across all frames
            masks_by_frame: dict[int, np.ndarray] = {}
            for frame_idx, _obj_ids, video_res_masks in (
                video_segmenter.predictor.propagate_in_video(state)
            ):
                mask = (video_res_masks[0] > 0.5).cpu().numpy().astype(np.uint8)
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                masks_by_frame[frame_idx] = mask

    n = len(frames_bgr)
    h, w = frames_bgr[0].shape[:2]
    return [masks_by_frame.get(i, np.zeros((h, w), dtype=np.uint8)) for i in range(n)]


def _preview_and_confirm_rerun(
    episode_idx: int,
    source_frames: list[np.ndarray],
    segmented_frames: list[np.ndarray],
    masks: list[np.ndarray],
    fps: int,
    source_key: str,
    new_key: str,
) -> bool:
    """Show a side-by-side preview in Rerun and ask for user confirmation.

    Logs the original frames, segmented frames, and masks to Rerun,
    then prompts the user on the console.

    Args:
        episode_idx: Episode index (for display purposes).
        source_frames: Original BGR frames.
        segmented_frames: SAM2-highlighted BGR frames.
        masks: Binary masks (H, W, uint8).
        fps: Frames per second (for timeline).
        source_key: Name of the source stream.
        new_key: Name of the new stream.

    Returns:
        ``True`` if the user accepted, ``False`` otherwise.
    """
    import gc

    import cv2
    import rerun as rr

    session_name = f"SAM2 Preview – Episode {episode_idx}"
    rr.init(session_name, spawn=True)
    gc.collect()

    for i in range(len(source_frames)):
        timestamp = i / fps
        rr.set_time("frame_index", sequence=i)
        rr.set_time("timestamp", timestamp=timestamp)

        # Log original frame (BGR → RGB for rerun)
        if source_frames[i] is not None:
            src_rgb = cv2.cvtColor(source_frames[i], cv2.COLOR_BGR2RGB)
            rr.log(f"{source_key}/original", rr.Image(src_rgb))

        # Log segmented frame
        seg_rgb = cv2.cvtColor(segmented_frames[i], cv2.COLOR_BGR2RGB)
        rr.log(f"{new_key}/segmented", rr.Image(seg_rgb))

        # Log mask as a greyscale image (if available)
        if masks and i < len(masks) and masks[i] is not None:
            rr.log(f"{new_key}/mask", rr.Image(masks[i] * 255))

    # Ask user for confirmation
    while True:
        answer = input(
            f"\n  Episode {episode_idx}: Accept tracking result? [y/n]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


# ---------------------------------------------------------------------------
# SAM3 stream
# ---------------------------------------------------------------------------


def add_sam3_stream(
    dataset: LeRobotDataset,
    source_key: str,
    new_key: str,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    fade_pixels: int = 16,
    min_brightness: float = 0.0,
) -> LeRobotDataset:
    """Add a SAM3 video-tracked segmentation stream to the dataset.

    For every episode:

    1. The first frame of ``source_key`` is shown in an interactive OpenCV
       window where the user selects an object using SAM3 point prompts.
    2. The SAM3 **video predictor** propagates the mask across all frames
       of the episode using instance interactivity.
    3. Each frame is highlighted (foreground preserved, background faded).
    4. The resulting stream is previewed in **Rerun** so the user can
       inspect the tracking quality.
    5. The user confirms (``y``) or rejects (``n``) the result.
       If rejected, the raw source frames are used for that episode.
    6. The confirmed frames are encoded as a new video stream.

    Args:
        dataset: Source LeRobotDataset (must contain at least one video key).
        source_key: Existing video key to segment
            (e.g. ``"observation.images.top_back"``).
        new_key: Name for the new segmented stream
            (e.g. ``"observation.images.sam3_top_back"``).
        output_dir: Directory for the new dataset.
        repo_id: Repository ID for the new dataset.
        vcodec: Video codec (default: ``"libsvtav1"``).
        pix_fmt: Pixel format (default: ``"yuv420p"``).
        g: GOP size (default: ``2``).
        crf: Constant rate factor (default: ``30``).
        fade_pixels: Number of pixels for the fade-to-black gradient.
        min_brightness: Background brightness (0.0 = black).

    Returns:
        New :class:`LeRobotDataset` with the SAM3-tracked stream added.
    """
    import gc
    import time

    import rerun as rr

    from lerobot.cameras.zmq.segment import (
        SAM3Segmenter,
        SAM3VideoSegmenter,
        highlight_object_overlay,
        interactive_select,
    )

    if source_key not in dataset.meta.video_keys:
        raise ValueError(
            f"source_key '{source_key}' must be a video stream. "
            f"Video keys in dataset: {dataset.meta.video_keys}."
        )

    if new_key in dataset.meta.features:
        raise ValueError(f"Key '{new_key}' already exists in dataset features.")

    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_sam3"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    # -- Load models --------------------------------------------------------
    logging.info("Loading SAM3 image segmenter (for interactive selection) …")
    image_segmenter = SAM3Segmenter()
    logging.info("Loading SAM3 video segmenter (for temporal propagation) …")
    video_segmenter = SAM3VideoSegmenter()

    total_episodes = dataset.meta.total_episodes
    fps = dataset.meta.fps
    processed_frames: dict[int, list[np.ndarray]] = {}
    episode_lengths: dict[int, int] = {}

    for ep_idx in range(total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        ep_length = ep["length"]
        episode_lengths[ep_idx] = ep_length

        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            f"Decoding {ep_length} frames from '{source_key}' …"
        )
        frames_bgr = _decode_all_frames_bgr(dataset, ep_idx, source_key)

        if not frames_bgr:
            logging.warning(f"Episode {ep_idx}: no frames decoded, skipping.")
            processed_frames[ep_idx] = []
            continue

        # -- Step 1: interactive selection on first frame -------------------
        # SAM3 video predictor needs points, not a mask, so use return_points=True
        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            "Select object to track (Enter=confirm, Esc=skip)"
        )
        _highlighted, mask, sel_points, sel_labels = interactive_select(
            frames_bgr[0],
            image_segmenter,
            window_name=f"SAM3 Track – Episode {ep_idx} ({source_key})",
            fade_pixels=fade_pixels,
            min_brightness=min_brightness,
            return_points=True,
        )

        if mask is None or not sel_points:
            logging.info(f"Episode {ep_idx}: selection cancelled, using raw frames.")
            processed_frames[ep_idx] = frames_bgr
            continue

        # -- Step 2: propagate using SAM3 video predictor -------------------
        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            f"Propagating mask across {ep_length} frames with SAM3 video predictor …"
        )
        masks = video_segmenter.propagate(frames_bgr, sel_points, sel_labels)
        del mask  # free initial mask

        # Apply highlighting to each frame (in-place to save memory)
        highlighted_frames: list[np.ndarray] = []
        for i in range(len(frames_bgr)):
            highlighted = highlight_object_overlay(frames_bgr[i], masks[i])
            highlighted_frames.append(highlighted)
            # Free source frame and mask immediately after use
            frames_bgr[i] = None  # type: ignore[assignment]
            masks[i] = None  # type: ignore[assignment]

        # Re-decode source frames (lightweight) only for the Rerun preview
        logging.info(
            f"Episode {ep_idx + 1}/{total_episodes}: "
            "Starting Rerun preview …"
        )
        preview_source = _decode_all_frames_bgr(dataset, ep_idx, source_key)
        accepted = _preview_and_confirm_rerun(
            episode_idx=ep_idx,
            source_frames=preview_source,
            segmented_frames=highlighted_frames,
            masks=[],  # masks already freed
            fps=fps,
            source_key=source_key,
            new_key=new_key,
        )
        del preview_source

        if accepted:
            logging.info(f"Episode {ep_idx}: confirmed ✓")
            processed_frames[ep_idx] = highlighted_frames
        else:
            logging.info(f"Episode {ep_idx}: rejected, using raw frames.")
            del highlighted_frames
            processed_frames[ep_idx] = _decode_all_frames_bgr(
                dataset, ep_idx, source_key
            )

    # -- Build new dataset structure ----------------------------------------
    source_feature = {
        "dtype": "video",
        "shape": dataset.meta.features[source_key]["shape"],
        "names": dataset.meta.features[source_key].get("names"),
    }
    new_features = dataset.meta.features.copy()
    new_features[new_key] = source_feature

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=True,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Copy parquet data files
    logging.info("Copying data files …")
    data_dir_src = dataset.root / DATA_DIR
    data_dir_dst = new_meta.root / DATA_DIR
    shutil.copytree(data_dir_src, data_dir_dst)

    # Copy existing video files
    logging.info("Copying existing video files …")
    _copy_videos(dataset, new_meta)

    # Copy episodes metadata
    logging.info("Copying episodes metadata …")
    episodes_src = dataset.root / "meta" / "episodes"
    episodes_dst = new_meta.root / "meta" / "episodes"
    if episodes_src.exists():
        shutil.copytree(episodes_src, episodes_dst, dirs_exist_ok=True)

    # Copy tasks and stats
    if dataset.meta.tasks is not None:
        write_tasks(dataset.meta.tasks, new_meta.root)
    if dataset.meta.stats is not None:
        write_stats(dataset.meta.stats, new_meta.root)

    # Group episodes by source video file (same chunk/file layout)
    file_to_episodes: dict[tuple[int, int], list[int]] = {}
    for ep_idx in range(total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        chunk_idx = ep[f"videos/{source_key}/chunk_index"]
        file_idx = ep[f"videos/{source_key}/file_index"]
        file_to_episodes.setdefault((chunk_idx, file_idx), []).append(ep_idx)

    # -- Encode SAM3-tracked stream videos ----------------------------------
    sam3_video_metadata: dict[int, dict] = {}

    for (src_chunk_idx, src_file_idx), eps_in_file in tqdm(
        sorted(file_to_episodes.items()), desc="Encoding SAM3 stream videos"
    ):
        assert new_meta.video_path is not None
        out_video_path = new_meta.root / new_meta.video_path.format(
            video_key=new_key, chunk_index=src_chunk_idx, file_index=src_file_idx
        )

        episode_timestamps = _encode_frame_list_video(
            frames_per_episode=processed_frames,
            episode_lengths=episode_lengths,
            episodes=eps_in_file,
            output_path=out_video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
        )

        for ep_idx, (from_ts, to_ts) in zip(eps_in_file, episode_timestamps, strict=True):
            sam3_video_metadata[ep_idx] = {
                f"videos/{new_key}/chunk_index": src_chunk_idx,
                f"videos/{new_key}/file_index": src_file_idx,
                f"videos/{new_key}/from_timestamp": from_ts,
                f"videos/{new_key}/to_timestamp": to_ts,
            }

    # Update episodes metadata with new video columns
    logging.info("Updating episodes metadata with SAM3 stream info …")
    _update_episodes_metadata_with_video(new_meta, sam3_video_metadata)

    # Populate video codec/resolution info
    first_sam3_path = new_meta.root / new_meta.video_path.format(
        video_key=new_key, chunk_index=0, file_index=0
    )
    if first_sam3_path.exists():
        new_meta.info["features"][new_key]["info"] = get_video_info(first_sam3_path)

    for key in dataset.meta.video_keys:
        if key in dataset.meta.features and dataset.meta.info["features"][key].get("info"):
            new_meta.info["features"][key]["info"] = dataset.meta.info["features"][key]["info"]

    new_meta.info.update(
        {
            "total_episodes": dataset.meta.total_episodes,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "splits": dataset.meta.info.get("splits", {"train": f"0:{dataset.meta.total_episodes}"}),
        }
    )
    write_info(new_meta.info, new_meta.root)

    logging.info(f"SAM3 stream '{new_key}' added. Dataset saved to: {output_dir}")
    return LeRobotDataset(repo_id=repo_id, root=output_dir)
