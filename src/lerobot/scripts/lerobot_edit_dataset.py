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

"""
Edit LeRobot datasets using various transformation tools.

Requires: pip install 'lerobot[dataset]'

This script allows you to delete episodes, split datasets, merge datasets,
remove features, modify tasks, recompute stats, and convert image datasets to video format.
When new_repo_id is specified, creates a new dataset.


Path semantics (v2): --root and --new_root are exact dataset folders containing
meta/, data/, videos/. When omitted, defaults to $HF_LEROBOT_HOME/{repo_id}.

Usage Examples:

Delete episodes 0, 2, and 5 from a dataset:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Delete episodes from a local dataset at a specific path:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --root /path/to/pusht \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Resize all video streams of a dataset to 320x240:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_320x240 \
        --operation.type resize_videos \
        --operation.width 320 \
        --operation.height 240

Resize only selected camera streams and switch codec to h264:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_256 \
        --new_root /path/to/pusht_256 \
        --operation.type resize_videos \
        --operation.width 256 \
        --operation.height 256 \
        --operation.video_keys "['observation.images.laptop']" \
        --operation.vcodec h264

Delete episodes and save to a new dataset at a specific path and with a new repo_id:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_filtered \
        --new_root /path/to/pusht_filtered \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Trim 5 frames from the start and 3 frames from the end of episode 0, and
2 frames from the start of episode 2:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_trimmed \
        --operation.type trim_episodes \
        --operation.episode_trim_specs '{"0": [5, 3], "2": [2, 0]}'

Trim episodes and use the visualization tool to find cut boundaries first:
    # Step 1: Visualize episode 0 to find the frame indices to trim
    lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0

    # Step 2: Trim based on the identified frame indices
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_trimmed \
        --operation.type trim_episodes \
        --operation.episode_trim_specs '{"0": [10, 5]}'

Append trimmed episodes from a source dataset to an existing target dataset (incremental workflow):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type trim_episodes \
        --operation.episode_trim_specs '{"0": [5, 3]}' \
        --operation.append_to_repo_id lerobot/pusht_existing

Split an episode into two at a specific frame position:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_split \
        --operation.type split_episodes \
        --operation.episode_split_specs '{"0": 15}'

Split multiple episodes at specific frame positions:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_split \
        --operation.type split_episodes \
        --operation.episode_split_specs '{"0": 15, "3": 20}'

Split dataset by fractions (pusht_train, pusht_val):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": 0.8, "val": 0.2}'

Split dataset by fractions and save split datasets to a specific folder (base_folder/train, base_folder/val):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_root /path/to/base_folder \
        --operation.type split \
        --operation.splits '{"train": 0.8, "val": 0.2}'

Split dataset by episode indices:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": [0, 1, 2, 3], "val": [4, 5]}'

Split into more than two splits:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": 0.6, "val": 0.2, "test": 0.2}'


Merge multiple datasets:
    lerobot-edit-dataset \
        --new_repo_id lerobot/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"

Merge multiple datasets to a specific output path:
    lerobot-edit-dataset \
        --new_repo_id lerobot/pusht_merged \
        --new_root /path/to/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"

Merge multiple datasets from a list of local dataset paths:
    lerobot-edit-dataset \
        --new_repo_id lerobot/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['pusht_train', 'pusht_val']" \
        --operation.roots "['/path/to/pusht_train', '/path/to/pusht_val']"

Remove camera feature:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type remove_feature \
        --operation.feature_names "['observation.image']"

Rename feature keys (e.g. rename a camera stream):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_renamed \
        --operation.type rename_features \
        --operation.rename_map '{"observation.images.top": "observation.images.main"}'


Modify tasks - set a single task for all episodes (WARNING: modifies in-place):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type modify_tasks \
        --operation.new_task "Pick up the cube and place it"

Modify tasks - set different tasks for specific episodes (WARNING: modifies in-place):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type modify_tasks \
        --operation.episode_tasks '{"0": "Task A", "1": "Task B", "2": "Task A"}'

Modify tasks - set default task with overrides for specific episodes (WARNING: modifies in-place):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type modify_tasks \
        --operation.new_task "Default task" \
        --operation.episode_tasks '{"5": "Special task for episode 5"}'

Convert image dataset to video format and save locally:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --new_root /path/to/output/pusht_video \
        --operation.type convert_image_to_video

Convert image dataset to video format and save with new repo_id:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --new_repo_id lerobot/pusht_video \
        --operation.type convert_image_to_video

Convert image dataset to video format and push to hub:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --new_repo_id lerobot/pusht_video \
        --operation.type convert_image_to_video \
        --push_to_hub true

Add a black stream (all-black frames with the same resolution as a source camera):
    lerobot-edit-dataset \
        --repo_id my_user/my_video_dataset \
        --new_repo_id my_user/my_video_dataset_with_black \
        --operation.type add_black_stream \
        --operation.source_key observation.images.top \
        --operation.new_key observation.images.black_top

Add a black stream for specific episodes only:
    lerobot-edit-dataset \
        --repo_id my_user/my_video_dataset \
        --new_repo_id my_user/my_video_dataset_with_black \
        --operation.type add_black_stream \
        --operation.source_key observation.images.top \
        --operation.new_key observation.images.black_top \
        --operation.episode_indices "[0, 1, 2]"

Append a black stream from one dataset to an existing target dataset (incremental workflow):
    lerobot-edit-dataset \
        --repo_id my_user/my_video_dataset \
        --operation.type add_black_stream \
        --operation.source_key observation.images.top \
        --operation.new_key observation.images.black_top \
        --operation.append_to_repo_id my_user/my_existing_dataset

Add a guide stream (repeats the first frame of a camera throughout each episode):
    lerobot-edit-dataset \
        --repo_id my_user/my_video_dataset \
        --new_repo_id my_user/my_video_dataset_with_guide \
        --operation.type add_guide_stream \
        --operation.source_key observation.images.laptop \
        --operation.new_key observation.images.guide_laptop

Add a guide stream and push to hub:
    lerobot-edit-dataset \
        --repo_id my_user/my_video_dataset \
        --new_repo_id my_user/my_video_dataset_with_guide \
        --operation.type add_guide_stream \
        --operation.source_key observation.images.top \
        --operation.new_key observation.images.guide_top \
        --push_to_hub true

Add a segmented scene stream (interactive SAM2 segmentation of first frame per episode):
    lerobot-edit-dataset \
        --repo_id my_user/my_video_dataset \
        --new_repo_id my_user/my_video_dataset_with_seg \
        --operation.type add_sam2_initial_segment \
        --operation.source_key observation.images.laptop \
        --operation.new_key observation.images.segmented_laptop

Show dataset information:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --operation.type info \
        --operation.show_features true

Show dataset information without feature details:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --operation.type info \
        --operation.show_features false

Copy all episodes from one dataset to a new dataset:
    lerobot-edit-dataset \
        --repo_id source/dataset \
        --new_repo_id target/new_dataset \
        --operation.type copy_episodes

Copy specific episodes only:
    lerobot-edit-dataset \
        --repo_id source/dataset \
        --new_repo_id target/new_dataset \
        --operation.type copy_episodes \
        --operation.episode_indices "[0, 2, 5]"

Copy only specific camera streams:
    lerobot-edit-dataset \
        --repo_id source/dataset \
        --new_repo_id target/new_dataset \
        --operation.type copy_episodes \
        --operation.camera_keys "['observation.images.top']"

Copy and rename camera streams:
    lerobot-edit-dataset \
        --repo_id source/dataset \
        --new_repo_id target/new_dataset \
        --operation.type copy_episodes \
        --operation.camera_key_mapping '{"observation.images.top": "observation.images.main"}'

Copy episodes and append to an existing dataset:
    lerobot-edit-dataset \
        --repo_id source/dataset \
        --operation.type copy_episodes \
        --operation.episode_indices "[0, 2, 5]" \
        --operation.append_to_repo_id target/existing_dataset

Resize all video streams to 320x240:
    lerobot-edit-dataset \
        --repo_id my_user/my_dataset \
        --new_repo_id my_user/my_dataset_320x240 \
        --operation.type resize_videos \
        --operation.resize_shape "[240, 320]"
Recompute dataset statistics:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type recompute_stats

Recompute stats for relative actions and push to hub:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type recompute_stats \
        --operation.relative_action true \
        --operation.chunk_size 50 \
        --operation.relative_exclude_joints "['gripper']" \
        --operation.num_workers 4 \
        --push_to_hub true

Check dataset consistency only (dry-run):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --root /path/to/pusht \
        --operation.type repair_inconsistencies \
        --operation.dry_run true

Repair dataset inconsistencies into a new dataset:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --root /path/to/pusht \
        --new_repo_id lerobot/pusht_repaired \
        --new_root /path/to/pusht_repaired \
        --operation.type repair_inconsistencies \
        --operation.overwrite_output true

Using JSON config file:
    lerobot-edit-dataset \
        --config_path path/to/edit_config.json
"""

import abc
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset

from lerobot.datasets.dataset_tools import (
    add_black_stream,
    check_dataset_inconsistencies,
    add_guide_stream,
    add_sam2_initial_segment,
    add_sam2_stream,
    add_sam3_stream,

    convert_image_to_video_dataset,
    copy_episodes,
    delete_episodes,
    merge_datasets,
    modify_tasks,
    repair_dataset_inconsistencies,
    recompute_stats,
    remove_feature,
    resize_video_streams,
    rename_features,
    resize_videos,
    split_dataset,
    split_episodes,
    trim_episodes,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


@dataclass
class OperationConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@OperationConfig.register_subclass("trim_episodes")
@dataclass
class TrimEpisodesConfig(OperationConfig):
    # Keys are episode indices as strings because JSON/CLI argument parsing
    # always produces string dict keys.  They are converted to int in
    # handle_trim_episodes before being passed to trim_episodes().
    episode_trim_specs: dict[str, tuple[int, int]] | None = None
    # When set, trimmed episodes are appended to this existing repo instead of
    # creating a new dataset.  Mutually exclusive with EditDatasetConfig.new_repo_id.
    append_to_repo_id: str | None = None


@OperationConfig.register_subclass("split_episodes")
@dataclass
class SplitEpisodesConfig(OperationConfig):
    # Keys are episode indices as strings because JSON/CLI argument parsing
    # always produces string dict keys. They are converted to int in
    # handle_split_episodes before being passed to split_episodes().
    episode_split_specs: dict[str, int] | None = None


@OperationConfig.register_subclass("delete_episodes")
@dataclass
class DeleteEpisodesConfig(OperationConfig):
    episode_indices: list[int] | None = None
    # Video codec for re-encoded segments. None = preserve source dataset codec.
    vcodec: str | None = None


@OperationConfig.register_subclass("resize_videos")
@dataclass
class ResizeVideosConfig(OperationConfig):
    """Configuration for resizing video streams of a dataset.

    Every targeted video file is decoded, resized frame-by-frame and re-encoded.
    Non-targeted video keys are copied as-is. The feature ``shape`` and the
    per-video entries in ``meta/info.json`` are updated to reflect the new
    resolution and (if changed) codec.
    """

    # Target frame width in pixels (must be positive).
    width: int = 0
    # Target frame height in pixels (must be positive).
    height: int = 0
    # Optional list of video feature keys to resize. None = resize all.
    video_keys: list[str] | None = None
    # Video codec for re-encoding. None = preserve source dataset codec.
    vcodec: str | None = None
    pix_fmt: str = "yuv420p"


@OperationConfig.register_subclass("copy_episodes")
@dataclass
class CopyEpisodesConfig(OperationConfig):
    """Configuration for copying episodes from one dataset to another.

    All non-camera features (e.g. action, observation.state) are always copied.
    Use *camera_keys* to restrict which camera/image streams are included, and
    *camera_key_mapping* to rename them in the target dataset.
    """

    # Episode indices to copy. None = copy all episodes.
    episode_indices: list[int] | None = None
    # Camera/image stream keys to include. None = include all camera streams.
    camera_keys: list[str] | None = None
    # Optional dict mapping source camera key names to target camera key names.
    camera_key_mapping: dict[str, str] | None = None
    # When set, copied episodes are appended to this existing repo instead of
    # creating a new dataset.  Mutually exclusive with EditDatasetConfig.new_repo_id.
    append_to_repo_id: str | None = None


@OperationConfig.register_subclass("split")
@dataclass
class SplitConfig(OperationConfig):
    splits: dict[str, float | list[int]] | None = None


@OperationConfig.register_subclass("merge")
@dataclass
class MergeConfig(OperationConfig):
    repo_ids: list[str] | None = None
    roots: list[str] | None = None


@OperationConfig.register_subclass("remove_feature")
@dataclass
class RemoveFeatureConfig(OperationConfig):
    feature_names: list[str] | None = None


# @OperationConfig.register_subclass("resize_videos")
# @dataclass
# class ResizeVideosConfig(OperationConfig):
#     resize_shape: tuple[int, int] | None = None
#     interpolation_mode: str = "bilinear"
#     vcodec: str | None = None


@OperationConfig.register_subclass("rename_features")
@dataclass
class RenameFeaturesConfig(OperationConfig):
    """Configuration for renaming features of a dataset.

    ``rename_map`` maps existing feature keys to their new names. Works for
    state / action / image / video features. Required features (``timestamp``,
    ``frame_index``, ``episode_index``, ``index``, ``task_index``) cannot be
    renamed.
    """

    rename_map: dict[str, str] | None = None


@OperationConfig.register_subclass("modify_tasks")
@dataclass
class ModifyTasksConfig(OperationConfig):
    new_task: str | None = None
    episode_tasks: dict[str, str] | None = None


@OperationConfig.register_subclass("convert_image_to_video")
@dataclass
class ConvertImageToVideoConfig(OperationConfig):
    output_dir: str | None = None
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int = 2
    crf: int = 30
    fast_decode: int = 0
    episode_indices: list[int] | None = None
    num_workers: int = 4
    max_episodes_per_batch: int | None = None
    max_frames_per_batch: int | None = None


@OperationConfig.register_subclass("add_black_stream")
@dataclass
class AddBlackStreamConfig(OperationConfig):
    """Configuration for adding an all-black video stream to a dataset.

    The new stream contains only black (zero-valued) frames for every frame of
    every episode.  Its spatial dimensions are taken from ``source_key`` so the
    stream is resolution-compatible with the rest of the dataset.  Useful as a
    placeholder when a second camera is not yet available.
    """

    source_key: str = ""
    new_key: str = ""
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int = 2
    crf: int = 30
    episode_indices: list[int] | None = None
    append_to_repo_id: str | None = None


@OperationConfig.register_subclass("add_guide_stream")
@dataclass
class AddGuideStreamConfig(OperationConfig):
    """Configuration for adding a guide video stream to a dataset.

    The guide stream repeats the first frame of ``source_key`` throughout every
    episode.  It is encoded as a regular video stream under ``new_key`` so that
    any policy or visualisation tool can use it as a reference image.
    """

    source_key: str = ""
    new_key: str = ""
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int = 2
    crf: int = 30
    episode_indices: list[int] | None = None
    append_to_repo_id: str | None = None


@OperationConfig.register_subclass("add_sam2_initial_segment")
@dataclass
class AddSam2InitialSegmentConfig(OperationConfig):
    """Configuration for adding a segmented scene stream.

    For each episode the first frame of ``source_key`` is shown in an
    interactive OpenCV window where the user segments an object with SAM2.
    The highlighted image is then repeated for the full episode duration,
    just like :class:`AddGuideStreamConfig`.
    """

    source_key: str = ""
    new_key: str = ""
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int = 2
    crf: int = 30
    fade_pixels: int = 16
    min_brightness: float = 0.0
    episode_indices: list[int] | None = None
    append_to_repo_id: str | None = None


@OperationConfig.register_subclass("add_sam2_stream")
@dataclass
class AddSam2StreamConfig(OperationConfig):
    """Configuration for adding a SAM2 video-tracked segmentation stream.

    For each episode the first frame of ``source_key`` is shown in an
    interactive OpenCV window where the user selects an object.  SAM2's
    video predictor then propagates the mask across **every** frame of
    the episode.  The result is previewed in Rerun before the user
    confirms or rejects it.
    """

    source_key: str = ""
    new_key: str = ""
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int = 2
    crf: int = 30
    fade_pixels: int = 16
    min_brightness: float = 0.0
    episode_indices: list[int] | None = None
    append_to_repo_id: str | None = None


@OperationConfig.register_subclass("add_sam3_stream")
@dataclass
class AddSam3StreamConfig(OperationConfig):
    """Configuration for adding a SAM3 video-tracked segmentation stream.

    For each episode the first frame of ``source_key`` is shown in an
    interactive OpenCV window where the user selects an object.  SAM3's
    video predictor then propagates the selected object across **every**
    frame of the episode.  The result is previewed in Rerun before the
    user confirms or rejects it.
    """

    source_key: str = ""
    new_key: str = ""
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    g: int = 2
    crf: int = 30
    fade_pixels: int = 16
    min_brightness: float = 0.0


@OperationConfig.register_subclass("recompute_stats")
@dataclass
class RecomputeStatsConfig(OperationConfig):
    skip_image_video: bool = True
    relative_action: bool = False
    relative_exclude_joints: list[str] | None = None
    chunk_size: int = 50
    num_workers: int = 0


@OperationConfig.register_subclass("repair_inconsistencies")
@dataclass
class RepairInconsistenciesConfig(OperationConfig):
    # If True, only report inconsistencies and do not write output dataset.
    dry_run: bool = False
    # Replace output directory if it already exists.
    overwrite_output: bool = False
    # Recompute stats.json after repairing the dataset.
    recompute_stats_after: bool = True
    # If recomputing stats, skip image/video features.
    skip_image_video_stats: bool = True


@OperationConfig.register_subclass("info")
@dataclass
class InfoConfig(OperationConfig):
    show_features: bool = False


@dataclass
class EditDatasetConfig:
    # Operation configuration.
    operation: OperationConfig
    # Input dataset identifier. Always required unless for Merge operation.
    repo_id: str | None = None
    # Root directory where the input dataset is stored. If not specified, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | None = None
    # Edited dataset identifier. When both new_repo_id (resp. new_root) and repo_id (resp. root) are identical, modifications are applied in-place and a backup of the original dataset is created. Required for Merge operation.
    new_repo_id: str | None = None
    # Root directory where the edited dataset will be stored. If not specified, defaults to $HF_LEROBOT_HOME/new_repo_id. For Split operation, this is the base directory for the split datasets.
    new_root: str | None = None
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False


def get_output_path(
    repo_id: str,
    new_repo_id: str | None,
    root: Path | str | None,
    new_root: Path | str | None,
) -> tuple[str, Path]:
    input_path = Path(root) if root else HF_LEROBOT_HOME / repo_id

    output_repo_id = new_repo_id if new_repo_id else repo_id
    output_path = Path(new_root) if new_root else HF_LEROBOT_HOME / output_repo_id

    # In case of in-place modification, create a backup of the original dataset (if it exists)
    if output_path == input_path:
        backup_path = input_path.with_name(input_path.name + "_old")

        if input_path.exists():
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(input_path, backup_path)

    return output_repo_id, output_path


def handle_delete_episodes(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, DeleteEpisodesConfig):
        raise ValueError("Operation config must be DeleteEpisodesConfig")

    if not cfg.operation.episode_indices:
        raise ValueError("episode_indices must be specified for delete_episodes operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id,
        new_repo_id=cfg.new_repo_id,
        root=cfg.root,
        new_root=cfg.new_root,
    )

    # In case of in-place modification, make the dataset point to the backup directory
    if output_dir == dataset.root:
        dataset.root = dataset.root.with_name(dataset.root.name + "_old")

    logging.info(f"Deleting episodes {cfg.operation.episode_indices} from {cfg.repo_id}")
    new_dataset = delete_episodes(
        dataset,
        episode_indices=cfg.operation.episode_indices,
        output_dir=output_dir,
        repo_id=output_repo_id,
        vcodec=cfg.operation.vcodec,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}, Frames: {new_dataset.meta.total_frames}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_resize_videos(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, ResizeVideosConfig):
        raise ValueError("Operation config must be ResizeVideosConfig")

    if cfg.operation.width <= 0 or cfg.operation.height <= 0:
        raise ValueError(
            "--operation.width and --operation.height must be positive integers "
            f"(got {cfg.operation.width}x{cfg.operation.height})"
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id,
        new_repo_id=cfg.new_repo_id,
        root=cfg.root,
        new_root=cfg.new_root,
    )

    # In case of in-place modification, make the source point to a backup dir.
    if output_dir == dataset.root:
        dataset.root = dataset.root.with_name(dataset.root.name + "_old")

    logging.info(
        f"Resizing videos of {cfg.repo_id} to "
        f"{cfg.operation.width}x{cfg.operation.height}"
    )
    new_dataset = resize_videos(
        dataset,
        width=cfg.operation.width,
        height=cfg.operation.height,
        output_dir=output_dir,
        repo_id=output_repo_id,
        video_keys=cfg.operation.video_keys,
        vcodec=cfg.operation.vcodec,
        pix_fmt=cfg.operation.pix_fmt,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(
        f"Episodes: {new_dataset.meta.total_episodes}, Frames: {new_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_copy_episodes(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, CopyEpisodesConfig):
        raise ValueError("Operation config must be CopyEpisodesConfig")

    if cfg.operation.append_to_repo_id is not None and cfg.new_repo_id is not None:
        raise ValueError(
            "Cannot specify both 'operation.append_to_repo_id' and 'new_repo_id'. "
            "Use 'operation.append_to_repo_id' to append to an existing dataset, "
            "or 'new_repo_id' to create a fresh one."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    if cfg.operation.append_to_repo_id is not None:
        append_root = Path(cfg.root) if cfg.root else HF_LEROBOT_HOME
        append_to_dir = append_root / cfg.operation.append_to_repo_id
        append_to_dataset = LeRobotDataset(cfg.operation.append_to_repo_id, root=append_to_dir)

        logging.info(
            f"Copying episodes from {cfg.repo_id} and appending to {cfg.operation.append_to_repo_id}"
        )
        result_dataset = copy_episodes(
            src_dataset=dataset,
            episode_indices=cfg.operation.episode_indices,
            camera_keys=cfg.operation.camera_keys,
            camera_key_mapping=cfg.operation.camera_key_mapping,
            append_to_dataset=append_to_dataset,
        )
        logging.info(f"Dataset saved to {append_to_dir}")
    else:
        output_repo_id, output_dir = get_output_path(
            cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
        )

        logging.info(f"Copying episodes from {cfg.repo_id} to {output_repo_id}")
        result_dataset = copy_episodes(
            src_dataset=dataset,
            episode_indices=cfg.operation.episode_indices,
            camera_keys=cfg.operation.camera_keys,
            camera_key_mapping=cfg.operation.camera_key_mapping,
            output_dir=output_dir,
            repo_id=output_repo_id,
        )
        logging.info(f"Dataset saved to {output_dir}")

    logging.info(
        f"Episodes: {result_dataset.meta.total_episodes}, Frames: {result_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {result_dataset.repo_id}")
        result_dataset.push_to_hub()
        logging.info("✓ Successfully pushed to hub!")


def handle_trim_episodes(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, TrimEpisodesConfig):
        raise ValueError("Operation config must be TrimEpisodesConfig")

    if not cfg.operation.episode_trim_specs:
        raise ValueError(
            "episode_trim_specs must be specified for trim_episodes operation. "
            "Provide a dict mapping episode indices to (trim_start, trim_end) tuples, e.g. "
            '\'{"0": [5, 3], "2": [2, 0]}\''
        )

    if cfg.operation.append_to_repo_id is not None and cfg.new_repo_id is not None:
        raise ValueError(
            "Cannot specify both 'operation.append_to_repo_id' and 'new_repo_id'. "
            "Use 'operation.append_to_repo_id' to append to an existing dataset, "
            "or 'new_repo_id' to create a fresh one."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    # Convert string keys to int (CLI args come in as strings)
    episode_trim_specs: dict[int, tuple[int, int]] = {}
    for k, v in cfg.operation.episode_trim_specs.items():
        episode_trim_specs[int(k)] = (int(v[0]), int(v[1]))

    if cfg.operation.append_to_repo_id is not None:
        # Append mode: load the target dataset and append trimmed episodes to it.
        append_root = Path(cfg.root) if cfg.root else HF_LEROBOT_HOME
        append_to_dir = append_root / cfg.operation.append_to_repo_id
        append_to_dataset = LeRobotDataset(cfg.operation.append_to_repo_id, root=append_to_dir)

        logging.info(
            f"Appending trimmed episodes from {cfg.repo_id} to {cfg.operation.append_to_repo_id}"
        )
        result_dataset = trim_episodes(
            dataset,
            episode_trim_specs=episode_trim_specs,
            append_to_dataset=append_to_dataset,
        )

        logging.info(f"Dataset saved to {append_to_dir}")
    else:
        # Normal mode: write trimmed output to a new (or in-place replaced) dataset.
        output_repo_id, output_dir = get_output_path(
            cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
        )

        if cfg.new_repo_id is None:
            dataset.root = Path(str(dataset.root) + "_old")

        logging.info(f"Trimming episodes {list(episode_trim_specs.keys())} in {cfg.repo_id}")
        result_dataset = trim_episodes(
            dataset,
            episode_trim_specs=episode_trim_specs,
            output_dir=output_dir,
            repo_id=output_repo_id,
        )

        logging.info(f"Dataset saved to {output_dir}")

    logging.info(
        f"Episodes: {result_dataset.meta.total_episodes}, Frames: {result_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {result_dataset.repo_id}")
        result_dataset.push_to_hub()


def handle_split_episodes(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, SplitEpisodesConfig):
        raise ValueError("Operation config must be SplitEpisodesConfig")

    if not cfg.operation.episode_split_specs:
        raise ValueError(
            "episode_split_specs must be specified for split_episodes operation. "
            "Provide a dict mapping episode indices to split frame positions, e.g. "
            '\'{"0": 15, "3": 20}\''
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
    )

    if cfg.new_repo_id is None:
        dataset.root = Path(str(dataset.root) + "_old")

    # Convert string keys to int (CLI args come in as strings)
    episode_split_specs: dict[int, int] = {int(k): int(v) for k, v in cfg.operation.episode_split_specs.items()}

    logging.info(f"Splitting episodes {list(episode_split_specs.keys())} in {cfg.repo_id}")
    result_dataset = split_episodes(
        dataset,
        episode_split_specs=episode_split_specs,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(
        f"Episodes: {result_dataset.meta.total_episodes}, Frames: {result_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {result_dataset.repo_id}")
        result_dataset.push_to_hub()


def handle_split(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, SplitConfig):
        raise ValueError("Operation config must be SplitConfig")

    if not cfg.operation.splits:
        raise ValueError(
            "splits dict must be specified with split names as keys and fractions/episode lists as values"
        )

    if cfg.new_repo_id is not None:
        logging.warning(
            "split uses the original dataset identifier --repo_id to generate split names. The --new_repo_id parameter is ignored."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    logging.info(f"Splitting dataset {cfg.repo_id} with splits: {cfg.operation.splits}")
    split_datasets = split_dataset(
        dataset,
        splits=cfg.operation.splits,
        output_dir=cfg.new_root,
    )

    for split_name, split_ds in split_datasets.items():
        logging.info(
            f"{split_name}: {split_ds.meta.total_episodes} episodes, {split_ds.meta.total_frames} frames"
        )

        if cfg.push_to_hub:
            logging.info(f"Pushing {split_name} split to hub as {split_ds.repo_id}")
            LeRobotDataset(split_ds.repo_id, root=split_ds.root).push_to_hub()


def handle_merge(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, MergeConfig):
        raise ValueError("Operation config must be MergeConfig")

    if not cfg.operation.repo_ids:
        raise ValueError("repo_ids must be specified for merge operation")

    if cfg.repo_id is not None or cfg.root is not None:
        logging.warning(
            "merge uses --new_repo_id and --new_root for the merged dataset. The --repo_id and --root parameters are ignored."
        )

    if cfg.operation.roots:
        if len(cfg.operation.roots) != len(cfg.operation.repo_ids):
            raise ValueError("repo_ids and roots must have the same length for merge operation")
        logging.info(f"Loading {len(cfg.operation.roots)} datasets to merge")
        datasets = [
            LeRobotDataset(repo_id=repo_id, root=root)
            for repo_id, root in zip(cfg.operation.repo_ids, cfg.operation.roots, strict=True)
        ]
    else:
        logging.info(f"Loading {len(cfg.operation.repo_ids)} datasets to merge")
        datasets = [LeRobotDataset(repo_id) for repo_id in cfg.operation.repo_ids]

    output_dir = Path(cfg.new_root) if cfg.new_root else HF_LEROBOT_HOME / cfg.new_repo_id

    logging.info(f"Merging datasets into {cfg.new_repo_id}")
    merged_dataset = merge_datasets(
        datasets,
        output_repo_id=cfg.new_repo_id,
        output_dir=output_dir,
    )

    logging.info(f"Merged dataset saved to {output_dir}")
    logging.info(
        f"Episodes: {merged_dataset.meta.total_episodes}, Frames: {merged_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {cfg.new_repo_id}")
        LeRobotDataset(merged_dataset.repo_id, root=output_dir).push_to_hub()


def handle_remove_feature(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RemoveFeatureConfig):
        raise ValueError("Operation config must be RemoveFeatureConfig")

    if not cfg.operation.feature_names:
        raise ValueError("feature_names must be specified for remove_feature operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id,
        new_repo_id=cfg.new_repo_id,
        root=cfg.root,
        new_root=cfg.new_root,
    )

    # In case of in-place modification, make the dataset point to the backup directory
    if output_dir == dataset.root:
        dataset.root = dataset.root.with_name(dataset.root.name + "_old")

    logging.info(f"Removing features {cfg.operation.feature_names} from {cfg.repo_id}")
    new_dataset = remove_feature(
        dataset,
        feature_names=cfg.operation.feature_names,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Remaining features: {list(new_dataset.meta.features.keys())}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_resize_videos(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, ResizeVideosConfig):
        raise ValueError("Operation config must be ResizeVideosConfig")

    if cfg.operation.resize_shape is None:
        raise ValueError("operation.resize_shape must be specified for resize_videos operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id,
        new_repo_id=cfg.new_repo_id,
        root=cfg.root,
        new_root=cfg.new_root,
    )

    if output_dir == dataset.root:
        dataset.root = dataset.root.with_name(dataset.root.name + "_old")

    logging.info(
        "Resizing all video streams to (H=%d, W=%d)",
        cfg.operation.resize_shape[0],
        cfg.operation.resize_shape[1],
    )
    new_dataset = resize_video_streams(
        dataset=dataset,
        resize_shape=cfg.operation.resize_shape,
        output_dir=output_dir,
        repo_id=output_repo_id,
        vcodec=cfg.operation.vcodec,
        interpolation_mode=cfg.operation.interpolation_mode,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}, Frames: {new_dataset.meta.total_frames}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_rename_features(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RenameFeaturesConfig):
        raise ValueError("Operation config must be RenameFeaturesConfig")

    if not cfg.operation.rename_map:
        raise ValueError(
            "--operation.rename_map must be a non-empty dict, e.g. "
            "'{\"observation.images.top\": \"observation.images.main\"}'"
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id,
        new_repo_id=cfg.new_repo_id,
        root=cfg.root,
        new_root=cfg.new_root,
    )

    # In case of in-place modification, make the source point to a backup dir.
    if output_dir == dataset.root:
        dataset.root = dataset.root.with_name(dataset.root.name + "_old")

    logging.info(
        f"Renaming features {cfg.operation.rename_map} of {cfg.repo_id}"
    )
    new_dataset = rename_features(
        dataset,
        rename_map=cfg.operation.rename_map,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Features after rename: {list(new_dataset.meta.features.keys())}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_modify_tasks(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, ModifyTasksConfig):
        raise ValueError("Operation config must be ModifyTasksConfig")

    new_task = cfg.operation.new_task
    episode_tasks_raw = cfg.operation.episode_tasks

    if new_task is None and episode_tasks_raw is None:
        raise ValueError("Must specify at least one of new_task or episode_tasks for modify_tasks operation")

    if cfg.new_repo_id is not None or cfg.new_root is not None:
        logging.warning(
            "modify_tasks modifies datasets in-place. The --new_repo_id and --new_root parameters are ignored."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    logging.warning(f"Modifying dataset in-place at {dataset.root}. Original data will be overwritten.")

    # Convert episode_tasks keys from string to int if needed (CLI passes strings)
    episode_tasks: dict[int, str] | None = None
    if episode_tasks_raw is not None:
        episode_tasks = {int(k): v for k, v in episode_tasks_raw.items()}

    logging.info(f"Modifying tasks in {cfg.repo_id}")
    if new_task:
        logging.info(f"  Default task: '{new_task}'")
    if episode_tasks:
        logging.info(f"  Episode-specific tasks: {episode_tasks}")

    modified_dataset = modify_tasks(
        dataset,
        new_task=new_task,
        episode_tasks=episode_tasks,
    )

    logging.info(f"Dataset modified at {dataset.root}")
    logging.info(f"Tasks: {list(modified_dataset.meta.tasks.index)}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {cfg.repo_id}")
        modified_dataset.push_to_hub()


def handle_convert_image_to_video(cfg: EditDatasetConfig) -> None:
    # Note: Parser may create any config type with the right fields, so we access fields directly
    # instead of checking isinstance()
    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    # Determine output directory and repo_id
    # Priority: 1) new_root, 2) new_repo_id, 3) operation.output_dir, 4) auto-generated name
    output_dir_config = getattr(cfg.operation, "output_dir", None)
    if output_dir_config:
        logging.warning(
            "--operation.output_dir is deprecated and will be removed in future versions. "
            "Please use --new_root instead."
        )

    if cfg.new_root:
        output_dir = Path(cfg.new_root)
        output_repo_id = cfg.new_repo_id or f"{cfg.repo_id}_video"
        logging.info(f"Saving to new_root: {output_dir} as {output_repo_id}")
    elif cfg.new_repo_id:
        output_repo_id = cfg.new_repo_id
        output_dir = HF_LEROBOT_HOME / cfg.new_repo_id
        logging.info(f"Saving to new dataset: {cfg.new_repo_id} at {output_dir}")
    elif output_dir_config:
        output_dir = Path(output_dir_config)
        output_repo_id = output_dir.name
        logging.info(f"Saving to local directory: {output_dir} as {output_repo_id}")
    else:
        output_repo_id = f"{cfg.repo_id}_video"
        output_dir = HF_LEROBOT_HOME / output_repo_id
        logging.info(f"Saving to auto-generated location: {output_dir} as {output_repo_id}")

    logging.info(f"Converting dataset {cfg.repo_id} to video format")

    new_dataset = convert_image_to_video_dataset(
        dataset=dataset,
        output_dir=output_dir,
        repo_id=output_repo_id,
        vcodec=getattr(cfg.operation, "vcodec", "libsvtav1"),
        pix_fmt=getattr(cfg.operation, "pix_fmt", "yuv420p"),
        g=getattr(cfg.operation, "g", 2),
        crf=getattr(cfg.operation, "crf", 30),
        fast_decode=getattr(cfg.operation, "fast_decode", 0),
        episode_indices=getattr(cfg.operation, "episode_indices", None),
        num_workers=getattr(cfg.operation, "num_workers", 4),
        max_episodes_per_batch=getattr(cfg.operation, "max_episodes_per_batch", None),
        max_frames_per_batch=getattr(cfg.operation, "max_frames_per_batch", None),
    )

    logging.info("Video dataset created successfully!")
    logging.info(f"Location: {output_dir}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}")
    logging.info(f"Frames: {new_dataset.meta.total_frames}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}...")
        new_dataset.push_to_hub()
        logging.info("✓ Successfully pushed to hub!")
    else:
        logging.info("Dataset saved locally (not pushed to hub)")


def handle_recompute_stats(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RecomputeStatsConfig):
        raise ValueError("Operation config must be RecomputeStatsConfig")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    logging.info(f"Recomputing stats for {cfg.repo_id}")
    if cfg.operation.relative_action:
        logging.info(
            f"Relative action stats enabled (chunk_size={cfg.operation.chunk_size}, "
            f"exclude_joints={cfg.operation.relative_exclude_joints})"
        )

    recompute_stats(
        dataset,
        skip_image_video=cfg.operation.skip_image_video,
        relative_action=cfg.operation.relative_action,
        relative_exclude_joints=cfg.operation.relative_exclude_joints,
        chunk_size=cfg.operation.chunk_size,
        num_workers=cfg.operation.num_workers,
    )

    logging.info(f"Stats written to {dataset.root}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {dataset.meta.repo_id}...")
        dataset.push_to_hub()


def handle_repair_inconsistencies(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RepairInconsistenciesConfig):
        raise ValueError("Operation config must be RepairInconsistenciesConfig")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    report = check_dataset_inconsistencies(dataset)
    logging.info(f"Consistency report: {report}")

    if cfg.operation.dry_run:
        return

    output_repo_id, output_dir = get_output_path(
        cfg.repo_id,
        new_repo_id=cfg.new_repo_id,
        root=cfg.root,
        new_root=cfg.new_root,
    )

    repaired_dataset = repair_dataset_inconsistencies(
        dataset,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
        dry_run=False,
        overwrite_output=cfg.operation.overwrite_output,
        recompute_stats_after=cfg.operation.recompute_stats_after,
        skip_image_video_stats=cfg.operation.skip_image_video_stats,
    )

    if repaired_dataset is None:
        return

    logging.info(f"Repaired dataset saved to {output_dir}")
    logging.info(
        f"Episodes: {repaired_dataset.meta.total_episodes}, Frames: {repaired_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {repaired_dataset.repo_id}...")
        repaired_dataset.push_to_hub()


def _get_dataset_size(repo_path):
    import os

    total = 0
    with os.scandir(repo_path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += _get_dataset_size(entry.path)
    return total


def handle_add_black_stream(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, AddBlackStreamConfig):
        raise ValueError("Operation config must be AddBlackStreamConfig")

    if not cfg.operation.source_key:
        raise ValueError("operation.source_key must be specified for add_black_stream operation")

    if not cfg.operation.new_key:
        raise ValueError("operation.new_key must be specified for add_black_stream operation")

    if cfg.operation.append_to_repo_id is not None and cfg.new_repo_id is not None:
        raise ValueError(
            "Cannot specify both 'operation.append_to_repo_id' and 'new_repo_id'. "
            "Use 'operation.append_to_repo_id' to append to an existing dataset, "
            "or 'new_repo_id' to create a fresh one."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    if cfg.operation.append_to_repo_id is not None:
        append_root = Path(cfg.root) if cfg.root else HF_LEROBOT_HOME
        append_to_dir = append_root / cfg.operation.append_to_repo_id
        append_to_dataset = LeRobotDataset(cfg.operation.append_to_repo_id, root=append_to_dir)

        logging.info(
            f"Appending black stream '{cfg.operation.new_key}' "
            f"(resolution from '{cfg.operation.source_key}') "
            f"from {cfg.repo_id} to {cfg.operation.append_to_repo_id}"
        )
        result_dataset = add_black_stream(
            dataset=dataset,
            source_key=cfg.operation.source_key,
            new_key=cfg.operation.new_key,
            vcodec=cfg.operation.vcodec,
            pix_fmt=cfg.operation.pix_fmt,
            g=cfg.operation.g,
            crf=cfg.operation.crf,
            episode_indices=cfg.operation.episode_indices,
            append_to_dataset=append_to_dataset,
        )
        logging.info(f"Dataset saved to {append_to_dir}")
    else:
        output_repo_id, output_dir = get_output_path(
            cfg.repo_id,
            new_repo_id=cfg.new_repo_id,
            root=cfg.root,
            new_root=cfg.new_root,
        )

        if cfg.new_repo_id is None:
            dataset.root = Path(str(dataset.root) + "_old")

        logging.info(
            f"Adding black stream '{cfg.operation.new_key}' "
            f"(resolution from '{cfg.operation.source_key}') to {cfg.repo_id}"
        )
        result_dataset = add_black_stream(
            dataset=dataset,
            source_key=cfg.operation.source_key,
            new_key=cfg.operation.new_key,
            output_dir=output_dir,
            repo_id=output_repo_id,
            vcodec=cfg.operation.vcodec,
            pix_fmt=cfg.operation.pix_fmt,
            g=cfg.operation.g,
            crf=cfg.operation.crf,
            episode_indices=cfg.operation.episode_indices,
        )
        logging.info(f"Dataset saved to {output_dir}")

    logging.info(
        f"Episodes: {result_dataset.meta.total_episodes}, "
        f"Frames: {result_dataset.meta.total_frames}, "
        f"Video keys: {result_dataset.meta.video_keys}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {result_dataset.repo_id}")
        result_dataset.push_to_hub()
        logging.info("✓ Successfully pushed to hub!")


def handle_add_guide_stream(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, AddGuideStreamConfig):
        raise ValueError("Operation config must be AddGuideStreamConfig")

    if not cfg.operation.source_key:
        raise ValueError("operation.source_key must be specified for add_guide_stream operation")

    if not cfg.operation.new_key:
        raise ValueError("operation.new_key must be specified for add_guide_stream operation")

    if cfg.operation.append_to_repo_id is not None and cfg.new_repo_id is not None:
        raise ValueError(
            "Cannot specify both 'operation.append_to_repo_id' and 'new_repo_id'. "
            "Use 'operation.append_to_repo_id' to append to an existing dataset, "
            "or 'new_repo_id' to create a fresh one."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    if cfg.operation.append_to_repo_id is not None:
        append_root = Path(cfg.root) if cfg.root else HF_LEROBOT_HOME
        append_to_dir = append_root / cfg.operation.append_to_repo_id
        append_to_dataset = LeRobotDataset(cfg.operation.append_to_repo_id, root=append_to_dir)

        logging.info(
            f"Appending guide stream '{cfg.operation.new_key}' "
            f"(sourced from '{cfg.operation.source_key}') "
            f"from {cfg.repo_id} to {cfg.operation.append_to_repo_id}"
        )
        result_dataset = add_guide_stream(
            dataset=dataset,
            source_key=cfg.operation.source_key,
            new_key=cfg.operation.new_key,
            vcodec=cfg.operation.vcodec,
            pix_fmt=cfg.operation.pix_fmt,
            g=cfg.operation.g,
            crf=cfg.operation.crf,
            episode_indices=cfg.operation.episode_indices,
            append_to_dataset=append_to_dataset,
        )
        logging.info(f"Dataset saved to {append_to_dir}")
    else:
        output_repo_id, output_dir = get_output_path(
            cfg.repo_id,
            new_repo_id=cfg.new_repo_id,
            root=cfg.root,
            new_root=cfg.new_root,
        )

        if cfg.new_repo_id is None:
            dataset.root = Path(str(dataset.root) + "_old")

        logging.info(
            f"Adding guide stream '{cfg.operation.new_key}' "
            f"(sourced from '{cfg.operation.source_key}') to {cfg.repo_id}"
        )
        result_dataset = add_guide_stream(
            dataset=dataset,
            source_key=cfg.operation.source_key,
            new_key=cfg.operation.new_key,
            output_dir=output_dir,
            repo_id=output_repo_id,
            vcodec=cfg.operation.vcodec,
            pix_fmt=cfg.operation.pix_fmt,
            g=cfg.operation.g,
            crf=cfg.operation.crf,
            episode_indices=cfg.operation.episode_indices,
        )
        logging.info(f"Dataset saved to {output_dir}")

    logging.info(
        f"Episodes: {result_dataset.meta.total_episodes}, "
        f"Frames: {result_dataset.meta.total_frames}, "
        f"Video keys: {result_dataset.meta.video_keys}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {result_dataset.repo_id}")
        result_dataset.push_to_hub()
        logging.info("✓ Successfully pushed to hub!")


def handle_add_sam2_initial_segment(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, AddSam2InitialSegmentConfig):
        raise ValueError("Operation config must be AddSam2InitialSegmentConfig")

    if not cfg.operation.source_key:
        raise ValueError(
            "operation.source_key must be specified for "
            "add_sam2_initial_segment operation"
        )
    if not cfg.operation.new_key:
        raise ValueError(
            "operation.new_key must be specified for "
            "add_sam2_initial_segment operation"
        )

    if cfg.operation.append_to_repo_id is not None and cfg.new_repo_id is not None:
        raise ValueError(
            "Cannot specify both 'operation.append_to_repo_id' and 'new_repo_id'. "
            "Use 'operation.append_to_repo_id' to append to an existing dataset, "
            "or 'new_repo_id' to create a fresh one."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    if cfg.operation.append_to_repo_id is not None:
        append_root = Path(cfg.root) if cfg.root else HF_LEROBOT_HOME
        append_to_dir = append_root / cfg.operation.append_to_repo_id
        append_to_dataset = LeRobotDataset(cfg.operation.append_to_repo_id, root=append_to_dir)

        logging.info(
            f"Appending segmented stream '{cfg.operation.new_key}' "
            f"(sourced from '{cfg.operation.source_key}') "
            f"from {cfg.repo_id} to {cfg.operation.append_to_repo_id}"
        )
        result_dataset = add_sam2_initial_segment(
            dataset=dataset,
            source_key=cfg.operation.source_key,
            new_key=cfg.operation.new_key,
            vcodec=cfg.operation.vcodec,
            pix_fmt=cfg.operation.pix_fmt,
            g=cfg.operation.g,
            crf=cfg.operation.crf,
            fade_pixels=cfg.operation.fade_pixels,
            min_brightness=cfg.operation.min_brightness,
            episode_indices=cfg.operation.episode_indices,
            append_to_dataset=append_to_dataset,
        )
        logging.info(f"Dataset saved to {append_to_dir}")
    else:
        output_repo_id, output_dir = get_output_path(
            cfg.repo_id,
            new_repo_id=cfg.new_repo_id,
            root=cfg.root,
            new_root=cfg.new_root,
        )

        if cfg.new_repo_id is None:
            dataset.root = Path(str(dataset.root) + "_old")

        logging.info(
            f"Adding segmented stream '{cfg.operation.new_key}' "
            f"(sourced from '{cfg.operation.source_key}') to {cfg.repo_id}"
        )
        result_dataset = add_sam2_initial_segment(
            dataset=dataset,
            source_key=cfg.operation.source_key,
            new_key=cfg.operation.new_key,
            output_dir=output_dir,
            repo_id=output_repo_id,
            vcodec=cfg.operation.vcodec,
            pix_fmt=cfg.operation.pix_fmt,
            g=cfg.operation.g,
            crf=cfg.operation.crf,
            fade_pixels=cfg.operation.fade_pixels,
            min_brightness=cfg.operation.min_brightness,
            episode_indices=cfg.operation.episode_indices,
        )
        logging.info(f"Dataset saved to {output_dir}")

    logging.info(
        f"Episodes: {result_dataset.meta.total_episodes}, "
        f"Frames: {result_dataset.meta.total_frames}, "
        f"Video keys: {result_dataset.meta.video_keys}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {result_dataset.repo_id}")
        result_dataset.push_to_hub()
        logging.info("✓ Successfully pushed to hub!")


def handle_add_sam2_stream(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, AddSam2StreamConfig):
        raise ValueError("Operation config must be AddSam2StreamConfig")

    if not cfg.operation.source_key:
        raise ValueError(
            "operation.source_key must be specified for "
            "add_sam2_stream operation"
        )
    if not cfg.operation.new_key:
        raise ValueError(
            "operation.new_key must be specified for "
            "add_sam2_stream operation"
        )

    if cfg.operation.append_to_repo_id is not None and cfg.new_repo_id is not None:
        raise ValueError(
            "Cannot specify both 'operation.append_to_repo_id' and 'new_repo_id'. "
            "Use 'operation.append_to_repo_id' to append to an existing dataset, "
            "or 'new_repo_id' to create a fresh one."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    if cfg.operation.append_to_repo_id is not None:
        append_root = Path(cfg.root) if cfg.root else HF_LEROBOT_HOME
        append_to_dir = append_root / cfg.operation.append_to_repo_id
        append_to_dataset = LeRobotDataset(cfg.operation.append_to_repo_id, root=append_to_dir)

        logging.info(
            f"Appending SAM2-tracked stream '{cfg.operation.new_key}' "
            f"(sourced from '{cfg.operation.source_key}') "
            f"from {cfg.repo_id} to {cfg.operation.append_to_repo_id}"
        )
        result_dataset = add_sam2_stream(
            dataset=dataset,
            source_key=cfg.operation.source_key,
            new_key=cfg.operation.new_key,
            vcodec=cfg.operation.vcodec,
            pix_fmt=cfg.operation.pix_fmt,
            g=cfg.operation.g,
            crf=cfg.operation.crf,
            fade_pixels=cfg.operation.fade_pixels,
            min_brightness=cfg.operation.min_brightness,
            episode_indices=cfg.operation.episode_indices,
            append_to_dataset=append_to_dataset,
        )
        logging.info(f"Dataset saved to {append_to_dir}")
    else:
        output_repo_id, output_dir = get_output_path(
            cfg.repo_id,
            new_repo_id=cfg.new_repo_id,
            root=cfg.root,
            new_root=cfg.new_root,
        )

        if cfg.new_repo_id is None:
            dataset.root = Path(str(dataset.root) + "_old")

        logging.info(
            f"Adding SAM2-tracked stream '{cfg.operation.new_key}' "
            f"(sourced from '{cfg.operation.source_key}') to {cfg.repo_id}"
        )
        result_dataset = add_sam2_stream(
            dataset=dataset,
            source_key=cfg.operation.source_key,
            new_key=cfg.operation.new_key,
            output_dir=output_dir,
            repo_id=output_repo_id,
            vcodec=cfg.operation.vcodec,
            pix_fmt=cfg.operation.pix_fmt,
            g=cfg.operation.g,
            crf=cfg.operation.crf,
            fade_pixels=cfg.operation.fade_pixels,
            min_brightness=cfg.operation.min_brightness,
            episode_indices=cfg.operation.episode_indices,
        )
        logging.info(f"Dataset saved to {output_dir}")

    logging.info(
        f"Episodes: {result_dataset.meta.total_episodes}, "
        f"Frames: {result_dataset.meta.total_frames}, "
        f"Video keys: {result_dataset.meta.video_keys}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {result_dataset.repo_id}")
        result_dataset.push_to_hub()
        logging.info("✓ Successfully pushed to hub!")


def handle_add_sam3_stream(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, AddSam3StreamConfig):
        raise ValueError("Operation config must be AddSam3StreamConfig")

    if not cfg.operation.source_key:
        raise ValueError(
            "operation.source_key must be specified for "
            "add_sam3_stream operation"
        )
    if not cfg.operation.new_key:
        raise ValueError(
            "operation.new_key must be specified for "
            "add_sam3_stream operation"
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir = get_output_path(
        cfg.repo_id, cfg.new_repo_id, Path(cfg.root) if cfg.root else None
    )

    if cfg.new_repo_id is None:
        dataset.root = Path(str(dataset.root) + "_old")

    logging.info(
        f"Adding SAM3-tracked stream '{cfg.operation.new_key}' "
        f"(sourced from '{cfg.operation.source_key}') to {cfg.repo_id}"
    )
    new_dataset = add_sam3_stream(
        dataset=dataset,
        source_key=cfg.operation.source_key,
        new_key=cfg.operation.new_key,
        output_dir=output_dir,
        repo_id=output_repo_id,
        vcodec=cfg.operation.vcodec,
        pix_fmt=cfg.operation.pix_fmt,
        g=cfg.operation.g,
        crf=cfg.operation.crf,
        fade_pixels=cfg.operation.fade_pixels,
        min_brightness=cfg.operation.min_brightness,
    )

    logging.info(f"Dataset with SAM3 stream saved to {output_dir}")
    logging.info(f"Video keys: {new_dataset.meta.video_keys}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        new_dataset.push_to_hub()
        logging.info("Successfully pushed to hub!")


def handle_info(cfg: EditDatasetConfig):
    if not isinstance(cfg.operation, InfoConfig):
        raise ValueError("Operation config must be InfoConfig")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    sys.stdout.write(f"======Info {dataset.meta.repo_id}\n")
    sys.stdout.write(f"Repository ID: {dataset.meta.repo_id} \n")
    sys.stdout.write(f"Total episode: {dataset.meta.total_episodes} \n")
    sys.stdout.write(f"Total task: {dataset.meta.total_tasks} \n")
    sys.stdout.write(f"Total frame(Actual Count): {dataset.meta.total_frames}({len(dataset)}) \n")
    sys.stdout.write(
        f"Average frame per episode: {dataset.meta.total_frames / dataset.meta.total_episodes:.1f}\n"
    )
    sys.stdout.write(
        f"Average episode time(sec): {(dataset.meta.total_frames / dataset.meta.total_episodes) / dataset.meta.fps:.1f}\n"
    )
    sys.stdout.write(f"FPS: {dataset.meta.fps}\n")

    total_file_size = _get_dataset_size(dataset.root)
    sys.stdout.write(f"Size: {total_file_size / (1024 * 1024):.1f} MB\n")
    if cfg.operation.show_features:
        import json

        feature_dump_str = json.dumps(
            dataset.meta.features, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": ")
        )
        sys.stdout.write("Features:\n")
        sys.stdout.write(f"{feature_dump_str}\n")


def _validate_config(cfg: EditDatasetConfig) -> None:
    if isinstance(cfg.operation, MergeConfig):
        if not cfg.new_repo_id:
            raise ValueError("--new_repo_id is required for merge operation (the merged dataset identifier)")
    else:
        if not cfg.repo_id:
            raise ValueError(
                f"--repo_id is required for {cfg.operation.type} operation (the input dataset identifier)"
            )


@parser.wrap()
def edit_dataset(cfg: EditDatasetConfig) -> None:
    _validate_config(cfg)
    operation_type = cfg.operation.type

    if operation_type == "delete_episodes":
        handle_delete_episodes(cfg)
    elif operation_type == "resize_videos":
        handle_resize_videos(cfg)
    elif operation_type == "copy_episodes":
        handle_copy_episodes(cfg)
    elif operation_type == "trim_episodes":
        handle_trim_episodes(cfg)
    elif operation_type == "split_episodes":
        handle_split_episodes(cfg)
    elif operation_type == "split":
        handle_split(cfg)
    elif operation_type == "merge":
        handle_merge(cfg)
    elif operation_type == "remove_feature":
        handle_remove_feature(cfg)
    elif operation_type == "rename_features":
        handle_rename_features(cfg)
    elif operation_type == "resize_videos":
        handle_resize_videos(cfg)
    elif operation_type == "modify_tasks":
        handle_modify_tasks(cfg)
    elif operation_type == "convert_image_to_video":
        handle_convert_image_to_video(cfg)
    elif operation_type == "add_black_stream":
        handle_add_black_stream(cfg)
    elif operation_type == "add_guide_stream":
        handle_add_guide_stream(cfg)
    elif operation_type == "add_sam2_initial_segment":
        handle_add_sam2_initial_segment(cfg)
    elif operation_type == "add_sam2_stream":
        handle_add_sam2_stream(cfg)
    elif operation_type == "add_sam3_stream":
        handle_add_sam3_stream(cfg)
    elif operation_type == "recompute_stats":
        handle_recompute_stats(cfg)
    elif operation_type == "repair_inconsistencies":
        handle_repair_inconsistencies(cfg)
    elif operation_type == "info":
        handle_info(cfg)
    else:
        available = ", ".join(OperationConfig.get_known_choices())
        raise ValueError(f"Unknown operation: {operation_type}\nAvailable operations: {available}")


def main() -> None:
    init_logging()
    edit_dataset()


if __name__ == "__main__":
    main()
