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
Manage LeRobot datasets: push to and pull from the Hugging Face Hub.

This script provides a CLI for downloading (pulling) datasets from
the Hugging Face Hub and uploading (pushing) local datasets to it.

Usage Examples:

Download a dataset from the Hub:
    lerobot-dataset-hub \\
        --operation.type pull \\
        --operation.repo_id lerobot/pusht

Download a dataset to a specific local directory:
    lerobot-dataset-hub \\
        --operation.type pull \\
        --operation.repo_id lerobot/pusht \\
        --operation.local_dir ./my_datasets/pusht

Download only specific episodes:
    lerobot-dataset-hub \\
        --operation.type pull \\
        --operation.repo_id lerobot/pusht \\
        --operation.episodes "[0, 1, 2]"

Download a dataset without videos:
    lerobot-dataset-hub \\
        --operation.type pull \\
        --operation.repo_id lerobot/pusht \\
        --operation.download_videos false

Download a specific revision/version:
    lerobot-dataset-hub \\
        --operation.type pull \\
        --operation.repo_id lerobot/pusht \\
        --operation.revision main

Force re-download even if cached:
    lerobot-dataset-hub \\
        --operation.type pull \\
        --operation.repo_id lerobot/pusht \\
        --operation.force_cache_sync true

Push a local dataset to the Hub:
    lerobot-dataset-hub \\
        --operation.type push \\
        --operation.repo_id my_user/my_dataset

Push a local dataset from a specific directory:
    lerobot-dataset-hub \\
        --operation.type push \\
        --operation.repo_id my_user/my_dataset \\
        --operation.local_dir ./path/to/local/dataset

Push a dataset as private:
    lerobot-dataset-hub \\
        --operation.type push \\
        --operation.repo_id my_user/my_dataset \\
        --operation.private true

Push a large dataset (many files):
    lerobot-dataset-hub \\
        --operation.type push \\
        --operation.repo_id my_user/my_dataset \\
        --operation.upload_large_folder true

Push without videos:
    lerobot-dataset-hub \\
        --operation.type push \\
        --operation.repo_id my_user/my_dataset \\
        --operation.push_videos false

Push to a specific branch:
    lerobot-dataset-hub \\
        --operation.type push \\
        --operation.repo_id my_user/my_dataset \\
        --operation.branch dev

Show dataset info (local or remote):
    lerobot-dataset-hub \\
        --operation.type info \\
        --operation.repo_id lerobot/pusht

Show dataset info with feature details:
    lerobot-dataset-hub \\
        --operation.type info \\
        --operation.repo_id lerobot/pusht \\
        --operation.show_features true

Using JSON config file:
    lerobot-dataset-hub \\
        --config_path path/to/hub_config.json
"""

import abc
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import draccus

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


@dataclass
class HubOperationConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@HubOperationConfig.register_subclass("pull")
@dataclass
class PullConfig(HubOperationConfig):
    """Download a dataset from the Hugging Face Hub."""

    repo_id: str = ""
    local_dir: str | None = None
    episodes: list[int] | None = None
    download_videos: bool = True
    revision: str | None = None
    force_cache_sync: bool = False


@HubOperationConfig.register_subclass("push")
@dataclass
class PushConfig(HubOperationConfig):
    """Push a local dataset to the Hugging Face Hub."""

    repo_id: str = ""
    local_dir: str | None = None
    branch: str | None = None
    tags: list[str] = field(default_factory=list)
    license: str = "apache-2.0"
    tag_version: bool = True
    push_videos: bool = True
    private: bool = False
    upload_large_folder: bool = False


@HubOperationConfig.register_subclass("info")
@dataclass
class InfoConfig(HubOperationConfig):
    """Show dataset information."""

    repo_id: str = ""
    local_dir: str | None = None
    show_features: bool = False


@dataclass
class DatasetHubConfig:
    operation: HubOperationConfig


def _get_dataset_size(repo_path: Path) -> int:
    """Calculate total size of a dataset directory in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(repo_path):
        for f in filenames:
            fp = Path(dirpath) / f
            total += fp.stat().st_size
    return total


def handle_pull(cfg: DatasetHubConfig) -> None:
    if not isinstance(cfg.operation, PullConfig):
        raise ValueError("Operation config must be PullConfig")

    op = cfg.operation
    if not op.repo_id:
        raise ValueError("repo_id must be specified for pull operation")

    root = Path(op.local_dir) if op.local_dir else None

    logging.info(f"Downloading dataset '{op.repo_id}' from Hugging Face Hub")
    if root:
        logging.info(f"  Local directory: {root}")
    if op.episodes:
        logging.info(f"  Episodes: {op.episodes}")
    if not op.download_videos:
        logging.info("  Skipping video files")
    if op.revision:
        logging.info(f"  Revision: {op.revision}")

    dataset = LeRobotDataset(
        repo_id=op.repo_id,
        root=root,
        episodes=op.episodes,
        revision=op.revision,
        force_cache_sync=op.force_cache_sync,
        download_videos=op.download_videos,
    )

    logging.info(f"Dataset downloaded to: {dataset.root}")
    logging.info(f"  Repository ID: {dataset.repo_id}")
    logging.info(f"  Episodes: {dataset.meta.total_episodes}")
    logging.info(f"  Frames: {dataset.meta.total_frames}")
    logging.info(f"  FPS: {dataset.meta.fps}")
    if dataset.meta.video_keys:
        logging.info(f"  Video keys: {dataset.meta.video_keys}")


def handle_push(cfg: DatasetHubConfig) -> None:
    if not isinstance(cfg.operation, PushConfig):
        raise ValueError("Operation config must be PushConfig")

    op = cfg.operation
    if not op.repo_id:
        raise ValueError("repo_id must be specified for push operation")

    root = Path(op.local_dir) if op.local_dir else None

    logging.info(f"Loading dataset '{op.repo_id}' for upload")
    dataset = LeRobotDataset(repo_id=op.repo_id, root=root)

    logging.info(f"Pushing dataset to Hugging Face Hub as '{op.repo_id}'")
    if op.private:
        logging.info("  Visibility: private")
    if op.branch:
        logging.info(f"  Branch: {op.branch}")
    if not op.push_videos:
        logging.info("  Skipping video files")
    if op.upload_large_folder:
        logging.info("  Using large folder upload mode")

    dataset.push_to_hub(
        branch=op.branch,
        tags=op.tags if op.tags else None,
        license=op.license,
        tag_version=op.tag_version,
        push_videos=op.push_videos,
        private=op.private,
        upload_large_folder=op.upload_large_folder,
    )

    logging.info(f"Dataset '{op.repo_id}' pushed successfully!")
    logging.info(f"  URL: https://huggingface.co/datasets/{op.repo_id}")


def handle_info(cfg: DatasetHubConfig) -> None:
    if not isinstance(cfg.operation, InfoConfig):
        raise ValueError("Operation config must be InfoConfig")

    op = cfg.operation
    if not op.repo_id:
        raise ValueError("repo_id must be specified for info operation")

    root = Path(op.local_dir) if op.local_dir else None
    dataset = LeRobotDataset(repo_id=op.repo_id, root=root)

    sys.stdout.write(f"====== Dataset: {dataset.meta.repo_id} ======\n")
    sys.stdout.write(f"Repository ID: {dataset.meta.repo_id}\n")
    sys.stdout.write(f"Local path: {dataset.root}\n")
    sys.stdout.write(f"Total episodes: {dataset.meta.total_episodes}\n")
    sys.stdout.write(f"Total tasks: {dataset.meta.total_tasks}\n")
    sys.stdout.write(f"Total frames: {dataset.meta.total_frames}\n")
    sys.stdout.write(f"FPS: {dataset.meta.fps}\n")

    avg_frames = dataset.meta.total_frames / max(dataset.meta.total_episodes, 1)
    sys.stdout.write(f"Avg frames/episode: {avg_frames:.1f}\n")
    sys.stdout.write(f"Avg episode duration: {avg_frames / dataset.meta.fps:.1f}s\n")

    if dataset.meta.video_keys:
        sys.stdout.write(f"Video keys: {dataset.meta.video_keys}\n")
    if dataset.meta.image_keys:
        sys.stdout.write(f"Image keys: {dataset.meta.image_keys}\n")

    if dataset.root.exists():
        total_size = _get_dataset_size(dataset.root)
        if total_size >= 1024 * 1024 * 1024:
            sys.stdout.write(f"Local size: {total_size / (1024 ** 3):.2f} GB\n")
        else:
            sys.stdout.write(f"Local size: {total_size / (1024 ** 2):.1f} MB\n")

    if op.show_features:
        feature_str = json.dumps(
            dataset.meta.features,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
        sys.stdout.write(f"Features:\n{feature_str}\n")


@parser.wrap()
def dataset_hub(cfg: DatasetHubConfig) -> None:
    operation_type = cfg.operation.type

    if operation_type == "pull":
        handle_pull(cfg)
    elif operation_type == "push":
        handle_push(cfg)
    elif operation_type == "info":
        handle_info(cfg)
    else:
        available = ", ".join(HubOperationConfig.get_known_choices())
        raise ValueError(f"Unknown operation: {operation_type}\nAvailable operations: {available}")


def main() -> None:
    init_logging()
    dataset_hub()


if __name__ == "__main__":
    main()
