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

"""Analyze dataset episodes for motor dead-times and camera freeze candidates."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.episode_analysis import analyze_episode_motion, plot_episode_motion
from lerobot.utils.utils import init_logging


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


def analyze_dataset_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    save: bool = False,
    output_dir: Path | None = None,
    verbose: bool = False,
    strip_motion: bool = False,
    min_idle_frames: int = 5,
    frame_padding: int = 0,
    motor_score_aggregation: str = "max",
    gripper_motor_index: int | None = 5,
    smoothing_window: int = 1,
    plot_output_html: Path | None = None,
    show_plot: bool = True,
    show_episode_progress: bool = True,
    **kwargs,
) -> dict:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write analysis files with `--output-dir path/to/directory`."
        )

    result = analyze_episode_motion(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        filter_initial_motion=strip_motion,
        filter_final_motion=strip_motion,
        min_idle_frames=min_idle_frames,
        frame_padding=frame_padding,
        motor_score_aggregation=motor_score_aggregation,
        gripper_motor_index=gripper_motor_index,
        smoothing_window=smoothing_window,
        show_progress=show_episode_progress,
    )
    result["repo_id"] = dataset.repo_id
    result["episode_index"] = int(episode_index)
    result_out = {k: v for k, v in result.items() if not k.startswith("_")}

    activity_window = result["activity_window"]
    logging.info(f"Motor movement starts at frame index: {activity_window['start_frame_index']}")
    logging.info(f"Motor movement ends at frame index: {activity_window['end_frame_index']}")

    for camera_key, frozen_frames in result["camera_freeze_candidates"].items():
        logging.info(f"Camera '{camera_key}' freeze candidates: {len(frozen_frames)} frames")

    gripper_transitions = result["gripper_transitions"]
    if gripper_transitions["available"]:
        logging.info(
            "Gripper motor %s transitions: %s total (%s open->closed, %s closed->open)",
            gripper_transitions["motor_index"],
            len(gripper_transitions["change_frames"]),
            len(gripper_transitions["open_to_closed_frames"]),
            len(gripper_transitions["closed_to_open_frames"]),
        )
    else:
        logging.info(
            "Gripper transition detection unavailable for motor %s (%s)",
            gripper_transitions["motor_index"],
            gripper_transitions["reason"],
        )

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = dataset.repo_id.replace("/", "_")
        output_path = output_dir / f"{repo_id_str}_episode_{episode_index:05d}_analysis.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result_out, f, indent=2)
        logging.info(f"Saved analysis report to: {output_path}")

    if verbose:
        plot_episode_motion(result, output_html_path=plot_output_html, show=show_plot)

    print(json.dumps(result_out, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    episode_group = parser.add_mutually_exclusive_group(required=True)
    episode_group.add_argument(
        "--episode-index",
        type=int,
        help="Analyze a single episode index.",
    )
    episode_group.add_argument(
        "--all-episodes",
        action="store_true",
        help="Analyze all episodes of the dataset.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to store analysis outputs (.json and optional .html plots).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help="Save an analysis .json file in the directory provided by `--output-dir`.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-run analysis even if output files already exist in --output-dir.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display interactive Plotly charts with motor state, derivative, and camera scores.",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Strip residual motion on both sides of the episode and prefer the main movement block.",
    )
    parser.add_argument(
        "--min-idle-frames",
        type=int,
        default=5,
        help="Minimum number of consecutive low-motion frames required to split motion into separate blocks "
             "when using --strip (default: 5).",
    )
    parser.add_argument(
        "--frame-padding",
        type=int,
        default=0,
        help="Number of frames to add before and after the detected activity window (default: 0).",
    )
    parser.add_argument(
        "--motor-score-aggregation",
        type=str,
        choices=["max", "sum"],
        default="max",
        help="How to aggregate motor transition scores across motors: 'max' (default) uses the maximum "
             "transition per frame, 'sum' uses the sum of all motor transitions per frame.",
    )
    parser.add_argument(
        "--gripper-motor-index",
        type=int,
        default=5,
        help=(
            "Motor index used for gripper open/closed transition detection. "
            "Set to a negative value to disable transition detection (default: 5)."
        ),
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Window size for moving-average smoothing of camera motion scores. "
             "Use odd values such as 3, 5, or 7 (default: 1 = no smoothing).",
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset FPS value. "
            "This argument is passed to the LeRobotDataset constructor as tolerance_s. "
            "Defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    repo_id = args.repo_id
    root = args.root
    tolerance_s = args.tolerance_s

    init_logging()
    # Keep analyze output focused: suppress decoder fallback chatter.
    logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)
    logging.getLogger("lerobot.datasets.video_utils").setLevel(logging.ERROR)

    if args.all_episodes and args.output_dir is None:
        parser.error("--all-episodes requires --output-dir to store analysis results.")

    if args.episode_index is not None:
        episode_indices = [args.episode_index]
    else:
        logging.info("Loading dataset metadata to determine total episodes")
        dataset_meta = LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)
        episode_indices = list(range(dataset_meta.meta.total_episodes))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_id_str = repo_id.replace("/", "_")

    for idx, episode_index in enumerate(episode_indices, start=1):
        if args.output_dir is not None and not args.override:
            output_json_path = args.output_dir / f"{repo_id_str}_episode_{episode_index:05d}_analysis.json"
            if output_json_path.exists():
                logging.info(f"Skipping episode {episode_index}: analysis file already exists at {output_json_path}")
                continue

        logging.info(f"Analyzing episode {episode_index} ({idx}/{len(episode_indices)})")

        dataset = LeRobotDataset(repo_id, episodes=[episode_index], root=root, tolerance_s=tolerance_s)

        plot_output_html = None
        show_plot = args.verbose and len(episode_indices) == 1 and args.output_dir is None
        if args.verbose and args.output_dir is not None:
            plot_output_html = args.output_dir / f"{repo_id_str}_episode_{episode_index:05d}_analysis.html"

        analyze_dataset_episode(
            dataset=dataset,
            episode_index=episode_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save=(args.output_dir is not None) or bool(args.save),
            output_dir=args.output_dir,
            verbose=args.verbose,
            strip_motion=args.strip,
            min_idle_frames=args.min_idle_frames,
            frame_padding=args.frame_padding,
            motor_score_aggregation=args.motor_score_aggregation,
            gripper_motor_index=None if args.gripper_motor_index < 0 else args.gripper_motor_index,
            smoothing_window=args.smoothing_window,
            plot_output_html=plot_output_html,
            show_plot=show_plot,
            show_episode_progress=True,
        )


if __name__ == "__main__":
    main()
