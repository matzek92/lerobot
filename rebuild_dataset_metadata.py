#!/usr/bin/env python3
"""
Rebuild dataset metadata by scanning actual files on disk.

Usage:
    python rebuild_dataset_metadata.py /path/to/dataset
"""

import logging
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rebuild_episodes_metadata(root: Path) -> None:
    """Rebuild episodes metadata by scanning actual parquet files."""
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        logger.error(f"No metadata directory found at {meta_dir}")
        return

    # Find all chunk directories
    chunks = {}
    for chunk_dir in sorted(meta_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        chunk_idx = int(chunk_dir.name.split("-")[1])
        chunks[chunk_idx] = []

        # Find all parquet files in this chunk
        for pq_file in sorted(chunk_dir.glob("file-*.parquet")):
            file_idx = int(pq_file.name.split("-")[1].split(".")[0])
            chunks[chunk_idx].append((file_idx, pq_file))

    if not chunks:
        logger.warning("No episode metadata files found")
        return

    # Collect all dataframes and rebuild indices
    all_chunks = []
    real_chunk_indices = []
    real_file_indices = []

    for chunk_idx in sorted(chunks.keys()):
        for file_idx, pq_file in chunks[chunk_idx]:
            try:
                df = pd.read_parquet(pq_file)
                
                # Update chunk and file indices in the dataframe
                if "meta/episodes/chunk_index" not in df.columns:
                    df["meta/episodes/chunk_index"] = chunk_idx
                else:
                    df["meta/episodes/chunk_index"] = chunk_idx
                    
                if "meta/episodes/file_index" not in df.columns:
                    df["meta/episodes/file_index"] = file_idx
                else:
                    df["meta/episodes/file_index"] = file_idx

                all_chunks.append(df)
                real_chunk_indices.extend([chunk_idx] * len(df))
                real_file_indices.extend([file_idx] * len(df))
                
                logger.info(f"✓ Found {len(df)} episodes in chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet")
            except Exception as e:
                logger.error(f"✗ Error reading {pq_file}: {e}")
                continue

    if not all_chunks:
        logger.error("No valid episode metadata files found")
        return

    # Combine all dataframes
    combined_df = pd.concat(all_chunks, ignore_index=True)

    logger.info(f"\nTotal episodes found: {len(combined_df)}")
    logger.info(f"Unique chunks: {len(chunks)}")
    logger.info(f"Total chunk/file pairs: {len(all_chunks)}")

    # Show the reconstructed index
    logger.info("\nReconstructed indices:")
    logger.info("  chunk_index: " + str(combined_df["meta/episodes/chunk_index"].unique()))
    logger.info("  file_index: " + str(combined_df["meta/episodes/file_index"].unique()))


def rebuild_data_metadata(root: Path) -> None:
    """Rebuild data references by scanning actual data files."""
    data_dir = root / "data"
    if not data_dir.exists():
        logger.warning(f"No data directory found at {data_dir}")
        return

    chunks = {}
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        chunk_idx = int(chunk_dir.name.split("-")[1])
        chunks[chunk_idx] = []

        for pq_file in sorted(chunk_dir.glob("file-*.parquet")):
            file_idx = int(pq_file.name.split("-")[1].split(".")[0])
            size_mb = pq_file.stat().st_size / (1024 * 1024)
            chunks[chunk_idx].append((file_idx, size_mb))
            logger.info(f"✓ Found data chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet ({size_mb:.2f} MB)")

    if chunks:
        logger.info(f"Total data chunks: {len(chunks)}")


def rebuild_videos_metadata(root: Path) -> None:
    """Rebuild video references by scanning actual video files."""
    videos_dir = root / "videos"
    if not videos_dir.exists():
        logger.warning(f"No videos directory found at {videos_dir}")
        return

    video_keys = set()
    for video_key_dir in videos_dir.glob("*/"):
        if not video_key_dir.is_dir():
            continue
        video_key = video_key_dir.name
        video_keys.add(video_key)

        chunks = {}
        for chunk_dir in sorted(video_key_dir.glob("chunk-*")):
            if not chunk_dir.is_dir():
                continue
            chunk_idx = int(chunk_dir.name.split("-")[1])
            chunks[chunk_idx] = []

            for video_file in sorted(chunk_dir.glob("file-*.mp4")) + sorted(chunk_dir.glob("file-*.webp")):
                file_idx = int(video_file.name.split("-")[1].split(".")[0])
                size_mb = video_file.stat().st_size / (1024 * 1024)
                chunks[chunk_idx].append((file_idx, size_mb))
                logger.info(
                    f"✓ Found video '{video_key}' chunk-{chunk_idx:03d}/file-{file_idx:03d} ({size_mb:.2f} MB)"
                )

    if video_keys:
        logger.info(f"Total video streams: {len(video_keys)}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild dataset metadata by scanning actual files on disk"
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the dataset root directory")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        logger.error(f"Dataset root not found: {root}")
        return

    logger.info(f"Scanning dataset at: {root}\n")

    logger.info("=" * 60)
    logger.info("EPISODES METADATA")
    logger.info("=" * 60)
    rebuild_episodes_metadata(root)

    logger.info("\n" + "=" * 60)
    logger.info("DATA FILES")
    logger.info("=" * 60)
    rebuild_data_metadata(root)

    logger.info("\n" + "=" * 60)
    logger.info("VIDEO FILES")
    logger.info("=" * 60)
    rebuild_videos_metadata(root)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("✓ Scan complete. Check results above for any missing files.")
    logger.info("✓ If any files were skipped, they may be corrupted or incomplete.")


if __name__ == "__main__":
    main()
