from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Global cache directory (consistent with nanochat, data downloaded only once)
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "vit_data")
DATASET_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
TIME_BUDGET = 600  # Training time budget (seconds), 10 minutes

TINY_IMAGENET_URL = "https://image-net.org/data/tiny-imagenet-200.zip"

# Mirror URL list (in priority order)
MIRROR_URLS = [
    "https://image-net.org/data/tiny-imagenet-200.zip",
    "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
]


def download(url: str, destination: Path) -> None:
    """Download file using wget with resume, auto-proxy, and retry support"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        # Simple validation: file size should be > 200MB (full zip is ~237MB)
        if destination.stat().st_size > 200 * 1024 * 1024:
            print(f"[prepare] archive already exists: {destination}")
            return
        else:
            print(f"[prepare] incomplete archive detected, re-downloading...")
            destination.unlink()

    # Build URL attempt list
    urls_to_try = [url]
    for mirror in MIRROR_URLS:
        if mirror != url:
            urls_to_try.append(mirror)

    last_error = None
    for try_url in urls_to_try:
        print(f"[prepare] downloading from {try_url}")
        try:
            # wget parameters:
            #   -c : resume download
            #   --timeout=60 : network timeout 60s (read/connect/DNS each 60s)
            #   --tries=3 : retry 3 times
            #   --waitretry=5 : retry interval 5 seconds
            #   --show-progress : show progress bar
            #   -O : output file path
            result = subprocess.run(
                [
                    "wget", "-c",
                    "--timeout=60",
                    "--tries=3",
                    "--waitretry=5",
                    "--show-progress",
                    "-O", str(destination),
                    try_url,
                ],
                check=True,
            )
            # Verify download integrity
            if destination.exists() and destination.stat().st_size > 200 * 1024 * 1024:
                print(f"[prepare] downloaded to {destination}")
                return
            else:
                raise RuntimeError("Downloaded file is too small, possibly incomplete")
        except Exception as e:
            last_error = e
            print(f"[prepare] failed from {try_url}: {e}")
            # Clean up incomplete file
            if destination.exists() and destination.stat().st_size < 200 * 1024 * 1024:
                destination.unlink(missing_ok=True)
            continue

    raise RuntimeError(
        f"[prepare] all download sources failed. Last error: {last_error}\n"
        f"You can manually download and place the file at:\n"
        f"  {destination}\n"
        f"Try one of these URLs in your browser or with a download tool:\n"
        + "\n".join(f"  - {u}" for u in urls_to_try)
    )


def extract(archive_path: Path, output_dir: Path) -> Path:
    dataset_dir = output_dir / "tiny-imagenet-200"
    if dataset_dir.exists():
        print(f"[prepare] extracted dataset already exists: {dataset_dir}")
        return dataset_dir
    print(f"[prepare] extracting {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"[prepare] extracted to {dataset_dir}")
    return dataset_dir


def build_val_split(dataset_dir: Path) -> Path:
    annotations_file = dataset_dir / "val" / "val_annotations.txt"
    images_dir = dataset_dir / "val" / "images"
    output_dir = dataset_dir / "val_split"

    if output_dir.exists():
        print(f"[prepare] validation split already exists: {output_dir}")
        return output_dir

    print(f"[prepare] reorganizing validation images into class folders")
    output_dir.mkdir(parents=True, exist_ok=True)

    with annotations_file.open("r", encoding="utf-8") as f:
        for line in f:
            image_name, wnid, *_ = line.strip().split("\t")
            class_dir = output_dir / wnid
            class_dir.mkdir(parents=True, exist_ok=True)
            src = images_dir / image_name
            dst = class_dir / image_name
            shutil.copy2(src, dst)

    print(f"[prepare] validation split written to {output_dir}")
    return output_dir


def write_metadata(dataset_dir: Path) -> None:
    wnids = (dataset_dir / "wnids.txt").read_text(encoding="utf-8").splitlines()
    words = {}
    with (dataset_dir / "words.txt").open("r", encoding="utf-8") as f:
        for line in f:
            wnid, names = line.strip().split("\t", 1)
            words[wnid] = names

    metadata = {
        "num_classes": len(wnids),
        "classes": wnids,
        "class_to_idx": {wnid: idx for idx, wnid in enumerate(wnids)},
        "class_descriptions": words,
        "train_dir": str((dataset_dir / "train").resolve()),
        "val_dir": str((dataset_dir / "val_split").resolve()),
    }
    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[prepare] metadata written to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Tiny ImageNet")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument(
        "--url",
        type=str,
        default=TINY_IMAGENET_URL,
        help="Tiny ImageNet archive URL",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    archive_path = data_dir / "tiny-imagenet-200.zip"

    download(args.url, archive_path)
    dataset_dir = extract(archive_path, data_dir)
    build_val_split(dataset_dir)
    write_metadata(dataset_dir)

    print("[prepare] done")
    print(f"[prepare] dataset dir: {dataset_dir}")
    print(f"[prepare] train dir: {dataset_dir / 'train'}")
    print(f"[prepare] val dir:   {dataset_dir / 'val_split'}")


if __name__ == "__main__":
    main()
