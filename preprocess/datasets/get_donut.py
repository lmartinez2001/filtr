import argparse
import logging
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise ImportError(
        "huggingface_hub is required to download DONUT. "
        "Install it with `pip install huggingface_hub`."
    ) from exc


LOGGER = logging.getLogger("get_donut")
REPO_ID = "LouisM2001/donut"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the DONUT dataset from Hugging Face and flatten shard directories.",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Destination directory where the dataset should be materialized.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset revision to download (branch, tag, or commit hash). Default: main.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download even if files are already cached by huggingface_hub.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files in the destination directory.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_destination(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "obj").mkdir(exist_ok=True)
    (path / "pcd").mkdir(exist_ok=True)


def copy_root_metadata(staging_dir: Path, destination: Path, overwrite: bool) -> None:
    LOGGER.info("Copying repository metadata files")
    for item in staging_dir.iterdir():
        if item.name in {"obj", "pcd", ".cache"}:
            continue

        target = destination / item.name
        if item.is_dir():
            if target.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Refusing to overwrite existing directory: {target}. "
                        "Use --overwrite to replace it."
                    )
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {target}. "
                    "Use --overwrite to replace it."
                )
            shutil.copy2(item, target)


def move_files(
    flat_source_dir: Path, pattern: str, flat_target_dir: Path, overwrite: bool
) -> int:
    moved = 0
    for source_path in sorted(flat_source_dir.rglob(pattern)):
        target_path = flat_target_dir / source_path.name
        if target_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {target_path}. "
                    "Use --overwrite to replace it."
                )
            target_path.unlink()

        shutil.move(str(source_path), str(target_path))
        moved += 1
    return moved


def remove_empty_shard_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def flatten_shards(staging_dir: Path, destination: Path, overwrite: bool) -> None:
    LOGGER.info("Flattening shard directories into %s", destination)

    obj_source_dir = staging_dir / "obj"
    pcd_source_dir = staging_dir / "pcd"
    obj_target_dir = destination / "obj"
    pcd_target_dir = destination / "pcd"

    obj_count = move_files(obj_source_dir, "*.npz", obj_target_dir, overwrite=overwrite)
    LOGGER.info("Moved %s object files into %s", obj_count, obj_target_dir)

    pcd_count = move_files(pcd_source_dir, "*.npy", pcd_target_dir, overwrite=overwrite)
    LOGGER.info("Moved %s point-cloud files into %s", pcd_count, pcd_target_dir)

    remove_empty_shard_dirs(obj_source_dir)
    remove_empty_shard_dirs(pcd_source_dir)


def main() -> int:
    args = parse_args()
    configure_logging()

    destination = args.destination.expanduser().resolve()
    ensure_destination(destination)

    LOGGER.info("Preparing to download dataset %s into %s", REPO_ID, destination)
    LOGGER.info("Using Hugging Face Hub snapshot_download with repo_type='dataset'")

    with TemporaryDirectory(
        prefix="donut_download_", dir=destination.parent
    ) as tmp_dir:
        staging_dir = Path(tmp_dir) / "snapshot"
        LOGGER.info(
            "Downloading dataset snapshot to temporary staging directory %s",
            staging_dir,
        )
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=args.revision,
            local_dir=staging_dir,
            allow_patterns=["obj/**", "pcd/**", "*.csv", "*.md", "*.txt", "*.json"],
            force_download=args.force_download,
            max_workers=1,
        )

        copy_root_metadata(staging_dir, destination, overwrite=args.overwrite)
        flatten_shards(staging_dir, destination, overwrite=args.overwrite)

    LOGGER.info("DONUT dataset download and unsharding complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.error("Failed to download DONUT: %s", exc)
        raise
