import argparse
import json
import logging
from pathlib import Path


LOGGER = logging.getLogger("create_splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-split JSON manifest files from split text files, point clouds, "
            "and persistence diagrams."
        )
    )
    parser.add_argument(
        "--diagram_dir",
        required=True,
        type=Path,
        help="Directory containing persistence diagrams saved as .npz files.",
    )
    parser.add_argument(
        "--pcd_dir",
        required=True,
        type=Path,
        help="Directory containing point clouds saved as .npy files.",
    )
    parser.add_argument(
        "--split_dir",
        required=True,
        type=Path,
        help="Directory containing split .txt files with one sample name per line.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where the generated split JSON files will be saved.",
    )
    parser.add_argument(
        "--pcd_suffix",
        default=".npy",
        type=str,
        help="File suffix for point clouds. Default: .npy",
    )
    parser.add_argument(
        "--diagram_suffix",
        default=".npz",
        type=str,
        help="File suffix for persistence diagrams. Default: .npz",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files in the output directory.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_suffix(value: str) -> str:
    return value if value.startswith(".") else f".{value}"


def read_split_file(split_path: Path) -> list[str]:
    with split_path.open("r", encoding="utf-8") as handle:
        sample_ids = [line.strip() for line in handle if line.strip()]

    if not sample_ids:
        raise ValueError(f"Split file is empty: {split_path}")

    return sample_ids


def build_record(
    sample_id: str,
    pcd_dir: Path,
    diagram_dir: Path,
    pcd_suffix: str,
    diagram_suffix: str,
) -> dict[str, str]:
    pcd_path = pcd_dir / f"{sample_id}{pcd_suffix}"
    diagram_path = diagram_dir / f"{sample_id}{diagram_suffix}"

    if not pcd_path.exists():
        raise FileNotFoundError(
            f"Point cloud file not found for '{sample_id}': {pcd_path}"
        )
    if not diagram_path.exists():
        raise FileNotFoundError(
            f"Persistence diagram file not found for '{sample_id}': {diagram_path}"
        )

    return {
        "id": sample_id,
        "pcd_path": str(pcd_path),
        "diagram_path": str(diagram_path),
    }


def write_split_json(
    records: list[dict[str, str]], output_path: Path, overwrite: bool
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    configure_logging()

    diagram_dir = args.diagram_dir.expanduser().resolve()
    pcd_dir = args.pcd_dir.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pcd_suffix = ensure_suffix(args.pcd_suffix)
    diagram_suffix = ensure_suffix(args.diagram_suffix)

    for path in (diagram_dir, pcd_dir, split_dir):
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Expected a directory: {path}")

    split_paths = sorted(split_dir.glob("*.txt"))
    if not split_paths:
        raise FileNotFoundError(f"No split .txt files found in: {split_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_path in split_paths:
        split_name = split_path.stem
        sample_ids = read_split_file(split_path)
        records = [
            build_record(
                sample_id=sample_id,
                pcd_dir=pcd_dir,
                diagram_dir=diagram_dir,
                pcd_suffix=pcd_suffix,
                diagram_suffix=diagram_suffix,
            )
            for sample_id in sample_ids
        ]

        output_path = output_dir / f"{split_name}.json"
        write_split_json(records, output_path, overwrite=args.overwrite)
        LOGGER.info("Wrote %s records to %s", len(records), output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
