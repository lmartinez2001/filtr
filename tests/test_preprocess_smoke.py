import json
import subprocess
import sys
from pathlib import Path


def test_create_splits_cli_generates_json_manifests(toy_dataset_root):
    output_dir = toy_dataset_root / "generated_manifests"
    script_path = Path("preprocess/datasets/create_splits.py")

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--diagram_dir",
            str(toy_dataset_root / "diagrams"),
            "--pcd_dir",
            str(toy_dataset_root / "pcd"),
            "--tokens_dir",
            str(toy_dataset_root / "tokens"),
            "--split_dir",
            str(toy_dataset_root / "splits"),
            "--output_dir",
            str(output_dir),
            "--output_suffix",
            "_out2048_in1024_pbert",
        ],
        check=True,
        cwd=Path.cwd(),
    )

    train_manifest = output_dir / "train_out2048_in1024_pbert.json"
    val_manifest = output_dir / "val_out2048_in1024_pbert.json"

    assert train_manifest.exists()
    assert val_manifest.exists()

    train_records = json.loads(train_manifest.read_text(encoding="utf-8"))
    val_records = json.loads(val_manifest.read_text(encoding="utf-8"))

    assert len(train_records) == 2
    assert len(val_records) == 1
    assert Path(train_records[0]["pcd_path"]).exists()
    assert Path(train_records[0]["diagram_path"]).exists()
    assert Path(train_records[0]["tokens_path"]).exists()
