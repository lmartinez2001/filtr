import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def toy_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "toy_data"
    pcd_dir = root / "pcd"
    diagram_dir = root / "diagrams"
    tokens_dir = root / "tokens"
    splits_dir = root / "splits"

    pcd_dir.mkdir(parents=True)
    diagram_dir.mkdir(parents=True)
    tokens_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    sample_ids = ["sample_0", "sample_1", "sample_2"]
    records = []

    for idx, sample_id in enumerate(sample_ids):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0 + 0.1 * idx, 0.0, 0.0],
                [0.0, 1.0 + 0.1 * idx, 0.0],
                [0.0, 0.0, 1.0 + 0.1 * idx],
                [1.0, 1.0, 0.5 + 0.1 * idx],
                [0.5, 1.0, 1.0 + 0.1 * idx],
            ],
            dtype=np.float32,
        )
        np.save(pcd_dir / f"{sample_id}.npy", points)

        pd1 = np.array(
            [
                [0.1, 0.4 + 0.1 * idx],
                [0.2, 0.7 + 0.1 * idx],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(
            diagram_dir / f"{sample_id}.npz",
            pd0=np.array([[0.0, np.inf]], dtype=np.float32),
            pd1=pd1,
        )

        tokens_payload = {
            "patch_tokens": np.full((4, 16), idx + 1, dtype=np.float32),
            "patch_pos_embeddings": np.full((4, 16), idx + 0.5, dtype=np.float32),
        }
        for block_idx in range(12):
            tokens_payload[f"layer_features_{block_idx}"] = np.full(
                (5, 16), idx + block_idx / 10.0, dtype=np.float32
            )
        np.savez_compressed(tokens_dir / f"{sample_id}.npz", **tokens_payload)

        records.append(
            {
                "id": sample_id,
                "pcd_path": f"pcd/{sample_id}.npy",
                "diagram_path": f"diagrams/{sample_id}.npz",
                "tokens_path": f"tokens/{sample_id}.npz",
            }
        )

    train_records = records[:2]
    val_records = records[2:]

    for split_name, split_records in (("train", train_records), ("val", val_records)):
        (root / f"{split_name}.json").write_text(
            json.dumps(split_records, indent=2) + "\n",
            encoding="utf-8",
        )
        (splits_dir / f"{split_name}.txt").write_text(
            "\n".join(record["id"] for record in split_records) + "\n",
            encoding="utf-8",
        )

    return root
