from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as data

import util.misc as utils


class BaseDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        index_filename: Optional[str],
        diagram_key: str,
        quantile_alpha: float,
    ) -> None:
        self.root = Path(root)
        if index_filename is None:
            index_filename = f"{split}.json"
        self.index_path = self.root / index_filename
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        with open(self.index_path, "r") as f:
            self.records: List[Dict[str, Any]] = json.load(f)

        self.diagram_key = diagram_key
        self.quantile_alpha = quantile_alpha

    def __len__(self) -> int:
        return len(self.records)

    def _load_pairs(self, rec: Dict[str, Any], sample_id: str) -> torch.Tensor:
        diag_path = self.root / rec["diagram_path"]
        pairs = np.load(diag_path)[self.diagram_key]
        pairs, _ = utils.h1_threshold_quantile(pairs, self.quantile_alpha)
        pairs = torch.from_numpy(pairs).float()

        if pairs.numel() > 0:
            if not (pairs.dim() == 2 and pairs.shape[1] == 2):
                raise ValueError(
                    f"diagram pairs must be (N,2); got {tuple(pairs.shape)} for id={sample_id}"
                )
            return pairs

        return pairs.reshape(0, 2)


class BaseTokenDataset(BaseDataset):
    backbones_without_cls_token: set[str] = set()

    def __init__(
        self,
        root: str,
        split: str,
        index_filename: Optional[str],
        token_key: str,
        pos_embed_key: str,
        diagram_key: str,
        quantile_alpha: float,
        block_features_key: str,
        use_intermediate_blocks: bool,
        backbone: str,
        supported_backbones: set[str],
        n_blocks: int = 12,
    ) -> None:
        super().__init__(root, split, index_filename, diagram_key, quantile_alpha)
        self.token_key = token_key
        self.pos_embed_key = pos_embed_key
        self.block_features_key = block_features_key
        self.use_intermediate_blocks = use_intermediate_blocks
        self.n_blocks = n_blocks
        self.backbone = backbone

        if self.backbone not in supported_backbones:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        if use_intermediate_blocks:
            print(f"[DATASET] Using intermediate block features from backbone '{self.backbone}'")

    def _intermediate_blocks_collate(self, features: np.ndarray) -> torch.Tensor:
        final_feat = []
        for block in range(self.n_blocks):
            block_feat = features[f"{self.block_features_key}_{block}"]
            if self.backbone in self.backbones_without_cls_token:
                final_feat.append(block_feat)
            else:
                final_feat.append(block_feat[1:, :])

        final_feat = np.sum(final_feat, axis=0) / self.n_blocks
        return torch.from_numpy(final_feat).float()

    def collate_fn(self, batch):
        return {
            "tokens": torch.stack([sample["tokens"] for sample in batch], dim=0),
            "pos_embeddings": torch.stack([sample["pos_embeddings"] for sample in batch], dim=0),
            "pairs": [sample["pairs"] for sample in batch],
        }


class BasePointCloudDataset(BaseDataset):
    def collate_fn(self, batch):
        return {
            "pcd": torch.stack([sample["pcd"] for sample in batch], dim=0),
            "pairs": [sample["pairs"] for sample in batch],
        }
