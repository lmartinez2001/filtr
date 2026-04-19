from __future__ import annotations
from typing import Dict, Tuple

import torch
import numpy as np

from datasets.base import BasePointCloudDataset, BaseTokenDataset
from datasets.utils import pc_norm


class ABCFilt(BaseTokenDataset):
    backbones_without_cls_token = {"pmae", "pgpt", "pcpmae"}

    def __init__(
        self,
        root: str,
        split: str,
        index_filename: str,
        token_key: str,
        pos_embed_key: str,
        diagram_key: str,
        quantile_alpha: float,
        block_features_key: str,
        use_intermediate_blocks: bool,
        backbone: str
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            index_filename=index_filename,
            token_key=token_key,
            pos_embed_key=pos_embed_key,
            diagram_key=diagram_key,
            quantile_alpha=quantile_alpha,
            block_features_key=block_features_key,
            use_intermediate_blocks=use_intermediate_blocks,
            backbone=backbone,
            supported_backbones={"pmae", "pbert", "pgpt", "pcpmae"},
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.records[idx]
        sample_id = rec.get("id", str(idx))

        # Load tokens
        tokens_path = self.root / rec["tokens_path"]
        features = np.load(tokens_path)

        # Use intermediate block features
        if self.use_intermediate_blocks:
            tokens = self._intermediate_blocks_collate(features)
        else:
            tokens = torch.from_numpy(features[self.token_key]).float()
        pos_embeddings = torch.from_numpy(features[self.pos_embed_key]).float()
        pairs = self._load_pairs(rec, sample_id)

        target: Dict[str, torch.Tensor] = {
            "pairs": pairs,
        }

        return tokens, pos_embeddings, target


class ABCFiltEnd2End(BasePointCloudDataset):
    def __init__(
        self,
        root: str,
        split: str,
        index_filename: str,
        diagram_key: str,
        quantile_alpha: float,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            index_filename=index_filename,
            diagram_key=diagram_key,
            quantile_alpha=quantile_alpha,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.records[idx]
        sample_id = rec.get("id", str(idx))

        # Load tokens
        pcd_path = self.root / rec["pcd_path"]
        pcd = np.load(pcd_path)
        pcd = torch.from_numpy(pcd).float()
        pcd = pc_norm(pcd)
        pairs = self._load_pairs(rec, sample_id)

        target: Dict[str, torch.Tensor] = {
            "pairs": pairs,
        }

        return pcd, target
