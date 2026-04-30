import torch
from torch.utils.data import DataLoader

from datasets.donut import Donut
from engine import evaluate, train_one_epoch
from models.filtr import FILTR, SetCriterion
from models.pd_transformer import Transformer


class DummyMatcher:
    def __call__(self, outputs, targets):
        indices = []
        num_queries = outputs["pred_pairs"].shape[1]
        device = outputs["pred_pairs"].device

        for target in targets:
            n_targets = min(target["pairs"].shape[0], num_queries)
            pred_idx = torch.arange(n_targets, dtype=torch.long, device=device)
            tgt_idx = torch.arange(n_targets, dtype=torch.long, device=device)
            indices.append((pred_idx, tgt_idx))

        return indices


def test_train_one_epoch_smoke(toy_dataset_root):
    dataset = Donut(
        root=str(toy_dataset_root),
        split="train",
        index_filename="train.json",
        token_key="patch_tokens",
        pos_embed_key="patch_pos_embeddings",
        diagram_key="pd1",
        quantile_alpha=0.0,
        block_features_key="layer_features",
        use_intermediate_blocks=False,
        backbone="pbert",
        n_blocks=12,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    model = FILTR(
        decoder=Transformer(
            d_model=16,
            nhead=4,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        ),
        num_queries=4,
        in_feature_dim=16,
        aux_loss=False,
        use_layer_norm_adapter=False,
    )
    criterion = SetCriterion(
        matcher=DummyMatcher(),
        weight_dict={"existence": 0.1, "recon": 1.0, "diag": 0.1},
        losses=["existence", "recon", "diag"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logged_batches = []

    stats = train_one_epoch(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        epoch=0,
        max_norm=0.0,
        log_batch_metrics=logged_batches.append,
    )

    assert "total_loss" in stats
    assert torch.isfinite(torch.tensor(stats["total_loss"]))
    assert logged_batches


def test_evaluate_smoke(toy_dataset_root):
    dataset = Donut(
        root=str(toy_dataset_root),
        split="train",
        index_filename="train.json",
        token_key="patch_tokens",
        pos_embed_key="patch_pos_embeddings",
        diagram_key="pd1",
        quantile_alpha=0.0,
        block_features_key="layer_features",
        use_intermediate_blocks=False,
        backbone="pbert",
        n_blocks=12,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    model = FILTR(
        decoder=Transformer(
            d_model=16,
            nhead=4,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            activation="relu",
            normalize_before=True,
            return_intermediate_dec=False,
        ),
        num_queries=4,
        in_feature_dim=16,
        aux_loss=False,
        use_layer_norm_adapter=False,
    )
    criterion = SetCriterion(
        matcher=DummyMatcher(),
        weight_dict={"existence": 0.1, "recon": 1.0, "diag": 0.1},
        losses=["existence", "recon", "diag"],
    )

    stats, figures = evaluate(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        device=torch.device("cpu"),
        max_n_figs=1,
    )

    assert "total_loss" in stats
    assert len(figures) == 1
