from torch.utils.data import DataLoader

from datasets.donut import Donut, DonutEnd2End


def test_donut_dataset_loads_sample_and_batch(toy_dataset_root):
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

    sample = dataset[0]
    assert sample["tokens"].shape == (4, 16)
    assert sample["pos_embeddings"].shape == (4, 16)
    assert sample["pairs"].shape[1] == 2

    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    assert batch["tokens"].shape == (2, 4, 16)
    assert batch["pos_embeddings"].shape == (2, 4, 16)
    assert len(batch["pairs"]) == 2


def test_donut_dataset_intermediate_block_mode_loads(toy_dataset_root):
    dataset = Donut(
        root=str(toy_dataset_root),
        split="train",
        index_filename="train.json",
        token_key="patch_tokens",
        pos_embed_key="patch_pos_embeddings",
        diagram_key="pd1",
        quantile_alpha=0.0,
        block_features_key="layer_features",
        use_intermediate_blocks=True,
        backbone="pbert",
        n_blocks=12,
    )

    sample = dataset[0]
    assert sample["tokens"].shape == (4, 16)
    assert sample["pairs"].shape[1] == 2


def test_donut_end2end_dataset_loads_sample_and_batch(toy_dataset_root):
    dataset = DonutEnd2End(
        root=str(toy_dataset_root),
        split="train",
        index_filename="train.json",
        diagram_key="pd1",
        quantile_alpha=0.0,
    )

    sample = dataset[0]
    assert sample["pcd"].shape[1] == 3
    assert sample["pairs"].shape[1] == 2

    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    assert batch["pcd"].shape[0] == 2
    assert batch["pcd"].shape[2] == 3
    assert len(batch["pairs"]) == 2
