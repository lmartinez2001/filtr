import torch
from torch import nn

from models.filtr import FILTR, FILTREnd2End


class DummyDecoder(nn.Module):
    d_model = 4

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class DummyEncoder(nn.Module):
    pass


class DummyBackbone(nn.Module):
    pass


def assert_valid_persistence_pairs(pairs, expected_shape):
    assert pairs.shape == expected_shape
    assert torch.isfinite(pairs).all()
    assert torch.all((pairs[..., 0] >= 0.0) & (pairs[..., 0] <= 1.0))
    assert torch.all(pairs[..., 1] > pairs[..., 0])


def test_filtr_format_pairs_outputs_valid_persistence_pairs_for_extreme_logits():
    model = FILTR(
        decoder=DummyDecoder(),
        num_queries=3,
        in_feature_dim=4,
        aux_loss=False,
        use_layer_norm_adapter=False,
    )
    raw_pairs = torch.tensor(
        [[[-100.0, -100.0], [0.0, 0.0], [100.0, 100.0]]],
        dtype=torch.float32,
    )

    formatted = model._format_pairs(raw_pairs)

    assert_valid_persistence_pairs(formatted, raw_pairs.shape)


def test_filtr_end2end_format_pairs_outputs_valid_persistence_pairs():
    model = FILTREnd2End(
        backbone=DummyBackbone(),
        encoder=DummyEncoder(),
        decoder=DummyDecoder(),
        num_queries=2,
        aux_loss=False,
    )
    raw_pairs = torch.tensor(
        [[[-20.0, 0.5], [20.0, -0.5]]],
        dtype=torch.float32,
    )

    formatted = model._format_pairs(raw_pairs)

    assert_valid_persistence_pairs(formatted, raw_pairs.shape)
