import pytest
import torch

from models.pd_matcher import HungarianMatcherPersistence


def test_hungarian_matcher_cpu_returns_expected_min_cost_assignment():
    matcher = HungarianMatcherPersistence(
        cost_coord=1.0,
        cost_no_exist=0.0,
        use_exist_in_assignment=False,
        use_cpu=True,
    )
    outputs = {
        "pred_pairs": torch.tensor(
            [[[0.90, 1.10], [0.10, 0.40], [0.45, 0.80]]],
            dtype=torch.float32,
        ),
        "pred_exist": torch.zeros(1, 3),
    }
    targets = [
        {
            "pairs": torch.tensor(
                [[0.10, 0.40], [0.45, 0.80]],
                dtype=torch.float32,
            )
        }
    ]

    indices = matcher(outputs, targets)

    pred_idx, gt_idx = indices[0]
    assert pred_idx.tolist() == [1, 2]
    assert gt_idx.tolist() == [0, 1]


def test_hungarian_matcher_handles_empty_ground_truth():
    matcher = HungarianMatcherPersistence(
        cost_coord=1.0,
        cost_no_exist=0.0,
        use_exist_in_assignment=False,
        use_cpu=True,
    )
    outputs = {
        "pred_pairs": torch.rand(1, 3, 2),
        "pred_exist": torch.zeros(1, 3),
    }
    targets = [{"pairs": torch.empty(0, 2)}]

    indices = matcher(outputs, targets)

    pred_idx, gt_idx = indices[0]
    assert pred_idx.numel() == 0
    assert gt_idx.numel() == 0
    assert pred_idx.device == outputs["pred_pairs"].device


def test_hungarian_matcher_can_use_existence_logits_in_assignment():
    matcher = HungarianMatcherPersistence(
        cost_coord=0.0,
        cost_no_exist=1.0,
        use_exist_in_assignment=True,
        use_cpu=True,
    )
    outputs = {
        "pred_pairs": torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        "pred_exist": torch.tensor([[-5.0, 5.0]]),
    }
    targets = [{"pairs": torch.tensor([[0.0, 1.0]])}]

    indices = matcher(outputs, targets)

    pred_idx, gt_idx = indices[0]
    assert pred_idx.tolist() == [1]
    assert gt_idx.tolist() == [0]


def test_hungarian_matcher_requires_nonzero_assignment_cost():
    with pytest.raises(AssertionError):
        HungarianMatcherPersistence(
            cost_coord=0.0,
            cost_no_exist=0.0,
            use_exist_in_assignment=True,
            use_cpu=True,
        )
