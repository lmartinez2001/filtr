import pytest
import torch

from models.filtr import SetCriterion


class FixedMatcher:
    def __init__(self, indices):
        self.indices = indices

    def __call__(self, outputs, targets):
        return self.indices


def test_set_criterion_returns_finite_weighted_losses_for_known_matches():
    outputs = {
        "pred_pairs": torch.tensor(
            [[[0.10, 0.50], [0.40, 0.90], [0.20, 0.80]]],
            dtype=torch.float32,
        ),
        "pred_exist": torch.tensor([[4.0, -2.0, 0.5]], dtype=torch.float32),
    }
    targets = [{"pairs": torch.tensor([[0.10, 0.50], [0.20, 0.80]])}]
    matcher = FixedMatcher(
        [
            (
                torch.tensor([0, 2], dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
            )
        ]
    )
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict={"existence": 0.5, "recon": 2.0, "diag": 0.25},
        losses=["existence", "recon", "diag"],
    )

    losses = criterion(outputs, targets)

    assert set(losses) == {"existence", "recon", "diag"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert torch.isclose(losses["recon"], torch.tensor(0.0))
    assert losses["diag"] > 0


def test_set_criterion_handles_empty_targets_without_nan_losses():
    outputs = {
        "pred_pairs": torch.tensor(
            [[[0.10, 0.30], [0.40, 0.45]]],
            dtype=torch.float32,
        ),
        "pred_exist": torch.tensor([[-3.0, -2.0]], dtype=torch.float32),
    }
    targets = [{"pairs": torch.empty(0, 2)}]
    matcher = FixedMatcher(
        [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    )
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict={"existence": 1.0, "recon": 1.0, "diag": 1.0},
        losses=["existence", "recon", "diag"],
    )

    losses = criterion(outputs, targets)

    assert all(torch.isfinite(loss) for loss in losses.values())
    assert torch.isclose(losses["recon"], torch.tensor(0.0))
    assert losses["existence"] > 0
    assert losses["diag"] > 0


def test_set_criterion_diagonal_loss_is_zero_when_all_predictions_are_matched():
    outputs = {
        "pred_pairs": torch.tensor([[[0.10, 0.30], [0.40, 0.70]]]),
        "pred_exist": torch.tensor([[2.0, 2.0]]),
    }
    targets = [{"pairs": torch.tensor([[0.10, 0.30], [0.40, 0.70]])}]
    matcher = FixedMatcher(
        [
            (
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
            )
        ]
    )
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict={"diag": 1.0},
        losses=["diag"],
    )

    losses = criterion(outputs, targets)

    assert torch.isclose(losses["diag"], torch.tensor(0.0))


def test_set_criterion_rejects_unknown_loss_name():
    with pytest.raises(ValueError, match="not recognized"):
        SetCriterion(matcher=FixedMatcher([]), weight_dict={}, losses=["missing"])
