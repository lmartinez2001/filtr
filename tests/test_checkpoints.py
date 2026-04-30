from types import SimpleNamespace

import pytest
import torch

from train import load_best_metric, load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_restores_training_state(tmp_path):
    checkpoint_path = tmp_path / "last.pth"
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        global_step=12,
        best_metric_name="total_loss",
        best_mode="min",
        best_metric_value=0.25,
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer,
        step_size=1,
    )
    step_logger = SimpleNamespace(global_step=0)

    start_epoch = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        step_logger=step_logger,
        device=torch.device("cpu"),
        weights_only=False,
        strict=True,
    )

    assert start_epoch == 5
    assert step_logger.global_step == 12
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)


def test_load_checkpoint_weights_only_does_not_resume_epoch_or_step(tmp_path):
    checkpoint_path = tmp_path / "weights.pth"
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=4,
        global_step=12,
        best_metric_name="total_loss",
        best_mode="min",
        best_metric_value=0.25,
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    step_logger = SimpleNamespace(global_step=0)

    start_epoch = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=None,
        step_logger=step_logger,
        device=torch.device("cpu"),
        weights_only=True,
        strict=True,
    )

    assert start_epoch == 0
    assert step_logger.global_step == 0
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)


def test_load_checkpoint_missing_file_raises(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(
            checkpoint_path=tmp_path / "missing.pth",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            step_logger=SimpleNamespace(global_step=0),
            device=torch.device("cpu"),
            weights_only=False,
            strict=True,
        )


def test_load_best_metric_prefers_best_checkpoint_when_present(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    last_path = tmp_path / "last.pth"
    best_path = tmp_path / "best.pth"

    save_checkpoint(
        checkpoint_path=last_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        global_step=3,
        best_metric_name="total_loss",
        best_mode="min",
        best_metric_value=0.50,
    )
    save_checkpoint(
        checkpoint_path=best_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=2,
        global_step=6,
        best_metric_name="total_loss",
        best_mode="min",
        best_metric_value=0.25,
    )

    assert load_best_metric(last_path) == 0.25
