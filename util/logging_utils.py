from __future__ import annotations

import logging

try:
    import wandb
except ImportError:
    wandb = None


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class StepLogger:
    def __init__(self, run_logger):
        self.global_step = 0
        self.run_logger = run_logger

    def __call__(self, stats: dict):
        log_payload = {f"train_step/{k}": v for k, v in stats.items()}
        self.run_logger.log(log_payload, step=self.global_step)
        self.global_step += 1


class DefaultLogger:
    def __init__(self):
        self.summary = {}

    def log(self, payload: dict, step: int | None = None, commit: bool = True):
        return None

    def image(self, figure):
        return figure

    def finish(self):
        return None


class WandbLogger:
    def __init__(self, project: str, config: dict, run_name: str):
        if wandb is None:
            raise ImportError(
                "Weights & Biases is not installed. Install `wandb` or set logger=default."
            )

        self.run = wandb.init(project=project, config=config, name=run_name)
        self.summary = self.run.summary

    def log(self, payload: dict, step: int | None = None, commit: bool = True):
        wandb.log(payload, step=step, commit=commit)

    def image(self, figure):
        return wandb.Image(figure)

    def finish(self):
        wandb.finish()
