import time
import yaml
import torch
import hydra
import random
import datetime
import logging
import numpy as np
import util.misc as utils

from pathlib import Path
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from util.logging_utils import configure_logging
from util.monitor import get_model_complexity_info, get_e2e_model_complexity_info
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from engine import evaluate, train_one_epoch, evaluate_end2end, train_one_epoch_end2end

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

class StepLogger():
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
            raise ImportError("Weights & Biases is not installed. Install `wandb` or set logger=default.")

        self.run = wandb.init(project=project, config=config, name=run_name)
        self.summary = self.run.summary

    def log(self, payload: dict, step: int | None = None, commit: bool = True):
        wandb.log(payload, step=step, commit=commit)

    def image(self, figure):
        return wandb.Image(figure)

    def finish(self):
        wandb.finish()


def build_logger(cfg: DictConfig):
    logger_name = cfg.logger.name
    if logger_name == "default":
        logger.info("Using default logger (no external tracking).")
        return DefaultLogger()
    if logger_name == "wandb":
        return WandbLogger(
            project=cfg.logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            run_name=cfg.exp_name,
        )
    raise ValueError(f"Unknown logger type: {logger_name}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):    
    configure_logging()
    run_logger = build_logger(cfg)
    step_logger = StepLogger(run_logger)
    device = torch.device(cfg.device)
    set_seed(cfg.seed, deterministic=cfg.training.deterministic)
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(cfg.seed)

    model = instantiate(cfg.model.network)
    model.to(device)
    
    criterion = instantiate(cfg.model.criterion)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=cfg.optimizer.lr, 
                                  weight_decay=cfg.optimizer.weight_decay)

    # ==> Datasets and loaders
    train_set = instantiate(cfg.dataset.train)
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        drop_last=True,
        collate_fn=train_set.collate_fn, 
        num_workers=cfg.training.num_workers,
        worker_init_fn=seed_worker,
        generator=data_loader_generator,
    )
    
    val_set = instantiate(cfg.dataset.val)
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=False,
        drop_last=True, # To change later on
        collate_fn=val_set.collate_fn, 
        num_workers=cfg.training.num_workers,
        worker_init_fn=seed_worker,
        generator=data_loader_generator,
    )

    lr_scheduler = build_scheduler(optimizer=optimizer, cfg=cfg, steps_per_epoch=len(train_loader))
    start_epoch = 0
    best_metric_name = cfg.training.best_metric
    best_mode = cfg.training.best_mode
    best_metric_value = float("inf") if best_mode == "min" else float("-inf")

    if cfg.training.resume_from is not None:
        checkpoint_path = Path(to_absolute_path(cfg.training.resume_from))
        start_epoch = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            step_logger=step_logger,
            device=device,
            weights_only=cfg.training.load_weights_only,
            strict=cfg.training.strict_checkpoint_load,
        )
        if not cfg.training.load_weights_only:
            resumed_best_metric = load_best_metric(checkpoint_path)
            if resumed_best_metric is not None:
                best_metric_value = resumed_best_metric

    output_dir = Path(cfg.output_dir) / cfg.exp_name
    if output_dir:
        logger.info("Creating output dir: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    # ==> Compute FLOPs
    if cfg.model.get("is_end2end", False):
        dummy_pts = torch.randn(1, 1024, 3).to(device)
        get_e2e_model_complexity_info(model=model, inputs=dummy_pts)
    else:
        sample = train_set[0]
        token_shape = sample["tokens"].shape
        pos_shape = sample["pos_embeddings"].shape

        if token_shape[-1] != cfg.model.network.in_feature_dim:
            raise ValueError(
                "Configured in_feature_dim does not match dataset features: "
                f"config={cfg.model.network.in_feature_dim}, dataset={token_shape[-1]}"
            )
        if pos_shape[-1] != token_shape[-1]:
            raise ValueError(
                "Token and positional embedding feature dimensions must match for FLOPs audit: "
                f"tokens={token_shape[-1]}, pos={pos_shape[-1]}"
            )

        dummy_tokens = torch.randn(1, *token_shape).to(device)
        dummy_pos = torch.randn(1, *pos_shape).to(device)
        get_model_complexity_info(model=model, tokens=dummy_tokens, pos=dummy_pos)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of training params: %s", n_parameters)
    logger.info("Using losses: %s", "".join([f"{loss}, " for loss in criterion.losses]))
    logger.info("Starting training for %s epochs", cfg.training.epochs)
    start_time = time.time()
    total_epochs = cfg.training.epochs
    track_cuda_memory = device.type == "cuda"
    for epoch in range(start_epoch, total_epochs):
        # print(f"==> Starting epoch {epoch+1}/{total_epochs}")
        if track_cuda_memory:
            torch.cuda.reset_accumulated_memory_stats(device)
        epoch_start = time.time()
        
        if cfg.model.get("is_end2end", False):
            train_stats = train_one_epoch_end2end(model=model, 
                                                  criterion=criterion, 
                                                  data_loader=train_loader, 
                                                  optimizer=optimizer,
                                                  scheduler=lr_scheduler, 
                                                  device=device, 
                                                  epoch=epoch, 
                                                  max_norm=cfg.training.clip_max_norm,
                                                  log_batch_metrics=step_logger)
        else:
            train_stats = train_one_epoch(model=model, 
                                        criterion=criterion, 
                                        data_loader=train_loader, 
                                        optimizer=optimizer,
                                        scheduler=lr_scheduler, 
                                        device=device, 
                                        epoch=epoch, 
                                        max_norm=cfg.training.clip_max_norm,
                                        log_batch_metrics=step_logger)
        
        epoch_duration = time.time() - epoch_start

        log_payload = {f"train_epoch/{k}": v for k, v in train_stats.items()}
        log_payload["train_epoch/epoch_duration"] = epoch_duration
        
        if epoch == 0 and track_cuda_memory:
            max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            logger.info("Max memory allocated: %.2f GB", max_mem_gb)

        run_logger.log(log_payload, step=step_logger.global_step)

        # ==> Save checkpoints
        checkpoint_paths = [output_dir / "last.pth"]
        if epoch % cfg.training.eval_every == 0:
            checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
        for checkpoint_path in checkpoint_paths:
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                epoch=epoch,
                global_step=step_logger.global_step,
                best_metric_name=best_metric_name,
                best_mode=best_mode,
                best_metric_value=best_metric_value,
            )


        # ==> Evauation 
        val_stats = None
        if epoch % cfg.training.eval_every == 0:
            logger.info("Starting evaluation")
            if cfg.model.get("is_end2end", False):
                val_stats, figs = evaluate_end2end(model=model, 
                                                  criterion=criterion, 
                                                  data_loader=val_loader, 
                                                  device=device)
            else:
                val_stats, figs = evaluate(model=model, 
                                        criterion=criterion, 
                                        data_loader=val_loader, 
                                        device=device)
            run_logger.log({f"val/{k}": v for k, v in val_stats.items()}, step=step_logger.global_step, commit=False)
            run_logger.log({"val/diagrams": [run_logger.image(fig) for fig in figs]}, step=step_logger.global_step)

            current_metric_value = val_stats.get(best_metric_name)
            if current_metric_value is None:
                logger.warning(
                    "Best-checkpoint metric '%s' not found in validation stats. Skipping best checkpoint update.",
                    best_metric_name,
                )
            elif is_better_metric(current_metric_value, best_metric_value, best_mode):
                best_metric_value = current_metric_value
                best_checkpoint_path = output_dir / "best.pth"
                save_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    epoch=epoch,
                    global_step=step_logger.global_step,
                    best_metric_name=best_metric_name,
                    best_mode=best_mode,
                    best_metric_value=best_metric_value,
                )
                save_checkpoint(
                    checkpoint_path=output_dir / "last.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    epoch=epoch,
                    global_step=step_logger.global_step,
                    best_metric_name=best_metric_name,
                    best_mode=best_mode,
                    best_metric_value=best_metric_value,
                )
                logger.info(
                    "Saved new best checkpoint to %s using %s=%s",
                    best_checkpoint_path,
                    best_metric_name,
                    best_metric_value,
                )


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    run_logger.summary["total_training_time"] = total_time_str
    logger.info("Training time %s", total_time_str)
    run_logger.finish()


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig, steps_per_epoch: int):
    total_steps = cfg.training.epochs * steps_per_epoch
    warmup_steps = cfg.scheduler.warmup_epochs * steps_per_epoch

    if cfg.scheduler.type == "cosine":
        warmup_scheduler = LinearLR(optimizer, 
                                    start_factor=cfg.scheduler.start_factor, 
                                    end_factor=1.0,
                                    total_iters=warmup_steps
                                    )
        main_scheduler = CosineAnnealingLR(optimizer, 
                                           T_max=total_steps - warmup_steps, 
                                           eta_min=cfg.scheduler.eta_min)
        scheduler = SequentialLR(optimizer=optimizer, 
                                 schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    elif cfg.scheduler.type == "none":
        logger.info("No scheduler is used.")
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")
    return scheduler


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step_logger: StepLogger,
    device: torch.device,
    weights_only: bool,
    strict: bool,
) -> int:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=strict)

    if weights_only:
        logger.info("Loaded model weights only.")
        return 0

    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("lr_scheduler")
    if scheduler_state is not None:
        if scheduler is None:
            logger.warning("Checkpoint contains scheduler state but current config disables the scheduler. Skipping scheduler restore.")
        else:
            scheduler.load_state_dict(scheduler_state)
    elif scheduler is not None:
        logger.warning("No scheduler state found in checkpoint. Continuing with a fresh scheduler.")

    step_logger.global_step = checkpoint.get("global_step", 0)
    start_epoch = checkpoint.get("epoch", -1) + 1
    logger.info("Resuming from epoch %s with global step %s", start_epoch, step_logger.global_step)
    return start_epoch


def load_best_metric(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    best_metric_value = checkpoint.get("best_metric_value")
    best_checkpoint_path = checkpoint_path.parent / "best.pth"
    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_metric_value = best_checkpoint.get("best_metric_value", best_metric_value)
    return best_metric_value


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric_name: str,
    best_mode: str,
    best_metric_value: float,
):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric_name": best_metric_name,
        "best_metric_mode": best_mode,
        "best_metric_value": best_metric_value,
    }, checkpoint_path)


def is_better_metric(current_value: float, best_value: float, mode: str) -> bool:
    if mode == "min":
        return current_value < best_value
    if mode == "max":
        return current_value > best_value
    raise ValueError(f"Unknown best metric mode: {mode}")


def set_seed(seed: int, deterministic: bool = False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    main()
