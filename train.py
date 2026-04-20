import time
import yaml
import wandb
import torch
import hydra
import random
import datetime
import numpy as np
import util.misc as utils

from pathlib import Path
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from util.monitor import get_model_complexity_info, get_e2e_model_complexity_info
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from engine import evaluate, train_one_epoch, evaluate_end2end, train_one_epoch_end2end

class StepLogger():
    def __init__(self):
        self.global_step = 0

    def __call__(self, stats: dict):
        log_payload = {f"train_step/{k}": v for k, v in stats.items()}
        wandb.log(log_payload, step=self.global_step)
        self.global_step += 1


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):    

    wandb.init(project="FILTR-test", config=OmegaConf.to_container(cfg, resolve=True), name=cfg.exp_name)
    logger = StepLogger()
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

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
        num_workers=cfg.training.num_workers
    )
    
    val_set = instantiate(cfg.dataset.val)
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.training.batch_size, 
        shuffle=False,
        drop_last=True, 
        collate_fn=val_set.collate_fn, 
        num_workers=cfg.training.num_workers
    )

    lr_scheduler = build_scheduler(optimizer=optimizer, cfg=cfg, steps_per_epoch=len(train_loader))

    output_dir = Path(cfg.output_dir) / cfg.exp_name
    if output_dir:
        print("==> creating output dir:", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    # ==> Compute FLOPs
    if cfg.model.get("is_end2end", False):
        dummy_pts = torch.randn(1, 1024, 3).to(device)
        get_e2e_model_complexity_info(model=model, inputs=dummy_pts)
    else:
        dummy_tokens = torch.randn(1, cfg.model.network.num_queries, cfg.model.network.in_feature_dim).to(device)
        dummy_pos = torch.randn_like(dummy_tokens).to(device)
        get_model_complexity_info(model=model, tokens=dummy_tokens, pos=dummy_pos)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[TRAINING] Number of training params:", n_parameters)
    print("[TRAINING] Using losses:", "".join([f"{loss}, " for loss in criterion.losses]))
    print(f"[TRAINING] Starting training for {cfg.training.epochs} epochs")
    start_time = time.time()
    total_epochs = cfg.training.epochs
    track_cuda_memory = device.type == "cuda"
    for epoch in range(total_epochs):
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
                                                  log_batch_metrics=logger)
        else:
            train_stats = train_one_epoch(model=model, 
                                        criterion=criterion, 
                                        data_loader=train_loader, 
                                        optimizer=optimizer,
                                        scheduler=lr_scheduler, 
                                        device=device, 
                                        epoch=epoch, 
                                        max_norm=cfg.training.clip_max_norm,
                                        log_batch_metrics=logger)
        
        epoch_duration = time.time() - epoch_start

        log_payload = {f"train_epoch/{k}": v for k, v in train_stats.items()}
        log_payload["train_epoch/epoch_duration"] = epoch_duration
        
        if epoch == 0 and track_cuda_memory:
            max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"[TRAINING] Max memory allocated: {max_mem_gb:.2f} GB")

        wandb.log(log_payload, step=logger.global_step)

        # ==> Save checkpoints
        checkpoint_paths = [output_dir / "last.pth"]
        if epoch % cfg.training.eval_every == 0:
            checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
        for checkpoint_path in checkpoint_paths:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "epoch": epoch,
                "global_step": logger.global_step,
            }, checkpoint_path)


        # ==> Evauation 
        val_stats = None
        if epoch % cfg.training.eval_every == 0:
            print("==> Starting evaluation")
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
            wandb.log({f"val/{k}": v for k, v in val_stats.items()}, step=logger.global_step, commit=False)
            wandb.log({"val/diagrams": [wandb.Image(fig) for fig in figs]}, step=logger.global_step)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # log total time to wandb as a summary metric
    wandb.run.summary["total_training_time"] = total_time_str
    print(f"Training time {total_time_str}")
    wandb.finish()


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
        print("[TRAINING] No scheduler is used.")
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")
    return scheduler


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    main()
