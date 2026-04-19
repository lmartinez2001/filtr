# Code adapted from DETR (https://github.com/facebookresearch/detr)
import os
import sys
import math
import torch
import util.misc as utils

from tqdm import tqdm
from typing import Iterable
from torch.nn import Module
from torch.optim import Optimizer
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler


def train_one_epoch(model: Module, 
                    criterion: Module,
                    data_loader: Iterable, 
                    optimizer: Optimizer,
                    scheduler: _LRScheduler,
                    device: torch.device, 
                    epoch: int, 
                    max_norm: float,
                    log_batch_metrics: callable):
    model.train()
    criterion.train()

    n_batches = len(data_loader) # Assumes drop_last=True in DataLoader
    meters = defaultdict(float)

    for inputs, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch", total=n_batches):
        tokens = inputs["tokens"].to(device)
        pos_embeddings = inputs["pos_embeddings"].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # list of dicts containing "pairs"

        outputs = model(features=tokens, pos=pos_embeddings)
        loss_dict = criterion(outputs, targets) # {existence: ..., recon: ..., }
        total_loss = sum(loss_dict.values()) # existence + recon + ...
        
        batch_stats = {k: v.item() for k, v in loss_dict.items()}
        batch_stats["total_loss"] = total_loss.item()

        if not math.isfinite(total_loss):
            print(f"Loss is {total_loss}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        for k, v in batch_stats.items():
            meters[k] += v

        if scheduler is not None:
            scheduler.step()

        # To log lr scheduling
        batch_stats["lr"] = optimizer.param_groups[0]["lr"]

        log_batch_metrics(batch_stats)

    epoch_stats = {k: v / max(1, n_batches) for k, v in meters.items()}
    return epoch_stats


def train_one_epoch_end2end(model: Module, 
                            criterion: Module,
                            data_loader: Iterable, 
                            optimizer: Optimizer,
                            scheduler: _LRScheduler,
                            device: torch.device, 
                            epoch: int, 
                            max_norm: float,
                            log_batch_metrics: callable):
    model.train()
    criterion.train()

    n_batches = len(data_loader)
    meters = defaultdict(float)

    for inputs, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch", total=n_batches):
        inputs = inputs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict.values())
        total_loss = sum(loss_dict.values())

        batch_stats = {k: v.item() for k, v in loss_dict.items()}
        batch_stats["total_loss"] = total_loss.item()

        if not math.isfinite(total_loss):
            print(f"Loss is {total_loss}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        for k, v in batch_stats.items():
            meters[k] += v

        if scheduler is not None:
            scheduler.step()
        
        batch_stats["lr"] = optimizer.param_groups[0]["lr"]
        log_batch_metrics(batch_stats)

    epoch_stats = {k: v / max(1, n_batches) for k, v in meters.items()}
    return epoch_stats


@torch.no_grad()
def evaluate(model: Module, 
             criterion: Module, 
             data_loader: DataLoader, 
             device: torch.device, 
             max_n_figs: int = 32):
    model.eval()
    criterion.eval()

    figs = []
    n_batches = len(data_loader)
    meters = defaultdict(float)

    for inputs, targets in tqdm(data_loader, desc="Evaluation", unit="batch", total=n_batches):
        tokens = inputs["tokens"].to(device)
        pos_embeddings = inputs["pos_embeddings"].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(features=tokens, pos=pos_embeddings)
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict.values())

        batch_stats = {k: v.item() for k, v in loss_dict.items()}
        batch_stats["total_loss"] = total_loss.item()

        for k, v in batch_stats.items():
            meters[k] += v

        if len(figs) < max_n_figs:
            fig = utils.plot_persistence_diagrams(outputs, targets, idx=0)
            figs.append(fig)

    epoch_stats = {k: v / max(1, n_batches) for k, v in meters.items()}
    return epoch_stats, figs


@torch.no_grad()
def evaluate_end2end(model: Module, 
                     criterion: Module, 
                     data_loader: DataLoader, 
                     device: torch.device,
                     max_n_figs: int = 32):
    model.eval()
    criterion.eval()

    figs = []
    n_batches = len(data_loader)
    meters = defaultdict(float)

    for inputs, targets in tqdm(data_loader, desc="Evaluation", unit="batch", total=n_batches):
        inputs = inputs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(inputs)
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict.values())

        batch_stats = {k: v.item() for k, v in loss_dict.items()}
        batch_stats["total_loss"] = total_loss.item()

        for k, v in batch_stats.items():
            meters[k] += v

        if len(figs) < max_n_figs:
            fig = utils.plot_persistence_diagrams(outputs, targets, idx=0)
            figs.append(fig)

    epoch_stats = {k: v / max(1, n_batches) for k, v in meters.items()}
    return epoch_stats, figs
