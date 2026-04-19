# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import argparse
from packaging import version
from types import SimpleNamespace
import yaml
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import numpy as np
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

def load_config(args_cmd: argparse.Namespace) -> SimpleNamespace:
    with open(args_cmd.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    if args_cmd.end2end:
        cfg_dict['end2end'] = True
    return SimpleNamespace(**cfg_dict, exp_name=args_cmd.exp_name)

def load_config_from_path(config_path: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)


def plot_persistence_diagrams(outputs: dict, targets: list, idx: int = 0) -> plt.Figure:
    pred_pairs = outputs["pred_pairs"][idx].cpu().numpy()
    pred_exist = torch.sigmoid(outputs["pred_exist"][idx]).cpu().numpy()
    pred_pairs_filt = pred_pairs[pred_exist > 0.5]
    gt_pairs = targets[idx]["pairs"].cpu().numpy()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12,6))
    fig.suptitle(f"{len(gt_pairs)} GT pairs, {len(pred_pairs_filt)} Predicted pairs")
    
    a1.scatter(*gt_pairs.T, label='Ground Truth', s=20, alpha=0.6)
    a1.scatter(*pred_pairs.T, label='Predictions', s=20, alpha=0.6)
    a1.set_xlabel('birth')
    a1.set_ylabel('death')
    a1.set_title('All predicted pairs')
    a1.legend()

    a2.scatter(*gt_pairs.T, label='Ground Truth', s=20, alpha=0.6)
    a2.scatter(*pred_pairs_filt.T, label='Predictions', s=20, alpha=0.6)
    a2.set_xlabel('birth')
    a2.set_ylabel('death')
    a2.set_title('Zoomed in predicted pairs')
    a2.legend()
    plt.close(fig)
    return fig

def h1_threshold_quantile(D, alpha=0.01):
    L = D[:, 1] - D[:, 0]
    tau = np.quantile(L, 1 - alpha)
    keep = L >= tau
    return D[keep], float(tau)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res