import torch


def pc_norm(pc: torch.Tensor) -> torch.Tensor:
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.norm(pc, dim=1))
    pc = pc / m
    return pc
