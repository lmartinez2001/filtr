import logging
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction

logger = logging.getLogger(__name__)


class PointNet2(nn.Module):
    def __init__(self, in_channels, full):
        super().__init__()
        self.full = full

        if full:
            logger.info("Using FULL PointNet2 Backbone")
        else:
            logger.info("Using LIGHTWEIGHT PointNet2 Backbone")

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channels,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        if full:
            self.sa3 = PointNetSetAbstraction(
                npoint=None, 
                radius=None, 
                nsample=None, 
                in_channel=256 + 3, 
                mlp=[256, 512, 1024], 
                group_all=True)
            
            self.proj = nn.Linear(1024, 256)

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2)
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        if self.full:
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            l3_points = l3_points.view(B, 1, -1)
            l3_points = self.proj(l3_points)
            return l3_points, l3_xyz, l2_xyz, l1_xyz
        else:
            l2_points = l2_points.transpose(1, 2)  # (B, N, C)
            l2_xyz = l2_xyz.transpose(1, 2)  # (B, N, 3)
            return l2_points, l2_xyz
    
