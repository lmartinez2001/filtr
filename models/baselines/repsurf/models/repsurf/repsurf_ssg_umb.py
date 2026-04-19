"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch.nn as nn
import torch.nn.functional as F
from models.encoders.repsurf.modules.repsurface_utils import (
    SurfaceAbstractionCD,
    UmbrellaSurfaceConstructor,
)


class RepSurf(nn.Module):
    def __init__(self, args):
        super(RepSurf, self).__init__()
        center_channel = (
            0 if not args.return_center else (6 if args.return_polar else 3)
        )
        repsurf_channel = 10
        self.full = args.full

        self.init_nsample = args.num_point
        self.return_dist = args.return_dist
        self.surface_constructor = UmbrellaSurfaceConstructor(
            args.group_size + 1,
            repsurf_channel,
            return_dist=args.return_dist,
            aggr_type=args.umb_pool,
            cuda=args.cuda_ops,
        )
        self.sa1 = SurfaceAbstractionCD(
            npoint=512,
            radius=0.2,
            nsample=32,
            feat_channel=repsurf_channel,
            pos_channel=center_channel,
            mlp=[64, 64, 128],
            group_all=False,
            return_polar=args.return_polar,
            cuda=args.cuda_ops,
        )
        self.sa2 = SurfaceAbstractionCD(
            npoint=128,
            radius=0.4,
            nsample=64,
            feat_channel=128 + repsurf_channel,
            pos_channel=center_channel,
            mlp=[128, 128, 256],
            group_all=False,
            return_polar=args.return_polar,
            cuda=args.cuda_ops,
        )
        if self.full:
            print("[MODEL] Using FULL RepSurf Backbone")
            self.sa3 = SurfaceAbstractionCD(
                npoint=None,
                radius=None,
                nsample=None,
                feat_channel=256 + repsurf_channel,
                pos_channel=center_channel,
                mlp=[256, 512, 1024],
                group_all=True,
                return_polar=args.return_polar,
                cuda=args.cuda_ops,
            )
            self.proj = nn.Linear(1024, 256)
        else:
            print("[MODEL] Using LIGHT RepSurf Backbone")
        # modelnet40
        # self.classfier = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(True),
        #     nn.Dropout(0.4),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(True),
        #     nn.Dropout(0.4),
        #     nn.Linear(256, args.num_class),
        # )

    def forward(self, points):
        points = points.transpose(1, 2) # (B, 3, N)
        B, _, _ = points.size()
        # init
        center = points[:, :3, :]

        normal = self.surface_constructor(center)

        center, normal, feature = self.sa1(center, normal, None)
        center, normal, feature = self.sa2(center, normal, feature)

        if self.full:
            center, normal, feature = self.sa3(center, normal, feature)
            feature = feature.view(B, 1, 1024)
            feature = self.proj(feature) # (B, 1, 256) to match decoder size
            return feature
        else:
            feature = feature.transpose(1, 2)  # (B, N, C)
            center = center.transpose(1, 2)  # (B, N, 3)
            return feature, center

    def get_loss(self, target, preds):
        clf_loss = F.cross_entropy(preds, target)
        return clf_loss
