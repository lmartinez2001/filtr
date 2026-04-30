from .dgcnn import DGCNN
from .pointnet import PointNet
from .pointnet2 import PointNet2
# from .repsurf.models.repsurf.repsurf_ssg_umb import RepSurf
from types import SimpleNamespace


# def build_backbone(args):
#     backbone_name = args.backbone_name
#     assert backbone_name in ["pointnet", "pointnet2", "dgcnn", "repsurf"], f'Unsupported backbone: {backbone_name}'

#     if backbone_name == "pointnet":
#         return PointNet(
#             input_dim=3,
#             intermediate_dim=64,
#             feature_dim=args.hidden_dim,
#             dropout=args.pnet["dropout"]
#         )
#     elif backbone_name == "dgcnn":
#         return DGCNN(
#             k=args.dgcnn["k"],
#             emb_dims=args.hidden_dim,
#             dropout=args.dgcnn["dropout"]
#         )
#     elif backbone_name == "pointnet2":
#         return PointNet2(
#             in_channels=args.pointnet2["in_channels"],
#             full=args.pointnet2["full"]
#         )
#     elif backbone_name == "repsurf":
#         return RepSurf(SimpleNamespace(**args.repsurf))