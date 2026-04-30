import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, dropout, input_dim, intermediate_dim, feature_dim):
        super().__init__()
        self.dropout = dropout

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim

        self.feature_dim = feature_dim

        self.shared1 = nn.Sequential(
            self._make_linear(self.input_dim, 64, shared=True),
            self._make_linear(64, self.intermediate_dim, shared=True),
        )

        self.shared2 = nn.Sequential(
            self._make_linear(self.intermediate_dim, 64, shared=True),
            self._make_linear(64, 128, shared=True),
            self._make_linear(128, self.feature_dim, shared=True),
        )


    def forward(self, x):
        # x: [B, N, 3]
        x = x.transpose(1, 2)  # [B, 3, N]
        x = self.shared1(x)  # [B, 64, N]
        x = self.shared2(x)  # [B, 1024, N]
        x = x.transpose(1, 2)  # [B, N, 1024]

        return x


    def _make_linear(self, in_features, out_features, shared=False):
        layers = []
        if shared:
            layers += [
                nn.Conv1d(
                    in_channels=in_features, out_channels=out_features, kernel_size=1
                )
            ]
        else:
            layers += [nn.Linear(in_features, out_features)]
        layers += [nn.BatchNorm1d(out_features), nn.ReLU()]
        return nn.Sequential(*layers)