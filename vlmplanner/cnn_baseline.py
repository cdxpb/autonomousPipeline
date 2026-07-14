# small CNN trained from scratch on the raw crop+segment mask, no SmolVLM2 at all.
# same conv trunk shape as pilotnet_regNheadv2.py's feature_extractor

import torch.nn as nn

from classifier_common import MultiTaskHead


class SmallCNNTrunk(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        # plain mean instead of AdaptiveAvgPool2d, see utils.py's SafePool comment: it's buggy on MPS
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, out_dim, kernel_size=3, stride=1), nn.BatchNorm2d(out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x).mean(dim=(-2, -1))


class CNNClassifier(nn.Module):
    def __init__(self, trunk_dim: int = 64, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.trunk = SmallCNNTrunk(trunk_dim)
        self.head = MultiTaskHead(trunk_dim, hidden, dropout)

    def forward(self, x):
        return self.head(self.trunk(x))
