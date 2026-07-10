import torch
import torch.nn as nn
from utils import SafePool


class ConditionalPilotNet(nn.Module):
    """Approach 5 retrained against the smaller (256x320) YOLO backbone.
    Architecture matches models/smaller_img/reg4heads_smaller.ipynb exactly
    (48-channel late conv layers, unlike pilotnet_regNheadv2.py's 64-channel
    version, these are NOT interchangeable checkpoints)."""

    def __init__(self):
        super(ConditionalPilotNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1), nn.BatchNorm2d(48), nn.ReLU()
        )

        self.pool = SafePool((5, 10))
        self.flattened_size = 48 * 5 * 10  # 2400

        self.straight_head = nn.Sequential(
            nn.Linear(self.flattened_size, 50), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(50, 10), nn.ReLU(), nn.Linear(10, 2)
        )
        self.left_head = nn.Sequential(
            nn.Linear(self.flattened_size, 50), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(50, 10), nn.ReLU(), nn.Linear(10, 2)
        )
        self.right_head = nn.Sequential(
            nn.Linear(self.flattened_size, 50), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(50, 10), nn.ReLU(), nn.Linear(10, 2)
        )
        self.lane_following_head = nn.Sequential(
            nn.Linear(self.flattened_size, 50), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(50, 10), nn.ReLU(), nn.Linear(10, 2)
        )

    def forward(self, img, intent):
        features = self.feature_extractor(img)
        features = self.pool(features)
        features = features.reshape(-1, self.flattened_size)

        # one-hot order: straight=0, left=1, right=2, lane_following=3 ("stop" maps to the
        # same one-hot as lane_following, moot since callers gate "stop" out first)
        # gather by index, not boolean-mask indexing (mask indexing breaks the ONNX export)
        intent_indices = torch.argmax(intent, dim=1)
        stacked = torch.stack([
            self.straight_head(features),
            self.left_head(features),
            self.right_head(features),
            self.lane_following_head(features),
        ], dim=1)  # [B, 4, 2]
        output_velocities = stacked[torch.arange(features.shape[0], device=img.device), intent_indices]
        return output_velocities
