import torch
import torch.nn as nn
from utils import SafePool

class ConditionalPilotNet(nn.Module):
    def __init__(self, num_frames=3): # Added num_frames parameter
        super(ConditionalPilotNet, self).__init__()

        self.num_frames = num_frames

        # 1. Image Feature Extractor (Modified for N input channels)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.num_frames, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.pool = SafePool((5, 10))
        self.image_feature_size = 64 * 5 * 10 # 3200

        # 2. State Encoder for Previous Velocities
        # Takes the previous 2 velocities and embeds them into a feature vector
        self.state_feature_size = 32
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, self.state_feature_size),
            nn.ReLU()
        )

        # 3. Combined size for the heads
        self.combined_size = self.image_feature_size + self.state_feature_size

        # 4. Heads (Now taking combined image + state features)
        def create_head():
            return nn.Sequential(
                nn.Linear(self.combined_size, 128), # Slightly wider to handle combined features
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 2) # Output 2 velocities
            )

        self.straight_head = create_head()
        self.left_head = create_head()
        self.right_head = create_head()

    def forward(self, img_sequence, prev_velocities, intent):
        # img_sequence shape: [batch, num_frames, H, W]
        # prev_velocities shape: [batch, 2]

        # Extract Image Features
        img_features = self.feature_extractor(img_sequence)
        img_features = self.pool(img_features)
        img_features = img_features.reshape(-1, self.image_feature_size)

        # Extract State Features
        state_features = self.state_encoder(prev_velocities)

        # Fusion: Concatenate image features with the physical state
        # Shape becomes [batch, 3200 + 32]
        fused_features = torch.cat((img_features, state_features), dim=1)

        # Intent Routing (Same as your original code)
        intent_indices = torch.argmax(intent, dim=1)
        batch_size = fused_features.shape[0]
        output_velocities = torch.zeros(batch_size, 2, device=img_sequence.device)

        is_straight = (intent_indices == 0)
        is_left = (intent_indices == 1)
        is_right = (intent_indices == 2)

        if is_straight.any():
            output_velocities[is_straight] = self.straight_head(fused_features[is_straight])
        if is_left.any():
            output_velocities[is_left] = self.left_head(fused_features[is_left])
        if is_right.any():
            output_velocities[is_right] = self.right_head(fused_features[is_right])

        return output_velocities