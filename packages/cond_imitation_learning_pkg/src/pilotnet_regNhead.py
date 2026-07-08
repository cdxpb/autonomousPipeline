import torch
import torch.nn as nn
from utils import SafePool

class ConditionalPilotNet(nn.Module):
    def __init__(self):
        super(ConditionalPilotNet, self).__init__()

        # Modified to accept 1 input channel (segmentation mask)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.pool = SafePool((5, 10))
        # Adjusted flattened_size based on expected output after pooling
        self.flattened_size = 64 * 5 * 10 # 3200

        # Define three separate regression heads for 'straight', 'left', 'right'
        # Each head takes only the image features (flattened_size) as input
        # Using the specified smaller head architecture
        self.straight_head = nn.Sequential(
            nn.Linear(self.flattened_size, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2) # Output 2 velocities as clarified
        )
        self.left_head = nn.Sequential(
            nn.Linear(self.flattened_size, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2) # Output 2 velocities as clarified
        )
        self.right_head = nn.Sequential(
            nn.Linear(self.flattened_size, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2) # Output 2 velocities as clarified
        )
        # The 'stop' intent will implicitly output zero velocities in the forward pass.

    def forward(self, img, intent):
        # img is now expected to be the processed segmentation mask (1-channel tensor)
        features = self.feature_extractor(img)
        features = self.pool(features)
        features = features.reshape(-1, self.flattened_size) # [batch_size, flattened_size]

        # Determine which intent is active for each sample in the batch
        # intent is one-hot encoded: [straight, left, right, stop]
        intent_indices = torch.argmax(intent, dim=1) # E.g., [0, 1, 0, 2, 3, ...] for batch_size samples

        # Initialize an output tensor of zeros
        batch_size = features.shape[0]
        output_velocities = torch.zeros(batch_size, 2, device=img.device)

        # Create masks for each intent (assuming order: Straight=0, Left=1, Right=2, Stop=3)
        is_straight = (intent_indices == 0)
        is_left = (intent_indices == 1)
        is_right = (intent_indices == 2)

        # Apply the corresponding head based on intent for each sample
        if is_straight.any():
            output_velocities = self.straight_head(features[is_straight])
        if is_left.any():
            output_velocities = self.left_head(features[is_left])
        if is_right.any():
            output_velocities = self.right_head(features[is_right])
        # 'Stop' intent samples (where intent_indices == 3) will remain zeros as initialized.

        return output_velocities
