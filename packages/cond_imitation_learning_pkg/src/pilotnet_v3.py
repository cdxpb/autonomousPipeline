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
        self.flattened_size = 64 * 5 * 10

        # Define a smaller head architecture for each intent
        # Output is 4 neurons for classification (actions 0, 1, 2, 3)
        def create_smaller_head():
            return nn.Sequential(
                nn.Linear(self.flattened_size, 50),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(50, 10),
                nn.ReLU(),
                nn.Linear(10, 4) # Output 4 classification neurons
            )

        self.straight_head = create_smaller_head()
        self.left_head = create_smaller_head()
        self.right_head = create_smaller_head()
        self.default_head = create_smaller_head() # For 'stop' intent or other cases

    def forward(self, img, intent):
        # img is now expected to be the processed segmentation mask (1-channel tensor)
        x = self.feature_extractor(img)
        x = self.pool(x)
        x = x.reshape(-1, self.flattened_size)

        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, 4, device=x.device)

        # Get boolean masks for each intent based on the one-hot encoding
        straight_mask = (intent[:, 0] == 1)
        left_mask = (intent[:, 1] == 1)
        right_mask = (intent[:, 2] == 1)
        stop_mask = (intent[:, 3] == 1) # Assuming index 3 for 'stop'

        # Apply corresponding head to the relevant part of the batch
        if straight_mask.any():
            outputs[straight_mask] = self.straight_head(x[straight_mask])
        if left_mask.any():
            outputs[left_mask] = self.left_head(x[left_mask])
        if right_mask.any():
            outputs[right_mask] = self.right_head(x[right_mask])
        if stop_mask.any(): # Handle 'stop' intent with the default head
            outputs[stop_mask] = self.default_head(x[stop_mask])

        return outputs
