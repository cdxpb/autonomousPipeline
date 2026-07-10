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
       self.lane_following_head = nn.Sequential(
           nn.Linear(self.flattened_size, 50),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(50, 10),
           nn.ReLU(),
           nn.Linear(10, 2) # Output 2 velocities as clarified
       )


   def forward(self, img, intent):
       # img is now expected to be the processed segmentation mask (1-channel tensor)
       features = self.feature_extractor(img)
       features = self.pool(features)
       features = features.reshape(-1, self.flattened_size) # [batch_size, flattened_size]

       # one-hot order: straight=0, left=1, right=2, lane_following=3
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

