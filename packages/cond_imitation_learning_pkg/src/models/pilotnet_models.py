import torch
import torch.nn as nn
from utils import SafePool

class PilotNetBasic(nn.Module):
    def __init__(self, in_channels=3):
        super(PilotNetBasic, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        
        self.pool = SafePool((5, 10))
        self.flattened_size = 64 * 5 * 10  
        self.num_intents = 4 # [Straight, Left, Right, Stop]

        self.regressor = nn.Sequential(
            nn.Linear(self.flattened_size + self.num_intents, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2)  
        )

    def forward(self, img, intent):
        # Image path
        x = self.feature_extractor(img)
        x = self.pool(x)
        x = x.reshape(-1, self.flattened_size)
        
        # Concatenate Image Features with the Intent Vector
        x = torch.cat((x, intent), dim=1)
        
        # Regress to wheels
        x = self.regressor(x)
        return x
class PilotNetV3(nn.Module):
    def __init__(self):
        super(PilotNetV3, self).__init__()

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


class PilotNetClassNHead(nn.Module):
   def __init__(self):
       super(PilotNetClassNHead, self).__init__()


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
           nn.Linear(10, 3) # Output 3 logits for classification
       )
       self.left_head = nn.Sequential(
           nn.Linear(self.flattened_size, 50),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(50, 10),
           nn.ReLU(),
           nn.Linear(10, 3) # Output 3 logits for classification
       )
       self.right_head = nn.Sequential(
           nn.Linear(self.flattened_size, 50),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(50, 10),
           nn.ReLU(),
           nn.Linear(10, 3) # Output 3 logits for classification
       )
       self.lane_following_head = nn.Sequential(
           nn.Linear(self.flattened_size, 50),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(50, 10),
           nn.ReLU(),
           nn.Linear(10, 3) # Output 3 logits for classification
       )


   def forward(self, img, intent):
       # img is now expected to be the processed segmentation mask (1-channel tensor)
       features = self.feature_extractor(img)
       features = self.pool(features)
       features = features.reshape(-1, self.flattened_size) # [batch_size, flattened_size]


       # Determine which intent is active for each sample in the batch
       # intent is one-hot encoded: [straight, left, right, lane_following]
       intent_indices = torch.argmax(intent, dim=1) # E.g., [0, 1, 0, 2, 3, ...] for batch_size samples


       # Initialize an output tensor of zeros
       batch_size = features.shape[0]
       output_velocities = torch.zeros(batch_size, 3, device=img.device) # Changed from 2 to 3


       # Create masks for each intent (assuming order: Straight=0, Left=1, Right=2, Lane_following=3)
       is_straight = (intent_indices == 0)
       is_left = (intent_indices == 1)
       is_right = (intent_indices == 2)
       is_lane_following = (intent_indices == 3)


       # Apply the corresponding head based on intent for each sample
       if is_straight.any():
           output_velocities[is_straight] = self.straight_head(features[is_straight])
       if is_left.any():
           output_velocities[is_left] = self.left_head(features[is_left])
       if is_right.any():
           output_velocities[is_right] = self.right_head(features[is_right])
       if is_lane_following.any(): # Add this block for lane_following
           output_velocities[is_lane_following] = self.lane_following_head(features[is_lane_following])


       return output_velocities


class PilotNetRegNHead(nn.Module):
    def __init__(self):
        super(PilotNetRegNHead, self).__init__()

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


class PilotNetRegNHeadV2(nn.Module):
   def __init__(self):
       super(PilotNetRegNHeadV2, self).__init__()


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



class PilotNetRegSmaller(nn.Module):
    """Approach 5 retrained against the smaller (256x320) YOLO backbone.
    Architecture matches models/smaller_img/reg4heads_smaller.ipynb exactly
    (48-channel late conv layers, unlike pilotnet_regNheadv2.py's 64-channel
    version, these are NOT interchangeable checkpoints)."""

    def __init__(self):
        super(PilotNetRegSmaller, self).__init__()

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

class PilotNetRegNCILTemporal(nn.Module):
    def __init__(self, num_frames=3): # Added num_frames parameter
        super(PilotNetRegNCILTemporal, self).__init__()

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



class FiLM(nn.Module):
   """
   Feature-wise Linear Modulation.


   Predicts a per-channel gamma/beta pair from a conditioning embedding and
   applies gamma * x + beta. Works for conv feature maps (B, C, H, W) -
   broadcasting gamma/beta over spatial dims - and for flat vectors (B, C).
   """
   def __init__(self, embedding_dim: int, num_features: int):
       super().__init__()
       self.gamma_fc = nn.Linear(embedding_dim, num_features)
       self.beta_fc = nn.Linear(embedding_dim, num_features)
       # Identity-ish init (gamma=1, beta=0) so FiLM starts as a no-op and
       # doesn't destabilize training before the conditioning is learned -
       # this matters more here since we stack several FiLM layers.
       nn.init.zeros_(self.gamma_fc.weight)
       nn.init.ones_(self.gamma_fc.bias)
       nn.init.zeros_(self.beta_fc.weight)
       nn.init.zeros_(self.beta_fc.bias)


   def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
       gamma = self.gamma_fc(cond)
       beta = self.beta_fc(cond)
       if x.dim() == 4:
           gamma = gamma.unsqueeze(-1).unsqueeze(-1)
           beta = beta.unsqueeze(-1).unsqueeze(-1)
       return gamma * x + beta




class PilotNetFiLM(nn.Module):
   def __init__(self, intent_dim: int = 4, cond_embedding_dim: int = 64):
       super().__init__()


       # --- Shared intent encoder ---
       # One small embedding shared by every FiLM layer downstream, each FiLM layer
       # below has its own lightweight linear head off this shared embedding.
       self.intent_encoder = nn.Sequential(
           nn.Linear(intent_dim, cond_embedding_dim),
           nn.ReLU(),
       )


       # --- Early conv layers: unconditioned ---
       self.conv1 = nn.Sequential(
           nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(),
           nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(),
       )


       # --- Deeper conv layers: FiLM-conditioned ---
       self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
       self.bn3 = nn.BatchNorm2d(48)
       self.film3 = FiLM(cond_embedding_dim, 48)


       self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
       self.bn4 = nn.BatchNorm2d(64)
       self.film4 = FiLM(cond_embedding_dim, 64)


       self.conv5 = nn.Conv2d(64, 48, kernel_size=3, stride=1)
       self.bn5 = nn.BatchNorm2d(48)
       self.film5 = FiLM(cond_embedding_dim, 48)


       self.act = nn.ReLU()


       # --- Dimensionality reduction ---
       self.pool = SafePool((4, 7))
       self.flattened_size = 48 * 4 * 7  # 1344


       # --- Bottleneck: explicit compression + FiLM on a "linear layer" ---
       self.bottleneck_dim = 128
       self.bottleneck_fc = nn.Linear(self.flattened_size, self.bottleneck_dim)
       self.film_bottleneck = FiLM(cond_embedding_dim, self.bottleneck_dim)


       # --- Regression head ---
       self.head = nn.Sequential(
           nn.Linear(self.bottleneck_dim, 50),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(50, 10),
           nn.ReLU(),
           nn.Linear(10, 2),
       )


   def forward(self, img: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
       cond = self.intent_encoder(intent)  # (B, cond_embedding_dim), shared by all FiLM layers below


       x = self.conv1(img)


       x = self.conv3(x)
       x = self.bn3(x)
       x = self.film3(x, cond)
       x = self.act(x)


       x = self.conv4(x)
       x = self.bn4(x)
       x = self.film4(x, cond)
       x = self.act(x)


       x = self.conv5(x)
       x = self.bn5(x)
       x = self.film5(x, cond)
       x = self.act(x)


       x = self.pool(x)
       x = x.reshape(x.size(0), -1)


       x = self.bottleneck_fc(x)
       x = self.film_bottleneck(x, cond)
       x = self.act(x)


       output_velocities = self.head(x)
       return output_velocities

