import torch
import torch.nn as nn
from utils import SafePool




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




class ConditionalPilotNetFiLM(nn.Module):
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

