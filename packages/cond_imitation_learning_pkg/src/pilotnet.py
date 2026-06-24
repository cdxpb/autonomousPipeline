import torch
import torch.nn as nn

class ConditionalPilotNet(nn.Module):
    def __init__(self, in_channels=3):
        super(ConditionalPilotNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d((5, 10))
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