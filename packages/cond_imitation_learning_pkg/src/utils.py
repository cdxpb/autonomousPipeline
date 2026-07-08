import torchvision.transforms as transforms

# Preprocessing Constants
CROP_TOP_ROWS = 175
INPUT_SHAPE = (112, 224) # Height, Width

# Intent mapping for one-hot encoding
INTENT_MAP = {
    "straight":       [1.0, 0.0, 0.0, 0.0],
    "left":           [0.0, 1.0, 0.0, 0.0],
    "right":          [0.0, 0.0, 1.0, 0.0],
    "stop":           [0.0, 0.0, 0.0, 1.0],
    "lane_following": [0.0, 0.0, 0.0, 1.0]
}

def crop_image(pil_image):
    width, height = pil_image.size
    return pil_image.crop((0, CROP_TOP_ROWS, width, height))

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(INPUT_SHAPE),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
    ])

def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize(INPUT_SHAPE),
        transforms.ToTensor(),
    ])

import torch.nn as nn
class SafePool(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(size)
    def forward(self, x):
        if x.device.type == 'mps':
            return self.pool(x.cpu()).to(x.device)
        return self.pool(x)