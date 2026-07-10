try:
    import torchvision.transforms as transforms
except ImportError:
    transforms = None

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

try:
    import torch
    import torch.nn as nn

    class SafePool(nn.Module):
        # matches nn.AdaptiveAvgPool2d(size) numerically but traces to plain slicing+mean,
        # so it works around AdaptiveAvgPool2d being buggy on MPS and torch.onnx.export
        # refusing sizes that don't divide evenly (e.g. 7x21 -> 5x10, used by PilotNet heads)
        def __init__(self, size):
            super().__init__()
            self.output_size = size

        def forward(self, x):
            oh, ow = self.output_size
            h, w = x.shape[-2:]
            rows = []
            for i in range(oh):
                hs, he = (i * h) // oh, -(-((i + 1) * h) // oh)
                cols = []
                for j in range(ow):
                    ws, we = (j * w) // ow, -(-((j + 1) * w) // ow)
                    cols.append(x[..., hs:he, ws:we].mean(dim=(-2, -1), keepdim=True))
                rows.append(torch.cat(cols, dim=-1))
            return torch.cat(rows, dim=-2)
except ImportError:
    pass