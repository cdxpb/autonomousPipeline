# crop + yolo segment, same as autonomous_node.py --approach 5.

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                        / "packages" / "cond_imitation_learning_pkg" / "src"))
from utils import crop_image  # noqa: E402

MASK_RESIZE = (224, 112)  # (width, height)
NUM_CLASSES = 4  # yellow line, white line, red line, background
BACKGROUND_CLASS = 3


class SegmentationPreprocessor:
    def __init__(self, model_path: str = "models/yolo_model/yolo_model.pt", device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="semantic")
        self.device = device

    def segment(self, pil_image: Image.Image) -> np.ndarray:
        cropped = crop_image(pil_image)
        results = self.model(cropped, device=self.device, verbose=False)
        mask = results[0].semantic_mask.data.cpu().numpy().astype(np.uint8)
        return np.array(Image.fromarray(mask).resize(MASK_RESIZE, Image.NEAREST))

    def colorize(self, mask: np.ndarray) -> Image.Image:
        mask_gray = (mask * (255 // (NUM_CLASSES - 1))).astype(np.uint8)
        colored = cv2.applyColorMap(mask_gray, cv2.COLORMAP_JET)
        return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))

    def process(self, pil_image: Image.Image) -> Image.Image:
        return self.colorize(self.segment(pil_image))


def occlude_side(mask: np.ndarray, side: str, frac: float = 0.45) -> np.ndarray:
    # unused experiment
    out = mask.copy()
    w = out.shape[1]
    if side == "left":
        out[:, :int(w * frac)] = BACKGROUND_CLASS
    elif side == "right":
        out[:, int(w * (1 - frac)):] = BACKGROUND_CLASS
    return out
