# same interface as vlm_planner.VLMOracle (classify_approach/classify_exit) but backed by
# a trained checkpoint. image is expected already cropped+segmented+colorized, e.g.
# autonomous_node.py's ui_model_view, reused instead of re-running yolo

import torch
from PIL import Image

from classifier_common import VisionFeatureExtractor, MultiTaskHead, get_device

JUNCTION, LEFT_OPEN, STRAIGHT_OPEN, RIGHT_OPEN, EXIT_CLEARED = range(5)
DIRECTION_TASK = {"left": LEFT_OPEN, "straight": STRAIGHT_OPEN, "right": RIGHT_OPEN}


class TrainedOracle:
    def __init__(self, checkpoint_path: str, device=None):
        self.device = device or get_device()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.extractor = VisionFeatureExtractor(ckpt["model_id"], device=self.device,
                                                 spatial_split=ckpt.get("spatial_split", 1))
        self.model = MultiTaskHead(ckpt["embed_dim"], ckpt["hidden"], ckpt["dropout"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.model_id = ckpt["model_id"]

    def _probs(self, image: Image.Image):
        emb = self.extractor.extract_batch([image])
        with torch.no_grad():
            logits = self.model(torch.from_numpy(emb).to(self.device))
            return torch.sigmoid(logits)[0].cpu().numpy()

    def classify_approach(self, image: Image.Image, direction: str = "straight"):
        probs = self._probs(image)
        junction_p = probs[JUNCTION]
        if junction_p < 0.5:
            return "no", float(1 - junction_p)
        dir_p = probs[DIRECTION_TASK.get(direction, STRAIGHT_OPEN)]
        return ("yes", float(dir_p)) if dir_p >= 0.5 else ("pass", float(1 - dir_p))

    def classify_exit(self, image: Image.Image):
        cleared_p = self._probs(image)[EXIT_CLEARED]
        return ("yes", float(cleared_p)) if cleared_p >= 0.5 else ("no", float(1 - cleared_p))
