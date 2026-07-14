# shared code for training/evaluating the classifier head

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps

TASKS = ["junction", "left_open", "straight_open", "right_open", "exit_cleared"]
DIRECTIONS = ("left", "straight", "right")
FLIP_SWAP = {"left": "right", "right": "left", "straight": "straight"}

AUG_SPECS_TRAIN = ["orig", "hflip", "jitter", "hflip_jitter"]
AUG_SPECS_VAL = ["orig"]

# extra copies for minority-class frames
EXTRA_AUG_SPECS_MINORITY = ["jitter2", "hflip_jitter2", "jitter3", "hflip_jitter3"]


# --- labels ---

def load_usable_labels(labels_path: Path) -> Dict[str, dict]:
    """Drop 'skip' entries (and anything malformed); keep approach/exit only."""
    raw = json.loads(Path(labels_path).read_text())
    return {f: l for f, l in raw.items() if l.get("type") in ("approach", "exit")}


def label_to_targets(label: dict) -> Tuple[np.ndarray, np.ndarray]:
    """-> (targets, mask), both float32 of length len(TASKS). mask[i]=1 iff
    this frame has ground truth for that task."""
    t = np.zeros(len(TASKS), dtype=np.float32)
    m = np.zeros(len(TASKS), dtype=np.float32)
    if label["type"] == "approach":
        t[0] = 1.0 if label["junction"] == "yes" else 0.0
        m[0] = 1.0
        if label["junction"] == "yes":
            for i, d in enumerate(DIRECTIONS, start=1):
                t[i] = 1.0 if label["directions"][d] else 0.0
                m[i] = 1.0
    elif label["type"] == "exit":
        t[4] = 1.0 if label["cleared"] else 0.0
        m[4] = 1.0
    return t, m


def flip_label(label: dict) -> dict:
    """Mirror a label to match a horizontally-flipped image."""
    l = json.loads(json.dumps(label))
    if l["type"] == "approach" and l["junction"] == "yes":
        d = l["directions"]
        l["directions"] = {"left": d["right"], "straight": d["straight"], "right": d["left"]}
    elif l["type"] == "exit":
        l["turn"] = FLIP_SWAP[l["turn"]]
    return l


# --- episode-aware train/val split ---

TIMESTAMP_RE = re.compile(r"^(\d+\.\d+)")


def frame_timestamp(frame_name: str) -> Optional[float]:
    m = TIMESTAMP_RE.match(frame_name)
    return float(m.group(1)) if m else None


def build_episodes(frames: List[str], gap_seconds: float = 2.0) -> List[List[str]]:
    ts = [(f, frame_timestamp(f)) for f in frames]
    if any(t is None for _, t in ts):
        return [[f] for f in frames]
    ts.sort(key=lambda x: x[1])
    episodes = [[ts[0][0]]]
    for (f, t), (_, t_prev) in zip(ts[1:], ts[:-1]):
        if t - t_prev > gap_seconds:
            episodes.append([])
        episodes[-1].append(f)
    return episodes


def split_frames(frames: List[str], val_frac: float, gap_seconds: float, seed: int
                  ) -> Tuple[List[str], List[str]]:
    episodes = build_episodes(frames, gap_seconds)
    rng = random.Random(seed)
    order = list(range(len(episodes)))
    rng.shuffle(order)
    target_val = round(val_frac * len(frames))

    val_frames: List[str] = []
    val_ids = set()
    for i in order:
        if len(val_frames) >= target_val:
            break
        val_frames.extend(episodes[i])
        val_ids.add(i)

    train_frames = [f for i, ep in enumerate(episodes) if i not in val_ids for f in ep]
    return train_frames, val_frames


# --- augmentation ---

def augment(image: Image.Image, label: dict, spec: str, rng: random.Random
            ) -> Tuple[Image.Image, dict]:
    img, lbl = image, label
    if "hflip" in spec:
        img = ImageOps.mirror(img)
        lbl = flip_label(lbl)
    if "jitter" in spec:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
        angle = rng.uniform(-6, 6)  # small: enough to help, not enough to flip left/right semantics
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return img, lbl


def manifest_for(frames: List[str], labels: Dict[str, dict], specs: List[str]
                  ) -> List[Tuple[str, str]]:
    """List of (frame, spec) keys to featurize, e.g. [("0001.jpg","orig"), ("0001.jpg","hflip"), ...]."""
    return [(f, spec) for f in frames for spec in specs]


def minority_classes(labels: Dict[str, dict]) -> Dict[str, bool]:
    """Per task, which boolean value is the minority class -> {task: is_positive_the_minority}."""
    pos = {t: 0 for t in TASKS}
    neg = {t: 0 for t in TASKS}
    for label in labels.values():
        t, m = label_to_targets(label)
        for i, task in enumerate(TASKS):
            if m[i] > 0.5:
                (pos if t[i] > 0.5 else neg)[task] += 1
    return {task: pos[task] < neg[task] for task in TASKS if pos[task] > 0 and neg[task] > 0}


def frame_is_minority(label: dict, minority: Dict[str, bool]) -> bool:
    t, m = label_to_targets(label)
    return any(m[i] > 0.5 and task in minority and (t[i] > 0.5) == minority[task]
               for i, task in enumerate(TASKS))


def manifest_for_train(frames: List[str], labels: Dict[str, dict], minority: Dict[str, bool]
                        ) -> List[Tuple[str, str]]:
    """Oversample minority-class frames with extra augmented copies on top of AUG_SPECS_TRAIN."""
    out = []
    for f in frames:
        specs = list(AUG_SPECS_TRAIN)
        if frame_is_minority(labels[f], minority):
            specs += EXTRA_AUG_SPECS_MINORITY
        out += [(f, spec) for spec in specs]
    return out


def episode_ids(frames: List[str], gap_seconds: float) -> Dict[str, int]:
    return {f: i for i, ep in enumerate(build_episodes(frames, gap_seconds)) for f in ep}


def print_class_diagnostics(labels: Dict[str, dict], frames: List[str], gap_seconds: float):
    """episodes per class, not frames"""
    ep_ids = episode_ids(frames, gap_seconds)
    for task_i, task in enumerate(TASKS):
        pos_eps, neg_eps = set(), set()
        for f in frames:
            t, m = label_to_targets(labels[f])
            if m[task_i] > 0.5:
                (pos_eps if t[task_i] > 0.5 else neg_eps).add(ep_ids[f])
        if pos_eps or neg_eps:
            print(f"  {task:14s}  positive: {len(pos_eps):2d} episodes   negative: {len(neg_eps):2d} episodes")


# --- frozen SmolVLM2 vision features ---

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class VisionFeatureExtractor:
    """frozen smolvlm2 vision encoder. spatial_split=1 mean-pools all patches,
    spatial_split=3 pools left/center/right thirds separately"""

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Instruct", device=None,
                 spatial_split: int = 1):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.device = device or get_device()
        self.spatial_split = spatial_split
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float32)
        self.model = self.model.to(self.device).eval()
        vc = self.model.config.vision_config
        self.grid_size = vc.image_size // vc.patch_size
        self.embed_dim = vc.hidden_size * spatial_split

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor.image_processor(images, do_image_splitting=False, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        b, num_images, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * num_images, c, h, w)
        with torch.no_grad():
            # Call the vision transformer directly instead of model.get_image_features():
            # that method's return contract changed across transformers releases (some
            # versions apply the pixel-shuffle connector and return a plain downsampled
            # tensor, others return the raw pre-connector last_hidden_state), which silently
            # fed the wrong patch grid/dim into this head depending on which env was running.
            # Calling vision_model directly always gives the raw grid_size x grid_size,
            # vision-hidden_size patches this classifier head was trained on.
            hidden = self.model.model.vision_model(pixel_values=pixel_values).last_hidden_state

            if self.spatial_split == 1:
                emb = hidden.mean(dim=1)
            else:
                b, n, d = hidden.shape
                g = self.grid_size
                grid = hidden.view(b, g, g, d)  # (batch, row, col, dim)
                col_bounds = [round(i * g / self.spatial_split) for i in range(self.spatial_split + 1)]
                parts = [grid[:, :, col_bounds[i]:col_bounds[i + 1], :].reshape(b, -1, d).mean(dim=1)
                         for i in range(self.spatial_split)]
                emb = torch.cat(parts, dim=-1)
        return emb.float().cpu().numpy()


# --- metrics ---

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> dict:
    """Precision/recall/F1/accuracy + confusion counts over masked-in entries only."""
    sel = mask > 0.5
    n = int(sel.sum())
    if n == 0:
        return dict(n=0, acc=None, precision=None, recall=None, f1=None, tp=0, fp=0, fn=0, tn=0)
    yt = y_true[sel] > 0.5
    yp = y_prob[sel] >= threshold
    tp = int(np.sum(yt & yp)); fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp)); tn = int(np.sum(~yt & ~yp))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(n=n, acc=(tp + tn) / n, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


# --- classification head ---

class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(TASKS)),
        )

    def forward(self, x):
        return self.net(x)
