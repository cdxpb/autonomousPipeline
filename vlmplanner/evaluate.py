# evaluates a trained classifier head, the small-CNN baseline, or zero/few-shot VLM
# prompting against the same annotated val split -- prints per-task metrics, no report
#
# python vlmplanner/evaluate.py head --checkpoint dataset/checkpoints/head_500m/checkpoint.pt \
#     --images "dataset/dataset/cil_dataset_20260622-125203/images" --labels dataset/annotations.json
# python vlmplanner/evaluate.py cnn --checkpoint dataset/checkpoints/head_cnn_baseline/checkpoint.pt \
#     --images "dataset/dataset/cil_dataset_20260622-125203/images" --labels dataset/annotations.json
# python vlmplanner/evaluate.py prompting \
#     --images "dataset/dataset/cil_dataset_20260622-125203/images" --labels dataset/annotations.json \
#     --val-from dataset/checkpoints/head_500m_spatial3/checkpoint.pt

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from classifier_common import (
    TASKS, AUG_SPECS_VAL, load_usable_labels, label_to_targets, manifest_for,
    get_device, VisionFeatureExtractor, MultiTaskHead, binary_metrics,
)
from segmentation import SegmentationPreprocessor, crop_image

DIRECTIONS = ("left", "straight", "right")


def print_metrics(name: str, per_task: dict, macro_f1: bool = False):
    print(f"\n{name}:")
    for t in TASKS:
        m = per_task[t]
        if m["n"] == 0:
            print(f"  {t:14s}  (no val samples)")
        else:
            print(f"  {t:14s}  n={m['n']:4d}  acc={m['acc']:.3f}  "
                  f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")
    if macro_f1:
        f1s = [m["f1"] for m in per_task.values() if m["f1"] is not None]
        print(f"  macro F1: {np.mean(f1s):.4f}")


def threshold_sweep(Y: np.ndarray, probs: np.ndarray, M: np.ndarray):
    print("\nThreshold sweep (same val set -- a cheap look at whether moving the cutoff\n"
          "buys precision, NOT a tuned/deployable threshold, since it's read off this same val set):")
    for i, t in enumerate(TASKS):
        if M[:, i].sum() == 0:
            continue
        row = []
        for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            m = binary_metrics(Y[:, i], probs[:, i], M[:, i], threshold=th)
            row.append(f"{th:.1f}:P{m['precision']:.2f}/R{m['recall']:.2f}/F{m['f1']:.2f}")
        print(f"  {t:14s}  " + "  ".join(row))


def eval_head(args):
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    images_dir = Path(args.images)
    labels = load_usable_labels(Path(args.labels))
    val_frames = [f for f in ckpt["val_frames"] if f in labels]
    print(f"Val frames: {len(val_frames)}")

    device = get_device()
    extractor = VisionFeatureExtractor(ckpt["model_id"], device=device,
                                        spatial_split=ckpt.get("spatial_split", 1))
    seg = SegmentationPreprocessor(ckpt["yolo_model"], device=device.type)
    model = MultiTaskHead(ckpt["embed_dim"], ckpt["hidden"], ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    input_mode = ckpt.get("input_mode", "cropseg")
    manifest = manifest_for(val_frames, labels, AUG_SPECS_VAL)  # AUG_SPECS_VAL == ["orig"]
    images, targets_list, masks_list = [], [], []
    for frame, _ in manifest:
        img = Image.open(images_dir / frame).convert("RGB")
        images.append(crop_image(img) if input_mode == "crop" else seg.process(img))
        t, m = label_to_targets(labels[frame])
        targets_list.append(t)
        masks_list.append(m)

    embeddings = np.concatenate(
        [extractor.extract_batch(images[i:i + 16]) for i in range(0, len(images), 16)])
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(embeddings).to(device))).cpu().numpy()

    Y, M = np.stack(targets_list), np.stack(masks_list)
    per_task = {t: binary_metrics(Y[:, i], probs[:, i], M[:, i]) for i, t in enumerate(TASKS)}
    print_metrics(f"Classifier head ({ckpt_path})", per_task)
    threshold_sweep(Y, probs, M)


def eval_cnn(args):
    from cnn_baseline import CNNClassifier

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    images_dir = Path(args.images)
    labels = load_usable_labels(Path(args.labels))
    val_frames = [f for f in ckpt["val_frames"] if f in labels]
    print(f"Val frames: {len(val_frames)}")

    # see train_cnn_baseline.py: MPS backward is unreliable for this model, stick to cpu
    seg = SegmentationPreprocessor(ckpt["yolo_model"], device="cpu")
    model = CNNClassifier(hidden=ckpt["hidden"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    manifest = manifest_for(val_frames, labels, AUG_SPECS_VAL)  # AUG_SPECS_VAL == ["orig"]
    masks, targets_list, masks_list = [], [], []
    for frame, _ in manifest:
        img = Image.open(images_dir / frame).convert("RGB")
        masks.append(seg.segment(img))
        t, m = label_to_targets(labels[frame])
        targets_list.append(t)
        masks_list.append(m)

    X = np.stack(masks).astype(np.float32)[:, None, :, :] / 255.0
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X))).cpu().numpy()

    Y, M = np.stack(targets_list), np.stack(masks_list)
    per_task = {t: binary_metrics(Y[:, i], probs[:, i], M[:, i]) for i, t in enumerate(TASKS)}
    print_metrics(f"CNN baseline ({ckpt['n_params']} params, {ckpt_path})", per_task)


def pick_exemplars(labels: dict, train_frames: list, images_dir: Path, input_mode: str = "raw"):
    approach_yes = approach_pass = approach_no = exit_yes = exit_no = None
    for f in train_frames:
        lbl = labels[f]
        if lbl["type"] == "approach":
            if lbl["junction"] == "no" and approach_no is None:
                approach_no = f
            elif lbl["junction"] == "yes":
                dirs = lbl["directions"]
                if approach_yes is None and any(dirs.values()):
                    approach_yes = (f, next(d for d in DIRECTIONS if dirs[d]))
                if approach_pass is None and not all(dirs.values()):
                    approach_pass = (f, next(d for d in DIRECTIONS if not dirs[d]))
        elif lbl["type"] == "exit":
            if lbl["cleared"] and exit_yes is None:
                exit_yes = f
            elif not lbl["cleared"] and exit_no is None:
                exit_no = f
        if approach_yes and approach_pass and approach_no and exit_yes and exit_no:
            break

    def load(f):
        img = Image.open(images_dir / f).convert("RGB")
        return crop_image(img) if input_mode == "crop" else img

    approach_exemplars = [
        (load(approach_yes[0]), approach_yes[1], "yes"),
        (load(approach_pass[0]), approach_pass[1], "pass"),
        (load(approach_no), "straight", "no"),
    ]
    exit_exemplars = [(load(exit_yes), "yes"), (load(exit_no), "no")]
    print(f"Exemplars: approach yes={approach_yes} pass={approach_pass} no={approach_no}  "
          f"exit yes={exit_yes} no={exit_no}")
    return approach_exemplars, exit_exemplars


def evaluate_oracle(oracle, val_frames: list, labels: dict, images_dir: Path, input_mode: str = "raw"):
    Y, P, M = [], [], []
    for f in val_frames:
        lbl = labels[f]
        t, m = label_to_targets(lbl)
        p = np.zeros(len(TASKS), dtype=np.float32)
        img = Image.open(images_dir / f).convert("RGB")
        if input_mode == "crop":
            img = crop_image(img)

        if lbl["type"] == "approach":
            answers = {d: oracle.classify_approach(img, direction=d)[0] for d in DIRECTIONS}
            p[0] = 1.0 if any(a in ("yes", "pass") for a in answers.values()) else 0.0
            for i, d in enumerate(DIRECTIONS, start=1):
                p[i] = 1.0 if answers[d] == "yes" else 0.0
        else:
            ans, _ = oracle.classify_exit(img)
            p[4] = 1.0 if ans == "yes" else 0.0

        Y.append(t)
        P.append(p)
        M.append(m)

    Y, P, M = np.stack(Y), np.stack(P), np.stack(M)
    return {t: binary_metrics(Y[:, i], P[:, i], M[:, i]) for i, t in enumerate(TASKS)}


def eval_prompting(args):
    from prompting_oracle import Config, ZeroShotOracle, FewShotOracle

    images_dir = Path(args.images)
    ckpt = torch.load(args.val_from, map_location="cpu", weights_only=False)
    labels = load_usable_labels(Path(args.labels))
    val_frames = [f for f in ckpt["val_frames"] if f in labels]
    train_frames = [f for f in ckpt["train_frames"] if f in labels]
    print(f"Val frames: {len(val_frames)} (from {args.val_from})")

    cfg = Config(model_id=args.model_id, precision=args.precision)

    if args.mode in ("zeroshot", "both"):
        oracle = ZeroShotOracle(cfg)
        per_task = evaluate_oracle(oracle, val_frames, labels, images_dir, args.input_mode)
        print_metrics(f"Zero-shot ({args.model_id}, {args.input_mode})", per_task, macro_f1=True)
        del oracle

    if args.mode in ("fewshot", "both"):
        approach_ex, exit_ex = pick_exemplars(labels, train_frames, images_dir, args.input_mode)
        oracle = FewShotOracle(cfg, approach_ex, exit_ex)
        per_task = evaluate_oracle(oracle, val_frames, labels, images_dir, args.input_mode)
        print_metrics(f"Few-shot ({args.model_id}, {args.input_mode})", per_task, macro_f1=True)


def main():
    p = argparse.ArgumentParser(description="Evaluate a classifier head, CNN baseline, or VLM prompting oracle")
    sub = p.add_subparsers(dest="cmd", required=True)

    head = sub.add_parser("head", help="trained classifier head")
    head.add_argument("--checkpoint", required=True)
    head.add_argument("--images", required=True)
    head.add_argument("--labels", default="dataset/annotations.json")
    head.set_defaults(func=eval_head)

    cnn = sub.add_parser("cnn", help="small-CNN comparison baseline")
    cnn.add_argument("--checkpoint", required=True)
    cnn.add_argument("--images", required=True)
    cnn.add_argument("--labels", default="dataset/annotations.json")
    cnn.set_defaults(func=eval_cnn)

    prompting = sub.add_parser("prompting", help="zero/few-shot SmolVLM2 prompting, no training")
    prompting.add_argument("--images", required=True)
    prompting.add_argument("--labels", default="dataset/annotations.json")
    prompting.add_argument("--val-from", default="dataset/checkpoints/head_500m_spatial3/checkpoint.pt",
                            help="reuse this checkpoint's exact val/train split for a fair comparison")
    prompting.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Instruct")
    prompting.add_argument("--precision", default="fp32", choices=["fp16", "fp32", "int8", "int4"])
    prompting.add_argument("--mode", default="both", choices=["zeroshot", "fewshot", "both"])
    prompting.add_argument("--input-mode", default="raw", choices=["raw", "crop"],
                            help="raw = full uncropped camera image (what the prompts were written "
                                 "for), crop = top-rows-off crop (same crop the trained classifier uses)")
    prompting.set_defaults(func=eval_prompting)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
