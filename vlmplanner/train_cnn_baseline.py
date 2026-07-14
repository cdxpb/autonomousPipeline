# trains the small-CNN comparison (cnn_baseline.py) on the raw crop+segment mask,
# no SmolVLM2. same labels/split/augmentation/oversampling as train_classifier.py.
#
# python vlmplanner/train_cnn_baseline.py \
#     --images "dataset/dataset/cil_dataset_20260622-125203/images" \
#     --labels dataset/annotations.json \
#     --out dataset/checkpoints/head_cnn_baseline

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from classifier_common import (
    TASKS, AUG_SPECS_VAL,
    load_usable_labels, label_to_targets, flip_label, augment, manifest_for,
    frame_timestamp, split_frames, get_device,
    minority_classes, manifest_for_train, print_class_diagnostics, binary_metrics,
)
from segmentation import SegmentationPreprocessor
from cnn_baseline import CNNClassifier


def load_mask_cache(path: Path, cache_key: str) -> dict:
    if not path.exists():
        return {}
    npz = np.load(path, allow_pickle=True)
    if str(npz["cache_key"]) != cache_key:
        print(f"cache at {path} was built with different preprocessing, ignoring it")
        return {}
    return {k: v for k, v in zip(npz["keys"], npz["masks"])}


def save_mask_cache(path: Path, cache: dict, cache_key: str):
    keys = list(cache.keys())
    masks = np.stack([cache[k] for k in keys])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.array(keys, dtype=object), masks=masks, cache_key=cache_key)


def ensure_masks(manifest, images_dir: Path, labels: dict, cache: dict, seg: SegmentationPreprocessor) -> dict:
    missing = [(f, spec) for f, spec in manifest if f"{f}|{spec}" not in cache]
    if not missing:
        return cache
    print(f"Segmenting {len(missing)} new (frame, augmentation) pairs...")
    t0 = time.time()
    for i, (frame, spec) in enumerate(missing):
        img = Image.open(images_dir / frame).convert("RGB")
        rng_seed = abs(hash(f"{frame}:{spec}")) % (2**32)
        img, _ = augment(img, labels[frame], spec, random.Random(rng_seed))
        cache[f"{frame}|{spec}"] = seg.segment(img)
        if i % 200 == 0:
            print(f"  {i}/{len(missing)}  ({time.time() - t0:.0f}s)")
    print(f"Done segmenting in {time.time() - t0:.0f}s")
    return cache


def build_xym(manifest, cache: dict, labels: dict):
    X, Y, M = [], [], []
    for frame, spec in manifest:
        X.append(cache[f"{frame}|{spec}"])
        lbl = labels[frame]
        if "hflip" in spec:
            lbl = flip_label(lbl)
        t, m = label_to_targets(lbl)
        Y.append(t)
        M.append(m)
    X = np.stack(X).astype(np.float32) / 255.0
    return X[:, None, :, :], np.stack(Y), np.stack(M)


def compute_pos_weight(Y: np.ndarray, M: np.ndarray, cap: float = 10.0) -> np.ndarray:
    w = np.ones(len(TASKS), dtype=np.float32)
    for i in range(len(TASKS)):
        sel = M[:, i] > 0.5
        pos = Y[sel, i].sum()
        neg = sel.sum() - pos
        if pos > 0:
            w[i] = np.clip(neg / pos, 0.1, cap)
    return w


def evaluate(model, X, Y, M, device, batch_size=64) -> dict:
    model.eval()
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            probs_list.append(torch.sigmoid(model(xb)).cpu().numpy())
    probs = np.concatenate(probs_list)
    per_task = {t: binary_metrics(Y[:, i], probs[:, i], M[:, i]) for i, t in enumerate(TASKS)}
    f1s = [m["f1"] for m in per_task.values() if m["f1"] is not None]
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return per_task, macro_f1


def main():
    p = argparse.ArgumentParser(description="Train the small-CNN comparison baseline (no SmolVLM2)")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", default="dataset/annotations.json")
    p.add_argument("--out", default="dataset/checkpoints/head_cnn_baseline")
    p.add_argument("--yolo-model", default="models/yolo_model/yolo_model.pt")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--episode-gap", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask-cache", default=None, help="default: <out>/masks_cache.npz")
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.mask_cache) if args.mask_cache else out_dir / "masks_cache.npz"
    if args.rebuild_cache and cache_path.exists():
        cache_path.unlink()

    labels = load_usable_labels(Path(args.labels))
    frames = sorted(labels.keys(), key=lambda f: (frame_timestamp(f) is None, frame_timestamp(f) or 0, f))
    print(f"Usable labeled frames: {len(frames)}")
    print_class_diagnostics(labels, frames, args.episode_gap)

    train_frames, val_frames = split_frames(frames, args.val_frac, args.episode_gap, args.seed)
    print(f"Train frames: {len(train_frames)}  Val frames: {len(val_frames)}")

    minority = minority_classes({f: labels[f] for f in train_frames})
    train_manifest = manifest_for_train(train_frames, labels, minority)
    val_manifest = manifest_for(val_frames, labels, AUG_SPECS_VAL)

    # small conv net trains on CPU: this conv+batchnorm stack hits a stride bug in
    # backward() on MPS (view size incompatible with stride), reproducible even after
    # removing AdaptiveAvgPool2d (a separately documented MPS bug, see utils.py's
    # SafePool) -- so some other op in here is also unreliable on MPS backward. Model is
    # tiny (~55k conv params) so CPU is still fast, not worth chasing further.
    device = torch.device("cpu")
    seg_device = get_device()
    print(f"Device: model={device} yolo={seg_device}")
    seg = SegmentationPreprocessor(args.yolo_model, device=seg_device.type)

    cache_key = f"{args.yolo_model}|cropseg_v1"
    cache = load_mask_cache(cache_path, cache_key)
    cache = ensure_masks(train_manifest + val_manifest, images_dir, labels, cache, seg)
    save_mask_cache(cache_path, cache, cache_key)

    Xtr, Ytr, Mtr = build_xym(train_manifest, cache, labels)
    Xval, Yval, Mval = build_xym(val_manifest, cache, labels)
    print(f"Train samples (incl. augmentation): {len(Xtr)}   Val samples: {len(Xval)}")

    pos_weight = compute_pos_weight(Ytr, Mtr)
    print("Per-task pos_weight (train split):", dict(zip(TASKS, np.round(pos_weight, 2))))

    torch.manual_seed(args.seed)
    model = CNNClassifier(hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_t = torch.tensor(pos_weight, device=device)

    Ytr_t = torch.from_numpy(Ytr)
    Mtr_t = torch.from_numpy(Mtr)

    n = len(Xtr)
    best_macro_f1 = -1.0
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = torch.from_numpy(Xtr[idx.numpy()]).to(device)
            yb, mb = Ytr_t[idx].to(device), Mtr_t[idx].to(device)
            logits = model(xb)
            raw = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight_t, reduction="none")
            loss = (raw * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n

        per_task, macro_f1 = evaluate(model, Xval, Yval, Mval, device)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_macro_f1=macro_f1))
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"no val improvement in {args.patience} epochs, stopping at epoch {epoch} "
                  f"(best was epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    per_task, macro_f1 = evaluate(model, Xval, Yval, Mval, device)
    print(f"\nBest val macro F1: {macro_f1:.4f}")
    for t, m in per_task.items():
        if m["n"] == 0:
            print(f"  {t:14s}  (no val samples)")
        else:
            print(f"  {t:14s}  n={m['n']:4d}  acc={m['acc']:.3f}  "
                  f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")

    n_params = sum(p.numel() for p in model.parameters())
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(dict(
        state_dict=best_state,
        tasks=TASKS,
        hidden=args.hidden,
        dropout=args.dropout,
        yolo_model=args.yolo_model,
        n_params=n_params,
        train_frames=train_frames,
        val_frames=val_frames,
        val_frac=args.val_frac,
        episode_gap=args.episode_gap,
        seed=args.seed,
    ), ckpt_path)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "val_metrics.json").write_text(json.dumps(
        {t: {k: v for k, v in m.items()} for t, m in per_task.items()}, indent=2))
    print(f"\nParams: {n_params}")
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
