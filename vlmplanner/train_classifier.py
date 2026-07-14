# trains the classifier head on frozen smolvlm2 features
#
# python vlmplanner/train_classifier.py \
#     --images "dataset/dataset/cil_dataset_20260622-125203/images" \
#     --labels dataset/annotations.json \
#     --out dataset/checkpoints/head_500m

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
    minority_classes, manifest_for_train, print_class_diagnostics,
    VisionFeatureExtractor, MultiTaskHead, binary_metrics,
)
from segmentation import SegmentationPreprocessor, crop_image


def load_feature_cache(path: Path, cache_key: str) -> dict:
    if not path.exists():
        return {}
    npz = np.load(path, allow_pickle=True)
    if str(npz["cache_key"]) != cache_key:
        print(f"cache at {path} was built with a different model/preprocessing, ignoring it")
        return {}
    return {k: v for k, v in zip(npz["keys"], npz["embeddings"])}


def save_feature_cache(path: Path, cache: dict, cache_key: str):
    keys = list(cache.keys())
    embeddings = np.stack([cache[k] for k in keys])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.array(keys, dtype=object), embeddings=embeddings, cache_key=cache_key)


def ensure_features(manifest, images_dir: Path, labels: dict, cache: dict,
                     extractor: VisionFeatureExtractor, seg: SegmentationPreprocessor,
                     batch_size: int = 16, input_mode: str = "cropseg") -> dict:
    missing = [(f, spec) for f, spec in manifest if f"{f}|{spec}" not in cache]
    if not missing:
        return cache

    print(f"Featurizing {len(missing)} new (frame, augmentation) pairs...")
    t0 = time.time()
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        images = []
        for frame, spec in batch:
            img = Image.open(images_dir / frame).convert("RGB")
            rng_seed = abs(hash(f"{frame}:{spec}")) % (2**32)
            img, _ = augment(img, labels[frame], spec, random.Random(rng_seed))
            images.append(crop_image(img) if input_mode == "crop" else seg.process(img))
        embeddings = extractor.extract_batch(images)
        for (frame, spec), emb in zip(batch, embeddings):
            cache[f"{frame}|{spec}"] = emb
        if (start // batch_size) % 10 == 0:
            print(f"  {start + len(batch)}/{len(missing)}  ({time.time() - t0:.0f}s)")
    print(f"Done featurizing in {time.time() - t0:.0f}s")
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
    return np.stack(X).astype(np.float32), np.stack(Y), np.stack(M)


def compute_pos_weight(Y: np.ndarray, M: np.ndarray, cap: float = 10.0) -> np.ndarray:
    w = np.ones(len(TASKS), dtype=np.float32)
    for i in range(len(TASKS)):
        sel = M[:, i] > 0.5
        pos = Y[sel, i].sum()
        neg = sel.sum() - pos
        if pos > 0:
            w[i] = np.clip(neg / pos, 0.1, cap)
    return w


def evaluate(model, X, Y, M, device) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    per_task = {t: binary_metrics(Y[:, i], probs[:, i], M[:, i]) for i, t in enumerate(TASKS)}
    f1s = [m["f1"] for m in per_task.values() if m["f1"] is not None]
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return per_task, macro_f1


def main():
    p = argparse.ArgumentParser(description="Train approach/exit classification head on frozen SmolVLM2 features")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", default="dataset/annotations.json")
    p.add_argument("--out", default="dataset/checkpoints/head_500m_spatial3",
                   help="use a distinct --out per --model-id when comparing backbones")
    p.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Instruct")
    p.add_argument("--yolo-model", default="models/yolo_model/yolo_model.pt",
                   help="segmentation model, same one used by autonomous_node.py --approach 5")
    p.add_argument("--spatial-split", type=int, default=3,
                   help="1 = mean-pool all patches (position-blind), 3 = pool left/center/right "
                        "thirds separately (default, much better for left/right tasks)")
    p.add_argument("--input-mode", default="cropseg", choices=["cropseg", "crop"],
                   help="cropseg = crop+YOLO segment+colorize (default), crop = just the crop, "
                        "no segmentation at all -- ablation to check if segmentation is pulling weight")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--episode-gap", type=float, default=2.0,
                   help="seconds between consecutive frame timestamps that marks a new episode")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50,
                   help="stop after this many epochs with no val macro-F1 improvement")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feature-cache", default=None, help="default: <out>/features_cache.npz")
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.feature_cache) if args.feature_cache else out_dir / "features_cache.npz"
    if args.rebuild_cache and cache_path.exists():
        cache_path.unlink()

    labels = load_usable_labels(Path(args.labels))
    frames = sorted(labels.keys(), key=lambda f: (frame_timestamp(f) is None, frame_timestamp(f) or 0, f))
    print(f"Usable labeled frames: {len(frames)}")
    print("Class support (episodes, not frames -- a minority backed by 1-2 episodes is memorized, not generalized):")
    print_class_diagnostics(labels, frames, args.episode_gap)

    train_frames, val_frames = split_frames(frames, args.val_frac, args.episode_gap, args.seed)
    print(f"Train frames: {len(train_frames)}  Val frames: {len(val_frames)}  "
          f"(episode-grouped split, gap={args.episode_gap}s)")

    minority = minority_classes({f: labels[f] for f in train_frames})
    train_manifest = manifest_for_train(train_frames, labels, minority)
    val_manifest = manifest_for(val_frames, labels, AUG_SPECS_VAL)

    device = get_device()
    print(f"Device: {device}")
    extractor = VisionFeatureExtractor(args.model_id, device=device, spatial_split=args.spatial_split)
    print(f"Embedding dim: {extractor.embed_dim}")
    seg = SegmentationPreprocessor(args.yolo_model, device=device.type)

    cache_key = f"{args.model_id}|{args.yolo_model}|{args.input_mode}_v1|split{args.spatial_split}"
    cache = load_feature_cache(cache_path, cache_key)
    cache = ensure_features(train_manifest + val_manifest, images_dir, labels, cache, extractor, seg,
                             input_mode=args.input_mode)
    save_feature_cache(cache_path, cache, cache_key)

    Xtr, Ytr, Mtr = build_xym(train_manifest, cache, labels)
    Xval, Yval, Mval = build_xym(val_manifest, cache, labels)
    print(f"Train samples (incl. augmentation): {len(Xtr)}   Val samples: {len(Xval)}")

    pos_weight = compute_pos_weight(Ytr, Mtr)
    print("Per-task pos_weight (train split):", dict(zip(TASKS, np.round(pos_weight, 2))))

    torch.manual_seed(args.seed)
    model = MultiTaskHead(extractor.embed_dim, args.hidden, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_t = torch.tensor(pos_weight, device=device)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Mtr_t = torch.from_numpy(Mtr).to(device)

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
            xb, yb, mb = Xtr_t[idx], Ytr_t[idx], Mtr_t[idx]
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

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(dict(
        state_dict=best_state,
        tasks=TASKS,
        embed_dim=extractor.embed_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        model_id=args.model_id,
        yolo_model=args.yolo_model,
        spatial_split=args.spatial_split,
        input_mode=args.input_mode,
        train_frames=train_frames,
        val_frames=val_frames,
        val_frac=args.val_frac,
        episode_gap=args.episode_gap,
        seed=args.seed,
    ), ckpt_path)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "val_metrics.json").write_text(json.dumps(
        {t: {k: v for k, v in m.items()} for t, m in per_task.items()}, indent=2))
    print(f"\nSaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
