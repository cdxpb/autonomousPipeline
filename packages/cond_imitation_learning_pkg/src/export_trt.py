#!/usr/bin/env python3
"""
Build TensorRT .engine files from the existing ONNX exports.

Deliberately avoids torch2trt: a torch2trt TRTModule is a torch.nn.Module
wrapping the engine, so it still pulls in the full PyTorch CUDA runtime at
inference time -- on a 4GB Jetson Nano that alone can spike RAM/swap by
~1.8GB just from PyTorch's CUDA kernel loading (see
https://forums.developer.nvidia.com/t/jetpack-4-6-1-l4t-r32-7-1-pytorch-allocates-all-the-memory-swap/273631).
trtexec builds a raw .engine file that loads via plain `tensorrt` +
`pycuda`, with no torch import at inference time at all.

Run this ON THE ROBOT (needs the Jetson's TensorRT install). Requires the
source .onnx files to already be present under MODEL_BASE (see
transfer_models.sh).

Usage:
    python3 export_trt.py --approach 7
    python3 export_trt.py --approach 5
    python3 export_trt.py --yolo
    python3 export_trt.py --all
"""
import argparse
import os
import shutil
import subprocess

MODEL_BASE = "/data/models"

APPROACH_ONNX = {
    5: f"{MODEL_BASE}/pilotnet/best_model_regNheadv2.onnx",
    7: f"{MODEL_BASE}/pilotnet/best_model_approach7.onnx",
}
YOLO_ONNX = f"{MODEL_BASE}/yolo_model/yolo_model.onnx"


def find_trtexec():
    exe = shutil.which("trtexec")
    if exe:
        return exe
    for candidate in ("/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec"):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "trtexec not found. It ships with TensorRT (usually "
        "/usr/src/tensorrt/bin/trtexec on JetPack) -- check your TensorRT install."
    )


def build_engine(trtexec, onnx_path, fp16=True, workspace_mb=1024):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX source not found: {onnx_path}")
    engine_path = onnx_path[:-len(".onnx")] + ".engine"
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")
    print(f"Building engine: {onnx_path} -> {engine_path}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved {engine_path}")
    return engine_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=int, choices=[5, 7], help="PilotNet approach to export")
    parser.add_argument("--yolo", action="store_true", help="Export the YOLO segmentation model")
    parser.add_argument("--all", action="store_true", help="Export YOLO + approach 5 + approach 7")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 (builds FP32 engine)")
    parser.add_argument("--workspace", type=int, default=1024, help="Builder workspace size in MiB")
    args = parser.parse_args()

    if not (args.approach or args.yolo or args.all):
        parser.error("Specify --approach {5,7}, --yolo, or --all")

    trtexec = find_trtexec()
    fp16 = not args.no_fp16

    if args.all or args.yolo:
        build_engine(trtexec, YOLO_ONNX, fp16=fp16, workspace_mb=args.workspace)

    if args.all:
        for approach, onnx_path in APPROACH_ONNX.items():
            build_engine(trtexec, onnx_path, fp16=fp16, workspace_mb=args.workspace)
    elif args.approach:
        build_engine(trtexec, APPROACH_ONNX[args.approach], fp16=fp16, workspace_mb=args.workspace)
