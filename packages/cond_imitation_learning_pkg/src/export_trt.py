#!/usr/bin/env python3
"""
Build TensorRT .engine files from the existing ONNX exports.
Run this ON THE ROBOT (needs the Jetson's TensorRT install). Requires the
source .onnx files already present under MODEL_BASE (see transfer_models.sh).

Two model sets, matching headless_autonomous_node.py's YOLO_SPEC/PILOTNET_BASENAME
tables exactly (keep both in sync if either changes):
  "smaller" (default): yolo_best_256x320, retrained backbone, faster.
  "full": the original 480x640 models.

Usage:
    python3 export_trt.py --approach 7
    python3 export_trt.py --approach 5 --model-set full
    python3 export_trt.py --yolo
    python3 export_trt.py --all
    python3 export_trt.py --all --model-set full
"""
import argparse
import os
import shutil
import subprocess

MODEL_BASE = "/data/models"

YOLO_SPEC = {
    "smaller": {"basename": "yolo_best_256x320", "argmax_fused": True},
    "full":    {"basename": "yolo_model",        "argmax_fused": False},
}
PILOTNET_BASENAME = {
    (5, "smaller"): "best_model_approach5_smaller",
    (5, "full"):    "best_model_regNheadv2",
    (7, "smaller"): "best_model_approach7_smaller",
    (7, "full"):    "best_model_approach7",
}


def patch_uint8_output_to_int32(onnx_path):
    """Patches a fused ArgMax->Cast(uint8) output to INT32, this Jetson's TensorRT can't
    import a uint8 Cast. Writes a separate *_trt.onnx, original stays untouched for the
    onnxruntime backend (which handles uint8 fine)."""
    patched_path = onnx_path[:-len(".onnx")] + "_trt.onnx"
    if os.path.exists(patched_path):
        return patched_path

    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        raise RuntimeError(
            f"{patched_path} not found, and the `onnx` package isn't installed here to "
            f"generate it. Easiest fix: run `python3 export_trt.py --yolo` once on your "
            f"Mac (onnx is already available there via the ros_native conda env) to "
            f"produce {os.path.basename(patched_path)}, then `./transfer_models.sh --all` "
            f"to sync it to the robot."
        )

    model = onnx.load(onnx_path)
    changed = False
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.UINT8:
                    attr.i = TensorProto.INT32
                    changed = True

    if not changed:
        return onnx_path

    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.UINT8:
            out.type.tensor_type.elem_type = TensorProto.INT32

    onnx.save(model, patched_path)
    print(f"Patched UINT8 Cast -> INT32 for TensorRT 8.2.1 compat: {patched_path}")
    return patched_path


def find_trtexec():
    exe = shutil.which("trtexec")
    if exe:
        return exe
    for candidate in ("/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec"):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "trtexec not found. It ships with TensorRT (usually "
        "/usr/src/tensorrt/bin/trtexec on JetPack), check your TensorRT install."
    )


def build_engine(trtexec, onnx_path, fp16=True, workspace_mb=1024, engine_path=None):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX source not found: {onnx_path}")
    if engine_path is None:
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


def build_yolo_engine(model_set, trtexec, fp16, workspace_mb):
    spec = YOLO_SPEC[model_set]
    onnx_path = f"{MODEL_BASE}/yolo_model/{spec['basename']}.onnx"
    engine_path = f"{MODEL_BASE}/yolo_model/{spec['basename']}.engine"
    source = patch_uint8_output_to_int32(onnx_path) if spec["argmax_fused"] else onnx_path
    build_engine(trtexec, source, fp16=fp16, workspace_mb=workspace_mb, engine_path=engine_path)


def build_pilotnet_engine(approach, model_set, trtexec, fp16, workspace_mb):
    onnx_path = f"{MODEL_BASE}/pilotnet/{PILOTNET_BASENAME[(approach, model_set)]}.onnx"
    build_engine(trtexec, onnx_path, fp16=fp16, workspace_mb=workspace_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=int, choices=[5, 7], help="PilotNet approach to export")
    parser.add_argument("--yolo", action="store_true", help="Export the YOLO segmentation model")
    parser.add_argument("--all", action="store_true", help="Export YOLO + approach 5 + approach 7")
    parser.add_argument("--model-set", choices=["smaller", "full"], default="smaller",
                         help="smaller (default): yolo_best_256x320. full: the original 480x640 models.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 (builds FP32 engine)")
    parser.add_argument("--workspace", type=int, default=1024, help="Builder workspace size in MiB")
    args = parser.parse_args()

    if not (args.approach or args.yolo or args.all):
        parser.error("Specify --approach {5,7}, --yolo, or --all")

    trtexec = find_trtexec()
    fp16 = not args.no_fp16

    if args.all or args.yolo:
        build_yolo_engine(args.model_set, trtexec, fp16, args.workspace)

    if args.all:
        for approach in (5, 7):
            build_pilotnet_engine(approach, args.model_set, trtexec, fp16, args.workspace)
    elif args.approach:
        build_pilotnet_engine(args.approach, args.model_set, trtexec, fp16, args.workspace)
