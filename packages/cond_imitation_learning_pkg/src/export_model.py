#!/usr/bin/env python3
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Export PilotNet models to ONNX or TensorRT.")
    parser.add_argument("--format", type=str, choices=["onnx", "trt"], required=True, help="Target format to export.")
    parser.add_argument("--approach", type=int, default=7, help="PilotNet approach (e.g. 5, 7).")
    parser.add_argument("--basename", type=str, default="best_model", help="Base name of the model to export.")
    return parser.parse_args()

def export_to_onnx(approach, basename):
    import torch
    import torch.nn as nn
    from models import get_pilotnet_model

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../../models/pilotnet")

    def _patch_safe_pool(module, kernel, stride):
        from utils import SafePool
        for name, child in module.named_children():
            if isinstance(child, SafePool):
                setattr(module, name, nn.AvgPool2d(kernel_size=kernel, stride=stride))
            else:
                _patch_safe_pool(child, kernel, stride)

    pt_path = os.path.join(MODEL_DIR, f"{basename}.pt")
    if not os.path.exists(pt_path):
        # fallback to .pth
        pt_path = os.path.join(MODEL_DIR, f"{basename}.pth")
        
    out_path = os.path.join(MODEL_DIR, f"{basename}.onnx")

    orig = get_pilotnet_model(approach)
    if os.path.exists(pt_path):
        orig.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=False))
    else:
        print(f"Warning: {pt_path} not found. Exporting randomly initialized model.")
        
    orig.eval()

    if approach == 7:
        _patch_safe_pool(orig, kernel=(1, 3), stride=(2, 3))
        model = orig
        dummy_intent = torch.randn(1, 4)
    elif approach == 5:
        # Wrapper for Approach 5
        class ConditionalPilotNetONNXWrapper(nn.Module):
            def __init__(self, orig_model):
                super().__init__()
                self.feature_extractor = orig_model.feature_extractor
                self.pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2))
                self.flattened_size = orig_model.flattened_size
                self.straight_head = orig_model.straight_head
                self.left_head = orig_model.left_head
                self.right_head = orig_model.right_head
                self.lane_following_head = getattr(orig_model, 'lane_following_head', orig_model.straight_head) # fallback

            def forward(self, img, intent):
                features = self.feature_extractor(img)
                features = self.pool(features)
                features = features.reshape(-1, self.flattened_size)
                out_s = self.straight_head(features)
                out_l = self.left_head(features)
                out_r = self.right_head(features)
                out_lf = self.lane_following_head(features)
                heads = torch.stack([out_s, out_l, out_r, out_lf], dim=1)  # (B, 4, 2)
                weights = intent.unsqueeze(-1)                             # (B, 4, 1)
                return (heads * weights).sum(dim=1)                        # (B, 2)
        model = ConditionalPilotNetONNXWrapper(orig)
        model.eval()
        dummy_intent = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    else:
        model = orig
        dummy_intent = torch.randn(1, 4)

    dummy_img = torch.randn(1, 1 if approach in [5,7] else 3, 112, 224)

    torch.onnx.export(
        model, (dummy_img, dummy_intent), out_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=["image_input", "intent_input"],
        output_names=["wheel_velocities"],
        dynamic_axes={"image_input": {0: "batch"}, "intent_input": {0: "batch"}, "wheel_velocities": {0: "batch"}},
    )
    print(f"Exported Approach {approach} to ONNX: {out_path}")

def patch_uint8_output_to_int32(onnx_path):
    patched_path = onnx_path[:-len(".onnx")] + "_trt.onnx"
    if os.path.exists(patched_path):
        return patched_path

    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        raise RuntimeError("onnx package required to patch for TRT")

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
    print(f"Patched UINT8 Cast -> INT32 for TensorRT: {patched_path}")
    return patched_path

def build_trt_engine(onnx_path, engine_path=None):
    import shutil
    import subprocess
    
    exe = shutil.which("trtexec")
    if not exe:
        for candidate in ("/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec"):
            if os.path.exists(candidate):
                exe = candidate
                break
    if not exe:
        raise FileNotFoundError("trtexec not found.")

    if engine_path is None:
        engine_path = onnx_path[:-len(".onnx")] + ".engine"

    cmd = [
        exe,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--workspace=1024"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Engine saved to {engine_path}")

def export_to_trt(approach, basename):
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../../models/pilotnet")
    onnx_path = os.path.join(MODEL_DIR, f"{basename}.onnx")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX source not found: {onnx_path}")
        
    try:
        patched_path = patch_uint8_output_to_int32(onnx_path)
    except Exception as e:
        print(f"Patching failed or skipped: {e}")
        patched_path = onnx_path
        
    build_trt_engine(patched_path)

if __name__ == "__main__":
    args = parse_args()
    if args.format == "onnx":
        export_to_onnx(args.approach, args.basename)
    elif args.format == "trt":
        export_to_trt(args.approach, args.basename)
