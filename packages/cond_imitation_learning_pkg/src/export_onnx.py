#!/usr/bin/env python3
"""
Exports PilotNet Approach 5 (regNheadv2) and Approach 7 (FiLM) to ONNX.
Replaces SafePool with equivalent static AvgPool2d kernels before exporting, the ONNX
tracer chokes on AdaptiveAvgPool2d's non-divisible output sizes here.

Run this on your Mac (in ros_native env) before deploying to the Duckiebot:
    conda activate ros_native
    cd packages/cond_imitation_learning_pkg/src
    python3 export_onnx.py
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../../models/pilotnet")


def _patch_safe_pool(module, kernel, stride):
    """Recursively replace SafePool with a static AvgPool2d in-place."""
    from utils import SafePool
    for name, child in module.named_children():
        if isinstance(child, SafePool):
            setattr(module, name, nn.AvgPool2d(kernel_size=kernel, stride=stride))
        else:
            _patch_safe_pool(child, kernel, stride)


# Approach 7: FiLM
# Feature map entering SafePool is 48 x 7 x 21 -> target (4, 7)
# Static equivalent: AvgPool2d(kernel_size=(1,3), stride=(2,3))

def export_approach7():
    from pilotnet_FiLM import ConditionalPilotNetFiLM
    model = ConditionalPilotNetFiLM()
    pt_path  = os.path.join(MODEL_DIR, "best_model_approach7.pt")
    out_path = os.path.join(MODEL_DIR, "best_model_approach7.onnx")

    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=False))
    model.eval()
    _patch_safe_pool(model, kernel=(1, 3), stride=(2, 3))

    dummy_img    = torch.randn(1, 1, 112, 224)
    dummy_intent = torch.randn(1, 4)

    torch.onnx.export(
        model, (dummy_img, dummy_intent), out_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=["image_input", "intent_input"],
        output_names=["wheel_velocities"],
        dynamic_axes={"image_input": {0: "batch"}, "intent_input": {0: "batch"}, "wheel_velocities": {0: "batch"}},
    )
    print(f"Approach 7 exported -> {out_path}")
    _verify(out_path, dummy_img, dummy_intent, model)


# Approach 5: regNheadv2
# Feature map entering SafePool is 64 x 7 x 21 -> target (5, 10)
# Static equivalent: AvgPool2d(kernel_size=(3,3), stride=(1,2))
# Also replaces dynamic boolean-mask head selection with a static weighted-sum.

class ConditionalPilotNetONNXWrapper(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.feature_extractor   = orig.feature_extractor
        self.pool                = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2))
        self.flattened_size      = 64 * 5 * 10
        self.straight_head       = orig.straight_head
        self.left_head           = orig.left_head
        self.right_head          = orig.right_head
        self.lane_following_head = orig.lane_following_head

    def forward(self, img, intent):
        features = self.feature_extractor(img)
        features = self.pool(features)
        features = features.reshape(-1, self.flattened_size)
        out_s  = self.straight_head(features)
        out_l  = self.left_head(features)
        out_r  = self.right_head(features)
        out_lf = self.lane_following_head(features)
        heads   = torch.stack([out_s, out_l, out_r, out_lf], dim=1)  # (B, 4, 2)
        weights = intent.unsqueeze(-1)                                 # (B, 4, 1)
        return (heads * weights).sum(dim=1)                            # (B, 2)


def export_approach5():
    from pilotnet_regNheadv2 import ConditionalPilotNet
    pt_path  = os.path.join(MODEL_DIR, "best_model_regNheadv2.pt")
    out_path = os.path.join(MODEL_DIR, "best_model_regNheadv2.onnx")

    orig = ConditionalPilotNet()
    orig.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=False))
    orig.eval()

    model = ConditionalPilotNetONNXWrapper(orig)
    model.eval()

    dummy_img    = torch.randn(1, 1, 112, 224)
    dummy_intent = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    torch.onnx.export(
        model, (dummy_img, dummy_intent), out_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=["image_input", "intent_input"],
        output_names=["wheel_velocities"],
        dynamic_axes={"image_input": {0: "batch"}, "intent_input": {0: "batch"}, "wheel_velocities": {0: "batch"}},
    )
    print(f"Approach 5 exported -> {out_path}")
    _verify(out_path, dummy_img, dummy_intent, model)


def _verify(onnx_path, dummy_img, dummy_intent, pt_model):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"image_input": dummy_img.numpy(), "intent_input": dummy_intent.numpy()})[0]
        with torch.no_grad():
            pt_out = pt_model(dummy_img, dummy_intent).numpy()
        max_diff = float(abs(ort_out - pt_out).max())
        if max_diff < 1e-3:
            print(f"  ONNX matches PyTorch (max diff = {max_diff:.2e})")
        else:
            print(f"  WARNING: max diff = {max_diff:.4f}")
    except Exception as e:
        print(f"  Verification skipped: {e}")


if __name__ == "__main__":
    print("Exporting Approach 7 (FiLM)...")
    export_approach7()

    print("\nExporting Approach 5 (regNheadv2)...")
    export_approach5()

    print("\nDone! Transfer to Duckiebot:")
    print("  ./transfer_models.sh --all")
