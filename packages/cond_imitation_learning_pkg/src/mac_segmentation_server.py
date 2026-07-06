#!/usr/bin/env python3
"""

run the server: python3 mac_segmentation_server.py --model_path /path/to/yolo_model.onnx --device mps
   (Note: For ONNX models, the 'device' argument is typically ignored by Ultralytics as it defaults to the CPU via ONNXRuntime, unless you install onnxruntime-silicon for CoreML).
"""

import io
import argparse
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
from PIL import Image

app = FastAPI(title="Mac MPS YOLO Segmentation Server")
model = None
args = None

@app.on_event("startup")
def startup_event():
    global model
    print(f"Loading YOLO model from {args.model_path} onto device={args.device}...")
    model = YOLO(args.model_path, task='semantic')
    
    # Pre-warm model with a dummy image to avoid latency on the first request
    # Note: size might differ depending on approach, but 224x112 is a safe default for prewarming
    dummy_img = np.zeros((112, 224, 3), dtype=np.uint8)
    _ = model(dummy_img, device=args.device, verbose=False)
    print("Model loaded and pre-warmed.")

@app.post("/predict/segmentation")
async def predict_segmentation(file: UploadFile = File(...)):
    """
    Accepts a JPEG image, runs YOLO semantic segmentation, 
    and returns the mask encoded as a PNG image to preserve exact class values.
    """
    # Read the incoming image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if cv_image is None:
        return Response(status_code=400, content="Invalid image format")

    # YOLO expects RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Run YOLO Inference
    results = model(rgb_image, device=args.device, verbose=False)
    
    # Extract semantic mask
    mask = results[0].semantic_mask.data.cpu()
    mask_np = mask.numpy().astype(np.uint8)
    
    # Encode mask as PNG for lossless compression (important so class indices aren't distorted by JPEG artifacts)
    is_success, buffer = cv2.imencode(".png", mask_np)
    if not is_success:
        return Response(status_code=500, content="Failed to encode mask")

    return Response(content=buffer.tobytes(), media_type="image/png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model (e.g., yolo_model.pt)")
    parser.add_argument("--device", type=str, default="mps", help="Device to run inference on (mps, cuda, cpu)")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
