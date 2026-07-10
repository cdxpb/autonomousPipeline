#!/usr/bin/env python3
"""
Mac-native VLM inference server. Offloads vlm_planner.VLMOracle out of the dashboard
process. Exposes both oracle questions behind one endpoint, selected via `mode`.

Usage:
    python vlm_server.py --model HuggingFaceTB/SmolVLM2-256M-Instruct --precision fp16 --port 8000
"""
import io
import argparse
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

from vlm_planner import VLMOracle, Config

app = FastAPI(title="Mac Native VLM Inference Server")


class InferenceServer:
    def __init__(self, model_id: str, precision: str):
        self.model_id   = model_id
        self.precision  = precision
        self.vlm_oracle = None

    def load_vlm(self):
        if self.vlm_oracle is None:
            print(f"Loading {self.model_id} [{self.precision}] ...")
            cfg = Config(model_id=self.model_id, precision=self.precision)
            self.vlm_oracle = VLMOracle(cfg)
            print("VLM Loaded.")


server = InferenceServer(model_id="HuggingFaceTB/SmolVLM2-256M-Instruct", precision="fp16")


@app.on_event("startup")
def startup_event():
    server.load_vlm()


@app.get("/health")
def health():
    return {"status": "ok", "loaded": server.vlm_oracle is not None}


@app.post("/predict/vlm")
async def predict_vlm(file: UploadFile = File(...),
                       direction: str = Form("straight"),
                       mode: str = Form("approach")):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    if mode == "exit":
        ans, confidence = server.vlm_oracle.classify_exit(image)
    else:
        ans, confidence = server.vlm_oracle.classify_approach(image, direction)

    return JSONResponse({
        "ans": ans,
        "confidence": confidence,
        "mode": mode,
    })


def main():
    parser = argparse.ArgumentParser(description="Mac native VLM inference server")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-256M-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "int8", "int4"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server.model_id  = args.model
    server.precision = args.precision

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
