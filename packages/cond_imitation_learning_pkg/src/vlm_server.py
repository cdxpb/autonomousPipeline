#!/usr/bin/env python3
import io
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

# Import the Oracle and Config
from vlm_planner import VLMOracle, Config

app = FastAPI(title="Mac Native Inference Server")

class InferenceServer:
    def __init__(self):
        self.vlm_oracle = None

    def load_vlm(self):
        if self.vlm_oracle is None:
            print("Loading SmolVLM2 to MPS...")
            # precision fp16 is optimal for M1 MPS
            cfg = Config(model_id="HuggingFaceTB/SmolVLM2-256M-Instruct", precision="fp16")
            self.vlm_oracle = VLMOracle(cfg)
            print("VLM Loaded.")

server = InferenceServer()

@app.on_event("startup")
def startup_event():
    server.load_vlm()

@app.post("/predict/vlm")
async def predict_vlm(file: UploadFile = File(...), direction: str = Form("straight")):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Run inference
    ans, confidence = server.vlm_oracle.at_intersection(image, direction)
    
    return JSONResponse({
        "ans": ans,
        "confidence": confidence
    })

if __name__ == "__main__":
    uvicorn.run("vlm_server:app", host="0.0.0.0", port=8000, reload=False)
