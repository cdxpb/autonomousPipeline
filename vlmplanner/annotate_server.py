"""
Frame annotation tool for the VLM planner's approach/exit label schema.

Labels, one entry per frame keyed by filename, in --labels-out:
    {"type": "approach", "junction": "no"}
    {"type": "approach", "junction": "yes", "directions": {"left": bool, "straight": bool, "right": bool}}
    {"type": "exit", "turn": "left"|"straight"|"right", "cleared": bool}
    {"type": "skip"}

Usage:
    python vlmplanner/annotate_server.py --images dataset/images --labels-out dataset/annotations.json
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

IMG_EXTS = {".jpg", ".jpeg", ".png"}

app = FastAPI()
STATE = {}  # populated in main(): images_dir, labels_path, frames


def _load_labels() -> dict:
    if STATE["labels_path"].exists():
        return json.loads(STATE["labels_path"].read_text())
    return {}


def _save_labels(labels: dict) -> None:
    path = STATE["labels_path"]
    if path.exists():
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    with open(fd, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    shutil.move(tmp_name, path)


def _validate(label: dict) -> None:
    if label.get("type") == "skip":
        return
    if label.get("type") == "approach":
        if label.get("junction") not in ("yes", "no"):
            raise HTTPException(400, "approach label needs junction: yes|no")
        if label["junction"] == "yes":
            dirs = label.get("directions")
            if not isinstance(dirs, dict) or set(dirs) != {"left", "straight", "right"}:
                raise HTTPException(400, "approach/junction=yes needs directions: {left,straight,right}")
            if not all(isinstance(v, bool) for v in dirs.values()):
                raise HTTPException(400, "directions values must be booleans")
    elif label.get("type") == "exit":
        if label.get("turn") not in ("left", "straight", "right"):
            raise HTTPException(400, "exit label needs turn: left|straight|right")
        if not isinstance(label.get("cleared"), bool):
            raise HTTPException(400, "exit label needs cleared: bool")
    else:
        raise HTTPException(400, "label.type must be 'approach' or 'exit'")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "annotate.html").read_text()


@app.get("/api/frames")
def frames():
    return {"frames": STATE["frames"]}


@app.get("/api/labels")
def labels():
    return _load_labels()


@app.post("/api/labels/{frame}")
def set_label(frame: str, label: dict):
    if frame not in STATE["frames"]:
        raise HTTPException(404, f"unknown frame: {frame}")
    _validate(label)
    labels = _load_labels()
    labels[frame] = label
    _save_labels(labels)
    return {"ok": True}


@app.delete("/api/labels/{frame}")
def delete_label(frame: str):
    labels = _load_labels()
    labels.pop(frame, None)
    _save_labels(labels)
    return {"ok": True}


def main():
    parser = argparse.ArgumentParser(description="VLM planner frame annotation UI")
    parser.add_argument("--images", required=True, help="Directory of frames (.jpg/.jpeg/.png)")
    parser.add_argument("--labels-out", default="dataset/annotations.json",
                        help="Where to persist labels (created if missing)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    images_dir = Path(args.images).resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"not a directory: {images_dir}")

    STATE["images_dir"]  = images_dir
    STATE["labels_path"] = Path(args.labels_out).resolve()
    STATE["labels_path"].parent.mkdir(parents=True, exist_ok=True)
    STATE["frames"] = sorted(f.name for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTS)

    if not STATE["frames"]:
        raise SystemExit(f"no images found in {images_dir}")

    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

    print(f"Frames : {len(STATE['frames'])} in {images_dir}")
    print(f"Labels : {STATE['labels_path']}")
    print(f"Open   : http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
