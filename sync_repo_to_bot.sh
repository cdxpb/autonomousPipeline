#!/bin/bash
# Sync repo source (not models, not .git) to the Duckiebot for native builds.
# Run from your Mac. Models are transferred separately via transfer_models.sh.

ROBOT=duckie@duckiexp.local
REMOTE_DIR="~/dev/autonomousPipeline"

echo "=== Syncing repo source to $ROBOT:$REMOTE_DIR ==="

ssh "$ROBOT" "mkdir -p $REMOTE_DIR"

rsync -av --progress \
    --exclude models \
    --exclude .git \
    --exclude notebooks \
    --exclude report \
    --exclude .ipynb_checkpoints \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.onnx' \
    --exclude '*.onnx.data' \
    ./ "$ROBOT:$REMOTE_DIR/"

echo
echo "Sync complete. Models still need: ./transfer_models.sh --all"
