#!/bin/bash
# ============================================================
# sync_repo_to_bot.sh – Sync the repo source (not models, not .git)
# to the Duckiebot's home dir for native (no-Docker) builds.
# Run this from your Mac. Models are transferred separately via
# transfer_models.sh (they're large and don't change often).
# ============================================================

ROBOT=duckie@duckiexp.local
REMOTE_DIR="~/dev/autonomousPipeline"

echo "======================================================="
echo " Syncing repo source to $ROBOT:$REMOTE_DIR"
echo "======================================================="

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
