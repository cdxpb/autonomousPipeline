#!/bin/bash
# ============================================================
# transfer_models.sh  –  Transfers ONLY the required model files to the Duckiebot
# Run this from your Mac ONCE before launching run_robot.sh on the Duckiebot
# ============================================================
# Usage:
#   ./transfer_models.sh            # Approach 7 (default)
#   ./transfer_models.sh --approach 5
#   ./transfer_models.sh --all      # Both approach 5 and 7
# ============================================================

ROBOT=duckie@duckiexp.local
REMOTE_DIR=/data/models
LOCAL_PILOTNET=models/pilotnet
LOCAL_YOLO=models/yolo_model
APPROACH=7
ALL=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --approach) APPROACH="$2"; shift ;;
        --all) ALL=1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "======================================================="
echo " Transferring models to $ROBOT:$REMOTE_DIR"
echo "======================================================="

# Ensure remote directories exist
ssh $ROBOT "mkdir -p $REMOTE_DIR/pilotnet $REMOTE_DIR/yolo_model"

# Always transfer YOLO ONNX (used by both approaches)
echo "→ Transferring YOLO ONNX model..."
rsync -av --progress "$LOCAL_YOLO/yolo_model.onnx" "$ROBOT:$REMOTE_DIR/yolo_model/"

if [ $ALL -eq 1 ] || [ "$APPROACH" -eq 5 ]; then
    echo "→ Transferring Approach 5 (regNheadv2) ONNX..."
    rsync -av --progress "$LOCAL_PILOTNET/best_model_regNheadv2.onnx" "$ROBOT:$REMOTE_DIR/pilotnet/"
    rsync -av --progress "$LOCAL_PILOTNET/best_model_regNheadv2.pt"   "$ROBOT:$REMOTE_DIR/pilotnet/"
fi

if [ $ALL -eq 1 ] || [ "$APPROACH" -eq 7 ]; then
    echo "→ Transferring Approach 7 (FiLM) ONNX..."
    rsync -av --progress "$LOCAL_PILOTNET/best_model_approach7.onnx" "$ROBOT:$REMOTE_DIR/pilotnet/"
    rsync -av --progress "$LOCAL_PILOTNET/best_model_approach7.pt"   "$ROBOT:$REMOTE_DIR/pilotnet/"
fi

echo ""
echo "Transfer complete!"
echo "On the Duckiebot, models are at: $REMOTE_DIR/"
