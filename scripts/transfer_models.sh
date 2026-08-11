#!/bin/bash
# Transfers ONLY the required model files to the Duckiebot.
# Run from your Mac ONCE before launching run_robot.sh on the Duckiebot.
#
# Usage:
#   ./transfer_models.sh                          # Approach 7, smaller model set (default)
#   ./transfer_models.sh --approach 5
#   ./transfer_models.sh --all                     # Both approach 5 and 7
#   ./transfer_models.sh --all --model-set full     # older 480x640 models

ROBOT=duckie@duckiexp.local
REMOTE_DIR=/data/models
LOCAL_PILOTNET=models/pilotnet
LOCAL_YOLO=models/yolo_model
APPROACH=7
ALL=0
MODEL_SET=smaller

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --approach)  APPROACH="$2";  shift ;;
        --all) ALL=1 ;;
        --model-set) MODEL_SET="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "=== Transferring models to $ROBOT:$REMOTE_DIR (model-set: $MODEL_SET) ==="

ssh $ROBOT "mkdir -p $REMOTE_DIR/pilotnet $REMOTE_DIR/yolo_model"

if [ "$MODEL_SET" == "full" ]; then
    echo "-> Transferring YOLO ONNX + PT model (480x640, original)..."
    rsync -av --progress "$LOCAL_YOLO/yolo_model.onnx" "$ROBOT:$REMOTE_DIR/yolo_model/"
    rsync -av --progress "$LOCAL_YOLO/yolo_model.pt"   "$ROBOT:$REMOTE_DIR/yolo_model/"

    if [ $ALL -eq 1 ] || [ "$APPROACH" -eq 5 ]; then
        echo "-> Transferring Approach 5 (regNheadv2, original) ONNX + PT..."
        rsync -av --progress "$LOCAL_PILOTNET/best_model_regNheadv2.onnx" "$ROBOT:$REMOTE_DIR/pilotnet/"
        rsync -av --progress "$LOCAL_PILOTNET/best_model_regNheadv2.pt"   "$ROBOT:$REMOTE_DIR/pilotnet/"
    fi
    if [ $ALL -eq 1 ] || [ "$APPROACH" -eq 7 ]; then
        echo "-> Transferring Approach 7 (original) ONNX + PT..."
        rsync -av --progress "$LOCAL_PILOTNET/best_model_approach7.onnx" "$ROBOT:$REMOTE_DIR/pilotnet/"
        rsync -av --progress "$LOCAL_PILOTNET/best_model_approach7.pt"   "$ROBOT:$REMOTE_DIR/pilotnet/"
    fi
else
    echo "-> Transferring YOLO ONNX + PT model (256x320)..."
    rsync -av --progress "$LOCAL_YOLO/yolo_best_256x320.onnx" "$ROBOT:$REMOTE_DIR/yolo_model/"
    rsync -av --progress "$LOCAL_YOLO/yolo_best_256x320.pt"   "$ROBOT:$REMOTE_DIR/yolo_model/"
    # Only exists after running `python3 export_trt.py --yolo` locally at least once.
    if [ -f "$LOCAL_YOLO/yolo_best_256x320_trt.onnx" ]; then
        rsync -av --progress "$LOCAL_YOLO/yolo_best_256x320_trt.onnx" "$ROBOT:$REMOTE_DIR/yolo_model/"
    fi

    if [ $ALL -eq 1 ] || [ "$APPROACH" -eq 5 ]; then
        echo "-> Transferring Approach 5 (smaller) ONNX + PT..."
        rsync -av --progress "$LOCAL_PILOTNET/best_model_approach5_smaller.onnx" "$ROBOT:$REMOTE_DIR/pilotnet/"
        rsync -av --progress "$LOCAL_PILOTNET/best_model_approach5_smaller.pt"   "$ROBOT:$REMOTE_DIR/pilotnet/"
    fi
    if [ $ALL -eq 1 ] || [ "$APPROACH" -eq 7 ]; then
        echo "-> Transferring Approach 7 (smaller) ONNX + PT..."
        rsync -av --progress "$LOCAL_PILOTNET/best_model_approach7_smaller.onnx" "$ROBOT:$REMOTE_DIR/pilotnet/"
        rsync -av --progress "$LOCAL_PILOTNET/best_model_approach7_smaller.pt"   "$ROBOT:$REMOTE_DIR/pilotnet/"
    fi
fi

echo
echo "Transfer complete! On the Duckiebot, models are at: $REMOTE_DIR/"
