#!/bin/bash
# Testing/Inference launcher script
# Loads a checkpoint, runs inference on test images, calculates metrics, and saves predictions

set -e

CONFIG_DIR="configs/experiments"
CHECKPOINT_DIR="outputs/experiments"
OUTPUT_DIR="outputs/tests"

if [ $# -eq 0 ]; then
    echo "Usage: ./test.sh <experiment_name> [checkpoint_name]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name  Name of the experiment config file (without .yaml)"
    echo "  checkpoint_name  Optional: 'best' or 'last' (default: best)"
    echo ""
    echo "What it does:"
    echo "  - Loads the trained model checkpoint"
    echo "  - Runs inference on test images"
    echo "  - Calculates metrics (Dice, IoU, etc.) comparing predictions vs ground truth"
    echo "  - Saves predicted masks to: outputs/tests/<experiment_name>/predictions/"
    echo "  - Saves metrics to YAML files"
    echo ""
    echo "Available experiments:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -n 1 basename | sed 's/.yaml//' || echo "  (none yet)"
    echo ""
    echo "Examples:"
    echo "  ./test.sh exp001_basic_unet         # Uses best.pth"
    echo "  ./test.sh exp001_basic_unet best    # Uses best.pth"
    echo "  ./test.sh exp001_basic_unet last    # Uses last.pth"
    exit 1
fi

EXPERIMENT=$1
CHECKPOINT_NAME=${2:-best}  # Default to 'best' if not provided

CONFIG_FILE="$CONFIG_DIR/${EXPERIMENT}.yaml"
CHECKPOINT_FILE="$CHECKPOINT_DIR/${EXPERIMENT}/checkpoints/${CHECKPOINT_NAME}.pth"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_FILE"
    echo ""
    echo "Available checkpoints for $EXPERIMENT:"
    ls -1 "$CHECKPOINT_DIR/${EXPERIMENT}/checkpoints/"*.pth 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "=========================================="
echo "Running Inference & Testing"
echo "=========================================="
echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Output: $OUTPUT_DIR/$EXPERIMENT/"
echo "=========================================="

python scripts/test.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_FILE"

echo ""
echo "=========================================="
echo "✓ Testing complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR/$EXPERIMENT/"
echo "  - Predicted masks: $OUTPUT_DIR/$EXPERIMENT/predictions/"
echo "  - Metrics summary: $OUTPUT_DIR/$EXPERIMENT/test_metrics.yaml"
echo "  - Per-image metrics: $OUTPUT_DIR/$EXPERIMENT/per_image_metrics.yaml"
echo "=========================================="
