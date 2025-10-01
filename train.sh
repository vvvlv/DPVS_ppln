#!/bin/bash
# Simple training launcher script

set -e

CONFIG_DIR="configs/experiments"

if [ $# -eq 0 ]; then
    echo "Usage: ./train.sh <experiment_name>"
    echo ""
    echo "Available experiments:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -n 1 basename | sed 's/.yaml//' || echo "  (none yet)"
    exit 1
fi

EXPERIMENT=$1
CONFIG_FILE="$CONFIG_DIR/${EXPERIMENT}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Training: $EXPERIMENT"
echo "Config: $CONFIG_FILE"
echo "=========================================="

python scripts/train.py --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training complete!"
echo "==========================================" 