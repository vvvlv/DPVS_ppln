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

# Extract output directory from config file
OUTPUT_DIR=$(grep -E "^\s*dir:" "$CONFIG_FILE" | head -1 | sed 's/.*dir:\s*"\?\([^"]*\)"\?.*/\1/' | tr -d '"')

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_log_${TIMESTAMP}.txt"

echo "Console output will be saved to: $LOG_FILE"
echo ""

# Run training and save output to log file while displaying it
python scripts/train.py --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# Capture the exit status
EXIT_STATUS=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Training complete!"
else
    echo "Training failed with exit code: $EXIT_STATUS"
fi
echo "Log saved to: $LOG_FILE"
echo "=========================================="

exit $EXIT_STATUS 