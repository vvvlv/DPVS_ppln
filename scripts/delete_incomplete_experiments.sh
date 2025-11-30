#!/bin/bash

# Script to delete experiment outputs that ran for less than 4 epochs
# Usage: ./delete_incomplete_experiments.sh [--dry-run]

set -e

# Configuration
EXPERIMENTS_DIR="outputs/experiments"
MIN_EPOCHS=5
DRY_RUN=false

# Parse arguments
if [ "$1" == "--dry-run" ] || [ "$1" == "-n" ]; then
    DRY_RUN=true
    echo "DRY RUN MODE - No files will be deleted"
    echo "========================================"
    echo ""
fi

# Check if experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Error: Experiments directory '$EXPERIMENTS_DIR' not found"
    exit 1
fi

# Counters
TOTAL_EXPERIMENTS=0
INCOMPLETE_COUNT=0
TO_DELETE=()

echo "Scanning experiments in: $EXPERIMENTS_DIR"
echo "Minimum epochs required: $MIN_EPOCHS"
echo ""

# Find all experiment directories
while IFS= read -r -d '' exp_dir; do
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    exp_name=$(basename "$exp_dir")
    
    # Check if metrics_history.yaml exists
    metrics_file="$exp_dir/metrics_history.yaml"
    
    if [ ! -f "$metrics_file" ]; then
        # No metrics file means incomplete or never started
        INCOMPLETE_COUNT=$((INCOMPLETE_COUNT + 1))
        TO_DELETE+=("$exp_dir")
        echo "[INCOMPLETE] $exp_name - No metrics_history.yaml found"
        continue
    fi
    
    # Count epochs in metrics_history.yaml
    # Count lines that start with "- epoch:"
    epoch_count=$(grep -c "^- epoch:" "$metrics_file" 2>/dev/null || echo "0")
    
    if [ "$epoch_count" -lt "$MIN_EPOCHS" ]; then
        INCOMPLETE_COUNT=$((INCOMPLETE_COUNT + 1))
        TO_DELETE+=("$exp_dir")
        echo "[INCOMPLETE] $exp_name - Only $epoch_count epoch(s) completed (need $MIN_EPOCHS)"
    else
        echo "[OK] $exp_name - $epoch_count epoch(s) completed"
    fi
    
done < <(find "$EXPERIMENTS_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Total experiments scanned: $TOTAL_EXPERIMENTS"
echo "Incomplete experiments (< $MIN_EPOCHS epochs): $INCOMPLETE_COUNT"
echo ""

if [ ${#TO_DELETE[@]} -eq 0 ]; then
    echo "No incomplete experiments found. Nothing to delete."
    exit 0
fi

echo "Experiments to delete:"
for i in "${!TO_DELETE[@]}"; do
    echo "  $((i+1)). $(basename "${TO_DELETE[$i]}")"
done
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN: Would delete ${#TO_DELETE[@]} experiment directory(ies)"
    echo "Run without --dry-run to actually delete them"
    exit 0
fi

# Confirm deletion
echo "WARNING: This will permanently delete ${#TO_DELETE[@]} experiment directory(ies)"
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Deletion cancelled."
    exit 0
fi

# Delete the directories
echo ""
echo "Deleting incomplete experiments..."
DELETED_COUNT=0
FAILED_COUNT=0

for exp_dir in "${TO_DELETE[@]}"; do
    exp_name=$(basename "$exp_dir")
    if rm -rf "$exp_dir"; then
        echo "  ✓ Deleted: $exp_name"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    else
        echo "  ✗ Failed to delete: $exp_name"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "========================================"
echo "Deletion Complete"
echo "========================================"
echo "Successfully deleted: $DELETED_COUNT"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "Failed to delete: $FAILED_COUNT"
    exit 1
fi

exit 0

