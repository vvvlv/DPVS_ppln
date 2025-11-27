#!/usr/bin/env bash

# Configure experiment comparisons here, then run:
#   ./scripts/compare_experiments.sh
#
# This script avoids long Python CLI calls by letting you define experiments
# and (optionally) a fixed sample image as shell variables.

set -e

########################################
# USER CONFIGURATION
########################################

# List of experiments to compare (rows in the plot)
EXPERIMENTS=(
  exp008_tinyswin_reg_dice_patch2_embeddingDim64
  exp008_tinyswin_reg_focal_patch2_embeddingDim64
  exp008_tinyswin_reg_focal_patch4_embeddingDim128
  exp006_focal_reg_11_simple_encoder_decoder
)

# Optional short display names (same length/order as EXPERIMENTS).
# If empty or length mismatch, full experiment IDs will be shown.
DISPLAY_NAMES=(
  "Dice p2 d64"
  "Focal p2 d64"
  "Focal p4 d128"
  "Baseline_encoder_decoder_reg"
)

# Base output directory for comparison plots
OUTPUT_DIR="outputs/comparison_plots/tinyswin_comparisons"

# Optional fixed test sample (e.g. "178_N.png"); leave empty for random
SAMPLE=""

# Dataset config used to resolve test image/label paths
DATASET_CONFIG="configs/datasets/fives512_g.yaml"

# Optional random seed for reproducible random sample selection
SEED=""

# Threshold for binarizing predictions
THRESHOLD=0.5



########################################
# SCRIPT
########################################

if [ "${#EXPERIMENTS[@]}" -lt 2 ]; then
  echo "Please configure at least two experiments in compare_experiments.sh (EXPERIMENTS array)."
  exit 1
fi

ARGS=(
  --dataset-config "$DATASET_CONFIG"
  --threshold "$THRESHOLD"
  --output-dir "$OUTPUT_DIR"
)

if [ -n "$SAMPLE" ]; then
  ARGS+=(--sample "$SAMPLE")
fi

if [ -n "$SEED" ]; then
  ARGS+=(--seed "$SEED")
fi

if [ "${#DISPLAY_NAMES[@]}" -eq "${#EXPERIMENTS[@]}" ] && [ "${#DISPLAY_NAMES[@]}" -gt 0 ]; then
  ARGS+=(--display-names "${DISPLAY_NAMES[@]}")
fi

python scripts/compare_test_results.py "${EXPERIMENTS[@]}" "${ARGS[@]}"


