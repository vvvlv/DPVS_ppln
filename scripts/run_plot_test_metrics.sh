#!/usr/bin/env bash
# Convenience wrapper to plot test metrics for the main experiment suites.
#
# Experiments covered:
#   - exp001_1_basic_unet_green
#   - exp003_unet_components_01_baseline ... exp003_unet_components_12_kernel7x7
#   - exp004_auc_01_unet_baseline ... exp004_auc_04_unet_deep
#
# Usage:
#   ./scripts/plot_all_metrics.sh
#   ./scripts/plot_all_metrics.sh --show   # also display the plots interactively

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"



# EXPERIMENTS=(
#   "exp001_1_basic_unet_green"
#   "exp003_unet_components_02_reduced_depths"
#   "exp003_unet_components_04_shallow"
#   "exp003_unet_components_05_deep"
#   "exp003_unet_components_06_limited_skip"
#   "exp003_unet_components_10_no_skip"
#   "exp003_unet_components_11_kernel5x5"
#   "exp003_unet_components_12_kernel7x7"
#   "exp004_auc_01_unet_baseline"
#   "exp004_auc_02_unet_kernel5"
#   "exp004_auc_03_unet_shallow"
#   "exp004_auc_04_unet_deep"
# )


EXPERIMENTS=(
  "exp003_unet_components_01_baseline"
  "exp003_unet_components_07_simple_encoder"
  "exp003_unet_components_08_simple_bottleneck"
  "exp003_unet_components_09_simple_decoder"
  "exp003_unet_components_10_no_skip"
  "exp005_focal_components_01_baseline"
  "exp005_focal_components_07_simple_encoder"
  "exp005_focal_components_08_simple_bottleneck"
  "exp005_focal_components_09_simple_decoder"
  "exp005_focal_components_10_no_skip"
  "exp006_focal_reg_01_baseline"
  "exp006_focal_reg_02_baseline_dropout_0.1"
  "exp006_focal_reg_03_baseline_dropout_0.005"
  "exp006_focal_reg_04_baseline_dropout_0.4"
  "exp006_focal_reg_11_simple_encoder_decoder"
)


SHOW_FLAG=""
OUTPUT_DIR="outputs/plots/unet_components_comparison"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --show)
      SHOW_FLAG="--show"
      shift
      ;;
    --output=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --output)
      shift
      OUTPUT_DIR="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./scripts/run_plot_test_metrics.sh [--show] [--output <dir>]"
      exit 1
      ;;
  esac
done

python scripts/plot_test_metrics.py \
  --select "${EXPERIMENTS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  ${SHOW_FLAG:+$SHOW_FLAG}

echo "✓ Plots saved under $OUTPUT_DIR"

