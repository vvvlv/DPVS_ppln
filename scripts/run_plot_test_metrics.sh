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
#   "exp003_unet_components_01_baseline"
#   "exp003_unet_components_07_simple_encoder"
#   "exp003_unet_components_08_simple_bottleneck"
#   "exp003_unet_components_09_simple_decoder"
#   "exp003_unet_components_10_no_skip"
# )
# OUTPUT_DIR="outputs/plots/001_unet_components_comparison"

# EXPERIMENTS=(
#   "exp003_unet_components_01_baseline"
#   "exp003_unet_components_07_simple_encoder"
#   "exp003_unet_components_08_simple_bottleneck"
#   "exp003_unet_components_09_simple_decoder"
#   "exp003_unet_components_10_no_skip"
#   "exp005_focal_components_01_baseline"
#   "exp005_focal_components_07_simple_encoder"
#   "exp005_focal_components_08_simple_bottleneck"
#   "exp005_focal_components_09_simple_decoder"
#   "exp005_focal_components_10_no_skip"
#   "exp006_focal_reg_01_baseline"
#   "exp006_focal_reg_02_baseline_dropout_0.1"
#   "exp006_focal_reg_03_baseline_dropout_0.005"
#   "exp006_focal_reg_04_baseline_dropout_0.4"
#   "exp006_focal_reg_11_simple_encoder_decoder"
# )

# OUTPUT_DIR="outputs/plots/unet_components_comparison"

# EXPERIMENTS=(
#   "exp003_unet_components_01_baseline"
#   "exp003_unet_components_13_simple_encoder_decoder"
#   "exp005_focal_components_01_baseline"
#   "exp005_focal_components_11_simple_encoder_decoder"
#   "exp006_focal_reg_01_baseline"
#   "exp006_focal_reg_11_simple_encoder_decoder"
# )
#
# OUTPUT_DIR="outputs/plots/unet_baseline_vs_singleConv"




# EXPERIMENTS=(
#   "exp003_unet_components_01_baseline"
#   "exp003_unet_components_13_simple_encoder_decoder"
#   "exp005_focal_components_01_baseline"
#   "exp005_focal_components_11_simple_encoder_decoder"
#   "exp006_focal_reg_01_baseline"
#   "exp006_focal_reg_11_simple_encoder_decoder"
#   "exp007_dice_reg_01_baseline_dropout_0.2"
#   "exp007_dice_reg_03_simple_encoder_decoder_dropout_0.2"
# )
#
# OUTPUT_DIR="outputs/plots/003_baseline_vs_singleConv_dice_focal_reg"





  # "exp007_dice_reg_01_baseline_dropout_0.2"
  # "exp005_focal_components_01_baseline"
  # "exp005_focal_components_11_simple_encoder_decoder"
  # "exp006_focal_reg_01_baseline"
  # "exp006_focal_reg_11_simple_encoder_decoder"

# EXPERIMENTS=(
#   "exp003_unet_components_01_baseline"
#   "exp003_unet_components_07_simple_encoder"
#   "exp003_unet_components_08_simple_bottleneck"
#   "exp003_unet_components_09_simple_decoder"
#   "exp003_unet_components_10_no_skip"
#   "exp003_unet_components_13_simple_encoder_decoder"
#   "exp007_dice_reg_01_baseline_dropout_0.2"
#   "exp007_dice_reg_05_simple_encoder_dropout_0.2"
#   "exp007_dice_reg_03_simple_encoder_decoder_dropout_0.2"
# )

# OUTPUT_DIR="outputs/plots/unet_noReg_vs_Reg"


# EXPERIMENTS=(
#   "exp003_unet_components_01_baseline"
#   "exp003_unet_components_13_simple_encoder_decoder"
#   "exp005_focal_components_01_baseline"
#   "exp005_focal_components_11_simple_encoder_decoder"
#   "exp007_dice_reg_01_baseline_dropout_0.2"
#   "exp007_dice_reg_03_simple_encoder_decoder_dropout_0.2"
#   "exp006_focal_reg_01_baseline"
#   "exp006_focal_reg_11_simple_encoder_decoder"
# )
#
# OUTPUT_DIR="outputs/plots/dice_vs_focal_tversky"







EXPERIMENTS=(
  "exp003_unet_components_01_baseline"
  "exp007_dice_reg_01_baseline_dropout_0.2"
  "exp006_focal_reg_11_simple_encoder_decoder"
  "exp008_tinyswin_reg_dice"
  "exp008_tinyswin_reg_dice_patch2_embeddingDim64"
  "exp008_tinyswin_reg_focal_patch2_embeddingDim64"
  "exp008_tinyswin_reg_focal_patch4_embeddingDim128"
  "exp008_tinyswin_reg_dice_deep_noConvstem"
  "exp008_tinyswin_reg_focal_deep_noConvstem"
  "exp008_tinyswin_reg_dice_deep_convstem"
  "exp008_tinyswin_reg_focal_deep_convstem"
  "exp008_tinyswin_reg_dice_deep_convstem_k3_layer2"
  "exp008_tinyswin_reg_focal_deep_convstem_k3_layer2"
)

OUTPUT_DIR="outputs/plots/exp008_tinyswin_compare"







SHOW_FLAG=""


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

