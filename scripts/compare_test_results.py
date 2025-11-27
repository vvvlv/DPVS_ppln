#!/usr/bin/env python3
"""
Compare test predictions from multiple experiments by visualizing the ground
truth mask, each experiment's prediction, and an overlap map that highlights:
    - True Positives (white)
    - False Positives (yellow)
    - False Negatives (red)

Currently optimized for comparing two experiments, but works with an arbitrary
number (>1). The script expects that test-time predictions are stored under
`outputs/tests/<experiment>/predictions/<image_name>.png` and that the ground
truth masks live under `<label_root>/<image_name>.png`.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


DEFAULT_TESTS_ROOT = Path("outputs/tests")
DEFAULT_LABEL_ROOT = Path("data/FIVES512_G/test/label")
DEFAULT_OUTPUT_DIR = Path("outputs/comparison_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare test predictions from multiple experiments."
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment names (must match folders under outputs/tests).",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Image / mask filename to visualize. If omitted, a random sample "
        "is selected from the label directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed when selecting a random sample.",
    )
    parser.add_argument(
        "--display-names",
        nargs="*",
        default=None,
        help="Optional short names for experiments (same length/order as experiments). "
        "If omitted, experiment IDs are used as labels.",
    )
    parser.add_argument(
        "--tests-root",
        type=Path,
        default=DEFAULT_TESTS_ROOT,
        help="Root directory containing outputs/tests (default: %(default)s).",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=None,
        help="Directory containing ground-truth masks (default derived from "
        "dataset config or data/FIVES512_G/test/label).",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Optional dataset config YAML to infer label directory "
        "(expects paths.test pointing to <root>/test).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to prediction masks (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure (PNG). If omitted, display.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory to save comparison figures when --output is not set "
        f"(default: {DEFAULT_OUTPUT_DIR}). Each experiment set gets its own "
        "subfolder inside this directory.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI when saving to file (default: %(default)s).",
    )
    return parser.parse_args()


def resolve_label_root(args: argparse.Namespace) -> Path:
    if args.label_root is not None:
        return args.label_root.resolve()

    if args.dataset_config is not None:
        config_path = args.dataset_config.resolve()
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        test_path = config.get("paths", {}).get("test")
        if not test_path:
            raise ValueError(
                f"Dataset config {config_path} does not define paths.test"
            )
        # Assume masks are stored under <test>/label
        label_root = Path(test_path).resolve() / "label"
        return label_root

    return (DEFAULT_LABEL_ROOT).resolve()


def resolve_image_root(label_root: Path) -> Path:
    """
    Infer image root from label root.
    Assumes structure: <root>/test/label and <root>/test/image.
    """
    return label_root.parent / "image"


def choose_sample(
    label_root: Path, explicit_sample: Optional[str], seed: Optional[int]
) -> str:
    if explicit_sample:
        sample_path = label_root / explicit_sample
        if not sample_path.exists():
            raise FileNotFoundError(f"Ground truth mask not found: {sample_path}")
        return explicit_sample

    if seed is not None:
        random.seed(seed)

    candidates = sorted([p.name for p in label_root.glob("*.png")])
    if not candidates:
        raise FileNotFoundError(f"No mask files found under {label_root}")

    return random.choice(candidates)


def load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as img:
        img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    # Always load as RGB for visualization; grayscale will be repeated across channels
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def binarize(mask: np.ndarray, threshold: float) -> np.ndarray:
    return (mask >= threshold).astype(np.uint8)


def create_comparison_overlay(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # Colors: TP white, FP yellow, FN red, background black
    overlay = np.zeros((*gt.shape, 3), dtype=np.float32)

    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)

    overlay[tp] = [1.0, 1.0, 1.0]  # white
    overlay[fp] = [1.0, 1.0, 0.0]  # yellow
    overlay[fn] = [1.0, 0.0, 0.0]  # red

    return overlay


def gather_predictions(
    experiments: List[str],
    tests_root: Path,
    sample_name: str,
    threshold: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for exp in experiments:
        pred_path = tests_root / exp / "predictions" / sample_name
        pred_raw = load_mask(pred_path)
        pred_bin = binarize(pred_raw, threshold)
        data[exp] = {
            "raw": pred_raw,
            "bin": pred_bin.astype(bool),
        }
    return data


def plot_results(
    input_image: np.ndarray,
    gt_mask: np.ndarray,
    predictions: Dict[str, Dict[str, np.ndarray]],
    row_labels: List[str],
    sample_name: str,
    output_path: Path | None,
    dpi: int,
    title: str | None = None,
) -> None:
    """
    Plot a grid where each row corresponds to an experiment:
      [Input | Label | Prediction | TP/FP/FN overlay]
    """
    experiments = list(predictions.keys())
    num_rows = len(experiments)
    num_cols = 4

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3 * num_cols, 2.2 * num_rows),
        constrained_layout=False,
    )

    axes = np.atleast_2d(axes)

    # Determine how to show input (RGB vs grayscale)
    input_is_gray = input_image.ndim == 2 or input_image.shape[2] == 1

    for row_idx, exp in enumerate(experiments):
        row_axes = axes[row_idx]

        # Input image
        if input_is_gray:
            row_axes[0].imshow(input_image.squeeze(), cmap="gray")
        else:
            row_axes[0].imshow(input_image)
        if row_idx == 0:
            row_axes[0].set_title("Input")
        row_axes[0].axis("off")

        # Ground truth
        row_axes[1].imshow(gt_mask, cmap="gray")
        if row_idx == 0:
            row_axes[1].set_title("Label")
        row_axes[1].axis("off")

        # Prediction and overlay
        pred_raw = predictions[exp]["raw"]
        pred_bin = predictions[exp]["bin"]
        overlay = create_comparison_overlay(gt_mask.astype(bool), pred_bin)

        row_axes[2].imshow(pred_raw, cmap="gray")
        if row_idx == 0:
            row_axes[2].set_title("Prediction")
        row_axes[2].axis("off")

        row_axes[3].imshow(overlay)
        if row_idx == 0:
            row_axes[3].set_title("TP (white) / FP (yellow) / FN (red)")
        row_axes[3].axis("off")

    # Tighten layout: reduce margins and spacing between subplots
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.90,
        bottom=0.06,
        wspace=0.08,
        hspace=0.12,
    )

    # Add experiment labels to the left of each row (vertical text)
    for row_idx, label in enumerate(row_labels):
        # y coordinate for center of this row in figure coordinates (top -> bottom)
        y = 1.0 - (row_idx + 0.5) / num_rows
        fig.text(
            0.03,
            y,
            label,
            va="center",
            ha="left",
            rotation=90,
            fontsize=9,
        )

    if title:
        fig.suptitle(title, fontsize=14)
    else:
        fig.suptitle(f"Sample {sample_name}", fontsize=14)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        print(f"Saved comparison figure to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def run_comparison(
    experiments: List[str],
    args: argparse.Namespace,
    tests_root: Path,
    label_root: Path,
) -> None:
    if len(experiments) < 2:
        raise ValueError("Each comparison must have at least two experiments.")

    sample_name = choose_sample(label_root, args.sample, args.seed)
    gt_path = label_root / sample_name
    gt_mask = load_mask(gt_path)
    gt_bin = binarize(gt_mask, threshold=0.5).astype(bool)

    image_root = resolve_image_root(label_root)
    image_path = image_root / sample_name
    input_image = load_image(image_path)

    print(f"\nComparing experiments: {', '.join(experiments)}")
    print(f"Sample chosen: {sample_name}")

    predictions = gather_predictions(
        experiments,
        tests_root,
        sample_name,
        args.threshold,
    )

    for exp, pred in predictions.items():
        if pred["raw"].shape != gt_mask.shape:
            raise ValueError(
                f"Shape mismatch for {exp}: "
                f"pred {pred['raw'].shape} vs gt {gt_mask.shape}"
            )

    # Decide output path
    output_path = args.output
    if output_path is None:
        # Group-specific folder based on experiment IDs
        group_name = "_vs_".join(experiments)
        base_dir = args.output_dir.resolve()
        output_dir = base_dir / group_name
        output_path = (output_dir / sample_name).with_suffix(".png")

    # Determine row labels (display names) for each experiment
    if args.display_names and len(args.display_names) == len(experiments):
        row_labels = args.display_names
    else:
        row_labels = experiments

    plot_results(
        input_image=input_image,
        gt_mask=gt_bin.astype(np.float32),
        predictions=predictions,
        row_labels=row_labels,
        sample_name=sample_name,
        output_path=output_path,
        dpi=args.dpi,
        title=args.title,
    )


def main() -> int:
    args = parse_args()

    tests_root = args.tests_root.resolve()
    label_root = resolve_label_root(args)

    run_comparison(args.experiments, args, tests_root, label_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

