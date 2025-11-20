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
DEFAULT_OUTPUT_DIR = Path("outputs/plots/experiment_comparisons")


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
    gt_mask: np.ndarray,
    predictions: Dict[str, Dict[str, np.ndarray]],
    sample_name: str,
    output_path: Path | None,
    dpi: int,
    title: str | None = None,
):
    experiments = list(predictions.keys())
    num_cols = 1 + len(experiments) * 2
    fig, axes = plt.subplots(
        1,
        num_cols,
        figsize=(4 * num_cols, 4),
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes)
    axes[0].imshow(gt_mask, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    for idx, exp in enumerate(experiments):
        pred_raw = predictions[exp]["raw"]
        pred_bin = predictions[exp]["bin"]
        overlay = create_comparison_overlay(gt_mask.astype(bool), pred_bin)

        col_pred = 1 + idx * 2
        col_overlay = col_pred + 1

        axes[col_pred].imshow(pred_raw, cmap="gray")
        axes[col_pred].set_title(f"{exp}\nPrediction")
        axes[col_pred].axis("off")

        axes[col_overlay].imshow(overlay)
        axes[col_overlay].set_title(f"{exp}\nTP/FP/FN")
        axes[col_overlay].axis("off")

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


def main() -> int:
    args = parse_args()

    if len(args.experiments) < 2:
        print("Please provide at least two experiments to compare.", file=sys.stderr)
        return 1

    tests_root = args.tests_root.resolve()
    label_root = resolve_label_root(args)

    sample_name = choose_sample(label_root, args.sample, args.seed)
    gt_path = label_root / sample_name
    gt_mask = load_mask(gt_path)
    gt_bin = binarize(gt_mask, threshold=0.5).astype(bool)

    predictions = gather_predictions(
        args.experiments,
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

    output_path = args.output
    if output_path is None:
        output_path = (
            DEFAULT_OUTPUT_DIR
            / f"compare_{'_vs_'.join(args.experiments)}_{sample_name}"
        ).with_suffix(".png")

    plot_results(
        gt_bin.astype(np.float32),
        predictions,
        sample_name,
        output_path=output_path,
        dpi=args.dpi,
        title=args.title,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

