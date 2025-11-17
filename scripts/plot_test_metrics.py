#!/usr/bin/env python3
"""
Plot test metrics across multiple experiments.

Usage examples:
    # Use the DEFAULT_EXPERIMENTS list defined below
    python scripts/plot_test_metrics.py

    # Plot every experiment that has outputs/tests/<exp>/test_metrics.yaml
    python scripts/plot_test_metrics.py --all

    # Plot only the experiments passed explicitly
    python scripts/plot_test_metrics.py --select exp004_auc_01_unet_baseline exp004_auc_02_unet_kernel5
"""

import argparse
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import yaml


# Edit this list to quickly choose which experiments to visualize by default.
DEFAULT_EXPERIMENTS: List[str] = [
    "exp001_1_basic_unet_green",
    "exp003_unet_components_01_baseline",
    "exp003_unet_components_02_reduced_depths",
    "exp003_unet_components_04_shallow",
    "exp003_unet_components_05_deep",
    "exp003_unet_components_06_limited_skip",
    "exp003_unet_components_07_simple_encoder",
    "exp003_unet_components_08_simple_bottleneck",
    "exp003_unet_components_09_simple_decoder",
    "exp003_unet_components_10_no_skip",
    "exp003_unet_components_11_kernel5x5",
    "exp003_unet_components_12_kernel7x7",
    "exp004_auc_01_unet_baseline",
    "exp004_auc_02_unet_kernel5",
    "exp004_auc_03_unet_shallow",
    "exp004_auc_04_unet_deep",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate test metrics from multiple experiments and plot each metric."
    )
    parser.add_argument(
        "--tests-dir",
        default="outputs/tests",
        type=str,
        help="Directory containing per-experiment test metrics (default: outputs/tests).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/plots/test_metrics",
        type=str,
        help="Directory where the plots will be saved (default: outputs/plots/test_metrics).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include every experiment that has a test_metrics.yaml file under --tests-dir.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        metavar="EXPERIMENT",
        help="Explicit list of experiments to include (space-separated).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def find_all_experiments(tests_dir: Path) -> List[str]:
    experiments = []
    if not tests_dir.exists():
        return experiments

    for subdir in sorted(tests_dir.iterdir()):
        metrics_file = subdir / "test_metrics.yaml"
        if metrics_file.exists():
            experiments.append(subdir.name)
    return experiments


def load_metrics(experiments: List[str], tests_dir: Path) -> Dict[str, Dict[str, float]]:
    data: Dict[str, Dict[str, float]] = {}
    for exp in experiments:
        metrics_path = tests_dir / exp / "test_metrics.yaml"
        if not metrics_path.exists():
            print(f"[WARN] Missing test_metrics.yaml for {exp}, skipping.")
            continue

        try:
            with open(metrics_path, "r") as f:
                summary = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            print(f"[WARN] Failed to parse YAML for {exp}: {exc}")
            continue

        avg_metrics = summary.get("average_metrics", {})
        if not avg_metrics:
            print(f"[WARN] No average_metrics found for {exp}, skipping.")
            continue

        data[exp] = {k: float(v) for k, v in avg_metrics.items()}
    return data


def plot_metrics(
    metrics_data: Dict[str, Dict[str, float]],
    output_dir: Path,
    show_plots: bool = False,
) -> None:
    if not metrics_data:
        print("[INFO] No metrics available to plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect the union of all metric names.
    metric_names: Set[str] = set()
    for per_exp_metrics in metrics_data.values():
        metric_names.update(per_exp_metrics.keys())

    metric_names_sorted = sorted(metric_names)

    for metric in metric_names_sorted:
        experiments = []
        values = []
        for exp_name, per_exp_metrics in metrics_data.items():
            if metric in per_exp_metrics:
                experiments.append(exp_name)
                values.append(per_exp_metrics[metric])

        if not experiments:
            continue

        plt.figure(figsize=(max(8, len(experiments) * 0.9), 5))
        bars = plt.bar(range(len(experiments)), values, color="#4C72B0")
        plt.xticks(range(len(experiments)), experiments, rotation=45, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Test {metric.upper()} across experiments")
        plt.ylim(0.6, 1.0)

        # Annotate bars with their values
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plot_path = output_dir / f"{metric}.png"
        plt.savefig(plot_path, dpi=200)
        if show_plots:
            plt.show()
        plt.close()
        print(f"[INFO] Saved plot: {plot_path}")

    # Combined figure with one subplot per metric
    if metric_names_sorted:
        num_metrics = len(metric_names_sorted)
        ncols = min(2, num_metrics)
        nrows = (num_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * 6, nrows * 4),
            squeeze=False
        )

        for idx, metric in enumerate(metric_names_sorted):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            experiments = []
            values = []
            for exp_name, per_exp_metrics in metrics_data.items():
                if metric in per_exp_metrics:
                    experiments.append(exp_name)
                    values.append(per_exp_metrics[metric])

            if not experiments:
                ax.axis("off")
                continue

            bars = ax.bar(range(len(experiments)), values, color="#4C72B0")
            ax.set_xticks(range(len(experiments)))
            ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0.6, 1.0)
            ax.set_title(metric.upper())
            ax.set_ylabel(metric.upper())

            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        # Hide any unused axes
        for idx in range(len(metric_names_sorted), nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].axis("off")

        fig.tight_layout()
        combined_path = output_dir / "metrics_combined.png"
        fig.savefig(combined_path, dpi=200)
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"[INFO] Saved combined plot: {combined_path}")


def main():
    args = parse_args()

    tests_dir = Path(args.tests_dir)
    output_dir = Path(args.output_dir)

    if args.all:
        experiments = find_all_experiments(tests_dir)
        print(f"[INFO] Using all experiments with metrics ({len(experiments)} found).")
    elif args.select:
        experiments = args.select
        print(f"[INFO] Using experiments provided via --select ({len(experiments)}).")
    else:
        experiments = DEFAULT_EXPERIMENTS
        print(f"[INFO] Using DEFAULT_EXPERIMENTS ({len(experiments)}).")

    if not experiments:
        print("[ERROR] No experiments specified. Use --all, --select, or edit DEFAULT_EXPERIMENTS.")
        return

    metrics_data = load_metrics(experiments, tests_dir)
    if not metrics_data:
        print("[ERROR] No metrics found. Make sure test_metrics.yaml exists for the selected experiments.")
        return

    plot_metrics(metrics_data, output_dir, show_plots=args.show)
    print("[INFO] Plot generation complete.")


if __name__ == "__main__":
    main()

