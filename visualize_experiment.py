#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment result visualization script

Supports three modes:
  1. Compare LLMs:   Compare optimization curves of all LLMs under one algorithm
  2. All iterations: Plot optimization curve for one LLM (single series)
  3. Single experiment: Plot detailed training curves for one experiment

Usage:
  # Compare all LLMs under ViT (auto-discovers subdirectories)
  python visualize_experiment.py --algorithm ViT

  # Compare specific LLMs
  python visualize_experiment.py --algorithm ViT --llm GPT-5.4 gemini-3-pro-preview claude-opus-4-6

  # Single LLM optimization curve (same as old behavior)
  python visualize_experiment.py --algorithm ViT --llm GPT-5.4 --single

  # Plot training curve for a specific experiment
  python visualize_experiment.py --algorithm ViT --llm GPT-5.4 --experiment 2

  # Specify output file
  python visualize_experiment.py --algorithm ViT -o my_plot.png
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Default output directory for all generated plots
VISUALIZATIONS_DIR = Path(__file__).parent / "visualizations"

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─── Color Palette ─────────────────────────────────────────

LLM_COLORS = {
    "GPT-5.4":               "#10A37F",  # OpenAI green
    "GPT-5":                 "#10A37F",
    "gpt-5":                 "#10A37F",
    "gemini-3-pro-preview":  "#4285F4",  # Google blue
    "gemini-3-pro":          "#4285F4",
    "claude-opus-4-6":       "#D97706",  # Anthropic amber
    "claude-sonnet-4":       "#D97706",
}

FALLBACK_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#E91E63", "#8BC34A", "#FF5722", "#607D8B",
]


def get_llm_color(llm_name: str, idx: int) -> str:
    """Get a consistent color for each LLM."""
    return LLM_COLORS.get(llm_name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ─── Evaluator Loader ─────────────────────────────────────────

def load_evaluator(algorithm: str):
    """Load the evaluator for the given algorithm to determine score direction."""
    algo_dir = Path(__file__).parent / "algorithms" / algorithm
    evaluator_path = algo_dir / "evaluator.py"
    if not evaluator_path.exists():
        return None

    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluator_module", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None

    # Find the evaluator class
    from autoresearch import BaseEvaluator
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and issubclass(attr, BaseEvaluator)
                and attr is not BaseEvaluator):
            return attr()
    return None


# ─── Data Parsing ─────────────────────────────────────────

def parse_epoch_metrics(stdout: str) -> list[dict]:
    """
    Extract per-epoch metrics from training log stdout.
    Matches format: [epNNN] train_loss=... train_acc1=... val_loss=... val_acc1=...
    """
    pattern = re.compile(
        r"\[ep(\d+)\]\s+"
        r"train_loss=([\d.]+)\s+"
        r"train_acc1=([\d.]+)\s+"
        r"val_loss=([\d.]+)\s+"
        r"val_acc1=([\d.]+)"
    )
    metrics = []
    for m in pattern.finditer(stdout):
        metrics.append({
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "train_acc1": float(m.group(3)),
            "val_loss": float(m.group(4)),
            "val_acc1": float(m.group(5)),
        })
    return metrics


def load_results_json(directory: Path) -> list[dict]:
    """Load results.json summary file."""
    results_path = directory / "results.json"
    if not results_path.exists():
        return []
    with open(results_path) as f:
        return json.load(f)


def load_single_experiment(directory: Path, exp_id: int) -> Optional[dict]:
    """Find a specific experiment from results.json, or load from experiment directory."""
    results_path = directory / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        for r in all_results:
            if r.get("experiment_id") == exp_id:
                return r.get("result", {}).get("training_result", {})

    for d in sorted(directory.iterdir()):
        if d.is_dir() and d.name.startswith(f"experiment_{exp_id:03d}_"):
            result_file = d / "training_result.json"
            if result_file.exists():
                with open(result_file) as f:
                    return json.load(f)
    return None


def extract_scores(all_results: list[dict], higher_is_better: bool = True):
    """
    Extract valid and failed iterations from results.

    Returns:
        valid_iterations: list of (exp_id, score)
        failed_iterations: list of (exp_id, score)
    """
    valid = []
    failed = []
    fail_score = 0.0 if higher_is_better else float("inf")

    for r in all_results:
        exp_id = r.get("experiment_id", 0)
        tr = r.get("result", {}).get("training_result", {})
        ev = r.get("result", {}).get("evaluation", {})
        score = ev.get("score")
        if score is None:
            score = tr.get("best_val_acc1", fail_score)

        success = tr.get("success", True)
        return_code = tr.get("return_code", 0)

        if higher_is_better:
            is_failed = (not success) or (return_code != 0) or (score is None) or (score == 0)
        else:
            is_failed = (not success) or (return_code != 0) or (score is None) or (score == float("inf"))

        if is_failed:
            failed.append((exp_id, score if score is not None else fail_score))
        else:
            valid.append((exp_id, float(score)))

    valid.sort(key=lambda x: x[0])
    failed.sort(key=lambda x: x[0])
    return valid, failed


# ─── Plotting Functions ─────────────────────────────────────────

def plot_single_experiment(algorithm: str, exp_id: int, training_result: dict, output: str):
    """
    Plot training curves for a single experiment:
    - Top subplot: train_loss / val_loss
    - Bottom subplot: train_acc1 / val_acc1
    """
    stdout = training_result.get("stdout", "")
    metrics = parse_epoch_metrics(stdout)

    if not metrics:
        print("❌ No epoch metrics found in training log")
        sys.exit(1)

    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m["val_loss"] for m in metrics]
    train_acc = [m["train_acc1"] for m in metrics]
    val_acc = [m["val_acc1"] for m in metrics]

    best_val_acc = training_result.get("best_val_acc1", max(val_acc))
    best_epoch = val_acc.index(max(val_acc)) + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"{algorithm} — Experiment #{exp_id}  (best val_acc1={best_val_acc:.4f})",
        fontsize=14, fontweight="bold",
    )

    # ── Loss Curve ──
    ax1.plot(epochs, train_loss, "o-", color="#2196F3", markersize=2, linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, val_loss, "s-", color="#F44336", markersize=2, linewidth=1.5, label="Val Loss")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Loss Curve", fontsize=11)

    # ── Accuracy Curve ──
    ax2.plot(epochs, train_acc, "o-", color="#2196F3", markersize=2, linewidth=1.5, label="Train Acc@1")
    ax2.plot(epochs, val_acc, "s-", color="#F44336", markersize=2, linewidth=1.5, label="Val Acc@1")
    ax2.axhline(y=best_val_acc, color="#4CAF50", linestyle="--", alpha=0.7, label=f"Best Val={best_val_acc:.4f}")
    ax2.axvline(x=best_epoch, color="#4CAF50", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Accuracy Curve", fontsize=11)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output}")
    plt.close()


def plot_single_llm(algorithm: str, all_results: list[dict], output: str,
                    higher_is_better: bool = True, score_label: str = "Score"):
    """
    Plot optimization curve for a single LLM across all iterations.
    """
    valid, failed = extract_scores(all_results, higher_is_better)

    if not valid:
        print("❌ No valid experiment results found")
        sys.exit(1)

    valid_ids = [it[0] for it in valid]
    valid_scores = [it[1] for it in valid]
    failed_ids = [it[0] for it in failed]
    failed_scores = [it[1] for it in failed]

    # Best-so-far envelope
    best_so_far = []
    cmp = max if higher_is_better else min
    current_best = valid_scores[0]
    for s in valid_scores:
        current_best = cmp(current_best, s)
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(valid_ids, valid_scores, "o-", color="#2196F3", markersize=6, linewidth=1.5,
            label=f"{score_label} (n={len(valid)})", zorder=3)
    ax.plot(valid_ids, best_so_far, "s--", color="#4CAF50", markersize=4, linewidth=2,
            label="Best so far", alpha=0.8, zorder=2)

    if failed:
        ax.scatter(failed_ids, failed_scores, marker="x", color="#9E9E9E", s=80, linewidths=2,
                   label=f"Failed (n={len(failed)})", zorder=4)

    # Annotate best point
    best_val = cmp(valid_scores)
    best_idx = valid_scores.index(best_val)
    ax.annotate(
        f"Best: {best_val:.4f}\n(iter #{valid_ids[best_idx]})",
        xy=(valid_ids[best_idx], best_val),
        xytext=(15, 15 if higher_is_better else -25),
        textcoords="offset points", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F44336"),
    )

    ax.axhline(y=valid_scores[0], color="#FF9800", linestyle=":", alpha=0.6,
               label=f"Baseline: {valid_scores[0]:.4f}")

    ax.set_xlabel("Iteration (Experiment ID)", fontsize=12)
    ax.set_ylabel(score_label, fontsize=12)
    ax.set_title(f"{algorithm} — Optimization Curve ({len(valid)} valid / {len(valid) + len(failed)} total)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right" if higher_is_better else "upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output}")
    plt.close()


def plot_compare_llms(algorithm: str, llm_data: dict[str, list[dict]], output: str,
                      higher_is_better: bool = True, score_label: str = "Score"):
    """
    Plot optimization curves for multiple LLMs on the same chart.

    Args:
        algorithm: Algorithm name
        llm_data: {llm_name: results_json_list}
        higher_is_better: Score direction
        score_label: Y-axis label
        output: Output image path
    """
    cmp = max if higher_is_better else min

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [3, 1]})
    ax_curve = axes[0]
    ax_bar = axes[1]

    direction_hint = "↑ higher is better" if higher_is_better else "↓ lower is better"
    fig.suptitle(
        f"{algorithm} — Multi-LLM Comparison  ({direction_hint})",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # ── Left panel: optimization curves ──
    summary_rows = []  # For stats
    all_scores = []  # For axis range

    for idx, (llm_name, results) in enumerate(sorted(llm_data.items())):
        valid, failed = extract_scores(results, higher_is_better)
        if not valid:
            print(f"   ⚠️ {llm_name}: no valid results, skipping")
            continue

        color = get_llm_color(llm_name, idx)
        valid_ids = [it[0] for it in valid]
        valid_scores = [it[1] for it in valid]
        all_scores.extend(valid_scores)

        # Best-so-far envelope
        best_so_far = []
        current_best = valid_scores[0]
        for s in valid_scores:
            current_best = cmp(current_best, s)
            best_so_far.append(current_best)

        # Score per iteration
        ax_curve.plot(
            valid_ids, valid_scores, "o-",
            color=color, markersize=5, linewidth=1.2, alpha=0.5,
            label=f"{llm_name}",
        )
        # Best-so-far line (thicker, more prominent)
        ax_curve.plot(
            valid_ids, best_so_far, "s-",
            color=color, markersize=3, linewidth=2.5,
            label=f"{llm_name} (best-so-far)",
        )

        # Failed experiments
        if failed:
            failed_ids = [it[0] for it in failed]
            failed_scores = [it[1] for it in failed]
            ax_curve.scatter(
                failed_ids, failed_scores,
                marker="x", color=color, s=60, linewidths=2, alpha=0.4,
            )

        # Collect summary
        baseline = valid_scores[0]
        best_val = best_so_far[-1]
        improvement = best_val - baseline
        summary_rows.append({
            "llm": llm_name,
            "baseline": baseline,
            "best": best_val,
            "improvement": improvement,
            "n_valid": len(valid),
            "n_failed": len(failed),
            "color": color,
        })

    ax_curve.set_xlabel("Iteration (Experiment ID)", fontsize=12)
    ax_curve.set_ylabel(score_label, fontsize=12)
    ax_curve.legend(fontsize=9, loc="lower right" if higher_is_better else "upper right",
                    ncol=1, framealpha=0.9)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_curve.yaxis.get_major_formatter().set_useOffset(False)  # Disable offset notation
    ax_curve.set_title("Optimization Curves", fontsize=12)

    # ── Right panel: summary bar chart ──
    if summary_rows:
        # Sort by best score
        summary_rows.sort(key=lambda r: r["best"], reverse=higher_is_better)

        llm_names = [r["llm"] for r in summary_rows]
        baselines = [r["baseline"] for r in summary_rows]
        bests = [r["best"] for r in summary_rows]
        colors = [r["color"] for r in summary_rows]

        y_pos = range(len(llm_names))

        # Bar chart: baseline vs best
        bar_height = 0.35
        bars_baseline = ax_bar.barh(
            [y - bar_height / 2 for y in y_pos], baselines,
            height=bar_height, color=[c + "60" for c in colors],
            label="Baseline", edgecolor=[c for c in colors], linewidth=1,
        )
        bars_best = ax_bar.barh(
            [y + bar_height / 2 for y in y_pos], bests,
            height=bar_height, color=colors,
            label="Best", edgecolor=colors, linewidth=1,
        )

        # Annotate bars with values
        for i, row in enumerate(summary_rows):
            ax_bar.text(
                row["best"], i + bar_height / 2,
                f" {row['best']:.4f}",
                va="center", ha="left", fontsize=9, fontweight="bold",
            )
            imp = row["improvement"]
            if higher_is_better:
                imp_text = f"+{imp:.4f}" if imp > 0 else f"{imp:.4f}"
            else:
                imp_text = f"{imp:.4f}" if imp < 0 else f"+{imp:.4f}"

            is_improved = (imp > 0 and higher_is_better) or (imp < 0 and not higher_is_better)
            imp_color = "#4CAF50" if is_improved else "#F44336"

            ax_bar.text(
                max(row["best"], row["baseline"]), i,
                f"  ({imp_text})",
                va="center", ha="left", fontsize=8, color=imp_color,
            )

        ax_bar.set_yticks(list(y_pos))
        ax_bar.set_yticklabels(llm_names, fontsize=10)
        ax_bar.set_xlabel(score_label, fontsize=11)
        ax_bar.set_title("Best Score Comparison", fontsize=12)
        ax_bar.legend(fontsize=9, loc="lower right" if higher_is_better else "lower left")
        ax_bar.grid(True, axis="x", alpha=0.3)
        ax_bar.invert_yaxis()  # Top = best

        # Print summary table to console
        print(f"\n{'─' * 70}")
        print(f"  {algorithm} — LLM Comparison Summary")
        print(f"{'─' * 70}")
        header = f"  {'LLM':<28s} {'Baseline':>10s} {'Best':>10s} {'Δ':>10s} {'Runs':>6s}"
        print(header)
        print(f"  {'─' * 66}")
        for row in summary_rows:
            imp = row["improvement"]
            if higher_is_better:
                imp_text = f"+{imp:.4f}" if imp > 0 else f"{imp:.4f}"
            else:
                imp_text = f"{imp:.4f}" if imp < 0 else f"+{imp:.4f}"
            runs = f"{row['n_valid']}/{row['n_valid'] + row['n_failed']}"
            print(f"  {row['llm']:<28s} {row['baseline']:>10.4f} {row['best']:>10.4f} {imp_text:>10s} {runs:>6s}")
        print(f"{'─' * 70}\n")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {output}")
    plt.close()


# ─── Main Entry ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment result visualization: compare LLMs or plot optimization/training curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all LLMs under ViT (auto-discovers subdirectories)
  python visualize_experiment.py --algorithm ViT

  # Compare specific LLMs
  python visualize_experiment.py --algorithm ViT --llm GPT-5.4 gemini-3-pro-preview

  # Single LLM optimization curve
  python visualize_experiment.py --algorithm ViT --llm GPT-5.4 --single

  # Plot training curve for a specific experiment
  python visualize_experiment.py --algorithm ViT --llm GPT-5.4 --experiment 2

  # Specify output path
  python visualize_experiment.py --algorithm nanochat -o nanochat_compare.png
        """,
    )
    parser.add_argument(
        "--algorithm", "-a", required=True,
        help="Algorithm name, corresponding to directory under experiments/ (e.g. ViT, nanochat)",
    )
    parser.add_argument(
        "--llm", nargs="*", default=None,
        help="LLM model names to compare. If not specified, auto-discovers all LLMs. "
             "Use --single to plot one LLM alone.",
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Plot single LLM optimization curve instead of multi-LLM comparison. "
             "Requires exactly one --llm argument.",
    )
    parser.add_argument(
        "--experiment", "-e", type=int, default=None,
        help="Plot detailed training curve for a specific experiment number. "
             "Requires exactly one --llm argument.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output image path (auto-generated by default)",
    )

    args = parser.parse_args()

    experiments_root = Path(__file__).parent / "experiments"
    algo_dir = experiments_root / args.algorithm

    # Ensure visualizations output directory exists
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not algo_dir.exists():
        print(f"❌ Algorithm directory not found: {algo_dir}")
        available = [d.name for d in experiments_root.iterdir() if d.is_dir()]
        if available:
            print(f"   Available algorithms: {', '.join(available)}")
        sys.exit(1)

    # Load evaluator to determine score direction
    evaluator = load_evaluator(args.algorithm)
    higher_is_better = evaluator.higher_is_better if evaluator else True
    score_label = "Score"
    if evaluator:
        if not higher_is_better:
            score_label = "Score (lower is better)"
        else:
            score_label = "Score (higher is better)"

    # ── Mode 1: Single experiment training curve ──
    if args.experiment is not None:
        if not args.llm or len(args.llm) != 1:
            print("❌ --experiment requires exactly one --llm argument")
            sys.exit(1)
        llm_name = args.llm[0]
        llm_dir = algo_dir / llm_name
        if not llm_dir.exists():
            print(f"❌ LLM directory not found: {llm_dir}")
            sys.exit(1)

        tr = load_single_experiment(llm_dir, args.experiment)
        if tr is None:
            print(f"❌ Experiment #{args.experiment} not found")
            sys.exit(1)

        title_label = f"{args.algorithm} ({llm_name})"
        output = args.output or str(VISUALIZATIONS_DIR / f"{args.algorithm}_{llm_name}_{args.experiment:03d}_training.png")
        plot_single_experiment(title_label, args.experiment, tr, output)
        return

    # ── Mode 2: Single LLM optimization curve ──
    if args.single:
        if not args.llm or len(args.llm) != 1:
            print("❌ --single requires exactly one --llm argument")
            sys.exit(1)
        llm_name = args.llm[0]
        llm_dir = algo_dir / llm_name
        if not llm_dir.exists():
            print(f"❌ LLM directory not found: {llm_dir}")
            sys.exit(1)

        all_results = load_results_json(llm_dir)
        if not all_results:
            print(f"❌ No results found in {llm_dir}")
            sys.exit(1)

        title_label = f"{args.algorithm} ({llm_name})"
        output = args.output or str(VISUALIZATIONS_DIR / f"{args.algorithm}_{llm_name}_optimization.png")
        plot_single_llm(title_label, all_results, output, higher_is_better, score_label)
        return

    # ── Mode 3: Multi-LLM comparison (default) ──
    # Discover or use specified LLMs
    if args.llm:
        llm_names = args.llm
    else:
        # Auto-discover: any subdirectory with results.json
        llm_names = sorted([
            d.name for d in algo_dir.iterdir()
            if d.is_dir() and (d / "results.json").exists()
        ])
        if not llm_names:
            print(f"❌ No LLM result directories found under {algo_dir}")
            available = [d.name for d in algo_dir.iterdir() if d.is_dir()]
            if available:
                print(f"   Found directories: {', '.join(available)}")
                print("   (None of them contain results.json)")
            sys.exit(1)

    # Load all results
    llm_data = {}
    for llm_name in llm_names:
        llm_dir = algo_dir / llm_name
        if not llm_dir.exists():
            print(f"   ⚠️ Directory not found: {llm_dir}, skipping")
            continue
        results = load_results_json(llm_dir)
        if not results:
            print(f"   ⚠️ No results in {llm_dir}, skipping")
            continue
        llm_data[llm_name] = results

    if not llm_data:
        print("❌ No valid LLM results to compare")
        sys.exit(1)

    if len(llm_data) == 1:
        # Only one LLM found, fall back to single-LLM plot
        llm_name = list(llm_data.keys())[0]
        title_label = f"{args.algorithm} ({llm_name})"
        output = args.output or str(VISUALIZATIONS_DIR / f"{args.algorithm}_{llm_name}_optimization.png")
        plot_single_llm(title_label, llm_data[llm_name], output, higher_is_better, score_label)
    else:
        output = args.output or str(VISUALIZATIONS_DIR / f"{args.algorithm}_compare_llms.png")
        plot_compare_llms(args.algorithm, llm_data, output, higher_is_better, score_label)


if __name__ == "__main__":
    main()
