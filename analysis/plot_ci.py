"""
plot_ci.py — Voyager vs AutoSkill confidence interval comparison graph
Usage:
    python plot_ci.py
    python plot_ci.py --metric tech_tree_score
    python plot_ci.py --out results.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Run definitions ──────────────────────────────────────────────────────────
LOG_BASE = Path(__file__).resolve().parent.parent / "logs"

RUNS: dict[str, list[str]] = {
    "Voyager (main)": [
        # original 5 (submission)
        "run-2026-04-03_14-13-42",
        "run-2026-04-03_19-03-57",
        "run-2026-04-04_21-18-15",
        "run-2026-04-05_15-29-49",
        "run-2026-04-06_02-00-16",
        # 5 additional independent runs (n=10 stability)
        "run-resub-vb-seed1",
        "run-resub-vb-seed2",
        "run-resub-vb-seed3",
        "run-resub-vb-seed4",
        "run-resub-vb-seed5",
    ],
    "AutoSkill (ours)": [
        # original 5 (submission)
        "run-2026-04-08_17-04-48",
        "run-2026-04-12_02-43-00",
        "run-2026-04-12_11-00-21",
        "run-2026-05-14_18-47-19",
        "run-2026-05-14_02-05-05",
        # 5 additional independent runs (n=10 stability)
        "run-resub-fix2-seed6",
        "run-resub-fix2-seed7",
        "run-resub-fix2-seed8",
        "run-resub-fix2-seed9",
        "run-resub-fix2-seed10",
    ],
}

# Colorblind-safe (Wong) palette — Voyager amber, AutoSkill blue
COLORS: dict[str, str] = {
    "Voyager (main)":   "#E69F00",
    "AutoSkill (ours)": "#0072B2",
}
LINESTYLES: dict[str, str] = {
    "Voyager (main)":   "--",
    "AutoSkill (ours)": "-",
}


# ── IEEE Access-style figure config ─────────────────────────────────────────

def _apply_style() -> None:
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        # Remove dashed minor ticks that "science" style adds on top/right
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
    })


# ── Data loading ─────────────────────────────────────────────────────────────

def load_series(run_dir: str, metric: str) -> np.ndarray:
    """Load a metric series from step_log.jsonl. Forward-fill missing values."""
    path = LOG_BASE / run_dir / "step_log.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    values = []
    last = 0.0
    for r in rows:
        v = r.get(metric)
        if v is not None:
            last = float(v)
        values.append(last)

    if not any(r.get(metric) is not None for r in rows):
        raise ValueError(f"Metric '{metric}' not found in {path}")

    return np.array(values)


def align_and_stack(runs: list[str], metric: str) -> np.ndarray:
    """Stack trajectories to a common length.

    The metrics we plot (TTS, distinct items) are monotonically non-decreasing,
    so we pad shorter runs with their last value rather than truncating to the
    shortest run. This makes the trajectory mean at the final step coincide
    with the per-run max-TTS mean reported in the tables.
    """
    series_list = [load_series(r, metric) for r in runs]
    max_len = max(len(s) for s in series_list)
    padded = []
    for s in series_list:
        if len(s) < max_len:
            s = np.concatenate([s, np.full(max_len - len(s), s[-1])])
        padded.append(s)
    return np.stack(padded, axis=0)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot(metric: str = "distinct_items", out: str | None = None) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    legend_handles = []
    final_points = []

    for label, run_list in RUNS.items():
        color = COLORS[label]
        ls = LINESTYLES[label]
        data = align_and_stack(run_list, metric)
        n_runs, n_steps = data.shape
        xs = np.arange(1, n_steps + 1)

        mean = data.mean(axis=0)
        std  = data.std(axis=0, ddof=min(1, n_runs - 1))
        lo = mean - std
        hi = mean + std

        ax.plot(xs, mean, color=color, linestyle=ls, linewidth=1.8,
                label=label, zorder=3)
        ax.fill_between(xs, lo, hi, alpha=0.18, color=color, linewidth=0,
                        zorder=2)
        final_points.append((label, color, xs[-1], float(mean[-1])))

        patch = mpatches.Patch(facecolor=color, alpha=0.4)
        line  = plt.Line2D([0], [0], color=color, linewidth=1.8, linestyle=ls)
        legend_handles.append((line, patch, label))

    # Separated final-value annotations (avoid overlap)
    final_points.sort(key=lambda t: t[3])
    used_y = []
    for label, color, x_end, y_end in final_points:
        y_disp = y_end
        # nudge if too close to a previously placed annotation
        for yp in used_y:
            if abs(y_disp - yp) < (max(p[3] for p in final_points) * 0.05):
                y_disp += (max(p[3] for p in final_points) * 0.05)
        used_y.append(y_disp)
        ax.annotate(
            f"{y_end:.1f}",
            xy=(x_end, y_end), xytext=(7, 0),
            textcoords="offset points",
            fontsize=12, color=color, fontweight="bold",
            va="center", ha="left",
        )

    # Axes
    ylabel = {
        "distinct_items": "Number of Distinct Items",
        "tech_tree_score": "Tech Tree Score (TTS)",
    }.get(metric, metric)
    ax.set_xlabel("Number of Prompting Iterations")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    # leave a touch of headroom on the right for annotations
    cur_xmax = ax.get_xlim()[1]
    ax.set_xlim(0, cur_xmax * 1.04)

    # Legend
    from matplotlib.legend_handler import HandlerTuple
    proxies = [(l, p) for l, p, _ in legend_handles]
    labels  = [lb for _, __, lb in legend_handles]
    ax.legend(
        proxies, labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower right",
        frameon=True, fancybox=False,
        edgecolor="black", framealpha=0.95,
    )

    fig.tight_layout(pad=0.3)

    if out is None:
        out = f"ci_{metric}.pdf"
    # Save both pdf and png (paper + preview)
    base = str(out).rsplit(".", 1)[0]
    for fmt in ("pdf", "png"):
        path = f"{base}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CI comparison graph")
    parser.add_argument(
        "--metric", default="distinct_items",
        choices=["distinct_items", "tech_tree_score"],
        help="Metric to plot (default: distinct_items)",
    )
    parser.add_argument("--out", default=None,
                        help="Output file path (default: ci_<metric>.pdf)")
    args = parser.parse_args()
    plot(metric=args.metric, out=args.out)
