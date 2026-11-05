"""
plot_dedup_efficiency.py — VI-E: Deduplication Efficiency
Single panel: skill library size over iterations (AutoSkill vs Voyager).
AutoSkill stays compact while reaching the same results.

Usage:
    python plot_dedup_efficiency.py
    python plot_dedup_efficiency.py --out dedup_efficiency.pdf
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

# ── Run definitions ───────────────────────────────────────────────────────────
LOG_BASE = Path(__file__).resolve().parent.parent / "logs"

RUNS: dict[str, list[str]] = {
    "Voyager (main)": [
        "run-2026-04-03_14-13-42",
        "run-2026-04-03_19-03-57",
        "run-2026-04-04_21-18-15",
        "run-2026-04-05_15-29-49",
        "run-2026-04-06_02-00-16",
        "run-resub-vb-seed1",
        "run-resub-vb-seed2",
        "run-resub-vb-seed3",
        "run-resub-vb-seed4",
        "run-resub-vb-seed5",
    ],
    "AutoSkill (ours)": [
        "run-2026-04-08_17-04-48",
        "run-2026-04-12_02-43-00",
        "run-2026-04-12_11-00-21",
        "run-2026-05-14_18-47-19",
        "run-2026-05-14_02-05-05",
        "run-resub-fix2-seed6",
        "run-resub-fix2-seed7",
        "run-resub-fix2-seed8",
        "run-resub-fix2-seed9",
        "run-resub-fix2-seed10",
    ],
}

# Colorblind-safe (Wong) — consistent with plot_ci.py
COLORS = {"Voyager (main)": "#E69F00", "AutoSkill (ours)": "#0072B2"}
LINESTYLES = {"Voyager (main)": "--",  "AutoSkill (ours)": "-"}


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
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
    })


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_step_log(run_dir: str) -> list[dict]:
    path = LOG_BASE / run_dir / "step_log.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def lib_size_series(run_dir: str) -> np.ndarray:
    rows = load_step_log(run_dir)
    return np.array([r.get("skill_library_size", 0) for r in rows], dtype=float)


def align_stack(series_list: list[np.ndarray]) -> np.ndarray:
    # Pad shorter runs with their last value so the final step reflects
    # every run's final library size (truncation would drop late growth).
    max_len = max(len(s) for s in series_list)
    return np.stack([np.pad(s, (0, max_len - len(s)), mode="edge")
                     for s in series_list], axis=0)


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot(out: str | None = None) -> None:
    _apply_style()

    lib_data: dict[str, np.ndarray] = {}
    for label, run_list in RUNS.items():
        mat = align_stack([lib_size_series(r) for r in run_list])
        lib_data[label] = mat

    n_steps = min(lib_data["Voyager (main)"].shape[1],
                  lib_data["AutoSkill (ours)"].shape[1])
    xs = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    legend_handles = []
    final_vals = {}

    for label, mat in lib_data.items():
        color = COLORS[label]
        ls = LINESTYLES[label]
        mean = mat[:, :n_steps].mean(axis=0)
        std  = mat[:, :n_steps].std(axis=0, ddof=1)

        ax.plot(xs, mean, color=color, linestyle=ls, linewidth=1.8,
                label=label, zorder=3)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=color,
                        linewidth=0, zorder=2)
        final_vals[label] = float(mean[-1])

        patch = mpatches.Patch(facecolor=color, alpha=0.4)
        line  = plt.Line2D([0], [0], color=color, linewidth=1.8, linestyle=ls)
        legend_handles.append((line, patch, label))

    # Final-value labels at right edge
    for label, val in final_vals.items():
        ax.annotate(
            f"{val:.0f}",
            xy=(xs[-1], val), xytext=(7, 0),
            textcoords="offset points",
            fontsize=12, color=COLORS[label], fontweight="bold",
            va="center", ha="left",
        )

    v_final = final_vals["Voyager (main)"]
    a_final = final_vals["AutoSkill (ours)"]

    ax.set_xlabel("Number of Prompting Iterations")
    ax.set_ylabel("Skill Library Size")
    ax.set_xlim(0, xs[-1] * 1.04)
    ax.set_ylim(0, None)

    from matplotlib.legend_handler import HandlerTuple
    proxies = [(l, p) for l, p, _ in legend_handles]
    labels  = [lb for _, __, lb in legend_handles]
    ax.legend(
        proxies, labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
        frameon=True, fancybox=False,
        edgecolor="black", framealpha=0.95,
    )

    fig.tight_layout(pad=0.3)

    base = (out or "dedup_efficiency.pdf").rsplit(".", 1)[0]
    for fmt in ("pdf", "png"):
        path = f"{base}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n=== Deduplication Efficiency Summary ===")
    for label, mat in lib_data.items():
        final = mat[:, :n_steps].mean(axis=0)[-1]
        std   = mat[:, :n_steps].std(axis=0, ddof=1)[-1]
        print(f"  {label:22s} final library size: {final:.1f} ± {std:.1f}")
    print(f"\n  AutoSkill keeps {v_final - a_final:.0f} fewer skills "
          f"({(1 - a_final/v_final)*100:.1f}% smaller library)")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot deduplication efficiency")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: dedup_efficiency.pdf)")
    args = parser.parse_args()
    plot(out=args.out)
