"""
plot_ablation_ci.py — Ablation study learning curves (4 conditions).

Plots cumulative TTS over 150 prompting iterations for:
  - Voyager (baseline, no dedup + no critique)
  - w/o Dedup (AutoSkill minus dedup gate)
  - w/o Critique (AutoSkill minus reuse-aware critique)
  - AutoSkill (full system)

n=5 runs per condition, mean ± 1 std bands.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOG_BASE = Path(__file__).resolve().parent.parent / "logs"

_N5: dict[str, list[str]] = {
    "w/o Dedup": [
        "run-2026-04-12_17-47-11",
        "run-2026-04-13_00-57-58",
        "run-2026-04-16_11-19-26",
        "run-2026-05-14_13-36-20",
        "run-2026-05-14_22-52-31-slot2",
    ],
    "w/o Critique": [
        "run-2026-04-12_21-26-54",
        "run-2026-04-13_04-53-52",
        "run-2026-04-13_09-04-02",
        "run-2026-05-14_07-14-01",
        "run-2026-05-14_22-27-55-slot1",
    ],
}
from plot_dedup_efficiency import RUNS as _MAIN
# n=10 per condition: original five ablation runs + round-2 seeds 6-10; w/o Curriculum = round-2 seeds 1-10
RUNS: dict[str, list[str]] = {
    "Voyager": _MAIN["Voyager (main)"],
    "w/o Dedup": _N5["w/o Dedup"] + [f"run-r2-mini-nd-seed{i}" for i in range(6, 11)],
    "w/o Critique": _N5["w/o Critique"] + [f"run-r2-mini-nc-seed{i}" for i in range(6, 11)],
    "w/o Curriculum": [f"run-r2-mini-ncur-seed{i}" for i in range(1, 11)],
    "AutoSkill": _MAIN["AutoSkill (ours)"],
}

# Wong colorblind-safe palette
COLORS = {
    "Voyager":      "#E69F00",   # amber (baseline)
    "w/o Dedup":    "#009E73",   # green
    "w/o Critique": "#CC79A7",   # pink/purple
    "w/o Curriculum": "#D55E00", # vermillion
    "AutoSkill":    "#0072B2",   # blue (full system)
}
LINESTYLES = {
    "Voyager":      "--",
    "w/o Dedup":    "-.",
    "w/o Critique": ":",
    "w/o Curriculum": (0, (5, 1, 1, 1, 1, 1)),
    "AutoSkill":    "-",
}
ZORDER = {
    "Voyager":      2,
    "w/o Dedup":    3,
    "w/o Critique": 3,
    "w/o Curriculum": 3,
    "AutoSkill":    4,
}


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


def load_series(run_dir: str, metric: str = "tech_tree_score") -> np.ndarray:
    path = LOG_BASE / run_dir / "step_log.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    values, last = [], 0.0
    for r in rows:
        v = r.get(metric)
        if v is not None:
            last = float(v)
        values.append(last)
    return np.array(values)


def align_and_stack(runs: list[str], metric: str) -> np.ndarray:
    series = [load_series(r, metric) for r in runs]
    max_len = max(len(s) for s in series)
    padded = []
    for s in series:
        if len(s) < max_len:
            s = np.concatenate([s, np.full(max_len - len(s), s[-1])])
        padded.append(s)
    return np.stack(padded, axis=0)


def plot(out: str = "ablation_ci.pdf") -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    legend_handles = []
    final_points = []

    for label, run_list in RUNS.items():
        color = COLORS[label]
        ls    = LINESTYLES[label]
        z     = ZORDER[label]
        data  = align_and_stack(run_list, "tech_tree_score")
        n_runs, n_steps = data.shape
        xs = np.arange(1, n_steps + 1)

        mean = data.mean(axis=0)
        std  = data.std(axis=0, ddof=min(1, n_runs - 1))
        lo   = mean - std
        hi   = mean + std

        ax.plot(xs, mean, color=color, linestyle=ls, linewidth=1.8,
                label=label, zorder=z + 1)
        ax.fill_between(xs, lo, hi, alpha=0.12, color=color, linewidth=0,
                        zorder=z)
        final_points.append((label, color, xs[-1], float(mean[-1])))

        patch = mpatches.Patch(facecolor=color, alpha=0.4)
        line  = plt.Line2D([0], [0], color=color, linewidth=1.8, linestyle=ls)
        legend_handles.append((line, patch, label))

    # Final-value annotations (avoid overlap by sorting then iteratively
    # pushing each label below or above its neighbours)
    final_points.sort(key=lambda t: t[3])  # low to high
    ymax = max(p[3] for p in final_points)
    spacing = ymax * 0.055
    placed = []  # list of (y_disp,)
    for label, color, x_end, y_end in final_points:
        y_disp = y_end
        # ensure separation from every already-placed annotation
        changed = True
        while changed:
            changed = False
            for yp in placed:
                if abs(y_disp - yp) < spacing:
                    y_disp = yp + spacing
                    changed = True
        placed.append(y_disp)
        ax.annotate(
            f"{y_end:.1f}",
            xy=(x_end, y_end), xytext=(8, (y_disp - y_end) * (300 / ymax)),
            textcoords="offset points",
            fontsize=11, color=color, fontweight="bold",
            va="center", ha="left",
        )

    ax.set_xlabel("Number of Prompting Iterations")
    ax.set_ylabel("Tech Tree Score (TTS)")
    # Zoom in on iterations >= 80, where the four conditions are visibly
    # separated; the early-iteration region (all curves bunched near zero)
    # is uninformative for the ablation comparison. We also tighten the
    # y-axis to the range actually occupied by the curves in this window
    # so the cross-method gap is not visually compressed.
    cur_xmax = max(p[2] for p in final_points)
    ax.set_xlim(80, cur_xmax * 1.05)
    ax.set_ylim(50, max(p[3] for p in final_points) * 1.08)

    from matplotlib.legend_handler import HandlerTuple
    proxies = [(l, p) for l, p, _ in legend_handles]
    labels  = [lb for _, _, lb in legend_handles]
    ax.legend(
        proxies, labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower right",
        frameon=True, fancybox=False,
        edgecolor="black", framealpha=0.95,
        ncol=1,
    )

    fig.tight_layout(pad=0.3)

    base = out.rsplit(".", 1)[0]
    for fmt in ("pdf", "png"):
        path = f"{base}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")


if __name__ == "__main__":
    plot(out="ablation_ci.pdf")
