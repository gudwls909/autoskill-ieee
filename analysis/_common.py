"""Shared helpers for Voyager vs AutoSkill comparison figures."""
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

LOG_BASE = Path(__file__).resolve().parent.parent / "logs"

RUNS = {
    "Voyager (main)": [
        "run-2026-04-03_14-13-42", "run-2026-04-03_19-03-57", "run-2026-04-04_21-18-15",
        "run-2026-04-05_15-29-49", "run-2026-04-06_02-00-16",
        "run-resub-vb-seed1", "run-resub-vb-seed2", "run-resub-vb-seed3",
        "run-resub-vb-seed4", "run-resub-vb-seed5",
    ],
    "AutoSkill (ours)": [
        "run-2026-04-08_17-04-48", "run-2026-04-12_02-43-00", "run-2026-04-12_11-00-21",
        "run-2026-05-14_18-47-19", "run-2026-05-14_02-05-05",
        "run-resub-fix2-seed6", "run-resub-fix2-seed7", "run-resub-fix2-seed8",
        "run-resub-fix2-seed9", "run-resub-fix2-seed10",
    ],
}

COLORS     = {"Voyager (main)": "#E69F00", "AutoSkill (ours)": "#0072B2"}
LINESTYLES = {"Voyager (main)": "--",      "AutoSkill (ours)": "-"}


def apply_style():
    # Use only the "science" base style — `ieee` adds dashed minor ticks on
    # all four sides that overlap with right-edge annotations in our figures.
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
        # Disable inward/outward minor ticks that "ieee" style adds on all sides
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.top": False,
        "ytick.right": False,
    })


def load_step_log(run):
    return [json.loads(l) for l in open(LOG_BASE / run / "step_log.jsonl") if l.strip()]


def load_skill_log(run):
    p = LOG_BASE / run / "skill_log.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def save_pair(fig, base):
    """Save as both PDF and PNG."""
    for fmt in ("pdf", "png"):
        path = f"{base}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")
