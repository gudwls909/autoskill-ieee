"""Figure A — Skill Library Efficiency: TTS per skill in library over iterations.

Higher = each skill in the library is more useful (more TTS gained per skill).
Shows that AutoSkill's compact, deduplicated library yields more value per slot.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib.legend_handler import HandlerTuple

from _common import RUNS, COLORS, LINESTYLES, apply_style, load_step_log, save_pair


def efficiency_series(run):
    steps = load_step_log(run)
    tts = []
    lib = []
    cur_tts = 0
    for s in steps:
        cur_tts = max(cur_tts, s.get("tech_tree_score", 0))
        tts.append(cur_tts)
        lib.append(max(s.get("skill_library_size", 1), 1))
    return np.array(tts) / np.array(lib)


def main(out):
    apply_style()
    series = {}
    for label, runs in RUNS.items():
        arrs = [efficiency_series(r) for r in runs]
        n = min(len(a) for a in arrs)
        series[label] = np.stack([a[:n] for a in arrs])
    n = min(s.shape[1] for s in series.values())
    xs = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    legend_handles = []

    for label, mat in series.items():
        color = COLORS[label]
        ls = LINESTYLES[label]
        mean = mat[:, :n].mean(axis=0)
        std  = mat[:, :n].std(axis=0, ddof=1)
        ax.plot(xs, mean, color=color, linestyle=ls, linewidth=1.8,
                label=label, zorder=3)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=color,
                        linewidth=0, zorder=2)
        ax.annotate(f"{mean[-1]:.2f}",
                    xy=(xs[-1], mean[-1]), xytext=(7, 0),
                    textcoords="offset points", fontsize=12, color=color,
                    fontweight="bold", va="center", ha="left")
        patch = mpatches.Patch(facecolor=color, alpha=0.4)
        line  = plt.Line2D([0], [0], color=color, linewidth=1.8, linestyle=ls)
        legend_handles.append((line, patch, label))

    ax.set_xlabel("Number of Prompting Iterations")
    ax.set_ylabel("TTS per Skill in Library")
    ax.set_xlim(0, xs[-1] * 1.05)
    ax.set_ylim(0, None)
    proxies = [(l, p) for l, p, _ in legend_handles]
    labels  = [lb for _, __, lb in legend_handles]
    ax.legend(proxies, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="upper left", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.95)
    fig.tight_layout(pad=0.3)
    save_pair(fig, out.rsplit(".", 1)[0])
    plt.close(fig)

    print("\n=== Skill Library Efficiency (final) ===")
    for label, mat in series.items():
        m = mat[:, :n].mean(axis=0)[-1]
        s = mat[:, :n].std(axis=0, ddof=1)[-1]
        print(f"  {label:22s} {m:.2f} ± {s:.2f}  TTS/skill")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fig_efficiency.pdf")
    args = p.parse_args()
    main(args.out)
