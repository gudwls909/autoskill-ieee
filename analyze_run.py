#!/usr/bin/env python3
"""
analyze_run.py — Voyager run metric visualizer.

Usage:
    python analyze_run.py                     # most recent run in ./logs/
    python analyze_run.py logs/run-YYYY-...   # specific run dir
    python analyze_run.py --compare A B ...   # overlay multiple runs (metrics 1,2 only)
    python analyze_run.py --table             # print markdown table and exit

Outputs:
    <run_dir>/metrics.png   (saved automatically)
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # headless-safe; comment out if you want interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def find_latest_run(base="logs"):
    runs = sorted(glob.glob(os.path.join(base, "run-*")))
    return runs[-1] if runs else None


def task_outcomes(step_log):
    """
    Collapse per-agent-iteration step_log entries into one entry per task proposal
    (the final iteration of each consecutive task block).
    """
    outcomes = []
    i = 0
    while i < len(step_log):
        task = step_log[i]["task"]
        j = i + 1
        while j < len(step_log) and step_log[j]["task"] == task:
            j += 1
        outcomes.append(step_log[j - 1])   # last iteration = final outcome
        i = j
    return outcomes


# ──────────────────────────────────────────────────────────────────
# Per-run metric extraction
# ──────────────────────────────────────────────────────────────────

def _load_branch(run_dir: str) -> str:
    meta_path = os.path.join(run_dir, "meta.json")
    try:
        return json.load(open(meta_path)).get("git_branch", "?")
    except Exception:
        return "?"


def extract_metrics(run_dir):
    step_log  = load_jsonl(os.path.join(run_dir, "step_log.jsonl"))
    skill_log = load_jsonl(os.path.join(run_dir, "skill_log.jsonl"))
    reuse_log = load_jsonl(os.path.join(run_dir, "reuse_log.jsonl"))

    outcomes = task_outcomes(step_log)
    n = len(outcomes)
    iters = list(range(1, n + 1))

    # ── Metric 1: tech tree score (cumulative) ─────────────────────
    tech_tree_scores = [e.get("tech_tree_score", 0) for e in outcomes]
    # fill forward so score never drops (fallback for older logs without the field)
    prev_tts = 0
    for i, v in enumerate(tech_tree_scores):
        if v == 0 and not outcomes[i].get("success"):
            tech_tree_scores[i] = prev_tts
        else:
            prev_tts = tech_tree_scores[i]

    # ── Metric 2: distinct items (cumulative) ──────────────────────
    distinct_items = [e.get("distinct_items", 0) for e in outcomes]

    # ── Metric 2: items gained per 50-task bin ─────────────────────
    BIN = 50
    bin_labels, bin_gains = [], []
    prev = 0
    for start in range(0, n, BIN):
        end = min(start + BIN, n)
        cur = outcomes[end - 1].get("distinct_items", prev)
        bin_gains.append(cur - prev)
        bin_labels.append(f"{start + 1}–{end}")
        prev = cur

    # ── Metric 3: skill library size ──────────────────────────────
    skill_lib_sizes = [e.get("skill_library_size") for e in outcomes]
    if all(s is None for s in skill_lib_sizes):
        # fallback: reconstruct from skill_log (supports both old "skill_add" and new "added")
        ADDED_KINDS = {"added", "skill_add"}
        task_to_lib = {}
        count = 0
        for se in skill_log:
            if se.get("kind") in ADDED_KINDS:
                count += 1
            task_to_lib[se.get("task", "")] = count
        running = 0
        for i, oc in enumerate(outcomes):
            running = task_to_lib.get(oc["task"], running)
            skill_lib_sizes[i] = running
    # fill None gaps with previous value
    prev_val = 0
    for i, v in enumerate(skill_lib_sizes):
        if v is None:
            skill_lib_sizes[i] = prev_val
        else:
            prev_val = v

    # ── Metric 4: cumulative skill reuse count ─────────────────────
    skill_reused = [e.get("skill_reused") for e in outcomes]
    if not any(skill_reused):
        # fallback: reconstruct from reuse_log
        task_reused = defaultdict(bool)
        for re in reuse_log:
            if re.get("true_called_known_skills"):
                task_reused[re.get("task", "")] = True
        skill_reused = [task_reused.get(oc["task"], False) for oc in outcomes]

    cumul_reuse = []
    cnt = 0
    for r in skill_reused:
        if r:
            cnt += 1
        cumul_reuse.append(cnt)

    # ── Metric 5: composite skill % ────────────────────────────────
    ADDED_KINDS = {"added", "skill_add"}
    added = [se for se in skill_log if se.get("kind") in ADDED_KINDS]
    n_added = len(added)
    n_composite = sum(1 for se in added if se.get("is_composite", False))

    # over-time version: per-task running %
    ADDED_KINDS = {"added", "skill_add"}
    task_to_skill = {}
    for se in skill_log:
        if se.get("kind") in ADDED_KINDS:
            task_to_skill.setdefault(se.get("task", ""), se)

    run_added = run_comp = 0
    seen_tasks = set()
    comp_pct_series = []
    for oc in outcomes:
        t = oc["task"]
        if t in task_to_skill and t not in seen_tasks:
            seen_tasks.add(t)
            run_added += 1
            if task_to_skill[t].get("is_composite", False):
                run_comp += 1
        comp_pct_series.append(
            (run_comp / run_added * 100) if run_added > 0 else 0.0
        )

    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir),
        "branch": _load_branch(run_dir),
        "n": n,
        "iters": iters,
        "tech_tree_scores": tech_tree_scores,
        "distinct_items": distinct_items,
        "bin_labels": bin_labels,
        "bin_gains": bin_gains,
        "skill_lib_sizes": skill_lib_sizes,
        "cumul_reuse": cumul_reuse,
        "comp_pct_series": comp_pct_series,
        # summary scalars
        "final_items": distinct_items[-1] if distinct_items else 0,
        "final_lib": skill_lib_sizes[-1] if skill_lib_sizes else 0,
        "final_reuse": cumul_reuse[-1] if cumul_reuse else 0,
        "reuse_rate_pct": cumul_reuse[-1] / n * 100 if n else 0,
        "composite_pct": n_composite / n_added * 100 if n_added else 0,
        "n_composite": n_composite,
        "n_added": n_added,
    }


# ──────────────────────────────────────────────────────────────────
# Single-run chart
# ──────────────────────────────────────────────────────────────────

COLORS = {
    "blue":   "#2196F3",
    "purple": "#9C27B0",
    "red":    "#F44336",
    "orange": "#FF5722",
    "green":  "#4CAF50",
    "amber":  "#FF9800",
    "header": "#37474F",
}


def plot_single(m, save=True):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Voyager Run Metrics — {m['run_name']}  (n={m['n']} task proposals)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # 1. Distinct items (cumulative)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(m["iters"], m["distinct_items"], color=COLORS["blue"], linewidth=2)
    ax1.set_title("Distinct Items Seen (cumulative)")
    ax1.set_xlabel("Task proposal #")
    ax1.set_ylabel("Unique items")
    ax1.grid(alpha=0.3)

    # 2. Items per 50-task bin
    ax2 = fig.add_subplot(gs[0, 1])
    avg = np.mean(m["bin_gains"]) if m["bin_gains"] else 0
    bar_cols = [COLORS["green"] if v >= avg else COLORS["amber"] for v in m["bin_gains"]]
    bars = ax2.bar(m["bin_labels"], m["bin_gains"], color=bar_cols, edgecolor="white", linewidth=0.5)
    ax2.set_title("New Distinct Items per 50 Tasks")
    ax2.set_xlabel("Iteration range")
    ax2.set_ylabel("Items gained")
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, m["bin_gains"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.2, str(v),
                 ha="center", va="bottom", fontsize=9)
    ax2.axhline(avg, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                label=f"avg {avg:.1f}")
    ax2.legend(fontsize=8)

    # 3. Skill library size
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(m["iters"], m["skill_lib_sizes"], color=COLORS["purple"], linewidth=2)
    ax3.set_title("Skill Library Size Over Time")
    ax3.set_xlabel("Task proposal #")
    ax3.set_ylabel("# Skills in library")
    ax3.grid(alpha=0.3)

    # 4. Cumulative skill reuse
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(m["iters"], m["cumul_reuse"], color=COLORS["red"], linewidth=2)
    ax4.set_title(f"Cumulative Skill Reuse Count  (rate={m['reuse_rate_pct']:.1f}%)")
    ax4.set_xlabel("Task proposal #")
    ax4.set_ylabel("# Tasks reusing a library skill")
    ax4.grid(alpha=0.3)

    # 5. Composite skill % over time
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(m["iters"], m["comp_pct_series"], color=COLORS["orange"], linewidth=2)
    ax5.set_title(f"Composite Skill % Over Time  (final={m['composite_pct']:.1f}%)")
    ax5.set_xlabel("Task proposal #")
    ax5.set_ylabel("% composite skills")
    ax5.set_ylim(0, 105)
    ax5.grid(alpha=0.3)

    # 6. Summary table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    rows = [
        ["Total task proposals",      str(m["n"])],
        ["Final distinct items",       str(m["final_items"])],
        ["Final skill library size",   str(m["final_lib"])],
        ["Total skill reuse events",   str(m["final_reuse"])],
        ["Skill reuse rate",           f"{m['reuse_rate_pct']:.1f}%"],
        ["Composite skill %",          f"{m['composite_pct']:.1f}%"],
        ["Composite / total skills",   f"{m['n_composite']} / {m['n_added']}"],
    ]
    tbl = ax6.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for j in range(2):
        tbl[(0, j)].set_facecolor(COLORS["header"])
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(2):
            tbl[(i, j)].set_facecolor(bg)
    ax6.set_title("Summary", fontweight="bold")

    out = os.path.join(m["run_dir"], "metrics.png")
    if save:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    return fig, out


# ──────────────────────────────────────────────────────────────────
# Multi-run comparison chart (metrics 1 & 3 overlaid)
# ──────────────────────────────────────────────────────────────────

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def plot_compare(metrics_list, out_path="compare.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Run Comparison", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "Tech Tree Score (cumulative)",  "tech_tree_scores"),
        (axes[0, 1], "Distinct Items (cumulative)",   "distinct_items"),
        (axes[1, 0], "Skill Library Size",             "skill_lib_sizes"),
        (axes[1, 1], "Cumulative Skill Reuse",         "cumul_reuse"),
    ]

    for ax, title, key in panels:
        for m, col in zip(metrics_list, PALETTE):
            label = f"{m['run_name']} [{m['branch']}]"
            ax.plot(m["iters"], m[key], color=col, linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Task proposal #")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# Markdown table printer
# ──────────────────────────────────────────────────────────────────

def print_table(metrics_list):
    cols = [
        ("Run", "run_name"),
        ("Branch", "branch"),
        ("Tasks", "n"),
        ("Distinct items", "final_items"),
        ("Skill lib size", "final_lib"),
        ("Reuse events", "final_reuse"),
        ("Reuse rate", "reuse_rate_pct"),
        ("Composite %", "composite_pct"),
    ]
    header = " | ".join(f"**{c[0]}**" for c in cols)
    sep    = " | ".join("---" for _ in cols)
    print(f"| {header} |")
    print(f"| {sep} |")
    for m in metrics_list:
        vals = []
        for label, key in cols:
            v = m[key]
            if isinstance(v, float):
                vals.append(f"{v:.1f}%")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze Voyager run metrics.")
    parser.add_argument("runs", nargs="*",
                        help="Run directory/directories (default: latest in ./logs/)")
    parser.add_argument("--compare", action="store_true",
                        help="Overlay multiple runs in a comparison chart")
    parser.add_argument("--table", action="store_true",
                        help="Print markdown summary table and exit")
    args = parser.parse_args()

    run_dirs = args.runs
    if not run_dirs:
        latest = find_latest_run()
        if not latest:
            print("No run found in ./logs/")
            sys.exit(1)
        run_dirs = [latest]

    all_metrics = [extract_metrics(d) for d in run_dirs]

    if args.table:
        print_table(all_metrics)
        return

    if args.compare and len(all_metrics) > 1:
        plot_compare(all_metrics)
        print_table(all_metrics)
    else:
        for m in all_metrics:
            fig, out = plot_single(m)
            # also print summary
            print(f"\n=== {m['run_name']}  [branch: {m['branch']}] ===")
            print(f"  Tasks:               {m['n']}")
            print(f"  Distinct items:      {m['final_items']}")
            print(f"  Skill library size:  {m['final_lib']}")
            print(f"  Skill reuse events:  {m['final_reuse']}  ({m['reuse_rate_pct']:.1f}%)")
            print(f"  Composite skills:    {m['n_composite']}/{m['n_added']}  ({m['composite_pct']:.1f}%)")
            plt.close(fig)


if __name__ == "__main__":
    main()
