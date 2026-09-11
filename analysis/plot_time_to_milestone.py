"""Figure B — Time-to-Tier-Milestone boxplot.

For each run, find the first step at which the agent unlocks T3 / T4 / T5 items.
Lower = faster to reach milestone.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from _common import RUNS, COLORS, apply_style, load_step_log, save_pair

# Tier definitions (subset from plot_tier_progression.py — only items needed here)
TECH_TREE = {}
for name in ["wooden_pickaxe", "wooden_axe", "wooden_shovel", "wooden_sword",
             "wooden_hoe", "stone_pickaxe", "stone_axe", "stone_shovel",
             "stone_sword", "stone_hoe", "iron_ingot", "copper_ingot",
             "gold_ingot", "brick", "terracotta", "glass", "paper", "bowl",
             "bucket", "shears", "flint_and_steel", "chest", "bookshelf",
             "ladder"]:
    TECH_TREE[name] = 3
for name in ["iron_pickaxe", "iron_axe", "iron_shovel", "iron_sword", "iron_hoe",
             "iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots",
             "shield", "anvil", "iron_bars", "compass", "clock", "lightning_rod",
             "spyglass", "copper_block", "gold_pickaxe", "gold_sword",
             "gold_helmet", "gold_chestplate", "gold_leggings", "gold_boots"]:
    TECH_TREE[name] = 4
for name in ["diamond_pickaxe", "diamond_axe", "diamond_shovel", "diamond_sword",
             "diamond_hoe", "diamond_helmet", "diamond_chestplate",
             "diamond_leggings", "diamond_boots", "enchanting_table", "jukebox",
             "golden_apple", "name_tag"]:
    TECH_TREE[name] = 5


def first_step_for_tier(run, tier):
    """Return first step (1-indexed) where any tier-`tier` item appears in seen_items."""
    for i, s in enumerate(load_step_log(run), start=1):
        items = s.get("seen_items") or []
        for it in items:
            if TECH_TREE.get(it.lower().strip()) == tier:
                return i
    return None


def main(out):
    apply_style()
    tiers = [3, 4, 5]
    data = {label: {t: [] for t in tiers} for label in RUNS}
    for label, runs in RUNS.items():
        for run in runs:
            for t in tiers:
                v = first_step_for_tier(run, t)
                if v is not None:
                    data[label][t].append(v)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    width = 0.34
    positions = np.arange(len(tiers))

    for idx, label in enumerate(RUNS):
        color = COLORS[label]
        offset = (-1 if idx == 0 else 1) * width / 2 - width * 0.02
        for j, t in enumerate(tiers):
            vals = data[label][t]
            if not vals:
                continue
            bp = ax.boxplot(
                [vals],
                positions=[positions[j] + offset],
                widths=width * 0.9,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.2),
                boxprops=dict(facecolor=color, alpha=0.55,
                              edgecolor="black", linewidth=0.7),
                whiskerprops=dict(color="black", linewidth=0.7),
                capprops=dict(color="black", linewidth=0.7),
                flierprops=dict(marker="o", markerfacecolor="white",
                                markeredgecolor="black", markersize=4),
            )
            # overlay individual runs
            rng = np.random.default_rng(j * 7 + idx)
            jitter = rng.uniform(-width * 0.18, width * 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), positions[j] + offset) + jitter, vals,
                       s=14, color="white", edgecolors="black",
                       linewidths=0.6, zorder=4, alpha=0.95)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Tier {t}" for t in tiers])
    ax.set_ylabel("First Step Tier Reached")
    ax.set_ylim(0, None)

    legend_handles = [mpatches.Patch(facecolor=COLORS[l], edgecolor="black",
                                     linewidth=0.7, alpha=0.55, label=l)
                      for l in RUNS]
    ax.legend(handles=legend_handles, loc="upper left",
              frameon=True, fancybox=False, edgecolor="black", framealpha=0.95)

    fig.tight_layout(pad=0.3)
    save_pair(fig, out.rsplit(".", 1)[0])
    plt.close(fig)

    print("\n=== Time-to-Tier-Milestone (first step) ===")
    for label in RUNS:
        for t in tiers:
            v = data[label][t]
            if v:
                print(f"  {label:22s} T{t}: {np.mean(v):5.1f} ± {np.std(v, ddof=1):4.1f}  (n={len(v)})")
            else:
                print(f"  {label:22s} T{t}: never reached")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fig_time_to_milestone.pdf")
    args = p.parse_args()
    main(args.out)
