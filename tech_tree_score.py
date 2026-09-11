#!/usr/bin/env python3
from __future__ import annotations
"""
tech_tree_score.py — Compute Tech Tree Depth Score from Voyager run logs.

Scoring philosophy:
  Each item is assigned a tier based on how many crafting/progression steps
  it takes from raw materials. Higher tier = deeper progression.
  Score = sum of tier values of all distinct items seen.

Usage:
  python3.10 tech_tree_score.py [run_dir ...]
  python3.10 tech_tree_score.py --compare run_A run_B
"""

import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Tech Tree Tier Definitions
# Items not listed default to tier 1 (raw collectibles).
# ---------------------------------------------------------------------------
TECH_TREE = {
    # Tier 1 — Raw collectibles
    "oak_log": 1, "birch_log": 1, "spruce_log": 1, "jungle_log": 1,
    "acacia_log": 1, "dark_oak_log": 1, "mangrove_log": 1, "wood": 1,
    "cobblestone": 1, "sand": 1, "gravel": 1, "dirt": 1, "clay": 1,
    "flint": 1, "bone": 1, "string": 1, "feather": 1, "leather": 1,
    "wheat_seeds": 1, "sugar_cane": 1, "bamboo": 1, "kelp": 1,
    "seagrass": 1, "lily_pad": 1, "apple": 1, "sweet_berries": 1,
    "egg": 1, "snowball": 1, "vine": 1, "chicken": 1, "beef": 1,
    "porkchop": 1, "mutton": 1, "rabbit": 1, "cod": 1, "salmon": 1,

    # Tier 2 — First processing step
    "oak_planks": 2, "birch_planks": 2, "spruce_planks": 2,
    "jungle_planks": 2, "acacia_planks": 2, "dark_oak_planks": 2,
    "mangrove_planks": 2, "stick": 2, "charcoal": 2,
    "coal": 2, "coal_ore": 2,
    "raw_iron": 2, "iron_ore": 2,
    "raw_copper": 2, "copper_ore": 2,
    "raw_gold": 2, "gold_ore": 2,
    "lapis_lazuli": 2, "lapis_ore": 2,
    "redstone": 2, "redstone_ore": 2,
    "diamond": 2, "diamond_ore": 2,
    "emerald": 2, "emerald_ore": 2,
    "quartz": 2, "nether_quartz_ore": 2,
    "ancient_debris": 2,
    "wheat": 2, "bread": 2, "torch": 2,
    "crafting_table": 2, "furnace": 2,
    "cooked_chicken": 2, "cooked_beef": 2, "cooked_porkchop": 2,
    "cooked_mutton": 2, "cooked_cod": 2, "cooked_salmon": 2,
    "cooked_rabbit": 2, "rabbit_stew": 2,

    # Tier 3 — Basic tools / first smelted metals
    "wooden_pickaxe": 3, "wooden_axe": 3, "wooden_shovel": 3,
    "wooden_sword": 3, "wooden_hoe": 3,
    "stone_pickaxe": 3, "stone_axe": 3, "stone_shovel": 3,
    "stone_sword": 3, "stone_hoe": 3,
    "iron_ingot": 3, "copper_ingot": 3, "gold_ingot": 3,
    "brick": 3, "terracotta": 3, "glass": 3, "paper": 3,
    "bowl": 3, "bucket": 3, "shears": 3, "flint_and_steel": 3,
    "chest": 3, "bookshelf": 3, "ladder": 3,

    # Tier 4 — Iron tier items
    "iron_pickaxe": 4, "iron_axe": 4, "iron_shovel": 4,
    "iron_sword": 4, "iron_hoe": 4,
    "iron_helmet": 4, "iron_chestplate": 4, "iron_leggings": 4,
    "iron_boots": 4,
    "shield": 4, "anvil": 4, "iron_bars": 4,
    "compass": 4, "clock": 4, "lightning_rod": 4,
    "spyglass": 4, "copper_block": 4,
    "gold_pickaxe": 4, "gold_sword": 4, "gold_helmet": 4,
    "gold_chestplate": 4, "gold_leggings": 4, "gold_boots": 4,

    # Tier 5 — Diamond tier items
    "diamond_pickaxe": 5, "diamond_axe": 5, "diamond_shovel": 5,
    "diamond_sword": 5, "diamond_hoe": 5,
    "diamond_helmet": 5, "diamond_chestplate": 5,
    "diamond_leggings": 5, "diamond_boots": 5,
    "enchanting_table": 5, "jukebox": 5,
    "golden_apple": 5, "name_tag": 5,

    # Tier 6 — Netherite / end-game
    "netherite_scrap": 6, "netherite_ingot": 6,
    "netherite_pickaxe": 6, "netherite_axe": 6, "netherite_shovel": 6,
    "netherite_sword": 6, "netherite_hoe": 6,
    "netherite_helmet": 6, "netherite_chestplate": 6,
    "netherite_leggings": 6, "netherite_boots": 6,
    "beacon": 6, "conduit": 6,
}

MAX_TIER = max(TECH_TREE.values())


def get_tier(item: str) -> int:
    """Return tier for an item, defaulting to 1."""
    item = item.lower().strip()
    if item in TECH_TREE:
        return TECH_TREE[item]
    # fuzzy: strip common suffixes
    for suffix in ("_block", "_ore", "_log", "_planks", "_ingot"):
        if item.endswith(suffix):
            base = item[: -len(suffix)]
            if base in TECH_TREE:
                return TECH_TREE[base]
    return 1


def tech_tree_score(items: list[str]) -> float:
    """Sum of tier values for a collection of distinct items."""
    return sum(get_tier(i) for i in items)


def max_tier_reached(items: list[str]) -> int:
    if not items:
        return 0
    return max(get_tier(i) for i in items)


# ---------------------------------------------------------------------------
# Task-name → item inference (for old logs without seen_items)
# ---------------------------------------------------------------------------
TASK_ITEM_PATTERNS = [
    (re.compile(r"mine \d+ (.+)", re.I), lambda m: m.group(1).lower().replace(" ", "_")),
    (re.compile(r"craft \d+ (.+)", re.I), lambda m: m.group(1).lower().replace(" ", "_")),
    (re.compile(r"smelt \d+ (.+)", re.I), lambda m: m.group(1).lower().replace(" ", "_")),
    (re.compile(r"obtain \d+ (.+)", re.I), lambda m: m.group(1).lower().replace(" ", "_")),
    (re.compile(r"kill \d+ (.+)", re.I), lambda m: m.group(1).lower().replace(" ", "_")),
    (re.compile(r"cook \d+ (.+)", re.I), lambda m: m.group(1).lower().replace(" ", "_")),
]

# common aliases for task names
TASK_ALIASES = {
    "wood_log": "oak_log", "wooden_planks": "oak_planks",
    "raw_iron": "iron_ingot",  # smelt raw iron → iron_ingot
    "raw_copper": "copper_ingot", "raw_gold": "gold_ingot",
    "iron": "iron_ingot", "gold": "gold_ingot", "copper": "copper_ingot",
    "log": "oak_log",
}


def infer_item_from_task(task: str) -> str | None:
    for pat, fn in TASK_ITEM_PATTERNS:
        m = pat.match(task.strip())
        if m:
            item = fn(m)
            return TASK_ALIASES.get(item, item)
    return None


# ---------------------------------------------------------------------------
# Load run data
# ---------------------------------------------------------------------------
def load_run(run_dir: str):
    step_log = os.path.join(run_dir, "step_log.jsonl")
    if not os.path.exists(step_log):
        return None
    steps = [json.loads(l) for l in open(step_log) if l.strip()]
    return steps


def compute_scores(run_dir: str):
    steps = load_run(run_dir)
    if not steps:
        print(f"[warn] no step_log in {run_dir}")
        return None

    scores_over_time = []
    cumulative_items = set()

    for step in steps:
        if not step.get("success"):
            scores_over_time.append(scores_over_time[-1] if scores_over_time else 0)
            continue

        # Prefer seen_items list (new logs), fall back to task inference (old logs)
        if "seen_items" in step and step["seen_items"]:
            cumulative_items = set(step["seen_items"])
        else:
            item = infer_item_from_task(step.get("task", ""))
            if item:
                cumulative_items.add(item)

        scores_over_time.append(tech_tree_score(cumulative_items))

    final_items = list(cumulative_items)
    return {
        "run_dir": run_dir,
        "total_steps": len(steps),
        "successful_steps": sum(1 for s in steps if s.get("success")),
        "final_score": scores_over_time[-1] if scores_over_time else 0,
        "max_tier": max_tier_reached(final_items),
        "distinct_items": len(final_items),
        "items_by_tier": {
            t: [i for i in final_items if get_tier(i) == t]
            for t in range(1, MAX_TIER + 1)
            if any(get_tier(i) == t for i in final_items)
        },
        "scores_over_time": scores_over_time,
    }


def _load_branch(run_dir: str) -> str:
    meta_path = os.path.join(run_dir, "meta.json")
    try:
        return json.load(open(meta_path)).get("git_branch", "?")
    except Exception:
        return "?"


def print_report(result: dict):
    run_name = os.path.basename(result["run_dir"])
    branch = _load_branch(result["run_dir"])
    print(f"\n{'='*60}")
    print(f"Run: {run_name}  [branch: {branch}]")
    print(f"{'='*60}")
    print(f"  Total steps:      {result['total_steps']}")
    print(f"  Successful steps: {result['successful_steps']}")
    print(f"  Distinct items:   {result['distinct_items']}")
    print(f"  Max tier reached: {result['max_tier']}")
    print(f"  Tech Tree Score:  {result['final_score']}")
    print()
    print("  Items by tier:")
    for tier, items in sorted(result["items_by_tier"].items()):
        print(f"    Tier {tier}: {', '.join(sorted(items))}")


def plot_comparison(results: list[dict]):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    # Left: Tech Tree Score over steps
    ax = axes[0]
    for i, r in enumerate(results):
        branch = _load_branch(r["run_dir"])
        label = f"{os.path.basename(r['run_dir'])} [{branch}]"
        ax.plot(r["scores_over_time"], label=label, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tech Tree Score")
    ax.set_title("Tech Tree Score over Steps")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Middle: Distinct items over steps
    ax_mid = axes[1]
    for i, r in enumerate(results):
        branch = _load_branch(r["run_dir"])
        label = f"{os.path.basename(r['run_dir'])} [{branch}]"
        steps = load_run(r["run_dir"])
        distinct = [s.get("distinct_items", 0) for s in (steps or [])]
        ax_mid.plot(distinct, label=label, color=colors[i % len(colors)], linewidth=2)
    ax_mid.set_xlabel("Step")
    ax_mid.set_ylabel("Distinct Items")
    ax_mid.set_title("Distinct Items over Steps")
    ax_mid.legend(fontsize=7)
    ax_mid.grid(True, alpha=0.3)

    # Right: Bar chart — tier breakdown
    ax2 = axes[2]
    n_tiers = MAX_TIER
    x = np.arange(1, n_tiers + 1)
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        branch = _load_branch(r["run_dir"])
        label = f"{os.path.basename(r['run_dir'])} [{branch}]"
        counts = [len(r["items_by_tier"].get(t, [])) for t in range(1, n_tiers + 1)]
        ax2.bar(x + i * width - 0.4 + width / 2, counts, width,
                label=label, color=colors[i % len(colors)], alpha=0.8)
    ax2.set_xlabel("Tier")
    ax2.set_ylabel("# Distinct Items")
    ax2.set_title("Items per Tier")
    ax2.set_xticks(x)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = "tech_tree_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="Run directories (default: all in logs/)")
    parser.add_argument("--compare", action="store_true", help="Plot comparison chart")
    args = parser.parse_args()

    # Resolve run dirs
    if args.runs:
        run_dirs = [r if os.path.isdir(r) else os.path.join("logs", r) for r in args.runs]
    else:
        log_root = "logs"
        run_dirs = sorted(
            os.path.join(log_root, d)
            for d in os.listdir(log_root)
            if os.path.isdir(os.path.join(log_root, d))
        ) if os.path.isdir(log_root) else []

    results = []
    for rd in run_dirs:
        r = compute_scores(rd)
        if r:
            print_report(r)
            results.append(r)

    if args.compare and len(results) >= 2:
        plot_comparison(results)
    elif len(results) == 1 and args.compare:
        print("[info] Need at least 2 runs to compare")

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"{'Run':<30} {'Branch':<12} {'Steps':>6} {'Score':>7} {'MaxTier':>8} {'Items':>6}")
        print(f"{'-'*70}")
        for r in results:
            name = os.path.basename(r["run_dir"])[:29]
            branch = _load_branch(r["run_dir"])
            print(f"{name:<30} {branch:<12} {r['total_steps']:>6} {r['final_score']:>7} "
                  f"{r['max_tier']:>8} {r['distinct_items']:>6}")
