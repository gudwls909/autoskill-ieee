"""
plot_tier_progression.py — VI-F: Tech Tree Tier Progression
Shows how many items from each tier (T3, T4, T5) accumulate over time,
comparing Voyager vs AutoSkill.

Three subplots (one per tier), each showing mean ± 1 std band for both groups.

Usage:
    python plot_tier_progression.py
    python plot_tier_progression.py --out tier_progression.png
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

# ── Tier definitions (from tech_tree_score.py) ───────────────────────────────
TECH_TREE = {
    "oak_log": 1, "birch_log": 1, "spruce_log": 1, "jungle_log": 1,
    "acacia_log": 1, "dark_oak_log": 1, "mangrove_log": 1, "wood": 1,
    "cobblestone": 1, "sand": 1, "gravel": 1, "dirt": 1, "clay": 1,
    "flint": 1, "bone": 1, "string": 1, "feather": 1, "leather": 1,
    "wheat_seeds": 1, "sugar_cane": 1, "bamboo": 1, "kelp": 1,
    "seagrass": 1, "lily_pad": 1, "apple": 1, "sweet_berries": 1,
    "egg": 1, "snowball": 1, "vine": 1, "chicken": 1, "beef": 1,
    "porkchop": 1, "mutton": 1, "rabbit": 1, "cod": 1, "salmon": 1,

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

    "wooden_pickaxe": 3, "wooden_axe": 3, "wooden_shovel": 3,
    "wooden_sword": 3, "wooden_hoe": 3,
    "stone_pickaxe": 3, "stone_axe": 3, "stone_shovel": 3,
    "stone_sword": 3, "stone_hoe": 3,
    "iron_ingot": 3, "copper_ingot": 3, "gold_ingot": 3,
    "brick": 3, "terracotta": 3, "glass": 3, "paper": 3,
    "bowl": 3, "bucket": 3, "shears": 3, "flint_and_steel": 3,
    "chest": 3, "bookshelf": 3, "ladder": 3,

    "iron_pickaxe": 4, "iron_axe": 4, "iron_shovel": 4,
    "iron_sword": 4, "iron_hoe": 4,
    "iron_helmet": 4, "iron_chestplate": 4, "iron_leggings": 4,
    "iron_boots": 4,
    "shield": 4, "anvil": 4, "iron_bars": 4,
    "compass": 4, "clock": 4, "lightning_rod": 4,
    "spyglass": 4, "copper_block": 4,
    "gold_pickaxe": 4, "gold_sword": 4, "gold_helmet": 4,
    "gold_chestplate": 4, "gold_leggings": 4, "gold_boots": 4,

    "diamond_pickaxe": 5, "diamond_axe": 5, "diamond_shovel": 5,
    "diamond_sword": 5, "diamond_hoe": 5,
    "diamond_helmet": 5, "diamond_chestplate": 5,
    "diamond_leggings": 5, "diamond_boots": 5,
    "enchanting_table": 5, "jukebox": 5,
    "golden_apple": 5, "name_tag": 5,

    "netherite_scrap": 6, "netherite_ingot": 6,
    "netherite_pickaxe": 6, "netherite_axe": 6, "netherite_shovel": 6,
    "netherite_sword": 6, "netherite_hoe": 6,
    "netherite_helmet": 6, "netherite_chestplate": 6,
    "netherite_leggings": 6, "netherite_boots": 6,
    "beacon": 6, "conduit": 6,
}


def get_tier(item: str) -> int:
    item = item.lower().strip()
    if item in TECH_TREE:
        return TECH_TREE[item]
    for suffix in ("_block", "_ore", "_log", "_planks", "_ingot"):
        if item.endswith(suffix):
            base = item[: -len(suffix)]
            if base in TECH_TREE:
                return TECH_TREE[base]
    return 1


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

COLORS = {
    "Voyager (main)":   "#E69F00",
    "AutoSkill (ours)": "#0072B2",
}
LINESTYLES = {
    "Voyager (main)":   "--",
    "AutoSkill (ours)": "-",
}

TIER_LABELS = {
    3: "Tier 3 — Tools & Ingots",
    4: "Tier 4 — Iron Tier",
    5: "Tier 5 — Diamond Tier",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_tier_series(run_dir: str, tier: int) -> np.ndarray:
    """
    Returns an array of length = n_steps where each value is the cumulative
    count of distinct items at the given tier seen up to that step.
    seen_items in step_log is cumulative (all items seen so far).
    Forward-fill on failed steps where seen_items may be absent/empty.
    """
    path = LOG_BASE / run_dir / "step_log.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    values = []
    last_count = 0
    for r in rows:
        items = r.get("seen_items")
        if items:
            count = sum(1 for it in items if get_tier(it) == tier)
            last_count = count
        values.append(last_count)

    return np.array(values, dtype=float)


def align_stack(series_list: list[np.ndarray]) -> np.ndarray:
    """Pad shorter runs with their final value (tier counts are monotone), so
    the endpoint of each curve equals the per-run final mean reported in the
    tier table rather than the value at the shortest run's last step."""
    max_len = max(len(s) for s in series_list)
    return np.stack(
        [np.concatenate([s, np.full(max_len - len(s), s[-1])]) if len(s) < max_len else s
         for s in series_list],
        axis=0,
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(out: str | None = None) -> None:
    _apply_style()
    tiers = [3, 4, 5]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharey=False)

    summary: dict[str, dict[int, float]] = {lb: {} for lb in RUNS}
    legend_handles = []

    for ax_idx, (ax, tier) in enumerate(zip(axes, tiers)):
        for label, run_list in RUNS.items():
            color = COLORS[label]
            ls = LINESTYLES[label]
            mat = align_stack([load_tier_series(r, tier) for r in run_list])
            n_runs, n_steps = mat.shape
            xs = np.arange(1, n_steps + 1)

            mean = mat.mean(axis=0)
            std  = mat.std(axis=0, ddof=min(1, n_runs - 1))

            ax.plot(xs, mean, color=color, linestyle=ls, linewidth=1.8,
                    label=label, zorder=3)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=color,
                            linewidth=0, zorder=2)

            ax.annotate(
                f"{mean[-1]:.1f}",
                xy=(xs[-1], mean[-1]),
                xytext=(5, 0), textcoords="offset points",
                fontsize=11, color=color, fontweight="bold",
                va="center", ha="left",
            )

            summary[label][tier] = (mean[-1], std[-1])

            if ax_idx == 0:
                patch = mpatches.Patch(facecolor=color, alpha=0.4)
                line  = plt.Line2D([0], [0], color=color, linewidth=1.8, linestyle=ls)
                legend_handles.append((line, patch, label))

        ax.set_title(TIER_LABELS[tier], pad=6)
        ax.set_xlabel("Prompting Iterations")
        if ax_idx == 0:
            ax.set_ylabel("Cumulative Items Discovered")
        ax.set_xlim(1, n_steps * 1.08)
        ax.set_ylim(0, None)

    # Single shared legend on first subplot
    from matplotlib.legend_handler import HandlerTuple
    proxies = [(l, p) for l, p, _ in legend_handles]
    labels  = [lb for _, __, lb in legend_handles]
    axes[0].legend(
        proxies, labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
        frameon=True, fancybox=False,
        edgecolor="black", framealpha=0.95,
    )

    fig.tight_layout(pad=0.3, w_pad=1.4)

    base = (out or "tier_progression.pdf").rsplit(".", 1)[0]
    for fmt in ("pdf", "png"):
        path = f"{base}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n=== Tier Progression Summary (final step) ===")
    print(f"{'Tier':<8} {'Voyager':>18} {'AutoSkill':>18} {'Delta':>10} {'Δ%':>8}")
    print("-" * 65)
    for tier in tiers:
        v_mean, v_std = summary["Voyager (main)"][tier]
        a_mean, a_std = summary["AutoSkill (ours)"][tier]
        delta = a_mean - v_mean
        pct   = (delta / v_mean * 100) if v_mean > 0 else float("inf")
        print(f"T{tier}      {v_mean:6.1f} ± {v_std:4.1f}     {a_mean:6.1f} ± {a_std:4.1f}    {delta:+6.1f}   {pct:+6.1f}%")

    # Step at which AutoSkill first reaches T5
    print("\n--- First step AutoSkill vs Voyager reaches T5 ≥ 1 item ---")
    for label, run_list in RUNS.items():
        first_steps = []
        for r in run_list:
            series = load_tier_series(r, 5)
            idxs = np.where(series >= 1)[0]
            first_steps.append(int(idxs[0]) + 1 if len(idxs) > 0 else None)
        reached = [s for s in first_steps if s is not None]
        if reached:
            print(f"  {label}: {reached}  → avg step {np.mean(reached):.1f}")
        else:
            print(f"  {label}: never reached T5 in any run")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tier progression comparison")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: tier_progression.png)")
    args = parser.parse_args()
    plot(out=args.out)
