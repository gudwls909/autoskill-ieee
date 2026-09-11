"""
Backfill tech_tree_score into an old step_log.jsonl that doesn't have it.

Usage:
    python backfill_tts.py <run_id>

Reconstructs seen_items cumulatively from per-step events files, then
writes tech_tree_score (and seen_items) into each step_log entry in-place.
"""
from __future__ import annotations
import json, os, sys, re
from pathlib import Path

# ── locate run dirs ───────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
LOGS   = BASE / "logs"
CKPT   = BASE / "ckpt"

def run_dirs(run_id: str):
    prefix = run_id if run_id.startswith("run-") else f"run-{run_id}"
    log_dir  = LOGS / prefix
    ckpt_dir = CKPT / prefix
    if not log_dir.exists():
        sys.exit(f"[error] log dir not found: {log_dir}")
    if not ckpt_dir.exists():
        sys.exit(f"[error] ckpt dir not found: {ckpt_dir}")
    return log_dir, ckpt_dir

# ── tech tree score ───────────────────────────────────────────────────────────
try:
    sys.path.insert(0, str(BASE))
    from tech_tree_score import tech_tree_score as _tts_fn
    def compute_tts(items): return _tts_fn(list(items))
except Exception:
    TIERS = {
        1: {"log","wood","dirt","sand","gravel","stone","cobblestone","coal","flint",
            "clay","wool","string","feather","egg","wheat_seeds","bone","leather",
            "ink_sac","oak_log","birch_log","spruce_log","jungle_log","acacia_log","dark_oak_log",
            "bamboo","seagrass","kelp","vine","lily_pad","cactus","sugar_cane","pumpkin",
            "melon","apple","raw_beef","raw_chicken","raw_cod","raw_salmon","raw_mutton",
            "raw_porkchop","raw_rabbit","spider_eye","rotten_flesh","gunpowder","slime_ball",
            "snowball","blaze_rod","ghast_tear","magma_cream","nether_star","ender_pearl",
            "chorus_fruit","prismarine_shard","prismarine_crystals","shulker_shell",
            "rabbit_hide","turtle_egg","phantom_membrane","honeycomb","sweet_berries",
            "glow_ink_sac","moss_block","pointed_dripstone","spore_blossom",
            "andesite","diorite","granite","tuff","calcite","dripstone_block",
            "raw_iron","raw_gold","raw_copper","iron_ore","gold_ore","copper_ore",
            "coal_ore","lapis_ore","redstone_ore","emerald_ore","diamond_ore",
            "nether_quartz_ore","nether_gold_ore","ancient_debris","deepslate",
            "cobbled_deepslate","gravel","dirt"},
        2: {"plank","planks","stick","crafting_table","furnace","chest","torch",
            "ladder","fence","gate","door","slab","stairs","glass","glass_pane",
            "brick","clay_ball","flower_pot","bowl","sign","banner","bed","carpet",
            "string_bow","lead","name_tag","paper","book","map","clock","compass",
            "fishing_rod","flint_and_steel","shears","bucket","wooden_pickaxe",
            "wooden_axe","wooden_shovel","wooden_sword","wooden_hoe",
            "oak_planks","birch_planks","spruce_planks","jungle_planks","acacia_planks","dark_oak_planks",
            "oak_slab","oak_stairs","oak_door","oak_fence","oak_fence_gate",
            "charcoal","dried_kelp","bread","cookie","cake","pumpkin_pie","mushroom_stew",
            "rabbit_stew","suspicious_stew","cooked_beef","cooked_chicken","cooked_cod",
            "cooked_salmon","cooked_mutton","cooked_porkchop","cooked_rabbit","baked_potato",
            "stone_pickaxe","stone_axe","stone_shovel","stone_sword","stone_hoe",},
        3: {"iron_ingot","iron_pickaxe","iron_axe","iron_shovel","iron_sword","iron_hoe",
            "iron_helmet","iron_chestplate","iron_leggings","iron_boots",
            "iron_door","iron_bars","iron_trapdoor","iron_block","bucket","cauldron",
            "compass","clock","shears","chain","lantern","soul_lantern","iron_nugget",
            "lapis_lazuli","redstone","redstone_torch","redstone_block",
            "emerald","gold_ingot","gold_nugget","gold_block",
            "copper_ingot","copper_block","lightning_rod","spyglass",
            "leather_helmet","leather_chestplate","leather_leggings","leather_boots",
            "chainmail_helmet","chainmail_chestplate","chainmail_leggings","chainmail_boots",
            "golden_helmet","golden_chestplate","golden_leggings","golden_boots",
            "golden_pickaxe","golden_axe","golden_shovel","golden_sword","golden_hoe",
            "shield","bow","arrow","crossbow","tipped_arrow","spectral_arrow",
            "anvil","smithing_table","stonecutter","blast_furnace","smoker","grindstone",
            "barrel","bookshelf","lectern","loom","cartography_table","fletching_table",
            "quartz","quartz_block","smooth_quartz","nether_brick"},
        4: {"diamond","diamond_pickaxe","diamond_axe","diamond_shovel","diamond_sword","diamond_hoe",
            "diamond_helmet","diamond_chestplate","diamond_leggings","diamond_boots","diamond_block",
            "enchanting_table","ender_chest","beacon","conduit",
            "nether_star","blaze_rod","blaze_powder","eye_of_ender","ender_eye",
            "experience_bottle","name_tag","saddle","totem_of_undying",
            "trident","nautilus_shell","heart_of_the_sea",
            "netherite_scrap","gilded_blackstone"},
        5: {"netherite_ingot","netherite_pickaxe","netherite_axe","netherite_shovel",
            "netherite_sword","netherite_hoe","netherite_helmet","netherite_chestplate",
            "netherite_leggings","netherite_boots","netherite_block"},
    }
    def _tier(item: str) -> int:
        for t in sorted(TIERS, reverse=True):
            if item in TIERS[t]: return t
        return 1
    def compute_tts(items):
        return sum(_tier(i) for i in items)

# ── extract seen items from one events file ───────────────────────────────────
def extract_from_events(path: Path) -> set[str]:
    with open(path) as f:
        events = json.load(f)
    items: set[str] = set()
    for ev_name, ev_data in events:
        if ev_name != "observe":
            continue
        items.update(ev_data.get("inventory", {}).keys())
        for slot in ev_data.get("status", {}).get("equipment", []):
            if slot:
                items.add(slot)
        for chest in ev_data.get("chests", {}).values():
            if isinstance(chest, dict):
                items.update(chest.keys())
    return items

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not run_id:
        sys.exit("Usage: python backfill_tts.py <run_id>")

    log_dir, ckpt_dir = run_dirs(run_id)
    step_log_path = log_dir / "step_log.jsonl"
    events_dir    = ckpt_dir / "events"

    # Read all step_log entries
    with open(step_log_path) as f:
        steps = [json.loads(l) for l in f if l.strip()]

    # Sort events files by their embedded timestamp (YYYYMMDD_HHMMSS suffix)
    ts_re = re.compile(r"_(\d{8}_\d{6})$")
    def ev_ts(p: Path):
        m = ts_re.search(p.stem)
        return m.group(1) if m else p.stem

    ev_files = sorted(events_dir.iterdir(), key=ev_ts)

    if len(ev_files) != len(steps):
        print(f"[warn] events={len(ev_files)}, step_log={len(steps)} — mismatch, will process min({len(ev_files)},{len(steps)})")

    n = min(len(ev_files), len(steps))
    seen: set[str] = set()
    updated = 0

    for i in range(n):
        seen.update(extract_from_events(ev_files[i]))
        entry = steps[i]

        if "tech_tree_score" in entry:
            # already has it; still update seen_items if missing
            if "seen_items" not in entry:
                entry["seen_items"] = sorted(seen)
                updated += 1
        else:
            entry["tech_tree_score"] = compute_tts(seen)
            entry["seen_items"]      = sorted(seen)
            entry["distinct_items"]  = entry.get("distinct_items", len(seen))
            updated += 1

    if updated == 0:
        print("[info] All entries already have tech_tree_score — nothing to do.")
        return

    # Reorder keys so tech_tree_score appears right after success
    def reorder(d: dict) -> dict:
        priority = ["task","step","success","tech_tree_score","distinct_items","seen_items",
                    "agent_iterations","duration_s","error","skill_library_size",
                    "skill_reused","ts"]
        out = {k: d[k] for k in priority if k in d}
        out.update({k: v for k, v in d.items() if k not in out})
        return out

    steps = [reorder(s) for s in steps]

    # Write back
    with open(step_log_path, "w") as f:
        for s in steps:
            f.write(json.dumps(s) + "\n")

    print(f"[done] Backfilled tech_tree_score for {updated}/{n} entries in {step_log_path}")

if __name__ == "__main__":
    main()
