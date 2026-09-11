async function craftItem(bot, name, count = 1) {
    // return if name is not string
    if (typeof name !== "string") {
        throw new Error("name for craftItem must be a string");
    }
    // return if count is not number
    if (typeof count !== "number") {
        throw new Error("count for craftItem must be a number");
    }
    const itemByName = mcData.itemsByName[name];
    if (!itemByName) {
        throw new Error(`No item named ${name}`);
    }
    let craftingTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 32,
    });
    if (!craftingTable) {
        // Auto-place crafting table from inventory if available
        const tableInInventory = bot.inventory.findInventoryItem(
            mcData.itemsByName.crafting_table.id
        );
        if (tableInInventory) {
            // Try several adjacent positions in case one is occupied / not placeable.
            // Use floored() so placeItem always receives integer block coordinates —
            // bot.entity.position has fractional y (e.g. 65.1) which breaks
            // the reference-block lookup inside placeItem.
            const base = bot.entity.position.floored();
            const offsets = [
                new Vec3(1, 0, 0), new Vec3(-1, 0, 0),
                new Vec3(0, 0, 1), new Vec3(0, 0, -1),
                new Vec3(1, 0, 1), new Vec3(-1, 0, -1),
                new Vec3(2, 0, 0), new Vec3(-2, 0, 0),
                new Vec3(0, 0, 2), new Vec3(0, 0, -2),
            ];
            for (const off of offsets) {
                try {
                    await placeItem(bot, "crafting_table", base.offset(off.x, off.y, off.z));
                } catch (e) {
                    // placement failed at this offset, try the next one
                }
                craftingTable = bot.findBlock({
                    matching: mcData.blocksByName.crafting_table.id,
                    maxDistance: 32,
                });
                if (craftingTable) break;
            }
            // Fallback: if all positions occupied (e.g. fully underground in stone),
            // mine one adjacent block to create space, then place there.
            if (!craftingTable) {
                const digOffsets = [
                    new Vec3(1, 0, 0), new Vec3(-1, 0, 0),
                    new Vec3(0, 0, 1), new Vec3(0, 0, -1),
                ];
                for (const off of digOffsets) {
                    const pos = base.offset(off.x, off.y, off.z);
                    const block = bot.blockAt(pos);
                    if (block && block.name !== "air" && block.diggable) {
                        try {
                            bot.chat(`Mining ${block.name} to make space for crafting_table`);
                            await bot.dig(block);
                            await placeItem(bot, "crafting_table", pos);
                        } catch (e) { /* try next */ }
                        craftingTable = bot.findBlock({
                            matching: mcData.blocksByName.crafting_table.id,
                            maxDistance: 32,
                        });
                        if (craftingTable) break;
                    }
                }
            }
        } else {
            // No crafting table nearby AND none in inventory: auto-craft one from
            // planks (a crafting table is made in the 2x2 grid, so it needs no table),
            // crafting planks from logs first if needed, then place it. Without this
            // the bot gets stuck at iron tier — every table-requiring craft (iron
            // pickaxe/armor, etc.) fails because no table can be obtained.
            let plankCount = bot.inventory.items()
                .filter((i) => i.name.endsWith("_planks"))
                .reduce((s, i) => s + i.count, 0);
            if (plankCount < 4) {
                const logItem = bot.inventory.items().find((i) => i.name.endsWith("_log"));
                if (logItem) {
                    const plankData = mcData.itemsByName[logItem.name.replace("_log", "_planks")];
                    if (plankData) {
                        try {
                            const r = bot.recipesFor(plankData.id, null, 1, null)[0];
                            if (r) await bot.craft(r, 2, null);
                        } catch (e) { /* ignore */ }
                    }
                }
                plankCount = bot.inventory.items()
                    .filter((i) => i.name.endsWith("_planks"))
                    .reduce((s, i) => s + i.count, 0);
            }
            if (plankCount >= 4) {
                try {
                    const tableRecipe = bot.recipesFor(
                        mcData.itemsByName.crafting_table.id, null, 1, null
                    )[0];
                    if (tableRecipe) await bot.craft(tableRecipe, 1, null);
                } catch (e) { /* ignore */ }
                const base = bot.entity.position.floored();
                for (const off of [
                    new Vec3(1, 0, 0), new Vec3(-1, 0, 0),
                    new Vec3(0, 0, 1), new Vec3(0, 0, -1),
                ]) {
                    try {
                        await placeItem(bot, "crafting_table", base.offset(off.x, off.y, off.z));
                    } catch (e) { /* try next offset */ }
                    craftingTable = bot.findBlock({
                        matching: mcData.blocksByName.crafting_table.id,
                        maxDistance: 32,
                    });
                    if (craftingTable) break;
                }
            }
            if (!craftingTable) {
                bot.chat("Craft without a crafting table");
            }
        }
    }
    // Auto-prerequisite: if the target recipe needs sticks and we don't have enough,
    // try to craft sticks now (from planks → from logs if necessary).
    const stickMeta = mcData.itemsByName["stick"];
    const plankMeta = mcData.itemsByName["oak_planks"];
    const logNames = ["oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log"];
    if (stickMeta) {
        const testRecipe = bot.recipesFor(itemByName.id, null, 1, craftingTable)[0];
        if (testRecipe) {
            // Count how many sticks the recipe needs
            let sticksNeeded = 0;
            for (const ingredient of (testRecipe.inShape || []).flat()) {
                if (ingredient && ingredient.id === stickMeta.id) sticksNeeded++;
            }
            const sticksHave = bot.inventory.items()
                .filter(i => i.id === stickMeta.id)
                .reduce((s, i) => s + i.count, 0);
            if (sticksNeeded > sticksHave) {
                const sticksShort = sticksNeeded - sticksHave;
                // Need planks to make sticks (2 planks → 4 sticks)
                const planksHave = plankMeta
                    ? bot.inventory.items().filter(i => i.id === plankMeta.id).reduce((s,i) => s+i.count, 0)
                    : 0;
                if (planksHave < 2) {
                    // Try to get logs and craft planks first
                    for (const logName of logNames) {
                        const logMeta = mcData.itemsByName[logName];
                        if (!logMeta) continue;
                        const logsHave = bot.inventory.items()
                            .filter(i => i.id === logMeta.id)
                            .reduce((s, i) => s + i.count, 0);
                        if (logsHave > 0) {
                            bot.chat(`Auto-crafting ${logName} → planks for sticks`);
                            try { await craftItem(bot, logName.replace("_log","_planks"), 1); } catch(e) {}
                            break;
                        }
                    }
                    // If still no planks, mine a wood log
                    const planksNow = plankMeta
                        ? bot.inventory.items().filter(i => i.id === plankMeta.id).reduce((s,i) => s+i.count, 0)
                        : 0;
                    if (planksNow < 2) {
                        bot.chat("Need wood for sticks — mining a log first");
                        try { await mineBlock(bot, "oak_log", 1); } catch(e) {}
                        try { await craftItem(bot, "oak_planks", 1); } catch(e) {}
                    }
                }
                bot.chat(`Auto-crafting ${sticksShort} missing sticks`);
                try { await craftItem(bot, "stick", Math.ceil(sticksShort / 4)); } catch(e) {}
            }
        }
    }

    if (craftingTable) {
        await bot.pathfinder.goto(
            new GoalLookAtBlock(craftingTable.position, bot.world)
        );
    }
    const recipe = bot.recipesFor(itemByName.id, null, 1, craftingTable)[0];
    if (recipe) {
        bot.chat(`I can make ${name}`);
        try {
            await bot.craft(recipe, count, craftingTable);
            bot.chat(`I did the recipe for ${name} ${count} times`);
        } catch (err) {
            bot.chat(`I cannot do the recipe for ${name} ${count} times`);
        }
    } else {
        failedCraftFeedback(bot, name, itemByName, craftingTable);
        _craftItemFailCount++;
        if (_craftItemFailCount > 10) {
            throw new Error(
                "craftItem failed too many times, check chat log to see what happened"
            );
        }
    }
}
