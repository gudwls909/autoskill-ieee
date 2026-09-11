async function smeltItem(bot, itemName, fuelName, count = 1) {
    // return if itemName or fuelName is not string
    if (typeof itemName !== "string" || typeof fuelName !== "string") {
        throw new Error("itemName or fuelName for smeltItem must be a string");
    }
    // return if count is not a number
    if (typeof count !== "number") {
        throw new Error("count for smeltItem must be a number");
    }

    // Map from raw ingredient → expected smelting output.
    // Used to early-exit when we already have enough of the output item.
    const SMELT_OUTPUT = {
        raw_iron: "iron_ingot",
        raw_gold: "gold_ingot",
        raw_copper: "copper_ingot",
        iron_ingot: "iron_ingot",      // already-smelted aliases
        gold_ingot: "gold_ingot",
        copper_ingot: "copper_ingot",
        sand: "glass",
        cobblestone: "stone",
        stone: "smooth_stone",
        clay_ball: "brick",
        netherrack: "nether_brick",
        ancient_debris: "netherite_scrap",
        kelp: "dried_kelp",
        wet_sponge: "sponge",
        cactus: "green_dye",
        wood: "charcoal",           // generic; oak_log → charcoal handled separately
        oak_log: "charcoal",
        spruce_log: "charcoal",
        birch_log: "charcoal",
        jungle_log: "charcoal",
        acacia_log: "charcoal",
        dark_oak_log: "charcoal",
        mangrove_log: "charcoal",
        beef: "cooked_beef",
        porkchop: "cooked_porkchop",
        chicken: "cooked_chicken",
        mutton: "cooked_mutton",
        cod: "cooked_cod",
        salmon: "cooked_salmon",
        rabbit: "cooked_rabbit",
        potato: "baked_potato",
    };
    const outputName = SMELT_OUTPUT[itemName] || itemName;
    const outputMeta = mcData.itemsByName[outputName];
    if (outputMeta) {
        const alreadyHave = bot.inventory.items()
            .filter(i => i.id === outputMeta.id)
            .reduce((s, i) => s + i.count, 0);
        if (alreadyHave >= count) {
            bot.chat(`Already have ${alreadyHave} ${outputName}, skipping smelt.`);
            return;
        }
        // Only smelt the remaining deficit
        count = count - alreadyHave;
    }

    const item = mcData.itemsByName[itemName];
    const fuel = mcData.itemsByName[fuelName];
    if (!item) {
        throw new Error(`No item named ${itemName}`);
    }
    if (!fuel) {
        throw new Error(`No item named ${fuelName}`);
    }
    let furnaceBlock = bot.findBlock({
        matching: mcData.blocksByName.furnace.id,
        maxDistance: 32,
    });
    if (!furnaceBlock) {
        // Auto-craft furnace from cobblestone if not in inventory
        // (furnace requires 8 cobblestone and a crafting table)
        let furnaceInInventory = bot.inventory.findInventoryItem(
            mcData.itemsByName.furnace.id
        );
        if (!furnaceInInventory) {
            const cobblestoneId = mcData.itemsByName.cobblestone?.id;
            const cobbleCount = cobblestoneId
                ? bot.inventory.items().filter(i => i.id === cobblestoneId).reduce((s,i) => s+i.count, 0)
                : 0;
            if (cobbleCount >= 8) {
                bot.chat("Auto-crafting furnace from cobblestone...");
                try {
                    await craftItem(bot, "furnace", 1);
                } catch (e) {
                    bot.chat(`Could not auto-craft furnace: ${e.message}`);
                }
                furnaceInInventory = bot.inventory.findInventoryItem(
                    mcData.itemsByName.furnace.id
                );
            }
        }
        if (furnaceInInventory) {
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
                    await placeItem(bot, "furnace", base.offset(off.x, off.y, off.z));
                } catch (e) {
                    // placement failed at this offset, try the next one
                }
                furnaceBlock = bot.findBlock({
                    matching: mcData.blocksByName.furnace.id,
                    maxDistance: 32,
                });
                if (furnaceBlock) break;
            }
            // Fallback: mine one adjacent block to create space for the furnace.
            if (!furnaceBlock) {
                const digOffsets = [
                    new Vec3(1, 0, 0), new Vec3(-1, 0, 0),
                    new Vec3(0, 0, 1), new Vec3(0, 0, -1),
                ];
                for (const off of digOffsets) {
                    const pos = base.offset(off.x, off.y, off.z);
                    const block = bot.blockAt(pos);
                    if (block && block.name !== "air" && block.diggable) {
                        try {
                            bot.chat(`Mining ${block.name} to make space for furnace`);
                            await bot.dig(block);
                            await placeItem(bot, "furnace", pos);
                        } catch (e) { /* try next */ }
                        furnaceBlock = bot.findBlock({
                            matching: mcData.blocksByName.furnace.id,
                            maxDistance: 32,
                        });
                        if (furnaceBlock) break;
                    }
                }
            }
        }
        if (!furnaceBlock) {
            throw new Error("No furnace nearby — craft and place a furnace first");
        }
    }
    await bot.pathfinder.goto(
        new GoalLookAtBlock(furnaceBlock.position, bot.world)
    );
    const furnace = await bot.openFurnace(furnaceBlock);
    let success_count = 0;
    for (let i = 0; i < count; i++) {
        if (!bot.inventory.findInventoryItem(item.id, null)) {
            bot.chat(`No ${itemName} to smelt in inventory`);
            break;
        }
        if (furnace.fuelSeconds < 15 && furnace.fuelItem()?.name !== fuelName) {
            if (!bot.inventory.findInventoryItem(fuel.id, null)) {
                bot.chat(`No ${fuelName} as fuel in inventory`);
                break;
            }
            await furnace.putFuel(fuel.id, null, 1);
            await bot.waitForTicks(20);
            if (!furnace.fuel && furnace.fuelItem()?.name !== fuelName) {
                throw new Error(`${fuelName} is not a valid fuel`);
            }
        }
        await furnace.putInput(item.id, null, 1);
        await bot.waitForTicks(12 * 20);
        if (!furnace.outputItem()) {
            throw new Error(`${itemName} is not a valid input`);
        }
        await furnace.takeOutput();
        success_count++;
    }
    furnace.close();
    if (success_count > 0) bot.chat(`Smelted ${success_count} ${itemName}.`);
    else {
        bot.chat(
            `Failed to smelt ${itemName}, please check the fuel and input.`
        );
        _smeltItemFailCount++;
        if (_smeltItemFailCount > 10) {
            throw new Error(
                `smeltItem failed too many times, please check the fuel and input.`
            );
        }
    }
}
