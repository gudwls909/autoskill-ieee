async function mineBlock(bot, name, count = 1) {
    if (typeof name !== "string") {
        throw new Error(`name for mineBlock must be a string`);
    }
    if (typeof count !== "number") {
        throw new Error(`count for mineBlock must be a number`);
    }

    // Normalize common LLM mistakes: item-drop names → actual block names.
    // deepslate_X variants are also mapped to the base ore name so the deepslate
    // inclusion logic below can add both variants automatically.
    const BLOCK_ALIASES = {
        raw_copper: "copper_ore",
        raw_iron: "iron_ore",
        raw_gold: "gold_ore",
        lapis_lazuli: "lapis_ore",
        lapis_lazuli_ore: "lapis_ore",
        redstone: "redstone_ore",
        emerald: "emerald_ore",
        diamond: "diamond_ore",
        coal: "coal_ore",
        quartz: "nether_quartz_ore",
        // deepslate variants → base ore so both layers are searched together
        deepslate_gold_ore: "gold_ore",
        deepslate_iron_ore: "iron_ore",
        deepslate_copper_ore: "copper_ore",
        deepslate_lapis_ore: "lapis_ore",
        deepslate_redstone_ore: "redstone_ore",
        deepslate_emerald_ore: "emerald_ore",
        deepslate_diamond_ore: "diamond_ore",
        deepslate_coal_ore: "coal_ore",
    };
    name = BLOCK_ALIASES[name] || name;

    const blockByName = mcData.blocksByName[name];
    if (!blockByName) {
        throw new Error(`No block named ${name}`);
    }

    // Include deepslate variant if available
    const matchIds = [blockByName.id];
    const deepslateVariant = mcData.blocksByName[`deepslate_${name}`];
    if (deepslateVariant) matchIds.push(deepslateVariant.id);

    const findTargets = () => {
        const positions = bot.findBlocks({ matching: matchIds, maxDistance: 32, count: 1024 });
        return positions.map(p => bot.blockAt(p)).filter(Boolean);
    };

    let targets = findTargets();
    if (targets.length === 0) {
        bot.chat(`No ${name} nearby, please explore first`);
        _mineBlockFailCount++;
        if (_mineBlockFailCount > 10) {
            throw new Error(
                "mineBlock failed too many times, make sure you explore before calling mineBlock"
            );
        }
        return;
    }

    // Navigate near the closest target first so collectBlock can reach it.
    // If pathfinding to the nearest target fails, those specific blocks are
    // unreachable — move in a random direction to find different ones instead.
    const nearest = targets.reduce((a, b) =>
        bot.entity.position.distanceTo(a.position) <= bot.entity.position.distanceTo(b.position) ? a : b
    );
    try {
        await bot.pathfinder.goto(new GoalNear(nearest.position.x, nearest.position.y, nearest.position.z, 3));
    } catch (e) {
        // Nearby blocks unreachable — move further away to find different ones
        bot.chat(`Cannot reach nearby ${name}, searching further...`);
        const angle = Math.random() * 2 * Math.PI;
        const destX = bot.entity.position.x + Math.cos(angle) * 60;
        const destZ = bot.entity.position.z + Math.sin(angle) * 60;
        try {
            await bot.pathfinder.goto(new GoalNearXZ(destX, destZ, 2));
        } catch (e2) {
            bot.chat(`Could not move toward ${name}, continuing anyway...`);
        }
    }
    // Explicitly stop pathfinder before handing control to collectBlock.
    // Without this, collectBlock's internal goto() can race with the residual
    // state of the previous goto(), causing "goal was changed" errors.
    bot.pathfinder.stop();

    // Re-scan with extended radius after moving — nearby trees may be exhausted
    const extendedPositions = bot.findBlocks({ matching: matchIds, maxDistance: 64, count: 1024 });
    targets = extendedPositions.map(p => bot.blockAt(p)).filter(Boolean);
    if (targets.length === 0) {
        bot.chat(`No ${name} found within 64 blocks`);
        _mineBlockFailCount++;
        if (_mineBlockFailCount > 10) {
            throw new Error("mineBlock failed too many times, make sure you explore before calling mineBlock");
        }
        return;
    }

    const collectWithTimeout = (tgts, opts, timeoutMs = 30000) =>
        Promise.race([
            bot.collectBlock.collect(tgts, opts),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error(
                    `mineBlock timed out after ${timeoutMs / 1000}s — block may be unreachable (e.g. underwater)`
                )), timeoutMs)
            ),
        ]);

    try {
        await collectWithTimeout(targets, { ignoreNoPath: true, count: count });
    } catch (err) {
        const msg = err.message || "";
        if (msg.includes("PathStopped") || msg.includes("No path") ||
            msg.includes("timed out") || msg.includes("goal was changed")) {
            // "goal was changed" = pathfinder interrupted mid-collect; safe to retry.
            bot.chat(`Path issue mining ${name}, retrying once...`);
            bot.pathfinder.stop();
            const retryTargets = findTargets();
            if (retryTargets.length > 0) {
                await collectWithTimeout(retryTargets, { ignoreNoPath: true, count: count });
            }
        } else {
            throw err;
        }
    }
    bot.save(`${name}_mined`);
}
