async function getItemFromChest(bot, chestPosition, itemsToGet) {
    // return if chestPosition is not Vec3
    if (!(chestPosition instanceof Vec3)) {
        bot.chat("chestPosition for getItemFromChest must be a Vec3");
        return;
    }
    await moveToChest(bot, chestPosition);
    const chestBlock = bot.blockAt(chestPosition);
    const chest = await bot.openContainer(chestBlock);
    for (const name in itemsToGet) {
        const itemByName = mcData.itemsByName[name];
        if (!itemByName) {
            bot.chat(`No item named ${name}`);
            continue;
        }

        const item = chest.findContainerItem(itemByName.id);
        if (!item) {
            bot.chat(`I don't see ${name} in this chest`);
            continue;
        }
        try {
            await chest.withdraw(item.type, null, itemsToGet[name]);
        } catch (err) {
            bot.chat(`Not enough ${name} in chest.`);
        }
    }
    await closeChest(bot, chestBlock);
}

async function depositItemIntoChest(bot, chestPosition, itemsToDeposit) {
    // return if chestPosition is not Vec3
    if (!(chestPosition instanceof Vec3)) {
        throw new Error(
            "chestPosition for depositItemIntoChest must be a Vec3"
        );
    }
    await moveToChest(bot, chestPosition);
    const chestBlock = bot.blockAt(chestPosition);
    const chest = await bot.openContainer(chestBlock);
    for (const name in itemsToDeposit) {
        const itemByName = mcData.itemsByName[name];
        if (!itemByName) {
            bot.chat(`No item named ${name}`);
            continue;
        }
        const item = bot.inventory.findInventoryItem(itemByName.id);
        if (!item) {
            bot.chat(`No ${name} in inventory`);
            continue;
        }
        try {
            await chest.deposit(item.type, null, itemsToDeposit[name]);
        } catch (err) {
            bot.chat(`Not enough ${name} in inventory.`);
        }
    }
    await closeChest(bot, chestBlock);
}

async function checkItemInsideChest(bot, chestPosition) {
    // return if chestPosition is not Vec3
    if (!(chestPosition instanceof Vec3)) {
        throw new Error(
            "chestPosition for checkItemInsideChest must be a Vec3"
        );
    }
    // moveToChest navigates to the chest AND returns the chestBlock.
    // We do NOT call bot.openContainer separately — closeChest (via
    // listItemsInChest) will open it exactly once, read its contents,
    // emit the "closeChest" event that updates Python's chest registry,
    // and close the window.  Calling openContainer twice on the same
    // window causes the second windowOpen event to never fire (20 s timeout).
    const chestBlock = await moveToChest(bot, chestPosition);
    await closeChest(bot, chestBlock);
}

async function moveToChest(bot, chestPosition) {
    if (!(chestPosition instanceof Vec3)) {
        throw new Error(
            "chestPosition for depositItemIntoChest must be a Vec3"
        );
    }
    if (chestPosition.distanceTo(bot.entity.position) > 32) {
        // Use the same safe-TP sequence as safeTP() in index.js:
        // disable physics BEFORE sending /tp to prevent "moved too quickly".
        bot.pathfinder.setGoal(null);  // clear goal before stop to prevent "goal changed" flood
        bot.pathfinder.stop();
        bot.physicsEnabled = false;
        if (bot.entity && bot.entity.velocity) bot.entity.velocity.set(0, 0, 0);
        await new Promise(r => setTimeout(r, 150));

        // For underground chests (y < 60) teleport directly to the chest Y
        // so pathfinder can navigate through the existing tunnel.
        // (Teleporting to surface y=65 leaves the bot above solid rock with
        //  no tunnel access, causing GoalLookAtBlock to fail.)
        const tpY = chestPosition.y;
        bot.chat(`/tp @s ${chestPosition.x} ${tpY} ${chestPosition.z}`);

        // Wait for the server's forcedMove confirmation (same as safeTP)
        await new Promise(resolve => {
            const timeout = setTimeout(resolve, 3000);
            bot.once("forcedMove", () => { clearTimeout(timeout); resolve(); });
        });
        if (bot.entity && bot.entity.velocity) bot.entity.velocity.set(0, 0, 0);
        await new Promise(r => setTimeout(r, 500));
        bot.physicsEnabled = true;
        await new Promise(r => setTimeout(r, 250));
    }
    const chestBlock = bot.blockAt(chestPosition);
    if (chestBlock.name !== "chest") {
        bot.emit("removeChest", chestPosition);
        throw new Error(
            `No chest at ${chestPosition}, it is ${chestBlock.name}`
        );
    }
    await bot.pathfinder.goto(
        new GoalLookAtBlock(chestBlock.position, bot.world, {})
    );
    return chestBlock;
}

async function listItemsInChest(bot, chestBlock) {
    const chest = await bot.openContainer(chestBlock);
    const items = chest.containerItems();
    if (items.length > 0) {
        const itemNames = items.reduce((acc, obj) => {
            if (acc[obj.name]) {
                acc[obj.name] += obj.count;
            } else {
                acc[obj.name] = obj.count;
            }
            return acc;
        }, {});
        bot.emit("closeChest", itemNames, chestBlock.position);
    } else {
        bot.emit("closeChest", {}, chestBlock.position);
    }
    return chest;
}

async function closeChest(bot, chestBlock) {
    try {
        const chest = await listItemsInChest(bot, chestBlock);
        await chest.close();
    } catch (err) {
        await bot.closeWindow(chestBlock);
    }
}

function itemByName(items, name) {
    for (let i = 0; i < items.length; ++i) {
        const item = items[i];
        if (item && item.name === name) return item;
    }
    return null;
}
