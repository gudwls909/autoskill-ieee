async function placeItem(bot, name, position) {
    // return if name is not string
    if (typeof name !== "string") {
        throw new Error(`name for placeItem must be a string`);
    }
    // return if position is not Vec3
    if (!(position instanceof Vec3)) {
        throw new Error(`position for placeItem must be a Vec3`);
    }
    // Always work with integer block coordinates — fractional positions
    // (e.g. from bot.entity.position.offset()) cause reference-block lookup failures.
    position = position.floored();
    const itemByName = mcData.itemsByName[name];
    if (!itemByName) {
        throw new Error(`No item named ${name}`);
    }
    const item = bot.inventory.findInventoryItem(itemByName.id);
    if (!item) {
        bot.chat(`No ${name} in inventory`);
        return;
    }
    const item_count = item.count;
    // find a reference block
    const faceVectors = [
        new Vec3(0, 1, 0),
        new Vec3(0, -1, 0),
        new Vec3(1, 0, 0),
        new Vec3(-1, 0, 0),
        new Vec3(0, 0, 1),
        new Vec3(0, 0, -1),
    ];
    // Fast-fail: if the target position is already occupied by a solid block,
    // throw immediately rather than waiting for a 5-second blockUpdate timeout.
    // Must throw (not return) so callers' try/catch can advance to the next offset.
    const targetBlock = bot.blockAt(position);
    if (targetBlock && targetBlock.boundingBox === "block") {
        const msg = `Cannot place ${name}: position ${position} is occupied by ${targetBlock.name}, try another position`;
        bot.chat(msg);
        throw new Error(msg);
    }

    let referenceBlock = null;
    let faceVector = null;
    for (const vector of faceVectors) {
        const block = bot.blockAt(position.minus(vector));
        if (block?.name !== "air") {
            referenceBlock = block;
            faceVector = vector;
            bot.chat(`Placing ${name} on ${block.name} at ${block.position}`);
            break;
        }
    }
    if (!referenceBlock) {
        const msg = `Cannot place ${name}: no adjacent solid block to place against`;
        bot.chat(msg);
        throw new Error(msg);
    }

    // You must use try catch to placeBlock
    try {
        // You must first go to the block position you want to place
        await bot.pathfinder.goto(new GoalPlaceBlock(position, bot.world, {}));
        // You must equip the item right before calling placeBlock
        await bot.equip(item, "hand");
        await bot.placeBlock(referenceBlock, faceVector);
        bot.chat(`Placed ${name}`);
        bot.save(`${name}_placed`);
    } catch (err) {
        // blockUpdate timeout is a server-timing issue: verify by inventory change
        // rather than trusting the error. If item count decreased, placement succeeded.
        const itemAfter = bot.inventory.findInventoryItem(itemByName.id);
        const countAfter = itemAfter ? itemAfter.count : 0;
        if (countAfter < item_count) {
            // Item was consumed — placement succeeded despite timeout error
            bot.chat(`Placed ${name}`);
            bot.save(`${name}_placed`);
        } else {
            bot.chat(
                `Error placing ${name}: ${err.message}, please find another position to place`
            );
            _placeItemFailCount++;
            if (_placeItemFailCount > 10) {
                throw new Error(
                    `placeItem failed too many times, please find another position to place.`
                );
            }
        }
    }
}
