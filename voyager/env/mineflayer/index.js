const fs = require("fs");
const express = require("express");
const bodyParser = require("body-parser");
const mineflayer = require("mineflayer");

const skills = require("./lib/skillLoader");
const { initCounter, getNextTime } = require("./lib/utils");
const obs = require("./lib/observation/base");
const OnChat = require("./lib/observation/onChat");
const OnError = require("./lib/observation/onError");
const { Voxels, BlockRecords } = require("./lib/observation/voxels");
const Status = require("./lib/observation/status");
const Inventory = require("./lib/observation/inventory");
const OnSave = require("./lib/observation/onSave");
const Chests = require("./lib/observation/chests");
const { plugin: tool } = require("mineflayer-tool");

let bot = null;

const app = express();

app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: false }));

app.post("/start", (req, res) => {
    if (bot) onDisconnect("Restarting bot");
    bot = null;
    console.log(req.body);
    bot = mineflayer.createBot({
        host: "localhost", // minecraft server ip
        port: req.body.port, // minecraft server port
        username: "bot",
        disableChatSigning: true,
        checkTimeoutInterval: 60 * 60 * 1000,
    });
    bot.once("error", onConnectionFailed);

    // Event subscriptions
    bot.waitTicks = req.body.waitTicks;
    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];
    bot.iron_pickaxe = false;
    bot.has_furnace = false;
    bot.has_crafting_table = false;

    // "kicked" already triggers mineflayer's internal disconnect — do NOT call
    // bot.end() again here or you get "handleDisconnection() called twice".
    bot.on("kicked", (reason) => {
        console.log(`Bot was kicked: ${reason}`);
        bot = null;
    });

    // mounting will cause physicsTick to stop
    bot.on("mount", () => {
        bot.dismount();
    });

    bot.once("spawn", async () => {
        try {
            bot.removeListener("error", onConnectionFailed);
            let itemTicks = 1;
            if (req.body.reset === "hard") {
                bot.iron_pickaxe = false;
                bot.has_furnace = false;
                bot.has_crafting_table = false;
                bot.chat("/clear @s");
                bot.chat("/effect clear @s");
                const inventory = req.body.inventory ? req.body.inventory : {};
                const equipment = req.body.equipment
                    ? req.body.equipment
                    : [null, null, null, null, null, null];
                for (let key in inventory) {
                    bot.chat(`/give @s minecraft:${key} ${inventory[key]}`);
                    itemTicks += 1;
                }
                if (inventory["iron_pickaxe"]) {
                    bot.iron_pickaxe = true;
                }
                // Restore persistent-tool flags from Python (seen_items-based).
                // If the bot has ever had furnace/crafting_table, ensure it always
                // gets one back after a hard reset (mirrors iron_pickaxe behaviour).
                const persistentTools = req.body.persistent_tools || [];
                if (persistentTools.includes("furnace") || inventory["furnace"]) {
                    bot.has_furnace = true;
                    if (!inventory["furnace"]) {
                        bot.chat("/give @s minecraft:furnace 1");
                        itemTicks += 1;
                    }
                }
                if (persistentTools.includes("crafting_table") || inventory["crafting_table"]) {
                    bot.has_crafting_table = true;
                    if (!inventory["crafting_table"]) {
                        bot.chat("/give @s minecraft:crafting_table 1");
                        itemTicks += 1;
                    }
                }
                const equipmentNames = [
                    "armor.head",
                    "armor.chest",
                    "armor.legs",
                    "armor.feet",
                    "weapon.mainhand",
                    "weapon.offhand",
                ];
                for (let i = 0; i < 6; i++) {
                    if (i === 4) continue;
                    if (equipment[i]) {
                        bot.chat(
                            `/item replace entity @s ${equipmentNames[i]} with minecraft:${equipment[i]}`
                        );
                        itemTicks += 1;
                    }
                }
            }

            // Soft-reset: update persistent flags from current inventory.
            // Hard-reset already set these above; soft-reset just keeps them live.
            if (req.body.reset !== "hard") {
                if (bot.inventory.items().find((item) => item.name === "iron_pickaxe"))
                    bot.iron_pickaxe = true;
                if (bot.inventory.items().find((item) => item.name === "furnace"))
                    bot.has_furnace = true;
                if (bot.inventory.items().find((item) => item.name === "crafting_table"))
                    bot.has_crafting_table = true;
            }

            const { pathfinder } = require("mineflayer-pathfinder");
            const tool = require("mineflayer-tool").plugin;
            const collectBlock = require("mineflayer-collectblock").plugin;
            const pvp = require("mineflayer-pvp").plugin;
            const minecraftHawkEye = require("minecrafthawkeye").default;
            bot.loadPlugin(pathfinder);
            bot.loadPlugin(tool);
            bot.loadPlugin(collectBlock);
            bot.loadPlugin(pvp);
            bot.loadPlugin(minecraftHawkEye);

            obs.inject(bot, [
                OnChat,
                OnError,
                Voxels,
                Status,
                Inventory,
                OnSave,
                Chests,
                BlockRecords,
            ]);
            skills.inject(bot);

            await bot.waitForTicks(bot.waitTicks * itemTicks);

            // safeTP: disable physics so mineflayer doesn't send stale displacement
            // packets while the server relocates the bot. Use forcedMove event
            // (fires on PlayerPositionAndLook packet) rather than waitForTicks,
            // because physicsTick only fires when physicsEnabled=true — calling
            // waitForTicks with physicsEnabled=false would hang forever.
            const safeTP = async (x, y, z) => {
                bot.pathfinder.setGoal(null);  // clear goal BEFORE stop to prevent "goal changed" flood
                bot.pathfinder.stop();
                // Disable physics FIRST, then zero velocity BEFORE sending /tp.
                // Previously waitForTicks(3) ran here with physics still on, which
                // let mineflayer emit position packets carrying stale velocity right
                // as the server processed the teleport → "moved too quickly" kick.
                bot.physicsEnabled = false;
                if (bot.entity && bot.entity.velocity) bot.entity.velocity.set(0, 0, 0);
                // Brief setTimeout (no physics ticks) to let any in-flight position
                // packets drain before the server relocates the bot.
                await new Promise(r => setTimeout(r, 150));
                bot.chat(`/tp @s ${x} ${y} ${z}`);
                await new Promise(resolve => {
                    const timeout = setTimeout(resolve, 3000);
                    bot.once("forcedMove", () => { clearTimeout(timeout); resolve(); });
                });
                if (bot.entity && bot.entity.velocity) bot.entity.velocity.set(0, 0, 0);
                await new Promise(r => setTimeout(r, 500));
                bot.physicsEnabled = true;
                await new Promise(r => setTimeout(r, 250)); // ~5 ticks, no waitForTicks needed
            };

            // Explicit position tp requested by Python (e.g. check_process restart)
            if (req.body.position) {
                const p = req.body.position;
                await safeTP(p.x, p.y, p.z);
            }

            // Safety check: if bot is inside a solid block (underground spawn),
            // tp to a known-safe surface coordinate. Only trigger on actual
            // suffocation — NOT on a y-threshold — to avoid kicking the bot when
            // it spawns legitimately below y=64 on the surface.
            try {
                if (bot && bot.entity) {
                    const block = bot.blockAt(bot.entity.position);
                    if (block && block.boundingBox === "block") {
                        console.log(`[safeTP] bot inside solid block ${block.name} at y=${bot.entity.position.y.toFixed(1)}, tping to safe spawn`);
                        await safeTP(64, 65, 0);
                    }
                }
            } catch (e) {
                console.log(`[safeTP check error] ${e.message}`);
            }

            if (!bot) {
                res.status(500).json({ error: "bot disconnected during spawn" });
                return;
            }

            res.json(bot.observe());

            initCounter(bot);
            bot.chat("/gamerule keepInventory true");
            bot.chat("/gamerule doDaylightCycle false");
            bot.chat("/time set day");
            bot.chat("/difficulty peaceful");
        } catch (e) {
            console.error("[spawn error]", e && e.stack ? e.stack : e);
            try { res.status(500).json({ error: String(e) }); } catch (_) {}
        }
    });

    function onConnectionFailed(e) {
        console.log(e);
        bot = null;
        res.status(400).json({ error: e });
    }
    function onDisconnect(message) {
        if (bot.viewer) {
            bot.viewer.close();
        }
        bot.end();
        console.log(message);
        bot = null;
    }
});

app.post("/step", async (req, res) => {
    // import useful package
    let response_sent = false;
    function sendResponse() {
        if (!response_sent) {
            response_sent = true;
            try { res.json(bot.observe()); } catch (_) { res.status(500).json({ error: "bot disconnected" }); }
        }
    }
    // Deduplicate noisy errors so chat is not flooded.
    let _lastErrorMsg = "";
    let _lastErrorCount = 0;
    function otherError(err) {
        const msg = err && err.message ? err.message : String(err);
        // "The goal was changed before it could be completed!" is a harmless
        // pathfinder abort that fires after every safeTP.  Swallow it entirely —
        // printing it to chat serves no purpose and spams the log.
        if (msg.includes("goal was changed")) return;

        // For any other error, emit once and deduplicate rapid repeats.
        if (msg === _lastErrorMsg) {
            _lastErrorCount++;
            if (_lastErrorCount > 3) return;   // suppress beyond 3 identical errors
        } else {
            _lastErrorMsg = msg;
            _lastErrorCount = 1;
        }
        console.log("Uncaught Error");
        bot.emit("error", handleError(err));
        bot.waitForTicks(bot.waitTicks).then(sendResponse);
    }

    // If the bot disconnects mid-step (e.g. "handleDisconnection called twice",
    // server timeout, etc.), send a response immediately so Python doesn't hang.
    bot.once("end", (reason) => {
        console.log(`Bot disconnected during step: ${reason}`);
        bot = null;  // prevent handleDisconnection() called twice on next /start
        sendResponse();
    });

    process.on("uncaughtException", otherError);

    const mcData = require("minecraft-data")(bot.version);
    mcData.itemsByName["leather_cap"] = mcData.itemsByName["leather_helmet"];
    mcData.itemsByName["leather_tunic"] =
        mcData.itemsByName["leather_chestplate"];
    mcData.itemsByName["leather_pants"] =
        mcData.itemsByName["leather_leggings"];
    mcData.itemsByName["leather_boots"] = mcData.itemsByName["leather_boots"];
    mcData.itemsByName["lapis_lazuli_ore"] = mcData.itemsByName["lapis_ore"];
    mcData.blocksByName["lapis_lazuli_ore"] = mcData.blocksByName["lapis_ore"];
    const {
        Movements,
        goals: {
            Goal,
            GoalBlock,
            GoalNear,
            GoalXZ,
            GoalNearXZ,
            GoalY,
            GoalGetToBlock,
            GoalLookAtBlock,
            GoalBreakBlock,
            GoalCompositeAny,
            GoalCompositeAll,
            GoalInvert,
            GoalFollow,
            GoalPlaceBlock,
        },
        pathfinder,
        Move,
        ComputedPath,
        PartiallyComputedPath,
        XZCoordinates,
        XYZCoordinates,
        SafeBlock,
        GoalPlaceBlockOptions,
    } = require("mineflayer-pathfinder");
    const { Vec3 } = require("vec3");

    // Set up pathfinder
    const movements = new Movements(bot, mcData);
    bot.pathfinder.setMovements(movements);

    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];

    function onTick() {
        bot.globalTickCounter++;
        if (bot.pathfinder.isMoving()) {
            bot.stuckTickCounter++;
            if (bot.stuckTickCounter >= 100) {
                onStuck(1.5);
                bot.stuckTickCounter = 0;
            }
        }
    }

    bot.on("physicTick", onTick);

    // initialize fail count
    let _craftItemFailCount = 0;
    let _killMobFailCount = 0;
    let _mineBlockFailCount = 0;
    let _placeItemFailCount = 0;
    let _smeltItemFailCount = 0;

    // Retrieve array form post bod
    const code = req.body.code;
    const programs = req.body.programs;
    bot.cumulativeObs = [];
    await bot.waitForTicks(bot.waitTicks);
    const r = await evaluateCode(code, programs);
    process.off("uncaughtException", otherError);
    if (r !== "success") {
        bot.emit("error", handleError(r));
    }
    await returnItems();
    // wait for last message
    await bot.waitForTicks(bot.waitTicks);
    sendResponse();
    bot.removeListener("physicTick", onTick);

    async function evaluateCode(code, programs) {
        // Echo the code produced for players to see it. Don't echo when the bot code is already producing dialog or it will double echo
        try {
            await eval("(async () => {" + programs + "\n" + code + "})()");
            return "success";
        } catch (err) {
            return err;
        }
    }

    function onStuck(posThreshold) {
        const currentPos = bot.entity.position;
        bot.stuckPosList.push(currentPos);

        // Check if the list is full
        if (bot.stuckPosList.length === 5) {
            const oldestPos = bot.stuckPosList[0];
            const posDifference = currentPos.distanceTo(oldestPos);

            if (posDifference < posThreshold) {
                teleportBot(); // execute the function
            }

            // Remove the oldest time from the list
            bot.stuckPosList.shift();
        }
    }

    function teleportBot() {
        bot.pathfinder.stop();
        const blocks = bot.findBlocks({
            matching: (block) => {
                return block.type === 0;
            },
            maxDistance: 1,
            count: 27,
        });

        if (blocks) {
            const randomIndex = Math.floor(Math.random() * blocks.length);
            const block = blocks[randomIndex];
            bot.chat(`/tp @s ${block.x} ${block.y} ${block.z}`);
        } else {
            bot.chat("/tp @s ~ ~1.25 ~");
        }
    }

    function returnItems() {
        bot.chat("/gamerule doTileDrops false");
        const crafting_table = bot.findBlock({
            matching: mcData.blocksByName.crafting_table.id,
            maxDistance: 128,
        });
        if (crafting_table) {
            bot.chat(
                `/setblock ${crafting_table.position.x} ${crafting_table.position.y} ${crafting_table.position.z} air destroy`
            );
            bot.chat("/give @s crafting_table");
        }
        const furnace = bot.findBlock({
            matching: mcData.blocksByName.furnace.id,
            maxDistance: 128,
        });
        if (furnace) {
            bot.chat(
                `/setblock ${furnace.position.x} ${furnace.position.y} ${furnace.position.z} air destroy`
            );
            bot.chat("/give @s furnace");
        }
        if (bot.inventoryUsed() >= 32) {
            // if chest is not in bot's inventory
            if (!bot.inventory.items().find((item) => item.name === "chest")) {
                bot.chat("/give @s chest");
            }
        }
        // if iron_pickaxe not in bot's inventory and bot.iron_pickaxe
        if (
            bot.iron_pickaxe &&
            !bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        ) {
            bot.chat("/give @s iron_pickaxe");
        }
        // Persistent utility tools: give back furnace / crafting_table if the bot
        // has ever had them and they are currently missing from inventory.
        // Mirrors the iron_pickaxe behaviour — once earned, never lose them.
        if (bot.has_furnace &&
            !bot.inventory.items().find((item) => item.name === "furnace")) {
            bot.chat("/give @s minecraft:furnace 1");
        }
        if (bot.has_crafting_table &&
            !bot.inventory.items().find((item) => item.name === "crafting_table")) {
            bot.chat("/give @s minecraft:crafting_table 1");
        }
        bot.chat("/gamerule doTileDrops true");
        // Re-apply game settings in case action code modified them
        bot.chat("/difficulty peaceful");
        bot.chat("/gamerule doDaylightCycle false");
    }

    function handleError(err) {
        let stack = err.stack;
        if (!stack) {
            return err;
        }
        console.log(stack);
        const final_line = stack.split("\n")[1];
        const regex = /<anonymous>:(\d+):\d+\)/;

        const programs_length = programs.split("\n").length;
        let match_line = null;
        for (const line of stack.split("\n")) {
            const match = regex.exec(line);
            if (match) {
                const line_num = parseInt(match[1]);
                if (line_num >= programs_length) {
                    match_line = line_num - programs_length;
                    break;
                }
            }
        }
        if (!match_line) {
            return err.message;
        }
        let f_line = final_line.match(
            /\((?<file>.*):(?<line>\d+):(?<pos>\d+)\)/
        );
        if (f_line && f_line.groups && fs.existsSync(f_line.groups.file)) {
            const { file, line, pos } = f_line.groups;
            const f = fs.readFileSync(file, "utf8").split("\n");
            // let filename = file.match(/(?<=node_modules\\)(.*)/)[1];
            let source = file + `:${line}\n${f[line - 1].trim()}\n `;

            const code_source =
                "at " +
                code.split("\n")[match_line - 1].trim() +
                " in your code";
            return source + err.message + "\n" + code_source;
        } else if (
            f_line &&
            f_line.groups &&
            f_line.groups.file.includes("<anonymous>")
        ) {
            const { file, line, pos } = f_line.groups;
            let source =
                "Your code" +
                `:${match_line}\n${code.split("\n")[match_line - 1].trim()}\n `;
            let code_source = "";
            if (line < programs_length) {
                source =
                    "In your program code: " +
                    programs.split("\n")[line - 1].trim() +
                    "\n";
                code_source = `at line ${match_line}:${code
                    .split("\n")
                    [match_line - 1].trim()} in your code`;
            }
            return source + err.message + "\n" + code_source;
        }
        return err.message;
    }
});

app.post("/stop", (req, res) => {
    bot.end();
    res.json({
        message: "Bot stopped",
    });
});

app.post("/pause", (req, res) => {
    // Pause disabled: server pause/unpause cycle causes "Invalid move player
    // packet" kicks due to mineflayer physics running while server is paused.
    // /difficulty peaceful + /gamerule doDaylightCycle false keep world stable.
    res.json({ message: "noop" });
});

// Server listening to PORT 3000

const DEFAULT_PORT = 3000;
const PORT = process.argv[2] || DEFAULT_PORT;
app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});
