#!/bin/bash
# run_resub_seed.sh — honest (no-discard) single-seed runner for the IEEE Access
# resubmission n=10 stability evidence (R2.3).
#
# Usage: run_resub_seed.sh as <seed>   # AutoSkill  (main branch copy,  slot2)
#        run_resub_seed.sh vb <seed>   # Voyager    (voyager-baseline copy, slot1)
#
# Keeps EVERY completed run: NO TTS-threshold discard, NO early-abort watchdog.
# Same per-run config as submission (gpt-5.4-mini, 150 steps, world spawn 64/65/0).
set -uo pipefail
METHOD="${1:?usage: as|vb <seed>}"; SEED="${2:?seed}"
CONDA_ENV=voyager4
RCON_PASS=voyager_rcon

if [ "$METHOD" = "as" ]; then
    REPO="$HOME/code/voyager_as"
    MC_DIR="$HOME/mc_server_119b"; MC_PORT=25566; RCON_PORT=25576; BRIDGE_PORT=3001
    RUN_ID="resub-as-seed${SEED}"
    LEARN_ARGS=(-max 150 -ablation none -mc-port "$MC_PORT" -server-port "$BRIDGE_PORT" \
                -mc-world-path "$MC_DIR/world" -id "$RUN_ID")
elif [ "$METHOD" = "pre" ]; then
    REPO="$HOME/code/voyager_pre"
    MC_DIR="$HOME/mc_server_119b"; MC_PORT=25566; RCON_PORT=25576; BRIDGE_PORT=3001
    RUN_ID="resub-pre-seed${SEED}"
    LEARN_ARGS=(-max 150 -id "$RUN_ID")   # 9f6fb41: mc_port/world hardcoded (patched to slot2)
elif [ "$METHOD" = "vb" ]; then
    REPO="$HOME/code/voyager_vb"
    MC_DIR="$HOME/mc_server_119"; MC_PORT=25565; RCON_PORT=25575; BRIDGE_PORT=3000
    RUN_ID="resub-vb-seed${SEED}"
    LEARN_ARGS=(-max 150 -id "$RUN_ID")   # voyager-baseline: mc_port/world hardcoded to slot1
elif [ "$METHOD" = "vb2" ]; then
    REPO="$HOME/code/voyager_vb2"
    MC_DIR="$HOME/mc_server_119b"; MC_PORT=25566; RCON_PORT=25576; BRIDGE_PORT=3001
    RUN_ID="resub-vb-seed${SEED}"
    LEARN_ARGS=(-max 150 -id "$RUN_ID")   # voyager-baseline copy, ports patched to slot2
else
    echo "bad method: $METHOD (use as|vb)"; exit 2
fi

SERVER_LOG="/tmp/mc_${METHOD}_seed${SEED}.log"
rcon(){ printf '%s\n' "$1" | mcrcon --password "$RCON_PASS" -p "$RCON_PORT" localhost 2>/dev/null || true; }

# ── kill stale processes for THIS slot only ─────────────────────────────────
for pid in $(pgrep -f "fabric-server-launch" 2>/dev/null); do
  [ "$(readlink /proc/$pid/cwd 2>/dev/null)" = "$MC_DIR" ] && kill -9 "$pid" 2>/dev/null
done
pkill -9 -f "learn.py.*-id $RUN_ID" 2>/dev/null || true
pkill -9 -f "node.*mineflayer/index.js.*$BRIDGE_PORT" 2>/dev/null || true
sleep 2
rm -rf "$MC_DIR/world"

# ── start MC server ─────────────────────────────────────────────────────────
echo "[$RUN_ID] starting MC server ($MC_DIR port $MC_PORT)..."
( cd "$MC_DIR" && exec java -Xmx6G -Xms2G -jar fabric-server-launch.jar nogui ) > "$SERVER_LOG" 2>&1 &
MC_PID=$!
cleanup(){ kill "$MC_PID" 2>/dev/null || true; wait "$MC_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

WAIT=0
until grep -q "Done" "$SERVER_LOG" 2>/dev/null; do
  sleep 2; WAIT=$((WAIT+2))
  kill -0 "$MC_PID" 2>/dev/null || { echo "[$RUN_ID] MC crashed:"; tail -20 "$SERVER_LOG"; exit 1; }
  [ $WAIT -gt 120 ] && { echo "[$RUN_ID] MC boot timeout:"; tail -20 "$SERVER_LOG"; exit 1; }
done
sleep 2
rcon "setworldspawn 64 65 0"; rcon "gamerule spawnRadius 0"
echo "[$RUN_ID] MC ready (${WAIT}s). launching learn.py (no discard)..."

# ── run learn.py (requires OPENAI_API_KEY in env) ───────────────────────────
cd "$REPO"
conda run -n "$CONDA_ENV" --no-capture-output python learn.py "${LEARN_ARGS[@]}"
CODE=$?
echo "[$RUN_ID] learn.py exit=$CODE"
exit $CODE
