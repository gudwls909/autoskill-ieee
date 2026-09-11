#!/bin/bash
# run_fix2_queue.sh START END — validate the craftItem (crafting-table auto-craft) fix.
# Fixed code (~/code/voyager, iclr). slot2 MC (25566, ~/mc_server_119b), fresh world +
# spawn 64/65/0 each run. Keeps every run. Requires OPENAI_API_KEY. Runs concurrently
# with the Voyager baseline on slot1.
set -u
START="${1:?start}"; END="${2:?end}"
REPO="$HOME/code/voyager"; MC_DIR="$HOME/mc_server_119b"
MC_PORT=25566; RCON_PORT=25576; BRIDGE_PORT=3001; RCON_PASS=voyager_rcon
LOG="$HOME/code/resub_fix2_queue.log"
rcon(){ printf '%s\n' "$1" | mcrcon --password "$RCON_PASS" -p "$RCON_PORT" localhost 2>/dev/null || true; }

echo "[fix2 $START-$END] begin $(date '+%F %T')" >> "$LOG"
for s in $(seq "$START" "$END"); do
  RUN_ID="resub-fix2-seed${s}"; SERVER_LOG="/tmp/mc_fix2_seed${s}.log"
  echo "[fix2] === seed $s start $(date '+%F %T') ===" >> "$LOG"
  for p in $(pgrep -f fabric-server-launch 2>/dev/null); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$MC_DIR" ] && kill -9 "$p" 2>/dev/null
  done
  for p in $(pgrep -f "mineflayer/index.js.*$BRIDGE_PORT" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 2; rm -rf "$MC_DIR/world"
  ( cd "$MC_DIR" && exec java -Xmx6G -Xms2G -jar fabric-server-launch.jar nogui ) > "$SERVER_LOG" 2>&1 &
  MC_PID=$!
  WAIT=0
  until grep -q "Done" "$SERVER_LOG" 2>/dev/null; do
    sleep 2; WAIT=$((WAIT+2))
    kill -0 "$MC_PID" 2>/dev/null || break
    [ $WAIT -gt 120 ] && break
  done
  sleep 2; rcon "setworldspawn 64 65 0"; rcon "gamerule spawnRadius 0"
  echo "[fix2] seed $s MC ready (${WAIT}s); learn.py" >> "$LOG"
  ( cd "$REPO" && conda run -n voyager4 --no-capture-output python learn.py -max 150 \
      -mc-port "$MC_PORT" -server-port "$BRIDGE_PORT" -mc-world-path "$MC_DIR/world" \
      -id "$RUN_ID" ) >> "$LOG" 2>&1
  echo "[fix2] === seed $s exit=$? $(date '+%F %T') ===" >> "$LOG"
  kill -9 "$MC_PID" 2>/dev/null; sleep 2
done
echo "[fix2] ALL DONE $(date '+%F %T')" >> "$LOG"
