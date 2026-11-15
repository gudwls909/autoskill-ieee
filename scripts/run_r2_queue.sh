#!/bin/bash
# run_r2_queue.sh SLOT JOB... — round-2 revision runs (R3.1 baseline, R3.3/R5.4 model swap).
# Each JOB is METHOD:SEED with METHOD in
#   ed   Voyager + embedding-similarity dedup (~/code/voyager_ed)
#   vb   Voyager baseline                     (~/code/voyager_vb)
#   as   AutoSkill                            (~/code/voyager)
#   nd / nc / ncur   AutoSkill ablations: w/o Dedup, w/o Critique, w/o composite-aware Curriculum
#   vp   Voyager on AutoSkill's shared infrastructure (all three AutoSkill components off)
# Run id = run-r2-${TAG}-${METHOD}-seed${SEED}; TAG (env) names the model setup, e.g. mini / g54 / qwen14b.
# Model overrides pass through env: MODEL_MAIN, MODEL_SMALL, LLM_BASE_URL, SKILL_SEMANTIC_MODEL.
# Same protocol as the paper runs: fresh world, spawn 64/65/0, 150 steps, every run kept.
set -u
SLOT="${1:?slot 1-6}"; shift; TAG="${TAG:?set TAG}"
case "$SLOT" in
  1) MC_DIR=~/mc_server_119;  MC_PORT=25565; RCON_PORT=25575; BRIDGE_PORT=3000 ;;
  2) MC_DIR=~/mc_server_119b; MC_PORT=25566; RCON_PORT=25576; BRIDGE_PORT=3001 ;;
  3) MC_DIR=~/mc_server_119c; MC_PORT=25567; RCON_PORT=25577; BRIDGE_PORT=3002 ;;
  4) MC_DIR=~/mc_server_119d; MC_PORT=25568; RCON_PORT=25578; BRIDGE_PORT=3003 ;;
  5) MC_DIR=~/mc_server_119e; MC_PORT=25569; RCON_PORT=25579; BRIDGE_PORT=3004 ;;
  6) MC_DIR=~/mc_server_119f; MC_PORT=25570; RCON_PORT=25580; BRIDGE_PORT=3005 ;;
  *) echo "bad slot"; exit 2 ;;
esac
RCON_PASS=voyager_rcon
LOG=~/code/r2_queue_slot${SLOT}.log
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"
export MC_PORT SERVER_PORT=$BRIDGE_PORT MC_WORLD="$MC_DIR/world"
rcon(){ printf '%s\n' "$1" | mcrcon --password "$RCON_PASS" -p "$RCON_PORT" localhost 2>/dev/null || true; }

for job in "$@"; do
  METHOD="${job%%:*}"; SEED="${job##*:}"
  case "$METHOD" in
    ed) REPO=~/code/voyager_ed ;; vb) REPO=~/code/voyager_vb ;; as|nd|nc|ncur|vp) REPO=~/code/voyager ;;
    *) echo "bad method $METHOD" >> "$LOG"; continue ;;
  esac
  RUN_ID="r2-${TAG}-${METHOD}-seed${SEED}"; SERVER_LOG="/tmp/mc_r2_slot${SLOT}.log"
  if [ -f "$REPO/logs/run-$RUN_ID/step_log.jsonl" ] && [ "$(wc -l < "$REPO/logs/run-$RUN_ID/step_log.jsonl")" -ge 140 ]; then
    echo "[slot$SLOT] skip $RUN_ID (done)" >> "$LOG"; continue
  fi
  echo "[slot$SLOT] === $RUN_ID start $(date '+%F %T') ===" >> "$LOG"
  for p in $(pgrep -f fabric-server-launch 2>/dev/null); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$(readlink -f $MC_DIR)" ] && kill -9 "$p" 2>/dev/null
  done
  for p in $(pgrep -f "mineflayer/index.js.*$BRIDGE_PORT" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 2; rm -rf "$MC_DIR/world"
  ( cd "$MC_DIR" && exec java -Xmx6G -Xms2G -jar fabric-server-launch.jar nogui ) > "$SERVER_LOG" 2>&1 &
  MC_PID=$!; WAIT=0
  until grep -q "Done" "$SERVER_LOG" 2>/dev/null; do
    sleep 2; WAIT=$((WAIT+2)); kill -0 "$MC_PID" 2>/dev/null || break; [ $WAIT -gt 180 ] && break
  done
  sleep 2; rcon "setworldspawn 64 65 0"; rcon "gamerule spawnRadius 0"
  case "$METHOD" in nd) ABL=no_dedup ;; nc) ABL=no_critique ;; ncur) ABL=no_curriculum ;; vp) ABL=voyager_infra ;; *) ABL=none ;; esac
  if [ "$REPO" = ~/code/voyager ]; then
    ARGS=(-max "${MAX:-150}" -ablation "$ABL" -mc-port "$MC_PORT" -server-port "$BRIDGE_PORT" -mc-world-path "$MC_DIR/world" -id "$RUN_ID")
  else
    ARGS=(-max "${MAX:-150}" -id "$RUN_ID")
  fi
  ( cd "$REPO" && conda run -n voyager4 --no-capture-output python learn.py "${ARGS[@]}" ) >> "$LOG" 2>&1
  echo "[slot$SLOT] === $RUN_ID exit=$? $(date '+%F %T') ===" >> "$LOG"
  kill -9 "$MC_PID" 2>/dev/null; sleep 2
done
echo "[slot$SLOT] ALL DONE $(date '+%F %T')" >> "$LOG"
