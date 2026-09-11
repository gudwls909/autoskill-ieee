#!/bin/bash
# reset_and_run.sh — Reset Minecraft world and start a fresh Voyager run
#
# Usage:
#   ./reset_and_run.sh              # reset world + start MC server + run learn.py
#   ./reset_and_run.sh --no-learn   # reset world + start MC server only (no learn.py)
#   ./reset_and_run.sh --max 80     # pass --max to learn.py
#   ./reset_and_run.sh --id exp1    # pass --id to learn.py

set -e

MC_SERVER_DIR="/home/sss/mc_server_119"
VOYAGER_DIR="/home/sss/code/voyager"
MC_SESSION="mc_server"      # tmux session name for the Minecraft server
MC_START_WAIT=20            # seconds to wait for server to finish starting

NO_LEARN=false
LEARN_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-learn) NO_LEARN=true; shift ;;
        --max)      LEARN_ARGS="$LEARN_ARGS -max $2"; shift 2 ;;
        --id)       LEARN_ARGS="$LEARN_ARGS -id $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo "========================================"
echo "  Voyager World Reset Script"
echo "========================================"

# ── 1. Stop existing MC server ────────────────────────────────────────────────
echo ""
echo "[1/4] Stopping Minecraft server..."

if tmux has-session -t "$MC_SESSION" 2>/dev/null; then
    echo "  Sending 'stop' to tmux session '$MC_SESSION'..."
    tmux send-keys -t "$MC_SESSION" "stop" Enter
    sleep 8
    # Kill the session if it's still alive
    if tmux has-session -t "$MC_SESSION" 2>/dev/null; then
        tmux kill-session -t "$MC_SESSION" 2>/dev/null || true
    fi
    echo "  Server stopped."
else
    # Fallback: kill any java process running the fabric server
    if pgrep -f "fabric-server-launch.jar" > /dev/null 2>&1; then
        echo "  No tmux session found — killing java process..."
        pkill -f "fabric-server-launch.jar" || true
        sleep 5
        echo "  Server killed."
    else
        echo "  No running server found. Continuing."
    fi
fi

# ── 2. Reset world from backup ────────────────────────────────────────────────
echo ""
echo "[2/4] Resetting world from backup..."

if [ ! -d "$MC_SERVER_DIR/world_backup" ]; then
    echo "ERROR: world_backup not found at $MC_SERVER_DIR/world_backup"
    exit 1
fi

cd "$MC_SERVER_DIR"
rm -rf world
cp -r world_backup world
echo "  World reset complete (world_backup → world)."

# ── 3. Start MC server in tmux ────────────────────────────────────────────────
echo ""
echo "[3/4] Starting Minecraft server in tmux session '$MC_SESSION'..."

tmux new-session -d -s "$MC_SESSION" -c "$MC_SERVER_DIR" \
    "java -Xmx2G -Xms1G -jar fabric-server-launch.jar nogui 2>&1 | tee logs/server_$(date +%Y%m%d_%H%M%S).log"

echo "  Waiting ${MC_START_WAIT}s for server to start..."
sleep "$MC_START_WAIT"

# Quick check: is the session still alive?
if ! tmux has-session -t "$MC_SESSION" 2>/dev/null; then
    echo "ERROR: MC server tmux session died unexpectedly."
    echo "Check logs at $MC_SERVER_DIR/logs/"
    exit 1
fi
echo "  Server should be ready."

# ── 4. Run learn.py ───────────────────────────────────────────────────────────
if [ "$NO_LEARN" = true ]; then
    echo ""
    echo "[4/4] Skipping learn.py (--no-learn flag set)."
    echo ""
    echo "========================================"
    echo "  Done! MC server running in tmux session '$MC_SESSION'."
    echo "  Attach with: tmux attach -t $MC_SESSION"
    echo "========================================"
    exit 0
fi

echo ""
echo "[4/4] Starting Voyager learn.py..."
cd "$VOYAGER_DIR"
python learn.py $LEARN_ARGS

echo ""
echo "========================================"
echo "  Run complete."
echo "========================================"
