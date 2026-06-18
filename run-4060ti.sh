#!/bin/bash
# Single exo node on the RTX 4060 Ti (--no-api), joining the "mycluster" cluster.
# Runs in its own process group; Ctrl+C cleanly stops it AND the python child
# that `uv run` spawns (so the zenoh port is always released).

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping exo (4060 Ti)..."
  [ -n "$PGID" ] && kill -TERM -- "-$PGID" 2>/dev/null
  sleep 2
  [ -n "$PGID" ] && kill -KILL -- "-$PGID" 2>/dev/null
  echo "Stopped."
}
trap cleanup INT TERM EXIT

setsid env EXO_DEFAULT_MODELS_DIR="$HOME/exo-models" CUDA_VISIBLE_DEVICES=1 EXO_HOME=.exo-gpu1 ENABLE_DISAGGREGATION=true \
  uv run exo --namespace mycluster --no-api \
  --zenoh-port 52424 --discovery-port 52423 \
  --connect-peer 169.254.150.225:52414,127.0.0.1:52414 &
PGID=$!

echo "4060 Ti node running (PGID=$PGID). Press Ctrl+C to stop."
wait
