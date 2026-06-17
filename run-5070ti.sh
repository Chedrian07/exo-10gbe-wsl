#!/bin/bash
# Single exo node on the RTX 5070 Ti, joining the "mycluster" cluster (8-bit).
# Runs in its own process group; Ctrl+C cleanly stops it AND the python child
# that `uv run` spawns (so the zenoh port is always released).

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping exo (5070 Ti)..."
  [ -n "$PGID" ] && kill -TERM -- "-$PGID" 2>/dev/null
  sleep 2
  [ -n "$PGID" ] && kill -KILL -- "-$PGID" 2>/dev/null
  echo "Stopped."
}
trap cleanup INT TERM EXIT

setsid env CUDA_VISIBLE_DEVICES=0 EXO_HOME=.exo-gpu0 ENABLE_DISAGGREGATION=true \
  uv run exo --namespace mycluster \
  --zenoh-port 52414 --discovery-port 52413 --api-port 52415 \
  --connect-peer 169.254.150.225:52414,127.0.0.1:52424 &
PGID=$!

echo "5070 Ti node running (PGID=$PGID). Press Ctrl+C to stop."
wait
