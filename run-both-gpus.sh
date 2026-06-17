#!/bin/bash
# Launch two exo GPU nodes on one WSL box (pipeline-parallel prefill, 8-bit).
#
# Start order no longer matters (the topology fix treats one-directional edges as
# bidirectional, so the --no-api node joins cycles). GPU1 is still started first
# + a short sleep as belt-and-suspenders.
#
# Each node runs in its OWN process group (setsid). Ctrl+C cleanly stops BOTH —
# including the python child that `uv run` spawns (which previously survived and
# kept holding the zenoh port).
#
# Reminder: placement runs on the MASTER, so make sure the Mac is also on main.

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping exo nodes..."
  for pgid in "$GPU1_PGID" "$GPU0_PGID"; do
    [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null
  done
  # give zenoh listeners a moment to release ports, then hard-kill survivors
  sleep 2
  for pgid in "$GPU1_PGID" "$GPU0_PGID"; do
    [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null
  done
  echo "Stopped."
}
trap cleanup INT TERM EXIT

echo "Starting GPU1 (4060 Ti) on zenoh :52424 ..."
setsid env CUDA_VISIBLE_DEVICES=1 EXO_HOME=.exo-gpu1 ENABLE_DISAGGREGATION=true \
  uv run exo --namespace mycluster --no-api \
  --zenoh-port 52424 --discovery-port 52423 \
  --connect-peer 169.254.150.225:52414,127.0.0.1:52414 &
GPU1_PGID=$!

# Give GPU1 time to bind its zenoh listener before GPU0 dials it.
sleep 3

echo "Starting GPU0 (5070 Ti) on zenoh :52414, API :52415 ..."
setsid env CUDA_VISIBLE_DEVICES=0 EXO_HOME=.exo-gpu0 ENABLE_DISAGGREGATION=true \
  uv run exo --namespace mycluster \
  --zenoh-port 52414 --discovery-port 52413 --api-port 52415 \
  --connect-peer 169.254.150.225:52414,127.0.0.1:52424 &
GPU0_PGID=$!

echo "Both starting (GPU1 PGID=$GPU1_PGID  GPU0 PGID=$GPU0_PGID)."
echo "Press Ctrl+C to stop both."
wait
