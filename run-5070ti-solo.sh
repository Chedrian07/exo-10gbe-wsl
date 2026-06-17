#!/bin/bash
# Standalone single-GPU node on the RTX 5070 Ti — for an isolated 4-bit decode
# benchmark with NO network in the path.
#
# Its own namespace ("solo5070") and no --connect-peer, so it forms a 1-node
# cluster and does NOT join the Mac/4060Ti cluster. A 27B-4bit model (~14 GB)
# fits this 16 GB GPU, so placement puts it entirely here.
#
# Runs in its own process group; Ctrl+C cleanly stops it AND the python child
# that `uv run` spawns (so ports 52414/52415 are always released).
#
# Usage:
#   1. Stop any other exo process first (this reuses ports 52414/52415).
#   2. ./run-5070ti-solo.sh
#   3. http://localhost:52415  ->  pick mlx-community/Qwen3.6-27B-4bit,
#      Minimum Devices = 1, Launch, then chat and read the tok/s (decode speed).

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping exo (solo)..."
  [ -n "$PGID" ] && kill -TERM -- "-$PGID" 2>/dev/null
  sleep 2
  [ -n "$PGID" ] && kill -KILL -- "-$PGID" 2>/dev/null
  echo "Stopped."
}
trap cleanup INT TERM EXIT

setsid env CUDA_VISIBLE_DEVICES=0 EXO_HOME=.exo-solo ENABLE_DISAGGREGATION=false \
  uv run exo --namespace solo5070 \
  --zenoh-port 52414 --discovery-port 52413 --api-port 52415 &
PGID=$!

echo "Solo 5070 Ti node running (PGID=$PGID). Press Ctrl+C to stop."
wait
