#!/bin/bash
# Launch two exo GPU nodes on one WSL box for pipeline-parallel prefill.
#
# NOTE: the topology fix (bidirectional cycle detection) means start order no
# longer matters — a single one-directional edge is enough to form a cycle, and
# the --no-api node (GPU1) now joins cycles even though it cannot be HTTP-probed.
# GPU1 is still started first (+ a short sleep) as belt-and-suspenders so both
# --connect-peer dials land, but it is no longer required for correctness.
#
# Reminder: the topology/placement fix runs on the MASTER node, so make sure the
# Mac (current master) is also on this branch.

set -e

echo "Starting GPU1 (4060 Ti) on zenoh :52424 ..."
CUDA_VISIBLE_DEVICES=1 EXO_HOME=.exo-gpu1 ENABLE_DISAGGREGATION=true \
  uv run exo --namespace mycluster --no-api \
  --zenoh-port 52424 --discovery-port 52423 \
  --connect-peer 169.254.150.225:52414,127.0.0.1:52414 &
GPU1_PID=$!

# Give GPU1 time to bind its zenoh listener before GPU0 dials it.
sleep 3

echo "Starting GPU0 (5070 Ti) on zenoh :52414, API :52415 ..."
CUDA_VISIBLE_DEVICES=0 EXO_HOME=.exo-gpu0 ENABLE_DISAGGREGATION=true \
  uv run exo --namespace mycluster \
  --zenoh-port 52414 --discovery-port 52413 --api-port 52415 \
  --connect-peer 169.254.150.225:52414,127.0.0.1:52424 &
GPU0_PID=$!

echo "GPU1 PID=$GPU1_PID  GPU0 PID=$GPU0_PID"
echo "Press Ctrl+C to stop both."
trap "kill $GPU1_PID $GPU0_PID 2>/dev/null" EXIT INT TERM
wait
