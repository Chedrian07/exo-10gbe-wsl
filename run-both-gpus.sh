#!/bin/bash
# Launch two exo GPU nodes on one WSL box for pipeline-parallel prefill.
# GPU1 (4060 Ti) MUST start first so GPU0's --connect-peer 127.0.0.1:52424
# succeeds — exo does not retry failed startup dials, so a race where GPU0
# starts before GPU1 leaves a missing edge and breaks 2-node cycle formation.

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
