#!/bin/bash
# Standalone single-GPU node on the RTX 5070 Ti — for an isolated 4-bit decode
# benchmark with NO network in the path.
#
# It uses its OWN namespace ("solo5070") and no --connect-peer, so it forms a
# 1-node cluster and does NOT join the Mac/4060Ti cluster. A 27B-4bit model
# (~14 GB) fits this 16 GB GPU, so placement puts it entirely here.
#
# Usage:
#   1. Stop any other exo process first (this reuses ports 52414/52415).
#   2. ./run-5070ti-solo.sh
#   3. Open http://localhost:52415  ->  pick mlx-community/Qwen3.6-27B-4bit,
#      Minimum Devices = 1, Launch, then chat and read the tok/s (decode speed).
#
# Compare that number against 8-bit on the Mac to decide if 4-bit is worth it.

CUDA_VISIBLE_DEVICES=0 EXO_HOME=.exo-solo ENABLE_DISAGGREGATION=false \
  uv run exo --namespace solo5070 \
  --zenoh-port 52414 --discovery-port 52413 --api-port 52415
