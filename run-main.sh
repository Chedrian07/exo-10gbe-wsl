#!/bin/bash
# DEFAULT launch for this Linux (WSL2) machine: the RTX 5070 Ti as the main node.
#
# ENABLE_DISAGGREGATION is on, so when the Mac joins the same cluster the master
# auto-assigns this GPU to PREFILL and the Mac to DECODE — no manual instance-link
# (see compute_auto_disaggregation_links). Standalone (Mac down) it just runs
# everything on the 5070 Ti. EXO_NO_BATCH=1 picks the faster single-request
# generator. Ctrl+C cleanly stops it (process group via setsid).
#
# Workflow:
#   1. ./run-main.sh                 (here)   +   run exo with the same --namespace on the Mac
#   2. Dashboard http://localhost:52415  ->  launch the SAME model on BOTH nodes (Min Devices = 1)
#   3. ~10s later the master logs "Auto-disaggregation: prefill=[...] -> decode=[...]"
#   4. >1000-token prompts then run prefill on the 5070 Ti (decode-node log: "Remote prefill: ...")
#
# Override the Mac address / namespace via env if needed:
#   MAC_IP=100.88.x.x NAMESPACE=mycluster ./run-main.sh

MAC_IP="${MAC_IP:-169.254.150.225}"   # Mac's 10GbE IP
NAMESPACE="${NAMESPACE:-mycluster}"

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping exo (5070 Ti main)..."
  [ -n "$PGID" ] && kill -TERM -- "-$PGID" 2>/dev/null
  sleep 2
  [ -n "$PGID" ] && kill -KILL -- "-$PGID" 2>/dev/null
  echo "Stopped."
}
trap cleanup INT TERM EXIT

setsid env EXO_DEFAULT_MODELS_DIR="$HOME/exo-models" CUDA_VISIBLE_DEVICES=0 EXO_HOME=.exo-main \
  ENABLE_DISAGGREGATION=true EXO_NO_BATCH=1 \
  uv run exo --namespace "$NAMESPACE" \
  --zenoh-port 52414 --discovery-port 52413 --api-port 52415 \
  --connect-peer "${MAC_IP}:52414" &
PGID=$!

echo "5070 Ti main node running (PGID=$PGID, namespace=$NAMESPACE, Mac=$MAC_IP)."
echo "Dashboard: http://localhost:52415   |   Press Ctrl+C to stop."
wait
