# Disaggregated CUDA‑prefill / Mac‑decode on a 10GbE cluster

Goal: replicate the [exo DGX‑Spark disaggregated inference](https://blog.exolabs.net/nvidia-dgx-spark/)
on this fork — run **prompt prefill on the NVIDIA/CUDA box** (compute‑bound) and **token decode on
the Mac** (memory‑bandwidth‑bound), streaming the KV cache over 10GbE.

## Hardware in this setup
- **WSL2 node** — RTX 5070 Ti (16GB, PCIe Gen5×16) + RTX 4060 Ti (16GB, PCIe **Gen4×4**), 54GB system RAM, CUDA 13, MLX‑CUDA (`mlx==0.32.0`, `mlx-cuda-13`).
- **Mac Studio** — M4 Max, 128GB unified memory. Reachable at `169.254.150.225` over 10GbE.
- The Mac's 128GB is **only** usable by the Mac — it cannot back the CUDA node's weights (different machine across the network).

## Why there is no CUDA speedup today (three gates, all on)
1. **Disaggregation is disabled by default** — `ENABLE_DISAGGREGATION` defaults to `false`
   (`src/exo/shared/constants.py:104`). Normal placement is **memory‑proportional**
   (`src/exo/master/placement_utils.py:47-75`), so the 128GB Mac is handed almost every layer and
   the CUDA box becomes a trivial pipeline stage — CUDA compute is never used for prefill.
2. **Roles are manual** — there is no auto device‑profiling that assigns prefill→GPU / decode→Mac.
   You must create an **instance link** (`POST /v1/instance-links`,
   `src/exo/master/main.py:_prefill_endpoint_for` at :82).
3. **Remote prefill only fires for long prompts** — `REMOTE_PREFILL_MIN_TOKENS = 1000`
   (`src/exo/worker/engines/mlx/generator/batch_generate.py:62`), and disaggregated mode runs **two
   independent full‑model instances** (one per machine) — *not* one model sharded across both.

## Hard constraints discovered
- **16GB per GPU.** Qwen3‑27B at **8‑bit ≈ 27GB does not fit one 16GB GPU.** Options:
  4‑bit (~14GB, fits one GPU), 8‑bit with managed‑memory paging into the 54GB host RAM (works,
  speed depends on PCIe paging), or **8‑bit split across both GPUs (16+16=32GB, no paging)** → Tier 2.
- **Keep KV‑cache quantization OFF.** `QuantizedKVCache | CacheList | DeepseekV4Cache` hit
  `raise NotImplementedError` in `src/exo/worker/engines/mlx/disaggregated/adapter.py:104`.
  8‑bit *weights* are fine — just never set `kv_bits`.
- **JACCL (Apple RDMA) cannot include the CUDA node** — `MlxJaccl` requires every node to support
  `MlxMetal` (`src/exo/master/placement.py`, `INSTANCE_META_BACKENDS`). Cross‑machine and the
  two local GPUs therefore use **`MlxRing`** over TCP, not JACCL.
- **PCIe is NOT a bottleneck for pipeline parallel.** A pipeline cut transfers only the boundary
  hidden state once per prefill: for S=8192, hidden=5120, bf16 ≈ **84MB → ~13ms** over the 4060 Ti's
  Gen4×4 link, vs seconds of compute (<0.3%). (Tensor parallel, by contrast, all‑reduces every
  layer ≈ 21GB/prefill — avoid it here.)

---

## Architecture

```
            PREFILL INSTANCE (CUDA, MlxRing)                 DECODE INSTANCE (Mac)
  ┌──────────────────────────────────────────────┐        ┌────────────────────────┐
  │  Tier 1: one 16GB GPU, full model             │        │  M4 Max 128GB          │
  │  Tier 2: 5070Ti + 4060Ti pipeline (32GB)      │  KV    │  full model, decode    │
  │  runs prefill, produces KV cache              │ ─────► │  injects KV, generates │
  └──────────────────────────────────────────────┘ 10GbE  └────────────────────────┘
          linked by  POST /v1/instance-links { prefill_instances:[…], decode_instances:[…] }
```

The KV transfer is a custom msgpack/TCP protocol streamed **per layer**
(`src/exo/worker/disaggregated/protocol.py`, `…/engines/mlx/disaggregated/adapter.py`):
`Header(num_layers,dtype,start_pos)` → `KVChunk(layer_idx,…,keys,values)` per layer → `Done`.

---

## Tier 1 — works with existing code (validate first)

Prefill on a **single** CUDA GPU (`world_size=1`) → Mac decode. This exercises the existing,
working disaggregated path. Start here to prove a real speedup on this hardware before investing in
Tier 2.

### Model choice for Tier 1
- **4‑bit (~14GB):** fits the 5070 Ti in‑VRAM, fastest, simplest. Recommended for the first run.
  With the auto‑VRAM reporting in this branch the node advertises ~16GB, so a 14GB model places
  cleanly — **no override needed.**
- **8‑bit (~27GB):** relies on MLX‑CUDA managed‑memory paging from the 54GB host RAM. Because the
  node now advertises ~16GB VRAM, placement would *reject* a 27GB model on one GPU — set
  `OVERRIDE_MEMORY_MB=28000` on that node to let the master accept it (it will then page through
  the 16GB GPU). Benchmark to see if PCIe paging hurts prefill; if it does, prefer Tier 2.

### Launch (run on BOTH machines; identical namespace is mandatory)
`--namespace` must match on every node — `discovery.rs:172` drops peers with a different namespace,
even with `--connect-peer`. `EXO_ZENOH_NAMESPACE: None` in the log is cosmetic; the CLI arg is honored
(`src/exo/main.py:83`).

**Mac (decode, master):**
```bash
ENABLE_DISAGGREGATION=true uv run exo --namespace mycluster -m
```

**WSL (prefill, 5070 Ti) — 4‑bit:**
```bash
CUDA_VISIBLE_DEVICES=0 \
ENABLE_DISAGGREGATION=true \
uv run exo --namespace mycluster --connect-peer 169.254.150.225
```
- `CUDA_VISIBLE_DEVICES=0` pins the 5070 Ti.
- No `OVERRIDE_MEMORY_MB` needed: this branch auto‑reports the GPU's ~16GB VRAM (respecting
  `CUDA_VISIBLE_DEVICES`) instead of 54GB system RAM, so the master sizes the GPU correctly.
  For the 8‑bit‑paging variant only, add `OVERRIDE_MEMORY_MB=28000` (see "Model choice" above).

### Bring up the two instances and link them
1. Inspect node/instance IDs: `curl -s localhost:52415/state | jq 'keys'` (the route is `/state`,
   not `/v1/state`; drill in with `/state/topology`, `/state/instances`). Check disaggregation is on
   with `curl -s localhost:52415/v1/feature-flags`.
2. Place the **decode** instance so it lands on the Mac, and the **prefill** instance so it lands on
   the CUDA node. Placement picks the smallest sufficient cycle, so pin explicitly when needed:
   - Preview a placement: `GET /placement?model_id=<id>&sharding=pipeline&min_nodes=1`.
   - Create an explicitly‑assigned instance: `POST /instance` with the returned `Instance` (lets you
     pin `shard_assignments.node_to_runner` to the node you want). `POST /place_instance` is the
     automatic alternative.
3. Confirm `ENABLE_DISAGGREGATION` is on: `curl localhost:52415/v1/feature-flags` → `{"disaggregation":true}`.
4. Link prefill → decode:
```bash
curl -X POST localhost:52415/v1/instance-links \
  -H 'content-type: application/json' \
  -d '{"prefill_instances":["<CUDA_INSTANCE_ID>"],"decode_instances":["<MAC_INSTANCE_ID>"]}'
```
5. Send a **>1000‑token** prompt to `/v1/chat/completions`. Below 1000 tokens decode prefills
   locally on the Mac by design.

### Benchmark
Compare prefill time / prompt‑TPS for an 8K‑token prompt, three ways:
- Mac alone (no link), CUDA‑prefill linked, and (later) Tier 2.
Watch the logs: `remote_prefill.py` logs `Remote prefill: N tokens at X tok/s, transfer=…ms`.
Use `bench/` or `POST /bench/chat/completions`.

---

## Tier 2 — 8‑bit across both GPUs (the goal; needs live‑cluster validation)

Run the **prefill instance itself** as a 2‑node pipeline‑parallel group across the two local GPUs so
the full 27GB 8‑bit model fits in 16+16GB with **no paging**, then link it to the Mac decode instance.

### Operational shape
Run **two** exo node‑processes on the WSL box, each pinned to one GPU, with separate data dirs/ports.
**IMPORTANT: start GPU1 BEFORE GPU0.** exo does not retry failed startup dials, so GPU0's
`--connect-peer 127.0.0.1:52424` will fail silently if GPU1 hasn't bound its port yet — this leaves
a missing topology edge and prevents the 2‑node cycle that placement requires. Use the bundled
launch script:

```bash
# Launches GPU1 first, waits 3s, then GPU0. Ctrl+C stops both.
./run-both-gpus.sh
```

Or manually:
```bash
# 5070 Ti node — keeps the API on :52415
CUDA_VISIBLE_DEVICES=0 EXO_HOME=.exo-gpu0 ENABLE_DISAGGREGATION=true \
  uv run exo --namespace mycluster \
  --zenoh-port 52414 --discovery-port 52413 --api-port 52415 \
  --connect-peer 169.254.150.225:52414,127.0.0.1:52424
# 4060 Ti node — --no-api so it binds no API port
CUDA_VISIBLE_DEVICES=1 EXO_HOME=.exo-gpu1 ENABLE_DISAGGREGATION=true \
  uv run exo --namespace mycluster --no-api \
  --zenoh-port 52424 --discovery-port 52423 \
  --connect-peer 169.254.150.225:52414,127.0.0.1:52414
```
**Running two exo processes on one host — avoid port collisions:**
- `--zenoh-port` is a TCP listener and **must differ** per process (52414 vs 52424). `--api-port`
  must differ too, so the second worker uses `--no-api` (drive everything through the Mac's or the
  first node's API). `--discovery-port` is bound `SO_REUSEPORT` so it *may* be shared, but distinct
  values are clearer.
- WSL2 blocks multicast discovery, so the two local nodes will **not** auto-discover each other —
  each must `--connect-peer` the other (`127.0.0.1:<other zenoh-port>`) as well as the Mac.
- **Always give an explicit port in `--connect-peer`** (`169.254.150.225:52414`): a bare host
  defaults to the dialer's own `--zenoh-port`, which is wrong once the ports differ.
- Separate `EXO_HOME` per process (distinct event log / downloads / lock files).

With the auto‑VRAM reporting in this branch each CUDA node advertises ~16GB, so a *single* node is
insufficient for the 27GB 8‑bit model and the master **automatically** places the prefill instance
as a **2‑node pipeline** across the two GPUs (layers split ~proportionally → both fit under 16GB).
No `OVERRIDE_MEMORY_MB` is required. MLX distributed uses `MlxRing` (TCP) for the two ranks; the
boundary transfer is ~84MB/prefill (negligible).

### Code change (implemented in this branch)
Previously the disaggregated server served on **rank 0 only**, and in pipeline parallel rank 0's
`cache` holds **only its own layers' KV** — so a multi‑rank prefill instance could not transfer the
*full* KV. Implemented:

1. **Serve on every rank.** Dropped the `device_rank != 0` guard in `runner.py:_start_prefill_server`,
   so each rank exposes its layers. `state.prefill_server_ports` already keys by `runner_id`.
2. **Emit GLOBAL layer indices.** `send_mlx_kv_cache` / `write_cache_to_wire`
   (`…/disaggregated/adapter.py`) take a `layer_offset` so each rank writes
   `layer_idx = layer_offset + local_idx`, and `total_layers` sets `Header.num_layers` to the global
   count. The runner passes `shard_metadata.start_layer` / `.n_layers` via `Engine.serve_prefill`.
3. **Return all prefill endpoints.** `_prefill_endpoints_for` (`master/main.py`) returns the **list**
   of every prefill‑rank `ip:port` for the chosen source, skipping sources where any rank is
   unreachable (partial KV is useless).
4. **Carry multiple endpoints on the task.** `prefill_endpoint: str | None` → `prefill_endpoints:
   list[str]` on `TextGenerationTaskParams`; both decode paths (`generate.py`, `batch_generate.py`)
   updated.
5. **Merge on decode.** `remote_prefill` fetches every endpoint **concurrently** (pipeline prefill is
   a collective — a sequential blocking fetch would deadlock), then merges the per‑rank
   `PrefillResult`s by global `layer_idx` and injects once. Unit tests in
   `…/disaggregated/tests/test_multirank_prefill.py` cover the global‑index wire format and the
   multi‑server merge.

**Still requires live‑cluster validation:** the actual MLX distributed pipeline collective across the
two GPUs (the unit tests use stub servers, not a real `mx.distributed` group).

### Tier‑2 caveats
- Pipeline bubble ≈ `(W−1)/(N+W−1)`; exo halves `prefill_step_size` for the group
  (`generate.py:195`), so an 8K prompt is ~4 chunks → ~20% bubble, shrinking on longer prompts.
- The 4060 Ti (Gen4×4, ~half the compute) is the slower stage and bounds steady‑state throughput;
  filling the 5070 Ti first (more layers on the fast GPU) keeps the stages roughly balanced.
- All KV crosses two PCIe hops per prefill but only ~84MB total — not a bottleneck.

---

## Code changes in this branch
- `src/exo/main.py` — log the **actual** discovery namespace in use (not just the env var).
- `src/exo/routing/event_router.py` — cold‑join catch‑up: after a partial drain, immediately request
  the next event‑log gap instead of waiting for the next out‑of‑order live event (fixes the slow
  "Requesting Event Log from 0…1000…2000…" crawl). Master caps replay at 1000/req
  (`master/main.py:454`).
- `src/exo/utils/info_gatherer/info_gatherer.py` + `src/exo/shared/types/profiling.py` — on a
  non‑Darwin CUDA node, report **GPU VRAM** (respecting `CUDA_VISIBLE_DEVICES`) as the node memory
  when `OVERRIDE_MEMORY_MB` is unset, so placement sizes GPUs correctly by default.
- Tier 2 multi‑rank KV streaming (serve‑on‑all‑ranks, global layer indices, all‑endpoint
  resolution, list‑valued `prefill_endpoints`, concurrent fetch+merge on decode) is **implemented**
  with unit tests; the real `mx.distributed` pipeline collective across the two GPUs still needs
  live‑cluster validation. Touches: `worker/runner/runner.py`, `engines/base.py`,
  `engines/image/builder.py`, `engines/mlx/disaggregated/adapter.py`,
  `runner/llm_inference/batch_generator.py`, `engines/mlx/generator/{remote_prefill,generate,batch_generate}.py`,
  `shared/types/text_generation.py`, `master/main.py`.

## Verification
- Local: `uv run basedpyright && uv run ruff check && uv run pytest` (Tier‑1 fixes + new unit tests).
- Cluster (manual): follow the Tier‑1 runbook, send an 8K‑token prompt, confirm a
  `Remote prefill: … tok/s` log line on the decode node and a measured prefill speedup vs Mac‑alone.
