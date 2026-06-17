import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import CacheSnapshot, snapshot_ssm_states
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillResult,
    ingest_into_mlx_cache,
    remote_prefill_fetch,
)
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.bootstrap import logger


# FORK(exo-10gbe-wsl): multi-rank concurrent fetch + per-rank result merge; replaces single-endpoint sequential fetch
def remote_prefill(
    prompt_tokens: mx.array,
    cache: KVCacheType,
    on_prefill_progress: Callable[[int, int], None] | None,
    *,
    endpoints: list[str],
    request_id: str,
    model_id: str,
    start_pos: int = 0,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Fetch a prompt's KV cache from one or more remote prefill ranks.

    ``endpoints`` holds one ``ip:port`` per rank of the linked prefill instance.
    A pipeline-parallel prefill instance has one server per rank, each holding a
    disjoint range of global layers. The ranks run prefill as a COLLECTIVE (the
    pipeline stages send/recv to each other), so every rank must be executing its
    forward pass at the same time. We therefore dispatch the request to all rank
    servers concurrently and only then read the responses — a sequential fetch
    would block on the first rank forever, deadlocking the collective. The
    per-rank results are merged by global layer index before injection.
    """
    if not endpoints:
        raise ValueError("remote_prefill requires at least one endpoint")

    t0 = time.perf_counter()
    total_prompt_tokens = int(prompt_tokens.shape[0])

    request = PrefillRequest(
        model_id=model_id,
        token_ids=cast(list[int], prompt_tokens.tolist()),
        start_pos=start_pos,
        request_id=request_id,
    )

    if len(endpoints) == 1:
        results = [remote_prefill_fetch(endpoints[0], request)]
    else:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            futures = [
                pool.submit(remote_prefill_fetch, endpoint, request)
                for endpoint in endpoints
            ]
            results = [future.result() for future in futures]

    t_received = time.perf_counter()

    # Merge per-rank results. Each rank contributes a disjoint set of global
    # layer indices; overlapping layers would mean a misconfigured shard split.
    merged = PrefillResult(header=results[0].header)
    for result in results:
        for layer_idx, chunks in result.kv_chunks.items():
            if layer_idx in merged.kv_chunks:
                raise RuntimeError(
                    f"Layer {layer_idx} returned by more than one prefill rank"
                )
            merged.kv_chunks[layer_idx] = chunks
        merged.arrays.update(result.arrays)
        merged.total_tokens = max(merged.total_tokens, result.total_tokens)

    caches = cast(list[KVCache | RotatingKVCache | ArraysCache], list(cache))
    final_offset = ingest_into_mlx_cache(merged, caches, start_pos=start_pos)
    t_done = time.perf_counter()

    if on_prefill_progress is not None:
        on_prefill_progress(total_prompt_tokens, total_prompt_tokens)

    num_tokens = final_offset - start_pos
    tps = num_tokens / max(t_done - t0, 0.001)

    logger.info(
        f"Remote prefill: {num_tokens} tokens from {len(endpoints)} rank(s) "
        f"(start_pos={start_pos}, final_offset={final_offset}) at {tps:.0f} tok/s, "
        f"transfer={(t_received - t0) * 1000:.0f}ms, "
        f"inject={(t_done - t_received) * 1000:.0f}ms"
    )
    return tps, num_tokens, [snapshot_ssm_states(cache)]
