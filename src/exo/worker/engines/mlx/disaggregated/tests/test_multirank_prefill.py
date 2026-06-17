"""Tier-2 disaggregated prefill: a pipeline-parallel prefill instance has one
server per rank, each holding a disjoint range of global layers. These tests
cover the two pieces that make that work:
  1. the wire emits GLOBAL layer indices (layer_offset / total_layers), and
  2. the decode side concurrently fetches every rank and merges by global layer.
"""

import io
from typing import BinaryIO, cast

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from exo.worker.disaggregated.protocol import (
    Done,
    KVChunk,
    read_header,
    read_message,
)
from exo.worker.disaggregated.server import PrefillRequest, PrefillServer
from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.types import KVCacheType


def _equal(a: mx.array, b: mx.array) -> bool:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.dtype == mx.bfloat16:
        return bool(
            np.array_equal(np.asarray(a.view(mx.uint16)), np.asarray(b.view(mx.uint16)))
        )
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _make_cache(seed: int, seq_len: int, n_heads: int, head_dim: int) -> KVCache:
    mx.random.seed(seed)
    cache = KVCache()
    with mx.stream(mx.Device(mx.cpu)):
        cache.keys = (
            mx.random.uniform(shape=(1, n_heads, seq_len, head_dim)) * 10
        ).astype(mx.bfloat16)
        cache.values = (
            mx.random.uniform(shape=(1, n_heads, seq_len, head_dim)) * 10
        ).astype(mx.bfloat16)
        mx.eval(cache.keys, cache.values)
    cache.offset = seq_len
    return cache


def test_write_cache_to_wire_emits_global_layer_index() -> None:
    # A rank holding the second half of a 4-layer model (global layers 2,3).
    local_cache: KVCacheType = [
        _make_cache(1, 5, 2, 4),
        _make_cache(2, 5, 2, 4),
    ]
    buf = io.BytesIO()
    write_cache_to_wire(
        cast(BinaryIO, buf),
        local_cache,
        request_id="r",
        model_id="m",
        start_pos=0,
        layer_offset=2,
        total_layers=4,
    )

    buf.seek(0)
    stream = cast(BinaryIO, buf)
    header = read_header(stream)
    assert header.num_layers == 4, "header must advertise the GLOBAL layer count"

    layer_ids: list[int] = []
    while True:
        msg = read_message(stream)
        if msg is None or isinstance(msg, Done):
            break
        if isinstance(msg, KVChunk):
            layer_ids.append(msg.layer_idx)
    assert layer_ids == [2, 3], "chunks must carry global layer indices"


@pytest.mark.slow
def test_remote_prefill_merges_multiple_ranks() -> None:
    n_layers = 4
    seq_len, n_heads, head_dim = 6, 2, 4
    # Distinct content per layer so a mis-ordered merge would be detected.
    gold = [_make_cache(i + 1, seq_len, n_heads, head_dim) for i in range(n_layers)]

    def make_resolve(shard: KVCacheType, offset: int):
        def resolve(job: PrefillRequest, wfile: BinaryIO) -> bool:
            write_cache_to_wire(
                wfile,
                shard,
                request_id=job.request_id,
                model_id="m",
                start_pos=0,
                layer_offset=offset,
                total_layers=n_layers,
            )
            return True

        return resolve

    rank0 = PrefillServer(
        resolve=make_resolve(gold[0:2], 0), host="127.0.0.1", port=52420
    )
    rank1 = PrefillServer(
        resolve=make_resolve(gold[2:4], 2), host="127.0.0.1", port=52421
    )
    try:
        dst: list[KVCache] = [KVCache() for _ in range(n_layers)]
        _tps, num_tokens, _snaps = remote_prefill(
            mx.arange(seq_len),
            cast(KVCacheType, dst),
            None,
            endpoints=["127.0.0.1:52420", "127.0.0.1:52421"],
            request_id="req-1",
            model_id="m",
            start_pos=0,
        )
        assert num_tokens == seq_len
        for i in range(n_layers):
            dk, dv = dst[i].keys, dst[i].values
            gk, gv = gold[i].keys, gold[i].values
            assert dk is not None and dv is not None
            assert gk is not None and gv is not None
            assert _equal(dk, gk), f"layer {i} keys mismatch after merge"
            assert _equal(dv, gv), f"layer {i} values mismatch after merge"
            assert dst[i].offset == seq_len
    finally:
        rank0.stop()
        rank1.stop()
