"""Tests for CUDA VRAM memory reporting in info_gatherer and MemoryUsage."""

import pytest

from exo.shared.types.profiling import MemoryUsage
from exo.utils.info_gatherer.info_gatherer import (
    _cuda_vram_bytes,  # pyright: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# MemoryUsage.from_cuda_vram — deterministic field-mapping tests
# ---------------------------------------------------------------------------


def test_from_cuda_vram_field_mapping():
    total = 16 * 1024**3  # 16 GiB
    free = 10 * 1024**3  # 10 GiB
    usage = MemoryUsage.from_cuda_vram(vram_total=total, vram_free=free)

    assert usage.ram_total.in_bytes == total
    assert usage.ram_available.in_bytes == free
    # Swap fields must be zero — GPU VRAM has no swap analogue.
    assert usage.swap_total.in_bytes == 0
    assert usage.swap_available.in_bytes == 0


def test_from_cuda_vram_full_free():
    """When VRAM is completely free, available equals total."""
    total = 8 * 1024**3
    usage = MemoryUsage.from_cuda_vram(vram_total=total, vram_free=total)
    assert usage.ram_total.in_bytes == usage.ram_available.in_bytes


def test_from_cuda_vram_no_free():
    """When VRAM is fully used, available is 0."""
    total = 8 * 1024**3
    usage = MemoryUsage.from_cuda_vram(vram_total=total, vram_free=0)
    assert usage.ram_available.in_bytes == 0
    assert usage.ram_total.in_bytes == total


# ---------------------------------------------------------------------------
# _cuda_vram_bytes — live GPU test (skipped when CUDA/pynvml unavailable)
# ---------------------------------------------------------------------------


def _pynvml_available() -> bool:
    try:
        import pynvml as nvml

        nvml.nvmlInit()
        try:
            count: int = nvml.nvmlDeviceGetCount()
            return count > 0
        finally:
            nvml.nvmlShutdown()
    except Exception:
        return False


@pytest.mark.skipif(not _pynvml_available(), reason="pynvml/CUDA not available")
def test_cuda_vram_bytes_plausible():
    """_cuda_vram_bytes returns plausible values when a GPU is present."""
    result = _cuda_vram_bytes()
    assert result is not None, (
        "_cuda_vram_bytes returned None despite CUDA being available"
    )
    total, free = result
    assert total > 0, f"VRAM total should be > 0, got {total}"
    assert free >= 0, f"VRAM free should be >= 0, got {free}"
    assert free <= total, f"VRAM free ({free}) must not exceed total ({total})"


@pytest.mark.skipif(not _pynvml_available(), reason="pynvml/CUDA not available")
def test_cuda_vram_bytes_respects_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch):
    """When CUDA_VISIBLE_DEVICES is set to a valid single integer, returns non-None."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = _cuda_vram_bytes()
    assert result is not None
    total, free = result
    assert total > 0
    assert 0 <= free <= total


@pytest.mark.skipif(not _pynvml_available(), reason="pynvml/CUDA not available")
def test_cuda_vram_bytes_invalid_cuda_visible_devices_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
    """Non-integer CUDA_VISIBLE_DEVICES (e.g. 'NoDevFiles') must return None."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "NoDevFiles")
    result = _cuda_vram_bytes()
    assert result is None


@pytest.mark.skipif(not _pynvml_available(), reason="pynvml/CUDA not available")
def test_cuda_vram_bytes_out_of_range_index_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
    """An out-of-range CUDA_VISIBLE_DEVICES index must return None."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9999")
    result = _cuda_vram_bytes()
    assert result is None


# ---------------------------------------------------------------------------
# Precedence: OVERRIDE_MEMORY_MB wins over CUDA VRAM
# ---------------------------------------------------------------------------


def test_from_psutil_respects_override():
    """from_psutil with an override sets ram_available to the override value."""
    override_bytes = 4 * 1024**3  # 4 GiB
    usage = MemoryUsage.from_psutil(override_memory=override_bytes)
    assert usage.ram_available.in_bytes == override_bytes
