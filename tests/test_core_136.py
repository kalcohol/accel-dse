"""0.136: tile I/O is a GEMM-kernel term only (flash attention keeps S / P on chip; its K / V stream is already DRAM
traffic) — was double-counted on attention under cta; opt-in Chip.kernel_vec (vector ops pay the kernel floor);
extval.kfloor_variant."""
from dataclasses import replace

from accel_dse.core import extval as X
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import Chip


def _dec(var):
    r = next(r for r in X.rows() if r["metric"] == "max_tok_s_total" and r["model"] == "llama-3.1-8b")
    return evaluate(X.scenario(r, var, phase="decode", batch=256, ctx=1100, out_len=2048), X._spec(r["model"], True))


def test_tile_io_not_on_attention():
    C = X.candidate_variant()
    a, b = _dec(C), _dec(X.tileio_variant(C))
    # decode attention array time is identical with and without tile I/O (was 4.5× with it)
    assert abs(b.stages[0].time.t_arr_attn - a.stages[0].time.t_arr_attn) < 1e-12
    assert b.tpot < a.tpot * 1.05


def test_kernel_vec_counts_vector_kernels():
    C = X.candidate_variant()
    s = _dec(X.kfloor_variant(C)).stages[0].time.t_sync
    # 32 layers × (6 GEMMs + 1 attention + 5 vector kernels) × 2 µs + head / embed kernels
    assert 32 * 12 * 2e-6 <= s < 32 * 12 * 2e-6 + 20e-6
    assert Chip().kernel_vec is False


from tests.pin_legacy import pin_module_unfused  # noqa: E402  (0.198: per-projection checks on the unfused IR)
pin_module_unfused(globals(), only={'test_kernel_vec_counts_vector_kernels'})
