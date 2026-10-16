"""0.198: fused QKV projection (vLLM / SGLang QKVParallelLinear, TRT-LLM Attention.qkv)."""
import sys
from dataclasses import replace
sys.path.insert(0, ".")
from accel_dse.core import evaluate as E
from accel_dse.core.catalog import get_model
from accel_dse.core.hardware import System
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.refhw import REF_HW


def _gemms(mid, tp, ph):
    m = get_model(mid)
    return [o for o in build_rank_ops(m, 0, ph, Shard(tp=tp)) if o.kind == "gemm" and o.name.startswith("attn.")]


def test_fused_same_work():
    ph = Phase("decode", 1, 1, 200)
    for mid, tp in (("llama-3.3-70b", 8), ("llama-3.1-8b", 1), ("qwen3-8b", 4)):
        f = _gemms(mid, tp, ph)
        with E.model_flags(FUSE_QKV=False):
            u = _gemms(mid, tp, ph)
        assert "attn.qkv" in {o.name for o in f} and not {"attn.q", "attn.k", "attn.v"} & {o.name for o in f}
        assert {"attn.q", "attn.k", "attn.v"} <= {o.name for o in u}
        assert sum(o.w_params for o in f) == sum(o.w_params for o in u)
        assert sum(o.m * o.k * o.n * o.count for o in f) == sum(o.m * o.k * o.n * o.count for o in u)


def test_mi300x_b1_one_kernel_cheaper():
    h = REF_HW["MI300X"]
    s = System(h.chip, h.mem_id)
    s = replace(s, chip=replace(h.chip, tile_io_Bpc=s.dram_GBps / h.chip.freq_ghz))
    m = get_model("llama-3.3-70b")
    ph = Phase("decode", 1, 1, 200)
    t = lambda ops: sum(E._op_seconds(o, s, "reconf", m)[0] for o in ops if o.name in ("attn.qkv", "attn.q", "attn.k", "attn.v"))
    f = t(_gemms("llama-3.3-70b", 8, ph))
    with E.model_flags(FUSE_QKV=False):
        u = t(_gemms("llama-3.3-70b", 8, ph))
    wbytes = 8192 * 1280 * 2 / (s.dram_GBps * 1e9)
    assert wbytes * (1 - 1e-9) <= f < 0.8 * u      # 10.3 µs → ≈ its 5.6 µs weight floor


def test_mla_and_grouped_untouched():
    names = {o.name for o in _gemms("deepseek-v3", 8, Phase("decode", 1, 1, 200))}
    assert "attn.qkv" not in names
