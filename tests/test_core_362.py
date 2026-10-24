"""0.362: DeepEmbedAttention must re-read reconstructed K/V that miss SRAM.

kkup / vvup write K [ctx, 256] and V [ctx, 768]. The flash path used to treat
those as equal width and, on decode, did not stream the attention at all, so a
DRAM-bound long context paid for the write and not the read. TP counted the
two matmuls once per rank instead of once per request.
"""
from __future__ import annotations

import math

from accel_dse.core import catalog as C
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Op, Phase, Shard, build_rank_ops
from accel_dse.core.memplan import act_stream
from accel_dse.core.scenario import Scenario

HYB = "rwkv8-pile-hybrid"
AB = 2.0
KQ, VD = 256, 768


def _budget(chip="100T") -> float:
    return CHIPS[chip].sram_bytes / 2.0


def test_equal_width_flash_stream_unchanged():
    """v_dim 0 keeps the historical 2·k Q/O and K/V byte count."""
    o = Op("qk", "attn", 0, m=128, k=128, n=65536, count=2)
    budget = _budget()
    a, wx = act_stream(o, AB, CHIPS["100T"].sram_bytes)
    qo = 2.0 * o.count * o.m * o.k * AB
    kv = 2.0 * o.count * o.n * o.k * AB
    assert qo + kv > budget
    br = max(1, int(budget // (o.k * (AB + 4))))
    assert wx == 0.0 and a == qo + kv * (math.ceil(o.m / br) - o.kv_pre)


def test_prefill_spills_when_only_the_real_v_misses():
    """Equal-width (V counted as another K) still fits; K+V at 256+768 does not."""
    m = C.get_model(HYB)
    budget = _budget()
    true_per = 2.0 * (KQ + VD) * AB
    equal_per = 4.0 * KQ * AB
    q = int(budget // true_per) + 1
    assert q * true_per > budget and q * equal_per <= budget
    qk = next(o for o in build_rank_ops(m, 0, Phase("prefill", 1, q, 0)) if o.name == "dea_qk")
    assert qk.stream and qk.v_dim == VD and qk.kv_pre == 0.0
    a, wx = act_stream(qk, AB, CHIPS["100T"].sram_bytes)
    br = max(1, int(budget // (KQ * (AB + 4))))
    assert q <= br and wx == 0.0
    assert a == q * true_per


def test_decode_rereads_spilled_kv_and_keeps_short_context_on_chip():
    m = C.get_model(HYB)
    budget = _budget()
    chip = "100T"

    def rows_of(ctx):
        return ctx + 1

    def one(ctx):
        return evaluate(Scenario().replace("model", HYB).replace("chip", chip).replace("serving.ctx", ctx))

    short = 4096
    rs = rows_of(short)
    assert rs * (KQ + VD) * AB + (KQ + VD) * AB <= budget
    assert one(short).stages[0].dram["act"] == 0.0

    ctx = 131072
    rows = rows_of(ctx)
    k_b = rows * KQ * AB
    v_b = rows * VD * AB
    assert k_b > budget and v_b > budget
    # in_elems=1 on each up-projection: the cache is kv_read, and one stand-in element still moves.
    sentinel = AB
    qo = (KQ + VD) * AB
    expect = m.n_layers * (2.0 * (k_b + v_b) + qo + 2.0 * sentinel)
    got = one(ctx).stages[0].dram["act"]
    assert got == expect


def test_tp_counts_dea_matmul_once():
    m = C.get_model(HYB)
    qk = next(o for o in build_rank_ops(m, 3, Phase("decode", 1, 1, 32), Shard(tp=2)) if o.name == "dea_qk")
    pv = next(o for o in build_rank_ops(m, 3, Phase("decode", 1, 1, 32), Shard(tp=2)) if o.name == "dea_pv")
    assert qk.replicated == 2 and pv.replicated == 2

    def useful(tp):
        r = evaluate(Scenario().replace("model", HYB).replace("layout.tp", tp).replace("serving.ctx", 4096))
        return sum(st.flops_u for st in r.stages) * tp

    assert useful(2) == useful(1)
