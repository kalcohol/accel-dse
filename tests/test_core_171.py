"""0.171: Evo 2 long-context prefill — context parallel for Hyena (savanna use_cp_hyena all-to-all) + ring attention,
vortex recurrence prefill, fused on-chip FFT-conv what-if with four-step spill."""
from dataclasses import replace

from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, _hyena_ops, fft_flops
from accel_dse.core.memplan import fft_spill
from accel_dse.core.model import from_release, with_hyena_prefill
from accel_dse.core.parallel import Layout
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving

H100 = REF_HW["H100-SXM"]
S7 = from_release("evo2-7b", "arcinstitute/evo2_7b")
LR = next(l.core for l in S7.layers if l.core.hy == "lr")


def _run(P, mode="reference", lay=Layout(), **kw):
    return evaluate(Scenario(model="evo2-7b", chip=H100.chip, mem_id=H100.mem_id, layout=lay, hyena_prefill=mode,
                             node_cards=8, serving=Serving(phase="prefill", batch=1, prompt=P, ctx=P), **kw))


def test_cp_hyena_all_to_all_and_full_length_fft():
    S, cp = 65536, 4
    ph = Phase("prefill", cp, S // cp, 0, cp=cp, cp_len=S)
    ops = _hyena_ops(S7, 2, LR, ph, Shard(tp=2), 1)
    comm = [o for o in ops if o.kind == "comm"]
    a2a = [o for o in comm if o.comm_kind == "alltoall"]
    assert len(a2a) == 4 and all(o.comm_group == cp and o.comm_stride == 2 for o in a2a)
    assert a2a[0].comm_bytes == (S // cp) * (4096 // 2) * 2                 # x1 / x2 / v / z: rank's block, bf16
    halo = [o for o in comm if o.comm_kind == "p2p"][0]
    assert halo.comm_bytes == 3 * 4096 // 2 * 2 * 2                          # 2 columns of the 3h/TP featurizer
    st = ops[0]
    one = _hyena_ops(S7, 2, LR, Phase("prefill", 1, S, 0), Shard(tp=2 * cp), 1)[0]
    # whole-sequence FFT on ch / cp channels == TP·cp channel split of the full sequence (same FFT work)
    assert abs(st.vec - one.vec) / one.vec < 1e-9 and st.act_bytes == one.act_bytes


def test_1m_prefill_feasibility():
    assert not _run(1048576).fits                                            # 0.170: 2140 GiB on one card
    r = _run(1048576, lay=Layout(tp=8, cp=8))
    assert r.fits and r.stages[0].time.t_link > 0                            # 64 cards: a2a + ring KV traffic
    assert not _run(1048576, lay=Layout(cp=8)).fits                          # modal-FFT tensors still too big
    f = _run(1048576, "fused", Layout(cp=8))
    assert f.fits and f.stages[0].mem.dram_need < 40 * 2**30


def test_recurrence_prefill_mechanism():
    m = with_hyena_prefill(S7, "recurrence")
    assert [l.core.hy_prefill for l in m.layers if l.core.hy == "mr"][0] == ""
    L = 8192
    o = _hyena_ops(m, 2, next(l.core for l in m.layers if l.core.hy == "lr"), Phase("prefill", 1, L, 0), Shard(), 1)[0]
    ref = _hyena_ops(S7, 2, LR, Phase("prefill", 1, L, 0), Shard(), 1)[0]
    assert o.launch == 14 * L and ref.launch == 0
    assert o.act_bytes == max(8 * 4096 * 16 * L, 4 * 4096 * L * 16) and o.act_bytes < ref.act_bytes
    # the per-token Python loop moves more bytes than the FFT path at 8k: not a speed-up, only less peak memory
    r8, c8 = _run(8192, "recurrence"), _run(8192)
    assert r8.stages[0].dram["act"] > c8.stages[0].dram["act"] and r8.ttft > c8.ttft
    assert r8.stages[0].mem.dram_need < c8.stages[0].mem.dram_need


def test_fused_spill_and_reference_unchanged():
    ch = H100.chip
    fit = ch.sram_bytes / 2 / ch.engines / 8
    assert fft_spill(16384, ch.sram_bytes, ch.engines) == 0.0 and 16384 < fit
    n = 2 * 1048576
    assert fft_spill(n, ch.sram_bytes, ch.engines) == 16.0 * n                # four-step: 2 passes → 1 extra
    assert fft_spill(n, 2**20, 1024) > 16.0 * n                                # small SRAM per engine: more passes
    ref, fu = _run(8192), _run(8192, "fused")
    assert abs(ref.ttft - 0.37786053076894655) < 1e-12   # 0.209 0.3778605 (STATE_SPLIT: Hyena state misses by its own share; was 0.37786546319722353) ← 0.193 0.377865 (H100 GEMMs on cta + tile I/O; was 0.3742307611) ← 0.170 0.50315 → 0.172 0.41291 (attn orientation) → 0.173 0.37423 (wide GEMM orientation / split-K) → 0.192 0.3742307611 (state pinned before weights; was …097030)
    assert fu.stages[0].dram["act"] < ref.stages[0].dram["act"] / 10 and fu.ttft < 0.6 * ref.ttft
    assert any("fused" in w for w in fu.warnings)
    assert fft_flops(8) == 5 * 8 * 3
