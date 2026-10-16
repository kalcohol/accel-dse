"""0.170: Evo 2 / StripedHyena 2 on core v2 — hyena SE / MR / LR operators (vortex engine), FFT-conv prefill with
modal-FFT state extraction, FIR / modal decode states, MHA-only KV cache."""
import math
from dataclasses import replace

from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, _hyena_ops, fft_flops
from accel_dse.core.model import from_release, load_release
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving

H100 = REF_HW["H100-SXM"]
IDS = (("evo2-40b", "arcinstitute/evo2_40b"), ("evo2-7b", "arcinstitute/evo2_7b"), ("evo2-1b", "arcinstitute/evo2_1b_base"))


def _scn(m, phase, P, **kw):
    return Scenario(model=m, chip=H100.chip, mem_id=H100.mem_id, serving=Serving(phase=phase, batch=1, prompt=P, ctx=P), **kw)


def test_params_match_checkpoints_and_layer_pattern():
    for mid, hf in IDS:
        s = from_release(mid, hf)
        rel = load_release(hf)
        assert s.params() == s.release_params, mid                      # exact vs the .pt headers (tied head deduped)
        c = rel["config"]
        kinds = [l.core.hy or l.core.kind for l in s.layers]
        assert kinds.count("se") == len(c["hcs_layer_idxs"]) and kinds.count("lr") == len(c["hcl_layer_idxs"])
        assert [i for i, k in enumerate(kinds) if k == "gqa"] == c["attn_layer_idxs"]
    s7 = from_release("evo2-7b", "arcinstitute/evo2_7b")
    # KV cache only in the 5 MHA layers: 5 × (k + v) × 4096
    assert sum(l.core.kv_elems_per_token() for l in s7.layers) == 5 * 2 * 4096
    assert s7.max_ctx == 1048576


def test_decode_state_bytes_by_hand():
    r = evaluate(_scn("evo2-7b", "decode", 8192))
    h = 4096
    hand = (27 * 3 * h * 2 * 2          # featurizer short FIR (3 taps → 2 cached columns of the 3h bf16 projections)
            + 9 * h * 6 * 2             # SE: 7 taps → 6 bf16 columns
            + 9 * h * 127 * 2           # MR: 128 taps → 127 bf16 columns
            + 9 * h * 16 * 4)           # LR: 16 fp32 modal states per channel
    assert r.stages[0].mem.state_total == hand
    assert r.fits and r.stages[0].dram["act"] == 0.0


def test_fft_prefill_ops_and_modal_state_cost():
    assert fft_flops(1024) == 5 * 1024 * 10 and fft_flops(1024, True) == 2.5 * 1024 * 10
    s = from_release("evo2-7b", "arcinstitute/evo2_7b")
    L = 4096
    ph = Phase("prefill", 1, L, 0)
    sh = Shard(tp=1)
    se, mr, lr = (next(l.core for l in s.layers if l.core.hy == k) for k in ("se", "mr", "lr"))
    base = 3 * 4096 * 3 * L * 2 + 2 * 4096 * L
    assert _hyena_ops(s, 0, se, ph, sh, 1)[0].vec == base + 2 * 7 * 4096 * L
    N = 2 * L
    o = _hyena_ops(s, 1, mr, ph, sh, 1)[0]
    assert math.isclose(o.vec, base + 4096 * (2 * fft_flops(N, True) + 6 * (N // 2 + 1)) + 256 * fft_flops(N, True) + 2 * 4096 * L)
    full = _hyena_ops(s, 2, lr, ph, sh, 1)[0]
    noc = _hyena_ops(s, 2, replace(lr, hy_prefill=""), ph, sh, 1)[0]
    # modal-FFT state extraction: fp32 / complex64 tensors of channels × 16 modes × 2L dominate the DRAM traffic
    assert full.act_x - noc.act_x == 4096 * 16 * L * (24 + 80)
    assert full.act_bytes == 2 * 4096 * 16 * N * 8 and noc.act_bytes == 2 * 4 * 4096 * 16 * L   # 0.171: + stock h build
    assert full.act_x > 10 * o.act_x


def test_prefill_scaling_and_1m_capacity():
    r8 = evaluate(_scn("evo2-7b", "prefill", 8192))
    st = r8.stages[0]
    assert r8.fits and st.dram["act"] > 10 * st.dram["weights"]          # FFT intermediates, not weights, move bytes
    # the as-released modal-FFT prefill materialises 2 × h·16·2L complex64 per LR layer: 1M bases cannot fit one card
    r1m = evaluate(_scn("evo2-7b", "prefill", 1048576))
    assert not r1m.fits
    need = 2 * 4096 * 16 * 2 * 1048576 * 8
    assert r1m.stages[0].mem.dram_need > need
    # decode: weights + caches only; KV grows only in the 5 attention layers
    d1, d2 = evaluate(_scn("evo2-7b", "decode", 8192)), evaluate(_scn("evo2-7b", "decode", 65536))
    for d, P in ((d1, 8192), (d2, 65536)):
        assert d.stages[0].mem.kv_total == (P + 1) * 5 * 2 * 4096 * 2     # ctx + the step's token, bf16
        assert d.stages[0].mem.state_total == d1.stages[0].mem.state_total  # hyena states do not grow with ctx
