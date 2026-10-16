"""0.172: long-context attention stays near its FLOP roofline (standing invariant).

Before 0.172 LLM attention kept the 0.40 orientation, so on the "wide" GPU references PV (N = head_dim) used
128 of 8448 array columns: attention ran ~34x its FLOP time at every length (llama-3.1-8b 131k TTFT 156 s)."""
import sys
sys.path.insert(0, ".")
from accel_dse.core.refhw import REF_HW
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.evaluate import evaluate
from accel_dse.core import evaluate as E
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.catalog import get_model

LENS = (4096, 32768, 131072)


def attn_ratio(chip, mem_id, P, model="llama-3.1-8b"):
    m = get_model(model)
    ops = [o for o in build_rank_ops(m, 0, Phase("prefill", 1, P, 0), Shard()) if o.kind == "attn"]
    # Op.flops is already causal-halved; build_rank_ops gives one layer
    ideal = len(m.layers) * sum(o.flops for o in ops) / 2 / chip.macs / chip.freq_ghz / 1e9
    r = evaluate(Scenario(model=model, chip=chip, mem_id=mem_id,
              serving=Serving(phase="prefill", batch=1, prompt=P, ctx=P)))
    return r.stages[0].time.t_arr_attn / ideal, r


def test_refhw_within_factor():
    """Standing invariant: prefill attention array time within [0.9, 3]× its FLOP time, 4k–131k, every reference chip."""
    for name, h in REF_HW.items():
        for P in LENS:
            x, _ = attn_ratio(h.chip, h.mem_id, P)
            assert 0.9 <= x <= 3.0, (name, P, round(x, 2))


def test_presets_within_factor():
    mem = REF_HW["H100-SXM"].mem_id
    for name in ("100T", "1P"):
        for P in LENS:
            x, _ = attn_ratio(CHIPS[name], mem, P)
            assert 0.9 <= x <= 3.0, (name, P, round(x, 2))


def test_old_behaviour_was_the_artifact():
    h = REF_HW["H100-SXM"]
    with E.attn_orient(False):
        old, ro = attn_ratio(h.chip, h.mem_id, 131072)
    new, rn = attn_ratio(h.chip, h.mem_id, 131072)
    assert old > 20 and ro.ttft > 100          # ≤ 0.171: 33.5× ideal, TTFT 156 s
    assert new < 1.5 and rn.ttft < 12          # 0.172: 1.02×, TTFT 8.0 s
