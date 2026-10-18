"""0.241: max_num_seqs blocking of the chunk server.  In a backlog the fused paths admit a prompt only while
running + prefilling < B; the batch fills after J* admissions of a busy period and the next prompt then waits
V = (out − B·n_B)·T_d(B) for the oldest decode — one lumped vacation per B admissions (pdqueue._slot_block).  The
DES counterfactual with the slot cap lifted (sglang_mixed λ18.6: wait 743 → 524 ms) shows it is the TTFT mechanism;
queue–service correlation is weak (per-prompt share +2.7 % with queue)."""
import math

from accel_dse.core import pdqueue, pdsim
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

# DES TTFT p90 (ms), n_req 4000, warmup 800, seeds 1..10 (/workspace/sd239/des_*.txt; DES unchanged since 0.235)
DES_TTFT90 = {"sglang_mixed": {17.37: 1219, 18.0: 1603, 18.6: 2163, 19.32: 3353}}
OLD_TTFT90 = {"sglang_mixed": {17.37: 976, 18.0: 1124, 18.6: 1298, 19.32: 1641}}   # 0.240 analytic


def _ctx(prof):
    s = Scenario(chip=CHIPS["1P"], layout=Layout(tp=2), model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200", mem_cap_GB=48.0,
                 serving=Serving(batch=32, prompt=16384, out_len=256),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6,
                             chunk_profile=prof))
    return pdsim.capture_ctx(s)


def test_slot_block_reductions():
    # batch never fills when B·n_B ≥ out, or unstable
    assert pdqueue._slot_block(4.0, 0.8, 10.0, 32, 8.2, 256, 0.003) is None
    assert pdqueue._slot_block(4.0, 1.0, 10.0, 32, 4.0, 256, 0.002) is None
    V, p, J, phi = pdqueue._slot_block(4.65, 0.837, 13.2, 32, 4.19, 256, 0.0019)
    assert math.isclose(V, (256 - 32 * 4.19) * 0.0019)
    assert math.isclose(p, phi / 32) and 0 < phi < 1
    # deterministic aging fill: k_J ≥ B first at J
    k = lambda j: j + 13.2 * max(0.0, 1 - j * 4.19 / 256)
    assert k(J) >= 32 > k(J - 1)
    # light load: blocking negligible; more load / fuller start → more blocking
    assert pdqueue._slot_block(3.75, 0.664, 4.8, 32, 4.34, 256, 0.0018)[3] < 0.02
    assert pdqueue._slot_block(4.83, 0.87, 13.2, 32, 4.19, 256, 0.0019)[3] > phi
    assert pdqueue._slot_block(4.65, 0.837, 20.0, 32, 4.19, 256, 0.0019)[3] > phi


def test_slot_block_moves_ttft_toward_des():
    # 0.246: E mixed moved to the slot account (test_core_246); the 0.241–0.243 block is checked behind its flag
    flag = pdqueue._FUSED_BACKLOG_CHUNKED
    pdqueue._FUSED_BACKLOG_CHUNKED = False
    try:
        _slot_block_moves_ttft_toward_des()
    finally:
        pdqueue._FUSED_BACKLOG_CHUNKED = flag


def _slot_block_moves_ttft_toward_des():
    for prof, pts in DES_TTFT90.items():
        rep, ctx = _ctx(prof)
        for lam, des in pts.items():
            r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, lam)
            new, old = r["ttft"]["p90"] * 1e3, OLD_TTFT90[prof][lam]
            assert r["slot_block"] is not None
            assert old < new < des, (prof, lam, old, new, des)   # derived, partial: never overshoots the DES


def test_slot_block_off_away_from_saturation():
    for prof in ("sglang_mixed", "vllm"):
        rep, ctx = _ctx(prof)
        c = pdqueue._coloc_chunked(ctx, rep["queue"]["lambda_rps"])
        sb = c["slot_block"]
        assert sb is None or sb["phi"] < 0.02
