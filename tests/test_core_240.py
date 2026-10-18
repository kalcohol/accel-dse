"""0.240: max_num_seqs caps a request's fused time — at most B prompts are admitted during one decode life, so
TPOT_req ≤ (B·S_B + (out − B·n_B)·T_d)/out (or S_B/n_B when B·n_B ≥ out).  Near saturation the DES TPOT p90 sits
flat at that value (case E, n4000 × 10) while the 0.239 law climbed to 50 ms."""
from accel_dse.core import pdqueue, pdsim
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

# DES TPOT p90 (ms), n_req 4000, warmup 800, seeds 1..10 (/workspace/sd239/des_*.txt; DES unchanged since)
DES_TPOT90 = {"sglang_mixed": {18.0: 23.3, 18.6: 23.4, 19.32: 23.4, 19.76: 23.4},
              "vllm": {18.0: 22.9, 18.6: 23.0, 19.32: 23.0}}


def _ctx(prof):
    s = Scenario(chip=CHIPS["1P"], layout=Layout(tp=2), model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200", mem_cap_GB=48.0,
                 serving=Serving(batch=32, prompt=16384, out_len=256),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6,
                             chunk_profile=prof))
    return pdsim.capture_ctx(s)


def test_tpot_cap_matches_des_near_saturation():
    for prof, pts in DES_TPOT90.items():
        rep, ctx = _ctx(prof)
        for lam, des in pts.items():
            r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, lam)
            assert r["tpot_p90"] <= r["tpot_cap"] + 1e-12 and r["tpot_p99"] <= r["tpot_cap"] + 1e-12
            assert r["tpot_mean"] <= r["tpot_p90"]
            assert abs(r["tpot_p90"] - des / 1e3) / (des / 1e3) < 0.04, (prof, lam, r["tpot_p90"] * 1e3, des)


def test_tpot_cap_inactive_at_operating_point():
    for prof in ("sglang_mixed", "vllm"):
        rep, ctx = _ctx(prof)
        c = pdqueue._coloc_chunked(ctx, rep["queue"]["lambda_rps"])
        assert not c["tpot_capped"] and c["tpot_p90"] < c["tpot_cap"]
