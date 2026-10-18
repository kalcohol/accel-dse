"""0.246: the budget-chunked fused cases (C(B) < 2 prompts: D/F, A vllm_v0) on the slot account.  At a full slot cap the
partial prompt holds a slot (max_num_seqs counts it): its chunks ride B − 1 decodes and a follower head joins the tail
only in a slot a death freed (the two-prompt iteration at B − 2); the follower starts at phase φ = the tail (resumed
chunk, longer own tail).  pack_at(B) priced it ~1 % low (D 45.3 / F 66.8 ms).  Long-run DES saturated iterations:
τ D 45.8 / F 67.5 ms, n_B 2.38 / 3.26, running k 62.7; SLO rates D vllm 55.45, F vllm 43.75 (6 × n64000)."""
from accel_dse.core import pdqueue, pdsim
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

HB = "hbm3e_8s_12h24g_9200"


def _ctx(prompt, prof, cap=None, batch=64, out=512):
    kw = {"mem_cap_GB": cap} if cap else {}
    s = Scenario(chip=CHIPS["1P"], layout=Layout(tp=2), model="qwen3-8b", mem_id=HB, **kw,
                 serving=Serving(batch=batch, prompt=prompt, out_len=out),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6,
                             chunk_profile=prof))
    return pdsim.capture_ctx(s)


def test_budget_chunked_account_matches_des_iteration():
    for prompt, lam, tau_des, nb_des in ((4096, 55.45, 0.0458, 2.37), (6000, 43.75, 0.0675, 3.26)):
        _, ctx = _ctx(prompt, "vllm", cap=48.0)
        r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, lam)
        sb = r["slot_backlog"]
        assert r.get("slot_block") is None
        assert abs(sb["tau"] / tau_des - 1) < 0.005, (prompt, sb["tau"])          # 0.245 pack_at(B): 45.27 / 66.84
        assert abs(sb["n_B"] / nb_des - 1) < 0.02, (prompt, sb["n_B"])
        # k̄ = B − lat/τ (prompts in flight, Little): DES running k 62.7 in saturated iterations
        assert 62.5 < sb["k_bar"] < 63.0, sb["k_bar"]
        assert 0.8 * 2.0 < r["ttft"]["p90"] < 1.15 * 2.0, (prompt, r["ttft"]["p90"])   # 1.82 / 1.71 s


def test_phase_iters_k_top():
    _, ctx = _ctx(4096, "vllm", cap=48.0)
    cp, B = ctx["cpool"], ctx["B"]
    C = pdqueue.chunk_at(ctx["chunk"], B - 1)
    a = pdqueue._phase_iters(cp, B - 1, 4096, 0, C, 0, True)
    b = pdqueue._phase_iters(cp, B - 1, 4096, 0, C, 0, True, k_top=B - 1)
    assert a == b
    # unshared tails never use k_top (one piece)
    c = pdqueue._phase_iters(cp, B - 1, 4096, 0, C, 0, False, k_top=B - 2)
    d = pdqueue._phase_iters(cp, B - 1, 4096, 0, C, 0, False)
    assert c == d
    # the topped tail at k_top: its wall is the B − 2 two-piece iteration
    e = pdqueue._phase_iters(cp, B - 1, 4096, 0, C, 0, True, k_top=B - 2)
    used = e["used"]
    T = pdqueue._packed_fused(cp, B - 2, [(4096, 0, 4096 - used, 4096), (4096, 0, 0, C - used)])
    assert abs((e["share"] - d["pre"]) - used / C * T) < 1e-12
