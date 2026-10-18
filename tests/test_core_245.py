"""0.245: vLLM V1 counts the prefilling requests in max_num_seqs too (scheduler.schedule(): the waiting loop breaks at
len(running) >= max_num_running_reqs), so the 0.244 slot account (pdqueue._slot_backlog) carries over to the fused
engines: an admission's n_B chunk iterations at a full batch already give n_B·B·e/out deaths, β = d − n_B·T_d(B).
Applied where the budget fits ≥ 2 whole prompts at a full batch (A/G); budget-chunked cases keep the 0.243 clustered
block.  Long-run DES (6 seeds × n64000): saturated admission share A 39.5 / G 48.0 ms; SLO rates A vllm 55.28,
G vllm 49.02 (0.244 analytic 56.20 / 49.99)."""
import math

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


def test_fused_account_packed_cases():
    for prompt, lam, tau_des, p90_des in ((4096, 55.28, 0.0395, 2.0), (5000, 49.02, 0.0480, 2.0)):
        _, ctx = _ctx(prompt, "vllm")
        r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, lam)
        sb = r["slot_backlog"]
        assert abs(sb["tau"] / tau_des - 1) < 0.01, (prompt, sb["tau"])
        assert 0.7 < sb["n_B"] < 0.95
        # β = d − n_B·T_d(B)·k̄/B: the chunk iterations' own deaths (0.246: k̄ decodes ride them, not B)
        tdB = ctx["cpool"].run("decode", ctx["B"]).step
        assert abs(sb["beta"] - (sb["d"] - sb["n_B"] * tdB * sb["k_bar"] / ctx["B"])) < 1e-12
        assert 0.8 * p90_des < r["ttft"]["p90"] < 1.15 * p90_des, (prompt, r["ttft"]["p90"])   # 0.244: 1.32 / 1.30 s


def test_budget_chunked_keeps_clustered_block():
    # 0.246: budget-chunked cases moved to the account (test_core_246); the 0.243 block stays behind the flag
    _, ctx = _ctx(4096, "vllm", cap=48.0)
    old = pdqueue._FUSED_BACKLOG_CHUNKED
    pdqueue._FUSED_BACKLOG_CHUNKED = False
    try:
        r = pdqueue._coloc_chunked({**ctx, "slo": (2.0, 0.05)}, 55.45)
    finally:
        pdqueue._FUSED_BACKLOG_CHUNKED = old
    assert "slot_backlog" not in r and r["slot_block"] is not None


def test_slot_backlog_no_block_when_chunks_cover_the_life():
    _, ctx = _ctx(4096, "vllm")
    cp, B = ctx["cpool"], ctx["B"]
    x = pdqueue._slot_backlog(cp, 13.0, 0.5, [(1.0, 0.04)], [1.0], None, 512, 0.0, B, tau=0.04, n_B=9.0)
    assert x["p_sat"] == 0.0 and x["beta"] == 0.0 and math.isinf(x["theta"])
    # n_B = 0 is the 0.244 SGLang non-mixed account (β = d)
    y = pdqueue._slot_backlog(cp, 13.0, 0.5, [(1.0, 0.04)], [1.0], None, 512, 0.0, B)
    assert abs(y["beta"] - y["d"]) < 1e-15
