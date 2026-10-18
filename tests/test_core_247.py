"""0.247: diagnostics of the slot account's floor (U ≤ 0) — the free-slot law and the drift μ(f) — and the replay
hook _FLOOR_W (death credit below 0 = w·uniform-age + (1 − w)·λ replay ⇒ μ → w·μ).  DES (A vllm λ52, n64000):
deaths/s at k 63 / 34.5 / 14.7 = 14.5 / 12.8 / 9.4 vs uniform-age 15.3 / 12.6 / 8.0 — a full batch is young, a
thin one old, so the floor's restoring drift near U = 0 is weaker than the model's (MODEL §194; fix deferred)."""
import math

from accel_dse.core import pdqueue, pdsim
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving


def _cp():
    s = Scenario(chip=CHIPS["1P"], layout=Layout(tp=2), model="qwen3-8b", mem_id="hbm3e_8s_12h24g_9200",
                 serving=Serving(batch=64, prompt=4096, out_len=512),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6,
                             chunk_profile="vllm"))
    _, ctx = pdsim.capture_ctx(s)
    return ctx["cpool"], ctx["B"]


def test_floor_diagnostics_and_replay_hook():
    cp, B = _cp()
    kw = dict(tau=0.0396, n_B=0.866, k_bar=62.76)
    x = pdqueue._slot_backlog(cp, 13.0, 0.514, [(1.0, 0.049)], [1.0], None, 512, 0.0, B, **kw)
    cdf = [c for _, c in x["free_cdf"]]
    assert all(b >= a for a, b in zip(cdf, cdf[1:])) and cdf[0] > x["p_sat"] and cdf[-1] <= 1 + 1e-9
    mus = [m for _, m in x["mu_f"]]
    assert all(b > a for a, b in zip(mus, mus[1:]))           # drift toward U = 0 grows with the free slots
    assert x["sigma2b"] > x["sigma2"]
    assert pdqueue._FLOOR_W == 1.0
    old = pdqueue._FLOOR_W
    try:
        pdqueue._FLOOR_W = 0.65
        y = pdqueue._slot_backlog(cp, 13.0, 0.514, [(1.0, 0.049)], [1.0], None, 512, 0.0, B, **kw)
    finally:
        pdqueue._FLOOR_W = old
    # a weaker floor drift → more mass near U = 0 → more backlog; above 0 unchanged
    assert y["p_sat"] > x["p_sat"] and math.isclose(y["theta"], x["theta"])
    assert all(math.isclose(a * 0.65, b, rel_tol=1e-12) for (_, a), (_, b) in zip(x["mu_f"], y["mu_f"]))
