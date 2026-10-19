"""0.248: the slot account's derived floor pieces (MODEL §195).
(1) Age-structured death credit below 0: admissions trigger slow iterations (Δ extra wall time each) that collect
    further births, so births cluster in token time (branching η = λΔ) and the dying (oldest) member covaries less with
    the batch count: w = 1 − (1 − g)(I_s − I₀)/(2 I_s), death credit c(k)·a(k), a(k) = w + (1 − w)·m/k.
    A λ52: w 0.867 (token-time micro-model 0.87, DES per-step ≈ 0.88).
(3) Queue-ful entry: U > 0 means q ≥ 1 with the slots full; U enters by a whole S̄ and the admission in service is
    not in U, so the U > 0 TTFT base is lat + β + (β/d)·(S̄ + E[S²]/(2S̄)) — slot-bound share β/d (A 0.89; E 16k
    prompts ≈ 0, where the queue-ful state is the budget queue already priced by T₀) (A λ52: 79.5 → 173 ms)."""
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


KW = dict(tau=0.0396, n_B=0.866, k_bar=62.76)


def _sb(cp, B, **extra):
    return pdqueue._slot_backlog(cp, 13.0, 0.514, [(1.0, 0.049)], [1.0], None, 512, 0.0, B, **KW, **extra)


def test_age_structured_death_weight():
    cp, B = _cp()
    assert pdqueue._AGE_DEATHS and pdqueue._SAT_ENTRY
    x0 = _sb(cp, B)                       # no mean batch given → uniform age (w = 1, 0.247 floor)
    x = _sb(cp, B, m_k=40.0)
    assert x0["w_age"] == 1.0
    assert 0.80 < x["w_age"] < 0.92, x["w_age"]
    # above 0 untouched; a(k) < 1 above the mean batch weakens the restoring drift near full → more backlog mass
    assert math.isclose(x["theta"], x0["theta"]) and math.isclose(x["beta"], x0["beta"])
    assert x["p_sat"] > x0["p_sat"]
    old = pdqueue._AGE_DEATHS
    try:
        pdqueue._AGE_DEATHS = False
        y = _sb(cp, B, m_k=40.0)
    finally:
        pdqueue._AGE_DEATHS = old
    assert y["w_age"] == 1.0 and math.isclose(y["p_sat"], x0["p_sat"])


def test_queue_ful_entry_base():
    cp, B = _cp()
    x = _sb(cp, B, m_k=40.0)
    s_bar = x["tau"] + x["beta"]
    e_s2 = x["sigma2"] / 13.0
    sb = x["beta"] / x["d"]                 # slot-bound share (E 16k: β ≈ 0 → no entry term; test_core_242)
    assert math.isclose(x["slot_bound"], sb) and 0.8 < sb < 0.95
    assert math.isclose(x["base"], 0.049 + x["beta"] + sb * (s_bar + e_s2 / (2 * s_bar)), rel_tol=1e-12)
    old = pdqueue._SAT_ENTRY
    try:
        pdqueue._SAT_ENTRY = False
        y = _sb(cp, B, m_k=40.0)
    finally:
        pdqueue._SAT_ENTRY = old
    assert math.isclose(y["base"], 0.049 + y["beta"]) and math.isclose(y["p_sat"], x["p_sat"])
    assert 0.08 < x["base"] - y["base"] < 0.11          # A: 0.89 × (S̄ 70 ms + S_e 35 ms)


def test_deepseek_v41_flash_1017_alias():
    # 0.248: DeepSeek-V4.1-Flash-1017 (2026-10-17, HF sha 1bae0e10, MIT): config.json identical to V4.1-Flash
    # (sorted diff empty) and safetensors index total_size 510286023000 = the stored bytes_total → an alias + release
    # note on deepseek-v4.1-flash, no separate release file
    import json
    from pathlib import Path
    from accel_dse.core import catalog
    assert catalog.get_model("deepseek-ai/DeepSeek-V4.1-Flash-1017") == catalog.get_model("deepseek-v4.1-flash")
    data = Path(catalog.DATA)
    e = [x for x in json.loads((data / "series_catalog.json").read_text())["entries"] if x["id"] == "deepseek-v4.1-flash"][0]
    rel = e["releases"][0]
    assert rel["hf_id"] == "deepseek-ai/DeepSeek-V4.1-Flash-1017" and "RLSF" in rel["note"]
    stored = json.loads((data / "releases" / "deepseek-ai__DeepSeek-V4.1-Flash.json").read_text())["bytes_total"]
    assert rel["bytes_total"] == stored
    assert not (data / "releases" / "deepseek-ai__DeepSeek-V4.1-Flash-1017.json").exists()
