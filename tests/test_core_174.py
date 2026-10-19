"""0.174: Meta Muse Spark Open (vLLM blog 2026-10-15 + vLLM PR #61879 @ 6cc692b5 + public AMD release headers)."""
import json
import sys
sys.path.insert(0, ".")
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.parallel import Layout
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.evaluate import evaluate

REL = "accel_dse/data/releases/"
IDS = ("muse-spark-open", "muse-spark-open-fp8", "muse-spark-open-mxfp4")


def _roles(f):
    d = json.load(open(REL + f))
    return d, {r: sum(x["params"] for x in v.values()) for r, v in d["roles"].items()}


def test_listed_and_params_match_headers():
    ids = {e["id"] for e in list_models()}
    assert set(IDS) <= ids
    for i in IDS:
        pc = get_model(i).param_check()
        assert abs(pc["rel_err"]) < 1e-5, (i, pc)


def test_blog_breakdown():
    """Blog table: 799B experts, 13.5B attention, 4.2B latent projections, 3.3B embed + head, 1.9B vision (separate)."""
    d, p = _roles("amd__Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8.json")
    assert abs(p["expert"] / 1e9 - 799) < 0.5
    assert abs(p["attn"] / 1e9 - 13.5) < 0.05
    assert abs(p["mlp"] / 1e9 - 4.2) < 0.05                    # pooler + unpooler (role "mlp" in the header map)
    assert abs((p["embed"] + p["lm_head"]) / 1e9 - 3.3) < 0.05
    assert 1.9 < p["vision"] / 1e9 < 2.1                       # 「来源」 header: vision 1.9B + audio 118M + projectors
    m = get_model("muse-spark-open")
    act = (m.active_params() - 2 * 202048 * 8192) / 1e9          # blog "42B active" (excl. embedding / head)
    assert 42 <= act < 43.5, act


def test_structure_from_pr_code():
    m = get_model("muse-spark-open")
    L = m.layers
    assert len(L) == 62
    full = [i for i, l in enumerate(L) if not l.core.window]
    assert len(full) == 16 and full[-1] == 61 and all(i % 4 == 1 for i in full)          # cycle of 4 anchored to the end
    assert all(l.core.window == 2048 for l in L if l.core.window)
    l0 = L[0]
    assert (l0.core.n_q, l0.core.n_kv, l0.core.qk_dim) == (128, 16, 64)
    q = next(x for x in l0.attn_linears if x.name == "q")
    assert q.n == 2 * 128 * 64                                    # q + per-head output gate (QKVOGate)
    f = l0.ffn
    assert (f.n_experts, f.top_k, f.d_expert, f.latent, f.n_shared) == (256, 8, 4096, 4096, 0)
    names = {x.name for x in l0.ffn_linears}
    assert {"router", "latent_up", "latent_down"} <= names


def test_kv_at_1m_just_over_a_quarter():
    """Blog: at 1M tokens the cache is just over ¼ of all-full attention (46 SWA layers hold 2048)."""
    m = get_model("muse-spark-open")
    S = 1 << 20
    kv = sum(l.core.kv_elems_per_token() * min(S, l.core.window or S) for l in m.layers)
    allfull = sum(l.core.kv_elems_per_token() * S for l in m.layers)
    assert 0.25 < kv / allfull < 0.27


def test_evaluates_and_capacity():
    lay = Layout(tp=8, ep=8)
    b = REF_HW["B200"]
    r = evaluate(Scenario(model="muse-spark-open", chip=b.chip, mem_id=b.mem_id, layout=lay,
                          serving=Serving(phase="decode", batch=1, ctx=8192)))
    assert not r.fits                                              # 1.64 TB BF16 > 8 × 192 GB
    m = REF_HW["MI300X"]
    r = evaluate(Scenario(model="muse-spark-open-mxfp4", chip=m.chip, mem_id=m.mem_id, layout=lay,
                          serving=Serving(phase="decode", batch=1, ctx=1 << 20)))
    assert r.fits and r.tpot > 0


def test_tpu_v7_sanity_not_fitted():
    """Labelled sanity only (MODEL §120): blog TPU v7 megakernel 270 / 350 tok/s (no spec, 4 / 8 chips) vs a weight-
    streaming roofline N · 7380 GB/s (Google Cloud TPU7x doc) / active bytes per token of the MXFP4 release (megakernel
    dtype not published 「假设」).  Reported must sit below the roofline."""
    d = json.load(open(REL + "amd__Muse-Spark-Open-816B-A42B-Quark-MXFP4.json"))
    b = {r: sum(x["bytes"] for x in v.values()) for r, v in d["roles"].items()}
    act = b["attn"] + b["mlp"] + b["router"] + b["lm_head"] + b["expert"] * 8 / 256
    for n, rep in ((4, 270), (8, 350)):
        roof = n * 7380e9 / act
        assert rep < roof, (n, rep, roof)


def test_vision_tower_matches_blog():
    """PR vision.py NaViT + adapter = 1.89B (blog 1.9B vision); header remainder (audio + embeds) is storage only."""
    v = get_model("muse-spark-open").vision
    assert abs((v.params - v.extra) / 1e9 - 1.9) < 0.02
    assert v.depth == 50 and v.patch == 14 and v.merge == 2 and not v.approx   # 0.255: resize sourced
