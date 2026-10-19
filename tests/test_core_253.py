"""0.253: HF freshness sweep (≥ 2026-10-15) — NVIDIA Nemotron-Labs Dory looped LM (vLLM PR #60149 models/dory.py),
merged DeepSeek-V4.1-Flash-1017 / MAI-Gym-9B — and DSpine sampled rounds + published-acceptance source
(sglang_dspine worker.py DSpineDraftSampler.sample / kernels.serial_rs_table)."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import catalog as C
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, dspark_ops
from accel_dse.core.ir import Shard
from accel_dse.core.model import spec_accept_from_tau
from accel_dse.core.scenario import Scenario

D, DS = "dory1.0-4b", "qwen3-8b-dspine"


def _ev(mid, phase="decode", sv=None, **wl):
    s = Scenario().replace("model", mid).replace("serving.phase", phase)
    for k, v in (sv or {}).items():
        s = s.replace("serving." + k, v)
    for k, v in wl.items():
        s = s.replace("workload." + k, v)
    return evaluate(s)


def test_dory_structure_exact():
    for mid, fm in ((D, "bf16"), (D + "-mxfp8", "fp8"), (D + "-nvfp4", "nvfp4")):
        m = C.get_model(mid)
        assert m.param_check()["rel_err"] == 0.0 and m.release_params == 3975098368
        assert m.coverage == "full" and not m.coverage_reasons and m.act_fmt == "bf16"     # W8A16 / W4A16 weight-only
        assert dict(m.formats)["mlp"].fmt == fm and dict(m.formats)["attn"].fmt == fm
        assert m.loop_spec == (16, 40, 2, "per_loop") and len(m.loop_phys) == 72
        assert len(m.layers) == 16 + 40 * 2 + 16                       # input + recurrent × 2 (config) + output
        assert sum(1 for L in m.layers if L.loop) == 40 and not any(L.kv_shared for L in m.layers)
        attn = [L for L in m.loop_phys if L.core.kind == "gqa"]
        assert len(attn) == 36 and sum(1 for L in attn if L.core.window is None) == 9   # every 4th attention is full
        assert attn[0].attn_linears[0].n == 2 * 16 * 256                # q + g_proj output gate
        assert m.final_norm_params == 0 and not m.tie_embeddings


def test_dory_loops_and_kv_mode():
    r1, r2, r4 = (_ev(D, loops=n) for n in (1, 2, 4))
    l4 = _ev(D, loops=4, loop_kv="last_loop")
    w = [r.stages[0].mem.stored_w for r in (r1, r2, r4, l4)]
    assert max(w) == min(w)                                             # shared weights: stored once at any loop count
    kv = [r.stages[0].mem.kv_total for r in (r1, r2, r4)]
    assert kv[0] < kv[1] < kv[2] and l4.stages[0].mem.kv_total == kv[0]   # last_loop: the one-loop KV
    assert r1.tpot < r2.tpot < r4.tpot and l4.tpot == r4.tpot           # same reads / writes per step
    assert _ev(D).tpot == r2.tpot and "loops" not in Scenario().to_dict()["workload"]
    assert any("TP = 1" in x for x in evaluate(Scenario().replace("model", D).replace("layout.tp", 2)).warnings)
    assert any("只用于循环 LM" in x for x in _ev("qwen3-8b", loops=2).warnings)


def test_merged_releases():
    um = {x["id"]: x for x in C.unlisted_models()}
    for mid, base in (("mai-gym-9b", "qwen3.5-9b"),):
        a, b = C.get_model(mid), C.get_model(base)
        assert mid in um and a.params() == b.params() and a.release_params == b.release_params
        assert a.param_check()["rel_err"] < 1e-6
    assert C.get_model("mai-gym-9b").vision is not None and not C.get_model("mai-gym-9b").mtp_layers
    listed = {x["id"] for x in C.list_models()}
    assert {D, D + "-mxfp8", D + "-nvfp4"} <= listed


def test_dspine_sampled_round_ops():
    m = C.get_model(DS)
    g = {o.name for o in dspark_ops(m, Phase("decode", 1, 16, 4096), Shard())}
    s = {o.name: o for o in dspark_ops(m, Phase("decode", 1, 16, 4096, samp=True), Shard())}
    new = set(s) - g
    assert new == {"dspine_rs_gather", "dspine_rs_bridge", "dspine_rs_write", "dspine_rs_logits", "dspine_rs_sample"}
    n, K, R, Cc, H = 15, 16, 512, 1024, 4096
    macs = (s["dspine_rs_bridge"].m * s["dspine_rs_bridge"].k * s["dspine_rs_bridge"].n
            + s["dspine_rs_write"].m * s["dspine_rs_write"].k * s["dspine_rs_write"].n
            + s["dspine_rs_logits"].m * s["dspine_rs_logits"].k * s["dspine_rs_logits"].n * s["dspine_rs_logits"].count)
    assert macs == n * K * (2 * R * Cc + H * R + K * H)                 # K·(2RC + HR + KH) per draft row
    assert s["dspine_rs_bridge"].w_fmt == "fp32" and s["dspine_rs_gather"].kv_read == n * K * (Cc + H) * 2


def test_dspine_sampling_and_published_acceptance():
    base = {"spec_k": 15}
    g = _ev(DS, sv=base)
    sm = _ev(DS, sv={**base, "spec_sampling": "sampled"})
    assert sm.tpot > g.tpot and sm.spec["accept"] == g.spec["accept"] == 0.7      # manual lever unchanged
    p0 = _ev(DS, sv={**base, "spec_accept_src": "published"})
    p1 = _ev(DS, sv={**base, "spec_accept_src": "published", "spec_sampling": "sampled"})
    assert p0.spec["accept"] == round(spec_accept_from_tau(7.83, 15), 4) and abs(p0.spec["tau"] - 7.83) < 0.01
    assert p1.spec["accept"] == round(spec_accept_from_tau(6.98, 15), 4) and abs(p1.spec["tau"] - 6.98) < 0.01
    assert p0.tpot < p1.tpot < g.tpot
    rd = _ev("qwen3.8-27b-ready", sv={"spec_k": 7, "spec_accept_src": "published", "spec_sampling": "sampled"})
    assert rd.spec["accept"] == 0.7 and any("公开接受长度" in w for w in rd.warnings)
    for bad in ({"serving": {"spec_sampling": "hot"}}, {"serving": {"spec_accept_src": "x"}}, {"workload": {"loop_kv": "x"}},
                {"workload": {"loops": 17}}):
        try:
            Scenario.from_dict({"model": DS, **bad})
            raise AssertionError(bad)
        except ValueError:
            pass
