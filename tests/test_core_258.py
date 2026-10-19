"""0.258: MOSS-Image CFG negative branch at its own shape (release pipeline: empty negative prompt → 10 chat-template
tokens, block 22 skipped, two serial transformer calls per step; workload.cfg_branches) and the text-refiner attention
core (ModelSpec.io_attn); SLC policy "dip" (thrash-resistant insertion, Qureshi et al. ISCA 2007)."""
from dataclasses import replace

from accel_dse.core import catalog as C
from accel_dse.core.energy import action_counts
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, Shard, full_io_ops
from accel_dse.core.memplan import DIP_EPS, dip_hit
from accel_dse.core.scenario import Scenario, Serving

M = "moss-image-base"
LP = "lpddr5x_4x64_8533_16g"


def _ev(mid=M, **wl):
    s = Scenario().replace("model", mid)
    for k, v in wl.items():
        s = s.replace("workload." + k, v)
    return evaluate(s)


def test_moss_declares_negative_branch_and_refiner_attention():
    for mid in (M, "moss-image-sft"):
        m = C.get_model(mid)
        assert m.workload.uncond == (10, (22,))
        assert [a[0] for a in m.io_attn] == ["refiner0", "refiner1"] and all(a[2:] == (24, 128, 1) for a in m.io_attn)
        assert m.param_check()["rel_err"] == 0.0 and m.coverage == "full"


def test_refiner_attention_ops_follow_text_rows():
    m = C.get_model(M)
    ph = Phase("full", 1, 4352, 0, frames=1)
    ops = {o.name: o for o in full_io_ops(m, ph, Shard(), "pre")}
    qk = ops["io.refiner0.qk"]
    assert qk.m == qk.n == 256 and qk.k == 128 and qk.count == 24
    ops_u = {o.name: o for o in full_io_ops(m, replace(ph, q=4096 + 10, txt=10), Shard(), "pre")}
    assert ops_u["io.refiner1.qk"].m == 10
    assert not any(o.name.startswith("io.refiner") and o.name.endswith(".qk")
                   for o in full_io_ops(m, replace(ph, mreuse=True), Shard(), "pre"))
    assert not any(".qk" in o.name for o in full_io_ops(m, ph, Shard(), "post"))


def test_cfg_branches_ref_vs_same():
    r, s = _ev(), _ev(cfg_branches="same")
    a = r.cfg_asym
    assert s.cfg_asym is None and r.microbatches == 2 and s.microbatches == 1
    assert a["uncond_text_tokens"] == 10 and a["skip_layers"] == [22]
    assert a["cond_tokens"] == 4352 and a["uncond_tokens"] == 4096 + 10
    w = s.workload
    same_seq = s._tflop_per_request() / (w.steps * w.seqs_per_request)
    assert abs(a["cond_tflop_seq"] - same_seq) <= 1e-9 * same_seq          # the conditional branch is unchanged
    assert a["uncond_tflop_seq"] < a["cond_tflop_seq"] * 41 / 42           # one block fewer and 246 rows fewer
    assert abs(r._tflop_per_request() - (a["cond_tflop_seq"] + a["uncond_tflop_seq"]) * w.steps) < 1e-6
    pl = r.pipeline
    den = w.steps * (a["cond_ms"] + a["uncond_ms"]) / 1e3
    assert abs(pl["denoise_s"] - den) < 1e-9 and abs(r.latency - (den + pl["te_s"] + pl["decode_s"] + pl["load_s"])) < 1e-9
    te = next(p for p in pl["parts"] if p["role"] == "text_encoder")
    assert te["neg_tokens"] == 10
    assert r.domain_summary()["forward_ms"] == a["step_ms"]
    # energy counts carry both branches: MAC work tracks the FLOP ratio
    mr, ms = action_counts(r)["counts"]["mac"], action_counts(s)["counts"]["mac"]
    tr = r.domain_summary()["dit_tflop_per_request"] / s.domain_summary()["dit_tflop_per_request"]
    assert abs(mr / ms - tr) < 0.01


def test_cfg_branches_fallbacks():
    s = Scenario().replace("model", M)
    for k, v in (("layout.pp", 2), ("serving.microbatches", 3), ("workload.step_skip", 5)):
        r = evaluate(s.replace(k, v))
        assert r.cfg_asym is None and any("CFG 负分支" in w for w in r.warnings), k
    assert _ev(cfg=1).cfg_asym is None
    assert _ev("qwen-image").cfg_asym is None                              # no declared negative branch
    try:
        s.replace("workload.cfg_branches", "x")
        raise AssertionError("bad cfg_branches accepted")
    except ValueError:
        pass


def test_dip_hit_formula():
    assert dip_hit(10, 5) == 1.0 and dip_hit(10, 10) == 1.0
    K, T = 8.0, 20.0
    assert abs(dip_hit(K, T) - (K - DIP_EPS * (T - K)) / T) < 1e-15
    assert dip_hit(1.0, 100.0) == 0.0                                       # ε(T − K) > K: nothing retained
    xs = [dip_hit(k, 32.0) for k in range(1, 33)]
    assert all(b >= a for a, b in zip(xs, xs[1:]))


def _llm(pol, gib):
    ch = replace(CHIPS["100T"], slc_mib=gib * 1024, slc_policy=pol) if gib else CHIPS["100T"]
    return evaluate(Scenario(model="qwen3-8b", chip=ch, mem_id=LP, serving=Serving(batch=8)))


def test_dip_slc_conserves_and_beats_lru():
    base = _llm("pin", 0)
    d0 = sum(st.dram["total"] for st in base.stages)
    prev = None
    for gib in (1, 4, 8, 16, 32):
        r, lru = _llm("dip", gib), _llm("lru", gib)
        st = r.stages[0]
        assert abs(sum(s.dram["total"] + s.dram.get("slc", 0.0) for s in r.stages) - d0) <= 1e-6 * d0
        f = st.mem.slc_frac
        if st.mem.slc_all:
            assert f == 1.0 and abs(r.tpot - lru.tpot) < 1e-12              # fits: as lru
        else:
            assert not lru.stages[0].mem.slc_all and r.tpot < lru.tpot      # lru thrashes, dip keeps a share
            parts = st.dram["slc_parts"]
            for k, v in parts.items():
                if v + st.dram[k] > 0:
                    assert abs(v / (v + st.dram[k]) - f) < 1e-9, k         # every read class hits by the same share
        assert prev is None or r.tpot <= prev
        prev = r.tpot
    assert action_counts(_llm("dip", 8))["counts"]["slc"] > 0
