"""0.93: per-checkpoint / per-hardware GEMM formats from public docs (gpt-oss W4A8 MXFP4×MXFP8 MoE on Blackwell;
DeepSeek-R1-0528-NVFP4-v2 keeps FP8 attention), per-role activation formats, trace-grounded layer overhead constant."""
import math
from accel_dse.core import extval as X
from accel_dse.core.catalog import entries, get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, build_rank_ops
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.model import with_formats
from accel_dse.core.refhw import REF_HW


def _row(rid):
    return next(r for r in X.rows(X.DATA_R4) if r["id"] == rid)


def test_act_fmt_for_default_and_override():
    for e in entries():
        m = get_model(e["id"])
        assert not any(k.startswith("act_") for k, _ in m.formats)        # catalog unchanged → bit-identical
        assert m.act_fmt_for("expert") == m.act_fmt
    m = with_formats(get_model("gpt-oss-20b"), dict(X.MXFP4_BLACKWELL))
    ops = build_rank_ops(m, 0, Phase("prefill", 1, 512, 512))
    ex = [o for o in ops if o.role == "expert"]
    assert ex and all(o.a_fmt == "fp8" and o.w_fmt == "mxfp4" for o in ex)
    assert all(o.a_fmt == "bf16" for o in ops if o.kind == "gemm" and o.role == "attn")


def test_gpt_oss_blackwell_moe_runs_at_fp8_rate():
    r = _row("trt120-b200-gpt-oss-20b-mxfp4-tp1-8192-1024")
    s = X.scenario(r, "catalog")
    assert ("act_expert", "fp8") in s.formats_override
    c = gemm_cost(REF_HW["B200"].chip, "reconf", 8192, 2880, 5760, w_fmt="mxfp4", a_fmt="fp8")
    assert c.exec_fmt == "fp8"
    h = dict(r, hw="H200-SXM")                                            # Hopper: bf16 activations (README)
    assert X._formats(h) == X.MXFP4


def test_dsr1_nvfp4_v2_attention_fp8():
    r = _row("trt120-b200-deepseek-r1-nvfp4-dep4-1024-1024")
    f = dict(X._formats(r))
    assert f["attn"] == "fp8" and f["expert"] == "nvfp4" and f["act"] == "fp8" and f["act_expert"] == "nvfp4"
    m = with_formats(get_model("deepseek-r1"), f)
    ops = [o for o in build_rank_ops(m, 10, Phase("decode", 8, 1, 1024)) if o.kind == "gemm"]
    for o in ops:
        if o.role == "attn":
            assert (o.w_fmt, o.a_fmt) == ("fp8", "fp8")
        if o.role in ("expert", "shared_expert"):
            assert (o.w_fmt, o.a_fmt) == ("nvfp4", "nvfp4")


def test_fmt_fix_only_moves_r4_and_improves_it():
    lr = lambda rows: [math.log(X.predict(r, "catalog_kernel")["pred"] / r["value"]) for r in rows]
    R = X.rows(X.DATA_R4)
    new = lr(R)
    X.FMT_FIX = False
    try:
        old = lr(R)
        others = [lr(X.rows(p)) for p in (X.DATA, X.DATA_MOE, X.DATA_R3)]
    finally:
        X.FMT_FIX = True
    assert others == [lr(X.rows(p)) for p in (X.DATA, X.DATA_MOE, X.DATA_R3)]
    e = lambda x: 100 * sum(map(abs, x)) / len(x)
    assert e(new) < e(old) and e(new) < 28


def test_trace_layer_constant():
    assert abs(X.TRACE_LAYER_US - 62.30) < 0.01


def test_attn_sched_default_off_and_candidate_bands():
    from accel_dse.core.hardware import CHIPS
    assert all(c.attn_sched == "" for c in CHIPS.values()) and all(h.chip.attn_sched == "" for h in REF_HW.values())
    v = X.candidate_variant()
    g = lambda rows, var: [math.log(X.predict(r, var)["pred"] / r["value"]) for r in rows]
    e = lambda x: 100 * sum(map(abs, x)) / len(x)
    R4 = X.rows(X.DATA_R4)
    # hold-out: 26.7 → 19.2 % (0.93).  0.172: LLM attention orientation choice brings catalog_kernel itself to 16.8 %
    # (old 26.6 %), so the attention-cta part of the candidate no longer buys anything on R4
    assert e(g(R4, v)) < 20 and e(g(R4, "catalog_kernel")) < 18
    lat = [r for r in X.rows(X.DATA_R3) if r["metric"] == "static_latency_s"]
    assert e(g(lat, v)) < 35                                                # 118 → 33 %
    one = [r for r in X.rows(X.DATA_MOE) if r["metric"] == "static_tok_s_total"]
    assert len(one) == 1 and e(g(one, v)) > e(g(one, "catalog_kernel"))    # the group that blocks adoption
