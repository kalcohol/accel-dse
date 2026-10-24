"""0.365: GDN in_proj_ba is the bf16 GEMM inside the attn role, not the role average.

The split is applied only when the release buckets match the builder parameter for
parameter. A non-empty deepstack list on a Qwen3.5 tower is still refused (0.366
models that list only on qwen3_vl / qwen3_vl_moe).
"""
from __future__ import annotations

from accel_dse.core.catalog import entries, get_model
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import load_release
from accel_dse.core.vision import vision_spec


def _weights(m):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0).weights


def _avg_bits(role):
    p = sum(v["params"] for v in role.values())
    b = sum(v["bytes"] for v in role.values())
    return 8.0 * b / p


def test_fp8_gdn_in_proj_ba_is_bf16():
    for mid, sib_id in (("qwen3-coder-next-fp8", "qwen3-next-80b-a3b"),
                        ("qwen3.8-27b-fp8", "qwen3.8-27b")):
        m = get_model(mid)
        rel = load_release(m.hf_id)
        role = rel["roles"]["attn"]
        lo = 8.0 * role["fp8"]["bytes"] / role["fp8"]["params"]
        assert m.fmt("attn").fmt == "fp8" and abs(m.fmt("attn").bits - lo) < 1e-12
        assert abs(m.fmt("attn").bits - _avg_bits(role)) > 0.02
        assert m.fmt("lin_ba").fmt == "bf16" and m.fmt("lin_ba").bits == 16.0
        assert m.act_fmt_for("lin_ba") == "fp8" and m.act_fmt == "fp8"
        bas = [l for L in m.layers for l in L.attn_linears if l.name == "lin_ba"]
        assert bas and all(l.role == "lin_ba" for l in bas)
        assert all(l.role != "lin_ba" for L in m.mtp_layers for l in L.attn_linears)
        li = next(i for i, L in enumerate(m.layers) if any(l.name == "lin_ba" for l in L.attn_linears))
        ops = build_rank_ops(m, li, Phase("decode", 1, 1, 16), Shard())
        ba = next(o for o in ops if o.name == "attn.lin_ba")
        qkv = next(o for o in ops if o.name == "attn.lin_qkv")
        assert ba.w_params == m.hidden * 2 * m.layers[li].core.n_state_heads
        assert ba.w_fmt == "bf16" and ba.w_bits == 16.0 and ba.a_fmt == "fp8"
        assert ba.w_params * ba.w_bits / 8 == ba.w_params * 2
        assert qkv.w_fmt == "fp8" and abs(qkv.w_bits - lo) < 1e-12 and qkv.a_fmt == "fp8"
        # same residual as the bf16 sibling: LLM storage omits the final RMSNorm
        sib = get_model(sib_id)
        assert "lin_ba" not in dict(sib.formats) and sib.fmt("attn").bits == 16.0
        assert _weights(sib) - load_release(sib.hf_id)["bytes_llm"] == -sib.hidden * 2
        assert abs((_weights(m) - rel["bytes_llm"]) - (-m.hidden * 2)) < 1.0
        assert m.param_check()["rel_err"] == 0.0


def test_mixed_attn_without_the_split_keeps_the_average():
    """Ornith keeps the whole linear-attention stack bf16. Qwen3.5-9B mixes a few fp32 tensors in."""
    for mid in ("ornith-1.5-35b-a3b-fp8", "qwen3.5-9b"):
        m = get_model(mid)
        role = load_release(m.hf_id)["roles"]["attn"]
        assert "lin_ba" not in dict(m.formats)
        assert all(l.role != "lin_ba" for L in m.layers for l in L.attn_linears)
        assert abs(m.fmt("attn").bits - _avg_bits(role)) < 1e-9


def test_catalog_split_is_only_those_two():
    got = sorted(e["id"] for e in entries() if "lin_ba" in dict(get_model(e["id"]).formats))
    assert got == ["qwen3-coder-next-fp8", "qwen3.8-27b-fp8"]


def test_qwen3_5_deepstack_stays_refused():
    """Non-empty deepstack_visual_indexes on a Qwen3.5 tower is not Qwen3VLVisionModel."""
    m = get_model("qwen3.8-27b-fp8")
    cfg = load_release(m.hf_id)["config"]
    base = vision_spec(m.hf_id, cfg)
    assert base is not None and base.deepstack == 0 and len(base.post) == 2
    # same tower, empty deepstack: qwen3_vl is the Qwen ViT path with no extra merger
    vl = {**cfg, "vision_config": {**cfg["vision_config"], "model_type": "qwen3_vl"}}
    got = vision_spec(m.hf_id, vl)
    assert got is not None and got.params == base.params and got.deepstack == 0 and got.label == "Qwen ViT"
    deep = {**cfg, "vision_config": {**cfg["vision_config"], "deepstack_visual_indexes": [8, 16, 24]}}
    assert vision_spec(m.hf_id, deep) is None
    bad = {**cfg, "vision_config": {**cfg["vision_config"], "model_type": "qwen3_vl",
                                    "deepstack_visual_indexes": [8, 16, 99]}}
    assert vision_spec(m.hf_id, bad) is None
