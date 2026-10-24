"""0.364: block-FP8 MTP eh_proj is the bf16 remainder of the mtp role, not the role average."""
from __future__ import annotations

from accel_dse.core.catalog import get_model
from accel_dse.core.ir import Phase, Shard, _mtp_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import load_release


def _mtp_bytes(m):
    n = len(m.mtp_layers)
    full = stage_storage(m, 0, m.n_layers, True, True, Shard(), 0, n_mtp=n)
    base = stage_storage(m, 0, m.n_layers, True, True, Shard(), 0, n_mtp=0)
    return full.weights - base.weights


def test_qwen38_fp8_eh_proj_is_bf16():
    m = get_model("qwen3.8-27b-fp8")
    rel = load_release(m.hf_id)
    role = rel["roles"]["mtp"]
    assert role["bf16"]["params"] == m.mtp_extra_params + m.mtp_layers[0].misc_params
    assert m.fmt("mtp").fmt == "bf16" and m.fmt("mtp").bits == 16.0
    # whole-role average stays fp8-named; the fix must not report that as the eh_proj format
    avg = 8.0 * sum(v["bytes"] for v in role.values()) / sum(v["params"] for v in role.values())
    assert avg < 9.0 and m.fmt("mtp").bits != avg
    eh = next(o for o in _mtp_ops(m, Phase(kind="decode", batch=1, q=1, ctx=16), Shard()) if o.name == "mtp.eh_proj")
    assert eh.w_fmt == "bf16" and eh.w_bits == 16.0 and eh.a_fmt == "fp8"
    assert eh.w_params == 2 * m.hidden * m.hidden
    assert eh.w_params * eh.w_bits / 8 == 2 * m.hidden * m.hidden * 2
    # leftover is the main-stack attn scale smear on the MTP attention GEMMs, not the old ~46 MB
    assert abs(_mtp_bytes(m) - rel["bytes_mtp"]) < 1_000_000


def test_mixed_mtp_decoder_keeps_role_average():
    """DeepSeek-V3 bf16 weights sit inside the MTP block. The exact split does not hold."""
    m = get_model("deepseek-v3")
    role = load_release(m.hf_id)["roles"]["mtp"]
    p = sum(v["params"] for v in role.values())
    b = sum(v["bytes"] for v in role.values())
    assert m.fmt("mtp").fmt == "fp8"
    assert m.fmt("mtp").bits == 8.0 * b / p


def test_bf16_mtp_and_pure_fp8_unchanged():
    flash = get_model("glm-4.7-flash")
    assert flash.fmt("mtp").fmt == "bf16" and abs(flash.fmt("mtp").bits - 16.0) < 1e-6
    hy = get_model("hy3")
    assert hy.fmt("mtp").fmt == "bf16" and abs(hy.fmt("mtp").bits - 16.0) < 1e-6
    ultra = get_model("nemotron-3-ultra-550b-a55b-bf16")
    assert ultra.fmt("mtp").fmt == "bf16" and abs(ultra.fmt("mtp").bits - 16.0) < 1e-6
    for mid in ("qwen3-14b-fp8", "qwen3-coder-480b-a35b-fp8", "qwen3-coder-next-fp8"):
        m = get_model(mid)
        assert m.fmt("attn").fmt == "fp8" and m.act_fmt == "fp8"
        assert "mtp" not in dict(m.formats) or m.fmt("mtp").fmt == "fp8"
    sib = get_model("qwen3.8-27b")
    assert sib.fmt("mtp").fmt == "bf16" and sib.fmt("mtp").bits == 16.0
    assert abs(_mtp_bytes(sib) - load_release(sib.hf_id)["bytes_mtp"]) < 100
