"""0.361: RWKV-8 Heron pile rc00 (pure) and rc00-hybrid.

Shapes and dtypes are the BlinkDL .pth headers. The time-mix is the existing
RWKV-7 recurrence. DeepEmbed is a per-token row. DeepEmbedAttention follows
rwkv_v8_rc00_hybrid_demo.py forward_seq (fp16 kk/vv cache, full-cache up
projections, one causal head).
"""
from __future__ import annotations

from accel_dse.core import catalog as C
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, build_rank_ops
from accel_dse.core.memplan import cache_new_bytes, stage_storage
from accel_dse.core.parallel import Shard
from accel_dse.core.scenario import Scenario

PURE = "rwkv8-pile"
HYB = "rwkv8-pile-hybrid"
H = 768
V = 50304


def _s(m, **kw):
    s = Scenario().replace("model", m)
    for k, v in kw.items():
        s = s.replace(k.replace("__", "."), v)
    return s


def _fmt(m, role):
    return dict(m.formats)[role].fmt


def test_rwkv8_params_exact():
    pure, hyb = C.get_model(PURE), C.get_model(HYB)
    assert pure.param_check()["rel_err"] == 0.0 and pure.params() == 631_289_856
    assert hyb.param_check()["rel_err"] == 0.0 and hyb.params() == 1_252_796_928
    for m in (pure, hyb):
        assert m.coverage == "full" and not m.coverage_reasons
        assert m.n_layers == 12 and m.hidden == H and m.vocab == V and not m.tie_embeddings
        assert m.layers[1].ffn.d_ff == 3072 and not m.layers[1].ffn.gated
        assert m.layers[1].deepemb == H
        c = m.layers[1].core
        assert c.kind == "linear" and c.lin == "rwkv7" and c.n_state_heads == 12
        assert c.state_dk == c.state_dv == 64 and c.lin_chunk == 64
        assert "rwkv_v1" not in {l.name for l in m.layers[0].attn_linears}
        assert _fmt(m, "attn") == "bf16" and _fmt(m, "mlp") == "bf16" and _fmt(m, "lm_head") == "bf16"
    assert pure.lookup_params == 12 * V * H and pure.lookup_bits == 16.0 and _fmt(pure, "embed") == "bf16"
    assert not pure.layers[0].core.dea_q
    assert pure.layers[0].misc_params - pure.layers[1].misc_params == 2 * H + 2 * H * 32
    dea = hyb.layers[0].core
    assert (dea.dea_q, dea.dea_k, dea.dea_kq, dea.dea_v, dea.dea_vd, dea.dea_k_emb, dea.dea_v_emb) == (
        256, 32, 256, 32, 768, 256, 768)
    assert hyb.lookup_params == 12 * V * (H + 256 + H) and _fmt(hyb, "embed") == "fp16"
    assert hyb.layers[0].misc_params - hyb.layers[1].misc_params == 2 * H * 32
    assert hyb.layers[0].misc_params == pure.layers[0].misc_params - 2 * H + 2 * (256 + 256 + H)
    names = {l.name: (l.k, l.n, l.split) for l in hyb.layers[1].attn_linears}
    assert names["dea_qq"] == (H, 256, "rep") and names["dea_kkup"] == (32, 256, "rep")
    assert names["dea_vvup"] == (32, H, "rep")


def test_rwkv8_catalog_keys():
    ents = {e["id"]: e for e in C.entries()}
    assert ents[PURE]["release"] == "BlinkDL/rwkv-8-pile:rc00"
    assert ents[HYB]["release"] == "BlinkDL/rwkv-8-pile:rc00-hybrid"
    assert ents[PURE]["hf_id"] == ents[HYB]["hf_id"] == "BlinkDL/rwkv-8-pile"
    assert ents[PURE]["provider"] == "rwkv" and ents[PURE]["family"] == "RWKV-8 Heron"
    assert ents[HYB]["family"] == "RWKV-8 Heron" and ents[PURE]["listed"] and ents[HYB]["listed"]
    assert C.get_model("BlinkDL/rwkv-8-pile").id == PURE
    assert C.get_model("BlinkDL/rwkv-8-pile:rc00-hybrid").id == HYB
    ids = [m["id"] for m in C.list_models()]
    assert ids.index(HYB) < ids.index(PURE) < ids.index("rwkv7-g1k-13.3b")
    assert C.get_model("microsoft-decision-1").coverage == "proxy"
    assert len([e for e in C.entries() if e["id"].startswith("startlux-decision")]) == 14


def test_pure_decode_is_context_invariant():
    rs = [evaluate(_s(PURE, serving__ctx=c, serving__batch=4)) for c in (4096, 131072)]
    assert len({round(r.tpot, 12) for r in rs}) == 1
    assert all(r.stages[0].mem.kv_total == 0 and r.stages[0].dram["kv_read"] == 0 for r in rs)
    m = C.get_model(PURE)
    ph = Phase("decode", 1, 1, 4096)
    de = next(o for o in build_rank_ops(m, 3, ph) if o.name == "deepemb")
    assert de.kind == "lookup" and de.kv_read == H * 2 and de.vec == H
    assert not any(o.name.startswith("attn.dea_") or o.name.startswith("dea_") for o in build_rank_ops(m, 3, ph))


def test_hybrid_cache_and_schedule():
    m = C.get_model(HYB)
    ctx, q, b = 4096, 1, 2
    st = stage_storage(m, 0, m.n_layers, True, True, Shard(), ctx)
    per = (12 * 64 * 2 + 4) * ctx
    assert st.kv_per_seq == per
    assert cache_new_bytes(m, ctx, 0) - cache_new_bytes(m, 0, 0) == per
    r = evaluate(_s(HYB, serving__ctx=ctx, serving__batch=b))
    # Decode plans the cache after the new token is appended (ctx + q, q = 1).
    assert r.stages[0].mem.kv_total == (12 * 64 * 2 + 4) * (ctx + 1) * b
    ph = Phase("decode", b, q, ctx)
    ops = build_rank_ops(m, 0, ph)
    up = next(o for o in ops if o.name == "attn.dea_kkup")
    assert up.m == b * (ctx + q) and up.k == 32 and up.n == 256
    vv = next(o for o in ops if o.name == "attn.dea_vvup")
    assert vv.m == b * (ctx + q) and (up.kv_read + vv.kv_read) == b * (ctx + q) * 64 * 2
    de = next(o for o in ops if o.name == "deepemb")
    assert de.kv_read == b * q * H * 2
    qk = next(o for o in ops if o.name == "dea_qk")
    assert qk.m == q and qk.k == 256 and qk.n == ctx + q and qk.count == b and qk.causal == 1.0
    ids = [o for li in range(m.n_layers) for o in build_rank_ops(m, li, ph) if o.name == "dea_ids"]
    assert len(ids) == 1 and ids[0].kind == "vector" and ids[0].kv_read == b * (ctx + q) * 4
    pre = build_rank_ops(m, 1, Phase("prefill", 1, 8, 0))
    assert next(o for o in pre if o.name == "attn.dea_kkup").m == 8
    assert abs(next(o for o in pre if o.name == "dea_qk").causal - 0.5) < 1e-12
    t = [evaluate(_s(HYB, serving__ctx=c)).tpot for c in (4096, 131072)]
    assert t[1] > t[0]
