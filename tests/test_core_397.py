"""0.397: Paddle U16 on ERNIE-4.5-300B-A47B-Base is bf16; non-PT VL twins match PT.

PaddlePaddle/Paddle flex_checkpoint/dcp/utils.py maps safetensors dtype U16 to
bfloat16 (2 bytes per element). Fused qkv_proj and up_gate_proj peel onto the
existing q/k/v and gate_up GEMMs. The three non-PT VL repos match the PT
headers. 0.396 Thinking still selects moe_k experts, not a literal 2.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import from_release, load_release
from accel_dse.core.vision import image_grid, video_pre, vision_ops

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "accel_dse" / "data" / "releases"

BASE_PARAMS = 299_484_163_264
BASE_BYTES = 599_021_810_432
BASE_TENSORS = 6855
PT_TENSORS = 10_230
H = 8192
HEADS = 64
KV = 8
HD = 128
DENSE = 28_672
EXPERT = 3_584
E = 64
K = 8
VOCAB = 103_424
ROUTER_PARAMS = 51 * (H * E + E)

VL28_LANG = 28_624_635_776
VL28_VISION = 774_658_560
VL28_BYTES = 58_816_290_304
VL28_TENSORS = 11_105
VL424_LANG = 422_722_779_520
VL424_VISION = 803_505_664
VL424_BYTES = 847_159_538_176
VL424_TENSORS = 20_475
VL424_INDEX = 847_161_324_032
THINK_HEAD = 103_424 * 2560


def _rows(op) -> float:
    return op.count * op.m * op.share


def _raise_gate(hf: str, rel: dict, gate: str) -> None:
    cfg = copy.deepcopy(rel["config"])
    cfg["moe_gate"] = gate
    try:
        from_release("probe", hf, rel, cfg)
    except ValueError as exc:
        assert "TopKGate" in str(exc)
        return
    raise AssertionError(f"{hf} accepted moe_gate={gate!r}")


def _gemm(model, layer: int, name: str, phase: Phase):
    ops = build_rank_ops(model, layer, phase, Shard())
    return next(op for op in ops if op.name == name)


def test_ernie_300b_base_u16_peels_onto_existing_gemms():
    m = get_model("ernie-4.5-300b-a47b-base")
    pt = get_model("ernie-4.5-300b-a47b-base-pt")
    rel = load_release(m.hf_id)
    rpt = load_release(pt.hf_id)
    cfg = rel["config"]
    assert cfg["moe_gate"] == "topk" and cfg["moe_k"] == K
    assert cfg["dtype"] == "bfloat16" and rel["torch_dtype"] == "bfloat16"
    assert "moe_layer_end_index" not in cfg and "num_nextn_predict_layers" not in cfg
    assert cfg.get("moe_num_shared_experts") in (None, 0)
    assert cfg["moe_capacity"] == [64, 64, 64] and cfg["moe_layer_start_index"] == 3
    assert rel["storage_alias"] == {"U16": "bf16"}
    assert rel["fused_linears"] == ["qkv_proj", "up_gate_proj"]
    assert "storage_alias" not in rpt and "fused_linears" not in rpt
    assert rel["roles"] == rpt["roles"]
    assert rel["n_tensors"] == BASE_TENSORS and rpt["n_tensors"] == PT_TENSORS
    # Three dense gate+up pairs, 51*64 expert pairs, and 54 q/k/v triples each lose two tensors.
    assert PT_TENSORS - (3 + 51 * E + 54 * 2) == BASE_TENSORS
    assert m.param_check()["rel_err"] == 0.0 and pt.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == pt.params() == rel["params_llm"] == BASE_PARAMS
    assert m.mtp_params() == rel["params_mtp"] == 0 and m.mtp_layers == ()
    assert m.vision is None and rel["bytes_total"] == rel["bytes_llm"] == BASE_BYTES
    router = rel["roles"]["router"]["fp32"]
    assert router["params"] == ROUTER_PARAMS and router["bytes"] == 4 * ROUTER_PARAMS
    for role, fmts in rel["roles"].items():
        if role == "router":
            assert set(fmts) == {"fp32"}
            continue
        assert set(fmts) == {"bf16"}
        assert fmts["bf16"]["bytes"] == 2 * fmts["bf16"]["params"]
        assert "u16" not in fmts
    assert rel["bytes_total"] == 2 * BASE_PARAMS + 2 * ROUTER_PARAMS
    assert m.n_layers == 54 and m.hidden == H and m.vocab == VOCAB and not m.tie_embeddings
    assert m.coverage == "full" and m.domain == "llm"
    assert m.fmt("attn").fmt == "bf16" and m.fmt("expert").fmt == "bf16"
    assert m.fmt("mlp").fmt == "bf16" and m.fmt("router").fmt == "fp32"
    assert m.fmt("router").bits == 32.0
    qkv_n = (HEADS + 2 * KV) * HD
    assert qkv_n == 10_240
    for i in range(3):
        assert m.layers[i].ffn.kind == "dense" and m.layers[i].ffn.d_ff == DENSE
        assert m.layers[i].misc_params == 2 * H
        ups = [lin for lin in m.layers[i].ffn_linears if lin.name == "gate_up"]
        assert [(lin.k, lin.n) for lin in ups] == [(H, 2 * DENSE)]
    for i in range(3, 54):
        ffn = m.layers[i].ffn
        assert ffn.kind == "moe" and ffn.top_k == K and ffn.top_k != 2
        assert ffn.n_experts == E and ffn.d_expert == EXPERT and ffn.n_shared == 0
        assert m.layers[i].misc_params == 2 * H + E
    q, k, v, o = m.layers[0].attn_linears
    assert (q.name, k.name, v.name, o.name) == ("q", "k", "v", "o")
    assert q.n + k.n + v.n == qkv_n and (q.k, k.k, v.k, o.k) == (H, H, H, H)
    assert (o.n, o.k) == (H, H)
    ph = Phase("decode", 1, 1, 16)
    dense = build_rank_ops(m, 0, ph, Shard())
    assert [op.name for op in dense if op.kind == "gemm"] == [
        "attn.qkv", "attn.o", "mlp.gate_up", "mlp.down"]
    qkv = next(op for op in dense if op.name == "attn.qkv")
    gate = next(op for op in dense if op.name == "mlp.gate_up")
    down = next(op for op in dense if op.name == "mlp.down")
    assert (qkv.k, qkv.n, qkv.w_fmt, qkv.w_bits) == (H, qkv_n, "bf16", 16.0)
    assert (gate.k, gate.n) == (H, 2 * DENSE) and (down.k, down.n) == (DENSE, H)
    assert not any(op.name in ("qkv_proj", "up_gate_proj") or "u16" in op.name for op in dense)
    expert = _gemm(m, 3, "expert.gate_up", ph)
    edown = _gemm(m, 3, "expert.down", ph)
    router_op = _gemm(m, 3, "moe.router", ph)
    assert (expert.k, expert.n, expert.w_fmt) == (H, 2 * EXPERT, "bf16")
    assert _rows(expert) == K and (edown.k, edown.n) == (EXPERT, H)
    assert (router_op.k, router_op.n, router_op.w_fmt, router_op.w_bits) == (H, E, "fp32", 32.0)
    text = " ".join(m.notes)
    assert "_SAFETENSORS_DTYPE_MAPPING" in text and "U16" in text and "gate_up" in text
    assert "不另写 U16 GEMM" in text and "不另写融合 QKV" in text
    assert "U16" not in " ".join(pt.notes)
    assert pt.layers[3].ffn.top_k == K
    shapes = [(L.ffn.kind, L.ffn.top_k, L.ffn.d_expert, L.ffn.d_ff) for L in m.layers]
    assert shapes == [(L.ffn.kind, L.ffn.top_k, L.ffn.d_expert, L.ffn.d_ff) for L in pt.layers]
    row = {e["id"]: e for e in list_models()}["ernie-4.5-300b-a47b-base"]
    assert row["family"] == "ERNIE-4.5" and row["domain"] == "llm" and row["param_err"] == 0.0
    aliases = [
        json.loads(p.read_text()).get("storage_alias")
        for p in RELEASES.glob("*.json")
    ]
    assert aliases.count({"U16": "bf16"}) == 1

    swapped = copy.deepcopy(cfg)
    swapped["moe_gate"] = "top2_fused"
    same = from_release(m.id, m.hf_id, rel, swapped)
    assert same.params() == m.params() and same.layers[3].ffn.top_k == K
    assert _rows(_gemm(same, 3, "expert.gate_up", ph)) == K
    assert (_gemm(same, 3, "expert.gate_up", ph).k, _gemm(same, 3, "expert.gate_up", ph).n) == (H, 2 * EXPERT)
    three = copy.deepcopy(cfg)
    three["moe_gate"] = "top2_fused"
    three["moe_k"] = 3
    alt = from_release(m.id, m.hf_id, rel, three)
    assert alt.layers[3].ffn.top_k == 3 and alt.layers[3].ffn.top_k != 2
    assert alt.params() == m.params()
    alt_up = _gemm(alt, 3, "expert.gate_up", ph)
    assert _rows(alt_up) == 3 and (alt_up.k, alt_up.n) == (H, 2 * EXPERT)
    cap = copy.deepcopy(cfg)
    cap["moe_capacity"] = [1, 1, 1]
    held = from_release(m.id, m.hf_id, rel, cap)
    held_up = _gemm(held, 3, "expert.gate_up", ph)
    assert held.params() == m.params() and held.layers[3].ffn.top_k == K
    assert (held_up.k, held_up.n, held_up.count, held_up.m, held_up.share) == (
        expert.k, expert.n, expert.count, expert.m, expert.share)
    bare_cfg = copy.deepcopy(cfg)
    bare_cfg.pop("moe_gate")
    bare = from_release(m.id, m.hf_id, rel, bare_cfg)
    assert bare.layers[3].ffn.top_k == K and bare.params() == m.params()
    assert "不是 2" not in " ".join(bare.notes) and "U16" in " ".join(bare.notes)
    stripped = dict(rel)
    stripped.pop("storage_alias")
    stripped.pop("fused_linears")
    quiet = from_release(m.id, m.hf_id, stripped, cfg)
    assert quiet.params() == m.params() and quiet.layers[0].ffn_linears[0].n == 2 * DENSE
    assert "U16" not in " ".join(quiet.notes)
    for bad in ("top2", "gshard", ""):
        _raise_gate(m.hf_id, rel, bad)


def test_ernie_vl_non_pt_matches_the_pt_bill():
    pairs = (
        ("ernie-4.5-vl-28b-a3b", "ernie-4.5-vl-28b-a3b-pt", VL28_LANG, VL28_VISION,
         VL28_BYTES, VL28_TENSORS, 6, True),
        ("ernie-4.5-vl-28b-a3b-base", "ernie-4.5-vl-28b-a3b-pt", VL28_LANG, VL28_VISION,
         VL28_BYTES, VL28_TENSORS, 6, True),
        ("ernie-4.5-vl-424b-a47b", "ernie-4.5-vl-424b-a47b-pt", VL424_LANG, VL424_VISION,
         VL424_BYTES, VL424_TENSORS, 8, False),
    )
    listed = {e["id"]: e for e in list_models()}
    for mid, ptid, lang, vis, nbytes, ntensors, k, tie in pairs:
        m = get_model(mid)
        pt = get_model(ptid)
        rel = load_release(m.hf_id)
        rpt = load_release(pt.hf_id)
        assert rel["config"]["moe_gate"] == "topk" and rel["config"]["moe_k"] == k
        assert "storage_alias" not in rel and "fused_linears" not in rel
        assert rel["roles"] == rpt["roles"]
        assert rel["n_tensors"] == rpt["n_tensors"] == ntensors
        assert rel["bytes_total"] == rpt["bytes_total"] == nbytes
        assert rel["bytes_total"] != VL424_INDEX
        assert m.param_check()["rel_err"] == 0.0 and pt.param_check()["rel_err"] == 0.0
        assert m.params() == pt.params() == lang and m.vision_params == pt.vision_params == vis
        assert m.mtp_params() == 0 and m.tie_embeddings is tie and pt.tie_embeddings is tie
        assert m.layers[0].ffn.kind == "dense"
        moe = next(i for i, layer in enumerate(m.layers) if layer.ffn.kind == "moe")
        assert m.layers[moe].ffn.top_k == k and pt.layers[moe].ffn.top_k == k
        assert m.layers[moe].ffn.n_experts_b == pt.layers[moe].ffn.n_experts_b
        assert m.layers[moe].ffn.d_expert == pt.layers[moe].ffn.d_expert
        assert m.layers[moe].ffn.d_expert_b == pt.layers[moe].ffn.d_expert_b
        assert video_pre(m) == video_pre(pt)
        assert video_pre(m)["ernie_temporal"] == 2
        g = image_grid(m.vision, 1024, 1024)
        sig = [(op.name, op.w_fmt, op.w_bits, op.m, op.k, op.n) for op in vision_ops(m.vision, g, 1)]
        pt_sig = [(op.name, op.w_fmt, op.w_bits, op.m, op.k, op.n) for op in vision_ops(pt.vision, g, 1)]
        assert sig == pt_sig and sig
        assert set(rel["roles"]["vision"]) == set(rpt["roles"]["vision"]) == {"fp16", "bf16"}
        assert listed[mid]["family"] == "ERNIE-4.5-VL" and listed[mid]["domain"] == "vlm"
        assert listed[mid]["param_err"] == 0.0
        ph = Phase("decode", 1, 1, 16)
        up = _gemm(m, moe, "expert.gate_up", ph)
        pt_up = _gemm(pt, moe, "expert.gate_up", ph)
        assert (up.k, up.n, up.count, up.m) == (pt_up.k, pt_up.n, pt_up.count, pt_up.m)
        assert _rows(up) == k


def test_thinking_moe_k_is_not_literal_two():
    m = get_model("ernie-4.5-21b-a3b-thinking")
    rel = load_release(m.hf_id)
    ph = Phase("decode", 1, 1, 16)
    base = _gemm(m, 1, "expert.gate_up", ph)
    assert rel["config"]["moe_gate"] == "top2_fused" and m.layers[1].ffn.top_k == 6
    assert _rows(base) == 6 and (base.k, base.n) == (2560, 2 * 1536)
    topk = copy.deepcopy(rel["config"])
    topk["moe_gate"] = "topk"
    swapped = from_release(m.id, m.hf_id, rel, topk)
    assert swapped.params() == m.params() and swapped.layers[1].ffn.top_k == 6
    assert _rows(_gemm(swapped, 1, "expert.gate_up", ph)) == 6
    three = copy.deepcopy(rel["config"])
    three["moe_gate"] = "top2_fused"
    three["moe_k"] = 3
    alt = from_release(m.id, m.hf_id, rel, three)
    alt_up = _gemm(alt, 1, "expert.gate_up", ph)
    assert alt.layers[1].ffn.top_k == 3 and alt.layers[1].ffn.top_k != 2
    assert alt.params() == m.params()
    assert _rows(alt_up) == 3 and (alt_up.k, alt_up.n) == (base.k, base.n)
    cap = copy.deepcopy(rel["config"])
    cap["moe_capacity"] = [1, 1, 1]
    held = from_release(m.id, m.hf_id, rel, cap)
    held_up = _gemm(held, 1, "expert.gate_up", ph)
    assert held.params() == m.params()
    assert (held_up.k, held_up.n, held_up.count, held_up.m, held_up.share) == (
        base.k, base.n, base.count, base.m, base.share)

    vl = get_model("ernie-4.5-vl-28b-a3b-thinking")
    vrel = load_release(vl.hf_id)
    assert vrel["config"]["moe_gate"] == "topk" and vl.tie_embeddings
    assert vl.params() == vrel["params_llm"] - THINK_HEAD
    assert "tied lm_head" in " ".join(vl.notes)
    assert set(vrel["roles"]["vision"]) == {"bf16"}
    g = image_grid(vl.vision, 1024, 1024)
    gemms = [op for op in vision_ops(vl.vision, g, 1) if op.kind == "gemm"]
    assert gemms and all(op.w_fmt == "bf16" and op.w_bits == 16.0 for op in gemms)
    both = Phase("prefill", 1, 4, mm=2)
    up = _gemm(vl, 1, "expert.gate_up", both)
    up_b = _gemm(vl, 1, "expert_b.gate_up", both)
    assert (up.k, up.n) == (2560, 2 * 1536) and _rows(up) == 12
    assert (up_b.k, up_b.n) == (2560, 2 * 512) and _rows(up_b) == 12
    v3 = copy.deepcopy(vrel["config"])
    v3["moe_k"] = 3
    narrow = from_release(vl.id, vl.hf_id, vrel, v3)
    assert narrow.layers[1].ffn.top_k == 3 and narrow.layers[1].ffn.top_k != 2
    assert narrow.params() == vl.params() and narrow.vision_params == vl.vision_params
    n_up = _gemm(narrow, 1, "expert.gate_up", both)
    n_b = _gemm(narrow, 1, "expert_b.gate_up", both)
    assert _rows(n_up) == 6 and _rows(n_b) == 6
    assert (n_up.k, n_up.n) == (up.k, up.n) and (n_b.k, n_b.n) == (up_b.k, up_b.n)
    vcap = copy.deepcopy(vrel["config"])
    vcap["moe_capacity"] = [1, 1, 1]
    vheld = from_release(vl.id, vl.hf_id, vrel, vcap)
    h_up = _gemm(vheld, 1, "expert.gate_up", both)
    h_b = _gemm(vheld, 1, "expert_b.gate_up", both)
    assert vheld.params() == vl.params()
    assert (h_up.k, h_up.n, h_up.count, h_up.m) == (up.k, up.n, up.count, up.m)
    assert (h_b.k, h_b.n, h_b.count, h_b.m) == (up_b.k, up_b.n, up_b.count, up_b.m)
    for bad in ("top2", "top2_fused", "gshard"):
        _raise_gate(vl.hf_id, vrel, bad)
