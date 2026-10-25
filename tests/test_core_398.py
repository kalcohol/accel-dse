"""0.398: LongCat-Flash MTP follows the public vLLM draft cache.

HF modeling_longcat_flash.py still ignores model.mtp.*. vLLM LongCatFlashMTP
loads one dense block. After the model_type rewrite to longcat_flash_mtp,
use_mla is false, so attention is DeepseekV2Attention: kv_b expands, then V
is padded from v_head_dim to qk_head_dim in the KV cache and sliced back
before o_proj. Released GEMMs stay at v_head_dim. eh_proj is 2h -> h.
The private embedding and mtp.norm stay in the remainder.

Also locks the 0.397 U16 width check and the 424B index overcount.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, build_rank_ops, _mtp_ops
from accel_dse.core.model import from_release, load_release

H = 6144
HEADS = 64
QK = 192
V = 128
Q_LORA = 1536
KV_LORA = 512
D_FF = 12288
VOCAB = 131072
PARAMS = 560_664_980_480
MTP = 1_197_899_776
VL424_BYTES = 847_159_538_176
VL424_INDEX = 847_161_324_032


def _sig(ops):
    return [
        (op.name, op.kind, op.m, op.k, op.n, op.w_params, op.w_bits, op.w_fmt, op.kv_read, op.kv_write)
        for op in ops
    ]


def _summarize():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_release.py"
    spec = importlib.util.spec_from_file_location("summarize_release_398", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_longcat_flash_mtp_pads_v_in_the_draft_cache():
    m = get_model("longcat-flash-chat")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0
    assert m.param_check()["mtp_ours"] == m.param_check()["mtp_release"] == MTP
    assert m.params() == m.release_params == rel["params_llm"] == PARAMS
    assert m.mtp_params() == rel["params_mtp"] == MTP
    assert m.lookup_params == 0 and m.vision_params == 0 and m.coverage == "full"
    assert len(m.layers) == 56 and len(m.mtp_layers) == 1
    extra = VOCAB * H + 2 * H * H + H
    assert m.mtp_extra_params == extra
    assert set(rel["roles"]["mtp"]) == {"bf16"}
    assert rel["roles"]["mtp"]["bf16"]["params"] == MTP
    assert rel["roles"]["mtp"]["bf16"]["bytes"] == 2 * MTP
    assert m.fmt("mtp").fmt == "bf16" and m.fmt("mtp").bits == 16.0

    mt = m.mtp_layers[0]
    dense = m.layers[1]
    main = m.layers[0]
    assert mt.ffn.kind == "dense" and mt.ffn.d_ff == D_FF and mt.ffn.n_experts == 0
    assert not any(lin.name == "router" for lin in mt.ffn_linears)
    assert [(lin.name, lin.k, lin.n) for lin in mt.attn_linears] == [
        (lin.name, lin.k, lin.n) for lin in dense.attn_linears]
    assert [(lin.name, lin.k, lin.n) for lin in mt.ffn_linears] == [
        (lin.name, lin.k, lin.n) for lin in dense.ffn_linears]
    named = {lin.name: lin for lin in mt.attn_linears}
    assert (named["q_a"].k, named["q_a"].n) == (H, Q_LORA)
    assert (named["q"].k, named["q"].n) == (Q_LORA, HEADS * QK)
    assert (named["kv_a"].k, named["kv_a"].n) == (H, KV_LORA + 64)
    assert (named["kv_up"].k, named["kv_up"].n) == (KV_LORA, HEADS * (V + V))
    assert (named["o"].k, named["o"].n) == (HEADS * V, H)
    ff = {lin.name: lin for lin in mt.ffn_linears}
    assert (ff["gate_up"].k, ff["gate_up"].n) == (H, 2 * D_FF)
    assert (ff["down"].k, ff["down"].n) == (D_FF, H)
    assert mt.misc_params == dense.misc_params + 2 * H
    assert mt.misc_params == 2 * H + Q_LORA + KV_LORA + 2 * H
    assert (mt.core.kind, mt.core.n_q, mt.core.n_kv, mt.core.qk_dim, mt.core.rope_dim) == (
        "gqa", HEADS, HEADS, QK, 64)
    assert mt.core.kv_lora == 0 and mt.core.v_dim == QK
    assert dense.core.v_dim == V and main.core.v_dim == V
    assert mt.core.kv_elems_per_token() == HEADS * (QK + QK)
    assert main.core.kv_elems_per_token() == HEADS * (QK + V)
    assert mt.params(H) + extra == MTP

    ph = Phase("decode", 1, 1, 64)
    ops = _mtp_ops(m, ph, Shard())
    eh = next(op for op in ops if op.name == "mtp.eh_proj")
    assert (eh.k, eh.n, eh.w_params, eh.w_bits, eh.w_fmt) == (2 * H, H, 2 * H * H, 16.0, "bf16")
    pv = next(op for op in ops if op.name == "pv")
    assert pv.n == QK
    sm = next(op for op in ops if op.name == "softmax")
    assert sm.kv_read == 65 * HEADS * (QK + QK) * 2
    assert sm.kv_write == HEADS * (QK + QK) * 2
    o = next(op for op in ops if op.name == "attn.o")
    assert (o.k, o.n, o.w_bits, o.w_fmt) == (HEADS * V, H, 16.0, "bf16")
    up = next(op for op in ops if op.name == "attn.kv_up")
    assert (up.k, up.n) == (KV_LORA, HEADS * (V + V))
    assert not any(op.kind == "lookup" or op.name.startswith("embed") for op in ops)
    assert not any(op.name == "moe.router" or op.name.startswith("expert.") for op in ops)
    head = next(op for op in ops if op.name == "mtp.head")
    assert (head.k, head.n) == (H, VOCAB)
    main_ops = build_rank_ops(m, 0, ph, Shard())
    main_sm = next(op for op in main_ops if op.name == "softmax")
    assert main_sm.kv_read == 65 * HEADS * (QK + V) * 2
    assert main_sm.kv_write == HEADS * (QK + V) * 2
    assert next(op for op in main_ops if op.name == "pv").n == V

    text = " ".join(m.notes)
    assert "longcat_flash_mtp" in text and "qk_head_dim" in text
    assert "不另做查表" in text and "未填充的 V" in text

    cfg = copy.deepcopy(rel["config"])
    cfg["mla_scale_q_lora"] = False
    cfg["mla_scale_kv_lora"] = False
    cfg["routed_scaling_factor"] = 1.0
    alt = from_release(m.id, m.hf_id, rel, cfg)
    assert alt.param_check()["rel_err"] == 0.0
    assert alt.params() == m.params() and alt.mtp_params() == m.mtp_params()
    assert alt.mtp_layers[0].core.v_dim == QK and alt.layers[0].core.v_dim == V
    assert _sig(_mtp_ops(alt, ph, Shard())) == _sig(ops)
    assert _sig(build_rank_ops(alt, 0, ph, Shard())) == _sig(main_ops)

    cleared = copy.deepcopy(rel)
    cleared["params_mtp"] = 0
    bare = from_release(m.id, m.hf_id, cleared, cleared["config"])
    assert bare.mtp_layers == () and bare.mtp_extra_params == 0
    assert bare.params() == m.params() and bare.mtp_params() == 0
    assert _mtp_ops(bare, ph, Shard()) == []

    kept = copy.deepcopy(rel)
    kept.pop("params_mtp")
    still = from_release(m.id, m.hf_id, kept, kept["config"])
    assert len(still.mtp_layers) == 1 and still.mtp_layers[0].core.v_dim == QK

    ident = copy.deepcopy(rel["config"])
    ident["zero_expert_type"] = None
    same = from_release(m.id, m.hf_id, rel, ident)
    assert same.params() == m.params() and same.layers[0].ffn.zero_experts == 256
    bad = copy.deepcopy(rel["config"])
    bad["zero_expert_type"] = "copy"
    try:
        from_release(m.id, m.hf_id, rel, bad)
    except ValueError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("non-identity zero_expert_type was accepted")
    narrow = copy.deepcopy(rel["config"])
    narrow["qk_nope_head_dim"] = 32
    narrow["qk_rope_head_dim"] = 32
    try:
        from_release(m.id, m.hf_id, rel, narrow)
    except ValueError as exc:
        assert "qk_head_dim" in str(exc)
    else:
        raise AssertionError("negative MTP V pad was accepted")


def test_longcat_flash_thinking_matches_the_chat_bill():
    chat = get_model("longcat-flash-chat")
    m = get_model("longcat-flash-thinking")
    rel = load_release(m.hf_id)
    crel = load_release(chat.hf_id)
    assert rel["roles"] == crel["roles"]
    assert rel["n_tensors"] == crel["n_tensors"] == 43756
    assert rel["bytes_total"] == crel["bytes_total"]
    assert rel["bytes_total"] != 561_862_880_256
    assert rel["config"] == crel["config"]
    assert rel["quantization_config"] is None and rel["torch_dtype"] is None
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == chat.params() == PARAMS
    assert m.mtp_params() == chat.mtp_params() == MTP
    assert m.mtp_extra_params == chat.mtp_extra_params
    assert len(m.mtp_layers) == 1 and m.mtp_layers[0].core.v_dim == QK
    assert m.layers[0].core.v_dim == V
    assert m.fmt("mtp").fmt == "bf16" and m.fmt("router").fmt == "fp32"
    ids = {row["id"]: row for row in list_models()}
    row = ids["longcat-flash-thinking"]
    assert row["family"] == "LongCat-Flash" and row["provider"] == "meituan"
    assert row["domain"] == "llm" and row["param_err"] == 0.0


def test_u16_width_and_424b_index_stay_locked():
    mod = _summarize()
    old = mod.CACHE
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod.CACHE = root
            d = root / "probe__u16"
            d.mkdir()
            cfg = {
                "architectures": ["Ernie4_5_MoeForCausalLM"],
                "model_type": "ernie4_5_moe",
                "num_hidden_layers": 1,
                "hidden_size": 4,
            }
            (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            qkv = {"dtype": "U16", "shape": [4, 8], "data_offsets": [0, 64]}
            up = {"dtype": "U16", "shape": [4, 8], "data_offsets": [64, 128]}
            (d / "headers.json").write_text(json.dumps({
                "model.safetensors": {
                    "model.layers.0.self_attn.qkv_proj.weight": qkv,
                    "model.layers.0.mlp.up_gate_proj.weight": up,
                },
            }), encoding="utf-8")
            got = mod.summarize("probe/u16")
            assert got["storage_alias"] == {"U16": "bf16"}
            assert got["fused_linears"] == ["qkv_proj", "up_gate_proj"]
            assert got["roles"]["attn"]["bf16"] == {"params": 32, "bytes": 64}
            assert got["roles"]["mlp"]["bf16"] == {"params": 32, "bytes": 64}
            assert "u16" not in got["roles"]["attn"] and "u16" not in got["roles"]["mlp"]
            qkv["data_offsets"] = [0, 32]
            (d / "headers.json").write_text(json.dumps({
                "model.safetensors": {
                    "model.layers.0.self_attn.qkv_proj.weight": qkv,
                    "model.layers.0.mlp.up_gate_proj.weight": up,
                },
            }), encoding="utf-8")
            try:
                mod.summarize("probe/u16")
            except ValueError as exc:
                assert "U16" in str(exc)
            else:
                raise AssertionError("U16 width other than 2 bytes was accepted")
            qkv["dtype"] = "BF16"
            qkv["data_offsets"] = [0, 64]
            up["dtype"] = "BF16"
            (d / "headers.json").write_text(json.dumps({
                "model.safetensors": {
                    "model.layers.0.self_attn.qkv_proj.weight": qkv,
                    "model.layers.0.mlp.up_gate_proj.weight": up,
                },
            }), encoding="utf-8")
            bf = mod.summarize("probe/u16")
            assert "storage_alias" not in bf
            assert bf["fused_linears"] == ["qkv_proj", "up_gate_proj"]
            assert bf["roles"]["attn"]["bf16"]["bytes"] == 2 * bf["roles"]["attn"]["bf16"]["params"]
    finally:
        mod.CACHE = old

    rel = load_release("baidu/ERNIE-4.5-VL-424B-A47B")
    assert rel["bytes_total"] == VL424_BYTES
    assert rel["bytes_total"] != VL424_INDEX
    assert VL424_INDEX - VL424_BYTES == rel["roles"]["norm"]["bf16"]["bytes"]
