"""0.391: Granite-4.2 MXFP4 / NVFP4 on the existing W4A4 bill.

Packed Linear weights are mxfp4 (4.25 bit) or nvfp4 (4.5 bit plus two fp32 scales).
Dynamic 4-bit inputs use those formats. Attention QK/PV and lm_head stay bf16.
A scheme that does not match the released pack keeps bf16 GEMM inputs.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.dtypes import gemm_exec
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops
from accel_dse.core.model import from_release, load_release
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving

# mid, bf16 twin, logical params, tensors, bytes, activation format, attention_multiplier
_ROWS = (
    ("granite-4.2-3b-mxfp4", "granite-4.2-3b", 3_659_737_600, 643, 2_699_187_200, "mxfp4", 0.015625),
    ("granite-4.2-3b-nvfp4", "granite-4.2-3b", 3_659_737_600, 1203, 2_797_493_440, "nvfp4", 0.015625),
    ("granite-4.2-8b-mxfp4", "granite-4.2-8b", 8_791_592_960, 643, 5_878_456_320, "mxfp4", 0.0078125),
    ("granite-4.2-8b-nvfp4", "granite-4.2-8b", 8_791_592_960, 1203, 6_127_495_360, "nvfp4", 0.0078125),
    ("granite-4.2-30b-mxfp4", "granite-4.2-30b", 29_276_770_304, 1027, 16_761_495_552, "mxfp4", 0.0078125),
    ("granite-4.2-30b-nvfp4", "granite-4.2-30b", 29_276_770_304, 1923, 17_650_691_584, "nvfp4", 0.0078125),
)


def test_granite_42_w4a4_matches_bf16_shapes():
    ids = [e["id"] for e in list_models() if e["family"] == "Granite-4.2"]
    assert ids == [
        "granite-4.2-30b", "granite-4.2-30b-mxfp4", "granite-4.2-30b-nvfp4", "granite-4.2-30b-fp8",
        "granite-4.2-8b", "granite-4.2-8b-mxfp4", "granite-4.2-8b-nvfp4", "granite-4.2-8b-fp8",
        "granite-4.2-3b", "granite-4.2-3b-mxfp4", "granite-4.2-3b-nvfp4", "granite-4.2-3b-fp8",
    ]
    for mid, twin, params, nt, total, name, mult in _ROWS:
        m, b = get_model(mid), get_model(twin)
        rel, old = load_release(m.hf_id), load_release(b.hf_id)
        assert m.param_check()["rel_err"] == 0.0 and b.param_check()["rel_err"] == 0.0, mid
        assert m.params() == b.params() == params == m.release_params == rel["params_llm"], mid
        assert rel["n_tensors"] == nt and rel["bytes_total"] == rel["bytes_llm"] == total, mid
        assert rel["params_mtp"] == 0 and rel["params_vision"] == 0, mid
        assert rel["bytes_total"] < old["bytes_total"], mid
        assert sum(st["bytes"] for role in rel["roles"].values() for st in role.values()) == total, mid
        layers = m.n_layers
        assert rel["n_tensors"] == (643 if layers == 40 else 1027) + (0 if name == "mxfp4" else layers * 14), mid
        for role, n_lin in (("attn", 4), ("mlp", 3)):
            body = rel["roles"][role][name]
            p, nb = body["params"], body["bytes"]
            assert m.fmt(role).fmt == name and m.fmt(role).bits == 8.0 * nb / p, mid
            if name == "mxfp4":
                assert nb * 32 == p * 17 and m.fmt(role).bits == 4.25, mid
            else:
                assert nb == p * 9 // 16 + 8 * layers * n_lin and m.fmt(role).bits > 4.5, mid
        for role in ("embed", "lm_head"):
            body = rel["roles"][role]["bf16"]
            assert body["bytes"] == 2 * body["params"] and m.fmt(role).fmt == "bf16", mid
        norm = rel["roles"]["norm"]["bf16"]
        assert norm["bytes"] == 2 * norm["params"] and "norm" not in dict(m.formats), mid
        qc = rel["quantization_config"]
        assert qc["quant_method"] == "compressed-tensors" and qc["ignore"] == ["lm_head"], mid
        assert qc.get("kv_cache_scheme") is None and qc["format"].startswith(name), mid
        group = next(iter(qc["config_groups"].values()))
        assert group["format"] == qc["format"] and group["targets"] == ["Linear"], mid
        assert group["output_activations"] is None, mid
        assert group["weights"]["num_bits"] == 4 and group["input_activations"]["num_bits"] == 4, mid
        assert group["weights"]["type"] == "float" and group["input_activations"]["type"] == "float", mid
        if name == "mxfp4":
            assert group["weights"]["group_size"] == 32 and group["weights"]["dynamic"] is False, mid
            assert group["input_activations"]["dynamic"] is True, mid
            assert group["weights"]["scale_dtype"] == "torch.uint8", mid
            assert group["weights"]["strategy"] == "group", mid
        else:
            assert group["weights"]["group_size"] == 16 and group["input_activations"]["dynamic"] == "local", mid
            assert group["weights"]["scale_dtype"] == "torch.float8_e4m3fn", mid
            assert group["weights"]["strategy"] == "tensor_group", mid
        cfg, base = rel["config"], old["config"]
        assert {k: v for k, v in cfg.items() if k != "quantization_config"} == {
            k: v for k, v in base.items() if k != "quantization_config"}, mid
        assert cfg["attention_multiplier"] == mult == base["attention_multiplier"], mid
        assert cfg["embedding_multiplier"] == cfg["residual_multiplier"] == cfg["logits_scaling"] == 1.0, mid
        assert cfg["tie_word_embeddings"] is False and m.tie_embeddings is False, mid
        assert m.act_fmt == "bf16" and m.kv_fmt == "bf16" and b.act_fmt == "bf16", mid
        assert m.act_fmt_for("attn") == name and m.act_fmt_for("mlp") == name, mid
        assert m.act_fmt_for("lm_head") == "bf16" and m.act_fmt_for("embed") == "bf16", mid
        assert {k: v.fmt for k, v in m.formats if k.startswith("act_")} == {
            "act_attn": name, "act_mlp": name}, mid
        assert m.coverage == "full" and m.vision is None and m.mtp_layers == (), mid
        for Lm, Lb in zip(m.layers, b.layers):
            assert [(x.name, x.k, x.n) for x in Lm.attn_linears] == [(x.name, x.k, x.n) for x in Lb.attn_linears]
            assert [(x.name, x.k, x.n) for x in Lm.ffn_linears] == [(x.name, x.k, x.n) for x in Lb.ffn_linears]
            assert Lm.core.n_q == Lb.core.n_q and Lm.core.n_kv == Lb.core.n_kv and Lm.core.qk_dim == Lb.core.qk_dim
            assert Lm.ffn.d_ff == Lb.ffn.d_ff and Lm.misc_params == Lb.misc_params
        gemm = [o for o in build_rank_ops(m, 0, Phase("prefill", 1, 16, 0), Shard()) if o.kind == "gemm" and o.w_params]
        names = {o.name: o for o in gemm}
        fused = ("attn.qkv", "attn.o", "mlp.gate_up", "mlp.down")
        assert set(names) >= set(fused), mid
        assert all(names[n].w_fmt == name and names[n].a_fmt == name for n in fused), mid
        assert names["attn.qkv"].w_bits == m.fmt("attn").bits and names["mlp.down"].w_bits == m.fmt("mlp").bits, mid
        attn = [o for o in build_rank_ops(m, 0, Phase("prefill", 1, 16, 0), Shard()) if o.kind == "attn"]
        assert {o.name for o in attn} >= {"qk", "pv"}, mid
        assert all(o.a_fmt == "bf16" for o in attn if o.name in ("qk", "pv")), mid
        head = [o for o in head_ops(m, Phase("prefill", 1, 16, 0)) if o.name == "lm_head"]
        assert head and all(o.w_fmt == "bf16" and o.a_fmt == "bf16" for o in head), mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["family"] == "Granite-4.2" and listed["coverage"] == "full" and listed["domain"] == "llm", mid
    fp8 = get_model("granite-4.2-8b-fp8")
    assert fp8.act_fmt == "fp8" and not any(k.startswith("act_") for k, _ in fp8.formats)
    head = [o for o in head_ops(fp8, Phase("prefill", 1, 16, 0)) if o.name == "lm_head"]
    assert head and all(o.a_fmt == "fp8" and o.w_fmt == "bf16" for o in head)
    b200, h100 = REF_HW["B200"].chip.formats, REF_HW["H100-SXM"].chip.formats
    assert gemm_exec("mxfp4", "mxfp4", b200)[:3] == ("mxfp4", 4.0, "none")
    assert gemm_exec("nvfp4", "nvfp4", b200)[:3] == ("nvfp4", 4.0, "none")
    assert gemm_exec("mxfp4", "mxfp4", h100)[:3] == ("bf16", 1.0, "upcast_both")
    assert gemm_exec("nvfp4", "nvfp4", h100)[:3] == ("bf16", 1.0, "upcast_both")
    decode = evaluate(Scenario(model="granite-4.2-8b-mxfp4", serving=Serving(phase="decode", batch=1, ctx=32)))
    assert decode.step > 0
    decode = evaluate(Scenario(model="granite-4.2-30b-nvfp4", serving=Serving(phase="decode", batch=1, ctx=16)))
    assert decode.step > 0


def test_granite_42_w4a4_rejects_a_different_scheme():
    for mid, name in (("granite-4.2-8b-mxfp4", "mxfp4"), ("granite-4.2-3b-nvfp4", "nvfp4")):
        rel = load_release(get_model(mid).hf_id)
        kept = from_release("synth", rel["repo"], rel=copy.deepcopy(rel))
        assert kept.act_fmt_for("attn") == name and kept.fmt("attn").fmt == name, mid
        assert kept.params() == get_model(mid).params(), mid

        def check(mut, rel=rel, name=name):
            broken = copy.deepcopy(rel)
            mut(broken["quantization_config"])
            spec = from_release("synth", broken["repo"], rel=broken)
            assert spec.fmt("attn").fmt == name and spec.fmt("mlp").fmt == name
            assert spec.act_fmt == "bf16" and spec.kv_fmt == "bf16"
            assert spec.act_fmt_for("attn") == "bf16" and spec.act_fmt_for("mlp") == "bf16"
            assert not any(k.startswith("act_") for k, _ in spec.formats)
            assert spec.params() == kept.params()

        check(lambda qc: qc["config_groups"]["group_0"]["weights"].__setitem__("group_size", 8))
        check(lambda qc: qc.__setitem__("format", "float-quantized"))
        check(lambda qc: qc.__setitem__("ignore", ["lm_head", "embed"]))
        check(lambda qc: qc["config_groups"]["group_0"]["input_activations"].__setitem__("dynamic", False))
        check(lambda qc: qc["config_groups"].__setitem__("group_1", qc["config_groups"]["group_0"]))
        check(lambda qc: qc["config_groups"]["group_0"].__setitem__("output_activations", {}))
        check(lambda qc: qc["config_groups"]["group_0"].__setitem__("targets", ["Embedding"]))
        # A non-empty KV scheme is the existing fp8-KV rule, and it also drops the W4A4 input.
        kv = copy.deepcopy(rel)
        kv["quantization_config"]["kv_cache_scheme"] = {"num_bits": 8}
        spec = from_release("synth", kv["repo"], rel=kv)
        assert spec.kv_fmt == "fp8" and spec.act_fmt_for("attn") == "bf16" and spec.fmt("attn").fmt == name
