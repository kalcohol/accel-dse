"""0.396: ERNIE-4.5 Thinking reads moe_gate; OmniASR CTC properties hold.

ERNIE-4.5-21B-A3B-Thinking publishes moe_gate as the string "top2_fused".
PaddlePaddle/ERNIE modeling_moe.py maps top2_fused and topk to TopKGate.
Selected experts stay moe_k (6). The 28B VL Thinking row publishes "topk"
and matches the existing dual-pool bill after the stored tied lm_head is
deducted. OmniASR CTC stays one bidirectional fp32 forward.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import audio_ops, image_grid, video_pre, vision_ops

THINK_PARAMS = 21_825_437_888
VL_LANG = 28_624_635_776
VL_HEAD = 103_424 * 2560
VL_VISION = 774_658_560
VOCAB = 10288


def _frames(au, seconds: float) -> int:
    n = round(seconds * au.rate)
    for k, st, _pad in au.ds:
        n = (n - k) // st + 1 if n >= k else 0
    return max(0, n)


def _raise_gate(hf: str, rel: dict, gate: str) -> None:
    cfg = copy.deepcopy(rel["config"])
    cfg["moe_gate"] = gate
    try:
        from_release("probe", hf, rel, cfg)
    except ValueError as exc:
        assert "TopKGate" in str(exc)
        return
    raise AssertionError(f"{hf} accepted moe_gate={gate!r}")


def test_ernie_thinking_reads_top2_fused_as_topk():
    m = get_model("ernie-4.5-21b-a3b-thinking")
    base = get_model("ernie-4.5-21b-a3b-base-pt")
    rel = load_release(m.hf_id)
    h, E, k = 2560, 64, 6
    assert rel["config"]["moe_gate"] == "top2_fused"
    assert isinstance(rel["config"]["moe_gate"], str)
    assert "top2_fused" not in rel["config"]
    assert "moe_layer_end_index" not in rel["config"]
    assert "num_nextn_predict_layers" not in rel["config"]
    assert rel["config"]["moe_k"] == k and rel["config"]["moe_capacity"] == [64, 64, 64]
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == rel["params_llm"] == THINK_PARAMS == base.params()
    assert m.mtp_params() == rel["params_mtp"] == 0 and m.mtp_layers == ()
    assert m.vision is None and m.vision_params == 0
    assert rel["n_tensors"] == 5492
    # Router weights are fp32 (4 bytes). Every other language parameter is bf16.
    assert rel["bytes_total"] == 43_659_726_592 == 2 * THINK_PARAMS + 2 * 4_425_408
    assert rel["bytes_llm"] == rel["bytes_total"] and rel["torch_dtype"] == "bfloat16"
    assert set(rel["roles"]["router"]) == {"fp32"}
    assert rel["roles"]["router"]["fp32"]["params"] == 27 * (h * E + E) == 4_425_408
    assert m.n_layers == 28 and m.hidden == h and m.vocab == 103424 and m.tie_embeddings
    assert m.coverage == "full" and m.domain == "llm"
    assert m.layers[0].ffn.kind == "dense"
    for i in range(1, 28):
        ffn = m.layers[i].ffn
        assert ffn.kind == "moe" and ffn.top_k == k and ffn.n_experts == E
        assert ffn.top_k != 2 and ffn.d_expert == 1536
        assert ffn.n_shared == 1 and ffn.d_shared == 3072
        router = next(lin for lin in m.layers[i].ffn_linears if lin.name == "router")
        assert (router.k, router.n) == (h, E)
        assert m.layers[i].misc_params == 2 * h + E
    assert m.fmt("router").fmt == "fp32" and m.fmt("router").bits == 32.0
    text = " ".join(m.notes)
    assert "top2_fused" in text and "TopKGate" in text and "不是 2" in text
    assert "top2_fused" not in " ".join(base.notes)
    assert base.layers[1].ffn.top_k == k and base.param_check()["rel_err"] == 0.0
    dense = build_rank_ops(m, 0, Phase("decode", 1, 1, 16), Shard())
    assert not any(op.name == "moe.router" for op in dense)
    routed = build_rank_ops(m, 1, Phase("decode", 1, 1, 16), Shard())
    rt = next(op for op in routed if op.name == "moe.router")
    assert (rt.k, rt.n, rt.w_fmt, rt.w_bits) == (h, E, "fp32", 32.0)
    gate = next(op for op in routed if op.name == "expert.gate_up")
    assert gate.count * gate.m * gate.share == k
    row = {e["id"]: e for e in list_models()}["ernie-4.5-21b-a3b-thinking"]
    assert row["family"] == "ERNIE-4.5" and row["domain"] == "llm" and row["param_err"] == 0.0

    cfg = copy.deepcopy(rel["config"])
    cfg["moe_capacity"] = [1, 1, 1]
    shrunk = from_release(m.id, m.hf_id, rel, cfg)
    assert shrunk.params() == m.params() and shrunk.layers[1].ffn.top_k == k
    cfg.pop("moe_gate")
    bare = from_release(m.id, m.hf_id, rel, cfg)
    assert bare.params() == m.params() and bare.layers[1].ffn.top_k == k
    assert "top2_fused" not in " ".join(bare.notes)
    for bad in ("top2", "gshard", ""):
        _raise_gate(m.hf_id, rel, bad)


def test_ernie_vl_thinking_matches_the_pt_bill():
    m = get_model("ernie-4.5-vl-28b-a3b-thinking")
    pt = get_model("ernie-4.5-vl-28b-a3b-pt")
    rel = load_release(m.hf_id)
    assert rel["config"]["moe_gate"] == "topk"
    assert rel["config"]["moe_k"] == 6
    assert m.param_check()["rel_err"] == 0.0
    assert m.params() == m.release_params == pt.params() == VL_LANG
    assert rel["params_llm"] == VL_LANG + VL_HEAD
    assert "tied lm_head" in " ".join(m.notes)
    assert m.mtp_params() == 0 and m.mtp_layers == ()
    assert m.vision_params == rel["params_vision"] == pt.vision_params == VL_VISION
    assert rel["n_tensors"] == 11106
    assert rel["bytes_total"] == 59_345_821_184
    assert rel["bytes_total"] != 33_906_952_576
    assert set(rel["roles"]) >= {"lm_head", "vision", "router"}
    assert rel["roles"]["lm_head"]["bf16"]["params"] == VL_HEAD
    assert set(rel["roles"]["vision"]) == {"bf16"}
    assert rel["roles"]["vision"]["bf16"]["params"] == VL_VISION
    assert rel["roles"]["vision"]["bf16"]["bytes"] == 2 * VL_VISION
    assert m.vision.family == "ernievl" and m.vision.w_Bpp == 2.0 and m.vision.wfmt == ()
    assert m.layers[0].ffn.kind == "dense"
    assert m.layers[1].ffn.top_k == 6 and m.layers[1].ffn.n_experts_b == 64
    assert m.layers[1].ffn.d_expert == 1536 and m.layers[1].ffn.d_expert_b == 512
    assert m.fmt("router").fmt == "fp32"
    text = " ".join(m.notes)
    assert "TopKGate" in text and "top2_fused" in text
    assert video_pre(m) == video_pre(pt)
    g = image_grid(m.vision, 1024, 1024)
    patch = next(op for op in vision_ops(m.vision, g, 1) if op.name == "vis.patch_embed")
    assert patch.w_fmt == "bf16" and patch.w_bits == 16.0
    row = {e["id"]: e for e in list_models()}["ernie-4.5-vl-28b-a3b-thinking"]
    assert row["family"] == "ERNIE-4.5-VL" and row["domain"] == "vlm" and row["param_err"] == 0.0
    assert pt.param_check()["rel_err"] == 0.0 and pt.layers[1].ffn.top_k == 6

    cfg = copy.deepcopy(rel["config"])
    cfg.pop("moe_gate")
    bare = from_release(m.id, m.hf_id, rel, cfg)
    assert bare.params() == m.params() and bare.layers[1].ffn.top_k == 6
    for bad in ("top2", "top2_fused", "gshard"):
        _raise_gate(m.hf_id, rel, bad)


def test_omniasr_ctc_properties_hold():
    ids = ("omniasr-ctc-300m-v2", "omniasr-ctc-1b-v2", "omniasr-ctc-3b-v2", "omniasr-ctc-7b-v2")
    for mid in ids:
        m = get_model(mid)
        rel = load_release(m.hf_id)
        au = m.vision.audio
        hidden = au.dims[0]
        assert rel["params_llm"] == 0 and rel["params_mtp"] == 0
        assert m.params() == rel["params_vision"] == au.params
        assert rel["bytes_total"] == 4 * m.params()
        assert m.fmt("attn").fmt == "fp32" and m.act_fmt == "fp32" and m.kv_fmt == "fp32"
        assert m.vision.w_Bpp == 4.0 and au.w_fmt == "fp32" and au.w_bits == 32.0
        assert m.ctc_only and not m.kv_cache and m.layers == ()
        assert au.tokens(1.0) == 49 and au.tokens(10.0) == 499
        assert au.prompt_tokens(1.0) == au.prompt_tokens(10.0) == 0
        for sec in (0.0, 0.25, 0.5, 1.0, 2.0, 10.0, 30.0):
            assert au.tokens(sec) == _frames(au, sec)
        k0 = au.ds[0][0]
        short = 0.0
        for samples in range(k0):
            sec = samples / au.rate
            if round(sec * au.rate) == samples:
                short = sec
        assert au.tokens(short) == 0 and au.tokens(0.0) == 0
        quiet = audio_ops(au, short, 1, "fp32")
        assert not any(op.name == "aud.ctc" or op.name.endswith(".qk") for op in quiet)
        one = audio_ops(au, 1.0, 1, "fp32")
        ctc = next(op for op in one if op.name == "aud.ctc")
        qk = next(op for op in one if op.name == "aud.0.qk")
        pv = next(op for op in one if op.name == "aud.0.pv")
        assert ctc.w_params == hidden * VOCAB + VOCAB
        assert ctc.w_params != hidden * VOCAB
        assert (ctc.m, ctc.k, ctc.n) == (49, hidden, VOCAB)
        assert (qk.m, qk.n, qk.k) == (49, 49, hidden // 16)
        assert qk.causal == 1.0 and pv.causal == 1.0
        assert qk.kv_write == 0.0 and qk.kv_read == 0.0
        assert pv.kv_write == 0.0 and pv.kv_read == 0.0
        assert all(op.w_fmt == "fp32" and op.a_fmt == "fp32" for op in one if op.kind == "gemm")
        batched = audio_ops(au, 1.0, 3, "fp32")
        ctc_b = next(op for op in batched if op.name == "aud.ctc")
        qk_b = next(op for op in batched if op.name == "aud.0.qk")
        assert ctc_b.m == 3 * 49 and ctc_b.k == ctc.k and ctc_b.n == ctc.n
        assert ctc_b.w_params == ctc.w_params
        assert (qk_b.m, qk_b.n) == (49, 49) and qk_b.count == 3 * qk.count
        pre = evaluate(Scenario(model=mid, serving=Serving(phase="prefill", batch=1, audio_s=1.0)))
        dec = evaluate(Scenario(model=mid, serving=Serving(phase="decode", batch=1, audio_s=1.0)))
        assert pre.asr["enc_positions"] == dec.asr["enc_positions"] == 49
        assert pre.asr["prompt_audio_tokens"] == 0
        assert any("没有自回归" in w for w in dec.warnings)
        assert dec.asr["encoder_s"] == dec.step and dec.step == pre.step and dec.step > 0
        assert pre.step > 0
        zero = evaluate(Scenario(model=mid, serving=Serving(phase="prefill", batch=1, audio_s=0.0)))
        assert zero.vision["audio"]["tokens"] == 0 and zero.step == 0.0
        tiny = evaluate(Scenario(model=mid, serving=Serving(phase="prefill", batch=1, audio_s=short)))
        assert tiny.vision["audio"]["tokens"] == 0
