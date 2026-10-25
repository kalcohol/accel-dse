"""0.392: Granite Speech conformer + Q-Former on the granite LM.

Shaw block attention and the BLIP-2 query bridge are billed from the public class.
The CTC-only 470m encoder has no autoregressive backbone and stays out of the catalog.
Reviewed 0.388-0.391 bills are unchanged.
"""
from __future__ import annotations

import copy

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import audio_ops, audio_spec

_LLM = {
    "granite-speech-5.0-2b": 1_837_275_136,
    "granite-speech-4.1-2b-plus": 1_631_752_192,
}
_VIS = {
    "granite-speech-5.0-2b": 475_866_460,
    "granite-speech-4.1-2b-plus": 480_060_764,
}
_CROSS = {"granite-speech-5.0-2b": 1024, "granite-speech-4.1-2b-plus": 2048}
_TENSORS = {"granite-speech-5.0-2b": 954, "granite-speech-4.1-2b-plus": 953}
_BYTES = {
    "granite-speech-5.0-2b": 4_626_414_392,
    "granite-speech-4.1-2b-plus": 4_223_757_112,
}


def test_granite_speech_catalog_matches_headers():
    ids = [e["id"] for e in list_models() if e["family"] == "Granite-Speech"]
    assert ids == ["granite-speech-5.0-2b", "granite-speech-4.1-2b-plus"]
    assert "granite-speech-5.0-470m-turboctc" not in {e["id"] for e in list_models()}
    for mid, llm in _LLM.items():
        m = get_model(mid)
        rel = load_release(m.hf_id)
        vis = _VIS[mid]
        assert m.param_check()["rel_err"] == 0.0, mid
        assert m.params() == llm == rel["params_llm"] == m.release_params, mid
        assert m.vision is not None and m.vision.params == m.vision.audio.params == vis == rel["params_vision"], mid
        assert m.vision.extra == 0 and m.vision.family == "audio", mid
        assert rel["params_mtp"] == 0 and m.mtp_layers == () and m.coverage == "full", mid
        assert rel["n_tensors"] == _TENSORS[mid] and rel["bytes_total"] == _BYTES[mid], mid
        assert rel["roles"]["buffer"]["int64"]["bytes"] == 131_200, mid
        assert sum(st["bytes"] for role in rel["roles"].values() for st in role.values()) == rel["bytes_total"], mid
        for role, body in rel["roles"].items():
            if role == "buffer":
                continue
            assert set(body) == {"bf16"} and body["bf16"]["bytes"] == 2 * body["bf16"]["params"], mid
        assert rel["roles"]["vision"]["bf16"]["params"] == vis, mid
        assert m.act_fmt == "bf16" and m.kv_fmt == "bf16" and m.vision.w_Bpp == 2.0, mid
        assert m.hidden == 2048 and m.n_layers == 40 and m.vocab == 100353, mid
        layer = m.layers[0]
        assert [(x.name, x.k, x.n) for x in layer.attn_linears] == [
            ("q", 2048, 2048), ("k", 2048, 512), ("v", 2048, 512), ("o", 2048, 2048)], mid
        assert layer.ffn.d_ff == 4096 and layer.core.n_q == 16 and layer.core.n_kv == 4, mid
        assert layer.core.qk_dim == 128 and layer.misc_params == 2 * 2048, mid
        au = m.vision.audio
        assert au.kind == "gspeech" and au.gs_cross == _CROSS[mid] and au.gs_q == 3 and au.gs_mid == 348, mid
        assert au.enc_positions(1.0) == 50 and au.tokens(1.0) == au.prompt_tokens(1.0) == 12, mid
        assert au.tokens(0.0) == 0 and au.enc_positions(0.0) == 0, mid
        listed = next(e for e in list_models() if e["id"] == mid)
        assert listed["domain"] == "asr" and listed["coverage"] == "full", mid
    base, plus = get_model("granite-speech-5.0-2b"), get_model("granite-speech-4.1-2b-plus")
    assert base.tie_embeddings is False and "lm_head" in load_release(base.hf_id)["roles"]
    assert plus.tie_embeddings is True and "lm_head" not in load_release(plus.hf_id)["roles"]
    assert load_release(plus.hf_id)["config"]["text_config"]["dtype"] == "float32"
    assert load_release(base.hf_id)["config"]["text_config"]["dtype"] == "bfloat16"
    assert base.max_ctx == 16384 and plus.max_ctx == 4096
    assert "Shaw" in " ".join(base.notes) and "float32" in " ".join(plus.notes)
    assert "不另计 GEMM" in " ".join(base.notes)


def test_granite_speech_ops_follow_the_class():
    for mid, cross in _CROSS.items():
        m = get_model(mid)
        ops = audio_ops(m.vision.audio, 1.0, 1)
        by = {}
        for op in ops:
            assert op.name not in by, op.name
            by[op.name] = op
        assert by["aud.in"].m == 50 and (by["aud.in"].k, by["aud.in"].n) == (160, 1024), mid
        assert by["aud.0.q"].m == 200 and (by["aud.0.q"].k, by["aud.0.q"].n) == (1024, 1024), mid
        assert by["aud.0.kv"].m == 200 and by["aud.0.kv"].n == 2048, mid
        assert by["aud.0.o"].m == 50 and by["aud.0.ff1_up"].m == 50 and by["aud.0.ff1_up"].n == 4096, mid
        assert (by["aud.0.shaw"].m, by["aud.0.shaw"].k, by["aud.0.shaw"].n, by["aud.0.shaw"].count) == (200, 128, 200, 8), mid
        assert by["aud.0.shaw"].w_params == 0 and by["aud.0.shaw"].kind == "attn", mid
        assert (by["aud.0.qk"].m, by["aud.0.qk"].n) == (200, 200) and (by["aud.0.pv"].m, by["aud.0.pv"].k) == (200, 200), mid
        assert by["aud.0.conv_up"].n == 4096 and by["aud.0.conv_down"].k == 2048 and by["aud.0.conv_down"].m == 50, mid
        assert by["aud.0.dconv"].kind == "vector" and by["aud.0.dconv"].vec == 50 * 2048 * 15, mid
        assert sum(1 for name in by if name.endswith(".ff1_up")) == 16, mid
        assert by["aud.mid.out"].m == 50 and (by["aud.mid.out"].k, by["aud.mid.out"].n) == (1024, 348), mid
        assert (by["aud.mid.back"].k, by["aud.mid.back"].n) == (348, 1024), mid
        assert sum(1 for name in by if name.startswith("aud.mid.")) == 3, mid
        assert by["aud.qf.0.q"].m == 12 and by["aud.qf.0.ck"].m == 60, mid
        assert (by["aud.qf.0.ck"].k, by["aud.qf.0.ck"].n) == (cross, 1024), mid
        assert (by["aud.qf.0.cqk"].m, by["aud.qf.0.cqk"].n, by["aud.qf.0.cqk"].count) == (3, 15, 64), mid
        assert by["aud.qf.1.ck"].k == cross and by["aud.proj"].m == 12, mid
        assert (by["aud.proj"].k, by["aud.proj"].n) == (1024, 2048), mid
        assert all(op.w_fmt == "bf16" and op.a_fmt == "bf16" for op in ops if op.kind == "gemm"), mid
        wide = audio_ops(m.vision.audio, 1.0, 2)
        assert next(op for op in wide if op.name == "aud.in").m == 100, mid
    pre = evaluate(Scenario(model="granite-speech-5.0-2b", serving=Serving(phase="prefill", batch=1, prompt=4, audio_s=1.0)))
    assert pre.vision["image_tokens"] == 12 and pre.vision["audio"]["tokens"] == 12
    assert pre.asr["enc_positions"] == 50 and pre.asr["prompt_audio_tokens"] == 12
    assert pre.step > 0
    quiet = evaluate(Scenario(model="granite-speech-5.0-2b", serving=Serving(phase="prefill", batch=1, prompt=4)))
    assert quiet.vision is None


def test_granite_speech_refuses_a_different_tower():
    rel = load_release(get_model("granite-speech-5.0-2b").hf_id)
    kept = audio_spec(rel["config"])
    assert kept is not None and kept.params == _VIS["granite-speech-5.0-2b"]

    def gone(mut):
        cfg = copy.deepcopy(rel["config"])
        mut(cfg)
        assert audio_spec(cfg) is None

    gone(lambda cfg: cfg["projector_config"].__setitem__("use_qformer_text_input", True))
    gone(lambda cfg: cfg.__setitem__("window_size", 16))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("context_size", 600))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("dim_head", 64))
    gone(lambda cfg: cfg.__setitem__("has_lora_adapter", True))
    gone(lambda cfg: cfg["projector_config"].__setitem__("encoder_hidden_size", 2048))
    gone(lambda cfg: cfg["encoder_config"].__setitem__("cat_hidden_layers", [3]))
    plus = copy.deepcopy(load_release(get_model("granite-speech-4.1-2b-plus").hf_id)["config"])
    plus["encoder_config"]["cat_hidden_layers"] = []
    assert audio_spec(plus) is None
    plus["encoder_config"]["cat_hidden_layers"] = [3, 3]
    plus["projector_config"]["encoder_hidden_size"] = 2048
    again = audio_spec(plus)
    assert again is not None and again.gs_cross == 2048 and again.params == _VIS["granite-speech-4.1-2b-plus"]
    assert audio_spec({"model_type": "granite_speech5_ctc"}) is None
    cfg = copy.deepcopy(rel["config"])
    for key, value in (("attention_multiplier", 3.0), ("embedding_multiplier", 1.0),
                       ("residual_multiplier", 1.0), ("logits_scaling", 1.0)):
        cfg["text_config"][key] = value
    spec = from_release("synth", rel["repo"], rel={**rel, "config": cfg})
    base = get_model("granite-speech-5.0-2b")
    assert spec.params() == base.params()
    assert [(x.name, x.k, x.n) for x in spec.layers[0].attn_linears] == [
        (x.name, x.k, x.n) for x in base.layers[0].attn_linears]
    assert spec.layers[0].misc_params == base.layers[0].misc_params


def test_reviewed_ships_keep_their_bills():
    w = get_model("wevisone-vl-4b")
    block = load_release(w.hf_id)["roles"]["vision"]["fp32"]
    assert block["bytes"] == 4 * block["params"] == 4 * w.vision.params
    assert w.vision.w_Bpp == 4.0 and w.act_fmt == "bf16" and w.fmt("attn").fmt == "fp32"
    v = get_model("lfm2.5-vl-1.6b")
    assert v.vision.family == "lfm2vl" and v.vision.merge == 2
    assert v.vision.post[0][0] == v.vision.hidden * v.vision.merge * v.vision.merge
    assert v.tie_embeddings is True and v.param_check()["rel_err"] == 0.0
    m2 = get_model("minimax-m2")
    layer = m2.layers[0]
    hd = layer.core.qk_dim
    assert hd == 128
    assert layer.misc_params == 2 * m2.hidden + layer.core.n_q * hd + layer.core.n_kv * hd + layer.ffn.n_experts
    assert load_release(m2.hf_id)["config"]["use_qk_norm"] is True
    dawn = get_model("atria-dawn")
    cfg = load_release(dawn.hf_id)["config"]
    assert cfg["head_dim"] == 64 and dawn.layers[0].core.qk_dim == 256 and dawn.layers[0].core.v_dim == 256
    assert dawn.layers[0].core.qk_dim != cfg["head_dim"]
    g = get_model("granite-4.2-8b-mxfp4")
    assert g.act_fmt == "bf16" and g.act_fmt_for("attn") == "mxfp4" and g.fmt("attn").bits == 4.25
    broken = copy.deepcopy(load_release(g.hf_id))
    broken["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = 8
    spec = from_release("synth", broken["repo"], rel=broken)
    assert spec.fmt("attn").fmt == "mxfp4" and spec.act_fmt_for("attn") == "bf16" and spec.act_fmt == "bf16"
