"""0.381: 0.380 Mistral-Small-3.2 review holds; dense ERNIE-4.5-0.3B closes.

Mistral-Small-3.2 shard headers (2026-10-24) still match the tied language count,
the Pixtral tower, and the 1024 image positions. ERNIE-4.5-0.3B-PT and Base-PT
are dense ernie4_5. head_dim is 128, not hidden/heads. Counts are the release
headers (2026-10-24). The public class has no MTP module.
"""
from __future__ import annotations

import copy

from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, head_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.model import from_release, load_release
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import image_grid


def _weights(m):
    return stage_storage(m, 0, m.n_layers, True, True, Shard(), 0).weights


def test_review_380_mistral_small_3_2_unchanged():
    h, heads, kv, hd, inter, vocab, layers = 5120, 32, 8, 128, 32768, 131072, 40
    q, kk = heads * hd, kv * hd
    attn = h * q + 2 * h * kk + q * h
    language = layers * (attn + 3 * h * inter + 2 * h) + vocab * h + h
    head = vocab * h
    vh, vi, vd, vp, vm, text_h = 1024, 4096, 24, 14, 2, h
    vision = (3 * vp * vp * vh + vh + vd * (2 * vh + 4 * vh * vh + 3 * vh * vi)
              + vh + (vm * vm * vh) * vh + vh * text_h + text_h * text_h)
    assert language == 22_901_314_560 and vision == 438_958_080 and head == 671_088_640
    m = get_model("mistral-small-3.2-24b-instruct-2506")
    rel = load_release(m.hf_id)
    assert m.param_check()["rel_err"] == 0.0 and m.params() == language == m.release_params
    assert rel["params_llm"] == language + head and rel["params_vision"] == vision
    assert rel["n_tensors"] == 585 and rel["bytes_total"] == 48_022_722_560
    assert "tie_word_embeddings" not in rel["config"] and m.tie_embeddings is True
    assert m.mtp_layers == () and m.vision is not None and m.vision.params == vision
    g = image_grid(m.vision, 1024, 1024)
    assert (g.px, g.ph, g.pw, g.merged, g.llm_tokens) == ((1036, 1036), 74, 74, 1369, 1406)
    assert _weights(m) - rel["bytes_llm"] == -2 * (head + h)
    r = evaluate(Scenario(model=m.id, serving=Serving(phase="prefill", prompt=128, images=1)))
    assert r.vision["image_tokens"] == 1406 and r.vision["resized_px"] == [1036, 1036]


def test_ernie_4_5_0_3b_dense():
    h, heads, kv, hd, inter, vocab, layers = 1024, 16, 2, 128, 3072, 103424, 18
    q, kk = heads * hd, kv * hd
    attn = h * q + 2 * h * kk + q * h
    language = layers * (attn + 3 * h * inter + 2 * h) + vocab * h + h
    assert language == 360_748_032
    assert h // heads == 64 and hd == 128
    assert attn == 4_718_592 and 3 * h * inter == 9_437_184
    ids = ("ernie-4.5-0.3b-pt", "ernie-4.5-0.3b-base-pt")
    rels = []
    for mid in ids:
        m = get_model(mid)
        rel = load_release(m.hf_id)
        rels.append(rel)
        pc = m.param_check()
        assert pc["rel_err"] == 0.0 and m.params() == language == m.release_params
        assert rel["params_llm"] == language and rel["params_mtp"] == 0 and rel["params_vision"] == 0
        assert rel["bytes_llm"] == rel["bytes_total"] == 721_496_064 and rel["bytes_mtp"] == 0
        assert rel["n_tensors"] == 164 and rel["quantization_config"] is None
        assert rel["torch_dtype"] == "bfloat16"
        assert rel["config"]["model_type"] == "ernie4_5"
        assert rel["config"]["tie_word_embeddings"] is True and rel["config"]["use_bias"] is False
        assert "num_nextn_predict_layers" not in rel["config"]
        assert "lm_head" not in rel["roles"] and set(rel["roles"]) == {"attn", "embed", "mlp", "norm"}
        assert rel["roles"]["attn"]["bf16"]["params"] == layers * attn
        assert rel["roles"]["mlp"]["bf16"]["params"] == layers * 3 * h * inter
        assert rel["roles"]["embed"]["bf16"]["params"] == vocab * h
        assert rel["roles"]["norm"]["bf16"]["params"] == layers * 2 * h + h
        assert m.tie_embeddings is True and m.mtp_layers == () and m.mtp_params() == 0
        assert m.vision is None and m.n_layers == layers and m.hidden == h and m.vocab == vocab
        assert m.arch == "GQA 稠密" and m.coverage == "full" and m.coverage_reasons == ()
        assert m.act_fmt == "bf16" and m.kv_fmt == "bf16"
        assert m.fmt("attn").fmt == "bf16" and m.fmt("mlp").fmt == "bf16"
        assert m.fmt("embed").fmt == "bf16" and m.fmt("lm_head").fmt == "bf16"
        assert all(L.ffn.kind == "dense" and L.ffn.d_ff == inter and L.core.window is None for L in m.layers)
        L0 = m.layers[0]
        assert (L0.core.n_q, L0.core.n_kv, L0.core.qk_dim, L0.core.v_dim) == (heads, kv, hd, hd)
        assert [(lin.name, lin.k, lin.n) for lin in L0.attn_linears] == [
            ("q", h, q), ("k", h, kk), ("v", h, kk), ("o", q, h)]
        assert m.active_params() == language
        assert _weights(m) - rel["bytes_llm"] == -2 * h
        head_op = next(o for o in head_ops(m, Phase("decode", 1, 1, 8)) if o.name == "lm_head")
        assert head_op.role == "embed" and (head_op.k, head_op.n) == (h, vocab)
        bad = copy.deepcopy(rel["config"])
        bad["use_bias"] = True
        try:
            from_release("synth", rel["repo"], rel=rel, cfg=bad)
        except ValueError as exc:
            assert "use_bias" in str(exc)
        else:
            raise AssertionError("use_bias true was accepted")
        invented = copy.deepcopy(rel["config"])
        invented["num_nextn_predict_layers"] = 1
        try:
            from_release("synth", rel["repo"], rel=rel, cfg=invented)
        except ValueError as exc:
            assert "MTP" in str(exc)
        else:
            raise AssertionError("num_nextn_predict_layers was accepted")
        evaluate(Scenario(model=m.id, serving=Serving(phase="prefill", prompt=128)))
        evaluate(Scenario(model=m.id, serving=Serving(phase="decode", batch=2, ctx=128)))
        for ph, extra in (("prefill", {"prompt": 64}), ("decode", {"batch": 2})):
            s = A.api_eval({"scenario": {"model": m.id, "serving": {"phase": ph, **extra}},
                            "sensitivity": False})["summary"]
            assert s["step_ms"] > 0 and s["tok_s"] > 0
    assert rels[0]["config"] == rels[1]["config"]
    assert rels[0]["roles"] == rels[1]["roles"]
    assert rels[0]["repo"] == "baidu/ERNIE-4.5-0.3B-PT"
    assert rels[1]["repo"] == "baidu/ERNIE-4.5-0.3B-Base-PT"
    rows = {d["id"]: d for d in list_models()}
    for mid in ids:
        row = rows[mid]
        assert row["family"] == "ERNIE-4.5" and row["provenance"] == "official"
        assert row["provider"] == "baidu" and row["domain"] == "llm"
        assert row["mtp_layers"] == 0 and row["param_err"] == 0.0 and row["coverage"] == "full"
