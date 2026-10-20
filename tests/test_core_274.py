"""0.274 speech round 5: Kyutai Moshi (temporal 7B + per-frame depformer + Mimi codec, full duplex) and the
cross-model speech insight (core/speech_insight.py, /api/insight_speech)."""
import json
from pathlib import Path

from accel_dse import api, serve
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import _component_time, evaluate
from accel_dse.core.hardware import CHIPS, System
from accel_dse.core.moshi import (FRAME_S, _seanet_rows, depformer_ops, depth_weight_bytes, mimi_decode_ops,
                                  mimi_encode_ops, mimi_weight_bytes)
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import _stages, speech_bounds

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
M = "moshiko-7b"
REL = json.loads((ROOT / "accel_dse/data/releases/kyutai__moshiko-pytorch-bf16.json").read_text())


def _scn(**kw):
    return Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(**{"batch": 1, "prompt": 1, "out_len": 1, **kw}))


def test_config_transcription_matches_header_shapes():
    """The HF repo has no config: lm / mimi blocks are transcribed from loaders.py; every number is checked against
    the released tensor shapes here."""
    c, st = REL["config"], REL["speech_tensors"]
    lm, mi = c["lm"], c["mimi"]
    assert c["model_type"] == "moshi" and "loaders.py" in c["_source"]
    d, dd = lm["dim"], lm["depformer_dim"]
    assert st["depformer.layers.0.self_attn.in_proj_weight"] == [3 * dd * lm["dep_q"], dd]          # weights_per_step
    assert st["depformer.layers.0.self_attn.out_proj.weight"] == [dd * lm["dep_q"], dd]
    assert sum(1 for n in st if n.startswith("depformer.layers.0.gating.") and n.endswith("linear_in.weight")) == lm["dep_q"]
    assert sum(1 for n in st if n.startswith("depformer_in.")) == lm["dep_q"]
    assert st["depformer_in.0.weight"] == [dd, d] and st["linears.0.weight"] == [lm["card"], dd]
    assert st["depformer_text_emb.weight"] == [lm["text_card"] + 1, dd]
    assert len({n.split(".")[2] for n in st if n.startswith("depformer.layers.")}) == lm["depformer_num_layers"]
    t = mi["transformer"]
    assert st["mimi.encoder_transformer.transformer.layers.0.self_attn.in_proj_weight"] == [3 * t["d_model"], t["d_model"]]
    assert st["mimi.encoder_transformer.transformer.layers.0.linear1.weight"] == [t["dim_feedforward"], t["d_model"]]
    assert st["mimi.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"] == [mi["quantizer"]["bins"], mi["quantizer"]["dimension"]]
    m = get_model(M)
    assert (m.hidden, m.n_layers, m.vocab) == (d, lm["num_layers"], lm["text_card"])


def test_exact_params_and_frame_bytes():
    m = get_model(M)
    assert m.params() == REL["params_llm"] == 6_973_366_272
    k = m.moshi
    assert k.depth_params == REL["params_depth"] == 714_362_880
    assert k.mimi_params == REL["params_codec"] == 96_151_393
    # per frame the depformer reads every GEMM weight once; embedding tables are row lookups
    tables = sum(v[0] * v[1] for n, v in REL["speech_tensors"].items() if n.startswith(("depformer_emb.", "depformer_text_emb")))
    w = sum(op.w_params for op in depformer_ops(k, 1) if op.kind == "gemm")
    assert w == 666_906_624 and w + tables == k.depth_params
    assert abs(depth_weight_bytes(k) - 2 * w) / (2 * w) < 0.01
    # Mimi fp32: encoder + decoder weights, unused codebooks (24 of 32) are not read
    assert mimi_weight_bytes(k) < 4 * k.mimi_params


def test_seanet_rows_per_frame():
    k = get_model(M).moshi
    assert k.samples_per_frame == 1920 and k.enc_rate == 2
    enc = {i: r[0] for i, r in _seanet_rows(k, "encoder").items()}
    assert enc[0] == 1920 and enc[3] == 480 and enc[6] == 96 and enc[9] == 16 and enc[12] == 2
    dec = {i: r[0] for i, r in _seanet_rows(k, "decoder").items()}
    assert dec[0] == 2 and dec[14] == 1920


def test_frame_kpi_formula():
    s = _scn()
    p = speech_kpis(s)
    assert p["kind"] == "moshi" and p["frame_ms"] == 80.0 and p["ctx_frames"] == 3000
    f = p["split_frame"]
    assert abs(p["frame_s"] - sum(f.values())) < 1e-12
    dec = evaluate(s.replace("serving.phase", "decode").replace("serving.ctx", 3000))
    assert abs(f["temporal_s"] - dec.tpot) < 1e-12
    k = get_model(M).moshi
    sys = System(s.chip, s.mem_id, s.mem_eff)
    assert abs(f["depformer_s"] - _component_time([(depformer_ops(k, 1), 1)], sys, s.mapping, "bf16")["s"]) < 1e-12
    assert abs(f["mimi_encode_s"] - _component_time([(mimi_encode_ops(k, 1), 1)], sys, s.mapping, "fp32")["s"]) < 1e-12
    assert abs(f["mimi_decode_s"] - _component_time([(mimi_decode_ops(k, 1), 1)], sys, s.mapping, "fp32")["s"]) < 1e-12
    assert abs(p["rtf"] - p["frame_s"] / FRAME_S) < 1e-12
    assert abs(p["first_audio_s"] - (0.16 + p["frame_s"])) < 1e-12
    # 100T + LPDDR5X: weight re-read bound, ~8 % short of real time at a full KV ring buffer
    assert 1.0 < p["rtf"] < 1.2 and p["streams_realtime"] == 0
    assert f["temporal_s"] > f["depformer_s"] > f["mimi_encode_s"]
    # a short conversation (KV 100 frames) is real time
    assert speech_kpis(_scn(ctx=100))["rtf"] < 1.0


def test_api_eval_has_speech_and_ui():
    out = api.api_eval({"scenario": {"model": M, "mem_id": LP, "chip": {"name": "100T"},
                                     "serving": {"batch": 1, "prompt": 1, "out_len": 1}}, "sensitivity": False})
    assert out["speech"]["kind"] == "moshi"
    js = (ROOT / "accel_dse/web/app.js").read_text()
    assert "p.kind === 'moshi'" in js and "m.family === 'Moshi'" in js
    assert M in (ROOT / "scripts/ui_headless_smoke.py").read_text()


def test_speech_insight_mechanism():
    base = _scn()
    out = speech_bounds(base, ("moshiko-7b", "cosyvoice2-0.5b", "whisper-large-v3"), 10.0)
    bw = out["dram_GBps_eff"]
    rows = {r["model"]: r for r in out["rows"]}
    mo = rows[M]
    assert mo["binding"] == "frame" and not mo["realtime"] and mo["rereads_per_audio_s"] == 12.5
    assert abs(mo["dram_GBps_eff_needed"] - bw * mo["x"]) < 1e-9
    assert abs(mo["slc_GB_needed"] - (mo["x"] - 1) * bw / 12.5) < 1e-9
    assert mo["verify"]["realtime"]                     # the engine's own SLC path confirms the formula
    assert rows["cosyvoice2-0.5b"]["realtime"] and "slc_GB_needed" not in rows["cosyvoice2-0.5b"]
    assert rows["whisper-large-v3"]["kind"] == "asr" and rows["whisper-large-v3"]["realtime"]
    st, r, _, _ = _stages({"kind": "tts", "rtf": 2.0, "steps": 125, "audio_out_s": 10.0, "first_audio_s": 0.1}, 10.0)
    assert st == {"lm": 2.0} and r == 12.5


def test_api_insight_speech_route():
    assert "/api/insight_speech" in serve.POST_ROUTES
    out = api.api_insight_speech({"scenario": {"model": M, "mem_id": LP, "chip": {"name": "100T"}},
                                  "models": ["ming-3.0-tts-tiny"], "verify": False})
    assert out["rows"][0]["model"] == "ming-3.0-tts-tiny" and "hash" in out
    for bad in ({"models": []}, {"audio_s": 0}, {"verify": 1}, {"nope": 1}):
        try:
            api.api_insight_speech({"scenario": {"model": M, "mem_id": LP, "chip": {"name": "100T"}}, **bad})
        except api.ApiError:
            continue
        raise AssertionError(bad)
