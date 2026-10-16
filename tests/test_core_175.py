"""0.175: Muse Spark Open follow-ups — NVFP4 release (derived), NaViT block-window attention, audio encoder, DSpark."""
import json
import sys
sys.path.insert(0, ".")
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.dtypes import FORMATS
from accel_dse.core.parallel import Layout
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.evaluate import evaluate
from accel_dse.core.vision import image_grid, tile_keys, vision_ops, audio_ops

LAY = Layout(tp=8, ep=8)


def _ev(mid, hw="B200", **kw):
    h = REF_HW[hw]
    return evaluate(Scenario(model=mid, chip=h.chip, mem_id=h.mem_id, layout=LAY, serving=Serving(**kw)))


def test_nvfp4_release_bytes_and_format():
    """NVFP4 = e2m1 (0.5 B) + e4m3 scale per 16 (1/16 B) per param + 2 fp32 per tensor; experts only (假设)."""
    assert FORMATS["nvfp4"].bits == 4.5
    d = json.load(open("accel_dse/data/releases/meta-models__Muse-Spark-Open-816B-A42B-NVFP4.json"))
    e = d["roles"]["expert"]["nvfp4"]
    assert e["bytes"] == e["params"] // 2 + e["params"] // 16 + 256 * 3 * 62 * 8
    assert "推导" in d["derived"] and "假设" in d["derived"]
    m = get_model("muse-spark-open-nvfp4")
    f = dict(m.formats)
    assert f["expert"].fmt == "nvfp4" and abs(f["expert"].bits - 4.5) < 1e-3 and f["attn"].fmt == "bf16"
    assert abs(m.param_check()["rel_err"]) < 1e-5
    # 4-bit experts: fits 8×B200 where BF16 does not; decode faster than FP8-expert on the same chips
    assert _ev("muse-spark-open-nvfp4", phase="decode", batch=1, ctx=8192).fits
    assert not _ev("muse-spark-open", phase="decode", batch=1, ctx=8192).fits


def test_summarize_release_detects_nvfp4():
    src = open("scripts/summarize_release.py").read()
    assert '"nvfp4" if nv4 else "mxfp4"' in src


def test_navit_tiles():
    """PR vision.py _get_sparse_attention_perm: window layers attend within 32×32 patch tiles (edge tiles partial)."""
    v = get_model("muse-spark-open").vision
    assert v.tile == 32 and len(v.full_idx) == 13 and v.full_idx[-1] == 49
    assert tile_keys(64, 64, 32) == 4 * 1024 ** 2
    assert tile_keys(33, 32, 32) == 1024 ** 2 + 32 ** 2
    g = image_grid(v, 1920, 1080)
    ops = {o.name: o for o in vision_ops(v, g, 1)}
    N = g.ph * g.pw
    assert ops["vis.0.qk"].causal < 0.2 and ops["vis.3.qk"].causal == 1.0          # layer 3 is full attention
    assert ops["vis.win_perm"].vec > 0


def test_audio_encoder():
    """PR audio.py AutoEncoder: 117.84M params; 60 s @ 24 kHz → 750 tokens; FLOPs hand-checked ≈ 0.59 TFLOP."""
    a = get_model("muse-spark-open").vision.audio
    assert a.params == 117_840_384 and a.samples_per_token == 1920 and a.tokens(60) == 750
    fl = sum(2 * o.m * o.k * o.n for o in audio_ops(a, 60, 1) if o.kind == "gemm")
    assert 0.55e12 < fl < 0.62e12
    r = _ev("muse-spark-open-nvfp4", phase="prefill", batch=1, prompt=1024, audio_s=60.0)
    r0 = _ev("muse-spark-open-nvfp4", phase="prefill", batch=1, prompt=1024)
    assert r.vision["audio"]["tokens"] == 750 and r.vision["image_tokens"] == 750 and r.ttft > r0.ttft
    # text-only model: audio ignored with a warning
    assert any("audio_s" in w for w in _ev("muse-spark-open-nvfp4", phase="prefill", batch=1, prompt=8).warnings) is False


def test_dspark_draft_exact_params():
    """Inferact/Muse-Spark-Open-DSpark safetensors header: 5,301,633,537 bf16 params (+ I64 d2t map)."""
    m = get_model("muse-spark-open-nvfp4-dspark")
    assert m.draft == "dspark" and m.draft_block == 7 and m.draft_targets == 5 and m.draft_markov == 512
    assert len(m.mtp_layers) == 5 and m.mtp_layers[0].core.window == 2048
    assert abs(m.mtp_params() - 5_301_633_537) <= 5
    base = _ev("muse-spark-open-nvfp4", phase="decode", batch=1, ctx=8192)
    sp = _ev("muse-spark-open-nvfp4-dspark", phase="decode", batch=1, ctx=8192, spec_k=7, spec_accept=0.9)
    assert sp.tpot < base.tpot
    # no spec → identical to the target alone (draft weights only add memory)
    assert abs(_ev("muse-spark-open-nvfp4-dspark", phase="decode", batch=1, ctx=8192).tpot - base.tpot) < 1e-9
    assert "muse-spark-open-nvfp4-dspark" in {e["id"] for e in list_models()}
