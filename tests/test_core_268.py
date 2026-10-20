"""0.268: √C slc_access_ref_mib deprecated (flagged, behaviour kept); Ming-3.0-asr-tiny (BailingMoeV3 KDA/MLA MoE
decoder + Whisper audio encoder) added with exact param match and a costed audio path."""
from dataclasses import replace
from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.energy import SLC_SQRTC_DEPRECATION, EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.insight import slc_energy_report
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.vision import audio_ops
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
M = "qwen3-30b-a3b"
ASR = "ming-3.0-asr-tiny"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_sqrtc_deprecated_flag_and_note():
    ch = replace(CHIPS["100T"], slc_mib=8 * 1024, slc_policy="dip")
    r = evaluate(Scenario(model=M, chip=ch, mem_id=LP, serving=Serving(batch=8)))
    sq = energy_report(r, _t(slc_access_ref_mib=1.0))["slc_access"]
    assert sq["deprecated"] is True and sq["note"] == SLC_SQRTC_DEPRECATION and "pJ_bit_mm_onchip" in sq["note"]
    wire = energy_report(r, _t(pJ_bit_mm_onchip=0.078))["slc_access"]
    assert "deprecated" not in wire and "note" not in wire           # the distance model is not flagged
    base = Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=8))
    notes = slc_energy_report(base, _t(slc_access_ref_mib=1.0), [M], gibs=(0, 8))["notes"]
    assert SLC_SQRTC_DEPRECATION in notes
    assert SLC_SQRTC_DEPRECATION not in slc_energy_report(base, _t(pJ_bit_mm_onchip=0.078), [M], gibs=(0, 8))["notes"]


def test_sqrtc_ui_marked():
    html, js = W.read(W.INDEX_HTML), W.read()          # 0.286: the control's own <label>, not a 1.5 kB text window
    assert W.label_text(html, "e-slc_access_ref_mib").startswith("已弃用")
    assert W.has(js, "slc_access.deprecated")


def test_ming_asr_exact_params_and_listing():
    m = get_model(ASR)
    assert m.param_check()["rel_err"] == 0.0 and m.params() == 7_893_392_800 and m.mtp_params() == 316_604_800
    kinds = [(l.core.kind, l.ffn.kind) for l in m.layers]
    assert len(kinds) == 24 and kinds[0] == ("linear", "dense") and kinds[3] == ("mla", "moe")
    assert sum(k == "mla" for k, _ in kinds) == 6
    v = m.vision
    assert v.family == "audio" and v.audio.kind == "whisper"
    # released header: tower + linear_proj_audio = 647,149,568 run; linear_proj (ignored on load) 8,653,824 stored
    assert v.audio.params == 647_149_568 and v.extra == 8_653_824 and v.params == 655_803_392
    e = next(x for x in list_models() if x["id"] == ASR)
    assert e["domain"] == "asr" and e["coverage"] == "full"


def test_ming_asr_tokens_chunks_and_cost():
    a = get_model(ASR).vision.audio
    # MingAudioSpec.token_count: 100 mel frames / s, 3000-frame chunks, conv2 s2 then projector conv k3 s2 p1
    assert a.tokens(1) == 25 and a.tokens(30) == 750 and a.tokens(30.5) == 763 and a.tokens(61) == 1525
    assert a.chunks(61) == (3000, 3000, 100)
    fl = sum(2 * o.m * o.k * o.n * (o.count or 1) for o in audio_ops(a, 30, 1) if o.kind in ("gemm", "attn"))
    assert 2.25e12 < fl < 2.32e12                                     # hand count 2.286 TFLOP per 30 s chunk
    h = REF_HW["B200"]
    ev = lambda **kw: evaluate(Scenario(model=ASR, chip=h.chip, mem_id=h.mem_id, serving=Serving(**kw)))
    r, r0 = ev(phase="prefill", batch=1, prompt=64, audio_s=30.0), ev(phase="prefill", batch=1, prompt=64)
    assert r.vision["audio"]["tokens"] == 750 and r.vision["image_tokens"] == 750 and r.ttft > r0.ttft
    ri = ev(phase="prefill", batch=1, prompt=64, images=1)
    assert any("serving.images 已忽略" in w for w in ri.warnings)
