"""0.347: adversarial MOSS first_audio + Omni 0.346 holds; sidecar_params → evaluate.fits; no visual UI."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.hardware import CHIPS
from accel_dse.core.qwen3_omni import code2wav_ops, talker_projection_ops
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.speech import speech_kpis
from accel_dse.core.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()
LP = "lpddr5x_4x64_8533_16g"


def test_omni_346_fixes_held():
    o = get_model("qwen3-omni-30b-a3b").qwen3_omni
    assert o.c2_window == 72 and o.thinker_hidden == 2048 and o.proj_inter == 2048
    qk = [x for x in code2wav_ops(o, 1, frames=1) if x.name.endswith(".attn.qk")]
    assert qk and all(x.n == 72 for x in qk)
    assert all(x.n == 72 for x in code2wav_ops(o, 1, frames=100) if x.name.endswith(".attn.qk"))
    gemm = [(x.name, x.k, x.n) for x in talker_projection_ops(o, 1, 64) if x.kind == "gemm"]
    assert gemm == [("talker.proj.fc1", 2048, 2048), ("talker.proj.fc2", 2048, 1024)]
    assert get_model("qwen3-omni-30b-a3b").sidecar_params == 0  # speech-card path


def test_moss_first_audio_uses_full_frame():
    s = Scenario(model="moss-tts", chip=CHIPS["1P"], mem_id=LP,
                 serving=Serving(batch=1, prompt=128, out_len=64, audio_s=10.0))
    k = speech_kpis(s, 2.0)
    assert k["kind"] == "moss_tts"
    assert abs(k["split_first"]["delay_s"] - k["delay_frames"] * k["frame_s"]) < 1e-12
    assert abs(k["first_audio_s"] - (k["ttft_s"] + k["delay_frames"] * k["frame_s"] + k["frame_s"])) < 1e-9


def test_sidecar_moss_and_moshi_in_evaluate_fits():
    moss = get_model("moss-tts")
    assert moss.sidecar_params == moss.moss_tts.audio_head_params
    assert moss.params() == moss.release_params  # sidecar excluded from params()
    s = Scenario(model="moss-tts", chip=CHIPS["1P"], mem_id=LP,
                 serving=Serving(batch=1, prompt=128, out_len=64, audio_s=10.0))
    pre = evaluate(s.replace("serving.phase", "prefill"))
    assert pre.fits
    # need includes sidecar (~0.25 GiB bf16)
    assert pre.stages[0].mem.dram_need >= moss.sidecar_params * 2

    z = get_model("moshiko-7b")
    assert z.sidecar_params == z.moshi.depth_params and z.params() == z.release_params
    s2 = Scenario(model="moshiko-7b", chip=CHIPS["1P"], mem_id=LP,
                  serving=Serving(batch=1, prompt=128, out_len=64, audio_s=10.0))
    dec = evaluate(s2.replace("serving.phase", "decode").replace("serving.ctx", 128))
    assert dec.fits and dec.stages[0].mem.dram_need >= z.sidecar_params * 2
    assert speech_kpis(s2, 2.0)["fits"]


def test_thin_wrap_other_speech_unchanged():
    assert get_model("qwen2.5-omni-7b").omni is not None
    assert get_model("qwen2.5-omni-7b").moss_tts is None
    assert get_model("moss-tts").moshi is None


def test_changelog_347():
    assert "## [0.347.0]" in CL
    sec = CL.split("## [0.347.0]")[1].split("## [")[0]
    assert "sidecar" in sec.lower() or "sidecar_params" in sec
    assert "MOSS" in sec or "moss" in sec.lower()
    assert "Nano" in sec or "nano" in sec.lower() or "deferred" in sec.lower() or "延" in sec
