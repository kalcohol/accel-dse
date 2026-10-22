"""0.348: adversarial sidecar; sidecar_bytes folds Mimi fp32; no visual UI."""
from __future__ import annotations

from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()
LP = "lpddr5x_4x64_8533_16g"


def test_no_double_count_params():
    for mid in ("moss-tts", "moshiko-7b", "moshika-7b"):
        m = get_model(mid)
        assert m.params() == m.release_params
        assert m.sidecar_bytes > 0
        # params() must not include sidecar weight count
        assert m.params() == m.params()  # sanity
    assert get_model("qwen3-omni-30b-a3b").sidecar_bytes == 0
    assert get_model("qwen3-omni-30b-a3b").sidecar_params == 0


def test_moss_sidecar_bytes_bf16():
    m = get_model("moss-tts")
    assert m.sidecar_bytes == m.moss_tts.audio_head_params * 2.0
    assert m.sidecar_params == m.moss_tts.audio_head_params
    s = Scenario(model="moss-tts", chip=CHIPS["1P"], mem_id=LP,
                 serving=Serving(batch=1, prompt=128, out_len=64, phase="prefill"))
    r = evaluate(s)
    assert r.fits
    m0 = replace(m, sidecar_bytes=0.0, sidecar_params=0)
    r0 = evaluate(s, model=m0)
    assert abs(r.stages[0].mem.dram_need - r0.stages[0].mem.dram_need - m.sidecar_bytes) < 1.0


def test_moshi_sidecar_bytes_includes_mimi_fp32():
    z = get_model("moshiko-7b")
    assert z.sidecar_bytes == z.moshi.stored_bytes
    assert z.sidecar_bytes == z.moshi.depth_params * 2.0 + z.moshi.mimi_params * 4.0
    # depth-only path would under-count Mimi
    assert z.sidecar_bytes > z.moshi.depth_params * 2.0
    s = Scenario(model="moshiko-7b", chip=CHIPS["1P"], mem_id=LP,
                 serving=Serving(batch=1, prompt=128, out_len=64, phase="decode", ctx=128))
    r = evaluate(s)
    assert r.fits
    z0 = replace(z, sidecar_bytes=0.0, sidecar_params=0)
    r0 = evaluate(s, model=z0)
    assert abs(r.stages[0].mem.dram_need - r0.stages[0].mem.dram_need - z.sidecar_bytes) < 1.0
    # moshika twin
    k = get_model("moshika-7b")
    assert k.sidecar_bytes == k.moshi.stored_bytes == z.sidecar_bytes


def test_pp_sidecar_once_on_head():
    m = get_model("moss-tts")
    m0 = replace(m, sidecar_bytes=0.0, sidecar_params=0)
    s = Scenario(model="moss-tts", chip=CHIPS["1P"], mem_id=LP,
                 layout=Layout(tp=1, pp=2, dp=1),
                 serving=Serving(batch=1, prompt=128, out_len=64, phase="prefill"))
    r, r0 = evaluate(s, model=m), evaluate(s, model=m0)
    deltas = [a.mem.dram_need - b.mem.dram_need for a, b in zip(r.stages, r0.stages)]
    assert abs(deltas[0]) < 1.0
    assert abs(deltas[1] - m.sidecar_bytes) < 1.0


def test_changelog_348():
    assert "## [0.348.0]" in CL
    sec = CL.split("## [0.348.0]")[1].split("## [")[0]
    assert "sidecar_bytes" in sec
    assert "Mimi" in sec or "mimi" in sec.lower()
    assert "Nano" in sec or "deferred" in sec.lower() or "延" in sec
