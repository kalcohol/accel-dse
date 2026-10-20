"""0.279: test_core_237 deterministic phase-law work guard; Index-Echo S2ST 2B / 9B (AuT + connector, ST LM,
Hidden2CVMapper, CosyVoice3); Nemotron-3.5-Super-VL video (RADIO tubelets + EVS)."""
import json
import math
from pathlib import Path

from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.echo import ECHO_REASON, mapper_ops
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario
from accel_dse.core.speech import speech_kpis
from accel_dse.core.vision import radio_video_patches, video_grid_of

ROOT = Path(__file__).resolve().parents[1]
MEM = "lpddr5x_4x64_8533_16g"
ECHO = {"2b": ("IndexTeam__Index-Echo-S2ST-2B.json", 1_881_825_088, 656_315_778, 24_586_624, 2048),
        "9b": ("IndexTeam__Index-Echo-S2ST-9B.json", 8_953_803_264, 656_315_776, 26_425_728, 4096)}


def test_echo_params_exact():
    for k, (f, llm, vis, br, dh) in ECHO.items():
        m = get_model(f"index-echo-s2st-{k}")
        rel = json.loads((ROOT / "accel_dse/data/releases" / f).read_text())
        assert m.release_params == m.params() == rel["params_llm"] == llm
        assert m.vision_params == rel["params_vision"] == vis
        assert rel["params_bridge"] == br == m.echo.mapper_params
        assert m.echo.d_h == dh and m.echo.d_e == 896 and m.echo.n_layer == 3
        # AuT (Qwen3-Omni audio tower, 647,927,168) + connector: 2B w1/w2 2048² + 2 scalars, 9B proj [4096, 2048]
        assert vis - 647_927_168 == (2 * 2048 * 2048 + 2 if k == "2b" else 4096 * 2048)


def test_echo_coverage_partial_with_reason():
    for k in ("2b", "9b"):
        m = get_model(f"index-echo-s2st-{k}")
        assert m.coverage == "partial" and m.coverage_reasons == [ECHO_REASON] or tuple(m.coverage_reasons) == (ECHO_REASON,)


def test_mapper_ops_weights_exact():
    e = get_model("index-echo-s2st-2b").echo
    sh = dict(e.tensors)
    ops = mapper_ops(e, 30)
    w = sum(o.w_params for o in ops)
    assert w == e.mapper_params - math.prod(sh["pos"]) - math.prod(sh["lang_emb.weight"])


def test_speech_token_clip():
    e = get_model("index-echo-s2st-2b").echo
    assert e.speech_tokens(10, 5) == 20            # ≥ 2m
    assert e.speech_tokens(10, 1000) == 200        # ≤ 20m
    assert e.speech_tokens(100, 5000) == 1500      # ≤ 1500
    assert e.speech_tokens(30, 250) == 250


def test_s2st_kpis_split_and_fit():
    for k in ("2b", "9b"):
        s = Scenario(model=f"index-echo-s2st-{k}", chip=CHIPS["100T"], mem_id=MEM).replace("serving.audio_s", 10.0)
        p = speech_kpis(s, 10.0)
        assert p["kind"] == "s2st" and p["fits"]
        assert abs(sum(p["split"].values()) - p["total_s"]) < 1e-12 and p["first_audio_s"] == p["total_s"]
        assert abs(p["rtf"] - p["total_s"] / 10.0) < 1e-12
        assert all(v >= 0 for v in p["split"].values())
        assert p["speech_tokens"] == 250 and p["tgt_tokens"] == 30
    assert "speech_tokenizer_v3" in ECHO_REASON and "CampPlus" in ECHO_REASON


def test_echo_catalog_listing():
    listed = {x["id"]: x for x in list_models()}
    for k in ("2b", "9b"):
        x = listed[f"index-echo-s2st-{k}"]
        assert x["domain"] == "omni"
        assert get_model(f"index-echo-s2st-{k}").hf_id.startswith("IndexTeam/")


def test_fetch_echo_present():
    src = (ROOT / "scripts/fetch_pt_release.py").read_text()
    assert "def _distcp_shapes" in src and "def fetch_echo" in src and "--echo" in src


def test_237_size_helper():
    import importlib.util
    spec = importlib.util.spec_from_file_location("t237", ROOT / "tests/test_core_237.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    assert t._size([1, (2, 3), {"a": [4, 5]}]) >= 3
    assert t._size(5) == 0 or t._size(5) == 1
    assert len(t._PHASE_FNS) == 8 and set(t._PHASE_WORK) == {"sglang_mixed", "sglang", "vllm_v0", "vllm"}


def test_radio_video_patch_rule():
    # _compute_target_patches_video: 16:9 → ph round(√(1024/1.778)) = 24, pw round(√1820.4) = 43 → up (24, 44) = 1056 >
    # 1024 → down (24, 42); square → 32 × 32
    assert radio_video_patches(1280, 720, 1024, 2) == (42, 24)
    assert radio_video_patches(1024, 1024, 1024, 2) == (32, 32)
    assert radio_video_patches(720, 1280, 1024, 2) == (24, 42)


def test_nemotron_video_tokens_and_cost():
    M = "nemotron-3.5-super-vl-120b-a12b-nvfp4"
    m = get_model(M)
    s = Scenario(model=M, chip=CHIPS["100T"], mem_id=MEM).replace("layout.tp", 2).replace("layout.ep", 2) \
        .replace("serving.phase", "prefill")
    sv = s.replace("serving.videos", 1).serving
    vg = video_grid_of(m, sv)
    assert (vg.frames, vg.t, vg.slice.ph, vg.slice.pw, vg.slice.merged) == (32, 16, 24, 42, 252)
    # EVS (q 0.7): keep max(252, ⌊16·252·0.3⌋) = 1209 video tokens + <img></img> per tubelet
    assert vg.llm_tokens == 1209 + 32
    r0, r1 = evaluate(s), evaluate(s.replace("serving.videos", 1))
    v = r1.vision["video"]
    assert v["slices"] == 16 and v["frames_per_slice"] == 2 and v["s"] > 0
    assert r1.ttft > r0.ttft
    # odd frame count: last tubelet padded
    assert video_grid_of(m, s.replace("serving.videos", 1).replace("serving.video_frames", 5).serving).t == 3
