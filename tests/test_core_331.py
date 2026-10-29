"""0.331: (1) one repo, several release files (catalog.RELEASE_KEYS) + PixText-Teacher; (2) PixText guidance interval
(3 of 4 MeanFlow steps / 22 of 25 Teacher steps guided) with the no-glyph negative branch at its own shape, through the
generic NativeWorkload.cfg_interval + uncond path (MOSS-Image unchanged); (3) design-report thermal sensitivity of the
air_oam perf/W front (0.858 vs 0.86 W/mm², limit not retuned)."""
import importlib.util
import os

from accel_dse.core import catalog as C
from accel_dse.core.domain import guided_steps
from accel_dse.core.energy import action_counts
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import load_release, release_path
from accel_dse.core.report import design_report, to_markdown
from accel_dse.core.scenario import Scenario
from accel_dse.core.thermal import COOLING

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHER_KEY = "ATH-MaaS/PixText:models/PixText-Teacher"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ev(mid, **kv):
    s = Scenario().replace("model", mid).replace("serving.batch", 1)
    for k, v in kv.items():
        s = s.replace(k.replace("__", "."), v)
    return evaluate(s)


def _f(t):
    """PixText DiT forward FLOPs at T tokens: GEMMs + dense attention (see test_core_330)."""
    return 2 * t * 32 * (3840 + 1280 + 7680 + 6826 + 3413) * 1280 + 4 * t * t * 1280 * 32


# ---------------------------------------------------------------- (1) release keys
def test_release_keys_regression():
    # 0.361: BlinkDL/rwkv-8-pile holds two checkpoints, same pattern as PixText.
    assert C.RELEASE_KEYS == {
        "pixtext-teacher": TEACHER_KEY,
        "rwkv8-pile": "BlinkDL/rwkv-8-pile:rc00",
        "rwkv8-pile-hybrid": "BlinkDL/rwkv-8-pile:rc00-hybrid",
    }
    es = C.entries()
    by = {e["id"]: e for e in es}
    for e in es:
        if e["id"] in C.RELEASE_KEYS:
            continue
        assert e["release"] == e["hf_id"], e["id"]          # every pre-0.331 model reads its hf_id's file, as before
        if "+" not in e["hf_id"]:
            assert release_path(e["hf_id"]).exists(), e["id"]
    t, mf = by["pixtext-teacher"], by["pixtext-meanflow"]
    assert t["hf_id"] == mf["hf_id"] == "ATH-MaaS/PixText"   # links / provider / freshness keep the public repo
    assert t["release"] == TEACHER_KEY and TEACHER_KEY in t["aliases"] and "ATH-MaaS/PixText" not in t["aliases"]
    assert mf["label"] == "PixText" and "ATH-MaaS/PixText" in mf["aliases"] and t["label"] == "PixText · Teacher"
    assert t["provider"] == mf["provider"] == "athmaas"
    assert release_path("ATH-MaaS/PixText").name == "ATH-MaaS__PixText.json"
    assert release_path(TEACHER_KEY).name == "ATH-MaaS__PixText_models__PixText-Teacher.json"
    assert release_path(TEACHER_KEY).exists() and release_path("ATH-MaaS/PixText").exists()
    assert C.get_model("ATH-MaaS/PixText").id == "pixtext-meanflow"
    assert C.get_model(TEACHER_KEY).id == "pixtext-teacher"
    # freshness orgs come from PROVIDERS only: the athmaas prefix is unchanged
    assert ("athmaas", "ATH-MaaS（PixText）", ("ATH-MaaS/",)) in C.PROVIDERS


def test_fetch_and_summarize_key_helpers():
    fh = _script("fetch_hf_release")
    assert fh.own_key("ATH-MaaS/PixText:models/PixText-Teacher!") == TEACHER_KEY
    assert fh.own_key("ATH-MaaS/PixText:models/PixText-MeanFlow") is None
    assert fh.release_stem(TEACHER_KEY) == release_path(TEACHER_KEY).stem
    assert fh.release_stem("ATH-MaaS/PixText") == release_path("ATH-MaaS/PixText").stem
    try:
        fh.own_key("ATH-MaaS/PixText!")
        raise AssertionError("'!' without a subfolder must fail")
    except ValueError:
        pass
    sr = _script("summarize_release")
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ATH-MaaS__PixText_models__PixText-Teacher"
        p.mkdir()
        (p / "source.json").write_text(json.dumps({"release_key": TEACHER_KEY}))
        assert sr._cache_key(p) == TEACHER_KEY
        q = Path(d) / "ATH-MaaS__PixText"
        q.mkdir()
        assert sr._cache_key(q) == "ATH-MaaS/PixText"
    lines = open(os.path.join(ROOT, "scripts", "release_repos.txt")).read().split()
    assert TEACHER_KEY + "!" in lines


def test_teacher_exact_params_and_47_forwards():
    rel = load_release(TEACHER_KEY)
    assert rel["source"]["release_key"] == TEACHER_KEY
    m = C.get_model("pixtext-teacher")
    assert m.params() == m.release_params == 951_979_840 and m.coverage == "full" and not m.coverage_reasons
    assert C.get_model("pixtext-meanflow").params() == 953_948_480          # unchanged
    io = {x.name for x in m.io_pre + m.io_post}
    assert not any(n.startswith("interval") for n in io)                   # Teacher: no interval embedder
    assert any("47 次 DiT 前向" in n for n in m.notes)
    r = _ev("pixtext-teacher")
    assert (r.workload.steps, r.workload.guided, r.workload.seqs_per_request) == (25, 22, 2)
    assert r.cfg_asym["guided_steps"] == 22 and not r.warnings
    tot = r._tflop_per_request() * 1e12
    assert abs(tot / (25 * _f(4488) + 22 * _f(4292)) - 1) < 0.01
    assert abs(r._tflop_per_request() - 540.45) < 0.01


# ---------------------------------------------------------------- (2) guidance interval
def test_guided_steps_strict_interval():
    assert guided_steps(4, (0.1, 1.0)) == 3        # k/4 = 0, .25, .5, .75 → 0 is not > 0.1
    assert guided_steps(25, (0.1, 1.0)) == 22      # k = 3 … 24 (k/25 > 0.1 strictly: 0.12 first)
    assert guided_steps(10, (0.1, 1.0)) == 8       # k/10 = 0.1 is not > 0.1 (strict)
    assert guided_steps(50, ()) == 50
    assert guided_steps(4, (0.0, 1.0)) == 3


def test_pixtext_meanflow_flop_old_to_new():
    r = _ev("pixtext-meanflow")
    assert (r.workload.steps, r.workload.guided, r.workload.seqs_per_request) == (4, 3, 2) and not r.warnings
    new = r._tflop_per_request()
    old = 82.87894831104                            # ≤ 0.330 main d31e573: 7 forwards at the conditional shape
    assert abs(new * 1e12 / (4 * _f(4488) + 3 * _f(4292)) - 1) < 0.01
    assert abs(new - 80.916) < 0.01 and abs(new / old - 1 + 0.0237) < 0.001   # 82.88 → 80.92 TF/req (−2.37 %)
    assert abs(7 * _f(4488) / (4 * _f(4488) + 3 * _f(4292)) - 1.0243) < 1e-4   # analytic: no-glyph shape alone
    assert r.cfg_asym["guided_steps"] == 3 and r.cfg_asym["guided_step_ms"] > 0


def test_moss_image_unchanged():
    r = _ev("moss-image-base")
    assert r.workload.guided == -1 and r.cfg_asym["guided_steps"] == r.workload.steps == 50
    assert abs(r._tflop_per_request() - 4953.8040889344) < 1e-6        # = 0.330 main


def test_energy_counts_follow_guided_steps():
    r = _ev("pixtext-meanflow")
    a = action_counts(r)["counts"]["mac"]
    r.cfg_asym["guided_steps"] = 4                  # every step guided → more MACs
    b = action_counts(r)["counts"]["mac"]
    assert b > a * 1.1


def test_same_branches_warns_interval_unmodelled():
    r = _ev("pixtext-meanflow", workload__cfg_branches="same")
    assert any("guidance 区间未建模" in w for w in r.warnings)


# ---------------------------------------------------------------- (3) thermal sensitivity
def test_report_front_thermal_margin_air_oam():
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None, cooling="air_oam", power_scope="card")
    lim = COOLING["air_oam"]["flux_W_mm2"]
    assert lim == 0.86                               # not retuned
    rep = design_report(o, {}, 0)
    th = next(s for s in rep["sections"] if s["id"] == "thermal")
    it = {i["key"]: i["value"] for i in th["items"]}
    # 0.333 re-pin (HBM 5.80 pJ/bit + per-channel standby, MODEL §278; front [1, 0, 15] → [4, 16], test_core_330): the
    # headline row is now over air (flux 0.9461, margin 0.0024 → −0.0861); front near-ties {0, 15} (0.858) → {16}
    # 0.465 re-pin: qwen3-8b vector time. flux 0.9461 → 0.9425, margin −0.0861 → −0.0825. Front stays {4, 16}.
    assert abs(it["flux_margin_W_mm2"] - (lim - it["flux_W_mm2"])) < 1e-12 and -0.083 < it["flux_margin_W_mm2"] < -0.082
    ft = th["front_thermal"]
    pts = {p["row"]: p for p in ft["points"]}
    assert set(pts) == set(o["front"]) == {4, 16}
    p = pts[16]
    assert round(p["flux_W_mm2"], 4) == 0.8543 and p["within_tie"] and p["margin"] < 0.01
    assert abs(p["margin_W_mm2"] - (0.86 - p["flux_W_mm2"])) < 1e-12
    assert not pts[4]["within_tie"] and pts[4]["margin"] > 0.01
    w = [x for x in rep["near_ties"] if x["kind"] == "front_thermal"]
    assert {x["text"].split("前沿点第 ")[1].split(" 行")[0] for x in w} == {"16"}
    md = to_markdown(rep)
    assert "前沿热敏感性" in md and "| 第 16 行 |" in md and "⚠" in md
