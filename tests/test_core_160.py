"""0.160: joint sizing (interacting minima) and runner robustness."""
import json
import math
import os
import importlib.util
from pathlib import Path

from accel_dse import api as A
from accel_dse.core.budget import Budget, DENSITY_PRESETS
from accel_dse.core.insight import SIZING, TIE, _JOINT_ORDER, _gain, _point, model_scenario, sizing_report
from accel_dse.core.energy import EnergyTable
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)


def _worst(base, models, vals, domain, mets):
    w = math.inf
    for mid in models:
        s = model_scenario(base, mid)
        r = _point(s, EnergyTable(), domain)
        t = s
        for k in _JOINT_ORDER:
            if k in vals:
                t = SIZING[k][3](t, vals[k])
        p = _point(t, EnergyTable(), domain)
        if r["fits"] and not p["fits"]:
            return -math.inf
        w = min([w] + [g for m in mets if (g := _gain(m, r, p)) is not None])
    return w


def test_joint_point_holds_and_stays_between_solo_and_current():
    base = Scenario()
    for models in (["esm2-650m"], ["wan2.1-1.3b"], ["qwen3-8b", "gpt-oss-20b"]):
        r = sizing_report(base, models, joint=True)
        j = r["joint"]
        assert j["holds"] and j["worst_gain"] >= 1 - TIE
        vals = {e["key"]: e["joint"] for e in j["resources"]}
        assert _worst(base, models, vals, r["domain"], list(r["metrics"])) >= 1 - TIE      # independently re-checked
        cur = {x["key"]: x["current"] for x in r["resources"]}
        for e in j["resources"]:
            assert e["solo_min"] <= e["joint"] <= cur[e["key"]]
        if not j["all_minima_hold"]:
            assert j["backoffs"] and j["all_minima_worst_gain"] < 1 - TIE


def test_minima_interact_on_default_chip():
    # mechanism: solo minima are not jointly feasible (shared elementwise / memory time) — the joint pass matters
    j = sizing_report(Scenario(), ["esm2-650m"], joint=True)["joint"]
    assert not j["all_minima_hold"] and j["all_minima_worst_gain"] < 0.9


def test_joint_area_total_and_api():
    dens = DENSITY_PRESETS["n7_tpuv4i"]["fields"]
    b = Budget(mm2_per_kmac=dens["mm2_per_kmac"][0], mm2_per_mib_sram=dens["mm2_per_mib_sram"][0])
    j = sizing_report(Scenario(), ["wan2.1-1.3b"], b, joint=True)["joint"]
    assert abs(j["mm2_saved_total"] - sum(e.get("mm2_saved", 0.0) for e in j["resources"])) < 1e-9
    assert "joint" not in A.api_insight_sizing({"models": ["esm2-650m"], "resources": ["lanes"]})
    r = A.api_insight_sizing({"models": ["esm2-650m"], "joint": True})
    assert r["joint"]["holds"]
    web = Path(A.__file__).parent / "web"
    assert "ins-size-joint" in W.html_ids(W.read(W.INDEX_HTML)) and W.has(W.read(), "j.all_minima_hold")   # 0.286: parsed


def _runner():
    p = Path(A.__file__).resolve().parent.parent / "scripts" / "run_tests.py"
    spec = importlib.util.spec_from_file_location("run_tests_script", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_runner_reports_dead_previous_run(tmp=Path("/tmp/accel_runner_test")):
    m = _runner()
    tmp.mkdir(exist_ok=True)
    m.STATUS = tmp / "status.json"
    m.DUR = tmp / "dur.json"
    dead = 2 ** 22 + 12345
    while m._alive(dead):
        dead += 1
    me = Path(__file__).resolve()
    m.STATUS.write_text(json.dumps({"state": "running", "pid": dead, "started": "x", "total": 1,
                                    "files": [me.name], "done": {}}))
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.main(["-j", "1", str(Path(__file__).resolve().parent / "test_core_084.py")])
    out = buf.getvalue()
    assert rc == 0 and "died without a summary" in out and "passed, 0 file(s) failed" in out
    assert json.loads(m.STATUS.read_text())["state"] == "done"
    assert m._alive(os.getpid())
