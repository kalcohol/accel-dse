"""0.158: sizing_report — MAC array (cores) and DRAM bandwidth; MAC area from density presets only."""
from accel_dse import api as A
from accel_dse.core.budget import Budget, DENSITY_PRESETS
from accel_dse.core.insight import SIZING, TIE, _gain, _point, model_scenario, sizing_report
from accel_dse.core.energy import EnergyTable
from accel_dse.core.scenario import Scenario


def _holds(base, models, key, v, mets, domain):
    for mid in models:
        s = model_scenario(base, mid)
        r, p = _point(s, EnergyTable(), domain), _point(SIZING[key][3](s, v), EnergyTable(), domain)
        if (r["fits"] and not p["fits"]) or any((g := _gain(m, r, p)) is not None and g < 1 - TIE for m in mets):
            return False
    return True


def test_cores_and_bw_bracketed():
    base = Scenario()
    for models in (["esm2-650m"], ["wan2.1-1.3b"]):
        r = sizing_report(base, models, resources=["cores", "dram_bw"])
        for x in r["resources"]:
            assert 0 < x["min_ok"] <= x["current"]
            assert _holds(base, models, x["key"], x["min_ok"], list(r["metrics"]), r["domain"])
            if not x["floor_hit"]:
                assert not _holds(base, models, x["key"], x["fail_at"], list(r["metrics"]), r["domain"])


def test_setters_isolate_the_resource():
    s = Scenario()
    c = SIZING["cores"][3](s, 4).chip
    assert c.engines == 4 and c.lanes == s.chip.lanes and c.rows == s.chip.rows      # vector width held
    b = SIZING["dram_bw"][3](s, SIZING["dram_bw"][2](s) / 2)
    assert abs(SIZING["dram_bw"][2](b) - SIZING["dram_bw"][2](s) / 2) < 1e-6 and b.mem_id == s.mem_id


def test_mac_area_only_from_density():
    dens = DENSITY_PRESETS["n7_tpuv4i"]["fields"]["mm2_per_kmac"][0]
    r = sizing_report(Scenario(), ["esm2-650m"], Budget(mm2_per_kmac=dens), resources=["cores", "dram_bw"])
    cores, bw = r["resources"]
    ch = Scenario().chip
    assert abs(cores["mm2_saved"] - (cores["current"] - cores["min_ok"]) * ch.rows * ch.cols / 1000 * dens) < 1e-9
    assert "mm2_saved" not in bw
    assert all("mm2_saved" not in x for x in sizing_report(Scenario(), ["esm2-650m"], resources=["cores"])["resources"])


def test_api_density_preset():
    r = A.api_insight_sizing({"models": ["esm2-650m"], "resources": ["cores"], "budget": {"density_preset": "n7_tpuv4i"}})
    assert r["resources"][0]["key"] == "cores" and "mm2_saved" in r["resources"][0]


def test_parallel_runner_covers_every_test_file():
    import ast
    from pathlib import Path
    root = Path(A.__file__).resolve().parent.parent
    src = (root / "scripts" / "run_tests.py").read_text()
    ast.parse(src)
    assert "pytest" not in src.split('"""', 2)[2]                    # 0.159: stdlib only
    assert 'glob("test_*.py")' in src and "TIMEOUT" in src and "-k" not in src.split("def run")[1].split("def main")[0]
