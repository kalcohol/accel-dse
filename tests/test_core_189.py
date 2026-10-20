"""0.189: compound provenance tags; static-energy line/warning; opt-in TDP power cap."""
from accel_dse import api as A
from accel_dse.core.provenance import valid_tag
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, energy_report, apply_energy_preset
from accel_dse.core.budget import DENSITY_PRESETS
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_tags():
    for t in ("来源", "推导/二手", "二手/假设"):
        assert valid_tag(t)
    for t in ("", "推导/", "二手/二手", "猜", None, "来源 "):
        assert not valid_tag(t)
    assert ENERGY_PRESETS["tpuv4i_7nm"]["fields"]["pJ_bit_phy_lpddr"][1] == "推导/二手"
    for p in list(ENERGY_PRESETS.values()) + list(DENSITY_PRESETS.values()):
        for f, v in p["fields"].items():
            assert valid_tag(v[1]), (f, v[1])


def test_static_line_and_warning():
    r = evaluate(Scenario())
    e = energy_report(r, _t(tdp_W=175.0))
    assert abs(e["static_share"] - e["J_by_action"]["idle"] / e["J_per_unit"]) < 1e-12
    assert abs(e["J_dynamic_per_unit"] + e["J_by_action"]["idle"] - e["J_per_unit"]) < 1e-12
    assert e["static_share"] > 0.5 and "static_warning" in e and "H100" in e["idle_source_note"]
    e = energy_report(r, _t(tdp_W=175.0, idle_frac_tdp=0.001))
    assert "static_warning" not in e and "用户填写" in e["idle_source_note"]
    e = energy_report(r, _t(idle_W=1.0))
    assert "idle_source_note" not in e


def test_power_cap():
    r = evaluate(Scenario())
    base = energy_report(r, _t(tdp_W=175.0))
    assert "power_cap" not in base                                       # opt-in
    assert "error" in energy_report(r, _t(), power_cap=True)["power_cap"]  # no TDP default
    hi = energy_report(r, _t(tdp_W=1e5), power_cap=True)["power_cap"]
    assert not hi["capped"] and hi["throughput_factor"] == 1.0
    pd = hi["P_dyn_W"]
    tdp = 50.0
    pc = energy_report(r, _t(tdp_W=tdp, idle_frac_tdp=0.1), power_cap=True)["power_cap"]
    if pd > tdp * 0.9:
        assert pc["capped"] and abs(pc["slowdown"] - pd / (tdp * 0.9)) < 1e-9
        assert abs(pc["P_dyn_W"] / pc["slowdown"] + pc["P_static_W"] - tdp) < 1e-6   # fits exactly at cap
    pc = energy_report(r, _t(tdp_W=10.0, idle_W=10.0), power_cap=True)["power_cap"]
    assert pc["feasible"] is False
    # monotone: lower TDP never gives more throughput
    f = [energy_report(r, _t(tdp_W=t), power_cap=True)["power_cap"]["throughput_factor"] for t in (2.0 * pd, pd, pd / 2)]
    assert f[0] >= f[1] >= f[2] and f[2] < 1


def test_api_power_cap():
    b = {"energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"tdp_W_card": 175}}
    assert "power_cap" not in A.api_eval(b)["energy"]
    assert "power_cap" in A.api_eval({**b, "power_cap": True})["energy"]
    try:
        A.api_eval({**b, "power_cap": "yes"}); assert False
    except A.ApiError:
        pass
    js = W.read()                                      # 0.286: parsed
    assert "e-power_cap" in W.html_ids(W.read(W.INDEX_HTML)) and W.has(js, "power_cap: true") and W.mentions(js, "static_warning")


def test_190_preset_fraction_rounded_still_cited():
    """0.190 (UI smoke finding): the UI sends the preset 0.1026 rounded; the note must still cite the H100 source."""
    e = energy_report(evaluate(Scenario()), _t(tdp_W=175.0, idle_frac_tdp=0.1026))
    assert "H100" in e["idle_source_note"]
    e = energy_report(evaluate(Scenario()), _t(tdp_W=175.0, idle_frac_tdp=0.11))
    assert "用户填写" in e["idle_source_note"]


def test_190_cli_power_cap():
    import subprocess, sys
    o = subprocess.run([sys.executable, "-m", "accel_dse.cli", "eval", "--pJ-mac", "0.5", "--pJ-bit-dram", "4",
                        "--idle-frac-tdp", "0.103", "--tdp-W", "5", "--power-cap"], capture_output=True, text=True)
    assert o.returncode == 0 and "power cap:" in o.stdout and "throughput ×0." in o.stdout
    o = subprocess.run([sys.executable, "-m", "accel_dse.cli", "eval", "--pJ-mac", "0.5", "--power-cap"],
                       capture_output=True, text=True)
    assert o.returncode == 0 and "power_cap 需要 tdp_W" in o.stdout
