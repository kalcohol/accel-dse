"""0.335: LPDDR device-side active pJ/bit (Micron IDD4R−IDD2N, MODEL §280) aligned with HBM 0.333 split;
0.334 thermal device_static / standby knobs apply to LPDDR too."""
import os

from accel_dse.core.energy import ENERGY_PRESETS, _LPDDR_ACTIVE_PJ, EnergyTable, hbm_static_W
from accel_dse.core.energy_dse import MEM_PROXY
from accel_dse.core.powercap import hbm_device_static_W, hbm_standby_W
from accel_dse.core.thermal import package_power

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]


def test_lpddr_preset_is_idd_active_not_grace_allin():
    v, tag, src = F["pJ_bit_dram_lpddr"]
    assert abs(v - _LPDDR_ACTIVE_PJ) < 1e-12 and abs(v - 3.2298828125000005) < 1e-9
    assert tag == "推导" and "IDD4R" in src and "Y52Q" in src and "Grace" in src and "4.0" in src
    v6, tag6, src6 = F["pJ_bit_dram_lpddr6"]
    assert abs(v6 - _LPDDR_ACTIVE_PJ / 1.21) < 1e-12 and tag6 == "推导" and "1.21" in src6
    assert "2.67" in MEM_PROXY["LPDDR6"] or "3.23" in MEM_PROXY["LPDDR6"]


def test_lpddr_static_feeds_thermal_knobs():
    t = EnergyTable(mW_lpddr_ch_standby=F["mW_lpddr_ch_standby"][0],
                    mW_lpddr_Gb_refresh=F["mW_lpddr_Gb_refresh"][0])
    h = hbm_static_W("lpddr5x_8x64_8533_16g", t)
    assert h["kind"] == "LPDDR" and h["standby_W"] > 0 and h["W"] > h["standby_W"]
    e = {"hbm_static": h}
    assert hbm_standby_W(e) == h["standby_W"] and hbm_device_static_W(e) == h["W"]
    # same 0.334 pattern as HBM (test_core_333): floor ignores standby offset; hot excludes full device static
    st, stb, ref = 20.0 + h["W"], h["standby_W"], h["W"] - h["standby_W"]
    old = package_power(100.0, 600.0, 1, st, None, 0.0)
    new = package_power(100.0, 600.0, 1, st, None, stb)
    both = package_power(100.0, 600.0, 1, st, None, stb, device_static_W=h["W"])
    assert new["static_add_W"] > old["static_add_W"] - 1e-9          # standby does not offset floor
    assert abs(both["static_add_W"] - new["static_add_W"]) < 1e-9    # device_static does not change floor
    assert both["pkg_W_hot"] < new["pkg_W_hot"] and ref > 0          # refresh also off-die in hot bound


def test_docs_and_no_ui_patch():
    md = open(os.path.join(ROOT, "docs", "MODEL.md"), encoding="utf-8").read()
    cl = open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8").read()
    assert "§280" in md and "## [0.335.0]" in cl and "3.23" in cl.split("## [0.335.0]")[1].split("## [")[0]
    assert "IDD4R" in md
