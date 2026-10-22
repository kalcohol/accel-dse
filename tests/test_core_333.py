"""0.333: HBM energy reconciliation against Olgun et al. arXiv 2610.14244 (MODEL §278) — measured H200 HBM3E split
into a static term (per-64-bit-channel standby + refresh, ∝ channels / capacity × time) and a per-bit dynamic term;
clocked HBM standby no longer offsets the area-scaled idle floor (thermal.package_power standby_W).
0.334: adversarial review — hot bound excludes full HBM device static (standby+refresh); stale HBM4 2.46 labels → 4.14."""
import os

from accel_dse.core.energy import (ENERGY_PRESETS, H200_HBM_MEAS, HBM_CH_BITS, EnergyTable, hbm_static_W)
from accel_dse.core.powercap import hbm_device_static_W, hbm_standby_W
from accel_dse.core.energy_dse import MEM_PROXY
from accel_dse.core.thermal import STATIC_FLOOR_W_MM2, VR_EFF, _hot_k, cap_power_fn, card_W_for_cap, package_power

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
H200 = "hbm3e_6s_8h24g_6400"


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def _tab(**kw):
    return EnergyTable(mW_hbm_ch_standby=F["mW_hbm_ch_standby"][0], mW_hbm_Gb_refresh=F["mW_hbm_Gb_refresh"][0], **kw)


def test_preset_values_labels_and_sources():
    v, tag, src = F["mW_hbm_ch_standby"]
    assert _close(v, 376.7736111111111) and tag == "推导" and "2610.14244" in src and "38.42" in src
    v, tag, src = F["pJ_bit_dram_hbm"]
    assert _close(v, 5.8017342581848474) and tag == "推导" and "2610.14244" in src and "5.09" in src
    assert "FGDRAM" in src or "O'Connor" in src or "Fine-Grained" in src
    v4, tag4, src4 = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]["pJ_bit_dram_hbm4"]
    assert _close(v4, 5.8017342581848474 / 1.4) and tag4 == "推导" and "0.94–1.00" in src4 and "40 %" in src4
    assert _close(F["mW_hbm_Gb_refresh"][0], 195 * 1.2 * 260 / 3900 / 8)         # refresh unchanged (1.95 mW/Gb)
    assert "mW_hbm_die_standby" not in F                                         # the per-die form is no longer filled
    assert HBM_CH_BITS == 64 and H200_HBM_MEAS["channels"] == 96 and len(H200_HBM_MEAS["units"]) == 3


def test_h200_split_reproduces_the_measurement_by_construction():
    # not a validation (the split is defined from these numbers): static at idle = measured idle mean, and the
    # per-bit term × measured read bandwidth = measured (total − idle) mean
    u = H200_HBM_MEAS["units"]
    h = hbm_static_W(H200, _tab())
    assert h["kind"] == "HBM" and h["channels"] == 96 and h["dies"] == 48
    assert _close(h["W"], sum(x[0] for x in u) / 3)
    pj = F["pJ_bit_dram_hbm"][0]
    assert _close(pj, sum((t - i) / (g * 8e9) * 1e12 for i, t, g in u) / 3)
    for i, t, g in u:                                   # each unit within 1.5 % of the mean active energy
        assert abs((t - i) / (g * 8e9) * 1e12 / pj - 1) < 0.015


def test_per_channel_standby_is_height_independent_and_doubles_for_hbm4():
    t = _tab()
    a, b = hbm_static_W("hbm3e_8s_8h24g_9200", t), hbm_static_W("hbm3e_8s_12h24g_9200", t)
    assert a["standby_W"] == b["standby_W"] and b["refresh_W"] > a["refresh_W"]     # refresh ∝ capacity only
    c = hbm_static_W("hbm4_8s_12h24g_10000", t)
    assert c["channels"] == 2 * b["channels"] == 256 and _close(c["standby_W"], 2 * b["standby_W"])
    # HBM4 per stack 12.06 W standby vs the paper's own extrapolation 11.84 W (cross-check, not fitted): within 3 %
    assert abs(c["standby_W"] / 8 / (176.2e-3 * 1.05 * 64) - 1) < 0.03


def test_user_die_standby_field_still_works_and_sums():
    f = F["mW_hbm_Gb_refresh"][0]
    d = hbm_static_W("hbm3e_8s_8h24g_9200", EnergyTable(mW_hbm_die_standby=48.0, mW_hbm_Gb_refresh=f))
    assert _close(d["standby_W"], 64 * 0.048)
    both = hbm_static_W("hbm3e_8s_8h24g_9200", _tab(mW_hbm_die_standby=48.0))
    assert _close(both["standby_W"], 64 * 0.048 + 128 * F["mW_hbm_ch_standby"][0] * 1e-3)
    assert hbm_static_W("hbm3e_8s_8h24g_9200", EnergyTable(mW_hbm_ch_standby=100.0))["standby_W"] == 12.8


def test_standby_does_not_offset_the_area_floor():
    die, st, stb = 800.0, 60.0, 40.0                     # 60 W static of which 40 W clocked HBM standby
    old = package_power(300.0, die, 1, st, None)          # ≤ 0.332 behaviour (standby_W = 0)
    new = package_power(300.0, die, 1, st, None, stb)
    fl = STATIC_FLOOR_W_MM2 * die * VR_EFF
    assert _close(old["static_add_W"], max(0.0, fl - st)) and _close(new["static_add_W"], max(0.0, fl - (st - stb)))
    assert new["pkg_W"] > old["pkg_W"] and _close(new["static_W"], st + new["static_add_W"])
    hot = (st - stb + new["static_add_W"]) * (_hot_k() - 1.0) / VR_EFF            # standby is not die leakage
    assert _close(new["pkg_W_hot"] - new["pkg_W"], hot)
    # 0.334: refresh is also off-die — pass device_static_W = standby+refresh; floor still uses standby only
    ref = 5.0
    both = package_power(300.0, die, 1, st, None, stb, device_static_W=stb + ref)
    assert _close(both["static_add_W"], new["static_add_W"])                      # floor unchanged
    hot2 = (st - (stb + ref) + both["static_add_W"]) * (_hot_k() - 1.0) / VR_EFF
    assert _close(both["pkg_W_hot"] - both["pkg_W"], hot2) and both["pkg_W_hot"] < new["pkg_W_hot"]
    # die area matters again when the stacks out-idle the die (the 0.333 test_core_327 decision)
    small, big = (package_power(st, a, 1, st, None, stb)["pkg_W"] for a in (500.0, 900.0))
    assert big > small
    pk = cap_power_fn(die, 1, "package")
    assert pk(300.0, st, None, stb) == new["pkg_W"] and pk(300.0, st, None) == old["pkg_W"]
    c = card_W_for_cap(pk, 700.0, st, None, stb)
    assert abs(pk(c, st, None, stb) - 700.0) < 1e-6


def test_hbm_standby_W_reads_only_hbm():
    assert hbm_standby_W({"hbm_static": {"kind": "HBM", "standby_W": 25.0}}) == 25.0
    assert hbm_standby_W({"hbm_static": {"kind": "LPDDR", "standby_W": 3.0}}) == 3.0  # 0.335
    assert hbm_standby_W({}) == 0.0
    e = {"hbm_static": {"kind": "HBM", "standby_W": 25.0, "refresh_W": 3.0, "W": 28.0}}
    assert hbm_device_static_W(e) == 28.0 and hbm_standby_W(e) == 25.0
    assert hbm_device_static_W({"hbm_static": {"kind": "LPDDR", "W": 3.0}}) == 3.0   # 0.335: LPDDR too
    assert hbm_standby_W({"hbm_static": {"kind": "LPDDR", "standby_W": 1.5}}) == 1.5


def test_docs_record_the_change():
    md = open(os.path.join(ROOT, "docs", "MODEL.md"), encoding="utf-8").read()
    cl = open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8").read()
    assert "§278" in md and "2610.14244" in md
    sec = cl.split("## [0.333.0]")[1].split("## [")[0]
    assert "3.44" in sec and "5.80" in sec and "376.8" in sec and "standby_W" in sec
    assert "§279" in md and "## [0.334.0]" in cl
    assert "4.14" in MEM_PROXY["HBM4"] and "2.46" not in MEM_PROXY["HBM4"]
    assert "§280" in md and "## [0.335.0]" in cl
    assert "2.67" in MEM_PROXY["LPDDR6"] or "3.23" in MEM_PROXY["LPDDR6"]
