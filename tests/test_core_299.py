"""0.299: NoC energy from the dataflow (opt-in), energy breakdown, NoC design insight (/api/insight_noc)."""
import json
import math
from dataclasses import replace

from accel_dse import api, serve
from accel_dse.core import pdqueue
from accel_dse.core.energy import (ACTIONS, BREAKDOWN_GROUPS, NOC_HOP_PJ_BIT_45NM, NOC_HOP_PJ_BIT_7NM,
                                   NOC_MM2_PER_KMAC_DEFAULT, SLC_MM2_PER_MIB_DEFAULT, EnergyTable,
                                   apply_energy_preset, energy_report, noc_pitch_mm, unit_pJ)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.mapping import gemm_cost, noc_multicast
from accel_dse.core.noc_insight import engine_cuts, noc_design_report
from accel_dse.core.powercap import CORE_ACTIONS
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_sourced_constants():
    assert abs(NOC_HOP_PJ_BIT_45NM - ((427.3 - 76.7) / 653.0 - 0.121)) < 1e-12
    assert abs(NOC_HOP_PJ_BIT_7NM - NOC_HOP_PJ_BIT_45NM / 2.6) < 1e-12 and 0.15 < NOC_HOP_PJ_BIT_7NM < 0.17
    assert abs(NOC_MM2_PER_KMAC_DEFAULT - 400 * 0.11 / 64) < 1e-12


def test_multicast_topologies():
    ch = CHIPS["1P"]
    assert noc_multicast(ch, 1) == (0.0, 0.0)
    assert noc_multicast(ch, 32) == (31.0, 31.0)                                     # mesh XY tree
    assert noc_multicast(ch, 4, 16) == (12.0, 12.0)                                  # 4 blocks of 4×4: 4 hops apart
    assert noc_multicast(replace(ch, noc_topology="ring"), 4, 16) == (48.0, 48.0)
    h, p = noc_multicast(replace(ch, noc_topology="crossbar"), 32)
    assert h == 31.0 and abs(p - 32 * 0.5 * math.sqrt(32)) < 1e-12


def test_wide_gemm_multicasts_a_only_to_active_engines():
    ch = replace(CHIPS["1P"], instance_sched="wide")
    c = gemm_cost(ch, "os", 8, 4096, 4096)           # N = 4096 = 32 cols tiles → all 32 engines
    assert abs(c.noc_hops - 8 * 4096 * 2 * 31) < 1e-6 and c.noc_pitch == c.noc_hops
    c2 = gemm_cost(ch, "os", 8, 4096, 256)           # 2 column tiles → 2 engines → 1 receiver
    assert abs(c2.noc_hops - 8 * 4096 * 2 * 1) < 1e-6
    assert gemm_cost(replace(ch, engines=1, sram_mib=8.0), "os", 8, 4096, 4096).noc_hops == 0.0


def test_default_off_and_counts():
    r = evaluate(Scenario(model=M, chip=CHIPS["1P"], serving=Serving(batch=8)))
    e = energy_report(r, _t())
    assert "noc" not in e["J_by_action"] and "noc" not in e and e["counts_per_unit"]["noc"] > 0
    assert {"noc", "noc_wire"} <= set(ACTIONS) and "pJ_bit_hop_noc" not in e["missing"]
    t = _t(pJ_bit_hop_noc=NOC_HOP_PJ_BIT_7NM)                   # router only: no wire without pJ_bit_mm_onchip
    e2 = energy_report(r, t)
    assert "noc_wire" not in e2["J_by_action"]
    assert abs(e2["J_by_action"]["noc"] - e2["counts_per_unit"]["noc"] * 8 * NOC_HOP_PJ_BIT_7NM * 1e-12) < 1e-15
    t3 = _t(pJ_bit_hop_noc=NOC_HOP_PJ_BIT_7NM, pJ_bit_mm_onchip=0.078)
    e3 = energy_report(r, t3)
    p = noc_pitch_mm(t3, r.scenario.chip)
    assert abs(p - math.sqrt(256 / 32 * SLC_MM2_PER_MIB_DEFAULT + 16 * NOC_MM2_PER_KMAC_DEFAULT)) < 1e-12
    assert abs(e3["J_by_action"]["noc_wire"] - e3["counts_per_unit"]["noc_wire"] * 8 * 0.078 * p * 1e-12) < 1e-15
    assert e3["noc"]["topology"] == "mesh" and e3["noc"]["pitch_mm"] == p


def test_summary_unchanged_component_noc_kept_for_energy():
    r = evaluate(Scenario.from_dict({"model": "wan2.1-14b"}))
    parts = r.pipeline["parts"]
    assert any("noc" in p.get("acts", {}) for p in parts)
    assert not any("noc" in p.get("acts", {}) for p in r.summary()["gen"]["pipeline"]["parts"])


def test_breakdown_sums_and_wire_split():
    r = evaluate(Scenario(model=M, chip=CHIPS["1P"], serving=Serving(batch=8)))
    for t in (_t(), _t(pJ_bit_hop_noc=0.16, pJ_bit_mm_onchip=0.078)):
        e = energy_report(r, t)
        b = e["breakdown"]
        assert set(b["J"]) == set(BREAKDOWN_GROUPS) and abs(b["total"] - e["J_per_unit"]) < 1e-12 * max(1, e["J_per_unit"])
    sa = e["sram_access"]
    assert abs(b["J"]["sram"] - e["J_by_action"]["sram"] / sa["scale"]) < 1e-15
    assert abs(b["J"]["wire"] - (e["J_by_action"]["sram"] - b["J"]["sram"] + e["J_by_action"]["noc_wire"])) < 1e-15


def test_pd_parity_and_dvfs():
    scn = Scenario(model=M, chip=CHIPS["100T"], serving=Serving(batch=8))
    t = _t(pJ_bit_hop_noc=0.16, pJ_bit_mm_onchip=0.078)
    pd = pdqueue._energy({"noc": 1e6, "noc_wire": 1e6}, 0.0, t, pools=((scn, 0.0),), pool_names=("decode",))
    assert abs(pd["J_by_action"]["noc"] - 1e6 * 0.16 * 8e-12) < 1e-15
    assert abs(pd["J_by_action"]["noc_wire"] - 1e6 * unit_pJ(t, "noc_wire", scn.chip) * 8e-12) < 1e-15
    assert CORE_ACTIONS["noc"] == "pJ_bit_hop_noc"


def test_insight_design_reading():
    assert engine_cuts(524288) == [(2, 512), (8, 256), (32, 128), (128, 64), (512, 32), (2048, 16)]
    r = noc_design_report(Scenario(model=M, chip=CHIPS["1P"], serving=Serving(batch=8)), _t(),
                          topologies=("mesh", "crossbar"), phases=("prefill",))
    assert len(r["notes"]) == 2 and r["pJ_bit_hop_noc"] == NOC_HOP_PJ_BIT_7NM
    rows = {(x["engines"], x["topology"]): x for x in r["rows"]}
    m = [rows[(e, "mesh")]["noc_share"] for e in (8, 32, 128, 512)]
    assert m == sorted(m) and m[-1] > 3 * m[1]                  # many small cores: NoC share grows with E
    assert rows[(32, "crossbar")]["J_per_unit"] > rows[(32, "mesh")]["J_per_unit"]
    assert rows[(32, "mesh")]["central"]["extra_J"] > 0 and rows[(32, "mesh")]["base"]
    try:
        noc_design_report(Scenario(model=M, chip=CHIPS["1P"]), EnergyTable())
    except ValueError:
        pass
    else:
        raise AssertionError("needs an energy table")


def test_api_static_and_ui():
    assert "/api/insight_noc" in serve.POST_ROUTES
    assert "/api/insight_noc" in W.list_values(W.read(W.SHIM_JS), "slowRoutes")
    st, p = serve.respond("POST", "/api/insight_noc", json.dumps(
        {"scenario": {"model": M, "chip": {"name": "100T"}}, "energy": {"energy_preset": "tpuv4i_7nm"},
         "phases": ["decode"], "topologies": ["mesh"]}).encode())
    assert st == 200 and json.loads(p)["rows"][0]["base"] is True
    assert serve.respond("POST", "/api/insight_noc", b'{"scenario": {"model": "qwen3-8b"}, "topologies": ["torus"]}')[0] == 400
    html, js = W.read(W.INDEX_HTML), W.read()
    ids = W.html_ids(html)
    assert {"ins-noc-run", "ins-noc-status", "ins-noc-summary", "ins-noc-tbl", "ins-noc-prompt", "energy-breakdown",
            "e-pJ_bit_hop_noc", "e-noc_mm2_per_kmac"} <= ids
    assert [c["fn"] for c in W.req_calls(js) if c["path"] == "/api/insight_noc"] == ["runInsightNoc"]
    assert W.object_dict(W.literal(js, "STAGE"))["insight_noc"] == "NoC 能耗"
    assert "pJ_bit_hop_noc" in W.strings(js) and W.mentions(js, "breakdown") and W.mentions(js, "renderEnergyBreakdown")
    assert hasattr(api, "api_insight_noc")
