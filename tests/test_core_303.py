"""0.303: NoC congestion (bisection bandwidth, opt-in) and LPDDR power-down / self-refresh residency."""
import math
from dataclasses import replace

from accel_dse import api
from accel_dse.core.energy import (ENERGY_PRESETS, EnergyTable, apply_energy_preset, energy_report, hbm_static_W,
                                   lpddr_residency)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, NOC_LINK_BPC_REF
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.mem_catalog import parse_mem_id
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
LP = "lpddr5x_4x64_8533_16g"
NEW = ("mW_lpddr_ch_pd", "mW_lpddr_ch_sr", "ns_lpddr_pd_gap", "ns_lpddr_sr_gap")


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _chip(e, side, topo, link=0.0, base="1P"):
    # 0.305: the default became XY + DRAM ingress; these tests pin the 0.303 bisection model they describe
    return replace(CHIPS[base], engines=e, rows=side, cols=side, noc_topology=topo, noc_link_Bpc=link,
                   noc_routing="bisection", noc_mem_attach="none")


def _ev(ch, phase="prefill", mem=None, batch=1):
    kw = {"mem_id": mem} if mem else {}
    return evaluate(Scenario(model=M, chip=ch, serving=Serving(phase=phase, batch=batch, prompt=2048, ctx=4096), **kw))


# ---------------------------------------------------------------- NoC congestion
def test_bisection_links_and_bw_cycles():
    assert NOC_LINK_BPC_REF == 64.0
    c = _chip(64, 128, "mesh", 64.0)
    assert c.noc_bisection_links(64) == 8.0
    assert replace(c, noc_topology="ring").noc_bisection_links(64) == 2.0
    assert replace(c, noc_topology="crossbar").noc_bisection_links(64) == 32.0
    assert math.isinf(c.noc_bisection_links(1))
    assert abs(c.noc_bw_cycles(8 * 64 * 10, 64) - 10.0) < 1e-12
    assert replace(c, noc_link_Bpc=0.0).noc_bw_cycles(1e9, 64) == 0.0       # unset = unlimited (≤ 0.302)
    try:
        replace(c, noc_link_Bpc=-1.0)
        raise AssertionError("negative link accepted")
    except ValueError:
        pass


def test_default_unchanged_and_penalty_only_when_exceeded():
    ch = CHIPS["1P"]
    assert ch.noc_link_Bpc == 0.0
    # unset link: identical to an explicit 0
    a, b = _ev(ch), _ev(replace(ch, noc_link_Bpc=0.0, noc_routing="bisection", noc_mem_attach="none"))
    assert a.throughput == b.throughput
    # few big engines on a ring at the reference link: slower; a huge link: identical to unlimited
    free = _ev(_chip(8, 256, "ring"))
    slow = _ev(_chip(8, 256, "ring", 64.0))
    wide = _ev(_chip(8, 256, "ring", 1e5))
    assert slow.throughput < free.throughput * 0.95
    assert abs(wide.throughput / free.throughput - 1) < 1e-9
    # crossbar with E/2 bisection links: (practically) not exceeded at 64 B/cycle (< 0.01 %)
    assert abs(_ev(_chip(8, 256, "crossbar", 64.0)).throughput / _ev(_chip(8, 256, "crossbar")).throughput - 1) < 1e-4
    # monotone in the link width
    assert _ev(_chip(8, 256, "ring", 32.0)).throughput <= slow.throughput + 1e-9


def test_noc_insight_and_dse_wiring():
    body = {"chip_preset": "1P", "scenario": {"model": M, "serving": {"phase": "prefill", "batch": 1, "prompt": 1024}},
            "energy": {"energy_preset": "tpuv4i_7nm"}, "phases": ["prefill"], "topologies": ["ring"],
            "noc_link_Bpc": 64}
    o = api.api_insight_noc(body)
    assert o["noc_link_Bpc"] == 64.0
    ok = [x for x in o["rows"] if "error" not in x]
    assert ok and all("noc_penalty" in x and "tok_s_free" in x for x in ok)
    assert any(x["noc_penalty"] > 0.05 for x in ok)
    o0 = api.api_insight_noc({k: v for k, v in body.items() if k != "noc_link_Bpc"})
    assert all("noc_penalty" not in x for x in o0["rows"])
    for bad in (-1, "x", True, 1e7):
        try:
            api.api_insight_noc({**body, "noc_link_Bpc": bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass
    d = api.api_insight_dse({"chip_preset": "1P", "scenario": body["scenario"], "energy": body["energy"],
                             "mac_scales": [1], "sram_scales": [1], "noc_link_Bpc": 64})
    assert d["noc_link_Bpc"] == 64.0
    html, js = W.read(W.INDEX_HTML), W.read()
    assert {"ins-noc-link", "ins-dse-link"} <= W.html_ids(html)
    assert "noc_link_Bpc" in W.function_source(js, "runInsightNoc")
    assert "noc_link_Bpc" in W.function_source(js, "runInsightDse")


# ---------------------------------------------------------------- LPDDR residency
def test_lpddr_lowpower_preset_values():
    f = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    assert abs(f["mW_lpddr_ch_pd"][0] - 5.415) < 1e-9          # Y52Q Table 16 IDD2P, all rails
    assert abs(f["mW_lpddr_ch_sr"][0] - 9.99) < 1e-9           # Y52Q Table 18, 85 °C max
    assert f["ns_lpddr_pd_gap"][0] == 15.0 and f["ns_lpddr_sr_gap"][0] == 232.5
    ok = {"来源", "推导", "二手", "假设"}
    for k in NEW:
        assert set(f[k][1].split("/")) <= ok and f[k][2], k
    # self-refresh (refresh included) below power-down + refresh, both below standby + refresh
    ref = f["mW_lpddr_Gb_refresh"][0] * 8
    assert f["mW_lpddr_ch_sr"][0] < f["mW_lpddr_ch_pd"][0] + ref < f["mW_lpddr_ch_standby"][0] + ref


def test_residency_mechanism():
    t, sp = _t(), parse_mem_id(LP)
    # 64 GB on 16 x16 channels → 4 ranks per channel: one rank active while the channel streams
    r = lpddr_residency(sp, t, 1.0, 0.0)
    assert r["channels"] == 16 and r["ranks_per_channel"] == 4
    assert abs(r["f_standby"] - 0.25) < 1e-12 and abs(r["f_powerdown"] - 0.75) < 1e-12 and r["f_selfrefresh"] == 0
    # long gaps → self-refresh minus the exit/minimum window; short → power-down; shorter than tXP → standby
    r = lpddr_residency(sp, t, 0.5, 1e-6)
    assert abs(r["f_selfrefresh"] - 0.5 * (1 - 232.5e-9 / 1e-6)) < 1e-12
    r = lpddr_residency(sp, t, 0.5, 100e-9)
    assert r["f_selfrefresh"] == 0 and abs(r["f_powerdown"] - (0.375 + 0.5 * (1 - 0.15))) < 1e-12
    r = lpddr_residency(sp, t, 0.5, 10e-9)
    assert abs(r["f_standby"] - (0.125 + 0.5)) < 1e-12
    for u, g in ((0.0, 1e-3), (0.3, 1e-7), (1.0, None), (0.7, None)):
        r = lpddr_residency(sp, t, u, g)
        assert abs(r["f_standby"] + r["f_powerdown"] + r["f_selfrefresh"] - 1) < 1e-12
    # W: fully busy = standby share only on the active rank; idle with long gaps → self-refresh bound
    hs0 = hbm_static_W(LP, t)
    full = hbm_static_W(LP, t, 1.0, 0.0)
    idle = hbm_static_W(LP, t, 0.0, 1.0)
    sl = 64
    assert abs(hs0["W"] - full["W_no_lowpower"]) < 1e-12
    assert idle["W"] < full["W"] < hs0["W"]
    assert abs(idle["W"] - sl * 9.99e-3 * (1 - 232.5e-9)) < 1e-6
    # unset fields → the 0.302 bound; HBM unaffected
    assert hbm_static_W(LP, replace(t, mW_lpddr_ch_pd=None), 0.0, 1.0)["W"] == hs0["W"]
    assert hbm_static_W("hbm3e_4s_8h24g_9200", t, 0.0, 1.0)["W"] == hbm_static_W("hbm3e_4s_8h24g_9200", t)["W"]


def test_report_uses_modelled_utilisation():
    t = _t()
    t0 = replace(t, mW_lpddr_ch_pd=None, mW_lpddr_ch_sr=None)
    dec = _ev(CHIPS["100T"], "decode", LP, 8)
    pre = _ev(CHIPS["100T"], "prefill", LP, 1)
    ed, ep = energy_report(dec, t), energy_report(pre, t)
    # memory-bound decode streams at the effective bandwidth → u = 1, no self-refresh; prefill idles → self-refresh
    assert ed["hbm_static"]["residency"]["util"] > 0.99 and ed["hbm_static"]["residency"]["f_selfrefresh"] == 0
    assert ep["hbm_static"]["residency"]["util"] < 0.9 and ep["hbm_static"]["residency"]["f_selfrefresh"] > 0.3
    for r, e in ((dec, ed), (pre, ep)):
        e0 = energy_report(r, t0)
        assert e["hbm_static_W_card"] < e0["hbm_static_W_card"]
        assert e["J_per_unit"] < e0["J_per_unit"]
        assert abs(e["hbm_static"]["W_no_lowpower"] - e0["hbm_static_W_card"]) < 1e-12
