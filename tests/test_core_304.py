"""0.304: PD-queue LPDDR residency with an evaluation window; per-link (XY) NoC contention, opt-in; 210 guard."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import (EnergyTable, apply_energy_preset, energy_report, hbm_static_W, hbm_static_W_window,
                                   lpddr_window)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, NOC_ROUTINGS
from accel_dse.core.pdqueue import _energy, _per_unit
from accel_dse.core.scenario import PDConfig, Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
LP = "lpddr5x_4x64_8533_16g"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _r(ph, b, chip="100T", mem=LP):
    return evaluate(Scenario(chip=CHIPS[chip], model=M, mem_id=mem).replace("serving.phase", ph).replace("serving.batch", b))


# ---------------------------------------------------------------- PD-queue LPDDR window
def test_queue_ledger_with_window_matches_single_eval():
    t = _t()
    for ph, b in (("decode", 8), ("prefill", 1), ("decode", 64)):
        r = _r(ph, b)
        e = energy_report(r, t)
        cs = r.scenario.layout.cards * e["window_s"] / e["units_per_window"]
        p = _energy(_per_unit(r), cs, t, pools=((r.scenario, cs),), pool_windows=(lpddr_window(r),))
        assert abs(p["J_per_token"] / e["J_per_unit"] - 1) < 1e-9, (ph, b)
        assert abs(p["J_by_action"]["hbm_static"] / e["J_by_action"]["hbm_static"] - 1) < 1e-9
        # without a window: the 0.303 power-down-only figure (no self-refresh; gaps assumed ≥ tXP)
        p0 = _energy(_per_unit(r), cs, t, pools=((r.scenario, cs),))
        assert "busy_frac" not in p0["dram_residency"]["0"] and p0["dram_residency"]["0"]["f_selfrefresh"] == 0


def test_window_mix_mechanism():
    t, r = _t(), _r("decode", 8)
    win = lpddr_window(r)
    assert 0.99 < win["util"] <= 1.0 and win["layers"] == 36 and win["window_s"] > 0
    # a pool iterating 40 % of the time: ρ = 0.4 → W = 0.4·W_iter + 0.6·W_idle (u = 0, long gap → self-refresh)
    u = 0.4 * win["util"]
    hs = hbm_static_W_window(LP, t, u, win)
    wi = hbm_static_W(LP, t, win["util"], (1 - win["util"]) * win["window_s"] / win["layers"])["W"]
    w0 = hbm_static_W(LP, t, 0.0, win["window_s"])["W"]
    assert abs(hs["residency"]["busy_frac"] - 0.4) < 1e-12
    assert abs(hs["W"] - (0.4 * wi + 0.6 * w0)) < 1e-12
    assert hs["residency"]["f_selfrefresh"] > 0.59
    fr = hs["residency"]
    assert abs(fr["f_standby"] + fr["f_powerdown"] + fr["f_selfrefresh"] - 1) < 1e-12
    # no window → 0.303 (power-down only); HBM → unchanged
    assert hbm_static_W_window(LP, t, u, None)["W"] == hbm_static_W(LP, t, u, None)["W"]
    hb = "hbm3e_4s_8h24g_9200"
    assert hbm_static_W_window(hb, t, u, win)["W"] == hbm_static_W(hb, t)["W"]


def test_pd_report_credits_selfrefresh():
    t = _t()
    s = Scenario(model=M, chip=CHIPS["100T"], mem_id=LP, serving=Serving(batch=16, prompt=2048, out_len=512),
                 pd=PDConfig(enabled=True, prefill_cards=2, decode_cards=6, load=0.3))
    q = disagg_report(s, energy=t)["queue"]
    res = q["energy"]["pd"]["dram_residency"]
    assert set(res) == {"decode", "prefill"}
    assert 0 < res["prefill"]["busy_frac"] < 0.5 and res["prefill"]["f_selfrefresh"] > 0.5
    assert 0 < res["decode"]["busy_frac"] <= 1.0
    for m in ("coloc_prefill_first", "coloc_chunked"):
        assert q["energy"][m]["dram_residency"]["0"]["f_selfrefresh"] > 0, m


# ---------------------------------------------------------------- per-link XY NoC contention
def _chip(e, side, topo, routing="xy", link=64.0):
    return replace(CHIPS["1P"], engines=e, rows=side, cols=side, noc_topology=topo, noc_link_Bpc=link,
                   noc_routing=routing)


def test_xy_link_load_formulas():
    assert NOC_ROUTINGS == ("bisection", "xy") and CHIPS["1P"].noc_routing == "bisection"
    P = 1024.0
    m = _chip(16, 64, "mesh")
    assert abs(m.noc_xy_load(P, 16) - P * 3 / 4) < 1e-9          # 4 × 4: column links carry 3 of 4 rows' shards
    assert abs(m.noc_xy_load(P, 2) - P / 2) < 1e-9               # 1 × 2 row
    assert abs(m.noc_xy_load(P, 4, 16) - P / 2 / 4) < 1e-9       # 2 × 2 blocks of 16 engines: ⌊√16⌋ = 4 links
    assert abs(m.noc_xy_load(P, 8, 4, "reduce") - P / 2) < 1e-9  # one tile per tree edge over 2 boundary links
    r = replace(m, noc_topology="ring")
    assert abs(r.noc_xy_load(P, 16) - P / 2) < 1e-9 and r.noc_block_links(16) == 1.0
    x = replace(m, noc_topology="crossbar")
    assert abs(x.noc_xy_load(P, 16) - P * 15 / 16) < 1e-9 and x.noc_block_links(4) == 4.0
    assert m.noc_xy_load(P, 1) == 0.0
    # ring all-gather = the 0.303 bisection figure (2 links each carry half)
    assert abs(r.noc_traffic_cycles(P, 16) - replace(r, noc_routing="bisection").noc_traffic_cycles(P, 16)) < 1e-9
    assert replace(m, noc_link_Bpc=0.0).noc_traffic_cycles(P, 16) == 0.0
    try:
        replace(m, noc_routing="yx")
        raise AssertionError("bad routing accepted")
    except ValueError:
        pass


def test_xy_vs_bisection_effect():
    sv = Serving(phase="prefill", batch=1, prompt=2048, ctx=4096)

    def thr(c):
        return evaluate(Scenario(model=M, chip=c, serving=sv)).throughput
    free = thr(_chip(32, 128, "mesh", link=0.0))
    bis = thr(_chip(32, 128, "mesh", "bisection"))
    xy = thr(_chip(32, 128, "mesh"))
    # every engine ingests the whole A through its column link: busier than the bisection average
    assert xy < bis * 0.8 and bis <= free
    # routing has no effect while the link is unlimited; default routing = 0.303
    assert thr(_chip(32, 128, "mesh", "xy", 0.0)) == free
    assert bis == thr(replace(CHIPS["1P"], engines=32, rows=128, cols=128, noc_link_Bpc=64.0))
    # ring: XY ≈ bisection (both: half the payload per link)
    rb, rx = thr(_chip(8, 256, "ring", "bisection")), thr(_chip(8, 256, "ring"))
    assert abs(rx / rb - 1) < 0.02


def test_api_and_ui_routing():
    body = {"chip_preset": "1P", "scenario": {"model": M, "serving": {"phase": "prefill", "batch": 1, "prompt": 1024}},
            "energy": {"energy_preset": "tpuv4i_7nm"}, "phases": ["prefill"], "topologies": ["mesh"],
            "noc_link_Bpc": 64, "noc_routing": "xy"}
    o = api.api_insight_noc(body)
    assert o["noc_routing"] == "xy" and any("逐链路 XY" in n for n in o["notes"])
    try:
        api.api_insight_noc({**body, "noc_routing": "yx"})
        raise AssertionError
    except api.ApiError:
        pass
    d = api.api_insight_dse({"chip_preset": "1P", "scenario": body["scenario"], "energy": body["energy"],
                             "mac_scales": [1], "sram_scales": [1], "noc_link_Bpc": 64, "noc_routing": "xy"})
    assert d["noc_routing"] == "xy"
    html, js = W.read(W.INDEX_HTML), W.read()
    assert {"ins-noc-routing", "ins-dse-routing"} <= W.html_ids(html)
    assert "noc_routing" in W.function_source(js, "runInsightNoc")
    assert "noc_routing" in W.function_source(js, "runInsightDse")


# ---------------------------------------------------------------- 210 guard
def test_210_guard_uses_cpu_time():
    import time

    from tests.test_core_210 import _contention, _timed
    assert _contention() >= 1.0

    def burn(_):
        t = time.process_time()
        while time.process_time() - t < 0.3:
            pass
    try:
        _timed(burn, {}, 0.2)
        raise AssertionError("cpu over the bound passed")
    except AssertionError as e:
        assert "cpu" in str(e)
    assert _timed(lambda b: 7, {}, 5.0) == 7
