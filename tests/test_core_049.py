"""0.49 — MoE expert-load skew (optional) and resource / area budgets (optional design constraints)."""
from __future__ import annotations

import argparse
from dataclasses import replace

from accel_dse import api, cli
from accel_dse.api import ApiError
from accel_dse.core.budget import Budget, area_estimate, budget_report
from accel_dse.core.energy import EnergyTable, action_counts, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import moe_routing, skew_from_load
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving

HBM = "hbm3e_8s_12h24g_9200"
MOE = dict(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(tp=2, dp=4, ep=8))


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_routing_skew_closed_form():
    u = moe_routing(128, 8, 4096, 8)
    assert u.pairs_local == u.pairs_mean == 4096 * 8 / 8
    s = moe_routing(128, 8, 4096, 8, 1.5)
    assert s.pairs_local == 1.5 * u.pairs_local and s.hit == u.hit and s.m_e == -(-s.pairs_local // s.hit)
    cap = moe_routing(128, 8, 4096, 8, 64.0)          # capped at T per local expert (16 experts × T) and T·k
    assert cap.pairs_local == min(4096 * 8, 16 * 4096) and cap.m_e <= 4096
    small = moe_routing(128, 8, 4, 8, 8.0)            # decode: 4 tokens → hit grows when T rows cannot hold the load
    assert small.pairs_local == 32 and small.hit >= 8 and small.m_e <= 4
    assert moe_routing(128, 8, 4096, 1, 3.0).pairs_local == 4096 * 8      # EP = 1: nothing to skew
    assert skew_from_load((1.0,) * 128, 8) == 1.0
    load = (4.0,) * 16 + (1.0,) * 112                 # rank 0 holds 16 hot experts
    assert _close(skew_from_load(load, 8), 64 / (176 / 8)) and _close(skew_from_load(load, 4), 80 / (176 / 4))
    assert skew_from_load(load, 1) == 1.0


def test_skew_default_off_and_ignored_where_meaningless():
    base = evaluate(Scenario(**MOE, serving=Serving(batch=64)))
    assert base.moe_skew == 1.0 and "moe_skew" not in base.summary()
    for s in (Scenario(model="qwen3-8b", serving=Serving(batch=8, moe_skew=2.0)),
              Scenario(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(tp=2, etp=2), serving=Serving(batch=8, moe_skew=2.0))):
        r = evaluate(s)
        r0 = evaluate(replace(s, serving=replace(s.serving, moe_skew=1.0)))
        assert r.tick == r0.tick and r.moe_skew == 1.0 and any("倾斜已忽略" in w for w in r.warnings)


def test_skew_slows_busiest_rank_and_conserves_work():
    pre = Serving(batch=64, phase="prefill", prompt=2048)
    rs = [evaluate(Scenario(**MOE, serving=replace(pre, moe_skew=k))) for k in (1.0, 1.5, 2.0, 4.0)]
    t = [r.ttft for r in rs]
    assert t[0] < t[1] < t[2] < t[3] and rs[2].moe_skew == 2.0 and rs[2].summary()["moe_skew"] == 2.0
    # a2a payload follows the busiest rank: link bytes grow by (skew − 1) × the uniform dispatch + combine bytes
    l0, l2 = rs[0].stages[0].time.link_bytes, rs[2].stages[0].time.link_bytes
    assert l2 > l0
    # work counts per token unchanged (skew only moves work between ranks); card·s per token grows with latency
    c0, c2 = action_counts(rs[0]), action_counts(rs[2])
    for a in ("mac", "vec", "sram", "dram", "link"):
        assert _close(c0["counts"][a] / c0["units"], c2["counts"][a] / c2["units"], 1e-6), a
    e0, e2 = energy_report(rs[0]), energy_report(rs[2])
    assert e2["counts_per_unit"]["idle_card_s"] > e0["counts_per_unit"]["idle_card_s"] * 1.2
    # measured per-expert load ≡ the equivalent skew factor for this EP width
    load = (4.0,) * 16 + (1.0,) * 112
    k8 = skew_from_load(load, 8)
    a = evaluate(Scenario(**MOE, serving=replace(pre, moe_expert_load=load)))
    b = evaluate(Scenario(**MOE, serving=replace(pre, moe_skew=k8)))
    assert a.ttft == b.ttft and _close(a.moe_skew, k8)
    try:
        evaluate(Scenario(**MOE, serving=replace(pre, moe_expert_load=(1.0,) * 64)))
        raise AssertionError("length")
    except ValueError as e:
        assert "128" in str(e)
    for bad in (dict(moe_skew=0.5), dict(moe_skew=65.0), dict(moe_skew=2.0, moe_expert_load=(1.0,)),
                dict(moe_expert_load=(0.0, 0.0)), dict(moe_expert_load=(-1.0, 2.0))):
        try:
            Serving(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    # API round trip (list → tuple) and sweep path
    out = api.api_eval({"scenario": {**{"model": MOE["model"], "mem_id": HBM}, "layout": {"tp": 2, "dp": 4, "ep": 8},
                                     "serving": {"batch": 64, "phase": "prefill", "prompt": 2048,
                                                 "moe_expert_load": list(load)}}})
    assert _close(out["summary"]["moe_skew"], k8) and out["scenario"]["serving"]["moe_expert_load"][0] == 4.0
    sw = api.api_sweep({"scenario": {"model": MOE["model"], "mem_id": HBM, "layout": {"tp": 2, "dp": 4, "ep": 8},
                                     "serving": {"phase": "prefill", "prompt": 2048, "batch": 64}},
                        "path": "serving.moe_skew", "values": [1, 2]})
    assert sw["rows"][0]["ttft_ms"] < sw["rows"][1]["ttft_ms"]


def test_budget_report_semantics():
    r = evaluate(Scenario(model="qwen3-8b", serving=Serving(batch=8)))        # 100T: 64 MiB SRAM, 50176 MACs
    ch = r.scenario.chip
    assert budget_report(r, Budget())["items"] == [] and budget_report(r, Budget())["ok"] is True
    rep = budget_report(r, Budget(sram_mib=32, macs=60000, cards=1, tflops=90))
    it = {i["key"]: i for i in rep["items"]}
    assert it["sram_mib"]["ok"] is False and _close(it["sram_mib"]["headroom"], -1.0)
    assert it["macs"]["ok"] is True and _close(it["macs"]["headroom"], 1 - ch.macs / 60000)
    assert it["tflops"]["ok"] is False and it["cards"]["ok"] is True and rep["ok"] is False
    assert rep["violations"] == ["sram_mib", "tflops"]
    # area proxy: SRAM · d + MACs/1024 · d + fixed; missing density → undecidable; missing fixed → lower bound
    b = Budget(die_mm2=100, mm2_per_mib_sram=0.5, mm2_per_kmac=0.2, mm2_fixed=20)
    a = area_estimate(ch, b)
    assert _close(a["mm2"], 64 * 0.5 + ch.macs / 1024 * 0.2 + 20)
    it = budget_report(r, b)["items"][0]
    assert it["ok"] is True and _close(it["value"], a["mm2"])
    assert budget_report(r, replace(b, mm2_per_kmac=None))["items"][0]["ok"] is None
    lb = budget_report(r, replace(b, mm2_fixed=None))["items"][0]
    assert lb["ok"] is None and "下界" in lb["note"]
    assert budget_report(r, replace(b, mm2_fixed=None, die_mm2=10))["items"][0]["ok"] is False
    sys_ = budget_report(evaluate(Scenario(model="qwen3-8b", layout=Layout(tp=4), serving=Serving(batch=8))),
                         replace(b, die_mm2=None, system_mm2=300))["items"][0]
    assert _close(sys_["value"], 4 * a["mm2"]) and sys_["ok"] is True
    slc = evaluate(Scenario(model="qwen3-8b", chip=replace(CHIPS["100T"], slc_mib=256.0), serving=Serving(batch=8)))
    assert "mm2_per_mib_slc" in area_estimate(slc.scenario.chip, b)["missing"]
    # power: needs the energy table; an incomplete table gives a lower bound
    p = Budget(power_W_card=1000)
    assert budget_report(r, p, energy_report(r))["items"][0]["ok"] is None
    part = energy_report(r, EnergyTable(pJ_bit_dram=6.0))
    it = budget_report(r, p, part)["items"][0]
    assert it["ok"] is None and "下界" in it["note"] and "pJ_bit_slc" not in it["note"]     # zero-count entries skipped
    assert budget_report(r, Budget(power_W_card=0.1), part)["items"][0]["ok"] is False
    full = energy_report(r, EnergyTable(pJ_mac=0.5, pJ_vec=1, pJ_bit_sram=0.2, pJ_bit_dram=6, pJ_bit_link=5, idle_W=30))
    it = budget_report(r, p, full)["items"][0]
    assert it["ok"] is True and _close(it["value"], full["avg_W_per_card"])
    for bad in (dict(sram_mib=0), dict(cards=2.5), dict(mm2_fixed=-1), dict(die_mm2=float("nan"))):
        try:
            Budget(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_budget_api_cli_search_flags():
    sc = {"model": "qwen3-8b", "serving": {"batch": 8}}
    out = api.api_eval({"scenario": sc})
    assert "budget" not in out
    out = api.api_eval({"scenario": sc, "budget": {"sram_mib": 32}})
    assert out["budget"]["ok"] is False and out["budget"]["violations"] == ["sram_mib"]
    for bad in ({"sram": 1}, {"cards": 0}, [1]):
        try:
            api.api_eval({"scenario": sc, "budget": bad})
            raise AssertionError(bad)
        except ApiError:
            pass
    # search rows: power varies by layout → budget-breaking rows flagged and ranked after the rest
    en = {"pJ_mac": 0.5, "pJ_vec": 1, "pJ_bit_sram": 0.2, "pJ_bit_dram": 6, "pJ_bit_link": 5, "idle_W": 30}
    body = {"scenario": {"model": "qwen3-30b-a3b", "mem_id": HBM}, "cards": 8, "energy": en}
    free = api.api_layouts(body)
    ws = sorted(energy_report(evaluate(Scenario(model="qwen3-30b-a3b", mem_id=HBM,
                                                layout=Layout(**r["layout_obj"]),
                                                serving=Serving(batch=r["batch"]))), EnergyTable(**en))["avg_W_per_card"]
                for r in free["rows"][:6])
    lim = (ws[0] + ws[-1]) / 2
    flagged = api.api_layouts({**body, "budget": {"power_W_card": lim}})
    oks = [r["budget_ok"] for r in flagged["rows"]]
    assert True in oks and False in oks
    assert all(o is False for o in oks[oks.index(False):])          # breaking rows ranked after the rest
    cmp = api.api_compare({**body, "budget": {"power_W_card": lim}})
    assert all("budget_ok" in r for r in cmp["rows"])
    sw = api.api_sweep({"scenario": {"model": "qwen3-8b", "serving": {"batch": 8}}, "path": "chip.sram_mib",
                        "values": [32, 64, 128], "budget": {"die_mm2": 60, "mm2_per_mib_sram": 0.5,
                                                            "mm2_per_kmac": 0.2, "mm2_fixed": 0}})
    assert [r["value"] for r in sw["rows"]] == [32, 64, 128]
    assert [r["budget_ok"] for r in sw["rows"]] == [True, True, False]
    # capacity fix respects the card budget
    fit = api.api_fit({"scenario": {"model": "qwen3-235b-a22b"}, "budget": {"cards": 4}})
    assert not fit["fits"] and "min_cards" not in fit and fit["budget_cards_limit"] == 4
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    b = cli._body(p.parse_args(["--model", "qwen3-30b-a3b", "--moe-skew", "1.5", "--budget-sram-mib", "128",
                                "--budget-cards", "8", "--mm2-per-kmac", "0.2", "--budget-power-W", "300"]))
    assert b["scenario"]["serving"]["moe_skew"] == 1.5
    assert b["budget"] == {"sram_mib": 128.0, "cards": 8, "mm2_per_kmac": 0.2, "power_W_card": 300.0}
    b = cli._body(p.parse_args(["--model", "qwen3-30b-a3b", "--moe-expert-load", "1,2,3"]))
    assert b["scenario"]["serving"]["moe_expert_load"] == [1.0, 2.0, 3.0] and "budget" not in b
