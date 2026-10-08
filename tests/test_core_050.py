"""0.50 — prefill / decode disaggregation (optional) and the three-tier interconnect with an optional D2D tier."""
from __future__ import annotations

import argparse
import contextlib
import io
from dataclasses import replace

from accel_dse import api, cli
from accel_dse.api import ApiError
from accel_dse.core.d2d_catalog import D2D_DEFAULT_STD, D2D_DEFAULT_UNITS, D2D_STANDARDS, d2d_GBps
from accel_dse.core.disagg import disagg_report, kv_bytes_per_request
from accel_dse.core.catalog import get_model
from accel_dse.core.energy import action_counts
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import Link
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving
from accel_dse.core.schedule import collective_seconds, fabric_collective, p2p_tier, tiered_collective

D2D, LNK, NET = Link(1536.0, 0.5), Link(400.0, 3.0), Link(50.0, 5.0)
B = 8e6


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _same(a, b):
    for x, y in zip(a.stages, b.stages):
        assert x.time == y.time
    assert a.step == b.step and a.throughput == b.throughput


# ------------------------------------------------------------------ D2D catalog (user-approved list, 0.50)
def test_d2d_catalog_is_the_approved_list():
    assert list(D2D_STANDARDS) == ["ucie-a-48", "ucie-a-64", "ucie-s-32", "bow-256", "bow-512", "nvlink-c2c"]
    assert D2D_DEFAULT_STD == "ucie-a-48" and D2D_DEFAULT_UNITS == 4
    assert D2D_STANDARDS["ucie-a-48"]["per_unit_GBps"] == 48 * 64 / 8           # lanes × GT/s ÷ 8 (1 b/transfer)
    assert D2D_STANDARDS["ucie-a-64"]["per_unit_GBps"] == 64 * 64 / 8
    assert D2D_STANDARDS["ucie-s-32"]["per_unit_GBps"] == 32 * 16 / 8
    assert D2D_STANDARDS["bow-256"]["per_unit_GBps"] == 16 * 16 / 8 and D2D_STANDARDS["bow-512"]["per_unit_GBps"] == 64
    assert D2D_STANDARDS["nvlink-c2c"]["per_unit_GBps"] == 450 and not D2D_STANDARDS["nvlink-c2c"]["open"]
    assert all(v["open"] for k, v in D2D_STANDARDS.items() if k != "nvlink-c2c")
    assert not any("ccix" in k for k in D2D_STANDARDS)
    assert d2d_GBps("ucie-a-48", 4) == 1536.0
    cat = api.api_catalog()
    assert [d["id"] for d in cat["d2d_standards"]] == list(D2D_STANDARDS) and cat["d2d_default_std"] == "ucie-a-48"
    assert cat["defaults"]["d2d_enabled"] is False and cat["defaults"]["node_cards"] == 0


def test_defaults_d2d_off_and_one_node():
    s = Scenario()
    assert not s.d2d_enabled and s.package_eff == 1 and s.node_cards == 0
    assert s.d2d_link.GBps == 1536.0                          # the grade used when D2D is turned on
    assert s.replace("d2d_std", "custom").d2d_link == s.d2d
    assert s.replace("d2d_std", "bow-512").replace("d2d_units", 8).d2d_link.GBps == 512.0
    base = Scenario(model="qwen3-8b", layout=Layout(tp=8), serving=Serving(batch=8))
    r = evaluate(base)
    # inert knobs: package_cards with D2D off (warned), any D2D grade, the network link while node_cards = 0
    off = evaluate(replace(base, package_cards=4))
    _same(r, off)
    assert any("未生效" in w for w in off.warnings)
    _same(r, evaluate(replace(base, d2d_std="ucie-s-32", net=Link(1.0, 99.0))))
    assert all(st.time.net_bytes == 0 and st.time.d2d_bytes == 0 for st in r.stages)
    for bad in (dict(d2d_std="ccix"), dict(d2d_units=0), dict(node_cards=-1), dict(d2d_enabled=1),
                dict(d2d_enabled=True, package_cards=4, node_cards=6)):
        try:
            Scenario(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_legacy_scenarios_upgrade():
    s = Scenario.from_dict({"package_cards": 4})
    assert s.d2d_enabled and s.package_eff == 4 and s.d2d_std == D2D_DEFAULT_STD
    s = Scenario.from_dict({"package_cards": 2, "d2d": {"GBps": 1000.0, "alpha_us": 0.5, "topology": "switch"}})
    assert s.d2d_enabled and s.d2d_std == "custom" and s.d2d_link.GBps == 1000.0
    assert Scenario.from_dict(Scenario().to_dict()) == Scenario()
    out = api.api_eval({"scenario": {"model": "qwen3-8b", "layout": {"tp": 4}, "package_cards": 2,
                                     "d2d": {"GBps": 1000}}})
    assert out["scenario"]["d2d_enabled"] and out["scenario"]["d2d_std"] == "custom"


# ------------------------------------------------------------------ three-tier collectives
def test_fabric_reduces_to_flat_and_two_tier():
    for kind in ("allreduce", "allgather", "alltoall", "p2p"):
        flat = collective_seconds(kind, B, 8, LNK)
        assert fabric_collective(kind, B, 8, LNK) == (*flat, 0.0, 0.0)
        assert fabric_collective(kind, B, 8, LNK, D2D, 8)[2] == 1.0
        assert fabric_collective(kind, B, 8, LNK, net=NET, node=1)[3] == 1.0      # 1 card per node: all network
        for P in (2, 4):
            two = tiered_collective(kind, B, 8, LNK, D2D, P)
            assert fabric_collective(kind, B, 8, LNK, D2D, P)[:3] == two          # 0.48 bit for bit
            assert fabric_collective(kind, B, 8, LNK, D2D, P, NET, 8)[:3] == two  # node = whole group


def test_three_level_closed_forms():
    g, P, N = 16, 2, 8                       # n0 = 2 (D2D), n1 = 4 (scale-up), n2 = 2 (network)
    bd, bl, bn = 1536e9, 400e9, 50e9
    ad, al, an = 0.5e-6, 3e-6, 5e-6
    bw, a, fd, fn = fabric_collective("allreduce", B, g, LNK, D2D, P, NET, N)
    v = [2 * 1 / 2 * B, 2 * 3 / 4 * B / 2, 2 * 1 / 2 * B / 8]
    assert _close(bw, v[0] / bd + v[1] / bl + v[2] / bn) and _close(a, 2 * ad + 2 * al + an)
    assert _close(fd, v[0] / sum(v)) and _close(fn, v[2] / sum(v))
    bw, a, fd, fn = fabric_collective("allgather", B, g, LNK, D2D, P, NET, N)
    v = [1 * 8 * B, 3 * 2 * B, 1 * 1 * B]
    assert _close(bw, v[0] / bd + v[1] / bl + v[2] / bn) and _close(a, ad + al + an) and _close(fn, v[2] / sum(v))
    bw, a, fd, fn = fabric_collective("alltoall", B, g, LNK, D2D, P, NET, N)
    sh = [1 / 16, 6 / 16, 8 / 16]
    assert _close(bw, max(sh[0] * B / bd, sh[1] * B / bl, sh[2] * B / bn)) and _close(a, max(ad, al, an))
    assert _close(fd, sh[0] / sum(sh)) and _close(fn, sh[2] / sum(sh))
    # stride: a DP group with stride 8 (= node) never shares a node → all network
    assert fabric_collective("allreduce", B, 2, LNK, D2D, P, NET, N, stride=8)[3] == 1.0
    assert p2p_tier(0, 2, 4, 8) == "d2d" and p2p_tier(1, 2, 4, 8) == "link" and p2p_tier(3, 2, 4, 8) == "net"
    assert p2p_tier(0, 4, 1, 0) == "link"


def test_node_tier_in_evaluate_and_energy():
    base = Scenario(model="qwen3-8b", layout=Layout(tp=8), serving=Serving(batch=8))
    one = evaluate(base)
    two = evaluate(replace(base, node_cards=4))
    t1, t2 = one.stages[0].time, two.stages[0].time
    assert t2.link_bytes == t1.link_bytes and 0 < t2.net_bytes < t2.link_bytes and t2.t_link > t1.t_link
    c1, c2 = action_counts(one)["counts"], action_counts(two)["counts"]
    assert c1["net"] == 0 and c2["net"] > 0 and _close(c2["link"] + c2["net"], c1["link"])
    three = evaluate(replace(base, node_cards=4, package_cards=2, d2d_enabled=True))
    t3 = three.stages[0].time
    assert t3.d2d_bytes > 0 and t3.net_bytes > 0 and t3.d2d_bytes + t3.net_bytes < t3.link_bytes
    c3 = action_counts(three)["counts"]
    assert _close(c3["link"] + c3["net"] + c3["d2d"], c1["link"])
    pp = evaluate(Scenario(model="qwen3-8b", layout=Layout(pp=2, tp=2), serving=Serving(batch=8), node_cards=2))
    assert pp.stages[0].time.net_bytes > 0                    # stage 0 → 1 hand-off crosses the node boundary
    assert any("node_cards" in w for w in evaluate(replace(base, node_cards=3)).warnings)
    st = api.api_eval({"scenario": {"model": "qwen3-8b", "layout": {"tp": 8}, "node_cards": 4}})["stages"][0]
    assert st["link_GB"]["net"] > 0 and _close(st["link_GB"]["scaleup"] + st["link_GB"]["net"] + st["link_GB"]["d2d"],
                                               st["link_GB"]["total"])


def test_api_cli_three_tier():
    sw = api.api_sweep({"scenario": {"model": "qwen3-8b", "layout": {"tp": 8}}, "path": "node_cards",
                        "values": [0, 4, 2]})
    tp = [r["tpot_ms"] for r in sw["rows"]]
    assert tp[0] < tp[1] <= tp[2]                 # decode at batch 1 is DRAM-bound: the extra α shows
    sw = api.api_sweep({"scenario": {"model": "qwen3-8b", "layout": {"tp": 8}, "serving": {"phase": "prefill"}},
                        "path": "package_cards", "values": [1, 8]})
    assert sw["rows"][1]["ttft_ms"] < sw["rows"][0]["ttft_ms"]           # sweeping the package size turns D2D on
    try:
        api.api_sweep({"scenario": {"model": "qwen3-8b"}, "path": "d2d_units", "values": [1, 2]})
        raise AssertionError
    except ApiError:
        pass
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    a = p.parse_args(["--model", "qwen3-8b", "--package-cards", "4", "--d2d-std", "ucie-a-64", "--d2d-units", "2",
                      "--node-cards", "8", "--net-GBps", "100", "--pJ-bit-net", "20"])
    b = cli._body(a)
    sc = b["scenario"]
    assert sc["d2d_enabled"] and sc["d2d_std"] == "ucie-a-64" and sc["d2d_units"] == 2 and sc["node_cards"] == 8
    assert sc["net"] == {"GBps": 100.0} and b["energy"] == {"pJ_bit_net": 20.0}
    plain = cli._body(p.parse_args(["--model", "qwen3-8b"]))["scenario"]
    assert not {"d2d_enabled", "d2d_std", "d2d_units", "node_cards", "net", "pd"} & set(plain)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["d2d"]) == 0
    assert "ucie-a-48" in buf.getvalue() and "vendor" in buf.getvalue()


# ------------------------------------------------------------------ PD
PD_BASE = dict(model="qwen3-8b", serving=Serving(batch=16))


def test_pd_off_by_default():
    assert not Scenario().pd.enabled
    out = api.api_eval({"scenario": {"model": "qwen3-8b"}})
    assert "pd" not in out


def test_pd_kv_closed_form_and_report():
    m = get_model("qwen3-8b")
    assert kv_bytes_per_request(m, 4096) == 36 * 2 * 8 * 128 * 2 * 4096        # layers · K,V · kv heads · dim · bf16
    s = Scenario(**PD_BASE, pd=PDConfig(enabled=True, prefill_cards=2, decode_cards=2))
    r = disagg_report(s)
    assert r["prefill"]["cards"] == 2 and r["decode"]["cards"] == 2 and r["cards"] == 4
    k = r["kv"]
    assert k["tier"] == "link" and _close(k["t_ms"], (3e-6 + k["bytes_per_req"] / 400e9) * 1e3)
    assert _close(r["ttft_ms"], r["prefill"]["ttft_ms"] + k["exposed_ms"])
    cap = r["cap_req_s"]
    assert _close(r["req_s"], min(cap.values())) and r["bottleneck"] == min(cap, key=cap.get)
    assert _close(r["goodput_per_card"], r["req_s"] * s.serving.out_len / 4)
    assert _close(cap["decode"], 2 * r["decode"]["tok_s_replica"] / s.serving.out_len)
    assert r["best_split"]["goodput_per_card"] >= r["goodput_per_card"]
    assert sum(1 for x in r["splits"]) == 3 and r["coloc"]["cards"] == 4
    # decode pool sees no prefill interference: PD TPOT = the pure decode step; colocated effective TPOT is worse
    assert _close(r["tpot_ms"], evaluate(s.replace("serving.phase", "decode")).tpot * 1e3)
    assert r["coloc"]["tpot_eff_ms"] > r["tpot_ms"]


def test_pd_layerwise_net_tier_and_kv_override():
    s = Scenario(**PD_BASE, pd=PDConfig(enabled=True, prefill_cards=2, decode_cards=2))
    full, lw = disagg_report(s), disagg_report(s.replace("pd.kv_layerwise", True))
    assert lw["kv"]["exposed_ms"] <= full["kv"]["exposed_ms"] and lw["kv"]["t_ms"] > full["kv"]["t_ms"]
    nt = disagg_report(s.replace("node_cards", 1))
    assert nt["kv"]["tier"] == "net" and _close(nt["kv"]["GBps_req"], 50.0) and nt["kv"]["t_ms"] > full["kv"]["t_ms"]
    ov = disagg_report(s.replace("pd.kv_GBps", 10.0))
    assert ov["kv"]["source"] == "pd.kv_GBps" and _close(ov["kv"]["GBps_req"], 10.0)
    tp = disagg_report(Scenario(**PD_BASE, layout=Layout(tp=2), pd=PDConfig(enabled=True, prefill_layout=Layout(tp=4),
                                                                           prefill_cards=4, decode_cards=4)))
    assert tp["prefill"]["layout"] != tp["decode"]["layout"] and _close(tp["kv"]["GBps_req"], 2 * 400.0)


def test_pd_validation_api_cli():
    for bad in (dict(prefill_cards=3, prefill_layout=Layout(tp=2)), dict(prefill_cards=-1), dict(kv_GBps=0.0),
                dict(enabled=1)):
        try:
            PDConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    try:
        Scenario(layout=Layout(tp=2), pd=PDConfig(enabled=True, decode_cards=3))
        raise AssertionError
    except ValueError:
        pass
    out = api.api_eval({"scenario": {"model": "qwen3-8b", "serving": {"batch": 16},
                                     "pd": {"enabled": True, "prefill_cards": 1, "decode_cards": 3}}})
    assert out["pd"]["cards"] == 4 and out["pd"]["decode"]["replicas"] == 3
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    a = p.parse_args(["--model", "qwen3-8b", "--pd", "--pd-prefill-tp", "2", "--pd-prefill-cards", "4",
                      "--pd-decode-cards", "2", "--pd-layerwise"])
    pd = cli._body(a)["scenario"]["pd"]
    assert pd["enabled"] and pd["prefill_layout"]["tp"] == 2 and pd["prefill_cards"] == 4 and pd["kv_layerwise"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["eval", "--model", "qwen3-8b", "--batch", "16", "--pd", "--pd-decode-cards", "3"]) == 0
    assert "PD" in buf.getvalue()
