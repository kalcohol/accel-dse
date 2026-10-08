"""0.48 — system-level cache (SLC) and the two-tier interconnect (die-to-die inside a package vs scale-up network)."""
from __future__ import annotations

import argparse
from dataclasses import replace

from accel_dse import api, cli
from accel_dse.api import ApiError
from accel_dse.core.energy import EnergyTable, action_counts, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Chip, Link
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving, Workload
from accel_dse.core.schedule import collective_seconds, p2p_crosses, tiered_collective

HBM = "hbm3e_8s_12h24g_9200"
NET, D2D = Link(400.0, 3.0), Link(2000.0, 0.5)


def _close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _same(a, b):
    assert a.tick == b.tick and a.step == b.step and a.latency == b.latency and a.bound == b.bound
    for x, y in zip(a.stages, b.stages):
        assert x.time.total == y.time.total and x.time.dram_bytes == y.time.dram_bytes
        assert x.time.link_bytes == y.time.link_bytes and x.mem.dram_need == y.mem.dram_need


def test_defaults_reproduce_single_tier_and_no_slc():
    base = [Scenario(model="qwen3-8b", layout=Layout(tp=4, pp=2), serving=Serving(batch=8)),
            Scenario(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(tp=2, dp=4, ep=8), serving=Serving(batch=32)),
            Scenario(model="wan2.1-14b", mem_id=HBM, layout=Layout(sp=4), workload=Workload(dit_fsdp=True,
                                                                                            vae_parallel=True))]
    for s in base:
        r = evaluate(s)
        # package_cards = 1: the D2D link is never used, whatever it is; slc_mib = 0: SLC knobs are inert
        _same(r, evaluate(replace(s, d2d=Link(10.0, 9.0))))
        _same(r, evaluate(s.replace("chip", replace(s.chip, slc_GBps=1.0, slc_policy="lru"))))
        assert all(st.time.t_slc == 0 and st.time.slc_bytes == 0 and st.time.d2d_bytes == 0 for st in r.stages)
        # every card in one package with D2D = network link → the single-tier result again
        one = evaluate(replace(s, package_cards=64, d2d=s.link, d2d_enabled=True, d2d_std="custom"))
        assert _close(one.latency or one.tick, r.latency or r.tick, 1e-9)
        assert all(st.time.d2d_bytes == st.time.link_bytes for st in one.stages)


def test_tiered_closed_forms():
    B = 64e6
    bd, bn, ad, an = 2000e9, 400e9, 0.5e-6, 3e-6
    # k == 1 and k == g reduce to the single-tier formulas on the network / the D2D link
    assert tiered_collective("allreduce", B, 8, NET, D2D, 1) == (*collective_seconds("allreduce", B, 8, NET), 0.0)
    assert tiered_collective("allreduce", B, 8, NET, D2D, 8) == (*collective_seconds("allreduce", B, 8, D2D), 1.0)
    assert tiered_collective("alltoall", B, 4, NET, D2D, 4, stride=4)[2] == 0.0      # stride ≥ package → k = 1
    # g = 8, k = 4, m = 2
    bw, a, f = tiered_collective("allreduce", B, 8, NET, D2D, 4)
    assert _close(bw, 2 * 3 / 4 * B / bd + 2 * 1 / 2 * B / 4 / bn) and _close(a, 2 * ad + an)
    assert _close(f, (1.5 * B) / (1.5 * B + B / 4))
    bw, a, f = tiered_collective("allgather", B, 8, NET, D2D, 4)
    assert _close(bw, 1 * B / bn + 3 * 2 * B / bd) and _close(a, ad + an) and _close(f, 6 / 7)
    bw, a, f = tiered_collective("alltoall", B, 8, NET, D2D, 4)
    assert _close(bw, max(3 / 8 * B / bd, 4 / 8 * B / bn)) and _close(a, an) and _close(f, 3 / 7)
    # stride: an SP group of 4 behind TP 2 in packages of 4 cards has k = 2 members per package
    bw, a, f = tiered_collective("alltoall", B, 4, NET, D2D, 4, stride=2)
    assert _close(bw, max(1 / 4 * B / bd, 2 / 4 * B / bn)) and _close(f, 1 / 3)
    # hierarchical allreduce never moves more over the network than the flat one
    flat = collective_seconds("allreduce", B, 8, NET)[0]
    assert tiered_collective("allreduce", B, 8, NET, D2D, 4)[0] < flat
    # PP hand-off: stages of 2 cards in packages of 4 → 0→1 stays, 1→2 crosses
    assert not p2p_crosses(0, 2, 4) and p2p_crosses(1, 2, 4) and p2p_crosses(0, 4, 4) and p2p_crosses(0, 8, 1)


def test_collectives_map_to_tiers_in_evaluate():
    def share(s):
        r = evaluate(s)
        t = r.stages[r.heaviest_stage].time
        return r, t.d2d_bytes / t.link_bytes
    pre = Serving(batch=8, phase="prefill", prompt=4096)
    r1, f1 = share(Scenario(model="qwen3-8b", layout=Layout(tp=8), serving=pre))
    r4, f4 = share(Scenario(model="qwen3-8b", layout=Layout(tp=8), serving=pre, package_cards=4, d2d_enabled=True))
    r8, f8 = share(Scenario(model="qwen3-8b", layout=Layout(tp=8), serving=pre, package_cards=8, d2d_enabled=True))
    assert f1 == 0 and 0 < f4 < 1 and f8 == 1
    lt = [r.stages[0].time.t_link for r in (r1, r4, r8)]
    assert lt[0] > lt[1] > lt[2]                 # TP allreduce: flat network > hierarchical > all on D2D
    assert _close(lt[2], lt[0] * 400 / 1536)
    # PP2·TP4 in packages of 4: TP inside the package, the PP hand-off crosses packages
    r, f = share(Scenario(model="qwen3-8b", layout=Layout(pp=2, tp=4), serving=Serving(batch=8), package_cards=4, d2d_enabled=True))
    st0 = r.stages[0].time
    assert 0 < st0.d2d_bytes < st0.link_bytes and r.stages[1].time.d2d_bytes == r.stages[1].time.link_bytes
    r, f = share(Scenario(model="qwen3-8b", layout=Layout(pp=2, tp=4), serving=Serving(batch=8), package_cards=8, d2d_enabled=True))
    assert r.stages[0].time.d2d_bytes == r.stages[0].time.link_bytes
    # MoE EP spanning two packages: dispatch / combine split between tiers
    _, f = share(Scenario(model="qwen3-30b-a3b", mem_id=HBM, layout=Layout(tp=2, dp=4, ep=8),
                          serving=Serving(batch=64), package_cards=4, d2d_enabled=True))
    assert 0 < f < 1
    # video: Ulysses SP (stride TP) + FSDP gathers (stride TP) + VAE tile gathers
    w = dict(model="wan2.1-14b", mem_id=HBM, layout=Layout(sp=8),
             workload=Workload(dit_fsdp=True, vae_parallel=True, vae_tiling=True))
    a, b = evaluate(Scenario(**w)), evaluate(Scenario(**w, package_cards=8, d2d_enabled=True))
    assert b.stages[0].time.d2d_bytes == b.stages[0].time.link_bytes
    assert b.stages[0].time.t_link < a.stages[0].time.t_link
    va = next(p for p in a.pipeline["parts"] if p["role"] == "vae")
    vb = next(p for p in b.pipeline["parts"] if p["role"] == "vae")
    assert vb["gather_s"] < va["gather_s"] and vb["acts"]["link"] == 0 and vb["acts"]["d2d"] == va["acts"]["link"]
    # protein DAP (stride 1)
    p1 = evaluate(Scenario(model="protenix", mem_id=HBM, layout=Layout(sp=4)))
    p4 = evaluate(Scenario(model="protenix", mem_id=HBM, layout=Layout(sp=4), package_cards=4, d2d_enabled=True))
    assert p4.stages[0].time.t_link < p1.stages[0].time.t_link and p4.stages[0].time.d2d_bytes > 0


def test_slc_pin_and_lru():
    s0 = Scenario(model="qwen3-8b", serving=Serving(batch=8))
    r0 = evaluate(s0)
    d0 = r0.stages[0].time.dram_bytes
    prev = 0.0
    for mib in (256, 4096, 8192, 16384):
        s = s0.replace("chip", replace(s0.chip, slc_mib=float(mib)))
        r = evaluate(s)
        t, mp = r.stages[0].time, r.stages[0].mem
        assert _close(t.dram_bytes + t.slc_bytes, d0, 1e-9)                # traffic conserved: hits + misses
        assert t.slc_bytes > prev and _close(t.t_slc, t.slc_bytes / 2000e9)
        assert mp.dram_need == r0.stages[0].mem.dram_need                  # no capacity added
        assert r.tpot <= r0.tpot and mp.slc_residency >= mp.residency
        prev = t.slc_bytes
    # pinned KV: the cache order is hot weights → experts → KV, streamed / written bytes bypass
    big = evaluate(s0.replace("chip", replace(s0.chip, slc_mib=16384.0)))
    assert big.stages[0].mem.slc_residency == 1.0 and big.stages[0].dram["kv_write"] == r0.stages[0].dram["kv_write"]
    # lru: all-or-nothing (cyclic access)
    lo = evaluate(s0.replace("chip", replace(s0.chip, slc_mib=8192.0, slc_policy="lru")))
    _same(lo, r0)
    hi = evaluate(s0.replace("chip", replace(s0.chip, slc_mib=32768.0, slc_policy="lru")))
    dh = hi.stages[0].dram
    assert hi.stages[0].mem.slc_all and _close(dh["total"], dh["kv_write"] + dh["lookup"])
    assert hi.tpot < r0.tpot
    # a slow SLC can bind
    slow = evaluate(s0.replace("chip", replace(s0.chip, slc_mib=16384.0, slc_GBps=50.0)))
    assert slow.bound == "SLC" and slow.tpot > r0.tpot * 0.5
    # video: pin keeps weights, streamed activations still go to DRAM
    v0 = Scenario(model="wan2.1-1.3b", mem_id=HBM, workload=Workload(pipeline=False))
    a, b = evaluate(v0), evaluate(v0.replace("chip", replace(v0.chip, slc_mib=8192.0)))
    assert b.stages[0].dram["act"] == a.stages[0].dram["act"] > 0
    assert b.stages[0].dram["weights"] < a.stages[0].dram["weights"]
    assert _close(b.stages[0].time.dram_bytes + b.stages[0].time.slc_bytes, a.stages[0].time.dram_bytes, 1e-9)
    # SLC larger than DRAM is flagged
    huge = evaluate(s0.replace("chip", replace(s0.chip, slc_mib=1e6)))
    assert any("SLC 容量" in w for w in huge.warnings)


def test_energy_counts_split_slc_and_d2d():
    s0 = Scenario(model="qwen3-8b", layout=Layout(tp=8), serving=Serving(batch=8), mem_id=HBM)
    c0 = action_counts(evaluate(s0))["counts"]
    s1 = replace(s0.replace("chip", replace(s0.chip, slc_mib=2048.0)), package_cards=4, d2d_enabled=True)
    r1 = evaluate(s1)
    c1 = action_counts(r1)["counts"]
    assert c0["slc"] == 0 and c0["d2d"] == 0
    assert _close(c1["slc"] + c1["dram"], c0["dram"], 1e-9) and c1["slc"] > 0
    assert _close(c1["d2d"] + c1["link"], c0["link"], 1e-9) and 0 < c1["d2d"] < c0["link"]
    t = EnergyTable(pJ_bit_dram=6.0, pJ_bit_slc=1.0, pJ_bit_d2d=0.5, pJ_bit_link=5.0)
    e = energy_report(r1, t)
    u = action_counts(r1)["units"]
    for k, pj in (("slc", 1.0), ("dram", 6.0), ("d2d", 0.5), ("link", 5.0)):
        assert _close(e["J_by_action"][k], c1[k] / u * pj * 8e-12)
    assert set(e["missing"]) == {"pJ_mac", "pJ_vec", "pJ_bit_sram", "idle_W", "pJ_bit_net"}


def test_validation_api_cli():
    for bad in (dict(slc_mib=-1.0), dict(slc_GBps=0.0), dict(slc_policy="fifo")):
        try:
            replace(CHIPS["100T"], **bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    for pc in (0, True, 2.5, 4096):
        try:
            api.api_eval({"scenario": {"model": "qwen3-8b", "package_cards": pc}})
            raise AssertionError(pc)
        except ApiError:
            pass
    out = api.api_eval({"scenario": {"model": "qwen3-8b", "layout": {"tp": 4}, "package_cards": 2,
                                     "d2d": {"GBps": 1000}, "chip": {"slc_mib": 512, "slc_policy": "pin"}},
                        "energy": {"pJ_bit_slc": 1, "pJ_bit_d2d": 0.3}})
    st = out["stages"][0]
    assert st["link_GB"]["d2d"] > 0 and st["slc_GB"]["weights"] > 0 and st["mem"]["slc_MiB"] == 512
    assert "slc" in st["t_ms"] and out["scenario"]["d2d"]["GBps"] == 1000.0
    assert set(out["energy"]["provided"]) == {"pJ_bit_slc", "pJ_bit_d2d"}
    sw = api.api_sweep({"scenario": {"model": "qwen3-8b", "layout": {"tp": 8}, "serving": {"phase": "prefill",
                        "prompt": 2048}}, "path": "package_cards", "values": [1, 2, 4, 8]})
    ttft = [x["ttft_ms"] for x in sw["rows"]]
    assert ttft[-1] < ttft[0]
    sw = api.api_sweep({"scenario": {"model": "qwen3-8b", "serving": {"batch": 8}}, "path": "chip.slc_mib",
                        "values": [0, 1024, 8192]})
    tp = [x["tpot_ms"] for x in sw["rows"]]
    assert tp[0] > tp[1] > tp[2]
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    b = cli._body(p.parse_args(["--model", "qwen3-8b", "--slc-mib", "256", "--slc-policy", "lru", "--package-cards", "4",
                                "--d2d-GBps", "1500", "--link-alpha-us", "5", "--pJ-bit-d2d", "0.4"]))
    sc = b["scenario"]
    assert sc["chip"] == {"slc_mib": 256.0, "slc_policy": "lru"} and sc["package_cards"] == 4
    assert sc["d2d"] == {"GBps": 1500.0} and sc["link"] == {"alpha_us": 5.0} and b["energy"] == {"pJ_bit_d2d": 0.4}
    plain = cli._body(p.parse_args(["--model", "qwen3-8b"]))["scenario"]
    assert not {"chip", "d2d", "link", "package_cards"} & set(plain)
    cat = api.api_catalog()
    assert cat["chips"]["100T"]["slc_mib"] == 0 and cat["defaults"]["package_cards"] == 1
