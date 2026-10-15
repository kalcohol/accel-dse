"""0.127 hostile review round 3 (hardware side): GEMM mapping lower bounds / monotonicity / reconf optimality,
energy MAC counts layout-invariant, hardware monotonicity, capacity accounting identity, fewest-cards fit (not only
powers of two), Pareto front vs brute force."""
import dataclasses
import random

from accel_dse import api
from accel_dse.core.dtypes import FORMATS, fmt, gemm_exec
from accel_dse.core.energy import energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.mapping import ORGS, gemm_cost
from accel_dse.core.memplan import RUNTIME_RESERVE
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario
from accel_dse.core.search import pareto


def test_gemm_mapping_bounds_and_monotone():
    rng = random.Random(127)
    chips = list(CHIPS.values())
    chips += [dataclasses.replace(c, instance_sched=s) for c in list(chips) for s in ("auto", "wide", "cta")
              if s != c.instance_sched]
    fms = [f for f in ("bf16", "fp8", "int4", "mxfp4", "int8") if f in FORMATS]
    for _ in range(1500):
        ch, org = rng.choice(chips), rng.choice(ORGS)
        m, k, n = rng.choice([1, 3, 16, 33, 500, 4096]), rng.choice([64, 512, 4096]), rng.choice([64, 1024, 18432])
        cnt, wf = rng.choice([1, 3, 8, 64]), rng.choice(fms)
        c = gemm_cost(ch, org, m, k, n, count=cnt, w_fmt=wf)
        _, rate, _, _ = gemm_exec(wf, "bf16", ch.formats)
        assert c.cycles >= m * k * n * cnt / (ch.macs * rate) * (1 - 1e-9)           # ≤ peak MAC rate
        assert c.cycles >= k * n * fmt(wf).bits / 8 * cnt / ch.port_Bpc * (1 - 1e-9)  # every weight through the port
        # 0.139: the multi-core picker minimises max(array, replicated-conversion) cycles — monotone in that score
        sc = lambda g: max(g.cycles, g.convert_elems * 2.0 / (ch.formats.dequant_elems_per_cycle or ch.lanes))
        assert sc(gemm_cost(ch, org, m, k, n, count=cnt + 1, w_fmt=wf)) >= sc(c) * (1 - 1e-9)
        if org == "reconf":
            for o in ("os", "ws_edge", "ws_broad", "os_vec"):
                assert sc(c) <= sc(gemm_cost(ch, o, m, k, n, count=cnt, w_fmt=wf)) * (1 + 1e-9)


def test_energy_mac_counts_layout_invariant():
    """Useful MACs per token do not depend on PP / TP / DP for a dense model (replicated ops only appear with
    replicated work, e.g. the MoE router on every TP rank)."""
    for ph in ("decode", "prefill"):
        base = Scenario(model="qwen3-8b").replace("serving.batch", 8)
        if ph == "prefill":
            base = base.replace("serving.phase", "prefill").replace("serving.prompt", 1024)
        e0 = energy_report(evaluate(base))["counts_per_unit"]["mac"]
        for lay in (Layout(1, 2, 1, 1, 1), Layout(2, 1, 1, 1, 1), Layout(2, 2, 1, 1, 1), Layout(1, 3, 1, 1, 1)):
            e = energy_report(evaluate(base.replace("layout", lay)))["counts_per_unit"]["mac"]
            assert abs(e / e0 - 1) < 1e-6, (ph, lay, e / e0)
    b = Scenario(model="qwen3-30b-a3b").replace("serving.batch", 8)
    e0 = energy_report(evaluate(b))["counts_per_unit"]["mac"]
    e = energy_report(evaluate(b.replace("layout", Layout(1, 1, 2, 2, 1))))["counts_per_unit"]["mac"]
    assert abs(e / e0 - 1) < 1e-6


def test_better_hardware_never_slower():
    paths = {"chip.sram_mib": [16, 64, 256], "chip.sram_port_Bpc": [512, 2048, 8192], "chip.freq_ghz": [0.8, 1.4, 2.0],
             "mem_eff": [0.5, 0.9], "link.GBps": [50, 400], "chip.slc_mib": [0, 256], "chip.gemv_macs": [0, 4096]}
    for mid, lay in (("qwen3-8b", Layout(1, 2, 1, 1, 1)), ("qwen3-30b-a3b", Layout(1, 2, 1, 2, 1)),
                     ("wan2.1-1.3b", Layout(1, 1, 1, 1, 1, 2))):
        base = Scenario(model=mid).replace("layout", lay).replace("serving.batch", 16)
        for p, vals in paths.items():
            ts = []
            for v in vals:
                r = evaluate(base.replace(p, v))
                ts.append(r.latency if r.workload is not None else r.step)
            assert all(b <= a * (1 + 1e-9) for a, b in zip(ts, ts[1:])), (mid, p, ts)


def test_capacity_identity_llm():
    for mid, lay in (("qwen3-8b", Layout(1, 1, 1, 1, 1)), ("qwen3-next-80b-a3b", Layout(1, 4, 1, 4, 1)),
                     ("deepseek-v4-flash", Layout(2, 8, 1, 8, 1)), ("esmfold", Layout(1, 1, 1, 1, 1))):
        r = evaluate(Scenario(model=mid).replace("layout", lay).replace("serving.batch", 4))
        for st in r.stages:
            m = st.mem
            rec = m.stored_w + m.kv_total + m.state_total + RUNTIME_RESERVE + m.act_total + m.pipe_w
            assert abs(m.dram_need / rec - 1) < 1e-12, (mid, st.index)


def test_fit_fewest_cards_not_only_pow2():
    r = api.api_fit({"scenario": {"model": "llama-3.3-70b"}})
    mc = r["min_cards"]
    assert mc["cards"] == 3 and mc["pow2_cards"] == 4 and mc["layout"] == "PP1·TP3"
    lay = Layout(**mc["layout_obj"])
    assert evaluate(Scenario(model="llama-3.3-70b").replace("layout", lay).replace("serving.batch", 1)).fits
    assert not any(evaluate(Scenario(model="llama-3.3-70b").replace("layout", Layout(pp, tp, 1, 1, 1))
                            .replace("serving.batch", 1)).fits for pp, tp in ((1, 2), (2, 1)))


def test_pareto_bruteforce():
    rng = random.Random(5)
    for _ in range(300):
        pts = [(rng.choice([1.0, 2.0, 3.0, rng.random() * 5]), rng.choice([1.0, 2.0, rng.random() * 5]), i)
               for i in range(rng.randint(1, 25))]
        front = pareto(pts)
        dom = lambda a, b: a[0] <= b[0] and a[1] >= b[1] and (a[0] < b[0] or a[1] > b[1])
        ref = [p for p in pts if not any(dom(q, p) for q in pts)]
        got = {(p[0], p[1]) for p in front}
        assert got == {(p[0], p[1]) for p in ref}, (pts, front)
        assert len(front) == len(got)
