"""0.120: per-kernel launch overhead (Chip.kernel_us, opt-in) on structure-model kernels incl. every chunk of a chunked
triangle attention; chunk-fitting unfused attention still streams q / k / v / out once (no chunk-faster-than-fused);
random invariants of modules untouched since 0.100 (queueing, lengths, memory catalog)."""
import dataclasses
import math
import random

from accel_dse import mem_catalog as MC
from accel_dse.core import lengths as Ln
from accel_dse.core import queueing as Q
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario, Workload


def _lat(ku=0.0, mem=None, **wl):
    s = Scenario(model="openfold", workload=Workload(seq_len=512, **wl), **({"mem_id": mem} if mem else {}))
    s = dataclasses.replace(s, chip=dataclasses.replace(s.chip, kernel_us=ku))
    return evaluate(s)


def test_chunk_never_faster_than_fused():
    for mem in (None, "hbm3e_8s_12h24g_9200"):
        for ku in (0.0, 0.53, 5.0):
            f = _lat(ku, mem).domain_summary()["latency_s"]
            for c in (4, 64, 512):
                t = _lat(ku, mem, tri_attn="chunked", tri_chunk=c).domain_summary()["latency_s"]
                assert t >= f * (1 - 1e-9), (mem, ku, c, t, f)
    assert abs(_lat(0.0, tri_attn="chunked", tri_chunk=4).domain_summary()["latency_s"]
               / _lat(0.0).domain_summary()["latency_s"] - 1) < 1e-9          # no overhead → same as fused


def test_launch_count_closed_form():
    # openfold trunk tri_att rows per module = N (no DAP); chunk c → ⌈N / c⌉ chunks × 3 kernels; fused → 1
    a = _lat(1.0, tri_attn="chunked", tri_chunk=4).stages[0].time.t_sync
    b = _lat(1.0, tri_attn="chunked", tri_chunk=8).stages[0].time.t_sync
    z = _lat(0.0, tri_attn="chunked", tri_chunk=4).stages[0].time.t_sync
    f1 = _lat(1.0).stages[0].time.t_sync
    assert a > b > f1 > z
    # halving the chunks halves the extra chunk launches: (a − f1) ≈ 2 (b − f1) up to the fused-vs-3 kernel offset
    assert abs((a - f1) / (b - f1) - 2) < 0.05


def test_default_off_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "84df3be517a53ef4"   # 0.173 re-baseline (was 0a954915c5fb003f)
    assert _fp({}) == "1b755ac357ec6d2e"   # 0.173 re-baseline (was 53ee9e08165795c6)


def test_invariants_queueing_lengths_memory():
    rng = random.Random(120)
    for _ in range(200):
        tau = rng.uniform(0.01, 2); rho = rng.uniform(0.05, 0.95); lam = rho / tau
        a = Q.md1(lam, tau)
        assert a["p50"] <= a["p90"] <= a["p99"]
        if 0 < a["p90"] <= 6 * tau:
            assert abs(Q.md1_cdf(lam, tau, a["p90"]) - 0.9) < 2e-3
        m1 = Q.mdc_wait(lam, tau, 1, cs2=1.0)
        assert abs(m1["mean"] - rho * tau / (1 - rho)) < 1e-9 * max(1.0, m1["mean"])
        mean, cv = rng.uniform(1, 20000), rng.uniform(0.05, 3)
        vs = [x for _, x in Ln.lognormal_bins(mean, cv)]
        assert vs == sorted(vs) and min(vs) >= 1 and sum(vs) == max(len(vs), round(len(vs) * mean))
    for t in ("LPDDR5", "LPDDR5X", "LPDDR6", "HBM3", "HBM3E", "HBM4", "HBM4E"):
        base = MC.default_spec(t)
        for r in MC.rates_for(t):
            for n in MC.counts_for(t):
                sp = dataclasses.replace(base, rate_MTps=r, n_units=n)
                assert math.isclose(sp.raw_GBps, sp.bus_bits * r / 8e3, rel_tol=1e-9)
                assert 0 < sp.effective_GBps <= sp.payload_GBps <= sp.raw_GBps + 1e-9
                rt = MC.parse_mem_id(sp.id)
                assert (rt.mem_type, rt.rate_MTps, rt.n_units) == (sp.mem_type, sp.rate_MTps, sp.n_units)
