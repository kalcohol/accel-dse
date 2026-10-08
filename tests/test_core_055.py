"""0.55 — colocated burstiness (reduced-clock batch chain, busy-period spread, pair term κ), occupancy
autocorrelation window, quasi-static chunked TTFT, worst-gap up-crossings, KV-capacity admission / preemption,
V4 scenario families + multi-seed grid."""

from __future__ import annotations

import dataclasses
import math
import random

from accel_dse.core.disagg import disagg_report
from accel_dse.core.pdqueue import (_busy_spread, _decode_birth_death, _kv_slots, _mg1_busy, _occ_tau, _pair_kappa,
                                    _shape_batch)
from accel_dse.core.pdsim import capture_ctx, compare, simulate
from accel_dse.core.queueing import md1, md1_cdf, mg1_mix_sum_quantile, mg1_sum_quantile
from accel_dse.core.scenario import PDConfig
from accel_dse.core.validation import V4_FAMILIES, v4_scenario


def _moments(X):
    return sum((i + 1) * x for i, x in enumerate(X)), sum((i + 1) ** 2 * x for i, x in enumerate(X))


def test_mg1_busy_matches_theory():
    """Busy-period count: E[X] = 1/(1−ρ), Var X = (ρ(1−ρ) + λ²E[S²])/(1−ρ)³ (single value = Borel, mixed = Takács)."""
    for svc in ([(1.0, 0.1)], [(0.5, 0.05), (0.3, 0.12), (0.2, 0.3)]):
        lam = 3.0
        X, dur = _mg1_busy(lam, svc, want_dur=True)
        ES = sum(w * s for w, s in svc)
        ES2 = sum(w * s * s for w, s in svc)
        rho = lam * ES
        m1, m2 = _moments(X)
        assert abs(sum(X) - 1) < 1e-9
        assert abs(m1 - 1 / (1 - rho)) < 2e-3 * m1
        var = (rho * (1 - rho) + lam * lam * ES2) / (1 - rho) ** 3
        assert abs((m2 - m1 * m1) - var) < 0.02 * var
        EV = sum(t * p for t, p in dur)
        assert abs(EV - ES / (1 - rho)) < 5e-3 * EV
    assert _mg1_busy(20.0, [(1.0, 0.1)]) == (None, None)


def test_md1_exact_cdf():
    """Exact M/D/1 wait CDF: atom 1 − ρ at 0, monotone, → 1; matches a Lindley recursion at the 90th percentile."""
    lam, tau = 0.8, 1.0
    assert abs(md1_cdf(lam, tau, 0.0) - 0.2) < 1e-9
    xs = [0.1 * i for i in range(80)]
    vals = [md1_cdf(lam, tau, x) for x in xs]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])) and vals[-1] > 0.95
    rng, w, ws = random.Random(5), 0.0, []
    for _ in range(300000):
        w = max(0.0, w + tau - rng.expovariate(lam))
        ws.append(w)
    ws.sort()
    sim90 = ws[int(0.9 * len(ws))]
    assert abs(md1(lam, tau)["p90"] - sim90) < 0.06 * sim90


def test_batch_chain_reduces_to_birth_death(ctx_cache={}):
    rep, ctx = capture_ctx(v4_scenario(0.6))
    cp, out, B = ctx["cpool"], ctx["out"], ctx["B"]
    lam = rep["queue"]["lambda_rps"] / ctx["r_c"]
    a = _decode_birth_death(cp, lam, out, B)
    b = _decode_birth_death(cp, lam, out, B, batch=(lam, [1.0], 1.0))
    assert abs(a["occupancy"] - b["occupancy"]) < 1e-9 * max(1, a["occupancy"])
    assert abs(a["tpot"] - b["tpot"]) < 1e-12
    # batches raise the occupancy at the same mean rate
    X = [0.5, 0.3, 0.2]
    EX = _moments(X)[0]
    c = _decode_birth_death(cp, lam, out, B, batch=(lam / EX, X, 1.0))
    assert c["occupancy"] > a["occupancy"]


def test_shape_batch_and_kappa():
    X = [0.6, 0.25, 0.1, 0.05]
    lam_b = 2.0
    EX = _moments(X)[0]
    F = sum((i + 1) * i * x for i, x in enumerate(X))
    for a2 in (0.0, 0.3, 1.0, 1.6):
        nu, pmf = _shape_batch(lam_b, X, lam_b * EX, a2)
        m1 = sum((i + 1) * x for i, x in enumerate(pmf))
        f2 = sum((i + 1) * i * x for i, x in enumerate(pmf))
        assert abs(sum(pmf) - 1) < 1e-9
        assert abs(nu * m1 - lam_b * EX) < 1e-9                  # same mean rate
        assert abs(nu * f2 - a2 * lam_b * F) < 1e-9 * max(1, F)   # pair term scaled by a²
    assert abs(_pair_kappa([(1.0, 512)]) - 2.0) < 1e-12
    n = 4000                                                      # exponential law (quantile bins) → κ ≈ 1
    expo = [(1.0, -math.log(1 - (i + 0.5) / n)) for i in range(n)]
    assert abs(_pair_kappa(expo) - 1.0) < 0.02
    # prefill-first: the whole busy period is excess → the instantaneous batch is exact (a² = 1)
    assert _busy_spread(2.0, [(1.0, 0.1, 0.0)], X, 1.0) == 1.0
    assert 0 < _busy_spread(2.0, [(1.0, 0.1, 0.05)], [0.7, 0.2, 0.1], 1.5) < 1


def test_occupancy_tau_infinite_server():
    """M/M/∞ occupancy has autocorrelation e^{−μt} exactly → τ_int = 1/μ."""
    lam, mu, N = 6.0, 0.5, 120
    a = lam / mu
    logp = [k * math.log(a) - a - math.lgamma(k + 1) for k in range(N)]
    pi = [math.exp(x) for x in logp]
    tau = _occ_tau(pi, 10 ** 6, (lam, [1.0]), lambda k: k * mu)
    assert abs(tau - 1 / mu) < 1e-3 / mu


def test_mix_quantile_single_env_matches():
    lam = 2.0
    taus, ws, lats = [0.1, 0.25], [0.6, 0.4], [0.1, 0.25]
    q1 = mg1_sum_quantile(lam, taus, ws, lats, 0.99)
    q2 = mg1_mix_sum_quantile(lam, [(1.0, list(zip(ws, taus, lats)))], 0.99)
    assert abs(q1 - q2) < 1e-6 * q1
    # a slower environment mixed in only lengthens the tail
    q3 = mg1_mix_sum_quantile(lam, [(0.5, list(zip(ws, taus, lats))),
                                    (0.5, [(w, 1.3 * t, 1.3 * t) for w, t in zip(ws, taus)])], 0.99)
    assert q3 > q2
    # vacations (decode-only iterations) add delay
    q4 = mg1_mix_sum_quantile(lam, [(1.0, list(zip(ws, taus, lats)), 0.02)], 0.99)
    assert q2 < q4 < q2 + 0.02 + 1e-9


def test_v4_families_and_seeds():
    assert set(V4_FAMILIES) == {"dense8b", "moe30b", "tp4_32b"}
    for fam in V4_FAMILIES:
        q = disagg_report(v4_scenario(0.6, family=fam))["queue"]
        assert all(q["modes"][m]["stable"] for m in ("pd", "coloc_prefill_first", "coloc_chunked")), fam
    r = compare(v4_scenario(0.3, family="moe30b"), n_req=150, warmup=40, seeds=(1, 2), modes=("pd",))
    row = r["modes"]["pd"]
    assert "noise" in row and row["complete"] and all(v >= 0 for v in row["noise"].values())


def _kv(scn, pol, gb):
    return dataclasses.replace(scn, pd=dataclasses.replace(scn.pd, kv_policy=pol, kv_capacity_GB=gb))


def test_kv_policy_off_is_identity_and_binds():
    base = v4_scenario(0.85)
    q0 = disagg_report(base)["queue"]["modes"]
    big = disagg_report(_kv(base, "wait", 1000.0))["queue"]["modes"]    # capacity holds the full batch → inactive
    for m in q0:
        assert big[m]["kv_cap"]["binds"] is False
        assert abs(big[m]["tpot_mean_ms"] - q0[m]["tpot_mean_ms"]) < 1e-9
        assert big[m]["ttft_ms"] == q0[m]["ttft_ms"]
    w = disagg_report(_kv(base, "wait", 12.0))["queue"]["modes"]["pd"]
    assert w["kv_cap"]["binds"] and w["kv_cap"]["slots_kv"] < 64 and w["decode"]["slots"] == w["kv_cap"]["slots_kv"]
    assert w["kv_cap"]["slot_wait_mean_ms"] > 0
    rc = disagg_report(_kv(base, "recompute", 12.0))["queue"]["modes"]["pd"]
    assert rc["kv_cap"]["slots_kv"] >= w["kv_cap"]["slots_kv"]        # optimistic footprint (S + generated)
    assert rc["kv_cap"]["preempt_per_req"] >= 0
    # derived capacity (free DRAM after weights) is reported and does not bind for this 8B case
    d = disagg_report(_kv(base, "wait", None))["queue"]["modes"]["pd"]["kv_cap"]
    assert d["source"] == "derived_free_dram" and d["capacity_tokens"] > 64 * 4608 and not d["binds"]
    try:
        PDConfig(kv_policy="swap")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_kv_policy_des():
    base = v4_scenario(0.85)
    for pol in ("wait", "recompute"):
        rep, ctx = capture_ctx(_kv(base, pol, 12.0))
        lam = rep["queue"]["lambda_rps"]
        s = simulate(ctx, lam, "pd", n_req=500, warmup=120, seed=3)
        assert s["complete"] and s["slot_wait"]["mean"] > 0 and s["preempt_per_req"] >= 0
        if pol == "wait":
            assert s["preempt_per_req"] == 0
        s0 = simulate(ctx | {"kv_policy": "off"}, lam, "coloc_chunked", n_req=300, warmup=80, seed=3)
        assert "slot_wait" not in s0
