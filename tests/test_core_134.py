"""0.134 opt-in node-deduplicating EP all-to-all (serving.moe_a2a) and the SM share of its kernels during prefill
overlap (serving.moe_a2a_sm): closed form vs Monte-Carlo, network bytes only, defaults unchanged, validation, CLI."""
import random

from accel_dse.core import extval as X
from accel_dse.core.ir import node_dedup_factor
from accel_dse.core.scenario import Scenario
from accel_dse.core.schedule import fabric_collective
from accel_dse.core.hardware import Link


def test_node_dedup_factor_vs_monte_carlo():
    rng = random.Random(134)
    for E, k, N in ((256, 8, 4), (256, 8, 8), (128, 4, 2), (64, 6, 16)):
        per = E // N
        tot = 0
        for _ in range(4000):
            nodes = {e // per for e in rng.sample(range(E), k)}
            tot += len(nodes - {0})          # remote distinct nodes seen from node 0
        mc = tot / 4000 / (k * (N - 1) / N)
        assert abs(node_dedup_factor(E, k, N) - mc) < 0.03, (E, k, N)
    assert node_dedup_factor(256, 8, 1) == 1.0 and node_dedup_factor(256, 1, 4) == 1.0
    assert abs(node_dedup_factor(256, 8, 4) - 0.4518) < 1e-3


def test_net_scale_only_network_tier():
    a = fabric_collective("alltoall", 1e8, 32, Link(400, 1), net=Link(50, 5), node=8)
    b = fabric_collective("alltoall", 1e8, 32, Link(400, 1), net=Link(50, 5), node=8, net_scale=0.5)
    assert abs(b[0] - a[0] / 2) < 1e-12 * a[0]          # network-bound: halves
    c = fabric_collective("allreduce", 1e8, 32, Link(400, 1), net=Link(50, 5), node=8, net_scale=0.5)
    assert c == fabric_collective("allreduce", 1e8, 32, Link(400, 1), net=Link(50, 5), node=8)


def test_prefill_rows_opt_in_and_defaults():
    R = {r["id"]: r for r in X.rows(X.DATA_MOE)}
    r = R["dsp-pf-4096"]
    base = X.predict(r, "catalog")["pred"]
    node = X.predict(r, X.a2a_variant("catalog", "auto"))["pred"]
    sm = X.predict(r, X.a2a_variant("catalog", "auto", X.DEEPEP_NORMAL_SM))["pred"]
    assert node > base * 1.2 and base < sm < node
    d = R["sgl-dec-2000-default"]                       # auto: decode keeps per-pair (low-latency kernels)
    assert X.predict(d, X.a2a_variant("catalog", "auto", X.DEEPEP_NORMAL_SM))["pred"] == X.predict(d, "catalog")["pred"]
    s = Scenario()
    assert s.serving.moe_a2a == "pair" and s.serving.moe_a2a_sm == 0.0


def test_validation_and_cli():
    for k, v in (("serving.moe_a2a", "ring"), ("serving.moe_a2a_sm", 0.95), ("serving.moe_a2a_sm", True)):
        try:
            Scenario().replace(k, v)
            raise AssertionError(k)
        except ValueError:
            pass
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "accel_dse/cli.py").read_text()
    assert "--moe-a2a" in src and "--moe-a2a-sm" in src
