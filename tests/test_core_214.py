"""0.214: two-die package lever (B200 approach) on the existing multi-card + D2D model."""
from accel_dse import api
from accel_dse.core.budget import shoreline_check
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario

RWKV = "rwkv7-g1k-7.2b"


def _w(model):
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": model}})["sensitivity"]["balance_whatif"]
    return w["base"], {x["key"]: x for x in w["rows"]}


def test_d2d_edge_reduces_shoreline_sides():
    s = Scenario(chip=CHIPS["1P"])
    one = shoreline_check(s, 400.0)
    two = shoreline_check(s.replace("package_cards", 2).replace("d2d_enabled", True).replace("layout.tp", 2), 400.0)
    assert one["sides_max"] == 4 and two["sides_max"] == 3 and two["d2d_edges"] == 1
    assert two["edge_typical_mm"] == one["edge_typical_mm"] and two["edge_max_mm"] < one["edge_max_mm"]


def test_two_die_rwkv_1p():
    b, rows = _w(RWKV)
    t, x2 = rows["two_die"], rows["dram_bw_x2"]
    assert t["dies"] == 2 and t["d2d_phy_mm2"] > 0 and t["d2d_J_per_tok"] is not None
    assert t["area_mm2"] < 0.65 * x2["area_mm2"]                 # per die ≈ half the 8-stack die (+ D2D PHY)
    assert t["shoreline"]["stacks"] == 4 and t["shoreline"]["sides_max"] == 3
    assert t["reachable"] and abs(t["critical_batch"] - x2["critical_batch"]) / x2["critical_batch"] < 0.1
    assert abs(t["tpot_ms"] / x2["tpot_ms"] - 1) < 0.05            # same totals: TPOT within 5 % (D2D latency)
    assert t["d2d_J_per_tok"] < 0.01 * t["J_per_tok"]              # D2D energy is a small share
    assert "UCIe" in t["label"] or "ucie" in t["label"]


def test_two_die_qwen_kv_bound_stays_unreachable():
    b, rows = _w("qwen3-8b")
    assert not rows["two_die"]["reachable"] and not rows["dram_bw_x2"]["reachable"]
