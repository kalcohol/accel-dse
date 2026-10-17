"""0.226: default tree acceptance = chain (guaranteed lower bound) with the [chain, indep] band always reported;
"rank" model from the Sequoia positional acceptance vector on an SGLang-style tree; what-if headline; fuzz
(band ordering chain ≤ rank ≤ indep, J/token × tok/s/card = W/card)."""
import random
import sys

from accel_dse import api
from accel_dse.core.energy import energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.schedule import (SEQUOIA_ACCEPT_VECTOR, rank_accept, rank_tree_tau, spec_expected_tokens,
                                     spec_tree_tau, tree_tokens)
from accel_dse.core.spec_refs import tree_check

sys.path.insert(0, "tests")


def _sc(model, b=4, k=0, a=0.7, dm="", topk=1, n=0, am="", roll="", track=0):
    s = Scenario(chip=CHIPS["1P"], model=model)
    for f, v in (("batch", b), ("spec_k", k), ("spec_accept", a), ("ctx", 4096), ("spec_topk", topk),
                 ("spec_draft_tokens", n), ("mamba_track_interval", track)):
        s = s.replace("serving." + f, v)
    for f, v in (("draft_model", dm), ("spec_tree_accept_model", am), ("spec_rollback", roll)):
        if v:
            s = s.replace("serving." + f, v)
    return s


# ---- 1. default + band
def test_default_is_chain():
    assert Serving().spec_tree_accept_model == "chain"
    assert "spec_tree_accept_model" not in api.text_view(_sc("qwen3.8-27b", 1, 3, topk=4, n=16)).to_dict()["serving"]


def test_band_always_reported():
    r = evaluate(_sc("qwen3.8-27b", 1, 3, 0.7, topk=4, n=16))
    sp = r.spec
    assert sp["accept_model"] == "chain" and sp["tau"] == sp["tau_chain"]
    assert sp["tau_chain"] <= sp["tau_rank"] <= sp["tau_indep"]
    assert abs(sp["tpot_ms_chain"] - r.tpot * 1e3) < 1e-9 and sp["tpot_ms_indep"] < sp["tpot_ms_chain"]
    for m in ("chain", "rank", "indep"):
        assert abs(sp["tpot_ms_" + m] * sp["tau_" + m] - r.step * 1e3) < 1e-9
    assert set(sp["check"]) >= {"chain", "rank", "indep"}
    assert any("区间" in w and "主值" in w for w in r.warnings)


# ---- 2. rank model
def test_rank_vector_and_scaling():
    P = SEQUOIA_ACCEPT_VECTOR
    assert len(P) == 31 and P[0] == 0.7732 and sum(P) < 1
    for a in (0.0, 0.3, P[0], 0.9, 1.0):
        ps = [rank_accept(a, j) for j in range(40)]
        assert ps[0] == a and sum(ps) <= 1 + 1e-12 and all(x >= 0 for x in ps)
    assert all(abs(rank_accept(P[0], j) - P[j]) < 1e-12 for j in range(len(P)))   # published vector at a = P_0


def test_rank_tree_limits():
    assert abs(rank_tree_tau(0.7, 3, 1, 4) - spec_expected_tokens(3, 0.7)) < 1e-12     # topk 1 → the chain
    assert rank_tree_tau(1.0, 3, 4, 16) == 4.0 and rank_tree_tau(0.0, 3, 4, 16) == 1.0
    assert rank_tree_tau(0.7, 3, 4, 16) <= rank_tree_tau(0.7, 3, 4, 32) + 1e-12  # more tokens never hurt


def test_rank_check_inside_band_no_fit():
    c, lo, hi = tree_check("rank"), tree_check("chain"), tree_check("indep")
    assert lo["err_max"] < c["err_min"] <= c["err_max"] < hi["err_min"]
    assert max(abs(c["err_min"]), abs(c["err_max"])) < 0.05       # 0.226 observed +0.8 … +3.6 %
    for r in c["rows"]:
        assert r["model_tau"] < r["tree_tau"] * 1.05


# ---- 3. what-if headline
def test_compare_headline_and_row_uses_chain():
    s = _sc("qwen3.8-27b", 8, 3, 0.7)
    c = api.spec_tree_compare(s)
    assert c["headline"] == "chain" and c["crossover"] == c["crossover_chain"]
    for x in c["rows"]:
        assert x["tree_chain"]["tpot_ms"] >= x["tree_rank"]["tpot_ms"] >= x["tree_indep"]["tpot_ms"]
        assert x["tree_wins_chain"] is False
    w = api.balance_whatif(s, evaluate(s))
    row = [x for x in w["rows"] if x["key"] == "spec_tree_tree"][0]
    b8 = [x for x in row["spec_tree_compare"]["rows"] if x["batch"] == 8][0]
    assert abs(row["tpot_ms"] - b8["tree_chain"]["tpot_ms"]) < 1e-9         # lever ranked on the chain headline
    assert row["tpot_ratio"] > 1                      # under the chain headline the tree is slower at batch 8
    c2 = api.spec_tree_compare(_sc("qwen3.8-27b", 8, 3, 0.7, am="rank"))
    assert c2["headline"] == "rank" and c2["crossover"] == c2["crossover_rank"]


# ---- 4. fuzz
def test_fuzz_band_order_and_invariant():
    from test_core_217 import _table
    t, rnd = _table(), random.Random(226)
    for _ in range(16):
        a, k, topk = rnd.random(), rnd.randint(1, 8), rnd.randint(2, 10)
        n = rnd.choice([0, 2, rnd.randint(2, min(64, 1 + topk + (k - 1) * topk * topk))])
        c, r, i = (spec_tree_tau(a, k, topk, n, m) for m in ("chain", "rank", "indep"))
        assert 1 - 1e-12 <= c <= r + 1e-12 and r <= i + 1e-12 and i <= tree_tokens(k, topk, n) + 1e-9
    for _ in range(12):
        m = rnd.choice(["qwen3.8-27b", "qwen3-32b", "glm-5.3-flash", "qwen3.8-27b-ready"])
        dm = "qwen3-0.6b" if m == "qwen3-32b" else ""
        k, topk = rnd.randint(1, 5), rnd.choice([1, 2, 4, 8])
        n = rnd.choice([0, min(64, 1 + k * topk)]) if topk > 1 else 0
        am, a, b = rnd.choice(["chain", "rank", "indep"]), rnd.choice([0.0, 0.5, 0.9, 1.0]), rnd.choice([1, 8, 32])
        r = evaluate(_sc(m, b, k, a, dm, topk, n, am, rnd.choice(["snapshot", "replay"]), rnd.choice([0, 256])))
        for v in (r.step, r.tpot, r.throughput):
            assert v == v and 0 < v < float("inf"), (m, k, topk, n, am, a, b)
        if r.spec and "tau_chain" in r.spec:
            assert r.spec["tau_chain"] - 1e-12 <= r.spec["tau_rank"] <= r.spec["tau_indep"] + 1e-12
            assert r.tokens_per_step == r.spec["tau_" + am]
        e = energy_report(r, t)
        if e.get("J_per_unit"):
            assert abs(e["J_per_unit"] * (b / r.tpot / r.scenario.layout.cards) / e["avg_W_per_card"] - 1) < 1e-3
