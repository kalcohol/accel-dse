"""0.225: published tree-drafting acceptance references + the check of the tree-acceptance 「假设」 model (no fit);
upper / lower acceptance bounds (serving.spec_tree_accept_model); chain-vs-tree what-if row with the per-batch
crossover; seeded fuzzer incl. the J/token × tok/s/card = W/card invariant."""
import random
import sys

from accel_dse import api
from accel_dse.core.energy import energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario
from accel_dse.core.schedule import spec_expected_tokens, spec_tree_accept, tree_tokens
from accel_dse.core.spec_refs import iid_accept, tree_check, tree_refs

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


# ---- 1. references + check
def test_refs_file_shape():
    d = tree_refs()
    assert len(d["matched"]) == 6 and all(x["tree_tau"] > x["chain_tau"] > 1 for x in d["matched"])
    t = d["matched_tree"]
    assert sum(t["nodes_per_depth"]) + 1 == t["draft_tokens"] == 26 and len(t["nodes_per_depth"]) == t["steps"]
    assert t["contains_top1_chain"]
    for x in d["unmatched"]:
        assert x["tau"] <= x["steps"] + 1 and x["draft_tokens"] <= 64


def test_iid_accept_inverts_chain_tau():
    for a in (0.1, 0.5, 0.8, 0.95):
        assert abs(iid_accept(spec_expected_tokens(5, a), 5) - a) < 1e-9


def test_check_brackets_published():
    up, lo = tree_check("indep"), tree_check("chain")
    assert 0.45 < up["err_min"] <= up["err_max"] < 0.7          # 0.224 model over-predicts the published tree τ
    assert -0.25 < lo["err_min"] <= lo["err_max"] < -0.1         # chain lower bound under-predicts
    for ru, rl in zip(up["rows"], lo["rows"]):
        assert rl["model_tau"] < ru["tree_tau"] < ru["model_tau"]   # published sits inside the band
        assert abs(rl["model_tau"] - ru["chain_tau"]) < 1e-6        # lower bound reproduces the chain


def test_check_is_not_a_fit():
    # the evaluation's acceptance does not depend on the references: same a, k, topk, n → same formula as 0.224
    a, k, t, n = 0.7, 3, 4, 16
    assert spec_tree_accept(a, k, t, n) == 1 - (1 - a) ** min(t, (n - 1) / k)
    assert spec_tree_accept(a, k, t, n, "chain") == a


def test_tree_depth_capped_by_draft_tokens():
    # n verified tokens → at most n − 1 deep: k = 5, topk 4, n = 3 accepts ≤ 2 drafts (+ bonus)
    from accel_dse.core.schedule import spec_tree_tau
    assert spec_tree_tau(1.0, 5, 4, 3) == 3.0 and spec_tree_tau(1.0, 5, 4, 0) == 6.0
    r = evaluate(_sc("qwen3.8-27b", 1, 5, 1.0, topk=4, n=3))
    assert r.tokens_per_step <= 3 + 1e-12


# ---- 2. bounds in the evaluation
def test_accept_model_bounds_and_default():
    dflt = evaluate(_sc("qwen3.8-27b", 8, 3, 0.7, topk=4, n=16))             # 0.226: default = chain (was indep)
    up = evaluate(_sc("qwen3.8-27b", 8, 3, 0.7, topk=4, n=16, am="indep"))
    lo = evaluate(_sc("qwen3.8-27b", 8, 3, 0.7, topk=4, n=16, am="chain"))
    ch = evaluate(_sc("qwen3.8-27b", 8, 3, 0.7))
    assert dflt.tpot == lo.tpot and dflt.spec["accept_model"] == "chain" and up.tokens_per_step > lo.tokens_per_step
    assert abs(lo.tokens_per_step - ch.tokens_per_step) < 1e-12
    assert abs(lo.step - up.step) < 1e-15                         # the model changes τ, not the step cost
    assert up.spec["tau_indep"] == up.tokens_per_step and up.spec["tau_chain"] == lo.tokens_per_step
    assert up.spec["check"]["indep"][1] > 0.45 and lo.spec["check"]["chain"][1] < 0
    assert any("EAGLE-1" in w for w in up.warnings)
    assert ch.spec and "tau_indep" not in ch.spec and not any("EAGLE-1" in w for w in ch.warnings)
    c2 = evaluate(_sc("qwen3.8-27b", 8, 3, 0.7, am="indep"))
    assert c2.tpot == ch.tpot and any("spec_tree_accept_model 已忽略" in w for w in c2.warnings)
    assert not any("spec_tree_accept_model 已忽略" in w for w in ch.warnings)


def test_accept_model_validation_and_cli():
    try:
        _sc("qwen3.8-27b", 8, 3, am="iid")
        raise AssertionError("bad accept model accepted")
    except ValueError as e:
        assert "spec_tree_accept_model" in str(e)
    from accel_dse import cli
    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    if p is not None:
        a = p.parse_args(["eval", "--model", "qwen3.8-27b", "--spec-tree-accept-model", "chain"])
        assert cli._body(a)["scenario"]["serving"]["spec_tree_accept_model"] == "chain"


def test_result_dict_spec():
    r = evaluate(_sc("qwen3.8-27b", 8, 3, 0.7, topk=4, n=16))
    d = api.result_dict(r)
    assert d["spec"]["topk"] == 4 and d["spec"]["n"] == 16 and d["spec"]["accept_model"] == "chain"
    assert "spec" not in api.result_dict(evaluate(_sc("qwen3.8-27b", 8)))


# ---- 3. chain-vs-tree what-if
def test_tree_compare_rows_and_crossover():
    c = api.spec_tree_compare(_sc("qwen3.8-27b", 8, 3, 0.7))
    assert c["topk"] == api.SPEC_TREE_TOPK and c["draft_tokens"] == tree_tokens(3, 4, 0)
    assert [x["batch"] for x in c["rows"]] == list(api.SPEC_TREE_BATCHES)
    for x in c["rows"]:
        ch, ti, tc = x["chain"], x["tree_indep"], x["tree_chain"]
        assert tc["tpot_ms"] > ch["tpot_ms"]          # lower bound: same τ, more verify / draft work → never wins
        assert ti["tpot_ms"] < tc["tpot_ms"] and ti["tau"] > tc["tau"] == ch["tau"]
        assert x["tree_wins_chain"] is False
    assert c["crossover_chain"] == []
    # 0.224 conclusion under the upper bound: tree wins at batch 1 / 8 and loses at 32 for qwen3.8-27b
    assert [x["tree_wins_indep"] for x in c["rows"]] == [True, True, False] and c["crossover_indep"] == [[8, 32]]
    t = api.spec_tree_compare(_sc("qwen3.8-27b", 8, 3, 0.7, topk=4, n=16))
    assert t["topk"] == 4 and t["draft_tokens"] == 16


def test_whatif_tree_row():
    s = _sc("qwen3.8-27b", 8, 3, 0.7)
    w = api.balance_whatif(s, evaluate(s))
    row = [x for x in w["rows"] if x["key"] == "spec_tree_tree"]
    assert row and row[0]["spec_tree_compare"]["rows"]
    s2 = _sc("qwen3.8-27b", 8, 3, 0.7, topk=4, n=16)
    w2 = api.balance_whatif(s2, evaluate(s2))
    assert [x for x in w2["rows"] if x["key"] == "spec_tree_chain"]
    s3 = _sc("qwen3.8-27b", 8)                         # no spec → no row
    assert not [x for x in api.balance_whatif(s3, evaluate(s3))["rows"] if x["key"].startswith("spec_tree")]


# ---- 4. fuzz
def test_fuzz_tree_models_invariant():
    from test_core_217 import _table
    t, rnd = _table(), random.Random(225)
    for _ in range(16):
        m = rnd.choice(["qwen3.8-27b", "qwen3-32b", "glm-5.3-flash", "qwen3.8-27b-ready"])
        dm = "qwen3-0.6b" if m == "qwen3-32b" else ""
        k, topk = rnd.randint(1, 5), rnd.choice([1, 2, 4, 8])
        n = rnd.choice([0, min(64, 1 + k * topk), 2]) if topk > 1 else 0
        am, a = rnd.choice(["indep", "chain"]), rnd.choice([0.0, 0.3, 0.7, 1.0])
        b, track, roll = rnd.choice([1, 8, 32]), rnd.choice([0, 256]), rnd.choice(["snapshot", "replay"])
        r = evaluate(_sc(m, b, k, a, dm, topk, n, am, roll, track))
        for v in (r.step, r.tpot, r.throughput):
            assert v == v and 0 < v < float("inf"), (m, k, topk, n, am, a, b)
        assert 1 - 1e-12 <= r.tokens_per_step <= tree_tokens(k, topk, n) + 1e-9
        if r.spec and "tau_chain" in r.spec:
            assert r.spec["tau_chain"] - 1e-12 <= r.tokens_per_step <= r.spec["tau_indep"] + 1e-12
        e = energy_report(r, t)
        if e.get("J_per_unit"):
            assert abs(e["J_per_unit"] * (b / r.tpot / r.scenario.layout.cards) / e["avg_W_per_card"] - 1) < 1e-3
