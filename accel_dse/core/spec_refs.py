"""0.225: published tree-drafting acceptance reference points and the check of the tree-acceptance 「假设」 model
against them (data/spec_tree_refs.json).  Reports only — nothing here is fitted or feeds the evaluation."""
from __future__ import annotations

import functools
import json
from pathlib import Path

from .schedule import spec_expected_tokens, spec_tree_tau

_PATH = Path(__file__).resolve().parents[1] / "data" / "spec_tree_refs.json"


@functools.lru_cache(maxsize=1)
def tree_refs() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))


def iid_accept(tau: float, k: int) -> float:
    """Per-draft i.i.d. acceptance a whose chain of k drafts gives τ = (1 − a^(k+1)) / (1 − a)."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if spec_expected_tokens(k, mid) < tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@functools.lru_cache(maxsize=4)
def tree_check(model: str = "indep") -> dict:
    """Each matched chain / tree pair: a from the chain τ at the tree's depth, the model's tree τ at the published tree
    shape (steps, topk, draft tokens) vs the published tree τ.  err = model / published − 1."""
    d = tree_refs()
    t = d["matched_tree"]
    k, topk, n = t["steps"], t["topk"], t["draft_tokens"]
    rows = []
    for x in d["matched"]:
        a = iid_accept(x["chain_tau"], k)
        tau = spec_tree_tau(a, k, topk, n, model)
        rows.append({**x, "a_iid": a, "model_tau": tau, "err": tau / x["tree_tau"] - 1.0})
    errs = [r["err"] for r in rows]
    return {"model": model, "tree": t, "rows": rows, "err_min": min(errs), "err_max": max(errs),
            "gain_published": [r["tree_tau"] / r["chain_tau"] - 1 for r in rows]}
