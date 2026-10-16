"""0.128: colocated worst-gap lattice (the load-0.4 itl_max 'bias' is the p99 landing between busy-period atoms),
catalog label consistency, sensitivity block / sweep row / export schema stability."""
import math
import re
from pathlib import Path

from accel_dse import api
from accel_dse.core import catalog as C
from accel_dse.core.pdqueue import _busy_q_from, _mg1_busy

ROOT = Path(__file__).resolve().parents[1]


def test_busy_duration_lattice_borel():
    """One prompt length: busy durations are n·lat with Borel(λ·lat) masses, and the max-of-m quantile is an atom."""
    lam, lat = 1.06, 0.0681
    _, dur = _mg1_busy(lam, [(1.0, lat)], want_dur=True)
    a = lam * lat
    for n, (t, p) in enumerate(dur[:6], start=1):
        assert abs(t - n * lat) < 1e-12
        assert abs(p - math.exp(-a * n) * (a * n) ** (n - 1) / math.factorial(n)) < 1e-8
    atoms = {t for t, _ in dur}
    for m in (0.5, 1.39, 2.8, 10.0):
        d = _busy_q_from(dur, m, 0.99)
        assert d in atoms
        F = sum(p for t, p in dur if t <= d + 1e-12)
        assert F ** m >= 0.99 - 1e-9
        assert sum(p for t, p in dur if t < d - 1e-12) ** m < 0.99      # smallest such atom


def test_catalog_labels_consistent():
    for e in C.entries():
        s = C.get_model(e["id"])
        lab = C.labels(s)
        assert lab["coverage"] in ("full", "partial", "proxy")
        assert bool(lab["coverage_reasons"]) == (lab["coverage"] != "full"), e["id"]
        assert re.match(r"^W \S+(/专家 \S+)? · A \S+( · KV \S+)?$", lab["dtype"]), (e["id"], lab["dtype"])
        assert (" · KV " in lab["dtype"]) == bool(s.kv_cache)
        f = dict(s.formats)
        if "expert" in f and "attn" in f:
            assert (f"专家 {f['expert'].fmt}" in lab["dtype"]) == (f["expert"].fmt != f["attn"].fmt)
    for d in C.offline_entries():
        assert d["source"].startswith("https://"), d["id"]


SWEEP_ROW_KEYS = {"value", "tpot_ms", "ttft_ms", "latency_ms", "tok_s", "tok_s_card", "bound", "fits", "array_util"}
SENS_KEYS = {"base_tok_s", "base_bound", "base_label", "threshold", "rows", "hinges", "domain", "scope", "balance_whatif"}   # 0.208: + balance_whatif
SENS_ROW_KEYS = {"key", "label", "tok_s", "ratio", "bound", "fits", "bound_flip", "hinge"}


def test_export_schema_stable():
    out = api.api_sweep({"path": "serving.batch", "values": [1, 8]})
    assert {"path", "rows", "base_hash"} <= set(out)
    for r in out["rows"]:
        assert set(r) == SWEEP_ROW_KEYS
    sens = api.api_sensitivity({})["sensitivity"]
    assert set(sens) == SENS_KEYS
    for r in sens["rows"]:
        assert set(r) == SENS_ROW_KEYS or "error" in r
        if "ratio" in r and r["ratio"] is not None:
            assert abs(r["ratio"] - r["tok_s"] / sens["base_tok_s"]) < 1e-12     # ratio = alternative / base (no sign flip)
            assert r["hinge"] == (abs(r["ratio"] - 1) > sens["threshold"] or r["bound_flip"])
    js = (ROOT / "accel_dse/web/app.js").read_text()
    assert re.search(r"const EXP_SCHEMA = 1;", js) and "schema: EXP_SCHEMA" in js
    # sweep sensitivity skips only cases tied to the swept knob
    assert set(api.SENS_SWEEP_SKIP.values()) <= {k for k, _, _ in api.SENS_CASES}
