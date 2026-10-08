"""V2 trend sanity (caveated) and V3 cross-tool agreement with GenZ."""
from __future__ import annotations

from accel_dse.core.validation import v2_trends, v3_genz


def test_v2_h100_like_trend_bands():
    bad = [r for r in v2_trends() if not r["ok"]]
    assert not bad, bad


def test_v3_genz_memory_bound_agreement():
    rows = v3_genz()
    for r in rows:
        if not r["comparable"]:
            continue
        # Memory-bound points agree within 15 % (we sit ~7 % above GenZ: we also stream the LM head).
        # Attention-heavy HBM points (long ctx / large batch) diverge by design: decode attention is
        # M = GQA-group GEMMs per sequence, which we tile on the array / GEMV unit while GenZ assumes
        # ideal FLOPS.  Documented in docs/MODEL.md §validation.
        memory_bound = r["mem_GBps"] < 1000 or (r["batch"] <= 8 and r["ctx"] <= 1024)
        tol = 0.15 if memory_bound else 0.60
        assert 0.9 <= r["ratio"] <= 1 + tol, r


def test_v3_os_mapping_slower_than_ideal_on_hbm():
    # the documented divergence: GenZ has no feed/SRAM-port bound; an OS-only array cannot reach HBM speed
    rows = [r for r in v3_genz() if r["mem_GBps"] > 1000 and r["mapping"] == "os" and r["batch"] == 1]
    assert rows and all(r["ratio"] > 2.0 for r in rows)
