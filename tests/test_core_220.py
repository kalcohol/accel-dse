"""0.220: LPDDR sourcing complete; SOCAMM2 192 GB back to shipping; adversarial what-if monotonicity fuzzer."""
import random

from accel_dse import api, mem_catalog as m
from accel_dse.core.provenance import valid_tag


def _tag(v):
    return v.split(":", 1)[0].strip()


def _entries():
    for t, cat in m.LPDDR_CATALOG.items():
        for fn, f in cat["forms"].items():
            for r in f["rates"]:
                yield t, fn, ("rate", r)
            for w in f["widths"]:
                yield t, fn, ("width", w)
            for w, cs in f["caps"].items():
                for g in cs:
                    yield t, fn, ("cap", w, g)


def test_no_research_doc_citation_left():
    left = [(t, fn, k) for t, fn, k in _entries() if m.lpddr_source(t, fn, *k) == m._LP_R]
    assert not left, left


def test_assumed_entries_claim_no_product():
    """An entry whose only citation is 假设 must not claim a shipping / sampling product."""
    for t, fn, k in _entries():
        v = m.lpddr_source(t, fn, *k)
        if _tag(v) != "假设" or k[0] == "rate":
            continue
        f = m.LPDDR_CATALOG[t]["forms"][fn]
        st = f["widths"][k[1]] if k[0] == "width" else f["caps"][k[1]][k[2]]
        assert st[1] in ("none", "announced"), (t, fn, k, st)


def test_corrections_0220():
    so = m.LPDDR_CATALOG["LPDDR5X"]["forms"]["SOCAMM2"]["caps"][128]
    assert so[192] == ("jedec", "shipping") and all(so[g] == ("jedec", "none") for g in (48, 64, 96, 128))
    assert "2026-04-20" in m.LPDDR_SOURCES[("LPDDR5X", "SOCAMM2", "cap", 128, 192)]
    l5 = m.LPDDR_CATALOG["LPDDR5"]["forms"]["discrete"]["caps"]
    assert l5[32][3] == l5[32][6] == l5[64][3] == ("jedec", "none")
    assert "2026-09-07" in m.LPDDR_SOURCES[("LPDDR6", "discrete", "rate", 12800)]
    assert "TrendForce" not in m.LPDDR_SOURCES[("LPDDR6", "discrete", "rate", 12800)]


# --- adversarial fuzzer: what-if lever monotonicity (physics, not calibration) -----------------------------------
# A lever that only adds a resource must never make the design worse on the axis it adds:
#   kv_fp8 / ctx_half (fewer KV bytes per sequence) → batch limit ≥ base, asymptote ≥ base;
#   hbm_* (more capacity, same bandwidth) → batch limit ≥ base, chip FLOP/B unchanged;
#   dram_bw_x2 (twice the bandwidth, same capacity) → TPOT at the configured batch ≤ base;
#   sram_x2 → batch limit ≥ base (SRAM never removes capacity).
MODELS = ["qwen3-8b", "llama-3.2-1b", "llama-3.1-8b", "rwkv7-g1k-7.2b", "qwen3-30b-a3b"]


def test_whatif_monotonicity_fuzz():
    rnd, n = random.Random(220), 0
    for _ in range(10):
        m_, chip = rnd.choice(MODELS), rnd.choice(["1P", "H100-like", "100T"])
        sv = {"batch": rnd.choice([1, 8, 64]), "ctx": rnd.choice([512, 2048, 8192])}
        fo = rnd.choice([[], [["kv", "fp8"]]])
        try:
            w = api.api_sensitivity({"chip_preset": chip, "scenario": {"model": m_, "serving": sv, "formats_override": fo}})
        except api.ApiError:
            continue
        w = w["sensitivity"].get("balance_whatif")
        if not w:
            continue
        b, eps = w["base"], 1e-9
        for x in w["rows"]:
            k = x["key"]
            ctx = (m_, chip, sv, fo, k)
            if k in ("kv_fp8", "ctx_half") or k.startswith("hbm_") or k == "sram_x2":
                assert x["batch_limit"] >= b["batch_limit"], ctx
            if k in ("kv_fp8", "ctx_half"):
                assert x["ai_asymptote"] >= b["ai_asymptote"] * (1 - 1e-6), ctx
            if k.startswith("hbm_"):
                assert abs(x["chip_flop_per_byte"] - b["chip_flop_per_byte"]) < eps * max(1, b["chip_flop_per_byte"]), ctx
            if k == "dram_bw_x2":
                assert x["tpot_ms"] <= b["tpot_ms"] * (1 + 1e-6), ctx
            n += 1
    assert n >= 15, n
