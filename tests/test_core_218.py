"""0.218: Samsung HBM4 24 GB entry; catalog-wide provenance audit; what-if rows state their priced batch and never
exceed their own batch limit."""
from accel_dse import api, mem_catalog
from accel_dse.core.d2d_catalog import D2D_STANDARDS
from accel_dse.core.energy import ENERGY_PRESETS
from accel_dse.core.budget import DENSITY_PRESETS
from accel_dse.core.hardware import CHIPS, CHIP_SOURCES, LINK_SOURCES
from accel_dse.core.provenance import valid_tag
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)


def _tag(v):
    return v.split(":", 1)[0].strip()


def test_samsung_hbm4_24gb():
    s = mem_catalog.make_spec("HBM4", count=8, height=12, die_Gb=16, rate=11700)
    assert s.capacity_GB == 24 * 8 and s.id == "hbm4_8s_12h16g_11700"
    assert mem_catalog.HBM_CATALOG["HBM4"]["cap_tags"][(12, 16)] == ("custom", "shipping")
    src = mem_catalog.HBM_CAP_SOURCES[("HBM4", 12, 16)]
    assert _tag(src) == "推导" and "Samsung" in src and "24 GB" in src


def test_hbm_every_claim_sourced():
    for g, c in mem_catalog.HBM_CATALOG.items():
        for (h, d), (spec, prod) in c["cap_tags"].items():
            if prod != "none" or (g, h, d) in mem_catalog.HBM_CAP_SOURCES:
                assert valid_tag(_tag(mem_catalog.HBM_CAP_SOURCES[(g, h, d)])), (g, h, d)
        for rate in c["rates"]:
            assert valid_tag(_tag(mem_catalog.HBM_RATE_SOURCES[(g, rate)])), (g, rate)
    for k, v in mem_catalog.HBM_CAP_SOURCES.items():
        assert (k[1], k[2]) in mem_catalog.HBM_CATALOG[k[0]]["cap_tags"], k


def test_catalog_wide_provenance():
    for t, c in mem_catalog.LPDDR_CATALOG.items():
        assert valid_tag(_tag(c["src"])) and len(c["src"]) > 20, t      # 0.219: per entry, see test_core_219
    for k, d in D2D_STANDARDS.items():
        assert valid_tag(d["src"]) and d.get("note"), k
    for n in CHIPS:
        assert valid_tag(CHIP_SOURCES[n][0]) and CHIP_SOURCES[n][1], n
    for k, (t, note) in LINK_SOURCES.items():
        assert valid_tag(t) and note, k
    for P in (ENERGY_PRESETS, DENSITY_PRESETS):
        for name, p in P.items():
            for f, (v, tag, note) in p["fields"].items():
                assert valid_tag(tag) and note, (name, f)


def test_whatif_priced_batch_within_limit():
    for m in ("qwen3-8b", "rwkv7-g1k-7.2b", "llama-3.2-1b"):
        for fo in ([], [["kv", "fp8"]]):
            w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": m, "formats_override": fo}})
            w = w["sensitivity"]["balance_whatif"]
            assert w["base"]["cost_basis"] == "configured" and w["base"]["cost_batch"] >= 1
            for x in w["rows"]:
                if x.get("cost_batch") is None:
                    continue
                assert x["cost_basis"] in ("critical_batch", "batch_limit", "configured"), x["key"]
                if x.get("batch_limit"):
                    assert x["cost_batch"] <= x["batch_limit"], (m, x["key"])
                    assert x["base_cost_batch"] <= max(w["base"]["batch_limit"], x["cost_batch"])


def test_ui_states_priced_batch():
    from pathlib import Path
    js = W.read()                                      # 0.286: parsed
    assert W.any_string_contains(js, "计价 batch") and W.has(js, "x.cost_basis")
