"""0.219: HBM 二手 entries upgraded to primary sources; LPDDR per-entry sources; corrections."""
from accel_dse import mem_catalog as m
from accel_dse.core.provenance import valid_tag


def _tag(v):
    return v.split(":", 1)[0].strip()


def test_hbm_upgraded_to_primary():
    assert _tag(m.HBM_RATE_SOURCES[("HBM3E", 9800)]) == "来源" and "2023-10-20" in m.HBM_RATE_SOURCES[("HBM3E", 9800)]
    assert m.HBM_CATALOG["HBM3E"]["rates"][9800] == ("custom", "sampling")
    for k in (("HBM3", 8, 16), ("HBM3", 12, 16)):
        assert _tag(m.HBM_CAP_SOURCES[k]) == "来源", k
    assert "2023-04-20" in m.HBM_CAP_SOURCES[("HBM3", 12, 16)]
    assert not [k for k, v in m.HBM_CAP_SOURCES.items() if _tag(v) == "二手"]


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


def test_lpddr_every_entry_sourced():
    n = cited = doc = 0
    for t, fn, k in _entries():
        v = m.lpddr_source(t, fn, *k)
        assert valid_tag(_tag(v)) and len(v) > 15, (t, fn, k)
        n += 1
        cited += (t, fn) + k in m.LPDDR_SOURCES
        doc += v == m._LP_R
    # 0.219: 28 cited, 14 no-product configs 假设, 24 still the research-doc 二手 (deferred)
    assert n == 66 and cited >= 28 and doc <= 24, (n, cited, doc)
    for k in m.LPDDR_SOURCES:      # every citation names a real catalog entry
        f = m.LPDDR_CATALOG[k[0]]["forms"][k[1]]
        assert (k[3] in f["rates"] if k[2] == "rate" else k[3] in f["widths"] if k[2] == "width"
                else k[4] in f["caps"][k[3]]), k


def test_lpddr_corrections():
    assert m.LPDDR_CATALOG["LPDDR5X"]["forms"]["SOCAMM2"]["caps"][128][192] == ("jedec", "sampling")
    for k in (("LPDDR5", "discrete", "rate", 6400), ("LPDDR5X", "discrete", "rate", 8533),
              ("LPDDR6", "discrete", "rate", 10667), ("LPDDR6", "discrete", "rate", 14400)):
        assert _tag(m.LPDDR_SOURCES[k]) == "来源" and "JEDEC" in m.LPDDR_SOURCES[k], k
