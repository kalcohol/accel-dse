"""Memory catalog: geometry hand checks, provenance tags, legacy id resolution."""
from __future__ import annotations

from accel_dse.mem_catalog import TAG_ORDER, catalog_dict, make_spec, parse_mem_id, weakest


def test_lpddr6_4x96_10667_handcheck():
    """LPDDR6 4 × x96 @10667: bus 384 b, raw 512.0 GB/s, payload 256/288, ×0.7 efficiency."""
    s = make_spec("LPDDR6", count=4, rate=10667)
    assert s.unit_width_bits == 96 and s.bus_bits == 384 and s.form == "discrete"
    raw = 384 * 10667 / 8 / 1000
    assert abs(s.raw_GBps - raw) < 1e-9
    assert abs(s.payload_GBps - raw * 256 / 288) < 1e-9
    assert abs(s.effective_GBps - s.payload_GBps * 0.70) < 1e-9
    assert s.capacity_GB == 64 and s.capacity_bytes == 64 * 2**30


def test_hbm3e_and_lpddr5x_geometry():
    h = make_spec("HBM3E", count=8, height=12, die_Gb=24, rate=9200)
    assert h.capacity_GB == 288 and abs(h.raw_GBps - 8 * 1024 * 9200 / 8000) < 1e-9
    assert h.payload_factor == 1.0 and h.tag == "vendor_shipping"
    assert h.id == "hbm3e_8s_12h24g_9200"
    lp = make_spec("LPDDR5X", count=8, width_bits=64, rate=8533, cap_GB=16)
    assert lp.bus_bits == 512 and abs(lp.raw_GBps - 546.112) < 1e-9 and lp.capacity_GB == 128
    assert make_spec("HBM4", count=1).unit_width_bits == 2048  # HBM4 doubles the interface


def test_provenance_tags_weakest_wins():
    assert TAG_ORDER[0] == "jedec" and TAG_ORDER[-1] == "speculative"
    assert weakest(["jedec", "vendor_sampling", "vendor_shipping"]) == "vendor_sampling"
    assert make_spec("LPDDR5X", width_bits=96, count=4).tag == "vendor_announced"
    assert make_spec("LPDDR6", width_bits=48, count=2).tag == "speculative"
    assert make_spec("HBM4E", count=4).tag == "vendor_sampling"
    s16 = make_spec("HBM3E", count=16)
    assert s16.tag == "speculative" and s16.warnings()


def test_ids_round_trip_and_legacy_ids_resolve():
    for t in catalog_dict()["types"]:
        s = make_spec(t["id"])
        assert parse_mem_id(s.id) == s, s.id
    s = parse_mem_id("lpddr_4x64_8533")
    assert s.id == "lpddr5x_4x64_8533_16g" and s.legacy_id == "lpddr_4x64_8533"
    h = parse_mem_id("hbm_hbm3e_4s")
    assert (h.mem_type, h.n_units, h.hbm_height, h.hbm_die_Gb) == ("HBM3E", 4, 12, 24)


def test_legacy_lpddr6_bus_width_preserved():
    """lpddr6_{n}x24 → x96 packages when n·24 % 96 == 0, else x48; bus width (→ bandwidth) preserved."""
    for n in (4, 6, 8, 12, 16, 20):
        for r in (10667, 14400):
            s = parse_mem_id(f"lpddr6_{n}x24_{r}")
            assert s.n_units * s.unit_width_bits == n * 24, (n, s.id)
            assert s.unit_width_bits == (96 if n * 24 % 96 == 0 else 48) and s.rate_MTps == r


def test_catalog_options_all_construct():
    """Every (type, form, width, rate, count) the UI can offer builds a spec with the expected bandwidth."""
    for t in catalog_dict()["types"]:
        for f in t["forms"]:
            for w in f["widths"]:
                for r in f["rates"]:
                    s = make_spec(t["id"], form=f["id"], width_bits=w["bits"], rate=r["MTps"], count=f["default_count"])
                    assert abs(s.raw_GBps - s.bus_bits * r["MTps"] / 8000) < 1e-6
