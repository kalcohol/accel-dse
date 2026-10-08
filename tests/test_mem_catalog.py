"""Memory catalog: geometry, dual provenance, LPDDR5T alias, meta mode, legacy ids."""
from __future__ import annotations

from accel_dse.mem_catalog import (
    LPDDR6_META_CARVEOUT,
    LPDDR6_PAYLOAD,
    PRODUCT_ORDER,
    SPEC_ORDER,
    TAG_ORDER,
    catalog_dict,
    make_spec,
    parse_mem_id,
    weakest,
)


def test_lpddr6_4x96_10667_handcheck():
    """LPDDR6 4 × x96 @10667: bus 384 b, raw 512.0 GB/s, payload 256/288, ×0.7 efficiency."""
    s = make_spec("LPDDR6", count=4, rate=10667)
    assert s.unit_width_bits == 96 and s.bus_bits == 384 and s.form == "discrete"
    raw = 384 * 10667 / 8 / 1000
    assert abs(s.raw_GBps - raw) < 1e-9
    assert abs(s.payload_GBps - raw * LPDDR6_PAYLOAD) < 1e-9
    assert abs(s.effective_GBps - s.payload_GBps * 0.70) < 1e-9
    assert s.capacity_GB == 64 and s.capacity_bytes == 64 * 2**30
    assert abs(s.payload_GBps - 455.1253333333333) < 1e-6  # Samsung/JEDEC ~114 GB/s × 4


def test_hbm3e_and_lpddr5x_geometry():
    h = make_spec("HBM3E", count=8, height=12, die_Gb=24, rate=9200)
    assert h.capacity_GB == 288 and abs(h.raw_GBps - 8 * 1024 * 9200 / 8000) < 1e-9
    assert h.payload_factor == 1.0 and h.tag == "vendor_shipping"
    assert h.spec_status == "custom" and h.product_status == "shipping"
    assert h.id == "hbm3e_8s_12h24g_9200"
    lp = make_spec("LPDDR5X", count=8, width_bits=64, rate=8533, cap_GB=16)
    assert lp.bus_bits == 512 and abs(lp.raw_GBps - 546.112) < 1e-9 and lp.capacity_GB == 128
    assert make_spec("HBM4", count=1).unit_width_bits == 2048


def test_provenance_dual_axes():
    assert TAG_ORDER[0] == "jedec" and TAG_ORDER[-1] == "speculative"
    assert SPEC_ORDER[0] == "jedec" and PRODUCT_ORDER[0] == "shipping"
    assert weakest(["jedec", "vendor_sampling", "vendor_shipping"]) == "vendor_sampling"
    # LPDDR5X x96 removed (Apple-custom); request falls back to x64 packages
    s96 = make_spec("LPDDR5X", width_bits=96, count=4)
    assert s96.unit_width_bits == 64 and "Apple" in (s96.legacy_note or "")
    assert make_spec("LPDDR6", width_bits=48, count=2).tag == "speculative"
    assert make_spec("HBM4E", count=4).tag == "vendor_sampling"
    # HBM4 12H × 32 Gb = JEDEC-allowed only (samples are 16H × 24 Gb)
    h = make_spec("HBM4", height=12, die_Gb=32)
    assert h.spec_status == "jedec" and h.product_status == "none" and h.tag == "jedec"
    s16 = make_spec("HBM3E", count=16)
    assert s16.product_status == "none" and s16.warnings()
    # SOCAMM2 256 GB / LPDDR5X x32 64 GB = sampling; LPCAMM2 9600 = shipping
    assert make_spec("LPDDR5X", form="SOCAMM2", cap_GB=256).product_status == "sampling"
    assert make_spec("LPDDR5X", width_bits=32, cap_GB=64).product_status == "sampling"
    assert make_spec("LPDDR5X", form="LPCAMM2", rate=9600).product_status == "shipping"
    # LPDDR6 8 / 12 GB unified as JEDEC-density · 无产品
    for g in (8, 12):
        c = make_spec("LPDDR6", cap_GB=g)
        assert c.product_status == "none" and c.spec_status == "jedec_likely"


def test_lpddr5t_alias_and_meta_mode():
    s = make_spec("LPDDR5T")
    assert s.mem_type == "LPDDR5X" and s.rate_MTps == 9600 and "LPDDR5T" in (s.legacy_note or "")
    assert parse_mem_id("lpddr5t_4x64_9600_16g").mem_type == "LPDDR5X"
    m = make_spec("LPDDR6", count=4, meta_mode=True)
    assert m.meta_mode and abs(m.capacity_GB - 64 * (1 - LPDDR6_META_CARVEOUT)) < 1e-9
    assert m.payload_factor == LPDDR6_PAYLOAD  # 8/9 independent of meta mode
    assert parse_mem_id(m.id) == m and m.id.endswith("_meta")
    try:
        make_spec("LPDDR5X", meta_mode=True)
    except ValueError as e:
        assert "meta_mode" in str(e)
    else:
        raise AssertionError("meta_mode on LPDDR5X must be rejected")


def test_ids_round_trip_and_legacy_ids_resolve():
    for t in catalog_dict()["types"]:
        s = make_spec(t["id"])
        assert parse_mem_id(s.id) == s, s.id
    s = parse_mem_id("lpddr_4x64_8533")
    assert s.id == "lpddr5x_4x64_8533_16g" and s.legacy_id == "lpddr_4x64_8533"
    h = parse_mem_id("hbm_hbm3e_4s")
    assert (h.mem_type, h.n_units, h.hbm_height, h.hbm_die_Gb) == ("HBM3E", 4, 12, 24)


def test_legacy_lpddr6_bus_width_preserved():
    """lpddr6_{n}x24 → x96 packages when n·24 % 96 == 0, else x48; bus width preserved."""
    for n in (4, 6, 8, 12, 16, 20):
        for r in (10667, 14400):
            s = parse_mem_id(f"lpddr6_{n}x24_{r}")
            assert s.n_units * s.unit_width_bits == n * 24, (n, s.id)
            assert s.unit_width_bits == (96 if n * 24 % 96 == 0 else 48) and s.rate_MTps == r


def test_catalog_options_all_construct():
    """Every (type, form, width, rate, count) the UI can offer builds a spec with expected BW."""
    cat = catalog_dict()
    assert "LPDDR5X x96" in cat["not_offered"][0]
    assert "spec_order" in cat and "product_order" in cat
    lp5x = next(t for t in cat["types"] if t["id"] == "LPDDR5X")
    assert [w["bits"] for w in lp5x["forms"][0]["widths"]] == [32, 64]
    lp6 = next(t for t in cat["types"] if t["id"] == "LPDDR6")
    assert [r["MTps"] for r in lp6["forms"][0]["rates"]] == [10667, 11733, 12800, 14400]
    assert lp6["meta_mode"]["default"] is False
    for t in cat["types"]:
        for f in t["forms"]:
            for w in f["widths"]:
                for r in f["rates"]:
                    s = make_spec(t["id"], form=f["id"], width_bits=w["bits"], rate=r["MTps"],
                                  count=f["default_count"])
                    assert abs(s.raw_GBps - s.bus_bits * r["MTps"] / 8000) < 1e-6
                    assert "spec" in w and "product" in w and "status_zh" in w
