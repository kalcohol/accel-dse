"""0.301: energy-aware design search over compute + memory jointly (DRAM type / count / capacity, SLC), workload mix,
SLO limits, user memory $/GB; non-blocking UI (stale discard)."""
import json

from accel_dse import api, serve
from accel_dse.core.budget import Budget, apply_preset
from accel_dse.core.energy import EnergyTable, apply_energy_preset
from accel_dse.core.energy_dse import MAX_POINTS, MEM_PROXY, energy_design_search, mem_axis
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.mem_catalog import parse_mem_id
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
BASE = "hbm3e_4s_8h24g_9200"


def _t():
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm"}))


def _b(**kw):
    return Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i", **kw}))


def _s(batch=64):
    return Scenario(model=M, chip=CHIPS["1P"], serving=Serving(phase="decode", batch=batch, prompt=4096, ctx=4096))


def _run(**kw):
    kw.setdefault("mac_scales", (1.0,))
    kw.setdefault("sram_scales", (1.0,))
    kw.setdefault("sides", (128,))
    return energy_design_search(_s(kw.pop("batch", 64)), _t(), kw.pop("budget", _b(die_mm2=1200, power_W_card=400)), **kw)


def test_mem_axis():
    assert mem_axis(BASE) == [BASE]
    ids = mem_axis(BASE, mem_types=["HBM3E", "HBM4", "LPDDR5X", "LPDDR6"], mem_count_scales=(0.5, 1, 2))
    assert len(ids) == 12 and BASE in ids and "hbm3e_8s_8h24g_9200" in ids and "lpddr5x_16x64_8533_16g" in ids
    for i in ids:
        assert parse_mem_id(i).id == i
    assert mem_axis(BASE, mem_types=["HBM3E"], mem_count_scales=(100,)) == ["hbm3e_12s_8h24g_9200"]   # snaps to ≤ 12
    caps = mem_axis(BASE, mem_types=["HBM3E", "LPDDR5X"], mem_caps=(36, 32, 7))
    assert caps == ["hbm3e_4s_12h24g_9200", "lpddr5x_8x64_8533_32g"]           # 7 GB exists nowhere: skipped
    assert mem_axis(BASE, mems=["hbm4_8s_12h24g_8000", "hbm4_8s_12h24g_8000"]) == ["hbm4_8s_12h24g_8000"]


def test_memory_axis_area_energy_bandwidth():
    o = _run(mem_types=["HBM3E", "LPDDR5X"], mem_count_scales=(1, 2))
    rows = {x["mem_id"]: x for x in o["rows"]}
    h4, h8, l8 = rows[BASE], rows["hbm3e_8s_8h24g_9200"], rows["lpddr5x_8x64_8533_16g"]
    b = _b()
    # die area moves by exactly the DRAM PHY (bus bits × mm2_per_dq_*, sourced densities) — uncore share scales with it
    assert h8["mm2"] > h4["mm2"] and l8["mm2"] < h4["mm2"]
    assert h8["mem_bus_bits"] == 2 * h4["mem_bus_bits"] and abs(h8["mem_GBps"] / h4["mem_GBps"] - 2) < 1e-12
    phy = lambda x: x["mem_bus_bits"] * (b.mm2_per_dq_hbm if x["mem_kind"] == "HBM" else b.mm2_per_dq_lpddr)
    assert abs((h8["mm2"] - h4["mm2"]) - (phy(h8) - phy(h4)) / (1 - b.uncore_frac)) < 1e-6
    # decode b64 is DRAM-bound: 2× HBM stacks → more tok/s; LPDDR5X 8×x64 → far less (bandwidth), TPOT up
    assert h8["thr"] > 1.1 * h4["thr"] and l8["thr"] < 0.3 * h4["thr"] and l8["tpot_ms"] > 3 * h4["tpot_ms"]
    # 0.302: the preset now carries LPDDR background (Micron IDD2N + refresh) — charged, the "not modelled" note gone
    assert h8["hbm_static_W"] > h4["hbm_static_W"] > 0 and l8["hbm_static_W"] > 0
    assert not any("LPDDR 后台" in n for n in o["notes"]) and any("PHY 面积" in n for n in o["notes"])
    assert o["rows"][o["base"]]["mem_id"] == BASE and len(o["mems"]) == 4


def test_slc_axis_and_proxy_labels():
    o = _run(mem_types=["HBM4", "LPDDR6"], slc_mibs=(0, 128))
    assert any(MEM_PROXY["HBM4"] == n for n in o["notes"]) and any(MEM_PROXY["LPDDR6"] == n for n in o["notes"])
    by = {(x["mem_id"], x["slc_mib"]): x for x in o["rows"]}
    a, c = by[("hbm4_8s_12h24g_8000", 0.0)], by[("hbm4_8s_12h24g_8000", 128.0)]
    assert abs(c["mm2"] - a["mm2"] - 128 * _b().mm2_per_mib_slc / (1 - _b().uncore_frac)) < 1e-6


def test_slo_and_shoreline_and_cap():
    o = _run(mem_types=["HBM3E", "LPDDR5X"], mem_count_scales=(1, 3), slo_tpot_ms=50)
    for x in o["rows"]:
        if x.get("tpot_ms") and x["tpot_ms"] > 50:
            assert not x["feasible"] and "TPOT" in x["why"]
    sh = _run(mems=["hbm4_12s_12h24g_8000"], budget=_b(), mac_scales=(0.5,))
    assert "岸线" in sh["rows"][0]["why"] and sh["rows"][0]["shoreline"] == "over"          # 12 stacks × 11 mm > 4 sides of an ≈ 1000 mm² square die
    try:
        _run(mem_types=["HBM3E", "HBM4", "LPDDR5X", "LPDDR6"], mem_count_scales=(0.5, 1, 2), slc_mibs=(0, 64, 128),
             mac_scales=(0.5, 1, 2), sides=None, sram_scales=(1.0,))
    except ValueError as e:
        assert str(MAX_POINTS) in str(e)
    else:
        raise AssertionError("grid cap")


def test_mix_objective():
    o = _run(mix_out_tokens=256, slo_ttft_ms=1e9)
    x = o["rows"][0]
    pf, dc = x["phases"]["prefill"], x["phases"]["decode"]
    assert o["phase"] == "mix" and x["unit"] == "请求" and o["mix"] == {"prompt": 4096, "out_tokens": 256}
    assert abs(1 / x["thr"] - (4096 / pf["thr"] + 256 / dc["thr"])) < 1e-9 * (1 / x["thr"])
    assert abs(x["J_per_unit"] - (4096 * pf["J_per_unit"] + 256 * dc["J_per_unit"])) < 1e-9 * x["J_per_unit"]
    assert abs(sum(x["share"].values()) - 1) < 1e-9 and x["ttft_ms"] == pf["lat_ms"] and x["tpot_ms"] == dc["lat_ms"]


def test_mem_usd_user_only():
    o = _run(mem_types=["HBM3E", "LPDDR5X"], mem_usd_per_GB={"HBM3E": 10, "LPDDR5X": 3}, max_mem_usd=900)
    by = {x["mem_id"]: x for x in o["rows"]}
    assert by[BASE]["mem_usd"] == 10 * by[BASE]["mem_GB"] and not by[BASE]["feasible"] and "内存 $" in by[BASE]["why"]
    assert by["lpddr5x_8x64_8533_16g"]["mem_usd"] == 3 * 128
    assert o["priced"] and any(x.get("front_usd") for x in o["rows"])
    assert "mem_usd" not in _run()["rows"][0]                        # nothing priced without user input
    for kw in ({"max_mem_usd": 10}, {"mem_usd_per_GB": {"DDR9": 1}}, {"mix_out_tokens": 0}, {"slo_tpot_ms": -1}):
        try:
            _run(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(kw)


def test_api_and_ui_nonblocking():
    body = {"chip_preset": "1P", "scenario": {"model": M, "serving": {"batch": 64}}, "energy": {"energy_preset": "tpuv4i_7nm"},
            "budget": {"density_preset": "n7_tpuv4i"}, "mac_scales": [1], "sram_scales": [1], "sides": [128],
            "mem_types": ["HBM3E", "LPDDR5X"], "mem_count_scales": [1], "slc_mibs": [0, 64], "mix_out_tokens": 128,
            "slo_tpot_ms": 100, "mem_usd_per_GB": {"HBM3E": 10, "LPDDR5X": 3}}
    st, p = serve.respond("POST", "/api/insight_dse", json.dumps(body).encode())
    out = json.loads(p)
    assert st == 200 and len(out["rows"]) == 4 and out["phase"] == "mix" and out["priced"]
    for bad in ({"mem_types": "HBM3E"}, {"mem_types": ["DDR9"]}, {"slc_mibs": [-1]}, {"mix_out_tokens": 1.5},
                {"mem_usd_per_GB": [1]}, {"mems": ["nope"]}):
        assert serve.respond("POST", "/api/insight_dse", json.dumps({**body, **bad}).encode())[0] == 400, bad
    html, js = W.read(W.INDEX_HTML), W.read()
    assert {"ins-dse-mem", "ins-dse-memn", "ins-dse-memcap", "ins-dse-slc", "ins-dse-out", "ins-dse-tpot", "ins-dse-ttft",
            "ins-dse-usd", "ins-dse-maxusd"} <= W.html_ids(html)
    src = W.function_source(js, "runInsightDse")
    for k in ("mem_types", "slc_mibs", "mix_out_tokens", "slo_tpot_ms", "mem_usd_per_GB", "mem_id"):
        assert k in src, k
    inv = W.function_source(js, "invalidate")           # a scenario edit drops the in-flight search (stale discard)
    assert "cancel" in inv and "insight_dse" in inv
    assert "/api/insight_dse" in W.list_values(W.read(W.SHIM_JS), "slowRoutes")   # static build: worker, progress pill
