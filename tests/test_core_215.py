"""0.215: capacity lever — taller HBM stacks at the same bandwidth; KV-bound decode never reaches balance."""
from accel_dse import api, mem_catalog
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario


def _w(**sc):
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", **sc}})["sensitivity"]["balance_whatif"]
    return w["base"], {x["key"]: x for x in w["rows"]}


def test_height_levers_same_bandwidth_more_capacity():
    b, rows = _w()
    for k, gb, status in (("hbm_12h24g", 144, "量产"), ("hbm_16h24g", 192, "已发布")):
        x = rows[k]
        assert f"{gb:g} GB" in x["label"] and status in x["label"]
        assert x["batch_limit"] > b["batch_limit"] and not x["reachable"]
        assert abs(x["chip_flop_per_byte"] - b["chip_flop_per_byte"]) < 1e-9         # same bandwidth
        assert abs(x["area_mm2"] - b["area_mm2"]) < 1e-9 and x["shoreline"]["status"] == b["shoreline"]["status"]


def test_kv_bound_asymptote_independent_of_capacity():
    s = Scenario(chip=CHIPS["1P"], model="qwen3-8b")
    a = [evaluate(s.replace("mem_id", m)).balance()["ai_asymptote"]
         for m in ("hbm3e_4s_8h24g_9200", "hbm3e_4s_12h24g_9200", "hbm3e_4s_16h24g_9200")]
    assert max(a) / min(a) < 1.001 and max(a) < 0.2 * evaluate(s).balance()["chip_flop_per_byte"]


def test_no_height_lever_when_not_capacity_bound():
    b, rows = _w(serving={"tpot_slo_ms": 5.0})
    if b["limited_by"] != "kv_capacity":
        assert not any(k.startswith("hbm_") for k in rows)


def test_catalog_status_tags():
    s = mem_catalog.make_spec("HBM3E", count=4, height=12, die_Gb=24, rate=9200)
    assert s.capacity_GB == 144 and s.hbm_height == 12
