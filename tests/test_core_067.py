"""0.67 (V6.2): MoE / MLA / large-EP published measurements, H800 reference hardware, exec_overlap="tbo"."""
import math
from dataclasses import replace

from accel_dse.core import extval as X
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import EXEC_OVERLAPS, Layout
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.hardware import CHIPS


def test_h800_is_h100_die_with_capped_nvlink():
    a, b = REF_HW["H100-SXM"], REF_HW["H800-SXM"]
    assert b.chip.peak_tflops_bf16 == a.chip.peak_tflops_bf16 and b.hbm_GBps == a.hbm_GBps
    assert b.link.GBps * 2 == b.link_GBps_bidir == 400.0 < a.link_GBps_bidir
    assert all("参考硬件" not in getattr(c, "name", "") for c in CHIPS.values())


def test_moe_rows_cited_and_resolvable():
    R = X.rows(X.DATA_MOE)
    assert len(R) >= 7 and len({r["id"] for r in R}) == len(R)
    for r in R:
        assert r["url"].startswith("https://") and r["hw"] in REF_HW and r["value"] > 0
        if r["metric"] in ("prefill_tok_s_node", "decode_tok_s_node"):
            assert r["dp"] * (r["tp"]) == r["nodes"] * r["node_cards"] and r["ep"] == r["dp"]


def test_tbo_between_stage_and_kernel_and_hides_collectives():
    h = REF_HW["H100-SXM"]
    from accel_dse.core.hardware import Link
    base = Scenario(model="deepseek-v3", chip=h.chip, mem_id=h.mem_id, link=h.link, mapping="reconf",
                    layout=Layout(dp=16, ep=16), node_cards=8, net=Link(50.0, 5.0))
    assert "tbo" in EXEC_OVERLAPS
    for sv in (dict(phase="decode", batch=16 * 64, ctx=2048), dict(phase="prefill", batch=16 * 4, prompt=4096)):
        s0 = replace(base, serving=Serving(**sv))
        r = {m: evaluate(replace(s0, exec_overlap=m)) for m in ("stage", "tbo", "kernel", "serial")}
        key = "tpot" if sv["phase"] == "decode" else "ttft"
        v = {m: getattr(x, key) for m, x in r.items()}
        assert v["stage"] <= v["tbo"] * (1 + 1e-12) <= v["kernel"] * (1 + 1e-12) <= v["serial"] * (1 + 1e-12)
        t = r["tbo"].stages[0].time
        assert t.t_link > 0 and t.total >= t.t_link


def test_moe_harness_bands():
    """Regression bands for MODEL §25.7 (not accuracy targets): large-EP rows (TBO software) agree with the
    comm-hiding modes; the dense-fitted serial mode under-predicts them ~2×; TRT-LLM DS-R1 (no TBO) is over-predicted."""
    R = X.rows(X.DATA_MOE)
    ep = [r for r in R if r["src"] != "trtllm-0.21-dsr1"]
    g = lambda v, rs: math.exp(sum(math.log(X.predict(r, v)["pred"] / r["value"]) for r in rs) / len(rs))
    assert 0.85 < g("catalog_tbo", ep) < 1.15
    assert g("catalog_serial", ep) < 0.65
    trt = [r for r in R if r["src"] == "trtllm-0.21-dsr1"]
    assert 1.0 < g("catalog_serial", trt) < 1.3 < g("catalog_kernel", trt) < 1.5 < g("catalog", trt)


def test_mixtral_tp8_bands():
    R = [r for r in X.rows(X.DATA_MOE) if r["src"] == "trtllm-0.17"]
    assert len(R) == 27 and all(r["etp"] == r["tp"] == 8 for r in R)
    g = lambda v, rs: math.exp(sum(math.log(X.predict(r, v)["pred"] / r["value"]) for r in rs) / len(rs))
    hop = [r for r in R if r["hw"] != "A100-SXM-80GB"]
    assert 0.7 < g("catalog_kernel", hop) < 1.0 < g("catalog", hop) < 1.35
    assert 0.8 < g("catalog_serial", [r for r in R if r["hw"] == "A100-SXM-80GB"]) < 1.05
