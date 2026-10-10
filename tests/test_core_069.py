"""0.69 (V6.3): continuous-batching proxy, MI300X / B200 参考硬件, layer_overhead_us, published EPLB skews."""
import math
from dataclasses import replace

import pytest

from accel_dse.core import extval as X
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving


def test_refhw_mi300x_b200_datasheet():
    a, b = REF_HW["MI300X"], REF_HW["B200"]
    assert abs(a.chip.peak_tflops_bf16 - 1307.4) < 1e-6 and abs(b.chip.peak_tflops_bf16 - 2250.0) < 1e-6
    assert a.chip.formats.rate("fp8") == 2.0 and b.chip.formats.rate("nvfp4") == 4.0
    assert all("参考硬件" not in getattr(c, "name", "") for c in CHIPS.values())


def test_layer_overhead_default_identical_and_linear():
    s = Scenario(model="llama-3.1-8b", serving=Serving(batch=4, ctx=1024))
    assert "layer_overhead_us" not in s.to_dict()
    with pytest.raises(ValueError):
        replace(s, layer_overhead_us=-1.0)
    r0, r1, r2 = (evaluate(replace(s, layer_overhead_us=u)) for u in (0.0, 50.0, 100.0))
    d1, d2 = r1.tpot - r0.tpot, r2.tpot - r0.tpot
    assert abs(d1 - 32 * 50e-6) < 1e-12 and abs(d2 - 2 * d1) < 1e-12
    for m in ("class", "kernel", "serial", "tbo"):       # exposed in every overlap mode
        a, b = (evaluate(replace(s, exec_overlap=m, layer_overhead_us=u)) for u in (0.0, 100.0))
        assert abs(b.tpot - a.tpot - 32 * 100e-6) < 1e-12


def test_cb_proxy_reduces_to_decode_when_no_prefill_share():
    r = next(x for x in X.rows() if x["metric"] == "max_tok_s_total")
    v, _ = X._cb_at({**r, "isl": 1, "osl": 1 << 20}, 8, "catalog")
    dec = evaluate(X.scenario({**r, "isl": 1, "osl": 1 << 20}, "catalog", phase="decode", batch=8,
                              ctx=1 + (1 << 19), out_len=1 << 20))
    assert abs(v - 8 / dec.step) / v < 1e-4


def test_r3_rows_cited():
    R = X.rows(X.DATA_R3)
    assert len(R) == 31 and len({r["id"] for r in R}) == 31
    assert all(r["url"].startswith("https://") and r["hw"] in REF_HW for r in R)


def test_r3_bands():
    """MODEL §25.8 regression bands: MI300X latency is ~3× slower than the roofline without a per-layer overhead;
    100 µs/layer (fitted on H100 short-prompt TTFT only) brings it within ~25 %; B200 NVFP4 kernel mode ~1.2×."""
    R = X.rows(X.DATA_R3)
    lat = [r for r in R if r["metric"] == "static_latency_s"]
    g = lambda v, rs: math.exp(sum(math.log(r["value"] / X.predict(r, v)["pred"]) for r in rs) / len(rs))
    assert g("catalog_kernel", lat) > 2.5
    assert 0.85 < g(X.overhead_variant("catalog_kernel", 100), lat) < 1.2
    b2 = [r for r in R if r["hw"] == "B200"]
    gb = math.exp(sum(math.log(X.predict_cb(r, "catalog_kernel")["pred"] / r["value"]) for r in b2) / len(b2))
    assert 1.0 < gb < 1.35


def test_split_instances_engine_parallel_and_off_by_default():
    from accel_dse.core.mapping import gemm_cost
    ch = REF_HW["H800-SXM"].chip
    assert ch.split_instances is False and all(not c.split_instances for c in CHIPS.values())
    sp = replace(ch, split_instances=True)
    a = gemm_cost(ch, "reconf", 128, 4097, 512, count=128)
    b = gemm_cost(sp, "reconf", 128, 4097, 512, count=128)
    ideal = 128 * 4097 * 512 * 128 / ch.macs
    assert b.cycles < a.cycles and ideal <= b.cycles < 1.1 * ideal
    assert gemm_cost(sp, "reconf", 4096, 4096, 4096).cycles == gemm_cost(ch, "reconf", 4096, 4096, 4096).cycles
