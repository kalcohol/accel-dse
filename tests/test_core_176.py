"""0.176: mechanism-scoped fixed cost for catalog_kernel (deepseek-ai/profile-data decode / prefill traces)."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import extval as X


def test_trace_constants():
    assert abs(X.TRACE_GAP_US["graph"] - 1.0254) < 1e-3          # 3,791 µs over 3,697 decode kernel boundaries
    assert abs(X.TRACE_GAP_US["eager"] - 2.4043) < 1e-3          # 8,230 µs over 3,423 prefill boundaries
    assert X.RUNTIME_BY_PHASE == {"decode": "graph", "prefill": "eager"}
    v = X.VARIANTS["catalog_kernel_fixed"]
    assert v["layer_us"] == 0 and v["kernel_vec"] and v["cuda_only"]
    assert abs(v["kernel_us"]["decode"] - (X.TRACE_KERNEL_FLOOR_US + X.TRACE_GAP_US["graph"])) < 1e-12
    assert "catalog_kernel" in X.VARIANTS and "layer_us" not in X.VARIANTS["catalog_kernel"]   # baseline unchanged


def test_scoped_by_phase_and_vendor():
    r = next(x for x in X.rows() if x["metric"] == "ttft_ms" and x["isl"] == 128 and x["hw"] == "H100-SXM" and x["tp"] == 1)
    assert X.scenario(r, "catalog_kernel_fixed", phase="prefill").chip.kernel_us == X.VARIANTS["catalog_kernel_fixed"]["kernel_us"]["prefill"]
    assert X.scenario(r, "catalog_kernel_fixed", phase="decode").chip.kernel_us == X.VARIANTS["catalog_kernel_fixed"]["kernel_us"]["decode"]
    a = next(x for x in X.rows(X.DATA_R3) if x["hw"] == "MI300X")
    # no ROCm trace → AMD rows identical to the no-fixed-cost baseline
    assert X.predict(a, "catalog_kernel_fixed")["pred"] == X.predict(a, "catalog_kernel")["pred"]


def test_128_token_ttft_moves_toward_measurement_but_residual_remains():
    rs = [x for x in X.rows() if x["metric"] == "ttft_ms" and x["isl"] == 128 and x["hw"] in ("H100-SXM", "H200-SXM") and x["tp"] == 1]
    for r in rs:
        old, new = X.predict(r, "catalog_kernel")["pred"], X.predict(r, "catalog_kernel_fixed")["pred"]
        assert old < new < r["value"]
        assert 1.2 < r["value"] - new < 2.5          # §122: 1.8–2.1 ms; 0.177 glue (§123) → 1.4–1.7 ms unexplained
