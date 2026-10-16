"""0.178: default = mechanism (per-kernel cost + glue) on CUDA rows; ROCm decode sourced; AMD prefill lump fallback."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import extval as X


def test_default_definition():
    d, old = X.VARIANTS["default"], X.VARIANTS["default_lump"]
    assert old["layer_us"] == X.TRACE_LAYER_US and "glue" not in old
    assert d["layer_us"] == 0 and d["glue"] and d["kernel_vec"] and d["cuda_only"]
    assert d["ceil_mul"] and d["tile_io"] == "dram"          # 0.193 (MODEL §140)
    assert X.VARIANTS["default_0192"] == {k: v for k, v in d.items() if k not in ("ceil_mul", "tile_io", "trtllm_fp4", "pa_rocm")}
    assert d["fallback_us"] == X.TRACE_LAYER_US and d["amd_kernel_us"] == X.ROCM_KERNEL_US
    assert {k: v for k, v in d.items() if k not in ("layer_us", "glue", "kernel_vec", "cuda_only", "fallback_us",
                                                      "kernel_us", "amd_kernel_us", "ceil_mul", "tile_io", "trtllm_fp4", "pa_rocm")} == {k: v for k, v in old.items() if k != "layer_us"}


def test_rocm_constants_sourced():
    assert abs(X.ROCM_KERNEL_FLOOR_US - 4.833) < 1e-3 and abs(X.ROCM_GRAPH_GAP_US - 0.0203) < 1e-3
    assert set(X.ROCM_KERNEL_US) == {"decode"}


def test_scoping():
    h = next(x for x in X.rows() if x["metric"] == "ttft_ms" and x["hw"] == "H100-SXM")
    s = X.scenario(h, "default", phase="prefill")
    assert s.chip.gpu_glue and s.layer_overhead_us == 0 and s.chip.kernel_us == X.VARIANTS["default"]["kernel_us"]["prefill"]
    a = next(x for x in X.rows(X.DATA_R3) if x["hw"] == "MI300X")
    d, p = X.scenario(a, "default", phase="decode"), X.scenario(a, "default", phase="prefill")
    assert d.chip.gpu_glue and d.chip.kernel_us == X.ROCM_KERNEL_US["decode"] and not d.layer_overhead_us
    assert not p.chip.gpu_glue and p.layer_overhead_us == X.TRACE_LAYER_US
