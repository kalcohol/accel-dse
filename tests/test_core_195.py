"""0.195: CTA occupancy tile I/O (opt-in), TRT-LLM B200 NVFP4 kernel rates (default)."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import extval as X
from accel_dse.core.mapping import cta_occupancy
from accel_dse.core.refhw import REF_HW


def test_occupancy_from_documented_limits():
    h = REF_HW["H100-SXM"].chip
    assert (h.occ_smem_kib, h.occ_reg_kib, h.occ_threads) == (228.0, 256.0, 2048)
    assert cta_occupancy(h, 128, 256, 2, 2) == 1          # 3 × 384 × 128 B = 144 KiB smem
    assert cta_occupancy(h, 128, 128, 2, 2) == 2          # 96 KiB smem, 64 KiB acc
    assert cta_occupancy(h, 64, 64, 2, 2) == 4            # 48 KiB smem → 4
    assert cta_occupancy(REF_HW["MI300X"].chip, 64, 64, 2, 2) == 1   # 64 KiB LDS
    assert X.VARIANTS["default"].get("tile_io") == "dram" and X.VARIANTS["default_occ"]["tile_io"] == "occupancy"


def test_trtllm_fp4_rates():
    tf = dict(X.TRTLLM_FP4_SHAPES)
    assert abs(tf["gemm:nvfp4@8192x8192"] * 9000 - 4110) < 1 and abs(tf["gemm:nvfp4@8192x1024"] * 9000 - 2397) < 1
    rs = [r for r in X.rows(X.DATA_R3) if r["hw"] == "B200"]
    assert "gemm:nvfp4@8192x8192" in dict(X.scenario(rs[0], "default", phase="prefill").chip.fmt_ceiling)
    h100 = [r for r in X.rows() if r["hw"].startswith("H100")][0]
    assert "gemm:nvfp4@8192x8192" not in dict(X.scenario(h100, "default", phase="prefill").chip.fmt_ceiling or ())
