"""0.194: net GEMM ceiling (tile I/O + per-kernel cost netted at the source shape), Hopper ceiling merge, small-GEMM check."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import extval as X

ROWS = {r["hw"]: r for r in X.rows() + X.rows(X.DATA_R3)}


def _tf(hw, n):
    sc = X.scenario(ROWS[hw], "default", phase="decode", batch=1, ctx=1, out_len=1)
    return X._model_tflops(sc.chip, sc.mem_id, n, n, n, "bf16")


def test_hopper_keeps_bf16_ceiling():
    fc = dict(X.scenario(ROWS["H100-SXM"], "default", phase="decode").chip.fmt_ceiling)
    assert "gemm:bf16" in fc and "fp8" in fc and "attn:bf16" in fc


def test_net_ceiling_reproduces_source_rate():
    assert abs(_tf("H100-SXM", 8192) / 744.0 - 1) < 0.005          # cuBLAS 744 TF at 8192³ (GEMM_CEILING source)
    assert abs(_tf("B200", 8192) / 1551.0 - 1) < 0.005


def test_small_gemm_points_not_fitted():
    """Published cuBLAS bf16 points the model was NOT matched to (fast.cu README, pyptx.dev): within 10 %."""
    for hw, n, meas in (("H100-SXM", 4096, 716), ("B200", 2048, 1006), ("B200", 4096, 1532)):
        assert abs(_tf(hw, n) / meas - 1) < 0.10, (hw, n, _tf(hw, n))
    assert _tf("B200", 1024) < _tf("B200", 2048) < _tf("B200", 4096)
