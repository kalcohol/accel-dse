"""0.199: NCCL small-message all-reduce latency (Ring LL, src/graph/tuning.cc v2.19.3)."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import evaluate as E, schedule as S
from accel_dse.core.hardware import Link


def test_nccl_ring_ll_values():
    assert S.nccl_ring_ll_us(1) == 0.0
    assert abs(S.nccl_ring_ll_us(2) - 7.8) < 1e-9 and abs(S.nccl_ring_ll_us(8) - 15.0) < 1e-9


def test_allreduce_alpha_grows_with_group():
    L = Link(450.0, 3.0, "switch", ar_lat="gpu")   # 0.201: GPU link
    with E.model_flags(CUSTOM_AR=False):           # 0.200: NCCL path (no custom all-reduce)
        a2 = S.collective_seconds("allreduce", 1024, 2, L)[1]
        a8 = S.collective_seconds("allreduce", 1024, 8, L)[1]
        a16 = S.collective_seconds("allreduce", 1024, 16, L)[1]
    assert abs(a8 - 15e-6) < 1e-12 and a8 > a2
    assert abs(S.collective_seconds("allreduce", 1024, 8, L)[1] - 6.8e-6) < 1e-12   # custom AR floor (default)
    assert abs(S.collective_seconds("allreduce", 1024, 16, L)[1] - a16) < 1e-12      # > 8 GPUs: NCCL
    assert S.collective_seconds("alltoall", 1024, 8, L)[1] == 3e-6      # other collectives unchanged
    with E.model_flags(NCCL_AR_LAT=False):
        assert S.collective_seconds("allreduce", 1024, 8, L)[1] == 3e-6


def test_ttft_tp8_row_moves_toward_measured():
    from accel_dse.core import extval as X
    r = next(r for r in X.rows() if r.get("id") == 52)        # A100 llama-2-70b TP8 128 tokens, 35.6 ms
    new = X.predict(r, "default")["pred"]
    with E.model_flags(NCCL_AR_LAT=False):
        X.cta_ceiling_chip.cache_clear()
        old = X.predict(r, "default")["pred"]
    X.cta_ceiling_chip.cache_clear()
    assert old < new < r["value"]
