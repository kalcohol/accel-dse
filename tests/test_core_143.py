"""0.143: DeepEP node-mode forwarding + achieved NVLink rate; per-shape DeepGEMM ceilings; UI accumulator default."""
import math

from accel_dse.core.hardware import Link
from accel_dse.core.schedule import DEEPEP_NVL_EFF, collective_wire, fabric_collective


def test_node_mode_forwards_remote_pairs_over_nvlink():
    P, G = 1e9, 32
    w_pair = collective_wire("alltoall", P, G, False, 1, 8)
    w_node = collective_wire("alltoall", P, G, False, 1, 8, net_scale=0.45)
    assert abs(w_pair["link"] - 7 / 32 * P) < 1 and abs(w_node["net"] - 0.45 * w_pair["net"]) < 1
    assert abs(w_node["link"] - (7 / 32 + 24 / 32 * 7 / 8) * P) < 1      # 0.875 P: remote pairs forwarded too
    nv, ib = Link(200.0, 1.0), Link(50.0, 5.0)
    bw = fabric_collective("alltoall", P, G, nv, net=ib, node=8, net_scale=0.45)[0]
    assert abs(bw - max(0.875 * P / (200e9 * DEEPEP_NVL_EFF), 0.75 * 0.45 * P / 50e9)) < 1e-12
    # per-pair mode unchanged
    assert abs(fabric_collective("alltoall", P, G, nv, net=ib, node=8)[0] - max(7 / 32 * P / 200e9, 24 / 32 * P / 50e9)) < 1e-12


def test_shape_ceiling_nearest_published_shape():
    from accel_dse.core.evaluate import _shape_cap
    from accel_dse.core.extval import DEEPGEMM_SHAPES
    from accel_dse.core.ir import Op
    g = Op("expert.down", "gemm", 0, m=4096, k=2048, n=7168, count=8)
    d = Op("attn.kv_b", "gemm", 0, m=4096, k=512, n=32768)
    assert _shape_cap(DEEPGEMM_SHAPES, g, "fp8") == 1099 / 1979
    assert _shape_cap(DEEPGEMM_SHAPES, d, "fp8") == 590 / 1979
    assert _shape_cap(DEEPGEMM_SHAPES, d, "bf16") is None and _shape_cap((), d, "fp8") is None


def test_catalog_exposes_accumulator_default():
    from accel_dse import api
    from accel_dse.core.hardware import CHIPS
    c = api.api_catalog()["chips"]
    for k, v in c.items():
        assert v["acc_kib_per_engine_default"] == 64.0
        assert math.isclose(v["acc_total_kib"], CHIPS[k].acc_total_kib) and v["acc_total_kib"] > 0
