"""0.201: GPU mechanisms of 0.196–0.200 vs the NPU design path; prefill LM head = last token only."""
import sys
from dataclasses import replace
sys.path.insert(0, ".")
from accel_dse.core import evaluate as E, schedule as S, ir
from accel_dse.core.hardware import Link, CHIPS, NET_DEFAULT
from accel_dse.core.scenario import Scenario


def test_npu_link_keeps_own_alpha():
    npu = Link(400.0, 1.0)                      # user interconnect, α 1 µs
    assert S.collective_seconds("allreduce", 1024, 8, npu)[1] == 1e-6
    gpu = Link(450.0, 3.0, ar_lat="gpu")
    assert abs(S.collective_seconds("allreduce", 1024, 8, gpu)[1] - 6.8e-6) < 1e-12
    with E.model_flags(GPU_AR_ONLY=False):      # ≤ 0.200: the CUDA floor leaked into every switch link
        assert abs(S.collective_seconds("allreduce", 1024, 8, npu)[1] - 6.8e-6) < 1e-12


def test_cross_node_net_not_custom_ar():
    assert NET_DEFAULT.ar_lat == ""
    assert S.collective_seconds("allreduce", 1024, 8, NET_DEFAULT)[1] == NET_DEFAULT.alpha_us * 1e-6


def test_refhw_links_marked_gpu():
    from accel_dse.core import refhw
    assert open(refhw.__file__).read().count('ar_lat="gpu"') == 7


def test_link_ar_lat_hash_stable_and_validated():
    from dataclasses import fields
    f = next(f for f in fields(Link) if f.name == "ar_lat")
    assert f.metadata.get("omit_default")
    try:
        Link(400.0, 3.0, ar_lat="cuda")
        raise AssertionError
    except ValueError:
        pass


def _key(r):
    return (r.tick, r.step, r.ttft, r.tpot, r.throughput, [st.flops_u for st in r.stages])


def test_cta_flags_inert_on_npu_chips():
    for name in ("100T", "1P"):
        for ph in ("decode", "prefill"):
            s = Scenario(chip=CHIPS[name])
            s = replace(s, serving=replace(s.serving, phase=ph))
            base = _key(E.evaluate(s))
            with E.model_flags(CTA_DRAM_FLOOR=False, CTA_SMEM_BW=False, CTA_OCC_FILL=False, CTA_L2_ACT=False):
                assert _key(E.evaluate(s)) == base


def test_fuse_qkv_conserves_work_on_npu():
    s = Scenario(chip=CHIPS["100T"])
    on = sum(st.flops_u for st in E.evaluate(s).stages)
    with E.model_flags(FUSE_QKV=False):
        off = sum(st.flops_u for st in E.evaluate(s).stages)
    assert on > 0 and abs(on - off) <= 1e-9 * off


def test_prefill_lm_head_last_token_only():
    from accel_dse.core.catalog import get_model
    md = get_model(Scenario().model)
    h = [{o.name: o for o in ir.head_ops(md, ir.Phase("prefill", batch=2, q=S_))}["lm_head"] for S_ in (128, 2048)]
    assert h[0].m == h[1].m == 2
    d = {o.name: o for o in ir.head_ops(md, ir.Phase("decode", batch=2, q=1, ctx=128))}["lm_head"]
    assert d.m == 2
