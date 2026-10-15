"""0.137: under exec_overlap "tbo" the per-kernel / per-layer overhead runs on the compute stream and is hidden by the
other micro-batch's all-to-all like the kernels themselves (was added after the max, so a link-bound TBO stage paid
it — the per-SM candidate's only regression, MoE prefill 8.8 → 8.9 %)."""
from accel_dse.core import extval as X
from accel_dse.core.schedule import StageTime


def test_tbo_hides_kernel_overhead_when_link_bound():
    st = StageTime(1.0, 1.0, 0.0, 0.1, 0.5, 2.0, 0.3, 0, 0, 0, overlap="tbo", t_kern=0.2)
    assert abs(st.total - (2.0 + 0.1)) < 1e-12          # link-bound: only the α part (0.3 − 0.2) stays exposed
    st2 = StageTime(1.9, 1.9, 0.0, 0.1, 0.5, 2.0, 0.3, 0, 0, 0, overlap="tbo", t_kern=0.2)
    assert abs(st2.total - (1.9 + 0.1 + 0.2 + 0.1)) < 1e-12   # compute + overhead exceeds link: overhead counts
    k = StageTime(1.0, 1.0, 0.0, 0.1, 0.5, 2.0, 0.3, 0, 0, 0, overlap="kernel", t_kern=0.2)
    assert abs(k.total - (1.0 + 0.1 + 2.0 + 0.3)) < 1e-12     # kernel mode unchanged


def test_moe_prefill_candidate_equals_catalog():
    C = X.candidate_variant()
    for r in X.rows(X.DATA_MOE):
        if r["metric"] == "prefill_tok_s_node":
            a, b = X.predict(r, "catalog")["pred"], X.predict(r, X.tileio_variant(C))["pred"]
            assert abs(a - b) / a < 1e-9, r["id"]
