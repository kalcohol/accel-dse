"""0.177: GPU glue kernels (ir.glue_ops, Chip.gpu_glue) and the re-scored catalog_kernel_fixed."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import extval as X
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops, head_ops


def _row():
    return next(r for r in X.rows(X.DATA_MOE) if r["src"] == "deepseek-profile-decode")


def test_glue_inventory_dsv3_decode():
    m = get_model("deepseek-v3")
    ph, sh = Phase("decode", 128, 1, 4096), Shard(dp=128, ep=128)
    base = build_rank_ops(m, 5, ph, sh)
    g = [o.name for o in build_rank_ops(m, 5, ph, sh, glue=True)[len(base):]]
    assert g == ["glue.cast_attn_in", "glue.cast_q_b", "glue.cast_o", "glue.q_a_norm", "glue.attn_combine",
                 "glue.router_topk", "glue.router_splitk_reduce", "glue.shared_act"]
    # dense layer (no dispatch): the FFN input is cast too
    assert "glue.cast_ffn_in" in [o.name for o in build_rank_ops(m, 0, ph, sh, glue=True)]
    assert sum(o.name.startswith("glue.sample") for o in head_ops(m, ph, sh, glue=True)) == 4
    assert not any(o.name.startswith("glue.") for o in head_ops(m, ph, sh))


def test_kernels_per_layer_vs_trace():
    r = _row()
    out = {}
    for g in (False, True):
        X.VARIANTS[f"_k1{g}"] = {**X.VARIANTS["catalog_kernel"], "layer_us": 0, "kernel_vec": True, "kernel_us": 1.0, "glue": g}
        res = evaluate(X.scenario(r, f"_k1{g}", phase="decode", batch=128, ctx=4096))
        out[g] = res.stages[res.heaviest_stage].time.t_kern * 1e6 / 61
    assert abs(out[False] - 17.21) < 0.05 and abs(out[True] - 29.05) < 0.05     # trace: 30.4


def test_off_by_default_and_variant():
    r = _row()
    a = evaluate(X.scenario(r, "catalog_kernel", phase="decode", batch=128, ctx=4096)).tpot
    X.VARIANTS["_g0"] = {**X.VARIANTS["catalog_kernel"], "glue": True}
    b = evaluate(X.scenario(r, "_g0", phase="decode", batch=128, ctx=4096)).tpot
    assert b > a                                   # glue work (no kernel cost) still adds a little
    v = X.VARIANTS["catalog_kernel_fixed"]
    assert v["glue"] and v["kernel_vec"] and v["layer_us"] == 0 and isinstance(v["kernel_us"], dict)
    assert "glue" not in X.VARIANTS["catalog_kernel"] and X.VARIANTS["default"]["glue"]   # 0.178
