"""0.181: continuous-batching proxy packs ⌊C/isl⌋ prompts per prefill iteration; TRT-LLM max_batch_size 2048 cap."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import extval as X


def _mix(isl, osl, hw="H100-SXM"):
    return next(r for r in X.rows(X.DATA_MOE) if r["src"] == "trtllm-0.17" and r["isl"] == isl and r["osl"] == osl
                and r["hw"] == hw)


def test_short_prompts_pack():
    r = _mix(128, 128)
    new = X.predict_cb(r, "default")
    assert new["batch"] == 2048                      # TRT-LLM max_batch_size default, not osl / 1 = 128
    X.PACK_PROMPTS["default"] = False
    try:
        old = X.predict_cb(r, "default", max_seqs=0)
    finally:
        X.PACK_PROMPTS.pop("default")
    assert old["batch"] == 128 and new["pred"] > 3 * old["pred"]


def test_trtllm_cap_and_long_prompts_unchanged():
    assert all(X.SOURCE_ESTIMATOR[k]["max_seqs"] == 2048 for k in X.SOURCE_ESTIMATOR if k.startswith("trtllm"))
    r = _mix(20000, 2000)                            # isl > C: one prompt spans chunks; packing does not apply
    a = X.predict_cb(r, "default")
    X.PACK_PROMPTS["default"] = False
    try:
        b = X.predict_cb(r, "default")
    finally:
        X.PACK_PROMPTS.pop("default")
    assert a["pred"] == b["pred"]
