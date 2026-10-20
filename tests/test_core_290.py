"""0.290 (MODEL §235): the ggml codebook / block formats on the hardware dtype matrix — IQ2_XXS dequant element-ops from
the ggml reference dequant, block-scale bytes, format-aware dequant headroom, and the 「反量化」 tag on VECTOR-bound
stages whose vector time is mostly weight dequant."""
import json
from dataclasses import replace
from pathlib import Path

from accel_dse import api
from accel_dse.core.catalog import get_model
from accel_dse.core.dtypes import DEQUANT_OPS, FORMATS, FormatSupport, dequant_ops, fmt, gemm_exec
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.model import load_release
from accel_dse.core.vecsanity import check, format_headroom
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

BOLT = "nemotron-3.5-bolt-74b-a9b-24gb"
ROOT = Path(__file__).resolve().parent.parent


def _eval(model, chip_preset="1P", chip=None, batch=8, phase="decode"):
    sc = {"model": model, "chip": chip_preset, "serving": {"phase": phase, "batch": batch}}
    if chip:
        sc = {**sc, "chip": chip}
    return evaluate(api.scenario_from_body({"chip_preset": chip_preset, "scenario": sc}))


def test_iq2_xxs_dequant_ops_from_the_ggml_reference():
    # ggml dequantize_row_iq2_xxs: per weight convert + scale (the 2-op convention) + sign; per 8 two table lookups
    # (iq2xxs_grid 256 × uint64, ksigns_iq2xs 128 B) + two index ops; per 32 the sub-block scale (5 ops)
    assert DEQUANT_OPS == {"iq2_xxs": 2 + 1 + 4 / 8 + 5 / 32} and dequant_ops("iq2_xxs") == 3.65625
    # Q8_0 (y = qs·d) and every other format keep the convention
    assert all(dequant_ops(f) == 2.0 for f in FORMATS if f != "iq2_xxs")


def test_ggml_block_bytes_incl_scales_match_the_release():
    # block_iq2_xxs = fp16 d + 32 × uint16 per 256 = 66 B; block_q8_0 = fp16 d + 32 int8 = 34 B (ggml-common.h)
    assert fmt("iq2_xxs").bytes == 66 / 256 and fmt("q8_0").bytes == 34 / 32
    roles = load_release("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB")["roles"]
    for role, f in (("expert", "iq2_xxs"), ("shared_expert", "iq2_xxs"), ("lm_head", "q8_0"), ("mlp", "q8_0")):
        r = roles[role][f]
        assert r["bytes"] == r["params"] * fmt(f).bytes, (role, f)


def test_iq2_gemm_conversion_ops_and_native_pair():
    ch = CHIPS["1P"]
    m, k, n = 8, 4096, 2048
    c2 = gemm_cost(ch, "os", m, k, n, w_fmt="iq2_xxs", a_fmt="bf16")
    c4 = gemm_cost(ch, "os", m, k, n, w_fmt="int4", a_fmt="bf16")
    assert gemm_exec("iq2_xxs", "bf16", ch.formats)[2] == "dequant_w"
    assert c2.convert_elems == c4.convert_elems == c2.convert_w == k * n
    assert c2.convert_x == c2.convert_wx == k * n * 1.65625 and c4.convert_x == c4.convert_wx == 0.0
    q8 = gemm_cost(ch, "os", m, k, n, w_fmt="q8_0", a_fmt="bf16")
    assert q8.convert_w == k * n and q8.convert_x == 0.0
    # a chip that decodes IQ2_XXS in the MAC datapath (a listed mixed pair) converts nothing
    nat = replace(ch, formats=replace(ch.formats, mixed=(("iq2_xxs", "bf16", 1.0),)))
    cn = gemm_cost(nat, "os", m, k, n, w_fmt="iq2_xxs", a_fmt="bf16")
    assert cn.convert_elems == cn.convert_x == 0.0 and cn.conversion == "none"


def test_format_headroom():
    ch, bw = CHIPS["1P"], 2000.0
    h4 = check(ch, bw)["dequant_headroom"][0]
    assert abs(format_headroom(ch, bw, "int4") / h4 - fmt("int4").bytes / 0.5) < 1e-12
    assert abs(format_headroom(ch, bw, "iq2_xxs") / h4 - (66 / 256) / 0.5 * 2 / 3.65625) < 1e-12
    assert format_headroom(ch, bw, "nvfp4") > h4 and format_headroom(ch, bw, "q8_0") > h4


def test_bolt_dequant_warning_names_iq2_headroom():
    r = _eval(BOLT)
    w = [x for x in r.warnings if "反量化" in x]
    assert len(w) == 1 and "iq2_xxs 余量" in w[0] and "int4 余量" in w[0], w
    # a model without a format below int4's headroom keeps the pre-0.290 text
    r = _eval("nemotron-3.5-lightning-30b-a3b-nvfp4", chip={"vector_lanes": 2048})
    w = [x for x in r.warnings if "反量化" in x]
    assert w and all("余量" in x and "nvfp4 余量" not in x for x in w), w


def test_iq2_vector_seconds_scale_with_the_ops():
    # one GEMM, same shape: the per-pass weight-dequant vector seconds of IQ2_XXS = 3.65625 / 2 × those of int4
    from accel_dse.core.evaluate import _op_seconds
    from accel_dse.core.hardware import System
    from accel_dse.core.ir import Op
    sys_ = System(chip=CHIPS["1P"])
    m = get_model(BOLT)
    vw = {f: _op_seconds(Op("x", "gemm", 0, m=8, k=4096, n=2048, w_fmt=f, a_fmt="bf16"), sys_, "os", m)[7]
          for f in ("iq2_xxs", "int4", "q8_0")}
    assert vw["int4"] > 0 and abs(vw["iq2_xxs"] / vw["int4"] - 3.65625 / 2) < 1e-12 and vw["q8_0"] == vw["int4"]


def test_dequant_bound_tag():
    # Bolt on 1P with 4096 vector lanes: VECTOR-bound, mostly weight dequant → tagged; default lanes → MAC, untagged
    r = _eval(BOLT, chip={"vector_lanes": 4096})
    st = r.stages[r.heaviest_stage]
    assert st.time.bound == "VECTOR" and api.dequant_bound(st)
    d = api.result_dict(r)["stages"][r.heaviest_stage]
    assert d["dequant_bound"] is True and abs(d["dequant_ms"] - st.conv_w * 1e3) < 1e-9
    base = _eval(BOLT)
    assert not api.dequant_bound(base.stages[base.heaviest_stage])
    assert api.result_dict(base)["stages"][0]["dequant_bound"] is False


def test_ui_shows_dequant_ops_and_tag():
    assert api.api_catalog()["dequant_ops"] == {"iq2_xxs": 3.65625}
    js = W.read()
    assert W.has(js, "boundTag(st.bound, st.dequant_bound)") and W.has(js, "boundTag(s.bound, S.dq)")
    assert W.has(js, "S.cat.dequant_ops") and W.has(js, "hv.dequant_bound")
    json.dumps(api.api_catalog()["dequant_ops"])
