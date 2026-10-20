"""0.281: per-role GEMM input formats for the remaining quantized releases — AMD Quark W8A8 PTPC FP8 and W4A4 MXFP4
(vLLM QuarkW8A8Fp8 per-token dynamic / OCP MX dynamic MXFP4 inputs), Drama-3 bsgemm-moe (block-FP8 experts with dynamic
activations) — and the fused-expert scale suffix (Muse Spark NVFP4 experts.*_weight_scale_2)."""
import importlib.util
import json
from pathlib import Path

from accel_dse.core.catalog import get_model, labels
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.model import RoleFormat, _quark_act_roles, _w4a4_roles, load_release
from accel_dse.core.parallel import Layout
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario, Serving

ROOT = Path(__file__).resolve().parents[1]


def _sr():
    spec = importlib.util.spec_from_file_location("summarize_release", ROOT / "scripts/summarize_release.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _acts(mid):
    return {k: v.fmt for k, v in get_model(mid).formats if k.startswith("act_")}


def _ev(mid, phase, hw="B200"):
    h = REF_HW[hw]
    return evaluate(Scenario(model=mid, chip=h.chip, mem_id=h.mem_id, layout=Layout(tp=8, ep=8),
                             serving=Serving(phase=phase, batch=1, ctx=4096, prompt=4096)))


def test_quark_and_bsgemm_activation_roles():
    assert _acts("muse-spark-open-fp8") == {"act_attn": "fp8", "act_expert": "fp8"}      # bf16 mlp / lm_head keep bf16 inputs
    assert _acts("muse-spark-open-mxfp4") == {"act_expert": "mxfp4"}
    assert _acts("drama-3") == {"act_expert": "fp8"}
    for mid in ("muse-spark-open-fp8", "muse-spark-open-mxfp4", "drama-3"):
        assert get_model(mid).act_fmt == "bf16"
    qc = load_release("amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8")["quantization_config"]["global_quant_config"]["input_tensors"]
    assert (qc["dtype"], qc["is_dynamic"], qc["qscheme"]) == ("fp8_e4m3", True, "per_channel")   # per-token dynamic


def test_quark_guards():
    fm = {"expert": RoleFormat("mxfp4", 4.25), "attn": RoleFormat("bf16", 16.0)}
    g = {"input_tensors": {"dtype": "fp4"}, "weight": {"dtype": "fp4"}}
    assert _quark_act_roles({"quant_method": "quark", "global_quant_config": g}, fm) == {"act_expert": RoleFormat("mxfp4", 4.25)}
    # per-layer overrides are not modelled → no activation formats (conservative bf16)
    assert _quark_act_roles({"quant_method": "quark", "global_quant_config": g, "layer_quant_config": {"x": {}}}, fm) == {}
    # W4A8 (MXFP4 weights × FP8 inputs) is another family
    g8 = {"input_tensors": {"dtype": "fp8_e4m3"}, "weight": {"dtype": "fp4"}}
    assert _quark_act_roles({"quant_method": "quark", "global_quant_config": g8}, fm) == {}
    # weight-only Quark (no input_tensors) keeps bf16
    assert _w4a4_roles({"quantization_config": {"quant_method": "quark", "global_quant_config": {"weight": {"dtype": "fp4"}}}}, fm) == {}


def test_gemm_inputs_and_ttft():
    m = get_model("muse-spark-open-fp8")
    ops = [o for o in build_rank_ops(m, 1, Phase("prefill", 1, 512, 512), Shard()) if o.kind == "gemm"]
    assert all(o.a_fmt == "fp8" for o in ops if o.role in ("attn", "expert") and o.w_fmt == "fp8")
    assert all(o.a_fmt == "bf16" for o in ops if o.w_fmt == "bf16")
    for mid in ("muse-spark-open-fp8", "muse-spark-open-mxfp4"):
        p, d = _ev(mid, "prefill"), _ev(mid, "decode")
        assert p.fits and d.fits and p.ttft > 0
    # B200 native fp8 / fp4: W8A8 and W4A4 prefill beat the dequant-to-bf16 path ≤ 0.280 (0.0762 / 0.0761 s)
    assert _ev("muse-spark-open-fp8", "prefill").ttft < 0.0762
    assert _ev("muse-spark-open-mxfp4", "prefill").ttft < 0.0761


def test_fused_expert_scale_suffix():
    sr = _sr()
    for n, b in (("m.layers.3.mlp.experts.gate_up_proj_weight_scale_2", "m.layers.3.mlp.experts.gate_up_proj"),
                 ("m.layers.3.mlp.experts.gate_up_proj_weight_scale", "m.layers.3.mlp.experts.gate_up_proj"),
                 ("m.layers.3.mlp.experts.down_proj_input_scale", "m.layers.3.mlp.experts.down_proj"),
                 ("x.q_proj.weight_scale_2", "x.q_proj"), ("x.weight_scale_inv", "x")):
        assert sr.SCALE_RE.search(n) and sr._base(n) == b, n
    assert not sr.SCALE_RE.search("blocks.0.scale_shift_table")


def test_muse_nvfp4_role_table():
    d = json.load(open(ROOT / "accel_dse/data/releases/meta-models__Muse-Spark-Open-816B-A42B-NVFP4.json"))
    ex = d["roles"]["expert"]
    assert set(ex) == {"bf16", "nvfp4"} and abs(8 * ex["nvfp4"]["bytes"] / ex["nvfp4"]["params"] - 4.5) < 1e-5
    m = get_model("muse-spark-open-nvfp4")
    assert abs(dict(m.formats)["expert"].bits - 4.871) < 2e-3 and abs(m.param_check()["rel_err"]) < 2e-6
    assert labels(m)["dtype"] == "W bf16/专家 nvfp4 · A bf16/专家 nvfp4 · KV bf16"
