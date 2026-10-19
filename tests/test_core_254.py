"""0.254: Reflection Beam-1.0-501B-A23B (BF16 / FP8 / NVFP4 as released; transformers models/beam + vLLM models/beam),
DFlash2 block drafts (Beam + Muse-Spark-Open; vLLM qwen3_dflash2.py DFlashGroupedConv / CandidateSelector,
models/beam/nvidia/dflash2.py, spec_decode/dflash2/speculator.py) and as-released activation formats (compressed-tensors
W8A8 fp8, ModelOpt NVFP4 W4A4)."""
import sys
sys.path.insert(0, ".")
from accel_dse.core import catalog as C
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops, dspark_ops
from accel_dse.core.model import spec_accept_from_tau
from accel_dse.core.scenario import Scenario

B, BD, MD = "beam-1.0-501b-a23b", "beam-1.0-501b-a23b-nvfp4-dflash2", "muse-spark-open-nvfp4-dflash2"


def _sc(mid, phase="decode", **sv):
    s = (Scenario().replace("model", mid).replace("serving.phase", phase).replace("layout.tp", 8)
         .replace("layout.ep", 8).replace("layout.etp", 1))
    for k, v in sv.items():
        s = s.replace("serving." + k, v)
    return s


def test_beam_structure_exact():
    for mid, fm in ((B, "bf16"), (B + "-fp8", "fp8"), (B + "-nvfp4", "nvfp4")):
        m = C.get_model(mid)
        assert m.param_check()["rel_err"] == 0.0 and m.release_params == 501267612352
        assert m.coverage == "full" and not m.coverage_reasons and not m.tie_embeddings
        assert len(m.layers) == 52 and sum(1 for L in m.layers if L.ffn.kind == "moe") == 49
        full = [i for i, L in enumerate(m.layers) if L.core.window is None]
        assert full == [5, 11, 17, 23, 29, 35, 41, 47, 51]                   # every 6th + the last (NoPE)
        assert {L.core.window for L in m.layers if L.core.window} == {512}
        assert m.layers[0].attn_linears[0].n == 2 * 48 * 128                 # q + gate_proj output gate
        moe = m.layers[3].ffn
        assert (moe.n_experts, moe.top_k, moe.d_expert, moe.n_shared) == (256, 8, 2560, 0)
        f = dict(m.formats)
        assert f["expert"].fmt == fm and f["router"].fmt == "fp32"
        assert f["attn"].fmt == ("bf16" if fm == "nvfp4" else fm)            # NVFP4 keeps attention / dense layers bf16
        assert f["mlp"].fmt == ("bf16" if fm == "nvfp4" else fm)
    assert C.get_model(B).act_fmt == "bf16" and C.get_model(B + "-fp8").act_fmt == "fp8"   # W8A8 per-token dynamic
    e = [x for x in C.entries() if x["id"].startswith("beam-")]
    assert len(e) == 4 and {x["family"] for x in e} == {"Beam-1.0"} and {x["provider"] for x in e} == {"reflection"}


def test_activation_formats_as_released():
    nv = C.get_model(B + "-nvfp4")
    assert nv.act_fmt == "bf16" and nv.act_fmt_for("expert") == "nvfp4" and nv.act_fmt_for("attn") == "bf16"
    ops = build_rank_ops(nv, 3, Phase("prefill", 1, 512, 512))
    ex = [o for o in ops if o.kind == "gemm" and o.role == "expert"]
    assert ex and all(o.a_fmt == "nvfp4" and o.w_fmt == "nvfp4" for o in ex)
    assert all(o.a_fmt == "bf16" for o in ops if o.kind == "gemm" and o.role == "attn")
    for mid in ("muse-spark-open-nvfp4", "step-5-preview-nvfp4"):
        assert C.get_model(mid).act_fmt_for("expert") == "nvfp4"           # ModelOpt quant_algo NVFP4 = W4A4
    assert C.get_model("dory1.0-4b-nvfp4").act_fmt_for("mlp") == "bf16"   # W4A16_NVFP4 stays weight-only
    lf = C.get_model("lfm2.5-24b-a2b-fp8")
    assert lf.act_fmt == "fp8"                                            # compressed-tensors FP8_BLOCK W8A8
    s = _sc("lfm2.5-24b-a2b-fp8", "prefill")
    assert evaluate(s).ttft < evaluate(s.replace("formats_override", (("act", "bf16"),))).ttft


def test_dflash2_drafts_exact():
    for mid, n, dfl, win, blk in ((BD, 1927484672, (2, 16, 256, 16, 16), 4096, 8),
                                  (MD, 4384162432, (2, 16, 256, 16, 8), 2048, 8)):
        m = C.get_model(mid)
        pc = m.param_check()
        assert pc["mtp_ours"] == pc["mtp_release"] == n
        assert m.draft == "dspark" and m.draft_anchor and m.draft_dflash2 == dfl and m.draft_block == blk
        assert len(m.mtp_layers) == 5 and {L.core.window for L in m.mtp_layers} == {win}
        assert m.draft_cold_params >= 2 * m.vocab * 256                   # predecessor / successor codebooks (cold)
        assert not m.draft_bridge and m.spec_published[1] == 7
    assert C.get_model(BD).spec_published == (4.876, 7) and C.get_model(BD).spec_published_t1 == 4.67
    assert C.get_model(MD).spec_published == (3.687, 7) and C.get_model(MD).spec_published_t1 == 3.33


def test_dflash2_ops():
    m = C.get_model(BD)
    ops = dspark_ops(m, Phase("decode", 2, 1, 8192), Shard())
    names = [o.name for o in ops]
    assert names.count("dflash2.attn_conv_proj") == names.count("dflash2.mlp_conv_proj") == 5
    cp = next(o for o in ops if o.name == "dflash2.attn_conv_proj")
    assert (cp.m, cp.k, cp.n) == (2 * 8, 5120, 2 * 2 * 320)               # kernel_projection h → 2·taps·groups
    assert all(o.causal == 1.0 for o in ops if o.name == "dspark_block_qk")   # non-causal block
    head = next(o for o in ops if o.name == "dspark.head")
    assert head.m == 2 * 7                                                # lm_head on the B − 1 draft rows
    ed = next(o for o in ops if o.name == "dflash2_edges")
    assert (ed.m, ed.k, ed.n, ed.count) == (16, 256, 16, 14)              # K × K × rank per draft row
    kv = [o for o in ops if o.name == "dspark_main_kv"]
    assert len(kv) == 5 and all(o.kv_write == 2 * 2 * 8 * 128 * 2 for o in kv)   # windowed draft cache rows written
    assert not any(n.startswith("dspine_serial") for n in names)


def test_dflash2_block_follows_spec_k():
    r = {k: evaluate(_sc(BD, spec_k=k, spec_accept=0.8)) for k in (3, 7, 15)}
    assert r[3].tpot != r[7].tpot != r[15].tpot
    assert any("块 = 1 + spec_k = 4" in w for w in r[3].warnings)
    w = evaluate(_sc(BD, spec_k=16)).warnings
    assert any("超过 DFlash2 训练块 16" in x for x in w)
    w = evaluate(_sc(MD, spec_k=9)).warnings
    assert any("超过 DFlash2 训练块 8" in x for x in w)
    pub = _sc(BD, spec_k=7, spec_accept_src="published")
    a = evaluate(pub)
    assert a.tpot == evaluate(_sc(BD, spec_k=7, spec_accept=round(spec_accept_from_tau(4.876, 7), 4))).tpot
    assert evaluate(pub.replace("serving.spec_sampling", "sampled")).tpot > a.tpot   # τ 4.67 (T = 0.7) < 4.876
