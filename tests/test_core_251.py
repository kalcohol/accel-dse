"""0.251 catalog freshness sweep (HF releases 2026-10-14 … 19): every new entry loads, builds the IR and evaluates on the
default chip without NaN; Hy3 MoE (tencent Hy-Memory-30B-A3B), FLUX 3 Image Klein (per-modality early blocks), Step-5
NVFP4, two Qwen3.5 fine-tunes merged into the same-architecture entries, Muse NVFP4 read from the public mirror."""
import json
import math
import sys
sys.path.insert(0, ".")
from accel_dse.core import catalog as C
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, build_rank_ops
from accel_dse.core.scenario import Scenario

NEW_LLM = ("hy-memory-30b-a3b", "hy-memory-30b-a3b-fp8", "step-5-preview-nvfp4", "userrepgen-4b", "flux-3-image-klein-pup")
NEW_GEN = ("flux-3-image-klein", "flux-3-image-klein-base")


def _finite(r):
    vals = [r.tpot, r.ttft, r.latency]
    for st in r.stages:
        t = st.time
        vals += [t.t_pipe, t.t_dram, t.t_array, t.flops, t.dram_bytes]
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def test_new_entries_load_build_and_evaluate():
    ids = {e["id"] for e in C.entries()}
    for mid in NEW_LLM + NEW_GEN:
        assert mid in ids, mid
        m = C.get_model(mid)
        assert abs(m.param_check()["rel_err"]) < 1e-9, mid                 # exact against the safetensors headers
        if mid in NEW_LLM:
            ops = build_rank_ops(m, 1, Phase("decode", 1, 1, 4096))
            assert ops and all(math.isfinite(o.flops) and o.flops >= 0 for o in ops), mid
            for ph in ("decode", "prefill"):
                r = evaluate(Scenario().replace("model", mid).replace("serving.phase", ph))
                assert _finite(r) and (r.tpot > 0 if ph == "decode" else r.ttft > 0), (mid, ph)
        else:
            r = evaluate(Scenario().replace("model", mid))
            assert _finite(r) and r.latency > 0, mid


def test_hy3_moe_mechanisms():
    """HYV3: expert / shared width = expert_hidden_dim 768 (header experts.*.gate_proj [768, 2048]), not
    moe_intermediate_size 1536; 1 shared expert; layer 0 dense (mlp_layer_types); q/k RMSNorm; fp32 expert_bias."""
    m = C.get_model("hy-memory-30b-a3b")
    f0, f1 = m.layers[0].ffn, m.layers[1].ffn
    assert f0.kind == "dense" and f0.d_ff == 6912
    assert (f1.kind, f1.n_experts, f1.top_k, f1.d_expert, f1.n_shared, f1.d_shared) == ("moe", 128, 8, 768, 1, 768)
    assert m.layers[1].core.n_q == 32 and m.layers[1].core.n_kv == 4 and m.layers[1].core.qk_dim == 128
    assert m.layers[1].misc_params == 2 * 2048 + 2 * 128 + 128
    assert m.coverage == "full" and m.max_ctx == 262144
    q = C.get_model("hy-memory-30b-a3b-fp8")
    assert q.act_fmt == "fp8" and q.kv_fmt == "fp8" and dict(q.formats)["expert"].fmt == "fp8"   # static W8A8 + fp8 KV
    a = evaluate(Scenario().replace("model", "hy-memory-30b-a3b").replace("serving.phase", "decode"))
    b = evaluate(Scenario().replace("model", "hy-memory-30b-a3b-fp8").replace("serving.phase", "decode"))
    assert b.tpot < a.tpot


def test_flux3_klein():
    """Flux3Transformer2DModel from the header: 5 per-modality early blocks (txt / image run at text-to-image; the
    image_cond reference stream is stored only) + 28 shared single blocks; distilled 16 steps no guidance, Base 50 × CFG."""
    k, kb = C.get_model("flux-3-image-klein"), C.get_model("flux-3-image-klein-base")
    assert len(k.layers) == 33 and k.hidden == 3072 and k.coverage == "partial"
    assert any("image_cond" in r for r in k.coverage_reasons)
    names = {l.name for l in k.layers[0].attn_linears}
    assert {"img_q", "txt_q", "img_o", "txt_o"} <= names and not any("cond" in n for n in names)
    assert k.layers[0].ffn.d_ff == 9216 and k.layers[5].ffn.d_ff == 9216
    cond = 5 * (4 * 3072 * 3072 + 3 * 3072 * 9216 + 2 * 128) + 3072 * 128 + 2 * 9216 * 3072
    assert k.io_misc_params >= cond                                    # reference stream: storage only
    assert kb.release_params == k.release_params - (3072 * 256 + 3072 * 3072)   # Base has no flow-map target embedder
    assert (k.workload.steps, k.workload.cfg, kb.workload.steps, kb.workload.cfg) == (16, 1, 50, 2)
    t, tb = (evaluate(Scenario().replace("model", i)).latency for i in ("flux-3-image-klein", "flux-3-image-klein-base"))
    assert 5.0 < tb / t < 7.0                                           # ≈ 100 / 16 forwards, pipeline overhead shared
    for hf in ("black-forest-labs/flux-3-image-klein", "black-forest-labs/flux-3-image-klein-base"):
        d = json.load(open(f"accel_dse/data/releases/{hf.replace('/', '__')}.json"))
        assert d["repo"] == hf and d["mirror"]["repo"].startswith("unsloth/")


def test_merged_same_architecture_and_nvfp4():
    m = {x["id"]: x for x in C.entries()}
    assert m["userrepgen-4b"]["merged_into"] == "startlux-decision-v1-4b" and not m["userrepgen-4b"]["listed"]
    assert m["flux-3-image-klein-pup"]["merged_into"] == "qwen3.5-9b" and not m["userrepgen-4b"]["decision"]
    for a, b in (("userrepgen-4b", "startlux-decision-v1-4b"), ("flux-3-image-klein-pup", "qwen3.5-9b")):
        x, y = C.get_model(a), C.get_model(b)
        assert x.layers == y.layers and x.hidden == y.hidden and x.vocab == y.vocab
        assert not x.mtp_layers and y.mtp_layers                       # fine-tunes ship no MTP head
    assert m["hy-memory-30b-a3b"]["provider"] == "tencent" and m["flux-3-image-klein"]["provider"] == "bfl"
    s, s8 = C.get_model("step-5-preview-nvfp4"), C.get_model("step-5-preview")
    assert s.layers == s8.layers and dict(s.formats)["expert"].fmt == "nvfp4" and dict(s8.formats)["expert"].fmt == "fp8"
    assert s.release_bytes < s8.release_bytes and s.coverage == s8.coverage == "partial"
