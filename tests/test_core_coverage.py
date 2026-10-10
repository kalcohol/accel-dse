"""Coverage closure (Unreleased): DSA indexer, gpt-oss sinks and chunked linear attention per the reference code;
already-complete models keep their default numbers (KPI hashes pinned at 548b9d5 / 0.65.1)."""
from __future__ import annotations

import hashlib
import json
import math

from accel_dse.core.catalog import entries, get_model, offline_entries
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, build_rank_ops
from accel_dse.core.scenario import Scenario, Serving

_SV = (Serving(), Serving(phase="prefill", batch=2, prompt=8192), Serving(batch=16, ctx=32768))
# 50 models that were 「完整」 at 0.65.1: hash of (tpot, ttft, throughput, per_card, tick, fits, bound) × 3 scenarios
_FULL_0651 = {
"kimi-k2.7-code": "9f352c1808e8",
"deepseek-v3": "30f458027c8f",
"deepseek-v3.1": "30f458027c8f",
"deepseek-r1": "30f458027c8f",
"glm-4.5": "1f94d269a563",
"glm-4.5-air": "0af21bc1860e",
"glm-4.6": "1f94d269a563",
"kimi-k2": "a2d3378dda05",
"kimi-k2.5": "9f352c1808e8",
"qwen3-235b-a22b": "761382c8458e",
"qwen3-30b-a3b": "53c497801aee",
"qwen3-32b": "b975b7b3760e",
"qwen3-8b": "11f6f3a5d21f",
"mistral-large": "4ce7bca4c910",
"mixtral-8x22b": "0d8711691e1e",
"magistral-small": "21ef75e78165",
"phi-4": "8fd64e78ceb8",
"yi-1.5-34b": "4a9848721f34",
"internlm3-8b": "8c512c35a43b",
"internlm2-5-20b": "bd72fb4fe967",
"seed-oss-36b": "f547c49c5769",
"qwen3-32b-fp8": "cb91adbc72c9",
"qwen3-8b-fp8": "25b06545a766",
"qwen3-30b-a3b-fp8": "94343d72fd49",
"qwen3-235b-a22b-fp8": "d3f9a72514ad",
"qwen3-32b-awq": "3e44c63965e3",
"qwen3-8b-awq": "d0c28036e214",
"qwen3-4b": "659c4bad45d8",
"qwen3-1.7b": "c062686f3bf2",
"qwen3-0.6b": "e61d067c924b",
"qwen2.5-72b": "51b90d8f5e68",
"qwen2.5-3b": "8b8d7db5d3eb",
"qwen2.5-1.5b": "45cf41450b64",
"llama-3.1-8b": "dbce22f912d4",
"llama-3.3-70b": "8e50c6cabdc8",
"llama-3.2-3b": "fb036a79bd1b",
"llama-3.2-1b": "17ab9028535e",
"wan2.1-14b": "f178ae033325",
"wan2.1-1.3b": "309919c00604",
"cogvideox-5b": "35a06a9a61eb",
"cogvideox-2b": "79dd0c3869bd",
"wan2.2-a14b": "237d74973e0a",
"hunyuanvideo": "1f529f8bd192",
"ltx-video": "b76914968ee6",
"mochi-1": "d134d7d69b76",
"opensora-stdit3": "d7ec0d162072",
"minimax-h3": "b786010ff458",
"esm2-3b": "34f7544287cd",
"esm2-650m": "1956cfc2e36f",
"esmfold": "7ad8b1001045",
}

CLOSED = ("gpt-oss-120b", "gpt-oss-20b", "deepseek-v3.2", "glm-5", "glm-5.2", "glm-5.3", "kimi-k3", "qwen3.8-2.4t",
          "qwen3.8-27b", "qwen3.5-397b-a17b", "qwen3-next-80b-a3b", "minimax-text-01", "minimax-m1-80k",
          "glm-5.3-flash", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4.1-flash",
          "qwen3.8-flash-next")
PROXY = ()
PROTEIN_PARTIAL = ("openfold", "alphafold2", "boltz-1", "protenix")


def _kpi(i):
    s = []
    for sv in _SV:
        try:
            r = evaluate(Scenario(model=i, serving=sv))
            s.append([float(f"{x:.10g}") for x in (r.tpot, r.ttft, r.throughput, r.per_card, r.tick)] + [r.fits, r.bound])
        except Exception as x:  # noqa: BLE001
            s.append(str(x)[:40])
    return hashlib.sha1(json.dumps(s).encode()).hexdigest()[:12]


def test_full_models_unchanged():
    bad = [i for i, h in _FULL_0651.items() if _kpi(i) != h]
    assert not bad, bad


def test_coverage_list():
    cov = {e["id"]: get_model(e["id"]).coverage for e in entries()}
    assert all(cov[i] == "full" and not get_model(i).coverage_reasons for i in CLOSED)
    assert all(cov[i] == "proxy" for i in PROXY)
    assert all(cov[i] == "partial" for i in PROTEIN_PARTIAL)
    assert sorted(i for i, c in cov.items() if c != "full") == sorted(PROXY + PROTEIN_PARTIAL)
    assert [o["id"] for o in offline_entries()] == ["alphafold3"]


def test_glm_shared_indexer_layers():
    """GLM-5.2 / 5.3 indexer_types: 57 'shared' layers reuse the previous full layer's top-k (no indexer weights,
    transformers GlmMoeDsa) — the template now matches the release safetensors exactly (was +0.07 %)."""
    for i in ("glm-5.2", "glm-5.3"):
        m = get_model(i)
        assert m.param_check()["rel_err"] == 0.0
        assert sum(1 for l in m.layers if l.core.idx_heads) == 21
        assert all(l.core.topk == 2048 for l in m.layers)


def test_indexer_cache_bytes():
    ds = get_model("deepseek-v3.2").layers[0].core
    assert ds.idx_fp8 and ds.idx_bytes_per_token(2) == 128 + 4      # fp8 key + one fp32 scale per 128
    g5 = get_model("glm-5").layers[0].core
    assert not g5.idx_fp8 and g5.idx_bytes_per_token(2) == 256


def test_gpt_oss_sink_softmax_row():
    from dataclasses import replace
    m = get_model("gpt-oss-20b")
    ph = Phase("decode", batch=4, q=1, ctx=4096)
    a = {o.name: o for o in build_rank_ops(m, 1, ph)}
    m0 = replace(m, layers=tuple(replace(l, core=replace(l.core, sink=False)) for l in m.layers))
    b = {o.name: o for o in build_rank_ops(m0, 1, ph)}
    c = m.layers[1].core
    rows = 4 * c.n_q                                    # batch × query heads, one extra logit each
    assert math.isclose(a["softmax"].vec - b["softmax"].vec, 5 * rows)


def test_gdn_chunk_flops_reference():
    """Qwen3-Next prefill: chunked Gated DeltaNet GEMM FLOPs = transformers torch_chunk_gated_delta_rule per chunk
    (C = 64): kβKᵀ C²dk + QKᵀ C²dk + UT solve ½C²(dv+dk) + k_cumdecay·S, Q·S, Kᵀ·v_new 3·C·dk·dv + A·v_new C²dv."""
    m = get_model("qwen3-next-80b-a3b")
    li = next(i for i, l in enumerate(m.layers) if l.core.kind == "linear")
    c = m.layers[li].core
    q = 8192
    ops = [o for o in build_rank_ops(m, li, Phase("prefill", batch=1, q=q, ctx=0)) if o.name.startswith("lin_")]
    C, dk, dv, H = 64, c.state_dk, c.state_dv, c.n_state_heads
    macs = (2 * C * C * dk + 0.5 * C * C * (dv + dk) + 3 * C * dk * dv + C * C * dv) * (q // C) * H
    assert math.isclose(sum(o.flops for o in ops), 2 * macs, rel_tol=1e-9)


def test_lightning_block_256():
    c = next(l.core for l in get_model("minimax-m1-80k").layers if l.core.kind == "linear")
    assert c.lin == "lightning" and c.lin_chunk == 256


def test_hc_and_v4_params_match_release_layers():
    """GLM-5.3-Flash (transformers glm5_next) and DeepSeek-V4-Flash / -Pro (official inference/model.py): the only
    difference to the release summary is the hc_*_scale tensors (3 per hyper-connection, + 1 for the V4 head), which
    summarize_release files under quantisation scales."""
    for i, extra in (("glm-5.3-flash", 45 * 6), ("deepseek-v4-flash", 43 * 6 + 1), ("deepseek-v4-pro", 61 * 6 + 1),
                     ("deepseek-v4.1-flash", 40 * 6)):
        pc = get_model(i).param_check()
        assert pc["ours"] - pc["release"] == extra and pc["mtp_ours"] == pc["mtp_release"], (i, pc)


def test_hc_ops_reference_counts():
    m = get_model("glm-5.3-flash")
    ops = {o.name: o for o in build_rank_ops(m, 0, Phase("prefill", batch=1, q=8192, ctx=0))}
    N, h = 4, m.hidden
    assert ops["attn_hc_fn"].flops == 2 * 8192 * N * h * (2 + N) * N     # fn GEMM [N·h → (2+N)·N]
    assert "ffn_hc_mix" in ops and m.layers[0].hc_iters == 20


def test_v4_attention_keys_reference():
    """V4 ratio-0 layer: window 128 only; ratio-128 layer: window + every compressed entry; ratio-4: window + top-k."""
    m = get_model("deepseek-v4-flash")
    c0, c4, c128 = (m.layers[i].core for i in (0, 2, 3))
    assert (c0.compress, c4.compress, c128.compress) == (1, 4, 128) and c4.idx_heads == 64 and not c128.idx_heads
    ctx = 65536
    assert c0.ctx_eff(ctx) == 128 and c128.ctx_eff(ctx) == 128 + 512 and c4.ctx_eff(ctx) == 128 + 512
    assert c0.sink and c4.cmp_coff == 2 and c128.cmp_coff == 1


def test_v41_reference_structure():
    """DeepSeek-V4.1-Flash (official inference/model.py + engram.py): window + top-k compressed keys
    (min(p, w) + min(k, ⌊p/r⌋)), compressed cache / index keys only on the kv sources, indexer only on index sources,
    candidate blocks from layer 20, Engram fp8 + e8m0 tables, DSpark draft."""
    m = get_model("deepseek-v4.1-flash")
    c = [l.core for l in m.layers]
    assert [i for i, x in enumerate(c) if x.kv_owner] == [2, 8, 14, 20]
    assert [i for i, x in enumerate(c) if x.idx_heads] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert c[20].cand == "src" and [i for i, x in enumerate(c) if x.cand == "use"] == [24, 28, 32, 36]
    for x, r in ((c[3], 2), (c[25], 1)):
        assert x.dual and x.compress == r
        assert x.keys_sum(3000) == sum(min(p, 128) + min(512, p // r) for p in range(1, 3001))
    assert not c[0].dual and c[0].ctx_eff(10 ** 5) == 128
    assert [i for i, l in enumerate(m.layers) if l.engram_cols] == [1, 14] and m.layers[1].engram_cols == 24
    assert m.lookup_params == 384006168 * 256 + 384016682 * 256 and m.lookup_bits == 8.25
    assert m.draft == "dspark" and m.draft_block == 5 and len(m.mtp_layers) == 3
    ops = {o.name: o for o in build_rank_ops(m, 1, Phase("decode", 1, 1, 4096))}
    assert ops["engram_lookup"].kv_read == 24 * (256 + 8)          # one token: 24 fp8 rows + scales
    assert ops["attn.engram_wkv"].flops == 2 * 24 * 256 * 5 * 5120


def test_qwen4exp_params_and_ops_reference():
    """Qwen3.8-Flash-Next (transformers modeling_qwen4_exp + SGLang qwen4_exp_mtp.py): params equal the safetensors
    summary exactly (no final norm; gated residuals, PLE, QSA indexer, final mixer); QSA keys per query; op counts."""
    m = get_model("qwen3.8-flash-next")
    pc = m.param_check()
    assert pc["ours"] == pc["release"] and pc["mtp_ours"] == pc["mtp_release"], pc
    assert m.coverage == "full" and not m.coverage_reasons and m.final_norm_params == 0
    assert m.lookup_params == 128 * 2500012 * 160 and [i for i, l in enumerate(m.layers) if l.ple_rows] == [1]
    c = m.layers[3].core
    assert c.qsa == 4 and c.topk == 2048 and c.ctx_eff(2051) == 2051 and c.ctx_eff(2053) == 2048 + 1
    assert c.keys_sum(5000) == sum(p if p // 4 <= 512 else 2048 + p % 4 for p in range(1, 5001))
    t, N, h = 8192, 4, m.hidden
    ops = {o.name: o for o in build_rank_ops(m, 1, Phase("prefill", batch=1, q=t, ctx=0))}
    assert ops["attn.gr_a_down"].flops == 2 * t * N * h * 320 and ops["attn.gr_a_inj"].flops == 2 * t * N * h * N
    assert ops["attn.ple_key"].flops == 2 * t * 2560 * N * h
    assert ops["ple_lookup"].kv_read == t * 16 * 160 * 2
    assert "attn_norm" not in ops and "ffn_norm" not in ops
    from accel_dse.core.ir import mtp_ops
    mo = {o.name: o for o in mtp_ops(m, Phase("decode", 1, 1, 1000))}
    assert mo["mtp.fc_hidden"].flops == 2 * N * h * h and mo["mtp.fc_embedding"].flops == 2 * h * h
