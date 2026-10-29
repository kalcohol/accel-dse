"""0.125 hostile review: grouped-linear GEMM shapes (DeepSeek-V4 o_a), per-family weight-read conservation,
stored weights vs release bytes, batch / layout search exactness for prefill and video (bound units)."""
from accel_dse.core import catalog as C
from accel_dse.core import ir
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard
from accel_dse.core.memplan import _layer_storage
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario
from accel_dse.core.search import _feasible, best_batch, brute_force_batch, search_layouts


def _ops(m, ph, sh=Shard()):
    o = []
    for li in range(len(m.layers)):
        o += ir.build_rank_ops(m, li, ph, sh)
    return o + ir.embed_ops(m, ph, sh) + ir.head_ops(m, ph, sh)


def test_grouped_linear_shapes():
    m = C.get_model("deepseek-v4-flash")
    L = m.layers[0]
    oa = next(l for l in L.attn_linears if l.name == "o_a")
    assert oa.groups == 8
    for tp in (1, 2, 4, 8, 16):
        sh = Shard(tp=tp, ep=tp)
        op = next(o for o in ir.build_rank_ops(m, 0, Phase("decode", 4, 1, 64), sh) if o.name == "attn.o_a")
        assert op.k * op.n * op.count == op.w_params
        assert op.w_params * tp == oa.params, (tp, op.w_params, oa.params)      # conserved, no ×groups
        assert op.k == oa.k // oa.groups
    # storage plan sees the same bytes as the parameter count
    hot1 = _layer_storage(m, L, Shard())[0]
    hot8 = _layer_storage(m, L, Shard(tp=8, ep=8))[0]
    assert hot1 > hot8 > 0


def test_llm_weight_reads_match_active_params():
    """Batch-1 decode reads every active weight once: Σ GEMM weights + embedding table = active params (tied heads
    read the table twice); misc vectors (norms, biases, sinks) ≤ 1 %."""
    for e in C.list_models():
        if not e["evaluable"] or e["model_domain"] not in (None, "llm"):
            continue
        if e["domain"] == "tts":     # 0.269: TTS head re-reads the flow DiT per Euler step; conserved in test_core_269
            continue
        m = C.get_model(e["id"])
        ops = _ops(m, Phase("decode", 1, 1, 64))
        for o in ops:
            if o.w_params and o.kind == "gemm":
                assert o.k * o.n * o.count == o.w_params, (e["id"], o.name)
        wp = sum(o.w_params for o in ops if o.kind != "lookup")
        tied = m.embed_params if m.head_params == 0 else 0
        # 0.253 looped LMs (Dory): every later loop re-reads the shared recurrent weights (stored once)
        from dataclasses import replace as _rp
        reread = sum(_rp(L, loop=0).params(m.hidden) for L in m.layers if L.loop)
        # 0.270 encoder-decoder (Whisper): cross-attention k / v run once per request at prefill on the encoder output
        # (cached), never in decode; 0.358: params() includes encoder, so subtract it from active_params comparison
        xkv = sum(l.params for L in m.layers for l in L.cross_linears if l.name in ("k", "v")) if m.enc_ctx else 0
        enc = m.vision_params if m.enc_ctx else 0  # 0.358: encoder not read in decode
        r = (wp - tied + m.embed_params) / (m.active_params() + reread - xkv - enc) - 1
        assert -0.01 < r < 1e-9, (e["id"], r)


def test_stored_weights_match_release_bytes():
    for mid in ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4.1-flash", "qwen3-8b", "glm-4.5-air"):
        m = C.get_model(mid)
        r = evaluate(Scenario(model=mid))
        assert abs(sum(s.mem.stored_w for s in r.stages) / m.release_bytes - 1) < 0.01, mid


def test_batch_search_exact_prefill_and_video():
    s = Scenario(model="qwen3-8b").replace("serving.phase", "prefill").replace("serving.prompt", 2048)
    bb, (bf, val) = best_batch(s, 128), brute_force_batch(s, 128)
    assert bb.batch == bf and abs(bb.result.throughput - val) <= 1e-12 * val
    v = Scenario(model="wan2.1-1.3b").replace("workload.te_cpu", True).replace("workload.overlap", True)
    bb, (bf, val) = best_batch(v, 8), brute_force_batch(v, 8)
    assert bb.batch == bf and abs(bb.result.throughput - val) <= 1e-12 * val


def test_layout_search_never_worse_than_known_point():
    base = Scenario(model="qwen3-8b").replace("serving.phase", "prefill").replace("serving.prompt", 1024)
    rows = search_layouts(base, 4, top=1)
    assert rows and rows[0].result.fits and rows[0].result.slo_ok
    best = rows[0].per_card
    for lay in (Layout(1, 4, 1, 1, 1), Layout(4, 1, 1, 1, 1), Layout(2, 2, 1, 1, 1), Layout(1, 1, 1, 1, 1)):
        s = base.replace("layout", lay)
        b, thr = brute_force_batch(s, 48)
        if b:
            assert best >= thr / 4 * (1 - 1e-12), (lay, best, thr / 4)


def test_fingerprints_change_only_v4():
    """0.125 rebaseline (e2176af2610c144c stage / 392692e6937f69ef default; 0.141: 0a954915c5fb003f / 53ee9e08165795c6, were c034bfe3e3a3afa2 / e51c7ec17bbdb998):
    the grouped-linear fix moves only the DeepSeek-V4 rows — without them the fingerprint equals the 0.124 one."""
    import hashlib
    import json
    from accel_dse.core.catalog import entries

    def fpx(top):
        h = hashlib.sha256()
        for e in entries()[:60]:
            if e["id"].startswith("deepseek-v4"):
                continue
            for extra in ({}, {"chip": {"slc_mib": 256}}, {"chip": {"slc_mib": 256, "slc_policy": "lru"}}, {"layout": {"tp": 2}}):
                try:
                    h.update(json.dumps(evaluate(Scenario.from_dict({"model": e["id"], **top, **extra})).summary(),
                                        sort_keys=True, default=str).encode())
                except Exception as ex:  # noqa: BLE001
                    h.update(type(ex).__name__.encode())
        return h.hexdigest()[:16]
    # 0.141 rebaseline (were d29ef7bf6936030d / c4e341987bf2461f)
    # 0.173 rebaseline (were 4690f724dabe9fa6 / ea2456afee7c3a86): split-K / wide GEMM plans also move non-v4 MoE / hc models
    # 0.198 rebaseline (were 45d9dbb41d0502ba / c68f3a7cb9ca4b7b): fused QKV
    # 0.199 rebaseline (were 4aa21d67471fd69e / a3548246a90a9ceb): NCCL all-reduce latency
    # 0.200 rebaseline (were 3e753f36c5bcdb36 / fefb5b23fdf2782c): custom all-reduce α
    # 0.201 rebaseline (were ba49c208f61b306c / 9bb4812eed9b116a): GPU all-reduce latency no longer on NPU links (= 0.198 values)
    # 0.202 rebaseline (were 4aa21d67471fd69e / a3548246a90a9ceb): WIDE_WAVES refill GPU-only (GPU_WAVES_ONLY)
    assert fpx({"exec_overlap": "stage"}) == "5c058c9ed0f8f61e" and fpx({}) == "f8346734e98af4a2"   # 0.473 re-baseline (Seed-OSS SwiGLU product; pipe was eb4847bbe1543d12, stage unchanged) ← 0.472 re-baseline (Mixtral expert mul; pipe was 0f2c94a6e5524bc5, stage unchanged) ← 0.468 re-baseline (Qwen3.5 MoE expert mul; pipe was 903976206252d461, stage unchanged) ← 0.467 re-baseline (Qwen3 head width; pipe was 666d0c5f91fbb6a4, stage unchanged) ← 0.466 re-baseline (Qwen2.5-3B and 1.5B RMSNorm; pipe was a1484d51a9ad1ed3, stage unchanged) ← 0.462 re-baseline (Mistral-Large RMSNorm; pipe was 8a5fa1d0881da387, stage unchanged) ← 0.437 re-baseline (umT5 RMSNorm; was dee4a1703e8fab85 / 9982d6b04e230874) ← 0.210 re-baseline (pipe KV/state split; fpx({}) was a148b9cd855d95fc) ← 0.209 re-baseline (STATE_SPLIT; fpx({}) was 8220364829b88de3; was 5b3efea75e12245d)
