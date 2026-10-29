"""0.116: Qwen-Image-2.1 text-prefix K/V reuse (model-declared, workload.prefix_kv), image text-encoder + VAE pipelines
(coverage full), image-mode audit (resolution rule, CFG, steps), fingerprints."""
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.pipeline import pipeline_for, vae_ops, Vae
from accel_dse.core.scenario import Scenario

IM = ("qwen-image-2.1-turbo", "qwen-image", "flux.2-klein-4b")


def _g(sc):
    return evaluate(Scenario.from_dict(sc)).summary()["gen"]


def test_prefix_kv_declared_and_closed_form():
    m = get_model("qwen-image-2.1-turbo")
    assert m.workload.cond_causal and m.workload.prefix_kv
    assert not any(get_model(i).workload.prefix_kv for i in ("qwen-image", "flux.2-klein-4b", "wan2.1-14b"))
    g = _g({"model": "qwen-image-2.1-turbo"})
    pk = g["prefix_kv"]
    N, T, L, d = 128 * 128, 256, 32, 4096
    l0 = m.layers[0]
    tok = sum(l.k * l.n for l in l0.attn_linears + l0.ffn_linears)
    io = 2 * N * sum(l.k * l.n for l in m.io_pre + m.io_post if l.rows == "img")
    q = N + T
    cached = 2 * L * N * tok + 4 * L * N * q * d + io                       # image rows; keys = cached prefix + image
    extract = 2 * L * q * tok + 4 * L * (N * q + T * (T + 1) / 2) * d + io  # block-causal mask
    assert abs(pk["cached_tflop_seq"] * 1e12 / cached - 1) < 2e-3
    assert abs(pk["extract_tflop_seq"] * 1e12 / extract - 1) < 2e-3
    assert pk["cache_GiB_card"] * 2**30 == L * T * 32 * (128 + 128) * 2
    assert abs(g["denoise_s"] - (pk["extract_ms"] + 7 * pk["cached_ms"]) / 1e3) < 1e-9
    off = _g({"model": "qwen-image-2.1-turbo", "workload": {"prefix_kv": "off"}})
    assert "prefix_kv" not in off and off["denoise_s"] > g["denoise_s"]
    assert off["dit_tflop_per_request"] > g["dit_tflop_per_request"]
    assert "prefix_kv" not in _g({"model": "qwen-image-2.1-turbo", "workload": {"steps": 1}})


def test_prefix_kv_on_needs_causal_condition():
    r = evaluate(Scenario.from_dict({"model": "qwen-image", "workload": {"prefix_kv": "on"}}))
    assert any("prefix_kv = on 已忽略" in w for w in r.warnings) and r.prefix_kv is None
    s = Scenario.from_dict({"model": "qwen-image-2.1-turbo"})
    assert "prefix_kv" not in s.to_dict()["workload"]                    # omit-when-default: hashes unchanged
    try:
        Scenario.from_dict({"model": "qwen-image", "workload": {"prefix_kv": "yes"}})
        raise AssertionError
    except ValueError:
        pass


def test_prefix_kv_conserved_under_tp_sp_pp():
    base = _g({"model": "qwen-image-2.1-turbo", "serving": {"batch": 2}})
    for lay in ({"tp": 2}, {"sp": 2}, {"tp": 2, "sp": 2}, {"pp": 2}, {"tp": 3}, {"dp": 2}):
        y = _g({"model": "qwen-image-2.1-turbo", "serving": {"batch": 2}, "layout": lay})
        assert abs(y["dit_tflop_per_request"] / base["dit_tflop_per_request"] - 1) < 1e-6, lay
        assert "prefix_kv" in y


def test_image_pipelines_full_coverage():
    for mid in IM:
        m = get_model(mid)
        assert m.coverage == "full" and not m.coverage_reasons and pipeline_for(mid) is not None
        g = _g({"model": mid})
        p = g["pipeline"]
        assert p["te_s"] > 0 and p["decode_s"] > 0 and g["latency_s"] > g["denoise_s"]
        assert _g({"model": mid, "workload": {"pipeline": False}})["pipeline"] is None
    # FLUX.2: ×8 VAE latent with the 2×2 packing (same 64×64 tokens as 0.115)
    w = _g({"model": "flux.2-klein-4b"})["workload"]
    assert w["latent"] == [1, 128, 128] and w["grid"] == [1, 64, 64]


def test_time_conv_skipped_on_single_latent_frame():
    v = Vae("qwen-image-vae", "wan", "x")
    o1, _, _ = vae_ops(v, (1, 64, 64), 1, 1, "bf16")
    o2, _, _ = vae_ops(v, (2, 64, 64), 5, 1, "bf16")
    assert not any("time_conv" in o.name for o in o1) and any("time_conv" in o.name for o in o2)


def test_image_resolution_rule_cfg_steps():
    for mid, s, p in (("qwen-image-2.1-turbo", 16, 1), ("qwen-image", 8, 2), ("flux.2-klein-4b", 8, 2)):
        for H, W in ((1000, 1500), (928, 1664), (2048, 2048)):
            w = _g({"model": mid, "workload": {"height": H, "width": W, "pipeline": False}})["workload"]
            lh, lw = 2 * (H // (2 * s)), 2 * (W // (2 * s))
            assert w["latent"] == [1, lh, lw] and w["grid"] == [1, lh // p, lw // p], (mid, H, W)
        a = _g({"model": mid, "workload": {"cfg": 1}})
        b = _g({"model": mid, "workload": {"cfg": 2}})
        assert abs(b["dit_tflop_per_request"] / a["dit_tflop_per_request"] - 2) < 1e-9
        d = {st: _g({"model": mid, "workload": {"steps": st, "prefix_kv": "off"}})["denoise_s"] for st in (1, 4, 20)}
        assert all(abs(d[st] / d[1] / st - 1) < 1e-9 for st in d)


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "f54f33dfa13cdd41"   # 0.437 re-baseline (umT5 RMSNorm; was cd25d689b7492f42) ← 0.209 re-baseline (STATE_SPLIT; was b47ec84b00fc0892) ← 0.202 re-baseline (GPU_WAVES_ONLY; was 6dca732e67c83a75) ← 0.201 re-baseline (GPU_AR_ONLY, = 0.198 value; was be3e00dcc13fcdda) ← 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "25e9cd50ab57455e"   # 0.464 re-baseline (Llama-2 and Yi-1.5 RMSNorm; was a9cc980add47b869) ← 0.462 re-baseline (Mistral-Large RMSNorm; was 1e15ae85430a04f3) ← 0.437 re-baseline (umT5 RMSNorm; was 99a72aef99a2e674) ← 0.210 re-baseline (pipe KV/state split; was 62fd08857bc4c055) ← 0.209 re-baseline (STATE_SPLIT; was b4db8aea9e5680b6) ← 0.202 re-baseline (GPU_WAVES_ONLY; was bc06528a82e19e2a) ← 0.201 re-baseline (GPU_AR_ONLY; was 7d5cdb465b922571) ← 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)
