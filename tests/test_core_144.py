"""0.144: catalog refresh (Aug–Oct 2026 open-weight releases) — Step-5-Preview (new step5 builder: SWA + DSA region-
sparse full layers, head-wise gate, 352-expert MoE, MTP ×3, FP8 experts), MiMo-V2.6-Flash (new mimo_v2 builder: SWA-128
with sinks + full GQA of different KV geometry, MXFP4 experts), Intern-S2-397B (Qwen3.5-MoE hybrid), MiniCPM-V-4.7-35B-A3B
(Qwen3.5-MoE text, no MTP weights), LingBot-Video MoE-DMD 30B-A3B (new MoE DiT builder) — exact parameter counts vs the
safetensors headers, as-released dtypes."""
from accel_dse import api as A
from accel_dse.core.catalog import get_model, list_models


def test_step5_exact_params_and_structure():
    m = get_model("step-5-preview")
    assert m.params() == m.release_params == 598_476_053_952 and m.coverage == "partial"   # vision: overview only
    assert m.params(include_mtp=True) - m.params() == m.release_mtp_params == 2_515_858_560
    swa = [l for l in m.layers if l.core.window == 512]
    sp = [l.core for l in m.layers if l.core.idx_heads]
    assert len(m.layers) == 92 and len(swa) == 69 and len(sp) == 23 and len(m.mtp_layers) == 3
    c = sp[0]
    assert (c.qsa, c.topk, c.idx_heads, c.idx_dim, c.idx_sum, c.n_kv, c.qk_dim) == (8, 4096, 16, 256, True, 4, 192)
    assert c.ctx_eff(1 << 20) == 4096 + (1 << 20) % 8 and c.ctx_eff(3000) == 3000
    assert c.idx_bytes_per_token(2) == 256 * 5 / 8        # fp32 + fp8 summary per completed 8-token region
    moe = [l.ffn for l in m.layers if l.ffn.kind == "moe"]
    assert len(moe) == 88 and moe[0].n_experts == 352 and moe[0].top_k == 8 and moe[0].d_shared == 1536
    assert any(l.name == "g_head" and l.n == 64 for l in m.layers[0].attn_linears)
    f = dict(m.formats)
    assert f["expert"].fmt == "fp8" and f["attn"].fmt == "bf16" and m.act_fmt == "fp8"


def test_mimo_v26_exact_params_and_structure():
    m = get_model("mimo-v2.6-flash")
    assert m.params() == m.release_params == 308_778_780_864 and m.coverage == "full"   # 0.145: ViT band windows
    assert m.params(include_mtp=True) - m.params() == m.release_mtp_params == 987_820_224
    swa = [l.core for l in m.layers if l.core.window == 128]
    full = [l.core for l in m.layers if l.core.window is None]
    assert len(swa) == 39 and len(full) == 9 and all(c.sink for c in swa) and not any(c.sink for c in full)
    assert (swa[0].n_kv, full[0].n_kv, full[0].qk_dim, full[0].v_dim) == (8, 4, 192, 128)
    assert m.layers[0].ffn.kind == "dense" and sum(l.ffn.kind == "moe" for l in m.layers) == 47
    f = dict(m.formats)
    assert f["expert"].fmt == "mxfp4" and f["attn"].fmt == "fp8" and m.vision_params > 0


def test_qwen35_family_vlms_exact():
    m = get_model("intern-s2-397b")
    assert m.params() == m.release_params == 396_371_516_160
    assert m.params(include_mtp=True) - m.params() == m.release_mtp_params == 6_595_568_128
    assert sum(l.core.kind == "linear" for l in m.layers) == 45
    m = get_model("minicpm-v-4.7-35b-a3b")
    assert m.params() == m.release_params == 34_659_889_792      # vision merger counted as vision (0.144 role fix)
    assert not m.mtp_layers                                      # config MTP, no MTP weights in the release


def test_lingbot_moe_dit_exact_and_routed():
    from accel_dse.core.ir import Phase, Shard, _full_layer_ops
    m = get_model("lingbot-video-moe-dmd")
    assert m.params() == m.release_params == 30_084_512_320 and m.domain == "gen"
    assert all(l.ffn.kind == "moe" and l.post_norm and l.cross is None for l in m.layers) and len(m.layers) == 48
    assert m.workload.steps == 8 and m.workload.cfg == 1 and m.workload.frames == 121
    ops = {o.name: o for o in _full_layer_ops(m, 0, m.layers[0], Phase("full", 1, 8192, 0), Shard())}
    e = ops["expert.gate_up"]
    assert (e.m, e.count, e.n) == (8192 * 8 // 128, 128, 2 * 768) and "post_norm" in ops and "moe.router" in ops
    assert dict(m.formats)["expert"].fmt == "bf16" and dict(m.formats)["router"].fmt == "fp32"


def test_new_models_listed_and_evaluate():
    ids = {d["id"]: d for d in list_models()}
    want = {"step-5-preview": "Step-5", "mimo-v2.6-flash": "MiMo-V2.6", "intern-s2-397b": "Intern-S2",
            "minicpm-v-4.7-35b-a3b": "MiniCPM-V 4.7", "lingbot-video-moe-dmd": "LingBot-Video"}
    for mid, fam in want.items():
        assert ids[mid]["family"] == fam and ids[mid]["provenance"] == "official"
    for mid, lay in (("step-5-preview", {"tp": 8, "ep": 8}), ("mimo-v2.6-flash", {"tp": 8, "ep": 8}),
                     ("intern-s2-397b", {"tp": 8, "ep": 8}), ("minicpm-v-4.7-35b-a3b", {"tp": 1})):
        for ph in ("decode", "prefill"):
            s = A.api_eval({"scenario": {"model": mid, "layout": lay, "serving": {"phase": ph, "batch": 8}},
                            "sensitivity": False})["summary"]
            assert s["step_ms"] > 0 and s["tok_s"] > 0
    for lay in ({"tp": 1}, {"tp": 2, "sp": 4}):
        s = A.api_eval({"scenario": {"model": "lingbot-video-moe-dmd", "layout": lay}, "sensitivity": False})["summary"]
        assert s["step_ms"] > 0 and s["tok_s"] > 0


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "cd25d689b7492f42"   # 0.209 re-baseline (STATE_SPLIT; was b47ec84b00fc0892) ← 0.202 re-baseline (GPU_WAVES_ONLY; was 6dca732e67c83a75) ← 0.201 re-baseline (GPU_AR_ONLY, = 0.198 value; was be3e00dcc13fcdda) ← 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "99a72aef99a2e674"   # 0.210 re-baseline (pipe KV/state split; was 62fd08857bc4c055) ← 0.209 re-baseline (STATE_SPLIT; was b4db8aea9e5680b6) ← 0.202 re-baseline (GPU_WAVES_ONLY; was bc06528a82e19e2a) ← 0.201 re-baseline (GPU_AR_ONLY; was 7d5cdb465b922571) ← 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)


def test_new_vision_encoders_exact_and_tokens():
    """Vision towers + projectors from the released headers: params exact (MiMo: + audio encoder stored only)."""
    from accel_dse.core.vision import image_grid, vision_ops
    # 0.145: MiniCPM 1024×768 = 448²-class overview + 2×2 slices, 63 tokens each + 14 special (was one 448² view, 64)
    want = {"intern-s2-397b": (456_010_480, 768), "step-5-preview": (1_981_014_784, 169),
            "minicpm-v-4.7-35b-a3b": (552_986_032, 329), "mimo-v2.6-flash": (989_721_600, 768)}
    for mid, (p, tok) in want.items():
        m = get_model(mid)
        assert m.vision.params == m.vision_params == p, mid
        assert image_grid(m.vision, 1024, 768).llm_tokens == tok, mid
    v = get_model("mimo-v2.6-flash").vision
    assert v.extra == 261_127_168 and v.kv_heads == 8
    qkv = next(o for o in vision_ops(v, image_grid(v, 1024, 768), 1) if o.name == "vis.0.qkv")
    assert qkv.n == (32 + 2 * 8) * 64
    v = get_model("minicpm-v-4.7-35b-a3b").vision
    ops = {o.name: o for o in vision_ops(v, image_grid(v, 448, 448), 1)}
    assert ops["vis.5.qkv"].m == 1024 and ops["vis.6.qkv"].m == 256 and ops["vis.merger.1"].m == 64
    ops = {o.name: o for o in vision_ops(get_model("step-5-preview").vision, image_grid(get_model("step-5-preview").vision, 1, 1), 1)}
    assert (ops["vis.merger.0"].m, ops["vis.merger.1"].m, ops["vis.merger.2"].m) == (676, 169, 169)
