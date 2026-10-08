"""0.42 — the remaining video-generation DiTs on core v2: Wan2.2 A14B (two noise experts), HunyuanVideo (dual- +
single-stream MMDiT), LTX-Video, Mochi-1 (asymmetric dual-stream), Open-Sora STDiT3 (factorized spatial / temporal
attention, as released) and MiniMax-H3 (single-stream omni transformer with joint audio rows, AdaLN cached).
Exact parameters from the safetensors headers, workload → tokens, FLOPs vs closed forms, stream-specific rows,
standby / cached weights, V0 invariants, SP conservation with joint audio rows, API round trip."""
from __future__ import annotations

import math

from accel_dse import api
from accel_dse.core.catalog import get_model, labels
from accel_dse.core.domain import resolve_workload
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops, full_io_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Workload

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"
NEW = ("wan2.2-a14b", "hunyuanvideo", "ltx-video", "mochi-1", "opensora-stdit3", "minimax-h3")


def _flops(mid: str, **wl) -> float:
    r = evaluate(Scenario(model=mid, mem_id=HBM, workload=Workload(**wl)))
    return r.domain_summary()["tflop_per_request"] * 1e12


def test_params_exact_dtype_and_coverage():
    for mid in NEW:
        lb = labels(get_model(mid))
        assert abs(lb["param_err"]) < 2e-5, (mid, lb["param_err"])             # every header tensor accounted
        assert lb["coverage"] == "partial" and lb["model_domain"] == "gen" and not lb["kv_cache"]
        assert any("VAE" in r for r in lb["coverage_reasons"]), mid
    assert labels(get_model("wan2.2-a14b"))["dtype"] == "W fp32 · A bf16"
    assert labels(get_model("hunyuanvideo"))["dtype"] == "W bf16 · A bf16"
    assert labels(get_model("minimax-h3"))["dtype"] == "W bf16 · A bf16"
    assert get_model("minimax-h3").fmt("io").fmt == "fp32"                     # mixed release: fp32 io kept
    for mid in ("ltx-video", "mochi-1", "opensora-stdit3"):
        assert labels(get_model(mid))["dtype"].startswith("W fp32"), mid
    assert [l.core.span for l in get_model("opensora-stdit3").layers[:2]] == ["spatial", "temporal"]
    hy = get_model("hunyuanvideo")
    assert sum(1 for l in hy.layers if l.fused_out) == 40 and hy.n_layers == 60
    mo = get_model("mochi-1")
    assert mo.layers[-1] != mo.layers[0] and len(mo.layers[-1].ffn_linears) == 2   # last block: context_pre_only


def test_workload_tokens_vae_rules_and_audio():
    W = Workload()
    tok = {mid: resolve_workload(get_model(mid), W) for mid in NEW}
    assert tok["wan2.2-a14b"].tokens == 21 * 45 * 80 and tok["wan2.2-a14b"].steps == 40
    assert tok["hunyuanvideo"].tokens == 33 * 45 * 80 + 256 and tok["hunyuanvideo"].seqs_per_request == 1
    assert tok["ltx-video"].tokens == 21 * 16 * 22                                 # 32×32×8 VAE, patch 1
    assert tok["mochi-1"].tokens == 28 * 30 * 53 + 256 and tok["mochi-1"].steps == 64
    st = tok["opensora-stdit3"]
    assert st.tokens == 30 * 45 * 80 and st.frames == 30 and st.info["attention"] == "factorized"   # 6 × (17 → 5)
    h3 = tok["minimax-h3"]
    assert h3.aux == round(124 / 24 * 40) * 2 == 414 and h3.frames == 37                         # 17·7+5 → 5·7+2
    assert h3.tokens == 37 * 24 * 42 + 512 + 414 and h3.steps == 49 and h3.seqs_per_request == 1
    w = resolve_workload(get_model("minimax-h3"), Workload(frames=121))
    assert w.frames == 37 and any("17n+5" in x for x in w.warnings)                              # snapped up to 124
    w = resolve_workload(get_model("opensora-stdit3"), Workload(frames=51))
    assert w.frames == 15 and not any("17" in x for x in w.warnings)                             # 51 = 3 × 17


def test_flops_vs_closed_forms():
    # STDiT3: factorized attention — spatial groups of H·W tokens, temporal groups of T tokens
    d, f, L, nc = 1152, 4608, 28, 300
    T, HW = 30, 45 * 80
    N = T * HW
    gemm = 2 * N * (4 * d * d + 2 * d * d + 2 * d * f) + 2 * nc * 2 * d * d
    per = 2 * L * gemm + L * 4 * N * HW * d + L * 4 * N * T * d + 2 * L * 4 * N * nc * d
    got = _flops("opensora-stdit3")
    assert abs(got / (per * 2 * 30) - 1) < 0.01, got / (per * 60)
    full3d = 2 * L * 4 * N * N * d * 2 * 30                                     # the same model with full 3D attention
    assert full3d > 5 * got                                                       # factorization is the point of STDiT
    # HunyuanVideo: 20 dual + 40 single blocks, all with (4d² + 2df) GEMM work per token and joint attention
    d, f, N = 3072, 12288, 33 * 45 * 80 + 256
    per = 60 * (2 * N * (4 * d * d + 2 * d * f) + 4 * N * N * d)
    got = _flops("hunyuanvideo")
    assert abs(got / (per * 50) - 1) < 0.01, got / (per * 50)
    # MiniMax-H3: heads·head_dim (7168) > hidden (5376); SwiGLU 14336; 49 forwards, no CFG
    d, hq, f, N = 5376, 7168, 14336, 37 * 24 * 42 + 512 + 414
    per = 50 * (2 * N * (4 * d * hq + 3 * d * f) + 4 * N * N * hq)
    got = _flops("minimax-h3")
    assert abs(got / (per * 49) - 1) < 0.01, got / (per * 49)


def test_v2_opensora_readme_h100():
    """Open-Sora 1.2 README: 720p 4 s on one H100 takes 130 s end to end (T5 + VAE included).  Our DiT FLOPs over
    130 s must imply a plausible H100 utilisation (5–60 % of 989 TFLOPS); full 3D attention would imply > 100 %."""
    eff = _flops("opensora-stdit3") / 130.0 / 989e12
    assert 0.05 <= eff <= 0.6, eff


def test_stream_rows_standby_and_cached_weights():
    hy = get_model("hunyuanvideo")
    ph = Phase("full", 1, 33 * 45 * 80 + 256, 256, frames=33)
    rows = {o.name: o.m for o in build_rank_ops(hy, 0, ph, Shard()) if o.kind == "gemm"}
    assert rows["attn.txt_q"] == 256 and rows["attn.img_q"] == 33 * 45 * 80 and rows["attn.img_mod"] == 1
    rows = {o.name: o.m for o in build_rank_ops(hy, 30, ph, Shard()) if o.kind == "gemm"}
    assert rows["attn.q"] == ph.q and rows["mlp.fused_out"] == ph.q               # single stream: every token
    assert not any(o.name == "attn_allreduce" for o in build_rank_ops(hy, 30, ph, Shard(tp=2)))   # one fused out
    # Wan2.2 A14B: both experts stored, one read per step → same latency per step as Wan2.1-14B
    w21, w22 = get_model("wan2.1-14b"), get_model("wan2.2-a14b")
    s21 = stage_storage(w21, 0, 40, True, True, Shard(), 0)
    s22 = stage_storage(w22, 0, 40, True, True, Shard(), 0)
    assert abs(s22.weights / s21.weights - 2) < 0.01 and s22.hot_w == s21.hot_w
    a = evaluate(Scenario(model="wan2.1-14b", mem_id=HBM, workload=Workload(steps=40)))
    b = evaluate(Scenario(model="wan2.2-a14b", mem_id=HBM))
    assert abs(b.latency / a.latency - 1) < 1e-9 and b.stages[0].mem.dram_need > a.stages[0].mem.dram_need
    assert w22.active_params() == w21.params()
    # MiniMax-H3: AdaLN branches (13 B) are not stored for inference
    h3 = get_model("minimax-h3")
    st = stage_storage(h3, 0, h3.n_layers, True, True, Shard(), 0)
    assert 12e9 < h3.cached_params < 14e9 and st.weights < (h3.params() - h3.cached_params) * 2.2
    pre = {o.name: o.m for o in full_io_ops(h3, Phase("full", 1, 38222, 512, frames=37, aux=414), Shard(), "pre")
           if o.kind == "gemm"}
    assert pre["io.audio_in"] == 414 and pre["io.text_in"] == 512 and pre["io.patch_embed"] == 37 * 24 * 42


def test_sp_conservation_joint_rows():
    for mid, ph in (("minimax-h3", Phase("full", 1, 38222, 512, frames=37, aux=414)),
                    ("hunyuanvideo", Phase("full", 1, 33 * 45 * 80 + 256, 256, frames=33)),
                    ("opensora-stdit3", Phase("full", 2, 108000, 300, frames=30))):
        m = get_model(mid)

        def tot(sh: Shard, k: int) -> float:
            return k * sum(o.flops for li in (0, m.n_layers - 1) for o in build_rank_ops(m, li, ph, sh)
                           if o.kind in ("gemm", "attn"))
        one = tot(Shard(), 1)
        for sh, k in ((Shard(sp=2), 2), (Shard(tp=2), 2), (Shard(tp=2, sp=2), 4)):
            assert abs(tot(sh, k) / one - 1) < 5e-3, (mid, sh, tot(sh, k) / one)


def test_v0_invariants_new_video():
    for mid in NEW:
        base = Scenario(model=mid, mem_id=HBM, mapping="os")
        r = evaluate(base)
        assert r.fits and math.isfinite(r.latency) and r.latency > 0 and r.per_card > 0, mid
        assert not r.warnings, (mid, r.warnings)
        lat = {o: evaluate(base.replace("mapping", o)).latency for o in ("os", "ws_edge", "ws_broad", "os_vec", "reconf")}
        assert lat["reconf"] <= min(lat.values()) * (1 + 1e-9), (mid, lat)
        assert evaluate(base.replace("chip.sram_mib", base.chip.sram_mib * 2)).latency <= r.latency * (1 + 1e-9), mid
        assert evaluate(base.replace("serving.batch", 2)).latency >= r.latency * (1 - 1e-9), mid
        s2 = evaluate(base.replace("layout", Layout(sp=2)))
        assert 0.45 < s2.latency / r.latency < 0.62, (mid, s2.latency / r.latency)
    a = evaluate(Scenario(model="hunyuanvideo", mem_id=LP))
    b = evaluate(Scenario(model="hunyuanvideo", mem_id=LP, workload=Workload(cfg=2)))
    assert 1.9 < b.latency / a.latency < 2.1                                       # what-if: real CFG on a distilled model


def test_api_listing_and_eval_new_video():
    out = api.api_models()
    ids = {m["id"] for m in out["models"]}
    assert set(NEW) <= ids and not (set(NEW) & {o["id"] for o in out["offline"]})
    assert all(o["domain"] == "protein" for o in out["offline"])                   # 0.42: video fully on core v2
    for mid in NEW:
        e = api.api_eval({"scenario": {"model": mid, "workload": {"steps": 4}}})
        g = e["summary"]["gen"]
        assert g["clip_s"] > 0 and g["frames_per_s_card"] > 0 and g["workload"]["steps"] == 4, mid
    g = api.api_eval({"scenario": {"model": "minimax-h3"}})["summary"]["gen"]["workload"]
    assert g["audio_tokens"] == 414 and g["joint_text"] == 512
