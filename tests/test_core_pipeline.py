"""0.44 — video pipeline components: text encoders and VAE decoders as op graphs from their released checkpoint
headers (accel_dse/data/pipeline), counted in clip latency and per-card storage by default (workload.pipeline);
pipeline = false reproduces the 0.43 DiT-only numbers.  Also the activation-dtype what-if (fp32 activations)."""
from __future__ import annotations

import math

from accel_dse import api
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.pipeline import (PIPELINES, _hunyuan_tiles, component_data, pipeline_for, stored_bytes, text_ops,
                                     vae_ops)
from accel_dse.core.scenario import Scenario, Workload

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"
VIDEO = ("wan2.1-14b", "wan2.1-1.3b", "wan2.2-a14b", "cogvideox-5b", "cogvideox-2b", "hunyuanvideo", "ltx-video",
         "mochi-1", "opensora-stdit3", "minimax-h3")


def test_every_video_model_has_a_pipeline_with_release_data():
    assert set(PIPELINES) == set(VIDEO)
    for mid in VIDEO:
        pl = pipeline_for(mid)
        for k in [t.key for t in pl.text] + [v.key for v in (pl.vae, pl.audio) if v]:
            d = component_data(k)
            assert d["dtype"] in ("fp32", "bf16", "fp16") and d["params_stored"] >= d["params_used"] > 0, k
    # stored sizes match the released files (GB): umT5 11.36, Wan VAE 0.51, Llama text tower 15.0, Qwen3-VL 66.7
    assert abs(stored_bytes("umt5-xxl") / 1e9 - 11.36) < 0.01
    assert abs(stored_bytes("wan-vae") / 1e9 - 0.51) < 0.01
    assert abs(stored_bytes("llava-llama3-8b") / 1e9 - 15.01) < 0.01
    assert abs(stored_bytes("qwen3-vl-32b") / 1e9 - 66.71) < 0.01
    assert abs(stored_bytes("t5-v1_1-xxl.deepfloyd") / 1e9 - 19.05) < 0.01      # fp32 .bin (no safetensors header)


def test_text_encoder_flops_closed_form():
    # umT5-XXL: 24 layers, d 4096, gated FFN 10240, 64 heads × 64, bidirectional; 2 prompts × 512 tokens (CFG)
    te = pipeline_for("wan2.1-14b").text[0]
    ops, _ = text_ops(te, 2, "bf16")
    rows, d, f, L, T = 1024, 4096, 10240, 24, 512
    ref = 2 * rows * L * (4 * d * d + 3 * d * f) + L * 2 * 64 * 4 * T * T * 64
    got = sum(o.flops for o in ops)
    assert abs(got / ref - 1) < 1e-9, got / ref
    # Qwen3-VL text tower: causal attention, GQA (8 KV heads) taken from the header shapes, vision tower not run
    te = pipeline_for("minimax-h3").text[0]
    ops, _ = text_ops(te, 1, "bf16")
    assert not any("visual" in o.name or "lm_head" in o.name for o in ops)
    k = [o for o in ops if o.name.endswith("layers.0.self_attn.k_proj.weight")][0]
    assert (k.k, k.n) == (5120, 8 * 128)
    assert all(o.causal == 0.5 for o in ops if o.kind == "attn")


def test_vae_resolution_schedules():
    def out(mid, lat, F=0):
        return vae_ops(pipeline_for(mid).vae, lat, F, 1, "bf16")[2]
    assert out("wan2.1-14b", (21, 90, 160))["out"] == (81, 720, 1280)          # causal 2t−1 twice, ×8 spatial
    assert out("cogvideox-5b", (13, 60, 90))["out"] == (49, 480, 720)
    assert out("ltx-video", (21, 16, 22))["out"] == (161, 128, 176)            # ×8 then 4×4 unpatchify in conv_out
    assert out("mochi-1", (28, 60, 106))["out"] == (163, 480, 848)             # temporal ×3, ×2, ×1 (t·e − (e−1))
    hy = out("hunyuanvideo", (33, 90, 160))
    assert hy["tiles"] == 11 * 4 * 7 and 2.5 < hy["overlap"] < 2.8              # diffusers tiling, overlap recomputed
    assert sum(n for _, n in _hunyuan_tiles((33, 90, 160))) == 308
    assert out("minimax-h3", (37, 48, 84))["chunks"] == 8                      # (37 + token_drop 3) / 5


def test_wan_vae_flops_hand_count():
    """Wan-VAE decoder at 480P (latent 21×60×104): every conv as an implicit GEMM at its level's voxel count."""
    L0, L1, L2, L3 = 21 * 60 * 104, 41 * 120 * 208, 81 * 240 * 416, 81 * 480 * 832
    g = lambda vox, k, n: 2 * vox * k * n
    k3 = 27
    ref = (g(L0, 16, 16) + g(L0, 16 * k3, 384) + 4 * g(L0, 384 * k3, 384) + g(L0, 384, 1152) + g(L0, 384, 384)
           + 6 * g(L0, 384 * k3, 384) + g(L0, 384 * 3, 768) + g(L1, 384 * 9, 192)                       # block 0
           + g(L1, 192 * k3, 384) + g(L1, 192, 384) + 5 * g(L1, 384 * k3, 384) + g(L1, 384 * 3, 768)
           + g(L2, 384 * 9, 192)                                                                          # block 1
           + 6 * g(L2, 192 * k3, 192) + g(L3, 192 * 9, 96)                                                # block 2
           + 6 * g(L3, 96 * k3, 96) + g(L3, 96 * k3, 3))                                                  # block 3
    ref += 21 * 4 * (60 * 104) ** 2 * 384                                         # mid-block attention per frame
    ops, _, info = vae_ops(pipeline_for("wan2.1-1.3b").vae, (21, 60, 104), 81, 1, "bf16")
    got = sum(o.flops for o in ops)
    assert abs(got / ref - 1) < 1e-9, got / ref
    assert info["act"] == "fp32"                                                  # reference WanVAE runs in fp32


def test_pipeline_false_is_dit_only_and_latency_adds_up():
    for mid in VIDEO:
        on = evaluate(Scenario(model=mid, mem_id=HBM))
        off = evaluate(Scenario(model=mid, mem_id=HBM, workload=Workload(pipeline=False)))
        p = on.pipeline
        assert off.pipeline is None and p and p["te_s"] > 0 and p["decode_s"] > 0, mid
        assert abs(on.latency - (off.latency + p["te_s"] + p["decode_s"])) < 1e-9 * on.latency, mid
        s = on.domain_summary()
        assert abs(s["denoise_s"] - off.latency) < 1e-9 * off.latency and s["step_ms"] == off.domain_summary()["step_ms"]
        assert s["tflop_per_request"] > s["dit_tflop_per_request"] == off.domain_summary()["tflop_per_request"]
        added = on.stages[0].mem.dram_need - off.stages[0].mem.dram_need
        w = p["te_w"] + p["vae_w"]
        assert w - 1 <= added and on.stages[0].mem.pipe_w == w, mid                # pp = 1: everything on one card
        assert any("pipeline = false" in x for x in off.warnings), mid
    # PP 2: text encoder on stage 0, VAE on the last stage
    r = evaluate(Scenario(model="wan2.1-14b", mem_id=HBM, layout=Layout(pp=2)))
    assert r.stages[0].mem.pipe_w == r.pipeline["te_w"] and r.stages[1].mem.pipe_w == r.pipeline["vae_w"]


def test_batch_dp_and_capacity():
    a = evaluate(Scenario(model="wan2.1-1.3b", mem_id=HBM))
    b = evaluate(Scenario(model="wan2.1-1.3b", mem_id=HBM).replace("serving.batch", 2))
    assert b.pipeline["decode_s"] > 1.9 * a.pipeline["decode_s"]                 # one decode per clip
    c = evaluate(Scenario(model="wan2.1-1.3b", mem_id=HBM, layout=Layout(dp=2)).replace("serving.batch", 2))
    assert c.pipeline["batch_per_replica"] == 1 and abs(c.pipeline["decode_s"] / a.pipeline["decode_s"] - 1) < 1e-9
    assert abs(c.pipeline["tflop"] / b.pipeline["tflop"] - 1) < 1e-9             # same total work, two replicas
    # Wan2.1-14B fp32 DiT (57 GB) + umT5 (11.4 GB) no longer fits 64 GB LPDDR; DiT-only still does
    r = evaluate(Scenario(model="wan2.1-14b", mem_id=LP, workload=Workload(placement="resident")))
    assert not r.fits and any("文本编码器 / VAE" in w for w in r.warnings)
    assert evaluate(Scenario(model="wan2.1-14b", mem_id=LP, workload=Workload(pipeline=False))).fits


def test_scenario_api_and_cli_toggle():
    s = Scenario(model="hunyuanvideo", workload=Workload(pipeline=False))
    assert Scenario.from_dict(s.to_dict()) == s and s.hash() != Scenario(model="hunyuanvideo").hash()
    try:
        Workload(pipeline=1)
        raise AssertionError("non-bool accepted")
    except ValueError:
        pass
    out = api.api_eval({"scenario": {"model": "mochi-1", "mem_id": HBM}})
    g = out["summary"]["gen"]
    assert [p["role"] for p in g["pipeline"]["parts"]] == ["text_encoder", "vae"]
    assert out["stages"][0]["mem"]["pipe_w_GiB"] > 17
    assert out["model"]["workload"]["pipeline"][0]["dtype"] == "fp32"
    out = api.api_eval({"scenario": {"model": "mochi-1", "mem_id": HBM, "workload": {"pipeline": False}}})
    assert out["summary"]["gen"]["pipeline"] is None
    from accel_dse.cli import main
    assert main(["eval", "--model", "ltx-video", "--dit-only", "--json"]) == 0


def test_fp32_activation_what_if():
    base = Scenario(model="esmfold", mem_id=LP)          # DRAM-bound on LPDDR: activation streaming doubles
    a = evaluate(base)
    b = evaluate(base.replace("formats_override", (("act", "fp32"),)))
    assert b.model.what_if and b.model.act_fmt == "fp32" and a.model.act_fmt == "bf16"
    assert b.latency > 1.5 * a.latency and b.stages[0].mem.dram_need > a.stages[0].mem.dram_need
    assert b.stages[0].dram["act"] > 1.9 * a.stages[0].dram["act"]
    h = Scenario(model="esmfold", mem_id=HBM)              # MAC-bound on HBM: never faster
    assert evaluate(h.replace("formats_override", (("act", "fp32"),))).latency >= evaluate(h).latency
    assert abs(b.domain_summary()["tflop_per_request"] / a.domain_summary()["tflop_per_request"] - 1) < 1e-12
    assert b.stages[0].convert_elems > a.stages[0].convert_elems                 # fp32 → bf16 per GEMM on the vector path
    out = api.api_eval({"scenario": {"model": "boltz-1", "mem_id": HBM, "formats_override": [["act", "fp32"]]}})
    assert out["model"]["act_fmt"] == "fp32" and out["model"]["what_if"]
