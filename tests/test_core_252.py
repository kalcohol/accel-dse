"""0.252: FLUX 3 Image Klein reference-image stream (image_cond, BFL flux3/dit.py + pipeline.py, diffusers
transformer_flux3.py) and tencent DSpine (DFlash anchor-row block draft + selected-prefix code bridge, sglang_dspine)."""
import math
import sys
sys.path.insert(0, ".")
from accel_dse.core import catalog as C
from accel_dse.core.domain import flux3_ref_size
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, build_rank_ops, dspark_ops
from accel_dse.core.model import spec_accept_from_tau
from accel_dse.core.scenario import Scenario

K, KB, DS = "flux-3-image-klein", "flux-3-image-klein-base", "qwen3-8b-dspine"


def _ev(mid, **wl):
    s = Scenario().replace("model", mid)
    for k, v in wl.items():
        s = s.replace("workload." + k, v)
    return evaluate(s)


def test_flux3_every_tensor_modelled():
    for mid in (K, KB):
        m = C.get_model(mid)
        assert m.coverage == "full" and not m.coverage_reasons and m.param_check()["rel_err"] == 0.0
        assert m.layers[0].core.span == "mode" and m.layers[-1].core.span == "full"
        assert m.layers[0].fused_out and m.layers[-1].fused_out                # parallel attn + SwiGLU blocks
        rows = {l.name: l.rows for l in m.io_pre}
        assert rows["ref_embed"] == rows["ref_mod"] == rows["single_ref_mod"] == "ref"
        assert rows["time_1"] == "tok"                                          # per-token time_in (BFL dit.py)
        assert ("target_time_1" in rows) == (mid == K)
        assert m.io_misc_params < 1e6                                           # norms / biases only
        assert m.workload.max_refs == 10


def test_flux3_ref_sizes_follow_release_rules():
    assert flux3_ref_size(1, 1024, 1024) == (1024, 1024)
    assert flux3_ref_size(1, 4000, 3000) == (2352, 1760)                       # ≤ 4 MiB canvas, 16-px grid
    assert flux3_ref_size(2, 1024, 1024) == (1024, 1024)
    assert flux3_ref_size(3, 3000, 2000) == (1248, 832)                        # each ≤ 1 MiB, floored to 16
    w, h = flux3_ref_size(1, 8000, 100)
    assert w <= 4096 and h >= 64 and w % 16 == h % 16 == 0


def test_flux3_reference_rows_and_attention():
    r0, r1, r4 = (_ev(K, ref_images=n) for n in (0, 1, 4))
    i0, i1, i4 = (r.workload.info for r in (r0, r1, r4))
    assert i0["ref_tokens"] == 0 and i0["seq_tokens"] == 4096 + 320 and i0["attention"] == "per_mode"
    assert i1["ref_tokens"] == 4096 and i1["seq_tokens"] == 8512 and i4["ref_tokens"] == 4 * 4096
    assert r0.latency < r1.latency < r4.latency and all(math.isfinite(r.latency) for r in (r0, r1, r4))
    assert 1.8 < r1.latency / r0.latency < 2.6                                 # sequence ×1.93, attention ∝ seq²
    # per-mode blocks: text / image / refs attend separately; single blocks jointly
    m = C.get_model(K)
    ph = Phase("full", 1, 4096 + 320 + 4096, 320, frames=1, ref=4096)
    early, single = build_rank_ops(m, 0, ph), build_rank_ops(m, 10, ph)
    qk = lambda ops: sorted(o.m * o.n for o in ops if o.name == "qk")
    assert qk(early) == sorted([320 * 320, 4096 * 4096, 4096 * 4096])
    assert qk(single) == [8512 * 8512]
    assert any(o.name == "attn.ref_q" and o.m == 4096 for o in early)


def test_flux3_reference_encode_and_cfg():
    r = _ev(K, ref_images=2)
    enc = [p for p in r.pipeline["parts"] if p["role"] == "vae_encode"]
    assert len(enc) == 1 and enc[0]["images"] == 2 and enc[0]["px"] == [1024, 1024] and r.pipeline["enc_s"] > 0
    assert 3.0 < enc[0]["tflop"] / 2 < 7.0                                     # SD-VAE-style encoder at 1 MP, fp32
    assert "enc_s" not in _ev(K).pipeline   # omitted at T2I (keeps the summary fingerprints)
    # Base: the [uncond, cond] pack carries the references in both samples → the same per-forward rows
    b0, b1 = _ev(KB), _ev(KB, ref_images=1)
    assert b1.workload.info["ref_tokens"] == 4096 and b1.workload.seqs_per_request == 2
    assert b1.latency > 2 * b0.latency * 0.9
    s = _ev(K, ref_images=1, ref_width=2048, ref_height=2048).workload.info
    assert s["ref_px"] == [2048, 2048] and s["ref_tokens"] == 128 * 128


def test_flux3_ref_limits():
    try:
        _ev(K, ref_images=11)
        raise AssertionError("11 references accepted")
    except ValueError as e:
        assert "10" in str(e)
    w = _ev("flux-2-klein-4b" if "flux-2-klein-4b" in {e["id"] for e in C.entries()} else "qwen-image", ref_images=1)
    assert any("ref_images 已忽略" in x for x in w.warnings)


def test_dspine_draft():
    m = C.get_model(DS)
    pc = m.param_check()
    assert pc["rel_err"] == 0.0 and pc["mtp_ours"] == pc["mtp_release"] == 1237763844
    assert m.draft == "dspark" and m.draft_anchor and m.draft_block == 16 and m.draft_markov == 0
    assert m.draft_bridge == (1024, 512, 16, 0, 1, 2, 3, 4) and m.spec_published == (7.83, 15)
    assert m.draft_table_bytes == 151936 * 3 * 512 * 2 and 0 <= m.draft_cold_params - (151936 * 1024 + 4) < 5   # + integer remainder of extra // layers
    assert len(m.mtp_layers) == 5 and m.mtp_layers[0].core.n_kv == 8
    ops = dspark_ops(m, Phase("decode", 1, 1, 4096))
    names = [o.name for o in ops]
    assert names.count("dspine.read") == 5 and names.count("dspine.write") == 5
    assert "dspark.markov_head" not in names and "dspark.confidence" not in names
    head = next(o for o in ops if o.name == "dspark.head")
    assert head.m == 15                                                        # drafts only, anchor row excluded
    blk = next(o for o in ops if o.name == "dspark_block_qk")
    assert abs(blk.causal - 17 / 32) < 1e-12
    a = spec_accept_from_tau(7.83, 15)
    assert 0.89 < a < 0.90


def test_dspine_serving():
    base = evaluate(Scenario().replace("model", "qwen3-8b").replace("serving.phase", "decode"))
    s = Scenario().replace("model", DS).replace("serving.phase", "decode").replace("serving.spec_accept", 0.8)
    r15 = evaluate(s.replace("serving.spec_k", 15))
    r7 = evaluate(s.replace("serving.spec_k", 7))
    assert r15.spec["k"] == 15 and r15.spec["n"] == 16 and r15.tpot < base.tpot / 2
    assert r7.spec["k"] == 15 and any("块大小固定" in w for w in r7.warnings)
    tp2 = evaluate(s.replace("serving.spec_k", 15).replace("layout.tp", 2))
    assert any("TP = 1" in w for w in tp2.warnings)
    p = evaluate(Scenario().replace("model", DS).replace("serving.phase", "prefill").replace("serving.spec_k", 15))
    assert math.isfinite(p.ttft) and p.ttft > 0
    assert Scenario().replace("serving.spec_k", 16).serving.spec_k == 16
