"""0.269: Ming-3.0-tts-tiny — TTS decode head (stop head + CFM flow DiT × flow_steps × CFG + aggregator + streaming
AudioVAE decode) replacing the vocab head; re-read hot weight class pinned first (0.192 bytes-saved rule)."""
from dataclasses import replace
from pathlib import Path

from accel_dse import api
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import Phase, head_ops
from accel_dse.core.memplan import StageStorage, plan
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.tts import tts_step_flops

ROOT = Path(__file__).resolve().parents[1]
LP = "lpddr5x_4x64_8533_16g"
TTS = "ming-3.0-tts-tiny"


def test_exact_params_vs_header():
    m = get_model(TTS)
    t = m.tts
    assert m.param_check()["rel_err"] == 0.0 and m.params() == 7_893_392_800 and m.mtp_params() == 0
    # released safetensors header, component by component
    assert (t.flow_params, t.agg_params, t.dec_params) == (204_526_688, 102_396_448, 361_127_186)
    assert (t.enc_params, t.spk_params, t.stop_params) == (419_261_824, 296_448, 3_074)
    assert m.params() + t.total_params == 8_981_004_468              # HF safetensors total
    assert m.vision is None and m.vision_params == 0
    assert abs(t.s_per_step - 0.08) < 1e-12 and t.cfg_mult == 2 and t.flow_steps == 10 and t.flow_rows == 3
    e = next(x for x in list_models() if x["id"] == TTS)
    assert e["domain"] == "tts" and e["coverage"] == "full" and e["family"] == "Ming-3.0-TTS"


def test_head_ops_replace_vocab_head():
    m = get_model(TTS)
    ops = head_ops(m, Phase("decode", 4, 300, 1))
    names = [o.name for o in ops]
    assert "lm_head" not in names and "tts.stop_head" in names
    assert sum(n.startswith("tts.flow") and n.endswith(".0.qkv") for n in names) == 10   # one DiT per Euler step
    f = tts_step_flops(m.tts)
    # hand count, one request: 10 evals × 6 rows (3 tokens × CFG 2) × 16 blocks × 2·12·1024² ≈ 24.2 GFLOP
    assert 24.0e9 < f["flow"] < 24.5e9 and 2.8e9 < f["vae"] < 3.0e9


def test_tts_kpis_and_reread_pinning():
    def ev(gib, b=1):
        ch = replace(CHIPS["100T"], slc_mib=int(gib * 1024), slc_policy="pin") if gib else CHIPS["100T"]
        return evaluate(Scenario(model=TTS, chip=ch, mem_id=LP, serving=Serving(batch=b, prompt=64, ctx=300)))
    r0, r05 = ev(0), ev(0.5)
    asr = evaluate(Scenario(model="ming-3.0-asr-tiny", chip=CHIPS["100T"], mem_id=LP,
                            serving=Serving(batch=1, prompt=64, ctx=300)))
    assert r0.tpot > 2.5 * asr.tpot                       # same decoder: the head dominates a B1 step
    t = r0.tts
    assert abs(t["rtf"] - r0.tpot / 0.08) < 1e-12 and abs(t["audio_s_per_s"] - r0.throughput * 0.08) < 1e-9
    assert abs(t["head_w_GB_per_step"] - (10 * 204_526_688 + 102_396_448 + 361_127_186 + 3_074) * 2 / 1e9) < 1e-9
    # a 0.5 GiB SLC holds the re-read flow DiT (409 MB) → B1 step more than halves
    assert r05.tpot < 0.5 * r0.tpot
    d = api.result_dict(r0)["summary"]
    assert d["tts"]["flow_steps"] == 10


def test_plan_pins_reread_first_and_noop_without():
    st = StageStorage(hot_w=1e9, expert_w=0.0, kv_per_seq=0.0, idx_per_seq=0.0, state_per_seq=1e6,
                      rr_w=4e8, rr_reads=10.0)
    mp = plan(st, 1, 3e8, 1e12, 1e6)
    assert mp.pinned_rr == mp.pinned_hot > 0 and mp.state_sram == 0.0     # 10 B/B beats state's 2 B/B
    mp2 = plan(st, 1, 3e8, 1e12, 1e6, slc_bytes=2e8)
    assert mp2.slc_rr > 0 and mp2.slc_rr + mp2.pinned_rr <= 4e8 + 1
    base = StageStorage(1e9, 0.0, 0.0, 0.0, 1e6)
    a, b = plan(base, 1, 3e8, 1e12, 1e6), plan(replace(base, rr_w=0.0), 1, 3e8, 1e12, 1e6)
    assert a == b and a.pinned_rr == 0.0


def test_ui():
    js = (ROOT / "accel_dse/web/app.js").read_text()
    assert "s.tts" in js and "tts: '语音合成'" in js and "m.domain === 'tts'" in js


def test_health_domains_follow_catalog():
    """0.269: /api/health domains = catalog.DOMAINS (incl. asr / tts) and every listed model's domain is in it."""
    from accel_dse.core.catalog import DOMAINS
    h = api.api_health()["domains"]
    assert h == list(DOMAINS) and {"asr", "tts"} <= set(h)
    assert {m["domain"] for m in list_models()} <= set(h)


def test_tts_weight_reads_conserved():
    """Batch-1 TTS step: backbone conservation as test_core_125 with the vocab head replaced by the TTS head;
    TTS head reads flow weights once per Euler step (CFG rides as rows), aggregator + VAE decoder + stop once."""
    from accel_dse.core import catalog as C, ir
    from accel_dse.core.ir import Phase, Shard
    m = C.get_model("ming-3.0-tts-tiny")
    t, ph = m.tts, Phase("decode", 1, 1, 64)
    ops = []
    for li in range(len(m.layers)):
        ops += ir.build_rank_ops(m, li, ph, Shard())
    head = ir.head_ops(m, ph, Shard())
    for o in ops + head:
        if o.w_params and o.kind == "gemm":
            assert o.k * o.n * o.count == o.w_params, o.name
    tts_w = sum(o.w_params for o in head if o.name.startswith("tts."))
    want = t.flow_params * t.flow_steps + t.agg_params + t.dec_params + t.stop_params
    assert -0.01 < tts_w / want - 1 < 1e-9, (tts_w, want)
    body = sum(o.w_params for o in ops if o.kind != "lookup")
    r = (body + m.embed_params) / (m.active_params() - m.head_params) - 1
    assert -0.01 < r < 1e-9, r
