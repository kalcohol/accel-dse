"""0.41 — video-generation DiT (Wan2.1, CogVideoX) and protein encoders (ESM-2) in the v2 pipeline:
exact parameters, workload → token count, FLOPs vs an independent count, V0 invariants, SP / TP conservation,
exact batch search, API round trip, and the LLM path untouched by the new domain."""
from __future__ import annotations

import json
import math
from pathlib import Path

from accel_dse import api
from accel_dse.core.catalog import get_model, labels
from accel_dse.core.domain import resolve_workload
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, build_rank_ops
from accel_dse.core.memplan import stage_storage
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving, Workload
from accel_dse.core.search import best_batch, brute_force_batch, search_layouts

ROOT = Path(__file__).resolve().parents[1]
HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"
VIDEO = ("wan2.1-14b", "wan2.1-1.3b", "cogvideox-5b", "cogvideox-2b")
PROTEIN = ("esm2-3b", "esm2-650m")


def _cfg(hf_id: str) -> dict:
    return json.loads((ROOT / "accel_dse" / "data" / "releases" / (hf_id.replace("/", "__") + ".json")).read_text())["config"]


def test_domain_params_exact_and_coverage_reasons():
    for mid in VIDEO + PROTEIN:
        lb = labels(get_model(mid))
        assert abs(lb["param_err"]) < 1e-9, (mid, lb["param_err"])      # safetensors headers, every tensor
        if mid in VIDEO:
            assert lb["coverage"] == "partial" and lb["model_domain"] == "gen"
            assert any("VAE" in r and "GB" in r for r in lb["coverage_reasons"]), mid   # unmodelled parts sized
            assert not lb["kv_cache"] and "KV" not in lb["dtype"]
        else:
            assert lb["coverage"] == "full" and lb["model_domain"] == "protein"
    assert labels(get_model("wan2.1-14b"))["dtype"].startswith("W fp32")             # as released
    assert labels(get_model("cogvideox-2b"))["dtype"] == "W fp16 · A fp16"
    assert labels(get_model("cogvideox-5b"))["dtype"] == "W bf16 · A bf16"


def test_workload_token_counts():
    W = Workload()
    assert resolve_workload(get_model("wan2.1-1.3b"), W).tokens == 21 * 30 * 52 == 32760      # 832×480, 81 f
    assert resolve_workload(get_model("wan2.1-14b"), W).tokens == 21 * 45 * 80 == 75600       # 1280×720
    cv = resolve_workload(get_model("cogvideox-5b"), W)
    assert cv.tokens == 13 * 30 * 45 + 226 and cv.seqs_per_request == 2 and cv.steps == 50 and cv.units == 49
    es = resolve_workload(get_model("esm2-650m"), W)
    assert es.tokens == 514 and es.steps == 1 and es.seqs_per_request == 1 and es.unit == "seq"
    w = resolve_workload(get_model("wan2.1-1.3b"), Workload(frames=33, height=480, width=832, steps=20, cfg=1))
    assert w.tokens == 9 * 30 * 52 and w.steps == 20 and w.seqs_per_request == 1 and w.warnings
    assert resolve_workload(get_model("esm2-650m"), Workload(seq_len=2000)).warnings       # beyond 1022 residues
    for bad in (dict(cfg=3), dict(frames=-1), dict(clip_slo_s=0.0)):
        try:
            Workload(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def _flops_per_request(mid: str) -> float:
    r = evaluate(Scenario(model=mid, mem_id=HBM))
    return r.domain_summary()["tflop_per_request"] * 1e12


def test_flops_vs_independent_count():
    """Per-request FLOPs from the op graph vs a closed-form count from config.json (io layers excluded: ≤ 1 %)."""
    c = _cfg("Wan-AI/Wan2.1-T2V-1.3B")
    d, f, L, nc = c["dim"], c["ffn_dim"], c["num_layers"], c["text_len"]
    n = 32760
    per_fwd = L * (2 * n * (4 * d * d + 2 * d * d + 2 * d * f) + 2 * nc * 2 * d * d + 4 * n * n * d + 4 * n * nc * d)
    ref = per_fwd * 2 * 50                                                    # CFG × steps
    got = _flops_per_request("wan2.1-1.3b")
    assert abs(got / ref - 1) < 0.01, (got, ref)
    c = _cfg("zai-org/CogVideoX-2b")
    d = c["num_attention_heads"] * c["attention_head_dim"]
    L, n = c["num_layers"], 17550 + 226
    per_fwd = L * (2 * n * (4 * d * d + 2 * d * 4 * d) + 4 * n * n * d)      # joint attention over text + video
    got = _flops_per_request("cogvideox-2b")
    assert abs(got / (per_fwd * 2 * 50) - 1) < 0.01, (got, per_fwd * 100)
    c = _cfg("facebook/esm2_t33_650M_UR50D")
    d, f, L, n = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], 514
    ref = L * (2 * n * (4 * d * d + 2 * d * f) + 4 * n * n * d) + 2 * n * (d * d + d * 33)
    got = _flops_per_request("esm2-650m")
    assert abs(got / ref - 1) < 0.01, (got, ref)


def test_v2_video_trend_wan_readme():
    """Wan2.1 README: T2V-1.3B makes a 5 s 480P clip on one RTX 4090 in ~4 min (T5 + VAE + offload included).
    Our per-clip DiT FLOPs over 240 s must imply a plausible fraction (40–100 %) of the 4090's ~165 TFLOPS
    dense bf16 (fp32 accumulate) — a sanity check of the workload / FLOP model, not a hardware calibration."""
    eff = _flops_per_request("wan2.1-1.3b") / 240.0 / 165e12
    assert 0.4 <= eff <= 1.0, eff


def test_v0_invariants_domain():
    for mid in VIDEO + PROTEIN:
        base = Scenario(model=mid, mem_id=HBM if mid in ("wan2.1-14b",) else LP, mapping="os")
        r = evaluate(base)
        assert r.fits and math.isfinite(r.latency) and r.latency > 0 and r.per_card > 0, mid
        s = r.domain_summary()
        assert s["unit"] in ("frame", "seq") and s["latency_s"] == r.latency
        lat = {o: evaluate(base.replace("mapping", o)).latency for o in ("os", "ws_edge", "ws_broad", "os_vec", "reconf")}
        assert lat["reconf"] <= min(lat.values()) * (1 + 1e-9), (mid, lat)
        assert evaluate(base.replace("chip.sram_mib", base.chip.sram_mib * 2)).latency <= r.latency * (1 + 1e-9), mid
        assert evaluate(base.replace("mem_id", HBM)).latency <= r.latency * (1 + 1e-9), mid
        assert evaluate(base.replace("serving.batch", 2)).latency >= r.latency * (1 - 1e-9), mid
        assert not r.warnings, (mid, r.warnings)
        assert "tpot_ms" in r.summary() and r.summary()["gen"]["unit"] == s["unit"]
    v = Scenario(model="wan2.1-1.3b", mem_id=LP)
    a, b = evaluate(v), evaluate(v.replace("workload", Workload(steps=25)))
    assert abs(b.latency / a.latency - 0.5) < 1e-9                             # latency ∝ denoise steps
    c = evaluate(v.replace("workload", Workload(cfg=1)))
    assert 0.45 < c.latency / a.latency < 0.55                                 # one forward per step instead of two
    assert evaluate(Scenario(model="wan2.1-14b", mem_id=HBM)).stages[0].convert_elems > 0   # fp32 → bf16 per GEMM


def test_sp_tp_conservation_and_memory():
    m = get_model("wan2.1-1.3b")
    ph = Phase("full", 1, 32760, 512)

    def tot(sh: Shard, ranks: int) -> float:
        return ranks * sum(o.flops for o in build_rank_ops(m, 3, ph, sh) if o.kind in ("gemm", "attn"))
    one = tot(Shard(), 1)
    for sh, k in ((Shard(tp=2), 2), (Shard(sp=2), 2), (Shard(tp=2, sp=2), 4), (Shard(sp=4), 4)):
        assert abs(tot(sh, k) / one - 1) < 2e-3, (sh, tot(sh, k) / one)    # cross K/V (512 rows) recomputed per SP rank
    w1 = stage_storage(m, 0, m.n_layers, True, True, Shard(), 0, 0)
    wsp = stage_storage(m, 0, m.n_layers, True, True, Shard(sp=2), 0, 0)
    wtp = stage_storage(m, 0, m.n_layers, True, True, Shard(tp=2), 0, 0)
    g = lambda s: s.hot_w + s.expert_w                                       # noqa: E731
    assert abs(g(wsp) / g(w1) - 1) < 1e-9 and g(wtp) < 0.6 * g(w1)            # SP replicates weights, TP splits them
    s1 = evaluate(Scenario(model="wan2.1-1.3b", mem_id=HBM))
    s2 = evaluate(Scenario(model="wan2.1-1.3b", mem_id=HBM, layout=Layout(sp=2)))
    assert 0.45 < s2.latency / s1.latency < 0.6                               # Ulysses: ~2× minus all-to-alls
    for bad in (Layout(sp=2), Layout(dp=2)):
        try:
            evaluate(Scenario(model="qwen3-8b", layout=bad))                  # SP / plain DP stay video / protein only
            raise AssertionError(bad)
        except ValueError:
            pass


def test_exact_batch_search_domain():
    for mid, cap, slo in (("esm2-650m", 48, dict(seq_slo_ms=200.0)), ("wan2.1-1.3b", 8, dict(clip_slo_s=1500.0))):
        base = Scenario(model=mid, mem_id=LP, mapping="reconf", workload=Workload(**slo))
        bb = best_batch(base, b_cap=cap)
        b, thr = brute_force_batch(base, cap)
        assert bb.batch == b and abs(bb.result.throughput - thr) <= 1e-9 * thr, (mid, bb.batch, b)
        prev = 0.0
        for k in range(1, cap + 1):                                           # P1 / P2
            r = evaluate(base.replace("serving.batch", k))
            assert r.latency >= prev * (1 - 1e-12)
            prev = r.latency
    rows = search_layouts(Scenario(model="esm2-650m", mem_id=LP), 4)
    assert rows and all(r.layout.cards == 4 and r.layout.ep == 1 for r in rows)
    assert any(r.layout.sp > 1 for r in rows) and any(r.layout.dp > 1 for r in rows)


def test_api_domain_round_trip_and_llm_unchanged():
    for mid in ("wan2.1-1.3b", "esm2-650m"):
        sc = {"scenario": {"model": mid, "workload": {"steps": 10} if mid.startswith("wan") else {"seq_len": 256}}}
        e = api.api_eval(sc)
        g = e["summary"]["gen"]
        assert "goodput" not in e and e["model"]["model_domain"] in ("gen", "protein")
        assert g["latency_s"] > 0 and (g.get("s_per_frame") or g.get("seq_per_s_card"))
        c = api.api_compare({**sc, "cards": 4})
        assert all(r["unit"] in ("frame", "seq") and r["tpot_ms"] is None for r in c["rows"] if r["batch"])
        assert api.api_fit(sc)["fits"]
        assert api.api_pareto(sc)["front"][0]["latency_ms"] > 0
        try:
            api.api_layouts({**sc, "objective": "goodput"})
            raise AssertionError("goodput must be rejected")
        except api.ApiError:
            pass
    big = api.api_fit({"scenario": {"model": "wan2.1-14b", "mem_id": "lpddr5x_2x64_8533_16g"}})
    assert not big["fits"] and big["min_cards"]["cards"] >= 2 and big["min_mem"]  # 57 GB fp32 weights > 32 GiB
    e = api.api_eval({"scenario": {"model": "qwen3-8b"}})
    assert "gen" not in e["summary"] and "goodput" in e and e["summary"]["domain"] == "llm"
    assert set(api.api_health()["domains"]) == {"llm", "vlm", "gen", "protein"}
