"""0.231: chunked-prefill engine profiles of the colocated chunked mode — vllm (default, 来源: iteration token budget
incl. the running decodes, decode-first, prompt chunk = budget − running batch, budget by card memory), vllm_v0
(budget 2048), legacy_512 (≤ 0.230 「假设」), fixed; DES parity; invariants; fuzz."""
import math
import random

from accel_dse.core.disagg import _card_gib, disagg_report
from accel_dse.core.energy import EnergyTable, apply_energy_preset
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import CHUNK_PROFILES, _chunk_plan, chunk_at, chunk_spec, vllm_budget
from accel_dse.core.pdsim import compare
from accel_dse.core.scenario import CHUNK_PROFILE_IDS, PDConfig, Scenario, Serving

HB = "hbm3e_8s_12h24g_9200"


def _pd(prompt=4096, out=512, batch=64, cap=0.0, **pd):
    kw = dict(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=0.6)
    kw.update(pd)
    return Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HB, layout=Layout(tp=2), mem_cap_GB=cap,
                    serving=Serving(batch=batch, prompt=prompt, out_len=out), pd=PDConfig(**kw))


def test_tiers_and_spec():
    assert set(CHUNK_PROFILES) == set(CHUNK_PROFILE_IDS)
    # vLLM arg_utils (API server): < 70 GiB 2048, ≥ 70 GiB 8192, ≥ 160 GiB 16384 「来源」
    assert [vllm_budget(g) for g in (16, 69.9, 70, 141, 159.9, 160, 288)] == [2048, 2048, 8192, 8192, 8192, 16384, 16384]
    sp = chunk_spec(PDConfig(), 80.0)
    assert sp["profile"] == "vllm" and sp["budget"] == 8192 and sp["tag"] == "来源"
    assert [chunk_at(sp, k) for k in (0, 1, 64, 8191, 8192, 9000)] == [8192, 8191, 8128, 1, 0, 0]
    assert chunk_spec(PDConfig(chunk_budget=4096), 300.0)["budget"] == 4096
    assert chunk_spec(PDConfig(chunk_profile="vllm_v0"), 300.0)["budget"] == 2048
    assert chunk_at(chunk_spec(PDConfig(chunk_profile="legacy_512"), 80.0), 63) == 512
    assert chunk_at(chunk_spec(PDConfig(chunk_profile="fixed", chunk_tokens=1000), 80.0), 63) == 1000
    assert abs(_card_gib(_pd()) - 288.0) < 1e-9 and abs(_card_gib(_pd(cap=48.0)) - 48e9 / 2 ** 30) < 1e-9


def test_config_validation_and_legacy():
    for bad in (dict(chunk_profile="sglang"), dict(chunk_budget=8), dict(chunk_budget=1.5), dict(chunk_budget=True)):
        try:
            PDConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert Scenario().hash() == Scenario.from_dict(Scenario().to_dict()).hash()
    assert "chunk_profile" not in Scenario(pd=PDConfig(enabled=True)).to_dict()["pd"]       # omit_default
    # ≤ 0.230 dicts: an explicit non-512 chunk_tokens meant a fixed chunk
    assert Scenario.from_dict({"pd": {"chunk_tokens": 1024}}).pd.chunk_profile == "fixed"
    assert Scenario.from_dict({"pd": {"chunk_tokens": 512}}).pd.chunk_profile == "vllm"
    assert Scenario.from_dict({"pd": {"chunk_tokens": 1024, "chunk_profile": "vllm"}}).pd.chunk_profile == "vllm"


def test_default_is_vllm_and_legacy_reproduces_512():
    q = disagg_report(_pd())["queue"]
    c = q["modes"]["coloc_chunked"]
    assert q["chunk_profile"]["profile"] == "vllm" and q["chunk_profile"]["budget"] == 16384
    assert q["chunk_tokens"] == 16384 - 64 and c["stable"] and c["prefill"]["chunks"] == 1
    ql = disagg_report(_pd(chunk_profile="legacy_512"))["queue"]
    cl = ql["modes"]["coloc_chunked"]
    assert ql["chunk_tokens"] == 512 and cl["prefill"]["chunks"] == 8 == _chunk_plan(4096, 0, 512)[0]
    # the other modes do not depend on the chunk profile
    for m in ("pd", "coloc_prefill_first"):
        assert q["modes"][m]["ttft_ms"] == ql["modes"][m]["ttft_ms"]
    # 48 GB card → 2048 tier: chunk = 2048 − running batch (fractional k_s → between the integer batches)
    q2 = disagg_report(_pd(cap=48.0))["queue"]
    c2 = q2["modes"]["coloc_chunked"]
    assert q2["chunk_profile"]["budget"] == 2048 and 2048 - 64 <= c2["chunk_tokens"] <= 2047
    assert 2 <= c2["prefill"]["chunks"] <= 3
    # bigger chunks: fewer chunk iterations → lower TTFT than 512 (same arrival rate)
    assert c["ttft_ms"]["mean"] < cl["ttft_ms"]["mean"] and c2["ttft_ms"]["mean"] < cl["ttft_ms"]["mean"]


def test_budget_below_batch_is_infeasible():
    q = disagg_report(_pd(chunk_budget=64))["queue"]
    c = q["modes"]["coloc_chunked"]
    assert not c["stable"] and "预算" in c["why"] and q["chunk_tokens"] == 0
    assert q["modes"]["pd"]["stable"]


def test_des_parity_small_budget_decodes_not_starved():
    # budget barely above the batch: prompt progress is a few tokens per iteration, decodes are always carried
    s = _pd(prompt=1024, out=128, batch=16, load=0.3, chunk_budget=16 + 240)
    r = compare(s, n_req=300, warmup=60, seed=231)
    x = r["modes"]["coloc_chunked"]
    assert x["sim"]["tpot_mean"] > 0 and math.isfinite(x["sim"]["tpot_mean"])
    assert abs(x["err"]["tpot_mean"]) < 0.1 and abs(x["err"]["ttft_p50"]) < 0.25, x["err"]


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def test_fuzz_profiles():
    rng = random.Random(231)
    for i in range(5):
        prof = ("vllm", "vllm_v0", "legacy_512", "fixed", "vllm")[i]
        kw = {"chunk_profile": prof}
        if prof == "fixed":
            kw["chunk_tokens"] = rng.choice((128, 1000, 3000))
        if prof == "vllm" and i:
            kw["chunk_budget"] = rng.randint(80, 6000)
        batch = rng.choice((16, 32, 64))
        rate = rng.uniform(2.0, 6.0)
        prev = None
        for prompt in (1024, 2048, 4096):
            s = _pd(prompt=prompt, out=256, batch=batch, rate_rps=rate, **kw)
            q = disagg_report(s, energy=_t(idle_W=rng.uniform(0, 30)))["queue"]
            c = q["modes"]["coloc_chunked"]
            assert q["chunk_tokens"] >= 0 and (c.get("chunk_tokens") or 0) >= 0, (prof, kw)
            if prof.startswith("vllm"):
                # decode-first: a feasible iteration carries every running decode and budget − k prompt tokens
                b = q["chunk_profile"]["budget"]
                for k in range(1, batch + 1):
                    assert chunk_at(q["chunk_profile"], k) + k <= max(b, k) and chunk_at(q["chunk_profile"], k) >= 0
            if not c["stable"]:
                prev = None
                continue
            # TTFT monotone in prompt length at a fixed arrival rate
            if prev is not None:
                assert c["ttft_ms"]["mean"] >= prev - 1e-9, (prof, kw, prompt)
            prev = c["ttft_ms"]["mean"]
            lam, o = q["lambda_rps"], s.serving.out_len
            for m, e in q["energy"].items():
                W = e["J_per_token"] / e["card_s_per_token"]
                assert 0 < W < 2000 and abs(e["J_per_token"] * lam * o / 8 / W - 1) < 1e-9, m
