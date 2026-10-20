"""0.287: memory-hierarchy ledger audit (MODEL §232).

* repo hygiene: a ``local`` symlink (the per-worktree link to the shared local data) is ignored by .gitignore — the
  0.286 squash briefly tracked one because ``local/`` only matches directories — and the tracked tree has no ``local``;
* ledger property: for random stage storages, batches, SRAM / SLC sizes, policies, state buffers, measured expert
  loads and touched-byte sums, every tier's bytes are non-negative and SRAM + SLC + DRAM = the step's demand per
  class, and no tier holds more than its capacity;
* the two hits of the audit: the co-located (chunked-prefill) iteration charged the prompt chunk's KV under the
  chunk's own memory plan — on-chip bytes the decode batch already holds — and an independent draft model was planned
  on the full SRAM / SLC / state buffer of the head-stage cards the target had already filled."""
import math
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args):
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return p if p.returncode == 0 else None


def test_gitignore_ignores_local_symlink():
    lines = [l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()]
    # a pattern without a trailing slash matches a symlink as well as a directory
    assert "/local" in lines or "local" in lines, "need a 'local' pattern without trailing slash"
    if _git("rev-parse", "--git-dir") is None:
        return                                   # sdist / bundle extract: the pattern check above is the guard
    for path in ("local",):
        p = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, timeout=30)
        assert p.returncode == 0, path


def test_tracked_tree_has_no_local_entry():
    p = _git("ls-files", "-z")
    if p is None:
        return
    bad = [f for f in p.stdout.split("\0") if f == "local" or f.startswith("local/")]
    assert not bad, bad


# ---------------------------------------------------------------------------------------------- ledger property
from accel_dse.core import memplan as M

G = 2.0 ** 30
CLASSES = ("hot", "exp", "kv_read", "kv_write", "state", "lookup", "act", "w_extra")


def _sram_served(mp, st, t) -> dict:
    """Bytes of each demand class served by the SRAM, from the plan's resident bytes alone (independent of
    step_dram_bytes): a resident byte share of a class serves that share of its reads; experts by the read share of
    the SRAM-pinned experts (measured load) or their byte share."""
    rr = min(st.rr_w, st.hot_w)
    t_rr = t["hot_rr"] if st.rr_w > 0 else 0.0
    if st.hot_w <= 0:
        hot = 0.0
    elif rr > 0:
        rest = st.hot_w - rr
        hot = (t["hot"] - t_rr) * ((mp.pinned_hot - mp.pinned_rr) / rest if rest > 0 else 0.0) + t_rr * mp.pinned_rr / rr
    else:
        hot = t["hot"] * mp.pinned_hot / st.hot_w
    e_share = 0.0 if st.expert_w <= 0 else (mp.sram_exp_hit if mp.sram_exp_hit >= 0 else mp.pinned_expert / st.expert_w)
    kv_share = 0.0 if mp.kv_total <= 0 else (mp.kv_sram - mp.state_sram) / mp.kv_total
    st_share = 0.0 if mp.state_total <= 0 else mp.state_sram / mp.state_total
    return {"hot": hot, "exp": t["exp"] * e_share, "kv_read": t["kv_read"] * kv_share,
            "kv_write": t["kv_write"] * kv_share, "state": t["state"] * st_share, "lookup": 0.0, "act": 0.0,
            "w_extra": 0.0}


def _check_ledger(mp, st, t, d, where):
    """Per class: SRAM + SLC + DRAM = demand, every tier ≥ 0; capacities respected."""
    tol = lambda x: 1e-6 * max(1.0, x) + 1.0
    sram = _sram_served(mp, st, t)
    hit = d.get("slc_parts", {})
    w_dem = t["hot"] + t["exp"] + t["w_extra"]
    w_sram = sram["hot"] + sram["exp"]
    groups = {"weights": (w_dem, w_sram), "kv_read": (t["kv_read"], sram["kv_read"]),
              "kv_write": (t["kv_write"], sram["kv_write"]), "state": (t["state"], sram["state"]),
              "lookup": (t["lookup"], 0.0), "act": (t["act"], 0.0)}
    for k, (dem, s_) in groups.items():
        dr, sl = d[k], hit.get(k, 0.0)
        assert dr >= -tol(dem) and sl >= -tol(dem) and s_ >= -tol(dem), (where, k, dr, sl, s_)
        assert abs(dr + sl + s_ - dem) <= tol(dem), (where, k, dem, dr, sl, s_)
    assert abs(d["total"] - sum(d[k] for k in groups)) <= tol(d["total"]), where
    assert abs(d.get("slc", 0.0) - sum(hit.values())) <= tol(d.get("slc", 0.0)), where
    e_dem, e_sram = t["exp"], sram["exp"]
    assert d["exp_dram"] >= -tol(e_dem) and d["exp_slc"] >= -tol(e_dem), where
    assert abs(d["exp_dram"] + d["exp_slc"] + e_sram - e_dem) <= tol(e_dem), (where, d["exp_dram"], d["exp_slc"], e_sram)
    # capacities: SRAM pins fit beside staging and any reservation; the SLC holds at most its size; nothing is held
    # on chip twice (SRAM + SLC of a class ≤ what is stored)
    assert mp.pinned_hot + mp.pinned_expert + mp.kv_sram <= max(0.0, mp.sram - mp.staging - mp.sram_resv) + mp.state_buf + 1, where
    assert mp.slc_hot + mp.slc_expert + mp.slc_kv <= mp.slc + 1 and mp.slc_held <= mp.slc + 1, where
    assert mp.pinned_hot + mp.slc_hot <= st.hot_w + 1 and mp.pinned_expert + mp.slc_expert <= st.expert_w + 1, where
    assert mp.kv_sram + mp.slc_kv <= mp.kv_total + mp.state_total + 1, where


def _zipf_p(R, n, k):
    raw = [R.paretovariate(R.choice([0.7, 1.0, 2.0])) for _ in range(n)]
    s = sum(raw)
    return tuple(min(1.0, k * x / s) for x in raw)


def test_ledger_property_random_plans():
    """Random storages / batches / SRAM / SLC / policies / state buffers / reservations / measured loads / touched
    sums: per class SRAM + SLC + DRAM = demand with no negative tier (the 0.286 bug class: bytes credited to two tiers,
    negative DRAM bytes, a division by zero)."""
    R = random.Random(287)
    pick = lambda hi: R.choice([0.0, R.uniform(0, hi), R.uniform(0, hi)])
    for it in range(4000):
        hot = pick(20) * G
        rr = min(hot, pick(3) * G) if R.random() < 0.3 else 0.0
        st = M.StageStorage(hot, pick(40) * G, pick(1) * G / 64, pick(0.1) * G / 64, pick(1) * G / 64, pick(5) * G, rr,
                            R.choice([1.0, 2.0, 10.0, 36.0]))
        E = st.expert_w
        exp_read = R.choice([None, pick(1.5) * E])
        exp_p = _zipf_p(R, R.choice([8, 64, 128]), R.choice([1, 2, 8])) if R.random() < 0.4 else None
        sram = pick(32) * G
        mp = M.plan(st, R.choice([0, 1, 8, 64, 256]), sram, 1e15, pick(0.5) * G, act_need=pick(4) * G,
                    stream=R.random() < 0.5, slc_bytes=R.choice([0.0, pick(32) * G]),
                    slc_policy=R.choice(["pin", "pin_bw", "lru", "dip"]), kv_frac=R.choice([1.0, 0.5]),
                    state_buf=R.choice([0.0, pick(4) * G]), exp_read=exp_read, exp_p=exp_p,
                    sram_resv=R.choice([0.0, 0.0, pick(1) * sram]), slc_resv=R.choice([0.0, pick(8) * G]),
                    buf_resv=R.choice([0.0, pick(1) * G]))
        t = {"hot": st.hot_w * R.choice([0.5, 1.0, 3.0]), "exp": exp_read if exp_read is not None else E,
             "kv_read": mp.kv_total * R.choice([1.0, 0.5]), "kv_write": pick(0.01) * G, "state": 2 * mp.state_total,
             "lookup": pick(0.01) * G, "hot_rr": rr * st.rr_reads, "act": pick(2) * G, "w_extra": pick(1) * G}
        t["hot"] += t["hot_rr"]
        d = M.step_dram_bytes(mp, st, t, R.choice([0.0, 0.5, 2.0]))
        _check_ledger(mp, st, t, d, it)


def test_untracked_stream_misses_not_vanishes():
    """A touched KV / state stream with no planned storage (kv_total = 0: the text-prefix K/V of a cached image-DiT
    step, held in the activation reserve) has nothing on chip → all of it is DRAM traffic (≤ 0.286: miss share 0, the
    bytes were in no tier)."""
    st = M.StageStorage(1 * G, 0.0, 0.0, 0.0, 0.0)
    mp = M.plan(st, 0, 4 * G, 100 * G, 0.0)
    t = {"hot": 1 * G, "exp": 0.0, "kv_read": 0.25 * G, "kv_write": 0.125 * G, "state": 0.0625 * G, "lookup": 0.0}
    d = M.step_dram_bytes(mp, st, t)
    assert d["weights"] == 0.0 and d["kv_read"] == 0.25 * G and d["kv_write"] == 0.125 * G and d["state"] == 0.0625 * G
    assert M.kv_state_miss(mp) == (1.0, 1.0)


def _ev(sc):
    from accel_dse import api
    from accel_dse.core.evaluate import evaluate
    return evaluate(api.scenario_from_body({"scenario": sc}))


def test_eval_stages_ledger_closes():
    """End to end: every LLM stage's DRAM / SLC bytes and the SRAM share implied by its plan sum to the touched bytes
    the stage recorded (StageResult.ledger), across policies, SRAM sizes, a state buffer and a measured load."""
    load = [1.0 / (i + 1) for i in range(128)]
    cases = [("qwen3-0.6b", {"sram_mib": 4096}, {}), ("qwen3-0.6b", {"slc_mib": 4096, "slc_policy": "pin"}, {}),
             ("qwen3-30b-a3b", {"sram_mib": 4096, "slc_mib": 8192, "slc_policy": "pin"}, {"moe_expert_load": load}),
             ("qwen3-30b-a3b", {"sram_mib": 2048, "slc_mib": 4096, "slc_policy": "dip"}, {"moe_expert_load": load}),
             ("qwen3-30b-a3b", {"slc_mib": 65536, "slc_policy": "lru"}, {}),
             ("qwen3-next-80b-a3b", {"sram_mib": 1024, "state_sram_mib": 64, "slc_mib": 2048, "slc_policy": "pin_bw"}, {}),
             # re-read (rr) weights: the ledger must carry hot_rr (fuzz hit: fireredtts3-base, SRAM 1 GiB, lru)
             ("fireredtts3-base", {"sram_mib": 1024, "slc_mib": 256, "slc_policy": "lru", "state_sram_mib": 16}, {})]
    for model, ch, sv in cases:
        for ph in ("decode", "prefill"):
            r = _ev({"model": model, "chip": ch, "serving": {"phase": ph, "batch": 8, "ctx": 4096, "prompt": 1024, **sv}})
            for s in r.stages:
                store, tch, _, _ = s.ledger
                t = {k: tch.get(k, 0.0) for k in CLASSES}
                t["hot_rr"] = tch.get("hot_rr", 0.0)
                assert store.rr_w <= 0 or t["hot_rr"] > 0, (model, ph)
                _check_ledger(s.mem, store, t, s.dram, (model, ch, ph, s.index))


def test_draft_and_target_share_head_stage_on_chip_tiers():
    """An independent draft model runs on the head stage's cards.  Its bytes are read k times per step, so it takes the
    shared SRAM / state buffer / SLC first (0.192 rule); the target's head stage plans what is left.  ≤ 0.286 both
    were planned on the whole chip (SRAM 4 GiB: draft 4091 MiB + target 4076 MiB resident)."""
    from accel_dse.core.memplan import held
    sv = {"phase": "decode", "batch": 8, "ctx": 4096, "spec_k": 3, "draft_model": "qwen3-0.6b"}
    for ch in ({}, {"sram_mib": 4096}, {"sram_mib": 1024, "state_sram_mib": 256},
               {"slc_mib": 4096, "slc_policy": "pin"}, {"slc_mib": 4096, "slc_policy": "dip"},
               {"slc_mib": 65536, "slc_policy": "lru"}):
        r = _ev({"model": "qwen3-8b", "chip": ch, "serving": sv})
        tm, rv = r.stages[-1].mem, r.draft["resv"]
        h = held(tm)
        sram = (ch.get("sram_mib") or r.scenario.chip.sram_mib) * 2 ** 20
        assert rv["sram_resv"] + h[0] <= sram - tm.staging + 1, (ch, rv, h)
        assert rv["buf_resv"] + h[1] <= r.scenario.chip.state_sram_mib * 2 ** 20 + 1, ch
        assert rv["slc_resv"] + h[2] <= r.scenario.chip.slc_mib * 2 ** 20 + 1, (ch, rv, h)
        assert tm.sram_resv == rv["sram_resv"]
    # the draft takes the SRAM beside the (larger) target staging; the target loses what the draft holds
    big = _ev({"model": "qwen3-8b", "chip": {"sram_mib": 4096}, "serving": sv})
    tm = big.stages[-1].mem
    cut_mib = 4096 - (tm.staging - big.draft["staging"]) / 2 ** 20
    alone = _ev({"model": "qwen3-0.6b", "chip": {"sram_mib": cut_mib}, "serving": {k: v for k, v in sv.items()
                                                                                if k not in ("spec_k", "draft_model")}})
    assert abs(big.draft["pass_s"] - alone.step) < 1e-12
    assert held(alone.stages[0].mem)[0] == big.draft["resv"]["sram_resv"] > 4000 * 2 ** 20
    assert tm.pinned_hot + tm.kv_sram < 2 ** 20
    # no draft → no reservation, identical plans
    plain = _ev({"model": "qwen3-8b", "chip": {"sram_mib": 4096}, "serving": {"phase": "decode", "batch": 8, "ctx": 4096}})
    assert plain.stages[-1].mem.sram_resv == 0.0 and plain.stages[-1].mem.pinned_hot > 4000 * 2 ** 20


def test_lru_draft_thrashes_with_target_sweep():
    """lru serves a cyclic sweep only if everything swept fits: the draft fits alone, the target beside it does not →
    the draft gets no SLC hits either (the cyclic-thrash bound)."""
    sv = {"phase": "decode", "batch": 1, "ctx": 1024, "spec_k": 2, "draft_model": "qwen3-0.6b"}
    r = _ev({"model": "qwen3-8b", "chip": {"slc_mib": 2048, "slc_policy": "lru"}, "serving": sv})
    alone = _ev({"model": "qwen3-0.6b", "chip": {"slc_mib": 2048, "slc_policy": "lru"},
                 "serving": {"phase": "decode", "batch": 1, "ctx": 1024}})
    nos = _ev({"model": "qwen3-0.6b", "chip": {}, "serving": {"phase": "decode", "batch": 1, "ctx": 1024}})
    assert alone.stages[0].mem.slc_all and not r.stages[-1].mem.slc_all
    assert abs(r.draft["pass_s"] - nos.step) < 1e-12 and r.draft["resv"]["slc_resv"] == 0.0


def test_fused_chunk_charged_under_decode_plan():
    """Co-located chunk iteration: the chunk's KV joins the decode batch's pool, so its re-reads / writes miss at the
    decode plan's share.  ≤ 0.286 the chunk used its own plan (one sequence on an empty chip): with 4 GiB SRAM its
    470 MB of KV traffic was charged 0 DRAM, beside a decode batch already holding the leftover SRAM."""
    from accel_dse.core.evaluate import chunk_mem_bytes
    from accel_dse.core.pdqueue import _fused_step

    def pair(ch):
        base = {"model": "qwen3-0.6b", "chip": ch, "serving": {"ctx": 4096, "prompt": 4096}}
        dec = _ev({**base, "serving": {**base["serving"], "phase": "decode", "batch": 64}})
        pre = _ev({**base, "serving": {**base["serving"], "phase": "prefill", "batch": 1, "prefix_cached": 2048}})
        return dec, pre
    dec, pre = pair({})                      # default chip: no KV on chip anywhere → old = new, bit for bit
    sd, sp = dec.stages[0], pre.stages[0]
    c = chunk_mem_bytes(sd, sp, 1.0, 0.5)
    assert c[0] == sp.dram["total"] - sp.dram["weights"] + 0.5 * sp.dram["kv_write"] and c[1] == 0.0 and c[2] == c[3]
    dec, pre = pair({"sram_mib": 4096})
    sd, sp = dec.stages[0], pre.stages[0]
    assert sp.dram["kv_read"] == sp.dram["kv_write"] == 0.0 and sd.mem.kv_sram > 0       # the old double booking
    tk = sp.ledger[1]
    miss = 1.0 - sd.mem.kv_sram / sd.mem.kv_total
    dram, slc, kv, _ = chunk_mem_bytes(sd, sp, 1.0)
    assert abs(kv - (tk["kv_read"] + tk["kv_write"]) * miss) < 1.0 and slc == 0.0
    assert _fused_step(dec, pre, 1.0, 0.0) > dec.step
    dec, pre = pair({"slc_mib": 4096, "slc_policy": "pin"})
    sd, sp = dec.stages[0], pre.stages[0]
    dram, slc, kv, _ = chunk_mem_bytes(sd, sp, 1.0)
    f_kv = sd.mem.slc_kv / (sd.mem.kv_total + sd.mem.state_total - sd.mem.kv_sram)
    assert abs(slc - sp.ledger[1]["kv_read"] * f_kv) < 1.0 and slc < sp.dram["slc_parts"]["kv_read"]


def test_ornith_1_5_122b_catalog_entry():
    """Freshness: ornith-ai/Ornith-1.5-122B (public 2026-10-20, MIT) — qwen3_5_moe as released bf16, exact params from
    the safetensors headers, Qwen3.5 image / video processor sizes from its own preprocessor configs, its own family
    (no Qwen3.5-122B-A10B entry to merge into)."""
    from accel_dse import api
    from accel_dse.core.catalog import MERGED, get_model, list_models
    from accel_dse.core.model import load_release
    rel = load_release("ornith-ai/Ornith-1.5-122B")
    assert rel["params_llm"] == 122111526912 and rel["params_mtp"] == 2523679232 and rel["params_vision"] == 451290864
    assert set(rel["roles"]["expert"]) == {"bf16"} and rel["quantization_config"] is None
    tc = rel["config"]["text_config"]
    assert (tc["hidden_size"], tc["num_hidden_layers"], tc["num_experts"], tc["num_experts_per_tok"]) == (3072, 48, 256, 8)
    x = {e["id"]: e for e in list_models()}["ornith-1.5-122b"]
    assert x["provider"] == "ornith" and x["family"] == "Ornith-1.5" and x["domain"] == "vlm" and x["coverage"] == "full"
    assert abs(x["param_err"]) < 1e-9 and "ornith-1.5-122b" not in MERGED and get_model("ornith-1.5-122b").vision
    base = {"model": "ornith-1.5-122b", "layout": {"tp": 4, "ep": 4},
            "serving": {"phase": "prefill", "batch": 1, "prompt": 1024}}
    r0 = api.api_eval({"scenario": base})
    r1 = api.api_eval({"scenario": {**base, "serving": {**base["serving"], "images": 1}}})
    v = r1["summary"]["vision"]
    assert v["image_tokens"] == 1024 and v["s"] > 0       # 1024² image, min/max px 65536 / 16777216, patch 16, merge 2
    assert r1["summary"]["ttft_ms"] > r0["summary"]["ttft_ms"]


def test_draft_sram_cut_rerun_keeps_slc_share():
    """When the target's staging is larger than the draft's, the draft is re-planned beside it with less SRAM; the
    bytes that pushes off the SRAM may not grow the draft's SLC share, which the target's plan already assumed (fuzz:
    SLC over-committed by exactly the staging difference, e.g. 7.5 MB here, 181 MB on qwen3.8-flash-next b256)."""
    from accel_dse.core.memplan import held
    for model in ("qwen3-8b", "qwen3-30b-a3b"):
        r = _ev({"model": model, "chip": {"name": "1P", "sram_mib": 256, "slc_mib": 4096, "slc_policy": "pin"},
                 "serving": {"phase": "decode", "batch": 4, "ctx": 4096, "spec_k": 3, "draft_model": "qwen3-0.6b"}})
        m, rv, c = r.stages[-1].mem, r.draft["resv"], r.scenario.chip
        assert m.staging > r.draft["staging"], model           # the re-run path is taken
        assert rv["slc_resv"] + held(m)[2] <= c.slc_mib * 2**20 + 1, model
        assert rv["sram_resv"] + held(m)[0] <= c.sram_mib * 2**20 - m.staging + 1, model
