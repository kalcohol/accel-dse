"""0.288 (MODEL §233): the Ornith family (ornith-ai) and the memory-tier ledger beyond LLM stages."""
from accel_dse import api
from accel_dse.core.catalog import MERGED, entries, get_model, list_models
from accel_dse.core.model import load_release

ORNITH_LISTED = {"ornith-1.5-122b": "bf16", "ornith-1.5-122b-fp8": "fp8", "ornith-1.5-397b-fp8": "fp8",
                 "ornith-1.5-35b-a3b-fp8": "fp8"}
ORNITH_MERGED = {"ornith-1.5-397b": "qwen3.5-397b-a17b", "ornith-1.5-35b-a3b": "startlux-decision-v1-35b-a3b",
                 "ornith-1.5-9b": "qwen3.5-9b", "ornith-1.0-397b": "qwen3.5-397b-a17b",
                 "ornith-1.0-35b": "startlux-decision-v1-35b-a3b", "ornith-1.0-9b": "qwen3.5-9b",
                 "ornith-1.0-397b-fp8": "ornith-1.5-397b-fp8", "ornith-1.0-35b-fp8": "ornith-1.5-35b-a3b-fp8"}


def _shapes(rel):
    tc = rel["config"].get("text_config", rel["config"])
    keys = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "intermediate_size", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
            "shared_expert_intermediate_size", "layer_types", "linear_num_value_heads", "linear_num_key_heads",
            "vocab_size")
    return {k: tc.get(k) for k in keys}, rel["params_llm"], rel["params_vision"]


def test_ornith_entries_exact_and_merged_by_architecture():
    idx = {e["id"]: e for e in entries()}
    listed = {x["id"]: x for x in list_models()}
    for mid, w in ORNITH_LISTED.items():
        x = listed[mid]
        assert x["provider"] == "ornith" and x["family"] == "Ornith-1.5" and x["coverage"] == "full", mid
        assert abs(x["param_err"]) < 1e-9 and set(load_release(idx[mid]["hf_id"])["roles"]["expert"]) == {w}, mid
    for mid, base in ORNITH_MERGED.items():
        assert MERGED[mid] == base and mid not in listed, mid
        a, b = load_release(idx[mid]["hf_id"]), load_release(idx[base]["hf_id"])
        assert _shapes(a) == _shapes(b), mid                    # same architecture → merged, not a new point
        assert a["params_mtp"] in (0, b["params_mtp"]), mid      # 1.0 ships no MTP head
        assert get_model(mid).vision.params == a["params_vision"], mid
    # FP8 = compressed-tensors W8A8 float: experts / full-attention fp8, linear attention / MTP / ViT bf16
    q = load_release("ornith-ai/Ornith-1.5-397B-FP8")
    assert q["quantization_config"]["quant_method"] == "compressed-tensors"
    assert set(q["roles"]["mtp"]) == {"bf16"} and set(q["roles"]["vision"]) == {"bf16"}
    assert set(q["roles"]["attn"]) == {"bf16", "fp8"} and set(q["roles"]["shared_expert"]) == {"fp8"}


def test_ornith_fp8_halves_expert_bytes_and_uses_fp8_inputs():
    def ev(m):
        return api.api_eval({"scenario": {"model": m, "layout": {"tp": 1},
                                          "serving": {"phase": "decode", "batch": 8, "ctx": 4096}}})["summary"]
    b, f = ev("ornith-1.5-35b-a3b"), ev("ornith-1.5-35b-a3b-fp8")
    assert f["dram_need_GiB"] < 0.6 * b["dram_need_GiB"] and f["tpot_ms"] < b["tpot_ms"]
    assert get_model("ornith-1.5-35b-a3b-fp8").act_fmt_for("expert") == "fp8"


# ------------------------------------------------------------------------------------- ledger beyond LLM stages
import random

from test_core_287 import CLASSES, _check_ledger


def _ev(sc):
    from accel_dse.core.evaluate import evaluate
    return evaluate(api.scenario_from_body({"scenario": sc}))


def _stage_ledger(stages, where):
    for s in stages or ():
        assert s.ledger is not None, where
        store, tch, _, _ = s.ledger
        t = {k: tch.get(k, 0.0) for k in CLASSES}
        t["hot_rr"] = tch.get("hot_rr", 0.0)
        t["w_extra"] += s.dram.get("fsdp_gather", 0.0)          # the FSDP gather write is demand of its own
        _check_ledger(s.mem, store, t, s.dram, where)


def test_domain_stage_ledger_closes():
    """DiT (video / image, incl. the cached-prefix and FSDP paths), protein and genomics trunks go through the same
    plan → step_dram_bytes path as LLM stages; their per-tier bytes close per class."""
    chips = ({"sram_mib": 4096}, {"slc_mib": 4096, "slc_policy": "pin"}, {"sram_mib": 1024, "slc_mib": 1024,
             "slc_policy": "dip"}, {"slc_mib": 16384, "slc_policy": "lru"},
             {"sram_mib": 16384, "slc_mib": 8192, "slc_policy": "pin_bw", "state_sram_mib": 64})
    cases = (("wan2.1-1.3b", {"sp": 2, "dp": 2}, {"dit_fsdp": True}), ("qwen-image-2.1-turbo", {}, {}),
             ("flux.2-klein-4b", {"pp": 2}, {}), ("esm2-650m", {}, {}), ("carbon-a-1.2b", {}, {}))
    for model, lay, wl in cases:
        for ch in chips:
            sc = {"model": model, "chip": ch, "layout": lay, "serving": {"batch": 2}, **({"workload": wl} if wl else {})}
            r = _ev(sc)
            _stage_ledger(r.stages, (model, ch))
            _stage_ledger(getattr(r, "stages_u", None), (model, ch, "uncond"))


def test_component_charges_every_touched_byte():
    """A component (text encoder, VAE, vision / audio encoder, speech-card module) has no on-chip credit: every
    touched byte — weights, activations, K/V, state, lookups — is DRAM traffic inside its roofline.  ≤ 0.287 the
    Talker's 22 MB of codec-embedding lookups per prefill (b 4) were charged to no tier and its K/V seconds were added
    after the roofline."""
    from types import SimpleNamespace
    from accel_dse.core.evaluate import _COMP_KEYS, _component_time, _sum_ops
    from accel_dse.core.hardware import CHIPS, System
    from accel_dse.core.omni import talker_ops
    sys_ = System(CHIPS["1P"])
    o = get_model("qwen2.5-omni-7b").omni
    ops = talker_ops(o, 4, 512, 0)
    d = _sum_ops(ops, sys_, "os", SimpleNamespace(act_fmt="bf16", kv_fmt="bf16"))
    assert d["lookup"] > 0 and d["kv_read"] + d["kv_write"] > 0
    dem = sum(d[k] for k in _COMP_KEYS)
    c = _component_time([(ops, 1)], sys_, "os", "bf16")
    assert abs(c["dram_GB"] * 1e9 - dem) <= 1e-6 * dem
    c2 = _component_time([(ops, 1)], sys_, "os", "bf16", extra_dram=1e9)
    assert abs(c2["dram_GB"] * 1e9 - dem - 1e9) <= 1e-6 * dem and c2["s"] >= c["s"]


def test_speech_card_pin_shares_close():
    """Omni speech card: DiT weights SRAM-pinned (re-read rule), the SRAM-miss share then the Talker in the SLC
    (omni_slc_plan) — per re-read evaluation SRAM + SLC + DRAM = 1, the SLC holds ≤ its size; CosyVoice / Kimi
    flow / DiT pins are a share in [0, 1] for any SRAM (incl. 0)."""
    from accel_dse.core.cosyvoice import cosy_pin
    from accel_dse.core.kimi_audio import kimi_pin
    from accel_dse.core.omni import omni_pin, omni_slc_plan, talker_weight_bytes
    R = random.Random(288)
    o = get_model("qwen2.5-omni-7b").omni
    wd, wt = o.dit_run_params * 4.0, talker_weight_bytes(o)
    c, k = get_model("cosyvoice2-0.5b").cosy, get_model("kimi-audio-7b-instruct").kimi
    for _ in range(2000):
        sram = R.choice([0.0, R.uniform(0, 2 * wd), R.uniform(0, 64 * 2**20)])
        slc = R.choice([0.0, R.uniform(0, 2 * (wd + wt))])
        b = R.randint(1, 64)
        miss = omni_pin(o, sram)
        hd, ht = omni_slc_plan(o, miss, slc, b)
        assert 0.0 <= miss <= 1.0 and 0.0 <= hd <= 1.0 and 0.0 <= ht <= 1.0
        sram_s, slc_s, dram_s = 1.0 - miss, miss * hd, miss * (1.0 - hd)
        assert min(sram_s, slc_s, dram_s) >= 0.0 and abs(sram_s + slc_s + dram_s - 1.0) < 1e-12
        assert hd * miss * wd + ht * wt <= slc + 1.0
        assert 0.0 <= cosy_pin(c, sram) <= 1.0 and 0.0 <= kimi_pin(k, sram) <= 1.0


def test_insight_panels_stale_tracked():
    """0.288 (AGENTS §236 deferred item): the design-insight panels (insight, SLC size, sizing, upgrade path) are in
    invalidate()'s stale list and tag a result computed for an older scenario (staleTag of the click-time version);
    the 0.286 guard (test_core_286) then checks their success and catch paths set dataset.done."""
    from tests import webstruct as W
    js = W.read()
    ids = {"ins-status", "ins-slc-status", "ins-size-status", "ins-path-status"}
    assert ids <= set(W.stale_tracked(js))
    fns = W.functions(js)
    for fn in ("runInsight", "runInsightSlc", "runInsightSizing", "runInsightPath"):
        assert W.has(fns[fn], "const v0 = S.ver ;") and W.mentions(fns[fn], "staleTag"), fn
