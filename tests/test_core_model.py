"""V1 — model specs vs official releases (params, formats, active params)."""
from __future__ import annotations

from accel_dse.core.catalog import entries, get_model, labels
from accel_dse.core.ir import Phase, build_rank_ops, embed_ops, head_ops


def test_v1_params_within_2pct_of_safetensors():
    bad = []
    for e in entries():
        s = get_model(e["id"])
        pc = s.param_check()
        assert pc["release"], e["id"]
        if abs(pc["rel_err"]) > 0.02:
            bad.append((e["id"], pc["rel_err"]))
    assert not bad, bad


def test_v1_catalog_covers_all_fetched_llms():
    ids = {e["id"] for e in entries()}
    for must in ("deepseek-v3", "qwen3-32b", "qwen3-8b", "gpt-oss-120b", "kimi-k2", "glm-4.5", "mixtral-8x22b",
                 "qwen3-32b-fp8", "qwen3-8b-awq", "llama-3.1-8b"):
        assert must in ids, must
    assert len(ids) >= 50


def test_v1_active_params_vs_model_cards():
    # (id, official active params B, source) — compare excl. embedding table (cards differ on convention)
    claims = [("deepseek-v3", 37.0), ("qwen3-30b-a3b", 3.3), ("qwen3-235b-a22b", 22.0), ("gpt-oss-120b", 5.1),
              ("gpt-oss-20b", 3.6), ("kimi-k2", 32.0), ("glm-4.5", 32.0), ("mixtral-8x22b", 39.0)]
    for mid, b in claims:
        s = get_model(mid)
        act = s.active_params() / 1e9
        act_x = (s.active_params() - s.embed_params) / 1e9
        err = min(abs(act - b), abs(act_x - b)) / b
        assert err <= 0.05, (mid, act, act_x, b)


def test_release_formats_as_released():
    f = lambda mid, role: get_model(mid).fmt(role)
    assert f("deepseek-v3", "expert").fmt == "fp8" and 8.0 <= f("deepseek-v3", "expert").bits < 8.1
    assert get_model("deepseek-v3").act_fmt == "fp8"          # W8A8 dynamic
    assert f("gpt-oss-120b", "expert").fmt == "mxfp4" and abs(f("gpt-oss-120b", "expert").bits - 4.25) < 0.01
    assert f("gpt-oss-120b", "attn").fmt == "bf16"
    assert f("qwen3-32b-awq", "mlp").fmt == "int4" and abs(f("qwen3-32b-awq", "mlp").bits - (4 + 20 / 128)) < 0.01
    assert f("kimi-k2.5", "expert").fmt == "int4" and abs(f("kimi-k2.5", "expert").bits - 4.5) < 0.01
    assert f("qwen3-32b", "mlp").fmt == "bf16" and get_model("qwen3-32b").act_fmt == "bf16"
    assert get_model("qwen3-32b-fp8").act_fmt == "fp8"
    # official quantised variants are separate entries with identical architecture
    a, b = get_model("qwen3-32b"), get_model("qwen3-32b-fp8")
    assert a.params() == b.params() and a.id != b.id


def test_three_axis_labels_and_proxy_badge():
    for e in entries():
        lab = labels(get_model(e["id"]))
        assert lab["provenance"] in ("official", "mirror")
        assert lab["coverage"] in ("full", "partial", "proxy")
        assert lab["proxy_badge"] == (lab["coverage"] == "proxy")
        assert lab["dtype"].startswith("W ")
    assert labels(get_model("deepseek-v4.1-flash"))["proxy_badge"]   # Unreleased: V4 / V4-Pro now 「完整」
    assert labels(get_model("llama-3.1-8b"))["provenance"] == "mirror"
    assert labels(get_model("qwen3-8b"))["coverage"] == "full"


def _indep_decode_flops(s, ctx):
    """Independent count: 2·(non-embedding active params) + attention 4·n_q·d·ctx per layer (GQA)."""
    lin = s.active_params() - s.embed_params - sum(l.misc_params for l in s.layers) - s.hidden
    att = sum(2 * l.core.n_q * (l.core.qk_dim + l.core.v_dim) * min(ctx, l.core.window or ctx) for l in s.layers)
    return 2 * lin + att


def test_v1_flops_per_token_independent_count_dense_and_moe():
    for mid in ("qwen3-8b", "qwen3-32b", "llama-3.1-8b", "mistral-large", "phi-4", "qwen3-30b-a3b"):
        s = get_model(mid)
        ctx = 2048
        ph = Phase("decode", 1, 1, ctx - 1)
        ops = embed_ops(s, ph) + [o for li in range(s.n_layers) for o in build_rank_ops(s, li, ph)] + head_ops(s, ph)
        ours = sum(o.flops for o in ops)
        ref = _indep_decode_flops(s, ctx)
        if s.is_moe:
            # one token touches exactly top_k experts → our op graph = active-param count
            pass
        assert abs(ours - ref) / ref < 0.005, (mid, ours, ref)


def test_mla_decode_flops_absorbed_form():
    s = get_model("deepseek-v3")
    ctx = 4096
    ph = Phase("decode", 1, 1, ctx - 1)
    ops = build_rank_ops(s, 5, ph)
    c = s.layers[5].core
    att = sum(o.flops for o in ops if o.kind == "attn")
    # absorbed MLA: scores over (kv_lora + rope), output in kv_lora space, per head
    ref = 2 * c.n_q * ctx * (c.kv_lora + c.rope_dim) + 2 * c.n_q * ctx * c.kv_lora
    assert abs(att - ref) / ref < 1e-9
    lin = sum(o.flops for o in ops if o.kind == "gemm" and not o.name.startswith("expert"))
    assert lin > 0


def test_lookup_tables_stored_not_active():
    """n-gram / engram lookup tables count toward stored params but not toward params touched per token."""
    from accel_dse.core.catalog import get_model
    s = get_model("deepseek-v4.1-flash")
    assert s.lookup_params > 1e11
    assert s.params() - s.active_params() > s.lookup_params
    assert s.active_params() < 0.05 * s.params()


def test_catalog_listing_grouping_coverage_and_merges():
    """Listing: every model carries a coverage level, non-full ⇔ concrete reasons; grouped vendor → family (LLM and
    VLM of one vendor together, domain only a badge) with sizes non-increasing; catalog-only video / protein rows sit
    under their vendor after its evaluable rows; merged entries really have the same structure and dtype."""
    from accel_dse.core.catalog import MERGED, catalog_listing, dtype_label, list_models, offline_entries
    ms = list_models()
    seen, prev = [], None
    for m in ms:
        assert m["coverage"] in ("full", "partial", "proxy")
        assert (m["coverage"] == "full") == (not m["coverage_reasons"]), m["id"]
        assert (m["domain"] == "vlm") == (m["vision_params_B"] > 0), m["id"]
        if m["provider"] != (prev and prev["provider"]):
            assert m["provider"] not in seen, m["provider"]      # each vendor group is contiguous
            seen.append(m["provider"])
        elif prev["family"] == m["family"]:
            assert m["params_B"] <= prev["params_B"] + 1e-9, (prev["id"], m["id"])
        prev = m
    assert {"llm", "vlm", "gen", "protein"} == {m["domain"] for m in ms}
    for v in ("alibaba", "deepseek", "kimi", "zhipu"):                # text + VLM releases share one vendor group
        assert {"llm", "vlm"} <= {m["domain"] for m in ms if m["provider"] == v}, v
    listed = {m["id"] for m in ms}
    for a, b in MERGED.items():
        assert a not in listed and b in listed
        sa, sb = get_model(a), get_model(b)
        assert sa.params() == sb.params() and dtype_label(sa) == dtype_label(sb), (a, b)
    off = offline_entries()
    ids = {o["id"] for o in off}
    assert {o["domain"] for o in off} <= {"gen", "protein"} and not (ids & listed)
    assert ids == {"alphafold3"}                                      # 0.43: weights gated → the only catalog-only row
    live = {"wan2.1-14b", "wan2.1-1.3b", "cogvideox-5b", "cogvideox-2b", "esm2-3b", "esm2-650m",
            "minimax-h3", "hunyuanvideo", "wan2.2-a14b", "ltx-video", "mochi-1", "opensora-stdit3",
            "esmfold", "alphafold2", "openfold", "boltz-1", "protenix"}
    assert live <= listed and not (live & ids)              # 0.41–0.43: release-backed video / protein models
    for i in live:
        assert get_model(i).domain in ("gen", "protein") and not get_model(i).kv_cache
    assert all(o["provider"] != "other" and not o["evaluable"] for o in off)
    for o in off:                                                     # catalog-only rows are not resolvable
        try:
            get_model(o["id"])
            raise AssertionError(o["id"])
        except KeyError:
            pass
    cat = catalog_listing()
    assert [c["id"] for c in cat if c["evaluable"]] == [m["id"] for m in ms] and len(cat) == len(ms) + len(off)
    seen, prev = [], None
    for c in cat:
        if c["provider"] != (prev and prev["provider"]):
            assert c["provider"] not in seen, c["provider"]
            seen.append(c["provider"])
        else:
            assert c["evaluable"] <= prev["evaluable"], (prev["id"], c["id"])   # evaluable rows first in a vendor
        prev = c
    vend = {c["id"]: c["provider"] for c in cat}
    assert vend["wan2.1-14b"] == vend["qwen3-8b"] and vend["esm2-3b"] == vend["llama-3.1-8b"]
    assert vend["cogvideox-5b"] == vend["glm-4.6"] and vend["protenix"] == vend["seed-oss-36b"]
    # Unreleased: chunked linear attention per the reference kernels → 「完整」 (was 「部分」)
    assert labels(get_model("kimi-k3"))["coverage"] == "full"
    assert labels(get_model("qwen3-next-80b-a3b"))["coverage_reasons"] == []
