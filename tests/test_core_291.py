"""0.291 (MODEL §236): Perplexity pplx-decider decision checkpoints (readout head instead of an lm_head, noncausal full
attention, joint requests with one readout per question), Jais 3 (k2_horizon: BF16 originals + IFM FP8) and Mellum2.1
Thinking — exact parameter checks against the public releases."""
from accel_dse import api
from accel_dse.core.catalog import entries, get_model, list_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, head_ops
from accel_dse.core.model import load_release
from accel_dse.core.vision import image_grid, vision_spec

DECIDERS = {"pplx-decider-v1.2": ("perplexity-ai/pplx-decider-v1.2", 5120, True, 1638400),
            "pplx-decider-flash-v1.2": ("perplexity-ai/pplx-decider-flash-v1.2", 2560, True, 1638400),
            "pplx-decider-v1.1-27b": ("perplexity-ai/pplx-decider-v1.1-27b", 5120, True, 4194304),
            "pplx-decider-v1-27b": ("perplexity-ai/pplx-decider-v1-27b", 5120, False, 262144)}
JAIS = ("jais-3-375b-a23b", "jais-3-32b", "jais-3-7b", "jais-3-3.7b", "jais-3-0.9b",
        "jais-3-375b-a23b-fp8", "jais-3-32b-fp8", "jais-3-7b-fp8", "jais-3-3.7b-fp8")
MELLUM = "mellum2.1-12b-a2.5b-thinking"


def _dec(model, q=1, labels=4, batch=2, prompt=1050):
    return evaluate(api.scenario_from_body({"scenario": {"model": model, "serving": {
        "phase": "prefill", "batch": batch, "prompt": prompt, "labels": labels, "questions": q}}}))


def test_entries_exact_listed_and_grouped():
    idx = {e["id"]: e for e in entries()}
    listed = {x["id"]: x for x in list_models()}
    for mid in list(DECIDERS) + list(JAIS) + [MELLUM]:
        assert mid in listed and abs(listed[mid]["param_err"]) < 1e-12, mid
        assert get_model(mid).coverage == "full", mid
    for mid, (hf, *_r) in DECIDERS.items():
        assert idx[mid]["hf_id"] == hf and idx[mid]["provider"] == "perplexity" and idx[mid]["decision"], mid
        assert idx[mid]["domain"] == "vlm"
    assert {idx[m]["family"] for m in DECIDERS} == {"pplx-decider v1.2", "pplx-decider v1 / v1.1"}
    assert all(idx[m]["provider"] == "inception" and idx[m]["family"] == "Jais 3" for m in JAIS)
    assert idx[MELLUM]["provider"] == "jetbrains" and not idx[MELLUM]["decision"]
    assert "jais-3-mova-36b-a4b-fp8" not in idx and "jais-3-mova-36b-a4b" not in idx      # MoVA not modelled


def test_decider_readout_replaces_lm_head():
    # readout.safetensors BF16 [255, hidden] (header); no lm_head in the weight index; tied embeddings (Flash) stay tied
    for mid, (hf, h, nc, _px) in DECIDERS.items():
        rel = load_release(hf)
        m = get_model(mid)
        assert rel["decision"]["readout"] == [255, h] and rel["roles"]["lm_head"] == {"bf16": {"params": 255 * h,
                                                                                                "bytes": 2 * 255 * h}}
        assert m.readout_rows == 255 and m.head_params == 255 * h and m.params() == rel["params_llm"]
        full = [L for L in m.layers if L.core is not None and L.core.kind == "gqa"]
        assert len(full) == m.n_layers // 4 and all(L.core.causal is (not nc) for L in full), mid
        assert rel["decision"]["attention_mode"] == ("noncausal_full_attention" if nc else "causal")
    assert get_model("pplx-decider-flash-v1.2").tie_embeddings


def test_readout_ops_rows_and_questions():
    m = get_model("pplx-decider-v1.2")
    ops = {o.name: o for o in head_ops(m, Phase("prefill", 3, 1000, labels=5, questions=7))}
    lh = ops["label_head"]
    assert (lh.m, lh.k, lh.n) == (21, 5120, 5) and "lm_head" not in ops
    # outside a decision readout the whole readout is projected (no vocab head exists)
    ops = {o.name: o for o in head_ops(m, Phase("decode", 4, 1, 512))}
    assert ops["label_head"].n == 255 and "lm_head" not in ops


def test_joint_requests_answer_more_questions_per_forward():
    one, joint = _dec("pplx-decider-v1.2", 1).decision(), _dec("pplx-decider-v1.2", 8).decision()
    assert joint["questions"] == 8 and one["questions"] == 1
    assert abs(joint["latency_ms"] / one["latency_ms"] - 1) < 1e-3          # 8 readouts of 4 rows: negligible
    assert abs(joint["decisions_per_s"] / one["decisions_per_s"] - 8) < 1e-2
    assert "联合请求" in joint["basis"] and "联合请求" not in one["basis"]


def test_noncausal_full_attention_costs_more_than_causal():
    # v1.1 and v1 share the backbone and dtypes; only the full-attention mask differs
    a = _dec("pplx-decider-v1.1-27b", prompt=16384, batch=1)
    b = _dec("pplx-decider-v1-27b", prompt=16384, batch=1)
    assert a.ttft > b.ttft


def test_decision_validation():
    for sv, frag in (({"phase": "prefill", "questions": 2}, "serving.questions"),
                     ({"phase": "prefill", "labels": 4, "questions": 0}, "serving.questions"),
                     ({"phase": "prefill", "labels": 4, "questions": True}, "serving.questions")):
        try:
            api.scenario_from_body({"scenario": {"model": "pplx-decider-v1.2", "serving": sv}})
            raise AssertionError(sv)
        except (ValueError, api.ApiError) as e:
            assert frag in str(e).replace("Serving.", "serving."), (sv, e)
    try:
        _dec("pplx-decider-v1.2", labels=256)
        raise AssertionError("labels > readout rows")
    except ValueError as e:
        assert "255" in str(e)
    _dec("pplx-decider-v1.2", labels=255)
    sc = api.scenario_from_body({"scenario": {"model": "pplx-decider-v1.2", "serving": {
        "phase": "prefill", "labels": 4, "questions": 3}}})
    assert sc.to_dict()["serving"]["questions"] == 3
    assert "questions" not in api.scenario_from_body({"scenario": {"model": "pplx-decider-v1.2", "serving": {
        "phase": "prefill", "labels": 4}}}).to_dict()["serving"]


def test_decider_image_budget_from_processor_config():
    # processor_config.json longest_edge: v1.2 / Flash 1638400 px (card: "a 1.6 MP image uses about 1,600 positions")
    for mid, (hf, _h, _nc, px) in DECIDERS.items():
        g = image_grid(vision_spec(hf, load_release(hf)["config"]), 4096, 4096)
        assert g.px[0] * g.px[1] <= px and (g.px[0] + 32) * (g.px[1] + 32) > px, mid
    assert image_grid(vision_spec(DECIDERS["pplx-decider-v1.2"][0], load_release(DECIDERS["pplx-decider-v1.2"][0])
                                  ["config"]), 2048, 2048).llm_tokens == 1600


def test_jais3_k2_horizon_shared_experts_and_router_bias():
    m = get_model("jais-3-375b-a23b")
    moe = [L for L in m.layers if L.ffn.kind == "moe"]
    assert len(moe) == 58 and [L.ffn.kind for L in m.layers[:3]] == ["dense"] * 3      # mlp_only_layers [0, 1, 2]
    f = moe[0].ffn
    assert (f.n_experts, f.top_k, f.d_expert, f.n_shared, f.d_shared) == (192, 8, 1792, 1, 1792)
    rel = load_release("inception42/Jais-3-375B-A23B")
    assert sum(v["params"] for v in rel["roles"]["shared_expert"].values()) == 58 * 3 * 6144 * 1792
    assert all(get_model(x).act_fmt == "fp8" for x in JAIS if x.endswith("-fp8"))
    assert all(get_model(x).act_fmt == "bf16" for x in JAIS if not x.endswith("-fp8"))
    fp8 = dict(get_model("jais-3-375b-a23b-fp8").formats)
    assert fp8["expert"].fmt == "fp8" and fp8["attn"].fmt == "bf16" and fp8["shared_expert"].fmt == "bf16"
    assert dict(get_model("jais-3-7b-fp8").formats)["attn"].fmt == "fp8"


def test_mellum_qk_norm_window_and_moe():
    m = get_model(MELLUM)
    c = load_release("JetBrains/Mellum2.1-12B-A2.5B-Thinking")["config"]
    assert m.layers[0].misc_params == 2 * c["hidden_size"] + 2 * c["head_dim"]          # q_norm + k_norm [128] each
    win = [L.core.window for L in m.layers]
    assert win.count(1024) == 21 and win.count(None) == 7 and win[3] is None
    f = m.layers[0].ffn
    assert (f.n_experts, f.top_k, f.d_expert) == (64, 8, 896) and m.kv_fmt == "bf16" and m.act_fmt == "bf16"


# ---- 0.291 scope 2: tests/runcache.py (per-run report cache shared across test processes) ----

def _rc_toy(x, scale=1.0):
    return {"v": [x * scale, x], "n": _RC_CALLS.append(1) or len(_RC_CALLS)}


_RC_CALLS: list = []


def test_runcache_hit_is_an_equal_independent_copy():
    import tests.runcache as rc
    f = rc.cached(_rc_toy)
    a = f(3.5, scale=2.0)
    b = f(3.5, scale=2.0)
    assert a == b and a is not b and len(_RC_CALLS) == 1          # second call served from the cache
    b["v"].append(9)
    assert f(3.5, scale=2.0) == a                                 # mutating a hit does not leak into the cache
    assert f(3.5, scale=3.0) != a and len(_RC_CALLS) == 2         # any argument change is a new key
    assert f.uncached is _rc_toy


def test_runcache_key_tracks_model_state():
    import contextvars
    import tests.runcache as rc
    from accel_dse.core import disagg as dg, evaluate as E, mapping as mp
    with E.model_flags():                                         # loads the modules model_flags touches
        pass
    s0 = rc.state()
    with E.model_flags(**E.LEGACY_0192, FUSE_QKV=False):         # pin_legacy flags
        assert rc.state() != s0
    assert rc.state() == s0
    tok = dg.CTX_SINK.set({})                                     # pdsim.capture_ctx sink
    try:
        assert rc.state() != s0
    finally:
        dg.CTX_SINK.reset(tok)
    old = mp._gemm_cost if hasattr(mp, "_gemm_cost") else None
    orig = E.evaluate
    E.evaluate = lambda *a, **k: orig(*a, **k)                    # a monkeypatched module function
    try:
        assert rc.state() != s0
    finally:
        E.evaluate = orig
    assert rc.state() == s0 and old is getattr(mp, "_gemm_cost", None)
    assert isinstance(dg.CTX_SINK, contextvars.ContextVar)
    assert len(rc.code_version()) == 64


def test_runcache_code_version_covers_sources_and_data():
    import pathlib
    import tempfile
    import tests.runcache as rc
    with tempfile.TemporaryDirectory() as d:
        r = pathlib.Path(d)
        (r / "core").mkdir(); (r / "data").mkdir()
        (r / "core" / "m.py").write_text("X = 1\n"); (r / "data" / "c.json").write_text("{}")
        h0 = rc._tree_hash(r)
        (r / "data" / "c.json").write_text("{ }")                   # a data byte changes the version
        h1 = rc._tree_hash(r)
        (r / "core" / "m.py").write_text("X = 2\n")                # so does a source byte
        h2 = rc._tree_hash(r)
        assert len({h0, h1, h2}) == 3
    assert rc.code_version() == rc._tree_hash(rc._ROOT)


def test_runcache_disk_shared_across_processes_and_unpicklable_passthrough():
    import os
    import subprocess
    import sys
    import tempfile
    import tests.runcache as rc
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, rc.ENV: d}
        code = ("import statistics, tests.runcache as rc; print(rc.cached(statistics.fmean)([1.0, 2.0, 4.5]));"
                "print(rc.STATS['miss'])")
        out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True,
                             check=True).stdout.split()
        assert out == [repr(7.5 / 3), "1"]
        ls = os.listdir(d)
        entries = [f for f in ls if "." not in f]
        assert len(entries) == 1 and f"{entries[0]}.s" in ls                # result + its compute seconds
        assert not [f for f in ls if f.endswith((".lock", ".tmp"))]       # lock released, write was atomic
        assert [f for f in ls if f.startswith("stats.")]                 # the child left its counters
        import statistics
        was, os.environ[rc.ENV] = os.environ.get(rc.ENV), d
        try:
            h = rc.STATS["hit_disk"]
            assert rc.cached(statistics.fmean)([1.0, 2.0, 4.5]) == 7.5 / 3 and rc.STATS["hit_disk"] == h + 1
        finally:
            if was is None:
                os.environ.pop(rc.ENV)
            else:
                os.environ[rc.ENV] = was
    u = rc.STATS["uncached"]
    g = rc.cached(lambda x: (lambda: x))                          # unpicklable result: returned, not cached
    assert g(1)() == 1 and rc.STATS["uncached"] == u + 1
