"""0.191: decision-model readout (serving.labels) + StartLux / Microsoft-Decision-1 proxy; RWKV-7 G1k family."""
from accel_dse.core import catalog as C
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, head_ops, build_rank_ops
from accel_dse.core.scenario import Scenario

RW = ("rwkv7-g1k-1.5b", "rwkv7-g1k-2.9b", "rwkv7-g1k-7.2b", "rwkv7-g1k-13.3b")
SL = [e["id"] for e in C.entries() if e["id"].startswith("startlux-decision")]


def _s(m, **kw):
    s = Scenario().replace("model", m)
    if "serving__phase" in kw:
        s = s.replace("serving.phase", kw.pop("serving__phase"))
    for k, v in kw.items():
        s = s.replace(k.replace("__", "."), v)
    return s


# ---------------------------------------------------------------- RWKV-7
def test_rwkv_params_exact_and_config():
    want = {"rwkv7-g1k-1.5b": (24, 2048, 8192, (96, 96, 64, 256)), "rwkv7-g1k-2.9b": (32, 2560, 10240, (96, 96, 64, 320)),
            "rwkv7-g1k-7.2b": (32, 4096, 16384, (128, 128, 96, 480)),
            "rwkv7-g1k-13.3b": (61, 4096, 16384, (192, 192, 128, 384))}
    for mid, (L, h, f, lr) in want.items():
        m = C.get_model(mid)
        pc = m.param_check()
        assert pc["rel_err"] == 0.0 and m.coverage == "full" and not m.kv_cache is False, mid
        assert m.n_layers == L and m.hidden == h and m.vocab == 65536 and not m.tie_embeddings
        c = m.layers[1].core
        assert c.kind == "linear" and c.lin == "rwkv7" and c.n_state_heads == h // 64 and c.state_dk == c.state_dv == 64
        assert m.layers[1].ffn.d_ff == f and not m.layers[1].ffn.gated
        names = {l.name: l for l in m.layers[1].attn_linears}
        assert (names["rwkv_w1"].n, names["rwkv_a1"].n, names["rwkv_v1"].n, names["rwkv_g1"].n) == lr
        assert "rwkv_v1" not in {l.name for l in m.layers[0].attn_linears}     # layer 0 makes v_first


def test_rwkv_decode_ctx_invariant():
    """Adversarial: no KV cache → decode time, DRAM bytes and memory need identical at 4k, 128k and 1M context."""
    for mid in RW:
        rs = [evaluate(_s(mid, serving__ctx=c, serving__batch=4)) for c in (4096, 131072, 1 << 20)]
        assert len({round(r.tpot, 12) for r in rs}) == 1, mid
        assert len({round(r.stages[0].mem.dram_need) for r in rs}) == 1
        assert all(r.stages[0].dram["kv_read"] == 0 and r.stages[0].dram["kv_write"] == 0 for r in rs)
    t = [evaluate(_s("qwen3-8b", serving__ctx=c)).tpot for c in (4096, 131072)]
    assert t[1] > t[0]                                           # a Transformer's does grow


def test_rwkv_state_bytes_and_ops():
    m = C.get_model("rwkv7-g1k-7.2b")
    for b in (1, 8):
        r = evaluate(_s("rwkv7-g1k-7.2b", serving__batch=b))
        per = 32 * (64 * 64 * 64 * 4 + 2 * 4096 * 4)
        assert r.stages[0].mem.state_total == per * b
    ph = Phase("decode", 1, 1, 4096)
    st = [o for o in build_rank_ops(m, 3, ph) if o.name == "state_update"][0]
    assert st.vec == 38 * 4096 + 64 * 9 * 64 * 64              # per token: 38/channel + 9·N² per head
    assert st.state_rw == 2 * (64 * 64 * 64 * 4 + 2 * 4096 * 4)


def test_rwkv_prefill_linear_in_prompt():
    m = C.get_model("rwkv7-g1k-2.9b")
    f = lambda q: sum(o.flops for o in build_rank_ops(m, 5, Phase("prefill", 1, q, 0)) if o.kind == "attn")
    assert abs(f(8192) / f(4096) - 2.0) < 1e-12 and abs(f(4096) / f(64) - 64.0) < 1e-12


def test_rwkv_state_follows_sram_residency():
    small = evaluate(_s("rwkv7-g1k-1.5b", **{"chip__sram_mib": 64.0}))
    big = evaluate(_s("rwkv7-g1k-1.5b", **{"chip__sram_mib": 8192.0}))
    assert small.stages[0].dram["state"] > 0 and big.stages[0].dram["state"] == 0
    assert big.tpot < small.tpot


# ---------------------------------------------------------------- decision readout
def test_startlux_catalog():
    assert len(SL) == 14
    for mid in SL:
        m = C.get_model(mid)
        assert m.param_check()["rel_err"] == 0.0 and m.coverage == "full" and m.vocab == 248320 and m.max_ctx == 262144
        assert C.entries and next(e for e in C.entries() if e["id"] == mid)["decision"]
    m = C.get_model("startlux-decision-v1-35b-a3b")
    f = m.layers[0].ffn
    assert (m.n_layers, m.hidden, f.n_experts, f.top_k, f.d_expert) == (40, 2048, 256, 8, 512)
    m = C.get_model("startlux-decision-v1-27b")
    full = [l.core for l in m.layers if l.core.kind == "gqa"]
    assert len(full) == 16 and full[0].n_q == 24 and full[0].n_kv == 4 and full[0].qk_dim == 256
    assert all((i + 1) % 4 == 0 for i, l in enumerate(m.layers) if l.core.kind == "gqa")
    assert C.get_model("startlux-decision-v1-9b").params() == C.get_model("qwen3.5-9b").params()


def test_microsoft_decision_proxy_and_no_jev():
    m = C.get_model("microsoft-decision-1")
    assert m.coverage == "proxy" and any("Qwen3.5-9B" in n for n in m.notes)
    e = next(x for x in C.entries() if x["id"] == "microsoft-decision-1")
    assert e["provider"] == "microsoft" and e["decision"]
    assert not any("jev" in x["id"].lower() for x in C.entries())


def test_labels_validation_and_serialisation():
    for bad in (True, -1, 1025, 2.0):
        try:
            Scenario().replace("serving.phase", "prefill").replace("serving.labels", bad); assert False, bad
        except ValueError:
            pass
    try:
        Scenario().replace("serving.labels", 3); assert False          # decode + labels
    except ValueError:
        pass
    assert "labels" not in str(Scenario().to_dict()) and Scenario().hash() == Scenario().replace("serving.labels", 0).hash()


def test_decision_readout_mechanism():
    m = C.get_model("startlux-decision-v1-9b")
    hp = head_ops(m, Phase("prefill", 4, 1024, 0, labels=3))
    hv = head_ops(m, Phase("prefill", 4, 1024, 0))
    g3 = [o for o in hp if o.kind == "gemm"][0]
    gv = [o for o in hv if o.kind == "gemm"][0]
    assert g3.n == 3 and gv.n == 248320 and g3.w_bytes * 248320 == gv.w_bytes * 3
    assert not any(o.kind == "comm" for o in hp)
    ops = build_rank_ops(m, 3, Phase("prefill", 4, 1024, 0, labels=3))
    assert sum(o.kv_write for o in ops) == 0
    assert sum(o.kv_write for o in build_rank_ops(m, 3, Phase("prefill", 4, 1024, 0))) > 0
    # shared prefix (ctx > 0): the cache is kept
    assert sum(o.kv_write for o in build_rank_ops(m, 3, Phase("prefill", 4, 512, 512, labels=3))) > 0


def test_decision_metrics_and_monotone():
    base = dict(serving__phase="prefill", serving__prompt=1024)
    prev = None
    for L in (1, 3, 26, 1024):
        r = evaluate(_s("startlux-decision-v1-4b", serving__labels=L, serving__batch=4, **base))
        d = r.summary()["decision"]
        assert abs(d["decisions_per_s"] - 4 / r.step) < 1e-9 and abs(d["latency_ms"] - r.step * 1e3) < 1e-9
        if prev is not None:
            assert r.step >= prev
        prev = r.step
    plain = evaluate(_s("startlux-decision-v1-4b", serving__batch=4, **base))
    lab = evaluate(_s("startlux-decision-v1-4b", serving__labels=3, serving__batch=4, **base))
    assert lab.step < plain.step and "decision" not in plain.summary()
    # one decision costs one prefill: no out_len / decode dependence
    a = evaluate(_s("startlux-decision-v1-4b", serving__labels=3, serving__out_len=1, **base))
    b = evaluate(_s("startlux-decision-v1-4b", serving__labels=3, serving__out_len=4096, **base))
    assert a.step == b.step


def test_api_decision():
    from accel_dse import api
    r = api.api_eval({"scenario": {"model": "startlux-decision-v1-9b",
                                   "serving": {"phase": "prefill", "prompt": 512, "labels": 2}}})
    assert r["summary"]["decision"]["decisions_per_s"] > 0


def test_instant_bidirectional_and_autothink_vision():
    """v1.1 Instant config: jev_prompt_attention=bidirectional -> full-attention
    cores non-causal (linear layers stay causal); AutoThink keeps causal and has
    a qwen3_5_vision encoder."""
    from accel_dse.core.catalog import get_model
    inst = get_model("startlux-decision-v1.1-9b-instant")
    auto = get_model("startlux-decision-v1.1-9b-autothink")
    nc = [l for l in inst.layers if l.core.kind in ("gqa", "mla") and not l.core.causal]
    assert len(nc) == 8
    assert all(l.core.causal for l in inst.layers if l.core.kind == "linear")
    assert all(getattr(l.core, "causal", True) for l in auto.layers)
    assert auto.vision is not None
