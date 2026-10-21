"""0.293 (MODEL §238): MoVA value-expert routing and residency.

* the release gives no value-routing statistic: the config's router_aux_loss_coef feeds the FFN router logits only
  (output_router_logits records K2HorizonSparseMoeBlock), the v_router carries a selection-only bias and no aux loss;
  uniform routing stays the labelled default, and ``serving.mova_expert_load`` takes a measured value-expert load;
* ``ir.mova_hits``: occupancy of a measured load (the 0.132 rule), uniform load = the uniform expectation;
* memplan routed hot class (``StageStorage.rt_w``): value-expert bytes are pinned by their read share — after the
  every-step weights, against the routed FFN experts by read share — and miss at their own share; a measured load
  uses the 0.286 A0 read-share curve; ledger closes per class (SRAM + SLC + DRAM = demand)."""
import math
import random

from accel_dse import api
from accel_dse.core import memplan as M
from accel_dse.core.catalog import get_model
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import mova_hits, mova_probs, moe_routing
from accel_dse.core.model import load_release

FP8, BF = "jais-3-mova-36b-a4b-fp8", "jais-3-mova-36b-a4b"
G = 2.0 ** 30
EV, KV_K = 64, 4
SKEW = [8.0] * 8 + [1.0] * 56


def _ev(model, chip="1P", **sv):
    body = {"model": model, "chip": chip, "serving": {"phase": "decode", "batch": 8, **sv}}
    return evaluate(api.scenario_from_body({"scenario": body}))


def _vop(model, b=8, v_p=(), **kw):
    from accel_dse.core.ir import Phase, build_rank_ops
    m = get_model(model)
    li = next(i for i, L in enumerate(m.layers) if L.core.v_experts)
    return next(o for o in build_rank_ops(m, li, Phase("decode", b, 1, ctx=4096, v_p=v_p, **kw)) if o.name == "attn.v_experts")


def test_release_has_no_value_routing_statistic():
    c = load_release("inception42/Jais-3-MoVA-36B-A4B")["config"]
    mova = {k for k in c if "mova" in k}
    assert mova == {"mova_num_experts", "mova_num_experts_per_tok"}, mova
    assert c["output_router_logits"] is False and c["router_aux_loss_coef"] == 0.001 and c["moe_gate_bias"] is True
    assert not any(k for k in c if "load" in k or "balance" in k), c.keys()


def test_mova_hits_occupancy():
    for t in (1, 2, 8, 64, 512):
        u = moe_routing(EV, KV_K, t, 1)
        flat = mova_hits(EV, KV_K, t, mova_probs((1.0,) * EV, KV_K))
        assert (flat.hit, flat.pairs_local, flat.m_e) == (u.hit, u.pairs_local, u.m_e), t
        sk = mova_hits(EV, KV_K, t, mova_probs(tuple(SKEW), KV_K))
        assert KV_K <= sk.hit <= u.hit and sk.pairs_local == t * KV_K, (t, sk)
        assert sk.m_e == math.ceil(t * KV_K / sk.hit)
    one = mova_hits(EV, KV_K, 3, mova_probs((1.0,) + (0.0,) * (EV - 1), KV_K))
    assert one.hit == KV_K                          # a token picks k distinct experts: never fewer than k
    assert mova_hits(EV, KV_K, 8, ()) == moe_routing(EV, KV_K, 8, 1)


def test_mova_expert_load_input():
    base = _ev(FP8)
    flat = _ev(FP8, mova_expert_load=[1.0] * EV)
    assert abs(flat.tpot / base.tpot - 1) < 1e-9
    sk = _ev(FP8, mova_expert_load=SKEW)
    assert sk.tpot < base.tpot
    assert _vop(FP8, v_p=mova_probs(tuple(SKEW), KV_K)).count < _vop(FP8).count
    try:
        _ev(FP8, mova_expert_load=[1.0] * 63)
        raise AssertionError("wrong length accepted")
    except ValueError as e:
        assert "64" in str(e)
    for bad in ([0.0, 0.0], [-1.0, 2.0]):
        try:
            _ev(FP8, mova_expert_load=bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    q = _ev("qwen3-30b-a3b")
    q2 = _ev("qwen3-30b-a3b", mova_expert_load=[1.0] * EV)
    assert q2.tpot == q.tpot and any("mova_expert_load" in w for w in q2.warnings)
    # the FFN expert load is a different router: it never reaches the value experts (Phase.v_p stays empty)
    from accel_dse.core.ir import SkewSpec
    v0 = _vop(FP8)
    v1 = _vop(FP8, skew=SkewSpec(2.0, ((0.5,) * 50, (0.1,) * 50)))
    assert (v1.m, v1.count, v1.share) == (v0.m, v0.count, v0.share)
    ffn = _ev(FP8, moe_expert_load=[5.0] * 10 + [1.0] * 90)
    assert ffn.tpot <= base.tpot * (1 + 1e-12)     # EP 1 here: the load only moves the FFN expert residency
    # scenario round trip: omitted by default (fingerprints unchanged), kept when set
    d = api.scenario_from_body({"scenario": {"model": FP8, "serving": {"mova_expert_load": SKEW}}})
    assert d.serving.mova_expert_load == tuple(SKEW)
    from accel_dse.core.scenario import Scenario, Serving
    assert "mova_expert_load" not in str(Scenario(model=FP8, serving=Serving()).to_dict())


def test_storage_routed_class():
    m = get_model(FP8)
    from accel_dse.core.ir import Shard
    st = M.stage_storage(m, 0, m.n_layers, True, True, Shard(), 4096)
    assert st.rt_w == 45 * EV * 2560 * 8 * 128 * 2           # bf16 value experts (FP8 release: routed FFN only)
    assert st.rt_w < st.hot_w and st.rr_w == 0
    bf = get_model(BF)
    assert M.stage_storage(bf, 0, bf.n_layers, True, True, Shard(), 4096).rt_w == 45 * EV * 2560 * 1024 * 4
    q = get_model("qwen3-30b-a3b")
    assert M.stage_storage(q, 0, q.n_layers, True, True, Shard(), 4096).rt_w == 0


def test_pin_order_by_read_share():
    """Every-step hot first; then the routed classes by read share per byte (value experts vs FFN experts)."""
    st = M.StageStorage(hot_w=4 * G, expert_w=8 * G, kv_per_seq=0.0, idx_per_seq=0.0, state_per_seq=0.0, rt_w=3 * G)
    mp = M.plan(st, 0, 2 * G, 1e15, 0.0, exp_read=0.5 * G, rt_read=1.5 * G)     # rho_e 1/16 < rho_rt 1/2
    assert mp.pinned_hot - mp.pinned_rt > 0.99 * G and mp.pinned_rt > 0 and mp.pinned_expert == 0
    assert mp.pinned_hot - mp.pinned_rt == st.hot_w - st.rt_w
    mp2 = M.plan(st, 0, 2 * G, 1e15, 0.0, exp_read=6 * G, rt_read=0.3 * G)      # rho_e 3/4 > rho_rt 1/10
    assert mp2.pinned_rt == 0 and mp2.pinned_expert > 0
    # every-step bytes miss less than under the ≤ 0.292 byte share (all hot alike)
    t = {"hot": 1 * G + 1.5 * G, "hot_rt": 1.5 * G, "exp": 0.5 * G, "kv_read": 0.0, "kv_write": 0.0, "state": 0.0,
         "lookup": 0.0}
    d = M.step_dram_bytes(mp, st, t)
    old = (t["hot"]) * (1 - mp.pinned_hot / st.hot_w) + t["exp"]
    assert d["weights"] < old
    # no routed class → the ≤ 0.292 rule exactly
    st0 = M.StageStorage(hot_w=4 * G, expert_w=8 * G, kv_per_seq=0.0, idx_per_seq=0.0, state_per_seq=0.0)
    a = M.plan(st0, 0, 2 * G, 1e15, 0.0, exp_read=0.5 * G)
    b = M.plan(st0, 0, 2 * G, 1e15, 0.0, exp_read=0.5 * G, rt_read=0.1 * G, rt_p=(0.5, 0.1))
    assert a == b and M.rt_miss(a, st0) == 1.0


def test_measured_value_load_raises_sram_read_share():
    st = M.StageStorage(hot_w=4 * G, expert_w=0.0, kv_per_seq=0.0, idx_per_seq=0.0, state_per_seq=0.0, rt_w=3.5 * G)
    p = mova_probs(tuple(SKEW), KV_K)
    u = M.plan(st, 0, 1.5 * G, 1e15, 0.0, rt_read=0.25 * G)
    z = M.plan(st, 0, 1.5 * G, 1e15, 0.0, rt_read=0.25 * G, rt_p=p)
    assert u.pinned_rt == z.pinned_rt > 0
    assert u.sram_rt_hit == u.pinned_rt / st.rt_w and z.sram_rt_hit > u.sram_rt_hit
    assert M.rt_miss(z, st) < M.rt_miss(u, st)


def _served(mp, st, t):
    rr = min(st.rr_w, st.hot_w)
    rt = min(st.rt_w, st.hot_w - rr)
    rest = st.hot_w - rr - rt
    t_rr, t_rt = (t["hot_rr"] if rr > 0 else 0.0), (t["hot_rt"] if rt > 0 else 0.0)
    if st.hot_w <= 0:
        hot = 0.0
    elif rr <= 0 and rt <= 0:
        hot = t["hot"] * mp.pinned_hot / st.hot_w
    else:
        hot = (t["hot"] - t_rr - t_rt) * ((mp.pinned_hot - mp.pinned_rr - mp.pinned_rt) / rest if rest > 0 else 0.0)
        hot += t_rr * (mp.pinned_rr / rr if rr > 0 else 0.0)
        hot += t_rt * (mp.sram_rt_hit if rt > 0 else 0.0)
    e = 0.0 if st.expert_w <= 0 else (mp.sram_exp_hit if mp.sram_exp_hit >= 0 else mp.pinned_expert / st.expert_w)
    kv = 0.0 if mp.kv_total <= 0 else (mp.kv_sram - mp.state_sram) / mp.kv_total
    s_ = 0.0 if mp.state_total <= 0 else mp.state_sram / mp.state_total
    return {"weights": hot + t["exp"] * e, "kv_read": t["kv_read"] * kv, "kv_write": t["kv_write"] * kv,
            "state": t["state"] * s_, "lookup": 0.0, "act": 0.0}


def _check(mp, st, t, d, where):
    tol = lambda x: 1e-6 * max(1.0, x) + 1.0
    sram, hit = _served(mp, st, t), d.get("slc_parts", {})
    dem = {"weights": t["hot"] + t["exp"] + t.get("w_extra", 0.0), "kv_read": t["kv_read"],
           "kv_write": t["kv_write"], "state": t["state"], "lookup": t["lookup"], "act": t.get("act", 0.0)}
    for k, x in dem.items():
        dr, sl, sr = d[k], hit.get(k, 0.0), sram[k]
        assert dr >= -tol(x) and sl >= -tol(x) and sr >= -tol(x), (where, k, dr, sl, sr)
        assert abs(dr + sl + sr - x) <= tol(x), (where, k, x, dr, sl, sr)
    assert mp.pinned_hot + mp.slc_hot <= st.hot_w + 1 and mp.pinned_rt + mp.slc_rt <= st.rt_w + 1, where
    assert mp.slc_held <= mp.slc + 1 and 0.0 <= mp.slc_frac_rt <= 1.0 + 1e-12, where
    assert mp.pinned_hot + mp.pinned_expert + mp.kv_sram <= max(0.0, mp.sram - mp.staging - mp.sram_resv) + mp.state_buf + 1, where


def test_ledger_property_with_routed_class():
    R = random.Random(293)
    pick = lambda hi: R.choice([0.0, R.uniform(0, hi), R.uniform(0, hi)])
    for it in range(3000):
        hot = pick(20) * G
        rt = min(hot, pick(15) * G) if R.random() < 0.8 else 0.0
        st = M.StageStorage(hot, pick(40) * G, pick(1) * G / 64, 0.0, pick(1) * G / 64, pick(5) * G, rt_w=rt)
        exp_read = R.choice([None, pick(1.2) * st.expert_w])
        rt_read = R.choice([None, pick(1.0) * rt])
        p = (tuple(min(1.0, KV_K * x / sum(SKEW)) for x in SKEW) if R.random() < 0.5 else None)
        mp = M.plan(st, R.choice([0, 1, 8, 64]), pick(32) * G, 1e15, pick(0.5) * G, act_need=pick(4) * G,
                    stream=R.random() < 0.5, slc_bytes=R.choice([0.0, pick(32) * G]),
                    slc_policy=R.choice(["pin", "pin_bw", "lru", "dip"]), state_buf=R.choice([0.0, pick(4) * G]),
                    exp_read=exp_read, rt_read=rt_read, rt_p=p)
        t_rt = rt_read if rt_read is not None else rt
        t = {"hot": (hot - rt) * R.choice([0.5, 1.0]) + t_rt, "hot_rt": t_rt,
             "exp": exp_read if exp_read is not None else st.expert_w, "kv_read": mp.kv_total * R.choice([1.0, 0.5]),
             "kv_write": pick(0.01) * G, "state": 2 * mp.state_total, "lookup": pick(0.01) * G, "act": pick(2) * G,
             "w_extra": pick(1) * G}
        d = M.step_dram_bytes(mp, st, t, R.choice([0.0, 0.5, 2.0]))
        _check(mp, st, t, d, it)


def test_eval_ledger_mova():
    cases = [({"sram_mib": 4096}, {}), ({"sram_mib": 4096}, {"mova_expert_load": SKEW}),
             ({"slc_mib": 16384, "slc_policy": "pin"}, {"mova_expert_load": SKEW}),
             ({"sram_mib": 2048, "slc_mib": 8192, "slc_policy": "dip"}, {"mova_expert_load": SKEW}),
             ({"slc_mib": 65536, "slc_policy": "lru"}, {})]
    for ch, sv in cases:
        for ph in ("decode", "prefill"):
            r = _ev(FP8, ch, phase=ph, ctx=4096, prompt=1024, **sv)
            for s in r.stages:
                store, tch, _, _ = s.ledger
                t = {k: tch.get(k, 0.0) for k in ("hot", "exp", "kv_read", "kv_write", "state", "lookup", "act",
                                                  "w_extra", "hot_rr", "hot_rt")}
                assert store.rt_w > 0 and t["hot_rt"] > 0, (ch, ph)
                _check(s.mem, store, t, s.dram, (ch, sv, ph))


def test_big_sram_pins_every_step_weights_first():
    """The design consequence: with SRAM for every-step weights, value experts no longer dilute them (≤ 0.292 pinned
    all hot bytes alike at their byte share)."""
    r = _ev(FP8, {"sram_mib": 4096}, batch=1)
    m = r.stages[0].mem
    st = r.stages[0].ledger[0]
    assert m.pinned_rt == 0.0 and m.pinned_hot > 3.9 * G
    assert M.hot_miss(m, st)[0] < 1.0 - m.pinned_hot / st.hot_w
    d = api._stage_dict(r.stages[0])["mem"]
    assert d["vexp_sram_MiB"] == 0.0 and "sram_vexp_hit" in d


# ------------------------------------------------------------------------------- per-layer expert loads (0.293)
Q3 = "qwen3-30b-a3b"
BOLT = "nemotron-3.5-bolt-74b-a9b-bf16"
EP4 = {"tp": 1, "ep": 4, "dp": 4}


def _evl(model, chip="H100-like", layout=None, **sv):
    body = {"model": model, "chip": chip, "serving": {"phase": "decode", "batch": 64, **sv}}
    if layout:
        body["layout"] = layout
    return evaluate(api.scenario_from_body({"scenario": body}))


def _moe_layers(model):
    return [L for L in get_model(model).layers if L.ffn.kind == "moe"]


def test_per_layer_loads_equal_rows_match_one_vector():
    ls = _moe_layers(Q3)
    load = [1.0 / (i + 1) for i in range(ls[0].ffn.n_experts)]
    for chip, lay in (("H100-like", EP4), ({"sram_mib": 4096, "slc_mib": 8192, "slc_policy": "dip"}, None),
                      ({"sram_mib": 4096, "slc_mib": 8192, "slc_policy": "pin"}, None)):
        a = _evl(Q3, chip, lay, moe_expert_load=load)
        b = _evl(Q3, chip, lay, moe_expert_load_layers=[load] * len(ls))
        assert abs(a.tpot / b.tpot - 1) < 1e-12, (chip, lay, a.tpot, b.tpot)
    from accel_dse.core.energy import action_counts
    a = action_counts(_evl(Q3, "H100-like", EP4, moe_expert_load=load))
    b = action_counts(_evl(Q3, "H100-like", EP4, moe_expert_load_layers=[load] * len(ls)))
    assert all(abs(a["counts"][k] - b["counts"][k]) <= 1e-9 * max(1.0, abs(a["counts"][k])) for k in a["counts"])
    v = [list(SKEW)] * 45
    x, y = _ev(FP8, mova_expert_load=SKEW), _ev(FP8, mova_expert_load_layers=v)
    assert abs(x.tpot / y.tpot - 1) < 1e-12


def test_per_layer_loads_route_each_layer():
    ls = _moe_layers(Q3)
    E = ls[0].ffn.n_experts
    hot = [1.0 / (i + 1) ** 1.2 for i in range(E)]
    flat = [1.0] * E
    t = {n: _evl(Q3, "H100-like", EP4, moe_expert_load_layers=lls).tpot
         for n, lls in (("hot", [hot] * len(ls)), ("flat", [flat] * len(ls)),
                        ("half", [hot if i % 2 else flat for i in range(len(ls))]))}
    assert t["flat"] < t["half"] < t["hot"], t
    from accel_dse.core.evaluate import _route_groups
    from accel_dse.core.ir import Phase
    from accel_dse.core.memo import layer_groups
    m = get_model(Q3)
    assert _route_groups(m, Phase("decode", 8)) == layer_groups(m)
    g = _route_groups(m, Phase("decode", 8, skew_l=tuple(1.0 if i % 2 else 2.0 for i in range(len(m.layers)))))
    assert len(set(g)) == 2 * len(set(layer_groups(m)))


def test_het_moe_takes_per_layer_loads():
    ls = _moe_layers(BOLT)
    assert len({L.ffn.n_experts for L in ls}) > 1
    try:
        _evl(BOLT, layout=EP4, moe_expert_load=[1.0] * ls[0].ffn.n_experts)
        raise AssertionError("one vector accepted for different expert counts")
    except ValueError as e:
        assert "moe_expert_load_layers" in str(e)
    lls = [[1.0 / (i + 1) for i in range(L.ffn.n_experts)] for L in ls]
    r, u = _evl(BOLT, layout=EP4, moe_expert_load_layers=lls), _evl(BOLT, layout=EP4)
    assert r.tpot > u.tpot                          # the busiest EP rank of every layer carries more pairs
    j = next(i for i, L in enumerate(ls) if L.ffn.n_experts != ls[0].ffn.n_experts)
    for bad in (lls[:-1], [lls[j]] + lls[1:]):
        try:
            _evl(BOLT, layout=EP4, moe_expert_load_layers=bad)
            raise AssertionError("bad per-layer loads accepted")
        except ValueError:
            pass
    q = _evl("qwen3-8b", moe_expert_load_layers=[[1.0]])
    assert any("MoE" in w for w in q.warnings)


def test_weighted_a0_curve():
    p = tuple(0.5 / (i + 1) for i in range(16))
    for f in (0.1, 0.37, 0.8):
        a = M.dip_exp_hit(p, 0.2, f)
        b = M.dip_exp_hit(p, 0.2, f, (3.0,) * 16)
        assert abs(a - b) < 1e-12
    # two items, the hot one twice the bytes: with f = 2/3 of the bytes resident (the hot item) the served share is
    # its read share q_hot·2 / (q_hot·2 + q_cold)
    q = M._dip_exp_hit_w([0.9, 0.1], [2.0, 1.0], 0.5, 2.0 / 3.0)
    import math as _m
    lo, hi = 0.0, 1e3
    for _ in range(200):
        T = (lo + hi) / 2
        mean = (2 * (1 - 0.1 ** T) + (1 - 0.9 ** T)) / 3
        lo, hi = (T, hi) if mean < 0.5 else (lo, T)
    qh, qc = 1 - 0.1 ** hi, 1 - 0.9 ** hi
    assert abs(q - 2 * qh / (2 * qh + qc)) < 1e-9 and q > 2.0 / 3.0
    prev = 0.0
    for i in range(1, 20):
        h = M.dip_exp_hit(p, 0.3, i / 20, tuple(1.0 + (j % 3) for j in range(16)))
        assert i / 20 - 1e-12 <= h <= 1.0 and h >= prev - 1e-12
        prev = h


def test_mova_per_layer_validation():
    try:
        _ev(FP8, mova_expert_load_layers=[[1.0] * EV] * 44)
        raise AssertionError("wrong layer count accepted")
    except ValueError as e:
        assert "45" in str(e)
    r = _ev("qwen3-8b", mova_expert_load_layers=[[1.0] * EV])
    assert any("mova_expert_load_layers" in w for w in r.warnings)
    one = [list(SKEW)] + [[1.0] * EV] * 44
    a, u = _ev(FP8, mova_expert_load_layers=one), _ev(FP8)
    assert a.tpot < u.tpot
