"""0.289 (MODEL §234): the Nemotron 3.5 family (Bolt 74B-A9B puzzle / ggml block formats, Lightning 30B-A3B, Super-VL
BF16), routing of heterogeneous MoE layers, and the pipe flow-shop state as a lazy view (protenix on a 4 GiB SRAM +
4 GiB SLC pin chip: > 700 s → seconds, bit-identical)."""
import math
import random

from accel_dse import api
from accel_dse.core import overlap as OV
from accel_dse.core.catalog import MERGED, entries, get_model, list_models
from accel_dse.core.evaluate import evaluate, extra_expert_bytes, union_routing
from accel_dse.core.hardware import CHIPS
from accel_dse.core.ir import moe_routing
from accel_dse.core.model import load_release
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _fused_step
from accel_dse.core.scenario import Scenario, Serving

BOLT = "nemotron-3.5-bolt-74b-a9b-24gb"
LISTED = {BOLT: "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
          "nemotron-3.5-lightning-30b-a3b-nvfp4": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"}
BF16_SIBLINGS = {"nemotron-3.5-bolt-74b-a9b-bf16": BOLT,
          "nemotron-3.5-lightning-30b-a3b-bf16": "nemotron-3.5-lightning-30b-a3b-nvfp4",
          "nemotron-3.5-super-vl-120b-a12b-bf16": "nemotron-3.5-super-vl-120b-a12b-nvfp4"}
HB = "hbm3e_8s_12h24g_9200"
PIN = {"sram_mib": 4096, "slc_mib": 4096, "slc_policy": "pin"}


def _llm(rel):
    c = rel["config"]
    return c.get("llm_config", c)


def test_nemotron_35_entries_exact_and_bf16_siblings_listed():
    idx = {e["id"]: e for e in entries()}
    listed = {x["id"]: x for x in list_models()}
    for mid, hf in LISTED.items():
        x = listed[mid]
        assert idx[mid]["hf_id"] == hf and x["provider"] == "nvidia" and abs(x["param_err"]) < 1e-12, mid
        assert get_model(mid).params() == load_release(hf)["params_llm"], mid
    for mid, base in BF16_SIBLINGS.items():
        # same architecture, other dtypes → listed separately (MERGED requires identical params and dtypes)
        assert mid not in MERGED and mid in listed and listed[mid]["dtype"] != listed[base]["dtype"], mid
        a, b = load_release(idx[mid]["hf_id"]), load_release(idx[base]["hf_id"])
        la, lb = _llm(a), _llm(b)
        for k in ("hidden_size", "block_configs", "layers_block_type", "n_routed_experts", "num_experts_per_tok",
                  "moe_intermediate_size", "moe_latent_size", "mamba_num_heads", "num_attention_heads",
                  "num_key_value_heads", "num_nextn_predict_layers", "vocab_size"):
            assert la.get(k) == lb.get(k), (mid, k)
        assert (a["params_llm"], a["params_mtp"], a["params_vision"]) == (b["params_llm"], b["params_mtp"],
                                                                         b["params_vision"]), mid
        assert get_model(mid).params() == a["params_llm"], mid
    # Bolt: the public header totals (model card: 74B total / 9B active, RADIO ViT-H)
    rel = load_release(LISTED[BOLT])
    assert (rel["params_llm"], rel["params_vision"]) == (74042720768, 842360326)
    assert load_release("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")["params_mtp"] == 1335325952


def test_bolt_puzzle_layers_follow_block_configs():
    m = get_model(BOLT)
    bc = _llm(load_release(LISTED[BOLT]))["block_configs"]
    assert len(m.layers) == len(bc) == 88 and not m.mtp_layers
    kinds = {"mamba": "linear", "attention": "gqa"}
    for L, b in zip(m.layers, bc):
        if b["block_type"] == "moe":
            f = L.ffn
            assert (f.kind, f.n_experts, f.top_k, f.d_expert, f.latent) == (
                "moe", b["n_routed_experts"], b["num_experts_per_tok"], b["moe_intermediate_size"], 1024)
        else:
            assert L.ffn.kind == "none" and L.core.kind == kinds[b["block_type"]]
    es = {L.ffn.n_experts for L in m.layers if L.ffn.kind == "moe"}
    assert min(es) == 240 and max(es) == 496 and len(es) > 1


def test_bolt_24gb_ggml_formats_as_released():
    rel = load_release(LISTED[BOLT])
    r = rel["roles"]
    # ggml blocks (ggml-common.h): IQ2_XXS 66 B / 256 weights, Q8_0 34 B / 32
    assert 8 * r["expert"]["iq2_xxs"]["bytes"] == 2.0625 * r["expert"]["iq2_xxs"]["params"]
    assert 8 * r["lm_head"]["q8_0"]["bytes"] == 8.5 * r["lm_head"]["q8_0"]["params"]
    assert set(r["shared_expert"]) == {"iq2_xxs"} and set(r["mlp"]) == {"q8_0"} and set(r["embed"]) == {"bf16"}
    assert set(r["attn"]) == {"bf16", "nvfp4", "q8_0"}       # Mamba in / out NVFP4 W4A16, GQA q/k/v/o Q8_0, conv bf16
    m = get_model(BOLT)
    fm = dict(m.formats)
    assert (fm["expert"].fmt, fm["expert"].bits) == ("iq2_xxs", 2.0625)
    assert (fm["lm_head"].fmt, fm["lm_head"].bits) == ("q8_0", 8.5)
    assert not any(k.startswith("act_") for k in fm)          # every quantized role is weight-only → bf16 inputs
    assert m.kv_fmt == "bf16" and m.state_fmt == "fp32"        # config: no kv_cache_scheme, mamba_ssm_cache_dtype fp32
    assert any("ggml" in x for x in m.coverage_reasons)
    # decode b1 (default chip): the 2-bit experts stream ~7 × fewer bytes than the bf16 release, which does not fit
    q, h = (evaluate(api.scenario_from_body({"scenario": {"model": x, "serving": {"phase": "decode", "batch": 1}}}))
            for x in (BOLT, "nemotron-3.5-bolt-74b-a9b-bf16"))
    assert q.fits and not h.fits and q.tpot < 0.5 * h.tpot


def _pair(model, lay, ov, k):
    b = Scenario(model=model, chip=CHIPS["1P"], mem_id=HB, layout=lay, exec_overlap=ov)
    return (evaluate(b.replace("serving", Serving(phase="decode", batch=k, ctx=4096))),
            evaluate(b.replace("serving", Serving(phase="prefill", batch=1, prompt=4096))))


def test_heterogeneous_moe_chunk_experts_per_layer():
    m = get_model(BOLT)
    d, p = _pair(BOLT, Layout(tp=2, ep=2), "stage", 4)
    sd, sp = d.stages[0], p.stages[0]
    assert sd.exp_route[0] == "het"
    ru = union_routing(sd, sp, 1 / 32)
    assert isinstance(ru, dict) and len(ru) == len({(L.ffn.n_experts, L.ffn.top_k) for L in m.layers if L.ffn.kind == "moe"})
    # independent per-layer sum: each layer's experts stream hit × (per-expert bytes); the chunk adds hit' − hit
    td, tu, sk = sd.exp_route[2], sd.exp_route[2] + max(1, round(sp.exp_route[2] / 32)), sd.exp_route[4]
    base = extra = 0.0
    for L in m.layers:
        f = L.ffn
        if f.kind == "moe":
            w = f.mats * f.expert_in(m.hidden) * f.d_expert
            r0, r1 = moe_routing(f.n_experts, f.top_k, td, 2, sk), moe_routing(f.n_experts, f.top_k, tu, 2, sk)
            base += w * r0.hit
            extra += w * (r1.hit - r0.hit)
    x = extra_expert_bytes(sd, sp, 1 / 32)
    assert abs(x - sd.dram["exp_dram"] * extra / base) <= 1e-9 * x
    # the first layer's E (240, ≤ 0.288) under-counts: larger layers gain more distinct experts per chunk
    f0 = next(L.ffn for L in m.layers if L.ffn.kind == "moe")
    old = sd.dram["exp_dram"] * (moe_routing(f0.n_experts, 9, tu, 2, sk).hit / moe_routing(f0.n_experts, 9, td, 2, sk).hit - 1)
    assert x > old
    for ov in ("stage", "pipe", "serial"):          # the fused pipe path routes each expert op with its own layer
        d, p = _pair(BOLT, Layout(tp=2, ep=2), ov, 4)
        assert _fused_step(d, p, 1 / 32, 0.0) > d.step
    # uniform MoE keeps the 0.103 tuple
    d, _ = _pair("qwen3-30b-a3b", Layout(tp=2, ep=2), "stage", 1)
    assert d.stages[0].exp_route[:2] == (128, 8)


def test_heterogeneous_moe_load_refused_skew_cap_strictest_layer():
    s = api.scenario_from_body({"scenario": {"model": BOLT, "layout": {"tp": 8, "ep": 8},
                                             "serving": {"phase": "decode", "batch": 8, "moe_expert_load": [1.0] * 240}}})
    try:
        evaluate(s)
        raise AssertionError("moe_expert_load accepted for layers of different expert counts")
    except ValueError as e:
        assert "different expert counts" in str(e)
    # EP 64 > 240 / 9: the 240-expert layers cap the skew at 64 · ceil(240 / 64) / 9 = 28.44 (496: 64 · 8 / 9 = 56.9)
    s = api.scenario_from_body({"scenario": {"model": BOLT, "layout": {"tp": 64, "ep": 64},
                                             "serving": {"phase": "decode", "batch": 64, "moe_skew": 60.0}}})
    r = evaluate(s)
    assert any("28.4444" in w for w in r.warnings), r.warnings


# ---- pipe flow shop: the ≤ 0.288 simulator verbatim (state = full tuple), the reference for bit identity ----
class _RefSim:
    def __init__(self, buf, cross_op=True):
        self.buf, self.cross_op = math.floor(buf), cross_op
        self.d_end = self.c_end = self.last = 0.0
        self.win, self.wsum, self.op, self.s0, self.hw = [], 0, False, 0.0, 0.0

    def feed_runs(self, runs, op_start=False):
        first = op_start
        for t, k in runs:
            prev = prev_dt = None
            i = 0
            if k < 8 * (len(self.win) + 8):
                self.feed([t], first); first = False
                if k > 1:
                    self.feed([t] * (k - 1))
                continue
            while i < k:
                c0 = self.c_end
                self.feed([t], first); first = False; i += 1
                if k - i <= 0:
                    break
                if (i & (i - 1)) == 0 and i >= 2:
                    prev, prev_dt = self.state(), self.c_end - c0
                elif prev is not None:
                    st, dt = self.state(), self.c_end - c0
                    if abs(dt - prev_dt) <= 1e-12 * max(self.c_end, 1e-30) and _ref_same(st, prev, self.c_end):
                        self.shift(dt * (k - i)); i = k
                        break
                    prev = None

    def feed(self, tiles, op_start=False):
        win, buf = self.win, self.buf
        first = op_start and not self.cross_op
        d_end, c_end, last, wsum = self.d_end, self.c_end, self.last, self.wsum
        for sz, m, c in tiles:
            if sz.__class__ is str:
                if sz == "open":
                    self.s0, self.op = c_end, True
                elif sz == "close":
                    self.hw += c_end - self.s0; self.op = False
                elif sz == "reset":
                    self.hw = 0.0
                else:
                    h = min(self.hw, c); self.hw -= h; c_end += c - h
                continue
            start = d_end
            if sz > 0:
                sz = math.ceil(sz)
                if wsum + sz > buf:
                    h, nw = 0, len(win)
                    while wsum + sz > buf and h < nw:
                        w = win[h]
                        wsum -= w[0]; last = w[1]; h += 1
                    if h:
                        del win[:h]
                if last > start:
                    start = last
                if first and c_end > start:
                    start = c_end
            first = False
            d_end = start + m
            if sz > 0 and d_end > c_end:
                c_end = d_end
            c_end += c
            win.append([sz, c_end]); wsum += sz
        self.d_end, self.c_end, self.last, self.wsum = d_end, c_end, last, wsum

    def state(self):
        c = self.c_end
        return (self.d_end - c, self.last - c, tuple((s, d - c) for s, d in self.win), 0.0, self.hw)

    def shift(self, dt):
        self.d_end += dt; self.c_end += dt; self.last += dt
        if not self.op:
            self.s0 += dt
        for w in self.win:
            w[1] += dt


def _ref_same(a, b, scale):
    tol = 1e-12 * max(scale, 1e-30)
    if len(a[2]) != len(b[2]) or abs(a[0] - b[0]) > tol or abs(a[1] - b[1]) > tol:
        return False
    if abs(a[3] - b[3]) > tol or abs(a[4] - b[4]) > tol:
        return False
    return all(x[0] == y[0] and abs(x[1] - y[1]) <= tol for x, y in zip(a[2], b[2]))


def _ref_stage_pipe_time(segments, f, sec_per_byte, buf, cross_op):
    sim = _RefSim(buf, cross_op)
    tile = min(OV.PIPE_TILE, buf / 2)
    for items, n in segments:
        if not items or n <= 0:
            continue
        per = [OV._tiles([it], f, sec_per_byte, 1.0, tile) for it in items]
        prev, prev_dt, done = None, None, 0
        while done < n:
            t0 = sim.c_end
            for tl in per:
                sim.feed_runs(tl, True)
            done += 1
            st, dt = sim.state(), sim.c_end - t0
            if prev is not None and prev_dt is not None and _ref_same(st, prev, sim.c_end) \
                    and abs(dt - prev_dt) <= 1e-12 * max(sim.c_end, 1e-30):
                sim.shift(dt * (n - done)); done = n
            prev, prev_dt = st, dt
    return sim.c_end


def test_pipe_flow_shop_bit_identical_to_the_full_state_reference():
    rnd = random.Random(289)
    for case in range(600):
        segs = []
        for _ in range(rnd.randint(1, 4)):
            items = []
            for _ in range(rnd.randint(1, 5)):
                u = rnd.random()
                if u < 0.08:
                    items.append((rnd.choice(["open", "close", "reset", "hide"]), {}, rnd.random() * 1e-4))
                elif u < 0.15:
                    items.append((0.0, {}, rnd.random() * 1e-4))
                else:
                    b = {k: rnd.choice([0, rnd.randint(1, 1 << 24)]) for k in ("hot", "exp", "act")}
                    items.append((rnd.random() * 1e-3 * rnd.choice([0, 1, 1]), b, 0.0))
            segs.append((items, rnd.choice([1, 2, 3, 7, 40, 200])))
        f = {"hot": rnd.choice([0.0, 0.3, 1.0]), "exp": 1.0, "act": rnd.choice([0.0, 1.0])}
        buf = rnd.choice([1 << 16, 1 << 20, 1 << 24, 1 << 30, 4 << 30])
        spb, co = rnd.choice([1e-12, 1e-11, 1e-10]), rnd.random() < 0.5
        assert OV._stage_pipe_time(segs, f, spb, buf, co, 1.0, 0.0, 0.0) == _ref_stage_pipe_time(segs, f, spb, buf, co), case


# 0.289 deterministic work of the two slowest pin-chip cases (fresh process, box): tiles fed to the flow shop and window
# entries compared entry-wise (state views of equal length).  ≤ 0.288 every state() built a tuple of the whole window
# (protenix: 6415 states × ~0.5 M entries — > 700 s).
_PIN_WORK = {"protenix": (3521581, 0, 6415), "boltz-1": (1195681, 242699, 2165)}


def test_pin_chip_structure_runtime_guard():
    work = {"feed": 0, "cmp": 0, "same": 0}
    f0, s0 = OV._Sim.feed, OV._same

    def feed(self, tiles, op_start=False):
        work["feed"] += len(tiles)
        return f0(self, tiles, op_start)

    def same(a, b, scale):
        work["same"] += 1
        if a[2][2] - a[2][1] == b[2][2] - b[2][1]:
            work["cmp"] += a[2][2] - a[2][1]
        return s0(a, b, scale)

    from tests.workmeter import guard
    OV._Sim.feed, OV._same = feed, same
    OV._PIPE_MEMO.clear()
    try:
        for mid, (nf, nc, ns) in _PIN_WORK.items():
            for k in work:
                work[k] = 0
            # 0.332: the 20 s CPU bound (60 s under -j) is now deterministic work at the 20 s quiet-box equivalent
            # (tests/workmeter.py); 2.4 s alone (protenix); > 700 s at 0.288
            with guard(20, f"289 {mid}"):
                r = evaluate(api.scenario_from_body({"scenario": {"model": mid, "chip": PIN,
                                                                  "serving": {"phase": "decode", "batch": 1}}}))
            assert r.step > 0
            assert work["feed"] <= 1.5 * nf and work["cmp"] <= 1.5 * nc + 1000 and work["same"] <= 1.5 * ns, (mid, work)
    finally:
        OV._Sim.feed, OV._same = f0, s0
    st = OV._Sim(1 << 30).state()
    assert isinstance(st[2], tuple) and len(st[2]) == 4 and isinstance(st[2][0], list)   # a view, not the window


def test_kimi_dit_cached_kv_on_the_attention_ops():
    from accel_dse.core.kimi_audio import dit_eval_ops
    k = get_model("kimi-audio-7b-instruct").kimi
    for T, kv in ((60, 0), (60, 120), (90, k.max_kv)):
        ops = dit_eval_ops(k, T, kv)
        assert sum(o.kv_read for o in ops if o.kind == "attn") == k.kv_bytes(kv)     # 0.288's extra_dram, now in the ops
