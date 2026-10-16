"""0.145: MiMo-V2.6-Pro + Ling-3.0-flash-VL (exact vs the released headers); MiniCPM-V slicing and MiMo banded ViT
windows modelled (both → full coverage); hostile review of the 0.144 builders (Step-5 idx_sum indexer, MiMo
sink / mixed KV, video MoE) with randomized invariants — and the Step-5 summary-write break it found."""
import math
import random
from dataclasses import replace

from accel_dse import api as A
from accel_dse.core import memplan as MP
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.ir import Phase, Shard, _attn_core_ops, _full_layer_ops
from accel_dse.core.vision import band_keys, image_grid, minicpm_views, vision_ops


def test_mimo_v26_pro_exact():
    m = get_model("mimo-v2.6-pro")
    assert m.params() == m.release_params == 1_021_248_126_336 and m.coverage == "full"
    # config omits num_nextn_predict_layers; the release ships model.mtp.layers.{0,1,2}
    assert len(m.mtp_layers) == 3 and m.params(include_mtp=True) - m.params() == m.release_mtp_params == 1_934_715_264
    swa = [l.core for l in m.layers if l.core.window == 128]
    assert len(m.layers) == 70 and len(swa) == 60 and all(c.sink for c in swa)
    assert sum(l.ffn.kind == "moe" for l in m.layers) == 69 and m.layers[1].ffn.n_experts == 384
    assert m.vision.params == m.vision_params == 1_033_761_792 and m.vision.extra == 294_681_600
    f = dict(m.formats)
    assert f["expert"].fmt == "mxfp4" and f["attn"].fmt == "fp8"


def test_ling30_flash_vl_distinct_and_exact():
    """Ling-3.0-flash-VL is not Ling-3.1-flash: h 2560 / 42 layers (groups of 6, 7 MLA) / 512 experts d 768 /
    2 dense, MLA without q-LoRA, no MTP; the BailingMM linear_proj is vision (0.145 role fix)."""
    m, o = get_model("ling-3.0-flash-vl"), get_model("ling-3.1-flash")
    assert m.params() == m.release_params == 124_414_211_552 and not m.mtp_layers and m.coverage == "full"
    assert (m.hidden, len(m.layers), o.hidden, len(o.layers)) == (2560, 42, 4096, 72)
    mla = [l for l in m.layers if l.core.kind == "mla"]
    assert len(mla) == 7 and sum(l.core.kind == "linear" for l in m.layers) == 35
    assert not any(l.name == "q_a" for l in mla[0].attn_linears)
    assert any(l.name == "o_gate" and l.n == 32 for l in mla[0].attn_linears)
    moe = [l.ffn for l in m.layers if l.ffn.kind == "moe"]
    assert len(moe) == 40 and (moe[0].n_experts, moe[0].top_k, moe[0].d_expert, moe[0].n_shared) == (512, 8, 768, 1)
    assert m.vision.params == m.vision_params == 434_248_944
    assert image_grid(m.vision, 1024, 768).llm_tokens == 768


def test_new_models_listed_and_evaluate():
    ids = {d["id"]: d for d in list_models()}
    assert ids["mimo-v2.6-pro"]["family"] == "MiMo-V2.6" and ids["ling-3.0-flash-vl"]["family"] == "Ling-3.0-VL"
    for mid in ("mimo-v2.6-pro", "ling-3.0-flash-vl"):
        assert ids[mid]["provenance"] == "official"
        for ph in ("decode", "prefill"):
            s = A.api_eval({"scenario": {"model": mid, "layout": {"tp": 8, "ep": 8},
                                         "serving": {"phase": ph, "batch": 8, "images": 1}}, "sensitivity": False})["summary"]
            assert s["step_ms"] > 0 and s["tok_s"] > 0


def _ref_views(w, h, side=448, p=14, ms=9):
    """MiniCPMV4_6ImageProcessor (transformers) find_best_resize / get_refine_size / get_sliced_grid, transcribed."""
    ed = lambda x, d: max(round(x / d) * d, d)                           # noqa: E731

    def best(hh, ww, up):
        if hh * ww > side * side or up:
            ar = ww / hh
            hh = int(side / math.sqrt(ar))
            ww = int(hh * ar)
        return ed(hh, p * 4), ed(ww, p * 4)
    lr, mult = math.log(w / h), min(math.ceil(w * h / (side * side)), ms)
    grid = None
    if mult > 1:
        grid, e0 = [1, 1], float("inf")
        for n in (mult - 1, mult, mult + 1):
            if n == 1 or n > ms:
                continue
            for r in range(1, n + 1):
                if n % r == 0:
                    e = abs(lr - math.log((n // r) / r))
                    if e < e0 or (e == e0 and r > grid[0]):
                        grid, e0 = [r, n // r], e
    sh, sw = best(h, w, grid is None)
    if grid is None:
        return (sh // p, sw // p), None, (0, 0)
    rh, rw = ed(h, grid[0]), ed(w, grid[1])
    bh, bw = best(rh / grid[0], rw / grid[1], True)
    return (sh // p, sw // p), (bh // p, bw // p), tuple(grid)


def test_minicpm_slicing_matches_processor_and_counts_tokens():
    rng = random.Random(145)
    for _ in range(3000):
        w, h = rng.randint(16, 8000), rng.randint(16, 8000)
        assert minicpm_views(w, h, 448, 14, 9) == _ref_views(w, h), (w, h)
    v = get_model("minicpm-v-4.7-35b-a3b").vision
    g = image_grid(v, 1024, 768)                     # 2×2 slices of 28×36 patches + a 28×36 overview
    assert g.views == ((28, 36, 1), (28, 36, 4)) and g.llm_tokens == 5 * 63 + 2 + 3 + 4 * 2 + 1
    ops = vision_ops(v, g, 2)
    assert sum(o.m for o in ops if o.name in ("vis.merger.1", "vis.v1.merger.1")) == 2 * 5 * 63
    assert image_grid(v, 448, 448).views == ((32, 32, 1),)          # no slices at ≤ 448² area
    assert get_model("minicpm-v-4.7-35b-a3b").coverage == "full"


def test_mimo_vit_band_windows():
    rng = random.Random(7)
    for _ in range(200):
        n, w = rng.randint(1, 600), rng.choice((0, 1, 5, 64, 700))
        assert band_keys(n, w) == sum(1 for i in range(n) for j in range(n) if w <= 0 or abs(i - j) <= w)
    v = get_model("mimo-v2.6-flash").vision
    assert v.win == 64 and v.full_idx == (0, 9, 18, 27) and v.sink and v.perm == 6
    g = image_grid(v, 1024, 768)
    ops = {o.name: o for o in vision_ops(v, g, 1)}
    N = g.ph * g.pw
    assert ops["vis.0.qk"].causal == 1.0 and ops["vis.1.qk"].causal == band_keys(N, 64) / (N * N) < 0.05
    # never more work than full attention; equal once the image fits in one band (N ≤ 65)
    full = replace(v, win=0)
    for wh in ((64, 64), (1024, 768), (4000, 3000)):
        gg = image_grid(v, *wh)
        a, b = (sum(o.flops for o in vision_ops(x, gg, 1)) for x in (v, full))
        assert a <= b and (a == b) == (gg.ph * gg.pw <= 65)


def test_step5_summary_writes_conserved():
    """Hostile review: the indexer wrote idx_dim bytes per new token while the cache stores one fp32 + fp8 summary
    (5·idx_dim) per completed 8-token region → 8/5× the stored bytes.  Now writes = stored growth, chunk-additive."""
    m = get_model("step-5-preview")
    li = next(i for i, l in enumerate(m.layers) if l.core.idx_heads)
    c = m.layers[li].core
    nl = sum(1 for l in m.layers if l.core.idx_heads)
    w = lambda ph: {o.name: o for o in _attn_core_ops(m, li, c, ph, Shard(), ph.batch)}["indexer_kv_write"].kv_write  # noqa
    rng = random.Random(5)
    for _ in range(40):
        S = rng.randint(2, 40000)
        cut = sorted(rng.sample(range(1, S), min(3, S - 1)))
        pts = [0] + cut + [S]
        one = w(Phase("prefill", 1, S, 0))
        assert abs(sum(w(Phase("prefill", 1, b - a, a)) for a, b in zip(pts, pts[1:])) - one) < 1e-6 * max(one, 1)
        assert one * nl == MP.stage_storage(m, 0, len(m.layers), False, False, Shard(), S).idx_per_seq
        assert MP.cache_new_bytes(m, S, cut[0]) <= MP.cache_new_bytes(m, S, 0)
    for n in (1, 7, 4096, 4103, 4104, 33000):
        assert c.keys_sum(n) == sum(min(p, c.ctx_eff(p)) for p in range(1, n + 1))


def test_mimo_mixed_kv_and_video_moe_invariants():
    rng = random.Random(9)
    for mid in ("mimo-v2.6-flash", "mimo-v2.6-pro"):
        m = get_model(mid)
        for S in (1, 128, rng.randint(129, 6000)):
            wr = sum(o.kv_write for i, l in enumerate(m.layers)
                     for o in _attn_core_ops(m, i, l.core, Phase("prefill", 1, S, 0), Shard(), 1))
            st = MP.cache_new_bytes(m, S, 0)
            assert wr >= st and (S > 128 or abs(wr - st) <= 1e-9 * st)
        for i, l in enumerate(m.layers[:9]):
            ops = {o.name: o for o in _attn_core_ops(m, i, l.core, Phase("decode", 4, 1, 5000), Shard(tp=8), 4)}
            assert ops["qk"].n == (128 if l.core.window else 5001)
    v = get_model("lingbot-video-moe-dmd")
    f = v.layers[0].ffn
    for _ in range(40):
        q, tp, sp, b = rng.randint(1, 60000), rng.choice((1, 2, 4)), rng.choice((1, 2, 4, 8)), rng.choice((1, 2))
        e = next(o for o in _full_layer_ops(v, 0, v.layers[0], Phase("full", b, q, 0), Shard(tp=tp, sp=sp))
                 if o.name == "expert.gate_up")
        t = b * math.ceil(q / sp)
        assert abs(e.m * e.count * e.share - t * f.top_k) <= 1e-6 * t * f.top_k and e.count <= f.n_experts


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "8410808fd4f1e578"   # 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "b8ca44d92e7589cd"   # 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)
