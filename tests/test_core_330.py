"""0.330: adversarial review of the 0.329 verdict flips (MODEL §275) and PixText-MeanFlow in the gen domain.

(1) The 0.329 HBM3 / HBM3E PHY (24 mm² per stack) + controller (3.575 mm² per stack) change: HBM3E = HBM3 bump map
    is now 「推导」 (JEDEC has no separate HBM3E standard; Synopsys HBM3 PHY to 9600 Mbps; GH100 / H200; AMD MI300X →
    MI325X 16 channels) — value unchanged; each flipped verdict re-derived from the area increment alone (per die
    stacks/die × (8 + 3.575) / 0.77, nothing counted twice), the MAC×2 4-stack TCO flip is a reticle crossing.
(2) PixText-MeanFlow (ATH-MaaS/PixText, Apache-2.0): GroupJiT pixel-space DiT from the release config + header."""
import importlib.util
import os
from dataclasses import replace

from accel_dse import api
from accel_dse.core import budget as bm
from accel_dse.core import catalog as C
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Link
from accel_dse.core.model import load_release
from accel_dse.core.scenario import Layout, Scenario, Serving
from accel_dse.core.thermal import COOLING

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H8, H4S = "hbm3e_8s_8h24g_9200", "hbm3e_4s_8h24g_9200"
STACK = (24.0 - 16.0 + 3.575) / (1 - 0.23)          # 0.329 full-area increment per HBM3E stack (mm²)


def _ex():
    spec = importlib.util.spec_from_file_location("chiplet_example", os.path.join(ROOT, "scripts", "chiplet_example.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _old_b():
    """≤ 0.328 HBM3 / 3E semantics on the auto (n5_derived) budget: HBM4 PHY density, no HBM3E controller."""
    return replace(bm.auto_area_budget(), mm2_per_dq_hbm3=None, mm2_per_hbm_mc=0.0)


def _s(tp=1, link=400.0, mem=H8, phase="decode"):
    return Scenario(model="qwen3-8b", chip=replace(CHIPS["1P"], io_su_lane_Gbps=100.0), mem_id=mem,
                    layout=Layout(tp=tp), link=Link(GBps=link),
                    serving=Serving(phase=phase, batch=8, prompt=2048, ctx=4096))


# ---------------------------------------------------------------- (1) sources / tags (value unchanged)
def test_hbm3e_bump_map_now_derived_value_unchanged():
    for k in ("n5_derived", "n7_tpuv4i", "n28_tpuv1"):
        v, tag, src = bm.DENSITY_PRESETS[k]["fields"]["mm2_per_dq_hbm3"]
        assert abs(v - 24 / 1024) < 1e-15 and tag == "来源/推导"
        for s in ("HBM3E = the HBM3 bump map 「推导」 (0.330, was 「假设」 in 0.329)", "JESD238", "9600 Mbps",
                  "GH100", "MI325X", "No public measured HBM3E PHY area"):
            assert s in src, s
    v, _, src = bm.DENSITY_PRESETS["n5_derived"]["fields"]["mm2_per_hbm_mc"]
    assert abs(v - 7.15) < 1e-12 and "3.575–7.15" in src and "JESD270-4" in src
    assert bm.MC_CHANNELS == {"HBM3": 16, "HBM3E": 16, "HBM4": 32, "HBM4E": 32}


# ---------------------------------------------------------------- (1) no double count: exact per-die increment
def test_per_die_increment_is_stacks_per_die_times_phy_plus_mc():
    for mem, n in ((H8, 8), (H4S, 4)):
        s = _s(mem=mem)
        new, old = bm.auto_die_mm2(s, bm.auto_area_budget()), bm.auto_die_mm2(s, _old_b())
        assert abs((new - old) - n * STACK) < 1e-6, (mem, new, old)
    # chiplet sweep (MAC×2, tp8 / 400 GB/s, D0 0.1): mono / 2-die / 4-die rows carry 4 / 2 / 1 stacks each
    rows = _ex().run("qwen3-8b", "decode", "tp", 0, None, out=lambda *_: None, cost="n5_cowos_l", d0=0.1, mem=H4S,
                     tp=8, link=400.0, tco="accel_2026", cooling=None)["rows"]
    mm = {(x["dies"], x.get("clusters")): x["mm2"] for x in rows if x["mac_x"] == 2.0}
    assert abs(mm[(1, 1)] - 898.99) < 0.05 and abs(mm[(1, 1)] - 4 * STACK - 838.86) < 0.05
    assert min(x["mm2"] for x in rows if x["mac_x"] == 2.0 and x["dies"] == 2) == _near(455.5)
    assert min(x["mm2"] for x in rows if x["mac_x"] == 2.0 and x["dies"] == 4) == _near(232.7)


def _near(x, tol=0.05):
    class A:
        def __eq__(self, y):
            return abs(y - x) <= tol
    return A()


# ---------------------------------------------------------------- (1) the MAC×2 TCO-memory flip = reticle crossing
def test_mac2_4stack_crosses_reticle():
    ex = _ex()
    tco = {}
    for mem in (H4S, H8):
        rows = ex.run("qwen3-8b", "decode", "tp", 0, None, out=lambda *_: None, cost="n5_cowos_l", d0=0.1, mem=mem,
                      tp=8, link=400.0, tco="accel_2026", cooling=None)["rows"]
        r2 = [x for x in rows if x["mac_x"] == 2.0]
        mono = next(x for x in r2 if x["dies"] == 1 and x.get("clusters") == 1)
        assert not mono["feasible"] and "光罩 858" in mono["why"]
        tco[mem] = min(x["tco"]["usd_per_Munit"] for x in r2 if x["feasible"])
    # 4-stack mono was 838.9 mm² (< 858, $/M 0.01164 — the 0.329 winner); now 899.0 → best 4-stack row is a split
    ret = bm.RETICLE_FIELD_MM[0] * bm.RETICLE_FIELD_MM[1]
    assert 898.99 - 4 * STACK < ret == 858.0 < 898.99
    assert abs(tco[H4S] - 0.01966) < 5e-5 and abs(tco[H8] - 0.01681) < 5e-5 and tco[H8] < tco[H4S]


# ---------------------------------------------------------------- (1) I/O: tp8 / 900 GB/s buildable by area alone
def test_io_flip_from_area_alone():
    s9, s12 = _s(tp=8, link=900.0), _s(tp=8, link=1200.0)
    old9, new9 = bm.shoreline_report(s9, None, _old_b()), bm.shoreline_report(s9, None, bm.auto_area_budget())
    assert old9["io"]["status"] == "over" and not old9["feasible"]
    assert new9["feasible"] and abs(new9["die_mm2"] - old9["die_mm2"] - 8 * STACK) < 1e-6
    assert old9["io"]["need_mm"] == new9["io"]["need_mm"]                       # lanes unchanged, only the edge grew
    assert not bm.shoreline_report(s12, None, bm.auto_area_budget())["feasible"]


# ---------------------------------------------------------------- (1) thermal: same whole-die definition both sides
def test_air_flux_whole_die_definition_and_thin_margin():
    c = COOLING["air_oam"]
    assert c["flux_W_mm2"] == 0.86 and "814" in c["src"]       # 700 W ÷ 814 mm² of the whole GH100 die (PHYs incl.)
    o = _ex().run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None, cooling="air_oam", power_scope="card")
    assert o["front"] == [1, 0, 15]
    rows = o["rows"]
    # rows 0 / 15 sit at 0.858 W/mm² vs 0.86 (0.2 % margin) — recorded as fragile in MODEL §275
    for i in (0, 15):
        f = rows[i]["thermal"]["flux_W_mm2"]
        assert rows[i]["feasible"] and 0.855 < f < 0.86, (i, f)


# ---------------------------------------------------------------- (2) PixText-MeanFlow
def test_pixtext_release_and_exact_params():
    rel = load_release("ATH-MaaS/PixText")
    assert rel["domain"] == "video" and rel["source"]["subfolder"] == "models/PixText-MeanFlow"
    assert rel["n_tensors"] == 465 and rel["params_llm"] == 953_948_480 and rel["bytes_llm"] == 2 * 953_948_480
    assert rel["source"]["repo_files"] == {"models/PixText-MeanFlow/model.safetensors": 1_907_943_360,
                                           "models/PixText-Teacher/model.safetensors": 1_904_005_664}
    m = C.get_model("pixtext-meanflow")
    assert m.params() == m.release_params == 953_948_480 and m.coverage == "full" and not m.coverage_reasons
    L = m.layers[0]
    assert len(m.layers) == 32 and m.hidden == 1280 and L.core.n_q == 16 and L.core.qk_dim == 80
    assert L.core.rope_dim == 80 and not L.core.causal and L.ffn.d_ff == 3413 and L.ffn.gated
    assert {x.name: (x.k, x.n, x.rows) for x in L.attn_linears} == {"qkv": (1280, 3840, "tok"), "o": (1280, 1280, "tok"),
                                                            "ada_mod": (1280, 7680, "tok")}
    io = {x.name: (x.k, x.n, x.rows) for x in m.io_pre + m.io_post}
    assert io["patch_embed"] == (768, 1280, "img") and io["glyph_embed"] == (256, 1280, "txt")
    assert io["interval_1_zero"] == io["interval_1"] == (256, 1280, "tok") and io["head"] == (1280, 768, "img")
    assert any("no additional text encoder" in n for n in m.notes)
    assert any("seven model forwards" in n for n in m.notes)
    ids = [i for i, *_ in C.DOMAIN_RELEASES]
    assert "pixtext-meanflow" in ids and not any("teacher" in i for i in ids if "pixtext" in i)


def test_pixtext_workload_and_flops():
    w = next(w for i, _, _, w in C.DOMAIN_RELEASES if i == "pixtext-meanflow")
    assert (w.height, w.width, w.vae_s, w.steps, w.cfg, w.text_tokens) == (1024, 1024, 1, 7, 1, 392)
    r = evaluate(Scenario().replace("model", "pixtext-meanflow").replace("serving.batch", 1))
    assert (r.workload.tokens, r.workload.ctx, r.workload.steps, r.workload.seqs_per_request) == (4488, 392, 7, 1)
    # per forward: GEMMs 2 · 4488 · 32 · (3840 + 1280 + 7680 + 6826 + 3413)·1280 + dense attention 4 · 4488² · 1280 · 32
    gemm = 2 * 4488 * 32 * (3840 + 1280 + 7680 + 6826 + 3413) * 1280
    attn = 4 * 4488 ** 2 * 1280 * 32
    per = r._tflop_per_request() / 7 * 1e12
    assert abs(per / (gemm + attn) - 1) < 0.01, per / (gemm + attn)
    # the allowed-mask pair count of the release case (pass 1): main → main + items, items → main + items + own
    # glyph, glyph → own glyph + own item; dense overcounts it by 9.29 %
    n = [20] + [16] * 11
    S, M = sum(n), 4096
    allowed = M * (M + S) + S * (M + S) + sum(k * k for k in n) + 2 * sum(k * k for k in n)
    assert allowed == 18_430_912 and abs((4488 ** 2) / allowed - 1.0929) < 1e-4


def test_pixtext_provider_and_family():
    assert ("athmaas", "ATH-MaaS（PixText）", ("ATH-MaaS/",)) in C.PROVIDERS
    assert ("athmaas", "PixText", ("pixtext",)) in C.FAMILIES


def test_summarize_release_detects_pixtext():
    spec = importlib.util.spec_from_file_location("summarize_release", os.path.join(ROOT, "scripts", "summarize_release.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.domain_of({"format_version": 1, "model": {"image_attention_routing": "x"}}) == "video"
    assert m.domain_of({"format_version": 1, "model": {}}) == "llm"
    assert m.classify_domain("timestep_embedder.mlp.0.weight", "video") == "cond"
    assert m.classify_domain("glyph_projection.proj.weight", "video") == "io"
    assert m.classify_domain("blocks.3.adaLN_modulation.1.weight", "video") == "cond"
