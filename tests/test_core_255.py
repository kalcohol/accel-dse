"""0.255: group-limited MoE routing (n_group / topk_group) per-rank spread + node-dedup fan-out; Muse Spark image /
video resize (transformers MuseGlimmer processors); Step-5 crops (Step3p7ImageProcessor plan); FLUX 3 Base static
mode-stream reuse (SGLang StaticStreams, workload.mode_cache)."""
import math
import random
import statistics as stt
import sys

from accel_dse.core import catalog as C
from accel_dse.core.ir import _group_rank_moments, node_dedup_factor
from accel_dse.core.vision import _muse_resize, image_grid, step_crops, video_grid, video_pre

sys.path.insert(0, __import__("os").path.dirname(__file__))
from test_core_252 import _ev   # noqa: E402


def _route(rng, E, k, G, g):
    S = E // G
    gs = rng.sample(range(G), g)
    return rng.sample([x * S + j for x in gs for j in range(S)], k)


def test_moe_groups_from_config():
    for i in ("deepseek-v3", "deepseek-r1", "deepseek-v3.2", "ling-3.1-flash", "beam-1.0-501b-a23b"):
        m = C.get_model(i)
        f = next(l.ffn for l in m.layers if l.ffn.kind == "moe")
        assert f.groups == (8, 4), i
    for i in ("kimi-k2.5", "glm-5"):         # n_group = topk_group = 1 → unrestricted
        m = C.get_model(i)
        assert all(l.ffn.groups == () for l in m.layers), i


def test_group_rank_moments_match_monte_carlo():
    E, k, G, g, T = 256, 8, 8, 4, 16
    rng = random.Random(0)
    for ep in (8, 4, 16):                     # rank inside one group (ep ≥ G) and a rank holding G/ep groups
        (dm, ds), (pm, ps) = _group_rank_moments(E, k, T, ep, (G, g))
        n_loc = E // ep
        D, P = [], []
        for _ in range(3000):
            hit = [0] * n_loc
            for _t in range(T):
                for e in _route(rng, E, k, G, g):
                    if e < n_loc:
                        hit[e] += 1
            D.append(sum(1 for h in hit if h))
            P.append(sum(hit))
        assert abs(stt.mean(D) - dm) / dm < 0.02 and abs(stt.mean(P) - pm) / pm < 0.02, ep
        assert abs(stt.pstdev(D) - ds) / ds < 0.04 and abs(stt.pstdev(P) - ps) / ps < 0.04, ep
        # means equal the ungrouped ones (expert marginals stay k/E); the spread grows
        assert math.isclose(pm, T * k / ep, rel_tol=1e-9)
        assert math.isclose(dm, n_loc * (1 - (1 - k / E) ** T), rel_tol=1e-6)


def test_group_rank_moments_reduce_when_all_groups_kept():
    # g = G (or n_group 1) is the unrestricted router: no grouped moments, the ungrouped rule applies
    assert _group_rank_moments(256, 8, 16, 8, (8, 8)) is None
    assert _group_rank_moments(256, 8, 16, 8, (1, 1)) is None
    assert _group_rank_moments(256, 8, 16, 6, (8, 4)) is None      # groups do not tile 6 ranks
    assert _group_rank_moments(256, 8, 16, 72, (8, 4)) is None     # 256 experts on EP 72 straddle the groups


def test_node_dedup_grouped_monte_carlo():
    E, k, G, g = 256, 8, 8, 4
    rng = random.Random(1)
    for N in (8, 4):
        tot = 0
        n = 20000
        for _ in range(n):
            tot += len({e // (E // N) for e in _route(rng, E, k, G, g)})
        mc = tot / n / k
        assert abs(node_dedup_factor(E, k, N, (G, g)) - mc) < 0.006, N
        assert node_dedup_factor(E, k, N, (G, g)) < node_dedup_factor(E, k, N)


def test_muse_resize_verbatim():
    # transformers muse_glimmer smart_resize: floor/ceil grid closest in aspect, ≤ max cells (28 px = one merged token)
    assert _muse_resize(1024, 1024, 28, 4096) == (1008, 1008)
    assert _muse_resize(1080, 1920, 28, 4096) == (1092, 1932)
    h, w = _muse_resize(3000, 4000, 28, 4096)
    assert (h // 28) * (w // 28) <= 4096 and (h, w) == (1540, 2044)
    assert _muse_resize(10, 10, 28, 4096) == (28, 28)
    m = C.get_model("muse-spark-open")
    assert m.coverage == "full" and not m.coverage_reasons
    g = image_grid(m.vision, 1024, 1024)
    assert (g.ph, g.pw, g.merged, g.llm_tokens) == (72, 72, 1296, 1298)     # + <|image_start|> / <|image_end|>
    vg = video_grid(m.vision, video_pre(m), 64, 1280, 720)
    # processor_config video: ≤ 256 merged tokens / frame, temporal patch 2 → 32 slices; + start / 31 seps / end
    assert vg.t == 32 and vg.slice.merged == 252 and vg.llm_tokens == 32 * 252 + 33
    assert video_grid(m.vision, video_pre(m), 1000, 640, 640).frames == 600   # num_frames cap


def test_step5_crops_plan():
    assert step_crops(728, 728, 728, 504, 3024) == (0, 0)          # fits the overview, not elongated
    assert step_crops(1024, 768, 728, 504, 3024) == (2, 2)          # 768 = 504 + 264 > 20 % → snapped up
    assert step_crops(4000, 3000, 728, 504, 3024) == (6, 5)         # scaled to 3024 × 2268 first
    assert step_crops(600, 300, 728, 504, 3024) == (2, 1)           # elongated small image: window = short side
    assert step_crops(10, 200, 728, 504, 3024) == (0, 0)            # extreme aspect → padded square ≤ 728
    v = C.get_model("step-5-preview").vision
    g = image_grid(v, 1920, 1080)
    assert g.views == ((52, 52, 1), (36, 36, 8)) and g.merged == 169 + 8 * 81
    assert g.llm_tokens == 171 + 8 * 83 + 1


def test_flux3_mode_cache():
    KB, K = "flux-3-image-klein-base", "flux-3-image-klein"
    off, on = _ev(KB, ref_images=1), _ev(KB, ref_images=1, mode_cache="on")
    assert off.mode_cache is None and on.mode_cache["rows"] == 4096 + 320
    mc = on.mode_cache
    assert mc["cached_tflop_seq"] < mc["extract_tflop_seq"]
    assert math.isclose(mc["extract_tflop_seq"], off.domain_summary()["dit_tflop_per_request"]
                        / off.workload.steps / off.workload.seqs_per_request, rel_tol=1e-6)
    assert on.latency < off.latency
    d = _ev(K, ref_images=1, mode_cache="on")          # distilled flow map: target time modulates every mode
    assert d.mode_cache is None and any("mode_cache" in w for w in d.warnings)
