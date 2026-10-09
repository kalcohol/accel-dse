"""0.61.2 — round-2 audit fixes: the occupancy correlation time of the decode chain is finite at large mean
occupancy (the top-down Poisson-equation sum cancelled to rounding noise below the mean and was divided by a
left-tail π ~ e^-a)."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

from accel_dse.core.pdqueue import _decode_birth_death, _occ_tau


def _poisson(a, N):
    return [math.exp(k * math.log(a) - a - math.lgamma(k + 1)) if k else math.exp(-a) for k in range(N)]


def test_occ_tau_mminf_large_occupancy():
    """M/M/∞ occupancy is an OU-like chain with τ_int = 1/μ exactly, for every offered load a."""
    for a in (5.0, 64.0, 256.0, 900.0):
        mu = 0.04
        lam = a * mu
        N = int(a + 12 * math.sqrt(a) + 40)
        pi = _poisson(a, N)
        tau = _occ_tau(pi, 10 ** 7, (lam, [1.0]), lambda k: k * mu)
        assert tau is not None and abs(tau - 1 / mu) < 1e-3 / mu, (a, tau)


def test_decode_chain_tau_finite_at_large_batch():
    """Constant step s, B ≫ offered load: the decode replica is M/M/∞ with μ = 1/(out·s) → τ_int = out·s."""
    R = NS(fits=True, tokens_per_step=1.0)
    for B, lam, out, s in [(2000, 10.0, 256, 0.1), (5000, 40.0, 100, 0.25)]:
        d = _decode_birth_death(None, lam, out, B, step_fn=lambda k: (s, R))
        assert abs(d["tau_int"] - out * s) < 1e-3 * out * s, (B, d["tau_int"])
    # B = 300 at a = 256 Erlangs (the cap binds in the right tail): exact rational evaluation of
    # Σ G(n)²/(π_n μ_n) / Var k  (fractions.Fraction, top-down G, no rounding) = 25.712311352000853
    d = _decode_birth_death(None, 10.0, 256, 300, step_fn=lambda k: (0.1, R))
    assert abs(d["tau_int"] - 25.712311352000853) < 1e-9, d["tau_int"]


def test_radix_che_node_larger_than_cache():
    """A radix node longer than the cache is never resident, nor is anything below it; the rest of the tree gets the
    whole capacity (Che and the DES radix LRU agree)."""
    import random

    from accel_dse.core.pdsim import RadixLRU, _tree_paths
    from accel_dse.core.prefixcache import tree_depth_tail, tree_groups, tree_resident
    # level 2 (5000 tokens) cannot fit in 3000: the 10 × 100-token roots all fit → H = [1, 0]  (was [0.0995, 0.0190])
    lv = [(100, 10, 1.0), (5000, 10, 1.0)]
    g = tree_groups(lv)
    assert tree_depth_tail(g, tree_resident(lv, 3000, g)) == [1.0, 0.0]
    # the root itself is too long → nothing ever matches (was [0.583, 0.065])
    lv2 = [(5000, 1, 0.0), (100, 50, 1.0)]
    g2 = tree_groups(lv2)
    assert tree_depth_tail(g2, tree_resident(lv2, 3000, g2)) == [0.0, 0.0]
    for levels, want in ((lv, [1.0, 0.0]), (lv2, [0.0, 0.0])):
        r = RadixLRU(3000, [L for L, _, _ in levels])
        for p in _tree_paths(random.Random(3), levels, 5000):
            r.access(p)
        assert r.used <= 3000 and all(len(k) <= (1 if levels is lv else 0) for k in r.od)
        tail = [sum(r.depth_n[k:]) / r.looks for k in range(1, 3)]
        assert abs(tail[0] - want[0]) < 0.01 and tail[1] == 0.0


def test_area_lower_bound_with_missing_density_is_definite_violation():
    """Missing a density leaves the area undecidable only while the known terms are under the limit."""
    from dataclasses import replace

    from accel_dse.core.budget import Budget, budget_report
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.scenario import Scenario, Serving
    r = evaluate(Scenario(model="qwen3-8b", serving=Serving(batch=8)))
    sram = r.scenario.chip.sram_mib
    b = Budget(die_mm2=100, mm2_per_mib_sram=0.5, mm2_fixed=20)          # mm2_per_kmac missing
    it = budget_report(r, b)["items"][0]
    assert it["ok"] is None and it["value"] is None                        # 0.5·SRAM + 20 ≤ 100 → undecidable
    tight = budget_report(r, replace(b, die_mm2=10))["items"][0]           # known terms alone exceed 10 mm²
    assert tight["ok"] is False and abs(tight["value"] - (0.5 * sram + 20)) < 1e-9 and "下界" in tight["note"]


def test_ltx_vae_resnets_after_upsample():
    """diffusers LTXVideoUpBlock3d: conv_in → upsamplers → resnets.  Decode of the 121 × 512 × 704 latent
    (16, 16, 22) = 38.05 TFLOP by torch.utils.flop_counter on AutoencoderKLLTXVideo (Lightricks/LTX-Video vae
    config, meta tensors; diffusers 0.41) — 0.44–0.61.1 gave 11.6 (resnets at the pre-upsample resolution)."""
    from accel_dse.core.pipeline import PIPELINES, vae_ops
    ops, _, info = vae_ops(PIPELINES["ltx-video"].vae, (16, 16, 22), 121, 1, "bf16")
    f = sum(o.flops for o in ops)
    assert abs(f / 38.05e12 - 1) < 1e-3, f
    assert info["out"] == (121, 128, 176)        # conv grid before the 4 × 4 unpatchify of conv_out
    by = {}
    for o in ops:
        if ".resnets." in o.name:
            blk = int(o.name.split("up_blocks.")[1].split(".")[0]) if "up_blocks." in o.name else -1
            by.setdefault(blk, o.m)
    assert by[3] == 121 * 128 * 176 and by[1] == 31 * 32 * 44     # each block's resnets at its upsampled grid


def test_esmfold_default_trunk_passes():
    """esm / transformers: num_recycles=None → the trunk loop runs max_recycles = 4 times in total.  384 residues:
    transformers EsmForProteinFolding under torch.utils.flop_counter (meta tensors) = 50.24 TFLOP; 512 → 97.32."""
    from accel_dse.core.evaluate import evaluate
    from accel_dse.core.scenario import Scenario, Workload
    for L, ref in ((384, 50.24e12), (512, 97.32e12)):
        r = evaluate(Scenario(model="esmfold", mem_id="hbm3e_8s_12h24g_9200", workload=Workload(seq_len=L)))
        assert r.workload.recycles == 4 if hasattr(r.workload, "recycles") else True
        f = r.domain_summary()["tflop_per_request"] * 1e12
        assert abs(f / ref - 1) < 1e-3, (L, f)
