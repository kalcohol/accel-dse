"""0.257: freshness sweep (scripts/freshness_sweep.py: every PROVIDERS HF org + OpenMOSS-Team); MOSS-Image
Base / SFT (single-stream DiT, Qwen3.5-9B hybrid text tower with Gated DeltaNet layers, FLUX.2 small-decoder VAE);
the PD report's per-pool energy ledger (pdqueue._energy pool_counts: per-pool DRAM kind + PHY, per-pool gating
utilisation, J_by_pool)."""
import importlib.util
import os
import random
from dataclasses import replace

from accel_dse.core import catalog as C
from accel_dse.core.energy import EnergyTable, apply_energy_preset
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _energy
from accel_dse.core.pipeline import PIPELINES, component_data, text_ops
from accel_dse.core.scenario import Scenario, Serving

HB = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sweep():
    spec = importlib.util.spec_from_file_location("freshness_sweep", os.path.join(_ROOT, "scripts", "freshness_sweep.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sweep_covers_every_provider_org():
    orgs = _sweep().sweep_orgs()
    assert "OpenMOSS-Team" in orgs and len(orgs) == len(set(orgs))
    for p in C.PROVIDERS:
        for pre in p[2]:
            if pre.endswith("/") and ":" not in pre:            # HF orgs (gh: prefixes are GitHub repos)
                assert pre[:-1] in orgs, pre


def test_moss_image_exact_and_merged():
    for mid in ("moss-image-base", "moss-image-sft"):
        m = C.get_model(mid)
        assert m.coverage == "full" and not m.coverage_reasons, mid
        assert m.param_check()["rel_err"] == 0.0 and m.release_params == 5082308624, mid
        assert len(m.layers) == 42 and m.workload.cfg == 2 and m.workload.steps == 50
    assert C.MERGED["moss-image-sft"] == "moss-image-base"


def test_moss_text_tower_is_hybrid():
    te = PIPELINES["moss-image-base"].text[0]
    assert te.full_layers == tuple(range(3, 32, 4)) and len(te.full_layers) == 8
    ops, _ = text_ops(te, 1, "bf16")
    names = [o.name for o in ops]
    assert sum(n.endswith(".softmax") for n in names) == 8                  # softmax attention only on full layers
    assert sum(n.endswith(".lin_akk") for n in names) == 24                 # Gated DeltaNet on the other 24
    assert component_data("moss-vae")["dtype"] == "fp32"
    assert PIPELINES["moss-image-base"].vae.family == "os_s384"


def _sc(chip="1P", mem=HB, g=None):
    ch = CHIPS[chip] if g is None else replace(CHIPS[chip], clk_ungated_share=g)
    return Scenario(model="qwen3-8b", chip=ch, mem_id=mem, layout=Layout(tp=2), serving=Serving(batch=64))


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _counts(rng):
    return {k: rng.uniform(0, 1) * s for k, s in (("mac", 3e9), ("vec", 1e8), ("sram", 2e9), ("dram", 3e9),
                                                    ("link", 1e7))}


def _split(rng, tot):
    a = {k: v * rng.uniform(0, 1) for k, v in tot.items()}
    return a, {k: tot[k] - a[k] for k in tot}


def test_ledger_sums_and_same_chip_matches_pooled():
    rng = random.Random(257)
    for _ in range(40):
        g = rng.choice((None, 0.25, 0.6))
        t = _t(**rng.choice(({}, {"idle_W": 30.0}, {"tdp_W": 400.0})))
        tot = _counts(rng)
        d, p = _split(rng, tot)
        cs = (rng.uniform(1e-4, 1e-3), rng.uniform(1e-5, 1e-3))
        pools = ((_sc(g=g), cs[0]), (_sc(g=g), cs[1]))
        old = _energy(tot, sum(cs), t, cs[1], pools=pools)
        new = _energy(tot, sum(cs), t, cs[1], pools=pools, pool_counts=(d, p), pool_names=("decode", "prefill"))
        # identical chips / memory / gating: every term is linear in the counts → the pooled form was already exact
        assert abs(new["J_per_token"] - old["J_per_token"]) <= 1e-9 * old["J_per_token"]
        for k, v in old["J_by_action"].items():
            assert abs(new["J_by_action"][k] - v) <= 1e-9 * max(v, 1e-30), k
        bp = new["J_by_pool"]
        assert abs(sum(x["J_per_token"] for x in bp.values()) - new["J_per_token"]) <= 1e-12 * new["J_per_token"]
        for k, v in new["J_by_action"].items():
            assert abs(sum(x["J_by_action"].get(k, 0.0) for x in bp.values()) - v) <= 1e-12 * max(v, 1e-30), k


def test_hetero_memory_and_gating_priced_per_pool():
    t = _t()
    d = {"mac": 1e8, "dram": 4e9, "sram": 1e9}
    p = {"mac": 2e9, "dram": 2e8, "sram": 5e8}
    tot = {k: d[k] + p[k] for k in d}
    cs = (6e-4, 2e-4)
    pools = ((_sc(mem=HB, g=0.25), cs[0]), (_sc(mem=LP, g=1.0), cs[1]))
    e = _energy(tot, sum(cs), t, cs[1], pools=pools, pool_counts=(d, p), pool_names=("decode", "prefill"))
    want = (d["dram"] * t.pJ_bit_dram_hbm + p["dram"] * t.pJ_bit_dram_lpddr) * 8e-12
    assert abs(e["J_by_action"]["dram"] - want) <= 1e-12 * want
    assert e["J_by_pool"]["decode"]["dram_field"] == "pJ_bit_dram_hbm"
    assert e["J_by_pool"]["prefill"]["dram_field"] == "pJ_bit_dram_lpddr"
    ch = pools[0][0].chip
    cyc = [ch.macs * ch.freq_ghz * 1e9 * c for c in cs]
    u = e["clock_gating"]["pool_mac_util"]
    assert abs(u["decode"] - d["mac"] / cyc[0]) < 1e-12 and abs(u["prefill"] - p["mac"] / cyc[1]) < 1e-12
    clk = t.pJ_clk_mac_cycle * 1e-12 * (cyc[0] * (0.25 + 0.75 * u["decode"]) + cyc[1])
    assert abs(e["J_by_action"]["clock"] - clk) <= 1e-12 * clk
    # the 0.230 pooled form priced all DRAM as HBM and gated with one utilisation
    old = _energy(tot, sum(cs), t, cs[1], pools=pools)
    # 0.333: was `old < e` while HBM was the cheaper kind per bit (3.44 vs LPDDR 4.0); with HBM 5.80 the pooled form
    # now over-prices — check the mechanism itself: pooled = all bytes at the HBM rate, ≠ the per-pool sum
    assert abs(old["J_by_action"]["dram"] - tot["dram"] * t.pJ_bit_dram_hbm * 8e-12) <= 1e-12 * old["J_by_action"]["dram"]
    assert abs(old["J_by_action"]["dram"] - e["J_by_action"]["dram"]) > 1e-6 * e["J_by_action"]["dram"]
    assert old["J_by_action"]["clock"] != e["J_by_action"]["clock"]


def test_idle_prefill_booked_to_prefill_pool():
    t = _t(idle_W=40.0, idle_W_prefill=90.0)
    d, p = {"mac": 1e8, "dram": 1e9}, {"mac": 1e9, "dram": 1e8}
    tot = {k: d[k] + p[k] for k in d}
    cs = (5e-4, 1e-4)
    pools = ((_sc(), cs[0]), (_sc(), cs[1]))
    e = _energy(tot, sum(cs), t, cs[1], pools=pools, pool_counts=(d, p), pool_names=("decode", "prefill"))
    bp = e["J_by_pool"]
    assert abs(bp["prefill"]["J_by_action"]["idle_prefill"] - 90.0 * cs[1]) < 1e-15
    assert "idle" not in bp["prefill"]["J_by_action"] and "idle_prefill" not in bp["decode"]["J_by_action"]
    assert abs(bp["decode"]["J_by_action"]["idle"] - 40.0 * cs[0]) < 1e-15
    old = _energy(tot, sum(cs), t, cs[1], pools=pools)
    assert abs(old["J_per_token"] - e["J_per_token"]) <= 1e-9 * e["J_per_token"]


def test_without_pool_counts_unchanged():
    t = _t()
    tot = {"mac": 1e9, "dram": 1e9}
    pools = ((_sc(), 3e-4), (_sc(mem=LP), 1e-4))
    a = _energy(tot, 4e-4, t, 1e-4, pools=pools)
    assert "J_by_pool" not in a and "pool_mac_util" not in a["clock_gating"]
    assert abs(a["J_by_action"]["dram"] - 1e9 * t.pJ_bit_dram_hbm * 8e-12) < 1e-15   # first pool's kind (0.230)
