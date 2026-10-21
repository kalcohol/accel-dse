"""0.230: the PD report's energy path (pdqueue._energy) carries what the main report does — per-kind DRAM + PHY,
idle_frac_tdp × TDP, SRAM leakage, HBM standby / refresh, the per-cycle clock term with gating; invariant: a
single-phase operating point priced by _energy equals energy_report; fuzz."""
import random
from dataclasses import replace

from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _energy, _per_unit
from accel_dse.core.scenario import PDConfig, Scenario, Serving

HB = "hbm3e_8s_12h24g_9200"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _same(r, t):
    e = energy_report(r, t)
    cs = r.scenario.layout.cards * e["window_s"] / e["units_per_window"]
    p = _energy(_per_unit(r), cs, t, pools=((r.scenario, cs),))
    return e, p


def test_single_phase_matches_main_report():
    for kw in ({}, {"idle_W": 20.0}, {"tdp_W": 700.0}, {"tdp_W": 700.0, "idle_W": 5.0}):
        for g in (1.0, 0.3):
            for ph, b in (("decode", 64), ("prefill", 4), ("decode", 1)):
                sc = Scenario(chip=replace(CHIPS["1P"], clk_ungated_share=g), model="qwen3-8b", mem_id=HB)
                r = evaluate(sc.replace("serving.phase", ph).replace("serving.batch", b))
                e, p = _same(r, _t(**kw))
                assert abs(p["J_per_token"] / e["J_per_unit"] - 1) < 1e-9, (kw, g, ph, b)
                for k, v in e["J_by_action"].items():
                    assert abs(p["J_by_action"].get(k, 0.0) - v) <= 1e-12 + 1e-9 * v, (k, kw, g, ph)
                assert abs(p["static_share"] - e["static_share"]) < 1e-9 if "static_share" in e else True


def test_lpddr_and_state_sram():
    sc = Scenario(chip=replace(CHIPS["1P"], state_sram_mib=256.0), model="rwkv7-g1k-7.2b")
    r = evaluate(sc.replace("serving.phase", "decode").replace("serving.batch", 16))
    for kw in ({}, {"idle_W": 20.0}):
        e, p = _same(r, _t(**kw))
        assert abs(p["J_per_token"] / e["J_per_unit"] - 1) < 1e-9 and p["J_by_action"]["sram_leak"] > 0


def _pd(load=0.6):
    return Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HB, layout=Layout(tp=2),
                    serving=Serving(batch=64, prompt=4096, out_len=512),
                    pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6, load=load))


def test_pd_report_has_new_terms():
    q = disagg_report(_pd(), energy=_t())["queue"]
    for m in ("pd", "coloc_prefill_first", "coloc_chunked"):
        j = q["energy"][m]["J_by_action"]
        assert j["clock"] > 0 and j["hbm_static"] > 0 and j["dram_phy"] > 0 and "sram_leak" in j, m
    q2 = disagg_report(_pd(), energy=_t(tdp_W=700.0))["queue"]
    e = q2["energy"]["pd"]
    # idle_frac_tdp × TDP is now charged (was ignored by the PD path ≤ 0.229); a measured idle replaces leak / HBM static
    assert abs(e["J_by_action"]["idle"] - 71.8 * e["card_s_per_token"]) < 1e-9 and "hbm_static" not in e["J_by_action"]
    # clock: every provisioned card-second at nominal clock (g = 1)
    ch = CHIPS["1P"]
    assert abs(e["J_by_action"]["clock"] - ch.macs * ch.freq_ghz * 1e9 * _t().pJ_clk_mac_cycle * 1e-12
               * e["card_s_per_token"]) < 1e-9


def test_fuzz_pd_energy():
    rng = random.Random(230)
    for _ in range(4):
        load = rng.uniform(0.3, 0.8)
        g = rng.choice((1.0, rng.uniform(0, 1)))
        s = _pd(load).replace("chip", replace(CHIPS["1P"], clk_ungated_share=g))
        t = _t(**rng.choice(({}, {"idle_W": rng.uniform(0, 50)}, {"tdp_W": rng.uniform(200, 900)})))
        q = disagg_report(s, energy=t)["queue"]
        lam, o = q["lambda_rps"], s.serving.out_len
        for m, e in q["energy"].items():
            # J/token × tok/s/card = W/card with tok/s/card = λ·o / cards = 1 / card_s_per_token
            W = e["J_per_token"] / e["card_s_per_token"]
            assert 0 < W < 2000 and abs(e["J_per_token"] * lam * o / 8 / W - 1) < 1e-9, m
            assert 0 <= e["clock_gating"]["mac_util"] <= 1
            assert 0 <= e["static_share"] < 1


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
