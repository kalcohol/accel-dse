"""0.47.1 — action counts × user-supplied energy table (Accelergy-style ERT; no built-in energy numbers)."""
from __future__ import annotations

import argparse

from accel_dse import api, cli
from accel_dse.api import ApiError
from accel_dse.core.energy import EnergyTable, action_counts, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving, Workload

HBM = "hbm3e_8s_12h24g_9200"
T = EnergyTable(pJ_mac=0.4, pJ_vec=1.5, pJ_bit_sram=0.2, pJ_bit_dram=6.0, pJ_bit_link=12.0, idle_W=40.0)


def test_llm_counts_conserved_and_closed_form():
    reps = {}
    for lay in (Layout(), Layout(tp=2), Layout(pp=2)):
        r = evaluate(Scenario(model="qwen3-8b", layout=lay, serving=Serving(batch=8)))
        e = energy_report(r)
        c = e["counts_per_unit"]
        # bf16 chip: MAC slots = executed FLOPs / 2, summed over ranks and micro-batches, per output token
        tok = r.throughput * r.step
        fl = sum(s.time.flops for s in r.stages) * r.microbatches * lay.tp * lay.sp * lay.dp
        assert abs(c["mac"] - fl / 2 / tok) / c["mac"] < 1e-9 and abs(tok - 8) < 1e-9
        assert abs(c["idle_card_s"] - lay.cards * r.step / tok) < 1e-15
        reps[lay.label] = c
        assert "J_per_unit" not in e and e["provided"] == [] and len(e["missing"]) == 6
    base = reps["PP1·TP1"]
    for k, c in reps.items():
        assert abs(c["mac"] / base["mac"] - 1) < 0.01, k                 # work is conserved across layouts
    assert abs(reps["PP1·TP2"]["dram"] / base["dram"] - 1) < 0.05           # TP: weights read once per step in total
    # PP2 runs 2 micro-batches and each re-streams its stage's weights (as the stage time does) → ≈ 2× DRAM / token
    assert 1.6 < reps["PP2·TP1"]["dram"] / base["dram"] < 2.0
    assert reps["PP1·TP2"]["link"] > 0 and base["link"] == 0
    # ~2 GB of weights per step / 8 tokens + KV: DRAM per token ≈ weights / batch
    r = evaluate(Scenario(model="qwen3-8b", serving=Serving(batch=8)))
    w = r.stages[0].mem.stored_w
    assert w / 8 < base["dram"] < w / 8 * 1.5


def test_energy_is_counts_times_table():
    r = evaluate(Scenario(model="qwen3-8b", serving=Serving(batch=8)))
    e = energy_report(r, T)
    c = e["counts_per_unit"]
    want = {"mac": c["mac"] * 0.4e-12, "vec": c["vec"] * 1.5e-12, "sram": c["sram"] * 8 * 0.2e-12,
            "dram": c["dram"] * 8 * 6e-12, "link": c["link"] * 8 * 12e-12, "idle": 40 * c["idle_card_s"]}
    assert all(abs(e["J_by_action"][k] - v) <= 1e-15 + 1e-12 * v for k, v in want.items())
    assert abs(e["J_per_unit"] - sum(want.values())) < 1e-12
    assert abs(e["avg_W_per_card"] - e["J_per_unit"] * e["units_per_window"] / e["window_s"]) < 1e-9
    part = energy_report(r, EnergyTable(pJ_bit_dram=6.0))
    assert set(part["J_by_action"]) == {"dram"} and "pJ_mac" in part["missing"]
    for bad in (dict(pJ_mac=-1), dict(idle_W=True), dict(pJ_vec="1"), dict(pJ_bit_dram=float("inf"))):
        try:
            EnergyTable(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_video_components_and_protein():
    kw = dict(model="wan2.1-1.3b", mem_id=HBM)
    full = action_counts(evaluate(Scenario(**kw)))
    dit = action_counts(evaluate(Scenario(**kw, workload=Workload(pipeline=False))))
    r = evaluate(Scenario(**kw))
    comp = sum(p["acts"]["mac"] for p in r.pipeline["parts"])
    assert abs(full["counts"]["mac"] - dit["counts"]["mac"] - comp) / comp < 1e-9
    cpu = action_counts(evaluate(Scenario(**kw, workload=Workload(te_cpu=True))))
    te = sum(p["acts"]["mac"] for p in r.pipeline["parts"] if p["role"] == "text_encoder")
    assert abs(full["counts"]["mac"] - cpu["counts"]["mac"] - te) / te < 1e-9          # host energy not counted
    # tile-parallel VAE: same arithmetic, plus the tile gathers on the link
    a = evaluate(Scenario(model="minimax-h3", mem_id=HBM, layout=Layout(sp=4)))
    b = evaluate(Scenario(model="minimax-h3", mem_id=HBM, layout=Layout(sp=4), workload=Workload(vae_parallel=True)))
    va, vb = ([p for p in x.pipeline["parts"] if p["role"] == "vae"][0]["acts"] for x in (a, b))
    assert va["mac"] == vb["mac"] and vb["link"] > 0 == va["link"]
    p = evaluate(Scenario(model="protenix", mem_id=HBM, layout=Layout(sp=4)))
    e = energy_report(p, T)
    assert e["unit"] == "seq" and e["counts_per_unit"]["link"] > 0 and e["J_per_unit"] > 0


def test_api_cli_energy():
    out = api.api_eval({"scenario": {"model": "qwen3-8b"}})
    assert "counts_per_unit" in out["energy"] and "J_per_unit" not in out["energy"]
    out = api.api_eval({"scenario": {"model": "qwen3-8b"}, "energy": {"pJ_bit_dram": 6, "idle_W": 40}})
    assert out["energy"]["provided"] == ["pJ_bit_dram", "idle_W"] and out["energy"]["J_per_unit"] > 0
    for bad in ({"pJ_bit_hbm": 1}, {"pJ_mac": -0.1}, [1]):
        try:
            api.api_eval({"scenario": {"model": "qwen3-8b"}, "energy": bad})
            raise AssertionError(bad)
        except ApiError:
            pass
    p = argparse.ArgumentParser()
    cli._scenario_args(p)
    a = p.parse_args(["--model", "qwen3-8b", "--pJ-mac", "0.4", "--idle-W", "40"])
    assert cli._body(a)["energy"] == {"pJ_mac": 0.4, "idle_W": 40.0}
    assert "energy" not in cli._body(p.parse_args(["--model", "qwen3-8b"]))
