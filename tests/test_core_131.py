"""0.131 energy / power model internal consistency (hostile review): idle-only power equals the table's W per card in
every domain, MAC / vector counts are TP-invariant, DRAM + SLC bytes are conserved by the cache, MoE skew moves no
work, bit-based entries are × 8, weight bytes per token fall ~1/B, the PD helper and the single report agree, and a
table with only entries the report cannot use yields no total (was 0 J / 0 W)."""
from accel_dse.core.energy import EnergyTable, action_counts, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.pdqueue import _energy, _per_unit
from accel_dse.core.scenario import Scenario


def ev(**kw):
    s = Scenario()
    for k, v in kw.items():
        s = s.replace(k, v)
    return evaluate(s)


def test_idle_only_power_is_table_W():
    for m in ("llama-3.1-8b", "wan2.1-1.3b", "esm2-650m", "carbon-a-1.2b", "qwen-image", "openfold"):
        e = energy_report(ev(model=m), EnergyTable(idle_W=50))
        assert abs(e["avg_W_per_card"] - 50) < 1e-9, m
        assert abs(e["J_per_unit"] - 50 * e["counts_per_unit"]["idle_card_s"]) < 1e-12


def test_counts_layout_and_cache_invariants():
    base = None
    for lay in (Layout(1, 1, 1, 1, 1), Layout(1, 2, 1, 1, 1), Layout(1, 4, 1, 1, 1), Layout(2, 1, 1, 1, 1)):
        c = energy_report(ev(model="llama-3.1-8b", layout=lay, **{"serving.batch": 8}))["counts_per_unit"]
        if base is None:
            base = c
        assert abs(c["mac"] / base["mac"] - 1) < 1e-9 and abs(c["vec"] / base["vec"] - 1) < 1e-9, lay.label
    c0 = energy_report(ev(model="llama-3.2-1b", **{"serving.batch": 4}))["counts_per_unit"]
    c1 = energy_report(ev(model="llama-3.2-1b", **{"serving.batch": 4, "chip.slc_mib": 4096}))["counts_per_unit"]
    assert abs((c1["dram"] + c1["slc"]) / (c0["dram"] + c0["slc"]) - 1) < 1e-3 and c1["slc"] > 0.9 * c0["dram"]
    d = [energy_report(ev(model="llama-3.1-8b", **{"serving.batch": b}))["counts_per_unit"]["dram"] for b in (1, 2, 4)]
    assert 1.8 < d[0] / d[1] < 2.0 and 1.8 < d[1] / d[2] < 2.0          # weights / B + the per-sequence KV


def test_moe_skew_moves_no_work():
    lay = Layout(1, 8, 1, 8, 1)
    a = energy_report(ev(model="deepseek-v4-flash", layout=lay, **{"serving.batch": 1}))["counts_per_unit"]
    b = energy_report(ev(model="deepseek-v4-flash", layout=lay, **{"serving.batch": 1, "serving.moe_skew": 3.0}))
    for k in ("mac", "vec", "sram", "dram", "link"):
        assert abs(b["counts_per_unit"][k] / a[k] - 1) < 1e-9, k
    assert b["counts_per_unit"]["idle_card_s"] > 1.4 * a["idle_card_s"]     # the slower window is charged


def test_units_and_pd_helper_agree():
    r = ev(model="llama-3.1-8b", **{"serving.batch": 8})
    t = EnergyTable(pJ_mac=0.5, pJ_vec=1, pJ_bit_sram=0.1, pJ_bit_dram=5, pJ_bit_link=2, idle_W=50)
    e = energy_report(r, t)
    c = e["counts_per_unit"]
    assert abs(e["J_by_action"]["dram"] - c["dram"] * 8 * 5e-12) < 1e-15
    assert abs(e["J_by_action"]["mac"] - c["mac"] * 0.5e-12) < 1e-15
    p = _energy(_per_unit(r), c["idle_card_s"], t)
    assert abs(p["J_per_token"] - e["J_per_unit"]) < 1e-12 * e["J_per_unit"] + 1e-15
    a = action_counts(r)
    assert abs(a["units"] - r.throughput * a["window_s"]) < 1e-9 * a["units"]


def test_unusable_entries_give_no_total():
    r = ev(model="llama-3.1-8b")
    for t in (EnergyTable(idle_W_prefill=40), EnergyTable(pJ_bit_host=10)):
        e = energy_report(r, t)
        assert "J_per_unit" not in e and "avg_W_per_card" not in e and e.get("note"), t
