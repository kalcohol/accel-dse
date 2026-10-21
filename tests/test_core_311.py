"""0.311: audit of the 0.310 memory-edge default (the +49 % at 100T + LPDDR5X 8×64, L32 is decode THROUGHPUT per card;
J/token −10 %), the host-I/O edge kept off the memory shoreline, the published H100-like rectangle, per-edge interleave of
the attach links, the ui_headless_smoke robustness, and the Ling-3.0 releases + the UBTree draft."""
import contextlib
import importlib.util
import io
import os
import time
from dataclasses import replace

from accel_dse.core.budget import auto_die_mm2, mem_edge_units, mem_sides_max
from accel_dse.core.catalog import get_model, list_models
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, die_edges, preset_die_mm2
from accel_dse.core.scenario import Scenario, Serving

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LP8 = "lpddr5x_8x64_8533_16g"
V309 = {"noc_mem_edges": 2, "noc_mem_io_edges": 0, "noc_mem_balance": "pooled"}
V310 = {"noc_mem_edges": 0, "noc_mem_io_edges": 0, "noc_mem_balance": "pooled"}


def _t():
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm"}))


def _run(chip="100T", mem=LP8, L=32, model="qwen3-8b", batch=64, **ck):
    s = Scenario(model=model, chip=replace(CHIPS[chip], noc_link_Bpc=L, **ck), mem_id=mem,
                 serving=Serving(phase="decode", batch=batch, prompt=2048, ctx=4096))
    ev = evaluate(s)
    return s, ev, energy_report(ev, _t())["J_per_unit"]


# ---------------------------------------------------------------- (1) the +49 % audit
def test_the_plus49_is_throughput_and_its_mechanism():
    s9, e9, j9 = _run(**V309)
    s0, e0, j0 = _run(**V310)
    s1, e1, j1 = _run()
    assert (s9.chip.mem_edges, s0.chip.mem_edges, s1.chip.mem_edges) == (2, 3, 3)
    # 16 engines, R = C = 4: ingress links 2R = 8 → 2R + C = 12 (pooled) → 0.311 interleave 11.636
    assert s9.chip.noc_mem_links() == 8 and s0.chip.noc_mem_links() == 12
    assert s1.chip.noc_mem_load == (11, 11, 10) and abs(s1.chip.noc_mem_links() - 11.636) < 1e-3
    assert abs(e9.per_card - 304.65) < 0.1 and abs(e0.per_card - 454.92) < 0.1 and abs(e1.per_card - 443.12) < 0.1
    assert abs(e0.per_card / e9.per_card - 1.4932) < 1e-3          # the 0.310 "+49 %" = tok/s per card
    assert abs(j0 / j9 - 1 + 0.1019) < 1e-3                        # J/token −10.2 %, not +49 %
    assert abs(e1.per_card / e9.per_card - 1.4545) < 1e-3 and abs(j1 * 1e3 - 63.285) < 1e-2
    # at L = 0 (no NoC limit) the value is the DRAM roofline: 3 edges lifted decode off the NoC-ingress bound
    assert abs(_run(L=0)[1].per_card - 454.92) < 0.1


def test_lpddr_third_edge_is_physical_and_leaves_io_edge():
    s, _, _ = _run()
    assert mem_sides_max(s) == 3                                   # 4 − 0 D2D − 1 host-I/O
    assert mem_sides_max(replace(s, chip=replace(s.chip, noc_mem_io_edges=0))) == 4
    a = auto_die_mm2(s)
    e = die_edges(s.chip, a)
    assert abs(a - 107.9) < 0.1 and len(e) == 4 and abs(e[0] - 10.39) < 0.02   # square n5_derived proxy
    from accel_dse import mem_catalog as mc
    n, mm = mem_edge_units(mc.parse_mem_id(LP8))                            # 32 x16 channels × 16 / 18.75 mm
    assert n == sum(s.chip.noc_mem_load) == 32 and abs(mm - 16 / 18.75) < 1e-9
    assert abs(n * mm / 3 - 9.1) < 0.05 and n * mm / 3 < e[0]       # 9.1 mm per memory edge on a 10.39 mm edge


def test_io_edge_and_published_rectangle():
    s, _, _ = _run(chip="1P", mem="hbm3e_8s_8h24g_9200")
    assert s.chip.mem_edges == 2                                   # 0.310: 4 (no edge left for host I/O)
    assert _run(chip="1P", mem="hbm3e_8s_8h24g_9200", **V310)[0].chip.mem_edges == 4
    h = _run(chip="H100-like", mem="hbm3e_6s_8h24g_9200")[0]
    assert h.chip.mem_edges == 2 and h.chip.noc_mem_load == (3, 3)  # 33 mm long edges: 3 sites each
    assert abs(die_edges(h.chip, preset_die_mm2(h.chip))[0] - 33.0) < 1e-9
    h8 = _run(chip="H100-like", mem="hbm3e_8s_8h24g_9200")[0]
    assert h8.chip.mem_edges == 3 and h8.chip.noc_mem_load == (3, 3, 2)


def test_single_stack_interleave_and_pooled_switch():
    s, e, _ = _run(chip="1P", mem="hbm3e_1s_8h24g_9200")
    p, ep, _ = _run(chip="1P", mem="hbm3e_1s_8h24g_9200", noc_mem_balance="pooled")
    assert s.chip.noc_mem_load == (1, 0) and s.chip.noc_mem_links() * 2 == p.chip.noc_mem_links()
    assert abs(e.per_card / ep.per_card - 0.5) < 1e-3              # one stack sits on one edge
    d = s.to_dict()["chip"]
    assert "noc_mem_load" not in d and "noc_mem_io_edges" not in d and "noc_mem_balance" not in d
    for bad in ({"noc_mem_balance": "x"}, {"noc_mem_io_edges": 3}):
        try:
            replace(CHIPS["1P"], **bad)
            raise AssertionError(bad)
        except ValueError:
            pass


# ---------------------------------------------------------------- (2) ui_headless_smoke robustness
def _smoke():
    spec = importlib.util.spec_from_file_location("ui_hs", os.path.join(ROOT, "scripts", "ui_headless_smoke.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_smoke_steps_and_guard():
    m = _smoke()
    st = m._Steps(time.time())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st.append(("panel", "ok"))
    assert st.last == "panel" and "STEP [" in buf.getvalue()

    class _Srv:
        returncode = None

        def poll(self):
            return None
    errs = []
    with m._Guard(errs, st, _Srv(), "/nonexistent.log"):
        raise m.StepTimeout("wait 'x' 60s")
    assert len(errs) == 1 and "after step 'panel'" in errs[0] and "StepTimeout" in errs[0] and "alive" in errs[0]
    src = open(os.path.join(ROOT, "scripts", "ui_headless_smoke.py")).read()
    # every page wait goes through the per-step helpers (1 raw call each, inside wait_js / wait_sel): 11 + 2 sites
    assert src.count("p.wait_for_function(") == 1 and src.count("p.wait_for_selector(") == 1
    assert src.count("wait_js(") >= 12 and src.count("wait_sel(") >= 3
    assert "stderr=subprocess.PIPE" not in src and "--budget" in src


# ---------------------------------------------------------------- (3) Ling-3.0 + UBTree
def test_ling30_releases_and_ubtree_pair():
    L = {x["id"]: x for x in list_models()}
    for mid in ("ling-3.0-tiny", "ling-3.0-flash", "ling-3.0-tiny-ubtree"):
        assert mid in L and abs(L[mid]["param_err"]) < 1e-3 and L[mid]["family"] == "Ling-3.0"
    m = get_model("ling-3.0-tiny-ubtree")
    assert m.draft == "dspark" and m.draft_anchor and m.draft_block == 16 and m.draft_targets == 5
    assert m.draft_ubtree == (64, 256, 512, 15, 64, 49) and m.spec_published == (9.113, 15)

    def ev(k, src="published", **kw):
        return evaluate(Scenario(model="ling-3.0-tiny-ubtree" if k else "ling-3.0-tiny", chip=CHIPS["1P"],
                                 mem_id="hbm3e_4s_8h24g_9200",
                                 serving=Serving(phase="decode", batch=1, ctx=4096, spec_k=k, spec_accept_src=src, **kw)))
    base, r = ev(0), ev(15)
    assert r.spec["n"] == 50 and abs(r.spec["tau"] - 9.113) < 0.01  # 1 + 49-node tree verified
    assert any("UBTree" in w and "「假设」" in w for w in r.warnings)
    assert r.tpot < base.tpot
    assert ev(15, spec_draft_tokens=64).spec["n"] == 64
    try:
        ev(15, spec_draft_tokens=10)              # < 1 + k
        raise AssertionError
    except ValueError:
        pass
