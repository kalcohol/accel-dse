"""0.101: pipe-as-default adversarial — chunked-prefill fused iteration follows the overlap mode; energy counts are
overlap-independent (only the static / idle part moves)."""
import math
from accel_dse.core.disagg import disagg_report
from accel_dse.core.energy import energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

HB = "hbm3e_8s_12h24g_9200"


def test_chunked_fused_step_follows_overlap():
    rate = {}
    for ov in ("stage", "pipe", "serial"):
        s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=HB, layout=Layout(tp=2), exec_overlap=ov,
                     serving=Serving(batch=64, prompt=4096, out_len=512),
                     pd=PDConfig(enabled=True, prefill_layout=Layout(tp=2), prefill_cards=2, decode_cards=6))
        rate[ov] = disagg_report(s)["queue"]["modes"]["coloc_chunked"]["slo_rate_rps"]
    assert rate["stage"] > rate["pipe"] > rate["serial"]            # 0.100 bug: pipe == stage (stage rule hard-coded)


def test_energy_counts_overlap_independent():
    for kw in (dict(model="qwen3-8b", mem_id=HB, serving=Serving(phase="decode", batch=64, ctx=4096)),
               dict(model="deepseek-v3", mem_id=HB, layout=Layout(tp=8, ep=8), node_cards=8,
                    serving=Serving(phase="prefill", batch=1, prompt=8192)),
               dict(model="wan2.1-1.3b"), dict(model="esmfold")):
        a, b = (energy_report(evaluate(Scenario(chip=CHIPS["1P"], exec_overlap=ov, **kw)))["counts_per_unit"]
                for ov in ("stage", "pipe"))
        for k in a:
            if k not in ("idle_card_s", "clock"):      # 0.228: clock (MAC-slot·cycles) is time-proportional too
                assert math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-12), (kw["model"], k)
        assert b["idle_card_s"] >= a["idle_card_s"] * (1 - 1e-12)   # static energy: the slower step idles longer
        assert b["clock"] >= a["clock"] * (1 - 1e-12)               # 0.228: … and stays clocked longer


# 0.302 suite time: the SLO-goodput rate searches are skipped except in the test that reads them (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals(), skip=("test_chunked_fused_step_follows_overlap",))
