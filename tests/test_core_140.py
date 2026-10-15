"""0.140: system-level invariants (multi-card / fabric / PD / search) and the PP-split refinement they motivated."""
import random
from dataclasses import replace

from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import plan_stages
from accel_dse.core.scenario import Scenario
from accel_dse.core.search import best_batch


def _pp2():
    return Scenario().replace("model", "qwen3-8b-fp8").replace("mem_id", "lpddr6_4x96_10667_16g") \
        .replace("serving.phase", "prefill").replace("serving.batch", 1).replace("serving.prompt", 256) \
        .replace("serving.ctx", 1024).replace("layout.tp", 8) \
        .replace("layout.pp", 2).replace("package_cards", 2).replace("d2d_enabled", True)


def test_more_dram_channels_never_slower_with_pp_refinement():
    s = _pp2()
    a = evaluate(s).ttft
    b = evaluate(s.replace("mem_id", "lpddr6_6x96_10667_16g")).ttft
    assert b <= a * (1 + 1e-9)


def test_pp_split_no_worse_than_equal_counts():
    for m in ("qwen3-next-80b-a3b", "qwen3-8b", "gpt-oss-20b"):
        s = Scenario().replace("model", m).replace("layout.pp", 2)
        r = evaluate(s)
        r_lc = evaluate(s.replace("pp_split", "layers"))
        assert max(st.time.total for st in r.stages) <= max(st.time.total for st in r_lc.stages) * (1 + 1e-9)


def test_auto_overlap_fabric_monotone_in_net():
    s = Scenario().replace("model", "qwen3-8b").replace("serving.phase", "prefill").replace("serving.batch", 8) \
        .replace("layout.tp", 8).replace("node_cards", 4).replace("fabric.enabled", True) \
        .replace("fabric.algo", "auto_overlap").replace("mem_id", "hbm3e_4s_12h24g_9600")
    ts = [evaluate(s.replace("net", replace(s.net, GBps=g))).ttft for g in (50, 100, 200, 400)]
    assert all(b <= a * (1 + 1e-9) for a, b in zip(ts, ts[1:]))


def test_randomized_system_monotonicity():
    rng = random.Random(140)
    n = 0
    while n < 10:
        s = Scenario().replace("model", rng.choice(("qwen3-8b", "qwen3-30b-a3b", "wan2.1-1.3b", "esm2-650m")))
        try:
            s = s.replace("layout.tp", rng.choice((1, 2, 4))).replace("layout.pp", rng.choice((1, 2)))
            r0 = evaluate(s)
        except ValueError:
            continue
        n += 1
        for s2 in (s.replace("link", replace(s.link, GBps=s.link.GBps * 4)),
                   s.replace("chip", replace(s.chip, freq_ghz=s.chip.freq_ghz * 1.5)),
                   s.replace("mem_id", "lpddr5x_8x64_8533_16g")):
            r1 = evaluate(s2)
            assert r1.throughput >= r0.throughput * (1 - 1e-6), s.to_dict()


def test_best_batch_never_below_known_feasible_point():
    s = Scenario().replace("model", "qwen3-8b").replace("mem_id", "hbm3e_4s_12h24g_9600") \
        .replace("serving.tpot_slo_ms", 30.0)
    bb = best_batch(s, 64)
    for b in (1, 7, 16, 33, 64):
        r = evaluate(s.replace("serving.batch", b))
        if r.fits and r.tpot * 1e3 <= 30.0:
            assert bb.result.throughput >= r.throughput * (1 - 1e-9)
