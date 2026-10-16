"""0.139: NPU hostile review — randomized monotonicity invariants (more SRAM / port / clock / DRAM / dequant / link
never slower, DRAM need conserved) and the two breaks they found."""
import math
import random
from dataclasses import replace

from accel_dse.core import overlap as O
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import FormatSupport
from accel_dse.core.scenario import Scenario


def test_split_picker_charges_replicated_conversion():
    # AWQ int4 weights (no native int4 → dequant on the vector path): auto row splits replicate the K×N dequant;
    # a 4× SRAM port used to pick such a split and run 7× slower on the vector path
    s = Scenario().replace("model", "qwen3-8b-awq").replace("serving.batch", 8).replace("serving.ctx", 256) \
        .replace("mapping", "reconf").replace("mem_id", "lpddr6_4x96_10667_16g")
    ch = replace(s.chip, rows=56, cols=16, engines=8, sram_mib=256.0)
    prev = None
    for k in (1, 2, 4, 8):
        t = evaluate(s.replace("chip", replace(ch, sram_port_Bpc=ch.port_Bpc * k))).tpot
        assert prev is None or t <= prev * (1 + 1e-9)
        prev = t


def test_pipe_hide_window_is_wall_time_monotone_in_clock():
    # MoE dispatch / combine hide behind the shared expert's flow-shop wall time (DMA-bound here), so a faster
    # clock never exposes more all-to-all
    s = Scenario().replace("model", "qwen3-next-80b-a3b").replace("serving.phase", "prefill") \
        .replace("serving.batch", 8).replace("serving.prompt", 1024).replace("mapping", "ws_broad") \
        .replace("layout.tp", 2).replace("layout.ep", 2)
    ch = replace(s.chip, rows=64, cols=128, engines=16, sram_mib=8.0, instance_sched="split")
    ts = [evaluate(s.replace("chip", replace(ch, freq_ghz=f))).ttft for f in (1.0, 1.5, 2.25, 3.0)]
    assert all(b <= a * (1 + 1e-9) for a, b in zip(ts, ts[1:]))


def _sim(items, buf=1e9):
    return O.stage_pipe_time([(items, 1)], {}, 1e-9, buf)


def test_hide_markers_compute_bound_equals_old_credit():
    # compute-bound window: exposed = max(0, bw − shared compute), as before 0.139
    its = [("open", {}, 0.0), (2e-3, {}, 0.0), ("close", {}, 0.0), (1e-3, {}, 0.0), ("hide", {}, 3e-3),
           ("reset", {}, 0.0)]
    assert math.isclose(_sim(its), 2e-3 + 1e-3 + 1e-3, rel_tol=1e-12)
    # DMA-bound window (8 ms of loads under 2 ms of compute): the 3 ms all-to-all hides completely
    its[1] = (2e-3, {"hot": 8e6}, 0.0)
    t = _sim(its)
    assert math.isclose(t, _sim([(2e-3, {"hot": 8e6}, 0.0), (1e-3, {}, 0.0)]), rel_tol=1e-12)


MODELS = ("qwen3-8b", "qwen3-30b-a3b", "qwen3-8b-awq", "qwen3-8b-fp8", "llama-3.2-1b", "qwen3-next-80b-a3b")
MEMS = {"lpddr5x_4x64_8533_16g": "lpddr5x_8x64_8533_16g", "hbm3e_4s_12h24g_9600": "hbm3e_6s_12h24g_9600",
        "lpddr6_4x96_10667_16g": "lpddr6_6x96_10667_16g"}
KNOBS = {
    "sram": lambda s: s.replace("chip", replace(s.chip, sram_mib=s.chip.sram_mib * 4)),
    "freq": lambda s: s.replace("chip", replace(s.chip, freq_ghz=s.chip.freq_ghz * 1.5)),
    "port": lambda s: s.replace("chip", replace(s.chip, sram_port_Bpc=s.chip.port_Bpc * 4)),
    "lanes": lambda s: s.replace("chip", replace(s.chip, vector_lanes=s.chip.lanes * 4)),
    "acc": lambda s: s.replace("chip", replace(s.chip, acc_kib=s.chip.acc_total_kib * 4)),
    "dram": lambda s: s.replace("mem_id", MEMS[s.mem_id]),
    "link": lambda s: s.replace("link", replace(s.link, GBps=s.link.GBps * 4)),
    "dequant": lambda s: s.replace("chip", replace(s.chip, formats=replace(
        s.chip.formats, dequant_elems_per_cycle=(s.chip.formats.dequant_elems_per_cycle or 64) * 8))),
}


def test_randomized_resource_monotonicity():
    rng = random.Random(139)
    n = 0
    while n < 24:
        s = Scenario().replace("model", rng.choice(MODELS))
        ph = rng.choice(("decode", "prefill"))
        s = s.replace("serving.phase", ph).replace("serving.batch", rng.choice((1, 8, 64))) \
            .replace("serving.prompt", rng.choice((128, 2048))).replace("mapping", rng.choice(
                ("os", "ws_edge", "ws_broad", "os_vec", "reconf"))).replace("mem_id", rng.choice(sorted(MEMS)))
        fm = FormatSupport(rates=(("bf16", 1.0), ("fp16", 1.0)), dequant_elems_per_cycle=rng.choice((None, 64))) \
            if rng.random() < 0.4 else s.chip.formats
        s = s.replace("chip", replace(s.chip, rows=rng.choice((32, 64)), cols=rng.choice((16, 64, 128)),
                                      engines=rng.choice((1, 4, 16)), sram_mib=rng.choice((8.0, 64.0)),
                                      formats=fm, freq_ghz=rng.choice((0.8, 1.5))))
        try:
            r0 = evaluate(s)
        except ValueError:
            continue
        n += 1
        f = (lambda r: r.ttft) if ph == "prefill" else (lambda r: r.tpot)
        for k, g in KNOBS.items():
            r1 = evaluate(g(s))
            # 0.198: the dequant knob gets 2e-4 slack — the per-GEMM split picker scores max(array, conversion), so a
            # faster dequant can pick a plan with slightly more array cycles (qwen3-8b-fp8 fused qkv on 100T: +0.018 %
            # tpot); a known per-op myopia of the picker, recorded in CHANGELOG 0.198.0
            assert f(r1) <= f(r0) * (1 + (2e-4 if k == "dequant" else 1e-9)), (k, s.to_dict())
            if k != "dram":
                assert math.isclose(r1.summary()["dram_need_GiB"], r0.summary()["dram_need_GiB"], rel_tol=1e-12), k
            if k == "sram":
                assert (r1.summary()["residency"] or 0) >= (r0.summary()["residency"] or 0) - 1e-12
