"""0.141: per-core accumulator / SRAM-port defaults (+ legacy migration), GPU-validation default variant,
moe_rank_max = "expected" default, fabric.algo = "auto_overlap" default, PD smallest-sufficient prefill batch."""
from dataclasses import replace

from accel_dse.core.disagg import disagg_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import ACC_KIB_PER_ENGINE, Chip, Fabric
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core import extval


def test_per_core_defaults_scale_with_engines():
    for e in (1, 4, 16):
        c = Chip("t", 1.0, 56, 56, e)
        assert c.acc_total_kib == ACC_KIB_PER_ENGINE * e
        assert c.port_Bpc == 4 * (56 + 56) * 2 * e
    c = Chip("t", 1.0, 56, 56, 16, acc_kib_per_engine=32.0)
    assert c.acc_total_kib == 32.0 * 16
    try:
        Chip("t", 1.0, 56, 56, 16, acc_kib=128.0, acc_kib_per_engine=32.0)
    except ValueError:
        pass
    else:
        raise AssertionError("acc_kib and acc_kib_per_engine are exclusive")


def test_more_engines_never_slower_under_auto():
    for m, k, n in ((1, 4096, 4096), (64, 4096, 14336), (2048, 4096, 4096), (7, 1000, 333)):
        prev = None
        for e in (1, 2, 4, 8, 16, 32):
            cy = gemm_cost(Chip("t", 1.0, 64, 64, e), "reconf", m, k, n).cycles
            if prev is not None:
                assert cy <= prev * (1 + 1e-9), (m, k, n, e)
            prev = cy


def test_legacy_scenario_migrates_to_old_hardware():
    old = {"model": "qwen3-8b", "chip": {"name": "x", "freq_ghz": 1.0, "rows": 56, "cols": 56, "engines": 16,
                                        "acc_kib": 1024.0, "sram_port_Bpc": None}}
    s = Scenario.from_dict(old)
    assert s.chip.port_Bpc == 4 * (56 + 56 * 16) * 2 == 7616 and s.chip.acc_total_kib == 1024.0
    assert Scenario.from_dict(s.to_dict()).hash() == s.hash()
    new = Scenario.from_dict({"model": "qwen3-8b"})
    assert "acc_kib" not in new.to_dict()["chip"] and Scenario.from_dict(new.to_dict()).hash() == new.hash()


def test_validation_default_is_candidate_ceil_variant():
    # 0.178: the ≤ 0.177 default (candidate + gceil, flat lump) is now "default_lump"; default = its mechanism (§125)
    assert extval.VARIANTS["default_lump"] == extval.VARIANTS[extval.ceil_variant(extval.candidate_variant())]


def test_new_defaults():
    s = Scenario()
    assert s.serving.moe_rank_max == "expected" and Fabric().algo == "auto_overlap"
    f = Scenario(model="llama-3.1-8b", fabric=Fabric(enabled=True), serving=Serving(batch=64))
    assert evaluate(f).step <= evaluate(f.replace("fabric.algo", "auto")).step * (1 + 1e-12)


def test_pd_prefill_batch_smallest_sufficient():
    s = Scenario().replace("pd.enabled", True)
    r = disagg_report(s)
    pb = r["prefill"]["batch"]
    assert r["ttft_ok"] and r["ttft_ms"] < 1000         # ≤ 0.140: 1298 ms at the same goodput
    assert pb == 1                                       # λ is decode-bound: batch 1 carries it


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
