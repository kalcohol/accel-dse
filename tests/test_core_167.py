"""0.167: TeaCache-style step caching for DiT generation (skipped steps: embeddings + head + residual add)."""
import json
import subprocess
import sys

from accel_dse.core.evaluate import STEP_CACHE_FORCED, evaluate
from accel_dse.core.scenario import Scenario, Workload

M = "wan2.1-14b"


def _s(k=0, mode="teacache", **kw):
    s = Scenario(model=M)
    for p, v in kw.items():
        s = s.replace(p, v)
    return s.replace("workload.step_skip", k).replace("workload.step_cache", mode)


def test_sourced_forced_steps_and_validation():
    assert STEP_CACHE_FORCED == {"teacache": (1, 1), "teacache_ret": (5, 0)}
    assert evaluate(_s(48)).step_cache["computed"] == 2                 # 50 steps, first + last forced
    assert evaluate(_s(45, "teacache_ret")).step_cache["computed"] == 5
    for k, mode in ((49, "teacache"), (46, "teacache_ret")):
        try:
            evaluate(_s(k, mode))
        except ValueError as e:
            assert "可跳步数" in str(e)
        else:
            raise AssertionError((k, mode))
    for bad in (dict(step_skip=-1), dict(step_skip=True), dict(step_cache="fb")):
        try:
            Workload(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_off_is_identical_and_latency_algebra():
    r0, r00 = evaluate(Scenario(model=M)), evaluate(_s(0))
    assert r0.latency == r00.latency and r0.step_cache is None and Scenario(model=M).hash() == _s(0).hash()
    r = evaluate(_s(20))
    sc = r.step_cache
    steps = r0.workload.steps
    t_full = (r0.latency - (r0.pipeline or {}).get("te_s", 0) - (r0.pipeline or {}).get("decode_s", 0)
              - (r0.pipeline or {}).get("load_s", 0)) / steps
    assert 0 < sc["t_skip_ms"] / 1e3 < 0.05 * t_full                   # embed + head ≪ 40 blocks
    assert abs((r0.latency - r.latency) - 20 * (t_full - sc["t_skip_ms"] / 1e3)) < 1e-6 * r0.latency
    # FLOPs: 30 full forwards + 20 skipped forwards
    f0, f = r0.domain_summary()["dit_tflop_per_request"], r.domain_summary()["dit_tflop_per_request"]
    assert abs(f - (f0 * 30 / 50 + 20 * sc["tflop_seq_skip"] * r0.workload.seqs_per_request)) < 1e-6 * f0


def test_residual_buffer_memory_and_ignores():
    r0, r = evaluate(Scenario(model=M)), evaluate(_s(10))
    tok, h = r0.workload.tokens, r0.model.hidden
    seqs = r0.workload.seqs_per_request                                   # cond + uncond → 2 cached residuals
    assert abs(r.step_cache["buffer_GiB_card"] * 2**30 - seqs * tok * h * 2) < 1                     # bf16 「假设」
    assert abs(r.stages[0].mem.dram_need - r0.stages[0].mem.dram_need - r.step_cache["buffer_GiB_card"] * 2**30) < 1
    rp = evaluate(_s(10, **{"layout.pp": 2}))
    assert rp.step_cache is None and any("step_skip 已忽略" in w for w in rp.warnings)
    rl = evaluate(Scenario().replace("workload.step_skip", 3))
    assert any("step_skip 已忽略" in w for w in rl.warnings)


def test_cli_and_monotone():
    lat = [evaluate(_s(k)).latency for k in (0, 10, 20, 30)]
    assert all(a > b for a, b in zip(lat, lat[1:]))
    out = subprocess.run([sys.executable, "-m", "accel_dse.cli", "eval", "--model", M, "--step-skip", "20",
                          "--step-cache", "teacache_ret", "--json"], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    assert d["scenario"]["workload"]["step_skip"] == 20
    assert d["summary"]["gen"]["step_cache"]["mode"] == "teacache_ret"
