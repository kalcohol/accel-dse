"""0.164: dram_bw sizing uses min(memory, DMA issue limit); opt-in host / runtime step overhead."""
from accel_dse import api as A
from accel_dse.core.evaluate import evaluate
from accel_dse.core.insight import SIZING, sizing_report
from accel_dse.core.scenario import Scenario


def _dma(s, engines, kib, ns):
    return s.replace("chip", s.chip.__class__(**{**s.chip.__dict__, "dma_engines": engines, "dma_outstanding_KiB": kib,
                                                  "dram_latency_ns": ns}))


def test_dram_bw_reports_delivered_bandwidth_with_dma_cap():
    s = Scenario()
    mem = SIZING["dram_bw"][2](s)
    cap = 4 * 4 * 1024 / (500e-9) / 1e9                       # 4 engines × 4 KiB / 500 ns = 32.8 GB/s
    d = _dma(s, 4, 4.0, 500.0)
    assert abs(SIZING["dram_bw"][2](d) - min(mem, cap)) < 1e-9 and SIZING["dram_bw"][2](d) < mem
    t = SIZING["dram_bw"][3](d, cap / 2)                        # below the cap the memory sets delivered bandwidth
    assert abs(SIZING["dram_bw"][2](t) - cap / 2) < 1e-6
    r = sizing_report(d, ["qwen3-8b"], resources=["dram_bw"])["resources"][0]
    assert abs(r["current"] - cap) < 1e-9 and r["min_ok"] <= r["current"]
    assert SIZING["dram_bw"][2](_dma(s, 64, 64.0, 100.0)) == mem   # cap far above memory → memory


def test_host_overhead_off_is_identical_and_modes():
    for ph in ("decode", "prefill"):
        s = Scenario().replace("serving.phase", ph)
        r0 = evaluate(s)
        assert r0.host is None and s.hash() == Scenario().replace("serving.phase", ph).replace("host_steps", 1).hash()
        ser = evaluate(s.replace("host_step_us", 3000.0))
        asy = evaluate(s.replace("host_step_us", 3000.0).replace("host_overlap", "async"))
        assert abs(ser.step - (r0.step + 3e-3)) < 1e-12 and asy.step == max(r0.step, 3e-3)
        assert abs(ser.throughput * ser.step - r0.throughput * r0.step) < 1e-6 * r0.throughput * r0.step
        big = evaluate(s.replace("host_step_us", 1e6).replace("host_overlap", "async"))
        assert big.host["host_bound"] and abs(big.step - 1.0) < 1e-12
    d = Scenario().replace("serving.phase", "decode")
    m8 = evaluate(d.replace("host_step_us", 8000.0).replace("host_steps", 8))
    assert abs(m8.step - (evaluate(d).step + 1e-3)) < 1e-12     # multi-step amortisation (decode only)
    p8 = evaluate(Scenario().replace("serving.phase", "prefill").replace("host_step_us", 8000.0).replace("host_steps", 8))
    assert p8.host["steps"] == 1 and abs(p8.host["s"] - 8e-3) < 1e-12
    b = d.replace("serving.batch", 64)
    sq = evaluate(b.replace("host_seq_us", 10.0))
    assert abs(sq.host["s"] - 64 * 10e-6 / b.layout.dp) < 1e-12


def test_host_validation_and_api():
    for kw in (("host_step_us", -1.0), ("host_overlap", "x"), ("host_steps", 0), ("host_steps", 1.5)):
        try:
            Scenario().replace(*kw)
            raise AssertionError(kw)
        except ValueError:
            pass
    r = A.api_eval({"scenario": {"host_step_us": 2000, "serving": {"phase": "decode"}}})
    assert r is not None and "host" in str(r)
