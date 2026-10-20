"""0.91 standing tests: slc_policy pin_bw (inclusive SLC as a second read path, bandwidth-balanced hit ratio; opt-in),
external validation set r4 (TensorRT-LLM 1.2.0), routed-expert format override, instance_sched diagnostic variant."""
import math
from dataclasses import replace
from pathlib import Path

from accel_dse.core import extval as X
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import SLC_POLICIES
from accel_dse.core.mapping import gemm_cost
from accel_dse.core.refhw import REF_HW
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

H = "hbm3e_8s_12h24g_9200"
ROOT = Path(__file__).resolve().parents[1]


def _ev(mib, pol, gb=2000.0, model="esm2-650m", ov="serial"):
    return evaluate(Scenario.from_dict({"model": model, "mem_id": H, "layout": {"tp": 2}, "exec_overlap": ov,
                                        "chip": {"slc_mib": mib, "slc_policy": pol, "slc_GBps": gb}}))


def test_pin_bw_never_worse_and_monotone():
    assert SLC_POLICIES == ("pin", "lru", "pin_bw", "dip")        # 0.258: + dip
    for gb in (500.0, 2000.0, 6595.0, 20000.0):
        none = _ev(0, "pin", gb).step
        prev = none
        for mib in (16, 64, 512, 4096):
            p, b = _ev(mib, "pin", gb).step, _ev(mib, "pin_bw", gb).step
            assert b <= min(p, none) * (1 + 1e-12)
            assert b <= prev * (1 + 1e-12)          # more SLC never hurts
            prev = b


def test_pin_bw_balances_paths_and_reduces_to_pin():
    r = _ev(4096, "pin_bw", 2000.0)
    t, d = r.stages[0].time, r.stages[0].dram
    assert abs(t.t_dram - t.t_slc) / t.t_dram < 1e-9                 # both paths finish together
    sysbw = (d["total"] + d["slc"]) / t.t_dram                        # = B_dram + B_slc
    a = _ev(4096, "pin", 2000.0)
    assert a.stages[0].time.t_slc > a.stages[0].time.t_dram           # pin overloads the slower SLC
    assert sysbw > 0
    # small, fast SLC: the balance point is beyond the resident set → identical to pin
    for mib in (16, 64):
        assert _ev(mib, "pin_bw", 20000.0).step == _ev(mib, "pin", 20000.0).step


def test_pin_bw_wiring():
    html = W.read(W.INDEX_HTML)                        # 0.286: option of the SLC-policy select; token match
    assert "pin_bw" in W.select_options(html, "c-slc_policy")
    js = W.read()
    assert W.has(js, "slc_policy === 'pin_bw'")
    cli = (ROOT / "accel_dse/cli.py").read_text(encoding="utf-8")
    assert '"pin_bw"' in cli
    w = _ev(512, "pin", 2000.0).warnings
    assert any("pin_bw" in x for x in w)


def test_r4_rows_cited():
    R = X.rows(X.DATA_R4)
    assert len(R) == 38 and len({r["id"] for r in R}) == 38
    assert all(r["url"] == "https://nvidia.github.io/TensorRT-LLM/1.2.0/developer-guide/perf-overview.html" for r in R)
    assert all(r["hw"] in REF_HW and r["metric"] == "max_tok_s_gpu" and r["value"] > 0 for r in R)
    dep = [r for r in R if "-dep" in r["id"]]
    assert dep and all(r["tp"] == 1 and r["dp"] == r["ep"] > 1 for r in dep)


def test_r4_band_and_per_gpu_metric():
    R = X.rows(X.DATA_R4)
    lr = [math.log(X.predict(r, "catalog_kernel")["pred"] / r["value"]) for r in R]
    g = math.exp(sum(lr) / len(lr))
    assert 0.9 < g < 1.1 and 100 * sum(map(abs, lr)) / len(lr) < 20       # MODEL §37: 0.83 / 33 %; 0.172: 1.016 / 16.8 %
    r = next(x for x in R if x["id"] == "trt120-b200-qwen3-235b-a22b-nvfp4-dep4-1024-1024")
    tot = X.predict({**r, "metric": "max_tok_s_total"}, "catalog_kernel")["pred"]
    assert abs(X.predict(r, "catalog_kernel")["pred"] * 4 - tot) < 1e-6 * tot


def test_expert_format_override():
    r = next(x for x in X.rows(X.DATA_R4) if x["model"] == "qwen3-30b-a3b")
    s = X.scenario(r, "catalog")
    assert ("expert", "nvfp4") in s.formats_override and ("expert", "fp8") in X.FP8
    assert X.MXFP4 == (("kv", "fp8"),)


def test_sched_variant_diagnostic_only():
    assert all(h.chip.instance_sched == "wide" for h in REF_HW.values())       # reference default unchanged
    c = REF_HW["B200"].chip
    m, k, n = 131072, 4096, 2048
    u = lambda ch: m * k * n / ch.macs / gemm_cost(ch, "reconf", m, k, n).cycles
    assert u(c) < 0.2 and u(replace(c, instance_sched="auto")) > 0.95
    v = X.sched_variant("catalog_kernel")
    assert v == "catalog_kernel+A" and X.scenario(X.rows(X.DATA_R4)[0], v).chip.instance_sched == "auto"


from tests.pin_legacy import pin_module  # noqa: E402  (0.173: study bands at the 0.172 mapping; current numbers in data/ext_validation.json)
pin_module(globals())
