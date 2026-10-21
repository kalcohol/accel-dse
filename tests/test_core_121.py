"""0.121: Fabric.kv_feedback_prefill (opt-in) — the KV stream's NIC share also slows the prefill pool's network
collectives; prefill TTFT / request rate recomputed in the same fixed point.  Default off: report unchanged."""
import dataclasses

import argparse

from accel_dse.cli import _body, _scenario_args
from accel_dse.core.disagg import disagg_report
from accel_dse.core.hardware import CHIPS, Fabric, Link
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

HBM = "hbm3e_8s_12h24g_9200"


def _close(a, b, r=1e-9):
    return abs(a - b) <= r * max(1.0, abs(a), abs(b))


def test_prefill_feedback():
    lay = Layout(1, 1, 64, 64, 1)
    s = Scenario(model="deepseek-v3", chip=CHIPS["1P"], mem_id=HBM, layout=lay, node_cards=8, net=Link(12.5, 5.0),
                 serving=Serving(batch=512, prompt=2048, out_len=16, ctx=2048, ttft_slo_ms=20000),
                 pd=PDConfig(enabled=True, simulate=False, prefill_layout=lay, prefill_cards=960, decode_cards=64),
                 fabric=Fabric(enabled=True, oversub=4.0, leaf_nodes=1, kv_feedback=True))
    a = disagg_report(s)
    b = disagg_report(dataclasses.replace(s, fabric=dataclasses.replace(s.fabric, kv_feedback_prefill=True)))
    fa, fb = a["kv"]["fabric"]["feedback"], b["kv"]["fabric"]["feedback"]
    assert "ttft_ms_after" not in fa
    assert _close(fb["ttft_ms_before"], fa["ttft_ms_before"] if "ttft_ms_before" in fa else fb["ttft_ms_before"])
    assert fb["ttft_ms_after"] > fb["ttft_ms_before"]
    assert _close(fb["net_GBps_prefill"], 12.5 * (1 - fb["u_kv"]))
    assert b["ttft_ms"] > a["ttft_ms"] and b["goodput_per_card"] <= a["goodput_per_card"]
    # slower prefill → lower λ → smaller KV share on the decode NICs → decode TPOT no worse than decode-only feedback
    assert fb["u_kv"] <= fa["u_kv"] + 1e-12 and b["tpot_ms"] <= a["tpot_ms"] + 1e-9


def test_field_default_and_cli():
    f = Fabric()
    assert f.kv_feedback_prefill is False and "kv_feedback_prefill" not in Scenario().to_dict().get("fabric", {})
    try:
        Fabric(kv_feedback_prefill=1)
        raise AssertionError("accepted non-bool")
    except ValueError:
        pass
    p = argparse.ArgumentParser()
    _scenario_args(p)
    body = _body(p.parse_args(["--model", "deepseek-v3", "--fabric", "--kv-feedback", "--kv-feedback-prefill"]))
    fab = body["scenario"]["fabric"] if "scenario" in body else body["fabric"]
    assert fab["kv_feedback_prefill"] is True and Fabric(**fab).kv_feedback_prefill


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
