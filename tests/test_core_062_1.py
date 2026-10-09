"""0.62.1 — round-6 convergence audit: doc examples re-derived from the code (MODEL.md §2 / §9 / §12 / §19 / §19.5),
neutral-value identities of the optional features, eval / sweep agreement with images."""

from __future__ import annotations

import glob
import json
import os

from accel_dse import api
from accel_dse.core.catalog import get_model, list_models, unlisted_models
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Fabric
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving

HBM = "hbm3e_8s_12h24g_9200"
LP = "lpddr5x_4x64_8533_16g"


def test_release_count_and_param_error():
    """MODEL §2 / §9 V1: 72 releases (55 LLM / VLM + 17 video / protein), all within ±0.5 %."""
    files = glob.glob(os.path.join(os.path.dirname(__file__), "..", "accel_dse", "data", "releases", "*.json"))
    doms = [json.load(open(f)).get("domain", "llm") for f in files]
    assert len(files) == 72 and doms.count("llm") == 55
    listed = {x["id"]: x for x in list_models()}
    assert len(listed) + len(unlisted_models()) == 72
    assert all(abs(x["param_err"]) <= 0.005 for x in listed.values())
    assert all(abs(x["param_err"]) < 1e-5 for x in listed.values() if get_model(x["id"]).domain in ("gen", "protein"))


def test_esmfold_fp32_activation_example():
    """MODEL §12: ESMFold 512 residues, 100T + LPDDR5X, 5.0 → 10.0 s (4 trunk passes since 0.61.2)."""
    s = Scenario(chip=CHIPS["100T"], model="esmfold", mem_id=LP)
    assert round(evaluate(s).latency, 1) == 5.0
    assert round(evaluate(s.replace("formats_override", (("act", "fp32"),))).latency, 1) == 10.0


def test_fabric_examples_after_bf16_combine():
    """MODEL §19 example A and §19.5 (re-derived: combine / all-reduce wire bytes ≥ bf16 since 0.61.1)."""
    def run(model, lay, phase, batch, fab=Fabric()):
        return evaluate(Scenario(model=model, chip=CHIPS["1P"], mem_id=HBM, layout=lay, node_cards=8,
                                 serving=Serving(phase=phase, batch=batch), fabric=fab))
    r = run("deepseek-v3", Layout(dp=64, ep=64), "prefill", 64)
    assert round(r.step * 1e3, 1) == 715.9 and r.bound == "LINK"
    r = run("deepseek-v3", Layout(dp=64, ep=64), "prefill", 64, Fabric(enabled=True, oversub=4.0, leaf_nodes=2))
    assert round(r.step * 1e3) == 2453
    r = run("kimi-k2", Layout(dp=1024, ep=1024), "decode", 8192, Fabric(enabled=True, oversub=4.0))
    # 0.63: per-head MLA absorption (W_UK / W_UV) makes decode compute 7.26 ms > link 6.30 ms (0.62.1: 7.22, LINK)
    assert round(r.step * 1e3, 2) == 7.26 and r.bound == "MAC"


def test_neutral_values_are_identities():
    """images = 0 ignores the image size; slc_mib = 0 ignores slc_GBps / policy; fabric off ignores its knobs;
    spec_k = 0 ignores spec_accept; pp_split is inert at PP = 1; package_cards is inert with D2D off."""
    s = Scenario(model="qwen3.8-27b", mem_id=HBM, layout=Layout(tp=2), serving=Serving(batch=8))

    def sig(x):
        d = evaluate(x).summary()
        d.pop("warnings")
        return json.dumps(d, sort_keys=True, default=str)
    base = sig(s)
    assert sig(s.replace("serving.image_w", 333).replace("serving.image_h", 777)) == base
    assert sig(s.replace("chip.slc_GBps", 999.0).replace("chip.slc_policy", "lru")) == base
    assert sig(s.replace("fabric", Fabric(enabled=False, algo="tree", overlap="ports", net_tiers=3, oversub=4.0))) == base
    assert sig(s.replace("serving.spec_accept", 0.3)) == base
    assert sig(s.replace("pp_split", "layers")) == base
    assert sig(s.replace("package_cards", 4)) == base


def test_sweep_images_matches_eval():
    sc = {"model": "kimi-k2.5", "mem_id": HBM, "layout": {"tp": 4, "dp": 2, "ep": 8},
          "serving": {"phase": "prefill", "batch": 2, "image_w": 1920, "image_h": 1080}}
    sw = api.api_sweep({"scenario": sc, "path": "serving.images", "values": [0, 1, 3]})
    s = api.scenario_from_body({"scenario": sc})
    for row, v in zip(sw["rows"], (0, 1, 3)):
        r = evaluate(s.replace("serving.images", v))
        assert row["ttft_ms"] == r.ttft * 1e3 and row["fits"] == r.fits
    assert sw["rows"][0]["ttft_ms"] < sw["rows"][1]["ttft_ms"] < sw["rows"][2]["ttft_ms"]
