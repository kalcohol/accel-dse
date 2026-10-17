"""0.217: energy invariant J/token × tok/s/card = W/card (main eval + every what-if row); unreachable levers costed
at their batch limit, not the configured batch."""
import random

from accel_dse import api
from accel_dse.api import WHATIF_ENERGY
from accel_dse.core.energy import ENERGY_PRESETS, EnergyTable, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario

MODELS = ["qwen3-8b", "llama-3.1-8b", "llama-3.2-1b", "qwen3-30b-a3b", "rwkv7-g1k-7.2b", "llama-3.3-70b"]
TOL = 1e-6


def _models():
    return MODELS


def _table():
    f = ENERGY_PRESETS[WHATIF_ENERGY]["fields"]
    return EnergyTable.from_dict({k: v for k, (v, _, _) in f.items() if k in EnergyTable.__dataclass_fields__})


def test_main_eval_invariant_fuzzed():
    rnd, t, n = random.Random(217), _table(), 0
    chips = sorted(CHIPS)
    for _ in range(24):
        m, c, b = rnd.choice(_models()), rnd.choice(chips), rnd.choice([1, 4, 16, 64])
        try:
            r = evaluate(Scenario(chip=CHIPS[c], model=m).replace("serving.batch", b))
        except Exception:
            continue
        if not r.tpot or r.scenario.serving.phase != "decode":
            continue
        e = energy_report(r, t)
        if not e.get("J_per_unit"):
            continue
        tok_s_card = b / r.tpot / r.scenario.layout.cards
        assert abs(e["J_per_unit"] * tok_s_card / e["avg_W_per_card"] - 1) < 1e-3, (m, c, b)
        n += 1
    assert n >= 8, n


def test_whatif_rows_invariant_and_cost_batch():
    for m in ("qwen3-8b", "rwkv7-g1k-7.2b", "llama-3.1-8b"):
        for fo in ([], [["kv", "fp8"]]):
            try:
                w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": m, "formats_override": fo}})
            except Exception:
                continue
            w = w["sensitivity"]["balance_whatif"]
            for x in w["rows"]:
                if x.get("J_per_tok") is None:
                    continue
                assert abs(x["J_per_tok"] * x["tok_s_card_at_cost_batch"] / x["W_card"] - 1) < 1e-3, (m, x["key"])
                if not x["reachable"]:
                    assert x["cost_basis"] == "batch_limit" and x["cost_batch"] == x["batch_limit"]
                    assert x["base_cost_batch"] <= w["base"]["batch_limit"]


def test_qwen_jtok_order_of_magnitude():
    w = api.api_sensitivity({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "formats_override": [["kv", "fp8"]]}})
    rows = {x["key"]: x for x in w["sensitivity"]["balance_whatif"]["rows"]}
    x = rows["hbm_12h24g"]
    assert x["cost_batch"] == 453 and 5e-3 < x["J_per_tok"] < 3e-2     # ~14.7 mJ, not 0.48 J (batch 1)
