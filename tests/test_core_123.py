"""0.123: RadixLRU with a path shorter than the tree (multi-turn prompts) stores each node at its own size; the
multi-turn closed forms vs the corrected simulator; turn history in the PD DES; pd.prefix_turn_prompt."""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_122 import _model, _sim  # noqa: E402

from accel_dse.cli import _body, _scenario_args  # noqa: E402
from accel_dse.core import pdsim as ps  # noqa: E402
from accel_dse.core.disagg import disagg_report  # noqa: E402
from accel_dse.core.scenario import PDConfig  # noqa: E402
from accel_dse.core.validation import v4_scenario  # noqa: E402


def test_radix_short_path_sizes():
    c = ps.RadixLRU(10_000, [100, 512, 64])
    assert c.access((1, 2)) == 0 and c.used == 612            # was 100 + 64: the leaf took the deepest level's size
    assert c.access((1, 2, 3)) == 2 and c.used == 676
    assert c.access((1,)) == 1


def test_closed_forms_vs_sim():
    lv = [(1024, 8, 1.0), (512, 100, 0.8)]
    for wb in (False, True):
        m, s = _model(lv, 6, 64, 512, wb, 70308), _sim(lv, 6, 64, 512, wb, 70308, 60000)
        assert abs(m / s - 1) <= 0.03, ("irm", wb, m, s)
        m, s = _model(lv, 6, 64, 512, wb, 300000, 256), _sim(lv, 6, 64, 512, wb, 300000, 80000, sessions=256)
        assert abs(m / s - 1) <= 0.03, ("sessions", wb, m, s)


def _scn(gb, ses, auto=True):
    s = v4_scenario(0.6, 0.0, False, "dense8b")
    pd = dataclasses.replace(s.pd, prefix_tree=((512, 1, 0.0), (1024, 200, 1.0)), prefix_cache_GB=gb, prefix_turns=4,
                             prefix_turn_user=128, prefix_writeback=True, prefix_sessions=ses, prefix_turn_prompt=auto)
    return dataclasses.replace(s, pd=pd)


def test_des_turns_match_closed_form():
    c = ps.compare(_scn(1.0, 256), n_req=1500, warmup=400)
    for mode, row in c["modes"].items():
        assert abs(row["sim"]["prefix_hit"] - row["ana"]["prefix_hit"]) < 0.03, (mode, row["sim"]["prefix_hit"],
                                                                                 row["ana"]["prefix_hit"])


def test_turn_prompt_and_validation():
    r = disagg_report(_scn(4.0, 64), queue=False)
    t = r["prefix_cache"]["turns"]
    assert t["auto_prompt"] and t["prompt_given"] == 4096 and t["mean_turn_prompt"] == 512 + 1024 + 1.5 * (512 + 128)
    assert r["scenario"]["serving"]["prompt"] == 2496 if "scenario" in r else True
    assert not any("各轮平均" in w for w in r["warnings"]) and not any("DES" in w and "轮次" in w for w in r["warnings"])
    try:
        PDConfig(prefix_tree=((1, 1, 0.0),), prefix_turn_prompt=True)
        raise AssertionError("turn_prompt without turns")
    except ValueError:
        pass
    p = argparse.ArgumentParser()
    _scenario_args(p)
    b = _body(p.parse_args(["--model", "qwen3-8b", "--pd", "--pd-prefix-tree", "512:1:0", "--pd-turns", "3",
                            "--pd-turn-prompt"]))
    assert (b.get("scenario") or b)["pd"]["prefix_turn_prompt"] is True


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
