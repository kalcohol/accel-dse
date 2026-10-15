"""0.122: multi-turn conversations on the radix prefix tree (pd.prefix_turns / prefix_turn_user / prefix_writeback /
prefix_sessions, opt-in) — closed form vs a RadixLRU simulation (IRM and sequential sessions); CHANGELOG headers."""
import argparse
import dataclasses
import random
import re
from pathlib import Path

from accel_dse import __version__
from accel_dse.cli import _body, _scenario_args
from accel_dse.core import prefixcache as P
from accel_dse.core.disagg import disagg_report
from accel_dse.core.pdsim import RadixLRU, _zipf_ids
from accel_dse.core.scenario import PDConfig
from accel_dse.core.validation import v4_scenario


def _ins(m0, kinds, t, wb):
    d = m0
    for j, (k, i) in enumerate(kinds):
        if (k == "s" and i <= t) or (k == "o" and (i <= t - 1 or (wb and i == t))):
            d = m0 + j + 1
        else:
            break
    return d


def _sim(levels, T, u, o, wb, C, n, sessions=0, seed=1):
    ext, _, qd = P.turn_tree(levels, T, u, o, wb)
    m0 = len(levels)
    kinds = [x for t in range(1, T) for x in ((("o", t),) if o else ()) + ((("s", t + 1),) if u else ())]
    rng = random.Random(seed)
    cache, cum, ctr = RadixLRU(C, [L for L, _, _ in ext]), P.tree_cum_tokens(ext), [0]

    def newpath():
        ctr[0] += 1
        hid = ctr[0] if sessions else 0          # IRM: the leaf path is the conversation
        return tuple(_zipf_ids(rng, nn, a, 1)[0] for _, nn, a in levels) + (hid,) * (len(ext) - m0)
    act = [[newpath(), 1] for _ in range(max(1, sessions))]
    tot = cnt = 0
    for i in range(n):
        if sessions:
            a = rng.randrange(sessions)
            path, t = act[a]
            act[a] = [newpath(), 1] if t == T else [path, t + 1]
        else:
            path, t = newpath(), rng.randint(1, T)
        d = min(cache.access(path[:_ins(m0, kinds, t, wb)]), qd[t - 1])
        if i >= n // 4:
            tot, cnt = tot + cum[d], cnt + 1
    return tot / cnt


def _model(levels, T, u, o, wb, C, sessions=0):
    ext, w, qd = P.turn_tree(levels, T, u, o, wb)
    if sessions:
        tail, _ = P.turn_session_tail(levels, T, u, o, wb, C, sessions)
    else:
        tail = P.turn_depth_tail(ext, P.turn_resident(ext, C, w), qd)
    return sum(q * c for q, c in zip(P.tree_depth_law(tail), P.tree_cum_tokens(ext)))


def test_turn_tree_shape():
    ext, w, qd = P.turn_tree([(512, 1, 0.0), (1024, 50, 1.0)], 4, 128, 256, False)
    assert [L for L, _, _ in ext] == [512, 1024, 256, 128, 256, 128, 256, 128]
    assert w[2:] == [3 / 4, 3 / 4, 2 / 4, 2 / 4, 1 / 4, 1 / 4]
    assert qd == [2, 2, 4, 6]                      # turn t matches turn t−1's prompt
    _, w2, qd2 = P.turn_tree([(512, 1, 0.0), (1024, 50, 1.0)], 4, 128, 256, True)
    assert qd2 == [2, 3, 5, 7] and w2[2] == 1.0    # + the previous output with write-back


def test_irm_vs_radix_sim():
    lv = [(512, 1, 0.0), (1024, 50, 1.0)]
    for wb in (False, True):
        m, s = _model(lv, 4, 128, 256, wb, 31754), _sim(lv, 4, 128, 256, wb, 31754, 60000)
        assert -0.08 <= m / s - 1 <= 0.02, (wb, m, s)


def test_sessions_vs_radix_sim_and_order():
    lv = [(2048, 1, 0.0), (256, 1000, 1.0)]
    prev = None
    for C in (20000, 80000):
        res = {}
        for wb in (False, True):
            m, s = _model(lv, 8, 100, 400, wb, C, 16), _sim(lv, 8, 100, 400, wb, C, 60000, sessions=16)
            assert -0.15 <= m / s - 1 <= 0.03, (C, wb, m, s)
            res[wb] = m
        assert res[True] >= res[False] - 1e-9
        assert prev is None or res[False] >= prev
        prev = res[False]
    # IRM over turns (no temporal locality) is the conservative bound of the session model
    assert _model(lv, 8, 100, 400, True, 20000, 16) >= _model(lv, 8, 100, 400, True, 20000)


def test_disagg_wiring_and_default():
    s = v4_scenario(0.6, 0.0, False, "dense8b")
    s = dataclasses.replace(s, serving=dataclasses.replace(s.serving, prompt=2496))
    pd0 = dataclasses.replace(s.pd, prefix_tree=((512, 1, 0.0), (1024, 200, 1.0)), prefix_cache_GB=2.0)
    a = disagg_report(dataclasses.replace(s, pd=pd0), queue=False)
    assert "turns" not in a["prefix_cache"]
    b = disagg_report(dataclasses.replace(s, pd=dataclasses.replace(pd0, prefix_turns=4, prefix_turn_user=128,
                                                                    prefix_writeback=True, prefix_sessions=64)), queue=False)
    t = b["prefix_cache"]["turns"]
    assert t["mean_turn_prompt"] == 512 + 1024 + 1.5 * (512 + 128) == 2496 and 0 < t["session_survival"] < 1
    assert not any("多轮前缀：各轮平均" in w for w in b["warnings"])
    assert PDConfig().to_dict() == PDConfig(prefix_turns=1).to_dict() if hasattr(PDConfig, "to_dict") else True
    for bad in (dict(prefix_turns=2), dict(prefix_turns=0, prefix_tree=((1, 1, 0.0),)),
                dict(prefix_writeback=1, prefix_tree=((1, 1, 0.0),))):
        try:
            PDConfig(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_cli_flags():
    p = argparse.ArgumentParser()
    _scenario_args(p)
    body = _body(p.parse_args(["--model", "qwen3-8b", "--pd", "--pd-prefix-tree", "512:1:0,1024:50:1", "--pd-turns", "4",
                               "--pd-turn-user", "64", "--pd-writeback", "--pd-sessions", "32"]))
    pd = (body.get("scenario") or body)["pd"]
    assert (pd["prefix_turns"], pd["prefix_turn_user"], pd["prefix_writeback"], pd["prefix_sessions"]) == (4, 64, True, 32)


def test_changelog_has_every_release_header():
    s = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    heads = set(re.findall(r"^## \[0\.(\d+)\.0\]", s, flags=re.M))
    cur = int(__version__.split(".")[1])
    assert all(str(v) in heads for v in range(100, cur + 1)), sorted(set(map(str, range(100, cur + 1))) - heads)
