"""0.124: the validators against brute-force references (RadixLRU, pipe flow shop, DES decode / prefill servers),
extval scoring robustness + stored table freshness, Fabric.channels (opt-in)."""
import dataclasses
import json
import math
import random
from collections import deque
from pathlib import Path

from accel_dse import __version__
from accel_dse.core import extval as X
from accel_dse.core import overlap as ov
from accel_dse.core import pdsim as ps
from accel_dse.core.hardware import CHIPS, Fabric, Link
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario


def _ref_radix(cap, lens, seq):
    order, size, out = [], {}, []
    for path in seq:
        m = min(len(lens), len(path))
        d = 0
        for k in range(1, m + 1):
            if path[:k] not in size:
                break
            d = k
        out.append(d)
        if cap <= 0:
            continue
        acc, mf = 0, m
        for k in range(m):
            acc += lens[k]
            if acc > cap:
                mf = k
                break
        for k in range(mf, 0, -1):
            key = path[:k]
            if key in size:
                order.remove(key)
            else:
                size[key] = lens[k - 1]
            order.append(key)
        while order and sum(size[x] for x in order) > cap:
            del size[order.pop(0)]
    return out, sorted(order)


def test_radix_vs_bruteforce():
    rng = random.Random(124)
    for _ in range(600):
        nl = rng.randint(1, 4)
        lens = [rng.randint(1, 40) for _ in range(nl)]
        cap = rng.choice([0, rng.randint(1, 50), rng.randint(20, 300)])
        br = [rng.randint(1, 3) for _ in range(nl)]
        seq = [tuple(rng.randrange(br[k]) for k in range(rng.randint(1, nl))) for _ in range(rng.randint(1, 40))]
        c = ps.RadixLRU(cap, lens)
        got = [c.access(p) for p in seq]
        exp, st = _ref_radix(cap, lens, seq)
        assert got == exp and sorted(c.od) == st and c.used == sum(c.od.values())


def _ref_ms(tiles, buf, cross, starts):
    buf = math.floor(buf)
    D, C, S, cp = 0.0, [], [], 0.0
    for i, (sz, m, c) in enumerate(tiles):
        start = D
        if sz > 0:
            sz = math.ceil(sz)
            tot, h = sz, i
            while h > 0 and tot + S[h - 1] <= buf:
                tot += S[h - 1]
                h -= 1
            if h > 0:
                start = max(start, C[h - 1])
            if i in starts and not cross:
                start = max(start, cp)
        S.append(sz)
        D = start + m
        cp = (max(cp, D) if sz > 0 else cp) + c
        C.append(cp)
    return cp


def test_pipe_vs_flow_shop_recurrence():
    rng = random.Random(7)
    for _ in range(800):
        n = rng.randint(1, 25)
        tiles = [(rng.choice([0, rng.uniform(0.1, 12)]), rng.uniform(0, 2), rng.uniform(0, 2)) for _ in range(n)]
        buf, cross = rng.choice([1, 3, 10, 50, 1e9]), rng.random() < 0.5
        starts = set(rng.sample(range(n), rng.randint(0, n)))
        a, b = ov.makespan(tiles, buf, cross, starts), _ref_ms(tiles, buf, cross, starts)
        assert abs(a - b) <= 1e-9 * max(1.0, b)


class _C:
    spec = None

    def decode_step(self, k):
        return (0.1 + 0.01 * k, 1.0)

    def batch_prefill_wall(self, b):
        return 0.5 + 0.2 * len(b)


def test_decode_replica_vs_reference():
    rng = random.Random(3)
    for _ in range(200):
        n, t, arr = rng.randint(1, 30), 0.0, []
        for _ in range(n):
            t += rng.expovariate(rng.choice([1, 5, 20]))
            arr.append(t)
        outs = [rng.randint(1, 20) for _ in range(n)]
        slots = rng.randint(1, 6)
        eng = ps.Engine(random.Random(1))
        rep = ps.DecodeReplica(eng, _C(), slots)
        fin = {}
        rep.on_finish = lambda r: fin.__setitem__(r.rid, eng.t)
        for i, a in enumerate(arr):
            eng.schedule(a, "arr", ps.Req(i, a, 10, outs[i]))
        while (ev := eng.pop()) is not None:
            if ev.kind == "arr":
                rep.offer_decode(ev.payload)
            else:
                ev.payload[0].dec_done(ev.payload[1])
        t, i, q, run, left, ref = 0.0, 0, deque(), [], {}, {}
        while len(ref) < n:
            while i < n and arr[i] <= t:
                q.append(i)
                i += 1
            while q and len(run) < slots:
                r = q.popleft()
                run.append(r)
                left[r] = outs[r]
            if run:
                t += 0.1 + 0.01 * len(run)
                for r in list(run):
                    left[r] -= 1
                    if left[r] <= 0:
                        run.remove(r)
                        ref[r] = t
            elif i < n:
                t = arr[i]
        assert fin.keys() == ref.keys() and all(abs(fin[k] - ref[k]) < 1e-9 for k in fin)


def test_extval_summary_and_table_fresh():
    s = X.summary([{"metric": "m", "ratio": 0.0}, {"metric": "m", "ratio": 2.0}])
    assert s["m"]["n"] == 1 and s["m"]["n_invalid"] == 1 and abs(s["m"]["gmean_ratio"] - 2.0) < 1e-12
    d = json.loads((Path(X.DATA).with_name("ext_validation.json")).read_text())
    cur = {x["id"]: x for x in X.table("catalog")}
    assert all(abs(cur[x["id"]]["ratio"] - x["ratio"]) <= 1e-4 * max(1.0, x["ratio"]) for x in d["rows"]["catalog"]), \
        "data/ext_validation.json is stale: re-run scripts/ext_validate.py"
    assert d["generated"] != "0.66.0"


def test_fabric_channels():
    lay = Layout(1, 8, 1, 1, 1)
    base = Scenario(model="qwen3-8b", chip=CHIPS["1P"], layout=lay, link=Link(900.0, 3.0), fabric=Fabric(enabled=True))
    t0 = evaluate(base).stages[0].time.t_link
    wide = dataclasses.replace(base, fabric=dataclasses.replace(base.fabric, channels=32, channel_GBps=100.0))
    narrow = dataclasses.replace(base, fabric=dataclasses.replace(base.fabric, channels=4, channel_GBps=50.0))
    assert evaluate(wide).stages[0].time.t_link == t0                     # 3200 GB/s ≥ the link: no cap
    assert evaluate(narrow).stages[0].time.t_link > t0                    # 200 GB/s < 900
    assert "channels" not in Scenario().to_dict().get("fabric", {})
    for bad in (dict(channels=4), dict(channel_GBps=10.0), dict(channels=-1, channel_GBps=1.0)):
        try:
            Fabric(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert __version__
