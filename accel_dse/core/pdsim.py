"""Request-level discrete-event simulator for PD / colocated serving (0.54, 「假设」).

Reuses the engine's evaluated step costs via ``pdqueue._Pool`` (prefill TTFT(b, S, p), decode step(k),
fused chunked iteration).  Not a cycle-accurate runtime: arrivals are Poisson, scheduling is FCFS,
prefill uses a static batch cap, decode is continuous-batching, colocated is prefill-first or chunked.
Prefix cache is an exact whole-prefix LRU over a Zipf working set (same IRM as core/prefixcache.py).

Used to quantify the closed-form (pdqueue) error (V4 serving) and, opt-in, as ``pd.simulate`` / ``--pd-sim``.
"""

from __future__ import annotations

import heapq
import bisect
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .pdqueue import B_CAPS, _Pool, _chunk_plan, _fused_step

# --------------------------------------------------------------------------- helpers


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, math.ceil(q * len(ys)) - 1))
    return ys[i]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    return {"mean": sum(xs) / len(xs), "p50": _pct(xs, 0.5), "p90": _pct(xs, 0.9), "p99": _pct(xs, 0.99)}


@dataclass(eq=False)          # identity: requests are compared / removed by object
class Req:
    rid: int
    arrive: float
    S: int
    out: int
    p: int = 0
    prefix_id: int = -1
    hit: bool = False
    # filled by the simulation
    prefill_start: float = 0.0
    prefill_done: float = 0.0
    kv_done: float = 0.0
    decode_start: float = 0.0
    decode_done: float = 0.0
    tokens_left: int = 0
    itl_max: float = 0.0
    token_sum: float = 0.0
    tokens_done: int = 0
    replica: int = 0
    last_tok: float = 0.0
    dec: int = 0
    dec_holds: bool = False
    preempts: int = 0
    ready_t: float = 0.0     # 0.56: started waiting for KV admission
    admit_t: float = 0.0     # 0.56: admitted (slot + KV)


# --------------------------------------------------------------------------- length / prefix sampling


class LRUCache:
    """Whole-prefix LRU; returns whether the looked-up id was resident (and records the access)."""

    def __init__(self, K: int):
        self.K = max(0, int(K))
        self.od: OrderedDict = OrderedDict()
        self.hits = self.looks = 0

    def access(self, pid: int) -> bool:
        self.looks += 1
        if self.K <= 0:
            return False
        if pid in self.od:
            self.od.move_to_end(pid)
            self.hits += 1
            return True
        self.od[pid] = 1
        if len(self.od) > self.K:
            self.od.popitem(last=False)
        return False

    @property
    def hit_rate(self) -> float:
        return self.hits / self.looks if self.looks else 0.0


def _zipf_ids(rng: random.Random, n: int, alpha: float, count: int) -> list[int]:
    w = [i ** -alpha for i in range(1, n + 1)]
    return rng.choices(range(n), weights=w, k=count)


# --------------------------------------------------------------------------- cost table


class Costs:
    """Memoised engine costs for one pool."""

    def __init__(self, pool: _Pool):
        self.pool = pool
        self._pre: dict = {}
        self._dec: dict = {}

    def prefill(self, b: int, S: int, p: int) -> float:
        k = (b, S, p)
        if k not in self._pre:
            self._pre[k] = self.pool.run("prefill", b, S, p).ttft
        return self._pre[k]

    def decode_step(self, k: int) -> tuple[float, float]:
        """(step time, tokens accepted per sequence per step)."""
        k = max(1, k)
        if k not in self._dec:
            r = self.pool.run("decode", k)
            self._dec[k] = (r.step, r.tokens_per_step)
        return self._dec[k]

    def batch_prefill_wall(self, batch: list[Req]) -> float:
        """Wall time of a static prefill batch (all leave together).  Mean of TTFT(n, S_i, p_i) —
        matches the M/G/1 mixture mean used by the closed form."""
        n = len(batch)
        return sum(self.prefill(n, r.S, r.p) for r in batch) / n


# --------------------------------------------------------------------------- event engine bits


@dataclass(order=True)
class Ev:
    t: float
    seq: int
    kind: str = field(compare=False)
    payload: object = field(compare=False, default=None)


class Engine:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.t = 0.0
        self._seq = 0
        self.hq: list[Ev] = []

    def schedule(self, t: float, kind: str, payload=None):
        self._seq += 1
        heapq.heappush(self.hq, Ev(t, self._seq, kind, payload))

    def pop(self) -> Ev | None:
        if not self.hq:
            return None
        ev = heapq.heappop(self.hq)
        self.t = ev.t
        return ev


# --------------------------------------------------------------------------- PD prefill replica (batch server)


class PrefillReplica:
    def __init__(self, eng: Engine, costs: Costs, cap: int, name: str):
        self.eng = eng
        self.costs = costs
        self.cap = max(1, cap)
        self.name = name
        self.q: deque[Req] = deque()
        self.busy = False
        self.on_done = None  # callback(req_list, t)

    def offer(self, r: Req):
        self.q.append(r)
        self._maybe_start()

    def _maybe_start(self):
        if self.busy or not self.q:
            return
        batch = []
        while self.q and len(batch) < self.cap:
            batch.append(self.q.popleft())
        self.busy = True
        wall = self.costs.batch_prefill_wall(batch)
        self.busy_time = getattr(self, "busy_time", 0.0) + wall
        for r in batch:
            r.prefill_start = self.eng.t
        self.eng.schedule(self.eng.t + wall, "prefill_done", (self, batch))

    def finish(self, batch: list[Req]):
        self.busy = False
        for r in batch:
            r.prefill_done = self.eng.t
        if self.on_done:
            self.on_done(batch, self.eng.t)
        self._maybe_start()


# --------------------------------------------------------------------------- KV link (single server per prefill replica)


class KVServer:
    def __init__(self, eng: Engine, beta_req: float, alpha: float):
        self.eng = eng
        self.beta = beta_req
        self.alpha = alpha
        self.q: deque[tuple[Req, float]] = deque()  # (req, bytes)
        self.busy = False
        self.on_done = None

    def offer(self, r: Req, nbytes: float):
        self.q.append((r, nbytes))
        self._maybe()

    def _maybe(self):
        if self.busy or not self.q:
            return
        r, nb = self.q.popleft()
        self.busy = True
        hook = getattr(self, "start_hook", None)
        dur = hook(self, r, nb) if hook else self.alpha + (nb / self.beta if self.beta > 0 else 0.0)
        self.eng.schedule(self.eng.t + dur, "kv_done", (self, r))

    def finish(self, r: Req):
        self.busy = False
        if not getattr(self, "start_hook", None) or r.kv_done == 0.0:
            r.kv_done = r.kv_done or self.eng.t
        if self.on_done:
            self.on_done(r, self.eng.t)
        self._maybe()


# --------------------------------------------------------------------------- serving replicas (one activity in flight)


class _Replica:
    """Shared machinery: ``running`` decodes advance by one engine step per iteration; exactly one activity
    (decode iteration / prefill batch / fused chunk iteration) is in flight; tokens are credited when it ends."""

    def __init__(self, eng: Engine, costs: Costs, slots: int):
        self.eng = eng
        self.costs = costs
        self.slots = slots
        self.running: list[Req] = []
        self.joinq: deque[Req] = deque()   # prefilled, waiting for a decode slot
        self.in_flight = False
        self.on_finish = None
        self.iters = 0
        self.busy_time = 0.0
        # 0.55 decode KV capacity (pd.kv_policy): tokens per replica (None = unbounded, the 0.54 behaviour)
        self.kv_cap: int | None = None
        self.kv_policy = "off"
        self.kv_res = 0                    # wait: reserved S + out of the running set
        self.reprefq: deque[Req] = deque()  # recompute / swap: preempted, waiting to be restored
        self.n_preempt = 0
        # 0.56: admission order — after_prefill (0.55: prefilled requests wait in joinq) | before_prefill (vLLM: slot
        # and KV are taken first; ``pending`` holds admitted requests still prefilling / pulling their KV)
        self.kv_admit = "after_prefill"
        self.pending: list[Req] = []
        self.admitq: deque[Req] = deque()  # PD before_prefill: prefilled, waiting for decode admission
        self.on_admit = None
        self.swap_Bps = 0.0                # swap: host link per replica (B/s)
        self.kv_bpt = 0.0                  # swap: KV bytes per token per replica

    @property
    def before(self) -> bool:
        return self.kv_admit == "before_prefill" and self.kv_policy != "off" and self.kv_cap is not None

    def offer_decode(self, r: Req):
        r.tokens_left = r.out
        if self.before and r in self.pending:     # PD before_prefill: admitted earlier, KV has arrived
            self.pending.remove(r)
            self._start_decoding(r)
        else:
            r.ready_t = self.eng.t
            self.joinq.append(r)
        if not self.in_flight:
            self.boundary()

    def request_admit(self, r: Req):
        """PD before_prefill: a prefilled request asks this decode replica for a slot + KV before its KV moves."""
        r.ready_t = self.eng.t
        self.admitq.append(r)
        self._admit_pd()

    def _admit_pd(self):
        while self.admitq and self._can_admit(self.admitq[0]):
            r = self.admitq.popleft()
            self._reserve(r)
            self.pending.append(r)
            if self.on_admit:
                self.on_admit(r)

    @staticmethod
    def _foot(r: Req) -> int:
        return r.S + r.tokens_done

    def _kv_used(self) -> int:
        return sum(r.S + r.tokens_done for r in self.running) + sum(r.S for r in self.pending)

    def _can_admit(self, r: Req) -> bool:
        if len(self.running) + len(self.pending) >= self.slots:
            return False
        cap = self.kv_cap if self.kv_policy != "off" else None
        if cap is None or not (self.running or self.pending):   # one sequence always runs (no deadlock)
            return True
        if self.kv_policy == "wait":
            return self.kv_res + r.S + r.out <= cap
        return not self.reprefq and self._kv_used() + r.S <= cap

    def _reserve(self, r: Req):
        if self.kv_policy == "wait" and self.kv_cap is not None:
            self.kv_res += r.S + r.out
        r.admit_t = self.eng.t

    def _start_decoding(self, r: Req):
        r.decode_start = self.eng.t
        r.last_tok = self.eng.t
        self.running.append(r)

    def _admit(self):
        self._admit_pd()
        while self.joinq and self._can_admit(self.joinq[0]):
            r = self.joinq.popleft()
            self._reserve(r)
            self._start_decoding(r)

    def _preempt(self) -> float:
        """recompute / swap: before an iteration that grows every running sequence by e tokens, evict the youngest
        until the KV fits (vLLM preemption: recompute drops its KV and re-prefills prompt + generated later; swap
        copies the KV to host memory first, which stalls the replica — returned as extra seconds 「假设」)."""
        if self.kv_policy not in ("recompute", "swap") or self.kv_cap is None:
            return 0.0
        e = max(1, int(round(self.costs.decode_step(max(1, len(self.running)))[1])))
        used = self._kv_used()
        stall = 0.0
        while len(self.running) > 1 and used + e * len(self.running) > self.kv_cap:
            v = self.running.pop()
            used -= self._foot(v)
            v.preempts += 1
            self.n_preempt += 1
            if self.kv_policy == "swap":
                stall += self._foot(v) * self.kv_bpt / self.swap_Bps if self.swap_Bps > 0 else 0.0
            self.reprefq.appendleft(v)
        return stall

    def _try_recompute(self) -> bool:
        """recompute / swap: restore the head preempted sequence (re-prefill, or swap its KV back in; stalls the
        replica) when it fits again."""
        if self.kv_policy not in ("recompute", "swap") or not self.reprefq \
                or len(self.running) + len(self.pending) >= self.slots:
            return False
        r = self.reprefq[0]
        if (self.running or self.pending) and self._kv_used() + self._foot(r) > self.kv_cap:
            return False
        self.reprefq.popleft()
        if self.kv_policy == "swap":
            dur = self._foot(r) * self.kv_bpt / self.swap_Bps if self.swap_Bps > 0 else 0.0
        else:
            S_re = max(256, int(math.ceil(self._foot(r) / 256)) * 256)   # memo granularity 256 tokens 「假设」
            dur = self.costs.prefill(1, S_re, 0)
        self._start(dur, "rep_recompute_done", r)
        return True

    def recompute_done(self, r: Req):
        self.in_flight = False
        self.running.append(r)               # last_tok untouched → its gap spans the preemption
        self.boundary()

    def _credit(self, members: list[Req], e: int):
        t = self.eng.t
        done = []
        for r in members:
            gap = (t - r.last_tok) / e
            if gap > r.itl_max:
                r.itl_max = gap
            r.last_tok = t
            r.tokens_done += e
            r.tokens_left -= e
            if r.tokens_left <= 0:
                done.append(r)
        if done:
            ds = set(id(r) for r in done)
            self.running = [r for r in self.running if id(r) not in ds]
            if self.kv_policy == "wait" and self.kv_cap is not None:
                self.kv_res -= sum(r.S + r.out for r in done)
            for r in done:
                r.decode_done = t
                if self.on_finish:
                    self.on_finish(r)

    def _start(self, dur: float, kind: str, payload):
        self.in_flight = True
        self.iters += 1
        self.busy_time += dur
        self.eng.schedule(self.eng.t + dur, kind, (self, payload))

    def _decode_iter(self):
        stall = self._preempt()
        k = len(self.running)
        step, e = self.costs.decode_step(k)
        self._start(step + stall, "rep_dec_done", (list(self.running), max(1, int(round(e)))))

    def dec_done(self, payload):
        members, e = payload
        self.in_flight = False
        self._credit(members, e)
        self.boundary()


class DecodeReplica(_Replica):
    """PD decode pool replica: continuous batching, no prefill."""

    def boundary(self):
        if self.in_flight:
            return
        if self._try_recompute():
            return
        self._admit()
        if self.running:
            self._decode_iter()


class ColocPrefillFirst(_Replica):
    """Colocated, vLLM without chunking: at every iteration boundary a waiting prefill batch (≤ cap) runs first and
    stalls all running decodes; otherwise one decode iteration."""

    def __init__(self, eng: Engine, costs: Costs, cap: int, slots: int):
        super().__init__(eng, costs, slots)
        self.cap = max(1, cap)
        self.pq: deque[Req] = deque()

    def offer_prefill(self, r: Req):
        r.ready_t = self.eng.t                # before_prefill: the admission wait includes prompt queueing
        self.pq.append(r)
        if not self.in_flight:
            self.boundary()

    def boundary(self):
        if self.in_flight:
            return
        if self._try_recompute():
            return
        self._admit()
        batch = []
        while self.pq and len(batch) < self.cap:
            if self.before:                       # vLLM: only requests that get a slot + KV start prefilling
                if not self._can_admit(self.pq[0]):
                    break
                r = self.pq.popleft()
                self._reserve(r)
                self.pending.append(r)
                batch.append(r)
            else:
                batch.append(self.pq.popleft())
        if batch:
            for r in batch:
                r.prefill_start = self.eng.t
            self._start(self.costs.batch_prefill_wall(batch), "rep_pf_done", batch)
        elif self.running:
            self._decode_iter()

    def pf_done(self, batch):
        self.in_flight = False
        for r in batch:
            r.prefill_done = r.kv_done = self.eng.t
            r.tokens_left = r.out
            if self.before:
                self.pending.remove(r)
                self._start_decoding(r)
            else:
                r.ready_t = self.eng.t
                self.joinq.append(r)
        self.boundary()


class ColocChunked(_Replica):
    """Colocated, Sarathi / vLLM chunked prefill: every iteration = running decodes + ≤ C tokens of the head-of-line
    prompt (FCFS, one prompt at a time); step = the engine's stage-level fused iteration (pdqueue._fused_step)."""

    def __init__(self, eng: Engine, costs: Costs, slots: int, C: int):
        super().__init__(eng, costs, slots)
        self.C = max(1, C)
        self.pq: deque[Req] = deque()
        self.cur: Req | None = None
        self.cur_left = 0
        self._pre1: dict = {}

    def _p1(self, S: int, p: int):
        k = (S, p)
        if k not in self._pre1:
            self._pre1[k] = self.costs.pool.run("prefill", 1, S, p)
        return self._pre1[k]

    def offer_prefill(self, r: Req):
        r.ready_t = self.eng.t                # before_prefill: the admission wait includes prompt queueing
        self.pq.append(r)
        if not self.in_flight:
            self.boundary()

    def boundary(self):
        if self.in_flight:
            return
        if self._try_recompute():
            return
        self._admit()
        if self.cur is None and self.pq and (not self.before or self._can_admit(self.pq[0])):
            self.cur = self.pq.popleft()
            if self.before:
                self._reserve(self.cur)
                self.pending.append(self.cur)
            self.cur.prefill_start = self.eng.t
            self.cur_left = self.cur.S - self.cur.p
        if self.cur is None:
            if self.running:
                self._decode_iter()
            return
        stall = self._preempt()
        new = max(1, self.cur.S - self.cur.p)
        take = min(self.C, self.cur_left)
        _, _, rr = _chunk_plan(self.cur.S, self.cur.p, self.C)
        pre1 = self._p1(self.cur.S, self.cur.p)
        dec = self.costs.pool.run("decode", max(1, len(self.running)))
        step = _fused_step(dec, pre1, take / new, rr)
        e = max(1, int(round(dec.tokens_per_step)))
        self.cur_left -= take
        fin = self.cur if self.cur_left <= 0 else None
        if fin is not None:
            self.cur = None
        self._start(step + stall, "rep_chunk_done", (list(self.running), e, fin))

    def chunk_done(self, payload):
        members, e, fin = payload
        self.in_flight = False
        self._credit(members, e)
        if fin is not None:
            fin.prefill_done = fin.kv_done = self.eng.t
            fin.tokens_left = fin.out
            if self.before:
                self.pending.remove(fin)
                self._start_decoding(fin)
            else:
                fin.ready_t = self.eng.t
                self.joinq.append(fin)
        self.boundary()


# --------------------------------------------------------------------------- top-level simulate


def _kv_setup(reps: list, ctx: dict) -> None:
    """pd.kv_policy (0.55): the same per-replica KV token capacity the closed form uses."""
    pol = ctx.get("kv_policy", "off")
    cap = (ctx.get("kv_cap") or {}).get("tokens")
    if pol == "off" or cap is None:
        return
    for rp in reps:
        rp.kv_policy, rp.kv_cap = pol, int(cap)
        rp.kv_admit = ctx.get("kv_admit", "after_prefill")
        if pol == "swap":
            rp.swap_Bps = (ctx.get("swap") or {}).get("Bps_replica", 0.0)
            rp.kv_bpt = (ctx.get("kv_cap") or {}).get("bytes_per_token", 0.0)


def _cap_of(pool: _Pool, x: dict | None) -> int:
    """Batch cap: the closed form's choice when it has one, else the largest cap that fits."""
    if x and x.get("_pre"):
        return x["_pre"]["b"]
    cap = 1
    for b in B_CAPS:
        if pool.run("prefill", b).fits:
            cap = b
    return cap


def _warm_caches(caches: list[LRUCache], n: int, alpha: float, rng: random.Random, mult: int = 10) -> None:
    """Fill each cache toward Che's steady state (cold misses ignored by the closed form)."""
    if not caches:
        return
    K = max(c.K for c in caches)
    ids = _zipf_ids(rng, n, alpha, max(K * mult, 1))
    for c in caches:
        for pid in ids:
            c.access(pid)
        c.hits = c.looks = 0


def _requests(rng: random.Random, ctx: dict, mode: str, lam: float, n_tot: int, lru: bool,
              prefix_n: int, prefix_alpha: float) -> list[Req]:
    lens = ctx["len"]
    pts = ctx["pts"] if mode == "pd" else ctx.get("pts_c", ctx["pts"])
    p_of = {S: p for _, S, p in pts}            # explicit / no prefix: one p per prompt value
    joint = [(w, S, o) for w, S, o in lens.points]
    cum, acc = [], 0.0
    for w, S, o in joint:
        acc += w
        cum.append(acc)
    pids = _zipf_ids(rng, prefix_n, prefix_alpha, n_tot) if lru else [-1] * n_tot
    reqs, t = [], 0.0
    for i in range(n_tot):
        t += rng.expovariate(lam)
        u = rng.random() * acc
        j = min(bisect.bisect_left(cum, u), len(cum) - 1)
        _, S, o = joint[j]
        reqs.append(Req(i, t, S, max(1, int(o)), 0 if lru else p_of.get(S, 0), pids[i]))
    return reqs


def simulate(ctx: dict, lam: float, mode: str = "pd", n_req: int = 2000, warmup: int = 400,
             seed: int = 1, prefix_K: int | None = None, prefix_K_dec: int | None = None,
             prefix_n: int = 0, prefix_alpha: float = 1.0, prefix_len: int = 0,
             affinity: bool = False, max_events: int = 5_000_000) -> dict:
    """DES of one serving mode at Poisson rate ``lam`` (requests/s).  ``ctx`` = the dict disagg builds for
    queue_report.  Statistics over requests ``warmup ≤ id < warmup + n_req`` (another ``warmup`` arrive after them
    so the tail is not measured on a draining system)."""
    from .pdqueue import _contended, _coloc_prefill_first, _pd_mode, _tier_util, _wsum
    rng = random.Random(seed)
    eng = Engine(rng)
    B, C = ctx["B"], ctx["C"]
    lru = prefix_len > 0 and prefix_n > 0 and prefix_K is not None
    n_tot = n_req + 2 * warmup
    reqs = _requests(rng, ctx, mode, lam, n_tot, lru, prefix_n, prefix_alpha)
    lo_id, hi_id = warmup, warmup + n_req
    done: list[Req] = []

    def record(r: Req):
        if lo_id <= r.rid < hi_id:
            done.append(r)

    caches_p = caches_d = None
    if mode == "pd":
        x = _pd_mode(ctx, lam)                 # same contended costs as the closed form (KV vs collectives)
        r_p, r_d = ctx["r_p"], ctx["r_d"]
        pre_scale = x["_pre"]["scale"] if x.get("_pre") else 1.0
        pp = Costs(ctx["ppool"])
        if pre_scale != 1.0:
            base = pp.prefill
            pp.prefill = lambda b, S, p, _f=base: _f(b, S, p) * pre_scale
        dp = Costs(ctx["dpool"])
        shared = ctx.get("shared", True)
        if shared and x.get("stable"):
            u_kv_d = x["kv"]["u_kv_decode"]
            if u_kv_d > 0:
                base_d = dp.decode_step

                def dstep(k, _f=base_d):
                    r = ctx["dpool"].run("decode", max(1, k))
                    st, e = _f(k)
                    return st * _contended(r, ctx["tier"], ctx["beta"], u_kv_d), e
                dp.decode_step = dstep
        beta_req = x["kv"]["GBps_req_avail"] * 1e9 if x.get("stable") else ctx["pair"] * ctx["beta"]
        cap = _cap_of(ctx["ppool"], x)
        prefs = [PrefillReplica(eng, pp, cap, f"p{i}") for i in range(r_p)]
        decs = [DecodeReplica(eng, dp, B) for _ in range(r_d)]
        _kv_setup(decs, ctx)
        kvs = [KVServer(eng, beta_req, ctx["alpha"]) for _ in range(r_p)]
        L, layerwise = ctx["L"], ctx["layerwise"]
        if layerwise:
            for kv in kvs:
                kv.alpha = ctx["alpha"] * L
        if lru:
            caches_p = [LRUCache(prefix_K) for _ in range(r_p)]
            caches_d = [LRUCache(prefix_K if prefix_K_dec is None else prefix_K_dec) for _ in range(r_d)]
            _warm_caches(caches_p + caches_d, prefix_n, prefix_alpha, rng)
        kv_xfer, kv_new, kv_full = ctx["kv_xfer"], ctx["kv_new"], ctx.get("kv_full", {})

        def nbytes(r: Req) -> float:
            if not lru:
                return kv_xfer[(r.S, r.p)]
            full = kv_full.get((r.S, 0)) or kv_full.get((r.S, r.p)) or kv_new.get((r.S, 0), 0.0)
            if r.hit and r.dec_holds and r.p:
                return full * (r.S - r.p) / r.S
            return full

        def after_prefill(batch, t):
            for r in batch:
                r.dec = (r.prefix_id % r_d) if (affinity and lru) else rng.randrange(r_d)
                if lru:
                    r.dec_holds = caches_d[r.dec].access(r.prefix_id)
                if decs[r.dec].before:            # vLLM: the decode side allocates KV, then pulls it
                    decs[r.dec].request_admit(r)
                else:
                    kvs[r.replica].offer(r, nbytes(r))

        for d in decs:
            d.on_admit = lambda r: kvs[r.replica].offer(r, nbytes(r))

        def kv_start(kv, r, nb):
            t_bw = nb / kv.beta if kv.beta > 0 else 0.0
            dur = kv.alpha + t_bw
            if layerwise:
                lat = r.prefill_done - r.prefill_start
                exp = max(ctx["alpha"] + t_bw / L, L * ctx["alpha"] + t_bw - lat * (L - 1) / L)
                eng.schedule(eng.t + exp, "kv_exposed", r)
            return dur

        for kv in kvs:
            kv.start_hook = kv_start
            kv.on_done = (lambda r, t: None) if layerwise else (lambda r, t: decs[r.dec].offer_decode(r))
        for pr in prefs:
            pr.on_done = after_prefill
        for d in decs:
            d.on_finish = record
        for r in reqs:
            eng.schedule(r.arrive, "arrive", r)
        n_ev = 0
        while len(done) < n_req and n_ev < max_events:
            ev = eng.pop()
            if ev is None:
                break
            n_ev += 1
            k = ev.kind
            if k == "arrive":
                r = ev.payload
                r.replica = (r.prefix_id % r_p) if (affinity and lru) else rng.randrange(r_p)
                if lru:
                    r.hit = caches_p[r.replica].access(r.prefix_id)
                    r.p = min(prefix_len, r.S - 1) if r.hit else 0
                prefs[r.replica].offer(r)
            elif k == "prefill_done":
                pr, batch = ev.payload
                pr.finish(batch)
            elif k == "kv_done":
                kv, r = ev.payload
                kv.finish(r)
            elif k == "kv_exposed":
                r = ev.payload
                r.kv_done = eng.t
                decs[r.dec].offer_decode(r)
            elif k == "rep_dec_done":
                rep, payload = ev.payload
                rep.dec_done(payload)
            elif k == "rep_recompute_done":
                rep, payload = ev.payload
                rep.recompute_done(payload)
        # Hit rates over the measurement window only (Che is steady-state; cold misses excluded).
        hit = (sum(1 for r in done if r.hit) / len(done)) if (lru and done) else None
        hit_d = (sum(1 for r in done if r.dec_holds) / len(done)) if (lru and done) else None
        ttfts = [r.kv_done - r.arrive for r in done]
        extra = {"batch_cap": cap, "prefix_hit_decode": hit_d,
                 "prefill_util": sum(p.busy_time for p in prefs) / max(eng.t, 1e-12) / len(prefs)}
    else:
        r_c = ctx["r_c"]
        costs = Costs(ctx["cpool"])
        if mode == "coloc_prefill_first":
            x = _coloc_prefill_first(ctx, lam)
            cap = _cap_of(ctx["cpool"], x)
            reps = [ColocPrefillFirst(eng, costs, cap, B) for _ in range(r_c)]
            done_kinds = {"rep_dec_done": "dec_done", "rep_pf_done": "pf_done", "rep_recompute_done": "recompute_done"}
        elif mode == "coloc_chunked":
            cap = None
            reps = [ColocChunked(eng, costs, B, C) for _ in range(r_c)]
            done_kinds = {"rep_dec_done": "dec_done", "rep_chunk_done": "chunk_done",
                          "rep_recompute_done": "recompute_done"}
        else:
            raise ValueError(f"unknown mode {mode!r}")
        _kv_setup(reps, ctx)
        if lru:
            caches_p = [LRUCache(prefix_K) for _ in range(r_c)]
            _warm_caches(caches_p, prefix_n, prefix_alpha, rng)
        for rep in reps:
            rep.on_finish = record
        for r in reqs:
            eng.schedule(r.arrive, "arrive", r)
        n_ev = 0
        while len(done) < n_req and n_ev < max_events:
            ev = eng.pop()
            if ev is None:
                break
            n_ev += 1
            if ev.kind == "arrive":
                r = ev.payload
                r.replica = (r.prefix_id % r_c) if (affinity and lru) else rng.randrange(r_c)
                if lru:
                    r.hit = caches_p[r.replica].access(r.prefix_id)
                    r.p = min(prefix_len, r.S - 1) if r.hit else 0
                reps[r.replica].offer_prefill(r)
            else:
                rep, payload = ev.payload
                getattr(rep, done_kinds[ev.kind])(payload)
        hit = (sum(1 for r in done if r.hit) / len(done)) if (lru and done) else None
        ttfts = [r.prefill_done - r.arrive for r in done]
        extra = {"batch_cap": cap, "busy": sum(rp.busy_time for rp in reps) / max(eng.t, 1e-12) / len(reps)}

    tpots = [(r.decode_done - r.decode_start) / r.out for r in done if r.out > 0]
    itls = [r.itl_max for r in done]
    st = {"mode": mode, "lambda_rps": lam, "n": len(done), "complete": len(done) >= n_req,
          "ttft": _stats(ttfts), "tpot": _stats(tpots), "itl_max": _stats(itls), "prefix_hit": hit, **extra}
    if ctx.get("kv_policy", "off") != "off":       # 0.55: decode-admission wait and preemptions
        st["slot_wait"] = _stats([r.admit_t - r.ready_t for r in done])
        st["preempt_per_req"] = sum(r.preempts for r in done) / max(1, len(done))
        st["preempted_frac"] = sum(1 for r in done if r.preempts) / max(1, len(done))
    for k in ("ttft", "tpot", "itl_max"):
        st[k + "_ms"] = {q: v * 1e3 for q, v in st[k].items()}
    return st


def slo_rate(ctx: dict, mode: str, ttft_slo: float, tpot_slo: float, start: float, n_req: int = 1200,
             warmup: int = 250, seed: int = 1, rel_tol: float = 0.03, **kw) -> float:
    """Largest λ whose simulated p90 TTFT and p90 request-average TPOT meet the SLOs (bisection, common random
    numbers across λ).  Same definition as pdqueue._slo_rate."""
    def ok(lam: float) -> bool:
        if lam <= 0:
            return True
        x = simulate(ctx, lam, mode, n_req=n_req, warmup=warmup, seed=seed, **kw)
        return x["complete"] and x["ttft"]["p90"] <= ttft_slo and x["tpot"]["p90"] <= tpot_slo
    if start <= 0:
        return 0.0
    lo, hi = 0.0, start
    for _ in range(12):
        if not ok(hi):
            break
        lo, hi = hi, hi * 1.5
    else:
        return lo
    if lo == 0.0:
        lo_try = hi
        for _ in range(12):
            lo_try *= 0.6
            if ok(lo_try):
                lo = lo_try
                break
            hi = lo_try
        else:
            return 0.0
    while hi - lo > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def capture_ctx(scn, energy=None):
    """Run disagg_report while capturing the live queueing ctx (Pools are not JSON-serialisable)."""
    from . import disagg as dg
    box = {}
    orig = dg.queue_report

    def wrap(ctx, *a, **k):
        box["ctx"] = ctx
        return orig(ctx, *a, **k)

    dg.queue_report = wrap
    try:
        rep = dg.disagg_report(scn, energy=energy)
    finally:
        dg.queue_report = orig
    return rep, box.get("ctx")


def rel_err(sim_v, ana_v):
    """(analytic − DES) / DES: > 0 means the closed form is pessimistic (larger latency / smaller capacity)."""
    if sim_v is None or ana_v is None or not math.isfinite(sim_v) or not math.isfinite(ana_v) or sim_v == 0:
        return float("nan")
    return (ana_v - sim_v) / abs(sim_v)


METRICS = (("ttft_p50", "ttft", "p50"), ("ttft_p90", "ttft", "p90"), ("ttft_p99", "ttft", "p99"),
           ("tpot_mean", "tpot", "mean"), ("tpot_p90", "tpot", "p90"), ("itl_max", "itl_max", "p99"))


def _ana_value(ana: dict, key: str) -> float:
    if key.startswith("ttft_"):
        return ana["ttft_ms"][key[5:]] / 1e3
    return {"tpot_mean": ana["tpot_mean_ms"], "tpot_p90": ana["tpot_p90_ms"], "itl_max": ana["itl_max_ms"]}[key] / 1e3


def compare(scn, n_req: int = 2000, warmup: int = 400, seed: int = 1,
            modes=("pd", "coloc_prefill_first", "coloc_chunked"), slo: bool = False, slo_n: int = 1000,
            seeds: tuple | None = None, slo_tol: float = 0.03) -> dict:
    """Closed form (pdqueue) vs DES for every mode at the scenario's offered load (and optionally SLO goodput).
    ``seeds`` (0.55): average the DES metrics over several independent runs and report their spread
    (``noise`` = sd / mean across seeds) — near saturation the decode occupancy relaxes over tens of seconds, so one
    short run carries ±10–20 % on TPOT / gap tails (V4 0.55).  The DES SLO rate is also averaged over ``seeds``
    (one 1500-request bisection scatters ±7 % at the SLO operating point); ``slo_tol`` = bisection tolerance."""
    seeds = tuple(seeds) if seeds else (seed,)
    rep, ctx = capture_ctx(scn)
    q = rep.get("queue") or {}
    if ctx is None or "modes" not in q:
        return {"error": q.get("error", "no queueing context")}
    lam, pd = q["lambda_rps"], scn.pd
    pc = rep.get("prefix_cache")
    lru = dict(prefix_len=pd.prefix_len, prefix_n=pd.prefix_count, prefix_alpha=pd.prefix_zipf,
               affinity=pd.prefix_affinity) if pc else {}
    sv = scn.serving
    cards = ctx["n_p"] + ctx["n_d"]
    rows = {}
    for mode in modes:
        ana = q["modes"].get(mode)
        if ana is None:
            continue
        kw = dict(lru)
        if pc:
            kw["prefix_K"] = pc["prefill"]["K"] if mode == "pd" else pc["coloc"]["K"]
            if mode == "pd":
                kw["prefix_K_dec"] = pc["decode"]["K"]
        sims = [simulate(ctx, lam, mode, n_req=n_req, warmup=warmup, seed=sd, **kw) for sd in seeds]
        row = {"stable_analytic": bool(ana.get("stable")), "complete": all(x["complete"] for x in sims),
               "sim": {}, "ana": {}, "err": {}}
        if len(sims) > 1:
            row["noise"] = {}
        for name, k, qk in METRICS:
            vals = [x[k][qk] for x in sims]
            sv_ = sum(vals) / len(vals)
            row["sim"][name] = sv_ * 1e3
            if len(vals) > 1 and sv_ > 0 and math.isfinite(sv_):
                row["noise"][name] = math.sqrt(sum((v - sv_) ** 2 for v in vals) / (len(vals) - 1)) / sv_
            if ana.get("stable"):
                av = _ana_value(ana, name)
                row["ana"][name] = av * 1e3
                row["err"][name] = rel_err(sv_, av)
        hits = [x["prefix_hit"] for x in sims if x["prefix_hit"] is not None]
        if pc and hits:
            key = "prefill" if mode == "pd" else "coloc"
            row["sim"]["prefix_hit"] = sum(hits) / len(hits)
            row["ana"]["prefix_hit"] = pc[key]["hit"]
            row["err"]["prefix_hit"] = pc[key]["hit"] - row["sim"]["prefix_hit"]          # absolute
            if mode == "pd":
                hd = [x["prefix_hit_decode"] for x in sims if x["prefix_hit_decode"] is not None]
                row["sim"]["prefix_hit_decode"] = sum(hd) / len(hd) if hd else None
                row["ana"]["prefix_hit_decode"] = pc["decode"]["hit"]
        if slo:
            start = max(ana.get("slo_rate_rps", 0.0), lam)
            rs = [slo_rate(ctx, mode, sv.ttft_slo_ms / 1e3, sv.tpot_slo_ms / 1e3, start, n_req=slo_n,
                           warmup=max(200, slo_n // 5), seed=s_, rel_tol=slo_tol, **kw) for s_ in seeds]
            r_sim = sum(rs) / len(rs)
            if len(rs) > 1 and r_sim > 0:
                row["noise"]["slo_goodput"] = math.sqrt(sum((v - r_sim) ** 2 for v in rs) / (len(rs) - 1)) / r_sim
            g_sim = r_sim * ctx["out"] / cards
            g_ana = ana.get("slo_goodput_per_card", 0.0)
            row["sim"]["slo_goodput"] = g_sim
            row["ana"]["slo_goodput"] = g_ana
            # for capacity, analytic > DES means optimistic → sign flipped so > 0 = pessimistic everywhere
            # both 0 (SLO unreachable at any rate, e.g. p90 prompt alone > TTFT SLO) → n/a, not a 0 % match
            row["err"]["slo_goodput"] = (g_sim - g_ana) / g_sim if g_sim > 0 else (float("nan") if g_ana == 0
                                                                                  else float("-inf"))
        rows[mode] = row
    return {"lambda_rps": lam, "load": q.get("load"), "modes": rows}


