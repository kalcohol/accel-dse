"""0.98 — hardware-derived overlap (``Scenario.exec_overlap = "pipe"``, opt-in).

The other overlap modes are all-or-nothing rules over stage totals: ``stage`` = max of the components, ``serial`` =
their sum, ``class`` / ``kernel`` / ``tbo`` fixed mixtures. ``pipe`` instead replays the stage's op sequence on two
engines, which is what a double-buffered NPU does:

  DMA engine   streams each op's DRAM bytes (weights not pinned in SRAM, uncached KV / state, spilled activations)
               in program order, in tiles;
  compute      array and vector units run each op tile by tile. A tile starts when its bytes have landed and the
               previous tile is done. Inside one op the array and the vector unit pipeline, so the op's compute is
               max(array, vector). Separate vector ops (norms, activations, residual) are separate steps: they depend
               on the GEMM before them.

Rules, all derived from the hardware or from data dependencies:
  * double buffering: the DMA may run ahead of compute by at most the SRAM left for streaming
    (B = SRAM − pinned weights − cached KV / state ≥ the plan's staging area, which is ≥ 2 MiB). More SRAM never
    shrinks B and never adds DRAM bytes, so the makespan is monotone in SRAM.
  * tiles: an op touching b bytes is cut into max(PIPE_MIN_TILES, ⌈b / PIPE_TILE⌉) tiles (fixed by the op, not by
    how much of it the SRAM pins). The first tile's load
    (prologue) and the last tile's compute (epilogue) are exposed unless a neighbouring op covers them.
    PIPE_TILE = 1 MiB, half the minimum staging area, so two tiles always fit 「假设」.
  * cross-op prefetch (``Chip.pipe_cross_op``, default True): the DMA engine runs ahead into the next ops' data.
    GPUs launch one kernel at a time and a kernel's loads start when it starts, so the reference chips can set it
    False: each op then pays its own prologue and epilogue.
  * DMA issue rate (optional ``Chip.dma_engines`` × ``dma_outstanding_KiB`` / ``dram_latency_ns``, Little's law):
    when all three are set, the DRAM bandwidth of the System is min(DRAM, engines × outstanding / latency). This
    applies in every overlap mode (it is a bandwidth, not an overlap rule) and is opt-in: no public NPU numbers.
  * collectives sit on the compute chain (the next op consumes their result), but the DMA engine keeps prefetching
    behind them. MoE dispatch / combine overlap the shared-expert compute of the same layer, which does not depend
    on them (DeepSeek-V3 / SGLang practice). All other collectives are exposed. Their α stays in t_sync.
  * a PP hand-off is exposed at the end of the stage. FSDP weight gathers are prefetches and overlap the whole chain.
The per-op DRAM bytes come from the op's touched bytes times the plan's miss fraction per class (hot weights, routed
experts, KV / state), plus its streamed activations. They are scaled so that the stage's DRAM (or SLC) time is unchanged.
"""
from __future__ import annotations

import math

PIPE_TILE = float(2 ** 20)    # bytes per DMA tile (half the minimum 2 MiB staging area) 「假设」
PIPE_MIN_TILES = 32           # tiles per op at least (bounds the prologue to 1/32 of the op's load) 「假设」


def items_of(ops, op_secs, comm_secs, act_bytes) -> list[tuple]:
    """(compute s, {class: bytes}, chain-comm s) per schedulable step of one op list.  The attention core (QK, PV and
    the softmax that carries the K/V reads) is one fused step (flash attention); MoE dispatch / combine may hide behind
    the layer's shared-expert compute.

    0.139: the hiding window is the shared-expert steps' *wall* time in the flow shop, not their compute seconds —
    marker steps ("open" / "close" around each run of shared-expert steps, "hide" for a dispatch / combine) let the
    simulator credit the window it actually replays.  A DMA-bound shared expert (weights streaming) is a longer window
    than its compute; crediting compute only made a faster array expose more of the all-to-all (slower with more
    clock).  Unchanged whenever the window is compute-bound (wall = compute)."""
    out, fused, opened, seen = [], None, False, False
    for o in ops:
        sc, xkv = 1.0, 0.0
        if isinstance(o, tuple):            # 0.102: (op, scale, extra KV bytes) — a chunk's share of a prefill op
            o, sc, xkv = o
        role = getattr(o, "role", "")
        if opened and role != "shared_expert" and not (o.kind == "attn" or (o.name == "softmax" and fused is not None)):
            if fused is not None:
                out.append((max(fused[0], fused[1]), fused[2], 0.0)); fused = None
            out.append(("close", {}, 0.0)); opened = False
        if o.kind == "comm":
            bw = comm_secs(o) * sc
            if o.name in ("moe_dispatch", "moe_combine") and seen:
                out.append(("hide", {}, bw))
            else:
                out.append((0.0, {}, bw))
            continue
        a, _, _, v = op_secs(o)[:4]
        a, v = a * sc, v * sc
        b = {}
        if o.kind == "gemm":
            # 0.269: re-read hot weights (TTS flow DiT) miss at their own share (memplan.hot_miss)
            # 0.293: routed hot weights (MoVA value experts) at theirs (memplan.rt_miss)
            b["exp" if o.role == "expert" else ("hot_rr" if o.name.startswith("tts.flow") else
                                                ("hot_rt" if o.name == "attn.v_experts" else "hot"))] = o.w_bytes * sc
        if o.kind == "lookup":
            b["lookup"] = o.kv_read * sc
        elif o.kv_read or o.kv_write or xkv:
            b["kv"] = (o.kv_read + o.kv_write) * sc + xkv
        if o.kind != "lookup" and o.state_rw:          # 0.210: recurrent state has its own miss share (f["state"])
            b["state"] = o.state_rw * sc
        if o.stream:
            b["act"] = act_bytes(o) * sc
        if o.kind == "attn" or (o.name == "softmax" and fused is not None):
            if fused is None:
                fused = [0.0, 0.0, {}]
            fused[0] += a; fused[1] += v
            for k, x in b.items():
                fused[2][k] = fused[2].get(k, 0.0) + x
            if o.name == "softmax":
                out.append((max(fused[0], fused[1]), fused[2], 0.0)); fused = None
            continue
        if fused is not None:
            out.append((max(fused[0], fused[1]), fused[2], 0.0)); fused = None
        c = max(a, v)
        if o.role == "shared_expert" and c > 0:
            if not opened:
                out.append(("open", {}, 0.0)); opened = True
            seen = True
        out.append((c, b, 0.0))
    if fused is not None:
        out.append((max(fused[0], fused[1]), fused[2], 0.0))
    if opened:
        out.append(("close", {}, 0.0))
    if seen:
        out.append(("reset", {}, 0.0))     # unused credit does not carry past the op list (as before 0.139)
    return out


def _tiles(items, f: dict, sec_per_byte: float, dma_scale: float, tile: float = PIPE_TILE) -> list[tuple]:
    """Runs of identical tiles: ((bytes, load s, compute s), count)."""
    out = []
    for c, b, comm in items:
        if c.__class__ is str:              # 0.139: shared-expert window markers / hideable all-to-all
            out.append(((c, 0.0, comm), 1))
            continue
        if comm > 0:
            out.append(((0.0, 0.0, comm), 1))
            continue
        raw = sum(b.values())
        nb = sum(x * f.get(k, 1.0) for k, x in b.items())
        if raw <= 0:
            if c > 0:
                out.append(((0.0, 0.0, c), 1))
            continue
        # the tiling is the op's (from its touched bytes), not the plan's: pinning more of an op shrinks each tile's
        # load but never re-tiles it (monotone in SRAM)
        k = max(PIPE_MIN_TILES, math.ceil(raw / tile))
        m = nb * sec_per_byte * dma_scale / k
        out.append(((nb / k, m, c / k), k))
    return out


class _Sim:
    """Two-engine flow shop, fed tile by tile: the DMA loads tiles in order, at most ``buf`` bytes ahead of compute;
    compute runs tiles in order once loaded.  cross_op False: an op's first load waits for the previous op's compute
    (kernel boundary).  Tile sizes are whole bytes (rounded up), so the window sums are exact."""
    __slots__ = ("buf", "cross_op", "d_end", "c_end", "last", "log", "h", "wsum", "s0", "hw", "op")

    def __init__(self, buf: float, cross_op: bool = True):
        self.buf, self.cross_op = math.floor(buf), cross_op
        self.d_end = self.c_end = self.last = 0.0      # last = compute end of the newest tile that left the window
        # [size, compute end] of tiles still holding buffer space = log[h:] (0.289: a head index instead of
        # ``del win[:h]``; a list object is never mutated in place once a state snapshot may reference it — see state)
        self.log, self.h = [], 0
        self.wsum = 0
        self.op = False
        self.s0 = self.hw = 0.0                        # 0.139: open shared-expert window start / unused hide credit

    def feed_runs(self, runs, op_start: bool = False) -> None:
        """Runs of identical tiles. Inside a run, once a tile leaves the same relative state as the tile before it,
        the rest of the run adds the same time per tile, so it is added in one step (exact, as for layers)."""
        first = op_start
        for t, k in runs:
            prev = prev_dt = None
            i = 0
            if k < 8 * (len(self.log) - self.h + 8):   # short run vs the window: replaying is cheaper than comparing
                self.feed([t], first); first = False
                if k > 1:
                    self.feed([t] * (k - 1))
                continue
            while i < k:
                c0 = self.c_end
                self.feed([t], first); first = False; i += 1
                if k - i <= 0:
                    break
                # compare tile i with tile i − 1 only around powers of two (O(window · log k) state work)
                if (i & (i - 1)) == 0 and i >= 2:
                    prev, prev_dt = self.state(), self.c_end - c0
                elif prev is not None:
                    st, dt = self.state(), self.c_end - c0
                    if abs(dt - prev_dt) <= 1e-12 * max(self.c_end, 1e-30) and _same(st, prev, self.c_end):
                        self.shift(dt * (k - i)); i = k
                        break
                    prev = None

    def feed(self, tiles, op_start: bool = False) -> None:
        # 0.112: attributes held in locals for the loop (same operations in the same order — bit-identical)
        log, buf, hd = self.log, self.buf, self.h
        first = op_start and not self.cross_op
        d_end, c_end, last, wsum = self.d_end, self.c_end, self.last, self.wsum
        ceil = math.ceil
        for sz, m, c in tiles:
            if sz.__class__ is str:                    # 0.139: window markers (no bytes, no buffer)
                if sz == "open":
                    self.s0, self.op = c_end, True
                elif sz == "close":
                    self.hw += c_end - self.s0; self.op = False
                elif sz == "reset":
                    self.hw = 0.0
                else:                                  # "hide": the all-to-all runs under the window's wall time
                    h = min(self.hw, c); self.hw -= h; c_end += c - h
                continue
            start = d_end
            if sz > 0:
                sz = ceil(sz)
                if wsum + sz > buf:
                    nw = len(log)
                    while wsum + sz > buf and hd < nw:   # tiles before must have been consumed
                        w = log[hd]
                        wsum -= w[0]; last = w[1]; hd += 1
                if last > start:
                    start = last
                if first and c_end > start:
                    start = c_end
            first = False
            d_end = start + m
            if sz > 0 and d_end > c_end:
                c_end = d_end
            c_end += c
            log.append([sz, c_end]); wsum += sz
        self.d_end, self.c_end, self.last, self.wsum = d_end, c_end, last, wsum
        if hd > 4096 and 2 * hd > len(log):
            self.log, self.h = log[hd:], 0             # compact into a new list (snapshots keep the old one)
        else:
            self.h = hd

    def state(self):
        """Everything that decides the future, relative to the compute clock.  0.289: the window part is a lazy view
        (log list, head, end, clock) — ``_same`` reads (size, compute end − clock) per entry only when everything
        cheaper matched, with the same float expressions.  ≤ 0.288 every call built the full tuple: with a large
        buffer (4 GiB SRAM) the window never evicts and holds every tile fed so far, so each layer repeat cost
        O(tiles so far) — protenix on SRAM 4 GiB + SLC 4 GiB did not finish in 700 s.  Valid because no list object
        is mutated once referenced: ``shift`` rebuilds the window into a new list, and compaction copies."""
        c = self.c_end
        return (self.d_end - c, self.last - c, (self.log, self.h, len(self.log), c),
                0.0, self.hw)    # 0.209: an open window's start is absolute (see shift), not state

    def shift(self, dt: float) -> None:
        # 0.209: s0 stays absolute — skipped time inside an open shared-expert window still accrues hide credit at
        # "close" (hw += c_end − s0), exactly as replay would.  ≤ 0.208 s0 was in the state tuple and moved with the
        # clock, so a long run inside an open window never matched and was replayed tile by tile (O(tokens): a
        # 100k-batch prefill took minutes).
        self.d_end += dt; self.c_end += dt; self.last += dt
        if not self.op:
            self.s0 += dt
        self.log = [[w[0], w[1] + dt] for w in self.log[self.h:]]     # 0.289: new list (snapshots keep the old)
        self.h = 0


def makespan(tiles: list[tuple], buf: float, cross_op: bool = True, op_starts: set | None = None) -> float:
    sim = _Sim(buf, cross_op)
    if op_starts is None:
        sim.feed(tiles)
    else:
        for j, t in enumerate(tiles):
            sim.feed([t], j in op_starts)
    return sim.c_end


def _same(a, b, scale: float) -> bool:
    tol = 1e-12 * max(scale, 1e-30)
    la, ha, na, ca = a[2]
    lb, hb, nb, cb = b[2]
    if na - ha != nb - hb or abs(a[0] - b[0]) > tol or abs(a[1] - b[1]) > tol:
        return False
    if abs(a[3] - b[3]) > tol or abs(a[4] - b[4]) > tol:
        return False
    # entry-wise (size, compute end − clock), the 0.288 tuple's values (same subtraction, same order)
    for i in range(na - ha):
        x, y = la[ha + i], lb[hb + i]
        if x[0] != y[0] or abs((x[1] - ca) - (y[1] - cb)) > tol:
            return False
    return True


def stage_pipe_time(segments, f: dict, sec_per_byte: float, buf: float, cross_op: bool = True,
                    dma_scale: float = 1.0, tail_comm: float = 0.0, hidden_link: float = 0.0) -> float:
    """segments = [(items, repeat)] in program order, replayed exactly.  A repeated layer group is fed one repeat at
    a time; once a repeat leaves the simulator in the same state relative to the compute clock as the repeat before
    it, every later repeat adds the same time (the flow shop is deterministic), so the rest are added in one step.
    This is exact up to a 1e-12 relative tolerance on the state compare. 0.99: replaces the 0.98 extrapolation of
    very long stages.

    0.112: memoised on the full input (a pure function — the layout search / sweeps replay identical stages many
    times); the key holds every float and the dict insertion order, so a hit returns the bit-identical value."""
    key = (tuple((tuple((c, tuple(b.items()), cm) for c, b, cm in items), n) for items, n in segments),
           tuple(f.items()), sec_per_byte, buf, cross_op, dma_scale, tail_comm, hidden_link)
    r = _PIPE_MEMO.get(key)
    if r is None:
        r = _stage_pipe_time(segments, f, sec_per_byte, buf, cross_op, dma_scale, tail_comm, hidden_link)
        if len(_PIPE_MEMO) >= PIPE_MEMO_MAX:
            _PIPE_MEMO.clear()
        _PIPE_MEMO[key] = r
    return r


PIPE_MEMO_MAX = 20000
_PIPE_MEMO: dict = {}


def _stage_pipe_time(segments, f, sec_per_byte, buf, cross_op, dma_scale, tail_comm, hidden_link) -> float:
    sim = _Sim(buf, cross_op)
    tile = min(PIPE_TILE, buf / 2)      # 0.99: double buffering needs two tiles in the buffer
    for items, n in segments:
        if not items or n <= 0:
            continue
        per = [_tiles([it], f, sec_per_byte, dma_scale, tile) for it in items]
        prev, prev_dt, done = None, None, 0
        while done < n:
            t0 = sim.c_end
            for tl in per:
                sim.feed_runs(tl, True)
            done += 1
            st, dt = sim.state(), sim.c_end - t0
            if prev is not None and prev_dt is not None and _same(st, prev, sim.c_end) \
                    and abs(dt - prev_dt) <= 1e-12 * max(sim.c_end, 1e-30):
                sim.shift(dt * (n - done)); done = n
            prev, prev_dt = st, dt
    return max(sim.c_end, hidden_link) + tail_comm


_ATTN_CORE = ("softmax",)


def merge_chunk(ops_d, ops_p, f: float, rr: float, route=None):
    """0.102: one co-located iteration = decode ops ⊕ an f share of a prefill's ops, op by op (the engine runs one
    batched forward). GEMMs merge into one GEMM of m_dec + round(f·m_pre) rows (the weights are read once). The
    chunk's attention core runs as its own fused step after the decode's, reading f of the prompt's KV writes plus
    rr × the prompt's KV write (the prefix and the earlier chunks). Vector ops and collectives of the chunk follow
    the decode op of the same name, with work and payload × f. A prefill GEMM without a decode partner runs as the
    chunk's own GEMM of round(f·m) rows. Returns entries for ``items_of``."""
    from collections import defaultdict, deque
    from dataclasses import replace as _rep
    pend = defaultdict(deque)
    for o in ops_p:
        pend[o.name].append(o)
    kvw = sum(o.kv_write for o in ops_p)
    out, attn_p, seen_attn = [], [], False
    for od in ops_d:
        q = pend.get(od.name)
        op = q.popleft() if q else None
        if od.kind == "gemm":
            if od.role == "expert" and route is not None and op is not None:
                # 0.103: routed experts — the union of decode + chunk tokens hits more distinct experts
                # (0.289: ``route`` may be a function of the op — per-layer expert shapes)
                rd, ru = route(od) if callable(route) else route
                out.append(_rep(od, m=ru.m_e, count=ru.hit, w_params=od.w_params * ru.hit / max(od.count, 1)))
            elif op is not None and op.kind == "gemm":
                out.append(_rep(od, m=od.m + max(1, round(f * op.m))))
            else:
                out.append(od)
            continue
        out.append(od)
        if od.kind == "attn" or od.name in _ATTN_CORE:
            seen_attn = True
            if op is not None:
                attn_p.append(op)
            if od.name in _ATTN_CORE and attn_p:
                for i, x in enumerate(attn_p):          # the chunk's flash-attention step (+ the KV it re-reads)
                    out.append((x, f, rr * kvw if x.name in _ATTN_CORE else 0.0))
                attn_p = []
            continue
        if op is not None:
            out.append((op, f, 0.0))
    for x in attn_p:
        out.append((x, f, 0.0))
    for qq in pend.values():
        for x in qq:
            if x.kind == "gemm":                          # a prefill-only GEMM (e.g. non-absorbed MLA): the chunk's
                out.append(_rep(x, m=max(1, round(f * x.m))))   # own GEMM at its real shape
            else:
                out.append((x, f, 0.0))
    return out
