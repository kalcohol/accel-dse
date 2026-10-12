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
    the layer's shared-expert compute."""
    out, fused, shared = [], None, 0.0
    for o in ops:
        if o.kind == "comm":
            bw = comm_secs(o)
            if o.name in ("moe_dispatch", "moe_combine") and shared > 0:
                hide = min(shared, bw)
                shared -= hide
                bw -= hide
            out.append((0.0, {}, bw))
            continue
        a, _, _, v, _, _, _, _ = op_secs(o)
        b = {}
        if o.kind == "gemm":
            b["exp" if o.role == "expert" else "hot"] = o.w_bytes
        if o.kind == "lookup":
            b["lookup"] = o.kv_read
        elif o.kv_read or o.kv_write or o.state_rw:
            b["kv"] = o.kv_read + o.kv_write + o.state_rw
        if o.stream:
            b["act"] = act_bytes(o)
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
        if o.role == "shared_expert":
            shared += c
        out.append((c, b, 0.0))
    if fused is not None:
        out.append((max(fused[0], fused[1]), fused[2], 0.0))
    return out


def _tiles(items, f: dict, sec_per_byte: float, dma_scale: float, tile: float = PIPE_TILE) -> list[tuple]:
    """(bytes, load s, compute s) per tile."""
    out = []
    for c, b, comm in items:
        if comm > 0:
            out.append((0.0, 0.0, comm))
            continue
        raw = sum(b.values())
        nb = sum(x * f.get(k, 1.0) for k, x in b.items())
        if raw <= 0:
            if c > 0:
                out.append((0.0, 0.0, c))
            continue
        # the tiling is the op's (from its touched bytes), not the plan's: pinning more of an op shrinks each tile's
        # load but never re-tiles it (monotone in SRAM)
        k = max(PIPE_MIN_TILES, math.ceil(raw / tile))
        m = nb * sec_per_byte * dma_scale / k
        out.extend([(nb / k, m, c / k)] * k)
    return out


class _Sim:
    """Two-engine flow shop, fed tile by tile: the DMA loads tiles in order, at most ``buf`` bytes ahead of compute;
    compute runs tiles in order once loaded.  cross_op False: an op's first load waits for the previous op's compute
    (kernel boundary).  Tile sizes are whole bytes (rounded up), so the window sums are exact."""
    __slots__ = ("buf", "cross_op", "d_end", "c_end", "last", "win", "wsum")

    def __init__(self, buf: float, cross_op: bool = True):
        self.buf, self.cross_op = math.floor(buf), cross_op
        self.d_end = self.c_end = self.last = 0.0      # last = compute end of the newest tile that left the window
        self.win = []                                  # [size, compute end] of tiles still holding buffer space
        self.wsum = 0

    def feed(self, tiles, op_start: bool = False) -> None:
        win, buf = self.win, self.buf
        first = op_start and not self.cross_op
        for sz, m, c in tiles:
            start = self.d_end
            if sz > 0:
                sz = math.ceil(sz)
                h = 0
                while self.wsum + sz > buf and h < len(win):    # tiles before must have been consumed
                    self.wsum -= win[h][0]; self.last = win[h][1]; h += 1
                if h:
                    del win[:h]
                if self.last > start:
                    start = self.last
                if first and self.c_end > start:
                    start = self.c_end
            first = False
            self.d_end = start + m
            if sz > 0 and self.d_end > self.c_end:
                self.c_end = self.d_end
            self.c_end += c
            win.append([sz, self.c_end]); self.wsum += sz

    def state(self):
        """Everything that decides the future, relative to the compute clock."""
        c = self.c_end
        return (self.d_end - c, self.last - c, tuple((s, d - c) for s, d in self.win))

    def shift(self, dt: float) -> None:
        self.d_end += dt; self.c_end += dt; self.last += dt
        for w in self.win:
            w[1] += dt


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
    if len(a[2]) != len(b[2]) or abs(a[0] - b[0]) > tol or abs(a[1] - b[1]) > tol:
        return False
    return all(x[0] == y[0] and abs(x[1] - y[1]) <= tol for x, y in zip(a[2], b[2]))


def stage_pipe_time(segments, f: dict, sec_per_byte: float, buf: float, cross_op: bool = True,
                    dma_scale: float = 1.0, tail_comm: float = 0.0, hidden_link: float = 0.0) -> float:
    """segments = [(items, repeat)] in program order, replayed exactly.  A repeated layer group is fed one repeat at
    a time; once a repeat leaves the simulator in the same state relative to the compute clock as the repeat before
    it, every later repeat adds the same time (the flow shop is deterministic), so the rest are added in one step.
    This is exact up to a 1e-12 relative tolerance on the state compare. 0.99: replaces the 0.98 extrapolation of
    very long stages."""
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
                sim.feed(tl, True)
            done += 1
            st, dt = sim.state(), sim.c_end - t0
            if prev is not None and prev_dt is not None and _same(st, prev, sim.c_end) \
                    and abs(dt - prev_dt) <= 1e-12 * max(sim.c_end, 1e-30):
                sim.shift(dt * (n - done)); done = n
            prev, prev_dt = st, dt
    return max(sim.c_end, hidden_link) + tail_comm
