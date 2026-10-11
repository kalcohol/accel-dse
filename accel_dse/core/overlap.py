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
PIPE_REPEATS = 3              # identical layers simulated before extrapolating the steady-state period
PIPE_MAX_TILES = 400_000      # stages up to this many tiles are replayed exactly; longer ones extrapolate


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


def _tiles(items, f: dict, sec_per_byte: float, dma_scale: float) -> list[tuple]:
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
        k = max(PIPE_MIN_TILES, math.ceil(raw / PIPE_TILE))
        m = nb * sec_per_byte * dma_scale / k
        out.extend([(nb / k, m, c / k)] * k)
    return out


def makespan(tiles: list[tuple], buf: float, cross_op: bool = True, op_starts: set | None = None) -> float:
    """Two-engine flow shop: the DMA loads tiles in order, at most ``buf`` bytes ahead of compute; compute runs tiles
    in order once loaded.  cross_op False: an op's first load waits for the previous op's compute (kernel boundary)."""
    n = len(tiles)
    cum = [0.0] * (n + 1)
    for i, t in enumerate(tiles):
        cum[i + 1] = cum[i] + t[0]
    done = [0.0] * n
    lim = buf * (1 + 1e-9) + 1e-6 * (cum[n] / max(n, 1))   # cum is a float prefix sum: compare with a tolerance
    d_end = c_end = 0.0
    p = 0
    for j, (sz, m, c) in enumerate(tiles):
        start = d_end
        if sz > 0:
            while cum[j + 1] - cum[p] > lim and p < j:     # tiles before p must have been consumed
                p += 1
            if p > 0:
                start = max(start, done[p - 1])
            if not cross_op and op_starts is not None and j in op_starts and j > 0:
                start = max(start, done[j - 1])
        d_end = start + m
        c_end = max(c_end, d_end if sz > 0 else c_end) + c
        done[j] = c_end
    return c_end


def stage_pipe_time(segments, f: dict, sec_per_byte: float, buf: float, cross_op: bool = True,
                    dma_scale: float = 1.0, tail_comm: float = 0.0, hidden_link: float = 0.0) -> float:
    """segments = [(items, repeat)] in program order.  Repeated layer groups are simulated PIPE_REPEATS times and
    extrapolated with their steady-state period (measured in isolation)."""
    def run(seq_items):
        tl, starts = [], set()
        for c, b, comm in seq_items:
            starts.add(len(tl))
            tl.extend(_tiles([(c, b, comm)], f, sec_per_byte, dma_scale))
        return makespan(tl, buf, cross_op, starts)
    def ntiles(its):
        return sum(1 if comm > 0 or sum(b.values()) <= 0 else max(PIPE_MIN_TILES, math.ceil(sum(b.values()) / PIPE_TILE))
                   for c, b, comm in its)
    total = sum(ntiles(its) * n for its, n in segments if its and n > 0)
    seq, extra = [], 0.0
    for items, n in segments:
        if not items or n <= 0:
            continue
        if total <= PIPE_MAX_TILES:         # exact: the whole stage is replayed (monotone in every input)
            seq.extend(items * n)
            continue
        r = min(n, PIPE_REPEATS)            # very long stages (DiT steps × layers): steady-state extrapolation
        seq.extend(items * r)
        if n > r:
            t2, t3 = run(items * (PIPE_REPEATS - 1)), run(items * PIPE_REPEATS)
            extra += (n - r) * max(0.0, t3 - t2)
    return max(run(seq) + extra, hidden_link) + tail_comm
