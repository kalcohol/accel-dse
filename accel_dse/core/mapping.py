"""L3b — mapping / dataflow: per-op time bounds on a given chip.

The datapath organisation is a **design variable** (no presupposed small-M
path).  Organisations:

  os         output-stationary systolic array only
  ws_edge    weight-stationary (K-split, adder tree), weights shifted in from
             one edge: load width = C·E elements / cycle
  ws_broad   weight-stationary with broadside load: R·C·E elements / cycle
  os_vec     OS array + dedicated GEMV/vector MAC unit; each GEMM takes the
             faster of the two (the unit does not add peak to the array)
  reconf     reconfigurable: per-op best of os / ws_edge / ws_broad / gemv

Per GEMM (M,K,N, count instances, bf16-equivalent rate multiplier r):

  OS     mac  = ceil(M/R)·ceil(N/Ce)·K / r
         feed = [ceil(M/R)·K·N·wb + ceil(N/Ce)·M·K·ab + M·N·ob] / port
  WS     tiles = ceil(K/R)·ceil(N/Ce);  load = R·Ce / (w_load·r)   (16-bit lanes; fp8 packs 2)
         mac  = tiles·max(ceil(M/r), load)   (+ R + Ce pipeline fill once per op)
         feed = [K·N·wb + ceil(N/Ce)·M·K·ab + 2·(ceil(K/R)−1)·M'·N·4 + M·N·ob] / port
                M' = max(0, M − acc_rows): rows whose fp32 partial sums overflow the
                accumulator (acc_kib 「假设」) and round-trip through SRAM
  GEMV   mac  = M·K·N / (V·r)
         feed = [K·N·wb + M·K·ab + M·N·ob] / port

  t_op = max(mac, feed) · count / (f · mac_eff)

The *feed* bound is the SRAM-port limit: an OS array at M=1 consumes Ce
weights per cycle, so its weight ingest ≤ Ce·wb·f regardless of DRAM speed —
this is what makes small-M mapping a first-order design question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

from .dtypes import FormatSupport, dequant_ops, fmt as _fmt, gemm_exec
from .hardware import Chip

ORGS = ("os", "ws_edge", "ws_broad", "os_vec", "reconf")
ORG_LABEL = {
    "os": "OS 脉动阵列",
    "ws_edge": "WS 边缘加载 (K-split)",
    "ws_broad": "WS 宽边加载 (K-split)",
    "os_vec": "OS + GEMV 单元",
    "reconf": "可重构 (逐算子最优)",
}


@dataclass(frozen=True)
class GemmCost:
    cycles: float          # max(mac, feed) incl. count, before mac_eff
    mac_cycles: float
    feed_cycles: float
    dataflow: str          # os | ws_edge | ws_broad | gemv
    exec_fmt: str
    conversion: str        # none | dequant_w | upcast_both | downcast
    convert_elems: float   # elements converted on the vector path
    convert_w: float = 0.0  # of which second-operand (weight) elements: once per pass, independent of tokens (0.63)
    # 0.290: element-ops above the 2-per-element convention (dtypes.DEQUANT_OPS: IQ2_XXS codebook dequant); 0 for every
    # other format, so the conversion seconds of those are unchanged
    convert_x: float = 0.0
    convert_wx: float = 0.0  # of which on the second operand (weights)
    # 0.299: on-chip NoC traffic between engines (energy only; time is unchanged): bytes × router traversals and
    # bytes × link length in engine-pitch units, from the dataflow (operand multicast, cross-group operand copies,
    # split-K partial-sum reduction).  See noc_multicast.
    noc_hops: float = 0.0
    noc_pitch: float = 0.0
    # 0.303: cycles the NoC payload needs across the bisection (Chip.noc_bw_cycles; 0 when noc_link_Bpc is unset) —
    # already folded into ``cycles`` (max with the array); kept for reporting
    noc_cycles: float = 0.0
    # 0.307 (Chip.noc_eject "link" only, else 0): op-total cycles of the busiest network link and of one engine's
    # local port — the stage adds the DRAM stream's share of those resources (Chip.noc_share_seconds)
    noc_link: float = 0.0
    noc_eject: float = 0.0

    @property
    def bound(self) -> str:
        if self.noc_cycles > max(self.feed_cycles, self.mac_cycles):     # 0.303 (only with noc_link_Bpc set)
            return "noc"
        return "feed" if self.feed_cycles > self.mac_cycles else "mac"


def _cd(a: float, b: float) -> int:
    return int(-(-a // b))


WIDE_WAVES = True          # 0.193 default (False = ≤ 0.192 gang padding, MODEL §139)
# 0.202: the wave refill ("a GEMM kernel hands idle SMs other row tiles") is a GPU-kernel mechanism; on an NPU the
# "wide" schedule is one lock-step ganged array fed by a broadcast A row tile (the feed term above still assumes that
# broadcast), so N stays padded to E·cols there.  ≤ 0.201 the refill also reached NPU "auto"→"wide" plans.
# False = ≤ 0.201.
GPU_WAVES_ONLY = True


def noc_multicast(ch: Chip, dests: int, unit: float = 1.0) -> tuple[float, float]:
    """0.299: (router traversals, link length in engine pitches) to deliver one byte from one source to ``dests``
    blocks of ``unit`` engines each (the source block holds it already, so dests − 1 receivers).  A reduction tree
    over the same blocks is the mirror image (same cost).  Blocks are contiguous: on a mesh adjacent √u × √u blocks
    are √u engine hops apart and a multicast tree spans dests − 1 block links (XY-tree multicast, Park et al. DAC 2012
    "router-level multicast": broadcast = k² crossbar + (k² − 1) link traversals); on a ring adjacent blocks are u hops
    apart; a crossbar is one switch output traversal per receiver, the source → centre leg once and a centre →
    receiver leg per receiver, each the mean ½ side of the √(dests·u)-pitch array (centre to a uniform point of a
    square) 「推导」."""
    if dests <= 1:
        return 0.0, 0.0
    if ch.noc_topology == "crossbar":
        return float(dests - 1), dests * 0.5 * math.sqrt(dests * unit)
    step = math.sqrt(unit) if ch.noc_topology == "mesh" else float(unit)
    return (dests - 1) * step, (dests - 1) * step


def _a_bytes(ch: Chip, df: str, m, k, n, ab) -> float:
    """0.299: bytes of the streamed (A) operand the feed term sends to the array: once per E·cols column tile for the
    systolic dataflows, once for GEMV (the mapping docstring formulas)."""
    return (m * k * ab) if df == "gemv" else _cd(n, ch.c_eff) * m * k * ab


def _wide_noc_cycles(ch: Chip, df: str, m, k, n, ab) -> float:
    """0.303: per-instance cycles of the ganged multicast of the A operand across the bisection of the receiving
    block (min(E, ⌈N / cols⌉) engines) — 0 when noc_link_Bpc is unset or nothing is multicast."""
    if not ch.noc_link_Bpc or ch.engines <= 1:
        return 0.0
    d = min(ch.engines, _cd(n, ch.cols))
    return ch.noc_traffic_cycles(_a_bytes(ch, df, m, k, n, ab), d) if d > 1 else 0.0


def _wide_port_cycles(ch: Chip, df: str, m, k, n, ab) -> tuple[float, float]:
    """0.307: per-instance (busiest link, local port) cycles of the ganged multicast (local-port model on)."""
    if not ch.noc_eject_on() or ch.engines <= 1:
        return 0.0, 0.0
    d = min(ch.engines, _cd(n, ch.cols))
    return ch.noc_port_cycles(_a_bytes(ch, df, m, k, n, ab), d) if d > 1 else (0.0, 0.0)


def _wide_noc(ch: Chip, df: str, m, k, n, ab, count) -> tuple[float, float]:
    """0.299: a ganged ("wide") GEMM: the weight columns sit in each engine's local slice (no NoC), the A row tile is
    all-gathered / multicast to every engine that holds output columns (min(E, ⌈N / cols⌉)); outputs stay local."""
    if ch.engines <= 1:
        return 0.0, 0.0
    h, p = noc_multicast(ch, min(ch.engines, _cd(n, ch.cols)))
    a = _a_bytes(ch, df, m, k, n, ab) * count
    return a * h, a * p


def _os(ch: Chip, m, k, n, r, wb, ab, ob):
    R, Ce = ch.rows, ch.c_eff
    mac = _cd(m, R) * _cd(n, Ce) * k / r
    if WIDE_WAVES and not ch.wide_pad and ch.instance_sched == "wide" and ch.engines > 1 and (ch.gpu or not GPU_WAVES_ONLY):
        # 0.192: a ganged array whose output that is not a multiple of E·cols leaves whole engines idle in the last column tile; a GEMM kernel hands
        # them other row tiles (R × cols output tile per engine, waves over E engines — the 0.92 cta tile/wave
        # quantisation without tuned tile shapes).  Before: N padded to E·cols, so efficiency depended on whether one
        # output dim happened to fit the gang (H200 4096 → 8192-token prefill per-token time fell 25 %)
        mac = min(mac, _cd(_cd(m, R) * _cd(n, ch.cols), ch.engines) * k / r)
    byt = _cd(m, R) * k * n * wb + _cd(n, Ce) * m * k * ab + m * n * ob
    return mac, byt / ch.port_Bpc


def _ws(ch: Chip, m, k, n, r, wb, ab, ob, broad: bool):
    R, Ce = ch.rows, ch.c_eff
    w_load = (R * Ce if broad else Ce) * r      # lanes are 16-bit: narrow formats pack r per lane
    load = R * Ce / w_load
    tiles = _cd(k, R) * _cd(n, Ce)
    mac = tiles * max(math.ceil(m / r), load)
    spill_rows = max(0, m - ch.acc_rows)        # rows whose fp32 partial sums leave the accumulator
    byt = k * n * wb + _cd(n, Ce) * m * k * ab + 2 * (_cd(k, R) - 1) * spill_rows * n * 4 + m * n * ob
    return mac, byt / ch.port_Bpc


def _gemv(ch: Chip, m, k, n, r, wb, ab, ob):
    mac = m * k * n / (ch.gemv * r)
    byt = k * n * wb + m * k * ab + m * n * ob
    return mac, byt / ch.port_Bpc


def _groups(e: int) -> tuple[int, ...]:
    """Core-group sizes g < E that divide E (E/g concurrent instances)."""
    return tuple(g for g in range(1, e) if e % g == 0)


def _actives(par: int) -> tuple[int, ...]:
    """Active group counts tried by auto: par, par/2, … (integer halvings) down to 1."""
    out, a = [], par
    while a >= 1:
        out.append(a); a //= 2
    return tuple(out)


def _candidates(org: str) -> tuple[str, ...]:
    return {"os": ("os",), "ws_edge": ("ws_edge",), "ws_broad": ("ws_broad",),
            "os_vec": ("os", "gemv"), "reconf": ("os", "ws_edge", "ws_broad", "gemv")}[org]


def gemm_cost(ch: Chip, org: str, m: int, k: int, n: int, *, count: int = 1, w_fmt: str = "bf16",
              a_fmt: str = "bf16", w_bits: float | None = None) -> GemmCost:
    """Cycles for ``count`` identical GEMMs [m,k]×[k,n] under organisation ``org`` (memoised: pure function)."""
    return _gemm_cost(ch, org, m, k, n, count, w_fmt, a_fmt, w_bits)


@lru_cache(maxsize=1 << 12)
def _sub_chip(ch: Chip, g: int, act: int) -> Chip:
    """0.277: the core-group sub-chip of the split / auto instance schedule (``g`` engines, port share 1 / ``act``) —
    memoised: building it (dataclasses.replace + Chip.__post_init__) per GEMM shape was ~40 % of a speech-streams run;
    pure function of (ch, g, act), so every cost is unchanged."""
    return replace(ch, engines=g, split_instances=False, instance_sched="wide",
                   sram_port_Bpc=ch.port_Bpc / act, sram_port_Bpc_per_engine=None, acc_kib=ch.acc_total_kib * g / ch.engines,
                   acc_kib_per_engine=None)


@lru_cache(maxsize=1 << 18)
def _gemm_cost(ch: Chip, org: str, m: int, k: int, n: int, count: int, w_fmt: str, a_fmt: str,
               w_bits: float | None) -> GemmCost:
    if org not in ORGS:
        raise ValueError(f"unknown mapping {org!r}; one of {ORGS}")
    if m <= 0 or k <= 0 or n <= 0 or count <= 0:
        return GemmCost(0.0, 0.0, 0.0, "none", a_fmt, "none", 0.0)
    ex, rate, conv, flag = gemm_exec(w_fmt, a_fmt, ch.formats)
    # 0.197: 0 = unset (Op default).  0.202: None is unset under both settings (the legacy branch divided None)
    wb = (w_bits if w_bits is not None and (w_bits or not W_BITS_ZERO_FIX) else _fmt(w_fmt).bits) / 8.0
    ab = _fmt(a_fmt).bytes
    ob = max(2.0, ab)       # outputs leave in bf16 (fp32 when the activations are fp32)
    if ch.instance_sched == "cta":
        return _cta(ch, m, k, n, count, rate, wb, ab, ob, ex, conv, flag, w_fmt, a_fmt)
    best = None
    for df in _candidates(org):
        if df == "os":
            mac, feed = _os(ch, m, k, n, rate, wb, ab, ob)
        elif df == "gemv":
            mac, feed = _gemv(ch, m, k, n, rate, wb, ab, ob)
        else:
            mac, feed = _ws(ch, m, k, n, rate, wb, ab, ob, broad=(df == "ws_broad"))
        fill_c = (ch.rows + ch.c_eff) / count if df.startswith("ws") else 0.0
        nc = _wide_noc_cycles(ch, df, m, k, n, ab)      # 0.303: NoC bisection bound (0 = unlimited)
        cyc = max(mac + fill_c, feed, nc)
        if best is None or cyc < best[0]:
            best = (cyc, mac, feed, df, nc)
    cyc, mac, feed, df, nc = best
    fill = (ch.rows + ch.c_eff) if df.startswith("ws") else 0.0   # pipeline fill once per op (instances stream back-to-back)
    lat = ch.noc_cycles() + ch.wave_overhead_cycles   # 0.72 (default 0): NoC hand-off over E cores + one launch
    sched = "split" if ch.split_instances else ch.instance_sched
    # 0.70 / 0.71: instances spread over core groups; 0.72: auto also for a single GEMM (count = 1)
    if sched != "wide" and ch.engines > 1 and (count > 1 or sched == "auto"):
        best_c = max(mac * count + fill, feed * count, nc * count) + lat
        alt = None
        # 0.139: a row (column) split makes every group convert its own copy of the K×N (M×K) operand on the
        # dequant / vector path, which overlaps the array (stage = max of the paths) — the picker scores each plan as
        # max(array, conversion) cycles, so a wider SRAM port can never select a split whose replicated conversion
        # outweighs the array cycles it saves
        conv_rate = ch.formats.dequant_elems_per_cycle or ch.lanes
        xw, xa = _conv_extra(m, k, n, w_fmt, a_fmt, flag)
        base_conv = (((k * n if flag & 1 else 0.0) + (m * k if flag & 2 else 0.0)) * count * 2.0
                     + (xw + xa) * count) / conv_rate
        best_s = max(best_c, base_conv)
        for g in (_groups(ch.engines) if sched == "auto" else (1,)):
            par = ch.engines // g
            # active groups: all, or fewer (idle groups free their share of the shared SRAM port / NoC; the port is
            # a chip-total budget split over the groups that are streaming, accumulators stay per core)
            for act in (_actives(par) if sched == "auto" else (par,)):
                sub = _sub_chip(ch, g, act)
                # auto: when instances < active groups, also split each instance's M rows over s groups (row-parallel;
                # every group streams the full K×N operand through its share of the port)
                # 0.72: or split its N columns over s groups (each group streams the full M×K operand)
                s_m = max(1, min(act // count, m)) if sched == "auto" else 1
                s_n = max(1, min(act // count, n)) if sched == "auto" else 1
                cands = [(1, m, n)] + ([(s_m, _cd(m, s_m), n)] if s_m > 1 else []) + \
                    ([(s_n, m, _cd(n, s_n))] if s_n > 1 else [])
                for ci, (sp, mm, nn) in enumerate(cands):
                    waves = _cd(count * sp, act)
                    # each group streams its `waves` instances back to back (pipeline fill once, as wide does);
                    # the sub-call already pays one launch + its own NoC span
                    one = _gemm_cost(sub, org, mm, k, nn, waves, w_fmt, a_fmt, w_bits)
                    # 0.72: ≥ 2 active groups advance in lock-step waves → a barrier per extra wave; one active group
                    # (wide on a subset of cores) streams back to back like wide
                    c1 = one.cycles + ((waves - 1) * ch.wave_overhead_cycles if act > 1 else 0.0)
                    # 0.303: a split's cross-group operand copies (row split: K×N, column split: M×K to sp groups)
                    # cross the chip bisection, overlapped with the groups' own work
                    # 0.304: noc_routing "xy" — the copy is an all-gather over the sp groups' trees (busiest link,
                    # spread over a group's boundary links) and shares links with the groups' own multicasts, so the
                    # two NoC loads add (concurrent traffic on one link) instead of overlapping
                    xb_ = count * k * n * wb if (sp > 1 and mm != m) else count * m * k * ab if sp > 1 else 0.0
                    if ch.noc_link_Bpc and ch.noc_routing == "xy":
                        xn = ch.noc_traffic_cycles(xb_, sp, g) + (one.noc_cycles if xb_ else 0.0)
                        xl, xe = ch.noc_port_cycles(xb_, sp, g)      # 0.307 (0, 0) unless noc_eject "link"
                    else:
                        xn = ch.noc_bw_cycles(xb_) if ch.noc_link_Bpc else 0.0
                        xl = xe = 0.0
                    c1 = max(c1, xn)
                    sc = max(c1, (one.convert_elems * count * sp / waves * 2.0
                                  + one.convert_x * count * sp / waves) / conv_rate) if flag else c1
                    if sc < best_s:
                        # 0.299: a row split copies each instance's K×N operand to its sp groups, a column split its
                        # M×K operand (the sub-call already holds the in-group multicast); instances spread over
                        # groups keep their operands in their own groups' slices 「假设」
                        xb = (count * k * n * wb if (sp > 1 and mm != m) else count * m * k * ab if sp > 1 else 0.0)
                        xh, xp = noc_multicast(ch, sp, g) if xb else (0.0, 0.0)
                        best_s, best_c, alt = sc, c1, (one, count * sp / waves, c1 - one.cycles, xb * xh, xb * xp,
                                                       max(one.noc_cycles, xn), one.noc_link + xl, one.noc_eject + xe)
        if alt is not None:
            # conversions: every (row-split) instance piece converts its own operands (row splits replicate K×N)
            one, per, extra, xh, xp, xn, xl, xe = alt
            nh, npi = one.noc_hops * per + xh, one.noc_pitch * per + xp
            if not extra:
                return GemmCost(one.cycles, one.mac_cycles, one.feed_cycles, one.dataflow, one.exec_fmt, one.conversion,
                                one.convert_elems * per, one.convert_w * per, one.convert_x * per, one.convert_wx * per,
                                nh, npi, xn, xl, xe)
            return GemmCost(one.cycles + extra, one.mac_cycles + extra, one.feed_cycles, one.dataflow, one.exec_fmt,
                            one.conversion, one.convert_elems * per, one.convert_w * per, one.convert_x * per,
                            one.convert_wx * per, nh, npi, xn, xl, xe)
    conv_elems = 0.0
    if flag & 1:
        conv_elems += k * n
    if flag & 2:
        conv_elems += m * k
    xw, xa = _conv_extra(m, k, n, w_fmt, a_fmt, flag)
    nh, npi = _wide_noc(ch, df, m, k, n, ab, count)
    pl, pe = _wide_port_cycles(ch, df, m, k, n, ab)      # 0.307 (0, 0) unless noc_eject "link"
    if lat:
        return GemmCost(max(mac * count + fill, feed * count, nc * count) + lat, mac * count + fill + lat, feed * count,
                        df, ex, conv, conv_elems * count, (k * n * count) if flag & 1 else 0.0, (xw + xa) * count,
                        xw * count, nh, npi, nc * count, pl * count, pe * count)
    return GemmCost(max(mac * count + fill, feed * count, nc * count), mac * count + fill, feed * count, df, ex, conv,
                    conv_elems * count, (k * n * count) if flag & 1 else 0.0, (xw + xa) * count, xw * count, nh, npi,
                    nc * count, pl * count, pe * count)


def _conv_extra(m, k, n, w_fmt, a_fmt, flag) -> tuple[float, float]:
    """0.290: conversion element-ops above the 2-per-element convention, (second operand, first operand) — nonzero
    only for formats in dtypes.DEQUANT_OPS (IQ2_XXS)."""
    xw = k * n * (dequant_ops(w_fmt) - 2.0) if flag & 1 else 0.0
    xa = m * k * (dequant_ops(a_fmt) - 2.0) if flag & 2 else 0.0
    return xw, xa


# 0.92: NVIDIA "Matrix Multiplication Background" user guide — cuBLAS thread-block tiles (most → least efficient)
CTA_TILES = ((256, 128), (128, 256), (128, 128), (256, 64), (64, 256), (128, 64), (64, 128), (64, 64))
SPLIT_K = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
# 0.193: splits beyond 16 = stream-K style K decomposition over idle SMs (CUTLASS Stream-K, Osama et al. PPoPP 2023:
# MAC-loop iterations divided over all SMs, partial tiles fixed up), still bounded by tiles·s ≤ 2·E and K slices ≥ 64.
# ≤ 0.192: 1–16 (the guide's split-K set) left a tiny-output long-K GEMM (DeepSeek-V4 hc mixers 1 × 28672 × 24) on
# a handful of SMs (MI300X 3257× its FLOP time)


# 0.195: CTA resources per tile config (CUTLASS sm80 DefaultGemmConfiguration, OpClassTensorOp: kStages = 3,
# threadblock K = 64 for 16-bit operands = 128 B / row (the TMA / cp.async 128 B swizzle), 256 threads = 8 warps)
# 「二手」; accumulator = BM × BN fp32 in registers (Hopper) / TMEM (Blackwell).  Max 32 resident CTAs per SM (CUDA
# Programming Guide, compute capability table).
OCC_STAGES, OCC_K_BYTES, OCC_CTA_THREADS, OCC_MAX_CTAS = 3, 128, 256, 32
CTA_DRAM_FLOOR = True   # 0.196 (False = ≤ 0.195)
CTA_SMEM_BW = True      # 0.196 (False = ≤ 0.195)
CTA_OCC_FILL = True     # 0.196 (False = ≤ 0.195)
CTA_L2_ACT = True       # 0.197 (False = ≤ 0.196)
W_BITS_ZERO_FIX = True  # 0.197 (False = ≤ 0.196: w_bits 0.0 → 0-byte B operand)


def cta_occupancy(ch: Chip, tm: int, tn: int, ab: float, wb: float) -> int:
    """Resident CTAs per SM for a tm × tn tile under the CUDA occupancy rules: min over shared memory (stages × (BM +
    BN) × 128 B of K), accumulator storage (BM·BN·4 B) and threads; ≥ 1."""
    smem = OCC_STAGES * (tm + tn) * OCC_K_BYTES / 1024
    acc = tm * tn * 4 / 1024
    k = min(int(ch.occ_smem_kib // smem), int(ch.occ_reg_kib // acc) if ch.occ_reg_kib else OCC_MAX_CTAS,
            (ch.occ_threads // OCC_CTA_THREADS) if ch.occ_threads else OCC_MAX_CTAS, OCC_MAX_CTAS)
    return max(1, k)


def _cta(ch: Chip, m, k, n, count, rate, wb, ab, ob, ex, conv, flag, w_fmt="bf16", a_fmt="bf16") -> GemmCost:
    """0.92 instance_sched "cta" (GPU reference hardware): the output is cut into thread-block tiles tm × tn (the
    guide's tile set; the library picks the fastest, optionally split-K), one tile per SM (engine) at a time →
    tile quantization (partial tiles do full work) and wave quantization (ceil(tiles / SMs) waves, the tail wave
    as long as a full one).  A tile costs max(MAC, operand feed) over its K slice on one SM (R × C MACs, 1 / E of the
    SRAM port) plus a non-overlapped epilogue writing its tm × tn outputs (fp32 partials under split-K, then one
    reduction pass).  Grouped / batched instances (count) share the waves.  No tuned constants."""
    E = ch.engines
    macs = ch.rows * ch.cols * rate
    port = ch.port_Bpc / E
    if ch.tile_io_Bpc > 0 and ch.fmt_ceiling:   # 0.135: the format ceiling is the main loop's rate, per tile
        fc = dict(ch.fmt_ceiling)
        cap = fc.get(f"gemm:{ex}", fc.get(ex))
        if cap:
            macs *= cap
    best = None
    for tm, tn in CTA_TILES:
        tiles = _cd(m, tm) * _cd(n, tn) * count
        for s in SPLIT_K:
            if s > 1 and (tiles * s > 2 * E or _cd(k, s) < 64):
                break                    # split-K only to fill an under-occupied GPU, K slices ≥ 64
            kk = _cd(k, s)
            mac_t = tm * tn * kk / macs
            feed_t = (tm * ab + tn * wb) * kk / port
            if ch.smem_Bpc_sm and CTA_SMEM_BW:
                # 0.196: both operands stream through the SM's shared memory once per K step (SS-mode MMA), at
                # smem_Bpc_sm — a tm × tn tile does tm·tn MACs per (tm·a + tn·b) bytes, so small tiles run SMEM-bound
                # (B200 bf16: 64×64 needs 256 B/clk for 4096 MAC/clk → 50 %, 128×256 needs 96 → MAC-bound)
                feed_t = max(feed_t, (tm * ab + tn * wb) * kk / ch.smem_Bpc_sm)
            epi = tm * tn * (ob if s == 1 else 4.0) / port
            once = 0.0
            if ch.tile_io_Bpc > 0:       # 0.135 opt-in: prologue stage + C writeback at the SM's DRAM share
                io = ch.tile_io_Bpc / E
                bk = min(kk, max(1, int(128 / max(ab, wb, 0.5))))
                t_epi = max(epi, tm * tn * (ob if s == 1 else 4.0) / io)
                t_pro = bk * (tm * ab + tn * wb) / io
                t_io = t_epi + t_pro
                if ch.tile_io == "occupancy" and ch.occ_smem_kib:
                    # 0.195: k resident CTAs per SM (cta_occupancy); one tile's I/O runs under the others' main loops,
                    # so a tile costs max(main, (main + I/O) / k) in steady state.  A single tile per SM (waves = 1)
                    # has nothing to overlap with.  k = 1 → "dram"; k → ∞ → no exposed I/O
                    kw = min(cta_occupancy(ch, tm, tn, ab, wb), _cd(tiles * s, E))
                    main = max(mac_t, feed_t)
                    epi = max(main, (main + t_io) / kw) - main
                    if CTA_OCC_FILL and kw > 1:
                        # 0.196: steady state only after the pipe fills — the k resident CTAs start together, so their
                        # k first prologues share the SM's DRAM share with no main loop to hide under, and the last
                        # epilogue drains alone: k·prologue + epilogue once per SM.  ≤ 0.195 hid it (2 waves of
                        # 128 × 128 at k = 2 lost all I/O)
                        once = kw * t_pro + t_epi
                elif ch.tile_io == "persistent":
                    # 0.138: persistent warp-specialised kernel (CUTLASS sm90 ping-pong): tile i's epilogue and tile
                    # i+1's prologue run under the neighbouring tile's main loop — exposed once per SM, and per tile
                    # only where the I/O outlasts the main loop
                    once, epi = t_io, max(0.0, t_io - max(mac_t, feed_t))
                else:
                    epi = t_io
            waves = _cd(tiles * s, E)
            red = (m * n * count * 4.0 * s + m * n * count * ob) / ch.port_Bpc if s > 1 else 0.0
            cyc = waves * (max(mac_t, feed_t) + epi) + once + red
            if best is None or cyc < best[0]:
                # mac share: MAC-bound tiles' compute (+ epilogue / reduction); feed = operand bytes / port (SRAM bytes)
                byt = tiles * s * (tm * ab + tn * wb) * kk
                best = (cyc, cyc if mac_t >= feed_t else 0.0, byt / ch.port_Bpc)
    cyc, mac_c, feed_c = best
    if ch.tile_io_Bpc > 0 and CTA_DRAM_FLOOR:
        # 0.196: under tile I/O the array term carries the kernel's DRAM traffic, so it is bounded below by the unique
        # operand + output bytes over the aggregate DRAM bandwidth (the roofline).  ≤ 0.195 the occupancy overlap
        # (main + I/O) / k let co-resident CTAs hide every tile's I/O under each other's main loops as if each SM had its
        # own DRAM channel — no aggregate bound (DeepSeek MLA absorbed bmm on H100: 8.1 µs vs 13 µs of bytes)
        act = (m * k * ab + m * n * ob) * count
        if CTA_L2_ACT and ch.sram_mib:
            # 0.197: the activation operand was just written by the previous kernel and the output is read by the next;
            # while they fit in L2 (write-back, sram_mib 「来源」 refhw) they never reach DRAM — the stage DRAM term
            # (memplan) already keeps them on chip (dram["act"] = 0).  Only the excess over L2 streams.  Weights are
            # streamed once per step (whole model ≫ L2) and stay in the floor.  Reuse across tiles is already
            # excluded: the floor counts unique bytes.
            act = max(0.0, act - ch.sram_mib * 2 ** 20)
        fl = (act + k * n * wb * count) / ch.tile_io_Bpc
        if fl > cyc:
            mac_c, cyc = 0.0, fl
    conv_elems = (k * n if flag & 1 else 0.0) + (m * k if flag & 2 else 0.0)
    xw, xa = _conv_extra(m, k, n, w_fmt, a_fmt, flag)
    return GemmCost(cyc, mac_c, feed_c, "cta", ex, conv,
                    conv_elems * count, (k * n * count) if flag & 1 else 0.0, (xw + xa) * count, xw * count)


def gemm_seconds(ch: Chip, org: str, m, k, n, **kw) -> float:
    c = gemm_cost(ch, org, m, k, n, **kw)
    return c.cycles / (ch.freq_ghz * 1e9 * ch.mac_eff)


def vector_seconds(ch: Chip, elem_ops: float) -> float:
    return elem_ops / (ch.lanes * ch.freq_ghz * 1e9)
