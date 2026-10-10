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

from .dtypes import FormatSupport, fmt as _fmt, gemm_exec
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
    conversion: str        # none | dequant_w | upcast_both
    convert_elems: float   # elements converted on the vector path
    convert_w: float = 0.0  # of which second-operand (weight) elements: once per pass, independent of tokens (0.63)

    @property
    def bound(self) -> str:
        return "feed" if self.feed_cycles > self.mac_cycles else "mac"


def _cd(a: float, b: float) -> int:
    return int(-(-a // b))


def _os(ch: Chip, m, k, n, r, wb, ab, ob):
    R, Ce = ch.rows, ch.c_eff
    mac = _cd(m, R) * _cd(n, Ce) * k / r
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


def _candidates(org: str) -> tuple[str, ...]:
    return {"os": ("os",), "ws_edge": ("ws_edge",), "ws_broad": ("ws_broad",),
            "os_vec": ("os", "gemv"), "reconf": ("os", "ws_edge", "ws_broad", "gemv")}[org]


def gemm_cost(ch: Chip, org: str, m: int, k: int, n: int, *, count: int = 1, w_fmt: str = "bf16",
              a_fmt: str = "bf16", w_bits: float | None = None) -> GemmCost:
    """Cycles for ``count`` identical GEMMs [m,k]×[k,n] under organisation ``org`` (memoised: pure function)."""
    return _gemm_cost(ch, org, m, k, n, count, w_fmt, a_fmt, w_bits)


@lru_cache(maxsize=1 << 18)
def _gemm_cost(ch: Chip, org: str, m: int, k: int, n: int, count: int, w_fmt: str, a_fmt: str,
               w_bits: float | None) -> GemmCost:
    if org not in ORGS:
        raise ValueError(f"unknown mapping {org!r}; one of {ORGS}")
    if m <= 0 or k <= 0 or n <= 0 or count <= 0:
        return GemmCost(0.0, 0.0, 0.0, "none", a_fmt, "none", 0.0)
    ex, rate, conv, flag = gemm_exec(w_fmt, a_fmt, ch.formats)
    wb = (w_bits if w_bits is not None else _fmt(w_fmt).bits) / 8.0
    ab = _fmt(a_fmt).bytes
    ob = max(2.0, ab)       # outputs leave in bf16 (fp32 when the activations are fp32)
    best = None
    for df in _candidates(org):
        if df == "os":
            mac, feed = _os(ch, m, k, n, rate, wb, ab, ob)
        elif df == "gemv":
            mac, feed = _gemv(ch, m, k, n, rate, wb, ab, ob)
        else:
            mac, feed = _ws(ch, m, k, n, rate, wb, ab, ob, broad=(df == "ws_broad"))
        fill_c = (ch.rows + ch.c_eff) / count if df.startswith("ws") else 0.0
        cyc = max(mac + fill_c, feed)
        if best is None or cyc < best[0]:
            best = (cyc, mac, feed, df)
    cyc, mac, feed, df = best
    fill = (ch.rows + ch.c_eff) if df.startswith("ws") else 0.0   # pipeline fill once per op (instances stream back-to-back)
    sched = "split" if ch.split_instances else ch.instance_sched
    if sched != "wide" and count > 1 and ch.engines > 1:      # 0.70 / 0.71: instances spread over core groups
        best_c = max(mac * count + fill, feed * count)
        alt = None
        for g in (_groups(ch.engines) if sched == "auto" else (1,)):
            par = ch.engines // g
            sub = replace(ch, engines=g, split_instances=False, instance_sched="wide",
                          sram_port_Bpc=ch.port_Bpc * g / ch.engines, acc_kib=ch.acc_kib * g / ch.engines)
            one = _gemm_cost(sub, org, m, k, n, 1, w_fmt, a_fmt, w_bits)
            waves = _cd(count, par)
            if one.cycles * waves < best_c:
                best_c, alt = one.cycles * waves, (one, waves)
        if alt is not None:
            one, waves = alt
            return GemmCost(one.cycles * waves, one.mac_cycles * waves, one.feed_cycles * waves, one.dataflow,
                            one.exec_fmt, one.conversion, one.convert_elems * count, one.convert_w * count)
    conv_elems = 0.0
    if flag & 1:
        conv_elems += k * n
    if flag & 2:
        conv_elems += m * k
    return GemmCost(max(mac * count + fill, feed * count), mac * count + fill, feed * count, df, ex, conv,
                    conv_elems * count, (k * n * count) if flag & 1 else 0.0)


def gemm_seconds(ch: Chip, org: str, m, k, n, **kw) -> float:
    c = gemm_cost(ch, org, m, k, n, **kw)
    return c.cycles / (ch.freq_ghz * 1e9 * ch.mac_eff)


def vector_seconds(ch: Chip, elem_ops: float) -> float:
    return elem_ops / (ch.lanes * ch.freq_ghz * 1e9)
