"""Evaluate one Scenario: per-stage op graphs → mapping → memory plan → schedule.

Everything goes through the per-rank IR; a single card is Layout().
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from .catalog import get_model
from .dtypes import fmt as _fmt
from .hardware import System
from .domain import ResolvedWorkload, resolve_workload
from .ir import (Op, Phase, Shard, _cdiv, build_rank_ops, embed_ops, full_io_ops, head_ops, layer_repeat,
                 mtp_ops)
from .mapping import gemm_cost, vector_seconds
from .memo import layer_groups, model_cache
from .memplan import RUNTIME_RESERVE, MemPlan, act_stream, plan, stage_storage, step_dram_bytes, touched
from .model import ModelSpec, with_formats
from .parallel import Layout, plan_stages, plan_stages_balanced
from .pipeline import TILING, pipeline_for, stored_bytes, te_layers, text_ops, vae_ops
from .scenario import Scenario
from .vision import expand_images, image_grid, vision_act_peak, vision_ops
from .ir import red_bytes, skew_from_load
from . import fabric
from .schedule import StageTime, collective_seconds, fabric_collective, p2p_tier, spec_expected_tokens


@dataclass
class StageResult:
    index: int
    layers: tuple[int, int]
    time: StageTime
    mem: MemPlan
    dram: dict
    convert_elems: float
    ops: list[Op] = field(default_factory=list, repr=False)
    flops_u: float = 0.0    # useful FLOPs of one rank (ops replicated on r ranks count 1/r) — request-FLOP KPI
    sram_bytes: float = 0.0  # bytes through the SRAM ↔ datapath port of one rank per tick (0.47.1 action counts)
    ideal_w: float = 0.0     # 0.63: array s at 100 % of the mean rank's useful work (Op.share) — energy MAC count
    vec_w: float = 0.0       # 0.63: vector s of the mean rank's token work (excl. weight conversion)
    conv_w: float = 0.0      # 0.63: vector s of weight conversion (once per pass on every busy rank)
    exp_acts: dict = field(default_factory=dict)  # 0.65: routed-expert part {sram, conv_w, link_frac} (idle EP ranks)


@dataclass
class Result:
    scenario: Scenario
    model: ModelSpec
    stages: list[StageResult]
    tick: float               # slowest stage time per micro-batch
    step: float               # one decode step (all micro-batches) or prefill latency
    tokens_per_step: float    # accepted tokens per sequence per step
    tpot: float               # s / output token / sequence (decode)
    ttft: float               # prefill latency (prefill phase)
    throughput: float         # tokens / s per replica (decode: output; prefill: prompt tokens)
    per_card: float
    fits: bool
    bound: str
    heaviest_stage: int
    warnings: list[str]
    microbatches: int
    latency: float = 0.0      # s per request: decode TPOT, prefill TTFT; video: one clip; protein: one batch
    workload: ResolvedWorkload | None = None
    pipeline: dict | None = None   # video (0.44): text-encoder / VAE-decode time, FLOP, storage (None = DiT only)
    moe_skew: float = 1.0          # effective EP load skew used (0.49)
    vision: dict | None = None     # VLM (0.62): images, image tokens, encoder time / FLOP / storage (None = text-only)

    @property
    def domain(self) -> str:
        return self.model.domain

    @property
    def slo_ok(self) -> bool:
        """Latency SLO of the scenario: LLM decode TPOT (prefill: none), video clip latency, protein batch latency."""
        sv, w = self.scenario.serving, self.scenario.workload
        if self.model.domain == "gen":
            return self.latency <= w.clip_slo_s
        if self.model.domain == "protein":
            return self.latency * 1e3 <= self._seq_slo_ms
        return self.tpot * 1e3 <= sv.tpot_slo_ms

    @property
    def _seq_slo_ms(self) -> float:
        w = self.scenario.workload
        return w.fold_slo_s * 1e3 if self.model.is_pair else w.seq_slo_ms

    def _tflop_per_request(self) -> float:
        """Useful FLOPs of one request over the whole replica: per-rank useful FLOPs × the TP·SP ranks of a stage,
        per sequence of the rank's micro-batch, × forward sequences per request × steps (0.45: was per rank)."""
        w, lay = self.workload, self.scenario.layout
        # 0.63: flops_u is the mean rank's useful work (Op.share) → × all TP·SP·DP ranks / sequences of the micro-batch
        b_mb = _cdiv(self.scenario.serving.batch * w.seqs_per_request, self.microbatches)
        per_seq = sum(st.flops_u for st in self.stages) * lay.tp * lay.sp * lay.dp / b_mb
        return per_seq * w.seqs_per_request * w.steps / 1e12

    def domain_summary(self) -> dict:
        """Domain KPIs of a non-autoregressive model (empty for LLMs)."""
        w = self.workload
        if w is None:
            return {}
        cards = self.scenario.layout.cards
        b = self.scenario.serving.batch
        heavy = self.stages[self.heaviest_stage]
        period = (self.pipeline or {}).get("period_s", self.latency)   # 0.47 overlap: steady-state request period
        d = {"unit": w.unit, "latency_s": self.latency, "requests_per_s": b / period, "period_s": period,
             "units_per_s": self.throughput, "units_per_s_card": self.per_card, "workload": w.info,
             "forward_ms": self.tick * max(self.microbatches, self.scenario.layout.pp) * 1e3,
             "seqs_per_forward": b * w.seqs_per_request, "act_GiB": heavy.mem.act_total / 2**30,
             "tflop_per_request": self._tflop_per_request()}
        if w.kind == "gen":
            pl = self.pipeline
            denoise = pl["denoise_s"] if pl else self.latency
            if pl:
                d["tflop_per_request"] += pl["tflop"] / b
                d["dit_tflop_per_request"] = d["tflop_per_request"] - pl["tflop"] / b
            d.update(pipeline=pl, denoise_s=denoise)
            d.update(clip_s=self.latency, s_per_frame=self.latency / w.units, step_ms=denoise / w.steps * 1e3,
                     frames_per_s_card=self.per_card, clips_per_hour_card=3600 * b / period / cards,
                     realtime_x=(w.info["video_s"] * b / self.latency) if w.info.get("video_s") else None,
                     slo_s=self.scenario.workload.clip_slo_s)
        else:
            d.update(batch_ms=self.latency * 1e3, seq_per_s_card=self.per_card,
                     residues_per_s_card=self.per_card * w.info["seq_len"], slo_ms=self._seq_slo_ms)
        return d

    def summary(self) -> dict:
        s = self.scenario
        heavy = self.stages[self.heaviest_stage]
        return {
            "model": self.model.id, "layout": s.layout.label, "cards": s.layout.cards, "mapping": s.mapping,
            "phase": s.serving.phase, "batch": s.serving.batch,
            "tpot_ms": self.tpot * 1e3, "ttft_ms": self.ttft * 1e3, "step_ms": self.step * 1e3,
            "tok_s": self.throughput, "tok_s_card": self.per_card, "fits": self.fits, "bound": self.bound,
            "heaviest_stage": self.heaviest_stage, "dram_need_GiB": heavy.mem.dram_need / 2**30,
            "residency": heavy.mem.residency, "microbatches": self.microbatches, "warnings": self.warnings,
            "array_util": heavy.time.array_util, "domain": self.model.domain, "slo_ok": self.slo_ok,
            "latency_ms": self.latency * 1e3, **({"gen": self.domain_summary()} if self.workload else {}),
            **({"moe_skew": self.moe_skew} if self.moe_skew != 1.0 else {}),
            **({"vision": {k: v for k, v in self.vision.items() if k != "acts"}} if self.vision else {}),
        }


def _op_seconds(op: Op, sys: System, org: str, model: ModelSpec) -> tuple:
    """(array s, mac-bound share s, feed-bound share s, vector s, convert elems, ideal s, SRAM-port bytes,
    vector s of per-pass weight conversion)."""
    ch = sys.chip
    f = ch.freq_ghz * 1e9 * ch.mac_eff
    if op.kind == "gemm":
        c = gemm_cost(ch, org, op.m, op.k, op.n, count=op.count, w_fmt=op.w_fmt, a_fmt=op.a_fmt, w_bits=op.w_bits)
    elif op.kind == "attn" and op.orient:
        # full-sequence (DiT / encoder) attention: both operands are activations in the activation dtype, and a
        # flash kernel may compute O = P·V or Oᵀ = Vᵀ·Pᵀ — take the cheaper orientation (PV has N = head_dim,
        # which leaves most of a wide array idle).  LLM prefill / decode attention keeps the 0.40 orientation.
        af = model.act_fmt
        c = min((gemm_cost(ch, org, a, op.k, b, count=op.count, w_fmt=af, a_fmt=af)
                 for a, b in ((op.m, op.n), (op.n, op.m))), key=lambda x: x.cycles)
    elif op.kind == "attn":
        c = gemm_cost(ch, org, op.m, op.k, op.n, count=op.count, w_fmt=op.w_fmt or model.kv_fmt, a_fmt=op.a_fmt)
    else:
        return 0.0, 0.0, 0.0, vector_seconds(ch, op.vec), 0.0, 0.0, 0.0, 0.0
    t = c.cycles * op.causal / f
    vec = vector_seconds(ch, op.vec + c.convert_elems * 2.0)   # 2 element-ops per converted element 「假设」
    mac_share = t if c.bound == "mac" else 0.0
    rate = ch.formats.rate(c.exec_fmt) or 1.0
    ideal = op.flops / 2.0 / (ch.macs * rate * ch.freq_ghz * 1e9)   # 100 % array utilisation
    vw = vector_seconds(ch, c.convert_w * 2.0) if op.kind == "gemm" else 0.0   # weight dequant: per pass, not per token
    return t, mac_share, t - mac_share, vec, c.convert_elems, ideal, c.feed_cycles * op.causal * ch.port_Bpc, vw


def stage_ops(model: ModelSpec, first: int, last: int, has_embed: bool, has_head: bool, ph: Phase, sh: Shard,
              spec_k: int) -> list[Op]:
    """Full op list of one rank of a stage (debug / tests / UI drill-down)."""
    ops: list[Op] = []
    if has_embed:
        ops += embed_ops(model, ph, sh)
    for li in range(first, last):
        ops += build_rank_ops(model, li, ph, sh)
    ops += _tail_ops(model, has_head, ph, sh, spec_k)
    return ops


def _tail_ops(model, has_head, ph, sh, spec_k):
    ops: list[Op] = []
    if has_head:
        ops += head_ops(model, ph, sh)
        if spec_k and model.mtp_layers and ph.kind == "decode":
            dph = Phase("decode", ph.batch, 1, ph.ctx, skew=ph.skew)
            for d in range(spec_k):
                ops += mtp_ops(model, dph, sh, depth=d)
    return ops


_SUM_KEYS = ("t_arr", "t_mac", "t_feed", "t_vec", "conv", "t_ideal", "flops", "link_bw", "sync", "link_bytes",
             "hot", "exp", "kv_read", "kv_write", "state", "lookup", "max_act", "act", "w_extra", "max_act_tot",
             "flops_u", "sram", "d2d_bytes", "net_bytes", "busy_d2d", "busy_link", "busy_net", "bw_max",
             "t_ideal_w", "t_vec_w", "t_conv_w", "sram_e", "conv_w_e", "link_bytes_e")
_MAX_KEYS = ("max_act", "max_act_tot", "bw_max")
_EP_COMM = ("moe_dispatch", "moe_combine", "expert_allreduce")
_TIERS = ("d2d", "link", "net")


_COLL_MEMO: dict = {}       # 0.61: System → {(kind, bytes, group, stride): collective_full(...)} (pure function)
_MEMO_CAP = 200_000


def _memos(sys: System, org: str, model: ModelSpec) -> tuple:
    """0.61: per-op cost memo — the same per-rank op shapes recur across layouts / batches of a search (e.g. B/dp
    equal at the doubling points): ~29× hits on _op_seconds, ~9× on the fabric collectives at 1024 cards.  Pure
    functions of (op, chip, mapping, model formats) / (collective, System); the collective memo is off while
    fabric_report logs collectives.  Looked up once per evaluate."""
    try:
        om = model_cache(model).setdefault("op_secs", {})
    except TypeError:           # a shim namespace (no weak reference): no memo across calls
        return {}, None
    if len(om) > 64:
        om.clear()
    om = om.setdefault((sys.chip, org), {})
    if len(om) > _MEMO_CAP:
        om.clear()
    cm = None
    if sys.fabric.enabled and fabric.log() is None:
        if len(_COLL_MEMO) > 256:
            _COLL_MEMO.clear()
        cm = _COLL_MEMO.setdefault(sys, {})
        if len(cm) > _MEMO_CAP:
            cm.clear()
    return om, cm


def _exp_acts(agg: dict) -> dict:
    """0.65: the routed-expert part of a stage step — SRAM-port bytes and weight-dequant vector seconds of the
    expert GEMMs, and the EP dispatch / combine / expert all-reduce share of the stage's collective bytes."""
    lb = agg["link_bytes"]
    return {"sram": agg["sram_e"], "conv_w": agg["conv_w_e"], "link_frac": agg["link_bytes_e"] / lb if lb > 0 else 0.0}


def _sum_ops(ops: list[Op], sys: System, org: str, model: ModelSpec, memos: tuple | None = None) -> dict:
    d = dict.fromkeys(_SUM_KEYS, 0.0)
    om, cm = memos if memos is not None else _memos(sys, org, model)
    for o in ops:
        if o.kind == "comm":
            if sys.fabric.enabled:      # 0.60: per-tier busy seconds (overlap = ports) + candidates (auto_overlap)
                if cm is not None:
                    ck = (o.comm_kind, o.comm_bytes, o.comm_group, o.comm_stride)
                    cr = cm.get(ck)
                    if cr is None:
                        cr = cm[ck] = fabric.collective_full(o.comm_kind, o.comm_bytes, o.comm_group, sys,
                                                             o.comm_stride)
                    bw, a, fd, fn, busy, full = cr
                else:
                    bw, a, fd, fn, busy, full = fabric.collective_full(o.comm_kind, o.comm_bytes, o.comm_group, sys,
                                                                       o.comm_stride)
                for t, x in busy.items():
                    d["busy_" + t] += x
                d["bw_max"] = max(d["bw_max"], bw)
                if full is not None:
                    c = d.setdefault("_cands", {})
                    c[full[0]] = [c[full[0]][0] + 1 if full[0] in c else 1, full]
            else:
                bw, a, fd, fn = _comm(sys, o.comm_kind, o.comm_bytes, o.comm_group, o.comm_stride)
            d["link_bw"] += bw; d["sync"] += a; d["link_bytes"] += o.comm_bytes; d["d2d_bytes"] += o.comm_bytes * fd
            if o.name in _EP_COMM:            # 0.65: what a DP rank without sequences still sends / receives under EP
                d["link_bytes_e"] += o.comm_bytes
            d["net_bytes"] += o.comm_bytes * fn
            continue
        r_ = om.get(o)
        if r_ is None:
            r_ = om[o] = _op_seconds(o, sys, org, model)
        a_, ma, fe, v, ce, idl, sb, vw = r_
        d["t_arr"] += a_; d["t_mac"] += ma; d["t_feed"] += fe; d["t_vec"] += v; d["conv"] += ce
        d["t_ideal"] += idl; d["sram"] += sb
        d["t_ideal_w"] += idl * o.share; d["t_vec_w"] += (v - vw) * o.share   # 0.63: mean-rank work (energy, KPIs)
        d["t_conv_w"] += vw * o.wshare        # 0.64: mean-rank weight elements (uneven TP / EP splits)
        if o.role == "expert":                # 0.65: routed-expert share (idle EP ranks, energy action counts)
            d["sram_e"] += sb
            d["conv_w_e"] += vw * o.wshare
        d["flops"] += o.flops
        d["flops_u"] += o.flops * o.share / max(1, o.replicated)
        d["max_act"] = max(d["max_act"], o.act_bytes / max(o.count, 1))
        if o.stream:
            a, wx = act_stream(o, _fmt(model.act_fmt).bytes, sys.chip.sram_bytes)
            d["act"] += a; d["w_extra"] += wx
            ab = _fmt(model.act_fmt).bytes
            tot = (((o.in_elems or o.m * o.k * o.count) + o.m * o.n * o.count) * ab if o.kind == "gemm"
                   else o.act_bytes if o.bmm
                   else 2.0 * o.count * (o.m + o.n) * o.k * ab)
            d["max_act_tot"] = max(d["max_act_tot"], tot)
    for k, v in touched(ops).items():
        d[k] += v
    return d


def _moe_skew(m: ModelSpec, sv, lay: Layout, warnings: list[str]) -> float:
    """Effective EP load skew of the scenario (0.49): ``serving.moe_skew`` or derived from ``moe_expert_load``."""
    if sv.moe_skew == 1.0 and not sv.moe_expert_load:
        return 1.0
    if not m.is_moe or lay.ep <= 1:
        warnings.append("MoE 负载倾斜已忽略：" + ("非 MoE 模型" if not m.is_moe else "EP = 1（专家全在每个 rank 上，无跨 rank 不均衡）"))
        return 1.0
    if sv.moe_expert_load:
        n = next(L.ffn.n_experts for L in m.layers if L.ffn.kind == "moe")
        if len(sv.moe_expert_load) != n:
            raise ValueError(f"serving.moe_expert_load needs {n} entries (one per routed expert), got "
                             f"{len(sv.moe_expert_load)}")
        return skew_from_load(sv.moe_expert_load, lay.ep)
    return float(sv.moe_skew)


def _comm(sys: System, kind: str, payload: float, group: int, stride: int = 1,
          members: tuple[int | None, int | None] | None = None) -> tuple[float, float, float, float]:
    """One collective on the system's three-tier fabric (0.50; 0.48 two-tier when node_cards = 0):
    (bandwidth s, α s, D2D share, network share of the bytes).  ``members``: (ranks inside one package, inside one
    node) when the group is not a uniform stride (None entries = derive from the stride)."""
    kp, kn = members if members is not None else (None, None)
    if sys.fabric.enabled:      # 0.59 topology-aware collectives 「假设」
        return fabric.collective(kind, payload, group, sys, stride, k_pkg=kp, k_node=kn)
    if sys.package_cards <= 1 and sys.node_cards <= 0:
        return (*collective_seconds(kind, payload, group, sys.link), 0.0, 0.0)
    return fabric_collective(kind, payload, group, sys.link, sys.d2d if sys.package_cards > 1 else None,
                             sys.package_cards, sys.net if sys.node_cards > 0 else None, sys.node_cards, stride,
                             k_pkg=kp, k_node=kn)


def _p2p(sys: System, payload: float, stage: int, stage_cards: int) -> tuple[float, float, float, float]:
    return _p2p_full(sys, payload, stage, stage_cards)[:4]


def _p2p_full(sys: System, payload: float, stage: int, stage_cards: int) -> tuple[float, float, float, float, str]:
    """PP hand-off of ``stage`` → next: (bw s, α s, D2D share, network share of the bytes, tier of the slowest pair).
    0.63 (external review): every rank j of the stage sends its slice to rank j of the next stage at once (cards
    numbered TP, SP, DP, PP) — each pair on its own tier (same package → D2D, same node → scale-up link, else the
    network, oversubscribed when the two nodes sit under different leaves / pods); the hand-off ends with the slowest
    pair.  Was one representative pair (the stage's last card → the next stage's first): PP2·TP64 on 8-card nodes,
    4:1 leaves — 1 ms instead of 4 ms (rank 63 → 64 share a leaf, rank 0 → 64 do not)."""
    pk, nd = sys.package_cards, sys.node_cards
    keys: dict = {}
    for j in range(max(1, stage_cards)):
        a = stage * stage_cards + j
        b = a + stage_cards
        if pk > 1 and a // pk == b // pk:
            t = "d2d"
        elif nd <= 0 or a // nd == b // nd:
            t = "link"
        else:
            t = "net"
        # pairs with the same tier and (node, node) positions cost the same; keep one representative per class
        k = (t, a // nd if (t == "net" and nd > 0) else 0, b // nd if (t == "net" and nd > 0) else 0)
        if k not in keys:
            keys[k] = [a, b, 0]
        keys[k][2] += 1
    n_pairs = sum(v[2] for v in keys.values())
    fd = sum(v[2] for k, v in keys.items() if k[0] == "d2d") / n_pairs
    fn = sum(v[2] for k, v in keys.items() if k[0] == "net") / n_pairs
    best = None
    for (t, _, _), (a, b, _) in keys.items():
        ln = {"d2d": sys.d2d, "link": sys.link, "net": sys.net}[t]
        if sys.fabric.enabled and payload > 0:      # 0.59: oversubscribed leaf uplinks + per-step hop latency
            bw, al, pr = fabric.p2p_alpha(fabric.p2p(payload, t, ln, sys, a, b, stage_cards), t, ln, sys, a, b,
                                          with_proto=True)
        else:
            bw, al = collective_seconds("p2p", payload, 2, ln)
            pr = None
        if best is None or bw + al > best[0] + best[1]:
            best = (bw, al, t, pr)
    bw, al, t, pr = best
    if sys.fabric.enabled and payload > 0 and fabric.log() is not None:
        nm = "p2p" if pr is None else f"p2p/{pr}"
        e = fabric.log().setdefault(("p2p", stage, round(payload)), {
            "kind": "p2p", "group": 2, "bytes": payload, "count": 0, "levels": [[t, 2]], "algo": nm,
            "top": "", "cands": {nm: [bw, al]}, "stage": stage})
        e["count"] += 1
    return bw, al, fd, fn, t


def _slc_time(sys: System, dram: dict) -> float:
    s = dram.get("slc", 0.0)
    return s / (sys.chip.slc_GBps * 1e9) if s > 0 else 0.0


def _links(scn: Scenario) -> tuple:
    if scn.fabric.enabled:      # 0.59: topology factors depend on the replica's cards (scale-up domain on one node)
        return (scn.link, scn.d2d_link if scn.package_eff > 1 else None, scn.package_eff,
                scn.net if scn.node_cards > 0 else None, scn.node_cards, scn.fabric, scn.layout.cards)
    if scn.package_eff <= 1 and scn.node_cards <= 0:
        return scn.link
    return (scn.link, scn.d2d_link if scn.package_eff > 1 else None, scn.package_eff,
            scn.net if scn.node_cards > 0 else None, scn.node_cards)


def _acc(a: dict, b: dict, n: int = 1) -> None:
    for k in _SUM_KEYS:
        if k in _MAX_KEYS:
            a[k] = max(a[k], b[k])
        else:
            a[k] += b[k] * n
    if "_cands" in b:           # 0.60 auto_overlap: collective classes with their counts (new lists: b may be memoised)
        c = a.setdefault("_cands", {})
        for key, (cnt, full) in b["_cands"].items():
            c[key] = [c[key][0] + cnt * n if key in c else cnt * n, full]


def _p2p_tier_of(fd: float, fn: float) -> str:
    return "net" if fn >= 1.0 else "d2d" if fd >= 1.0 else "link"


def _stage_link(sys: System, agg: dict, link_bw: float, sync: float, d2d_b: float, net_b: float, extra: list,
                window: float, stage: int = 0) -> tuple[float, float, float, float]:
    """0.60 (fabric on): stage link seconds under ``fabric.overlap`` and the ``auto_overlap`` re-pick.
    extra = [(bw, tier)] of the stage's PP / FSDP transfers already in link_bw.  Returns (link, sync, d2d, net)."""
    fab = sys.fabric
    busy = {t: agg["busy_" + t] for t in _TIERS}
    bw_max = agg["bw_max"]
    for bw, t in extra:
        busy[t] += bw
        bw_max = max(bw_max, bw)
    cl = list(agg.get("_cands", {}).values())
    ports = fab.overlap == "ports"

    def link_of(lb: float, bz: dict, bm: float) -> float:
        return max(max(bz.values()), bm) if ports else lb

    choice = None
    if fab.algo == "auto_overlap" and cl:
        base_lb, base_sync, base_bz = link_bw, sync, dict(busy)
        for cnt, (key, pay, dflt, cands) in cl:     # strip the default picks
            bw, a, bz, _, _ = cands[dflt]
            base_lb -= cnt * bw
            base_sync -= cnt * a
            for t, x in bz.items():
                base_bz[t] -= cnt * x
        names = [sorted(f[3]) for _, f in cl]

        def cost(sel):
            lb, sy, bz, bm = base_lb, base_sync, dict(base_bz), bw_max
            for (cnt, (key, pay, dflt, cands)), nm in zip(cl, sel):
                bw, a, b_, _, _ = cands[nm]
                lb += cnt * bw
                sy += cnt * a
                for t, x in b_.items():
                    bz[t] += cnt * x
                bm = max(bm, bw)
            return max(window, link_of(lb, bz, bm)) + sy, lb, sy, bz, bm

        dsel = [f[2] for _, f in cl]
        n_comb = math.prod(len(x) for x in names)
        if n_comb <= 729:
            best = min(itertools.product(*names), key=lambda sel: (cost(sel)[0], sel != tuple(dsel)))
        else:                   # coordinate descent from the default picks
            best = list(dsel)
            for _ in range(3):
                for i in range(len(best)):
                    best[i] = min(names[i], key=lambda nm: (cost(best[:i] + [nm] + best[i + 1:])[0], nm != best[i]))
            best = tuple(best)
        _, link_bw, sync, busy, bw_max = cost(best)
        for (cnt, (key, pay, dflt, cands)), nm in zip(cl, best):
            if nm != dflt:
                _, _, _, fd0, fn0 = cands[dflt]
                _, _, _, fd1, fn1 = cands[nm]
                d2d_b += cnt * pay * (fd1 - fd0)
                net_b += cnt * pay * (fn1 - fn0)
        choice = {key: nm for (cnt, (key, *_)), nm in zip(cl, best)}
    link = link_of(link_bw, busy, bw_max)
    if fabric.log() is not None:
        if choice:
            for key, nm in choice.items():
                if key in fabric.log():
                    fabric.log()[key]["algo"] = nm
        fabric.log().setdefault("_stages", []).append(
            {"stage": stage, "link_sum_us": link_bw * 1e6, "link_ports_us": max(max(busy.values()), bw_max) * 1e6,
             "busy_us": {t: v * 1e6 for t, v in busy.items()}, "window_us": window * 1e6, "sync_us": sync * 1e6,
             "link_us": link * 1e6})
    return link, sync, d2d_b, net_b



_DRAM_KEYS = ("hot", "exp", "kv_read", "kv_write", "state", "lookup", "act", "w_extra")


def _plan_by_cost(scn, m, sys, pp, groups, ph, sh, mm, ops_memo, store_memo, ctx, n_mtp, b_rank, first_ops, last_ops,
                  p2p_payload, repeat: bool = False):
    """0.62 PP split (``scn.pp_split``): "cost" (default) balances the per-stage time of the stage rule over layers,
    embedding / io-pre (stage 0), head / io-post / MTP (last stage) and the PP hand-off (core/parallel.py
    ``plan_stages_balanced``); "layers" = equal layer counts (≤ 0.61).  Returns (stages, info) -- info None when the
    equal-count split is used outright."""
    lc = plan_stages(m.n_layers, pp)
    if pp == 1 or scn.pp_split == "layers":
        return lc, None
    log = fabric.log(); fabric.set_log(None)         # the proxy pass must not count collectives for fabric_report
    try:
        bw = sys.dram_GBps * 1e9

        def vec(a, k=1.0):
            return (a["t_arr"] * k, a["t_vec"] * k, sum(a[x] for x in _DRAM_KEYS) * k / bw, a["link_bw"] * k,
                    a["sync"] * k)

        per_g: dict = {}
        costs = []
        for li in range(m.n_layers):
            g = groups[li]
            if g not in per_g:
                okey = (g, ph, sh, scn.mapping, scn.chip, _links(scn))
                hit = ops_memo.get(okey)
                if hit is None:
                    hit = ops_memo[okey] = _sum_ops(build_rank_ops(m, li, ph, sh), sys, scn.mapping, m, mm)
                per_g[g] = vec(hit, layer_repeat(m.layers[li], ph) if repeat else 1)
            costs.append(per_g[g])
        first = vec(_sum_ops(first_ops, sys, scn.mapping, m, mm)) if first_ops else (0.0,) * 5
        last = vec(_sum_ops(last_ops, sys, scn.mapping, m, mm)) if last_ops else (0.0,) * 5
        t_p, a_p, _, _ = _p2p(sys, p2p_payload, 0, scn.layout.cards // pp)
    finally:
        fabric.set_log(log)

    def static(st_):
        return st_.weights + (st_.kv_per_seq + st_.idx_per_seq + st_.state_per_seq) * b_rank

    def stor(first_, last_, e, h, nm):
        key = (first_, last_, e, h, sh, ctx, nm)
        x = store_memo.get(key)
        if x is None:
            x = store_memo[key] = stage_storage(m, first_, last_, e, h, sh, ctx, nm)
        return x

    L = m.n_layers
    mem_g: dict = {}
    mem = []
    for li in range(L):
        g = groups[li]
        if g not in mem_g:
            mem_g[g] = static(stor(li, li + 1, False, False, 0))
        mem.append(mem_g[g])
    mem_first = static(stor(0, 1, True, False, 0)) - static(stor(0, 1, False, False, 0))
    mem_last = static(stor(L - 1, L, False, True, n_mtp)) - static(stor(L - 1, L, False, False, 0))

    def peak(plan_):
        return max(sum(mem[st.first:st.last]) + (mem_first if st.has_embed else 0.0) + (mem_last if st.has_head else 0.0)
                   for st in plan_)

    lc_peak = peak(lc)
    cands = plan_stages_balanced(costs, pp, first, last, (0.0, 0.0, 0.0, t_p, a_p), mem, mem_first, mem_last,
                                 max(lc_peak, sys.dram_bytes), alternatives=True)
    if not isinstance(cands, tuple):
        return lc, None
    return cands, {"lc_peak": lc_peak}


def _pick_split(run, cands: tuple, lc) -> list:
    """The proxy proposed splits other than equal counts: evaluate them and the equal-count split with the full model
    and keep the fastest that fits (equal counts win ties; the proxy ignores SRAM / SLC residency, activations and
    fabric effects).  fabric_report's collective log keeps the chosen run's entries only."""
    log = fabric.log()
    base = copy.deepcopy(log) if log is not None else None
    runs = []
    for plan_ in (lc, *cands):
        if log is not None:
            log.clear(); log.update(copy.deepcopy(base))
        st = run(plan_)
        runs.append((st, copy.deepcopy(log) if log is not None else None))

    def key(i):
        st = runs[i][0]
        return (not all(s.mem.fits for s in st), max(s.time.total for s in st), i)
    best = min(range(len(runs)), key=key)
    if log is not None:
        log.clear(); log.update(runs[best][1])
    return runs[best][0]


def _pp_imbalance_warn(stages, tick: float, warnings: list, split: str = "cost") -> None:
    """0.61.4: flag a lopsided pipeline (0.62: by default the split is already cost-balanced, so this mostly fires for
    pp_split = "layers" or when whole layers are too coarse to balance)."""
    if len(stages) < 2 or tick <= 0:
        return
    lo = min(s.time.total for s in stages)
    if tick > 1.5 * lo:
        if split == "layers":
            warnings.append(f"流水级按层数均分（pp_split = layers，不按代价平衡）：最慢级 / 最快级 = {tick / max(lo, 1e-30):.1f}×，"
                            "节拍取最慢级——异构层栈（蛋白质 trunk / 扩散、首层稠密等）的 PP 结果偏悲观；默认 cost 按代价平衡")
        else:
            warnings.append(f"流水级已按代价平衡，仍不均：最慢级 / 最快级 = {tick / max(lo, 1e-30):.1f}×（整层粒度或容量上限所限），"
                            "节拍取最慢级")

def _streams(m: ModelSpec) -> int:
    """Residual streams passed between pipeline stages (mHC: hc_mult streams of width hidden)."""
    return max((l.hc for l in m.layers), default=0) or 1


def evaluate(scn: Scenario, model: ModelSpec | None = None) -> Result:
    m = model or get_model(scn.model)
    if scn.formats_override:
        wi = model_cache(m).setdefault("what_if", {})
        key = tuple(scn.formats_override)
        if key not in wi:
            wi[key] = with_formats(m, dict(scn.formats_override))
        m = wi[key]
    warnings: list[str] = []
    if scn.serving.images and not scn.serving.image_tokens:
        if m.vision is None:
            warnings.append("serving.images 已忽略：该模型没有视觉编码器（只有 VLM 计图像）")
        else:
            scn = expand_images(scn, m)     # 0.62: image positions join prompt / ctx (KV, prefill length)
    sv, lay = scn.serving, scn.layout
    if not lay.valid_for(m.is_moe, full=not m.kv_cache, pair=m.is_pair):
        raise ValueError(f"layout {lay.label} invalid for {'MoE' if m.is_moe else 'dense'} model "
                         + ("(structure models: tp = 1 — TP of the pair stack not modelled; use DAP (sp) / DP / PP)"
                            if m.is_pair else
                            "(video / protein models: ep = etp = 1)" if not m.kv_cache else
                            "(MoE needs ep·etp = tp·dp; dense needs dp=ep=etp=1; sp only for video / protein models)"))
    if lay.pp > m.n_layers:
        raise ValueError("pp exceeds layer count")
    sys = System(scn.chip, scn.mem_id, scn.mem_eff, scn.link, scn.d2d_link, scn.package_eff, scn.net, scn.node_cards,
                 scn.fabric, lay.cards)
    if scn.chip.slc_bytes and scn.chip.slc_bytes >= sys.dram_bytes:
        warnings.append("SLC 容量不小于每卡 DRAM 容量——不现实的设计点（SLC 不增加容量，按包含式缓存计）")
    if sys.package_cards > 1 and lay.cards > sys.package_cards and lay.cards % sys.package_cards:
        warnings.append(f"package_cards = {sys.package_cards} 不整除总卡数 {lay.cards}：末尾封装未满，"
                        "按每个通信组的整除部分近似分层")
    if scn.package_cards > 1 and not scn.d2d_enabled:
        warnings.append(f"package_cards = {scn.package_cards} 未生效：D2D 关闭（单片大 die，每卡一个封装）——"
                        "需 d2d_enabled = true 才按芯粒封装分层")
    if scn.d2d_enabled and scn.package_cards == 1:
        warnings.append("d2d_enabled = true 但 package_cards = 1：每个封装只有一个 die，D2D 层无流量")
    if sys.node_cards > 0 and lay.cards > sys.node_cards and lay.cards % sys.node_cards:
        warnings.append(f"node_cards = {sys.node_cards} 不整除总卡数 {lay.cards}：末尾节点未满，"
                        "按每个通信组的整除部分近似分层")
    if not m.kv_cache:
        return _evaluate_full(scn, m, sys, warnings)
    spec_k = sv.spec_k if (sv.phase == "decode" and m.mtp_layers) else 0
    need_ctx = sv.ctx + 1 + spec_k if sv.phase == "decode" else sv.prompt
    if m.max_ctx and need_ctx > m.max_ctx:    # 0.61.4: evaluated as asked, but say the release does not cover it
        warnings.append(f"上下文 {need_ctx} 超过发布 config 的 max_position_embeddings {m.max_ctx}"
                        "（需要 RoPE 外推 / YaRN 等，发布未必支持；KV 与注意力仍按所给长度计）")
    if sv.spec_k and not m.mtp_layers and sv.phase == "decode":
        warnings.append("spec_k 已忽略：该模型没有 MTP 模块（独立草稿模型的投机解码未建模）")
    pp = lay.pp
    skew = _moe_skew(m, sv, lay, warnings)
    if sv.phase == "decode":
        mb = sv.microbatches or min(pp, sv.batch)
        mb = max(1, min(mb, sv.batch))
        q = 1 + spec_k
        ph = Phase("decode", _cdiv(sv.batch, mb), q, sv.ctx, skew=skew)
        ctx_cap = sv.ctx + q
    else:
        mb = sv.microbatches or min(pp, sv.batch)
        mb = max(1, min(mb, sv.batch))
        ph = Phase("prefill", _cdiv(sv.batch, mb), sv.prompt - sv.prefix_cached, sv.prefix_cached, skew=skew)
        ctx_cap = sv.prompt
    sh = lay.shard
    stages = []
    n_mtp = min(spec_k, len(m.mtp_layers)) if spec_k else 0
    b_rank = _cdiv(sv.batch, lay.dp)          # sequences whose KV lives on one rank (all micro-batches)
    cache: dict = {}
    groups = layer_groups(m)
    store_memo = model_cache(m).setdefault("stage_storage", {})
    ops_memo = model_cache(m).setdefault("layer_sums", {})
    if len(ops_memo) > 200_000:
        ops_memo.clear()
    if fabric.log() is not None:     # fabric_report: run every collective (no layer memo)
        ops_memo = {}
    mm = _memos(sys, scn.mapping, m)
    def _run(plan_):
        stages = []
        for st in plan_:
            agg = dict.fromkeys(_SUM_KEYS, 0.0)
            if st.has_embed:
                _acc(agg, _sum_ops(embed_ops(m, ph, sh), sys, scn.mapping, m, mm))
            counts: dict = {}
            for li in range(st.first, st.last):
                g = groups[li]
                first_li, n = counts.get(g, (li, 0))
                counts[g] = (first_li, n + 1)
            if fabric.log() is not None:     # count every stage's collectives (ops_memo is a private dict here)
                cache.clear()
                ops_memo.clear()
            for g, (li, n) in counts.items():
                if g not in cache:
                    okey = (g, ph, sh, scn.mapping, scn.chip, _links(scn))
                    cache[g] = ops_memo.get(okey)
                    if cache[g] is None:
                        fabric.set_mult(n)
                        cache[g] = ops_memo[okey] = _sum_ops(build_rank_ops(m, li, ph, sh), sys, scn.mapping, m, mm)
                        fabric.set_mult(1)
                _acc(agg, cache[g], n)
            tail = _tail_ops(m, st.has_head, ph, sh, spec_k)
            if tail:                        # 0.61: non-head stages have no tail ops (summing zeros is a no-op)
                _acc(agg, _sum_ops(tail, sys, scn.mapping, m, mm))
            skey = (st.first, st.last, st.has_embed, st.has_head, sh, ctx_cap, n_mtp)
            store = store_memo.get(skey)
            if store is None:
                store = store_memo[skey] = stage_storage(m, st.first, st.last, st.has_embed, st.has_head, sh, ctx_cap,
                                                         n_mtp)
            # 0.63: activations streamed (stream=True: staging ≤ the SRAM's activation half, spills in agg["act"]);
            # a working set beyond the runtime reserve needs its excess in DRAM 「假设」
            mp = plan(store, b_rank, sys.chip.sram_bytes, sys.dram_bytes, agg["max_act"],
                      act_need=max(0.0, agg["max_act_tot"] - RUNTIME_RESERVE), stream=True,
                      slc_bytes=sys.chip.slc_bytes, slc_policy=sys.chip.slc_policy)
            dram = step_dram_bytes(mp, store, agg)
            link_bw, sync, link_bytes, d2d_b = agg["link_bw"], agg["sync"], agg["link_bytes"], agg["d2d_bytes"]
            net_b = agg["net_bytes"]
            extra = []
            if pp > 1 and not st.has_head:
                act = ph.batch * ph.q * m.hidden * red_bytes(m) / lay.dp * _streams(m)   # 0.61.1: ≥ bf16; × mHC streams
                bw, a, fd, fn, tier = _p2p_full(sys, act, st.index, lay.cards // pp)
                link_bw += bw; sync += a; link_bytes += act; d2d_b += act * fd; net_b += act * fn
                extra.append((bw, tier))
            t_dram = dram["total"] / (sys.dram_GBps * 1e9)
            if sys.fabric.enabled:
                link_bw, sync, d2d_b, net_b = _stage_link(
                    sys, agg, link_bw, sync, d2d_b, net_b, extra,
                    max(agg["t_arr"], agg["t_vec"], t_dram, _slc_time(sys, dram)), st.index)
            stt = StageTime(agg["t_arr"], agg["t_mac"], agg["t_feed"], agg["t_vec"], t_dram, link_bw, sync,
                            dram["total"], link_bytes, agg["flops"], agg["t_ideal"], _slc_time(sys, dram),
                            dram.get("slc", 0.0), d2d_b, net_b)
            stages.append(StageResult(st.index, (st.first, st.last), stt, mp, dram, agg["conv"], flops_u=agg["flops_u"],
                                      sram_bytes=agg["sram"], ideal_w=agg["t_ideal_w"], vec_w=agg["t_vec_w"],
                                      conv_w=agg["t_conv_w"], exp_acts=_exp_acts(agg)))
        return stages

    plan_, pinfo = _plan_by_cost(scn, m, sys, pp, groups, ph, sh, mm, ops_memo, store_memo, ctx_cap, n_mtp, b_rank,
                                 embed_ops(m, ph, sh), _tail_ops(m, True, ph, sh, spec_k),
                                 ph.batch * ph.q * m.hidden * red_bytes(m) / lay.dp * _streams(m))
    stages = _pick_split(_run, plan_, plan_stages(m.n_layers, pp)) if pinfo else _run(plan_)
    vis = _vision_cost(scn, m, sys, stages) if (m.vision is not None and sv.images and sv.image_tokens) else None
    heavy = max(range(len(stages)), key=lambda i: stages[i].time.total)
    tick = stages[heavy].time.total
    _pp_imbalance_warn(stages, tick, warnings, scn.pp_split)
    cap_heavy = max(range(len(stages)), key=lambda i: stages[i].mem.dram_need)
    fits = all(s.mem.fits for s in stages)
    if not fits:
        warnings.append(f"容量不足：流水级 {cap_heavy} 每卡需要 {stages[cap_heavy].mem.dram_need / 2**30:.1f} GiB，"
                        f"超过每卡 {sys.dram_bytes / 2**30:.0f} GiB"
                        + (f"（含视觉编码器权重 {stages[cap_heavy].mem.pipe_w / 2**30:.2f} GiB）"
                           if vis and stages[cap_heavy].mem.pipe_w else ""))
    if m.coverage == "proxy":
        warnings.append("「架构代理」模型：结果为近似值（见建模说明）")
    if sv.phase == "decode":
        step = max(mb, pp) * tick
        e_tok = spec_expected_tokens(spec_k, sv.spec_accept) if spec_k else 1.0
        tpot = step / e_tok
        thr = sv.batch * e_tok / step
        ttft = 0.0
    else:
        # 0.62: the vision encoder runs before the LLM prefill of the step (serial 「假设」)
        step = (mb + pp - 1) * tick + (vis["s"] if vis else 0.0)
        e_tok = 1.0
        tpot = 0.0
        ttft = step
        thr = sv.batch * sv.prompt / step
    return Result(scn, m, stages, tick, step, e_tok, tpot, ttft, thr, thr / lay.cards, fits,
                  stages[heavy].time.bound, heavy, warnings, mb, moe_skew=skew, vision=vis)



def _vision_cost(scn: Scenario, m: ModelSpec, sys: System, stages: list) -> dict:
    """VLM vision encoder (0.62; ``core/vision.py``).  Placement 「假设」: data-parallel encoder on the first pipeline
    stage (vLLM ``mm_encoder_tp_mode = data``): every card of stage 0 holds the encoder weights (bf16) and encodes its
    share of the step's images (a DP rank's requests on its TP cards); the encode runs before the LLM prefill, serially.
    Decode steps carry only the weights.  Adds the weights (and, at prefill, the encoder's largest activation beyond the
    stage's own) to stage 0's DRAM need."""
    sv, lay = scn.serving, scn.layout
    v = m.vision
    g = image_grid(v, sv.image_w, sv.image_h)
    w = v.params * 2.0
    out = {"images": sv.images, "px": [sv.image_w, sv.image_h], "resized_px": list(g.px), "patches": [g.ph, g.pw],
           "merged": g.merged, "tokens_per_image": g.llm_tokens, "image_tokens": sv.image_tokens,
           "text_prompt": sv.prompt - sv.image_tokens, "label": v.label, "params_B": v.params / 1e9,
           "weights_GB": w / 1e9, "s": 0.0, "tflop": 0.0, "per_card": 0, "cards": 0}
    act = 0.0
    if sv.phase == "prefill":
        per_rank = _cdiv(sv.batch, lay.dp) * sv.images       # images of one DP rank's requests in the step
        c0 = max(1, lay.tp * lay.sp)                          # that rank's cards in the first stage
        per_card = _cdiv(per_rank, c0)
        busy = _cdiv(per_rank, per_card)
        dp_busy = min(lay.dp, _cdiv(sv.batch, _cdiv(sv.batch, lay.dp)))
        key = ("vision", g, per_card, scn.mapping, scn.chip, scn.mem_id, scn.mem_eff)
        memo = model_cache(m).setdefault("vision", {})
        c = memo.get(key)
        if c is None:
            if len(memo) > 4096:
                memo.clear()
            c = memo[key] = _component_time([(vision_ops(v, g, per_card), 1)], sys, scn.mapping, "bf16")
        act = vision_act_peak(v, g, per_card)
        n = busy * dp_busy
        # 0.63 (external review): a card's time is set by its ⌈images / card⌉, but only real images are work — MAC /
        # vector counts and FLOPs × images / per_card (3 images on TP 2 counted 4); byte counts per busy card
        k_w = sv.batch * sv.images / per_card
        out.update(s=c["s"], tflop=c["tflop"] * k_w, tflop_per_image=c["tflop"] / per_card, per_card=per_card,
                   cards=n, bound=c["bound"], dram_GB=c["dram_GB"] * n,
                   acts={k: x * (k_w if k in ("mac", "vec") else n) for k, x in c["acts"].items()})
    mp = stages[0].mem
    need = mp.dram_need + w + max(0.0, act - mp.act_total)
    stages[0].mem = replace(mp, dram_need=need, fits=need <= mp.dram_cap, pipe_w=mp.pipe_w + w)
    return out

def _cfg_chain(seqs: int, mb: int, per_req: int) -> int:
    """Largest set of microbatches linked by a shared request (sequences request-major, microbatches of ⌈S/mb⌉
    consecutive sequences): the microbatches of one connected group depend on each other at every denoise step."""
    if per_req <= 1 or mb <= 1:
        return 1
    msz = _cdiv(seqs, mb)
    nmb = _cdiv(seqs, msz)
    parent = list(range(nmb))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for r in range(_cdiv(seqs, per_req)):
        a, b = (r * per_req) // msz, (min(seqs, (r + 1) * per_req) - 1) // msz
        for j in range(a + 1, b + 1):
            parent[find(j)] = find(a)
    sizes: dict = {}
    for i in range(nmb):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    return max(sizes.values())


def _evaluate_full(scn: Scenario, m: ModelSpec, sys: System, warnings: list[str]) -> Result:
    """Non-autoregressive request: video clip (steps × CFG forwards over every latent token) or protein batch
    (one encoder forward).  Same op-graph → mapping → memplan → schedule path as the LLM phases.

      forward sequences  S = batch · cfg (video) or batch (protein); microbatches mb = min(pp, S) unless set
      video clip latency = steps · max(mb, k + pp − 1) · t_stage  (k = microbatches holding the CFG forwards of one
                           request, which must all finish a step before the next; k = 1 → max(mb, pp): the pipeline
                           is kept full across steps by independent micro-batches)
      protein latency    = (mb + pp − 1) · t_stage              (one pass, like an LLM prefill)
      throughput         = batch · units / latency               (frames/s or sequences/s per replica)
    """
    sv, lay = scn.serving, scn.layout
    wl = resolve_workload(m, scn.workload)
    warnings.extend(wl.warnings)
    if sv.spec_k:
        warnings.append("投机解码对视频 / 蛋白质模型无意义，已忽略")
    seqs = sv.batch * wl.seqs_per_request
    pp = lay.pp
    mb = max(1, min(sv.microbatches or min(pp, seqs), seqs))
    pd = wl.pair
    if pd is not None and lay.sp > 1 and scn.workload.sample_split and pd.samples > 1:
        pd = replace(pd, split=True)
    ph = Phase("full", _cdiv(seqs, mb), wl.tokens, wl.ctx, frames=wl.frames, aux=wl.aux, pair=pd)
    sh = lay.shard
    ab = _fmt(m.act_fmt).bytes
    groups = layer_groups(m)
    ops_memo = model_cache(m).setdefault("layer_sums", {})
    store_memo = model_cache(m).setdefault("stage_storage", {})
    if len(ops_memo) > 200_000:
        ops_memo.clear()
    if fabric.log() is not None:     # fabric_report: run every collective (no layer memo)
        ops_memo = {}
    stages = []
    resid = _cdiv(seqs, lay.dp) * _cdiv(wl.tokens, lay.sp) * m.hidden * ab   # residual streams of in-flight sequences
    pair_act = 0.0
    if wl.pair is not None:      # structure models: the pair representation N²·c_z rides along with the single track
        pair_act = _cdiv(wl.pair.n, lay.sp) * wl.pair.n * m.workload.pair_dim * ab   # DAP: residue-axis slice
        resid += _cdiv(seqs, lay.dp) * pair_act
    fsdp = lay.sp * lay.dp if scn.workload.dit_fsdp and wl.kind == "gen" and lay.sp * lay.dp > 1 else 0
    if scn.workload.dit_fsdp and not fsdp:
        warnings.append("workload.dit_fsdp 已忽略：" + ("只用于视频 DiT" if wl.kind != "gen" else
                                                       "需要 SP · DP > 1（FSDP 在同一流水级的数据 / 序列并行卡之间分片）"))
    def _run(plan_):
        stages = []
        for st in plan_:
            agg = dict.fromkeys(_SUM_KEYS, 0.0)
            if fabric.log() is not None:
                ops_memo.clear()
            if st.has_embed:
                _acc(agg, _sum_ops(full_io_ops(m, ph, sh, "pre"), sys, scn.mapping, m))
            counts: dict = {}
            for li in range(st.first, st.last):
                g = groups[li]
                first_li, n = counts.get(g, (li, 0))
                counts[g] = (first_li, n + 1)
            for g, (li, n) in counts.items():
                okey = (g, ph, sh, scn.mapping, scn.chip, _links(scn))
                hit = ops_memo.get(okey)
                if hit is None:
                    fabric.set_mult(n * layer_repeat(m.layers[li], ph))
                    hit = ops_memo[okey] = _sum_ops(build_rank_ops(m, li, ph, sh), sys, scn.mapping, m)
                    fabric.set_mult(1)
                _acc(agg, hit, n * layer_repeat(m.layers[li], ph))
            if st.has_head:
                _acc(agg, _sum_ops(full_io_ops(m, ph, sh, "post"), sys, scn.mapping, m))
            skey = (st.first, st.last, st.has_embed, st.has_head, sh, 0, 0)
            store = store_memo.get(skey)
            if store is None:
                store = store_memo[skey] = stage_storage(m, st.first, st.last, st.has_embed, st.has_head, sh, 0, 0)
            mp = plan(store, 0, sys.chip.sram_bytes, sys.dram_bytes, agg["max_act"],
                      act_need=agg["max_act_tot"] + resid, stream=True,
                      slc_bytes=sys.chip.slc_bytes, slc_policy=sys.chip.slc_policy)
            dram = step_dram_bytes(mp, store, agg)
            link_bw, sync, link_bytes, d2d_b = agg["link_bw"], agg["sync"], agg["link_bytes"], agg["d2d_bytes"]
            net_b = agg["net_bytes"]
            extra_f: dict = {}
            if fsdp:
                # DiT weights FSDP-sharded over the stage's g = SP·DP ranks (Wan --dit_fsdp): each card keeps w / g plus
                # two gathered layers (prefetch); every stage forward all-gathers the active weights layer by layer
                # (overlapping compute through the stage max rule, α per layer) and writes the gathered copy to DRAM
                # before the usual weight streaming reads it.  An idle expert (Wan2.2) is sharded but not gathered.
                w, nl = mp.stored_w, max(1, st.last - st.first)
                sb = _standby_bytes(m, lay, st.first, st.last)
                act_w = w - sb
                keep = w / fsdp + min(act_w * (fsdp - 1) / fsdp, 2 * act_w / nl)
                need = mp.dram_need - w + keep
                mp = replace(mp, stored_w=w / fsdp, dram_need=need, fits=need <= mp.dram_cap)
                bw, a_, fd, fn = _comm(sys, "allgather", act_w / fsdp, fsdp, stride=lay.tp)  # SP·DP ranks, stride TP
                link_bw += bw; sync += a_ * nl; link_bytes += act_w / fsdp; d2d_b += act_w / fsdp * fd
                net_b += act_w / fsdp * fn
                if sys.fabric.enabled:
                    extra_f = fabric.collective_full("allgather", act_w / fsdp, fsdp, sys, lay.tp)[4]
                gw = act_w * (fsdp - 1) / fsdp
                dram = {**dram, "weights": dram["weights"] + gw, "total": dram["total"] + gw, "fsdp_gather": gw}
            extra = [(x, t) for t, x in extra_f.items()]
            if pp > 1 and not st.has_head:
                act = ph.batch * (_cdiv(ph.q, lay.sp) * m.hidden * ab + pair_act) / lay.dp
                bw, a, fd, fn, tier = _p2p_full(sys, act, st.index, lay.cards // pp)
                link_bw += bw; sync += a; link_bytes += act; d2d_b += act * fd; net_b += act * fn
                extra.append((bw, tier))
            t_dram = dram["total"] / (sys.dram_GBps * 1e9)
            if sys.fabric.enabled:
                link_bw, sync, d2d_b, net_b = _stage_link(
                    sys, agg, link_bw, sync, d2d_b, net_b, extra,
                    max(agg["t_arr"], agg["t_vec"], t_dram, _slc_time(sys, dram)), st.index)
            stt = StageTime(agg["t_arr"], agg["t_mac"], agg["t_feed"], agg["t_vec"], t_dram, link_bw, sync,
                            dram["total"], link_bytes, agg["flops"], agg["t_ideal"], _slc_time(sys, dram),
                            dram.get("slc", 0.0), d2d_b, net_b)
            stages.append(StageResult(st.index, (st.first, st.last), stt, mp, dram, agg["conv"], flops_u=agg["flops_u"],
                                      sram_bytes=agg["sram"], ideal_w=agg["t_ideal_w"], vec_w=agg["t_vec_w"],
                                      conv_w=agg["t_conv_w"], exp_acts=_exp_acts(agg)))
        return stages

    p2p_pay = ph.batch * (_cdiv(ph.q, lay.sp) * m.hidden * ab + pair_act) / lay.dp
    plan_, pinfo = _plan_by_cost(scn, m, sys, pp, groups, ph, sh, _memos(sys, scn.mapping, m), ops_memo, store_memo, 0, 0,
                                 0, full_io_ops(m, ph, sh, "pre"), full_io_ops(m, ph, sh, "post"), p2p_pay, repeat=True)
    stages = _pick_split(_run, plan_, plan_stages(m.n_layers, pp)) if pinfo else _run(plan_)
    pipe = None
    if wl.kind == "gen":
        pipe = _pipeline_cost(scn, m, wl, sys, warnings)
        if pipe:
            pipe = _place_components(scn, m, pipe, stages, sys, warnings)
    heavy = max(range(len(stages)), key=lambda i: stages[i].time.total)
    tick = stages[heavy].time.total
    _pp_imbalance_warn(stages, tick, warnings, scn.pp_split)
    cap_heavy = max(range(len(stages)), key=lambda i: stages[i].mem.dram_need)
    fits = all(s.mem.fits for s in stages)
    if not fits:
        warnings.append(f"容量不足：流水级 {cap_heavy} 每卡需要 {stages[cap_heavy].mem.dram_need / 2**30:.1f} GiB，"
                        f"超过每卡 {sys.dram_bytes / 2**30:.0f} GiB"
                        + (f"（含文本编码器 / VAE 权重 {stages[cap_heavy].mem.pipe_w / 2**30:.1f} GiB）"
                           if stages[cap_heavy].mem.pipe_w else ""))
    if wl.kind == "gen":
        # 0.63 (external review): the CFG forwards of one request (cond / uncond, + image guidance) must all finish a
        # denoise step before its latent update starts the next one.  Microbatches that share a request form a
        # dependent chain of k microbatches: one step of the chain takes ≥ k + pp − 1 stage ticks, and the stages
        # serve mb microbatches per step → per step max(mb, k_max + pp − 1) ticks (= max(mb, pp) when every request
        # sits in one microbatch).  Was max(mb, pp): cond / uncond in different microbatches ran as independent.
        k_chain = _cfg_chain(seqs, mb, wl.seqs_per_request)
        per_step = max(mb, k_chain + pp - 1) if k_chain > 1 else max(mb, pp)
        denoise = wl.steps * per_step * tick
        latency = denoise + (pipe["te_s"] + pipe["decode_s"] + pipe["load_s"] if pipe else 0.0)
        if pipe:
            # 0.61.4: tflop_replica is one DP rank's share ⌈B/dp⌉ of the requests → × B / share (was × dp, which
            # counted idle ranks: batch 1 on DP 2 showed the text encoder + VAE twice per request)
            b_ = scn.serving.batch
            pipe = {**pipe, "denoise_s": denoise, "tflop": pipe["tflop_replica"] * b_ / _cdiv(b_, lay.dp)}
    else:
        latency = (mb + pp - 1) * tick
    period = latency
    if wl.kind == "gen" and scn.workload.overlap:
        # cross-request overlap (0.47): with the text encoder on the host CPU (te_cpu) the host encodes request
        # n+1 while the cards denoise / decode request n — two independent resources, a saturated queue →
        # steady-state period = max(card busy, host busy); clip latency unchanged.  Anything that shares the cards
        # (encoder / VAE on a card, offload reloads) stays serial: nothing else is overlapped.
        if pipe and pipe.get("te_cpu"):
            period = max(latency - pipe["te_s"], pipe["te_s"])
            pipe = {**pipe, "period_s": period, "overlap": True}
        else:
            warnings.append("workload.overlap：文本编码器与 DiT 同在卡上（未开 te_cpu），没有独立资源可跨请求重叠——按串行计")
    thr = sv.batch * wl.units / period
    return Result(scn, m, stages, tick, latency, wl.units, 0.0, 0.0, thr, thr / lay.cards, fits,
                  stages[heavy].time.bound, heavy, warnings, mb, latency, wl, pipe)


def _component_time(groups: list[tuple[list[Op], int]], sys: System, org: str, af: str) -> dict:
    """One pipeline component on one card: same op → mapping → streaming path as a stage; its weights are larger
    than SRAM, so every weight is read from DRAM once per op execution (tiles / chunks re-read).
    ``groups``: [(ops, executions)] (a tiled VAE decode runs one op list per tile)."""
    d = dict.fromkeys(_SUM_KEYS, 0.0)
    shim = SimpleNamespace(act_fmt=af, kv_fmt=af)
    for ops, n in groups:
        _acc(d, _sum_ops(ops, sys, org, shim), n)
    dram = d["hot"] + d["exp"] + d["w_extra"] + d["act"]
    stt = StageTime(d["t_arr"], d["t_mac"], d["t_feed"], d["t_vec"], dram / (sys.dram_GBps * 1e9), 0.0, 0.0, dram,
                    0.0, d["flops"], d["t_ideal"])
    ch = sys.chip
    return {"s": stt.total, "tflop": d["flops"] / 1e12, "dram_GB": dram / 1e9, "bound": stt.bound,
            # action counts (0.47.1, core/energy.py): bf16-equivalent MAC slots, vector element-ops, SRAM-port bytes
            "acts": {"mac": d["t_ideal"] * ch.macs * ch.freq_ghz * 1e9, "vec": d["t_vec"] * ch.lanes * ch.freq_ghz * 1e9,
                     "sram": d["sram"], "dram": dram, "link": 0.0, "d2d": 0.0, "net": 0.0}}


def _pipeline_cost(scn: Scenario, m: ModelSpec, wl: ResolvedWorkload, sys: System, warnings: list[str]) -> dict | None:
    """Text encoder(s) + VAE decode of one request batch (0.44; ``core/pipeline.py``).  Runs on one card of each data-
    parallel replica (its share of the batch), serially with the denoise loop 「假设」."""
    pl = pipeline_for(m.id)
    if pl is None:
        return None
    if not scn.workload.pipeline:
        warnings.append("workload.pipeline = false：只评估 DiT 去噪主干——文本编码器与 VAE 解码不计时间与存储")
        return None
    b = _cdiv(scn.serving.batch, scn.layout.dp)
    info = wl.info
    tiling = scn.workload.vae_tiling
    if tiling and pl.vae.family not in (*TILING, "hunyuan", "h3"):
        warnings.append(f"workload.vae_tiling：{pl.vae.label} 的参考实现没有空间分块解码（按参考默认不分块计）")
    lay = scn.layout
    par = lay.pp * lay.tp * lay.sp if scn.workload.vae_parallel else 1
    if scn.workload.vae_parallel and par == 1:
        warnings.append("workload.vae_parallel：每副本只有 1 张卡，VAE 解码不拆分")
    key = (b, tuple(info["latent"]), info["frames"], info["cfg"], wl.aux, scn.mapping, scn.chip, scn.mem_id,
           scn.mem_eff, tiling, par, _links(scn), lay.dp if par > 1 else 1)
    memo = model_cache(m).setdefault("pipeline", {})
    if key in memo:
        return memo[key]
    prompts = b * (info["cfg"] if pl.neg_prompt else 1)
    parts = []
    te_s = te_w = te_act = te_lw = 0.0
    te_nl = 0
    for te in pl.text:
        nl, lw = te_layers(te)
        te_nl += nl; te_lw = max(te_lw, lw)
        ops, peak = text_ops(te, prompts, m.act_fmt)
        c = _component_time([(ops, 1)], sys, scn.mapping, m.act_fmt)
        w = stored_bytes(te.key)
        te_s += c["s"]; te_w += w; te_act = max(te_act, peak)
        parts.append({"role": "text_encoder", "label": te.label, "key": te.key, "prompts": prompts,
                      "tokens": te.tokens, "stored_GB": w / 1e9, "note": te.note, **c})
    dec_s = vae_w = vae_act = 0.0
    vae_par = 1
    lat = tuple(info["latent"])
    for v, rows in ((pl.vae, 0), (pl.audio, wl.aux)):
        if v is None:
            continue
        ops, peak, vi = vae_ops(v, lat, info["frames"], b, m.act_fmt, audio_rows=rows, tiling=tiling)
        c = _component_time(vi.pop("groups", None) or [(ops, 1)], sys, scn.mapping, vi["act"])
        extra = {}
        if par > 1 and v is pl.vae:
            if "rounds" in vi:
                cp = {**_parallel_decode(vi, par, b, lat, info, sys, scn.mapping, _replica_members(sys, lay)),
                      "tflop": c["tflop"]}
                gb, gd, gn = cp.pop("gather_B"), cp.pop("gather_d2d"), cp.pop("gather_net")
                cp["acts"] = {**c["acts"], "link": gb - gd - gn, "d2d": gd, "net": gn}   # replica totals
                extra = {"par": par, "single_s": c["s"], "gather_s": cp.pop("gather_s"), "rank_tiles": cp.pop("tiles")}
                c = cp
                vae_par = par
            else:
                warnings.append(f"workload.vae_parallel：{pl.vae.label} 未分块解码，不拆分（需 vae_tiling，"
                                "HunyuanVideo / MiniMax-H3 总是分块）")
        w = stored_bytes(v.key)
        dec_s += c["s"]; vae_w += w; vae_act = max(vae_act, peak)
        parts.append({"role": "audio_vae" if v is pl.audio else "vae", "label": v.label, "key": v.key,
                      "stored_GB": w / 1e9, "note": v.note, "act": vi["act"],
                      **{k: vi[k] for k in ("tiles", "overlap", "chunks", "tiling") if k in vi}, **extra, **c})
    out = {"te_s": te_s, "decode_s": dec_s, "tflop_replica": sum(p["tflop"] for p in parts),
           "te_w": te_w, "vae_w": vae_w, "te_act": te_act, "vae_act": vae_act, "parts": parts,
           "te_layers": te_nl, "te_layer_w": te_lw, "vae_par": vae_par,
           "batch_per_replica": b, "note": pl.note}
    if len(memo) > 4096:
        memo.clear()
    memo[key] = out
    return out


def _replica_members(sys: System, lay: Layout) -> tuple[int | None, int | None] | None:
    """Cards of one data-parallel replica (PP·TP·SP) inside one package / one node (cards numbered TP, SP, DP, PP:
    with DP > 1 and PP > 1 the replica is not contiguous — only its TP·SP block is)."""
    if sys.package_cards <= 1 and sys.node_cards <= 0:
        return None
    par = lay.pp * lay.tp * lay.sp
    blk = par if (lay.dp == 1 or lay.pp == 1) else lay.tp * lay.sp

    def inside(cap: int) -> int:
        return math.gcd(min(cap, blk), par)
    return (inside(sys.package_cards) if sys.package_cards > 1 else 1,
            inside(sys.node_cards) if sys.node_cards > 0 else None)


def _parallel_decode(vi: dict, par: int, b: int, lat: tuple, info: dict, sys: System, org: str,
                     members: int | None = None) -> dict:
    """Tile-parallel VAE decode over the ``par`` = PP·TP·SP cards of a replica (0.47, ``workload.vae_parallel``).

    The scheme of the MiniMax-H3 release (``FL2VA/video_vae`` ``vae_parallel_tiling = 1``, ``klvae.tiled_decode``):
    within each independent tiled call (a decode "round": an H3 temporal clip, a HunyuanVideo temporal tile, the
    single spatial pass of a diffusers ``enable_tiling()`` decode) rank r decodes tiles r, r + par, … then the decoded
    pixel tiles are all-gathered so every rank can blend.  Tiles are independent → arithmetic unchanged; time = the
    slowest rank's tiles + one all-gather per round (payload = its largest share of decoded pixels in the decode
    dtype, α each).  VAE weights are replicated on every card of the replica (``_place_components``).  For decoders
    other than H3 this is a design option (xDiT's DistVAE patch-parallel decode is image-VAE-only; its CogVideoX
    example asserts parallel VAE off)."""
    ops = vi["tile_ops"]
    ab = _fmt(vi["act"]).bytes
    px = info["frames"] * info["height"] * info["width"] * 3 / math.prod(lat)   # output elements per latent voxel
    loads: dict = {}
    gather = sent = sent_d = sent_n = 0.0
    for rd in vi["rounds"]:
        share = [rd[r::par] for r in range(par)]
        for r, tl in enumerate(share):
            for t in tl:
                loads.setdefault(r, {}).setdefault(t, 0)
                loads[r][t] += 1
        top = max(sum(math.prod(t) for t in tl) for tl in share) * px * ab * b
        bw, a_, fd, fn = _comm(sys, "allgather", top, par, members=members)
        gather += bw + a_
        sent += par * (par - 1) * top            # every rank's share to every other rank (upper bound)
        sent_d += par * (par - 1) * top * fd
        sent_n += par * (par - 1) * top * fn
    memo: dict = {}
    best = None
    for r, cnt in loads.items():
        k = tuple(sorted(cnt.items()))
        if k not in memo:
            memo[k] = _component_time([(ops[t], n * b) for t, n in k], sys, org, vi["act"])
        if best is None or memo[k]["s"] > best[0]["s"]:
            best = (memo[k], sum(cnt.values()))
    c = dict(best[0])                # time / DRAM of the slowest rank (the caller restores the replica FLOPs)
    c["s"] += gather
    c.update(gather_s=gather, tiles=best[1], gather_B=sent, gather_d2d=sent_d, gather_net=sent_n)
    return c


_PLACE_LABEL = {"resident": "常驻", "shard": "文本编码器分片（FSDP）", "offload": "顺序卸载（CPU offload）",
                "shard+offload": "分片 + 卸载"}


def _standby_bytes(m: ModelSpec, lay: Layout, first: int, last: int) -> float:
    """Idle-expert weights of a stage per card (Wan2.2 A14B: the second 14B expert, resident but not run)."""
    return m.standby_params * (last - first) / m.n_layers * m.fmt("attn").bits / 8 / lay.tp


def _place_components(scn: Scenario, m: ModelSpec, pipe: dict, stages: list, sys: System,
                      warnings: list[str]) -> dict:
    """Where the pipeline components live on the cards of one data-parallel replica (0.45; docs/MODEL.md §11.4).

      resident       text encoder on the first stage's card, VAE (+ audio VAE) on the last stage's card, both
                     resident next to the DiT (0.44 behaviour)
      shard          text-encoder weights sharded over all c = PP·TP·SP cards of the replica (Wan ``--t5_fsdp``):
                     te_w / c per card + 2 gathered layers (prefetch); each card all-gathers the layers while it
                     encodes — time max(compute, te_w·(c−1)/c / link) + α per layer
      offload        components and DiT time-share the card (diffusers ``enable_model_cpu_offload`` / Wan
                     ``--offload_model``): capacity = max(DiT, text encoder, VAE) instead of the sum; every request
                     re-loads TE, DiT stage and VAE weights host → card at ``workload.host_GBps`` (H2D only, the host
                     keeps its copy 「假设」); a multi-expert DiT (Wan2.2 A14B) also parks its idle expert on the host
                     (Wan2.2 ``--offload_model`` swaps the experts at the noise boundary) → standby not resident,
                     both experts still loaded once per request
      shard+offload  both
      auto           the first of resident → shard → offload → shard+offload that fits (else the smallest need)
    ``workload.te_cpu`` (0.46, Wan ``--t5_cpu``): the text encoder runs on the host CPU — no encoder weights or
    activations on any card, encode time = encoder FLOPs / ``workload.host_TFLOPS`` (「假设」, set from a measurement),
    the embeddings' copy to the card is negligible; then shard is moot and auto tries resident → offload.
    ``workload.dit_fsdp`` (0.46): the DiT stage weights are already sharded over SP·DP in the stage plan (see
    ``_evaluate_full``); an offloaded card then re-loads only its shard."""
    lay, w = scn.layout, scn.workload
    c = lay.pp * lay.tp * lay.sp
    last = len(stages) - 1
    base = [s.mem for s in stages]
    alpha = sys.link.alpha_us * 1e-6
    g = lay.sp * lay.dp if w.dit_fsdp and lay.sp * lay.dp > 1 else 1
    standby = [_standby_bytes(m, lay, *s.layers) / g for s in stages]

    cpu = w.te_cpu
    te_tflop = sum(p["tflop"] for p in pipe["parts"] if p["role"] == "text_encoder")
    te_act = 0.0 if cpu else pipe["te_act"]

    def option(place: str) -> tuple[list[MemPlan], dict]:
        shard = "shard" in place and c > 1 and not cpu
        off = "offload" in place
        te_card = (pipe["te_w"] / c + 2 * pipe["te_layer_w"]) if shard else pipe["te_w"]
        te_card = 0.0 if cpu else min(te_card, pipe["te_w"])
        mems = []
        for i, mp in enumerate(base):
            te_here = shard or i == 0
            vae_here = i == last or pipe.get("vae_par", 1) > 1
            w_te = te_card if te_here else 0.0
            w_vae = pipe["vae_w"] if vae_here else 0.0
            if off:
                need = max(mp.dram_need - standby[i],
                           RUNTIME_RESERVE + w_te + te_act if te_here and not cpu else 0.0,
                           RUNTIME_RESERVE + w_vae + pipe["vae_act"] if vae_here else 0.0)
            else:
                act = max(te_act if te_here else 0.0, pipe["vae_act"] if vae_here else 0.0)
                need = mp.dram_need + w_te + w_vae + max(0.0, act - mp.act_total)
            mems.append(replace(mp, dram_need=need, fits=need <= mp.dram_cap, pipe_w=w_te + w_vae))
        if not shard:
            gather = 0.0
        elif sys.package_cards <= 1 and sys.node_cards <= 0 and not sys.fabric.enabled:
            gather = pipe["te_w"] * (c - 1) / c / (sys.link.GBps * 1e9) + alpha * pipe["te_layers"]
        else:       # 0.48 / 0.50: the same all-gather volume on the tiered fabric, α per encoder layer
            bw_, a_, _, _ = _comm(sys, "allgather", pipe["te_w"] / c, c, members=_replica_members(sys, lay))
            gather = bw_ + a_ * pipe["te_layers"]
        te_s = (te_tflop / w.host_TFLOPS if cpu else max(pipe["te_s"], gather) if shard else pipe["te_s"])
        host = w.host_GBps * 1e9
        load = ((te_card + max(mp.stored_w for mp in base) + pipe["vae_w"]) / host) if off else 0.0
        # 0.62: host-link bytes of one reload over the whole replica (every card re-reads its own DiT stage, encoder
        # share and VAE copy; load_s above is the slowest card's) -- the energy report charges them at pJ_bit_host
        cps = lay.cards // len(base)
        load_b = sum(cps * (mp.stored_w + (te_card if (shard or i == 0) else 0.0)
                            + (pipe["vae_w"] if (i == last or pipe.get("vae_par", 1) > 1) else 0.0))
                     for i, mp in enumerate(base)) if off else 0.0
        return mems, {"place": "shard" if place == "shard+offload" and not shard else
                      ("resident" if place == "shard" and not shard else place),
                      "te_card_w": te_card, "gather_s": gather, "te_s": te_s, "te_compute_s": pipe["te_s"],
                      "load_s": load, "load_bytes": load_b, "te_cards": c if shard else 1, "te_cpu": cpu,
                      "host_TFLOPS": w.host_TFLOPS if cpu else None}

    if w.placement == "auto":
        multi = c > 1 and not cpu
        cands = ["resident"] + (["shard"] if multi else []) + ["offload"] + (["shard+offload"] if multi else [])
        opts = [(p, *option(p)) for p in cands]
        pick = next((o for o in opts if all(mp.fits for mp in o[1])), None)
        if pick is None:
            pick = min(opts, key=lambda o: max(mp.dram_need for mp in o[1]))
    else:
        pick = (w.placement, *option(w.placement))
    req, mems, info = pick
    for s_, mp in zip(stages, mems):
        s_.mem = mp
    if w.placement == "auto" and info["place"] != "resident":
        warnings.append(f"组件放置 auto → {_PLACE_LABEL[info['place']]}：常驻放不下"
                        + (f"；每请求从主机重载权重 {info['load_s']:.2f} s（{w.host_GBps:g} GB/s「假设」）" if info["load_s"] else ""))
    parts = pipe["parts"]
    if cpu:     # encoder parts report their host time (the card-side numbers stay in te_compute_s)
        parts = [{**p, "s": p["tflop"] / w.host_TFLOPS, "bound": "host CPU", "dram_GB": 0.0, "on_host": True}
                 if p["role"] == "text_encoder" else p for p in parts]
    return {**pipe, **info, "parts": parts, "placement": w.placement,
            "place_label": _PLACE_LABEL[info["place"]] + ("，文本编码器在主机 CPU" if cpu else ""),
            "host_GBps": w.host_GBps}

