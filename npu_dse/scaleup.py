"""Multi-card tensor-parallel (TP) + pipeline-parallel (PP) + C2C + KV fabric.

Honesty labels: all C2C / IB / RoCE bandwidths, latencies, hide factors, and
link topology are *assumed / uncalibrated*. No PDK.

---------------------------------------------------------------------------
TP sharding (Megatron-style, stated simply)
---------------------------------------------------------------------------
Per transformer layer:
  Attention: Q/K/V column-parallel; O row-parallel → all-reduce
  MLP:       gate/up column-parallel; down row-parallel → all-reduce
Each layer: **2 all-reduces** on volume ``B · S · H · act_bytes``.
Per-card FLOPs / body W / KV ≈ total / (tp * pp) when PP splits layers.

---------------------------------------------------------------------------
Pipeline parallel (light, v0.8)
---------------------------------------------------------------------------
``pp ∈ {1,2,4,8}``; total cards = tp * pp (DP=1).
Layers split across stages (≈ L/pp each).
Activation send between stages: ``B·S·H·act_bytes`` per boundary (pp−1 sends
per forward pass through the stack).
Decode with microbatch ``mb=1``: bubble fraction ``(pp-1)/pp`` — decode PP is
often dependency-bound; wall ≈ stage_time / (1 − bubble) = stage_time · pp.
Prefill may use ``mb ≥ pp`` to cut bubble crudely:
  bubble = (pp-1) / (mb + pp - 1).
Limitation documented: crude analytical bubble; not a real schedule.

---------------------------------------------------------------------------
Expert parallel (MoE EP, v0.9)
---------------------------------------------------------------------------
``ep ∈ {1,2,4,8}``; total cards = tp * pp * ep (DP=1).
Experts sharded: each rank holds E/ep experts (E % ep == 0 required).
Active weight stream per rank: attn/tp + shared + (top_k/ep)*FFN (balanced).
All-to-all token dispatch+combine (Megatron-MoE style, crude, documented):

  V = B * S * H * act_bytes
  dispatch+combine ≈ 2 * (ep-1)/ep * V * (top_k / E_local_adjust)

Default ``E_local_adjust = 1`` (identity) → factor is just ``top_k``.
When ep=1: A2A bytes = 0. A2A rides the C2C (or dedicated EP) link.

---------------------------------------------------------------------------
KV fabric (IB / RoCE MVP, analytical, v0.8)
---------------------------------------------------------------------------
Modes: ``none`` (default) | ``roce_v2`` | ``ib``.
Presets: 100/200/400 Gbps class → assumed effective GB/s + fixed µs latency.
Use cases:
  1. Remote KV read on decode: ``remote_kv_frac`` of KV bytes from fabric
     (default 0 local; 1.0 for disagg decode worker).
     t_fabric = latency + bytes/BW; local DRAM KV reduced by (1−frac);
     wall uses max(compute, local_dram, c2c, fabric) honestly.
  2. Prefill→decode KV transfer (PD disagg one-shot): full KV after prefill;
     metric ``t_kv_xfer`` (also optional add-on to TTFT via flag).

When ``kv_fabric=none``, ``pp=1``, ``ep=1``: identical to v0.7 TP-only path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .evaluate import (
    EvalConfig,
    InferenceResult,
    PhaseResult,
    _mem_time_with_hide,
    effective_compute_time_s,
    evaluate_decode_phase,
    evaluate_inference,
)
from .memory import ExternalMemory, SRAMConfig
from .model_shape import ModelShape
from .npu import NPUConfig
from .traffic import ContentionResult, PhaseTraffic, contend

KvFabricMode = Literal["none", "roce_v2", "ib"]
EmbedPolicy = Literal["replicate", "shard"]
LinkDuplex = Literal["unidirectional", "bidirectional"]

VALID_TP = (1, 2, 4, 8)
VALID_PP = (1, 2, 4, 8)
VALID_EP = (1, 2, 4, 8)

# Megatron-style: all-reduce after row-parallel O and after row-parallel down.
COLLECTIVES_PER_LAYER = 2

# v0.29 — per-collective fixed latency α (launch + sync + hop), ASSUMED.
# Pure-bandwidth collectives were optimistic for decode (tiny messages):
# research/related_tools_2026-10 §bug-1. Default 3 µs (range 2–5 µs typical
# for NVLink/C2C all-reduce at small sizes). α=0 reproduces ≤0.28 exactly.
DEFAULT_C2C_LATENCY_US = 3.0
DEFAULT_SYNC_OVERLAP = 0.0  # fraction of α hidden behind compute; 0 = fully exposed

AttnParallel = Literal["tp", "dp"]
# v0.31 MoE expert sharding inside one PP stage (tp·ep cards):
#   "tp_ep"  — experts partitioned over the ep groups; every expert's weights
#              TP-split by tp inside its group (Megatron TP×EP). Default.
#   "ep_all" — experts partitioned over all tp·ep ranks (EP = tp·ep), each
#              expert whole on one rank (vLLM/SGLang "EP over all ranks").
MoeShard = Literal["tp_ep", "ep_all"]
SpecDraft = Literal["mtp", "model"]
DEFAULT_SPEC_ACCEPT = 0.7  # 「假设」per-token draft acceptance rate


# ---------------------------------------------------------------------------
# C2C link presets (assumed effective BW)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C2CLink:
    """Assumed chip-to-chip peer link.

    ``effective_bw_GBps`` is *sustained* bytes/s usable for collectives,
    already folding protocol/efficiency — labeled assumed, not a datasheet.
    """

    name: str
    effective_bw_GBps: float
    duplex: LinkDuplex = "bidirectional"
    note: str = "assumed effective per-peer BW"

    def effective_bw_Bps(self) -> float:
        return self.effective_bw_GBps * 1e9

    def summary(self) -> str:
        return (
            f"{self.name}: {self.effective_bw_GBps:.0f} GB/s effective "
            f"({self.duplex}) [{self.note}]"
        )


C2C_100 = C2CLink("c2c_100", 100.0, "bidirectional")
C2C_200 = C2CLink("c2c_200", 200.0, "bidirectional")
C2C_400 = C2CLink("c2c_400", 400.0, "bidirectional")
C2C_800 = C2CLink("c2c_800", 800.0, "bidirectional")
C2C_100_UNI = C2CLink("c2c_100_uni", 100.0, "unidirectional")
C2C_200_UNI = C2CLink("c2c_200_uni", 200.0, "unidirectional")

C2C_PRESETS: dict[str, C2CLink] = {
    c.name: c
    for c in (C2C_100, C2C_200, C2C_400, C2C_800, C2C_100_UNI, C2C_200_UNI)
}


def get_c2c(name_or_gbps: str | int | float) -> C2CLink:
    if isinstance(name_or_gbps, (int, float)):
        key = f"c2c_{int(name_or_gbps)}"
        if key in C2C_PRESETS:
            return C2C_PRESETS[key]
        return C2CLink(f"c2c_{int(name_or_gbps)}", float(name_or_gbps))
    if name_or_gbps in C2C_PRESETS:
        return C2C_PRESETS[name_or_gbps]
    raise KeyError(f"unknown C2C preset {name_or_gbps!r}; known={sorted(C2C_PRESETS)}")


# ---------------------------------------------------------------------------
# KV fabric presets (IB / RoCE — assumed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KvFabricLink:
    """Assumed IB or RoCEv2 link for remote KV.

    Line-rate class in Gbps converted to effective GB/s with an assumed
    efficiency fold-in. Latency is a fixed per-message µs cost (assumed).
    """

    name: str
    mode: KvFabricMode
    link_gbps: float
    effective_bw_GBps: float
    latency_us: float
    note: str = "assumed effective BW + fixed latency"

    def effective_bw_Bps(self) -> float:
        return self.effective_bw_GBps * 1e9

    def latency_s(self) -> float:
        return self.latency_us * 1e-6

    def transfer_time_s(self, nbytes: int | float) -> float:
        """t = latency + bytes/BW (one message). Returns 0 if nbytes<=0."""
        if nbytes <= 0:
            return 0.0
        bw = self.effective_bw_Bps()
        return self.latency_s() + (float(nbytes) / bw if bw > 0 else 0.0)

    def summary(self) -> str:
        return (
            f"{self.name}: {self.mode} {self.link_gbps:.0f}Gbps → "
            f"{self.effective_bw_GBps:.2f} GB/s eff, "
            f"lat={self.latency_us:.1f}µs [{self.note}]"
        )


def _gbps_to_eff_GBps(gbps: float, efficiency: float = 0.80) -> float:
    """Line Gbps → assumed effective GB/s (Gbps/8 * eff)."""
    return (gbps / 8.0) * efficiency


# Assumed presets: 100/200/400 Gbps class. IB slightly lower latency than RoCE.
KV_FABRIC_PRESETS: dict[str, KvFabricLink] = {}
for _mode, _lat in (("roce_v2", 5.0), ("ib", 2.0)):
    _prefix = "roce" if _mode == "roce_v2" else "ib"
    for _g in (100, 200, 400):
        _name = f"{_prefix}_{_g}g"
        KV_FABRIC_PRESETS[_name] = KvFabricLink(
            name=_name,
            mode=_mode,  # type: ignore[arg-type]
            link_gbps=float(_g),
            effective_bw_GBps=_gbps_to_eff_GBps(float(_g)),
            latency_us=_lat,
        )

# Default pick per mode (200G class)
KV_FABRIC_DEFAULT_BY_MODE: dict[str, str] = {
    "roce_v2": "roce_200g",
    "ib": "ib_200g",
}


def get_kv_fabric(
    name_or_mode: str,
    *,
    gbps: int | float | None = None,
) -> KvFabricLink | None:
    """Resolve a fabric link. ``none`` → None. Mode alone → default 200G preset.

    Accepts preset names (``roce_200g``, ``ib_400g``) or modes (``roce_v2``, ``ib``).
    Optional ``gbps`` overrides the class when a mode is given.
    """
    if name_or_mode in ("none", "", None):
        return None
    if name_or_mode in KV_FABRIC_PRESETS:
        link = KV_FABRIC_PRESETS[name_or_mode]
        if gbps is not None and abs(link.link_gbps - float(gbps)) > 0.5:
            # Rebuild at requested class keeping mode/latency family
            prefix = "roce" if link.mode == "roce_v2" else "ib"
            key = f"{prefix}_{int(gbps)}g"
            if key in KV_FABRIC_PRESETS:
                return KV_FABRIC_PRESETS[key]
            lat = 5.0 if link.mode == "roce_v2" else 2.0
            return KvFabricLink(
                name=key,
                mode=link.mode,
                link_gbps=float(gbps),
                effective_bw_GBps=_gbps_to_eff_GBps(float(gbps)),
                latency_us=lat,
            )
        return link
    if name_or_mode in ("roce_v2", "ib"):
        g = int(gbps) if gbps is not None else 200
        prefix = "roce" if name_or_mode == "roce_v2" else "ib"
        key = f"{prefix}_{g}g"
        if key in KV_FABRIC_PRESETS:
            return KV_FABRIC_PRESETS[key]
        lat = 5.0 if name_or_mode == "roce_v2" else 2.0
        return KvFabricLink(
            name=key,
            mode=name_or_mode,  # type: ignore[arg-type]
            link_gbps=float(g),
            effective_bw_GBps=_gbps_to_eff_GBps(float(g)),
            latency_us=lat,
        )
    raise KeyError(
        f"unknown kv_fabric {name_or_mode!r}; "
        f"modes=none|roce_v2|ib presets={sorted(KV_FABRIC_PRESETS)}"
    )


# ---------------------------------------------------------------------------
# Parallel / TP config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TPConfig:
    """Tensor + pipeline + expert parallel + C2C + KV fabric knobs (assumed unless derived)."""

    tp: int = 1
    pp: int = 1
    ep: int = 1  # MoE expert-parallel degree; cards = tp*pp*ep
    mb: int = 1  # microbatches; decode often 1 (bubble-heavy)
    c2c: C2CLink = C2C_400
    c2c_hide: float = 0.0
    kv_fabric: KvFabricMode = "none"
    fabric_gbps: float = 200.0  # class when fabric ≠ none
    remote_kv_frac: float = 0.0  # 1.0 = disagg decode worker
    pd_kv_xfer: bool = False  # enable prefill→decode one-shot transfer metric
    add_xfer_to_ttft: bool = False  # if True, ttft includes t_kv_xfer
    embed_policy: EmbedPolicy = "replicate"
    pp_link: C2CLink | None = None  # None → reuse c2c for inter-stage acts
    # E_local_adjust in A2A formula (default 1 → multiply by top_k only)
    ep_e_local_adjust: float = 1.0
    # v0.29 exposed sync: t_sync = n_collectives × α × (1 − overlap)  [assumed]
    c2c_latency_us: float = DEFAULT_C2C_LATENCY_US
    sync_overlap: float = DEFAULT_SYNC_OVERLAP
    # v0.29 attention parallel: "tp" (heads sharded; KV per min(tp, n_kv)) or
    # "dp" (attention data-parallel: KV partitioned by batch, attn W replicated)
    attn_parallel: AttnParallel = "tp"
    # v0.31 MoE expert sharding (see MoeShard) — only matters for MoE shapes
    moe_shard: MoeShard = "tp_ep"
    # v0.31 decode micro-batches for PP (0 = auto: min(B, pp)); B=1 → 1
    decode_mb: int = 0
    # v0.31 speculative decoding / MTP (spec_k=0 → off)  [assumed acceptance]
    spec_k: int = 0
    spec_accept: float = DEFAULT_SPEC_ACCEPT
    spec_draft: SpecDraft = "mtp"
    spec_draft_frac: float = 0.1  # draft="model": draft step cost / target step

    def __post_init__(self) -> None:
        if self.tp not in VALID_TP:
            raise ValueError(f"tp must be one of {VALID_TP}, got {self.tp}")
        if self.pp not in VALID_PP:
            raise ValueError(f"pp must be one of {VALID_PP}, got {self.pp}")
        if self.ep not in VALID_EP:
            raise ValueError(f"ep must be one of {VALID_EP}, got {self.ep}")
        if self.mb < 1:
            raise ValueError(f"mb must be >= 1, got {self.mb}")
        if not (0.0 <= self.c2c_hide <= 1.0):
            raise ValueError("c2c_hide must be in [0, 1]")
        if self.kv_fabric not in ("none", "roce_v2", "ib"):
            raise ValueError("kv_fabric must be 'none', 'roce_v2', or 'ib'")
        if not (0.0 <= self.remote_kv_frac <= 1.0):
            raise ValueError("remote_kv_frac must be in [0, 1]")
        if self.ep_e_local_adjust <= 0:
            raise ValueError("ep_e_local_adjust must be > 0")
        if self.c2c_latency_us < 0:
            raise ValueError("c2c_latency_us must be >= 0")
        if not (0.0 <= self.sync_overlap <= 1.0):
            raise ValueError("sync_overlap must be in [0, 1]")
        if self.attn_parallel not in ("tp", "dp"):
            raise ValueError("attn_parallel must be 'tp' or 'dp'")
        if self.moe_shard not in ("tp_ep", "ep_all"):
            raise ValueError("moe_shard must be 'tp_ep' or 'ep_all'")
        if self.decode_mb < 0:
            raise ValueError("decode_mb must be >= 0 (0 = auto min(B, pp))")
        if not (0 <= self.spec_k <= 8):
            raise ValueError("spec_k must be in [0, 8]")
        if not (0.0 <= self.spec_accept <= 1.0):
            raise ValueError("spec_accept must be in [0, 1]")
        if self.spec_draft not in ("mtp", "model"):
            raise ValueError("spec_draft must be 'mtp' or 'model'")
        if not (0.0 <= self.spec_draft_frac <= 1.0):
            raise ValueError("spec_draft_frac must be in [0, 1]")

    @property
    def n_cards(self) -> int:
        return self.tp * self.pp * self.ep

    def fabric_link(self) -> KvFabricLink | None:
        if self.kv_fabric == "none":
            return None
        return get_kv_fabric(self.kv_fabric, gbps=self.fabric_gbps)

    def interstage_link(self) -> C2CLink:
        return self.pp_link if self.pp_link is not None else self.c2c

    def decode_mb_eff(self, batch: int) -> int:
        """Decode micro-batches in flight through the pipeline (v0.31).

        pp=1 → 1. Otherwise the user knob ``decode_mb`` (0 = auto ``min(B, pp)``),
        clamped to [1, B]: B=1 keeps the single-request semantics."""
        if self.pp <= 1:
            return 1
        b = max(int(batch), 1)
        mb = self.decode_mb if self.decode_mb > 0 else min(b, self.pp)
        return max(1, min(int(mb), b))

    def spec_tokens_per_step(self) -> float:
        """Expected tokens emitted per verify step: (1 − a^(k+1)) / (1 − a)."""
        return spec_expected_tokens(self.spec_k, self.spec_accept)

    def bubble_fraction(self, *, phase: str = "decode") -> float:
        """Pipeline bubble fraction.

        Decode (mb often 1): (pp-1)/pp — documented limitation (bubble-heavy).
        Prefill: (pp-1)/(mb+pp-1); pass mb≥pp to reduce bubbles crudely.
        """
        if self.pp <= 1:
            return 0.0
        mb = self.mb
        if phase == "prefill" and mb < self.pp:
            # still allow user mb; formula handles it
            pass
        return (self.pp - 1) / (mb + self.pp - 1)


# ---------------------------------------------------------------------------
# Collective / activation math
# ---------------------------------------------------------------------------


def spec_expected_tokens(k: int, a: float) -> float:
    """Expected accepted tokens per speculative step (incl. the bonus token).

    E = 1 + a + a² + … + a^k = (1 − a^(k+1)) / (1 − a); a = 1 → k + 1; k = 0 → 1.
    「假设」i.i.d. per-position acceptance.
    """
    k = max(int(k), 0)
    a = float(a)
    if k == 0:
        return 1.0
    if a >= 1.0 - 1e-12:
        return float(k + 1)
    return (1.0 - a ** (k + 1)) / (1.0 - a)


def moe_expert_degree(shape: ModelShape, tp: int, ep: int, moe_shard: str = "tp_ep") -> tuple[int, int]:
    """(expert-parallel degree, expert TP degree) inside one PP stage (v0.31).

    tp_ep : experts over ``ep`` groups, each expert TP-split by ``tp``.
    ep_all: experts over all ``tp·ep`` ranks, experts not split.
    Dense shapes: (1, tp) — FFN is TP-split as before.
    """
    if not shape.is_moe:
        return 1, max(int(tp), 1)
    if moe_shard == "ep_all":
        return max(int(tp), 1) * max(int(ep), 1), 1
    return max(int(ep), 1), max(int(tp), 1)


def moe_sharded(shape: ModelShape, tp: int, ep: int, moe_shard: str = "tp_ep") -> bool:
    """True when the MoE expert-sharded accounting applies (ep>1 or ep_all with tp>1)."""
    return shape.is_moe and (ep > 1 or (moe_shard == "ep_all" and tp > 1))


def attn_dp_degree(shape: ModelShape, tp: int, ep: int, attn_parallel: str = "tp") -> int:
    """Attention data-parallel replicas inside one PP stage (v0.31).

    attn=dp splits the batch across the tp ranks; MoE ep>1 additionally runs
    attention data-parallel across the ep groups (standard TP×EP / DeepSeek
    "attention DP + expert EP"), so the batch (and KV) splits over ep too.
    """
    d = max(int(tp), 1) if attn_parallel == "dp" else 1
    if shape.is_moe and ep > 1:
        d *= int(ep)
    return d


def moe_a2a_bytes_per_rank(
    shape: ModelShape,
    tokens: int,
    ep_degree: int,
    *,
    moe_layers: int,
    e_local_adjust: float = 1.0,
) -> int:
    """Per-rank MoE dispatch+combine bytes for ``tokens`` tokens on one stage (v0.31).

    Each rank dispatches its share ``tokens / D`` (D = expert-parallel degree,
    attention data-parallel over the groups), every token goes to ``top_k``
    experts spread uniformly, a fraction ``(D-1)/D`` leaves the rank/group:

        per layer = 2 · (D−1)/D · (tokens / D) · top_k · H · act_bytes / E_local_adjust

    For tp_ep the destination group's tp ranks each need the full activation,
    which is why D = ep (not tp·ep) there. ≤0.30 used the full batch per rank
    (over-counted by ×D). 0 when D ≤ 1 or the shape is dense.
    """
    D = int(ep_degree)
    if D <= 1 or not shape.is_moe or tokens <= 0 or moe_layers <= 0:
        return 0
    if e_local_adjust <= 0:
        raise ValueError("e_local_adjust must be > 0")
    per_layer = (
        2.0 * (D - 1) / D * (tokens / D) * shape.top_k * shape.hidden
        * (shape.act_bits // 8) / e_local_adjust
    )
    return int(moe_layers * per_layer)


def ring_allreduce_bytes(volume: int, tp: int) -> int:
    """Per-rank bytes moved on the fabric for one ring all-reduce.

    Formula: ``2 * (tp - 1) / tp * volume``. Returns 0 when tp <= 1.
    """
    if tp <= 1 or volume <= 0:
        return 0
    return int(2 * (tp - 1) / tp * volume)


def moe_alltoall_bytes(
    shape: ModelShape,
    batch: int,
    seq: int,
    ep: int,
    *,
    e_local_adjust: float = 1.0,
    layers: int | None = None,
) -> int:
    """Per-rank MoE all-to-all dispatch+combine bytes (Megatron-MoE style).

    Documented formula (assumed balanced top_k routing)::

        V = B * S * H * act_bytes
        dispatch+combine = 2 * (ep - 1) / ep * V * (top_k / E_local_adjust)

    Default ``E_local_adjust = 1`` → the factor is simply ``top_k``.
    Multiplied by ``layers`` (default ``n_layers``) for a full phase; callers
    with PP should pass ``layers = n_layers // pp``.

    Returns 0 when ``ep <= 1`` or the shape is not MoE.
    """
    if ep <= 1 or not shape.is_moe or batch <= 0 or seq <= 0:
        return 0
    if e_local_adjust <= 0:
        raise ValueError("e_local_adjust must be > 0")
    vol = activation_volume_bytes(shape, batch, seq)
    factor = shape.top_k / e_local_adjust
    per_layer = int(2 * (ep - 1) / ep * vol * factor)
    n_layers = shape.n_layers if layers is None else layers
    return max(n_layers, 0) * per_layer


def activation_volume_bytes(
    shape: ModelShape,
    batch: int,
    seq: int,
) -> int:
    """Residual-stream activation volume B·S·H·act_bytes (one all-reduce / PP send)."""
    return int(batch) * int(seq) * shape.hidden * (shape.act_bits // 8)


def collectives_bytes_per_phase(
    shape: ModelShape,
    batch: int,
    seq: int,
    tp: int,
    *,
    pp: int = 1,
    collectives_per_layer: int = COLLECTIVES_PER_LAYER,
) -> int:
    """Total per-rank C2C bytes for TP all-reduces in one phase.

    With PP, only layers on this stage participate → scale by 1/pp.
    """
    if tp <= 1:
        return 0
    vol = activation_volume_bytes(shape, batch, seq)
    per = ring_allreduce_bytes(vol, tp)
    # Layers on this PP stage (integer floor; remainder ignored in MVP)
    layers_here = shape.n_layers // max(pp, 1)
    return layers_here * collectives_per_layer * per


def pp_activation_send_bytes(
    shape: ModelShape,
    batch: int,
    seq: int,
    pp: int,
) -> int:
    """Bytes sent on inter-stage links for one forward through the pipeline.

    ``(pp - 1)`` boundaries × ``B·S·H·act_bytes`` (per-message volume).
    Counted once per phase (not per-rank multiplied) — each boundary is one send.
    """
    if pp <= 1:
        return 0
    vol = activation_volume_bytes(shape, batch, seq)
    return (pp - 1) * vol


def pipeline_bubble_fraction(pp: int, mb: int) -> float:
    if pp <= 1:
        return 0.0
    if mb < 1:
        raise ValueError("mb must be >= 1")
    return (pp - 1) / (mb + pp - 1)


def sync_collectives_per_token(
    n_layers: int,
    *,
    tp: int,
    pp: int,
    ep: int = 1,
    is_moe: bool = False,
    collectives_per_layer: int = COLLECTIVES_PER_LAYER,
) -> int:
    """Number of latency-bound collectives on one PP stage per forward/token.

    TP:  ``collectives_per_layer`` (Megatron: 2 all-reduce) × L/pp
    EP:  2 all-to-all (dispatch + combine) × L/pp (MoE only)
    PP:  1 activation send/recv per stage
    """
    layers_here = int(n_layers) // max(int(pp), 1)
    n = 0
    if tp > 1:
        n += collectives_per_layer * layers_here
    if ep > 1 and is_moe:
        n += 2 * layers_here
    if pp > 1:
        n += 1
    return n


def sync_time_s(n_collectives: int, latency_us: float, overlap: float = 0.0) -> float:
    """Exposed sync time = n × α × (1 − overlap)  [α assumed]."""
    if n_collectives <= 0 or latency_us <= 0:
        return 0.0
    return n_collectives * latency_us * 1e-6 * (1.0 - overlap)


def kv_card_fraction(
    shape: ModelShape,
    tp: int,
    *,
    pp: int = 1,
    attn_parallel: str = "tp",
    batch: int = 1,
    ep: int = 1,
) -> tuple[int, int, float, str]:
    """Fraction (num, den) of the full KV cache held/read by one card.

    Returns ``(num, den, replication, reason)``; per-card KV = total·num//den.
    ``replication`` = stored copies vs ideal sharding (1.0 = none).

      - attention TP (attn_parallel="tp"), GQA/MHA: each rank holds
        ceil(n_kv/tp) KV heads → tp > n_kv duplicates KV heads.
      - attention TP, MLA: the joint latent is **not** head-sharded → every TP
        rank holds the full latent (replication = tp).
      - attention DP: KV partitioned by batch over the DP replicas
        (ceil(B/d)/B per rank). d = tp for attn_parallel="dp", and v0.31:
        MoE ep>1 runs attention data-parallel across the ep groups (d ×= ep).
    PP always divides by pp (layers per stage).
    """
    pp = max(int(pp), 1)
    tp = max(int(tp), 1)
    t_a = 1 if attn_parallel == "dp" else tp
    d = attn_dp_degree(shape, tp, ep, attn_parallel)
    if t_a <= 1 and d <= 1:
        return 1, pp, 1.0, ""
    b = max(int(batch), 1)
    whys: list[str] = []
    per_b, den_b, rep_b = 1, 1, 1.0
    if d > 1:
        per_b = math.ceil(b / d)
        den_b = b
        rep_b = per_b * d / b
        whys.append(f"attn-DP: KV 按 batch 切分 ceil({b}/{d})/{b}")
    per_h, den_h, rep_h = 1, 1, 1.0
    if t_a > 1:
        if shape.is_mla:
            rep_h = float(t_a)
            whys.append(f"MLA latent 不可按 TP 切分：每卡全量 (×{t_a})")
        else:
            nkv = max(int(shape.n_kv_heads), 1)
            per_h = math.ceil(nkv / t_a)
            den_h = nkv
            rep_h = per_h * t_a / nkv
            if rep_h > 1.0 + 1e-12:
                whys.append(f"GQA n_kv={nkv} < tp={t_a}: 每卡 {per_h} 个 KV 头（复制 ×{rep_h:g}）")
    return per_b * per_h, den_b * den_h * pp, rep_b * rep_h, "；".join(whys)


# ---------------------------------------------------------------------------
# Per-card capacity
# ---------------------------------------------------------------------------


def _ep_ffn_layer_avg_bytes(shape: ModelShape, moe_layer_params: int, tp: int) -> int:
    """Layer-average FFN bytes on one EP rank (v0.30 helper, kept for callers).

    MoE layers hold ``moe_layer_params``; dense-prefix layers keep a dense
    SwiGLU that is TP-sharded (/tp).
    """
    L = max(shape.n_layers, 1)
    nd = shape.n_dense_layers
    wb = shape.weight_bits
    if nd <= 0:
        return moe_layer_params * wb // 8
    total = nd * (shape.dense_ffn_params() // max(tp, 1)) + (L - nd) * moe_layer_params
    return total * wb // 8 // L


def moe_layer_card_params(
    shape: ModelShape,
    tp: int,
    ep: int,
    *,
    attn_parallel: str = "tp",
    moe_shard: str = "tp_ep",
    routed_experts: float | None = None,
) -> float:
    """Layer-average params on one card of a PP stage, MoE expert-sharded (v0.31).

      attention     : attn / tp   (attn_parallel="dp" → replicated, /1)
      dense prefix  : dense SwiGLU / tp
      shared experts: n_shared · P / tp
      routed experts: tp_ep  → (E/ep) · P / tp      (TP×EP: split inside group)
                      ep_all → (E/(tp·ep)) · P      (EP over all ranks)
    ``routed_experts`` overrides the local routed-expert count (e.g. expected
    touched experts for a decode step); default = all local experts (storage).
    """
    tp = max(int(tp), 1)
    t_a = 1 if attn_parallel == "dp" else tp
    ep_deg, etp = moe_expert_degree(shape, tp, ep, moe_shard)
    p = shape.ffn_weight_params_per_expert()
    n_loc = shape.n_experts // ep_deg if routed_experts is None else routed_experts
    moe = shape.n_shared_experts * p / tp + n_loc * p / etp
    dense = shape.dense_ffn_params() / tp
    L = max(shape.n_layers, 1)
    nd = shape.n_dense_layers
    ffn = (nd * dense + (L - nd) * moe) / L
    return shape.attn_weight_params_per_layer() / t_a + ffn


def mtp_card_bytes(
    shape: ModelShape,
    tp: int,
    ep: int = 1,
    *,
    attn_parallel: str = "tp",
    moe_shard: str = "tp_ep",
    n_layers: int = 1,
) -> int:
    """Stored bytes of ``n_layers`` MTP (nextn) modules on one (last-stage) card.

    One module = one transformer layer (MoE layer for MoE shapes, sharded like
    the body) + eh_proj (2H·H, sharded like attention). v0.31.
    """
    if n_layers <= 0:
        return 0
    tp = max(int(tp), 1)
    t_a = 1 if attn_parallel == "dp" else tp
    if shape.is_moe:
        ep_deg, etp = moe_expert_degree(shape, tp, ep, moe_shard)
        p = shape.ffn_weight_params_per_expert()
        ffn = shape.n_shared_experts * p / tp + (shape.n_experts // ep_deg) * p / etp
    else:
        ffn = shape.ffn_weight_params_per_expert() / tp
    params = shape.attn_weight_params_per_layer() / t_a + ffn + 2 * shape.hidden * shape.hidden / t_a
    return int(n_layers * params * shape.weight_bits / 8)


def lm_head_tp_split(tp: int, embed_policy: str, attn_parallel: str = "tp") -> int:
    """Vocab-parallel split of the LM head GEMM / weight read (v0.31).

    Attention TP: the TP group shares the same tokens → vocab-parallel head
    (each rank reads + computes its V/tp slice, even when the weights are stored
    replicated). Attention DP: each rank owns different tokens → a replicated
    head is read in full by every rank; a sharded head stays vocab-parallel.
    """
    tp = max(int(tp), 1)
    return tp if (attn_parallel != "dp" or embed_policy != "replicate") else 1


def lm_head_card_bytes(shape: ModelShape, tp: int, embed_policy: str, attn_parallel: str = "tp") -> int:
    """LM head bytes read per decode step on one (last-stage) card (v0.31)."""
    b = shape.lm_head_params() * shape.weight_bits // 8
    return b // lm_head_tp_split(tp, embed_policy, attn_parallel)


def per_card_weight_bytes(
    shape: ModelShape,
    tp: int,
    embed_policy: EmbedPolicy,
    *,
    pp: int = 1,
    ep: int = 1,
    attn_parallel: str = "tp",
    moe_shard: str = "tp_ep",
    mtp_layers: int = 0,
) -> int:
    """Weights stored on one card (capacity accounting).

    Body: PP stage holds L/pp layers; TP shards attention + dense FFN.
    MoE (v0.31 TP×EP): experts partitioned over ep groups and TP-split by tp
    inside the group (``moe_shard="tp_ep"``), or partitioned over all tp·ep
    ranks (``"ep_all"``) — see ``moe_layer_card_params``.
    attn_parallel="dp": attention weights replicated on every rank.
    Embedding (+ untied LM head): replicate or shard by tp.
    ``mtp_layers`` > 0 adds MTP modules on the last stage (spec decode on).
    """
    layers = shape.n_layers // max(pp, 1)
    if moe_sharded(shape, tp, ep, moe_shard):
        p = moe_layer_card_params(shape, tp, ep, attn_parallel=attn_parallel, moe_shard=moe_shard)
        body_card = int(layers * p * shape.weight_bits / 8)
    else:
        body = shape.body_weight_params() * shape.weight_bits // 8
        body_card = body // (tp * max(pp, 1))
        if attn_parallel == "dp" and tp > 1:
            attn_all = shape.n_layers * (
                shape.attn_weight_params_per_layer() * shape.weight_bits // 8
            )
            body_card += attn_all * (tp - 1) // (tp * max(pp, 1))
    # v0.30: untied LM head (HF tie_word_embeddings=false) stored separately
    embed = shape.embed_params() * shape.weight_bits // 8
    embed_card = embed if embed_policy == "replicate" else embed // tp
    mtp = mtp_card_bytes(shape, tp, ep, attn_parallel=attn_parallel, moe_shard=moe_shard,
                         n_layers=mtp_layers) if mtp_layers > 0 else 0
    return body_card + embed_card + mtp


def per_card_kv_bytes(
    shape: ModelShape,
    seq_len: int,
    batch: int,
    tp: int,
    *,
    pp: int = 1,
    ep: int = 1,
    attn_parallel: str = "tp",
) -> int:
    """KV cache bytes on one card (v0.29 head/latent sharding; v0.31 EP DP).

    TP shards KV **heads**: ceil(n_kv/tp)/n_kv per rank (duplication when
    tp > n_kv). MLA latent is replicated on every TP rank. attn_parallel="dp"
    partitions by batch. PP: each stage holds KV for its layers (/pp).
    v0.31: MoE ep>1 → attention data-parallel across ep groups → KV also
    partitioned by batch over ep. See ``kv_card_fraction``.
    """
    num, den, _rep, _why = kv_card_fraction(
        shape, tp, pp=pp, attn_parallel=attn_parallel, batch=batch, ep=ep
    )
    return shape.kv_cache_bytes(seq_len, batch) * num // den


def spec_mtp_layers(shape: ModelShape, tp_cfg: TPConfig) -> int:
    """MTP modules stored when spec decode uses the MTP head (v0.31): the
    model's ``n_mtp_layers`` (≥1 assumed when the config has none)."""
    if tp_cfg.spec_k <= 0 or tp_cfg.spec_draft != "mtp":
        return 0
    return max(int(shape.n_mtp_layers), 1)


def capacity_check(
    shape: ModelShape,
    mem: ExternalMemory,
    tp_cfg: TPConfig,
    *,
    seq_len: int,
    batch: int = 1,
) -> tuple[bool, int, int, str]:
    """Return (oom, per_card_bytes, capacity, detail)."""
    w = per_card_weight_bytes(
        shape, tp_cfg.tp, tp_cfg.embed_policy, pp=tp_cfg.pp, ep=tp_cfg.ep,
        attn_parallel=tp_cfg.attn_parallel, moe_shard=tp_cfg.moe_shard,
        mtp_layers=spec_mtp_layers(shape, tp_cfg),
    )
    kv = per_card_kv_bytes(
        shape, seq_len, batch, tp_cfg.tp, pp=tp_cfg.pp, ep=tp_cfg.ep,
        attn_parallel=tp_cfg.attn_parallel,
    )
    _n, _d, kv_rep, _why = kv_card_fraction(
        shape, tp_cfg.tp, pp=tp_cfg.pp, attn_parallel=tp_cfg.attn_parallel, batch=batch,
        ep=tp_cfg.ep,
    )
    used = w + kv
    cap = mem.capacity_bytes
    oom = used > cap
    detail = (
        f"per_card W={w/2**30:.3f}GB + KV={kv/2**30:.3f}GB = {used/2**30:.3f}GB "
        f"vs cap={cap/2**30:.1f}GB (GB=2^30 B) tp={tp_cfg.tp} pp={tp_cfg.pp} ep={tp_cfg.ep} "
        f"cards={tp_cfg.n_cards} embed={tp_cfg.embed_policy} "
        f"attn={tp_cfg.attn_parallel} moe={tp_cfg.moe_shard} kv_rep={kv_rep:g} "
        f"{'OOM' if oom else 'fits'} [assumed]"
    )
    return oom, used, cap, detail


# ---------------------------------------------------------------------------
# Scale-up phase / system result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleupPhaseResult:
    """Per-phase multi-card roofline with C2C / PP / KV fabric."""

    single: PhaseResult
    flops_card: int
    weight_bytes_card: int
    kv_bytes_card: int
    act_bytes_card: int
    compute_cycles_card: float
    collective_bytes: int
    pp_act_bytes: int
    a2a_bytes: int
    remote_kv_bytes: int
    local_kv_bytes: int
    t_compute_s: float
    t_memory_s: float
    t_c2c_s: float
    t_pp_act_s: float
    t_a2a_s: float
    t_fabric_s: float
    t_stage_s: float
    bubble_frac: float
    t_roofline_s: float
    t_sum_s: float
    wall: str
    contention: ContentionResult
    # v0.29 exposed sync + KV replication (defaults = ≤0.28 behaviour)
    t_sync_s: float = 0.0
    n_sync: int = 0
    kv_replication: float = 1.0
    kv_note: str = ""
    # v0.31 decode micro-batching / spec decode / LM head
    mb_eff: int = 1               # decode micro-batches in flight (PP)
    batch_eff: int = 0            # micro-batch size this phase was evaluated at
    tokens_per_seq: int = 1       # 1 + spec_k for a verify step
    t_draft_s: float = 0.0        # draft cost per tick (spec decode)
    t_tick_s: float = 0.0         # t_stage + t_draft (one pipeline tick)
    head_bytes_card: int = 0      # LM head bytes read per step on one card
    moe_tokens: float = 0.0       # tokens routed through the MoE this tick
    experts_read_card: float = 0.0  # expected local routed experts read (decode)


@dataclass(frozen=True)
class ScaleupResult:
    shape: ModelShape
    npu: NPUConfig
    sram: SRAMConfig
    mem: ExternalMemory
    cfg: EvalConfig
    tp_cfg: TPConfig
    prefill: ScaleupPhaseResult
    decode: ScaleupPhaseResult
    oom: bool
    per_card_bytes: int
    capacity_detail: str
    single_card: InferenceResult
    t_kv_xfer_s: float = 0.0
    kv_xfer_bytes: int = 0
    # v0.31 speculative decoding: expected tokens emitted per verify step
    spec_tokens_per_step: float = 1.0

    @property
    def ttft_s(self) -> float:
        t = self.prefill.t_roofline_s
        if self.tp_cfg.add_xfer_to_ttft:
            t = t + self.t_kv_xfer_s
        return t

    @property
    def tpot_step_s(self) -> float:
        """Time between decode steps seen by one user (pipeline traversal incl.)."""
        return self.decode.t_roofline_s

    @property
    def tpot_s(self) -> float:
        """Effective per-token latency: step time / expected tokens per step."""
        return self.decode.t_roofline_s / max(self.spec_tokens_per_step, 1e-12)

    def summary_lines(self) -> list[str]:
        tp = self.tp_cfg.tp
        pp = self.tp_cfg.pp
        c2c = self.tp_cfg.c2c
        fab = self.tp_cfg.fabric_link()
        fab_s = fab.summary() if fab else "none"
        lines = [
            f"=== SCALE-UP TP={tp} PP={pp} EP={self.tp_cfg.ep} "
            f"cards={self.tp_cfg.n_cards} | "
            f"{self.shape.name} | {self.mem.kind} | "
            f"sku={self.npu.sku_name or 'custom'} | "
            f"C2C={c2c.name} {c2c.effective_bw_GBps:.0f}GB/s "
            f"hide={self.tp_cfg.c2c_hide:.2f}[assumed] | "
            f"kv_fabric={fab_s} remote_frac={self.tp_cfg.remote_kv_frac:.2f} | "
            f"mb={self.tp_cfg.mb} embed={self.tp_cfg.embed_policy} ===",
            f"  sharding: TP col/row Megatron; PP layers≈L/{pp}; "
            f"EP experts E/{self.tp_cfg.ep}; "
            f"bubble_dec={self.decode.bubble_frac:.3f} "
            f"bubble_pref={self.prefill.bubble_frac:.3f} "
            f"[decode mb_eff={self.decode.mb_eff} (micro-batch {self.decode.batch_eff}); "
            f"TPOT_step=max(mb,pp)·tick] moe={self.tp_cfg.moe_shard} "
            f"spec_k={self.tp_cfg.spec_k} E[tok/step]={self.spec_tokens_per_step:.3f}",
            f"  capacity: {self.capacity_detail}",
            f"  TTFT: {self.ttft_s*1e3:.4f} ms [{self.prefill.wall}-bound]  "
            f"c={self.prefill.t_compute_s*1e3:.4f} "
            f"dram={self.prefill.t_memory_s*1e3:.4f} "
            f"c2c={self.prefill.t_c2c_s*1e3:.4f} "
            f"pp_act={self.prefill.t_pp_act_s*1e3:.4f} "
            f"a2a={self.prefill.t_a2a_s*1e3:.4f} "
            f"fab={self.prefill.t_fabric_s*1e3:.4f} ms  "
            f"coll={self.prefill.collective_bytes} a2a_B={self.prefill.a2a_bytes}",
            f"  TPOT: {self.tpot_s*1e3:.4f} ms/tok [{self.decode.wall}-bound]  "
            f"c={self.decode.t_compute_s*1e3:.4f} "
            f"dram={self.decode.t_memory_s*1e3:.4f} "
            f"c2c={self.decode.t_c2c_s*1e3:.4f} "
            f"pp_act={self.decode.t_pp_act_s*1e3:.4f} "
            f"a2a={self.decode.t_a2a_s*1e3:.4f} "
            f"fab={self.decode.t_fabric_s*1e3:.4f} ms  "
            f"remote_KV={self.decode.remote_kv_bytes} "
            f"coll={self.decode.collective_bytes} a2a_B={self.decode.a2a_bytes}",
            f"  t_kv_xfer (PD one-shot): {self.t_kv_xfer_s*1e3:.4f} ms "
            f"bytes={self.kv_xfer_bytes} "
            f"{'(added to TTFT)' if self.tp_cfg.add_xfer_to_ttft else '(separate metric)'}",
            f"  per-card decode: W={self.decode.weight_bytes_card} "
            f"KV={self.decode.kv_bytes_card} Act={self.decode.act_bytes_card}  "
            f"FLOPs/card={self.decode.flops_card}",
        ]
        return lines


def _classify_wall(vals: dict[str, float]) -> str:
    top = max(vals, key=vals.get)  # type: ignore[arg-type]
    ordered = sorted(vals.values(), reverse=True)
    if ordered[0] > 0 and len(ordered) > 1 and (ordered[0] - ordered[1]) / ordered[0] < 0.05:
        return "balanced"
    return top


def _ep_weight_stream_card(
    shape: ModelShape,
    single_w: int,
    *,
    tp: int,
    pp: int,
    ep: int,
    resident_layers: int,
    steady_decode: bool,
    attn_parallel: str = "tp",
    moe_shard: str = "tp_ep",
    tokens: float = 1.0,
) -> int:
    """Per-card body weight DRAM bytes (LM head excluded; added by caller).

    Dense / unsharded MoE: single_card_body_W / (tp*pp) (legacy; the single-card
    stream is already batch-aware for MoE in v0.31).
    MoE expert-sharded (v0.31 TP×EP / EP-all): per layer
        attn/t_a + shared/tp + E_hit_local(tokens) · P / expert_tp
    with E_hit_local = expected touched local experts for this step's tokens
    (「假设」uniform routing). Prefill cold-start loads all stored local experts.
    """
    if not moe_sharded(shape, tp, ep, moe_shard):
        base = single_w // (tp * max(pp, 1))
        if attn_parallel == "dp" and tp > 1 and single_w > 0:
            # attention weights replicated on every rank: each card streams the
            # full attention of its L/pp layers instead of 1/tp of it.
            attn_b = shape.attn_weight_params_per_layer() * shape.weight_bits // 8
            layers_here = shape.n_layers // max(pp, 1)
            base += layers_here * attn_b * (tp - 1) // tp
        return base

    L = shape.n_layers
    layers_here = L // max(pp, 1)
    R = max(0, min(resident_layers, L))
    wb = shape.weight_bits
    if steady_decode:
        ep_deg, _etp = moe_expert_degree(shape, tp, ep, moe_shard)
        hit = shape.experts_touched(tokens, shape.n_experts // ep_deg)
        per_layer = int(moe_layer_card_params(
            shape, tp, ep, attn_parallel=attn_parallel, moe_shard=moe_shard,
            routed_experts=hit) * wb / 8)
        if R >= L and L > 0:
            return 0
        if R >= 1:
            miss = max(layers_here - (R // max(pp, 1)), 0)
            return miss * per_layer
        return layers_here * per_layer
    # Prefill cold-start: load stored local experts once for layers_here
    stored = int(moe_layer_card_params(
        shape, tp, ep, attn_parallel=attn_parallel, moe_shard=moe_shard) * wb / 8)
    return layers_here * stored


def _scale_phase(
    single_phase: PhaseResult,
    *,
    shape: ModelShape,
    mem: ExternalMemory,
    cfg: EvalConfig,
    tp_cfg: TPConfig,
    batch: int,
    seq: int,
    phase: str,
    decode_mb_eff: int = 1,
    t_draft_s: float = 0.0,
    draft_frac_of_stage: float = 0.0,
) -> ScaleupPhaseResult:
    """One phase on one card of the slowest stage.

    ``batch`` is the micro-batch this phase is evaluated at (decode: ceil(B/mb)).
    Decode (v0.31): with ``mb = decode_mb_eff`` micro-batches cycling through
    ``pp`` stages a user's token interval is ``max(mb, pp) · t_tick`` with
    ``t_tick = t_stage(micro-batch) + t_draft``; mb=1 → ``pp · t_stage`` (≤0.30).
    Prefill keeps ``bubble = (pp-1)/(mb+pp-1)``.
    """
    tp = tp_cfg.tp
    pp = tp_cfg.pp
    ep = tp_cfg.ep
    if ep > 1 and not shape.is_moe:
        raise ValueError("ep>1 requires a MoE shape (n_experts>1)")
    ep_deg, _etp = moe_expert_degree(shape, tp, ep, tp_cfg.moe_shard)
    sharded = moe_sharded(shape, tp, ep, tp_cfg.moe_shard)
    if sharded and shape.n_experts % ep_deg != 0:
        raise ValueError(
            f"n_experts={shape.n_experts} must be divisible by the expert-parallel "
            f"degree {ep_deg} (ep={ep}, moe_shard={tp_cfg.moe_shard})"
        )

    tr = single_phase.traffic
    q = max(int(getattr(tr, "tokens_per_seq", 1)), 1)
    # KV: head / latent / batch sharding (v0.29; v0.31 attention DP over ep)
    kv_num, kv_den, kv_rep, kv_why = kv_card_fraction(
        shape, tp, pp=pp, attn_parallel=tp_cfg.attn_parallel, batch=batch, ep=ep
    )
    # Compute / expert work: MoE EP also splits expert FLOPs
    n_compute = tp * pp * ep if (shape.is_moe and ep > 1) else (tp * pp)
    steady = phase == "decode"
    tokens = batch * seq * q  # tokens through this stage per tick

    # v0.31: LM head runs on the last stage only (slowest card = last stage);
    # compute = its token share (attention-DP degree) × vocab slice (tp under
    # attention TP); weight read per ``lm_head_card_bytes``.
    h_f = int(getattr(tr, "head_flops", 0))
    h_c = float(getattr(tr, "head_cycles", 0.0))
    t_a_h = tp if tp_cfg.attn_parallel != "dp" else 1  # tokens shared by the TP group
    # 「假设」PP partition balanced around the head (last stage holds fewer
    # layers) → head cost amortised over the pp stages, like the body.
    head_div = attn_dp_degree(shape, tp, ep, tp_cfg.attn_parallel) * t_a_h * max(pp, 1)
    flops_card = (tr.flops - h_f) // n_compute + h_f // head_div
    head_total = int(getattr(tr, "head_weight_bytes", 0))
    head_card = (lm_head_card_bytes(shape, tp, tp_cfg.embed_policy, tp_cfg.attn_parallel)
                 // max(pp, 1) if head_total > 0 else 0)
    w_card = _ep_weight_stream_card(
        shape,
        tr.weight_bytes_dram - head_total,
        tp=tp,
        pp=pp,
        ep=ep,
        resident_layers=tr.resident_layers,
        steady_decode=steady,
        attn_parallel=tp_cfg.attn_parallel,
        moe_shard=tp_cfg.moe_shard,
        tokens=tokens,
    ) + head_card
    experts_read = 0.0
    if shape.is_moe and steady:
        experts_read = shape.experts_touched(tokens, shape.n_experts // ep_deg)
    kv_card = tr.kv_bytes_dram * kv_num // kv_den
    kv_read_card = tr.kv_bytes_read * kv_num // kv_den
    kv_write_card = tr.kv_bytes_write * kv_num // kv_den
    act_card = tr.act_bytes_dram
    cycles_card = (tr.compute_cycles - h_c) / n_compute + h_c / head_div

    # Remote vs local KV (decode remote read; writes stay local by default)
    frac = tp_cfg.remote_kv_frac if tp_cfg.kv_fabric != "none" else 0.0
    remote_kv = int(frac * kv_read_card)
    local_kv_read = kv_read_card - remote_kv
    local_kv_total = local_kv_read + kv_write_card

    card_traffic = PhaseTraffic(
        name=tr.name,
        flops=flops_card,
        weight_bytes_dram=w_card,
        kv_bytes_dram=local_kv_total,
        act_bytes_dram=act_card,
        compute_cycles=cycles_card,
        mean_gemm_util=tr.mean_gemm_util,
        m_eff=tr.m_eff,
        weight_path=tr.weight_path,
        partitions=tr.partitions,
        kv_bytes_read=local_kv_read,
        kv_bytes_write=kv_write_card,
        double_buffer_eligible=tr.double_buffer_eligible,
        layer_weight_bytes=(
            tr.layer_weight_bytes // n_compute
            if n_compute > 1
            else tr.layer_weight_bytes
        ),
        resident_layers=tr.resident_layers,
        head_weight_bytes=head_card,
        tokens_per_seq=q,
        head_flops=h_f // head_div,
        head_cycles=h_c / head_div,
    )
    cont = contend(card_traffic, mem)
    t_comp = effective_compute_time_s(cycles_card, cfg)
    t_mem = _mem_time_with_hide(
        cont,
        cfg.contention_mode,
        hide_factor=cfg.weight_hide_factor,
        double_buffer_eligible=card_traffic.double_buffer_eligible,
    )

    # TP collectives (layers on this stage only). v0.31: with MoE ep>1 the ep
    # groups run attention data-parallel → each TP group reduces tokens/ep.
    dp_ep = ep if (shape.is_moe and ep > 1) else 1
    tok_group = math.ceil(batch * q / dp_ep)
    coll = collectives_bytes_per_phase(shape, tok_group, seq, tp, pp=pp)
    bw = tp_cfg.c2c.effective_bw_Bps()
    hide = tp_cfg.c2c_hide
    t_c2c_raw = coll / bw if bw > 0 else 0.0
    t_c2c = t_c2c_raw * (1.0 - hide)

    # PP activation sends
    pp_bytes = pp_activation_send_bytes(shape, batch * q, seq, pp)
    pp_bw = tp_cfg.interstage_link().effective_bw_Bps()
    t_pp_act = (pp_bytes / pp_bw) if (pp > 1 and pp_bw > 0) else 0.0

    # MoE all-to-all dispatch+combine (v0.31 per-rank share, MoE layers only)
    layers_here = shape.n_layers // max(pp, 1)
    moe_layers_here = (
        round(layers_here * shape.n_moe_layers / max(shape.n_layers, 1)) if shape.is_moe else 0
    )
    a2a = moe_a2a_bytes_per_rank(
        shape, tokens, ep_deg if sharded else 1,
        moe_layers=moe_layers_here, e_local_adjust=tp_cfg.ep_e_local_adjust,
    )
    t_a2a_raw = a2a / bw if (bw > 0 and a2a > 0) else 0.0
    t_a2a = t_a2a_raw * (1.0 - hide)

    # KV fabric remote read
    fab = tp_cfg.fabric_link()
    if fab is not None and remote_kv > 0:
        t_fabric = fab.transfer_time_s(remote_kv)
    else:
        t_fabric = 0.0

    # v0.29 exposed sync: per-collective latency α is NOT hidden by max()
    n_sync = sync_collectives_per_token(
        shape.n_layers, tp=tp, pp=pp, ep=ep_deg if sharded else 1, is_moe=shape.is_moe
    )
    t_sync = sync_time_s(n_sync, tp_cfg.c2c_latency_us, tp_cfg.sync_overlap)

    # Stage time before bubble: max of BW contributors + exposed sync
    t_stage = max(t_comp, t_mem, t_c2c, t_pp_act, t_a2a, t_fabric) + t_sync
    mb_eff = 1
    t_draft = 0.0
    if phase == "decode":
        mb_eff = max(int(decode_mb_eff), 1) if pp > 1 else 1
        t_draft = float(t_draft_s) + float(draft_frac_of_stage) * t_stage
        t_tick = t_stage + t_draft
        slots = max(mb_eff, pp)
        bubble = 1.0 - mb_eff / slots if pp > 1 else 0.0
        t_roof = t_tick * slots
    else:
        t_tick = t_stage
        bubble = tp_cfg.bubble_fraction(phase=phase)
        if bubble >= 1.0 - 1e-15:
            t_roof = t_stage * float(pp)
        else:
            t_roof = t_stage / (1.0 - bubble)

    wall = _classify_wall(
        {
            "compute": t_comp,
            "memory": t_mem,
            "c2c": t_c2c,
            "pp_act": t_pp_act,
            "a2a": t_a2a,
            "fabric": t_fabric,
            "sync": t_sync,
        }
    )
    if pp > 1 and bubble > 0.05 and t_roof > t_tick * 1.05:
        if wall in ("compute", "memory") and t_roof > max(t_comp, t_mem) * 1.2:
            wall = "bubble"

    return ScaleupPhaseResult(
        single=single_phase,
        flops_card=flops_card,
        weight_bytes_card=w_card,
        kv_bytes_card=kv_card,
        act_bytes_card=act_card,
        compute_cycles_card=cycles_card,
        collective_bytes=coll,
        pp_act_bytes=pp_bytes,
        a2a_bytes=a2a,
        remote_kv_bytes=remote_kv,
        local_kv_bytes=local_kv_total,
        t_compute_s=t_comp,
        t_memory_s=t_mem,
        t_c2c_s=t_c2c,
        t_pp_act_s=t_pp_act,
        t_a2a_s=t_a2a,
        t_fabric_s=t_fabric,
        t_stage_s=t_stage,
        bubble_frac=bubble,
        t_roofline_s=t_roof,
        t_sum_s=t_comp + t_mem + t_c2c + t_pp_act + t_a2a + t_fabric + t_sync,
        wall=wall,
        contention=cont,
        t_sync_s=t_sync,
        n_sync=n_sync,
        kv_replication=kv_rep,
        kv_note=kv_why,
        mb_eff=mb_eff,
        batch_eff=int(batch),
        tokens_per_seq=q,
        t_draft_s=t_draft,
        t_tick_s=t_tick,
        head_bytes_card=head_card,
        moe_tokens=float(tokens) if shape.is_moe else 0.0,
        experts_read_card=experts_read,
    )


def evaluate_scaleup(
    shape: ModelShape,
    npu: NPUConfig,
    sram: SRAMConfig,
    mem: ExternalMemory,
    cfg: EvalConfig | None = None,
    tp_cfg: TPConfig | None = None,
) -> ScaleupResult:
    """Multi-card TP (+ optional PP) + C2C (+ optional KV fabric) evaluation.

    When tp=1, pp=1, ep=1, kv_fabric=none: matches single-card within
    int-division tolerance (FLOPs/bytes // 1).
    """
    cfg = cfg or EvalConfig()
    tp_cfg = tp_cfg or TPConfig()

    single = evaluate_inference(shape, npu, sram, mem, cfg)
    pref = _scale_phase(
        single.prefill,
        shape=shape,
        mem=mem,
        cfg=cfg,
        tp_cfg=tp_cfg,
        batch=cfg.batch,
        seq=cfg.prompt_len,
        phase="prefill",
    )
    # v0.31 decode: PP micro-batches (B split into mb micro-batches of b) and
    # speculative verify width q = 1 + spec_k.
    B = max(int(cfg.batch), 1)
    mb = tp_cfg.decode_mb_eff(B)
    b = math.ceil(B / mb)
    q = 1 + int(tp_cfg.spec_k)
    if b == cfg.batch and q == 1:
        dec_single = single.decode
    else:
        dec_single = evaluate_decode_phase(
            shape, npu, sram, mem, cfg, batch=b, tokens_per_seq=q
        )
    t_draft = 0.0
    draft_frac = 0.0
    if tp_cfg.spec_k > 0:
        if tp_cfg.spec_draft == "mtp":
            t_draft = tp_cfg.spec_k * _mtp_draft_step_s(shape, npu, sram, mem, cfg, tp_cfg, b)
        else:
            draft_frac = tp_cfg.spec_k * tp_cfg.spec_draft_frac
    dec = _scale_phase(
        dec_single,
        shape=shape,
        mem=mem,
        cfg=cfg,
        tp_cfg=tp_cfg,
        batch=b,
        seq=1,
        phase="decode",
        decode_mb_eff=mb,
        t_draft_s=t_draft,
        draft_frac_of_stage=draft_frac,
    )

    # PD disagg one-shot KV transfer (full cache at decode ctx)
    fab = tp_cfg.fabric_link()
    kv_xfer_bytes = 0
    t_kv_xfer = 0.0
    if tp_cfg.pd_kv_xfer and fab is not None:
        # Full KV after prefill (use decode_seq_len as populated length)
        kv_xfer_bytes = shape.kv_cache_bytes(cfg.decode_seq_len, cfg.batch)
        t_kv_xfer = fab.transfer_time_s(kv_xfer_bytes)

    oom, used, _cap, detail = capacity_check(
        shape, mem, tp_cfg, seq_len=cfg.decode_seq_len, batch=cfg.batch
    )
    return ScaleupResult(
        shape=shape,
        npu=npu,
        sram=sram,
        mem=mem,
        cfg=cfg,
        tp_cfg=tp_cfg,
        prefill=pref,
        decode=dec,
        oom=oom,
        per_card_bytes=used,
        capacity_detail=detail,
        single_card=single,
        t_kv_xfer_s=t_kv_xfer,
        kv_xfer_bytes=kv_xfer_bytes,
        spec_tokens_per_step=tp_cfg.spec_tokens_per_step(),
    )


def _mtp_draft_step_s(
    shape: ModelShape,
    npu: NPUConfig,
    sram: SRAMConfig,
    mem: ExternalMemory,
    cfg: EvalConfig,
    tp_cfg: TPConfig,
    batch: int,
) -> float:
    """One MTP draft token: one transformer layer (+ shared LM head) at M=batch
    on the last stage's tp·ep cards, same sharding as the body (v0.31)."""
    mtp_shape = replace(shape, n_layers=1, n_dense_layers=0, name=f"{shape.name}@mtp")
    d_cfg = replace(tp_cfg, pp=1, spec_k=0, decode_mb=0, kv_fabric="none")
    ph = evaluate_decode_phase(mtp_shape, npu, sram, mem, cfg, batch=batch, tokens_per_seq=1)
    dp = _scale_phase(ph, shape=mtp_shape, mem=mem, cfg=cfg, tp_cfg=d_cfg,
                      batch=batch, seq=1, phase="decode")
    return dp.t_stage_s


# Alias kept for clarity in newer call sites
ParallelConfig = TPConfig
evaluate_parallel = evaluate_scaleup


# ---------------------------------------------------------------------------
# Video / protein multi-card TP / PP / EP + communication model (v0.20)
# ---------------------------------------------------------------------------
#
# Reuses LLM scale-up patterns (ring collectives, PP bubble, C2C hide) with
# domain-honest caveats. No invented silicon numbers.
#
# TP  — weight shard; activation all-reduce / all-gather per DiT block or
#       protein encoder layer. Collective volume ~ B·T·H·act_bytes (video) or
#       B·L·H·act_bytes (protein). Algo: ``ring`` (default Megatron-style
#       ``2*(tp-1)/tp * V``) or ``tree`` (assumed recursive-doubling /
#       reduce+bcast: ``2*ceil(log2(tp))*V`` — BW-oriented analytical alt).
# PP  — microbatch pipeline across layer stages (L/pp). Bubble:
#         bubble = (pp-1)/(mb+pp-1)
#       Video denoise steps are sequential (decode-like) → default mb=1 gives
#       bubble=(pp-1)/pp; denoise pipelines poorly across steps (documented).
#       Protein forward may use mb≥pp to cut bubble.
# EP  — only if MoE experts exist on the shape. DiT / protein shapes have none
#       → ep>1 is accepted as a no-op (cards still satisfy tp*pp*ep==chips)
#       with an honesty note; a2a_bytes=0.
# ---------------------------------------------------------------------------


CollectiveAlgo = Literal["ring", "tree"]


def tree_allreduce_bytes(volume: int, tp: int) -> int:
    """Per-rank bytes for one tree / recursive-doubling style all-reduce.

    Assumed analytical alternative to ring: ``2 * ceil(log2(tp)) * volume``.
    Returns 0 when tp <= 1. Not a calibrated NoC model — labeled assumed.
    """
    if tp <= 1 or volume <= 0:
        return 0
    return int(2 * math.ceil(math.log2(tp)) * volume)


def collective_bytes_one(volume: int, tp: int, algo: CollectiveAlgo = "ring") -> int:
    """One collective (all-reduce / all-gather) per-rank byte volume."""
    if algo == "ring":
        return ring_allreduce_bytes(volume, tp)
    if algo == "tree":
        return tree_allreduce_bytes(volume, tp)
    raise ValueError(f"collective algo must be 'ring' or 'tree', got {algo!r}")


@dataclass(frozen=True)
class DomainTPResult:
    """TP/PP/EP overlay for non-LLM domains (DiT / protein).

    Mirrors LLM ``ScaleupPhaseResult`` spirit with domain honesty labels:
      - weights + FLOPs / cycles sharded across tp*pp (EP no-op without MoE)
      - TP collectives on residual-stream activations (layers on this PP stage)
      - PP inter-stage activation sends + analytical bubble
      - wall = max(compute, memory, c2c, pp_act); capacity W/(tp*pp) + other
    """

    tp: int
    pp: int
    ep: int
    flops_card: int
    weight_bytes_card: int
    act_bytes_card: int
    kv_bytes_card: int
    collective_bytes: int
    pp_act_bytes: int
    a2a_bytes: int
    t_compute_s: float
    t_memory_s: float
    t_c2c_s: float
    t_pp_act_s: float
    t_a2a_s: float
    t_stage_s: float
    bubble_frac: float
    t_roofline_s: float
    t_sum_s: float
    wall: str
    capacity_needed_bytes: int
    oom: bool
    mean_gemm_util: float
    collective_algo: str
    assumptions: tuple[str, ...]
    t_sync_s: float = 0.0  # v0.29 exposed per-collective latency
    n_sync: int = 0


def scale_domain_parallel(
    single: PhaseResult,
    *,
    mem: ExternalMemory,
    cfg: EvalConfig,
    tp: int,
    pp: int = 1,
    ep: int = 1,
    mb: int = 1,
    act_volume_bytes: int,
    n_layers: int,
    capacity_weight_bytes: int,
    capacity_other_bytes: int = 0,
    shard_other: bool = False,
    c2c: C2CLink | None = None,
    c2c_hide: float = 0.0,
    collectives_per_layer: int = COLLECTIVES_PER_LAYER,
    collective_algo: CollectiveAlgo = "ring",
    has_moe_experts: bool = False,
    phase: str = "forward",
    domain: str = "domain",
    c2c_latency_us: float = DEFAULT_C2C_LATENCY_US,
    sync_overlap: float = DEFAULT_SYNC_OVERLAP,
) -> DomainTPResult:
    """Apply TP/PP/EP scaling to a single-card video/protein PhaseResult.

    ``tp*pp*ep`` is the caller's chip_count constraint (enforced upstream).
    When ``tp=pp=ep=1``: identical to single-card (collectives=0, bubble=0).

    ``phase``:
      - ``\"denoise\"`` / ``\"decode\"``: sequential dependency (video denoise) —
        bubble uses mb as given (typically 1 → heavy bubble).
      - ``\"forward\"`` / ``\"prefill\"``: protein-like; mb≥pp cuts bubble.
    """
    if tp < 1 or pp < 1 or ep < 1:
        raise ValueError(f"tp/pp/ep must be >= 1, got {tp}/{pp}/{ep}")
    if mb < 1:
        raise ValueError(f"mb must be >= 1, got {mb}")
    if collective_algo not in ("ring", "tree"):
        raise ValueError(f"collective_algo must be ring|tree, got {collective_algo!r}")
    c2c = c2c or C2C_400
    hide = max(0.0, min(float(c2c_hide), 1.0))
    assumptions: list[str] = []

    tr = single.traffic
    n_shard = tp * pp  # EP does not shard compute without MoE experts

    # --- EP honesty ---
    ep_active = has_moe_experts and ep > 1
    if ep > 1 and not has_moe_experts:
        assumptions.append(
            f"{domain}: ep={ep} accepted (tp*pp*ep chip constraint) but "
            f"no MoE experts on shape → EP no-op (a2a=0; compute shard tp*pp only)"
        )
    elif ep_active:
        assumptions.append(
            f"{domain}: EP={ep} MoE expert shard active (rare for DiT/protein)"
        )
        n_shard = tp * pp * ep
    elif ep == 1:
        assumptions.append(
            f"{domain}: ep=1 (EP unused — typical for DiT/protein; no experts)"
        )

    if tp == 1 and pp == 1 and not ep_active:
        needed = capacity_weight_bytes + capacity_other_bytes
        assumptions.append(
            f"{domain} chip_count=1 → single-card roofline (no TP/PP collectives)"
        )
        return DomainTPResult(
            tp=1,
            pp=1,
            ep=ep,
            flops_card=tr.flops,
            weight_bytes_card=tr.weight_bytes_dram,
            act_bytes_card=tr.act_bytes_dram,
            kv_bytes_card=tr.kv_bytes_dram,
            collective_bytes=0,
            pp_act_bytes=0,
            a2a_bytes=0,
            t_compute_s=single.t_compute_s,
            t_memory_s=single.t_memory_s,
            t_c2c_s=0.0,
            t_pp_act_s=0.0,
            t_a2a_s=0.0,
            t_stage_s=max(single.t_compute_s, single.t_memory_s),
            bubble_frac=0.0,
            t_roofline_s=single.t_roofline_s,
            t_sum_s=single.t_sum_s,
            wall=single.wall,
            capacity_needed_bytes=needed,
            oom=needed > mem.capacity_bytes,
            mean_gemm_util=tr.mean_gemm_util,
            collective_algo=collective_algo,
            assumptions=tuple(assumptions),
        )

    layers_here = n_layers // max(pp, 1)
    assumptions.append(
        f"{domain} multi-card: tp={tp} pp={pp} ep={ep} → shard={n_shard} "
        f"(weight/FLOP /{n_shard}); layers/stage≈{layers_here} "
        f"(of {n_layers}); collective={collective_algo}"
    )

    # --- TP activation collectives (layers on this PP stage) ---
    if collective_algo == "ring":
        algo_note = (
            f"ring all-reduce: 2*(tp-1)/tp * V per collective "
            f"(Megatron-style; V=act residual stream)"
        )
    else:
        algo_note = (
            f"tree all-reduce (assumed): 2*ceil(log2(tp))*V per collective "
            f"— analytical alt to ring; not a calibrated NoC"
        )
    assumptions.append(
        f"TP activation collectives: {collectives_per_layer}×/layer × "
        f"{layers_here} layers on volume={act_volume_bytes} B; {algo_note}; "
        f"C2C={c2c.name} {c2c.effective_bw_GBps:.0f} GB/s hide={hide:.2f} [assumed]"
    )

    # --- PP bubble honesty ---
    bubble = pipeline_bubble_fraction(pp, mb) if pp > 1 else 0.0
    if pp > 1:
        if phase in ("denoise", "decode"):
            assumptions.append(
                f"PP bubble={bubble:.3f} with mb={mb}: video denoise / sequential "
                f"steps are decode-like — dependency-bound across denoise steps; "
                f"pipelines poorly (bubble≈(pp-1)/pp when mb=1) [assumed analytical]"
            )
        else:
            assumptions.append(
                f"PP bubble={bubble:.3f} with mb={mb}: "
                f"bubble=(pp-1)/(mb+pp-1); protein/forward may raise mb≥pp "
                f"to cut bubble [assumed crude schedule]"
            )
        assumptions.append(
            f"PP inter-stage acts: (pp-1)×V = {(pp - 1) * act_volume_bytes} B "
            f"on C2C (same link as TP unless overridden) [assumed]"
        )

    w_card = tr.weight_bytes_dram // n_shard
    flops_card = tr.flops // n_shard
    cycles_card = tr.compute_cycles / n_shard
    act_card = tr.act_bytes_dram
    # other (pair / KV-like): shard across tp*pp when requested (crude)
    other_div = n_shard if shard_other else 1
    kv_card = tr.kv_bytes_dram // other_div

    card_traffic = PhaseTraffic(
        name=tr.name,
        flops=flops_card,
        weight_bytes_dram=w_card,
        kv_bytes_dram=kv_card,
        act_bytes_dram=act_card,
        compute_cycles=cycles_card,
        mean_gemm_util=tr.mean_gemm_util,
        m_eff=tr.m_eff,
        weight_path=tr.weight_path,
        partitions=tr.partitions,
        kv_bytes_read=tr.kv_bytes_read // other_div,
        kv_bytes_write=tr.kv_bytes_write // other_div,
        double_buffer_eligible=tr.double_buffer_eligible,
        layer_weight_bytes=(
            tr.layer_weight_bytes // n_shard if n_shard > 1 else tr.layer_weight_bytes
        ),
        resident_layers=tr.resident_layers,
    )
    cont = contend(card_traffic, mem)
    t_comp = effective_compute_time_s(cycles_card, cfg)
    t_mem = _mem_time_with_hide(
        cont,
        cfg.contention_mode,
        hide_factor=cfg.weight_hide_factor,
        double_buffer_eligible=card_traffic.double_buffer_eligible,
    )

    # TP collectives
    per = collective_bytes_one(act_volume_bytes, tp, collective_algo)
    coll = layers_here * collectives_per_layer * per
    bw = c2c.effective_bw_Bps()
    t_c2c_raw = coll / bw if bw > 0 else 0.0
    t_c2c = t_c2c_raw * (1.0 - hide)

    # PP activation sends
    pp_bytes = (pp - 1) * act_volume_bytes if pp > 1 else 0
    t_pp_act = (pp_bytes / bw) if (pp > 1 and bw > 0) else 0.0

    # EP A2A — only when MoE experts present
    a2a = 0
    t_a2a = 0.0
    if ep_active:
        # Domain shapes rarely MoE; keep hook for completeness
        factor = 1.0  # no top_k on VideoShape/ProteinShape
        a2a = int(2 * (ep - 1) / ep * act_volume_bytes * factor) * layers_here
        t_a2a_raw = a2a / bw if bw > 0 else 0.0
        t_a2a = t_a2a_raw * (1.0 - hide)

    # v0.29 exposed sync (per-collective α; assumed) — outside the max()
    n_sync = (collectives_per_layer * layers_here if tp > 1 else 0) + (
        2 * layers_here if ep_active else 0
    ) + (1 if pp > 1 else 0)
    ov = max(0.0, min(float(sync_overlap), 1.0))
    t_sync = sync_time_s(n_sync, float(c2c_latency_us), ov)
    if t_sync > 0:
        assumptions.append(
            f"exposed sync: {n_sync} collectives × α={c2c_latency_us:g} µs × "
            f"(1−overlap {ov:g}) = {t_sync*1e6:.1f} µs per forward [assumed α]"
        )

    t_stage = max(t_comp, t_mem, t_c2c, t_pp_act, t_a2a) + t_sync
    if bubble >= 1.0 - 1e-15:
        t_roof = t_stage * float(pp)
    else:
        t_roof = t_stage / (1.0 - bubble) if bubble > 0 else t_stage

    wall = _classify_wall(
        {
            "compute": t_comp,
            "memory": t_mem,
            "c2c": t_c2c,
            "pp_act": t_pp_act,
            "a2a": t_a2a,
            "sync": t_sync,
        }
    )
    if pp > 1 and bubble > 0.05 and t_roof > t_stage * 1.05:
        if wall in ("compute", "memory") and t_roof > max(t_comp, t_mem) * 1.2:
            wall = "bubble"

    other = capacity_other_bytes // other_div
    if shard_other and capacity_other_bytes > 0:
        assumptions.append(
            f"capacity other (e.g. pair L²) sharded /{other_div} — crude analytical "
            f"assumption, not a real pair-tensor partition schedule"
        )
    w_cap = capacity_weight_bytes // n_shard
    needed = w_cap + other
    assumptions.append(
        f"per-card capacity: W={w_cap/1e9:.4f} GB + other={other/1e9:.4f} GB "
        f"vs cap={mem.capacity_bytes/1e9:.1f} GB [assumed]"
    )
    assumptions.append(
        "modeled: TP weight shard + act collectives, PP layer split + bubble + "
        "inter-stage acts, C2C BW/hide; "
        "per-collective latency α (exposed sync); "
        "ignored: real NoC topology, denoise cross-step PP schedule, "
        "pair-tensor partition schedule, calibrated link latency (α assumed)"
    )

    return DomainTPResult(
        tp=tp,
        pp=pp,
        ep=ep,
        flops_card=flops_card,
        weight_bytes_card=w_card,
        act_bytes_card=act_card,
        kv_bytes_card=kv_card,
        collective_bytes=coll,
        pp_act_bytes=pp_bytes,
        a2a_bytes=a2a,
        t_compute_s=t_comp,
        t_memory_s=t_mem,
        t_c2c_s=t_c2c,
        t_pp_act_s=t_pp_act,
        t_a2a_s=t_a2a,
        t_stage_s=t_stage,
        bubble_frac=bubble,
        t_roofline_s=t_roof,
        t_sum_s=t_comp + t_mem + t_c2c + t_pp_act + t_a2a + t_sync,
        wall=wall,
        capacity_needed_bytes=needed,
        oom=needed > mem.capacity_bytes,
        mean_gemm_util=tr.mean_gemm_util,
        collective_algo=collective_algo,
        assumptions=tuple(assumptions),
        t_sync_s=t_sync,
        n_sync=n_sync,
    )


def scale_domain_tp(
    single: PhaseResult,
    *,
    mem: ExternalMemory,
    cfg: EvalConfig,
    tp: int,
    act_volume_bytes: int,
    n_layers: int,
    capacity_weight_bytes: int,
    capacity_other_bytes: int = 0,
    shard_other: bool = False,
    c2c: C2CLink | None = None,
    c2c_hide: float = 0.0,
    collectives_per_layer: int = COLLECTIVES_PER_LAYER,
    domain: str = "domain",
) -> DomainTPResult:
    """Backward-compatible TP-only wrapper → ``scale_domain_parallel(pp=1, ep=1)``."""
    return scale_domain_parallel(
        single,
        mem=mem,
        cfg=cfg,
        tp=tp,
        pp=1,
        ep=1,
        mb=1,
        act_volume_bytes=act_volume_bytes,
        n_layers=n_layers,
        capacity_weight_bytes=capacity_weight_bytes,
        capacity_other_bytes=capacity_other_bytes,
        shard_other=shard_other,
        c2c=c2c,
        c2c_hide=c2c_hide,
        collectives_per_layer=collectives_per_layer,
        domain=domain,
    )
