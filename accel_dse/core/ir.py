"""L1b + L2 — op-graph IR, emitted per rank.

``build_rank_ops(model, layer_idx, phase, shard)`` returns the ordered op list
that ONE rank executes for one transformer layer in one forward step.  The
whole-model (single-card) IR is the same call with ``Shard()`` (all degrees 1)
— there is no second code path.  Conservation tests (tests/test_core_ir.py)
check that summing per-rank ops over ranks reproduces the global op graph plus
the explicitly-declared replication.

Op kinds
  gemm   – weight GEMM  y[M,N] = x[M,K] W[K,N]  (``count`` identical instances,
           e.g. hit experts or per-KV-head attention GEMMs)
  attn   – activation×activation GEMM (QK^T, PV, indexer scoring); no weights
  state  – linear-attention recurrent-state update (vector/elementwise work)
  vector – elementwise work (norms, rope, softmax, activation)  in element-ops
  comm   – collective (all-reduce / all-to-all / all-gather / p2p) bytes per rank
  lookup – embedding-table row reads
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .dtypes import fmt as _fmt
from .model import AttnCore, Layer, Linear, ModelSpec, PairCore


@dataclass(frozen=True)
class Shard:
    """Parallel degrees seen by one rank of one pipeline stage.

    tp   – attention tensor parallel (heads split)
    dp   – attention data parallel (independent attention replicas in the stage)
    ep   – expert parallel degree;  etp – expert tensor parallel
           (MoE layers span all tp·dp ranks of the stage: ep·etp == tp·dp)
    Dense FFN / shared experts are split by ``tp`` and replicated over ``dp``.
    sp   – sequence parallel (Ulysses) for non-autoregressive models: each rank holds 1/sp of the
           tokens for every GEMM; all-to-all before / after self-attention so that each rank attends
           over the whole sequence for 1/sp of the (TP-local) heads.  LLMs: sp = 1.
    """
    tp: int = 1
    dp: int = 1
    ep: int = 1
    etp: int = 1
    sp: int = 1

    def __post_init__(self):
        for k in ("tp", "dp", "ep", "etp", "sp"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be ≥ 1")

    @property
    def ranks(self) -> int:
        return self.tp * self.dp

    def check_moe(self) -> None:
        if self.ep * self.etp != self.tp * self.dp:
            raise ValueError(f"ep·etp ({self.ep}·{self.etp}) must equal tp·dp ({self.tp}·{self.dp})")


@dataclass(frozen=True)
class PairDims:
    """Grid sizes of a structure-model forward (0.43): residues / tokens n, MSA rows, extra-MSA rows, templates,
    atoms, trunk passes (recycles incl. the first), diffusion steps and batched diffusion samples."""
    n: int
    msa: int = 0
    xmsa: int = 0
    tmpl: int = 0
    atoms: int = 0
    recycles: int = 1
    diff_steps: int = 0
    samples: int = 1
    split: bool = False      # DAP > 1: diffusion samples split over the DAP ranks instead of replicated (0.46)


@dataclass(frozen=True)
class Phase:
    """One forward step of a pipeline stage (one micro-batch).

    kind  – decode | prefill | full
    batch – sequences in this step (global over attention-DP replicas)
    q     – new tokens per sequence (decode: 1, or 1+k when verifying k drafts;
            prefill: prompt length S; full: every token of the sequence)
    ctx   – context already in the cache per sequence (decode: current length);
            full: conditioning tokens per sequence (cross-attention K/V source)
    frames, aux – full only: latent frames of the video grid; audio rows packed into the sequence
    ``full`` = one non-autoregressive forward over the whole sequence (DiT denoise step,
    protein encoder): no KV cache, bidirectional attention.
    """
    kind: str
    batch: int
    q: int = 1
    ctx: int = 0
    frames: int = 0      # full: latent frames of the video grid (factorized spatial / temporal attention)
    aux: int = 0         # full: joint audio rows in the sequence (MiniMax-H3)
    pair: PairDims | None = None   # full: structure-model grids (AlphaFold-family / ESMFold trunks)
    skew: float = 1.0    # MoE: token-expert pairs on the busiest EP rank / the mean (0.49; 1 = uniform routing)

    def __post_init__(self):
        if self.kind not in ("decode", "prefill", "full"):
            raise ValueError("phase kind must be decode|prefill|full")
        if self.batch < 1 or self.q < 1 or self.ctx < 0:
            raise ValueError("batch,q ≥ 1 and ctx ≥ 0 required")

    @property
    def tokens(self) -> int:
        return self.batch * self.q


@dataclass(frozen=True)
class Op:
    name: str
    kind: str
    layer: int
    m: int = 0
    k: int = 0
    n: int = 0
    count: int = 1
    role: str = ""            # weight role (format lookup)
    w_params: int = 0         # weight params read (per rank, all instances)
    w_bits: float = 0.0
    w_fmt: str = ""
    a_fmt: str = "bf16"
    causal: float = 1.0       # useful fraction of the M×N work (causal masks)
    kv_read: float = 0.0      # bytes
    kv_write: float = 0.0
    state_rw: float = 0.0     # recurrent state bytes read+written
    vec: float = 0.0          # vector element-ops
    comm_kind: str = ""       # allreduce | alltoall | allgather | p2p
    comm_group: int = 1
    comm_bytes: float = 0.0   # payload per rank
    comm_stride: int = 1      # card-index stride between group members (cards numbered TP, SP, DP, PP; 0.48 tiers)
    act_bytes: float = 0.0    # activation in+out bytes (for SRAM-port / spill accounting)
    replicated: float = 1     # how many ranks of the stage compute this identical op (fractional: idle sample slots)
    stream: bool = False      # full phase: activations may exceed SRAM → DRAM streaming accounted (memplan.act_stream)
    orient: bool = False      # full-phase attention: mapping may take either GEMM orientation (O = P·V or Oᵀ = Vᵀ·Pᵀ)
    bmm: bool = False         # activation×activation batched GEMM whose operands are plain tensors (triangle update,
                              # outer-product mean, pair-weighted averaging): streamed as A, B in and C out
    in_elems: float = 0.0     # conv as implicit GEMM: elements of the real input tensor (the M×K im2col matrix is
                              # virtual — DRAM streaming moves the input once, 0 = M·K·count)

    @property
    def flops(self) -> float:
        return 2.0 * self.m * self.k * self.n * self.count * self.causal

    @property
    def w_bytes(self) -> float:
        return self.w_params * self.w_bits / 8.0


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _lin_local(l: Linear, tp: int) -> tuple[int, int]:
    """(K, N) of the per-rank slice of a linear (GEMM dims; grouped → per-group K)."""
    k = l.k // l.groups
    if l.split == "col":
        return k, _cdiv(l.n, tp)
    if l.split == "row":
        return _cdiv(k, tp), l.n
    if l.split == "head":
        heads = l.n // max(l.unit, 1)
        return k, _cdiv(heads, tp) * l.unit
    return k, l.n


def _repl(l: Linear, tp: int) -> int:
    if l.split == "rep":
        return tp
    if l.split == "head":
        heads = l.n // max(l.unit, 1)
        return max(1, tp // heads) if heads < tp else 1
    return 1


def gemm(model: ModelSpec, name: str, layer: int, m: int, k: int, n: int, role: str, *, count: int = 1,
         params: int | None = None, replicated: int = 1, stream: bool = False) -> Op:
    rf = model.fmt(role)
    p = k * n * count if params is None else params
    ab = _fmt(model.act_fmt).bytes
    return Op(name, "gemm", layer, m=m, k=k, n=n, count=count, role=role, w_params=p, w_bits=rf.bits,
              w_fmt=rf.fmt, a_fmt=model.act_fmt, act_bytes=(m * k + m * n) * count * ab, replicated=replicated,
              stream=stream)


# ------------------------------------------------------------------ MoE routing
@dataclass(frozen=True)
class Routing:
    tokens: int          # tokens entering the MoE layer (global over the EP group)
    pairs_local: float   # token-expert pairs landing on this rank
    hit: int             # distinct local experts touched
    m_e: int             # rows per hit expert GEMM
    n_local: int         # experts stored on this rank
    pairs_mean: float = 0.0  # mean pairs per rank (= pairs_local without skew)


def moe_routing(n_experts: int, top_k: int, tokens: int, ep: int, skew: float = 1.0) -> Routing:
    """Uniform-routing expectation (「假设」).  E_hit = n·(1−(1−k/E)^T).

    ``skew`` (0.49, default 1 = uniform): the busiest EP rank receives skew × the mean T·k/EP token-expert pairs
    (capped at T·k and at T per local expert).  Its distinct-expert count stays the uniform expectation (the extra
    load lands on popular experts) unless T rows per expert cannot hold it.  The busiest rank sets the stage time
    (all ranks wait for it at the combine all-to-all)."""
    n_local = _cdiv(n_experts, ep)
    if tokens <= 0:
        return Routing(0, 0.0, 0, 0, n_local)
    p_miss = (1.0 - top_k / n_experts) ** tokens
    e_hit_local = n_local * (1.0 - p_miss)
    e_hit_glob = n_experts * (1.0 - p_miss)
    hit = max(math.ceil(e_hit_local - 1e-9), math.ceil(e_hit_glob / ep - 1e-9), 1)
    hit = min(hit, n_local)
    pairs_mean = tokens * top_k / ep
    pairs_local = pairs_mean
    if skew > 1.0 and ep > 1:
        pairs_local = min(pairs_mean * skew, float(tokens * top_k), float(n_local * tokens))
        hit = min(n_local, max(hit, math.ceil(pairs_local / tokens - 1e-9)))
    m_e = max(1, math.ceil(pairs_local / hit))
    return Routing(tokens, pairs_local, hit, m_e, n_local, pairs_mean)


def skew_from_load(load: tuple[float, ...], ep: int) -> float:
    """Busiest-rank / mean load of a measured per-expert token distribution under contiguous expert placement
    (expert e on EP rank e // ceil(E / EP), the default linear placement of vLLM / SGLang without EPLB)."""
    if ep <= 1 or not load:
        return 1.0
    n_local = _cdiv(len(load), ep)
    ranks = [sum(load[r * n_local:(r + 1) * n_local]) for r in range(ep)]
    mean = sum(load) / ep
    return max(1.0, max(ranks) / mean) if mean > 0 else 1.0


# ------------------------------------------------------------------ attention core
def _attn_core_ops(model: ModelSpec, li: int, core: AttnCore, ph: Phase, sh: Shard, b_loc: int) -> list[Op]:
    ops: list[Op] = []
    ab = _fmt(model.act_fmt).bytes
    kvb = _fmt(model.kv_fmt).bytes
    q, tp = ph.q, sh.tp
    if core.kind == "linear":
        h_loc = _cdiv(core.n_state_heads, tp)
        t = b_loc * q
        st_bytes = h_loc * core.state_dk * core.state_dv * _fmt(model.state_fmt).bytes
        chunk = 64 if ph.kind == "prefill" else 1          # 「假设」 chunked-scan length
        vec = t * h_loc * core.state_dk * core.state_dv * 4 + t * h_loc * chunk * (core.state_dk + core.state_dv) * 2
        conv = _cdiv(core.conv_channels, tp) * core.conv_kernel * t * 2
        ops.append(Op("state_update", "state", li, vec=vec + conv,
                      state_rw=2 * st_bytes * b_loc, kv_read=0.0))
        return ops
    if core.kind == "none":
        return ops
    if ph.kind == "decode":
        ctx_tot = ph.ctx + q
        ce = core.ctx_eff(ctx_tot)
        causal = 1.0
    else:
        ctx_tot = ph.ctx + q
        ce = core.ctx_eff(ctx_tot)
        causal = 0.5 * (1 + ph.ctx / ctx_tot) if core.window is None and core.topk is None and core.compress == 1 else 1.0
        if core.window is not None or core.topk is not None or core.compress > 1:
            ce = min(ce, ctx_tot)
    if core.kind == "gqa":
        kv_loc = _cdiv(core.n_kv, tp)
        hq_loc = _cdiv(core.n_q, tp)
        g = max(1, hq_loc // kv_loc)
        cnt = b_loc * kv_loc
        ops.append(Op("qk", "attn", li, m=q * g, k=core.qk_dim, n=ce, count=cnt, causal=causal,
                      act_bytes=cnt * (q * g * core.qk_dim) * ab))
        ops.append(Op("pv", "attn", li, m=q * g, k=ce, n=core.v_dim, count=cnt, causal=causal,
                      act_bytes=cnt * (q * g * core.v_dim) * ab))
        kv_tok = kv_loc * (core.qk_dim + core.v_dim) / core.compress
        sm_elems = cnt * q * g * ce * causal
    else:  # mla (latent cache, replicated on every attention-TP rank)
        hq_loc = _cdiv(core.n_q, tp)
        lat = core.kv_lora + core.rope_dim
        if ph.kind == "decode":
            # absorbed: scores over the latent, output in latent space (W_UK/W_UV absorbed into kv_b op)
            ops.append(Op("qk_latent", "attn", li, m=q * hq_loc, k=lat, n=ce, count=b_loc, causal=causal,
                          act_bytes=b_loc * q * hq_loc * lat * ab))
            ops.append(Op("pv_latent", "attn", li, m=q * hq_loc, k=ce, n=core.kv_lora, count=b_loc, causal=causal,
                          act_bytes=b_loc * q * hq_loc * core.kv_lora * ab))
        else:
            # naive MHA over decompressed K/V (kv_b applied to all ctx tokens — see layer ops)
            ops.append(Op("qk", "attn", li, m=q, k=core.qk_dim, n=ce, count=b_loc * hq_loc, causal=causal,
                          act_bytes=b_loc * hq_loc * q * core.qk_dim * ab))
            ops.append(Op("pv", "attn", li, m=q, k=ce, n=core.v_dim, count=b_loc * hq_loc, causal=causal,
                          act_bytes=b_loc * hq_loc * q * core.v_dim * ab))
        kv_tok = lat / core.compress
        sm_elems = b_loc * q * hq_loc * ce * causal
    if core.idx_heads:
        n_keys = math.ceil(ctx_tot / core.compress)
        ops.append(Op("indexer_score", "attn", li, m=q * core.idx_heads, k=core.idx_dim, n=n_keys, count=b_loc,
                      causal=causal if core.compress == 1 else 1.0, replicated=tp,
                      kv_read=b_loc * n_keys * core.idx_dim * kvb if ph.kind == "decode" else 0.0,
                      vec=b_loc * q * n_keys * 2))
        ops.append(Op("indexer_kv_write", "vector", li, kv_write=b_loc * q * core.idx_dim * kvb / core.compress, replicated=tp))
    # KV cache traffic: read the attended entries (sparse/window aware), write the new ones
    if ph.kind == "decode":
        attended = min(ce, ctx_tot) if core.topk is None else min(ce, ctx_tot)
        kv_read = b_loc * attended * kv_tok * kvb
    else:
        kv_read = b_loc * ph.ctx * kv_tok * kvb           # prefix (cached) context only
    kv_write = b_loc * q * kv_tok * kvb
    ops.append(Op("softmax", "vector", li, vec=sm_elems * 5, kv_read=kv_read, kv_write=kv_write,
                  replicated=(tp if core.kind == "mla" else 1)))
    return ops


# ------------------------------------------------------------------ layer ops
def build_rank_ops(model: ModelSpec, li: int, ph: Phase, sh: Shard = Shard(), layer: Layer | None = None) -> list[Op]:
    """Ops of one rank for transformer layer ``li`` (or an explicit MTP ``layer``)."""
    L = layer if layer is not None else model.layers[li]
    if ph.kind == "full":
        return _full_layer_ops(model, li, L, ph, sh)
    h = model.hidden
    ab = _fmt(model.act_fmt).bytes
    tp, dp = sh.tp, sh.dp
    b_loc = _cdiv(ph.batch, dp)
    t = b_loc * ph.q                      # tokens through this rank's attention replica
    ops: list[Op] = []
    ops.append(Op("attn_norm", "vector", li, vec=t * h * 4))
    for l in L.attn_linears:
        k, n = _lin_local(l, tp)
        m = t
        if l.name == "kv_b" and ph.kind == "prefill":
            m = b_loc * (ph.ctx + ph.q)      # decompress cached prefix + new tokens
        ops.append(gemm(model, f"attn.{l.name}", li, m, k, n, "attn", count=l.groups if l.groups > 1 else 1,
                        params=k * n * (l.groups if l.groups > 1 else 1), replicated=_repl(l, tp)))
    ops.extend(_attn_core_ops(model, li, L.core, ph, sh, b_loc))
    if L.core.kind in ("gqa", "mla"):
        hq = _cdiv(L.core.n_q, tp)
        ops.append(Op("rope", "vector", li, vec=t * hq * max(L.core.rope_dim, L.core.qk_dim // 2) * 3))
    if tp > 1:
        ops.append(Op("attn_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * ab))
    ops.append(Op("ffn_norm", "vector", li, vec=t * h * 4))
    f = L.ffn
    if f.kind == "dense":
        for l in L.ffn_linears:
            k, n = _lin_local(l, tp)
            ops.append(gemm(model, f"mlp.{l.name}", li, t, k, n, "mlp"))
        ops.append(Op("act", "vector", li, vec=t * _cdiv(f.d_ff, tp) * 4))
        if tp > 1:
            ops.append(Op("mlp_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * ab))
    elif f.kind == "moe":
        sh.check_moe()
        T = ph.tokens                       # global tokens in the stage step
        for l in L.ffn_linears:
            k, n = _lin_local(l, tp)
            role = "router" if l.name.startswith(("router", "shared_expert_gate")) else (
                "shared_expert" if l.name.startswith("shared") else "mlp")
            ops.append(gemm(model, f"moe.{l.name}", li, t, k, n, role, replicated=_repl(l, tp)))
        r = moe_routing(f.n_experts, f.top_k, T, sh.ep, ph.skew)
        d_loc = _cdiv(f.d_expert, sh.etp)
        ein = f.expert_in(h)
        if r.hit:
            mats_up = 2 if f.gated else 1
            ops.append(gemm(model, "expert.gate_up", li, r.m_e, ein, mats_up * d_loc, "expert", count=r.hit))
            ops.append(gemm(model, "expert.down", li, r.m_e, d_loc, ein, "expert", count=r.hit))
            ops.append(Op("expert_act", "vector", li, vec=r.pairs_local * d_loc * 4))
        if sh.ep > 1:
            pay = r.pairs_local * ein * ab
            ops.append(Op("moe_dispatch", "comm", li, comm_kind="alltoall", comm_group=sh.ep, comm_bytes=pay,
                          comm_stride=sh.etp))
            ops.append(Op("moe_combine", "comm", li, comm_kind="alltoall", comm_group=sh.ep, comm_bytes=pay,
                          comm_stride=sh.etp))
        if sh.etp > 1:
            ops.append(Op("expert_allreduce", "comm", li, comm_kind="allreduce", comm_group=sh.etp,
                          comm_bytes=r.pairs_local * ein * ab))
        if tp > 1 and f.n_shared:
            ops.append(Op("shared_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * ab))
    ops.append(Op("residual", "vector", li, vec=t * h * 2))
    return ops


def embed_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard()) -> list[Op]:
    b_loc = _cdiv(ph.batch, sh.dp)
    t = b_loc * ph.q
    rf = model.fmt("embed")
    return [Op("embed_lookup", "lookup", -1, kv_read=t * model.hidden * rf.bits / 8, vec=t * model.hidden)]


def head_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard()) -> list[Op]:
    """LM head, vocab-parallel over attention TP.  Prefill: only the last token
    of each sequence is projected; decode: every new/verified token."""
    b_loc = _cdiv(ph.batch, sh.dp)
    m = b_loc if ph.kind == "prefill" else b_loc * ph.q
    n = _cdiv(model.vocab, sh.tp)
    role = "embed" if model.tie_embeddings else "lm_head"
    ops = [Op("final_norm", "vector", model.n_layers, vec=b_loc * ph.q * model.hidden * 4),
           gemm(model, "lm_head", model.n_layers, m, model.hidden, n, role),
           Op("sample", "vector", model.n_layers, vec=m * n * 3)]
    if sh.tp > 1:
        ops.append(Op("logits_gather", "comm", model.n_layers, comm_kind="allgather", comm_group=sh.tp,
                      comm_bytes=m * 2 * 4))  # (max, idx) per rank for greedy/top-k 「假设」
    return ops


def mtp_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard(), depth: int = 0) -> list[Op]:
    """One MTP draft step (module ``depth``): eh_proj + MTP layer + shared head."""
    if not model.mtp_layers:
        return []
    L = model.mtp_layers[min(depth, len(model.mtp_layers) - 1)]
    li = model.n_layers + 1 + depth
    b_loc = _cdiv(ph.batch, sh.dp)
    t = b_loc * ph.q
    h = model.hidden
    ops = [gemm(model, "mtp.eh_proj", li, t, 2 * h, h, "mtp", replicated=sh.tp)]
    ops += build_rank_ops(model, li, ph, sh, layer=L)
    ops += [gemm(model, "mtp.head", li, t, h, _cdiv(model.vocab, sh.tp), "lm_head")]
    return ops


# ------------------------------------------------------------------ full (non-autoregressive) forward
def _prefix(model: ModelSpec) -> int:
    return model.workload.prefix_tokens if model.workload else 0


def _video_rows(model: ModelSpec, ph: Phase) -> int:
    """Video tokens of the sequence (minus the joint text prefix and the joint audio rows)."""
    return ph.q - _prefix(model) - ph.aux


def _rows(model: ModelSpec, l: Linear, ph: Phase, b: int, sp: int) -> int:
    """GEMM rows of one rank: tokens are split over sp (Ulysses); conditioning / per-sequence rows are replicated.
    Stream-specific weights (dual-stream MMDiT, modality io) see only their rows: img = video tokens,
    txt = the joint text prefix, aud = the joint audio rows."""
    if l.rows == "ctx":
        return b * ph.ctx
    if l.rows == "seq":
        return b
    if l.rows == "img":
        return b * _cdiv(_video_rows(model, ph), sp)
    if l.rows == "txt":
        return b * _cdiv(_prefix(model), sp)
    if l.rows == "aud":
        return b * _cdiv(ph.aux, sp)
    return b * _cdiv(ph.q, sp)


def _full_attn(model: ModelSpec, li: int, core: AttnCore, ph: Phase, sh: Shard, b: int, cross: bool) -> list[Op]:
    """Bidirectional attention of a full-sequence forward (flash-style: the N×N scores are never stored).
    Self-attention under SP: each rank attends over the whole sequence for ceil(heads_tp / sp) heads;
    cross-attention: local queries over all ``ctx`` keys for every TP-local head (no all-to-all)."""
    ab = _fmt(model.act_fmt).bytes
    heads_tp = _cdiv(core.n_q, sh.tp)
    groups = 1
    if cross:
        h_loc, nq, nk, tag = heads_tp, _cdiv(ph.q, sh.sp), ph.ctx, "x"
    else:
        h_loc, nq, nk, tag = _cdiv(heads_tp, sh.sp), ph.q, ph.q, ""
        if core.span != "full" and ph.frames:
            # factorized ST attention over the video grid: spatial = one group per latent frame (H·W tokens),
            # temporal = one group per spatial position (T tokens)
            vid, T = _video_rows(model, ph), ph.frames
            hw = _cdiv(vid, T)
            groups, nq = (T, hw) if core.span == "spatial" else (hw, T)
            nk = nq
    causal = 0.5 if core.causal else 1.0
    cnt = b * h_loc * groups
    ops = [Op(tag + "qk", "attn", li, m=nq, k=core.qk_dim, n=nk, count=cnt, causal=causal,
              act_bytes=cnt * nq * core.qk_dim * ab, stream=True, orient=True),
           Op(tag + "pv", "attn", li, m=nq, k=nk, n=core.v_dim, count=cnt, causal=causal,
              act_bytes=cnt * nq * core.v_dim * ab, orient=True),
           Op(tag + "softmax", "vector", li, vec=cnt * nq * nk * causal * 5)]
    t_loc = b * _cdiv(ph.q, sh.sp)
    if not cross:
        ops.append(Op("qk_norm_rope", "vector", li, vec=t_loc * heads_tp * (core.qk_dim * 2 * 3 + core.rope_dim * 2 * 3)))
        if sh.sp > 1:
            ops.append(Op("sp_a2a_qkv", "comm", li, comm_kind="alltoall", comm_group=sh.sp,
                          comm_bytes=t_loc * heads_tp * (2 * core.qk_dim + core.v_dim) * ab, comm_stride=sh.tp))
            ops.append(Op("sp_a2a_o", "comm", li, comm_kind="alltoall", comm_group=sh.sp,
                          comm_bytes=t_loc * heads_tp * core.v_dim * ab, comm_stride=sh.tp))
    return ops


def layer_repeat(L: Layer, ph: Phase) -> int:
    """Sequential executions of layer ``L`` per forward request (structure models: trunk recycles, diffusion steps,
    shared-weight iterations).  The evaluator accumulates the layer's op sums this many times; ops are per execution."""
    pd = ph.pair
    if pd is None or not (L.repeat or L.iters > 1):
        return 1
    r = pd.recycles if L.repeat == "recycle" else pd.diff_steps if L.repeat == "diff" else 1
    return max(0, r) * L.iters


def _pair_rows(tag: str, pd: PairDims, b: int, s: int, ps: int, dap: int = 1) -> int:
    """Rows of a structure-model GEMM on one rank; ``s`` multiplies token / atom grids, ``ps`` the pair / MSA grids
    (per-sample pair copies).  DAP (0.45): the pair / template grids are split along their first residue axis and
    the MSA grids along one axis over ``dap`` ranks; token / atom grids are replicated."""
    n = pd.n
    nd = _cdiv(n, dap)
    return {"res": b * s * n, "tok": b * s * n, "seq": b * s, "pair": b * ps * nd * n,
            "msa": b * ps * _cdiv(pd.msa, dap) * n, "xmsa": b * ps * _cdiv(pd.xmsa, dap) * n,
            "tmpl": b * pd.tmpl * nd * n, "atom": b * s * pd.atoms,
            "apair": b * s * _cdiv(pd.atoms, 32) * 32 * 128}[tag]


_DAP_SPLIT = frozenset({"pair", "msa", "xmsa", "tmpl"})


def _pair_core_ops(model: ModelSpec, li: int, c: PairCore, pd: PairDims, b: int, s: int, ps: int,
                   tag: str, dap: int = 1, transposed_msa: bool = True) -> list[Op]:
    """Activation × activation work of one module on one rank.  ``dap`` > 1: dynamic axial parallelism (FastFold,
    Cheng et al. 2022) — the module runs on its local slice of the residue / MSA axis and the communication of the
    FastFold DAP kernels is added (payload per rank; like every collective it overlaps compute per the stage rule
    max(compute, DRAM, link) and adds α per collective):
      trimul   all-gather of one projected operand (N × N × c)
      tri_att  all-gather of the pair bias (N × N × H) + one all-to-all transpose of the pair rep (start ↔ end node)
      row_att  all-gather of the pair bias (MSA split along its rows → attention local)
      col_att  two all-to-all transposes of the MSA rep (row ↔ column split)
      opm      all-gather of the right projection (S × N × c2); AF3-style MSA modules without column attention keep
               the MSA row-split, so an all-to-all transpose of the MSA is added first
      pwa      all-gather of the pair-derived weights (N × N × H)
      seq_att  all-gather of the pair bias (single track itself replicated on every DAP rank)"""
    ab = _fmt(model.act_fmt).bytes
    n = pd.n
    nd = _cdiv(n, dap)
    bt = b * pd.tmpl if c.over == "tmpl" else b * ps      # pair-grid batch: templates or per-sample pair copies
    comm: list[Op] = []

    def cm(kind: str, payload: float, what: str) -> None:
        if dap > 1 and payload > 0:
            comm.append(Op(f"{tag}.dap_{what}", "comm", li, comm_kind=kind, comm_group=dap, comm_bytes=payload * ab))

    def att(groups: int, nq: int, nk: int, d: int, vd: int, bias: bool) -> list[Op]:
        cnt = groups * c.heads
        return [Op(tag + ".qk", "attn", li, m=nq, k=d, n=nk, count=cnt, act_bytes=cnt * nq * d * ab, stream=True,
                   orient=True),
                Op(tag + ".pv", "attn", li, m=nq, k=nk, n=vd, count=cnt, act_bytes=cnt * nq * vd * ab, orient=True),
                Op(tag + ".softmax", "vector", li, vec=cnt * nq * nk * (5 + (1 if bias else 0)))]

    def bmm(m: int, k: int, nn: int, cnt: int) -> Op:
        return Op(tag, "attn", li, m=m, k=k, n=nn, count=cnt, act_bytes=cnt * (m * k + k * nn + m * nn) * ab,
                  stream=True, bmm=True)

    d, vd = c.dim, c.v_dim or c.dim
    if c.kind == "trimul":
        cm("allgather", bt * nd * n * c.dim, "gather")
        return [bmm(nd, n, n, bt * c.dim)] + comm
    if c.kind == "tri_att":
        cm("allgather", bt * nd * n * c.heads, "bias")
        cm("alltoall", bt * nd * n * (c.width or model.workload.pair_dim), "transpose")
        return att(bt * nd, n, n, d, vd, True) + comm
    if c.kind == "row_att":
        g = pd.xmsa if c.over == "xmsa" else pd.msa
        cm("allgather", b * ps * nd * n * c.heads, "bias")
        return att(b * ps * _cdiv(g, dap), n, n, d, vd, True) + comm
    if c.kind == "col_att":
        S = pd.xmsa if c.over == "xmsa" else pd.msa
        cm("alltoall", 2 * b * ps * _cdiv(S, dap) * n * c.width, "transpose")
        return att(b * ps * nd, 1 if c.glob else S, S, d, vd, False) + comm
    if c.kind == "pt_att":          # template point-wise attention: each pair position attends over the T templates
        return att(b * nd * n, 1, max(1, pd.tmpl), d, vd, False)
    if c.kind == "opm":
        S = pd.xmsa if c.over == "xmsa" else pd.msa
        if not transposed_msa:
            cm("alltoall", b * ps * _cdiv(S, dap) * n * c.width, "transpose")
        cm("allgather", b * ps * S * nd * c.dim2, "gather")
        return [bmm(nd * c.dim, S, n * c.dim2, b * ps)] + comm
    if c.kind == "pwa":
        cm("allgather", b * ps * nd * n * c.heads, "weights")
        return [bmm(nd, n, pd.msa * c.dim, b * ps * c.heads),
                Op(tag + ".softmax", "vector", li, vec=b * ps * c.heads * nd * n * 5)] + comm
    if c.kind == "seq_att":
        cm("allgather", b * nd * n * c.heads, "bias")
        return att(b * s, n, n, d, vd, True) + comm
    if c.kind == "local_att":
        wq, wk = c.win
        return att(b * s * _cdiv(pd.atoms, wq), wq, wk, d, vd, True)
    raise ValueError(f"unknown pair core {c.kind}")


def _pair_layer_ops(model: ModelSpec, li: int, L: Layer, ph: Phase, sh: Shard) -> list[Op]:
    """One execution of a structure-model block on one rank.  Every weight GEMM runs on the rows of its grid; each
    linear also carries its layer-norm / gating / bias work (「假设」 4 element-ops per input element + 4 per output
    element).  ``sh.sp`` > 1 = DAP degree (0.45): pair / MSA / template grids split over the ranks, token / atom
    grids replicated (``replicated`` = DAP degree); TP of the pair stack is not modelled (c_z = 128 channels)."""
    pd = ph.pair
    if pd is None:
        raise ValueError("structure-model layer needs Phase.pair")
    if sh.tp > 1:
        raise ValueError("structure models: TP of the pair stack is not modelled; use DAP (sp) / DP / PP")
    dap = sh.sp
    b = _cdiv(ph.batch, sh.dp)
    s = pd.samples if L.samples in ("tok", "all") else 1
    ps = pd.samples if L.samples == "all" else 1
    rep_tok = dap              # token / atom grids: replicated on every DAP rank
    if dap > 1 and pd.split and L.samples == "tok" and s > 1:
        # 0.46: the diffusion module's samples are independent trajectories given the trunk output (single + pair
        # conditioning, already on every rank): each DAP rank runs ⌈S / D⌉ of them — exact, no extra communication;
        # the pair-grid work of the block (pair-bias projections) stays DAP-split as before
        s_loc = _cdiv(s, dap)
        rep_tok = dap * s_loc / s
        s = s_loc
    ops: list[Op] = []
    for l in L.pair_linears:
        m = _pair_rows(l.rows, pd, b, s, ps, dap)
        if m <= 0:
            continue
        rep = 1 if l.rows in _DAP_SPLIT else rep_tok
        o = gemm(model, f"{L.stack}.{l.name}", li, m, l.k, l.n, l.role or "pair", stream=True, replicated=rep)
        ops.append(o)
        ops.append(Op(f"{L.stack}.{l.name}.ew", "vector", li, vec=m * (l.k + l.n) * 4, replicated=rep))
    has_col = any(c.kind == "col_att" for c in L.pair_cores)
    for i, c in enumerate(L.pair_cores):
        cops = _pair_core_ops(model, li, c, pd, b, s, ps, f"{L.stack}.{c.kind}{i}", dap, has_col)
        if dap > 1 and c.kind in ("seq_att", "local_att"):
            cops = [replace(o, replicated=rep_tok) if o.kind != "comm" else o for o in cops]
        ops += cops
    return ops


def _repl_full(l: Linear, tp: int, sp: int) -> int:
    """Ranks computing the same rows of a full-forward linear: per-sequence / conditioning rows are not split by SP."""
    return _repl(l, tp) * (sp if l.rows in ("ctx", "seq") else 1)


def _full_layer_ops(model: ModelSpec, li: int, L: Layer, ph: Phase, sh: Shard) -> list[Op]:
    if L.pair_linears or L.pair_cores:
        return _pair_layer_ops(model, li, L, ph, sh)
    h = model.hidden
    ab = _fmt(model.act_fmt).bytes
    tp, sp = sh.tp, sh.sp
    b = _cdiv(ph.batch, sh.dp)
    t = b * _cdiv(ph.q, sp)                    # tokens through this rank's GEMMs
    mod = 4 if model.adaln else 0              # AdaLN: x·(1+scale)+shift, gate (element-ops per element)
    ops: list[Op] = [Op("attn_norm", "vector", li, vec=t * h * (4 + mod))]

    def lin(prefix: str, l: Linear, role: str) -> Op:
        k, n = _lin_local(l, tp)
        return gemm(model, f"{prefix}.{l.name}", li, _rows(model, l, ph, b, sp), k, n, role, stream=True,
                    replicated=_repl_full(l, tp, sp))

    ops += [lin("attn", l, "attn") for l in L.attn_linears]
    ops += _full_attn(model, li, L.core, ph, sh, b, cross=False)
    if tp > 1 and not L.fused_out:
        ops.append(Op("attn_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * ab))
    if L.cross is not None:
        ops.append(Op("cross_norm", "vector", li, vec=t * h * 4))
        ops += [lin("cross", l, "attn") for l in L.cross_linears]
        ops += _full_attn(model, li, L.cross, ph, sh, b, cross=True)
        if tp > 1:
            ops.append(Op("cross_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * ab))
    ops.append(Op("ffn_norm", "vector", li, vec=t * h * (4 + mod)))
    ops += [lin("mlp", l, "mlp") for l in L.ffn_linears]
    ops.append(Op("act", "vector", li, vec=t * _cdiv(L.ffn.d_ff, tp) * 8))     # GELU (tanh) / SwiGLU 「假设」 8 ops / element
    if tp > 1:
        ops.append(Op("mlp_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * ab))
    ops.append(Op("residual", "vector", li, vec=t * h * (2 + (2 if model.adaln else 0))))
    return ops


def full_io_ops(model: ModelSpec, ph: Phase, sh: Shard, side: str) -> list[Op]:
    """Input side (first stage): token lookup / patch, text and timestep embedders.
    Output side (last stage): final norm, un-patchify head / MLM head.  io GEMMs are column-split over TP
    with an all-gather of their (small) outputs."""
    b = _cdiv(ph.batch, sh.dp)
    t = b * _cdiv(ph.q, sh.sp)
    ab = _fmt(model.act_fmt).bytes
    li = -1 if side == "pre" else model.n_layers
    ops: list[Op] = []
    if side == "pre" and model.vocab:
        ops.append(Op("embed_lookup", "lookup", li, kv_read=t * model.hidden * model.fmt("embed").bits / 8,
                      vec=t * model.hidden * 2))
    if side == "post":
        ops.append(Op("final_norm", "vector", li, vec=t * model.hidden * (4 + (4 if model.adaln else 0))))
    for l in (model.io_pre if side == "pre" else model.io_post):
        k, n = _lin_local(l, sh.tp)
        m = _rows(model, l, ph, b, sh.sp)
        ops.append(gemm(model, "io." + l.name, li, m, k, n, l.role or "io", stream=True,
                        replicated=_repl_full(l, sh.tp, sh.sp)))
        if sh.tp > 1:
            ops.append(Op("io_allgather", "comm", li, comm_kind="allgather", comm_group=sh.tp, comm_bytes=m * n * ab))
    if side == "post" and model.vocab:     # MLM logits for every token (tied decoder)
        n = _cdiv(model.vocab, sh.tp)
        ops.append(gemm(model, "mlm_decoder", li, t, model.hidden, n, "embed", stream=True))
        ops.append(Op("act", "vector", li, vec=t * model.hidden * 8 + t * n * 3))
    return ops


def total(ops: list[Op], attr: str) -> float:
    return sum(getattr(o, attr) for o in ops)
