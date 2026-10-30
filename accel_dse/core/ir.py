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

import functools
import math
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from .dtypes import fmt as _fmt
from .model import AttnCore, Layer, Linear, ModelSpec, PairCore
from . import vector_rules as vr


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
    tri_chunk: int = 0       # 0.119 triangle attention: 0 = fused (flash-style); −1 = logits materialised whole;
                             # c > 0 = chunked over c rows (reference chunk_layer)


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
    ref: int = 0         # 0.252 full: reference-image rows in the sequence (FLUX 3 image_cond tokens, own weights)
    pair: PairDims | None = None   # full: structure-model grids (AlphaFold-family / ESMFold trunks)
    skew: float = 1.0    # MoE: token-expert pairs on the busiest EP rank / the mean (0.49; 1 = uniform routing)
    v_p: tuple = ()      # 0.293 MoVA: per-value-expert inclusion probability k_v·load_e/Σload (measured; () = uniform)
    # 0.293 per-layer measured loads (serving.*_expert_load_layers): per main-stack layer index the MoE routing skew
    # (float | SkewSpec) / the value-expert probabilities, None where the layer has none; () = the phase-wide skew / v_p
    skew_l: tuple = ()
    v_pl: tuple = ()
    cp: int = 1          # 0.89: context-parallel chunk phase (evaluate) — cp ranks, true prompt length cp_len
    cp_len: int = 0
    ep_nodes: int = 1    # 0.134: nodes the EP group spans when the all-to-all sends a token once per node (else 1)
    kvpre: int = 0       # 0.116 full: cached text-prefix tokens (prefix K/V reuse); q then holds the image tokens only
    kvx: bool = False    # 0.116 full: extract step of the prefix K/V reuse (writes the cache)
    mreuse: bool = False  # 0.255 full: FLUX 3 reused step — text / reference mode rows come from step 1 (SGLang StaticStreams)
    roll: str = ""       # 0.223 spec verify on linear layers: "" = intermediate-state snapshot (default), "replay" = ReplaySSM
    acc: float = 0.0     # 0.223 per-draft acceptance (replay / track: the ring advances by the expected accepted tokens)
    topk: int = 0        # 0.224 tree drafting: branches per draft step (0 / 1 = chain); q = verified tree tokens
    track: int = 0       # 0.224 radix-track interval (tokens) of hybrid-layer states; 0 = off
    questions: int = 1   # 0.291 joint decision request: readouts (question markers) per sequence
    labels: int = 0      # 0.191 decision readout (prefill only): logits of ≤ labels option tokens at the last
    samp: bool = False   # 0.253 decode verify: sampled draft rounds (DSpine eager serial_rs_table; serving.spec_sampling)
    # 0.384 multimodal tokens per sequence inside this step (ERNIE-4.5-VL expert pools). 0 = text only.
    mm: int = 0
    txt: int = 0         # 0.258 full: joint text rows of this forward (0 = the model's prefix_tokens) — CFG negative branch
    skip: tuple = ()     # 0.258 full: block indices this forward skips (MOSS-Image uncond_skip_layers)
                         # position, no KV cache kept (StartLux use_cache=False path) when nothing is prefix-shared

    def __post_init__(self):
        if self.kind not in ("decode", "prefill", "full"):
            raise ValueError("phase kind must be decode|prefill|full")
        if self.batch < 1 or self.q < 1 or self.ctx < 0:
            raise ValueError("batch,q ≥ 1 and ctx ≥ 0 required")
        if self.mm < 0 or self.mm > self.q:
            raise ValueError("phase.mm must be an integer in [0, q]")

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
    sfu: float = 0.0          # 0.103: transcendentals among them (exp, rsqrt, reciprocal) — Chip.sfu_per_cycle
    comm_kind: str = ""       # allreduce | alltoall | allgather | p2p
    comm_group: int = 1
    comm_bytes: float = 0.0   # payload per rank
    comm_net: float = 1.0     # 0.134: factor on an all-to-all's network-tier bytes (per-node token dedup)
    comm_stride: int = 1      # card-index stride between group members (cards numbered TP, SP, DP, PP; 0.48 tiers)
    act_bytes: float = 0.0    # activation in+out bytes (for SRAM-port / spill accounting)
    replicated: float = 1     # how many ranks of the stage compute this identical op (fractional: idle sample slots)
    stream: bool = False      # full phase: activations may exceed SRAM → DRAM streaming accounted (memplan.act_stream)
    orient: bool = False      # full-phase attention: mapping may take either GEMM orientation (O = P·V or Oᵀ = Vᵀ·Pᵀ)
    bmm: bool = False         # activation×activation batched GEMM whose operands are plain tensors (triangle update,
                              # outer-product mean, pair-weighted averaging): streamed as A, B in and C out
    share: float = 1.0        # 0.63: mean-rank useful work / this (busiest) rank's useful work of the op — uneven
                              # splits (ceil heads / columns / tokens / batch, padded expert rows) cost the busiest rank's
                              # time but only their share of work: energy / FLOP KPIs use work × share (rank_shares)
    kv_pre: float = 0.0       # streamed LLM prefill attention (0.63): share of the attended K/V already read from the
                              # cache (counted in kv_read) — its first pass is not charged again by act_stream
    wshare: float = 1.0       # 0.64: mean-rank weight elements / this (busiest) rank's — weight dequant (once per pass)
                              # is weight work, not token work: uneven TP / EP splits (rank_shares) scale it by this
    launch: float = 0.0       # 0.120: kernel launches of this op when Chip.kernel_us > 0 (0 = the 0.92 rule: one per GEMM /
                              # fused attention named qk*); structure-model ops set it explicitly
    chunks: int = 1           # 0.119: an unfused bmm run in this many sequential chunks (peak / SRAM fit per chunk)
    fft_n: int = 0            # 0.171 fused hyena FFT conv: transform size and transforms (sequence × channel) whose
    fft_rows: float = 0.0     # spectra spill only when they do not fit on chip (memplan.fft_spill)
    lg_bytes: float = 0.0     # 0.120: logits bytes inside an unfused attention op (stay on chip when one chunk fits)
    act_x: float = 0.0        # 0.119: extra activation DRAM bytes when the op spills (softmax read + write of logits)
    in_elems: float = 0.0     # conv as implicit GEMM: elements of the real input tensor (the M×K im2col matrix is
                              # virtual — DRAM streaming moves the input once, 0 = M·K·count)
    v_dim: int = 0            # 0.362: attn value width when it is not k (0 = k). Flash streaming counts K and V
                              # separately; equal-width attention leaves this 0 and stays on the 2×k path.

    @property
    def flops(self) -> float:
        return 2.0 * self.m * self.k * self.n * self.count * self.causal

    @property
    def w_bytes(self) -> float:
        return self.w_params * self.w_bits / 8.0


def red_bytes(model: ModelSpec) -> float:
    """0.61.1: bytes per element of reductions / the residual stream on the wire (TP / ETP all-reduce partial sums, MoE
    combine, PP hand-off): ≥ 2 (bf16) even when GEMM inputs are fp8 — DeepSeek-V3 report §3.3: dispatch in FP8,
    combine in BF16; all-reduces of row-parallel outputs are bf16 in vLLM / SGLang 「假设」.  fp8 dispatch keeps 1 B."""
    return max(_fmt(model.act_fmt).bytes, 2.0)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def gqa_local(n_q: int, n_kv: int, tp: int) -> tuple[int, int]:
    """0.63 (external review): (KV heads, query heads) on the busiest TP rank of a GQA layer; uneven TP covers every
    head (0.62: ⌈KV/TP⌉ KV heads × ⌊⌈Q/TP⌉ / ⌈KV/TP⌉⌋ query heads — Qwen3-8B TP3 covered 27 of 32 query heads).
    Two head-complete shardings, the cheaper one is used (fewest KV heads — KV cache bytes / reads — then fewest
    query rows per rank):
      groups whole: TP ≤ KV → ⌈KV/TP⌉ KV heads with their G = Q/KV query heads; TP > KV → 1 KV head, its G query
        heads over ⌊TP/KV⌋ ranks;
      heads dealt: contiguous blocks of ⌈Q/TP⌉ / ⌊Q/TP⌋ query heads; a rank holds the KV head of every group its
        block touches (a straddling group is replicated on both ranks).
    The attention kernel tiles per KV head with ⌈Q_loc / KV_loc⌉ query rows (padded rows cost time only).
    Divisible TP: KV/TP (or 1) and Q/TP as before."""
    hq = _cdiv(n_q, tp)
    if n_kv <= 0 or n_q % n_kv or tp <= 1:
        return _cdiv(n_kv, tp), hq
    g = n_q // n_kv
    base, rem = divmod(n_q, tp)
    kv_b, start = 1, 0
    for r in range(tp):
        size = base + (1 if r < rem else 0)
        if size:
            kv_b = max(kv_b, (start + size - 1) // g - start // g + 1)
        start += size
    if tp <= n_kv:
        kv_a = _cdiv(n_kv, tp); hq_a = kv_a * g
    else:
        kv_a, hq_a = 1, _cdiv(g, tp // n_kv)
    key = lambda kv, q: (kv, kv * _cdiv(q, kv), q)
    return min((kv_a, hq_a), (kv_b, hq), key=lambda c: key(*c))


# 0.198: fused QKV projection — serving engines run q, k, v as ONE GEMM over the concatenated weight [h, (Hq + 2·Hkv)·d]
# per rank 「来源」: vLLM model_executor/layers/linear.py QKVParallelLinear (models/llama.py qkv_proj), SGLang
# srt/layers/linear.py QKVParallelLinear (srt/models/llama.py qkv_proj), TensorRT-LLM layers/attention.py
# Attention.qkv = ColumnLinear(hidden, tp·(Hq + 2·Hkv)·d) (PyTorch backend _torch/modules/attention.py qkv_proj).
# Same FLOPs / weight bytes; one kernel instead of three (≤ 0.197: three GEMMs, each paying its own tile / wave /
# DRAM-floor cost — MI300X 70B b1 attn.k / attn.v 2.9 µs each vs 0.56 µs of weights).  False = ≤ 0.197.
FUSE_QKV = True
# 0.222: speculative verify on hybrid / linear-attention layers (decode with q = 1 + k > 1).  The verify pass cannot
# overwrite the committed recurrent state (rejected tokens would corrupt it): it writes one intermediate SSM state per
# draft token and the accepted one is scattered back after verification 「来源」 SGLang main 9c7b338
# mem_cache/memory_pool.py ("Cache intermediate SSM states per draft token during target verify", shape [layers, slots,
# speculative_num_draft_tokens, HV, V, K], ssm dtype) + hybrid_linear_attn_backend.update_mamba_state_after_mtp_verify
# → fused_mamba_state_scatter (read the accepted intermediate, write the committed state).  Per layer per sequence:
# read 1 + q intermediate writes + commit read 1 + write 1 = (q + 3) states (≤ 0.221: 2 — the AR read + write).  The
# opt-in ReplaySSM path (no per-draft writes) is not the default and is not modelled 「假设」 that the ReaDy-pinned SGLang
# (2948168) uses the same default path.  Memory: q intermediate states per sequence (memplan, spec_q).
SPEC_STATE_SNAPSHOT = True
# 0.223 / 0.224: opt-in ReplaySSM spec verify (serving.spec_rollback = "replay"; SGLang --enable-linear-replayssm-spec)
# 「来源」 SGLang main kernels/ops/attention/fla/gdn_replayssm_spec_decode.py + kda_replayssm_spec_decode.py,
# gdn_backend._replayssm_target_verify / _replayssm_fold_target_verify, mem_cache/memory_pool.py:
#  GDN (circular ring): the verify reconstructs the window output-only from the frozen checkpoint h0 + the committed
#   (d, k, g) records (no state write; rejected drafts roll back by a cursor move).  Cursor rule
#   _advance_gdn_spec_cursors_kernel: on a flush step the exact-fold kernel sequentially replays the write_pos committed
#   records into the fp32 checkpoint (launched first) and write_pos := accepted; else write_pos += accepted; the next
#   step flushes when write_pos + 2·max_spec_len > L (early flush).  0.224: the history / fold rate per step come from
#   that rule at the expected acceptance E (replay_cycle, mean-field 「推导」) — 0.223 used h̄ = (L − q)/2 「假设」.
#   Ring entry per head: d, k (bf16) + their low parts rawv / rawk (bf16, activation-dtype rings) + g fp32.
#  KDA (fold-every-commit, replayssm_spec_fold = spec ∧ is_kda): the recurrent verify kernel runs with
#   disable_state_update and stores the raw window (rawk, rawv bf16, g fp32 per K, beta fp32 + d, k rings); the
#   commit (commit_kda_replayssm_after_verify) replays the accepted prefix into the checkpoint every step.
#  Ring length L: power of two (assert) and ≥ 2 · draft tokens ("Config enforces max_cache_len >= 2 * max_spec_len",
#   GDN and KDA) with linear_replayssm_cache_len default 16 → L = pow2 ≥ max(16, 2q) 「来源」 (0.223: KDA rule applied to
#   GDN 「假设」, not a power of two).
#  Linear chains only (topk ≤ 1): "Tree verify (topk > 1), KDA, and NPU/CPU fall back to the recurrent verify
#   kernel"; other recurrences (lightning, conv) have no ReplaySSM path → snapshot (evaluate warns).
# 0.224 radix tracking (Phase.track = serving.mamba_track_interval, SGLang default 256 「来源」 server_args; 0 = off):
# the accepted state is also written to a track slot every `interval` tokens (snapshot: scatter to mamba_track_indices;
# GDN replay: the materialize kernel replays history up to the boundary — "Materialize circular history only at
# capacity or radix-track boundaries"; KDA: the per-step fold writes it).  Whether a deployment tracks (radix cache on
# for hybrid models) is the lever 「假设」.
REPLAY_RING_DEFAULT = 16


def replay_ring_len(q: int) -> int:
    """ReplaySSM ring length L for a verify window of q tokens (power of two ≥ max(16, 2q))."""
    n = max(REPLAY_RING_DEFAULT, 2 * q)
    return 1 << (n - 1).bit_length()


@functools.lru_cache(maxsize=4096)
def _replay_cycle(L: int, q: int, E6: int) -> tuple:
    E = E6 / 1e6
    wp, flag, hist, folds, fold_n, n = 0.0, False, 0.0, 0, 0.0, 2048
    for _ in range(n):
        if flag:
            folds += 1
            fold_n += wp
            wp = E
        else:
            hist += wp
            wp += E
        flag = wp + 2 * q > L
    return hist / n, folds / n, fold_n / n


def replay_cycle(L: int, q: int, E: float) -> tuple:
    """0.224 GDN ReplaySSM cursor cycle (SGLang _advance_gdn_spec_cursors_kernel) at a mean committed E tokens per
    step: (history records read per verify, flush steps per step, records folded per step)."""
    return _replay_cycle(L, q, int(round(E * 1e6)))


def spec_expected(q: int, a: float) -> float:
    """Expected tokens committed per verify of q = 1 + k tokens at per-draft acceptance a."""
    return float(q) if a >= 1.0 else (1.0 - a ** q) / (1.0 - a)
# 0.222: the DSpark Markov head is semi-autoregressive: draft position j adds markov_w2·markov_w1[token_{j−1}] to its
# base logits and samples, and token_{j−1} is position j−1's sample — gamma sequential steps, each a w2 GEMV (or GEMM
# over the batch) reading the whole V × r w2, + add + argmax / sample 「来源」 SGLang main 9c7b338
# speculative/dspark_components/dspark_draft.py draft_markov_block (`for step in range(gamma)`, _draft_step:
# MarkovGreedyStep / step_bias + MarkovSampleStep).  ≤ 0.221: one GEMM over b·B rows (w2 read once).  False = ≤ 0.221.
DSPARK_SEQ_MARKOV = True
_QKV_NAMES = ("attn.q", "attn.k", "attn.v", "attn.qkv")


def _qkv_fusable(L, tp: int):
    """(K, [Nq, Nk, Nv] per rank, replicated) when the layer's q / k / v are plain (ungrouped) linears over the same
    input, else None."""
    by = {l.name: l for l in L.attn_linears}
    if not all(x in by for x in ("q", "k", "v")):
        return None
    ls = [by[x] for x in ("q", "k", "v")]
    if any(l.groups > 1 for l in ls):
        return None
    kn = [_lin_local(l, tp, L.core) for l in ls]
    rp = {_repl(l, tp) for l in ls}
    if len({k for k, _ in kn}) != 1 or len(rp) != 1:
        return None
    return kn[0][0], [n for _, n in kn], rp.pop()


def _lin_local(l: Linear, tp: int, core: AttnCore | None = None) -> tuple[int, int]:
    """(K, N) of the per-rank slice of a linear (GEMM dims; grouped → per-group K).  ``core`` (a GQA layer's attention
    core, 0.63): the query / output projections follow the busiest rank's query heads (head-aligned, gqa_local)."""
    if core is not None and core.kind == "gqa" and tp > 1 and l.groups == 1 and core.n_q:
        kv, hq = gqa_local(core.n_q, core.n_kv, tp)
        if l.split == "head" and l.name in ("k", "v") and core.n_kv and l.n == core.n_kv * max(l.unit, 1):
            return l.k, kv * l.unit
        if l.split == "col" and l.name in ("q", "o_gate") and l.n % core.n_q == 0:
            return l.k, hq * (l.n // core.n_q)
        if l.split == "row" and l.name == "o" and l.k % core.n_q == 0:
            return hq * (l.k // core.n_q), l.n
    k = l.k // l.groups
    if l.groups > 1:              # 0.125 (review): block-diagonal — per group K = k/g, N = n/g (was N = n: ×g work)
        g_loc, n_g = groups_local(l, tp)
        return k, n_g
    if l.split == "col":
        return k, _cdiv(l.n, tp)
    if l.split == "row":
        return _cdiv(k, tp), l.n
    if l.split == "head":
        heads = l.n // max(l.unit, 1)
        return k, _cdiv(heads, tp) * l.unit
    return k, l.n


def groups_local(l: Linear, tp: int) -> tuple[int, int]:
    """0.125: (groups on the busiest rank, N per group) of a grouped linear under TP — whole groups per rank while
    TP ≤ groups (⌈g/TP⌉), else one group with its N split over TP/g ranks."""
    g, n_g = l.groups, l.n // l.groups
    if tp <= g:
        return _cdiv(g, tp), n_g
    return 1, _cdiv(n_g, _cdiv(tp, g))


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
    af = model.act_fmt_for(role)
    ab = _fmt(af).bytes
    return Op(name, "gemm", layer, m=m, k=k, n=n, count=count, role=role, w_params=p, w_bits=rf.bits,
              w_fmt=rf.fmt, a_fmt=af, act_bytes=(m * k + m * n) * count * ab, replicated=replicated,
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


@dataclass(frozen=True)
class SkewSpec:
    """0.132: EP load skew with the busiest rank's routing shape.  ``hot_p``: P(a token routes to each expert stored on
    the busiest rank) from a measured load (exact occupancy); ``spread``: scalar-skew assumption (popular | uniform)."""
    skew: float = 1.0
    hot_p: tuple = ()
    spread: str = "popular"
    rank_max: str = "off"          # 0.133: "expected" = E[max over EP ranks] of per-step distinct experts / pairs

    def __float__(self):
        return float(self.skew)


def moe_routing(n_experts: int, top_k: int, tokens: int, ep: int, skew=1.0, groups: tuple = ()) -> Routing:
    """Uniform-routing expectation (「假设」).  E_hit = n·(1−(1−k/E)^T).

    ``skew`` (0.49, default 1 = uniform): the busiest EP rank receives skew × the mean T·k/EP token-expert pairs
    (capped at T·k and at T per local expert).  Its distinct-expert count stays the uniform expectation (the extra
    load lands on popular experts) unless T rows per expert cannot hold it.  The busiest rank sets the stage time
    (all ranks wait for it at the combine all-to-all).

    0.132 (``SkewSpec``): with a measured per-expert load the busiest rank's distinct experts are the occupancy
    max over ranks of Σ_e 1 − (1 − p_e)^T over the rank's experts (p_e = k · load_e / Σ load, a token's inclusion probability; Monte-Carlo
    routing: mean error 1.6 % vs 22 % for the popular-expert rule, MODEL §78).  Opt-in ``spread = "uniform"`` for a
    scalar skew: p = k · skew / E on every local expert.

    0.255 ``groups`` = (n_group, topk_group) of a group-limited router: every expert keeps the marginal k / E, so the
    mean hits / pairs are unchanged; only the per-rank spread (``_rank_max``) widens."""
    spec = skew if isinstance(skew, SkewSpec) else None
    skew = float(skew)
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
        if spec is not None and spec.hot_p:
            ranks = spec.hot_p if isinstance(spec.hot_p[0], tuple) else (spec.hot_p,)
            occ = max(sum(1.0 - (1.0 - min(1.0, p)) ** tokens for p in r) for r in ranks)
            hit = math.ceil(occ - 1e-9)
        elif spec is not None and spec.spread == "uniform":
            hit = math.ceil(n_local * (1.0 - (1.0 - min(1.0, top_k * skew / n_experts)) ** tokens) - 1e-9)
        hit = min(n_local, max(1, hit, math.ceil(pairs_local / tokens - 1e-9)))
    if spec is not None and spec.rank_max == "expected" and ep > 1:
        d_max, p_max = _rank_max(n_experts, top_k, tokens, ep, spec.hot_p, skew, tuple(groups))
        pairs_local = min(max(pairs_local, p_max), float(tokens * top_k), float(n_local * tokens))
        hit = min(n_local, max(1, hit, math.ceil(d_max - 1e-9), math.ceil(pairs_local / tokens - 1e-9)))
    m_e = max(1, math.ceil(pairs_local / hit))
    return Routing(tokens, pairs_local, hit, m_e, n_local, pairs_mean)


def identity_zero_routing(n_experts: int, zero: int, top_k: int, tokens: int, ep: int, skew=1.0) -> Routing:
    """Uniform top-k over weighted experts plus identity zero-experts (LongCat-Flash).

    The router scores ``n_experts + zero`` classes. An identity expert copies the token and has no GEMM.
    Under the same uniform assumption as ``moe_routing``, each class is equally likely, so a token's
    expected weighted selections are ``top_k * n_experts / pool`` and each weighted expert's inclusion
    probability is ``top_k / pool``. Scalar skew scales the busiest rank's weighted pairs the same way
    ``moe_routing`` scales pairs. A ``SkewSpec`` is not a measured LongCat load, so it falls back to
    ``moe_routing`` on the weighted experts only.
    """
    if zero <= 0 or isinstance(skew, SkewSpec):
        return moe_routing(n_experts, top_k, tokens, ep, skew)
    pool = n_experts + zero
    sk = float(skew)
    n_local = _cdiv(n_experts, ep)
    if tokens <= 0:
        return Routing(0, 0.0, 0, 0, n_local)
    p = min(1.0, top_k / pool)
    e_hit_local = n_local * (1.0 - (1.0 - p) ** tokens)
    e_hit_glob = n_experts * (1.0 - (1.0 - p) ** tokens)
    hit = max(math.ceil(e_hit_local - 1e-9), math.ceil(e_hit_glob / ep - 1e-9), 1)
    hit = min(hit, n_local)
    frac = n_experts / pool
    pairs_mean = tokens * top_k * frac / ep
    pairs_local = pairs_mean
    if sk > 1.0 and ep > 1:
        pairs_local = min(pairs_mean * sk, float(tokens * top_k * frac), float(n_local * tokens))
        hit = min(n_local, max(1, hit, math.ceil(pairs_local / tokens - 1e-9)))
    m_e = max(1, math.ceil(pairs_local / hit))
    return Routing(tokens, pairs_local, hit, m_e, n_local, pairs_mean)


def _emax_normal(ms: tuple) -> float:
    """E[max] of independent normals N(mu, sd^2): Blom's order-statistic constant for identical ranks, else
    numeric integration E[max] = lo + ∫ (1 − Π Φ((x − mu)/sd)) dx."""
    from statistics import NormalDist
    nd = NormalDist()
    sdm = max(sd for _, sd in ms)
    if sdm <= 0:
        return max(mu for mu, _ in ms)
    if len(set(ms)) == 1:
        n = len(ms)
        mu, sd = ms[0]
        return mu + sd * nd.inv_cdf((n - 0.375) / (n + 0.25))
    lo = min(mu for mu, _ in ms) - 6 * sdm
    hi = max(mu for mu, _ in ms) + 6 * sdm
    N = 600
    dx = (hi - lo) / N
    acc = 0.0
    for i in range(N):
        x = lo + (i + 0.5) * dx
        P = 1.0
        for mu, sd in ms:
            P *= nd.cdf((x - mu) / sd) if sd > 0 else (1.0 if x >= mu else 0.0)
        acc += 1.0 - P
    return lo + acc * dx


@functools.lru_cache(maxsize=4096)
def _rank_max(n_experts: int, top_k: int, tokens: int, ep: int, hot_p: tuple, skew: float, groups: tuple = ()) -> tuple:
    """0.133 opt-in (serving.moe_rank_max = "expected"): the stage waits for the slowest EP rank, and under random
    routing the per-step maxima over ranks of distinct experts read and token-expert pairs exceed their means even
    with uniform popularity.  Per rank: distinct experts D = Σ_e 1[e hit], mean Σ q_e, var Σ q_e(1 − q_e) with
    q_e = 1 − (1 − p_e)^T (independent-indicator approximation); pairs = Σ_tokens hypergeometric count, mean T·k·π,
    var T·k·π(1 − π)(E − k)/(E − 1) with π = Σ_e p_e / k.  E[max] over ranks under a normal approximation
    (_emax_normal).  A scalar skew > 1 raises one rank's pairs to skew × mean (its distinct experts as moe_routing).
    Returns (E[max D], E[max pairs])."""
    n_local = _cdiv(n_experts, ep)
    if hot_p:
        ranks = hot_p if isinstance(hot_p[0], tuple) else None
    else:
        ranks = None
    gm = None if (ranks is not None or not groups) else _group_rank_moments(n_experts, top_k, tokens, ep, groups)
    if gm is not None:
        dm, pm = [gm[0]] * ep, [gm[1]] * ep
        p_max = _emax_normal(tuple(pm))
        if skew > 1.0:
            p_max = max(p_max, tokens * top_k / ep * skew)
        return _emax_normal(tuple(dm)), p_max
    if ranks is None:
        ranks = tuple((top_k / n_experts,) * max(0, min(n_local, n_experts - r * n_local)) for r in range(ep))
    dm, pm = [], []
    for r in ranks:
        q = [1.0 - (1.0 - min(1.0, p)) ** tokens for p in r]
        dm.append((round(sum(q), 9), round(math.sqrt(sum(x * (1 - x) for x in q)), 9)))
        pi = min(1.0, sum(r) / top_k)
        var = tokens * top_k * pi * (1 - pi) * (n_experts - top_k) / max(1, n_experts - 1)
        pm.append((round(tokens * top_k * pi, 9), round(math.sqrt(max(0.0, var)), 9)))
    p_max = _emax_normal(tuple(pm))
    if skew > 1.0 and not hot_p:
        p_max = max(p_max, tokens * top_k / ep * skew)
    return _emax_normal(tuple(dm)), p_max


def _hyp_pmf(N: int, K: int, n: int) -> list[tuple[int, float]]:
    """Hypergeometric pmf: j successes in n draws without replacement from N items of which K are successes."""
    out = []
    for j in range(max(0, n - (N - K)), min(n, K) + 1):
        out.append((j, math.comb(K, j) * math.comb(N - K, n - j) / math.comb(N, n)))
    return out


def _group_rank_moments(E: int, k: int, T: int, ep: int, groups: tuple):
    """0.255 group-limited routing (「推导」 under the symmetric-routing 「假设」 of moe_routing): per token, topk_group = g
    of n_group = G contiguous groups (S = E/G experts each) are chosen uniformly, then the k experts uniformly among
    the g·S experts of those groups (DeepseekV3TopkRouter / BeamTopKRouter / vLLM grouped_topk select groups by the sum
    of their top-2 scores; the expert marginal stays k / E).  Contiguous EP placement (vLLM / SGLang default).
      * ep ≥ G (G | ep): a rank holds n_r = S·G/ep experts of ONE group, chosen with p = g/G → per-token pairs
        X = 1[chosen]·Y, Y ~ Hyp(gS, n_r, k);
      * ep < G (ep | G): a rank holds a = G/ep whole groups, J ~ Hyp(G, a, g) of them chosen, X | J ~ Hyp(gS, J·S, k).
    Distinct experts over T tokens, exact (tokens i.i.d.): an expert stays unhit by one token w.p. q1 = 1 − pg + pg·x
    (pg = g/G, x = 1 − k/(gS)); two experts of the same group w.p. 1 − pg + pg·y, y = (gS − k)(gS − k − 1)/(gS(gS − 1))
    (k drawn without replacement); of two different groups w.p. P0 + P1·x + P2·y (P0 / P1 / P2 = neither / one / both
    groups chosen, hypergeometric over G).  Var D = N·e1(1 − e1) + Σ_pairs (e2 − e1²), e = per-token probability ^ T.
    Returns ((E D, sd D), (E pairs, sd pairs)) per rank, or None when the groups do not tile the EP ranks (or g = G:
    the ungrouped rule applies)."""
    G, g = groups
    if not (G > 1 and 0 < g < G and E % G == 0) or ep <= 1:
        return None
    S = E // G
    gS = g * S
    if k > gS:
        return None
    if ep >= G and ep % G == 0 and E % ep == 0:     # (E % ep ≠ 0, e.g. 256 on EP 72: ranks straddle groups → ungrouped)
        nr = E // ep
        p = g / G
        mY = k * nr / gS
        vY = k * (nr / gS) * (1 - nr / gS) * (gS - k) / max(1, gS - 1)
        ex, ex2 = p * mY, p * (vY + mY * mY)
        n_grp, a = nr, 1
    elif ep < G and G % ep == 0:
        a = G // ep
        ex = ex2 = 0.0
        for j, pj in _hyp_pmf(G, a, g):
            pi = j * S / gS
            m_ = k * pi
            v_ = k * pi * (1 - pi) * (gS - k) / max(1, gS - 1)
            ex += pj * m_
            ex2 += pj * (v_ + m_ * m_)
        n_grp = S
    else:
        return None
    pairs = (T * ex, math.sqrt(max(0.0, T * (ex2 - ex * ex))))
    x = 1.0 - k / gS
    y = (gS - k) * (gS - k - 1) / (gS * max(1, gS - 1))
    pg = g / G
    p2 = g * (g - 1) / (G * (G - 1))
    p0 = (G - g) * (G - g - 1) / (G * (G - 1))
    e1 = (1 - pg + pg * x) ** T
    e2s = (1 - pg + pg * y) ** T                       # two experts of one group both unhit
    e2d = (p0 + (1 - p0 - p2) * x + p2 * y) ** T        # two experts of different groups both unhit
    N = a * n_grp
    var = N * e1 * (1 - e1) + a * n_grp * (n_grp - 1) * (e2s - e1 * e1) + a * (a - 1) * n_grp * n_grp * (e2d - e1 * e1)
    dist = (N * (1 - e1), math.sqrt(max(0.0, var)))
    return (round(dist[0], 9), round(dist[1], 9)), (round(pairs[0], 9), round(pairs[1], 9))


def node_dedup_factor(n_experts: int, top_k: int, nodes: int, groups: tuple = (), pool: int = 0) -> float:
    """0.134: network bytes of a node-deduplicating EP all-to-all (DeepEP normal kernels, DeepSeek-V3 report §3.2: a
    token crosses the scale-out network once per destination node, then NVLink forwards it to the node's experts) /
    those of per-pair sends.  With k distinct experts drawn uniformly from E, spread evenly over N nodes:
    E[remote nodes] = (N − 1)(1 − C(E − E/N, k)/C(E, k)) vs E[remote pairs] = k(N − 1)/N, ratio N(1 − miss)/k."""
    if nodes <= 1 or top_k <= 1:
        return 1.0
    if groups:
        # 0.255 group-limited routing (DeepSeek-V3 report §2.1.2 node-limited routing: "each token will be sent to at
        # most M nodes"; the groups tile the nodes): P(a remote node is hit) under the grouped symmetric draw
        G, g = groups
        S = n_experts // G
        gS = g * S
        if G > 1 and 0 < g < G and n_experts % G == 0 and top_k <= gS:
            miss = lambda n: math.comb(gS - n, top_k) / math.comb(gS, top_k) if gS - n >= top_k else 0.0
            if G % nodes == 0:                       # a node holds G / N whole groups
                hit = sum(pj * (1.0 - miss(j * S)) for j, pj in _hyp_pmf(G, G // nodes, g))
                return min(1.0, nodes * hit / top_k)
            if nodes % G == 0 and n_experts % nodes == 0:   # a node holds n = E / N experts of one group
                return min(1.0, nodes * (g / G) * (1.0 - miss(n_experts // nodes)) / top_k)
    pop = pool if pool > n_experts else n_experts
    per = n_experts / nodes
    miss = 1.0
    for i in range(top_k):
        miss *= max(0.0, (pop - per - i) / (pop - i))
    return min(1.0, nodes * (1.0 - miss) / top_k)


def hot_rank_probs(load: tuple, ep: int, top_k: int) -> tuple:
    """0.132: per-rank tuples of inclusion probabilities k · load_e / Σ load (contiguous placement, as skew_from_load).
    moe_routing takes the max occupancy over ranks: the rank reading the most distinct experts need not be the rank
    with the most pairs (a concentrated hot rank reads few experts while a flat cool rank reads many)."""
    n_local = _cdiv(len(load), ep)
    tot = sum(load) or 1.0
    return tuple(tuple(min(1.0, top_k * x / tot) for x in load[r * n_local:(r + 1) * n_local]) for r in range(ep))


def skew_from_load(load: tuple[float, ...], ep: int) -> float:
    """Busiest-rank / mean load of a measured per-expert token distribution under contiguous expert placement
    (expert e on EP rank e // ceil(E / EP), the default linear placement of vLLM / SGLang without EPLB)."""
    if ep <= 1 or not load:
        return 1.0
    n_local = _cdiv(len(load), ep)
    ranks = [sum(load[r * n_local:(r + 1) * n_local]) for r in range(ep)]
    mean = sum(load) / ep
    return max(1.0, max(ranks) / mean) if mean > 0 else 1.0


# ------------------------------------------------------------------ StripedHyena 2 operators (0.170)
def fft_flops(n: int, real: bool = False) -> float:
    """Radix-2 Cooley–Tukey count「推导」: complex FFT of size n = 5·n·log2 n real flops; a real-input / real-output
    transform (rfft / irfft) ≈ half."""
    if n <= 1:
        return 0.0
    f = 5.0 * n * math.log2(n)
    return f / 2 if real else f


def _hyena_ops(model: ModelSpec, li: int, core: AttnCore, ph: Phase, sh: Shard, b_loc: int) -> list[Op]:
    """One hyena operator between its projections (vortex engine.py, as released: prefill_style fft).

    Featurizer: short FIR (conv_kernel taps) over the 3h projection channels, u = x1·v, output gate x2·z.
    se: direct depthwise FIR of hy_len taps.  mr: FFT convolution at fft size N = 2L (engine.parallel_fir takes the
    fftconv path for filters ≥ 128 taps): rfft(u), ·k_f, irfft, + D·u.  lr: the long filter h = Σ_m R·p_m^t is built
    over L (exp per mode), rfft(h) once per forward, complex fft(x1v), ·H, irfft, + D·x1v; for generation the state
    after the prompt comes from ``prefill_via_modal_fft``: fft(p_m^t, N) per channel × mode (once, repeated over the
    batch) and ifft(X·S, N) per sequence × channel × mode, of which only position L−1 is kept.
    Each torch.fft call / elementwise op is its own kernel over fp32 / complex64 tensors, so its operands round-trip
    DRAM (``act_x``: bytes per kernel below, L = prompt tokens, per channel; complex64 = 8 B, fp32 = 4 B「推导」);
    ``act_bytes`` = the largest live intermediate (lr: the modal ifft input and output, 2 × b·ch·M·N·8 B).
    Prefill with ph.ctx > 0 (chunked): the convolution runs over the chunk and the cached state adds M ops / channel
    / token (the release has no chunked-prefill path「假设」).
    Decode (engine.step_fir / step_iir): FIR dot over the cached taps, modal recurrence state = p·state + x1v and
    y = x2·(Σ R·state + D·x1v); exp(log_poles) per step; caches read + written each step."""
    tp = sh.tp
    ch = _cdiv(core.conv_channels // 3, tp)
    t = b_loc * ph.q
    M = core.hy_modes
    st_bytes = core.state_bytes(tp, _fmt(model.state_fmt).bytes)
    vec = _cdiv(core.conv_channels, tp) * core.conv_kernel * t * 2 + 2 * ch * t     # short FIR, x1·v, x2·z
    sfu = 0.0
    act_x = act_peak = 0.0
    launch = 0.0
    fft_rows = 0.0
    comm = []
    if ph.kind == "decode":
        if core.hy in ("se", "mr"):
            vec += 2 * core.hy_len * ch * t + (2 * ch * t if core.hy == "mr" else 0)
        else:
            vec += ch * t * (4 * M + 2)
            sfu += ch * t * M
    else:
        cp = ph.cp if ph.cp > 1 else 1
        # 0.171 context parallel (savanna use_cp_hyena): all-to-all "split_to_full" of x1, x2, v → each CP rank holds
        # the whole sequence for ch / cp channels, convolves, all-to-alls z back ("full_to_split")
        L = ph.cp_len if cp > 1 else ph.q
        ch = _cdiv(ch, cp)
        N = 2 * L
        mode = core.hy_prefill
        fused = mode == "fused"
        if core.hy == "se":
            vec += 2 * core.hy_len * ch * t * cp
        elif core.hy == "mr":
            g = _cdiv(_cdiv(core.hy_groups or core.conv_channels // 3, tp), cp)
            vec += b_loc * ch * (fft_flops(N, True) * 2 + 6 * (N // 2 + 1)) + g * fft_flops(N, True) + 2 * ch * L * b_loc
            if fused:
                fft_rows += b_loc * ch
            else:
                act_x += b_loc * ch * L * (12 + 16 + 16) + g * L * 12      # rfft(u) 4→8, ·k_f 16→8, irfft 8→8
        else:
            vec += ch * (2 * M * L + fft_flops(N, True))                          # h = Σ R·p^t, rfft(h)
            sfu += ch * M * L
            if not fused:
                act_x += ch * (8 * M * L + 12 * L)          # stock compute_filter: (ch, M, L) fp32, then rfft(h)
                act_peak = 2.0 * 4 * ch * M * L             # log_poles·t and its exp, both (ch, M, L) fp32 live
            if mode == "recurrence":
                # engine.prefill_via_direct_recurrence: x1v repeated to (b, ch, L, M, 2) in x1v's bf16, then a Python
                # loop over L of 14 elementwise kernels on the (b, ch, M, 2) fp32 state (3 statements) — no FFT conv
                Sx = b_loc * ch * M
                vec += L * Sx * 12 + b_loc * ch * L * 3
                act_x += 6.0 * b_loc * ch * L * M + L * Sx * 120.0
                act_peak = max(act_peak, 4.0 * b_loc * ch * L * M)
                launch = 14 * L
            else:
                vec += b_loc * ch * ((fft_flops(N, True) if fused else fft_flops(N)) + 6 * (N // 2 + 1)
                                     + fft_flops(N, True) + 3 * L)
                if fused:
                    # what-if「假设」: one kernel per operator, spectra on chip; modal state = Σ_t p^(L−1−t)·x1v_t in
                    # the same pass (2·M flop / token / channel), the filter built on chip (vortex HCL tiled build)
                    vec += b_loc * ch * 2 * M * L
                    fft_rows += b_loc * ch
                else:
                    act_x += b_loc * ch * L * (20 + 24 + 16)    # fft(x1v), ·H, irfft
                    if mode == "fft":
                        vec += ch * M * fft_flops(N) + b_loc * ch * M * (6 * N + fft_flops(N))
                        sfu += ch * M * L
                        act_x += ch * M * L * (4 + 20) + b_loc * ch * M * L * (16 + 32 + 32)   # exp, fft; repeat, X·S, ifft
                        act_peak = max(act_peak, 2.0 * b_loc * ch * M * N * 8)
            if ph.ctx and cp == 1:
                vec += ch * t * 2 * M
        if cp > 1:
            ab = _fmt(model.act_fmt).bytes
            pay = b_loc * ph.q * _cdiv(core.conv_channels // 3, tp) * ab
            comm = [Op(f"cp_hyena_a2a_{n}", "comm", li, comm_kind="alltoall", comm_group=cp, comm_bytes=pay,
                       comm_stride=tp) for n in ("x1", "x2", "v", "z")]
            # featurizer short FIR: the previous rank's last (taps − 1) columns (ExchangeOverlappingRegionsCausal)
            comm.append(Op("cp_hyena_halo", "comm", li, comm_kind="p2p", comm_group=2, comm_stride=tp,
                           comm_bytes=b_loc * _cdiv(core.conv_channels, tp) * max(0, core.conv_kernel - 1) * ab))
    return [Op("state_update", "state", li, vec=vec, sfu=sfu, state_rw=2 * st_bytes * b_loc, kv_read=0.0,
               act_x=act_x, act_bytes=act_peak, stream=act_x > 0 or fft_rows > 0, launch=launch,
               fft_n=N if fft_rows else 0, fft_rows=fft_rows)] + comm


def _rwkv7_ops(model: ModelSpec, li: int, core: AttnCore, ph: Phase, sh: Shard, b_loc: int) -> list[Op]:
    """0.191 RWKV-7 time-mix core per layer (heads split over TP; the GEMMs are in the layer's linears).

    Recurrent step (release rwkv7_recurrent; paper eq. for the generalised delta rule), per token per head, N = 64:
      S·diag(w)            N²   (multiply)
      S·(−κ̂)               2N²  (matrix-vector)        then ⊗ (κ̂·a) and add: 2N²
      v·kᵀ added           2N²
      out = S·r            2N²                         → 9N² vector ops (fp32 state, read + written once)
    Per token per channel outside the state: token-shift delta + 6 lerps (13), channel-mix shift + lerp (3), κ̂ =
    L2-normalise(k·k_k) (4), k·(1 + (a − 1)·k_a) (3), κ̂·a (1), value-residual mix (3), bonus r·k·r_k summed ·v (4),
    GroupNorm ln_x (5), (y + bonus)·g (2) → 38; transcendental: exp(w_log), sigmoid of w / a / v-gate, tanh of the
    w chain, sigmoid of the g chain → ≈ 4 per channel + the low-rank widths.
    Prefill (release rwkv7_chunked, chunk C = 64): per chunk and head, with Q̃ = κ̂·c, B̃ = b/c, K̃ = k/c, R̃ = r·c:
      Q̃B̃ᵀ, Q̃K̃ᵀ, R̃K̃ᵀ, R̃B̃ᵀ  [C,N,C];  unit-lower-triangular solve for U  [C,C,N] (½);  tril(Q̃K̃ᵀ)V, tril(R̃K̃ᵀ)V,
      tril(R̃B̃ᵀ)U  [C,C,N];  Q̃P₀, R̃P₀  [C,N,N];  K̃ᵀV, B̃ᵀU  [N,C,N]; cumulative decay + 4 rescales (5·C·N),
      3 masks (3·C²), state add (N²)."""
    tp = sh.tp
    h_loc = _cdiv(core.n_state_heads, tp)
    N = core.state_dk
    q = ph.q
    t = b_loc * q
    c_loc = h_loc * N
    st_bytes = h_loc * N * N * _fmt(model.state_fmt).bytes + _cdiv(core.conv_channels, tp) * _fmt(model.state_fmt).bytes
    tok_vec = 38 * c_loc
    sfu = t * 4 * c_loc
    ops: list[Op] = []
    if ph.kind == "prefill" and core.lin_chunk:
        C = core.lin_chunk
        nc = _cdiv(q, C)
        cnt = b_loc * h_loc * nc
        g = lambda nm, m_, k_, n_, cz=1.0: Op(nm, "attn", li, m=m_, k=k_, n=n_, count=cnt, causal=cz,
                                              w_fmt="bf16", a_fmt="bf16", act_bytes=cnt * (m_ * k_ + m_ * n_) * 2)
        ops += [g("rwkv_qb", C, N, C), g("rwkv_qk", C, N, C), g("rwkv_rk", C, N, C), g("rwkv_rb", C, N, C),
                g("rwkv_ut_solve", C, C, N, 0.5), g("rwkv_qkv", C, C, N), g("rwkv_rkv", C, C, N), g("rwkv_rbu", C, C, N),
                g("rwkv_qp0", C, N, N), g("rwkv_rp0", C, N, N), g("rwkv_kv", N, C, N), g("rwkv_bu", N, C, N)]
        vec = t * tok_vec + cnt * (5 * C * N + 3 * C * C + N * N)
    else:
        vec = t * (tok_vec + h_loc * 9 * N * N)
    ops.append(Op("state_update", "state", li, vec=vec, sfu=sfu, state_rw=2 * st_bytes * b_loc, kv_read=0.0))
    return ops


# ------------------------------------------------------------------ attention core
def _attn_core_ops(model: ModelSpec, li: int, core: AttnCore, ph: Phase, sh: Shard, b_loc: int,
                   alias: bool = False) -> list[Op]:
    ops: list[Op] = []
    ab = _fmt(model.act_fmt).bytes
    kvb = _fmt(model.kv_fmt).bytes
    q, tp = ph.q, sh.tp
    if core.kind == "linear" and core.lin == "hyena":
        return _hyena_ops(model, li, core, ph, sh, b_loc)
    if core.kind == "linear" and core.lin == "rwkv7":
        return _rwkv7_ops(model, li, core, ph, sh, b_loc)
    if core.kind == "linear":
        h_loc = _cdiv(core.n_state_heads, tp)
        t = b_loc * q
        # 0.61.1: + the conv state (kernel − 1 columns per channel) — stored by memplan, now also read / written
        st_bytes = (h_loc * core.state_dk * core.state_dv
                    + _cdiv(core.conv_channels, tp) * max(0, core.conv_kernel - 1)) * _fmt(model.state_fmt).bytes
        dk, dv = core.state_dk, core.state_dv
        conv = _cdiv(core.conv_channels, tp) * core.conv_kernel * t * 2
        delta = core.lin in ("gdn", "kda")
        # per token per head, outside the matmuls: β-scaling of k, v (delta rule), q·k L2 norm (3/elem), gate decay on q, k
        tok_vec = (dk + dv if delta else 0) + (6 * dk if delta else 0) + 2 * dk + 2 * dv
        if ph.kind == "prefill" and core.lin_chunk:
            # chunked form of the reference kernels (transformers torch_chunk_gated_delta_rule / chunk_kimi_delta_attention,
            # chunk 64; MiniMax modeling_minimax_text_01 lightning attention, BLOCK 256) — per chunk of C tokens, per head:
            #   delta rule: A_kk = (k·β)Kᵀ [C,dk,C]; UT solve of A on [v·β | k·β·decay] [C,C,dv+dk] (unitriangular, ½);
            #               v' = k_cumdecay·S [C,dk,dv]
            #   all:        A_qk = Q·Kᵀ [C,dk,C]; intra A_qk·V [C,C,dv]; inter Q·S [C,dk,dv]; state Kᵀ·V [dk,C,dv]
            # operands in bf16, fp32 accumulate (FLA kernels the transformers code dispatches to)
            C = core.lin_chunk
            nc = _cdiv(q, C)
            cnt = b_loc * h_loc * nc
            g = lambda nm, m_, k_, n_, cz=1.0: Op(nm, "attn", li, m=m_, k=k_, n=n_, count=cnt, causal=cz,
                                                  w_fmt="bf16", a_fmt="bf16", act_bytes=cnt * (m_ * k_ + m_ * n_) * 2)
            if delta:
                ops += [g("lin_akk", C, dk, C), g("lin_ut_solve", C, C, dv + dk, 0.5), g("lin_vprime", C, dk, dv)]
            if core.lin == "ssd" and core.state_groups:
                # 0.278 Mamba-2 SSD (mamba_chunk_scan_combined): C·Bᵀ per chunk is formed once per B / C group
                # (n_groups) and shared by its heads 「来源」 ssd_bmm chunk_size × ngroups; the rest is per head
                gl = _cdiv(core.state_groups, tp)
                cg = b_loc * gl * nc
                ops.append(Op("lin_aqk", "attn", li, m=C, k=dk, n=C, count=cg, w_fmt="bf16", a_fmt="bf16",
                              act_bytes=cg * (C * dk + C * C) * 2))
            else:
                ops.append(g("lin_aqk", C, dk, C))
            ops += [g("lin_intra", C, C, dv), g("lin_inter", C, dk, dv), g("lin_state", dk, C, dv)]
            # per chunk: decay masks (build 3/elem, apply to A_qk / A_kk); KDA per-channel gate exp on q, k, k·β
            # (FLA chunk_kda factorisation); v_new subtract; state decay
            ch_vec = (3 + (2 if delta else 1)) * C * C + (3 * C * dk if core.lin == "kda" else 0) + C * dv + dk * dv
            vec = t * h_loc * tok_vec + cnt * ch_vec
        else:
            # recurrent step (fused_recurrent kernels): state decay, kᵀ·S read (delta), rank-1 update, q·S read
            vec = t * h_loc * (tok_vec + (7 if delta else 5) * dk * dv)
        if core.lin == "conv":
            # 0.114 LFM2 gated short conv: B·x before and C·y after the depthwise conv (1 op / element each)
            vec += t * _cdiv(core.conv_channels, tp) * 2
        # 0.222 spec verify (decode, q > 1): per-draft-token intermediate states + accepted-state commit (SPEC_STATE_SNAPSHOT)
        upd = (7 if delta else 5) * dk * dv            # one sequential recurrent token update per head (vec ops)
        if ph.kind == "decode":
            E = spec_expected(q, ph.acc) if q > 1 else 1.0
            tr = E / ph.track if ph.track else 0.0     # 0.224: radix-track boundaries crossed per step (TRACK note)
        if ph.roll == "replay" and ph.kind == "decode" and q > 1 and core.lin in ("gdn", "kda"):
            # 0.224 ReplaySSM verify (REPLAY_RING_DEFAULT note): GDN circular ring with early flush, KDA fold-every-commit
            L_r = replay_ring_len(q)
            if core.lin == "gdn":
                hbar, fpos, fold_n = replay_cycle(L_r, q, E)
                ent = h_loc * ((dk + dv) * 4.0 + 4.0)
                # verify: output-only reconstruction of the window against hbar history records (Q·K / decay / D
                # accumulate per record and window token 「推导」) + exact sequential fold of fold_n entries on flush steps
                vec += b_loc * h_loc * (hbar * q * (dk + dv) * 2 + fold_n * upd)
                rw = b_loc * (st_bytes * (1.0 + 2.0 * fpos) + ent * (hbar + q + fold_n))
                # radix track: materialize the checkpoint + history up to the boundary into the track slot
                vec += b_loc * h_loc * tr * hbar * upd
                rw += b_loc * tr * (2.0 * st_bytes + ent * hbar)
            else:
                # KDA fold-every-commit: recurrent verify (state read, no write), raw window written; the commit replays
                # the E accepted records into the checkpoint (read + write)
                ent = h_loc * (4.0 * dv + 8.0 * dk + 4.0)
                vec += b_loc * h_loc * E * upd
                rw = b_loc * (3.0 * st_bytes + ent * (q + E) + tr * st_bytes)
            ops.append(Op("state_update", "state", li, vec=vec + conv, state_rw=rw, kv_read=0.0))
            return ops
        n_st = (q + 3) if (SPEC_STATE_SNAPSHOT and ph.kind == "decode" and q > 1) else 2
        if ph.kind == "decode" and ph.track:
            n_st += tr                                 # 0.224: accepted state copied to the radix track slot
        ops.append(Op("state_update", "state", li, vec=vec + conv,
                      state_rw=n_st * st_bytes * b_loc, kv_read=0.0))
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
        if not core.causal and core.window is None and core.topk is None and core.compress == 1:
            causal = 1.0                     # 0.106 attn_mode = bidir: every query sees all ctx_tot keys
        if core.window is not None or core.topk is not None or core.compress > 1:
            ce = min(ce, ctx_tot)
            # 0.61.1: the first positions attend fewer keys than the cap (min(p, window / top-k / ⌈p/c⌉ + w)):
            # charge the exact mean over the prompt positions, not the cap for every query (was ≤ 2× at prompt ≤ cap)
            causal = core.keys_mean(ph.ctx, q) / ce
    if core.kind == "gqa":
        kv_loc, hq_loc = gqa_local(core.n_q, core.n_kv, tp)
        g = max(1, _cdiv(hq_loc, kv_loc))     # 0.63: query rows per KV-head tile (uneven TP pads, never drops)
        cnt = b_loc * kv_loc
        ops.append(Op("qk", "attn", li, m=q * g, k=core.qk_dim, n=ce, count=cnt, causal=causal,
                      act_bytes=cnt * (q * g * core.qk_dim) * ab))
        ops.append(Op("pv", "attn", li, m=q * g, k=ce, n=core.v_dim, count=cnt, causal=causal,
                      act_bytes=cnt * (q * g * core.v_dim) * ab))
        kv_tok = kv_loc * (core.qk_dim + core.v_dim) / core.compress
        sm_elems = cnt * q * g * ce * causal
        if core.sink:
            # gpt-oss reference sdpa: the per-head sink logit is concatenated to every score row, softmaxed, dropped
            sm_elems += cnt * q * g
    else:  # mla (latent cache, replicated on every attention-TP rank)
        hq_loc = _cdiv(core.n_q, tp)
        lat = core.kv_lora + core.rope_dim
        if ph.kind == "decode":
            # absorbed: scores over the latent, output in latent space (W_UK / W_UV as per-head kv_b.uk / kv_b.uv GEMMs)
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
        if core.sink:                      # DeepSeek-V4 sparse_attn: learnable attn_sink logit per head
            sm_elems += b_loc * q * hq_loc
    if core.idx_heads:
        n_keys = math.ceil(ctx_tot / core.compress)
        # 0.61.1: prefill — the query at position p scores ⌈p/c⌉ compressed keys (exact mean; was the full square for
        # top-k / compressed layers); decode keeps every cached key
        icaus = 1.0
        if ph.kind != "decode":
            cs = lambda m_: (lambda a, r: core.compress * a * (a + 1) // 2 + r * (a + 1))(*divmod(m_, core.compress))
            icaus = (cs(ctx_tot) - cs(ph.ctx)) / q / n_keys
        ib = core.idx_bytes_per_token(kvb) * core.compress          # bytes per cached indexer key
        pool_vec = 0.0
        if core.qsa:
            b_ = core.qsa
            n_keys = max(ctx_tot // b_, 1)
            if core.idx_score_type == "max":
                # 0.355 MiniMax M3.1-Flash block-level MSA: per-block max of indexer scores (no pooling / k norm per
                # forward; indexer q LayerNorm + RoPE, score all cached keys, max over each block, top-(topk/b) blocks)
                fs = lambda m_: (lambda a_, c_: b_ * a_ * (a_ - 1) // 2 + a_ * (c_ + 1))(*divmod(m_, b_))
                icaus = 1.0 if ph.kind == "decode" else min(1.0, (fs(ctx_tot) - fs(ph.ctx)) / q / n_keys)
                pool_vec = b_loc * (ctx_tot // b_) * core.idx_dim    # per-block max reduction (1/elem)
            else:
                # Qwen4Exp QSA: score ⌊p/b⌋ mean-pooled key blocks (every forward pools the cached raw keys once: mean
                # 1/elem, k LayerNorm 4/elem, RoPE on the block start; the reference loop recomputes this per query —
                # identical values, counted once), q LayerNorm + RoPE, ReLU + head sum, top-(budget/b) blocks
                fs = lambda m_: (lambda a_, c_: b_ * a_ * (a_ - 1) // 2 + a_ * (c_ + 1))(*divmod(m_, b_))
                icaus = 1.0 if ph.kind == "decode" else min(1.0, (fs(ctx_tot) - fs(ph.ctx)) / q / n_keys)
                pool_vec = b_loc * (ctx_tot // b_) * (b_ * core.idx_dim + 4 * core.idx_dim + 3 * core.idx_rope)
            if core.idx_sum:
                # 0.144 Step-5 (inference/kernel.py update_summaries_*): only regions completed in this pass are
                # compressed (z-gated weighted sum 3/elem, k LayerNorm 4/elem, RoPE, fp8 round 2/elem) and cached;
                # the indexer reads one fp8 summary (idx_dim bytes) per completed region
                pool_vec = b_loc * (ctx_tot // b_ - ph.ctx // b_) * (b_ * core.idx_dim * 3 + 6 * core.idx_dim
                                                                     + 3 * core.idx_rope)
                ib = float(core.idx_dim)
        if core.dual:
            # V4.1 Indexer: the query at position p scores ⌊p/r⌋ compressed positions (index keys of the kv source)
            r_ = core.compress
            n_keys = max(ctx_tot // r_, 1)
            fs = lambda m_: (lambda a_, b_: r_ * a_ * (a_ - 1) // 2 + a_ * (b_ + 1))(*divmod(m_, r_))
            icaus = 1.0 if ph.kind == "decode" else min(1.0, (fs(ctx_tot) - fs(ph.ctx)) / q / n_keys)
            ib = core.idx_kbytes
            nl = b_loc * (ctx_tot // r_ - ph.ctx // r_)               # latents completed in this pass
            if core.idx_owner:   # k = k_norm(wk(latent)), RoPE, fp4 quant (simulated) — on the new latents only
                ops.append(Op("idx_wk", "gemm", li, m=nl, k=core.kv_lora, n=core.idx_dim, w_fmt="bf16", w_bits=16,
                              a_fmt="bf16", act_bytes=nl * (core.kv_lora + core.idx_dim) * 2, replicated=tp))
                pool_vec += nl * (4 * core.idx_dim + 3 * core.idx_rope + 2 * core.idx_dim)
            if core.cand == "src":   # select_candidate_blocks: amax per block (1/position) + top-k over blocks
                pool_vec += b_loc * q * n_keys * icaus * (1 + 1 / max(core.cand_block, 1))
            elif core.cand == "use":  # mask to the source's candidate blocks (1/position)
                pool_vec += b_loc * q * n_keys * icaus
        if core.idx_kpool:
            # transformers Glm5NextTextIndexer.get_pooled_states: every forward rebuilds the pools over all cached keys —
            # gate + APE (1/elem), softmax over the kpool slots (5/elem), weighted sum (2/elem) per key channel; then the
            # q·k GEMM, ReLU / head weighting and top-(topk/kpool) run over ⌈ctx/kpool⌉ pools (tail pool appended raw)
            pool_vec = b_loc * ctx_tot * core.idx_dim * 8
            n_keys = math.ceil(ctx_tot / core.idx_kpool)
            if ph.kind != "decode":
                kp = core.idx_kpool                 # the query at position p sees ⌊p/kp⌋ complete pools
                fl = lambda m_: (lambda a_, r_: kp * a_ * (a_ - 1) // 2 + a_ * (r_ + 1))(*divmod(m_, kp))
                icaus = (fl(ctx_tot) - fl(ph.ctx)) / q / max(n_keys, 1)
            icaus = min(1.0, icaus)
        t_new = b_loc * q
        # per scored (query, key) pair, after the q·k GEMM (DeepSeek-V3.2 inference/model.py fp8_index; transformers
        # GlmMoeDsaIndexer): ReLU per head, × head weight and sum over heads (3·heads), fp8 key-scale multiply (fp8
        # path), top-k selection 1 compare per candidate (vector-count convention, like softmax's 5 per element)
        ih_loc = _cdiv(core.idx_heads, tp) if core.idx_split else core.idx_heads
        rep_i = 1 if core.idx_split else tp
        ksh = core.idx_kshard and tp > 1
        if ksh:     # 0.87 key_split: index keys round-robin over the TP ranks (causal mean per rank unchanged)
            n_keys, rep_i = _cdiv(n_keys, tp), 1
        per_pair = 3 * ih_loc + 1 + (1 if core.idx_fp8 else 0)
        # per new token: k LayerNorm (4/elem, as the layer norms), RoPE on q heads + k (3/elem, as attention RoPE);
        # fp8 path: Hadamard rotation of q heads + k (d·log2 d adds, fast_hadamard_transform) and block fp8 quant
        # (2/elem, the converted-element convention of evaluate._op_seconds)
        rows = core.idx_heads + 1
        per_tok = 4 * core.idx_dim + 3 * rows * core.idx_rope
        if core.idx_fp8:
            per_tok += rows * core.idx_dim * (math.log2(core.idx_dim) + 2)
        if core.qsa:
            if core.idx_score_type == "max":
                # 0.355 MiniMax M3.1-Flash: q LayerNorm + RoPE, top-k over block-wise max
                per_tok = core.idx_heads * (4 * core.idx_dim + 3 * core.idx_rope)
                per_pair = 1    # max reduction per block, then top-k
            else:
                # Qwen4Exp QSA: q LayerNorm + RoPE; per pair ReLU + head sum, top-k
                per_tok = core.idx_heads * (4 * core.idx_dim + 3 * core.idx_rope)
                per_pair = 2 * core.idx_heads + 1
        if core.dual:      # V4.1: RoPE + fp4-simulated quant on the local q heads; weights GEMM is a Linear
            per_tok = ih_loc * (3 * core.idx_rope + 2 * core.idx_dim)
        if core.idx_cmp:
            # DeepSeek-V4 Indexer: RoPE + Hadamard + fp4-simulated quant on the local q heads; keys from its own
            # Compressor (gated softmax pooling over 2·ratio overlapped slots per channel: ape 1 + softmax 5 + sum 2;
            # per compressed entry RMSNorm 4, RoPE, Hadamard, fp4 quant 2)
            d_ = core.idx_dim
            per_tok = ih_loc * (3 * core.idx_rope + d_ * (math.log2(d_) + 2)) + 2 * d_ * 8 \
                + (4 * d_ + 3 * core.idx_rope + d_ * (math.log2(d_) + 2)) / core.compress
        ops.append(Op("indexer_score", "attn", li, m=q * ih_loc, k=core.idx_dim, n=n_keys, count=b_loc,
                      causal=icaus, replicated=rep_i, w_fmt="fp8" if core.idx_fp8 else "",
                      a_fmt="fp8" if core.idx_fp8 else "bf16",
                      kv_read=(b_loc * (ctx_tot if core.idx_kpool else n_keys) * ib) if ph.kind == "decode" else
                      (b_loc * (ph.ctx // core.compress) * ib if core.dual else 0.0),
                      vec=t_new * n_keys * icaus * per_pair + (0.0 if ksh else t_new * per_tok) + pool_vec))
        if ksh:
            # q-side prep (LayerNorm / RoPE / Hadamard / fp8 quant of the index q heads) on every rank; local top-k
            # (in the score vec), then all-gather (fp32 score, int32 index) of k_loc candidates per rank and a merge
            # over TP·k_loc candidates (1 compare each, the top-k convention) on every rank
            k_loc = min(core.topk or n_keys, n_keys)
            ops.append(Op("indexer_q_prep", "vector", li, replicated=tp, vec=t_new * per_tok))
            ops.append(Op("indexer_topk_gather", "comm", li, comm_kind="allgather", comm_group=tp,
                          comm_bytes=t_new * k_loc * 8))
            ops.append(Op("indexer_topk_merge", "vector", li, replicated=tp, vec=t_new * tp * k_loc))
        if core.idx_split and tp > 1:     # dist.all_reduce(index_score): fp32 [tokens, compressed keys]
            ops.append(Op("indexer_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp,
                          comm_bytes=t_new * n_keys * icaus * 4))
        if core.idx_sum and core.qsa:
            # 0.145 hostile review: Step-5 writes one fp32 + fp8 summary (5·idx_dim bytes) per region completed in
            # this pass — was idx_dim bytes per new token (8/5× the stored growth for qsa 8)
            iw = b_loc * (ctx_tot // core.qsa - ph.ctx // core.qsa) * core.idx_dim * 5.0
        else:
            iw = (nl * ib if core.idx_owner else 0.0) if core.dual else (t_new * ib / core.compress / (tp if ksh else 1))
        ops.append(Op("indexer_kv_write", "vector", li, replicated=1 if ksh else tp, kv_write=iw))
    if core.cmp_coff:
        # DeepSeek-V4 Compressor (KV path): gated softmax pooling of coff·head_dim channels per token (ape 1 + softmax 5
        # + weighted sum 2); per compressed entry RMSNorm (4), RoPE, fp8 quant of the non-rope dims (2)
        # V4.1: no APE (7/elem), ratio 1 = plain projection (no pooling); fp4-simulated quant of the whole entry
        hd_ = core.kv_lora
        pool = 0 if core.compress == 1 else core.cmp_coff * hd_ * (7 if core.cmp_noape else 8)
        ops.append(Op("compressor_pool", "vector", li, replicated=tp,
                      vec=b_loc * q * (pool + (4 * hd_ + 3 * core.rope_n + 2 * hd_) / core.compress)))
    # KV cache traffic: read the attended entries (sparse/window aware), write the new ones
    if ph.kind == "decode":
        # 0.61.1: ``attended`` counts cache entries (compressed layers: ⌈ctx/c⌉ + window), each a full entry of
        # kv_tok·c bytes — was × kv_tok (per original token), i.e. c× too few bytes (DeepSeek-V4: 4× / 128×)
        attended = min(ce, ctx_tot)
        kv_read = b_loc * attended * kv_tok * core.compress * kvb
    else:
        # prefix (cached) context only; a sliding-window-only layer needs the last ``window`` tokens of it (0.61.1)
        pre = min(ph.ctx, core.window) if (core.window and core.compress == 1 and core.topk is None) else ph.ctx
        kv_read = b_loc * pre * kv_tok * kvb
    kv_write = b_loc * q * kv_tok * kvb if core.causal else 0.0   # 0.106 bidir: encoder forward, no KV cache
    if core.win_bytes:
        # V4.1: window ring buffer per layer + the shared compressed cache (written by its kv source only)
        r_ = max(core.compress, 1)
        if ph.kind == "decode":
            kv_read = b_loc * (min(core.window, ctx_tot) * core.win_bytes
                               + (min(core.topk, ctx_tot // r_) * core.cmp_bytes if core.dual else 0.0))
        else:
            kv_read = b_loc * (min(core.window, ph.ctx) * core.win_bytes
                               + ((ph.ctx // r_) * core.cmp_bytes if core.dual else 0.0))
        kv_write = b_loc * q * core.win_bytes + (b_loc * (ctx_tot // r_ - ph.ctx // r_) * core.cmp_bytes
                                                 if core.kv_owner else 0.0)
    if alias:
        # 0.369 Gemma-4: this layer reads the donor layer's KV tensors and does not write its own cache.
        kv_write = 0.0
    ops.append(Op("softmax", "vector", li, vec=sm_elems * 5, sfu=sm_elems, kv_read=kv_read, kv_write=kv_write,
                  replicated=(tp if core.kind == "mla" else 1)))
    if ph.cp > 1 and ph.kind == "prefill" and core.topk is not None:
        # 0.89: the chunk phase charges every query of the busiest zigzag rank top-k keys (its positions are all past
        # the first top-k); the mean rank also holds the early positions that see fewer → Op.share = mean-rank keys /
        # busiest-rank keys, so the work KPI is exact (step time stays the busiest rank's)
        f = core.keys_sum(ph.cp_len) / (ph.cp * q * core.keys_mean(ph.ctx, q))
        ops = [replace(o, share=o.share * f) if o.name in ("qk", "pv", "softmax") else o for o in ops]
    return ops


# ------------------------------------------------------------------ layer ops
# 0.323: a run-scoped memo of build_rank_ops (None = off).  The layer IR is a pure function of (model, layer, phase,
# shard, glue) — no chip field enters it — and every argument and Op is a frozen dataclass, yet a capped energy DSE
# rebuilt it for each of the 16 bisection clocks of every design.  Scoped (rank_memo()) rather than global so module
# flags that tests toggle (e.g. SPEC_STATE_SNAPSHOT) are never served stale; callers get a fresh list.
_RANK_MEMO: dict | None = None


@contextmanager
def rank_memo():
    """Memoise build_rank_ops for the duration of the block (nests: an outer memo is kept)."""
    global _RANK_MEMO
    if _RANK_MEMO is not None:
        yield
        return
    _RANK_MEMO = {}
    try:
        yield
    finally:
        _RANK_MEMO = None


def build_rank_ops(model: ModelSpec, li: int, ph: Phase, sh: Shard = Shard(), layer: Layer | None = None,
                   glue: bool = False) -> list[Op]:
    """Ops of one rank for transformer layer ``li`` (or an explicit MTP ``layer``), with work shares (rank_shares).
    0.177: ``glue`` adds the GPU-runtime kernels of glue_ops."""
    memo = _RANK_MEMO
    if memo is None:
        return _build_rank_ops(model, li, ph, sh, layer, glue)
    key = (model, li, ph, sh, layer, glue)
    hit = memo.get(key)
    if hit is None:
        hit = memo[key] = tuple(_build_rank_ops(model, li, ph, sh, layer, glue))
    return list(hit)


def _build_rank_ops(model: ModelSpec, li: int, ph: Phase, sh: Shard, layer: Layer | None, glue: bool) -> list[Op]:
    ops = rank_shares(_rank_ops(model, li, ph, sh, layer), sh,
                      lambda: _rank_ops(model, li, ph, Shard(), layer))
    if ph.labels and ph.kind == "prefill" and ph.ctx == 0:
        ops = [replace(o, kv_write=0.0) if o.kv_write else o for o in ops]   # 0.191: one forward, nothing cached
    return ops + glue_ops(ops, li, ph) if glue else ops


_QUANT_ACT = ("fp8", "fp8_tensor", "nvfp4", "mxfp4", "int8")


def _cast_family(name: str) -> str | None:
    """Input tensor a quantised-activation GEMM reads: one cast kernel per distinct tensor.  Down projections read the
    activation output, which the fused SwiGLU + cast kernel quantises (trace: *swiglu_forward_and_per_token_cast_to_fp8*);
    absorbed-MLA kv_b bmms and the router read bf16."""
    if "down" in name or "kv_b" in name or "router" in name:
        return None
    if name.endswith(".o") or name.endswith("o_proj"):
        return "o"
    if "q_b" in name:
        return "q_b"
    if name.startswith("attn."):
        return "attn_in"
    if any(x in name for x in ("gate_up", ".up", "gate")):
        return "ffn_in"
    return None


def glue_ops(ops: list[Op], li: int, ph: Phase) -> list[Op]:
    """0.177: kernels a GPU serving stack runs per layer that the op list leaves implicit.  Kernel inventory from the
    public DeepSeek-V3 decode trace (deepseek-ai/profile-data, compute stream, per layer-forward): per_token_cast_to_fp8
    2.95, layer_norm 2.98 (attn / ffn / q_a), flash_fwd_splitkv_mla_combine 0.98, top2_sum_gate 0.96, splitKreduce 0.96
    (cuBLAS bf16 router GEMM), batched + shared SwiGLU-and-cast 1.95, DeepEP dispatch_ll / combine_ll 1.92 each
    (send + recv halves).  Bytes are the tensors each kernel reads and writes (bf16 in, fp8 + fp32 per-128 scales out)."""
    out: list[Op] = []
    gem = [o for o in ops if o.kind == "gemm"]
    names = {o.name for o in ops}
    dispatch = "moe_dispatch" in names
    seen: set = set()
    for o in gem:
        if o.a_fmt not in _QUANT_ACT:
            continue
        f = _cast_family(o.name)
        if f is None or f in seen or (f == "ffn_in" and dispatch):     # DeepEP low-latency dispatch casts in-kernel
            continue
        seen.add(f)
        r, k = o.m * max(o.count, 1) if f != "ffn_in" else o.m, o.k
        out.append(Op(f"glue.cast_{f}", "vector", li, vec=r * k * 3, act_bytes=r * k * 3 + r * k / 128 * 4))
    qb = next((o for o in gem if "q_b" in o.name), None)
    if qb is not None:                                                     # MLA q_a_layernorm (RMSNorm over q_lora)
        out.append(Op("glue.q_a_norm", "vector", li, vec=qb.m * qb.k * 4, act_bytes=qb.m * qb.k * 4))
    if ph.kind == "decode":
        qk = next((o for o in ops if o.kind == "attn" and o.name.startswith("qk")), None)
        pv = next((o for o in ops if o.kind == "attn" and o.name.startswith("pv")), None)
        if qk is not None and pv is not None:
            # split-KV combine: read the per-split partial O (fp32) + LSE, write O — two splits 「假设」 (lower bound)
            rows = pv.m * max(pv.count, 1)
            out.append(Op("glue.attn_combine", "vector", li, vec=rows * pv.n * 2 * 3,
                          act_bytes=rows * (2 * (pv.n + 1) * 4 + pv.n * 2)))
    rt = next((o for o in gem if "router" in o.name), None)
    if rt is not None:
        out.append(Op("glue.router_topk", "vector", li, vec=rt.m * rt.n * 8, act_bytes=rt.m * rt.n * 4))
        if ph.kind == "decode":
            out.append(Op("glue.router_splitk_reduce", "vector", li, vec=rt.m * rt.n * 2,
                          act_bytes=rt.m * rt.n * 4 * 3))
    sg = next((o for o in gem if "shared_gate_up" in o.name), None)
    if sg is not None:
        out.append(Op("glue.shared_act", "vector", li, vec=sg.m * sg.n * 2, act_bytes=sg.m * sg.n * 2 * 1.5))
    return out


def _hc_ops(model: ModelSpec, li: int, L: Layer, t: int, tp: int, which: str) -> list[Op]:
    """One mHC hyper-connection (transformers Glm5NextTextHyperConnection + the decoder-layer mix), per token:
    RMSNorm of the N·h flattened streams; fn GEMM [N·h → (2+N)·N]; sigmoid pre / post, softmax + Sinkhorn-Knopp comb
    (1 + 2·(iters − 1) normalisations of N×N); collapse Σ pre·streams; after the block post ⊗ out + combᵀ·residual.
    The residual streams are replicated on every TP rank.  fn weights are in misc_params (read with the hot set)."""
    N, h = L.hc, model.hidden
    nh, nw = N * h, L.hc_rows or (2 + N) * N          # 0.320: Raon_mHC selector rows 2N + (N−1)²
    vec = t * (4 * nh                                   # unweighted RMSNorm over N·h
               + 2 * N * 4 + N * N * 5                  # sigmoid pre / post, softmax comb
               + (2 * L.hc_iters - 1) * 2 * N * N       # Sinkhorn row / column normalisations (sum + divide)
               + 2 * nh                                 # collapse
               + nh * (2 * N + 2))                      # post ⊗ out (N·h) + combᵀ·residual (2N·h) + add (N·h)
    return [Op(f"{which}_hc_fn", "gemm", li, m=t, k=nh, n=nw, w_fmt="bf16", w_bits=16, a_fmt="bf16",
               act_bytes=t * (nh + nw) * 2, replicated=tp),
            Op(f"{which}_hc_mix", "vector", li, vec=vec, replicated=tp)]


def _gres_op(model: ModelSpec, li: int, L: Layer, t: int, tp: int, which: str) -> Op:
    """One Qwen4Exp GatedResidual (transformers Qwen4ExpTextGatedResidual) around a block, outside its Linears
    (down / up / inject GEMMs are Linears of the layer): grouped RMSNorm over N·h (4/elem), SiLU on the low rank (4),
    sigmoid input mix (4/elem), weighted stream mean (2/elem), injection sigmoid (4 / stream); after the block
    out ⊗ injection (1/elem) + residual add (1/elem).  Replicated over TP like the residual streams."""
    Nh = L.gres_n * model.hidden
    r = next((l.n for l in L.attn_linears if l.name == "gr_a_down"), 0)
    return Op(f"{which}_gres", "vector", li, replicated=tp,
              vec=t * (4 * Nh + 4 * r + 4 * Nh + 2 * Nh + 4 * L.gres_n + 2 * Nh))


def _dea_ops(model: ModelSpec, li: int, L: Layer, ph: Phase, sh: Shard, b_loc: int) -> list[Op]:
    """0.361 DeepEmbedAttention, one layer (BlinkDL rwkv_v8_rc00_hybrid_demo.py forward_seq).

    New tokens only: qq / kk / vv.  The cache is the fp16 kk and vv rows (demo DTYPE = torch.half), appended
    each step.  kkup / vvup run over every cached row (ctx + q), then the ennk / emm row of every cached token
    id, LayerNorm, and one causal head: scores = 32 * tanh((q @ k^T) / 512), softmax, @ v.  The up-projections
    are weight GEMMs; their left-hand side is the cache, counted as kv_read / kv_write (fp16, 2 B) rather than
    as a second activation read.  The reconstructed K (kq) and V (vd) are activations.  When they do not fit
    the SRAM activation half, the producer writes them and dea_qk reads them back (memplan.act_stream; V is
    not another copy of K).  Reference is one GPU: projections and the attention matmuls are replicated, the
    cache is not TP-split.
    """
    c = L.core
    if not c.dea_q:
        return []
    tp = sh.tp
    q = ph.q
    t = b_loc * q
    s = ph.ctx + q
    rows = b_loc * s
    by = {l.name: l for l in L.attn_linears}
    cache_b = 2.0

    def proj(name: str, m: int) -> Op:
        lin = by[name]
        k, n = _lin_local(lin, tp, c)
        return gemm(model, f"attn.{name}", li, m, k, n, "attn", params=k * n, replicated=_repl(lin, tp))

    if ph.kind == "decode" or s <= 0:
        causal = 1.0
    else:
        # Same mean as full-softmax GQA: 0.5 * (1 + ctx / (ctx + q)).
        causal = 0.5 * (1.0 + ph.ctx / s)
    bits = model.lookup_bits or model.fmt("embed").bits or 16.0
    ops = [proj("dea_qq", t), proj("dea_kk", t), proj("dea_vv", t)]
    ops[1] = replace(ops[1], kv_write=t * c.dea_k * cache_b)
    ops[2] = replace(ops[2], kv_write=t * c.dea_v * cache_b)
    ops.append(Op("dea_lnq", "vector", li, vec=t * c.dea_q * 4, sfu=t, replicated=tp))
    # One int32 token-id history for the stack (demo state[n_layer * 3], torch.int).  Appended each step and
    # read once to index ennk / emm.  Resident with the fp16 cache; not a lookup table, so it stays on kv_read.
    if not any(model.layers[j].core.dea_q for j in range(min(li, len(model.layers)))):
        ops.append(Op("dea_ids", "vector", li, kv_read=rows * 4.0, vec=rows, replicated=tp))
    kk = replace(proj("dea_kkup", rows), in_elems=1, kv_read=rows * c.dea_k * cache_b)
    vv = replace(proj("dea_vvup", rows), in_elems=1, kv_read=rows * c.dea_v * cache_b)
    ops.append(kk)
    ops.append(Op("dea_ennk", "lookup", li, kv_read=rows * c.dea_k_emb * bits / 8.0, vec=rows * c.dea_k_emb,
                  replicated=tp))
    ops.append(Op("dea_lnk", "vector", li, vec=rows * c.dea_kq * 4, sfu=rows, replicated=tp))
    ops.append(vv)
    ops.append(Op("dea_tanh", "vector", li, vec=rows * c.dea_vd, sfu=rows * c.dea_vd, replicated=tp))
    ops.append(Op("dea_emm", "lookup", li, kv_read=rows * c.dea_v_emb * bits / 8.0, vec=rows * c.dea_v_emb,
                  replicated=tp))
    ops.append(Op("dea_lnv", "vector", li, vec=rows * c.dea_vd * 4, sfu=rows, replicated=tp))
    sm = b_loc * q * s * causal
    # One head. K is [s, kq] and V is [s, vd]; they are not the same width, so flash streaming must not
    # treat V as another K. Replicated under TP like the projections (the reference is one GPU).
    ops.append(Op("dea_qk", "attn", li, m=q, k=c.dea_kq, n=s, count=b_loc, causal=causal, v_dim=c.dea_vd,
                  replicated=tp, act_bytes=b_loc * q * c.dea_kq * cache_b))
    ops.append(Op("dea_softcap", "vector", li, vec=sm * 3, sfu=sm, replicated=tp))
    ops.append(Op("dea_softmax", "vector", li, vec=sm * 5, sfu=sm, replicated=tp))
    ops.append(Op("dea_pv", "attn", li, m=q, k=s, n=c.dea_vd, count=b_loc, causal=causal, replicated=tp,
                  act_bytes=b_loc * q * c.dea_vd * cache_b))
    return ops


# SituAndMul (moonshotai/Kimi-K3 modeling_kimi_linear.py): the arithmetic is
# fp32. A constant reciprocal is not an element. tanh is one transcendental,
# the same count as dea_tanh. sigmoid is the SiLU census without the final
# multiply by x: exp, one add, one reciprocal.
_SITU_VEC = 11
_SITU_SFU = 4


def _situ_op(li: int, n: float) -> Op:
    """beta * tanh(gate / beta) * sigmoid(gate) * (linear_beta * tanh(up / linear_beta)).

    Width is the intermediate size, not the fused gate_up width. Casts to
    fp32 and back are the dtype, not extra element ops.
    """
    if n < 0:
        raise ValueError("kimi situ width is negative")
    return Op("situ", "vector", li, vec=float(n) * _SITU_VEC, sfu=float(n) * _SITU_SFU, a_fmt="fp32")


def _evo2_mlp_vecs(kind: str, nact: int, li: int, af: str) -> list[Op]:
    """ParallelGatedMLP product, and F.gelu on layer 0 only.

    F.gelu defaults to approximate='none': x * 0.5 * (1 + erf(x / sqrt(2))).
    That is four arithmetic ops and one erf. The class does not upcast.
    Later layers set the activation to Identity, so only the multiply remains.
    """
    if kind not in ("evo2_gelu", "evo2_id"):
        raise ValueError("evo2 vector kind is not gelu or identity")
    ops: list[Op] = []
    if kind == "evo2_gelu":
        ops.append(Op("gelu", "vector", li, vec=nact * 5, sfu=float(nact), a_fmt=af))
    ops.append(Op("mlp_mul", "vector", li, vec=float(nact), a_fmt=af))
    return ops


def _evo2_dense(L: Layer) -> bool:
    return L.vec_kind in ("evo2_gelu", "evo2_id")


def _whisper_dec(L: Layer) -> bool:
    """WhisperDecoderLayer: published erf-GELU between fc1 and fc2, fp32 LayerNorm."""
    return L.vec_kind == "whisper"


def _emit_dense_silu(model: ModelSpec, L: Layer, ops: list, li: int, t: int, tp: int,
                     message: str) -> None:
    """down(silu(gate) * up) between fused gate_up and down.

    Width is the local intermediate, not the fused gate_up rows. SiLU stays
    ``act``. The product is ``mlp_mul``. ``a_fmt`` follows ``act_fmt``.
    The shape check and the reject text come from ``vector_rules``.
    """
    f = L.ffn
    if not f.gated or [ln.name for ln in L.ffn_linears] != ["gate_up", "down"]:
        raise ValueError(message)
    nact = t * _cdiv(f.d_ff, tp)
    af = model.act_fmt
    for lin in L.ffn_linears:
        k, n = _lin_local(lin, tp)
        ops.append(gemm(model, f"mlp.{lin.name}", li, t, k, n, lin.role or "mlp"))
        if lin.name == "gate_up":
            ops.append(Op("act", "vector", li, vec=nact * 4, sfu=nact * 2, a_fmt=af))
            ops.append(Op("mlp_mul", "vector", li, vec=float(nact), a_fmt=af))


def _carbon_qk_rms(L: Layer, li: int, t: int, tp: int) -> list[Op]:
    """RMSNorm on head_dim after q_proj and k_proj.

    Qwen3RMSNorm, Qwen3VLTextRMSNorm, and Exaone4RMSNorm all say the norm
    is on the head dim, unlike Olmo. One rsqrt per head. The vector width
    is the projected q or k width (local heads * head_dim), not hidden.
    On the 0.467 Qwen3 rows the projected q width is not hidden. Llama
    Carbon and Granite checkpoints have no q/k norm. The reduction is the
    published T5 recipe (fp32).
    """
    who = vr.qk_rms_who(L.vec_kind)
    if not who:
        return []
    hd = L.core.qk_dim
    if hd <= 0:
        raise ValueError(f"{who} head_dim is not positive")
    by = {l.name: l for l in L.attn_linears}
    if "q" not in by or "k" not in by:
        raise ValueError(f"{who} attention is not q then k")
    ops: list[Op] = []
    for name in ("q", "k"):
        lin = by[name]
        _k, n = _lin_local(lin, tp, L.core)
        if n % hd != 0:
            raise ValueError(f"{who} qk norm width is not a whole number of heads")
        heads = n // hd
        ops.append(Op(f"{name}_rms", "vector", li, vec=float(t * n * 4),
                      sfu=float(t * heads), a_fmt="fp32", replicated=_repl(lin, tp)))
    return ops


def _rank_ops(model: ModelSpec, li: int, ph: Phase, sh: Shard = Shard(), layer: Layer | None = None) -> list[Op]:
    L = layer if layer is not None else model.layers[li]
    if ph.kind == "full":
        return _full_layer_ops(model, li, L, ph, sh)
    h = model.hidden
    ab = _fmt(model.act_fmt).bytes
    rb = red_bytes(model)
    tp, dp = sh.tp, sh.dp
    b_loc = _cdiv(ph.batch, dp)
    t = b_loc * ph.q                      # tokens through this rank's attention replica
    ops: list[Op] = []
    if L.ple_rows:
        # Qwen4Exp PLE (transformers Qwen4ExpTextPLELayer / NGramEmbedding): n-gram hash ids (shift gathers; per
        # n-gram size n multiplies + n − 1 xor, heads × (mod + offset add)); table rows (bf16) — memory-bound lookup;
        # key / value projections are Linears; norm_key / norm_query / norm_conv grouped RMSNorms (4/elem), gate dot
        # (2/elem), signed sqrt + sigmoid (6 / stream), gated value (1), dilated depthwise conv (2·k) + SiLU (4), adds
        Nh = L.gres_n * h
        ng = L.ple_dil
        hpn = L.ple_rows // max(ng - 1, 1)
        hv = ng + sum(n + (n - 1) + 2 * hpn for n in range(2, ng + 1))
        st = (L.ple_conv - 1) * L.ple_dil * Nh * 2               # conv state (bf16), read + written each step
        # 0.280: row bytes at the release's table dtype (embed role: bf16 Qwen; ≈ nvfp4 e2m1 + e4m3 / 16 FP4-Local)
        ops.append(Op("ple_lookup", "lookup", li, kv_read=t * L.ple_rows * L.ple_row_dim * model.fmt("embed").bits / 8,
                      vec=t * hv, replicated=tp))
        ops.append(Op("ple_mix", "state", li, replicated=tp,
                      vec=t * (Nh * (4 + 4 + 2 + 1 + 4 + 2 * L.ple_conv + 4 + 1 + 1) + 6 * L.gres_n),
                      state_rw=(2 * st * b_loc) if ph.kind == "decode" else st * b_loc))
    if L.engram_cols:
        # V4.1 Engram (engram.py NgramHashState + model.py Engram): hash ids per token (4 multiplies, 3 xor, 24 mod,
        # 24 offset adds); fetch engram_cols rows of fp8 + e8m0 scale per 32 from the row-sharded table (memory-bound
        # lookup), dequant (2/elem), all-reduce of the bf16 rows over TP; wkv GEMM (a Linear); gate: rstd of stream
        # and key per copy (2+2/elem), dot (3/elem), signed sqrt + sigmoid (6/copy), h + gate·value (2/elem)
        ec, ed, N = L.engram_cols, L.engram_dim, max(L.hc, 1)
        mx = model.engram_ngram
        ops.append(Op("engram_lookup", "lookup", li, kv_read=t * ec * (ed + ed / 32) / max(tp, 1),
                      vec=t * (mx + (mx - 1) + 2 * ec) + t * ec * ed * 2))
        if tp > 1:
            ops.append(Op("engram_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp,
                          comm_bytes=t * ec * ed * 2))
        ops.append(Op("engram_gate", "vector", li, replicated=tp, vec=t * (N * h * 9 + 6 * N)))
    if L.hc:
        ops += _hc_ops(model, li, L, t, tp, "attn")
    if L.gres_n:
        ops.append(_gres_op(model, li, L, t, tp, "attn"))
    else:
        if vr.pre_norm_fp32(L.vec_kind):
            # WhisperDecoderLayer.self_attn_layer_norm is nn.LayerNorm. The
            # reduction is fp32. One rsqrt per token row. Affine is already
            # in misc. The op name stays attn_norm. ernie300_swiglu keeps
            # the 0.458 Ernie4_5_MoeRMSNorm on MoE layers. The kind test is
            # vector_rules.pre_norm_fp32.
            ops.append(Op("attn_norm", "vector", li, vec=t * h * 4, sfu=t, a_fmt="fp32"))
        elif not vr.post_norm_fp32(L.vec_kind):
            # EXAONE-4 is post-norm. The same attn_norm op is emitted after
            # the attention all-reduce, not before qkv.
            ops.append(Op("attn_norm", "vector", li, vec=t * h * 4, sfu=t))
    if L.core.dea_q:
        ops.extend(_dea_ops(model, li, L, ph, sh, b_loc))
    fq = _qkv_fusable(L, tp) if FUSE_QKV else None
    for l in L.attn_linears:
        if l.name.startswith("dea_"):
            continue                          # 0.361: emitted by _dea_ops (kkup / vvup run over the whole cache)
        if fq and l.name in ("k", "v"):
            continue                          # 0.198: folded into attn.qkv
        if fq and l.name == "q":
            k_, ns, rp = fq
            ops.append(gemm(model, "attn.qkv", li, t, k_, sum(ns), "attn", params=k_ * sum(ns), replicated=rp))
            continue
        if l.name == "v_experts":
            ops += _mova_v_ops(model, li, L.core, l, t, tp, layer_vp(ph, li))
            continue
        k, n = _lin_local(l, tp, L.core)
        m = t
        if l.name == "kv_b" and ph.kind == "prefill":
            m = b_loc * (ph.ctx + ph.q)      # decompress cached prefix + new tokens
        elif l.name == "kv_b" and ph.kind == "decode" and L.core.kind == "mla":
            # 0.63 (external review): absorbed decode applies kv_b per head as two batched GEMMs — query absorb
            # q_nope·W_UK[h] → latent and output o_lat·W_UV[h] → v_dim (DeepSeek-V3 inference/model.py, absorb path:
            # einsum "bshd,hdc->bshc" / "bshc,hdc->bshd"); same params / FLOPs as the wide [kv_lora, H·(nope+v)] GEMM
            c = L.core
            nope = c.qk_dim - c.rope_dim
            hq = _cdiv(c.n_q, tp)
            ops.append(gemm(model, "attn.kv_b.uk", li, t, nope, c.kv_lora, "attn", count=hq,
                            params=hq * nope * c.kv_lora, replicated=_repl(l, tp)))
            ops.append(gemm(model, "attn.kv_b.uv", li, t, c.kv_lora, c.v_dim, "attn", count=hq,
                            params=hq * c.kv_lora * c.v_dim, replicated=_repl(l, tp)))
            continue
        gc = groups_local(l, tp)[0] if l.groups > 1 else 1
        ops.append(gemm(model, f"attn.{l.name}", li, m, k, n, l.role or "attn", count=gc, params=k * n * gc,
                        replicated=_repl(l, tp)))
    # 0.445 Carbon-X: q_norm / k_norm sit on the projected heads, before the score.
    # Fused qkv still keeps the unfused q and k linears, so the widths come from those.
    ops.extend(_carbon_qk_rms(L, li, t, tp))
    ops.extend(_attn_core_ops(model, li, L.core, ph, sh, b_loc, L.kv_alias))
    if L.core.out_gate_fn == "softplus":
        # 0.292 k2_horizon: attn_output · softplus(gate_proj x, β = ln 2) per element before o_proj 「来源」
        # (K2HorizonAttention / K2HorizonMoVAAttention): softplus = log1p(exp(βx))/β — exp + log on the SFU, 4 ops
        # around them + the multiply
        nq = gqa_local(L.core.n_q, L.core.n_kv, tp)[1] * L.core.v_dim
        ops.append(Op("attn_out_gate", "vector", li, vec=t * nq * 5, sfu=t * nq * 2))
    if L.core.kind in ("gqa", "mla"):
        hq = gqa_local(L.core.n_q, L.core.n_kv, tp)[1] if L.core.kind == "gqa" else _cdiv(L.core.n_q, tp)
        if L.core.rope_n and L.core.kind == "gqa":
            # Qwen4Exp: q / k per-head RMSNorm (4/elem) + partial RoPE (3/elem) on q heads and k heads; σ output gate
            c_ = L.core
            kv_ = gqa_local(c_.n_q, c_.n_kv, tp)[0]
            ops.append(Op("rope", "vector", li, vec=t * (hq + kv_) * (c_.qk_dim * 4 + c_.rope_n * 3)
                          + t * hq * c_.v_dim * 5))
        elif L.core.rope_n:
            # DeepSeek-V4: RoPE on q heads + kv, inverse RoPE on o heads; per-head q RMSNorm and kv RMSNorm (4/elem)
            c_ = L.core
            ops.append(Op("rope", "vector", li, vec=t * (2 * hq + 1) * c_.rope_n * 3 + t * (hq + 1) * c_.qk_dim * 4))
        elif not L.core.no_rope:
            # 0.371: EXAONE-4 hybrid full layers are global NoPE (Exaone4Attention). rope_dim 0
            # would still bill half a head, so no_rope drops the vector. KV width is unchanged.
            ops.append(Op("rope", "vector", li, vec=t * hq * max(L.core.rope_dim, L.core.qk_dim // 2) * 3))
    if tp > 1:
        ops.append(Op("attn_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * rb))
    if vr.post_norm_fp32(L.vec_kind):
        # Exaone4DecoderLayer.post_attention_layernorm is T5 RMSNorm on the
        # full hidden, after the row-parallel o_proj all-reduce. One rsqrt
        # per token row. The residual add stays the combined residual below.
        ops.append(Op("attn_norm", "vector", li, vec=t * h * 4, sfu=t, a_fmt="fp32"))
    if L.cross is not None and model.enc_ctx:
        ops += _cross_decode_ops(model, li, L, ph, sh, b_loc)
    if L.hc:
        ops += _hc_ops(model, li, L, t, tp, "ffn")
        if L.hc_head == "mix":
            # DeepSeek-V4 ParallelHead.hc_head: rsqrt-norm, fn GEMM [N·h → N], sigmoid pre, weighted collapse
            N = L.hc
            ops.append(Op("hc_head_fn", "gemm", li, m=t, k=N * h, n=N, w_fmt="bf16", w_bits=16, a_fmt="bf16",
                          act_bytes=t * (N * h + N) * 2, replicated=tp))
            ops.append(Op("hc_head", "vector", li, vec=t * (4 * N * h + 4 * N + 2 * N * h), replicated=tp))
        elif L.hc_head == "pre":      # V4.1: final hc_pre with the last pre_mix (weighted collapse, no projection)
            ops.append(Op("hc_head", "vector", li, vec=t * 2 * L.hc * h, replicated=tp))
        elif li == len(model.layers) - 1 and layer is None and model.draft == "":   # Glm5NextTextHyperHead: unweighted mean of N streams
            ops.append(Op("hc_head", "vector", li, vec=t * L.hc * h, replicated=tp))
    if L.gres_n:
        ops.append(_gres_op(model, li, L, t, tp, "mlp"))
        if L.gres_head:
            # final hyper_connection_mixer (no combine): grouped RMSNorm, SiLU, sigmoid, weighted stream mean
            Nh = L.gres_n * h
            ops.append(Op("gres_head", "vector", li, replicated=tp, vec=t * (4 * Nh + 4 * Nh + 2 * Nh)))
    else:
        if vr.pre_norm_fp32(L.vec_kind):
            # WhisperDecoderLayer.final_layer_norm is the pre-MLP LayerNorm,
            # not a second copy after both linears. fp32, one rsqrt per row.
            # ernie300_swiglu keeps the 0.458 post-attention RMSNorm.
            ops.append(Op("ffn_norm", "vector", li, vec=t * h * 4, sfu=t, a_fmt="fp32"))
        elif not vr.post_norm_fp32(L.vec_kind):
            # EXAONE-4 post_feedforward_layernorm is emitted after mlp.down
            # and the MLP all-reduce.
            ops.append(Op("ffn_norm", "vector", li, vec=t * h * 4, sfu=t))
    # One compare. SituAndMul below is unchanged: it is not silu(gate)*up.
    situ = L.vec_kind == "kimi_situ"
    f = L.ffn
    if f.kind == "dense":
        if _evo2_dense(L):
            # l3(act(l1(z)) * l2(z)). The fused GEMMs stay gate_up then down.
            # The activation and the product sit between them, on one branch.
            if not f.gated or [l.name for l in L.ffn_linears] != ["gate_up", "down"]:
                raise ValueError("evo2 mlp is not gate_up then down")
            nact = t * _cdiv(f.d_ff, tp)
            af = model.act_fmt
            for l in L.ffn_linears:
                k, n = _lin_local(l, tp)
                ops.append(gemm(model, f"mlp.{l.name}", li, t, k, n, l.role or "mlp"))
                if l.name == "gate_up":
                    ops.extend(_evo2_mlp_vecs(L.vec_kind, nact, li, af))
        elif (dense_msg := vr.dense_message(L.vec_kind, L.product_rule)) is not None:
            # 0.480: carbon, exaone, and the dense *_swiglu kinds share one emitter.
            # Kind-specific reject text still comes from the rule table.
            _emit_dense_silu(model, L, ops, li, t, tp, dense_msg)
        elif situ:
            # KimiMLP: down(SituAndMul(cat(gate, up))). The class casts the
            # two halves to fp32. There is no separate silu and no second multiply.
            if not f.gated or [l.name for l in L.ffn_linears] != ["gate_up", "down"]:
                raise ValueError("kimi situ mlp is not gate_up then down")
            nact = t * _cdiv(f.d_ff, tp)
            for l in L.ffn_linears:
                k, n = _lin_local(l, tp)
                ops.append(gemm(model, f"mlp.{l.name}", li, t, k, n, l.role or "mlp"))
                if l.name == "gate_up":
                    ops.append(_situ_op(li, nact))
        elif _whisper_dec(L):
            # WhisperDecoderLayer: fc2(ACT2FN["gelu"](fc1(x))). gelu is
            # x * 0.5 * (1 + erf(x / sqrt(2))): five element-ops, one erf.
            # The class does not upcast, so a_fmt follows the activation
            # dtype. Not gated, so there is no mlp_mul. The product sits
            # between up and down because fc2 reads gelu(fc1).
            if f.gated or [l.name for l in L.ffn_linears] != ["up", "down"]:
                raise ValueError("whisper decoder mlp is not up then down")
            nact = t * _cdiv(f.d_ff, tp)
            af = model.act_fmt
            for l in L.ffn_linears:
                k, n = _lin_local(l, tp)
                ops.append(gemm(model, f"mlp.{l.name}", li, t, k, n, l.role or "mlp"))
                if l.name == "up":
                    ops.append(Op("gelu", "vector", li, vec=nact * 5, sfu=float(nact), a_fmt=af))
        elif (bad := vr.dense_forbidden(L.vec_kind, L.product_rule)):
            raise ValueError(bad)
        else:
            for l in L.ffn_linears:
                k, n = _lin_local(l, tp)
                ops.append(gemm(model, f"mlp.{l.name}", li, t, k, n, l.role or "mlp"))
            ops.append(Op("act", "vector", li, vec=t * _cdiv(f.d_ff, tp) * 4, sfu=t * _cdiv(f.d_ff, tp) * 2))   # silu: exp + rcp
        if tp > 1:
            ops.append(Op("mlp_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * rb))
        if vr.post_norm_fp32(L.vec_kind):
            # post_feedforward_layernorm sees the full hidden. A row-parallel
            # down must all-reduce first.
            ops.append(Op("ffn_norm", "vector", li, vec=t * h * 4, sfu=t, a_fmt="fp32"))
    elif f.kind == "moe":
        sh.check_moe()
        T = ph.tokens                       # global tokens in the stage step
        # 0.384: pool B tokens are ph.mm per sequence. Absent on every earlier MoE (n_experts_b = 0).
        mm = min(T, ph.batch * ph.mm) if f.n_experts_b else 0
        text_T = (T - mm) if f.n_experts_b else T
        for l in L.ffn_linears:
            if mm == 0 and l.name.startswith("router_b"):
                continue
            k, n = _lin_local(l, tp)
            role = l.role or ("router" if l.name.startswith(("router", "shared_expert_gate")) else (
                "shared_expert" if l.name.startswith("shared") else "mlp"))
            ops.append(gemm(model, f"moe.{l.name}", li, t, k, n, role, replicated=_repl(l, tp)))
            # 0.480: shared-expert silu(gate)*up. Shared SiLU stays unbilled.
            shared_emit, shared_err = vr.shared_site(L.vec_kind, L.product_rule, l.name, f)
            if shared_err:
                raise ValueError(shared_err)
            if shared_emit:
                n_loc = _cdiv(f.n_shared * f.d_shared, tp)
                ops.append(Op("mlp_mul", "vector", li, vec=float(t * n_loc), a_fmt=model.act_fmt))
            if situ and l.name == "shared_gate_up":
                # KimiSparseMoeBlock shared expert is one KimiMLP whose
                # intermediate is moe_intermediate_size * num_shared_experts.
                # SituAndMul runs on every token, before down. It is not
                # the routed top-k width.
                if not f.gated or not f.n_shared or not f.d_shared:
                    raise ValueError("kimi situ shared expert is not situ")
                n_loc = _cdiv(f.n_shared * f.d_shared, tp)
                ops.append(_situ_op(li, t * n_loc))
        if f.d_ff:
            # Parallel dense SwiGLU (LongCat-Flash sublayer 0): every token, not a shared expert.
            ops.append(Op("act", "vector", li, vec=t * _cdiv(f.d_ff, tp) * 4, sfu=t * _cdiv(f.d_ff, tp) * 2))
            if tp > 1:
                ops.append(Op("mlp_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp,
                              comm_bytes=t * h * rb))

        def _pool(name: str, n_exp: int, d_exp: int, tokens: int, zero: int, role: str) -> None:
            # 0.480: routed silu(gate)*up guards live in vector_rules.
            expert_err = vr.expert_failure(L.vec_kind, L.product_rule, f)
            if expert_err:
                raise ValueError(expert_err)
            if situ and not f.gated:
                raise ValueError("kimi situ expert is not situ")
            if tokens <= 0 or n_exp <= 0:
                return
            if zero and not f.groups:
                r = identity_zero_routing(n_exp, zero, f.top_k, tokens, sh.ep, layer_skew(ph, li))
            else:
                r = moe_routing(n_exp, f.top_k, tokens, sh.ep, layer_skew(ph, li), f.groups)
            d_loc = _cdiv(d_exp, sh.etp)
            ein = f.expert_in(h)
            if r.hit:
                mats_up = 2 if f.gated else 1
                # padded rows (hit · m_e ≥ token-expert pairs) cost time but are not work: share = useful fraction (0.63)
                u = min(1.0, r.pairs_local / (r.hit * r.m_e))
                ops.append(replace(gemm(model, f"{name}.gate_up", li, r.m_e, ein, mats_up * d_loc, role,
                                        count=r.hit), share=u))
                act = Op(f"{name}_act" if name == "expert" else f"{name}.act", "vector", li,
                         vec=r.pairs_local * d_loc * 4, sfu=r.pairs_local * d_loc * 2)
                if vr.bills_expert_product(L.vec_kind, L.product_rule):
                    # silu(gate)*up on the useful pairs, before down.
                    # SiLU stays the historical act op. Latent experts use
                    # the published expert activation. Padded GEMM rows,
                    # routing-weight scaling, and shared SiLU are not this op.
                    fmt_key = vr.expert_fmt_key(L.vec_kind, L.product_rule)
                    a_fmt = model.act_fmt_for(role) if fmt_key == "expert_role" else model.act_fmt
                    ops.append(act)
                    ops.append(Op("mlp_mul", "vector", li, vec=float(r.pairs_local * d_loc),
                                  a_fmt=a_fmt))
                if situ:
                    # KimiBlockSparseMLP: SituAndMul(cat(w1, w3)) then w2.
                    # The historical SiLU expert_act is not this formula.
                    # Width is the useful token-expert pairs times local
                    # moe_intermediate_size, not the latent width and not
                    # the padded GEMM rows. The routing-weight multiply and
                    # the latent RMSNorm are not this op.
                    ops.append(_situ_op(li, r.pairs_local * d_loc))
                ops.append(replace(gemm(model, f"{name}.down", li, r.m_e, d_loc, ein, role, count=r.hit),
                                   share=u))
                if (not vr.bills_expert_product(L.vec_kind, L.product_rule)
                        and not situ):
                    ops.append(act)
            if sh.ep > 1:
                pay = r.pairs_local * ein * ab
                fn = node_dedup_factor(n_exp, f.top_k, ph.ep_nodes, f.groups, n_exp + zero)
                tag = "" if name == "expert" else "_b"
                ops.append(Op(f"moe_dispatch{tag}", "comm", li, comm_kind="alltoall", comm_group=sh.ep,
                              comm_bytes=pay, comm_stride=sh.etp, comm_net=fn))
                ops.append(Op(f"moe_combine{tag}", "comm", li, comm_kind="alltoall", comm_group=sh.ep,
                              comm_bytes=pay * rb / ab, comm_stride=sh.etp, comm_net=fn))
            if sh.etp > 1:
                ops.append(Op("expert_allreduce" if name == "expert" else "expert_b_allreduce", "comm", li,
                              comm_kind="allreduce", comm_group=sh.etp, comm_bytes=r.pairs_local * ein * rb))

        _pool("expert", f.n_experts, f.d_expert, text_T, f.zero_experts, f.expert_role or "expert")
        if mm and f.n_experts_b:
            _pool("expert_b", f.n_experts_b, f.d_expert_b, mm, 0, "expert")
        if tp > 1 and f.n_shared:
            ops.append(Op("shared_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * rb))
    if L.ple_in:
        # 0.369 Gemma-4: GELU(gate) * this layer's PLE slice, then RMSNorm(h). The GEMMs are ple_gate / ple_up.
        ops.append(Op("ple_mix", "vector", li, vec=t * (L.ple_in * 4 + h * 4), sfu=t * L.ple_in))
    if L.deepemb:
        # 0.361 RWKV-8 channel mix: value(relu(key)^2) * enn[idx] for the new tokens only (demo CMix_*).
        bits = model.lookup_bits or model.fmt("embed").bits or 16.0
        ops.append(Op("deepemb", "lookup", li, kv_read=t * L.deepemb * bits / 8.0, vec=t * L.deepemb))
    ops.append(Op("residual", "vector", li, vec=t * h * 2))
    return _llm_stream(ops, ph)


def layer_skew(ph: "Phase", li: int):
    """0.293: the routing skew of layer li — its own (per-layer measured load) or the phase-wide one (MTP layers, li
    beyond the main stack, use the phase-wide skew)."""
    s = ph.skew_l[li] if 0 <= li < len(ph.skew_l) else None
    return ph.skew if s is None else s


def layer_vp(ph: "Phase", li: int) -> tuple:
    """0.293: value-expert probabilities of layer li (per-layer measured load) or the phase-wide ones."""
    v = ph.v_pl[li] if 0 <= li < len(ph.v_pl) else None
    return ph.v_p if v is None else v


def mova_hits(E: int, k: int, t: int, v_p: tuple = ()) -> Routing:
    """0.293: value-expert routing of t tokens (attention TP: every rank routes all its tokens, ep = 1).  Uniform
    (``v_p`` empty): ``moe_routing`` (「假设」).  Measured load: distinct experts = the occupancy Σ_e 1 − (1 − p_e)^t
    (p_e = k·load_e/Σload, the 0.132 rule of ``moe_routing``'s SkewSpec path), at least k (a token picks k distinct
    experts) and at most E; pairs stay t·k."""
    r = moe_routing(E, k, t, 1)
    if not v_p or t <= 0:
        return r
    occ = sum(1.0 - (1.0 - min(1.0, p)) ** t for p in v_p)
    hit = min(E, max(min(E, k), math.ceil(occ - 1e-9)))
    return Routing(t, r.pairs_local, hit, max(1, math.ceil(r.pairs_local / hit)), E, r.pairs_mean)


def mova_probs(load: tuple, k: int) -> tuple:
    """0.293: inclusion probabilities k·load_e/Σload of a measured value-expert load (hot_rank_probs at ep = 1)."""
    return hot_rank_probs(tuple(load), 1, k)[0]


def _mova_v_ops(model: ModelSpec, li: int, c: AttnCore, l: Linear, t: int, tp: int, v_p: tuple = ()) -> list[Op]:
    """0.292 MoVA routed value projection (modeling_k2_horizon combine_routed_experts 「来源」): the v_router GEMM is
    an ordinary Linear; here each touched value expert is one GEMM over its routed rows (uniform-routing expectation,
    ``moe_routing`` over this rank's t tokens — attention TP keeps every token on every rank, each rank holds every
    expert's slice of its KV heads), then silu, × routing weight, scatter-add (6 ops / element, exp + rcp on the SFU),
    and the sigmoid + top-k selection (6 ops / expert, as the DiT MoE router).  Only touched experts' weights stream."""
    E, k = c.v_experts, c.v_topk
    n_loc = gqa_local(c.n_q, c.n_kv, tp)[0] * c.v_dim if tp > 1 else c.n_kv * c.v_dim
    r = mova_hits(E, k, t, v_p)       # 0.293: measured load → occupancy (serving.mova_expert_load)
    ops = []
    if r.hit:
        u = min(1.0, r.pairs_local / (r.hit * r.m_e))
        ops.append(replace(gemm(model, "attn.v_experts", li, r.m_e, l.k, n_loc, l.role or "attn", count=r.hit,
                                replicated=_repl(l, tp)), share=u))
    ops.append(Op("v_expert_combine", "vector", li, vec=r.pairs_local * n_loc * 6 + t * E * 6,
                  sfu=r.pairs_local * n_loc * 2 + t * E))
    return ops


def _cross_decode_ops(model: ModelSpec, li: int, L: Layer, ph: Phase, sh: Shard, b_loc: int) -> list[Op]:
    """0.270 encoder-decoder cross-attention in the autoregressive phases (transformers WhisperAttention with
    ``key_value_states`` + EncoderDecoderCache 「来源」): every step projects q from the new tokens and attends to all
    ``enc_ctx`` encoder positions (no causal mask); the cross K/V are projected once per request — at the first
    prefill pass (ctx = 0) from the encoder output (rows = enc_ctx per sequence) and written to the cache — and only
    read afterwards (decode, chunked-prefill continuations).  TP: heads column-split like self-attention, o row-split
    + all-reduce."""
    c, E, h, tp = L.cross, model.enc_ctx, model.hidden, sh.tp
    ab, kvb, rb = _fmt(model.act_fmt).bytes, _fmt(model.kv_fmt).bytes, red_bytes(model)
    t, q = b_loc * ph.q, ph.q
    first = ph.kind == "prefill" and ph.ctx == 0
    # WhisperDecoderLayer.encoder_attn_layer_norm is nn.LayerNorm on the
    # decoder hidden, before cross q. fp32 reduction, one rsqrt per row.
    if _whisper_dec(L):
        ops = [Op("cross_norm", "vector", li, vec=t * h * 4, sfu=t, a_fmt="fp32")]
    else:
        ops = [Op("cross_norm", "vector", li, vec=t * h * 4, sfu=t)]
    for l in L.cross_linears:
        if l.name in ("k", "v") and not first:
            continue
        k, n = _lin_local(l, tp, c)
        m = b_loc * E if l.name in ("k", "v") else t
        if l.name == "o":
            continue
        ops.append(gemm(model, f"cross.{l.name}", li, m, k, n, "attn", params=k * n, replicated=_repl(l, tp)))
    kv_loc, hq_loc = gqa_local(c.n_q, c.n_kv, tp)
    g = max(1, _cdiv(hq_loc, kv_loc))
    cnt = b_loc * kv_loc
    ops.append(Op("cross_qk", "attn", li, m=q * g, k=c.qk_dim, n=E, count=cnt, causal=1.0,
                  act_bytes=cnt * q * g * c.qk_dim * ab))
    ops.append(Op("cross_pv", "attn", li, m=q * g, k=E, n=c.v_dim, count=cnt, causal=1.0,
                  act_bytes=cnt * q * g * c.v_dim * ab))
    ent = b_loc * E * kv_loc * (c.qk_dim + c.v_dim) * kvb
    sm = cnt * q * g * E
    ops.append(Op("cross_softmax", "vector", li, vec=sm * 5, sfu=sm, kv_read=0.0 if first else ent,
                  kv_write=ent if first else 0.0))
    o = next(l for l in L.cross_linears if l.name == "o")
    k, n = _lin_local(o, tp, c)
    ops.append(gemm(model, "cross.o", li, t, k, n, "attn", params=k * n, replicated=_repl(o, tp)))
    if tp > 1:
        ops.append(Op("cross_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * rb))
    return ops


def _llm_stream(ops: list[Op], ph: Phase) -> list[Op]:
    """0.63 (external review): LLM activations are streamed like the full-sequence forwards — every weight GEMM, and
    the prefill attention (flash: Q / O blocks in SRAM, K / V re-read per block when they do not fit), may spill its
    activations to DRAM (memplan.act_stream; nothing moves while they fit the SRAM's activation half).  Decode
    attention reads its K / V from the cache once (kv_read) and keeps its tiny Q / O on chip: not streamed.
    Prefill: the cached prefix's K / V first pass is already in kv_read (kv_pre = ctx / (ctx + q))."""
    out = []
    pre = ph.ctx / (ph.ctx + ph.q) if ph.kind == "prefill" and ph.ctx + ph.q > 0 else 0.0
    for o in ops:
        if o.kind == "gemm" and not o.stream:
            o = replace(o, stream=True)
        elif o.kind == "attn" and ph.kind == "prefill" and o.name in ("qk", "qk_latent") and not o.stream:
            o = replace(o, stream=True, kv_pre=pre)
        elif o.kind == "attn" and o.name == "dea_qk" and not o.stream:
            # Reconstructed K/V are activations of the up-projections, not a prefix already in kv_read.
            # Decode and prefill both re-read them when they miss SRAM (v_dim != k; memplan.act_stream).
            o = replace(o, stream=True, kv_pre=0.0)
        out.append(o)
    return out


def embed_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard()) -> list[Op]:
    return rank_shares(_embed_ops(model, ph, sh), sh, lambda: _embed_ops(model, ph, Shard()))


def _longcat_ngram_hash_ops(order: int, splits: int) -> int:
    """Scalar ops per token of HF ``NgramEmbedding._get_ngram_ids`` plus one shift per order.

    For each order ``i`` in ``2..n``: one right shift, and ``k`` hashes. Each hash does
    ``i - 1`` multiply-adds and one modulo. The vocab-mod table is computed once at load.
    """
    return sum(1 + splits * (2 * (i - 1) + 1) for i in range(2, order + 1))


def _embed_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard()) -> list[Op]:
    b_loc = _cdiv(ph.batch, sh.dp)
    t = b_loc * ph.q
    rf = model.fmt("embed")
    ops = [Op("embed_lookup", "lookup", -1, kv_read=t * model.hidden * rf.bits / 8, vec=t * model.hidden)]
    if model.ngram_dim:
        # 0.400 LongCat-Flash-Lite. Tables are vocab-parallel; the bmm matches vLLM oe_projection
        # (one replicated GEMM per embedder, same FLOPs as k*(n-1) separate linears).
        num = model.ngram_splits * (model.ngram_order - 1)
        dim = model.ngram_dim
        bits = model.lookup_bits or rf.bits
        ops.append(Op("ngram_lookup", "lookup", -1, kv_read=t * num * dim * bits / 8,
                      vec=t * _longcat_ngram_hash_ops(model.ngram_order, model.ngram_splits)))
        ops.append(gemm(model, "ngram.proj", -1, t, dim, model.hidden, "ngram", count=num,
                        replicated=sh.tp, stream=True))
        ops.append(Op("ngram_mean", "vector", -1, vec=t * model.hidden * (num + 1)))
    if model.ple_packed:
        # 0.369 Gemma-4: one packed PLE row per token, then RMSNorm on each layer slice (width ple).
        bits = model.lookup_bits or rf.bits
        w = model.ple_packed
        ops.append(Op("ple_lookup", "lookup", -1, kv_read=t * w * bits / 8, vec=t * w))
        ops.append(Op("ple_norm", "vector", -1, vec=t * w * 4, sfu=t * model.n_layers))
    books = model.audio_codebooks
    if books and ph.kind == "decode":
        # 0.402 LongCat-Flash-Omni. The next decode token sums one row from each input codebook.
        # Prefill does not read these tables (input_process_forward_decode).
        n_in = books[0]
        bits = model.lookup_bits or rf.bits
        ops.append(Op("audio_code_lookup", "lookup", -1, kv_read=t * n_in * model.hidden * bits / 8,
                      vec=t * n_in * model.hidden))
    return ops


def head_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard(), glue: bool = False) -> list[Op]:
    ops = rank_shares(_head_ops(model, ph, sh, glue), sh, lambda: _head_ops(model, ph, Shard(), glue))
    if ph.kind == "prefill" and ph.cp > 1:
        # 0.89: only the CP rank holding a sequence's last chunk (zigzag rank 0) projects its last token — the chunk
        # phase charges that rank's time on every rank; the mean rank does 1/cp of it (work KPI exact)
        ops = [replace(o, share=o.share / ph.cp) if o.kind != "comm" else o for o in ops]
    return ops


def _head_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard(), glue: bool = False) -> list[Op]:
    """LM head, vocab-parallel over attention TP.  Prefill: only the last token
    of each sequence is projected; decode: every new/verified token."""
    b_loc = _cdiv(ph.batch, sh.dp)
    m = b_loc if ph.kind == "prefill" else b_loc * ph.q
    n = _cdiv(model.vocab, sh.tp)
    role = "embed" if model.tie_embeddings else "lm_head"
    ts = getattr(model, "tts", None)
    if ts is not None:
        # 0.269 TTS: the last hidden state drives stop head + flow DiT + aggregator + streaming AudioVAE decode
        # (core/tts.py) instead of the vocab head; one latent step per sequence (prefill: the first step)
        from .tts import tts_head_ops
        if getattr(ts, "kind", "") == "firered":      # 0.275 FireRedTTS3 (core/fireredtts3.py)
            from .fireredtts3 import fire_head_ops as tts_head_ops
        # 0.453 / 0.455: FireRedTTS3's backbone norm is Qwen3RMSNorm. Ming's
        # backbone norm is Qwen2RMSNorm. The TTS head still reads one row per
        # sequence, not every prompt token. One rsqrt, fp32. The vec is not
        # scaled by the prompt length.
        if vr.tts_fp32_backbone(model.vec_kind) and model.final_norm_params != 0:
            final = Op("final_norm", "vector", model.n_layers,
                       vec=b_loc * model.hidden * 4, sfu=float(b_loc), a_fmt="fp32")
        else:
            final = Op("final_norm", "vector", model.n_layers,
                       vec=0 if model.final_norm_params == 0 else b_loc * model.hidden * 4)
        return [final] + tts_head_ops(ts, b_loc, model.n_layers, replicated=sh.tp, af=model.act_fmt)
    if model.readout_rows and not (ph.labels and ph.kind == "prefill"):
        # 0.291 a decision checkpoint (pplx-decider) has no vocab head: outside a decision readout every row it
        # projects reads the whole readout (all answer-code rows)
        L = model.readout_rows
        return [Op("final_norm", "vector", model.n_layers,
                   vec=0 if model.final_norm_params == 0 else b_loc * ph.q * model.hidden * 4),
                gemm(model, "label_head", model.n_layers, m, model.hidden, L, "lm_head", stream=True),
                Op("label_softmax", "vector", model.n_layers, vec=m * L * 3, sfu=m * L)]
    if ph.labels and ph.kind == "prefill":
        # 0.291: joint requests read out at every question marker (m = sequences × questions); a readout checkpoint
        # reads its readout rows (role lm_head), not the vocab head
        m = m * ph.questions
        if model.readout_rows:
            role = "lm_head"
        # 0.191 decision readout (StartLux-Decision inference model.py): hidden at the last prompt position @ the
        # option-letter rows of the output head (index_select, replicated) → softmax over ≤ labels; nothing sampled,
        # no vocab-wide logits, no logits gather
        L = ph.labels
        return [Op("final_norm", "vector", model.n_layers,
                   vec=0 if model.final_norm_params == 0 else b_loc * ph.q * model.hidden * 4),
                gemm(model, "label_head", model.n_layers, m, model.hidden, L, role, stream=True),
                Op("label_softmax", "vector", model.n_layers, vec=m * L * 3, sfu=m * L)]
    if vr.fp32_final_norm(model.vec_kind):
        # LlamaRMSNorm / Qwen3RMSNorm / Qwen3VLTextRMSNorm / Exaone4RMSNorm /
        # GraniteRMSNorm / Qwen2RMSNorm / MistralRMSNorm / Moshi rms_norm_f32 /
        # Ernie4_5_MoeRMSNorm / Whisper decoder LayerNorm on every token row,
        # then the head reads the last rows. Kimi-Audio publishes two
        # Qwen2RMSNorms (norm and mimo_norm); final_norm_params is 2 * hidden.
        # Whisper's final_norm_params is also 2 * hidden, but that is one
        # LayerNorm (weight + bias), so the rsqrt count stays one per token
        # row. The reduction width is hidden, not 2 * hidden. The kind set
        # is vector_rules.fp32_final_norm.
        rows = b_loc * ph.q
        nrm = vr.final_norm_copies(model.vec_kind, model.final_norm_params, model.hidden)
        final = Op("final_norm", "vector", model.n_layers,
                   vec=0 if model.final_norm_params == 0 else rows * model.hidden * 4 * nrm,
                   sfu=0 if model.final_norm_params == 0 else float(rows * nrm), a_fmt="fp32")
    else:
        final = Op("final_norm", "vector", model.n_layers,
                   vec=0 if model.final_norm_params == 0 else b_loc * ph.q * model.hidden * 4)
    ops = [final,
           gemm(model, "lm_head", model.n_layers, m, model.hidden, n, role, stream=True),
           # 0.321: + 1 multiply per logit for a learned logit scale (Dory)
           Op("sample", "vector", model.n_layers, vec=m * n * (4 if model.logit_scale else 3), sfu=m * n)]
    if model.kv_cache:
        # 0.273 extra output heads of an autoregressive model (Kimi-Audio mimo_output: the audio-token head, sampled
        # every step next to the text head), vocab-parallel like the lm_head
        books = model.audio_codebooks
        if books and books[1]:
            # 0.402 LongCat-Flash-Omni. Residual codebook rows are read between audio heads
            # (head_idx > 0) on the same sampled rows as the output linears.
            n_res = books[1]
            bits = model.lookup_bits or model.fmt("embed").bits
            ops.append(Op("audio_residual_lookup", "lookup", model.n_layers,
                          kv_read=m * n_res * model.hidden * bits / 8,
                          vec=m * n_res * model.hidden))
        for l in model.io_post:
            nn_ = _cdiv(l.n, sh.tp)
            ops += [gemm(model, "io." + l.name, model.n_layers, m, l.k, nn_, l.role or "lm_head", stream=True),
                    Op("sample." + l.name, "vector", model.n_layers, vec=m * nn_ * 3, sfu=m * nn_)]
            if sh.tp > 1:
                ops.append(Op("logits_gather." + l.name, "comm", model.n_layers, comm_kind="allgather",
                              comm_group=sh.tp, comm_bytes=m * 2 * 4))
    if glue:
        # 0.177: sampling passes over the logits (trace: apply_penalty, topk, mask_top_p, sort / gatherTopK) — fp32
        # logits read + written per pass
        ops += [Op(f"glue.sample_{x}", "vector", model.n_layers, vec=m * n * 2, act_bytes=m * n * 8)
                for x in ("penalty", "topk", "top_p", "gather")]
    if sh.tp > 1:
        ops.append(Op("logits_gather", "comm", model.n_layers, comm_kind="allgather", comm_group=sh.tp,
                      comm_bytes=m * 2 * 4))  # (max, idx) per rank for greedy/top-k 「假设」
    return ops


def mtp_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard(), depth: int = 0) -> list[Op]:
    """One MTP draft step (module ``depth``): eh_proj + MTP layer + shared head."""
    return rank_shares(_mtp_ops(model, ph, sh, depth), sh, lambda: _mtp_ops(model, ph, Shard(), depth))


def dspark_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard()) -> list[Op]:
    """DeepSeek-V4.1 DSpark block draft (official inference/model.py forward_spec), one rank."""
    return rank_shares(_dspark_ops(model, ph, sh), sh, lambda: _dspark_ops(model, ph, Shard()))


def _dspark_ops(model: ModelSpec, ph: Phase, sh: Shard) -> list[Op]:
    """Prefill (start_pos 0): main_proj + main_norm over the prompt's target-layer stream means, and every draft
    layer only seeds its window cache from main_x (wkv, kv_norm, RoPE, fp8 quant).  Decode: one pass of the
    draft_block positions [last token, noise × (block − 1)] through all draft layers (window of main KV + the whole
    block, non-causal; the main_x KV of the step is written to the ring), hc_pre, norm, the shared head on every
    position, then the Markov head per position (rank-r embed + head, logits add, sample) and the confidence head."""
    b_loc = _cdiv(ph.batch, sh.dp)
    tp, h, N = sh.tp, model.hidden, max(model.layers[0].hc, 1)
    nt = model.draft_targets
    li = model.n_layers + 1
    hd = model.mtp_layers[0].core.qk_dim
    m_rows = b_loc * (ph.q if ph.kind == "prefill" else 1)
    ops = [Op("dspark_target_mean", "vector", li, vec=m_rows * nt * N * h, replicated=tp),
           gemm(model, "dspark.main_proj", li, m_rows, nt * h, h, "mtp", replicated=tp),
           Op("dspark_main_norm", "vector", li, vec=m_rows * h * 4, replicated=tp)]
    for j, L in enumerate(model.mtp_layers):
        lj = li + j
        c = L.core
        # main_x KV: wkv + kv_norm (4/elem) + RoPE + quant (2/elem), written into the ring buffer
        ops.append(gemm(model, "dspark.main_wkv", lj, m_rows, h, model.draft_kv or hd, "mtp", replicated=tp))
        if c.window is None or (model.draft_dflash2 and not c.win_bytes):
            # 0.254 DFlash2: a paged sliding-window GQA draft cache (per_layer_sliding_window) — rows appended like a
            # full cache; storage keeps min(ctx, window) (memplan)
            # 0.221 full-attention draft layer (ReaDy): the step's main_x K/V rows are appended to the draft's own KV
            # cache in the KV format (2·KV·hd per row, draft_kv)
            kvw = m_rows * (model.draft_kv or 2 * hd) * model.fmt("kv").bits / 8
        else:
            kvw = (m_rows * min(c.window, m_rows // max(b_loc, 1)) / max(m_rows // max(b_loc, 1), 1) * c.win_bytes)
        ops.append(Op("dspark_main_kv", "vector", lj, replicated=tp, vec=m_rows * (6 * hd + 3 * c.rope_n),
                      kv_write=kvw))
    if ph.kind == "prefill":
        return ops
    B = model.draft_block
    dph = Phase("decode", ph.batch, B, ph.ctx, skew=ph.skew, v_p=ph.v_p, skew_l=ph.skew_l, v_pl=ph.v_pl)
    ops.append(Op("dspark_embed", "lookup", li, kv_read=b_loc * B * h * model.fmt("embed").bits / 8, vec=b_loc * B * h))
    for j, L in enumerate(model.mtp_layers):
        lj = li + j
        lops = _rank_ops(model, lj, dph, sh, layer=L)
        # the draft block's own KV is temporary (not written to the ring); the block keys are attended non-causally
        lops = [replace(o, kv_write=0.0) if o.name == "softmax" else o for o in lops]
        hq = _cdiv(L.core.n_q, tp)
        # 0.252 DSpine / DFlash draft_block_causal: row i attends block rows ≤ i → (B + 1) / 2B of the B × B scores
        # 0.254 DFlash2 non-causal; 0.311 UBTree's DFlash backbone non-causal 「假设」 (serving code unreleased)
        cz = (B + 1) / (2 * B) if model.draft_anchor and not (model.draft_dflash2 or model.draft_ubtree) else 1.0
        for nm in ("dspark_block_qk", "dspark_block_pv"):
            lops.append(Op(nm, "attn", lj, m=B * hq, k=hd, n=B, count=b_loc, causal=cz, w_fmt="bf16", a_fmt="bf16"))
        lops.append(Op("dspark_block_softmax", "vector", lj, vec=b_loc * B * hq * B * 5 * cz, sfu=b_loc * B * hq * B * cz))
        ops += lops
        if model.draft_dflash2:
            ops += _dflash2_conv_ops(model, lj, b_loc * B, sh)
        if model.draft_bridge and j in model.draft_bridge[3:]:
            ops += _dspine_bridge_ops(model, lj, b_loc * B, sh)
        if model.draft_reader and j == model.draft_reader[6]:
            ops += _ready_reader_ops(model, lj, b_loc, B, sh)
    V = _cdiv(model.vocab, tp)
    r = model.draft_markov
    if model.draft_anchor:
        # 0.252 DSpine (sglang_dspine worker.py DSpineDraftSampler): the target lm_head on the B − 1 draft rows (the
        # anchor row is the committed token), top-K candidates, serial_table (per row: K lm_head rows + 2K TSV rows +
        # K TG rows gathered, K·h + K²·rank MACs, fp32 accumulate), greedy serial chain; no Markov / confidence head
        n = b_loc * (B - 1)
        C_, R_, K_ = model.draft_bridge[:3] if model.draft_bridge else (0, 0, 1)
        ops += [Op("dspark_hc_pre", "vector", li, vec=b_loc * B * 4 * h, replicated=tp),
                gemm(model, "dspark.head", li, n, h, V, "lm_head", stream=True),
                Op("dspine_topk", "vector", li, vec=n * V * 2)]
        if model.draft_bridge:
            ops += [Op("dspine_serial_table", "lookup", li, kv_read=n * K_ * (h + 3 * R_) * 2,
                       vec=n * (K_ * h + K_ * K_ * R_) * 2, replicated=tp),
                    Op("dspine_serial_chain", "vector", li, vec=n * K_ * K_, replicated=tp)]
            if ph.samp:
                # 0.253 sampled round (DSpineDraftSampler.sample, kernels.serial_rs_table 「来源」): per draft row and each
                # of the K senders, e = √C·codebook[sender] → pred_gate / prev (R × C each) → message → write (H × R) →
                # refined state, RMS-normed, dotted with the K candidates' lm_head rows: K·(2RC + HR + KH) MACs per row in
                # true fp32 (TF32 off); gathers K codebook + K lm_head rows per row; then serial_sample_chain (temperature,
                # top-k / top-p / min-p over the K × K table).  Eager, outside the draft graph; the greedy chain still runs
                f32 = dict(w_fmt="fp32", a_fmt="fp32", replicated=tp)
                ops += [Op("dspine_rs_gather", "lookup", li, kv_read=n * K_ * (C_ + h) * 2, vec=n * K_ * (C_ + h)),
                        Op("dspine_rs_bridge", "gemm", li, m=n * K_, k=C_, n=2 * R_, w_params=2 * R_ * C_, w_bits=32,
                           **f32),
                        Op("dspine_rs_write", "gemm", li, m=n * K_, k=R_, n=h, w_params=h * R_, w_bits=32, **f32),
                        Op("dspine_rs_logits", "attn", li, m=K_, k=h, n=K_, count=n, **f32),
                        Op("dspine_rs_sample", "vector", li, vec=n * K_ * K_ * 8, sfu=n * K_ * K_, replicated=tp)]
        if model.draft_ubtree:
            # 0.311 UBTree selector + tree build (config ubtree_config; SGLang code unreleased → 「假设」): hidden_projection
            # h → R per draft row (per-depth affine scales: vector); the K candidates' target-embedding rows gathered and
            # run through the predecessor and successor MLPs (h → F, SiLU, F → R); bigram edges per row = K × K × R
            # (predecessor[q] ⊙ h_p · successor[c]) + unary logits; pruned tree build over the rows' K × K scores
            # (best-first, budget nodes) — per row K² score reads + per node a K-wide pick
            K3, R3, F3, _, _, nodes = model.draft_ubtree
            nk = n * K3
            ops += [gemm(model, "ubtree.hidden_proj", li, n, h, R3, "mtp", replicated=tp),
                    Op("ubtree_depth_affine", "vector", li, vec=n * R3 * 2, replicated=tp),
                    Op("ubtree_cand_embed", "lookup", li, kv_read=nk * h * model.fmt("embed").bits / 8, vec=nk * h,
                       replicated=tp),
                    gemm(model, "ubtree.pred_in", li, nk, h, F3, "mtp", replicated=tp),
                    gemm(model, "ubtree.succ_in", li, nk, h, F3, "mtp", replicated=tp),
                    Op("ubtree_mlp_act", "vector", li, vec=2 * nk * F3 * 4, sfu=2 * nk * F3, replicated=tp),
                    gemm(model, "ubtree.pred_out", li, nk, F3, R3, "mtp", replicated=tp),
                    gemm(model, "ubtree.succ_out", li, nk, F3, R3, "mtp", replicated=tp),
                    Op("ubtree_edges", "attn", li, m=K3, k=R3, n=K3, count=n, w_fmt="bf16", a_fmt="bf16",
                       replicated=tp),
                    Op("ubtree_tree_build", "vector", li, vec=n * K3 * K3 * 2 + b_loc * nodes * K3 * 4,
                       replicated=tp)]
        if model.draft_dflash2:
            # 0.254 DFlash2 candidate selector (models/beam/nvidia/dflash2_ops.candidate_edges + speculator.py
            # _selector_walk_kernel 「来源」): hidden_projection h → R on the draft rows; per row and each of the K
            # predecessors (anchor token for the first row) one predecessor codebook row ⊙ h_p, dotted with the K
            # candidates' successor rows (K·K·R MACs, fp32 accumulate) + the unary logits; then the walk: per row one
            # gumbel-noised argmax over K (greedy: no noise — the same kernel either way, so no extra sampled-round op)
            _, _, R2, K2, _ = model.draft_dflash2
            ops += [gemm(model, "dflash2.selector_proj", li, n, h, R2, "mtp", replicated=tp),
                    Op("dflash2_codebooks", "lookup", li, kv_read=n * 2 * K2 * R2 * 2, vec=n * K2 * R2, replicated=tp),
                    Op("dflash2_edges", "attn", li, m=K2, k=R2, n=K2, count=n, w_fmt="bf16", a_fmt="bf16",
                       replicated=tp),
                    Op("dflash2_walk", "vector", li, vec=n * K2 * 8, sfu=n * K2, replicated=tp)]
        return ops
    ops += [Op("dspark_hc_pre", "vector", li, vec=b_loc * B * (2 * N * h + 4 * h), replicated=tp),
            gemm(model, "dspark.head", li, b_loc * B, h, V, "lm_head", stream=True),
            Op("dspark_markov_embed", "lookup", li, kv_read=b_loc * B * r * 2, vec=b_loc * B * r)]
    if DSPARK_SEQ_MARKOV:
        # 0.222: gamma sequential Markov steps (each re-reads w2; its own kernel)
        for j in range(B):
            ops += [gemm(model, f"dspark.markov_head.{j}", li, b_loc, r, V, "mtp", stream=True),
                    Op(f"dspark_sample.{j}", "vector", li, vec=b_loc * V * 3)]
    else:
        ops += [gemm(model, "dspark.markov_head", li, b_loc * B, r, V, "mtp", stream=True),
                Op("dspark_sample", "vector", li, vec=b_loc * B * V * 3)]
    ops += [
            gemm(model, "dspark.confidence", li, b_loc * B, h + r, 1, "mtp", replicated=tp)]
    return ops


def _dflash2_conv_ops(model: ModelSpec, lj: int, m: int, sh: Shard) -> list[Op]:
    """0.254 the two DFlash2 grouped dynamic convolutions of one draft layer over the m = batch · B block rows
    (qwen3_dflash2.DFlashGroupedConv; models/beam/nvidia/dflash2.py fuses side 1 into the residual norm): per conv a
    kernel_projection GEMM h → 2·taps·groups (ReplicatedLinear → every TP rank), then side 0 before and side 1 after
    the sub-block — per channel taps coefficient loads (base + group delta) and taps MACs (fp32), causal inside the
    block.  Per layer: 2 convs × 2 sides × taps × (mul + add + coefficient add) per element of the m × h rows."""
    taps, grp = model.draft_dflash2[:2]
    h, tp = model.hidden, sh.tp
    G = h // grp
    return [gemm(model, "dflash2.attn_conv_proj", lj, m, h, 2 * taps * G, "mtp", replicated=tp),
            gemm(model, "dflash2.mlp_conv_proj", lj, m, h, 2 * taps * G, "mtp", replicated=tp),
            Op("dflash2_conv", "vector", lj, vec=m * h * 2 * 2 * taps * 3, replicated=tp)]


def _dspine_bridge_ops(model: ModelSpec, lj: int, m: int, sh: Shard) -> list[Op]:
    """0.252 one DSpine code-bridge write over the m = batch · B block rows (sglang_dspine models/dspine.py
    DSpineCodeBridge._forward_fused): prepare_read (residual add + RMS), read GEMM h → C, normalize_codes (L2 + the
    anchor row's codebook row), packed GEMM C → 3R ([gate; pred_gate; prev]), feature_message (sigmoid gate of the
    predecessor), write GEMM R → h, feature_finish (rms-scaled residual add + the next RMSNorm).  TP = 1 in the reference
    (DSpineWorkerV2 refuses tp > 1) → the GEMMs are replicated on every TP rank."""
    C, R = model.draft_bridge[:2]
    h, tp = model.hidden, sh.tp
    b = max(1, m // max(model.draft_block, 1))
    return [Op("dspine_read_prep", "vector", lj, vec=m * h * 6, replicated=tp),
            gemm(model, "dspine.read", lj, m, h, C, "mtp", replicated=tp),
            Op("dspine_codes", "lookup", lj, kv_read=b * C * 2, vec=m * C * 4, replicated=tp),
            gemm(model, "dspine.packed", lj, m, C, 3 * R, "mtp", replicated=tp),
            Op("dspine_message", "vector", lj, vec=m * R * 4, sfu=m * R, replicated=tp),
            gemm(model, "dspine.write", lj, m, R, h, "mtp", replicated=tp),
            Op("dspine_finish", "vector", lj, vec=m * h * 8, replicated=tp)]


def _ready_reader_ops(model: ModelSpec, lj: int, b_loc: int, B: int, sh: Shard) -> list[Op]:
    """0.221 ReaDy recurrent-state reader (MODEL_SPEC.txt / reader_config.json), one decode proposal of B positions.
    Per sequence the committed state of ``nl`` target layers × ``Hs`` heads (dk × dv, state format — fp32) is gathered
    into a CUDA-graph buffer (read + write), RMS-normalised and value-mapped dv → r per head (read again): state DRAM
    bytes = 3 × nl·Hs·dk·dv·bytes 「推导」 (release: 'read the current committed state into stable CUDA-graph buffers for
    every proposal'; the Triton backend fuses gather + normalisation, matrix products stay separate).  Per position: a
    rank-r query per head (h → nl·Hs·r), lifted r → dk, read against the value-mapped state (dk × r), + per-head log
    magnitude → nl·Hs·(r + 1) features → down (→ bn) → SiLU → up (→ h) residual.  State heads are TP-sharded like the
    target's linear-attention layers; the small GEMMs are replicated."""
    r, nl, Hs, dk, dv, bn, _ = model.draft_reader
    tp, h = sh.tp, model.hidden
    hs = nl * _cdiv(Hs, tp)                   # state heads on this rank
    sb = _fmt(model.state_fmt).bytes
    rows = b_loc * B
    return [Op("ready_state_gather", "vector", lj, state_rw=3 * b_loc * hs * dk * dv * sb, vec=b_loc * hs * dk * dv * 4),
            Op("ready_value_map", "attn", lj, m=dk, k=dv, n=r, count=b_loc * hs, w_fmt="fp32", a_fmt="fp32"),
            gemm(model, "ready.query", lj, rows, h, nl * Hs * r, "mtp", replicated=tp),
            Op("ready_query_up", "attn", lj, m=B, k=r, n=dk, count=b_loc * hs, w_fmt="bf16", a_fmt="bf16"),
            Op("ready_state_read", "attn", lj, m=B, k=dk, n=r, count=b_loc * hs, w_fmt="fp32", a_fmt="fp32"),
            Op("ready_norm", "vector", lj, vec=rows * nl * Hs * (r + 1) * 6 + b_loc * hs * 4, sfu=b_loc * hs,
               replicated=tp),
            gemm(model, "ready.down", lj, rows, nl * Hs * (r + 1), bn, "mtp", replicated=tp),
            gemm(model, "ready.up", lj, rows, bn, h, "mtp", replicated=tp),
            Op("ready_residual", "vector", lj, vec=rows * (bn * 4 + h), sfu=rows * bn, replicated=tp)]


def _mtp_ops(model: ModelSpec, ph: Phase, sh: Shard = Shard(), depth: int = 0) -> list[Op]:
    if not model.mtp_layers:
        return []
    L = model.mtp_layers[min(depth, len(model.mtp_layers) - 1)]
    li = model.n_layers + 1 + depth
    b_loc = _cdiv(ph.batch, sh.dp)
    t = b_loc * ph.q
    h = model.hidden
    if L.gres_n:
        # Qwen4Exp MTP input fusion (SGLang qwen4_exp_mtp.py): fc_embedding on the normed embedding, fc_hidden on each
        # of the N normed hc streams, broadcast add; GemmaRMSNorms over h and N·h (4/elem)
        N = L.gres_n
        ops = [gemm(model, "mtp.fc_embedding", li, t, h, h, "mtp", replicated=sh.tp, stream=True),
               gemm(model, "mtp.fc_hidden", li, t * N, h, h, "mtp", replicated=sh.tp, stream=True),
               Op("mtp.fuse", "vector", li, vec=t * (4 * h + 4 * N * h + N * h), replicated=sh.tp)]
    else:
        eh_role = model.mtp_eh_role or "mtp"
        ops = [gemm(model, "mtp.eh_proj", li, t, 2 * h, h, eh_role, replicated=sh.tp, stream=True)]
    ops += _rank_ops(model, li, ph, sh, layer=L)
    ops += [gemm(model, "mtp.head", li, t, h, _cdiv(model.vocab, sh.tp), "lm_head", stream=True)]
    return ops


# ------------------------------------------------------------------ full (non-autoregressive) forward
def _prefix(model: ModelSpec, ph: Phase | None = None) -> int:
    """Joint text rows of a full forward: the phase's own (0.258: CFG negative branch) else the model's prefix."""
    if ph is not None and ph.txt:
        return ph.txt
    return model.workload.prefix_tokens if model.workload else 0


def _video_rows(model: ModelSpec, ph: Phase) -> int:
    """Video tokens of the sequence (minus the joint text prefix and the joint audio rows)."""
    return ph.q - ph.aux - ph.ref if ph.kvpre else ph.q - _prefix(model, ph) - ph.aux - ph.ref


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
    if ph.mreuse and l.rows in ("txt", "ref"):    # 0.255: reused text / reference streams run no rows
        return 0
    if l.rows == "txt":         # 0.116: a cached prefix runs no text rows
        return 0 if ph.kvpre else b * _cdiv(_prefix(model, ph), sp)
    if l.rows == "aud":
        return b * _cdiv(ph.aux, sp)
    if l.rows == "ref":         # 0.252 FLUX 3 reference-image rows (image_cond stream)
        return b * _cdiv(ph.ref, sp)
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
        if core.span == "mode":
            # 0.252 FLUX 3 per-mode blocks (BFL flux3/dit.py content_mode_blocks / txt_mode_blocks; diffusers
            # Flux3ModeTransformerBlock "self-attention restricted to a single token mode"): text, noisy image and the
            # reference images (all references of a sample jointly) each attend within themselves only
            ops = []
            gs = (_video_rows(model, ph),) if ph.mreuse else (_prefix(model, ph), _video_rows(model, ph), ph.ref)
            for g in gs:
                if g > 0:
                    ops += _full_attn(model, li, replace(core, span="full"), replace(ph, q=g, aux=0, ref=0, kvpre=0),
                                      sh, b, cross)
            # 0.255 reused step (SGLang StaticStreams): only the image mode block runs (QK-norm / RoPE on its rows)
            tail = replace(ph, q=gs[0], aux=0, ref=0, kvpre=0, mreuse=False) if ph.mreuse else ph
            return [o for o in ops if o.name != "qk_norm_rope" and not o.name.startswith("sp_a2a")] + \
                [o for o in _full_attn(model, li, replace(core, span="full"), tail, sh, b, cross)
                 if o.name == "qk_norm_rope" or o.name.startswith("sp_a2a")]
        if core.span != "full" and ph.frames:
            # factorized ST attention over the video grid: spatial = one group per latent frame (H·W tokens),
            # temporal = one group per spatial position (T tokens)
            vid, T = _video_rows(model, ph), ph.frames
            hw = _cdiv(vid, T)
            groups, nq = (T, hw) if core.span == "spatial" else (hw, T)
            nk = nq
    causal = 0.5 if core.causal else 1.0
    kv_r = kv_w = 0.0
    wl = model.workload
    if not cross and wl is not None and wl.cond_causal and groups == 1:
        T = _prefix(model, ph)
        if ph.kvpre:
            # 0.116 cached step (QwenImage21 kv_cache_mode "cached"): image queries over [cached prefix ‖ image];
            # the prefix K/V (post-RoPE) are read from the cache
            nk = ph.q + ph.kvpre
            kv_r = b * h_loc * ph.kvpre * (core.qk_dim + core.v_dim) * ab
        elif T:
            # block-causal mask: text rows causal over the text keys, image rows over every key
            N = ph.q - T
            causal = (N * ph.q + T * (T + 1) / 2) / (ph.q * ph.q)
            if ph.kvx:          # extract step (kv_cache_mode "extract"): the prefix K/V are written to the cache
                kv_w = b * h_loc * T * (core.qk_dim + core.v_dim) * ab
    cnt = b * h_loc * groups
    ops = [Op(tag + "qk", "attn", li, m=nq, k=core.qk_dim, n=nk, count=cnt, causal=causal, kv_read=kv_r, kv_write=kv_w,
              act_bytes=cnt * nq * core.qk_dim * ab, stream=True, orient=True),
           Op(tag + "pv", "attn", li, m=nq, k=nk, n=core.v_dim, count=cnt, causal=causal,
              act_bytes=cnt * nq * core.v_dim * ab, orient=True),
           Op(tag + "softmax", "vector", li, vec=cnt * nq * nk * causal * 5, sfu=cnt * nq * nk * causal)]
    t_loc = b * _cdiv(ph.q, sh.sp)
    if not cross:
        # qk_norm True (default): the historical lump, QK-norm plus RoPE, 3 element-ops each.
        # qk_norm False: RoPE only, on the published q and k tensors (before repeat_kv).
        # 3 element-ops per element. k uses n_kv heads. MHA (n_kv == n_q) matches the
        # previous query-head times 2 count. ESM-2 and Carbon-A have no QK-norm.
        if core.qk_norm or core.rope_dim:
            if core.qk_norm:
                qk_part = core.qk_dim * 2 * 3
                rope_part = core.rope_dim * 2 * 3
                ops.append(Op("qk_norm_rope", "vector", li,
                              vec=t_loc * heads_tp * (qk_part + rope_part)))
            else:
                hk = _cdiv(core.n_kv, sh.tp)
                ops.append(Op("rope", "vector", li,
                              vec=t_loc * (heads_tp + hk) * core.rope_dim * 3))
        if core.q_scale:
            # query * attention_head_size**-0.5, before RoPE. One multiply. The class does not upcast.
            ops.append(Op("q_scale", "vector", li, vec=t_loc * heads_tp * core.qk_dim, a_fmt=model.act_fmt))
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
    the MSA grids along one axis over ``dap`` ranks; token / atom grids are replicated.  ``spair`` (0.61.3, Protenix
    diffusion: the pair conditioning expanded over the samples) = one pair grid per diffusion sample, DAP-split along
    residues for all samples (the samples a rank owns under ``sample_split`` still need every sample's slice)."""
    n = pd.n
    nd = _cdiv(n, dap)
    return {"res": b * s * n, "tok": b * s * n, "seq": b * s, "pair": b * ps * nd * n,
            "msa": b * ps * _cdiv(pd.msa, dap) * n, "xmsa": b * ps * _cdiv(pd.xmsa, dap) * n,
            "tmpl": b * pd.tmpl * nd * n, "tres": b * pd.tmpl * n, "spair": b * pd.samples * nd * n, "atom": b * s * pd.atoms,
            "apair": b * s * _cdiv(pd.atoms, 32) * 32 * 128}[tag]


_DAP_SPLIT = frozenset({"pair", "spair", "msa", "xmsa", "tmpl"})


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
                   orient=True, launch=1),          # 0.120: one fused kernel (QK·softmax·PV)
                Op(tag + ".pv", "attn", li, m=nq, k=nk, n=vd, count=cnt, act_bytes=cnt * nq * vd * ab, orient=True),
                Op(tag + ".softmax", "vector", li, vec=cnt * nq * nk * (5 + (1 if bias else 0)), sfu=cnt * nq * nk)]

    def bmm(m: int, k: int, nn: int, cnt: int) -> Op:
        return Op(tag, "attn", li, m=m, k=k, n=nn, count=cnt, act_bytes=cnt * (m * k + k * nn + m * nn) * ab,
                  stream=True, bmm=True, launch=1)

    d, vd = c.dim, c.v_dim or c.dim
    if c.kind == "trimul":
        cm("allgather", bt * nd * n * c.dim, "gather")
        return [bmm(nd, n, n, bt * c.dim)] + comm
    if c.kind == "tri_att":
        cm("allgather", bt * nd * n * c.heads, "bias")
        cm("alltoall", bt * nd * n * (c.width or model.workload.pair_dim), "transpose")
        if pd.tri_chunk:
            return _tri_unfused(tag, li, bt * nd, n, d, vd, c.heads, pd.tri_chunk, ab) + comm
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
                Op(tag + ".softmax", "vector", li, vec=b * ps * c.heads * nd * n * 5, sfu=b * ps * c.heads * nd * n)] + comm
    if c.kind == "seq_att":
        cm("allgather", b * nd * n * c.heads, "bias")
        return att(b * s, n, n, d, vd, True) + comm
    if c.kind == "local_att":
        wq, wk = c.win
        return att(b * s * _cdiv(pd.atoms, wq), wq, wk, d, vd, True)
    raise ValueError(f"unknown pair core {c.kind}")


def _tri_unfused(tag: str, li: int, rows: int, n: int, d: int, vd: int, heads: int, chunk: int, ab: float) -> list[Op]:
    """0.119: triangle attention as the reference PyTorch kernels run it (OpenFold / Boltz / Protenix / ESMFold
    ``Attention``): logits = q·kᵀ (+ pair bias) written as a tensor, softmax in place, then ·v.  ``rows`` = pair rows
    (attention batch) on this rank; ``chunk`` < 0 materialises all rows at once, c > 0 runs ⌈rows / c⌉ chunks of c
    rows (``chunk_layer``).  Same FLOPs as the fused kernel; the per-chunk logits set the activation peak and, when they
    exceed the SRAM budget, the logits make 4 DRAM passes (write, softmax read + write, read by P·V)."""
    cr = rows if chunk < 0 else min(rows, chunk)
    nch = math.ceil(rows / cr)
    cnt = rows * heads
    lg = cnt * n * n * ab                                   # all logits of this module execution
    peak = cr * heads * (2 * n * d + n * n + n * vd) * ab   # one chunk: q, k, logits, v (+ out)
    # 0.120: three kernels (QKᵀ, softmax, ·V) per chunk when Chip.kernel_us > 0
    return [Op(tag + ".qk", "attn", li, m=n, k=d, n=n, count=cnt, act_bytes=peak, stream=True, bmm=True,
               chunks=nch, act_x=2 * lg, launch=nch, lg_bytes=lg),
            Op(tag + ".pv", "attn", li, m=n, k=n, n=vd, count=cnt, act_bytes=peak, stream=True, bmm=True, chunks=nch,
               launch=nch, lg_bytes=lg),
            Op(tag + ".softmax", "vector", li, vec=cnt * n * n * 6, sfu=cnt * n * n, launch=nch)]


def _esm_ln(name: str, rows: int, width: int, li: int, rep: float, who: str = "esmfold") -> Op:
    """nn.LayerNorm / OpenFold LayerNorm / AF2 haiku LayerNorm (F.layer_norm). fp32 reduction, one rsqrt per row.

    Affine weights are already in the layer's misc parameters. eps is not billed.
    """
    if rows <= 0 or width <= 0:
        raise ValueError(f"{who} {name} has no rows")
    return Op(name, "vector", li, vec=float(rows * width * 4), sfu=float(rows), a_fmt="fp32", replicated=rep)


def _esm_relu(name: str, n: int, li: int, rep: float, af: str, who: str = "esmfold") -> Op:
    """nn.ReLU is max(x, 0). One clip, no transcendental, no upcast."""
    if n <= 0:
        raise ValueError(f"{who} {name} has no elements")
    return Op(name, "vector", li, vec=float(n), a_fmt=af, replicated=rep)


def _bz_silu(name: str, n: int, li: int, rep: float, af: str) -> Op:
    """nn.SiLU is x * sigmoid(x). Four element-ops, two of them on the SFU, no upcast."""
    if n <= 0:
        raise ValueError(f"boltz-1 {name} has no elements")
    return Op(name, "vector", li, vec=float(n * 4), sfu=float(n * 2), a_fmt=af, replicated=rep)


def _bz_mul(name: str, n: int, li: int, rep: float, af: str) -> Op:
    """silu(fc1) * fc2. One multiply, no transcendental, no upcast."""
    if n <= 0:
        raise ValueError(f"boltz-1 {name} has no elements")
    return Op(name, "vector", li, vec=float(n), a_fmt=af, replicated=rep)


def _of_mask(name: str, n: int, li: int, rep: float, af: str) -> Op:
    """One published multiply by a mask that already has the activation dtype.

    OpenFold MSATransition / PairTransition: linear_2(x) * mask. Width is linear_2's N.
    OpenFold TemplateEmbedder and AlphaFold 2 TemplateEmbedding: the pointwise-attention
    output times (sum(template_mask) > 0). Width is linear_o's N (c_z), on the pair grid.
    Building the mask is not an op. No transcendental and no upcast.
    """
    if n <= 0:
        raise ValueError(f"openfold {name} has no elements")
    return Op(name, "vector", li, vec=float(n), a_fmt=af, replicated=rep)


def _esmfold_state_copies(model: ModelSpec) -> int:
    """lddt reads structure['states'], one row-block per StructureModule iteration."""
    found = [L.iters for L in model.layers if L.vec_kind == "esm_sm"]
    if len(found) != 1 or found[0] < 1:
        raise ValueError("esmfold lddt rows are not the structure-module block count")
    return found[0]


# TriangularSelfAttentionBlock linears (facebookresearch/esm tri_self_attn_block.py and the
# transformers port). Norm companions are billed separately; sigmoid gates are not.
ESMFOLD_TRUNK_LINEARS = frozenset({
    "mlp_pair.mlp.1", "mlp_pair.mlp.3", "mlp_seq.mlp.1", "mlp_seq.mlp.3",
    "pair_to_sequence.linear", "seq_attention.g_proj", "seq_attention.o_proj", "seq_attention.proj",
    "sequence_to_pair.o_proj", "sequence_to_pair.proj",
    "tri_att_end.linear", "tri_att_end.mha.linear_g", "tri_att_end.mha.linear_k",
    "tri_att_end.mha.linear_o", "tri_att_end.mha.linear_q", "tri_att_end.mha.linear_v",
    "tri_att_start.linear", "tri_att_start.mha.linear_g", "tri_att_start.mha.linear_k",
    "tri_att_start.mha.linear_o", "tri_att_start.mha.linear_q", "tri_att_start.mha.linear_v",
    "tri_mul_in.linear_a_g", "tri_mul_in.linear_a_p", "tri_mul_in.linear_b_g", "tri_mul_in.linear_b_p",
    "tri_mul_in.linear_g", "tri_mul_in.linear_z",
    "tri_mul_out.linear_a_g", "tri_mul_out.linear_a_p", "tri_mul_out.linear_b_g", "tri_mul_out.linear_b_p",
    "tri_mul_out.linear_g", "tri_mul_out.linear_z",
})
_ESM_TRUNK_LN_BEFORE = {
    "pair_to_sequence.linear": "pair_to_seq_ln",
    "seq_attention.proj": "seq_ln",
    "mlp_seq.mlp.1": "mlp_seq_ln",
    "sequence_to_pair.proj": "seq_to_pair_ln",
    "tri_mul_out.linear_a_g": "tri_mul_out_ln_in",
    "tri_mul_out.linear_z": "tri_mul_out_ln_out",
    "tri_mul_in.linear_a_g": "tri_mul_in_ln_in",
    "tri_mul_in.linear_z": "tri_mul_in_ln_out",
    "tri_att_start.linear": "tri_att_start_ln",
    "tri_att_end.linear": "tri_att_end_ln",
    "mlp_pair.mlp.1": "mlp_pair_ln",
}
_ESM_TRUNK_RELU_AFTER = {
    "mlp_seq.mlp.1": "mlp_seq_relu",
    "mlp_pair.mlp.1": "mlp_pair_relu",
}
# StructureModule loop body (openfold structure_module.py _forward_monomer). linear_in is not here:
# it runs once, before the block loop.
ESMFOLD_SM_LINEARS = frozenset({
    "angle_resnet.layers.0.linear_1", "angle_resnet.layers.0.linear_2",
    "angle_resnet.layers.1.linear_1", "angle_resnet.layers.1.linear_2",
    "angle_resnet.linear_in", "angle_resnet.linear_initial", "angle_resnet.linear_out",
    "bb_update.linear",
    "ipa.linear_b", "ipa.linear_kv", "ipa.linear_kv_points", "ipa.linear_out",
    "ipa.linear_q", "ipa.linear_q_points",
    "transition.layers.0.linear_1", "transition.layers.0.linear_2", "transition.layers.0.linear_3",
})
_ESM_SM_RELU_BEFORE = {
    "angle_resnet.linear_initial": "angle_init_relu",
    "angle_resnet.linear_in": "angle_in_relu",
    "angle_resnet.layers.0.linear_1": "angle_b0_relu_1",
    "angle_resnet.layers.0.linear_2": "angle_b0_relu_2",
    "angle_resnet.layers.1.linear_1": "angle_b1_relu_1",
    "angle_resnet.layers.1.linear_2": "angle_b1_relu_2",
    "angle_resnet.linear_out": "angle_out_relu",
}
_ESM_SM_RELU_AFTER = {
    "transition.layers.0.linear_1": "trans_relu_1",
    "transition.layers.0.linear_2": "trans_relu_2",
}
_ESM_SM_LN_AFTER = {
    "ipa.linear_out": "ipa_ln",
    "transition.layers.0.linear_3": "trans_ln",
}


def _strip_core(name: str) -> str:
    """OpenFold nests Evoformer pair modules under ``core.``. AF2's mapped names do not."""
    return name[5:] if name.startswith("core.") else name


def _mha(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}.linear_{leaf}" for leaf in ("g", "k", "o", "q", "v"))


def _tri_mul(side: str) -> tuple[str, ...]:
    return tuple(f"tri_mul_{side}.linear_{leaf}" for leaf in ("a_g", "a_p", "b_g", "b_p", "g", "z"))


def _tri_att(side: str) -> tuple[str, ...]:
    return (f"tri_att_{side}.linear",) + _mha(f"tri_att_{side}.mha")


_OF_PAIR_TAIL = (
    ("pair_transition.linear_1", "pair_transition.linear_2")
    + _tri_att("end") + _tri_att("start") + _tri_mul("in") + _tri_mul("out")
)
OPENFOLD_TMPL_LINEARS = frozenset(_OF_PAIR_TAIL)
OPENFOLD_EVO_LINEARS = frozenset(
    ("msa_transition.linear_1", "msa_transition.linear_2",
     "outer_product_mean.linear_1", "outer_product_mean.linear_2", "outer_product_mean.linear_out")
    + _OF_PAIR_TAIL
    + _mha("msa_att_col._msa_att.mha")
    + ("msa_att_row.linear_z",)
    + _mha("msa_att_row.mha")
)
OPENFOLD_EXTRA_LINEARS = frozenset(
    ("msa_transition.linear_1", "msa_transition.linear_2",
     "outer_product_mean.linear_1", "outer_product_mean.linear_2", "outer_product_mean.linear_out")
    + _OF_PAIR_TAIL
    + _mha("msa_att_col.global_attention")
    + ("msa_att_row.linear_z",)
    + _mha("msa_att_row.mha")
)
# LayerNorm sits on the activation that feeds the named linear. Width is that linear's K.
_OF_BLOCK_LN_BEFORE = {
    "msa_transition.linear_1": "msa_trans_ln",
    "outer_product_mean.linear_1": "opm_ln",
    "pair_transition.linear_1": "pair_trans_ln",
    "tri_att_end.linear": "tri_att_end_ln",
    "tri_att_start.linear": "tri_att_start_ln",
    "tri_mul_in.linear_a_g": "tri_mul_in_ln_in",
    "tri_mul_in.linear_z": "tri_mul_in_ln_out",
    "tri_mul_out.linear_a_g": "tri_mul_out_ln_in",
    "tri_mul_out.linear_z": "tri_mul_out_ln_out",
    "msa_att_col._msa_att.mha.linear_g": "msa_col_ln",
    "msa_att_col.global_attention.linear_g": "msa_col_ln",
    "msa_att_row.linear_z": "msa_row_z_ln",
    "msa_att_row.mha.linear_g": "msa_row_m_ln",
}
# ReLU follows the expansion linear. MSATransition and PairTransition only. Width is N.
_OF_BLOCK_RELU_AFTER = {
    "msa_transition.linear_1": "msa_trans_relu",
    "pair_transition.linear_1": "pair_trans_relu",
}
# Output multiply follows linear_2. OpenFold only: the published forward always multiplies.
# AlphaFold 2's Transition expands the mask and returns the activation without this product.
_OF_BLOCK_MASK_AFTER = {
    "msa_transition.linear_2": "msa_trans_mask",
    "pair_transition.linear_2": "pair_trans_mask",
}
# AlphaFold 2 haiku names for the same StructureModule loop (initial_projection is the preamble).
ALPHAFOLD2_SM_LINEARS = frozenset({
    "affine_update",
    "ipa.linear_b", "ipa.linear_kv", "ipa.linear_kv_points", "ipa.linear_out",
    "ipa.linear_q", "ipa.linear_q_points",
    "rigid_sidechain.input_projection", "rigid_sidechain.input_projection_1",
    "rigid_sidechain.resblock1", "rigid_sidechain.resblock1_1",
    "rigid_sidechain.resblock2", "rigid_sidechain.resblock2_1",
    "rigid_sidechain.unnormalized_angles",
    "transition", "transition_1", "transition_2",
})
_OF_SM_RELU_BEFORE = {
    **_ESM_SM_RELU_BEFORE,
    "rigid_sidechain.input_projection": "angle_in_relu",
    "rigid_sidechain.input_projection_1": "angle_init_relu",
    "rigid_sidechain.resblock1": "angle_b0_relu_1",
    "rigid_sidechain.resblock2": "angle_b0_relu_2",
    "rigid_sidechain.resblock1_1": "angle_b1_relu_1",
    "rigid_sidechain.resblock2_1": "angle_b1_relu_2",
    "rigid_sidechain.unnormalized_angles": "angle_out_relu",
}
_OF_SM_RELU_AFTER = {
    **_ESM_SM_RELU_AFTER,
    "transition": "trans_relu_1",
    "transition_1": "trans_relu_2",
}
_OF_SM_LN_AFTER = {
    **_ESM_SM_LN_AFTER,
    "transition_2": "trans_ln",
}
OPENFOLD_HEAD_LINEARS = frozenset({
    "distogram.linear", "experimentally_resolved.linear", "masked_msa.linear",
    "plddt.linear_1", "plddt.linear_2", "plddt.linear_3", "tm.linear",
})
ALPHAFOLD2_HEAD_LINEARS = frozenset({
    "distogram.linear", "experimentally_resolved.linear", "masked_msa.linear",
    "plddt.act_0", "plddt.act_1", "plddt.logits", "tm.linear",
})
_OF_PLDDT_LN = {"plddt.linear_1": "plddt_ln", "plddt.act_0": "plddt_ln"}
_OF_PLDDT_RELU = {
    "plddt.linear_1": "plddt_relu_1", "plddt.linear_2": "plddt_relu_2",
    "plddt.act_0": "plddt_relu_1", "plddt.act_1": "plddt_relu_2",
}
OPENFOLD_EMBED_LINEARS = frozenset({
    "evoformer.linear", "extra_msa_embedder.linear",
    "input_embedder.linear_msa_m", "input_embedder.linear_relpos",
    "input_embedder.linear_tf_m", "input_embedder.linear_tf_z_i", "input_embedder.linear_tf_z_j",
    "recycling_embedder.linear",
    "template_angle_embedder.linear_1", "template_angle_embedder.linear_2",
    "template_pair_embedder.linear",
    "template_pointwise_att.mha.linear_k", "template_pointwise_att.mha.linear_o",
    "template_pointwise_att.mha.linear_q", "template_pointwise_att.mha.linear_v",
})


def _bz_mha(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}.linear_{leaf}" for leaf in ("g", "k", "o", "q", "v"))


def _bz_tri_names(side: str) -> tuple[str, ...]:
    return (f"tri_mul_{side}.g_in", f"tri_mul_{side}.g_out", f"tri_mul_{side}.p_in", f"tri_mul_{side}.p_out")


def _bz_tri_att_names(side: str) -> tuple[str, ...]:
    return (f"tri_att_{side}.linear",) + _bz_mha(f"tri_att_{side}.mha")


_BZ_TRI = _bz_tri_names("in") + _bz_tri_names("out") + _bz_tri_att_names("end") + _bz_tri_att_names("start")
BOLTZ_PAIR_LINEARS = frozenset(
    ("attention.proj_g", "attention.proj_k", "attention.proj_o", "attention.proj_q", "attention.proj_v",
     "attention.proj_z.1",
     "transition_s.fc1", "transition_s.fc2", "transition_s.fc3",
     "transition_z.fc1", "transition_z.fc2", "transition_z.fc3")
    + _BZ_TRI
)
BOLTZ_MSA_LINEARS = frozenset(
    ("msa_transition.fc1", "msa_transition.fc2", "msa_transition.fc3",
     "outer_product_mean.proj_a", "outer_product_mean.proj_b", "outer_product_mean.proj_o",
     "pair_weighted_averaging.proj_g", "pair_weighted_averaging.proj_m",
     "pair_weighted_averaging.proj_o", "pair_weighted_averaging.proj_z",
     "z_transition.fc1", "z_transition.fc2", "z_transition.fc3")
    + _BZ_TRI
)
BOLTZ_CONF_LINEARS = frozenset({
    "confidence_heads.to_pae_logits", "confidence_heads.to_pde_logits",
    "confidence_heads.to_plddt_logits", "confidence_heads.to_resolved_logits",
    "dist_bin_pairwise_embed",
    "input_embedder.atom_attention_encoder.atom_to_token_trans.0",
    "input_embedder.atom_attention_encoder.c_to_p_trans_k.1",
    "input_embedder.atom_attention_encoder.c_to_p_trans_q.1",
    "input_embedder.atom_attention_encoder.embed_atom_features",
    "input_embedder.atom_attention_encoder.embed_atompair_ref_pos",
    "input_embedder.atom_attention_encoder.p_mlp.1",
    "input_embedder.atom_attention_encoder.p_mlp.3",
    "input_embedder.atom_attention_encoder.p_mlp.5",
    "msa_module.msa_proj", "msa_module.s_proj", "rel_pos.linear_layer",
    "s_diffusion_to_s", "s_init", "s_recycle", "s_to_z",
    "s_to_z_prod_in1", "s_to_z_prod_in2", "s_to_z_prod_out", "s_to_z_transpose",
    "z_init_1", "z_init_2", "z_recycle",
})
BOLTZ_DIFF_LINEARS = frozenset({
    "out_token_feat_update.transition_block.a_to_b",
    "out_token_feat_update.transition_block.adaln.s_bias",
    "out_token_feat_update.transition_block.adaln.s_scale",
    "out_token_feat_update.transition_block.b_to_a",
    "out_token_feat_update.transition_block.output_projection.0",
    "out_token_feat_update.transition_block.swish_gate.0",
    "score_model.atom_attention_decoder.a_to_q_trans",
    "score_model.atom_attention_decoder.atom_feat_to_atom_pos_update.1",
    "score_model.atom_attention_encoder.atom_to_token_trans.0",
    "score_model.atom_attention_encoder.r_to_q_trans",
    "score_model.s_to_a_linear.1",
    "score_model.single_conditioner.fourier_to_single",
    "score_model.single_conditioner.single_embed",
    "score_model.single_conditioner.transitions.0.fc1",
    "score_model.single_conditioner.transitions.0.fc2",
    "score_model.single_conditioner.transitions.0.fc3",
    "score_model.single_conditioner.transitions.1.fc1",
    "score_model.single_conditioner.transitions.1.fc2",
    "score_model.single_conditioner.transitions.1.fc3",
})
BOLTZ_MSA_REST = frozenset({"msa_proj", "s_proj"})
# LayerNorm feeds the named linear. Width is that linear's K, except where the comment says otherwise.
_BZ_PAIR_LN = {
    "attention.proj_q": "attn_s_ln",
    "attention.proj_z.1": "attn_z_ln",
    "transition_s.fc1": "trans_s_ln",
    "transition_z.fc1": "trans_z_ln",
    "tri_att_start.linear": "tri_att_start_ln",
    "tri_att_end.linear": "tri_att_end_ln",
    "tri_mul_in.p_in": "tri_mul_in_ln_in",
    "tri_mul_out.p_in": "tri_mul_out_ln_in",
    "tri_mul_in.p_out": "tri_mul_in_ln_out",
    "tri_mul_out.p_out": "tri_mul_out_ln_out",
}
_BZ_PAIR_SILU = {"transition_s.fc1": "trans_s_silu", "transition_z.fc1": "trans_z_silu"}
_BZ_PAIR_MUL = {"transition_s.fc2": "trans_s_mul", "transition_z.fc2": "trans_z_mul"}
_BZ_MSA_LN = {
    "msa_transition.fc1": "msa_trans_ln",
    "z_transition.fc1": "z_trans_ln",
    "outer_product_mean.proj_a": "opm_ln",
    "pair_weighted_averaging.proj_m": "pwa_m_ln",
    "pair_weighted_averaging.proj_z": "pwa_z_ln",
    "tri_att_start.linear": "tri_att_start_ln",
    "tri_att_end.linear": "tri_att_end_ln",
    "tri_mul_in.p_in": "tri_mul_in_ln_in",
    "tri_mul_out.p_in": "tri_mul_out_ln_in",
    "tri_mul_in.p_out": "tri_mul_in_ln_out",
    "tri_mul_out.p_out": "tri_mul_out_ln_out",
}
_BZ_MSA_SILU = {"msa_transition.fc1": "msa_trans_silu", "z_transition.fc1": "z_trans_silu"}
_BZ_MSA_MUL = {"msa_transition.fc2": "msa_trans_mul", "z_transition.fc2": "z_trans_mul"}
_BZ_CACHE_PREFIX = "diffusion.score_model.pairwise_conditioner."
_BZ_CACHE_LN = {
    _BZ_CACHE_PREFIX + "dim_pairwise_init_proj.1": "pair_init_ln",
    _BZ_CACHE_PREFIX + "transitions.0.fc1": "pair_trans0_ln",
    _BZ_CACHE_PREFIX + "transitions.1.fc1": "pair_trans1_ln",
}
_BZ_CACHE_SILU = {
    _BZ_CACHE_PREFIX + "transitions.0.fc1": "pair_trans0_silu",
    _BZ_CACHE_PREFIX + "transitions.1.fc1": "pair_trans1_silu",
}
_BZ_CACHE_MUL = {
    _BZ_CACHE_PREFIX + "transitions.0.fc2": "pair_trans0_mul",
    _BZ_CACHE_PREFIX + "transitions.1.fc2": "pair_trans1_mul",
}
_BZ_DIFF_LN = {
    "score_model.single_conditioner.single_embed": "norm_single",
    "score_model.single_conditioner.transitions.0.fc1": "single_trans0_ln",
    "score_model.single_conditioner.transitions.1.fc1": "single_trans1_ln",
    "score_model.s_to_a_linear.1": "s_to_a_ln",
}
_BZ_DIFF_SILU = {
    "score_model.single_conditioner.transitions.0.fc1": "single_trans0_silu",
    "score_model.single_conditioner.transitions.1.fc1": "single_trans1_silu",
}
_BZ_DIFF_MUL = {
    "score_model.single_conditioner.transitions.0.fc2": "single_trans0_mul",
    "score_model.single_conditioner.transitions.1.fc2": "single_trans1_mul",
}
_BZ_CACHE_SHAPES = (
    (_BZ_CACHE_PREFIX + "dim_pairwise_init_proj.1", 256, 128, "pair"),
    (_BZ_CACHE_PREFIX + "transitions.0.fc1", 128, 256, "pair"),
    (_BZ_CACHE_PREFIX + "transitions.0.fc2", 128, 256, "pair"),
    (_BZ_CACHE_PREFIX + "transitions.0.fc3", 256, 128, "pair"),
    (_BZ_CACHE_PREFIX + "transitions.1.fc1", 128, 256, "pair"),
    (_BZ_CACHE_PREFIX + "transitions.1.fc2", 128, 256, "pair"),
    (_BZ_CACHE_PREFIX + "transitions.1.fc3", 256, 128, "pair"),
)


def _bz_by(layer: Layer) -> dict:
    return {ln.name: ln for ln in layer.pair_linears}


def _bz_lin(by: dict, name: str, k: int, n: int, rows: str) -> None:
    lin = by.get(name)
    if lin is None or (lin.k, lin.n, lin.rows) != (k, n, rows):
        raise ValueError(f"boltz-1 {name} shape is not [{k}, {n}] on {rows}")


def _bz_transition(by: dict, prefix: str, k: int, hidden: int, rows: str) -> None:
    _bz_lin(by, f"{prefix}.fc1", k, hidden, rows)
    _bz_lin(by, f"{prefix}.fc2", k, hidden, rows)
    _bz_lin(by, f"{prefix}.fc3", hidden, k, rows)


def _bz_triangle(by: dict) -> None:
    for side in ("in", "out"):
        _bz_lin(by, f"tri_mul_{side}.p_in", 128, 256, "pair")
        _bz_lin(by, f"tri_mul_{side}.g_in", 128, 256, "pair")
        _bz_lin(by, f"tri_mul_{side}.p_out", 128, 128, "pair")
    for side in ("start", "end"):
        _bz_lin(by, f"tri_att_{side}.linear", 128, 4, "pair")


def _bz_check_pair(layer: Layer) -> None:
    if {ln.name for ln in layer.pair_linears} != BOLTZ_PAIR_LINEARS:
        raise ValueError("boltz-1 pairformer linears are not the published set")
    by = _bz_by(layer)
    _bz_lin(by, "attention.proj_q", 384, 384, "res")
    _bz_lin(by, "attention.proj_z.1", 128, 16, "pair")
    _bz_transition(by, "transition_s", 384, 1536, "res")
    _bz_transition(by, "transition_z", 128, 512, "pair")
    _bz_triangle(by)


def _bz_check_msa(layer: Layer) -> None:
    if {ln.name for ln in layer.pair_linears} != BOLTZ_MSA_LINEARS:
        raise ValueError("boltz-1 msa linears are not the published set")
    by = _bz_by(layer)
    _bz_transition(by, "msa_transition", 64, 256, "msa")
    _bz_transition(by, "z_transition", 128, 512, "pair")
    _bz_lin(by, "outer_product_mean.proj_a", 64, 32, "msa")
    _bz_lin(by, "pair_weighted_averaging.proj_m", 64, 256, "msa")
    _bz_lin(by, "pair_weighted_averaging.proj_z", 128, 8, "pair")
    _bz_triangle(by)


def _bz_check_recycle(layer: Layer) -> None:
    if len(layer.pair_linears) != 1:
        raise ValueError("boltz-1 recycle projection is not one linear")
    lin = layer.pair_linears[0]
    if (lin.k, lin.n, lin.rows) not in ((384, 384, "res"), (128, 128, "pair")):
        raise ValueError("boltz-1 recycle projection shape is not the published pair")


def _bz_check_conf(layer: Layer) -> None:
    if {ln.name for ln in layer.pair_linears} != BOLTZ_CONF_LINEARS:
        raise ValueError("boltz-1 confidence linears are not the published set")
    by = _bz_by(layer)
    _bz_lin(by, "s_recycle", 384, 384, "res")
    _bz_lin(by, "z_recycle", 128, 128, "pair")
    _bz_lin(by, "s_diffusion_to_s", 768, 384, "res")
    _bz_lin(by, "confidence_heads.to_plddt_logits", 384, 50, "res")
    _bz_lin(by, "confidence_heads.to_pae_logits", 128, 64, "res")


def _bz_check_cache(layer: Layer) -> None:
    for name, k, n, rows in _BZ_CACHE_SHAPES:
        found = [ln for ln in layer.pair_linears if ln.name == name]
        if len(found) != 1 or (found[0].k, found[0].n, found[0].rows) != (k, n, rows):
            raise ValueError(f"boltz-1 {name} shape is not [{k}, {n}] on {rows}")


def _bz_check_diff(layer: Layer) -> None:
    if {ln.name for ln in layer.pair_linears} != BOLTZ_DIFF_LINEARS:
        raise ValueError("boltz-1 diffusion conditioner linears are not the published set")
    by = _bz_by(layer)
    _bz_lin(by, "score_model.single_conditioner.single_embed", 839, 768, "res")
    _bz_lin(by, "score_model.single_conditioner.fourier_to_single", 256, 768, "res")
    _bz_transition(by, "score_model.single_conditioner.transitions.0", 768, 1536, "res")
    _bz_transition(by, "score_model.single_conditioner.transitions.1", 768, 1536, "res")
    _bz_lin(by, "score_model.s_to_a_linear.1", 768, 768, "res")
    _bz_lin(by, "score_model.atom_attention_decoder.a_to_q_trans", 768, 128, "atom")
    _bz_lin(by, "out_token_feat_update.transition_block.a_to_b", 768, 1536, "res")


# Protenix v0.5.0. Plain LayerNorm is OpenFoldLayerNorm / F.layer_norm (fp32 reduction).
# Transition is layernorm, then silu(a) * b. AtomAttentionEncoder uses F.relu / nn.ReLU.
# AdaLN and ConditionedTransitionBlock (diffusion SwiGLU + sigmoid) are not this recipe.
_PX_N_QUERIES = 32
_PX_N_KEYS = 128


def _px_leaves(prefix: str) -> str:
    return f"{prefix}." if prefix else ""


def _px_mha(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}.linear_{leaf}" for leaf in ("g", "k", "o", "q", "v"))


def _px_tri(prefix: str) -> tuple[str, ...]:
    p = _px_leaves(prefix)
    names: list[str] = []
    for side in ("end", "start"):
        names.append(f"{p}tri_att_{side}.linear")
        names.extend(_px_mha(f"{p}tri_att_{side}.mha"))
    for side in ("in", "out"):
        for leaf in ("a_g", "a_p", "b_g", "b_p", "g", "z"):
            names.append(f"{p}tri_mul_{side}.linear_{leaf}")
    return tuple(names)


def _px_trans(prefix: str) -> tuple[str, ...]:
    return (f"{prefix}.linear_no_bias", f"{prefix}.linear_no_bias_a", f"{prefix}.linear_no_bias_b")


def _px_dit_names(atom: bool) -> tuple[str, ...]:
    names = [f"attention_pair_bias.attention.linear_{leaf}" for leaf in ("g", "k", "o", "q", "v")]
    names += [
        "attention_pair_bias.layernorm_a.linear_nobias_s",
        "attention_pair_bias.layernorm_a.linear_s",
    ]
    if atom:
        names += [
            "attention_pair_bias.layernorm_kv.linear_nobias_s",
            "attention_pair_bias.layernorm_kv.linear_s",
        ]
    names += [
        "attention_pair_bias.linear_a_last",
        "attention_pair_bias.linear_nobias_z",
        "conditioned_transition_block.adaln.linear_nobias_s",
        "conditioned_transition_block.adaln.linear_s",
        "conditioned_transition_block.linear_nobias_a1",
        "conditioned_transition_block.linear_nobias_a2",
        "conditioned_transition_block.linear_nobias_b",
        "conditioned_transition_block.linear_s",
    ]
    return tuple(names)


_PX_PAIR_BODY = (
    "attention_pair_bias.attention.linear_g", "attention_pair_bias.attention.linear_k",
    "attention_pair_bias.attention.linear_o", "attention_pair_bias.attention.linear_q",
    "attention_pair_bias.attention.linear_v", "attention_pair_bias.linear_nobias_z",
) + _px_trans("pair_transition") + _px_trans("single_transition") + _px_tri("")
PROTENIX_PAIR_LINEARS = frozenset(_PX_PAIR_BODY)
PROTENIX_MSA_LINEARS = frozenset(
    ("msa_stack.msa_pair_weighted_averaging.linear_no_bias_mg",
     "msa_stack.msa_pair_weighted_averaging.linear_no_bias_mv",
     "msa_stack.msa_pair_weighted_averaging.linear_no_bias_out",
     "msa_stack.msa_pair_weighted_averaging.linear_no_bias_z")
    + _px_trans("msa_stack.transition_m")
    + ("outer_product_mean_msa.linear_1", "outer_product_mean_msa.linear_2", "outer_product_mean_msa.linear_out")
    + _px_trans("pair_stack.pair_transition")
    + _px_tri("pair_stack")
)
PROTENIX_MSA_LAST_LINEARS = frozenset(
    ("outer_product_mean_msa.linear_1", "outer_product_mean_msa.linear_2", "outer_product_mean_msa.linear_out")
    + _px_trans("pair_stack.pair_transition")
    + _px_tri("pair_stack")
)
PROTENIX_MSA_REST = frozenset({"linear_no_bias_m", "linear_no_bias_s"})
PROTENIX_CONF_LINEARS = frozenset({
    "linear_no_bias_d", "linear_no_bias_pae", "linear_no_bias_pde",
    "linear_no_bias_s1", "linear_no_bias_s2",
})
PROTENIX_EMBED_LINEARS = frozenset({
    "atom_attention_encoder.linear_no_bias_cl", "atom_attention_encoder.linear_no_bias_cm",
    "atom_attention_encoder.linear_no_bias_d", "atom_attention_encoder.linear_no_bias_f",
    "atom_attention_encoder.linear_no_bias_q", "atom_attention_encoder.linear_no_bias_ref_pos",
    "atom_attention_encoder.small_mlp.1", "atom_attention_encoder.small_mlp.3",
    "atom_attention_encoder.small_mlp.5",
})
PROTENIX_DIFF_LINEARS = frozenset({
    "atom_attention_decoder.linear_no_bias_a", "atom_attention_decoder.linear_no_bias_out",
    "atom_attention_encoder.linear_no_bias_cl", "atom_attention_encoder.linear_no_bias_cm",
    "atom_attention_encoder.linear_no_bias_d", "atom_attention_encoder.linear_no_bias_f",
    "atom_attention_encoder.linear_no_bias_q", "atom_attention_encoder.linear_no_bias_r",
    "atom_attention_encoder.linear_no_bias_ref_pos", "atom_attention_encoder.linear_no_bias_s",
    "atom_attention_encoder.linear_no_bias_z",
    "atom_attention_encoder.small_mlp.1", "atom_attention_encoder.small_mlp.3",
    "atom_attention_encoder.small_mlp.5",
    "diffusion_conditioning.linear_no_bias_n", "diffusion_conditioning.linear_no_bias_s",
    "diffusion_conditioning.linear_no_bias_z", "diffusion_conditioning.relpe.linear_no_bias",
    "diffusion_conditioning.transition_s1.linear_no_bias",
    "diffusion_conditioning.transition_s1.linear_no_bias_a",
    "diffusion_conditioning.transition_s1.linear_no_bias_b",
    "diffusion_conditioning.transition_s2.linear_no_bias",
    "diffusion_conditioning.transition_s2.linear_no_bias_a",
    "diffusion_conditioning.transition_s2.linear_no_bias_b",
    "diffusion_conditioning.transition_z1.linear_no_bias",
    "diffusion_conditioning.transition_z1.linear_no_bias_a",
    "diffusion_conditioning.transition_z1.linear_no_bias_b",
    "diffusion_conditioning.transition_z2.linear_no_bias",
    "diffusion_conditioning.transition_z2.linear_no_bias_a",
    "diffusion_conditioning.transition_z2.linear_no_bias_b",
    "linear_no_bias_s",
})
PROTENIX_DIT_LINEARS = frozenset(_px_dit_names(False))
PROTENIX_ATOM_DIT_LINEARS = frozenset(_px_dit_names(True))
_PX_PAIR_LN = {
    "attention_pair_bias.attention.linear_q": "attn_s_ln",
    "attention_pair_bias.linear_nobias_z": "attn_z_ln",
    "single_transition.linear_no_bias_a": "trans_s_ln",
    "pair_transition.linear_no_bias_a": "trans_z_ln",
    "tri_att_start.linear": "tri_att_start_ln",
    "tri_att_end.linear": "tri_att_end_ln",
    "tri_mul_in.linear_a_p": "tri_mul_in_ln_in",
    "tri_mul_out.linear_a_p": "tri_mul_out_ln_in",
    "tri_mul_in.linear_z": "tri_mul_in_ln_out",
    "tri_mul_out.linear_z": "tri_mul_out_ln_out",
}
_PX_PAIR_SILU = {
    "single_transition.linear_no_bias_a": "trans_s_silu",
    "pair_transition.linear_no_bias_a": "trans_z_silu",
}
_PX_PAIR_MUL = {
    "single_transition.linear_no_bias_b": "trans_s_mul",
    "pair_transition.linear_no_bias_b": "trans_z_mul",
}
_PX_MSA_LN = {
    "msa_stack.transition_m.linear_no_bias_a": "msa_trans_ln",
    "msa_stack.msa_pair_weighted_averaging.linear_no_bias_mv": "pwa_m_ln",
    "msa_stack.msa_pair_weighted_averaging.linear_no_bias_z": "pwa_z_ln",
    "outer_product_mean_msa.linear_1": "opm_ln",
    "pair_stack.pair_transition.linear_no_bias_a": "pair_trans_ln",
    "pair_stack.tri_att_start.linear": "tri_att_start_ln",
    "pair_stack.tri_att_end.linear": "tri_att_end_ln",
    "pair_stack.tri_mul_in.linear_a_p": "tri_mul_in_ln_in",
    "pair_stack.tri_mul_out.linear_a_p": "tri_mul_out_ln_in",
    "pair_stack.tri_mul_in.linear_z": "tri_mul_in_ln_out",
    "pair_stack.tri_mul_out.linear_z": "tri_mul_out_ln_out",
}
_PX_MSA_SILU = {
    "msa_stack.transition_m.linear_no_bias_a": "msa_trans_silu",
    "pair_stack.pair_transition.linear_no_bias_a": "pair_trans_silu",
}
_PX_MSA_MUL = {
    "msa_stack.transition_m.linear_no_bias_b": "msa_trans_mul",
    "pair_stack.pair_transition.linear_no_bias_b": "pair_trans_mul",
}
_PX_DIFF_LN = {
    "diffusion_conditioning.transition_z1.linear_no_bias_a": "z1_ln",
    "diffusion_conditioning.transition_z2.linear_no_bias_a": "z2_ln",
    "diffusion_conditioning.transition_s1.linear_no_bias_a": "s1_ln",
    "diffusion_conditioning.transition_s2.linear_no_bias_a": "s2_ln",
}
_PX_DIFF_SILU = {
    "diffusion_conditioning.transition_z1.linear_no_bias_a": "z1_silu",
    "diffusion_conditioning.transition_z2.linear_no_bias_a": "z2_silu",
    "diffusion_conditioning.transition_s1.linear_no_bias_a": "s1_silu",
    "diffusion_conditioning.transition_s2.linear_no_bias_a": "s2_silu",
}
_PX_DIFF_MUL = {
    "diffusion_conditioning.transition_z1.linear_no_bias_b": "z1_mul",
    "diffusion_conditioning.transition_z2.linear_no_bias_b": "z2_mul",
    "diffusion_conditioning.transition_s1.linear_no_bias_b": "s1_mul",
    "diffusion_conditioning.transition_s2.linear_no_bias_b": "s2_mul",
}
_PX_MLP_RELU = {
    "atom_attention_encoder.small_mlp.1": "mlp_relu_1",
    "atom_attention_encoder.small_mlp.3": "mlp_relu_2",
    "atom_attention_encoder.small_mlp.5": "mlp_relu_3",
}


def _px_ln(name: str, rows: int, width: int, li: int, rep: float) -> Op:
    return _esm_ln(name, rows, width, li, rep, "protenix")


def _px_silu(name: str, n: int, li: int, rep: float, af: str) -> Op:
    """Transition SiLU is x * sigmoid(x). Four element-ops, two of them on the SFU."""
    if n <= 0:
        raise ValueError(f"protenix {name} has no elements")
    return Op(name, "vector", li, vec=float(n * 4), sfu=float(n * 2), a_fmt=af, replicated=rep)


def _px_mul(name: str, n: int, li: int, rep: float, af: str) -> Op:
    """silu(linear_a) * linear_b. One multiply, no transcendental."""
    if n <= 0:
        raise ValueError(f"protenix {name} has no elements")
    return Op(name, "vector", li, vec=float(n), a_fmt=af, replicated=rep)


def _px_window_rows(atoms: int, width: int) -> int:
    """AtomAttentionEncoder dense trunks. Query width is 32, key width is 128."""
    if atoms <= 0:
        return 0
    trunks = (atoms + _PX_N_QUERIES - 1) // _PX_N_QUERIES
    return trunks * width


def _px_by(layer: Layer) -> dict:
    return {ln.name: ln for ln in layer.pair_linears}


def _px_lin(by: dict, name: str, k: int, n: int, rows: str) -> None:
    lin = by.get(name)
    if lin is None or (lin.k, lin.n, lin.rows) != (k, n, rows):
        raise ValueError(f"protenix {name} shape is not [{k}, {n}] on {rows}")


def _px_transition(by: dict, prefix: str, k: int, hidden: int, rows: str) -> None:
    _px_lin(by, f"{prefix}.linear_no_bias_a", k, hidden, rows)
    _px_lin(by, f"{prefix}.linear_no_bias_b", k, hidden, rows)
    _px_lin(by, f"{prefix}.linear_no_bias", hidden, k, rows)


def _px_triangle(by: dict, prefix: str) -> None:
    p = _px_leaves(prefix)
    for side in ("in", "out"):
        _px_lin(by, f"{p}tri_mul_{side}.linear_a_p", 128, 128, "pair")
        hidden = by[f"{p}tri_mul_{side}.linear_z"]
        if (hidden.k, hidden.n, hidden.rows) != (128, 128, "pair"):
            raise ValueError(f"protenix {p}tri_mul_{side}.linear_z shape is not the published hidden")
    for side in ("start", "end"):
        _px_lin(by, f"{p}tri_att_{side}.linear", 128, 4, "pair")


def _px_check_pair(layer: Layer) -> None:
    if {ln.name for ln in layer.pair_linears} != PROTENIX_PAIR_LINEARS:
        raise ValueError("protenix pairformer linears are not the published set")
    by = _px_by(layer)
    _px_lin(by, "attention_pair_bias.attention.linear_q", 384, 384, "res")
    _px_lin(by, "attention_pair_bias.linear_nobias_z", 128, 16, "pair")
    _px_transition(by, "single_transition", 384, 1536, "res")
    _px_transition(by, "pair_transition", 128, 512, "pair")
    _px_triangle(by, "")


def _px_check_msa(layer: Layer, last: bool) -> None:
    want = PROTENIX_MSA_LAST_LINEARS if last else PROTENIX_MSA_LINEARS
    if {ln.name for ln in layer.pair_linears} != want:
        raise ValueError("protenix msa linears are not the published set")
    by = _px_by(layer)
    if not last:
        _px_transition(by, "msa_stack.transition_m", 64, 256, "msa")
        _px_lin(by, "msa_stack.msa_pair_weighted_averaging.linear_no_bias_mv", 64, 64, "msa")
        _px_lin(by, "msa_stack.msa_pair_weighted_averaging.linear_no_bias_z", 128, 8, "pair")
    _px_lin(by, "outer_product_mean_msa.linear_1", 64, 32, "msa")
    _px_transition(by, "pair_stack.pair_transition", 128, 512, "pair")
    _px_triangle(by, "pair_stack")


def _px_check_recycle(layer: Layer) -> None:
    if len(layer.pair_linears) != 1:
        raise ValueError("protenix recycle projection is not one linear")
    lin = layer.pair_linears[0]
    if (lin.k, lin.n, lin.rows) not in ((384, 384, "res"), (128, 128, "pair")):
        raise ValueError("protenix recycle projection shape is not the published pair")


def _px_check_embed(layer: Layer) -> None:
    if {ln.name for ln in layer.pair_linears} != PROTENIX_EMBED_LINEARS:
        raise ValueError("protenix input embedder linears are not the published set")
    by = _px_by(layer)
    _px_lin(by, "atom_attention_encoder.linear_no_bias_cl", 128, 16, "atom")
    _px_lin(by, "atom_attention_encoder.linear_no_bias_cm", 128, 16, "atom")
    _px_lin(by, "atom_attention_encoder.linear_no_bias_q", 128, 384, "atom")
    _px_lin(by, "atom_attention_encoder.small_mlp.1", 16, 16, "apair")
    _px_lin(by, "atom_attention_encoder.small_mlp.3", 16, 16, "apair")
    _px_lin(by, "atom_attention_encoder.small_mlp.5", 16, 16, "apair")


def _px_check_diff(layer: Layer) -> None:
    if {ln.name for ln in layer.pair_linears} != PROTENIX_DIFF_LINEARS:
        raise ValueError("protenix diffusion linears are not the published set")
    by = _px_by(layer)
    _px_lin(by, "diffusion_conditioning.linear_no_bias_z", 256, 128, "pair")
    _px_lin(by, "diffusion_conditioning.linear_no_bias_s", 833, 384, "res")
    _px_lin(by, "diffusion_conditioning.linear_no_bias_n", 256, 384, "res")
    _px_transition(by, "diffusion_conditioning.transition_z1", 128, 256, "pair")
    _px_transition(by, "diffusion_conditioning.transition_z2", 128, 256, "pair")
    _px_transition(by, "diffusion_conditioning.transition_s1", 384, 768, "res")
    _px_transition(by, "diffusion_conditioning.transition_s2", 384, 768, "res")
    _px_lin(by, "atom_attention_encoder.linear_no_bias_s", 384, 128, "atom")
    _px_lin(by, "atom_attention_encoder.linear_no_bias_z", 128, 16, "spair")
    _px_lin(by, "atom_attention_encoder.linear_no_bias_cl", 128, 16, "atom")
    _px_lin(by, "atom_attention_encoder.linear_no_bias_cm", 128, 16, "atom")
    _px_lin(by, "atom_attention_encoder.linear_no_bias_q", 128, 768, "atom")
    _px_lin(by, "atom_attention_encoder.small_mlp.1", 16, 16, "apair")
    _px_lin(by, "linear_no_bias_s", 384, 768, "res")
    _px_lin(by, "atom_attention_decoder.linear_no_bias_a", 768, 128, "atom")
    _px_lin(by, "atom_attention_decoder.linear_no_bias_out", 128, 3, "atom")


def _px_check_conf(layer: Layer) -> None:
    if {ln.name for ln in layer.pair_linears} != PROTENIX_CONF_LINEARS:
        raise ValueError("protenix confidence linears are not the published set")
    by = _px_by(layer)
    _px_lin(by, "linear_no_bias_pae", 128, 64, "pair")
    _px_lin(by, "linear_no_bias_pde", 128, 64, "pair")
    _px_lin(by, "linear_no_bias_s1", 449, 128, "res")


def _px_check_dit(layer: Layer) -> None:
    names = {ln.name for ln in layer.pair_linears}
    by = _px_by(layer)
    if names == PROTENIX_DIT_LINEARS:
        _px_lin(by, "attention_pair_bias.linear_nobias_z", 128, 16, "spair")
        _px_lin(by, "attention_pair_bias.attention.linear_q", 768, 768, "res")
        return
    if names == PROTENIX_ATOM_DIT_LINEARS:
        _px_lin(by, "attention_pair_bias.linear_nobias_z", 16, 4, "apair")
        _px_lin(by, "attention_pair_bias.attention.linear_q", 128, 128, "atom")
        return
    raise ValueError("protenix diffusion-transformer linears are not the published set")


def _px_res_rows(pd: PairDims, b: int, s: int, dap: int) -> int:
    return _pair_rows("res", pd, b, s, 1, dap)


def _px_before(kind: str, name: str, rows: str, m: int, k: int, n: int, li: int, rep: float, af: str,
               pd: PairDims, b: int, s: int, dap: int, rep_tok: float) -> list[Op]:
    """LayerNorm / ReLU that the published module runs before this linear."""
    if kind == "px_pair":
        ln_name = _PX_PAIR_LN.get(name)
        return [_px_ln(ln_name, m, k, li, rep)] if ln_name else []
    if kind in ("px_msa", "px_msa_last"):
        ln_name = _PX_MSA_LN.get(name)
        return [_px_ln(ln_name, m, k, li, rep)] if ln_name else []
    if kind == "px_recycle":
        if (k, n, rows) == (384, 384, "res"):
            return [_px_ln("s_ln", m, k, li, rep)]
        return [_px_ln("z_ln", m, k, li, rep)]
    if kind == "px_dit":
        # Plain layernorm_z on the pair bias. The query norm is AdaLN and is not billed.
        if name == "attention_pair_bias.linear_nobias_z":
            return [_px_ln("attn_z_ln", m, k, li, rep)]
        return []
    if kind == "px_embed":
        return _px_atom_relu_before(name, m, k, li, rep, af, pd, b, s, rep_tok)
    if kind == "px_diff":
        if name == "diffusion_conditioning.linear_no_bias_z":
            return [_px_ln("cond_z_ln", m, k, li, rep)]
        if name == "diffusion_conditioning.linear_no_bias_s":
            # Cat(s_trunk, s_inputs) is normalized once per step, before the sample axis exists.
            tok = _px_res_rows(pd, b, 1, dap)
            return [_px_ln("cond_s_ln", tok, k, li, float(dap))] if tok > 0 else []
        if name == "diffusion_conditioning.linear_no_bias_n":
            # Fourier embedding is one vector per sample. The GEMM stays on token rows.
            return [_px_ln("noise_ln", b * s, k, li, rep_tok)] if s > 0 else []
        if name == "atom_attention_encoder.linear_no_bias_s":
            tok = _px_res_rows(pd, b, s, dap)
            return [_px_ln("enc_s_ln", tok, k, li, rep_tok)] if tok > 0 else []
        if name == "atom_attention_encoder.linear_no_bias_z":
            return [_px_ln("enc_z_ln", m, k, li, rep)]
        if name == "linear_no_bias_s":
            return [_px_ln("tok_s_ln", m, k, li, rep)]
        if name == "atom_attention_decoder.linear_no_bias_a":
            # layernorm_a reads the token activation. The decoder GEMM is on atom rows.
            tok = _px_res_rows(pd, b, s, dap)
            return [_px_ln("a_ln", tok, k, li, rep_tok)] if tok > 0 else []
        if name == "atom_attention_decoder.linear_no_bias_out":
            return [_px_ln("q_ln", m, k, li, rep)]
        ln_name = _PX_DIFF_LN.get(name)
        out = [_px_ln(ln_name, m, k, li, rep)] if ln_name else []
        out += _px_atom_relu_before(name, m, k, li, rep, af, pd, b, s, rep_tok)
        return out
    if kind == "px_conf":
        if name == "linear_no_bias_s1":
            # input_strunk_ln runs once, before the sample loop.
            tok = _px_res_rows(pd, b, 1, dap)
            return [_px_ln("input_s_ln", tok, 384, li, float(dap))] if tok > 0 else []
        if name == "linear_no_bias_pae":
            return [_px_ln("pae_ln", m, k, li, rep)]
        if name == "linear_no_bias_pde":
            return [_px_ln("pde_ln", m, k, li, rep)]
        return []
    raise ValueError(f"protenix vector kind {kind} is not published")


def _px_atom_relu_before(name: str, m: int, k: int, li: int, rep: float, af: str, pd: PairDims, b: int,
                         s: int, rep_tok: float) -> list[Op]:
    """F.relu on the windowed atom single, and nn.ReLU inside small_mlp."""
    if name == "atom_attention_encoder.linear_no_bias_cl":
        n_el = b * s * _px_window_rows(pd.atoms, _PX_N_QUERIES) * k
        return [_esm_relu("cl_relu", n_el, li, rep_tok, af, "protenix")] if n_el > 0 else []
    if name == "atom_attention_encoder.linear_no_bias_cm":
        n_el = b * s * _px_window_rows(pd.atoms, _PX_N_KEYS) * k
        return [_esm_relu("ck_relu", n_el, li, rep_tok, af, "protenix")] if n_el > 0 else []
    relu_name = _PX_MLP_RELU.get(name)
    if relu_name:
        return [_esm_relu(relu_name, m * k, li, rep, af, "protenix")]
    return []


def _px_after(kind: str, name: str, m: int, n: int, li: int, rep: float, af: str, pd: PairDims, b: int,
              s: int, ps: int, dap: int, rep_tok: float) -> list[Op]:
    """SiLU, the transition multiply, and the post-linear ReLU."""
    if kind == "px_pair":
        out = []
        silu_name = _PX_PAIR_SILU.get(name)
        if silu_name:
            out.append(_px_silu(silu_name, m * n, li, rep, af))
        mul_name = _PX_PAIR_MUL.get(name)
        if mul_name:
            out.append(_px_mul(mul_name, m * n, li, rep, af))
        return out
    if kind in ("px_msa", "px_msa_last"):
        out = []
        silu_name = _PX_MSA_SILU.get(name)
        if silu_name:
            out.append(_px_silu(silu_name, m * n, li, rep, af))
        mul_name = _PX_MSA_MUL.get(name)
        if mul_name:
            out.append(_px_mul(mul_name, m * n, li, rep, af))
        return out
    if kind in ("px_embed", "px_diff") and name == "atom_attention_encoder.linear_no_bias_q":
        # F.relu(linear_q(q_l)) on the atom rows, width c_token.
        return [_esm_relu("q_relu", m * n, li, rep, af, "protenix")]
    if kind == "px_diff":
        out = []
        silu_name = _PX_DIFF_SILU.get(name)
        if silu_name:
            out.append(_px_silu(silu_name, m * n, li, rep, af))
        mul_name = _PX_DIFF_MUL.get(name)
        if mul_name:
            out.append(_px_mul(mul_name, m * n, li, rep, af))
        return out
    if kind == "px_conf" and name == "linear_no_bias_pde":
        # plddt_ln / resolved_ln read the broadcast atom single, once per sample.
        atoms = _pair_rows("atom", pd, b, s, ps, dap)
        if atoms <= 0:
            return []
        return [_px_ln("plddt_ln", atoms, 384, li, rep_tok),
                _px_ln("resolved_ln", atoms, 384, li, rep_tok)]
    if kind in ("px_dit", "px_recycle", "px_conf", "px_embed", "px_diff"):
        return []
    raise ValueError(f"protenix vector kind {kind} is not published")


def _of_who(model: ModelSpec) -> str:
    if model.id in ("openfold", "alphafold2"):
        return model.id
    return "openfold"


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
    kind = L.vec_kind
    who = _of_who(model)
    embed_by: dict[str, object] = {}
    if kind == "esm_proj":
        # modeling_esmfold.py EsmForProteinFolding.esm_s_mlp: LayerNorm(language width),
        # Linear to the trunk width, nn.ReLU, Linear. Rows are residues (bos/eos already stripped).
        if [l.name for l in L.pair_linears] != ["mlp.1", "mlp.3"]:
            raise ValueError("esmfold projection GEMMs are not mlp.1 then mlp.3")
        if L.vec_width <= 0 or L.pair_linears[0].k != L.vec_width:
            raise ValueError("esmfold projection LayerNorm width is not the language width")
    elif kind == "esm_trunk":
        if {l.name for l in L.pair_linears} != ESMFOLD_TRUNK_LINEARS:
            raise ValueError("esmfold trunk block linears are not the published set")
    elif kind == "esm_sm":
        if {l.name for l in L.pair_linears} != ESMFOLD_SM_LINEARS:
            raise ValueError("esmfold structure-module loop linears are not the published set")
    elif kind == "esm_sm_pre":
        if [l.name for l in L.pair_linears] != ["linear_in"]:
            raise ValueError("esmfold structure-module preamble is not linear_in")
        if L.vec_width <= 0:
            raise ValueError("esmfold structure pair LayerNorm width is missing")
    elif kind == "esm_recycle":
        if [l.name for l in L.pair_linears] != ["trunk2sm_s", "trunk2sm_z"]:
            raise ValueError("esmfold recycle projections are not trunk2sm_s then trunk2sm_z")
        by_name = {l.name: l for l in L.pair_linears}
        for nm, oname in (("trunk2sm_s", "recycle_s_ln"), ("trunk2sm_z", "recycle_z_ln")):
            lin = by_name[nm]
            rows = _pair_rows(lin.rows, pd, b, s, ps, dap)
            if rows <= 0:
                continue
            rrep = 1 if lin.rows in _DAP_SPLIT else rep_tok
            ops.append(_esm_ln(oname, rows, lin.k, li, rrep))
    elif kind == "esm_lddt":
        if [l.name for l in L.pair_linears] != ["1", "2", "3"]:
            raise ValueError("esmfold lddt head is not three linears")
    elif kind == "of_block":
        normed = {_strip_core(l.name) for l in L.pair_linears}
        if normed not in (OPENFOLD_EVO_LINEARS, OPENFOLD_EXTRA_LINEARS, OPENFOLD_TMPL_LINEARS):
            raise ValueError(f"{who} block linears are not the published set")
        by_norm = {_strip_core(l.name): l for l in L.pair_linears}
        for side in ("in", "out"):
            hidden = by_norm[f"tri_mul_{side}.linear_z"]
            proj = by_norm[f"tri_mul_{side}.linear_a_p"]
            if hidden.k != proj.n:
                raise ValueError(f"{who} triangle multiply hidden width is not the projection width")
    elif kind == "of_sm":
        names = {l.name for l in L.pair_linears}
        if names not in (ESMFOLD_SM_LINEARS, ALPHAFOLD2_SM_LINEARS):
            raise ValueError(f"{who} structure-module loop linears are not the published set")
    elif kind == "of_sm_pre":
        if [l.name for l in L.pair_linears] not in (["linear_in"], ["initial_projection"]):
            raise ValueError(f"{who} structure-module preamble is not the published projection")
        pre_lin = L.pair_linears[0]
        if pre_lin.k != pre_lin.n or pre_lin.rows != "res" or L.vec_width <= 0:
            raise ValueError(f"{who} structure-module preamble shape is not the published projection")
    elif kind == "of_heads":
        if {l.name for l in L.pair_linears} not in (OPENFOLD_HEAD_LINEARS, ALPHAFOLD2_HEAD_LINEARS):
            raise ValueError(f"{who} heads linears are not the published set")
    elif kind == "of_embed":
        embed_by = {l.name: l for l in L.pair_linears}
        if set(embed_by) != OPENFOLD_EMBED_LINEARS:
            raise ValueError(f"{who} embed linears are not the published set")
    elif kind == "bz_pair":
        _bz_check_pair(L)
    elif kind == "bz_msa":
        _bz_check_msa(L)
    elif kind == "bz_recycle":
        _bz_check_recycle(L)
    elif kind == "bz_conf":
        _bz_check_conf(L)
    elif kind == "bz_cache":
        _bz_check_cache(L)
    elif kind == "bz_diff":
        _bz_check_diff(L)
    elif kind == "px_pair":
        _px_check_pair(L)
    elif kind == "px_msa":
        _px_check_msa(L, False)
    elif kind == "px_msa_last":
        _px_check_msa(L, True)
    elif kind == "px_recycle":
        _px_check_recycle(L)
    elif kind == "px_embed":
        _px_check_embed(L)
    elif kind == "px_diff":
        _px_check_diff(L)
    elif kind == "px_conf":
        _px_check_conf(L)
    elif kind == "px_dit":
        _px_check_dit(L)
    state_copies = _esmfold_state_copies(model) if kind == "esm_lddt" else 1
    for l in L.pair_linears:
        m = _pair_rows(l.rows, pd, b, s, ps, dap)
        if m <= 0:
            continue
        if kind == "esm_lddt":
            # EsmFold.lddt_head consumes structure["states"], stacked over num_blocks.
            m *= state_copies
        rep = 1 if l.rows in _DAP_SPLIT else rep_tok
        if kind == "esm_proj" and l.name == "mlp.1":
            ops.append(_esm_ln("esm_s_ln", m, l.k, li, rep))
        elif kind == "esm_trunk":
            ln_name = _ESM_TRUNK_LN_BEFORE.get(l.name)
            if ln_name:
                ops.append(_esm_ln(ln_name, m, l.k, li, rep))
        elif kind == "esm_sm":
            relu_name = _ESM_SM_RELU_BEFORE.get(l.name)
            if relu_name:
                ops.append(_esm_relu(relu_name, m * l.k, li, rep, model.act_fmt))
        elif kind == "esm_sm_pre" and l.name == "linear_in":
            # StructureModule._forward_monomer: layer_norm_s, layer_norm_z, then linear_in, once.
            ops.append(_esm_ln("sm_s_ln", m, l.k, li, rep))
            pz = _pair_rows("pair", pd, b, s, ps, dap)
            if pz > 0:
                ops.append(_esm_ln("sm_z_ln", pz, L.vec_width, li, 1))
        elif kind == "esm_lddt" and l.name == "1":
            # nn.Sequential: LayerNorm, Linear, Linear, Linear. No ReLU.
            ops.append(_esm_ln("lddt_ln", m, l.k, li, rep))
        elif kind == "of_block":
            ln_name = _OF_BLOCK_LN_BEFORE.get(_strip_core(l.name))
            if ln_name:
                ops.append(_esm_ln(ln_name, m, l.k, li, rep, who))
        elif kind == "of_sm":
            relu_name = _OF_SM_RELU_BEFORE.get(l.name)
            if relu_name:
                ops.append(_esm_relu(relu_name, m * l.k, li, rep, model.act_fmt, who))
        elif kind == "of_sm_pre":
            # StructureModule: layer_norm_s, layer_norm_z, then the input projection, once per recycle.
            ops.append(_esm_ln("sm_s_ln", m, l.k, li, rep, who))
            pz = _pair_rows("pair", pd, b, s, ps, dap)
            if pz > 0:
                ops.append(_esm_ln("sm_z_ln", pz, L.vec_width, li, 1, who))
        elif kind == "of_heads":
            ln_name = _OF_PLDDT_LN.get(l.name)
            if ln_name:
                # Aux heads read the final single representation, not the stacked IPA states.
                ops.append(_esm_ln(ln_name, m, l.k, li, rep, who))
        elif kind == "of_embed" and l.name == "recycling_embedder.linear":
            # First MSA row (residue grid) then the pair, including the zero-initialised first pass.
            evo = embed_by["evoformer.linear"]
            rm = _pair_rows("res", pd, b, s, ps, dap)
            if rm > 0:
                ops.append(_esm_ln("recycle_m_ln", rm, evo.k, li, rep_tok, who))
            ops.append(_esm_ln("recycle_z_ln", m, l.n, li, rep, who))
        elif kind == "of_embed" and l.name == "template_pointwise_att.mha.linear_k":
            # Template pair stack's final LayerNorm, once per recycle, before pointwise attention.
            width = embed_by["template_pair_embedder.linear"].n
            ops.append(_esm_ln("tmpl_out_ln", m, width, li, rep, who))
        elif kind == "bz_pair":
            ln_name = _BZ_PAIR_LN.get(l.name)
            if ln_name:
                ops.append(_esm_ln(ln_name, m, l.k, li, rep, "boltz-1"))
        elif kind == "bz_msa":
            ln_name = _BZ_MSA_LN.get(l.name)
            if ln_name:
                ops.append(_esm_ln(ln_name, m, l.k, li, rep, "boltz-1"))
        elif kind == "bz_recycle":
            if (l.k, l.n, l.rows) == (384, 384, "res"):
                ops.append(_esm_ln("s_ln", m, l.k, li, rep, "boltz-1"))
            else:
                ops.append(_esm_ln("z_ln", m, l.k, li, rep, "boltz-1"))
        elif kind == "bz_conf" and l.name == "s_recycle":
            # ConfidenceModule.imitate_trunk norms s before repeat_interleave(multiplicity).
            rows = _pair_rows("res", pd, b, 1, 1, dap)
            ops.append(_esm_ln("conf_s_ln", rows, l.k, li, rep_tok, "boltz-1"))
        elif kind == "bz_conf" and l.name == "z_recycle":
            rows = _pair_rows("pair", pd, b, 1, 1, dap)
            ops.append(_esm_ln("conf_z_ln", rows, l.k, li, 1, "boltz-1"))
        elif kind == "bz_conf" and l.name == "s_diffusion_to_s":
            ops.append(_esm_ln("s_diff_ln", m, l.k, li, rep, "boltz-1"))
        elif kind == "bz_conf" and l.name == "confidence_heads.to_plddt_logits":
            ops.append(_esm_ln("final_s_ln", m, l.k, li, rep, "boltz-1"))
        elif kind == "bz_conf" and l.name == "confidence_heads.to_pae_logits":
            # final_z_norm runs on the pair grid after the sample expansion. The head GEMM stays on residue rows.
            rows = _pair_rows("pair", pd, b, s, ps, dap)
            ops.append(_esm_ln("final_z_ln", rows, 128, li, 1, "boltz-1"))
        elif kind == "bz_cache":
            ln_name = _BZ_CACHE_LN.get(l.name)
            if ln_name:
                ops.append(_esm_ln(ln_name, m, l.k, li, rep, "boltz-1"))
        elif kind == "bz_diff" and l.name == "score_model.single_conditioner.fourier_to_single":
            # FourierEmbedding is one vector per sample, not one per token. The GEMM row count stays as released.
            ops.append(_esm_ln("fourier_ln", b * s, l.k, li, rep_tok, "boltz-1"))
        elif kind == "bz_diff" and l.name == "score_model.atom_attention_decoder.a_to_q_trans":
            rows = _pair_rows("res", pd, b, s, ps, dap)
            ops.append(_esm_ln("a_ln", rows, 768, li, rep_tok, "boltz-1"))
        elif kind == "bz_diff" and l.name == "out_token_feat_update.transition_block.a_to_b":
            ops.append(_esm_ln("tok_upd_ln", m, l.k, li, rep, "boltz-1"))
            ops.append(_esm_ln("tok_fourier_ln", b * s, 256, li, rep_tok, "boltz-1"))
        elif kind == "bz_diff":
            ln_name = _BZ_DIFF_LN.get(l.name)
            if ln_name:
                ops.append(_esm_ln(ln_name, m, l.k, li, rep, "boltz-1"))
        elif kind.startswith("px_"):
            ops.extend(_px_before(kind, l.name, l.rows, m, l.k, l.n, li, rep, model.act_fmt,
                                  pd, b, s, dap, rep_tok))
        o = gemm(model, f"{L.stack}.{l.name}", li, m, l.k, l.n, l.role or "pair", stream=True, replicated=rep)
        ops.append(o)
        if kind == "esm_proj" and l.name == "mlp.1":
            # nn.ReLU is max(x, 0). One clip, no transcendental, no upcast.
            ops.append(_esm_relu("esm_s_relu", m * l.n, li, rep, model.act_fmt))
        elif kind == "esm_trunk":
            relu_name = _ESM_TRUNK_RELU_AFTER.get(l.name)
            if relu_name:
                ops.append(_esm_relu(relu_name, m * l.n, li, rep, model.act_fmt))
        elif kind == "esm_sm":
            relu_name = _ESM_SM_RELU_AFTER.get(l.name)
            if relu_name:
                ops.append(_esm_relu(relu_name, m * l.n, li, rep, model.act_fmt))
            ln_name = _ESM_SM_LN_AFTER.get(l.name)
            if ln_name:
                ops.append(_esm_ln(ln_name, m, l.n, li, rep))
        elif kind == "of_block":
            bare = _strip_core(l.name)
            relu_name = _OF_BLOCK_RELU_AFTER.get(bare)
            if relu_name:
                ops.append(_esm_relu(relu_name, m * l.n, li, rep, model.act_fmt, who))
            if model.id == "openfold":
                mul_name = _OF_BLOCK_MASK_AFTER.get(bare)
                if mul_name:
                    ops.append(_of_mask(mul_name, m * l.n, li, rep, model.act_fmt))
        elif kind == "of_sm":
            relu_name = _OF_SM_RELU_AFTER.get(l.name)
            if relu_name:
                ops.append(_esm_relu(relu_name, m * l.n, li, rep, model.act_fmt, who))
            ln_name = _OF_SM_LN_AFTER.get(l.name)
            if ln_name:
                ops.append(_esm_ln(ln_name, m, l.n, li, rep, who))
        elif kind == "of_heads":
            relu_name = _OF_PLDDT_RELU.get(l.name)
            if relu_name:
                ops.append(_esm_relu(relu_name, m * l.n, li, rep, model.act_fmt, who))
        elif kind == "of_embed" and l.name == "template_angle_embedder.linear_1":
            ops.append(_esm_relu("tmpl_angle_relu", m * l.n, li, rep, model.act_fmt, who))
        elif kind == "of_embed" and l.name == "template_pointwise_att.mha.linear_o" \
                and model.id in ("openfold", "alphafold2"):
            # Published forward always multiplies the attention output. Pair rows, width c_z.
            ops.append(_of_mask("tmpl_out_mask", m * l.n, li, rep, model.act_fmt))
        elif kind == "bz_pair":
            silu_name = _BZ_PAIR_SILU.get(l.name)
            if silu_name:
                ops.append(_bz_silu(silu_name, m * l.n, li, rep, model.act_fmt))
            mul_name = _BZ_PAIR_MUL.get(l.name)
            if mul_name:
                ops.append(_bz_mul(mul_name, m * l.n, li, rep, model.act_fmt))
        elif kind == "bz_msa":
            silu_name = _BZ_MSA_SILU.get(l.name)
            if silu_name:
                ops.append(_bz_silu(silu_name, m * l.n, li, rep, model.act_fmt))
            mul_name = _BZ_MSA_MUL.get(l.name)
            if mul_name:
                ops.append(_bz_mul(mul_name, m * l.n, li, rep, model.act_fmt))
        elif kind == "bz_cache":
            silu_name = _BZ_CACHE_SILU.get(l.name)
            if silu_name:
                ops.append(_bz_silu(silu_name, m * l.n, li, rep, model.act_fmt))
            mul_name = _BZ_CACHE_MUL.get(l.name)
            if mul_name:
                ops.append(_bz_mul(mul_name, m * l.n, li, rep, model.act_fmt))
        elif kind == "bz_diff":
            silu_name = _BZ_DIFF_SILU.get(l.name)
            if silu_name:
                ops.append(_bz_silu(silu_name, m * l.n, li, rep, model.act_fmt))
            mul_name = _BZ_DIFF_MUL.get(l.name)
            if mul_name:
                ops.append(_bz_mul(mul_name, m * l.n, li, rep, model.act_fmt))
        elif kind.startswith("px_"):
            ops.extend(_px_after(kind, l.name, m, l.n, li, rep, model.act_fmt, pd, b, s, ps, dap, rep_tok))
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
    if L.vec_kind == "whisper":
        # The decoder is autoregressive (prefill / decode). A full-sequence
        # phase would bill SiLU after both linears and drop the fp32 mark.
        raise ValueError("whisper decoder vectors are the autoregressive path")
    if not (L.attn_linears or L.ffn_linears or L.cross_linears or L.cross is not None or L.core.kind != "none"):
        # 0.75: a parameter-holder layer (misc_params only — e.g. ESMFold's esm_s combine / esm I/O norms, read with
        # the hot set by memplan) has no block: it emitted a phantom pre-norm / FFN-norm / residual (+ TP all-reduces)
        return []
    # 0.439: an ESMFold language layer's vectors use the ESM-2 width, not the trunk width
    # stored on ModelSpec.hidden. Every other full-forward layer keeps model.hidden.
    h = L.vec_width if L.vec_kind == "esm2" and L.vec_width else model.hidden
    ab = _fmt(model.act_fmt).bytes
    rb = red_bytes(model)
    tp, sp = sh.tp, sh.sp
    b = _cdiv(ph.batch, sh.dp)
    t = b * _cdiv(ph.q, sp)                    # tokens through this rank's GEMMs
    if ph.mreuse and L.core.span == "mode":    # 0.255: only the image mode block runs on a reused step
        t = b * _cdiv(_video_rows(model, ph), sp)
    mod = 4 if model.adaln else 0              # AdaLN: x·(1+scale)+shift, gate (element-ops per element)
    if L.vec_kind == "esm2":
        # nn.LayerNorm before attention. fp32 reduction, one rsqrt per row. Affine is already in misc.
        if model.adaln or L.ffn.kind != "dense" or L.ffn.gated:
            raise ValueError("esm2 vector billing is pre-LN dense erf-GELU")
        if L.vec_width and L.attn_linears and L.vec_width != L.attn_linears[0].k:
            raise ValueError("esm2 vector width is not the layer width")
        ops = [Op("attn_norm", "vector", li, vec=t * h * 4, sfu=float(t), a_fmt="fp32")]
    elif L.vec_kind == "generanno":
        # GenerannoRMSNorm before attention. fp32 pow(2).mean, one rsqrt, times weight.
        if model.adaln or L.ffn.kind != "dense" or not L.ffn.gated:
            raise ValueError("generanno vector billing is RMSNorm and SwiGLU")
        ops = [Op("attn_rms", "vector", li, vec=t * h * 4, sfu=float(t), a_fmt="fp32")]
    else:
        ops = [Op("attn_norm", "vector", li, vec=t * h * (4 + mod))]

    def lin(prefix: str, l: Linear, role: str) -> Op:
        k, n = _lin_local(l, tp)
        return gemm(model, f"{prefix}.{l.name}", li, _rows(model, l, ph, b, sp), k, n, role, stream=True,
                    replicated=_repl_full(l, tp, sp))

    ops += [lin("attn", l, l.role or "attn") for l in L.attn_linears]
    ops += _full_attn(model, li, L.core, ph, sh, b, cross=False)
    if tp > 1 and not L.fused_out:
        ops.append(Op("attn_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * rb))
    if L.cross is not None:
        ops.append(Op("cross_norm", "vector", li, vec=t * h * 4))
        ops += [lin("cross", l, "attn") for l in L.cross_linears]
        ops += _full_attn(model, li, L.cross, ph, sh, b, cross=True)
        if tp > 1:
            ops.append(Op("cross_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * rb))
    if L.vec_kind == "esm2":
        ops.append(Op("ffn_norm", "vector", li, vec=t * h * 4, sfu=float(t), a_fmt="fp32"))
    elif L.vec_kind == "generanno":
        ops.append(Op("mlp_rms", "vector", li, vec=t * h * 4, sfu=float(t), a_fmt="fp32"))
    else:
        ops.append(Op("ffn_norm", "vector", li, vec=t * h * (4 + mod)))
    f = L.ffn
    if f.kind == "moe":
        # 0.144 MoE DiT (LingBot-Video LingBotVideoSparseMoeBlock): router + shared expert GEMMs on every local token;
        # routed experts as grouped GEMMs over this rank's tokens (no EP in the reference: Ulysses SP splits tokens,
        # every rank holds all experts, TP splits d_expert); uniform-routing expectation of the hit experts
        # 0.384: pool B (ERNIE-4.5-VL) uses ph.mm of each sequence. n_experts_b = 0 keeps this block unchanged.
        mm_tok = min(t, b * ph.mm) if f.n_experts_b else 0
        text_tok = (t - mm_tok) if f.n_experts_b else t
        for l in L.ffn_linears:
            if mm_tok == 0 and l.name.startswith("router_b"):
                continue
            role = "router" if l.name.startswith("router") else ("shared_expert" if l.name.startswith("shared") else "mlp")
            if l.role == "mlp_bf16" or l.role.startswith("mtp_"):
                role = l.role
            ops.append(lin("moe", l, role))
        if f.zero_experts and not f.groups:
            r = identity_zero_routing(f.n_experts, f.zero_experts, f.top_k, text_tok, 1)
        else:
            r = moe_routing(f.n_experts, f.top_k, text_tok, 1)
        d_loc, ein = _cdiv(f.d_expert, tp), f.expert_in(h)
        if r.hit:
            u = min(1.0, r.pairs_local / (r.hit * r.m_e))
            er = f.expert_role or "expert"
            ops.append(replace(gemm(model, "expert.gate_up", li, r.m_e, ein, (2 if f.gated else 1) * d_loc,
                                    er, count=r.hit, stream=True), share=u))
            ops.append(replace(gemm(model, "expert.down", li, r.m_e, d_loc, ein, er, count=r.hit, stream=True),
                               share=u))
        pairs_b, d_loc_b = 0.0, 0
        if mm_tok and f.n_experts_b:
            rb_ = moe_routing(f.n_experts_b, f.top_k, mm_tok, 1)
            d_loc_b = _cdiv(f.d_expert_b, tp)
            if rb_.hit:
                ub = min(1.0, rb_.pairs_local / (rb_.hit * rb_.m_e))
                ops.append(replace(gemm(model, "expert_b.gate_up", li, rb_.m_e, ein,
                                        (2 if f.gated else 1) * d_loc_b, "expert", count=rb_.hit, stream=True),
                                   share=ub))
                ops.append(replace(gemm(model, "expert_b.down", li, rb_.m_e, d_loc_b, ein, "expert",
                                        count=rb_.hit, stream=True), share=ub))
                pairs_b = rb_.pairs_local
        dense_ff = _cdiv(f.d_ff, tp) if f.d_ff else 0
        n_route = f.n_experts + f.zero_experts + (f.n_experts_b if mm_tok else 0)
        ops.append(Op("act", "vector", li, vec=((r.pairs_local * d_loc + pairs_b * d_loc_b
                      + (t * _cdiv(f.n_shared * f.d_shared, tp))) * 4 + t * n_route * 6 + t * dense_ff * 4),
                      sfu=((r.pairs_local * d_loc + pairs_b * d_loc_b
                            + t * _cdiv(f.n_shared * f.d_shared, tp)) * 2
                           + t * n_route + t * dense_ff * 2)))   # SiLU; sigmoid router + top-k (6 / class)
    else:
        if _evo2_dense(L):
            if not f.gated or [l.name for l in L.ffn_linears] != ["gate_up", "down"]:
                raise ValueError("evo2 mlp is not gate_up then down")
            nact = t * _cdiv(f.d_ff, tp)
            af = model.act_fmt
            for l in L.ffn_linears:
                ops.append(lin("mlp", l, l.role if l.role == "mlp_bf16" or l.role.startswith("mtp_") else "mlp"))
                if l.name == "gate_up":
                    ops.extend(_evo2_mlp_vecs(L.vec_kind, nact, li, af))
        else:
            ops += [lin("mlp", l, l.role if l.role == "mlp_bf16" or l.role.startswith("mtp_") else "mlp")
                    for l in L.ffn_linears]
        if _evo2_dense(L):
            # gelu and the product were inserted between gate_up and down.
            pass
        elif L.vec_kind == "esm2":
            # modeling_esm.gelu: x * 0.5 * (1 + erf(x / sqrt(2))). Four arithmetic ops and one erf.
            # The class does not upcast, so a_fmt follows the activation dtype.
            nact = t * _cdiv(L.ffn.d_ff, tp)
            ops.append(Op("gelu", "vector", li, vec=nact * 5, sfu=float(nact), a_fmt=model.act_fmt))
        elif L.vec_kind == "generanno":
            # ACT2FN["silu"] is x * sigmoid(x): exp, one add, a reciprocal, then the multiply
            # by x (4 element-ops, 2 of them on the SFU). The gate*up product is one multiply.
            # The class does not upcast, so a_fmt follows the billed activation dtype.
            nact = t * _cdiv(L.ffn.d_ff, tp)
            af = model.act_fmt
            ops.append(Op("silu", "vector", li, vec=nact * 4, sfu=float(nact * 2), a_fmt=af))
            ops.append(Op("mlp_mul", "vector", li, vec=float(nact), a_fmt=af))
        else:
            ops.append(Op("act", "vector", li, vec=t * _cdiv(L.ffn.d_ff, tp) * 8))  # GELU (tanh) / SwiGLU 「假设」 8 ops / element
    if L.post_norm:      # 0.144 sandwich norms (RMSNorm on the attention and FFN outputs before the gated residual)
        ops.append(Op("post_norm", "vector", li, vec=2 * t * h * 4))
    if tp > 1:
        ops.append(Op("mlp_allreduce", "comm", li, comm_kind="allreduce", comm_group=tp, comm_bytes=t * h * rb))
    ops.append(Op("residual", "vector", li, vec=t * h * (2 + (2 if model.adaln else 0))))
    if L.emb_ln:
        # EsmEncoder.emb_layer_norm_after, once, on the last language layer's output.
        # ESMFold does not run EsmLMHead, so this is not the MLM-head LayerNorm.
        if L.vec_kind != "esm2":
            raise ValueError("emb_layer_norm_after is an esm2 vector")
        ops.append(Op("emb_ln", "vector", li, vec=t * h * 4, sfu=float(t), a_fmt="fp32"))
    return ops


def full_io_ops(model: ModelSpec, ph: Phase, sh: Shard, side: str) -> list[Op]:
    return rank_shares(_full_io_ops(model, ph, sh, side), sh, lambda: _full_io_ops(model, ph, Shard(), side))


def _full_io_ops(model: ModelSpec, ph: Phase, sh: Shard, side: str) -> list[Op]:
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
        if model.vec_kind == "esm2":
            # EsmEncoder.emb_layer_norm_after. Same LayerNorm recipe as the block norms.
            ops.append(Op("final_norm", "vector", li, vec=t * model.hidden * 4, sfu=float(t), a_fmt="fp32"))
        elif model.vec_kind == "generanno":
            # GenerannoModel.norm. Same RMSNorm recipe as the block norms.
            ops.append(Op("final_rms", "vector", li, vec=t * model.hidden * 4, sfu=float(t), a_fmt="fp32"))
        else:
            ops.append(Op("final_norm", "vector", li, vec=t * model.hidden * (4 + (4 if model.adaln else 0))))
    for l in (model.io_pre if side == "pre" else model.io_post):
        k, n = _lin_local(l, sh.tp)
        m = _rows(model, l, ph, b, sh.sp)
        if ph.mreuse and l.rows == "tok":     # 0.255: per-token time MLP only for the active (image) mode
            m = b * _cdiv(_video_rows(model, ph), sh.sp)
        ops.append(gemm(model, "io." + l.name, li, m, k, n, l.role or "io", stream=True,
                        replicated=_repl_full(l, sh.tp, sh.sp)))
        if sh.tp > 1:
            ops.append(Op("io_allgather", "comm", li, comm_kind="allgather", comm_group=sh.tp, comm_bytes=m * n * ab))
    if side == "pre":
        ops += _io_attn_ops(model, ph, sh, b, li)
    if side == "post" and model.vocab and model.vocab_head:     # MLM logits for every token (tied decoder)
        n = _cdiv(model.vocab, sh.tp)
        if model.vec_kind == "esm2":
            # EsmLMHead: dense (already emitted) → gelu → LayerNorm → decoder + bias.
            # GEMM order stays dense then decoder. gelu and the head LayerNorm sit between them.
            ops.append(Op("lm_gelu", "vector", li, vec=t * model.hidden * 5, sfu=float(t * model.hidden),
                          a_fmt=model.act_fmt))
            ops.append(Op("lm_ln", "vector", li, vec=t * model.hidden * 4, sfu=float(t), a_fmt="fp32"))
        ops.append(gemm(model, "mlm_decoder", li, t, model.hidden, n, "embed", stream=True))
        if model.vec_kind == "esm2":
            # decoder(x) + bias. One add on this rank's vocab columns. No transcendental.
            ops.append(Op("lm_bias", "vector", li, vec=float(t * n), a_fmt=model.act_fmt))
        else:
            ops.append(Op("act", "vector", li, vec=t * model.hidden * 8 + t * n * 3))
    return ops


def _io_attn_ops(model: ModelSpec, ph: Phase, sh: Shard, b: int, li: int) -> list[Op]:
    """0.258: attention cores of io blocks (``ModelSpec.io_attn``; MOSS-Image text refiner: SDPA over the sample's text
    rows, bidirectional, RMS QK-norm + RoPE — pipeline_moss_image.py _TextBlock, 「来源」).  Heads split over TP, then
    over SP like the block attention (the text rows are SP-split in the refiner GEMMs; the all-to-all of these few
    rows is not added 「近似」).  No rows on a cached / reused step (the text stream does not run).
    0.260: an optional 6th field names the per-token vector work: "qknorm_rope" (default, MOSS), "qknorm"
    (MiniMax-H3 token refiner: RMS QK-norm, no RoPE, no mask 「来源」 — diffusers transformer_minimax_h3.py
    MiniMaxH3TokenRefinerBlock, "No AdaLN and no rotary embedding"; 0.261, was 「假设」), "" (HunyuanVideo
    token refiner: diffusers HunyuanVideoIndividualTokenRefinerBlock — plain Attention, no QK-norm, no RoPE)."""
    ops: list[Op] = []
    if not model.io_attn or ph.kvpre or ph.mreuse:
        return ops
    ab = _fmt(model.act_fmt).bytes
    for ent in model.io_attn:
        name, rows, heads, dh, blocks = ent[:5]
        norm = ent[5] if len(ent) > 5 else "qknorm_rope"
        T = _prefix(model, ph) if rows == "txt" else 0
        if T <= 0:
            continue
        h_loc = _cdiv(_cdiv(heads, sh.tp), sh.sp)
        cnt = b * h_loc * blocks
        ops += [Op(f"io.{name}.qk", "attn", li, m=T, k=dh, n=T, count=cnt, act_bytes=cnt * T * dh * ab, stream=True,
                   orient=True),
                Op(f"io.{name}.pv", "attn", li, m=T, k=T, n=dh, count=cnt, act_bytes=cnt * T * dh * ab, orient=True),
                Op(f"io.{name}.softmax", "vector", li, vec=cnt * T * T * 5, sfu=cnt * T * T)]
        per = (dh * 2 * 3 if "qknorm" in norm else 0) + (dh * 2 * 3 if "rope" in norm else 0)
        if per:
            ops.append(Op(f"io.{name}.qk_norm_rope", "vector", li,
                          vec=b * _cdiv(T, sh.sp) * _cdiv(heads, sh.tp) * blocks * per))
    return ops


def rank_shares(ops: list[Op], sh: Shard, global_ops) -> list[Op]:
    """0.63 (external review) — work conservation.  The per-rank op list is the busiest rank's (ceil splits of heads,
    columns, tokens, batch; padded expert rows), which sets the stage time.  Each op gets ``share`` = mean-rank useful
    work / this rank's useful work, by op name against the unsharded op graph (``global_ops()``, same phase):
    share = W_global / (N · Σ_rank W / replicated), N = TP·DP·SP ranks of the stage, W = FLOPs (vector element-ops for
    vector ops).  So N · Σ work·share / replicated = the global work exactly, and replicated execution (e.g. the MLA
    latent on every TP rank) still counts on every rank.  Divisible splits: share = 1."""
    n = sh.tp * sh.dp * sh.sp
    if n == 1:
        return ops
    # 0.198: fused / unfused QKV are one work item for the share match (the fusion test depends on the shard's head
    # replication, so the unsharded reference may be fused where a rank is not)
    _nk = lambda nm: "attn.qkv" if nm in _QKV_NAMES else nm
    gops = list(global_ops())     # 0.113: built once (was rebuilt for the weight pass below) — same ops, same order
    g_f: dict = {}; g_v: dict = {}
    for o in gops:
        if o.kind != "comm":
            g_f[_nk(o.name)] = g_f.get(_nk(o.name), 0.0) + o.flops * o.share     # an op's own share = its useful fraction
            g_v[_nk(o.name)] = g_v.get(_nk(o.name), 0.0) + o.vec * o.share       # (padded expert rows), 1 otherwise
    u_f: dict = {}; u_v: dict = {}
    for o in ops:
        if o.kind != "comm":
            r = max(o.replicated, 1e-12)
            u_f[_nk(o.name)] = u_f.get(_nk(o.name), 0.0) + o.flops * o.share / r
            u_v[_nk(o.name)] = u_v.get(_nk(o.name), 0.0) + o.vec * o.share   # vector-only ops: ``replicated`` marks the
            # replicated latent-cache bytes (MLA softmax), the score work itself is split by heads
    # 0.64: weight share — Σ_rank w_params = copies · W_global, copies = ⌊N · w_rank / W_global⌋ (≥ 1; the integer
    # replication of the weight over DP / SP / replicated-over-TP ranks — a ceil split inflates the busiest rank by
    # < one copy), wshare = copies · W_global / (N · w_rank).  假设: a partially replicated split (e.g. 2 KV heads on
    # TP 3) is read as uneven, not replicated
    g_w: dict = {}; u_w: dict = {}
    for o in gops:
        if o.kind != "comm" and o.w_params:
            g_w[_nk(o.name)] = g_w.get(_nk(o.name), 0.0) + o.w_params
    for o in ops:
        if o.kind != "comm" and o.w_params:
            u_w[_nk(o.name)] = u_w.get(_nk(o.name), 0.0) + o.w_params
    out = []
    for o in ops:
        x = 1.0
        xw = 1.0
        if o.kind != "comm" and u_w.get(_nk(o.name), 0) > 0 and g_w.get(_nk(o.name), 0) > 0:
            cp = max(1, math.floor(n * u_w[_nk(o.name)] / g_w[_nk(o.name)] + 1e-9))
            xw = min(1.0, cp * g_w[_nk(o.name)] / (n * u_w[_nk(o.name)]))
        if xw < 1.0 - 1e-12:
            o = replace(o, wshare=xw)
        if o.kind != "comm" and _nk(o.name) in g_f:
            if o.flops > 0 and u_f[_nk(o.name)] > 0:
                x = g_f[_nk(o.name)] / (n * u_f[_nk(o.name)])
            elif o.vec > 0 and u_v[_nk(o.name)] > 0:
                x = g_v[_nk(o.name)] / (n * u_v[_nk(o.name)])
        out.append(o if abs(x - 1.0) < 1e-12 else replace(o, share=o.share * x))
    return out


def total(ops: list[Op], attr: str) -> float:
    return sum(getattr(o, attr) for o in ops)
