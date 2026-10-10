"""L1a — model specification parsed from official Hugging Face configs.

A ``ModelSpec`` is the *verified* architectural description of one released
checkpoint: every linear (GEMM weight) with its TP split rule, the attention
core semantics (GQA / MLA / sparse / compressed / linear-recurrent), the FFN
(dense / MoE incl. shared + latent experts) and the **as-released storage
format per tensor role** taken from the safetensors headers
(``accel_dse/data/releases/*.json``).

Coverage axis (one of the three model labels):
  * ``full``     every parameter-bearing op and the attention/KV semantics are
                 modelled; params within ±2 % of the release.
  * ``partial``  all GEMMs modelled; some attention mechanism (sparse indexer,
                 sliding window variants, attention sinks …) is approximated —
                 listed in ``notes``.
  * ``proxy``    「架构代理」: the release contains mechanisms we only
                 approximate (hyper-connections, n-gram/engram tables,
                 compressed-KV attention …). Shown with a badge; never mixed
                 silently with full-coverage results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RELEASES = DATA / "releases"


# ---------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class Linear:
    """y[M,N] = x[M,K] @ W[K,N].  ``split`` = TP rule:
    col  – N split across attention/FFN TP ranks
    row  – K split (partial sums → all-reduce / reduce-scatter)
    rep  – replicated on every TP rank (small low-rank projections, routers)
    head – N split in units of ``unit`` (KV heads: ceil(n_kv/tp) per rank)
    ``rows`` = which rows feed the GEMM in a non-autoregressive (``full``) forward:
    tok  – every token of the sequence (default; the only kind LLM layers use)
    img  – the video tokens only (sequence minus the text prefix of joint-attention DiTs)
    ctx  – the conditioning tokens (text-encoder output feeding cross-attention K/V)
    seq  – one row per sequence (timestep embedding / AdaLN modulation)
    txt  – the joint text rows of the sequence (dual-stream MMDiT text weights, text refiners)
    aud  – the joint audio rows of the sequence (MiniMax-H3 audio io)
    structure models (0.43; Phase.pair gives the grid sizes):
    res  – one row per residue / token (single representation)
    pair – N² rows (pair representation)        msa / xmsa – S·N rows (MSA / extra-MSA stack)
    tmpl – T·N² rows (template pair stack)      atom – A rows (atoms);  apair – A·window rows (local atom pairs)
    """
    name: str
    k: int
    n: int
    split: str = "col"
    unit: int = 0          # for split == "head": columns per head
    groups: int = 1        # block-diagonal (grouped) linear: params = k*n (k,n are totals)
    rows: str = "tok"
    role: str = ""         # weight-format role override (non-LLM io GEMMs stored in another dtype); "" = default

    @property
    def params(self) -> int:
        return self.k * self.n // self.groups


@dataclass(frozen=True)
class AttnCore:
    """Per-token attention semantics of one layer.

    kind: gqa | mla | linear | none
      gqa    – n_q heads of qk_dim/v_dim, n_kv KV heads (GQA packing g = n_q/n_kv)
      mla    – latent cache (kv_lora + rope); decode uses absorbed form
      linear – recurrent state per sequence: n_state_heads × dk × dv (fp32 「假设」)
    window      – sliding-window length (None = full context)
    topk        – sparse attention: each query attends to ≤ topk selected tokens
    compress    – KV compression ratio (DeepSeek-V4 CSA): cache holds ctx/compress
    idx_heads/idx_dim – lightning indexer (DSA) scoring cost per (query, key)
    """
    kind: str
    n_q: int = 0
    n_kv: int = 0
    qk_dim: int = 0
    v_dim: int = 0
    kv_lora: int = 0
    rope_dim: int = 0
    window: int | None = None
    topk: int | None = None
    compress: int = 1
    idx_heads: int = 0
    idx_dim: int = 0
    idx_rope: int = 0          # indexer RoPE dims (q per head and k)
    idx_fp8: bool = False      # DeepSeek-V3.2 reference indexer: Hadamard-rotated q / k quantised to fp8 e4m3 with one
                               # fp32 scale per 128 elements; key cache fp8 + scales (inference/model.py, Indexer)
    # DeepSeek-V4.1 (official inference/model.py): every attending query reads its window AND top-k compressed entries
    # (keys = min(p, w) + min(top-k, ⌊p/r⌋), exact); entry bytes per cache; compressed KV / index keys shared by layers
    dual: bool = False
    win_bytes: float = 0.0     # bytes per window-cache entry (fp8 + e8m0 scale per 32)
    cmp_bytes: float = 0.0     # bytes per compressed entry (fp4 + e4m3 scale per 16)
    kv_owner: bool = False     # this layer runs the Compressor and stores the compressed cache (kv_source_layers)
    idx_owner: bool = False    # this layer derives + stores the index keys (wk on the latent; fp4)
    idx_kbytes: float = 0.0    # bytes per index-key entry
    cmp_noape: bool = False    # Compressor without the learned position bias (V4.1)
    cand: str = ""             # two-level top-k: src (select_candidate_blocks) | use (mask to the source's blocks)
    cand_block: int = 0
    cand_topk: int = 0
    idx_split: bool = False    # DeepSeek-V4 reference: indexer heads column-parallel over TP + all-reduce of the scores
    cmp_coff: int = 0          # DeepSeek-V4 Compressor: 1, or 2 for the overlapping ratio-4 windows (0 = no compressor)
    idx_cmp: bool = False      # DeepSeek-V4 indexer keys come from its own (Hadamard-rotated) compressor
    rope_n: int = 0            # rotary dims actually rotated per head when it differs from the MLA latent split
    qsa: int = 0               # Qwen4Exp QSA: key-block size; query sees every token until > top-k/qsa complete blocks,
                               # then top-k tokens of the best blocks + the incomplete tail block
    idx_kpool: int = 0         # GLM-5.3-Flash: indexer scores gated k-pools of this many keys (top-k/kpool pools)
    lin: str = ""              # linear-attention recurrence: gdn | kda (delta rule, scalar / per-channel gate) | lightning
    lin_chunk: int = 0         # prefill chunk (block) length of the reference chunked kernel
    sink: bool = False         # learned attention sink per head (gpt-oss): one extra softmax logit per query row
    n_state_heads: int = 0
    state_dk: int = 0
    state_dv: int = 0
    conv_channels: int = 0
    conv_kernel: int = 0
    causal: bool = True        # False: bidirectional (DiT / protein encoders)
    cross: bool = False        # cross-attention: keys are the conditioning tokens (``full`` phase ``ctx``)
    span: str = "full"         # full-phase self-attention extent: full (3D) | spatial (per latent frame) |
                               # temporal (per spatial position) — factorized ST-DiT (Open-Sora STDiT3)

    def kv_elems_per_token(self) -> float:
        """KV-cache elements stored per token (one layer, whole model width)."""
        if self.kind == "gqa":
            return self.n_kv * (self.qk_dim + self.v_dim) / self.compress
        if self.kind == "mla":
            return (self.kv_lora + self.rope_dim) / self.compress
        return 0.0

    def idx_elems_per_token(self) -> float:
        return float(self.idx_dim) / self.compress if self.idx_heads else 0.0

    def idx_bytes_per_token(self, kvb: float) -> float:
        """Indexer key-cache bytes per token: fp8 key + fp32 scale per 128 (DeepSeek-V3.2 reference), else the
        KV-cache dtype (transformers GlmMoeDsaIndexer caches the bf16 key)."""
        if not self.idx_heads:
            return 0.0
        if self.idx_kpool:      # transformers Glm5NextTextIndexer packs [k, gate scores, valid] in the hidden dtype
            return (2 * self.idx_dim + 1) * kvb / self.compress
        per = self.idx_dim + 4 * math.ceil(self.idx_dim / 128) if self.idx_fp8 else self.idx_dim * kvb
        return per / self.compress

    def state_elems(self) -> int:
        """Recurrent state per sequence (linear attention) incl. conv state."""
        if self.kind != "linear":
            return 0
        return self.n_state_heads * self.state_dk * self.state_dv + self.conv_channels * max(0, self.conv_kernel - 1)

    def ctx_eff(self, ctx: int) -> int:
        """Keys actually attended per query at context length ``ctx``."""
        if self.qsa:
            return ctx if ctx // self.qsa <= self.topk // self.qsa else self.topk + ctx % self.qsa
        if self.dual:
            return min(ctx, self.window or 0) + min(self.topk if self.topk is not None else ctx, ctx // self.compress)
        c = ctx
        if self.window is not None:
            c = min(c, self.window)
        if self.compress > 1:
            c = math.ceil(ctx / self.compress) + (self.window or 0)
        if self.topk is not None:
            c = min(c, self.topk + (self.window or 0) if self.compress > 1 else self.topk)
        return c

    def keys_sum(self, n: int) -> int:
        """0.61.1: Σ_{p=1..n} keys attended by the query at position p, min(p, ctx_eff(p)) — exact closed form.
        Prefill of a windowed / top-k / compressed layer charges the mean over its positions (the first queries see
        fewer than the cap).  compress > 1: min(p, ⌈p/c⌉ + w, top-k + w)."""
        if n <= 0:
            return 0
        c, w = self.compress, self.window or 0
        if self.qsa:                            # Σ keys(p): p up to p0 − 1, then top-k + p mod b
            b, p0 = self.qsa, (self.topk // self.qsa + 1) * self.qsa
            if n < p0:
                return n * (n + 1) // 2
            m = n - p0 + 1                      # positions p0..n, p0 ≡ 0 (mod b)
            a, r = divmod(m, b)
            return (p0 - 1) * p0 // 2 + m * self.topk + a * (b * (b - 1) // 2) + r * (r - 1) // 2
        if self.dual:                           # Σ min(p, w) + Σ min(top-k, ⌊p/c⌋)
            ws = n * (n + 1) // 2 if n <= w else w * (w + 1) // 2 + (n - w) * w

            def floor_sum(m: int) -> int:       # Σ_{p=1..m} ⌊p/c⌋
                a, r = divmod(m, c)
                return c * a * (a - 1) // 2 + a * (r + 1)
            k = self.topk
            if k is None or n < (k + 1) * c:
                return ws + floor_sum(n)
            pk = k * c                           # ⌊p/c⌋ ≥ k from p = k·c on
            return ws + floor_sum(pk - 1) + (n - pk + 1) * k
        if c <= 1:
            cap = min(x for x in (self.window, self.topk, n) if x is not None)
            return n * (n + 1) // 2 if n <= cap else cap * (cap + 1) // 2 + (n - cap) * cap
        top = self.topk + w if self.topk is not None else None

        def ceil_sum(m: int) -> int:           # Σ_{p=1..m} ⌈p/c⌉
            a, r = divmod(m, c)
            return c * a * (a + 1) // 2 + r * (a + 1)

        def last(ok) -> int:                   # largest p in [0, n] with ok(p) (ok monotone: true then false)
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
            return lo

        g = lambda p: -(-p // c) + w
        p1 = last(lambda p: p <= g(p))         # below p1 every key exists in the compressed cache + window
        hsum = lambda m: m * (m + 1) // 2 if m <= p1 else p1 * (p1 + 1) // 2 + (ceil_sum(m) - ceil_sum(p1)) + w * (m - p1)
        if top is None:
            return hsum(n)
        p3 = last(lambda p: min(p, g(p)) <= top)
        return hsum(p3) + (n - p3) * top

    def keys_mean(self, ctx: int, q: int) -> float:
        """Mean keys per query over the q positions after a cached prefix of ``ctx`` (0.61.1)."""
        return (self.keys_sum(ctx + q) - self.keys_sum(ctx)) / q


@dataclass(frozen=True)
class PairCore:
    """Activation × activation work of one structure-model module (0.43; AlphaFold-family trunks).  The weight GEMMs
    of the module are ordinary ``Linear``s in ``Layer.pair_linears``; this records only the work without weights.

    kind
      trimul    – triangle multiplicative update: ``dim`` channels of an [N×N]·[N×N] product (outgoing / incoming)
      tri_att   – triangle attention (starting / ending node): N groups of N keys, ``heads`` × ``dim``, pair bias
      row_att   – attention along the residue axis of each row of ``over`` (msa / xmsa: S groups; tmpl: T·N groups)
      col_att   – MSA column attention: N groups of S keys (``glob``: one averaged query per column, AF2 extra MSA)
      opm       – outer-product mean: [N·dim × S]·[S × N·dim2] (then a Linear from dim·dim2 to the pair width)
      pwa       – pair-weighted averaging (AF3 MSA module): ``heads`` × [N×N]·[N × S·dim]
      seq_att   – attention over the N tokens with a pair bias (single track, diffusion transformer, IPA)
      local_att – atom attention in windows of ``win[0]`` queries × ``win[1]`` keys
    ``v_dim`` – value width when it differs from ``dim`` (IPA: scalar + point + pair values).
    ``width`` – channel width of the representation the module reads (c_z / c_m / c_t, from the input dim of its
                query / projection weight): the payload of a DAP transpose (0.45)."""
    kind: str
    heads: int = 1
    dim: int = 0
    dim2: int = 0
    over: str = "pair"
    v_dim: int = 0
    glob: bool = False
    win: tuple[int, int] = (0, 0)
    width: int = 0


@dataclass(frozen=True)
class Ffn:
    kind: str                    # dense | moe | none
    d_ff: int = 0                # dense intermediate
    gated: bool = True           # SwiGLU (3 mats) vs 2 mats
    n_experts: int = 0
    top_k: int = 0
    d_expert: int = 0
    n_shared: int = 0
    d_shared: int = 0
    latent: int = 0              # latent MoE: experts operate in this width (Kimi-K3)

    @property
    def mats(self) -> int:
        return 3 if self.gated else 2

    def expert_in(self, hidden: int) -> int:
        return self.latent or hidden


@dataclass(frozen=True)
class Layer:
    attn_linears: tuple[Linear, ...]
    core: AttnCore
    ffn: Ffn
    ffn_linears: tuple[Linear, ...] = ()     # dense / shared / router / latent projections
    misc_params: int = 0                     # norms, gates, conv, hc — not GEMMs
    hc: int = 0                              # mHC streams (GLM-5.3-Flash): attn + ffn hyper-connection compute per layer
    hc_iters: int = 0                        # Sinkhorn-Knopp iterations of the comb weight
    gres_n: int = 0                          # Qwen4Exp gated residual: streams (attn + mlp GatedResidual per layer)
    gres_head: bool = False                  # last layer: final hyper_connection_mixer (no combine)
    ple_rows: int = 0                        # Qwen4Exp PLE layer: n-gram rows per token (heads × (ngram − 1))
    ple_row_dim: int = 0
    ple_conv: int = 0                        # dilated depthwise conv kernel (dilation = ngram_size)
    ple_dil: int = 0
    engram_cols: int = 0                     # V4.1 Engram: hash rows fetched per token (n-gram sizes × heads)
    engram_dim: int = 0                      # row width (fp8 + e8m0 scale per 32)
    hc_head: str = ""                        # last layer only: final stream collapse — mean (GLM-5.3-Flash) | mix (V4)
    cross_linears: tuple[Linear, ...] = ()   # cross-attention projections (video DiT: q from tokens, k/v from text)
    cross: AttnCore | None = None
    fused_out: bool = False                  # parallel attention + MLP block with one fused output GEMM
                                             # (HunyuanVideo single-stream): one TP all-reduce per block
    # structure models (0.43): a block of the AlphaFold-family trunk / diffusion module
    pair_linears: tuple[Linear, ...] = ()    # every weight GEMM of the block (rows tag = the grid it runs on)
    pair_cores: tuple[PairCore, ...] = ()
    repeat: str = ""                         # "" once | recycle (× trunk passes) | diff (× diffusion steps)
    iters: int = 1                           # fixed repetitions per pass with shared weights (structure module: 8)
    samples: str = ""                        # rows × diffusion samples: "" no | tok (token / atom grids; diffusion
                                             # module, pair conditioning shared) | all (every grid; confidence head)
    misc_role: str = ""                      # format role of misc_params ("" = attn; structure trunks: pair)
    stack: str = ""                          # display label of the stack (esm / trunk / msa / pairformer / diffusion …)

    def expert_params(self, hidden: int) -> int:
        f = self.ffn
        if f.kind != "moe":
            return 0
        return f.n_experts * f.mats * f.expert_in(hidden) * f.d_expert

    def params(self, hidden: int) -> int:
        return (sum(l.params for l in self.attn_linears) + sum(l.params for l in self.ffn_linears)
                + sum(l.params for l in self.cross_linears) + sum(l.params for l in self.pair_linears)
                + self.expert_params(hidden) + self.misc_params)


@dataclass(frozen=True)
class RoleFormat:
    fmt: str            # format name in core.dtypes.FORMATS (or what-if override)
    bits: float         # effective stored bits / param incl. scales (from release bytes)


@dataclass(frozen=True)
class NativeWorkload:
    """Native workload of a non-autoregressive release (video DiT clip / protein sequence).  Defaults come from the
    official config / README (cited in the catalog); the scenario's ``workload`` block overrides them."""
    kind: str = ""                 # gen | protein
    frames: int = 0                # output video frames
    height: int = 0                # output pixels
    width: int = 0
    fps: int = 0
    vae_t: int = 4                 # VAE temporal / spatial compression
    vae_s: int = 8
    patch: tuple[int, int, int] = (1, 2, 2)   # (t, h, w) latent patch
    text_tokens: int = 0           # conditioning tokens (padded text length)
    prefix_tokens: int = 0         # text tokens concatenated into the attention sequence (joint attention)
    steps: int = 50                # denoise steps (reference sampler default)
    cfg: int = 2                   # forward passes per step (classifier-free guidance cond + uncond)
    vae_frames: str = "causal"     # latent-frame rule: causal (F−1)/t+1 | chunk17 (Open-Sora 1.2: 17 → 5 per chunk) |
                                   # h3 (MiniMax-H3: F snapped to 17n+5 → 5n+2)
    audio_per_s: int = 0           # joint audio rows per second of video (MiniMax-H3: 40 latents/s × channels)
    audio_channels: int = 0
    seq_len: int = 0               # protein residues
    msa: int = 0                   # structure models (0.43): MSA rows / extra-MSA rows / templates
    xmsa: int = 0
    templates: int = 0
    tmpl_msa: bool = False         # AF2 / OpenFold (0.61.3): the T template torsion-angle rows are concatenated to the
                                   # MSA representation and run through every Evoformer block (MSA grid = msa + T)
    atoms_per_res: float = 0.0     # all-atom models: heavy atoms per residue
    recycles: int = 0              # trunk passes (incl. the first)
    diff_steps: int = 0            # diffusion sampler steps
    samples: int = 0               # diffusion samples per request (batched)
    pair_dim: int = 0              # pair-representation width c_z (residual-stream / p2p accounting)
    max_seq: int = 0               # longest trained sequence (residues)
    special_tokens: int = 0        # protein: <cls> + <eos>
    source: str = ""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_id: str
    arch: str
    hidden: int
    vocab: int
    layers: tuple[Layer, ...]
    tie_embeddings: bool = False
    mtp_layers: tuple[Layer, ...] = ()
    mtp_extra_params: int = 0                # eh_proj etc. per MTP module
    lookup_params: int = 0                   # n-gram / engram tables (lookup only)
    lookup_bits: float = 0.0                 # stored bits per lookup param incl. scales (0 = embed format)
    engram_ngram: int = 0                    # V4.1 engram_max_ngram_size (hash multiplies per token)
    draft: str = ""                          # dspark: V4.1 block draft head (one pass of draft_block positions)
    draft_block: int = 0
    draft_targets: int = 0                   # target layers whose stream means feed main_proj
    draft_markov: int = 0                    # markov head rank
    formats: tuple[tuple[str, RoleFormat], ...] = ()
    act_fmt: str = "bf16"
    kv_fmt: str = "bf16"
    state_fmt: str = "fp32"
    coverage: str = "full"
    notes: tuple[str, ...] = ()
    release_params: int | None = None        # logical LLM params in the safetensors (excl. MTP, vision)
    release_mtp_params: int | None = None
    release_bytes: int | None = None
    quantized_release: bool = False
    what_if: bool = False                    # formats overridden by the user
    coverage_reasons: tuple[str, ...] = ()   # what is approximated (empty ⇔ coverage == full)
    vision_params: int = 0                   # VLM vision encoder + projector params (bf16 as released; 0.62 modelled)
    vision: object = None                    # core.vision.VisionSpec (0.62; None = no vision tower / text-only model)
    max_ctx: int = 0                         # config max_position_embeddings (0 = unknown) — warning only (0.61.4)
    # ---- non-autoregressive domains (video generation DiT / protein encoders); defaults = LLM
    domain: str = "llm"                      # llm | gen | protein
    kv_cache: bool = True                    # autoregressive KV cache (False: one full-sequence forward)
    io_pre: tuple[Linear, ...] = ()          # input-side GEMMs on the first stage (patch / text / timestep embedders)
    io_post: tuple[Linear, ...] = ()         # output-side GEMMs on the last stage (un-patchify head, LM-head dense)
    io_misc_params: int = 0                  # io biases / unused stored tables (counted in storage, no GEMM)
    final_norm_params: int | None = None     # None → hidden (LLM RMSNorm)
    adaln: bool = False                      # AdaLN modulation (shift/scale/gate) around each norm
    workload: NativeWorkload | None = None
    standby_params: int = 0                  # stored but idle per forward (Wan2.2 A14B: the other noise-level expert)
    cached_params: int = 0                   # in the release but not loaded for inference (MiniMax-H3 AdaLN branches:
                                             # modulation outputs precomputed per timestep, README)

    # ----- derived
    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def fmt(self, role: str) -> RoleFormat:
        d = dict(self.formats)
        if role in d:
            return d[role]
        for fb in {"shared_expert": "mlp", "expert": "mlp", "mlp": "attn", "router": "attn",
                   "lm_head": "embed", "mtp": "expert"}.get(role, "attn"), "attn":
            if fb in d:
                return d[fb]
        return RoleFormat("bf16", 16.0)

    @property
    def embed_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def head_params(self) -> int:
        return 0 if self.tie_embeddings else self.vocab * self.hidden

    @property
    def is_pair(self) -> bool:
        """Structure model (AlphaFold-family / ESMFold trunk): has pair-representation blocks."""
        return any(L.pair_linears or L.pair_cores for L in self.layers)

    def params(self, include_mtp: bool = False) -> int:
        p = sum(l.params(self.hidden) for l in self.layers) + self.embed_params + self.head_params + self.lookup_params
        p += self.hidden if self.final_norm_params is None else self.final_norm_params
        p += sum(l.params for l in self.io_pre + self.io_post) + self.io_misc_params
        p += self.standby_params + self.cached_params
        if include_mtp:
            p += self.mtp_params()
        return p

    def mtp_params(self) -> int:
        return sum(l.params(self.hidden) for l in self.mtp_layers) + self.mtp_extra_params * len(self.mtp_layers)

    def active_params(self) -> int:
        """Params touched per token (routed experts: top_k of n; lookup tables excluded — a few rows/token)."""
        p = self.params() - self.lookup_params - self.standby_params - self.cached_params
        for l in self.layers:
            if l.ffn.kind == "moe":
                f = l.ffn
                p -= (f.n_experts - f.top_k) * f.mats * f.expert_in(self.hidden) * f.d_expert
        return p

    @property
    def is_moe(self) -> bool:
        return any(l.ffn.kind == "moe" for l in self.layers)

    def param_check(self) -> dict:
        ours = self.params()
        ref = self.release_params
        rel = None if not ref else (ours - ref) / ref
        return {"ours": ours, "release": ref, "rel_err": rel,
                "mtp_ours": self.mtp_params(), "mtp_release": self.release_mtp_params}


# ---------------------------------------------------------------- helpers
def _tc(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and tc.get("hidden_size"):
        return {**{k: v for k, v in cfg.items() if k != "text_config"}, **tc}
    return cfg


def _g(c: dict, *keys, default=None):
    for k in keys:
        v = c.get(k)
        if v is not None:
            return v
    return default


def _ffn_dense(h: int, d: int, gated: bool = True) -> tuple[Ffn, tuple[Linear, ...]]:
    if gated:
        lin = (Linear("gate_up", h, 2 * d, "col"), Linear("down", d, h, "row"))
    else:
        lin = (Linear("up", h, d, "col"), Linear("down", d, h, "row"))
    return Ffn("dense", d_ff=d, gated=gated), lin


def _ffn_moe(h: int, E: int, k: int, de: int, n_sh: int = 0, d_sh: int = 0, *, gated: bool = True,
             latent: int = 0, shared_gate: bool = False) -> tuple[Ffn, tuple[Linear, ...]]:
    lin = [Linear("router", h, E, "rep")]
    if n_sh and d_sh:
        lin += [Linear("shared_gate_up", h, 2 * n_sh * d_sh if gated else n_sh * d_sh, "col"),
                Linear("shared_down", n_sh * d_sh, h, "row")]
    if shared_gate:
        lin.append(Linear("shared_expert_gate", h, 1, "rep"))
    if latent:
        lin += [Linear("latent_up", h, latent, "rep"), Linear("latent_down", latent, h, "rep")]
    return Ffn("moe", gated=gated, n_experts=E, top_k=k, d_expert=de, n_shared=n_sh, d_shared=d_sh,
               latent=latent), tuple(lin)


def _gqa(h: int, H: int, KV: int, hd: int, *, vd: int | None = None, out_gate: bool = False,
         window: int | None = None) -> tuple[tuple[Linear, ...], AttnCore]:
    vd = vd or hd
    lin = (Linear("q", h, H * hd * (2 if out_gate else 1), "col"),
           Linear("k", h, KV * hd, "head", unit=hd),
           Linear("v", h, KV * vd, "head", unit=vd),
           Linear("o", H * vd, h, "row"))
    return lin, AttnCore("gqa", n_q=H, n_kv=KV, qk_dim=hd, v_dim=vd, window=window)


def _mla(c: dict, h: int) -> tuple[tuple[Linear, ...], AttnCore]:
    H = c["num_attention_heads"]
    ql = c.get("q_lora_rank")
    kvl = c["kv_lora_rank"]
    nope = c.get("qk_nope_head_dim") or 0
    rope = c.get("qk_rope_head_dim") or 0
    vd = c.get("v_head_dim") or nope
    qd = nope + rope
    lin = []
    if ql:
        lin += [Linear("q_a", h, ql, "rep"), Linear("q_b", ql, H * qd, "col")]
    else:
        lin += [Linear("q", h, H * qd, "col")]
    lin += [Linear("kv_a", h, kvl + rope, "rep"), Linear("kv_b", kvl, H * (nope + vd), "col"),
            Linear("o", H * vd, h, "row")]
    misc = (ql or 0) + kvl
    core = AttnCore("mla", n_q=H, n_kv=1, qk_dim=qd, v_dim=vd, kv_lora=kvl, rope_dim=rope)
    return tuple(lin), core, misc


def _indexer(c: dict, h: int, ql: int | None) -> tuple[tuple[Linear, ...], int, int, int]:
    ih, idim, topk = c.get("index_n_heads") or 0, c.get("index_head_dim") or 0, c.get("index_topk")
    if not ih:
        return (), 0, 0, None
    lin = (Linear("idx_q", ql or h, ih * idim, "rep"), Linear("idx_k", h, idim, "rep"),
           Linear("idx_w", h, ih, "rep"))
    return lin, ih, idim, topk


def _list(v):
    if isinstance(v, str):
        try:
            return json.loads(v.replace("'", '"'))
        except Exception:
            return None
    return v


# ---------------------------------------------------------------- family builders
def _build_std(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Llama-like GQA decoders, optionally MoE (Qwen2/3, Llama, Mistral, Mixtral,
    Phi-3/4, Yi, InternLM2/3, Seed-OSS, Qwen3-MoE, GLM-4.5, gpt-oss)."""
    h = c["hidden_size"]
    H = c["num_attention_heads"]
    KV = c.get("num_key_value_heads") or H
    hd = c.get("head_dim") or h // H
    L = c["num_hidden_layers"]
    notes, cov = [], "full"
    mt = c.get("model_type", "")
    E = _g(c, "num_local_experts", "num_experts", "n_routed_experts", default=0) or 0
    k = _g(c, "num_experts_per_tok", "experts_per_token", "num_experts_per_token", default=0)
    de = _g(c, "moe_intermediate_size", "intermediate_size")
    n_sh = c.get("n_shared_experts") or (1 if c.get("shared_expert_intermediate_size") else 0)
    d_sh = c.get("shared_expert_intermediate_size") or (de if c.get("n_shared_experts") else 0)
    first_dense = c.get("first_k_dense_replace") or 0
    mlp_only = set(c.get("mlp_only_layers") or [])
    types = _list(c.get("layer_types")) or []
    window = c.get("sliding_window")
    layers = []
    for i in range(L):
        w = None
        if types and i < len(types) and types[i] == "sliding_attention":
            w = window
        elif mt in ("mistral", "mixtral") and window and window < (c.get("max_position_embeddings") or 0):
            w = window
        alin, core = _gqa(h, H, KV, hd, window=w)
        if mt == "gpt_oss":
            core = replace(core, sink=True)
        if E and i >= first_dense and i not in mlp_only:
            ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=bool(c.get("shared_expert_intermediate_size")))
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        misc = 2 * h + (2 * hd if mt.startswith("qwen3") else 0)
        if c.get("attention_bias") or mt == "qwen2" or mt == "gpt_oss":
            misc += H * hd + 2 * KV * hd + (h if mt == "gpt_oss" else 0)
        if mt == "gpt_oss":
            misc += H  # attention sinks
            misc += E * (2 * de + h)  # expert biases
        layers.append(Layer(alin, core, ffn, flin, misc))
    if mt == "gpt_oss":
        notes.append("attention sink 按参考实现建模（每个 query 行的 softmax 多一个可学习 logit，随后丢弃）；"
                     "128-token 交替滑窗已建模")
    if any(l.core.window for l in layers) and mt != "gpt_oss":
        notes.append(f"滑窗 {window} 已建模")
    mtp = []
    n_mtp = c.get("num_nextn_predict_layers") or 0
    for _ in range(n_mtp):
        alin, core = _gqa(h, H, KV, hd)
        ffn, flin = _ffn_moe(h, E, k, de, n_sh, d_sh) if E else _ffn_dense(h, c["intermediate_size"])
        mtp.append(Layer(alin, core, ffn, flin, 2 * h))
    return layers, mtp, notes, cov


def _build_mla(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """DeepSeek-V3/R1/V3.1/V3.2, Kimi-K2.x, GLM-5 (MLA + optional DSA indexer)."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    E = c.get("n_routed_experts") or 0
    first_dense = c.get("first_k_dense_replace") or 0
    notes, cov = [], "full"
    ilin, ih, idim, topk = _indexer(c, h, c.get("q_lora_rank"))
    itypes = c.get("indexer_types") or []
    fp8 = c.get("model_type") == "deepseek_v32"
    if ih:
        n_sh = sum(1 for t in itypes[:L] if t == "shared")
        notes.append(f"DSA lightning indexer（{ih}×{idim}）按参考实现逐项建模：q·k 打分 GEMM"
                     + ("（fp8，Hadamard 旋转 + 每 128 元素 fp32 scale）" if fp8 else "")
                     + f"、ReLU、按头加权求和、top-{topk} 选择；稀疏注意力只读 top-{topk} 个 token"
                     + (f"；{n_sh} 层为 shared（复用前一 full 层的 top-k，无索引器权重 / 缓存）" if n_sh else ""))

    def layer(i: int) -> Layer:
        alin, core, misc = _mla(c, h)
        if ih and i < len(itypes) and itypes[i] == "shared":
            core = replace(core, topk=topk)       # transformers GlmMoeDsa: reuse the previous full layer's top-k
        elif ih:
            alin = alin + ilin
            core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk, idx_fp8=fp8,
                           idx_rope=c.get("qk_rope_head_dim") or 0)
            misc += 2 * idim
        if E and i >= first_dense:
            ffn, flin = _ffn_moe(h, E, c["num_experts_per_tok"], c["moe_intermediate_size"],
                                 c.get("n_shared_experts") or 0, c["moe_intermediate_size"])
            misc += E  # e_score_correction_bias
        else:
            ffn, flin = _ffn_dense(h, c["intermediate_size"])
        return Layer(alin, core, ffn, flin, misc + 2 * h)

    layers = [layer(i) for i in range(L)]
    mtp = [replace(layer(L), misc_params=layer(L).misc_params + 3 * h) for _ in range(c.get("num_nextn_predict_layers") or 0)]
    return layers, mtp, notes, cov


def _gdn(c: dict, h: int) -> tuple[tuple[Linear, ...], AttnCore, int]:
    """Gated DeltaNet (Qwen3-Next / Qwen3.5)."""
    nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
    kern = c.get("linear_conv_kernel_dim") or 4
    qkv = 2 * nk * dk + nv * dv
    lin = (Linear("lin_qkv", h, qkv, "col"), Linear("lin_z", h, nv * dv, "col"),
           Linear("lin_ba", h, 2 * nv, "col"), Linear("lin_out", nv * dv, h, "row"))
    core = AttnCore("linear", n_state_heads=nv, state_dk=dk, state_dv=dv, conv_channels=qkv, conv_kernel=kern,
                    lin="gdn", lin_chunk=64)   # transformers torch_chunk_gated_delta_rule(chunk_size=64) / FLA
    misc = qkv * kern + 2 * nv + dv
    return lin, core, misc


def _kda(c: dict, h: int) -> tuple[tuple[Linear, ...], AttnCore, int]:
    """Kimi Delta Attention (Kimi-Linear / Kimi-K3 / GLM-5.x-Next)."""
    lc = _list(c.get("linear_attn_config")) or {}
    H, hd = lc.get("num_heads", c["num_attention_heads"]), lc.get("head_dim", 128)
    kern = lc.get("short_conv_kernel_size", 4)
    d = H * hd
    # output gate: full g_proj (Kimi-K3 release) or low-rank g_a / g_b (transformers Glm5NextTextLinearAttention)
    gate = ((Linear("kda_g_a", h, hd, "rep"), Linear("kda_g_b", hd, d, "col")) if c.get("model_type", "").startswith("glm5_next")
            else (Linear("kda_g", h, d, "col"),))
    lin = (Linear("kda_q", h, d, "col"), Linear("kda_k", h, d, "col"), Linear("kda_v", h, d, "col")) + gate + (
        Linear("kda_b", h, H, "col"), Linear("kda_f_a", h, hd, "rep"), Linear("kda_f_b", hd, d, "col"),
        Linear("kda_o", d, h, "row"))
    core = AttnCore("linear", n_state_heads=H, state_dk=hd, state_dv=hd, conv_channels=3 * d, conv_kernel=kern,
                    lin="kda", lin_chunk=64)   # transformers chunk_kimi_delta_attention(chunk_size=64) / FLA chunk_kda
    misc = 3 * d * kern + d + H + hd
    return lin, core, misc


def _build_hybrid(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Hybrid linear/full attention: Qwen3-Next, Qwen3.5/3.8, Kimi-K3, GLM-5.x-Next, MiniMax."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    mt = c.get("model_type", "")
    notes, cov = [], "full"
    types = _list(c.get("layer_types"))
    lc = _list(c.get("linear_attn_config")) or {}
    if not types:
        if lc.get("full_attn_layers"):
            full = {x - 1 for x in lc["full_attn_layers"]}   # Kimi: 1-based
            types = ["full_attention" if i in full else "linear_attention" for i in range(L)]
        elif c.get("full_attention_interval"):
            n = c["full_attention_interval"]
            types = ["full_attention" if (i + 1) % n == 0 else "linear_attention" for i in range(L)]
        elif c.get("attn_type_list"):
            types = ["full_attention" if t == 1 else "linear_attention" for t in c["attn_type_list"]]
    E = _g(c, "num_experts", "n_routed_experts", "num_local_experts", default=0)
    k = _g(c, "num_experts_per_tok", "num_experts_per_token")
    de = _g(c, "moe_intermediate_size", "intermediate_size")
    mlp_types = _list(c.get("mlp_layer_types"))
    first_dense = c.get("first_k_dense_replace") or 0
    kda = mt in ("kimi_linear",) or mt.startswith("glm5_next")
    is_minimax = mt.startswith("minimax")

    def ffn_for(i: int):
        dense = (mlp_types and mlp_types[i] == "dense") or (i < first_dense) or not E
        if dense:
            return _ffn_dense(h, c["intermediate_size"])
        if mt == "kimi_linear":
            return _ffn_moe(h, E, k, de, c.get("num_shared_experts") or 0, de,
                            latent=c.get("routed_expert_hidden_size") or 0)
        n_sh = c.get("n_shared_experts") or c.get("num_shared_experts") or (1 if c.get("shared_expert_intermediate_size") else 0)
        d_sh = c.get("shared_expert_intermediate_size") or (de if n_sh else 0)
        return _ffn_moe(h, E, k, de, n_sh, d_sh, shared_gate=bool(c.get("shared_expert_intermediate_size")))

    def full_attn(i: int):
        if c.get("kv_lora_rank"):
            alin, core, misc = _mla(c, h)
            ilin, ih, idim, topk = _indexer(c, h, c.get("q_lora_rank"))
            if ih and (not types or types[i] != "full_attention" or mt.startswith("glm5")):
                alin = alin + ilin
                core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk, idx_rope=c.get("qk_rope_head_dim") or 0)
                kp = c.get("index_kpool") or 0
                if kp and c.get("index_kpool_compress"):
                    # transformers Glm5NextTextIndexer: gate GEMM h → idx_dim, learned per-slot APE, k LayerNorm (w + b);
                    # top-(topk / kpool) pools, expanded back to tokens, + the incomplete tail pool (≤ kpool − 1 tokens)
                    alin = alin + (Linear("idx_kpool_gate", h, idim, "rep"),)
                    misc += kp * idim + 2 * idim
                    core = replace(core, idx_kpool=kp,
                                   topk=topk + (kp - 1 if c.get("index_kpool_always_select_tail") else 0))
            if c.get("mla_use_output_gate"):
                alin = alin + (Linear("o_gate", h, core.n_q * core.v_dim, "col"),)
            return alin, core, misc
        H = c["num_attention_heads"]
        KV = c.get("num_key_value_heads") or H
        hd = c.get("head_dim") or h // H
        alin, core = _gqa(h, H, KV, hd, out_gate=bool(c.get("attn_output_gate")) or is_minimax)
        return alin, core, 2 * hd

    def lin_attn(i: int):
        if kda:
            return _kda(c, h)
        if is_minimax:
            H = c["num_attention_heads"]
            hd = c.get("head_dim") or h // H
            lin = (Linear("lin_qkv", h, 3 * H * hd, "col"), Linear("lin_gate", h, H * hd, "col"),
                   Linear("lin_out", H * hd, h, "row"))
            # MiniMax remote code modeling_minimax_text_01.py: BLOCK = 256, kv state fp32
            return lin, AttnCore("linear", n_state_heads=H, state_dk=hd, state_dv=hd, lin="lightning",
                                 lin_chunk=256), H * hd
        return _gdn(c, h)

    layers = []
    for i in range(L):
        t = types[i] if types and i < len(types) else "full_attention"
        alin, core, misc = full_attn(i) if t != "linear_attention" else lin_attn(i)
        ffn, flin = ffn_for(i)
        if ffn.kind == "moe" and mt in ("kimi_linear",) or (ffn.kind == "moe" and c.get("n_routed_experts")):
            misc += E
        hc = c.get("hc_mult") or 0
        if hc and mt.startswith("glm5_next"):
            # transformers Glm5NextTextHyperConnection ×2 per layer (attn_hc, ffn_hc): fn [(2+N)·N, N·h], base, scale
            misc += 2 * ((2 + hc) * hc * (hc * h + 1) + 3)
            layers.append(Layer(alin, core, ffn, flin, misc + 2 * h, hc=hc, hc_iters=c.get("hc_sinkhorn_iters") or 1))
        else:
            layers.append(Layer(alin, core, ffn, flin, misc + 2 * h))
    n_lin = sum(1 for l in layers if l.core.kind == "linear")
    notes.append(f"混合注意力：{n_lin}/{L} 层线性注意力（chunk 形式按参考实现上阵列，递归状态 fp32 按参考），"
                 f"{L - n_lin} 层全注意力")
    n_mtp = c.get("mtp_num_hidden_layers") or c.get("num_nextn_predict_layers") or 0
    mtp = []
    for _ in range(n_mtp):
        alin, core, misc = full_attn(L - 1)
        ffn, flin = ffn_for(L - 1)
        mtp.append(Layer(alin, core, ffn, flin, misc + 4 * h))
    return layers, mtp, notes, cov


def _build_dsv4_ref(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """DeepSeek-V4-Flash / -Pro per the official inference/model.py (identical in both repos): MQA head_dim 512 (one
    shared K = V entry, attention sink, per-head q RMSNorm, inverse RoPE on o), grouped low-rank output, sliding
    window 128 + per-layer Compressor (ratio 4 overlapping / 128), ratio-4 layers with a compressed-key indexer
    (heads column-parallel, top-k), hash routing (gate scores still computed; experts from tid2eid), mHC ×hc_mult
    with a learned final collapse."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H, hd = c["num_attention_heads"], c["head_dim"]
    ql, ol, og = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    rope = c.get("qk_rope_head_dim") or 64
    win = c.get("sliding_window") or 128
    ratios = _list(c.get("compress_ratios")) or [0] * (L + 1)
    ih, idim, topk = c.get("index_n_heads") or 0, c.get("index_head_dim") or 0, c.get("index_topk")
    hc, iters = c.get("hc_mult") or 1, c.get("hc_sinkhorn_iters") or 1
    E, k, de = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    n_hash = c.get("num_hash_layers") or 0
    hc_p = 2 * ((2 + hc) * hc * (hc * h + 1) + 3)          # hc_attn / hc_ffn: fn, base, scale
    head_p = hc * hc * h + hc + 1                            # hc_head_fn / base / scale
    notes = [f"DeepSeek-V4 按官方 inference/model.py 逐项建模：滑窗 {win} + 压缩器（比 4 重叠 / 128）、比 4 层压缩 key 索引器"
             f"（{ih} 头 TP 切分 + 分数 all-reduce，top-{topk}）、attention sink、mHC ×{hc}（Sinkhorn {iters} 次）、"
             f"前 {n_hash} 层哈希路由（gate 分数照算，专家由 tid2eid 给定）"]

    def layer(i: int, last: bool = False) -> Layer:
        r = ratios[i] if i < len(ratios) else 0
        r = r if r and r > 1 else 1
        alin = [Linear("q_a", h, ql, "rep"), Linear("q_b", ql, H * hd, "col"), Linear("kv", h, hd, "rep"),
                Linear("o_a", H * hd, og * ol, "col", groups=og), Linear("o_b", og * ol, h, "row")]
        core = AttnCore("mla", n_q=H, n_kv=1, qk_dim=hd, v_dim=hd, kv_lora=hd, rope_dim=0, rope_n=rope,
                        window=win, compress=r, sink=True)
        misc = ql + hd + H                                   # q_norm, kv_norm, attn_sink
        if r > 1:
            cw = 2 if r == 4 else 1
            alin += [Linear("cmp_kv", h, cw * hd, "rep"), Linear("cmp_gate", h, cw * hd, "rep")]
            misc += cw * r * hd + hd                         # ape, norm
            core = replace(core, cmp_coff=cw)
            if r == 4 and ih:
                alin += [Linear("idx_q", ql, ih * idim, "col"), Linear("idx_cmp_kv", h, cw * idim, "rep"),
                         Linear("idx_cmp_gate", h, cw * idim, "rep"), Linear("idx_w", h, ih, "col")]
                misc += cw * r * idim + idim
                core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk, idx_split=True, idx_cmp=True,
                               idx_rope=rope)
        ffn, flin = _ffn_moe(h, E, k, de, c.get("n_shared_experts") or 0, de)
        misc += 2 * h + hc_p + (0 if i < n_hash else E)      # norms, hc, gate bias (hash layers: tid2eid instead)
        if last:
            misc += head_p
        return Layer(tuple(alin), core, ffn, flin, misc, hc=hc, hc_iters=iters, hc_head="mix" if last else "")

    layers = [layer(i, i == L - 1) for i in range(L)]
    mtp = []
    for j in range(c.get("num_nextn_predict_layers") or 0):
        m_ = layer(L + j)
        mtp.append(replace(m_, misc_params=m_.misc_params + 3 * h + head_p, hc_head="mix"))
    return layers, mtp, notes, "full"


def _build_qwen4exp(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """Qwen3.8-Flash-Next (qwen4_exp) per transformers modeling_qwen4_exp: Gated DeltaNet / gated-q GQA with a QSA
    indexer (pooled key blocks, top budget/ratio blocks + tail); hc_count residual streams with two GatedResidual
    mixers per layer (low-rank input mix + block injection; no per-layer RMSNorms) and a final mixer; PLE n-gram
    layer(s) (hashed table rows → key / value projections, stream-gated, dilated depthwise conv); sparse MoE with
    a sigmoid-gated shared expert."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    types = c.get("layer_types") or []
    N, r = c["hc_count"], c["hc_lowrank"]
    Nh = N * h
    H, KV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    rope = int(hd * (c.get("partial_rotary_factor") or 1))
    ih, ikv, idim = c["indexer_n_heads"], c["indexer_kv_heads"], c["indexer_head_dim"]
    budget, ratio = c["indexer_budget"], c["indexer_compress_ratio"]
    ple_ids = set(_list(c.get("ple_layer_ids")) or [])
    ng, hpn, ped = c.get("ngram_size") or 0, c.get("heads_per_ngram") or 0, c.get("ple_embed_dim") or 0
    prow = (ng - 1) * hpn
    pk = c.get("ple_conv_kernel_size") or 0
    E, k, de = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    notes = [f"Qwen4Exp 按 transformers modeling_qwen4_exp 逐项建模：门控残差 ×{N}（低秩 {r}）、QSA 索引注意力"
             f"（{ih} 头 × {idim}，块 {ratio}，预算 {budget} + 尾块）、PLE n-gram 层 {sorted(i - 1 for i in ple_ids)}"
             f"（每 token {prow} 行 × {ped // max(prow, 1)}，膨胀 conv {pk}）"]

    def gres(tag: str, combine: bool = True):
        lin = (Linear(f"gr_{tag}_down", Nh, r, "rep"), Linear(f"gr_{tag}_up", r, Nh, "rep"))
        if combine:
            lin += (Linear(f"gr_{tag}_inj", Nh, N, "rep"),)
        return lin, Nh                                # + hc_norm weight

    def full(i: int):
        alin, core = _gqa(h, H, KV, hd, out_gate=True)
        alin = alin + (Linear("idx_qk", h, (ih + ikv) * idim, "rep"),)
        core = replace(core, qsa=ratio, topk=budget, idx_heads=ih, idx_dim=idim, idx_rope=rope, rope_n=rope)
        return alin, core, 2 * hd + 2 * idim

    def layer(i: int, typ: str, last: bool = False) -> Layer:
        alin, core, misc = full(i) if typ == "full_attention" else _gdn(c, h)
        ga, ma = gres("a")
        gm, mm = gres("m")
        alin = tuple(alin) + ga + gm
        misc += ma + mm
        ffn, flin = _ffn_moe(h, E, k, de, 1, c["shared_expert_intermediate_size"], shared_gate=True)
        kw = {}
        if i + 1 in ple_ids:
            alin += (Linear("ple_key", ped, Nh, "rep"), Linear("ple_value", ped, h, "rep"))
            misc += 3 * Nh + Nh * pk
            kw = dict(ple_rows=prow, ple_row_dim=ped // prow, ple_conv=pk, ple_dil=ng)
        if last:
            gf, mf = gres("f", combine=False)
            alin += gf
            misc += mf
        return Layer(alin, core, ffn, flin, misc, gres_n=N, gres_head=last, **kw)

    layers = [layer(i, types[i] if i < len(types) else "linear_attention", i == L - 1) for i in range(L)]
    mc = c.get("mtp") or {}
    mtp = []
    for j in range(c.get("mtp_num_hidden_layers") or 0):
        typ = (mc.get("layer_types") or ["full_attention"])[j]
        m_ = layer(L + j, typ)
        gf, mf = gres("f", combine=False)
        # MTP (SGLang qwen4_exp_mtp.py _fuse_residual_linear_shared): fc_embedding(norm(embed)) + fc_hidden applied
        # to each of the N normed streams (fc weights = mtp_extra 2·h·h), pre-fc norms (h + N·h), one QSA full-attention
        # layer without PLE, its own hyper_connection_mixer
        mtp.append(replace(m_, attn_linears=m_.attn_linears + gf, misc_params=m_.misc_params + mf + h + Nh,
                           gres_head=True))
    return layers, mtp, notes, "full"


def _build_dsv41_ref(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """DeepSeek-V4.1-Flash per the official inference/model.py + engram.py: every layer MQA-512 over its sliding
    window (+ attention sink); compress_ratios > 0 layers also attend to top-k entries of a compressed KV that only
    kv_source_layers compute (Compressor: ratio 1 plain projection, ratio > 1 fp32 softmax pooling, no APE) and
    store; index_source_layers run an indexer (index keys from wk on the source latent), the layers in between reuse
    its top-k; candidate_source_layer adds block pre-selection (select_candidate_blocks) used by later indexers;
    Engram n-gram tables (fp8 rows + e8m0 scales) at engram_layer_ids; mHC ×hc_mult with the final collapse by the
    last pre_mix (no head projection); DSpark block draft head (mtp.*)."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H, hd = c["num_attention_heads"], c["head_dim"]
    ql, ol, og = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    rope = c.get("qk_rope_head_dim") or 64
    win = c.get("sliding_window") or 128
    ratios = _list(c.get("compress_ratios")) or [0] * L
    ih, idim, topk = c["index_n_heads"], c["index_head_dim"], c["index_topk"]
    kv_src = set(_list(c.get("kv_source_layer_ids")) or [])
    idx_src = set(_list(c.get("index_source_layer_ids")) or [])
    cand_src = c.get("candidate_source_layer_id", -1)
    hc, iters = c.get("hc_mult") or 1, c.get("hc_sinkhorn_iters") or 1
    E, k, de = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    eng_ids = set(_list(c.get("engram_layer_ids")) or [])
    ecols = (c.get("engram_max_ngram_size", 1) - 1) * (c.get("engram_n_heads") or 0)
    edim = c.get("engram_head_dim") or 0
    hc_p = 2 * ((2 + hc) * hc * (hc * h + 1) + 3)
    kvb = 2.0       # reference caches are default-dtype (bf16) buffers; fp8 / fp4 act_quant is in-place simulation
    notes = [f"DeepSeek-V4.1 按官方 inference/model.py + engram.py 逐项建模：每层滑窗 {win} + attention sink；"
             f"压缩层另读 top-{topk} 压缩条目（keys = min(p, w) + min(top-k, ⌊p/r⌋)），压缩 KV 只在 {sorted(kv_src)} 层计算 / 存储，"
             f"索引器只在 {sorted(idx_src)} 层运行（其余层复用 top-k），第 {cand_src} 层候选块预选；Engram 查表层 "
             f"{sorted(eng_ids)}（每 token {ecols} 行 × {edim} fp8 + e8m0 scale）；mHC ×{hc}；DSpark 块草稿头"]

    def attn(i: int, r: int, backbone: bool = True):
        alin = [Linear("q_a", h, ql, "rep"), Linear("q_b", ql, H * hd, "col"), Linear("kv", h, hd, "rep"),
                Linear("o_a", H * hd, og * ol, "col", groups=og), Linear("o_b", og * ol, h, "row")]
        core = AttnCore("mla", n_q=H, n_kv=1, qk_dim=hd, v_dim=hd, kv_lora=hd, rope_dim=0, rope_n=rope,
                        window=win, sink=True, win_bytes=hd * kvb)
        misc = ql + hd + H
        if r and backbone:
            own = i in kv_src
            core = replace(core, dual=True, compress=r, topk=topk, cmp_bytes=hd * kvb, kv_owner=own,
                           cmp_coff=1 if own else 0, cmp_noape=True)
            if own:
                alin += [Linear("cmp_kv", h, hd, "rep")] + ([Linear("cmp_gate", h, hd, "rep")] if r > 1 else [])
                misc += hd
            if i in idx_src:
                alin += [Linear("idx_q", ql, ih * idim, "col"), Linear("idx_w", h, ih, "col")]
                core = replace(core, idx_heads=ih, idx_dim=idim, idx_split=True, idx_rope=rope, idx_owner=own,
                               idx_kbytes=idim * kvb,
                               cand="src" if i == cand_src else ("use" if 0 <= cand_src < i else ""),
                               cand_block=c.get("candidate_block_size") or 0,
                               cand_topk=c.get("candidate_topk_blocks") or 0)
                if own:
                    misc += hd * idim + idim         # wk (on compressed latents) + k_norm
        return alin, core, misc

    layers = []
    for i in range(L):
        r = ratios[i] if i < len(ratios) else 0
        alin, core, misc = attn(i, r)
        ec = 0
        if i in eng_ids:
            alin.append(Linear("engram_wkv", ecols * edim, h * (hc + 1), "rep"))
            misc += 2 * hc * h                       # q_weight, k_weight
            ec = ecols
        ffn, flin = _ffn_moe(h, E, k, de, c.get("n_shared_experts") or 0, de)
        misc += 2 * h + hc_p + 2 * E                 # norms, hc, gate bias + bias_vl
        layers.append(Layer(tuple(alin), core, ffn, flin, misc, hc=hc, hc_iters=iters, engram_cols=ec,
                            engram_dim=edim, hc_head="pre" if i == L - 1 else ""))
    mtp = []
    n_mtp = c.get("num_nextn_predict_layers") or 0
    Ed, kd = c.get("dspark_n_routed_experts") or E, c.get("dspark_num_experts_per_tok") or k
    for j in range(n_mtp):
        alin, core, misc = attn(L + j, 0, backbone=False)
        ffn, flin = _ffn_moe(h, Ed, kd, de, c.get("n_shared_experts") or 0, de)
        misc += 2 * h + hc_p + 2 * Ed
        mtp.append(Layer(tuple(alin), core, ffn, flin, misc, hc=hc, hc_iters=iters))
    return layers, mtp, notes, "full"


def _build_dsv4(c: dict) -> tuple[list[Layer], list[Layer], list[str], str]:
    """DeepSeek-V4 family: MQA head_dim 512, grouped low-rank output, compressed
    sparse attention (CSA) with per-layer compress ratio, sliding window, hash
    routing on the first layers, hyper-connections.  「架构代理」."""
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    H, hd = c["num_attention_heads"], c["head_dim"]
    ql, ol, og = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    rope = c.get("qk_rope_head_dim") or 64
    win = c.get("sliding_window") or 128
    ratios = _list(c.get("compress_ratios")) or [1] * L
    ih, idim, topk = c.get("index_n_heads") or 0, c.get("index_head_dim") or 0, c.get("index_topk")
    hc = c.get("hc_mult") or 1
    E, k, de = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
    notes = ["「架构代理」DeepSeek-V4：压缩稀疏注意力按有效上下文 ctx/ratio 近似（+ 窗口，索引层 ≤ top-k）；"
             "超连接（×%d 残差流）只计参数、不计混合计算；哈希路由层按 top-k MoE 处理" % hc]
    layers = []
    for i in range(L):
        r = ratios[i] if i < len(ratios) else 1
        r = r if r and r > 1 else 1
        alin = [Linear("q_a", h, ql, "rep"), Linear("q_b", ql, H * hd, "col"), Linear("kv", h, hd, "rep"),
                Linear("o_a", H * hd, og * ol, "col", groups=og), Linear("o_b", og * ol, h, "row")]
        core = AttnCore("mla", n_q=H, n_kv=1, qk_dim=hd, v_dim=hd, kv_lora=hd - rope, rope_dim=rope,
                        window=win, compress=r)
        misc = ql + hd + H + 2 * (2 * hc + hc * hc) * hc * h // max(1, hc) // 1
        if r > 1:
            cw = 2 if r == 4 else 1      # overlapping compression for small ratios
            alin += [Linear("cmp_kv", h, cw * hd, "rep"), Linear("cmp_gate", h, cw * hd, "rep")]
            misc += cw * r * hd + hd
            if r == 4 and ih:
                alin += [Linear("idx_q", ql, ih * idim, "rep"), Linear("idx_cmp_kv", h, cw * idim, "rep"),
                         Linear("idx_cmp_gate", h, cw * idim, "rep"), Linear("idx_w", h, ih, "rep")]
                core = replace(core, idx_heads=ih, idx_dim=idim, topk=topk)
        ffn, flin = _ffn_moe(h, E, k, de, c.get("n_shared_experts") or 0, de)
        layers.append(Layer(tuple(alin), core, ffn, flin, misc + 2 * h + E))
    mtp = [replace(layers[-1], misc_params=layers[-1].misc_params + 2 * h * h + 3 * h)
           for _ in range(c.get("num_nextn_predict_layers") or 0)]
    return layers, mtp, notes, "proxy"


def _builder(c: dict):
    mt = (c.get("model_type") or "").lower()
    if mt.startswith("qwen4_exp"):
        return _build_qwen4exp, "混合注意力（GDN + QSA）+ 门控残差 / PLE"
    if mt.startswith("deepseek_v41"):
        return _build_dsv41_ref, "DeepSeek-V4.1 (CSA/MQA-512 + Engram)"
    if mt == "deepseek_v4":
        return _build_dsv4_ref, "DeepSeek-V4 (CSA/MQA-512)"
    if mt.startswith("deepseek_v4"):
        return _build_dsv4, "DeepSeek-V4 (CSA/MQA-512)"
    if mt in ("kimi_linear", "qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
              "qwen4_exp", "qwen4_exp_text", "minimax_text_01", "minimax_m1", "minimax") or mt.startswith("glm5_next"):
        return _build_hybrid, "混合注意力（linear + full）"
    if c.get("kv_lora_rank"):
        return _build_mla, "MLA" + (" + DSA" if c.get("index_n_heads") else "") + (" MoE" if c.get("n_routed_experts") else "")
    return _build_std, "GQA" + (" MoE" if _g(c, "num_local_experts", "num_experts", "n_routed_experts") else " 稠密")


# ---------------------------------------------------------------- release formats
_ROLE_KEYS = ("attn", "mlp", "expert", "shared_expert", "router", "embed", "lm_head", "mtp")


def _release_formats(rel: dict) -> dict[str, RoleFormat]:
    out = {}
    for role in _ROLE_KEYS:
        v = rel.get("roles", {}).get(role)
        if not v:
            continue
        fmt, e = max(v.items(), key=lambda kv: kv[1]["params"])
        p = sum(x["params"] for x in v.values())
        b = sum(x["bytes"] for x in v.values())
        if p:
            name = {"uint8": "int8", "int32": "int4"}.get(fmt, fmt)
            out[role] = RoleFormat(name if name in ("fp32", "bf16", "fp16", "fp8", "mxfp4", "int8", "int4") else "bf16",
                                   8.0 * b / p)
    return out


def _kv_fmt(rel: dict) -> str:
    qc = rel.get("quantization_config") or {}
    if isinstance(qc, dict) and qc.get("kv_cache_scheme"):
        return "fp8"
    return "bf16"


def _quant_release(rel: dict) -> bool:
    return bool(rel.get("quantization_config"))


def _act_fmt(rel: dict) -> str:
    """Activation format: W8A8 block-FP8 releases with dynamic activation scaling
    run activations in fp8 (DeepSeek reference inference); all else bf16."""
    qc = rel.get("quantization_config") or {}
    if isinstance(qc, dict) and qc.get("quant_method") == "fp8" and qc.get("activation_scheme") == "dynamic":
        return "fp8"
    return "bf16"


# ---------------------------------------------------------------- public API
def release_path(hf_id: str) -> Path:
    return RELEASES / (hf_id.replace(":", "_").replace("/", "__") + ".json")


@lru_cache(maxsize=None)
def load_release(hf_id: str) -> dict | None:
    p = release_path(hf_id)
    return json.loads(p.read_text()) if p.exists() else None


def _coverage_reasons(c: dict, spec: ModelSpec, gap: float) -> tuple[str, ...]:
    """Concrete per-model list of approximated mechanisms (shown with the coverage label)."""
    mt = c.get("model_type", "")
    L = len(spec.layers)
    r = []
    cmp = sum(1 for l in spec.layers if l.core.compress > 1 and not (l.core.cmp_coff or l.core.dual))
    idx = [l.core for l in spec.layers if l.core.idx_heads]
    if cmp:
        r.append(f"压缩稀疏注意力 CSA（{cmp}/{L} 层）：按有效上下文 ctx/压缩比 + 滑窗近似"
                 + (f"，索引层只读 top-{idx[0].topk} 个 token" if idx else ""))
    if c.get("num_hash_layers") and mt != "deepseek_v4":
        r.append(f"哈希路由（前 {c['num_hash_layers']} 层）：按 top-k MoE 计")
    hc = next((c.get(k) for k in ("hc_mult", "hc_count", "mhc") if c.get(k)), None)
    if hc and not any(l.hc or l.gres_n for l in spec.layers):
        r.append(f"超连接（多流残差 ×{hc}）：参数计入，多流混合计算未计")
    if spec.lookup_params and not any(l.engram_cols or l.ple_rows for l in spec.layers):
        r.append(f"n-gram / engram 查表（{spec.lookup_params / 1e9:.1f}B 参数）：计存储，每 token 只读少量行")
    if abs(gap) > 0.02:
        r.append(f"参数与发布相差 {gap:+.1%}：模板未复现的部分按发布计存储")
    return tuple(r)


def from_release(model_id: str, hf_id: str, rel: dict | None = None, cfg: dict | None = None) -> ModelSpec:
    rel = rel if rel is not None else load_release(hf_id)
    if cfg is None:
        if rel is None:
            raise ValueError(f"no release metadata for {hf_id}; run scripts/fetch_hf_release.py")
        cfg = rel["config"]
    c = _tc(cfg)
    build, arch = _builder(c)
    layers, mtp, notes, cov = build(c)
    h = c["hidden_size"]
    tie = bool(c.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    spec = ModelSpec(
        id=model_id, hf_id=hf_id, arch=arch, hidden=h, vocab=c["vocab_size"], layers=tuple(layers),
        tie_embeddings=tie, mtp_layers=tuple(mtp), mtp_extra_params=2 * h * h if mtp else 0,
        coverage=cov, notes=tuple(notes))
    if rel is not None:
        fm = _release_formats(rel)
        spec = replace(spec, formats=tuple(sorted(fm.items())), kv_fmt=_kv_fmt(rel), act_fmt=_act_fmt(rel),
                       release_params=rel.get("params_llm"), release_mtp_params=rel.get("params_mtp"),
                       release_bytes=rel.get("bytes_llm"), quantized_release=_quant_release(rel))
        # lookup-only tables (n-gram / engram embeddings): stored, read a few rows per token
        emb_rel = sum(v["params"] for v in rel.get("roles", {}).get("embed", {}).values())
        if emb_rel > 1.05 * spec.embed_params:
            spec = replace(spec, lookup_params=emb_rel - spec.embed_params,
                           notes=spec.notes + (f"查表 {(emb_rel - spec.embed_params) / 1e9:.1f}B 参数（n-gram / engram），"
                                               "按发布 safetensors 头计入存储；每 token 只读少量行",))
        # MTP modules: stored size as released (incl. embed/head copies some releases store)
        rmtp = rel.get("params_mtp") or 0
        if rmtp and not spec.mtp_layers:
            # weights ship an MTP module the config does not declare (e.g. Qwen3-Next) → infer one
            last = spec.layers[-1]
            spec = replace(spec, mtp_layers=(replace(last, misc_params=last.misc_params + 2 * h),),
                           notes=spec.notes + ("MTP 模块由发布权重推断（config 中没有 MTP 字段）",))
        head_copy = 0
        if tie and "lm_head" in rel.get("roles", {}):
            head_copy = sum(v["params"] for v in rel["roles"]["lm_head"].values())
        if spec.mtp_layers and rmtp:
            per = spec.mtp_layers[0].params(h)
            n_eff = max(1, min(len(spec.mtp_layers), round(rmtp / max(per, 1))))
            extra = max(0, (rmtp - n_eff * per) // n_eff)
            spec = replace(spec, mtp_layers=spec.mtp_layers[:n_eff], mtp_extra_params=extra)
        if (c.get("model_type") or "").startswith("qwen4_exp"):
            spec = replace(spec, final_norm_params=0)    # the final hyper_connection_mixer replaces the last norm
        if (c.get("model_type") or "").startswith("deepseek_v41"):
            # Engram tables stay fp8 with an e8m0 scale per 32 (ParallelEngramEmbedding) → 8.25 bits / param stored
            spec = replace(spec, lookup_bits=8 + 8 / 32, engram_ngram=c.get("engram_max_ngram_size") or 0,
                           draft="dspark" if c.get("dspark_block_size") else "",
                           draft_block=c.get("dspark_block_size") or 0,
                           draft_targets=len(_list(c.get("dspark_target_layer_ids")) or []),
                           draft_markov=c.get("dspark_markov_rank") or 0)
        exotic = [k for k in ("hc_mult", "hc_count", "mhc") if c.get(k)] \
            if not any(l.hc or l.gres_n for l in spec.layers) else []
        lk_open = spec.lookup_params and not spec.lookup_bits and not any(l.ple_rows for l in spec.layers)
        if exotic or lk_open:
            why = []
            if exotic:
                why.append("超连接（多流残差）：参数计入，混合计算未计")
            if lk_open:
                why.append("n-gram / engram 查表：只计存储")
            spec = replace(spec, coverage="proxy", notes=spec.notes + ("「架构代理」：" + "；".join(why),))
        if head_copy:
            spec = replace(spec, release_params=spec.release_params - head_copy,
                           notes=spec.notes + (f"发布另存了一份 tied lm_head（{head_copy / 1e9:.3f}B），"
                                               "已去重",))
        if spec.release_params:
            gap = spec.release_params - spec.params()
            if abs(gap) / spec.release_params > 0.02:
                # parameters present in the release that the template does not reproduce
                spec = replace(spec, coverage="proxy" if cov != "full" or abs(gap) / spec.release_params > 0.05 else cov,
                               notes=spec.notes + (f"参数与发布相差 {gap / 1e9:+.2f}B "
                                                   f"（{gap / spec.release_params:+.1%}），见 docs/MODEL.md",))
        rel_gap = (spec.release_params - spec.params()) / spec.release_params if spec.release_params else 0.0
        mp = c.get("max_position_embeddings")
        from .vision import vision_spec
        vs = vision_spec(hf_id, cfg)
        spec = replace(spec, coverage_reasons=_coverage_reasons(c, spec, rel_gap),
                       vision=vs, vision_params=vs.params if vs is not None else (rel.get("params_vision") or 0),
                       max_ctx=int(mp) if isinstance(mp, (int, float)) and not isinstance(mp, bool) and mp > 0 else 0)
        if spec.coverage == "full" and spec.coverage_reasons:
            spec = replace(spec, coverage="partial")
    return spec


def with_formats(spec: ModelSpec, overrides: dict[str, str]) -> ModelSpec:
    """What-if: replace role formats (labelled ``what_if``)."""
    from .dtypes import fmt as _fmt
    d = dict(spec.formats)
    for role, name in overrides.items():
        if role in ("act", "kv"):
            continue
        d[role] = RoleFormat(name, _fmt(name).bits)
    return replace(spec, formats=tuple(sorted(d.items())), what_if=True,
                   act_fmt=overrides.get("act", spec.act_fmt), kv_fmt=overrides.get("kv", spec.kv_fmt))
