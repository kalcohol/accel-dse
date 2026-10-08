"""GQA dense / MoE transformer shape descriptors.

Built-in ``ILLUSTRATIVE_*`` / ``TOY_SHAPE`` are placeholders for handcheck.
Real public HF dims are registered via ``npu_dse.catalog`` / ``series``
(``hf:<id>`` in metadata). FLOPs/BW remain uncalibrated either way.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelShape:
    """GQA Transformer — dense (n_experts=1) or MoE (n_experts>1).

    Parameter count (approx, ignoring biases / norms / embeddings detail):
      attn: L * (H*H + 2*H*(n_kv*d) + H*H)   # Q, K, V, O  with GQA
      ffn dense:  L * (2*H*F + F*H)            # SwiGLU: gate, up, down
      ffn MoE:    L * (n_shared + E) * (2*H*F + F*H)   # stored (all experts)
      active MoE: L * (n_shared + top_k) * FFN_expert   # per-token (balanced)
      embed: V*H  (counted once; LM head often tied — we count once)

    MoE traffic honesty (assumed balanced routing):
      Each token streams attn + shared FFN + top_k expert FFNs — NOT all E.
      Total params still count all experts for capacity accounting.
      With expert-parallel (EP) in scaleup: experts sharded E/ep per rank;
      active stream uses top_k/ep FFN worth per rank; A2A dispatch/combine
      modeled separately (see scaleup.moe_alltoall_bytes).

    Notes:
      - head_dim * n_heads == hidden is typical but not enforced.
      - n_kv <= n_heads (GQA / MQA).
      - n_experts=1, top_k=1 → dense (legacy path).
    """

    name: str
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    # dtype bits for weights / activations / KV (assumed, uncalibrated)
    weight_bits: int = 16
    act_bits: int = 16
    kv_bits: int = 16
    # MoE (defaults = dense)
    n_experts: int = 1
    top_k: int = 1
    n_shared_experts: int = 0
    # MLA / compressed KV (0 = standard GQA full K+V cache)
    # When > 0: per-token KV bytes = L * (kv_lora_rank + qk_rope_head_dim) * kv_bits/8
    # (joint latent c_kv + decoupled RoPE key k_pe, as cached by DeepSeek-V3).
    kv_lora_rank: int = 0
    # v0.30 MLA / low-rank attention projections (HF config names; 0 = absent).
    #   q_lora_rank      : Q = W_qb · norm(W_qa · x)   (DeepSeek-V2/V3, Kimi, GLM-5, V4)
    #   qk_nope_head_dim : per-head non-RoPE q/k width (MLA)
    #   qk_rope_head_dim : per-head decoupled RoPE width (shared k_pe for MLA)
    #   v_head_dim       : per-head value width (MLA; else head_dim)
    #   o_lora_rank / o_groups : grouped low-rank output projection (DeepSeek-V4
    #                      configs; structure inferred from field names — assumed)
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    o_lora_rank: int = 0
    o_groups: int = 0
    # v0.30 dense prefix layers in MoE models (HF ``first_k_dense_replace``):
    # the first n_dense_layers use a dense SwiGLU FFN of width dense_intermediate.
    n_dense_layers: int = 0
    dense_intermediate: int = 0
    # Embedding / LM head: tied → counted once; untied (HF tie_word_embeddings
    # = false, all public HF LLM packs here) → embed + lm_head both stored.
    tie_embeddings: bool = True
    # v0.31 MTP / nextn predict layers (HF ``num_nextn_predict_layers``). Their
    # weights are NOT part of the body; counted only when speculative decoding
    # with draft="mtp" is enabled (capacity + draft step cost).
    n_mtp_layers: int = 0

    def __post_init__(self) -> None:
        if self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads must be <= n_heads")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("head counts must be positive")
        if self.weight_bits <= 0 or self.act_bits <= 0 or self.kv_bits <= 0:
            raise ValueError("bit widths must be positive")
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        if self.n_shared_experts < 0:
            raise ValueError("n_shared_experts must be non-negative")
        if self.kv_lora_rank < 0:
            raise ValueError("kv_lora_rank must be non-negative")
        for f in ("q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                  "o_lora_rank", "o_groups", "n_dense_layers", "dense_intermediate",
                  "n_mtp_layers"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be non-negative")
        if self.n_dense_layers > self.n_layers:
            raise ValueError("n_dense_layers must be <= n_layers")
        if self.head_dim * self.n_heads != self.hidden:
            # Allow mismatch but warn via property; keep flexible for toys.
            pass

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1

    @property
    def is_mla(self) -> bool:
        """True when using compressed latent KV (kv_lora_rank > 0)."""
        return self.kv_lora_rank > 0

    @property
    def kv_dim(self) -> int:
        """Per-layer K (or V) projection output width."""
        return self.n_kv_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    # --- attention projections (v0.30: MLA / low-rank aware) ---------------
    @property
    def attn_kind(self) -> str:
        """'mla' (latent KV), 'lowrank' (low-rank Q/O, no kv latent, e.g.
        DeepSeek-V4 configs), or 'gqa' (standard MHA/GQA/MQA)."""
        if self.kv_lora_rank > 0 and (self.qk_nope_head_dim > 0 or self.v_head_dim > 0):
            return "mla"
        if self.q_lora_rank > 0 or self.o_lora_rank > 0:
            return "lowrank"
        return "gqa"

    @property
    def qk_head_dim(self) -> int:
        """Per-head q/k width (MLA: nope + rope; else head_dim)."""
        if self.attn_kind == "mla":
            return self.qk_nope_head_dim + self.qk_rope_head_dim
        return self.head_dim

    @property
    def v_dim_per_head(self) -> int:
        if self.attn_kind == "mla" and self.v_head_dim > 0:
            return self.v_head_dim
        return self.head_dim

    def attn_proj_shapes(self) -> list[tuple[str, int, int]]:
        """Attention projection matrices as (name, K_in, N_out) per layer.

        gqa:     q (H→nh·d), k (H→n_kv·d), v (H→n_kv·d), o (nh·d→H)
        mla:     q_a (H→q_lora) + q_b (q_lora→nh·(nope+rope))   [or q (H→nh·qk) if no q_lora]
                 kv_a (H→kv_lora+rope), kv_b (kv_lora→nh·(nope+v)), o (nh·v→H)
        lowrank: as gqa but q/o low-rank when q_lora_rank / o_lora_rank given;
                 grouped O: nh·d·o_lora + o_groups·o_lora·H (assumed from field names)
        Norm weights (q_a/kv_a layernorm) ignored (< 0.01 %).
        """
        h, nh = self.hidden, self.n_heads
        kind = self.attn_kind
        out: list[tuple[str, int, int]] = []
        qk = self.qk_head_dim
        if self.q_lora_rank > 0:
            out += [("q_a", h, self.q_lora_rank), ("q_b", self.q_lora_rank, nh * qk)]
        else:
            out += [("q", h, nh * qk)]
        if kind == "mla":
            rope = self.qk_rope_head_dim
            out += [
                ("kv_a", h, self.kv_lora_rank + rope),
                ("kv_b", self.kv_lora_rank, nh * (self.qk_nope_head_dim + self.v_dim_per_head)),
            ]
        else:
            out += [("k", h, self.kv_dim), ("v", h, self.kv_dim)]
        vo = nh * self.v_dim_per_head
        if self.o_lora_rank > 0:
            g = max(self.o_groups, 1)
            out += [("o_a", vo, self.o_lora_rank), ("o_b", g * self.o_lora_rank, h)]
        else:
            out += [("o", vo, h)]
        return out

    def attn_weight_params_per_layer(self) -> int:
        """Attention projection params per layer (v0.30: MLA / low-rank aware).

        Legacy GQA: Q + K + V + O = H·qd + 2·H·kvd + qd·H (unchanged).
        """
        return sum(k * n for _name, k, n in self.attn_proj_shapes())

    # --- FFN (v0.30: dense-prefix layers in MoE models) ----------------------
    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense_layers if self.is_moe else 0

    def dense_ffn_params(self) -> int:
        """SwiGLU params of one dense-prefix FFN (0 if no dense prefix)."""
        if self.n_dense_layers <= 0:
            return 0
        f = self.dense_intermediate or self.intermediate
        return 3 * self.hidden * f

    def ffn_weight_params_per_expert(self) -> int:
        """SwiGLU params for one expert (or the dense FFN): gate+up+down."""
        h, f = self.hidden, self.intermediate
        return 2 * h * f + f * h

    def _moe_ffn_stored(self) -> int:
        return (self.n_shared_experts + self.n_experts) * self.ffn_weight_params_per_expert()

    def _moe_ffn_active(self) -> int:
        return (self.n_shared_experts + self.top_k) * self.ffn_weight_params_per_expert()

    def _ffn_avg(self, moe_part: int) -> int:
        """Layer-average FFN params: dense prefix + MoE layers (exact when no prefix)."""
        nd = self.n_dense_layers
        if nd <= 0:
            return moe_part
        total = nd * self.dense_ffn_params() + (self.n_layers - nd) * moe_part
        return total // self.n_layers

    def ffn_weight_params_per_layer(self) -> int:
        """Total *stored* FFN params per layer (all experts + shared).

        v0.30: layer average when the model has dense prefix layers.
        """
        return self._ffn_avg(self._moe_ffn_stored())

    def ffn_active_weight_params_per_layer(self) -> int:
        """FFN params touched per token (balanced routing: shared + top_k)."""
        return self._ffn_avg(self._moe_ffn_active())

    def embed_params(self) -> int:
        """Embedding table (+ untied LM head)."""
        e = self.vocab * self.hidden
        return e if self.tie_embeddings else 2 * e

    def lm_head_params(self) -> int:
        """Output projection (LM head) V·H — read once per decode step (v0.31).

        Tied embeddings reuse the embedding matrix as the head; untied models
        store a separate head (already inside ``embed_params``)."""
        return self.vocab * self.hidden

    def experts_touched(self, tokens: float, n_local: int | None = None) -> float:
        """Expected distinct routed experts hit by ``tokens`` tokens (v0.31).

        「假设」uniform, independent top_k routing:
            E_hit = n_local · (1 − (1 − top_k/E)^tokens)
        tokens=1 → n_local·top_k/E (the ≤0.30 "top_k/ep" stream); large batches
        approach all n_local experts (every local expert's weights are read
        once per step). Dense shapes return 0.
        """
        if not self.is_moe or tokens <= 0:
            return 0.0
        n = self.n_experts if n_local is None else int(n_local)
        p_miss = (1.0 - self.top_k / self.n_experts) ** float(tokens)
        return n * (1.0 - p_miss)

    def ffn_stream_params_per_layer_tokens(self, tokens: float) -> int:
        """Layer-average FFN params read for one step of ``tokens`` tokens
        (shared + expected touched routed experts; dense FFN for dense shapes)."""
        if not self.is_moe:
            return self.ffn_weight_params_per_layer()
        p = self.ffn_weight_params_per_expert()
        moe = int(round((self.n_shared_experts + self.experts_touched(tokens)) * p))
        return self._ffn_avg(moe)

    def stream_weight_bytes_per_layer_tokens(self, tokens: float) -> int:
        """Per-layer weight bytes streamed for a decode step of ``tokens`` tokens
        (v0.31 batch-aware MoE expert coverage; tokens=1 → legacy stream unit)."""
        if not self.is_moe or tokens <= 1:
            return self.stream_weight_bytes_per_layer()
        return (
            self.attn_weight_params_per_layer()
            + self.ffn_stream_params_per_layer_tokens(tokens)
        ) * self.weight_bits // 8

    def mtp_layer_params(self) -> int:
        """Stored params of one MTP (nextn) module: one transformer layer (MoE
        layer for MoE shapes) + eh_proj (2H→H). Embedding / head are shared."""
        moe = self._moe_ffn_stored() if self.is_moe else self.ffn_weight_params_per_expert()
        return self.attn_weight_params_per_layer() + moe + 2 * self.hidden * self.hidden

    def weight_params_per_layer(self) -> int:
        """Total *stored* params per layer (attn + all experts)."""
        return self.attn_weight_params_per_layer() + self.ffn_weight_params_per_layer()

    def active_weight_params_per_layer(self) -> int:
        """Params streamed / activated per layer per token (attn + active FFN)."""
        return (
            self.attn_weight_params_per_layer()
            + self.ffn_active_weight_params_per_layer()
        )

    def body_weight_params(self) -> int:
        """Exact stored body params (attention + dense-prefix FFN + MoE FFN)."""
        nd = self.n_dense_layers
        return (
            self.n_layers * self.attn_weight_params_per_layer()
            + nd * self.dense_ffn_params()
            + (self.n_layers - nd) * self._moe_ffn_stored()
        )

    def total_weight_params(self) -> int:
        """Transformer body (all experts) + embedding (+ untied LM head)."""
        return self.body_weight_params() + self.embed_params()

    def active_weight_params(self) -> int:
        """Active body params + embed (illustrative 'active parameter' count)."""
        nd = self.n_dense_layers
        return (
            self.n_layers * self.attn_weight_params_per_layer()
            + nd * self.dense_ffn_params()
            + (self.n_layers - nd) * self._moe_ffn_active()
            + self.embed_params()
        )

    def weight_bytes(self) -> int:
        return self.total_weight_params() * self.weight_bits // 8

    def weight_bytes_per_layer(self) -> int:
        """Stored bytes per layer (all experts) — capacity accounting."""
        return self.weight_params_per_layer() * self.weight_bits // 8

    def active_weight_bytes_per_layer(self) -> int:
        """Bytes streamed per layer per token under balanced top_k routing.

        Used for decode/prefill weight DRAM traffic and staging sizing.
        Expert-parallel residency of unused experts is *not* modeled
        (assumed: stream top_k each token).
        """
        return self.active_weight_params_per_layer() * self.weight_bits // 8

    def stream_weight_bytes_per_layer(self) -> int:
        """Alias: per-layer external stream unit (active for MoE, =stored if dense)."""
        return self.active_weight_bytes_per_layer()

    def kv_bytes_per_token(self) -> int:
        """Bytes to store K+V (or MLA latent) for one token across all layers.

        GQA/MQA (kv_lora_rank=0): 2 * L * n_kv * head_dim * kv_bytes.
        MLA (kv_lora_rank>0): L * kv_lora_rank * kv_bytes — joint compressed
        latent (DeepSeek-style c_kv illustrative; assumed, not a checkpoint claim).
        """
        if self.kv_lora_rank > 0:
            return (
                self.n_layers
                * (self.kv_lora_rank + self.qk_rope_head_dim)
                * self.kv_bits
                // 8
            )
        return (
            2
            * self.n_layers
            * self.n_kv_heads
            * self.head_dim
            * self.kv_bits
            // 8
        )

    def experts_per_rank(self, ep: int) -> int:
        """Local expert count when experts are sharded across ``ep`` ranks."""
        if ep < 1:
            raise ValueError("ep must be >= 1")
        if self.n_experts % ep != 0:
            raise ValueError(
                f"n_experts={self.n_experts} must be divisible by ep={ep}"
            )
        return self.n_experts // ep

    def ffn_active_weight_params_per_layer_ep(self, ep: int = 1) -> int:
        """Active FFN params streamed per layer on one EP rank (balanced).

        Shared experts replicated; top_k expert evaluations load-balanced
        across ``ep`` ranks → ``top_k / ep`` expert FFNs worth of stream
        (integer via ``(top_k * ffn_params) // ep``).
        When ep=1: identical to ``ffn_active_weight_params_per_layer``.
        """
        if ep < 1:
            raise ValueError("ep must be >= 1")
        shared = self.n_shared_experts * self.ffn_weight_params_per_expert()
        top = self.top_k * self.ffn_weight_params_per_expert()
        return self._ffn_avg(shared + top // ep)

    def active_weight_params_per_layer_ep(self, ep: int = 1) -> int:
        return (
            self.attn_weight_params_per_layer()
            + self.ffn_active_weight_params_per_layer_ep(ep)
        )

    def stream_weight_bytes_per_layer_ep(self, ep: int = 1) -> int:
        """Per-layer external stream unit on one EP rank (active experts)."""
        return self.active_weight_params_per_layer_ep(ep) * self.weight_bits // 8

    def weight_bytes_per_layer_ep(self, ep: int = 1) -> int:
        """Stored bytes per layer on one EP rank (E/ep experts + shared + attn)."""
        e_local = self.experts_per_rank(ep)
        n = self.n_shared_experts + e_local
        ffn = self._ffn_avg(n * self.ffn_weight_params_per_expert())
        return (
            self.attn_weight_params_per_layer() + ffn
        ) * self.weight_bits // 8

    def kv_cache_bytes(self, seq_len: int, batch: int = 1) -> int:
        if seq_len < 0 or batch < 0:
            raise ValueError("seq_len and batch must be non-negative")
        return self.kv_bytes_per_token() * seq_len * batch

    def with_bits(
        self,
        weight_bits: int | None = None,
        kv_bits: int | None = None,
        act_bits: int | None = None,
        name_suffix: str | None = None,
    ) -> "ModelShape":
        """Return a copy with overridden storage/traffic bit widths.

        FLOPs / MAC peak are unchanged (bytes-only scaling). See dtype.py.
        """
        wb = self.weight_bits if weight_bits is None else int(weight_bits)
        kb = self.kv_bits if kv_bits is None else int(kv_bits)
        ab = self.act_bits if act_bits is None else int(act_bits)
        if wb <= 0 or kb <= 0 or ab <= 0:
            raise ValueError("bit widths must be positive")
        # Strip any prior @suffix from name, then attach a fresh one.
        base_name = self.name.split("@", 1)[0]
        suffix = name_suffix or f"w{wb}kv{kb}"
        return replace(
            self,
            name=f"{base_name}@{suffix}",
            weight_bits=wb,
            kv_bits=kb,
            act_bits=ab,
        )

    def summary(self) -> str:
        p = self.total_weight_params()
        pa = self.active_weight_params()
        if p >= 1e9:
            ps, ws = f"{p/1e9:.3f}B", f"{self.weight_bytes()/1e9:.3f}GB"
        else:
            ps, ws = f"{p/1e6:.3f}M", f"{self.weight_bytes()/1e6:.3f}MB"
        moe = ""
        if self.is_moe:
            moe = (
                f" MoE E={self.n_experts} top_k={self.top_k}"
                f" shared={self.n_shared_experts}"
                f" active≈{pa/1e9:.3f}B"
                f" stream_W/layer={self.active_weight_bytes_per_layer()/2**20:.1f}MiB"
            )
        mla = ""
        if self.is_mla:
            mla = (
                f" MLA kv_lora_rank={self.kv_lora_rank}"
                + (f" q_lora={self.q_lora_rank} qk={self.qk_nope_head_dim}+{self.qk_rope_head_dim}"
                   f" v={self.v_head_dim}" if self.attn_kind == "mla" else "")
                + f" attn/layer={self.attn_weight_params_per_layer()/1e6:.1f}M"
                f" kv/tok={self.kv_bytes_per_token()/1024:.1f}KiB"
            )
        return (
            f"{self.name}: L={self.n_layers} H={self.hidden} heads={self.n_heads}/"
            f"{self.n_kv_heads} d={self.head_dim} F={self.intermediate} V={self.vocab} "
            f"| params≈{ps} weights≈{ws}{moe}{mla} "
            f"(W={self.weight_bits}/KV={self.kv_bits}/A={self.act_bits}-bit, assumed)"
        )


# ---------------------------------------------------------------------------
# Built-in shapes (illustrative placeholders — NOT a specific checkpoint)
# ---------------------------------------------------------------------------

TOY_SHAPE = ModelShape(
    name="toy",
    n_layers=2,
    hidden=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    intermediate=128,
    vocab=100,
    weight_bits=16,
    act_bits=16,
    kv_bits=16,
)
"""Tiny shape for hand-calculator verification. See examples/handcheck.md."""

# Rough ~27B-class dense GQA dims (illustrative; not claiming a vendor model).
# Params (derived) ≈ 28.7B with L=40, H=8192, heads=64, n_kv=8, d=128, F=22016.
ILLUSTRATIVE_27B = ModelShape(
    name="illustrative_27B",
    n_layers=40,
    hidden=8192,
    n_heads=64,
    n_kv_heads=8,
    head_dim=128,
    intermediate=22016,
    vocab=128256,
    weight_bits=16,
    act_bits=16,
    kv_bits=16,
)

# Illustrative MoE: E=8, top_k=2, dense-ish GQA attention, SwiGLU per expert.
# Total ≈45.5B / active ≈13.0B (assumed balanced routing; stream top_k FFN only).
# NOT a vendor Mixtral/DeepSeek claim — placeholder for active-W vs dense sweeps.
ILLUSTRATIVE_MOE = ModelShape(
    name="illustrative_moe",
    n_layers=40,
    hidden=4096,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    intermediate=11008,
    vocab=128256,
    weight_bits=16,
    act_bits=16,
    kv_bits=16,
    n_experts=8,
    top_k=2,
    n_shared_experts=0,
)

# Illustrative MLA: same body dims as illustrative_27B, but compressed KV latent.
# kv_lora_rank=512 << GQA full K+V width (2*n_kv*d = 2048) -> ~4x smaller KV/token.
# NOT a DeepSeek checkpoint claim — placeholder for long-ctx KV DSE.
ILLUSTRATIVE_MLA = ModelShape(
    name="illustrative_mla",
    n_layers=40,
    hidden=8192,
    n_heads=64,
    n_kv_heads=8,
    head_dim=128,
    intermediate=22016,
    vocab=128256,
    weight_bits=16,
    act_bits=16,
    kv_bits=16,
    kv_lora_rank=512,
)
