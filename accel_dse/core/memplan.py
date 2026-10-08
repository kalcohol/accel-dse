"""L4 — memory planning per rank of a pipeline stage.

Storage (per rank, as released):
  stored weights  = all local params of the stage (every local expert), at the
                    release's stored bits incl. scales
  KV cache        = Σ layers ctx·kv_elems_per_token (per-rank share; MLA latent
                    replicated on every attention-TP rank)
  recurrent state = linear-attention state per sequence (fp32 「假设」)
DRAM capacity check uses the *heaviest* stage.

SRAM policy (「假设」, design variable via chip.sram_mib):
  1. staging     = 2 × (largest per-op activation footprint), ≥ 2 MiB
  2. weights     pinned hottest-first: always-touched weights (attention, dense,
                 shared experts, router, head) then routed experts
  3. leftover    holds KV / recurrent state (partial hit: the cached fraction)
Per-step DRAM traffic = unpinned touched weights + uncached KV/state reads +
KV writes to DRAM-resident cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .dtypes import fmt as _fmt
from .ir import Op, Phase, Shard, _cdiv, _lin_local
from .model import Layer, ModelSpec

RUNTIME_RESERVE = 1.0 * 2**30   # 「假设」 runtime / activations reserve in DRAM


@dataclass(frozen=True)
class StageStorage:
    hot_w: float          # bytes of always-touched weights (per rank)
    expert_w: float       # bytes of routed-expert weights stored (per rank)
    kv_per_seq: float     # KV bytes per sequence at the planned context (per rank)
    idx_per_seq: float    # indexer-key bytes per sequence
    state_per_seq: float  # recurrent state bytes per sequence (per rank)

    @property
    def weights(self) -> float:
        return self.hot_w + self.expert_w


def _layer_storage(model: ModelSpec, L: Layer, sh: Shard) -> tuple[float, float]:
    hot = 0.0
    a = model.fmt("attn").bits / 8
    for l in L.attn_linears:
        k, n = _lin_local(l, sh.tp)
        hot += k * n * l.groups * a
    f = L.ffn
    for l in L.ffn_linears:
        k, n = _lin_local(l, sh.tp)
        role = "router" if l.name.startswith(("router", "shared_expert_gate")) else (
            "shared_expert" if l.name.startswith("shared") else "mlp")
        hot += k * n * model.fmt(role).bits / 8
    hot += L.misc_params * 2.0  # norms / gates / conv / hc vectors (bf16 「假设」)
    exp = 0.0
    if f.kind == "moe":
        n_loc = _cdiv(f.n_experts, sh.ep)
        exp = n_loc * f.mats * f.expert_in(model.hidden) * _cdiv(f.d_expert, sh.etp) * model.fmt("expert").bits / 8
    return hot, exp


def stage_storage(model: ModelSpec, first: int, last: int, has_embed: bool, has_head: bool, sh: Shard,
                  ctx: int, n_mtp: int = 0) -> StageStorage:
    hot = exp = kv = idx = st = 0.0
    kvb = _fmt(model.kv_fmt).bytes
    stb = _fmt(model.state_fmt).bytes
    layers = list(model.layers[first:last])
    if has_head:
        layers += list(model.mtp_layers[:n_mtp])
    for L in layers:
        h_, e_ = _layer_storage(model, L, sh)
        hot += h_; exp += e_
        c = L.core
        ce = min(ctx, c.window) if (c.window and c.compress == 1) else ctx
        if c.kind == "gqa":
            kv += ce * _cdiv(c.n_kv, sh.tp) * (c.qk_dim + c.v_dim) / c.compress * kvb
        elif c.kind == "mla":
            kv += (math.ceil(ctx / c.compress) + (c.window or 0 if c.compress > 1 else 0)) * (c.kv_lora + c.rope_dim) * kvb
        elif c.kind == "linear":
            st += (_cdiv(c.n_state_heads, sh.tp) * c.state_dk * c.state_dv + _cdiv(c.conv_channels, sh.tp) * max(0, c.conv_kernel - 1)) * stb
        if c.idx_heads:
            idx += math.ceil(ctx / c.compress) * c.idx_dim * kvb
    store_extra = 0.0
    emb_b = model.fmt("embed").bits / 8
    table = _cdiv(model.vocab, sh.tp) * model.hidden * emb_b
    if has_embed:
        # the embedding table is looked up (a few rows / token) → cold storage,
        # unless it is tied to the LM head on this same stage (then streamed every step)
        if model.tie_embeddings and has_head:
            hot += table
        else:
            store_extra += table
        store_extra += model.lookup_params * emb_b / sh.tp
    if has_head:
        if not model.tie_embeddings:
            hot += _cdiv(model.vocab, sh.tp) * model.hidden * model.fmt("lm_head").bits / 8
        elif not has_embed:
            hot += table          # tied head on a different stage keeps its own copy
        if n_mtp:
            hot += n_mtp * model.mtp_extra_params * model.fmt("mtp").bits / 8 / sh.tp
    return StageStorage(hot, exp + store_extra, kv, idx, st)


@dataclass(frozen=True)
class MemPlan:
    stored_w: float
    kv_total: float
    state_total: float
    dram_need: float
    dram_cap: float
    fits: bool
    sram: float
    staging: float
    pinned_hot: float
    pinned_expert: float
    kv_sram: float          # bytes of KV+state kept on chip
    residency: float        # pinned / stored weights

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def plan(store: StageStorage, batch_local: int, sram_bytes: float, dram_cap: float, max_act: float) -> MemPlan:
    kv_t = (store.kv_per_seq + store.idx_per_seq) * batch_local
    st_t = store.state_per_seq * batch_local
    need = store.weights + kv_t + st_t + RUNTIME_RESERVE
    staging = max(2 * 2**20, 2 * max_act)
    free = max(0.0, sram_bytes - staging)
    p_hot = min(store.hot_w, free); free -= p_hot
    p_exp = min(store.expert_w, free); free -= p_exp
    kv_s = min(kv_t + st_t, free)
    res = (p_hot + p_exp) / store.weights if store.weights else 1.0
    return MemPlan(store.weights, kv_t, st_t, need, dram_cap, need <= dram_cap, sram_bytes, staging,
                   p_hot, p_exp, kv_s, res)


def touched(ops: list[Op]) -> dict:
    """Per-step byte sums of an op list (independent of the plan)."""
    d = {"hot": 0.0, "exp": 0.0, "kv_read": 0.0, "kv_write": 0.0, "state": 0.0, "lookup": 0.0}
    for o in ops:
        if o.kind == "gemm":
            d["exp" if o.role == "expert" else "hot"] += o.w_bytes
        if o.kind == "lookup":
            d["lookup"] += o.kv_read
        else:
            d["kv_read"] += o.kv_read
        d["kv_write"] += o.kv_write
        d["state"] += o.state_rw
    return d


def step_dram_bytes(mp: MemPlan, store: StageStorage, t: dict) -> dict:
    """DRAM bytes moved in one stage step given the plan and touched-byte sums."""
    miss_hot = 0.0 if store.hot_w <= 0 else (1.0 - mp.pinned_hot / store.hot_w)
    miss_exp = 0.0 if store.expert_w <= 0 else (1.0 - mp.pinned_expert / store.expert_w)
    w = t["hot"] * miss_hot + t["exp"] * miss_exp
    kvst_total = mp.kv_total + mp.state_total
    kv_miss = 0.0 if kvst_total <= 0 else 1.0 - mp.kv_sram / kvst_total
    kv_r, kv_w, st = t["kv_read"] * kv_miss, t["kv_write"] * kv_miss, t["state"] * kv_miss
    return {"weights": w, "kv_read": kv_r, "kv_write": kv_w, "state": st, "lookup": t["lookup"],
            "total": w + kv_r + kv_w + st + t["lookup"], "w_touched": t["hot"] + t["exp"]}
