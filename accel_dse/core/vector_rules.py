"""Shared silu(gate)*up product rules.

Phase A (0.480) freezes new ``vec_kind`` string literals for SwiGLU and
RMSNorm-style product billing. A new HF family extends an allowlist below
or adds a rule id to ``RULES``. It does not add a ``vec_kind`` literal, a
``_tag_*_swiglu_vectors`` function, or an ``ir.py`` elif on a new kind.

Phase B (0.481) folds the hand-written validators named by ``LEGACY_TAGGERS``
into ``AllowRow.predicate``. One dispatcher walks ``TAG_ORDER``. The two
order slots stay hand-written. Do not add another nested swiglu tagger.

Catalog rows stay on ``EXTRA_TAIL``. Moving a row into the front 60 is a
separate fingerprint rebaseline. A changelog may say the fingerprints are
unchanged only after the stage fingerprint, the default pipe fingerprint,
and that version's declared KPI list have all been re-run and are green.

``TAG_ORDER`` is the required ``from_release`` order for this cluster.
``kimi_situ`` and the Llama-3 tagger are order slots between ``qwen35_moe``
and ``muse_swiglu``. They are not silu product rules. The silu allowlists
are disjoint. Carbon, Qwen, Whisper, and the 0.458 ERNIE dense tagger stay
outside this list and still run first.
"""

from __future__ import annotations

from dataclasses import dataclass


RULE_DENSE = "dense_silu_product"
RULE_ROUTED_GATED = "routed_gated_silu_product"
RULE_ROUTED_PLAIN = "routed_plain_silu_product"
RULE_ROUTED_LATENT = "routed_latent_silu_product"
RULE_ROUTED_VISION = "routed_vision_silu_product"
RULE_ROUTED_SHARED_ONE = "routed_shared_one_silu_product"
RULE_SHARED_POSITIVE = "shared_positive_silu_product"
RULE_SHARED_ONE = "shared_one_silu_product"

_DEFAULT_DENSE = "dense silu product is not gate_up then down"
_DEFAULT_EXPERT = "routed silu product is not swiglu"
_DEFAULT_SHARED = "shared silu product is not swiglu"


@dataclass(frozen=True)
class ProductRule:
    """One insertion site for silu(gate)*up.

    ``dense`` inserts ``act`` then ``mlp_mul`` between fused gate_up and down.
    ``expert`` moves ``expert_act`` (or ``expert_b.act``) to before down and
    inserts ``mlp_mul`` on the useful token-expert pairs.
    ``shared`` inserts ``mlp_mul`` on every token after ``shared_gate_up``.
    Shared SiLU stays unbilled. Widths are the published intermediate width,
    not the fused gate_up rows and not the padded GEMM rows.
    """

    rule_id: str
    dense: bool = False
    expert: bool = False
    expert_fmt: str = "act"
    expert_profile: str = ""
    shared: bool = False
    shared_profile: str = ""
    shared_linear: str = "shared_gate_up"
    dense_message: str = _DEFAULT_DENSE
    expert_message: str = _DEFAULT_EXPERT
    shared_message: str = _DEFAULT_SHARED


RULES: dict[str, ProductRule] = {
    RULE_DENSE: ProductRule(RULE_DENSE, dense=True),
    RULE_ROUTED_GATED: ProductRule(
        RULE_ROUTED_GATED, expert=True, expert_profile="gated"),
    RULE_ROUTED_PLAIN: ProductRule(
        RULE_ROUTED_PLAIN, expert=True, expert_profile="plain"),
    RULE_ROUTED_LATENT: ProductRule(
        RULE_ROUTED_LATENT, expert=True, expert_fmt="expert_role",
        expert_profile="latent"),
    RULE_ROUTED_VISION: ProductRule(
        RULE_ROUTED_VISION, expert=True, expert_profile="vision"),
    RULE_ROUTED_SHARED_ONE: ProductRule(
        RULE_ROUTED_SHARED_ONE, expert=True, expert_profile="shared_one"),
    RULE_SHARED_POSITIVE: ProductRule(
        RULE_SHARED_POSITIVE, shared=True, shared_profile="positive"),
    RULE_SHARED_ONE: ProductRule(
        RULE_SHARED_ONE, shared=True, shared_profile="one"),
}


@dataclass(frozen=True)
class KindBinding:
    """A frozen ``vec_kind`` bound to one or more rule ids.

    The kind string is not an extension point. Messages stay kind-specific
    so a bad geometry still raises the historical text.
    """

    vec_kind: str
    rule_ids: tuple[str, ...]
    dense_message: str = ""
    expert_message: str = ""
    shared_message: str = ""
    dense_forbidden: str = ""


def _bind(kind: str, rule_ids: tuple[str, ...], **kwargs: str) -> KindBinding:
    return KindBinding(kind, rule_ids, **kwargs)


_DENSE = (RULE_DENSE,)
_PLAIN = (RULE_ROUTED_PLAIN,)

# Kinds whose dense MLP is the carbon / exaone silu product. Exaone is not
# in the pre-norm carbon set: its norms stay post-attention / post-FFN.
CARBON_DENSE_KINDS = frozenset({
    "carbon_llama", "carbon_qwen", "qwen3vl_text", "granite", "qwen3",
    "step_audio", "mistral_small", "qwen2_text",
})

_CARBON_MESSAGE = {
    "qwen3vl_text": "qwen3-vl mlp is not gate_up then down",
    "exaone4": "exaone4 mlp is not gate_up then down",
    "granite": "granite mlp is not gate_up then down",
    "qwen3": "qwen3 mlp is not gate_up then down",
    "step_audio": "step audio mlp is not gate_up then down",
    "mistral_small": "mistral small mlp is not gate_up then down",
    "qwen2_text": "qwen2 text mlp is not gate_up then down",
}


def _carbon_binding(kind: str) -> KindBinding:
    return _bind(kind, _DENSE, dense_message=_CARBON_MESSAGE.get(
        kind, "carbon mlp is not gate_up then down"))


_KIND_LIST: tuple[KindBinding, ...] = (
    *(_carbon_binding(k) for k in (
        "carbon_llama", "carbon_qwen", "qwen3vl_text", "granite", "qwen3",
        "step_audio", "mistral_small", "qwen2_text")),
    _carbon_binding("exaone4"),
    _bind("seed_swiglu", _DENSE,
          dense_message="seed swiglu mlp is not gate_up then down"),
    _bind("olmo3_swiglu", _DENSE,
          dense_message="olmo3 swiglu mlp is not gate_up then down"),
    _bind("jais3_swiglu", _DENSE,
          dense_message="jais3 swiglu mlp is not gate_up then down"),
    _bind("ernievl_swiglu", (RULE_DENSE, RULE_ROUTED_VISION),
          dense_message="ernievl swiglu mlp is not gate_up then down",
          expert_message="ernievl swiglu expert is not swiglu"),
    _bind("qwen35_moe", (RULE_ROUTED_GATED, RULE_SHARED_POSITIVE),
          expert_message="qwen35 moe expert is not swiglu",
          shared_message="qwen35 moe shared expert is not swiglu"),
    _bind("muse_swiglu", (RULE_ROUTED_LATENT,),
          expert_message="muse swiglu expert is not latent swiglu"),
    _bind("mixtral_swiglu", _PLAIN,
          expert_message="mixtral swiglu expert is not swiglu"),
    _bind("minimax_swiglu", _PLAIN,
          expert_message="minimax swiglu expert is not swiglu"),
    _bind("ernie300_swiglu", _PLAIN,
          expert_message="ernie300 swiglu expert is not swiglu",
          dense_forbidden="ernie300 swiglu layer is not moe"),
    _bind("glm5_swiglu", (RULE_DENSE, RULE_ROUTED_SHARED_ONE, RULE_SHARED_ONE),
          dense_message="glm5 swiglu mlp is not gate_up then down",
          expert_message="glm5 swiglu expert is not swiglu",
          shared_message="glm5 swiglu shared expert is not swiglu"),
)

BINDINGS: dict[str, KindBinding] = {b.vec_kind: b for b in _KIND_LIST}

# Frozen product kinds. Structure-domain kinds and situ / gelu kinds are
# not in this set. Do not add a key.
FROZEN_PRODUCT_KINDS = frozenset(BINDINGS)


def _expert_fail(profile: str, ffn: object) -> bool:
    gated = bool(getattr(ffn, "gated"))
    latent = getattr(ffn, "latent")
    n_shared = getattr(ffn, "n_shared")
    d_shared = getattr(ffn, "d_shared")
    d_ff = getattr(ffn, "d_ff")
    n_experts_b = getattr(ffn, "n_experts_b")
    d_expert_b = getattr(ffn, "d_expert_b")
    zero = getattr(ffn, "zero_experts")
    groups = getattr(ffn, "groups")
    role = getattr(ffn, "expert_role")
    if profile == "gated":
        return not gated
    if profile == "latent":
        return (not gated) or latent <= 0 or bool(n_shared)
    if profile == "plain":
        return bool(
            (not gated) or latent or n_shared or d_shared or d_ff
            or n_experts_b or d_expert_b or zero or groups or role)
    if profile == "vision":
        return bool(
            (not gated) or latent or n_shared or d_shared or d_ff
            or (not n_experts_b) or (not d_expert_b) or zero or groups or role)
    if profile == "shared_one":
        return bool(
            (not gated) or latent or n_shared != 1 or (not d_shared)
            or d_ff or n_experts_b or d_expert_b or zero or groups or role)
    raise KeyError(profile)


def _shared_fail(profile: str, ffn: object) -> bool:
    gated = bool(getattr(ffn, "gated"))
    n_shared = getattr(ffn, "n_shared")
    d_shared = getattr(ffn, "d_shared")
    if profile == "positive":
        return (not gated) or (not n_shared) or (not d_shared)
    if profile == "one":
        return (not gated) or n_shared != 1 or (not d_shared)
    raise KeyError(profile)


def resolve(vec_kind: str, product_rule: str = "") -> tuple[tuple[ProductRule, ...], KindBinding | None]:
    """Kind binding wins. ``product_rule`` applies only when the kind has none.

    An unknown rule id is a caller error. An empty rule and an unbound kind
    select the historical tail (activation after down, no ``mlp_mul``).
    """
    binding = BINDINGS.get(vec_kind)
    if binding is not None:
        return tuple(RULES[i] for i in binding.rule_ids), binding
    if product_rule:
        rule = RULES.get(product_rule)
        if rule is None:
            raise ValueError(f"unknown product rule {product_rule}")
        return (rule,), None
    return (), None


def dense_message(vec_kind: str, product_rule: str = "") -> str | None:
    """Reject-message when this layer inserts the dense silu product.

    ``None`` means this layer does not use that emitter.
    """
    rules, binding = resolve(vec_kind, product_rule)
    if not any(r.dense for r in rules):
        return None
    if binding is not None and binding.dense_message:
        return binding.dense_message
    for rule in rules:
        if rule.dense:
            return rule.dense_message
    return None


def dense_forbidden(vec_kind: str, product_rule: str = "") -> str:
    """Non-empty when a dense FFN on this kind is a hard error."""
    binding = BINDINGS.get(vec_kind)
    if binding is None:
        return ""
    return binding.dense_forbidden


def expert_failure(vec_kind: str, product_rule: str, ffn: object) -> str:
    """Empty when the expert pool may bill the product, or the kind has no expert rule."""
    rules, binding = resolve(vec_kind, product_rule)
    for rule in rules:
        if not rule.expert:
            continue
        if _expert_fail(rule.expert_profile, ffn):
            if binding is not None and binding.expert_message:
                return binding.expert_message
            return rule.expert_message
    return ""


def expert_fmt_key(vec_kind: str, product_rule: str = "") -> str:
    """``act`` follows ``ModelSpec.act_fmt``. ``expert_role`` follows ``act_fmt_for(role)``."""
    rules, _binding = resolve(vec_kind, product_rule)
    for rule in rules:
        if rule.expert:
            return rule.expert_fmt
    return "act"


def bills_expert_product(vec_kind: str, product_rule: str = "") -> bool:
    rules, _binding = resolve(vec_kind, product_rule)
    return any(r.expert for r in rules)


def shared_site(vec_kind: str, product_rule: str, linear_name: str,
                ffn: object) -> tuple[bool, str]:
    """``(emit, error)``. No site returns ``(False, "")``.

    The caller inserts ``mlp_mul`` only when ``emit`` is true. A non-empty
    error is the historical ValueError text.
    """
    rules, binding = resolve(vec_kind, product_rule)
    matched = False
    for rule in rules:
        if not rule.shared or linear_name != rule.shared_linear:
            continue
        matched = True
        if _shared_fail(rule.shared_profile, ffn):
            if binding is not None and binding.shared_message:
                return False, binding.shared_message
            return False, rule.shared_message
    if not matched:
        return False, ""
    return True, ""


# Allowlists. These are the catalog-id extension point for an existing kind.
QWEN35_IDS = frozenset({
    "qwen3.8-2.4t", "qwen3.5-397b-a17b", "intern-s2-397b",
    "minicpm-v-4.7-35b-a3b", "startlux-decision-v1-35b-a3b",
    "ornith-1.5-122b", "ornith-1.5-122b-fp8",
    "ornith-1.5-397b", "ornith-1.5-397b-fp8",
    "ornith-1.5-35b-a3b", "ornith-1.5-35b-a3b-fp8",
    "ornith-1.0-397b", "ornith-1.0-397b-fp8",
    "ornith-1.0-35b", "ornith-1.0-35b-fp8",
})
MUSE_IDS = frozenset({
    "muse-spark-open",
    "muse-spark-open-fp8",
    "muse-spark-open-mxfp4",
    "muse-spark-open-nvfp4",
    "muse-spark-open-nvfp4-dspark",
    "muse-spark-open-nvfp4-dflash2",
})
MIXTRAL_IDS = frozenset({"mixtral-8x22b"})
SEED_IDS = frozenset({"seed-oss-36b"})
OLMO3_IDS = frozenset({
    "olmo-3-32b-think",
    "olmo-3.1-32b-instruct",
    "olmo-3-1125-32b",
})
JAIS3_IDS = frozenset({
    "jais-3-32b",
    "jais-3-32b-fp8",
})
ERNIEVL_IDS = frozenset({
    "ernie-4.5-vl-424b-a47b-pt",
    "ernie-4.5-vl-424b-a47b",
})
MINIMAX_IDS = frozenset({"minimax-text-01", "minimax-m1-80k"})
ERNIE300_IDS = frozenset({
    "ernie-4.5-300b-a47b-pt",
    "ernie-4.5-300b-a47b-base-pt",
    "ernie-4.5-300b-a47b-base",
})
GLM5_BF16_IDS = frozenset({
    "glm-5",
    "glm-5.2",
    "atria-dawn",
})
GLM5_FP8_IDS = frozenset({
    "glm-5.3",
    "atria-dawn-fp8",
})
GLM5_IDS = GLM5_BF16_IDS | GLM5_FP8_IDS

# Catalog ids whose layers stay unbound. ``product_rule`` names an existing
# RULES id. Kind binding still wins, so these ids must not also sit on an
# AllowRow. Do not add a vec_kind. Do not add a swiglu tagger. The row stays
# where it already is; this table does not move a catalog row.
UNBOUND_PRODUCT_RULES: dict[str, str] = {
    "magistral-small": RULE_DENSE,
}


@dataclass(frozen=True)
class AllowRow:
    """Catalog ids that bill one frozen kind through the named rule ids.

    ``predicate`` is the Phase B checker (``_pred_<name>``). ``legacy_tagger``
    is that checker's historical name. Extend ``model_ids`` only when the
    predicate already accepts the release. Do not add a swiglu tagger.
    """

    name: str
    vec_kind: str
    rule_ids: tuple[str, ...]
    model_ids: frozenset[str]
    legacy_tagger: str
    predicate: str


def _row(name: str, ids: frozenset[str], tagger: str) -> AllowRow:
    binding = BINDINGS[name]
    return AllowRow(
        name, binding.vec_kind, binding.rule_ids, ids, tagger, "_pred_" + name)


ALLOW_ROWS: tuple[AllowRow, ...] = (
    _row("qwen35_moe", QWEN35_IDS, "_tag_qwen35_moe_vectors"),
    _row("muse_swiglu", MUSE_IDS, "_tag_muse_swiglu_vectors"),
    _row("mixtral_swiglu", MIXTRAL_IDS, "_tag_mixtral_swiglu_vectors"),
    _row("seed_swiglu", SEED_IDS, "_tag_seed_swiglu_vectors"),
    _row("olmo3_swiglu", OLMO3_IDS, "_tag_olmo3_swiglu_vectors"),
    _row("jais3_swiglu", JAIS3_IDS, "_tag_jais3_swiglu_vectors"),
    _row("ernievl_swiglu", ERNIEVL_IDS, "_tag_ernievl_swiglu_vectors"),
    _row("minimax_swiglu", MINIMAX_IDS, "_tag_minimax_swiglu_vectors"),
    _row("ernie300_swiglu", ERNIE300_IDS, "_tag_ernie300_swiglu_vectors"),
    _row("glm5_swiglu", GLM5_IDS, "_tag_glm5_swiglu_vectors"),
)

_ALLOW_BY_KIND = {row.vec_kind: row for row in ALLOW_ROWS}

# Required from_release order. The two slots are not silu allow rows.
# Unrelated taggers stay outside and run before ``_tag_product_cluster``.
TAG_ORDER: tuple[str, ...] = (
    "_tag_qwen35_moe_vectors",
    "_tag_kimi_situ_vectors",
    "_tag_llama3_vectors",
    "_tag_muse_swiglu_vectors",
    "_tag_mixtral_swiglu_vectors",
    "_tag_seed_swiglu_vectors",
    "_tag_olmo3_swiglu_vectors",
    "_tag_jais3_swiglu_vectors",
    "_tag_ernievl_swiglu_vectors",
    "_tag_minimax_swiglu_vectors",
    "_tag_ernie300_swiglu_vectors",
    "_tag_glm5_swiglu_vectors",
)

ORDER_SLOTS = frozenset({
    "_tag_kimi_situ_vectors",
    "_tag_llama3_vectors",
})

LEGACY_TAGGERS: tuple[str, ...] = tuple(
    row.legacy_tagger for row in ALLOW_ROWS)


def allow_row(vec_kind: str) -> AllowRow:
    row = _ALLOW_BY_KIND.get(vec_kind)
    if row is None:
        raise KeyError(vec_kind)
    return row


# Block-norm placement. The dense/MoE emitter asks these predicates.
# Pre-norm fp32 covers the carbon set, Whisper decoder LayerNorm, and the
# 0.458 ERNIE RMSNorm kept on ernie300_swiglu MoE layers. Exaone is post-norm
# and is not in that set. Untagged MoE and kimi_situ are not in either set.
# The strings are existing kinds. This table does not add a vec_kind.
PRE_NORM_FP32 = CARBON_DENSE_KINDS | frozenset({
    "whisper",
    "ernie300_swiglu",
})
POST_NORM_FP32 = frozenset({"exaone4"})
FP32_FINAL_NORM = PRE_NORM_FP32 | POST_NORM_FP32
TTS_FP32_BACKBONE = frozenset({"qwen3", "qwen2_text"})
QK_RMS_WHO = {
    "carbon_qwen": "carbon-x",
    "qwen3vl_text": "qwen3-vl",
    "exaone4": "exaone4",
    "qwen3": "qwen3",
}


def pre_norm_fp32(kind: str) -> bool:
    """True when attn_norm and ffn_norm reduce in fp32 before the sublayer."""
    return kind in PRE_NORM_FP32


def post_norm_fp32(kind: str) -> bool:
    """True when the same norms are emitted after the all-reduce."""
    return kind in POST_NORM_FP32


def fp32_final_norm(kind: str) -> bool:
    """True when final_norm is the published fp32 reduction."""
    return kind in FP32_FINAL_NORM


def tts_fp32_backbone(kind: str) -> bool:
    """True when a TTS head reads one fp32 backbone norm per sequence."""
    return kind in TTS_FP32_BACKBONE


def qk_rms_who(kind: str) -> str:
    """Reject-name for a head-dim q/k RMSNorm, or empty when the kind has none."""
    return QK_RMS_WHO.get(kind, "")


def final_norm_copies(kind: str, final_norm_params: int | None, hidden: int) -> int:
    """Kimi-Audio's qwen2_text row publishes two final RMSNorms. Everyone else is one."""
    if kind == "qwen2_text" and final_norm_params == 2 * hidden:
        return 2
    return 1


def _check_tables() -> None:
    seen: dict[str, str] = {}
    preds: list[str] = []
    for row in ALLOW_ROWS:
        if row.rule_ids != BINDINGS[row.vec_kind].rule_ids:
            raise RuntimeError(f"{row.vec_kind} allow row drifted from its binding")
        if row.predicate != "_pred_" + row.name:
            raise RuntimeError(f"{row.vec_kind} predicate is not _pred_{row.name}")
        preds.append(row.predicate)
        for mid in row.model_ids:
            prev = seen.get(mid)
            if prev is not None:
                raise RuntimeError(f"{mid} is in both {prev} and {row.vec_kind}")
            seen[mid] = row.vec_kind
    if len(preds) != len(set(preds)):
        raise RuntimeError("allow row predicates are not unique")
    named = [row.legacy_tagger for row in ALLOW_ROWS]
    expected = [name for name in TAG_ORDER if name not in ORDER_SLOTS]
    if named != expected:
        raise RuntimeError("allow rows are not in TAG_ORDER")
    if PRE_NORM_FP32 & POST_NORM_FP32:
        raise RuntimeError("pre-norm and post-norm kinds overlap")
    if "kimi_situ" in FP32_FINAL_NORM or "evo2_gelu" in FP32_FINAL_NORM:
        raise RuntimeError("norm predicates captured a protected kind")
    for binding in BINDINGS.values():
        for rule_id in binding.rule_ids:
            if rule_id not in RULES:
                raise RuntimeError(f"{binding.vec_kind} names unknown rule {rule_id}")
    for mid, rule_id in UNBOUND_PRODUCT_RULES.items():
        if rule_id not in RULES:
            raise RuntimeError(f"{mid} names unknown product rule {rule_id}")
        if rule_id in FROZEN_PRODUCT_KINDS:
            raise RuntimeError(f"{mid} product rule is a vec_kind")
        if mid in seen:
            raise RuntimeError(f"{mid} is both an allow row and an unbound product rule")
        if mid in FROZEN_PRODUCT_KINDS:
            raise RuntimeError(f"{mid} collides with a frozen kind")


_check_tables()


def rule_ids() -> tuple[str, ...]:
    """Stable rule-id list. New billing adds an id here, not a vec_kind."""
    return tuple(RULES)
