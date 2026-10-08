#!/usr/bin/env python3
"""Bake accel_dse/data/series_catalog.json from out/hf_series_lookup.json.

Rule: no invented architecture dims — only lookup JSON / raw_configs / cited schema.
Workload clip geometry (video frames/latent) and protein seq_len use documented
defaults when not present in HF config.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOKUP = ROOT / "out" / "hf_series_lookup.json"
OUT = ROOT / "accel_dse" / "data" / "series_catalog.json"

# Gated / skip list (require HF token or dims not public)
SKIP_HF_IDS = {
    "meta-llama/Llama-4-Scout-17B-16E",
    "meta-llama/Llama-4-Maverick-17B-128E",
    "google/gemma-3-27b-it",
    "EvolutionaryScale/esm3-sm-open-v1",
}
SKIP_LABELS = {
    "Qwen 3.8 series",  # ambiguous umbrella
    "Chai-1",
    "RFdiffusion",
    "ESM-3 sm-open-v1",
    "Gemma-3 / Gemma-2",
    "Llama-4-Scout (17Bx16E)",
    "Llama-4-Maverick (17Bx128E)",
}

# Video clip geometry defaults (NOT architecture dims; labeled in metadata)
VIDEO_WORKLOAD_DEFAULTS = {
    "n_frames": 16,
    "latent_h": 32,
    "latent_w": 32,
    "n_denoise": 50,
}

# v0.30: structural keys taken from cached raw HF configs when the lookup
# extract omitted them (no KV-shape keys — KV accounting stays lookup-driven).
RAW_FILL_KEYS = (
    "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
    "o_lora_rank", "o_groups", "first_k_dense_replace", "tie_word_embeddings",
    "num_nextn_predict_layers",
)

# Protein seq_len workload default
PROTEIN_SEQ_DEFAULT = 512

# AF2/OpenFold companion schema fields cited from DeepMind alphafold config.py
# (lookup already cites that source for channels/blocks; heads were omitted in extract)
AF2_SCHEMA_HEADS = 8
AF2_SCHEMA_INTERMEDIATE_MULT = 4  # SwiGLU/FFN width proxy for DSE accounting


def _slug(hf_id: str | None, label: str) -> str:
    if hf_id:
        tail = hf_id.split("/")[-1]
    else:
        tail = label
    s = tail.lower()
    s = s.replace("_", "-")
    # strip common suffixes for shorter aliases
    for suf in (
        "-instruct",
        "-instruct-2411",
        "-instruct-2503",
        "-2506",
        "-2507",
        "-v0.1",
        "-preview",
        "-ur50d",
        "-stage3",
    ):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def _aliases_for(hf_id: str | None, label: str, primary: str) -> list[str]:
    """Build product aliases: short id, HF tail, series/ style."""
    aliases: list[str] = []
    if hf_id:
        tail = hf_id.split("/")[-1]
        aliases.append(tail)
        aliases.append(tail.lower())
        aliases.append(hf_id)
        aliases.append(hf_id.lower())
        # underscore variants
        aliases.append(tail.replace("-", "_"))
        aliases.append(tail.lower().replace("-", "_"))
    # label-ish
    lab = re.sub(r"[^a-zA-Z0-9]+", "-", label).strip("-").lower()
    if lab and lab != primary:
        aliases.append(lab)
    # series/ prefix
    aliases.append(f"series/{primary}")
    # dedupe preserving order, drop primary
    seen = {primary.lower()}
    out = []
    for a in aliases:
        k = a.lower()
        if k in seen or not a:
            continue
        seen.add(k)
        out.append(a)
    return out


def _head_dim(dims: dict, hidden: int, n_heads: int) -> int | None:
    hd = dims.get("head_dim")
    if isinstance(hd, int) and hd > 0:
        return hd
    qk_nope = dims.get("qk_nope_head_dim") or 0
    qk_rope = dims.get("qk_rope_head_dim") or 0
    if isinstance(qk_nope, int) and qk_nope > 0:
        return int(qk_nope) + int(qk_rope or 0)
    vhd = dims.get("v_head_dim")
    if isinstance(vhd, int) and vhd > 0:
        return vhd
    ahd = dims.get("attention_head_dim")
    if isinstance(ahd, int) and ahd > 0:
        return ahd
    if n_heads > 0 and hidden % n_heads == 0:
        return hidden // n_heads
    return None


def _llm_shape_dict(dims: dict, name: str) -> dict | None:
    """Map HF-style dims → ModelShape fields. Return None if incomplete."""
    n_layers = dims.get("num_hidden_layers")
    hidden = dims.get("hidden_size")
    n_heads = dims.get("num_attention_heads")
    if not all(isinstance(x, int) and x > 0 for x in (n_layers, hidden, n_heads)):
        return None
    vocab = dims.get("vocab_size")
    if not isinstance(vocab, int) or vocab <= 0:
        return None

    n_kv = dims.get("num_key_value_heads")
    if not isinstance(n_kv, int) or n_kv <= 0:
        n_kv = n_heads

    head_dim = _head_dim(dims, hidden, n_heads)
    if head_dim is None:
        return None

    # MoE expert width preferred when MoE
    n_experts = (
        dims.get("n_routed_experts")
        or dims.get("num_experts")
        or dims.get("num_local_experts")
        or 1
    )
    if not isinstance(n_experts, int) or n_experts < 1:
        n_experts = 1
    top_k = dims.get("num_experts_per_tok") or dims.get("num_experts_per_token") or 1
    if not isinstance(top_k, int) or top_k < 1:
        top_k = 1
    if top_k > n_experts:
        top_k = n_experts
    n_shared = dims.get("n_shared_experts") or dims.get("num_shared_experts") or 0
    if not isinstance(n_shared, int) or n_shared < 0:
        n_shared = 0

    is_moe = n_experts > 1
    if is_moe and isinstance(dims.get("moe_intermediate_size"), int):
        intermediate = dims["moe_intermediate_size"]
    elif isinstance(dims.get("intermediate_size"), int):
        intermediate = dims["intermediate_size"]
    else:
        return None
    if intermediate <= 0:
        return None

    kv_lora = dims.get("kv_lora_rank") or 0
    if not isinstance(kv_lora, int) or kv_lora < 0:
        kv_lora = 0

    # v0.30: MLA / low-rank projection dims, dense-prefix layers, untied LM head.
    # Only emitted when present in the public config (lookup dims, else raw_configs).
    def _pos_int(k: str) -> int:
        v = dims.get(k)
        return v if isinstance(v, int) and v > 0 else 0

    extra: dict = {}
    for k in ("q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
              "o_lora_rank", "o_groups"):
        if _pos_int(k):
            extra[k] = _pos_int(k)
    if kv_lora > 0 and not (extra.get("qk_nope_head_dim") or extra.get("v_head_dim")):
        extra = {k: v for k, v in extra.items() if k not in ("qk_rope_head_dim",)}
    fkd = _pos_int("first_k_dense_replace")
    if n_experts > 1 and fkd and isinstance(dims.get("intermediate_size"), int):
        extra["n_dense_layers"] = min(fkd, n_layers)
        extra["dense_intermediate"] = dims["intermediate_size"]
    if dims.get("tie_word_embeddings") is False:
        extra["tie_embeddings"] = False
    # v0.31: MTP / nextn predict layers (DeepSeek-V3, GLM-4.5+); weights only
    # counted when speculative decoding with draft=mtp is enabled.
    if _pos_int("num_nextn_predict_layers"):
        extra["n_mtp_layers"] = _pos_int("num_nextn_predict_layers")

    return {
        "kind": "llm",
        "name": name,
        "n_layers": n_layers,
        "hidden": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "vocab": vocab,
        "n_experts": n_experts,
        "top_k": top_k,
        "n_shared_experts": n_shared,
        "kv_lora_rank": kv_lora,
        **extra,
        "weight_bits": 16,
        "act_bits": 16,
        "kv_bits": 16,
    }


def _patch_spatial(patch) -> int:
    if isinstance(patch, int) and patch > 0:
        return patch
    if isinstance(patch, (list, tuple)) and len(patch) >= 2:
        # [t, h, w] or [h, w]
        vals = [int(x) for x in patch if isinstance(x, (int, float)) and int(x) > 0]
        if vals:
            return max(vals[-2:]) if len(vals) >= 2 else vals[-1]
    return 2  # common DiT default when config omits — labeled below


def _video_shape_dict(dims: dict, name: str) -> dict | None:
    n_layers = dims.get("num_layers") or dims.get("depth") or dims.get("num_hidden_layers")
    hidden = dims.get("hidden_size") or dims.get("dim")
    n_heads = dims.get("num_attention_heads") or dims.get("num_heads")
    if not all(isinstance(x, int) and x > 0 for x in (n_layers, hidden, n_heads)):
        return None

    head_dim = dims.get("attention_head_dim") or dims.get("head_dim")
    if not isinstance(head_dim, int) or head_dim <= 0:
        if hidden % n_heads == 0:
            head_dim = hidden // n_heads
        else:
            return None

    if isinstance(dims.get("ffn_dim"), int) and dims["ffn_dim"] > 0:
        intermediate = dims["ffn_dim"]
    elif isinstance(dims.get("intermediate_size"), int) and dims["intermediate_size"] > 0:
        intermediate = dims["intermediate_size"]
    elif isinstance(dims.get("mlp_ratio"), (int, float)) and dims["mlp_ratio"] > 0:
        intermediate = int(hidden * float(dims["mlp_ratio"]))
    else:
        intermediate = 4 * hidden  # DiT default MLP ratio; noted in extras

    patch = _patch_spatial(dims.get("patch_size"))
    # Ensure latent divisible by patch
    lat_h = VIDEO_WORKLOAD_DEFAULTS["latent_h"]
    lat_w = VIDEO_WORKLOAD_DEFAULTS["latent_w"]
    # bump latent if needed so divisible
    if lat_h % patch:
        lat_h = ((lat_h + patch - 1) // patch) * patch
    if lat_w % patch:
        lat_w = ((lat_w + patch - 1) // patch) * patch

    extras = {
        "in_channels": dims.get("in_channels") or dims.get("in_dim"),
        "out_channels": dims.get("out_channels") or dims.get("out_dim"),
        "architecture_family": dims.get("architecture_family"),
        "workload_defaults": "n_frames/latent/n_denoise are DSE clip defaults (not HF architecture dims)",
    }
    if "ffn_dim" not in dims and "intermediate_size" not in dims and "mlp_ratio" not in dims:
        extras["intermediate_note"] = "intermediate=4*hidden (DiT MLP ratio default; not in config)"

    return {
        "kind": "video",
        "name": name,
        "n_frames": VIDEO_WORKLOAD_DEFAULTS["n_frames"],
        "latent_h": lat_h,
        "latent_w": lat_w,
        "patch_size": patch,
        "hidden": hidden,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "n_denoise": VIDEO_WORKLOAD_DEFAULTS["n_denoise"],
        "batch": 1,
        "weight_bits": 16,
        "act_bits": 16,
        "extras": extras,
    }


def _protein_shape_dict(dims: dict, name: str, label: str) -> dict | None:
    """Map protein / structure dims onto ProteinShape (coarse approximation)."""
    fam = (dims.get("architecture_family") or "").lower()
    extras: dict = {"architecture_family": dims.get("architecture_family")}

    # ESM-2 style LM
    if dims.get("num_hidden_layers") and dims.get("hidden_size") and dims.get("num_attention_heads"):
        n_layers = dims["num_hidden_layers"]
        hidden = dims["hidden_size"]
        n_heads = dims["num_attention_heads"]
        intermediate = dims.get("intermediate_size")
        if not isinstance(intermediate, int) or intermediate <= 0:
            return None
        head_dim = hidden // n_heads if hidden % n_heads == 0 else None
        if head_dim is None:
            return None
        return {
            "kind": "protein",
            "name": name,
            "seq_len": PROTEIN_SEQ_DEFAULT,
            "hidden": hidden,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "intermediate": intermediate,
            "pair_dim": 0,
            "msa_depth": 1,
            "batch": 1,
            "weight_bits": 16,
            "act_bits": 16,
            "extras": {
                **extras,
                "seq_len_note": "workload default seq_len=512 (not architecture)",
                "mode": "seq_lm",
            },
        }

    # ESMFold: LM trunk + folding
    if dims.get("lm_num_hidden_layers") and dims.get("trunk_num_blocks"):
        # Prefer folding trunk for structure DSE; keep LM dims in extras
        n_layers = int(dims["trunk_num_blocks"])
        hidden = int(dims.get("sequence_state_dim") or dims.get("lm_hidden_size"))
        pair_dim = int(dims.get("pairwise_state_dim") or 0)
        lm_h = dims.get("lm_hidden_size")
        lm_heads = dims.get("lm_num_attention_heads")
        if isinstance(lm_h, int) and isinstance(lm_heads, int) and lm_heads > 0 and lm_h % lm_heads == 0:
            n_heads = lm_heads
            head_dim = lm_h // lm_heads
            # scale if trunk hidden differs — use trunk hidden with same head_dim if divisible
            if hidden % head_dim == 0:
                n_heads = hidden // head_dim
            else:
                n_heads = max(1, hidden // 64)
                head_dim = hidden // n_heads if hidden % n_heads == 0 else 64
        else:
            n_heads = 16
            head_dim = hidden // n_heads if hidden % n_heads == 0 else 64
            extras["heads_note"] = "trunk heads not in config; used hidden-aligned head split"
        intermediate = dims.get("lm_intermediate_size")
        if not isinstance(intermediate, int):
            intermediate = 4 * hidden
            extras["intermediate_note"] = "trunk FFN width not in config; using 4×H"
        extras["lm"] = {
            "lm_num_hidden_layers": dims.get("lm_num_hidden_layers"),
            "lm_hidden_size": dims.get("lm_hidden_size"),
            "lm_intermediate_size": dims.get("lm_intermediate_size"),
            "lm_num_attention_heads": dims.get("lm_num_attention_heads"),
            "max_recycles": dims.get("max_recycles"),
        }
        extras["mode"] = "esmfold_trunk"
        extras["seq_len_note"] = "workload default seq_len=512"
        return {
            "kind": "protein",
            "name": name,
            "seq_len": PROTEIN_SEQ_DEFAULT,
            "hidden": hidden,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": head_dim if hidden % n_heads == 0 else (hidden // max(n_heads, 1)),
            "intermediate": intermediate,
            "pair_dim": pair_dim,
            "msa_depth": 1,
            "batch": 1,
            "weight_bits": 16,
            "act_bits": 16,
            "extras": extras,
        }

    # AF2 / OpenFold / AF3 / Boltz / Protenix — pairformer/evoformer style
    n_layers = (
        dims.get("evoformer_num_block")
        or dims.get("pairformer_blocks")
        or dims.get("pairformer_blocks_claimed")
        or dims.get("n_blocks")
        or dims.get("trunk_num_blocks")
    )
    hidden = (
        dims.get("seq_channel")
        or dims.get("single_channel")
        or dims.get("token_s")
        or dims.get("c_s")
        or dims.get("sequence_state_dim")
    )
    pair_dim = (
        dims.get("pair_channel")
        or dims.get("token_z")
        or dims.get("c_z")
        or dims.get("pairwise_state_dim")
        or 0
    )
    if not isinstance(n_layers, int) or n_layers < 1:
        return None
    if not isinstance(hidden, int) or hidden < 1:
        return None
    if not isinstance(pair_dim, int):
        pair_dim = 0

    n_heads = dims.get("pairformer_heads") or dims.get("token_transformer_heads")
    if not isinstance(n_heads, int) or n_heads < 1:
        # AF2 / OpenFold: public schema no_heads=8 (alphafold config.py)
        if any(k in fam for k in ("af2", "openfold", "evoformer", "af3", "pairformer", "af3-like")):
            n_heads = AF2_SCHEMA_HEADS
            extras["heads_source"] = (
                "n_heads=8 from DeepMind alphafold config.py public schema "
                "(companion to cited channel/block dims in lookup)"
            )
        else:
            return None

    if hidden % n_heads == 0:
        head_dim = hidden // n_heads
    else:
        # pick largest divisor
        for h in range(n_heads, 0, -1):
            if hidden % h == 0:
                n_heads = h
                head_dim = hidden // h
                break
        else:
            return None

    intermediate = AF2_SCHEMA_INTERMEDIATE_MULT * hidden
    extras["intermediate_note"] = (
        f"intermediate={intermediate} (=4×hidden) — FFN width not in extracted "
        "lookup dims; DSE FFN accounting proxy only (Evoformer heavily approximated)"
    )
    extras["mode"] = "pairformer_approx"
    extras["seq_len_note"] = "workload default seq_len=512"
    extras["msa_channel"] = dims.get("msa_channel") or dims.get("msa_s")
    extras["msa_blocks"] = dims.get("msa_blocks") or dims.get("msa_blocks_claimed")
    if dims.get("note"):
        extras["note"] = dims["note"]

    msa_depth = 1
    if isinstance(dims.get("msa_blocks"), int):
        extras["msa_blocks"] = dims["msa_blocks"]

    return {
        "kind": "protein",
        "name": name,
        "seq_len": PROTEIN_SEQ_DEFAULT,
        "hidden": hidden,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "pair_dim": pair_dim,
        "msa_depth": msa_depth,
        "batch": 1,
        "weight_bits": 16,
        "act_bits": 16,
        "extras": extras,
    }


def _family_for_llm(shape: dict) -> str:
    if shape.get("kv_lora_rank", 0) > 0:
        return "mla" if shape.get("n_experts", 1) == 1 else "moe"
    if shape.get("n_experts", 1) > 1:
        return "moe"
    return "dense"


def _primary_id(hf_id: str | None, label: str, category: str) -> str:
    """Canonical product id."""
    # Prefer well-known short aliases for user-named models
    SPECIAL = {
        "zai-org/GLM-5.3": "glm-5.3",
        "zai-org/GLM-5.3-Flash": "glm-5.3-flash",
        "deepseek-ai/DeepSeek-V4.1-Flash": "deepseek-v4.1-flash",
        "deepseek-ai/DeepSeek-V4-Pro": "deepseek-v4-pro",
        "moonshotai/Kimi-K3": "kimi-k3",
        "moonshotai/Kimi-K2.7-Code": "kimi-k2.7-code",
        "Qwen/Qwen3.8-2.4T-A95B": "qwen3.8-2.4t",
        "Qwen/Qwen3.8-27B": "qwen3.8-27b",
        "Qwen/Qwen3.8-Flash-Next": "qwen3.8-flash-next",
        "MiniMaxAI/MiniMax-H3": "minimax-h3",
    }
    if hf_id in SPECIAL:
        return SPECIAL[hf_id]
    # Protein without HF id
    LABEL_IDS = {
        "AlphaFold3": "alphafold3",
        "AlphaFold2": "alphafold2",
        "ESMFold": "esmfold",
        "Boltz-1": "boltz-1",
        "Protenix (base)": "protenix",
        "OpenFold (AF2 reimpl)": "openfold",
        "CogVideoX-5B": "cogvideox-5b",
        "CogVideoX-2B": "cogvideox-2b",
        "HunyuanVideo": "hunyuanvideo",
        "Wan2.1-T2V-14B": "wan2.1-14b",
        "Wan2.1-T2V-1.3B": "wan2.1-1.3b",
        "Wan2.2-T2V-A14B": "wan2.2-a14b",
        "LTX-Video": "ltx-video",
        "Mochi-1": "mochi-1",
        "Open-Sora STDiT3": "opensora-stdit3",
        "Open-Sora STDiT2": "opensora-stdit2",
        "ESM-2 3B": "esm2-3b",
        "ESM-2 650M": "esm2-650m",
        "MiniMax H3 / Hailuo DiT": "minimax-h3",
    }
    if label in LABEL_IDS:
        return LABEL_IDS[label]
    return _slug(hf_id, label)


def _metadata(rec: dict, shape: dict) -> str:
    hf = rec.get("hf_id") or "—"
    src = rec.get("source_url") or "schema/paper"
    status = rec.get("status")
    fam = (rec.get("dims") or {}).get("architecture_family") or ""
    claimed = (rec.get("dims") or {}).get("param_count_claimed") or ""
    parts = [
        f"hf:{hf}",
        f"source:{src}",
        f"status:{status}",
    ]
    if fam:
        parts.append(f"arch:{fam}")
    if claimed:
        parts.append(f"claimed:{claimed}")
    # hybrid / engram extras
    dims = rec.get("dims") or {}
    extra_bits = []
    if dims.get("full_attn_layers") or dims.get("linear_key_head_dim"):
        extra_bits.append("hybrid_linear_attn")
    if dims.get("engram_vocab_size") or dims.get("engram_n_heads"):
        extra_bits.append("engram")
    if dims.get("index_n_heads") or "DSA" in fam or "dsa" in fam.lower():
        extra_bits.append("DSA/indexer")
    if shape.get("kind") == "video":
        extra_bits.append("video_clip_workload_defaults")
    if shape.get("kind") == "protein":
        extra_bits.append("protein_pair_approx")
        if shape.get("extras"):
            if shape["extras"].get("intermediate_note"):
                extra_bits.append("ffn_proxy_4xH")
            if shape["extras"].get("heads_source"):
                extra_bits.append("heads_from_af2_schema")
    if extra_bits:
        parts.append("extras:" + "+".join(extra_bits))
    parts.append("FLOPs/BW uncalibrated (DSE accounting only)")
    return " | ".join(parts)


def main() -> None:
    records = json.loads(LOOKUP.read_text())
    entries = []
    skipped = []

    for rec in records:
        label = rec.get("user_label") or ""
        status = rec.get("status")
        hf_id = rec.get("hf_id")
        category = rec.get("category")
        dims = rec.get("dims")

        if label in SKIP_LABELS or (hf_id in SKIP_HF_IDS):
            skipped.append({"label": label, "hf_id": hf_id, "reason": "gated_or_ambiguous_or_missing"})
            continue
        if status in ("ambiguous", "missing"):
            skipped.append({"label": label, "hf_id": hf_id, "reason": f"status={status}"})
            continue
        if status not in ("found", "nearest"):
            skipped.append({"label": label, "hf_id": hf_id, "reason": f"status={status}"})
            continue
        if not dims:
            skipped.append({"label": label, "hf_id": hf_id, "reason": "no_dims"})
            continue

        # v0.30: fill structural keys missing from the lookup extract from the
        # cached raw HF config (out/raw_configs/<org>__<name>.json); never
        # overrides lookup values.
        if hf_id:
            raw_p = ROOT / "out" / "raw_configs" / (hf_id.replace("/", "__") + ".json")
            if raw_p.is_file():
                try:
                    raw = json.loads(raw_p.read_text())
                except ValueError:
                    raw = {}
                if isinstance(raw, dict):
                    if isinstance(raw.get("text_config"), dict):
                        raw = {**raw, **raw["text_config"]}
                    dims = dict(dims)
                    for k in RAW_FILL_KEYS:
                        if k not in dims and k in raw:
                            dims[k] = raw[k]

        primary = _primary_id(hf_id, label, category)
        shape_name = primary.replace("/", "-")

        if category == "llm":
            shape = _llm_shape_dict(dims, shape_name)
            if shape is None:
                skipped.append({"label": label, "hf_id": hf_id, "reason": "incomplete_llm_dims"})
                continue
            family = _family_for_llm(shape)
            domain = "llm"
        elif category == "video":
            shape = _video_shape_dict(dims, shape_name)
            if shape is None:
                skipped.append({"label": label, "hf_id": hf_id, "reason": "incomplete_video_dims"})
                continue
            family = "dit_video"
            domain = "video"
        elif category == "protein":
            # OpenFold: use AF2 dims (lookup says same target dims)
            use_dims = dims
            if label.startswith("OpenFold") and (not dims.get("evoformer_num_block")):
                # find AF2 record dims from already-loaded records
                for _r in records:
                    if _r.get("user_label") == "AlphaFold2" and _r.get("dims"):
                        use_dims = dict(_r["dims"])
                        use_dims["architecture_family"] = "AF2 Evoformer (OpenFold)"
                        use_dims["note"] = "Same target dims as AF2 (OpenFold reimpl)"
                        break
            shape = _protein_shape_dict(use_dims, shape_name, label)
            if shape is None:
                skipped.append({"label": label, "hf_id": hf_id, "reason": "incomplete_protein_dims"})
                continue
            # Fix head_dim consistency
            if shape["hidden"] % shape["n_heads"] != 0:
                # recompute
                for h in (shape["n_heads"], 8, 16, 4, 2, 1):
                    if shape["hidden"] % h == 0:
                        shape["n_heads"] = h
                        shape["head_dim"] = shape["hidden"] // h
                        break
            family = "protein"
            domain = "protein"
        else:
            skipped.append({"label": label, "hf_id": hf_id, "reason": f"unknown_category={category}"})
            continue

        aliases = _aliases_for(hf_id, label, primary)
        # Extra product aliases from task
        EXTRA = {
            "glm-5.3": ["glm5.3", "GLM-5.3"],
            "glm-5.3-flash": ["glm5.3-flash"],
            "deepseek-v4.1-flash": ["deepseek-4.1-flash", "ds-v4.1-flash"],
            "deepseek-v4-pro": ["deepseek-4-pro", "ds-v4-pro"],
            "kimi-k3": ["kimi_k3", "Kimi-K3"],
            "kimi-k2.7-code": ["kimi-k2.7", "kimi_k2.7_code"],
            "qwen3.8-2.4t": ["qwen3.8-2.4t-a95b", "qwen3.8"],
            "qwen3.8-27b": ["qwen3.8-27B"],
            "qwen3.8-flash-next": ["qwen3.8-flash"],
            "minimax-h3": ["hailuo-h3", "MiniMax-H3"],
        }
        for a in EXTRA.get(primary, []):
            if a.lower() not in {x.lower() for x in aliases} and a.lower() != primary.lower():
                aliases.append(a)

        entries.append(
            {
                "id": primary,
                "family": family,
                "domain": domain,
                "aliases": aliases,
                "metadata": _metadata(rec, shape),
                "hf_id": hf_id,
                "user_label": label,
                "section": rec.get("section"),
                "status": status,
                "shape": shape,
            }
        )

    # Deduplicate by id (keep first / user_requested preferred)
    by_id = {}
    for e in entries:
        if e["id"] not in by_id:
            by_id[e["id"]] = e
        else:
            # prefer user_requested
            if e.get("section") == "user_requested":
                by_id[e["id"]] = e

    catalog = {
        "version": "0.31.0",
        "source": "out/hf_series_lookup.json",
        "honesty": (
            "Architecture dims from public HF config.json / model cards / cited schema. "
            "FLOPs/BW remain uncalibrated. Hybrid linear-attn / Engram / DSA stored in "
            "metadata only — core DSE uses ModelShape GQA/MoE/MLA accounting."
        ),
        "entries": list(by_id.values()),
        "skipped": skipped,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=2) + "\n")
    n_llm = sum(1 for e in catalog["entries"] if e["domain"] == "llm")
    n_vid = sum(1 for e in catalog["entries"] if e["domain"] == "video")
    n_pro = sum(1 for e in catalog["entries"] if e["domain"] == "protein")
    print(f"Wrote {OUT}")
    print(f"entries: llm={n_llm} video={n_vid} protein={n_pro} total={len(catalog['entries'])}")
    print(f"skipped: {len(skipped)}")
    for s in skipped:
        print(f"  SKIP {s}")


if __name__ == "__main__":
    main()
