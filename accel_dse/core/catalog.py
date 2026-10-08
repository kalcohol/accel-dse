"""Model catalog for core v2: catalog ids → official HF releases → ModelSpec.

Sources: LLM entries of ``data/series_catalog.json`` (labels / aliases) plus
official quantised releases as *separate* entries (as-released dtype) and a
few small references.  Gated repos (meta-llama) use public byte-identical
mirrors (unsloth/*), labelled as mirror provenance.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .model import DATA, ModelSpec, from_release, load_release

EXTRA = [
    # id, hf_id, label, section
    ("qwen3-32b-fp8", "Qwen/Qwen3-32B-FP8", "Qwen3 32B FP8 (官方)", "official_quant"),
    ("qwen3-8b-fp8", "Qwen/Qwen3-8B-FP8", "Qwen3 8B FP8 (官方)", "official_quant"),
    ("qwen3-30b-a3b-fp8", "Qwen/Qwen3-30B-A3B-FP8", "Qwen3 30B-A3B FP8 (官方)", "official_quant"),
    ("qwen3-235b-a22b-fp8", "Qwen/Qwen3-235B-A22B-FP8", "Qwen3 235B-A22B FP8 (官方)", "official_quant"),
    ("qwen3-32b-awq", "Qwen/Qwen3-32B-AWQ", "Qwen3 32B AWQ int4 (官方)", "official_quant"),
    ("qwen3-8b-awq", "Qwen/Qwen3-8B-AWQ", "Qwen3 8B AWQ int4 (官方)", "official_quant"),
    ("qwen3-4b", "Qwen/Qwen3-4B", "Qwen3 4B", "reference"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B", "Qwen3 1.7B", "reference"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B", "Qwen3 0.6B", "reference"),
    ("qwen2.5-72b", "Qwen/Qwen2.5-72B-Instruct", "Qwen2.5 72B Instruct", "reference"),
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B Instruct", "reference"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct", "reference"),
    ("llama-3.1-8b", "unsloth/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct", "reference"),
    ("llama-3.3-70b", "unsloth/Llama-3.3-70B-Instruct", "Llama 3.3 70B Instruct", "reference"),
    ("llama-3.2-3b", "unsloth/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", "reference"),
    ("llama-3.2-1b", "unsloth/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct", "reference"),
]
MIRRORS = {"unsloth/": "meta-llama (gated) → public mirror unsloth/*, same safetensors"}


@lru_cache(maxsize=1)
def entries() -> list[dict]:
    raw = json.loads((DATA / "series_catalog.json").read_text())
    out = []
    for e in raw["entries"]:
        if e.get("domain") != "llm" or not e.get("hf_id"):
            continue
        out.append({"id": e["id"], "hf_id": e["hf_id"], "label": e.get("user_label") or e["id"],
                    "section": e.get("section", ""), "aliases": tuple(e.get("aliases") or ())})
    for i, h, lab, sec in EXTRA:
        out.append({"id": i, "hf_id": h, "label": lab, "section": sec, "aliases": (h,)})
    return [e for e in out if load_release(e["hf_id"]) is not None]


def _index() -> dict[str, dict]:
    idx = {}
    for e in entries():
        idx[e["id"]] = e
        for a in e["aliases"]:
            idx.setdefault(a.lower(), e)
    return idx


@lru_cache(maxsize=None)
def get_model(model_id: str) -> ModelSpec:
    e = _index().get(model_id) or _index().get(model_id.lower())
    if e is None:
        raise KeyError(f"unknown model {model_id!r}")
    return from_release(e["id"], e["hf_id"])


def provenance(spec: ModelSpec) -> str:
    for k, v in MIRRORS.items():
        if spec.hf_id.startswith(k):
            return "mirror"
    return "official"


def dtype_label(spec: ModelSpec) -> str:
    f = dict(spec.formats)
    w = f.get("expert") or f.get("mlp") or f.get("attn")
    parts = []
    if "expert" in f and "attn" in f and f["expert"].fmt != f["attn"].fmt:
        parts.append(f"W {f['attn'].fmt}/专家 {f['expert'].fmt}")
    elif w:
        parts.append(f"W {w.fmt}")
    parts.append(f"A {spec.act_fmt}")
    parts.append(f"KV {spec.kv_fmt}")
    return " · ".join(parts)


def labels(spec: ModelSpec) -> dict:
    """Three-axis model label: provenance × coverage × as-released dtype."""
    pc = spec.param_check()
    return {
        "provenance": provenance(spec),
        "coverage": spec.coverage,
        "proxy_badge": spec.coverage == "proxy",
        "dtype": dtype_label(spec),
        "params_B": pc["ours"] / 1e9,
        "active_B": spec.active_params() / 1e9,
        "release_params_B": (pc["release"] or 0) / 1e9,
        "param_err": pc["rel_err"],
        "notes": list(spec.notes),
        "arch": spec.arch,
        "what_if": spec.what_if,
        "roles": {r: f.fmt for r, f in spec.formats},
        "is_moe": spec.is_moe,
        "mtp_layers": len(spec.mtp_layers),
        "n_layers": spec.n_layers,
    }


def list_models() -> list[dict]:
    out = []
    for e in entries():
        s = get_model(e["id"])
        out.append({"id": e["id"], "hf_id": e["hf_id"], "label": e["label"], "section": e["section"],
                    **labels(s)})
    return out
