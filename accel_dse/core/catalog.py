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
    # id, hf_id  (listing label = official repo name)
    ("qwen3-32b-fp8", "Qwen/Qwen3-32B-FP8"),
    ("qwen3-8b-fp8", "Qwen/Qwen3-8B-FP8"),
    ("qwen3-30b-a3b-fp8", "Qwen/Qwen3-30B-A3B-FP8"),
    ("qwen3-235b-a22b-fp8", "Qwen/Qwen3-235B-A22B-FP8"),
    ("qwen3-32b-awq", "Qwen/Qwen3-32B-AWQ"),
    ("qwen3-8b-awq", "Qwen/Qwen3-8B-AWQ"),
    ("qwen3-4b", "Qwen/Qwen3-4B"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    ("qwen2.5-72b", "Qwen/Qwen2.5-72B-Instruct"),
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("llama-3.1-8b", "unsloth/Llama-3.1-8B-Instruct"),
    ("llama-3.3-70b", "unsloth/Llama-3.3-70B-Instruct"),
    ("llama-3.2-3b", "unsloth/Llama-3.2-3B-Instruct"),
    ("llama-3.2-1b", "unsloth/Llama-3.2-1B-Instruct"),
]
MIRRORS = {"unsloth/": "meta-llama (gated) → public mirror unsloth/*, same safetensors"}

# ---------------------------------------------------------------- listing: domain → provider → family
DOMAINS = {"llm": "LLM（文本）", "vlm": "VLM（多模态，评估语言主干）", "gen": "图像 / 视频生成（DiT）"}
PROVIDERS = [   # key, label, HF orgs — listing order
    ("qwen", "Qwen（阿里）", ("Qwen/",)), ("deepseek", "DeepSeek（深度求索）", ("deepseek-ai/",)),
    ("kimi", "Kimi（月之暗面）", ("moonshotai/",)), ("glm", "GLM（智谱）", ("zai-org/",)),
    ("minimax", "MiniMax（稀宇）", ("MiniMaxAI/",)), ("llama", "Llama（Meta）", ("unsloth/", "meta-llama/")),
    ("gpt-oss", "gpt-oss（OpenAI）", ("openai/",)), ("mistral", "Mistral", ("mistralai/",)),
    ("phi", "Phi（微软）", ("microsoft/",)), ("seed", "Seed（字节跳动）", ("ByteDance-Seed/",)),
    ("yi", "Yi（零一万物）", ("01-ai/",)), ("internlm", "InternLM（上海 AI 实验室）", ("internlm/",)),
    ("wan", "Wan（阿里）", ("Wan-AI/",)), ("cogvideo", "CogVideoX（智谱）", ("THUDM/",)),
    ("hunyuan", "HunyuanVideo（腾讯）", ("hunyuanvideo-community/",)), ("ltx", "LTX-Video（Lightricks）", ("Lightricks/",)),
    ("mochi", "Mochi（Genmo）", ("genmo/",)), ("opensora", "Open-Sora（HPC-AI Tech）", ("hpcai-tech/",)),
]
FAMILIES = [    # provider, family, id prefixes — newest family first; within a family: size ↓, base → FP8 → AWQ
    ("qwen", "Qwen3.8", ("qwen3.8",)), ("qwen", "Qwen3.5", ("qwen3.5",)), ("qwen", "Qwen3-Next", ("qwen3-next",)),
    ("qwen", "Qwen3 MoE", ("qwen3-235b", "qwen3-30b")), ("qwen", "Qwen3 稠密", ("qwen3-",)), ("qwen", "Qwen2.5", ("qwen2.5",)),
    ("deepseek", "DeepSeek-V4", ("deepseek-v4",)), ("deepseek", "DeepSeek-V3", ("deepseek-v3", "deepseek-r1")),
    ("kimi", "Kimi-K3", ("kimi-k3",)), ("kimi", "Kimi-K2", ("kimi-k2",)),
    ("glm", "GLM-5", ("glm-5",)), ("glm", "GLM-4.5 / 4.6", ("glm-4",)),
    ("minimax", "MiniMax-M1 / Text-01", ("minimax-m1", "minimax-text")), ("minimax", "MiniMax-H", ("minimax-h",)),
    ("llama", "Llama 3", ("llama-3",)), ("gpt-oss", "gpt-oss", ("gpt-oss",)),
    ("mistral", "Mistral Large", ("mistral-large",)), ("mistral", "Mixtral", ("mixtral",)), ("mistral", "Magistral", ("magistral",)),
    ("phi", "Phi-4", ("phi-4",)), ("seed", "Seed-OSS", ("seed-oss",)),
    ("wan", "Wan2.2", ("wan2.2",)), ("wan", "Wan2.1", ("wan2.1",)), ("cogvideo", "CogVideoX", ("cogvideox",)),
    ("hunyuan", "HunyuanVideo", ("hunyuanvideo",)), ("ltx", "LTX-Video", ("ltx-video",)), ("mochi", "Mochi 1", ("mochi",)),
    ("opensora", "Open-Sora", ("opensora",)),
]
# Same architecture as a listed entry → resolvable by id and validated (V1), but not listed separately.
MERGED = {"deepseek-v3.1": "deepseek-v3", "deepseek-r1": "deepseek-v3", "kimi-k2.7-code": "kimi-k2.5",
          "glm-5": "glm-5.2", "glm-4.5": "glm-4.6", "minimax-text-01": "minimax-m1-80k"}
# Generic dense GQA models whose sizes are already covered by Qwen3 / Llama 3: no extra design insight.
UNLISTED = {"yi-1.5-34b": "与 Qwen3-32B 同类（稠密 GQA）", "internlm3-8b": "与 Qwen3-8B / Llama-3.1-8B 同类",
            "internlm2-5-20b": "与 Qwen3-32B / Magistral-Small 同类", "qwen2.5-3b": "与 Qwen3-4B 同类",
            "qwen2.5-1.5b": "与 Qwen3-1.7B 同类", "opensora-stdit2": "已由 STDiT3 取代（同尺寸）"}
_VARIANT = {"": 0, "fp8": 1, "awq": 2}


def provider_of(hf_id: str) -> tuple[str, str]:
    for key, label, orgs in PROVIDERS:
        if hf_id.startswith(orgs):
            return key, label
    return "other", hf_id.split("/")[0]


def family_of(provider: str, model_id: str) -> tuple[int, str]:
    for i, (prov, fam, prefixes) in enumerate(FAMILIES):
        if prov == provider and model_id.startswith(prefixes):
            return i, fam
    return len(FAMILIES), ""


def _variant(model_id: str) -> int:
    tail = model_id.rsplit("-", 1)[-1]
    return _VARIANT.get(tail if tail in _VARIANT else "", 0)


def _repo_name(hf_id: str) -> str:
    return hf_id.split("/", 1)[1]


@lru_cache(maxsize=1)
def entries() -> list[dict]:
    """Every fetched LLM / VLM release (resolvable by id; all are validated in V1)."""
    raw = json.loads((DATA / "series_catalog.json").read_text())
    out = []
    for e in raw["entries"]:
        if e.get("domain") != "llm" or not e.get("hf_id"):
            continue
        out.append({"id": e["id"], "hf_id": e["hf_id"], "aliases": tuple(e.get("aliases") or ())})
    for i, h in EXTRA:
        out.append({"id": i, "hf_id": h, "aliases": (h,)})
    out = [e for e in out if load_release(e["hf_id"]) is not None]
    for e in out:
        e["label"] = _repo_name(e["hf_id"])
        e["provider"], e["provider_label"] = provider_of(e["hf_id"])
        e["family_rank"], e["family"] = family_of(e["provider"], e["id"])
        e["domain"] = "vlm" if (load_release(e["hf_id"]).get("params_vision") or 0) > 0 else "llm"
        e["merged_into"] = MERGED.get(e["id"])
        e["unlisted"] = UNLISTED.get(e["id"])
        e["listed"] = not e["merged_into"] and not e["unlisted"]
    return out


@lru_cache(maxsize=1)
def offline_entries() -> list[dict]:
    """Image / video generation (DiT) entries kept in the catalog but not yet on core v2."""
    raw = json.loads((DATA / "series_catalog.json").read_text())
    out = []
    for e in raw["entries"]:
        if e.get("domain") != "video" or not e.get("hf_id") or e["id"] in UNLISTED:
            continue
        sh = e.get("shape") or {}
        prov, plab = provider_of(e["hf_id"])
        rank, fam = family_of(prov, e["id"])
        out.append({"id": e["id"], "hf_id": e["hf_id"], "label": _repo_name(e["hf_id"]), "domain": "gen",
                    "domain_label": DOMAINS["gen"], "provider": prov, "provider_label": plab, "family": fam or plab,
                    "family_rank": rank, "status": "暂未接入 v2",
                    "arch": f"DiT · {sh.get('n_layers', '?')} 层 · hidden {sh.get('hidden', '?')}",
                    "_size": (sh.get("n_layers") or 0) * (sh.get("hidden") or 0) ** 2})
    order = {k: i for i, (k, _, _) in enumerate(PROVIDERS)}
    out.sort(key=lambda d: (order.get(d["provider"], 99), d["family_rank"], -d["_size"]))
    for d in out:
        del d["_size"]
    return out


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


_COVER_FILTER = ("「架构代理」", "混合注意力：", "DSA lightning", "attention sink")


def labels(spec: ModelSpec) -> dict:
    """Three-axis model label: provenance × coverage (with concrete reasons) × as-released dtype."""
    pc = spec.param_check()
    return {
        "provenance": provenance(spec),
        "coverage": spec.coverage,
        "coverage_reasons": list(spec.coverage_reasons),
        "proxy_badge": spec.coverage == "proxy",
        "dtype": dtype_label(spec),
        "params_B": pc["ours"] / 1e9,
        "active_B": spec.active_params() / 1e9,
        "release_params_B": (pc["release"] or 0) / 1e9,
        "param_err": pc["rel_err"],
        "vision_params_B": spec.vision_params / 1e9,
        "notes": [n for n in spec.notes if not n.startswith(_COVER_FILTER)],
        "arch": spec.arch,
        "what_if": spec.what_if,
        "roles": {r: f.fmt for r, f in spec.formats},
        "is_moe": spec.is_moe,
        "mtp_layers": len(spec.mtp_layers),
        "n_layers": spec.n_layers,
    }


def list_models() -> list[dict]:
    """Listed (evaluable) models, ordered domain → provider → family → size ↓ → base / FP8 / AWQ."""
    es = entries()
    by_id = {e["id"]: e for e in es}
    merged: dict[str, list[str]] = {}
    for e in es:
        if e["merged_into"]:
            merged.setdefault(e["merged_into"], []).append(e["id"])
    out = []
    for e in es:
        if not e["listed"]:
            continue
        s = get_model(e["id"])
        same = []
        for mid in merged.get(e["id"], []):
            o = get_model(mid)
            same.append({"id": mid, "label": by_id[mid]["label"],
                         "identical": o.params() == s.params() and dtype_label(o) == dtype_label(s)})
        out.append({"id": e["id"], "hf_id": e["hf_id"], "label": e["label"], "domain": e["domain"],
                    "domain_label": DOMAINS[e["domain"]], "provider": e["provider"],
                    "provider_label": e["provider_label"], "family": e["family"], "same_as": same, **labels(s),
                    "_key": (list(DOMAINS).index(e["domain"]), [k for k, _, _ in PROVIDERS].index(e["provider"])
                             if e["provider"] != "other" else 99, e["family_rank"], -s.params(), _variant(e["id"]))})
    out.sort(key=lambda d: d["_key"])
    for d in out:
        del d["_key"]
    return out


def unlisted_models() -> list[dict]:
    """Fetched releases not listed separately, with the reason (merged into a same-structure entry, or redundant)."""
    by_id = {e["id"]: e for e in entries()}
    out = []
    for e in entries():
        if e["merged_into"]:
            a, b = get_model(e["id"]), get_model(e["merged_into"])
            d = a.params() / b.params() - 1
            rd = (a.release_params or 0) / (b.release_params or 1) - 1
            same = "同结构、同 dtype" + ("" if abs(rd) < 1e-4 else f"（发布参数差 {rd:+.2%}）")
            out.append({"id": e["id"], "label": e["label"],
                        "reason": f"与 {by_id[e['merged_into']]['label']} {same}，已合并（仍可按 id 评估）"})
    out += [{"id": e["id"], "label": e["label"], "reason": e["unlisted"] + "，尺寸已覆盖，不单列（仍可按 id 评估）"}
            for e in entries() if e["unlisted"]]
    return out
