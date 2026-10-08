"""Model catalog for core v2: catalog ids → official HF releases → ModelSpec.

Sources: LLM entries of ``data/series_catalog.json`` (labels / aliases) plus
official quantised releases as *separate* entries (as-released dtype) and a
few small references.  Gated repos (meta-llama) use public byte-identical
mirrors (unsloth/*), labelled as mirror provenance.  Video generation and protein entries of the same file are
listed as catalog-only rows (not yet on core v2).  Listing order: vendor (厂商) → family → size.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .domain import from_domain_release
from .model import DATA, ModelSpec, NativeWorkload, from_release, load_release

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

# Non-autoregressive releases on core v2 (0.41): id, hf_id, builder, native workload (official config / README).
# The remaining video / protein entries of series_catalog.json stay catalog-only until modelled.
_WAN_SRC = ("Wan2.1 官方 README / generate.py：t2v {res}、81 帧、16 fps、sample_steps 50、guide_scale 5.0（CFG）、"
            "VAE stride (4, 8, 8)、patch (1, 2, 2)、text_len 512")
_COG_SRC = ("transformer/config.json：sample_frames 49、潜空间 60×90（×8 = 480×720）、temporal_compression_ratio 4；"
            "diffusers CogVideoXPipeline 默认 50 步、guidance_scale 6（CFG）、8 fps")
_ESM_SRC = "config.json：max_position_embeddings 1026（1022 残基 + <cls>/<eos>）；默认 512 残基为工作负载「假设」"
DOMAIN_RELEASES = [
    ("wan2.1-14b", "Wan-AI/Wan2.1-T2V-14B", "wan",
     NativeWorkload("gen", frames=81, height=720, width=1280, fps=16, source=_WAN_SRC.format(res="720P（1280×720）"))),
    ("wan2.1-1.3b", "Wan-AI/Wan2.1-T2V-1.3B", "wan",
     NativeWorkload("gen", frames=81, height=480, width=832, fps=16, source=_WAN_SRC.format(res="480P（832×480）"))),
    ("cogvideox-5b", "zai-org/CogVideoX-5b", "cogvideox",
     NativeWorkload("gen", frames=49, height=480, width=720, fps=8, source=_COG_SRC)),
    ("cogvideox-2b", "zai-org/CogVideoX-2b", "cogvideox",
     NativeWorkload("gen", frames=49, height=480, width=720, fps=8, source=_COG_SRC)),
    ("esm2-3b", "facebook/esm2_t36_3B_UR50D", "esm", NativeWorkload("protein", seq_len=512, steps=1, cfg=1, source=_ESM_SRC)),
    ("esm2-650m", "facebook/esm2_t33_650M_UR50D", "esm", NativeWorkload("protein", seq_len=512, steps=1, cfg=1, source=_ESM_SRC)),
]
_DOMAIN_IDS = {i for i, *_ in DOMAIN_RELEASES}

# ---------------------------------------------------------------- listing: vendor (厂商) → family; domain is only a badge
DOMAINS = {"llm": "LLM（文本）", "vlm": "VLM（多模态，评估语言主干）", "gen": "视频生成（DiT）", "protein": "蛋白质"}
OFFLINE_DOMAINS = {"video": "gen", "protein": "protein"}     # series_catalog domain → listing domain (not on core v2)
PROVIDERS = [   # key, label, orgs (HF org, or gh:<owner>/ for GitHub-only releases) — listing order
    ("alibaba", "阿里（Qwen · Wan）", ("Qwen/", "Wan-AI/")), ("deepseek", "DeepSeek（深度求索）", ("deepseek-ai/",)),
    ("kimi", "Kimi（月之暗面）", ("moonshotai/",)), ("zhipu", "智谱（GLM · CogVideoX）", ("zai-org/", "THUDM/")),
    ("minimax", "MiniMax（稀宇）", ("MiniMaxAI/",)),
    ("meta", "Meta（Llama · ESM）", ("unsloth/", "meta-llama/", "facebook/")),
    ("openai", "OpenAI（gpt-oss）", ("openai/",)), ("mistral", "Mistral AI", ("mistralai/",)),
    ("microsoft", "微软（Phi）", ("microsoft/",)),
    ("bytedance", "字节跳动（Seed · Protenix）", ("ByteDance-Seed/", "gh:bytedance/")),
    ("yi", "零一万物（Yi）", ("01-ai/",)), ("internlm", "上海 AI 实验室（InternLM）", ("internlm/",)),
    # vendors with catalog-only (not yet on core v2) releases
    ("tencent", "腾讯（HunyuanVideo）", ("hunyuanvideo-community/",)), ("lightricks", "Lightricks（LTX-Video）", ("Lightricks/",)),
    ("genmo", "Genmo（Mochi）", ("genmo/",)), ("hpcai", "HPC-AI Tech（Open-Sora）", ("hpcai-tech/",)),
    ("deepmind", "Google DeepMind（AlphaFold）", ("gh:google-deepmind/",)), ("boltz", "Boltz（MIT）", ("gh:jwohlwend/",)),
    ("openfold", "OpenFold（aqlaboratory）", ("gh:aqlaboratory/",)),
]
FAMILIES = [    # provider, family, id prefixes — newest family first; within a family: size ↓, base → FP8 → AWQ
    ("alibaba", "Qwen3.8", ("qwen3.8",)), ("alibaba", "Qwen3.5", ("qwen3.5",)), ("alibaba", "Qwen3-Next", ("qwen3-next",)),
    ("alibaba", "Qwen3 MoE", ("qwen3-235b", "qwen3-30b")), ("alibaba", "Qwen3 稠密", ("qwen3-",)),
    ("alibaba", "Qwen2.5", ("qwen2.5",)), ("alibaba", "Wan2.2", ("wan2.2",)), ("alibaba", "Wan2.1", ("wan2.1",)),
    ("deepseek", "DeepSeek-V4", ("deepseek-v4",)), ("deepseek", "DeepSeek-V3", ("deepseek-v3", "deepseek-r1")),
    ("kimi", "Kimi-K3", ("kimi-k3",)), ("kimi", "Kimi-K2", ("kimi-k2",)),
    ("zhipu", "GLM-5", ("glm-5",)), ("zhipu", "GLM-4.5 / 4.6", ("glm-4",)), ("zhipu", "CogVideoX", ("cogvideox",)),
    ("minimax", "MiniMax-M1 / Text-01", ("minimax-m1", "minimax-text")), ("minimax", "MiniMax-H", ("minimax-h",)),
    ("meta", "Llama 3", ("llama-3",)), ("meta", "ESM-2", ("esm2",)), ("meta", "ESMFold", ("esmfold",)),
    ("openai", "gpt-oss", ("gpt-oss",)),
    ("mistral", "Mistral Large", ("mistral-large",)), ("mistral", "Mixtral", ("mixtral",)), ("mistral", "Magistral", ("magistral",)),
    ("microsoft", "Phi-4", ("phi-4",)), ("bytedance", "Seed-OSS", ("seed-oss",)), ("bytedance", "Protenix", ("protenix",)),
    ("tencent", "HunyuanVideo", ("hunyuanvideo",)), ("lightricks", "LTX-Video", ("ltx-video",)),
    ("genmo", "Mochi 1", ("mochi",)), ("hpcai", "Open-Sora", ("opensora",)),
    ("deepmind", "AlphaFold", ("alphafold",)), ("boltz", "Boltz", ("boltz",)), ("openfold", "OpenFold", ("openfold",)),
]
_PROV_RANK = {k: i for i, (k, _, _) in enumerate(PROVIDERS)}
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
    old = {e["id"]: e for e in raw["entries"]}
    for i, h, b, wl in DOMAIN_RELEASES:
        al = tuple(old.get(i, {}).get("aliases") or ()) + (h,)
        out.append({"id": i, "hf_id": h, "aliases": al, "builder": b, "workload": wl})
    out = [e for e in out if load_release(e["hf_id"]) is not None]
    for e in out:
        e["label"] = _repo_name(e["hf_id"])
        e["provider"], e["provider_label"] = provider_of(e["hf_id"])
        e["family_rank"], e["family"] = family_of(e["provider"], e["id"])
        if e.get("builder"):
            e["domain"] = e["workload"].kind
        else:
            e["domain"] = "vlm" if (load_release(e["hf_id"]).get("params_vision") or 0) > 0 else "llm"
        e["merged_into"] = MERGED.get(e["id"])
        e["unlisted"] = UNLISTED.get(e["id"])
        e["listed"] = not e["merged_into"] and not e["unlisted"]
    return out


def _meta(e: dict) -> dict[str, str]:
    """``metadata`` string of series_catalog.json ('k:v | k:v | …') → dict."""
    out = {}
    for part in (e.get("metadata") or "").split(" | "):
        k, _, v = part.partition(":")
        if v:
            out[k.strip()] = v.strip()
    return out


def _org_key(e: dict, meta: dict) -> str:
    """HF id, or ``gh:<owner>/<repo>`` for releases published on GitHub only (AlphaFold, Boltz, …)."""
    if e.get("hf_id"):
        return e["hf_id"]
    src = meta.get("source", "")
    if src.startswith("https://github.com/"):
        return "gh:" + "/".join(src[len("https://github.com/"):].split("/")[:2])
    return ""


@lru_cache(maxsize=1)
def offline_entries() -> list[dict]:
    """Video generation (DiT) and protein entries of series_catalog.json: listed in the catalog under their vendor,
    but not yet on core v2 (not evaluable).  Dims are the historical catalog's (public configs / papers)."""
    raw = json.loads((DATA / "series_catalog.json").read_text())
    out = []
    for e in raw["entries"]:
        dom = OFFLINE_DOMAINS.get(e.get("domain"))
        if dom is None or e["id"] in UNLISTED or e["id"] in _DOMAIN_IDS:
            continue
        meta, sh = _meta(e), e.get("shape") or {}
        org = _org_key(e, meta)
        prov, plab = provider_of(org)
        rank, fam = family_of(prov, e["id"])
        dims = f"{sh.get('n_layers', '?')} 层 · hidden {sh.get('hidden', '?')}"
        if sh.get("pair_dim"):
            dims += f" · pair {sh['pair_dim']}"
        out.append({"id": e["id"], "hf_id": e.get("hf_id"), "source": meta.get("source", ""),
                    "label": _repo_name(e["hf_id"]) if e.get("hf_id") else e.get("user_label") or e["id"],
                    "domain": dom, "domain_label": DOMAINS[dom], "provider": prov, "provider_label": plab,
                    "family": fam or plab, "family_rank": rank, "status": "暂未接入 v2", "evaluable": False,
                    "arch": ("DiT · " if dom == "gen" else "") + dims, "arch_detail": meta.get("arch", ""),
                    "_size": (sh.get("n_layers") or 0) * (sh.get("hidden") or 0) ** 2})
    out.sort(key=lambda d: (_PROV_RANK.get(d["provider"], 99), d["family_rank"], -d["_size"]))
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
    if e.get("builder"):
        return from_domain_release(e["id"], e["hf_id"], e["builder"], e["workload"])
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
    if spec.kv_cache:
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
        "model_domain": spec.domain,
        "kv_cache": spec.kv_cache,
        "workload": _workload_dict(spec),
        "heads": spec.layers[0].core.n_q if spec.layers else 0,
    }


def _workload_dict(spec: ModelSpec) -> dict | None:
    w = spec.workload
    if w is None:
        return None
    if w.kind == "gen":
        return {"kind": "gen", "frames": w.frames, "height": w.height, "width": w.width, "fps": w.fps,
                "steps": w.steps, "cfg": w.cfg, "patch": list(w.patch), "vae": [w.vae_t, w.vae_s, w.vae_s],
                "text_tokens": w.text_tokens, "joint_text": bool(w.prefix_tokens), "source": w.source}
    return {"kind": "protein", "seq_len": w.seq_len, "max_seq": w.max_seq, "source": w.source}


def list_models() -> list[dict]:
    """Listed (evaluable) models, ordered vendor → family → size ↓ → base / FP8 / AWQ.  LLM and VLM releases of
    one vendor (even of one family) sit together; ``domain`` is only a badge."""
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
                    "provider_label": e["provider_label"], "family": e["family"], "same_as": same, "evaluable": True,
                    **labels(s), "_key": (_PROV_RANK.get(e["provider"], 99), e["family_rank"], -s.params(),
                                          _variant(e["id"]))})
    out.sort(key=lambda d: d["_key"])
    for d in out:
        del d["_key"]
    return out


def catalog_listing() -> list[dict]:
    """The full catalog in display order: vendor → (evaluable families, then catalog-only families) → size ↓.
    Evaluable rows are ``list_models()`` items; catalog-only rows (video generation / protein, not yet on core v2)
    are ``offline_entries()`` items with ``evaluable: False``."""
    ms, off = list_models(), offline_entries()
    pos = {m["id"]: i for i, m in enumerate(ms)}
    opos = {o["id"]: i for i, o in enumerate(off)}
    rows = [(_PROV_RANK.get(m["provider"], 99), 0, pos[m["id"]], m) for m in ms]
    rows += [(_PROV_RANK.get(o["provider"], 99), 1, opos[o["id"]], o) for o in off]
    return [r[-1] for r in sorted(rows, key=lambda r: r[:3])]


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
