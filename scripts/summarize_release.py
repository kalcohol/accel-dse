#!/usr/bin/env python3
"""Re-summarise cached safetensors headers into accel_dse/data/releases/*.json.

Turns raw header entries into *logical* parameters and *stored* bytes per
(role, format), so the core can use as-released dtypes and V1 can compare
param counts.

Packed formats (logical params per stored element):
  * U8/I8 weight with an fp4 hint in quantization_config (mxfp4 / fp4 / e2m1)
    or gpt-oss ``*_blocks`` tensors           -> x2   (format mxfp4)
  * I32 ``weight_packed`` / ``qweight`` (compressed-tensors pack-quantized,
    AWQ, GPTQ)                                 -> x8   (format int4)
  * I8 otherwise                               -> x1   (format int8)
Scale / zero-point / shape tensors are not parameters: their bytes are
attributed to the weight they belong to.  I64 tensors are index buffers
(e.g. hash-routing tables) and are excluded from params, counted as bytes
under role ``buffer``.  Vision / multimodal projector tensors are reported
separately and excluded from the LLM total.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "local" / "hf_cache"
OUT = ROOT / "accel_dse" / "data" / "releases"

SCALE_RE = re.compile(r"(\.weight_scale_inv$|\.scale_inv$|\.weight_scale(_2)?$|\.scale$|_scales$|\.scales$|"
                      r"\.qzeros$|\.g_idx$|\.weight_shape$|\.input_scale$|\.weight_global_scale$|"
                      r"\.input_global_scale$|\.weight_zero_point$|_scale$)")
ROLE = [
    ("vision", r"(vision|visual|vit\.|image_|mm_projector|audio|multi_modal_projector|patch_embed)"),
    ("mtp", r"(^mtp\.|\.mtp\.|nextn|shared_head|\.eh_proj|\.enorm|\.hnorm)"),
    ("embed", r"(embed_tokens|\bwte\b|tok_embeddings|word_embeddings|^embed\.|\.embed\.weight|ple_embedding)"),
    ("lm_head", r"(lm_head|^output\.weight$|embed_out|^head\.weight$)"),
    ("norm", r"(norm\.|norm$|_norm|ln_|layernorm|\.ln\d)"),
    ("router", r"(\.gate\.weight$|\.router\.|gate\.e_score_correction|\.gate\.bias$|shared_expert_gate|\.gate\.tid2eid)"),
    ("shared_expert", r"shared_expert"),
    ("expert", r"(\.experts\.|block_sparse_moe\.experts|routed_expert)"),
    ("attn", r"(self_attn|attention|\.attn\.|linear_attn|\.mixer\.|\.attn_)"),
    ("mlp", r"(\.mlp\.|feed_forward|\.ffn\.)"),
]


def classify(name: str, n_layers: int | None = None) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if n_layers and m and int(m.group(1)) >= n_layers and not re.search(ROLE[0][1], name):
        return "mtp"
    for role, pat in ROLE:
        if re.search(pat, name):
            return role
    return "other"


def _fp4_hint(qc) -> bool:
    s = json.dumps(qc or {}).lower()
    return any(k in s for k in ("mxfp4", "fp4", "e2m1", "nvfp4"))


def _base(name: str) -> str:
    """Weight a scale tensor belongs to."""
    b = SCALE_RE.sub("", name)
    b = re.sub(r"_(blocks|scales)$", "", b)
    return b


def summarize(repo: str) -> dict:
    d = CACHE / repo.replace("/", "__")
    cfg = json.loads((d / "config.json").read_text())
    headers = json.loads((d / "headers.json").read_text())
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    qc = cfg.get("quantization_config") or tc.get("quantization_config")
    fp4 = _fp4_hint(qc) or "fp4" in json.dumps(cfg.get("expert_dtype") or tc.get("expert_dtype") or "")
    tensors = {}
    for h in headers.values():
        for name, m in h.items():
            if name != "__metadata__":
                n = 1
                for s in m["shape"]:
                    n *= int(s)
                tensors[name] = (m["dtype"], m["shape"], n, m["data_offsets"][1] - m["data_offsets"][0])
    nl = tc.get("num_hidden_layers") or cfg.get("num_hidden_layers") or cfg.get("num_layers")
    weights: dict[str, dict] = {}   # base name -> {role, fmt, params, bytes}
    scale_bytes: dict[str, int] = {}
    buffers = 0
    for name, (dt, shape, n, nb) in tensors.items():
        is_blocks = name.endswith("_blocks")
        if dt == "I64":
            buffers += nb
            continue
        if not is_blocks and SCALE_RE.search(name) and _base(name) != name:
            scale_bytes[_base(name)] = scale_bytes.get(_base(name), 0) + nb
            continue
        if is_blocks or (dt in ("U8", "I8") and (fp4 or name.endswith("weight_packed"))):
            fmt, params = "mxfp4", n * 2
        elif dt == "I32" and re.search(r"(weight_packed|qweight)$", name):
            fmt, params = "int4", n * 8
        elif dt == "I8":
            fmt, params = "int8", n
        elif dt.startswith("F8_E4M3") or dt.startswith("F8_E5M2"):
            fmt, params = "fp8", n
        elif dt == "F8_E8M0":   # stray scale without recognisable suffix
            scale_bytes[_base(name)] = scale_bytes.get(_base(name), 0) + nb
            continue
        else:
            fmt, params = {"BF16": "bf16", "F16": "fp16", "F32": "fp32", "U8": "uint8", "I32": "int32"}.get(dt, dt.lower()), n
        b = _base(name) if is_blocks else re.sub(r"\.(weight_packed|qweight)$", ".weight", name)
        weights[b] = {"role": classify(name, nl), "fmt": fmt, "params": params, "bytes": nb}
    for b, sb in scale_bytes.items():
        key = b if b in weights else (b + ".weight" if b + ".weight" in weights else None)
        if key is None:
            # e.g. "x.weight_scale" -> base "x" ; weight stored as "x.weight"
            cands = [w for w in weights if w.startswith(b)]
            key = cands[0] if len(cands) == 1 else None
        if key is None:
            weights.setdefault("_orphan_scales", {"role": "scale", "fmt": "scale", "params": 0, "bytes": 0})["bytes"] += sb
        else:
            weights[key]["bytes"] += sb
    roles: dict[str, dict[str, dict[str, int]]] = {}
    for w in weights.values():
        e = roles.setdefault(w["role"], {}).setdefault(w["fmt"], {"params": 0, "bytes": 0})
        e["params"] += w["params"]
        e["bytes"] += w["bytes"]
    if buffers:
        roles["buffer"] = {"int64": {"params": 0, "bytes": buffers}}
    llm_roles = [r for r in roles if r not in ("vision", "mtp", "buffer", "scale")]
    tot = lambda rs, k: sum(v[k] for r in rs for v in roles.get(r, {}).values())
    return {
        "repo": repo,
        "fetched": time.strftime("%Y-%m-%d"),
        "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype") or tc.get("torch_dtype") or tc.get("dtype"),
        "quantization_config": qc,
        "n_tensors": len(tensors),
        "params_llm": tot(llm_roles, "params"),
        "params_mtp": tot(["mtp"], "params"),
        "params_vision": tot(["vision"], "params"),
        "bytes_llm": tot(llm_roles, "bytes"),
        "bytes_mtp": tot(["mtp"], "bytes"),
        "bytes_total": sum(t[3] for t in tensors.values()),
        "roles": roles,
        "config": cfg,
    }


def main(argv):
    repos = argv or sorted(p.name.replace("__", "/", 1) for p in CACHE.iterdir() if (p / "headers.json").exists())
    OUT.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        s = summarize(repo)
        (OUT / (repo.replace("/", "__") + ".json")).write_text(json.dumps(s, indent=1, sort_keys=True))
        fm = {f for r, v in s["roles"].items() if r in ("expert", "attn", "mlp", "shared_expert") for f in v}
        print(f"{repo:48s} llm={s['params_llm']/1e9:9.3f}B mtp={s['params_mtp']/1e9:6.2f}B vis={s['params_vision']/1e9:5.2f}B "
              f"bytes={s['bytes_llm']/2**30:8.1f}GiB fmts={sorted(fm)}")


if __name__ == "__main__":
    main(sys.argv[1:])
