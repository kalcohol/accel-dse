#!/usr/bin/env python3
"""Fetch release metadata for catalog models from public Hugging Face repos.

For each repo: config.json, (quantization_config / torch_dtype inside it),
model.safetensors.index.json, and every safetensors **header** via HTTP Range
(no weights downloaded). Writes a compact per-model summary used by
accel_dse.core (dtype per tensor role) and by the V1 validation (param counts).

Raw files are cached under local/hf_cache/ (gitignored).
Usage: python3 scripts/fetch_hf_release.py [repo_id ...]   (default: all in models.toml list)
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import struct
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "local" / "hf_cache"
OUT = ROOT / "accel_dse" / "data" / "releases"
HF = "https://huggingface.co"
UA = {"User-Agent": "accel-dse-release-fetch/0.40"}


def _get(url: str, rng: tuple[int, int] | None = None, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            h = dict(UA)
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1]}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def _json(url: str):
    return json.loads(_get(url).decode())


def st_header(repo: str, fname: str) -> dict:
    url = f"{HF}/{repo}/resolve/main/{fname}"
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    return json.loads(_get(url, (8, 8 + n - 1)).decode())


def fetch(repo: str) -> dict:
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    cfg = _json(f"{HF}/{repo}/resolve/main/config.json")
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    try:
        idx = _json(f"{HF}/{repo}/resolve/main/model.safetensors.index.json")
        files = sorted(set(idx["weight_map"].values()))
    except urllib.error.HTTPError:
        idx = None
        files = ["model.safetensors"]
    hdr_path = d / "headers.json"
    headers = json.loads(hdr_path.read_text()) if hdr_path.exists() else {}
    todo = [f for f in files if f not in headers]
    with cf.ThreadPoolExecutor(8) as ex:
        for f, h in zip(todo, ex.map(lambda f: st_header(repo, f), todo)):
            headers[f] = h
    hdr_path.write_text(json.dumps(headers))
    tensors = {}
    for f, h in headers.items():
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            tensors[name] = (meta["dtype"], meta["shape"], meta["data_offsets"][1] - meta["data_offsets"][0])
    return {"repo": repo, "config": cfg, "index_total_size": (idx or {}).get("metadata", {}).get("total_size"),
            "tensors": tensors}


def summarize(raw: dict) -> dict:
    by_dtype: dict[str, list[int]] = {}
    roles: dict[str, dict[str, int]] = {}
    for name, (dt, shape, nbytes) in raw["tensors"].items():
        n = 1
        for s in shape:
            n *= int(s)
        e = by_dtype.setdefault(dt, [0, 0, 0])
        e[0] += 1; e[1] += n; e[2] += nbytes
        r = classify(name)
        rr = roles.setdefault(r, {})
        rr[dt] = rr.get(dt, 0) + n
    cfg = raw["config"]
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    return {
        "repo": raw["repo"],
        "fetched": time.strftime("%Y-%m-%d"),
        "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype") or tc.get("torch_dtype") or tc.get("dtype"),
        "quantization_config": cfg.get("quantization_config") or tc.get("quantization_config"),
        "index_total_size": raw["index_total_size"],
        "n_tensors": len(raw["tensors"]),
        "by_dtype": {k: {"tensors": v[0], "elements": v[1], "bytes": v[2]} for k, v in sorted(by_dtype.items())},
        "roles": roles,
        "config": cfg,
    }


_ROLE = [
    ("scale", r"(scale_inv|weight_scale|_scales?$|\.scales$|input_scale|blocks_scale|qzeros|g_idx)"),
    ("vision", r"(vision|visual|vit\.|image_|mm_projector|audio)"),
    ("mtp", r"(\.mtp\.|^mtp|nextn|layers\.(6[1-9]|[7-9][0-9])\.(eh_proj|enorm|hnorm|shared_head))"),
    ("embed", r"(embed_tokens|wte|tok_embeddings|word_embeddings)"),
    ("lm_head", r"(lm_head|output\.weight$|embed_out)"),
    ("norm", r"(norm|ln_|layernorm)"),
    ("router", r"(\.gate\.weight$|router|gate\.e_score_correction|\.gate\.bias)"),
    ("shared_expert", r"shared_expert"),
    ("expert", r"(experts|block_sparse_moe\.experts|\.w[123]\.)"),
    ("attn", r"(self_attn|attention|attn|linear_attn|\.mixer\.)"),
    ("mlp", r"(mlp|feed_forward|ffn)"),
]


def classify(name: str) -> str:
    for role, pat in _ROLE:
        if re.search(pat, name):
            return role
    return "other"


def main(argv: list[str]) -> int:
    repos = argv or [l.strip() for l in (ROOT / "scripts" / "release_repos.txt").read_text().splitlines()
                     if l.strip() and not l.startswith("#")]
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for repo in repos:
        try:
            t0 = time.time()
            s = summarize(fetch(repo))
            (OUT / (repo.replace("/", "__") + ".json")).write_text(json.dumps(s, indent=1, sort_keys=True))
            tot = sum(v["elements"] for v in s["by_dtype"].values())
            print(f"OK   {repo:55s} tensors={s['n_tensors']:6d} elems={tot/1e9:9.3f}B "
                  f"dtypes={list(s['by_dtype'])} q={bool(s['quantization_config'])} {time.time()-t0:.1f}s", flush=True)
            ok += 1
        except Exception as e:  # gated / missing
            print(f"SKIP {repo:55s} {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"done {ok}/{len(repos)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
