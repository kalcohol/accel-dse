#!/usr/bin/env python3
"""Fetch release metadata for catalog models from public Hugging Face repos.

For each repo: config.json, (quantization_config / torch_dtype inside it),
model.safetensors.index.json, and every safetensors **header** via HTTP Range
(no weights downloaded). Writes a compact per-model summary used by
accel_dse.core (dtype per tensor role) and by the V1 validation (param counts).

Raw files are cached under local/hf_cache/ (gitignored).
Usage: python3 scripts/fetch_hf_release.py [repo[:subfolder][@revision] ...]   (default: scripts/release_repos.txt)
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    return st_header_url(f"{HF}/{repo}/resolve/main/{fname}")


def st_header_url(url: str) -> dict:
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    return json.loads(_get(url, (8, 8 + n - 1)).decode())


def parse_spec(spec: str) -> tuple[str, str, str]:
    """``repo[:subfolder][@revision]`` → (repo, subfolder, revision).  Diffusers releases keep the denoiser in
    ``transformer/``; a few official repos ship safetensors only on the HF safetensors-conversion PR
    (e.g. ``facebook/esm2_t36_3B_UR50D@refs/pr/2``)."""
    rev = "main"
    if "@" in spec:
        spec, rev = spec.split("@", 1)
    repo, _, sub = spec.partition(":")
    return repo, sub, rev


def _tree_sizes(repo: str) -> dict[str, int]:
    """Sizes of every weight file in the repo (main), for the components we do not model (text encoder, VAE …)."""
    try:
        items = _json(f"{HF}/api/models/{repo}/tree/main?recursive=1")
    except Exception:
        return {}
    return {x["path"]: x.get("size", 0) for x in items
            if x.get("type") == "file" and x["path"].endswith((".safetensors", ".pth", ".bin", ".pt", ".ckpt"))}


def fetch(spec: str) -> dict:
    repo, sub, rev = parse_spec(spec)
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{HF}/{repo}/resolve/{rev.replace('/', '%2F')}/" + (sub + "/" if sub else "")
    cfg = _json(base + "config.json")
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    idx, files = None, None
    for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        try:
            idx = _json(base + name)
            files = sorted(set(idx["weight_map"].values()))
            break
        except urllib.error.HTTPError:
            continue
    if files is None:
        for name in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
            try:
                _get(base + name, (0, 7))
                files = [name]
                break
            except urllib.error.HTTPError:
                continue
    if files is None:
        raise RuntimeError("no safetensors in " + spec)
    bin_total = None
    if rev != "main":   # cross-check against the official main-branch checkpoint index
        try:
            bin_total = _json(f"{HF}/{repo}/resolve/main/" + (sub + "/" if sub else "") +
                              "pytorch_model.bin.index.json")["metadata"]["total_size"]
        except Exception:
            pass
    hdr_path = d / "headers.json"
    headers = json.loads(hdr_path.read_text()) if hdr_path.exists() else {}
    todo = [f for f in files if f not in headers]
    with cf.ThreadPoolExecutor(8) as ex:
        for f, h in zip(todo, ex.map(lambda f: st_header_url(base + f), todo)):
            headers[f] = h
    hdr_path.write_text(json.dumps(headers))
    meta = {"spec": spec, "subfolder": sub, "revision": rev, "main_bin_total_size": bin_total,
            "repo_files": _tree_sizes(repo) if (sub or rev != "main" or "diffusion" in files[0]) else {}}
    (d / "source.json").write_text(json.dumps(meta, indent=1))
    tensors = {}
    for f, h in headers.items():
        for name, m in h.items():
            if name == "__metadata__":
                continue
            tensors[name] = (m["dtype"], m["shape"], m["data_offsets"][1] - m["data_offsets"][0])
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
            fetch(repo)
            # the release summary proper (logical params, packed formats, domain roles) is summarize_release.py's
            import summarize_release
            s = summarize_release.summarize(parse_spec(repo)[0])
            (OUT / (parse_spec(repo)[0].replace("/", "__") + ".json")).write_text(json.dumps(s, indent=1, sort_keys=True))
            print(f"OK   {repo:55s} tensors={s['n_tensors']:6d} params={s['params_llm']/1e9:9.3f}B "
                  f"q={bool(s['quantization_config'])} {time.time()-t0:.1f}s", flush=True)
            ok += 1
        except Exception as e:  # gated / missing
            print(f"SKIP {repo:55s} {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"done {ok}/{len(repos)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
