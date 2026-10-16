#!/usr/bin/env python3
"""0.170: release summaries for Arc Institute Evo 2 (StripedHyena 2) → accel_dse/data/releases/arcinstitute__*.json.

Evo 2 ships one PyTorch zip checkpoint per model on Hugging Face (``arcinstitute/evo2_*``; the HF config.json only
names the architecture).  Sources, both public and read without downloading weights:
  * tensor names / shapes / dtypes: ``scripts/fetch_torch_ckpt.py`` over HTTP range (zip directory + data.pkl only)
    → local/evo2/<repo>.json
  * hyper-parameters: the inference configs of github.com/ArcInstitute/evo2 (``evo2/configs/*.yml``), the files the
    ``Evo2(<name>)`` loader pairs with each checkpoint.
The yml keys are kept verbatim under ``config`` (+ HF-style aliases the core reads: model_type, hidden_size, …).
usage: build_evo2_release.py   (after fetch_torch_ckpt.py for each repo)
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "accel_dse" / "data" / "releases"
YML = "https://raw.githubusercontent.com/ArcInstitute/evo2/main/evo2/configs/{}.yml"
# repo → yml: the HF README lists evo2_40b / evo2_7b as the 1M-context checkpoints and evo2_1b_base as 8192-context;
# layer counts (50 / 32 / 25) match num_layers of the paired yml and the checkpoint's blocks.*
MODELS = {"arcinstitute/evo2_40b": "evo2-40b-1m", "arcinstitute/evo2_7b": "evo2-7b-1m",
          "arcinstitute/evo2_1b_base": "evo2-1b-8k"}
DT = {"BF16": ("bf16", 2), "F32": ("fp32", 4), "F16": ("fp16", 2)}


def _yml(text: str) -> dict:
    """Flat ``key: value`` yml (the evo2 configs have no nesting) → dict with ints / floats / bools / lists."""
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or ":" not in line:
            continue
        k, v = (s.strip() for s in line.split(":", 1))
        if v.startswith("["):
            out[k] = [int(x) for x in v.strip("[]").split(",") if x.strip()]
        elif v in ("True", "False", "true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def role(name: str) -> str:
    if name.startswith("embedding_layer"):
        return "embed"
    if name.startswith("unembed"):
        return "lm_head"
    if name.endswith(("filter.t", "inv_freq")):
        return "buffer"            # non-parameter buffers stored in the state dict (time index, RoPE frequencies)
    if "norm" in name:
        return "norm"
    if ".mlp." in name:
        return "mlp"
    return "attn"                  # sequence mixers: hyena projections / filters / out_filter_dense, MHA Wqkv / out


def summarize(repo: str, yml: str) -> dict:
    hdr = json.loads((ROOT / "local" / "evo2" / (repo.split("/")[1] + ".json")).read_text())
    cfg = _yml(urllib.request.urlopen(YML.format(yml), timeout=60).read().decode())
    cfg.update({"model_type": "stripedhyena2", "architectures": ["StripedHyena2"], "num_hidden_layers": cfg["num_layers"],
                "intermediate_size": cfg["inner_mlp_size"], "max_position_embeddings": cfg["max_seqlen"],
                "tie_word_embeddings": cfg["tie_embeddings"], "config_source": YML.format(yml)})
    roles: dict = {}
    for name, t in hdr["tensors"].items():
        fmt, b = DT[t["dtype"]]
        n = math.prod(t["shape"])
        r = roles.setdefault(role(name), {}).setdefault(fmt, {"params": 0, "bytes": 0})
        r["params"] += n
        r["bytes"] += n * b
    p = sum(v["params"] for k, d in roles.items() if k != "buffer" for v in d.values())
    by = sum(v["bytes"] for k, d in roles.items() if k != "buffer" for v in d.values())
    return {"repo": repo, "config": cfg, "roles": roles, "params_llm": p, "bytes_llm": by, "params_mtp": 0,
            "bytes_mtp": 0, "params_vision": 0, "bytes_total": hdr["size"], "n_tensors": len(hdr["tensors"]),
            "aliases_dropped": hdr["aliases_dropped"], "checkpoint": hdr["url"], "torch_dtype": "bfloat16",
            "quantization_config": None, "fetched": time.strftime("%Y-%m-%d")}


def main():
    for repo, yml in MODELS.items():
        s = summarize(repo, yml)
        (OUT / (repo.replace("/", "__") + ".json")).write_text(json.dumps(s, indent=1, sort_keys=True))
        print(repo, s["params_llm"], {k: sum(v["params"] for v in d.values()) for k, d in s["roles"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
