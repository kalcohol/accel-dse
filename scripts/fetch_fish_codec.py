#!/usr/bin/env python3
"""0.277: Fish Audio Drama 3 DAC codec (codec.pth, a 1.87 GB torch zip pickle) → tensor names / shapes / dtypes.

Read with scripts/fetch_torch_ckpt.py (HTTP range requests on the zip central directory + data.pkl; no weights
downloaded, no torch).  Writes local/hf_cache/fishaudio__drama-3/fish_codec.json and merges the decode-path shapes into
accel_dse/data/releases/fishaudio__drama-3.json (keys from :func:`codec_extra`, the same ones summarize_release.py adds
when fish_codec.json is in the cache):
  codec_tensors          decode path: quantizer.{semantic_quantizer,quantizer}.* (codebooks + out_proj),
                         quantizer.post_module.* (window transformer; causal_mask / freqs_cis buffers dropped),
                         quantizer.upsample.*, decoder.*
  params_codec           every weight tensor of codec.pth (encoder + quantizer + decoder; buffers dropped)
  params_codec_decode    the decode path above;  codec_dtype  F32 (all weight tensors);  codec_file_bytes
Usage: python3 scripts/fetch_fish_codec.py [fishaudio/drama-3]
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUFFERS = ("causal_mask", "freqs_cis")
DECODE = ("quantizer.semantic_quantizer.", "quantizer.quantizer.", "quantizer.post_module.", "quantizer.upsample.",
          "decoder.")


def codec_extra(hdr: dict) -> dict:
    t = {n: v for n, v in hdr["tensors"].items() if not n.endswith(BUFFERS)}
    dec = {n: list(v["shape"]) for n, v in t.items() if n.startswith(DECODE)}
    dts = sorted({v["dtype"] for v in t.values()})
    return {"codec_tensors": dec, "params_codec": sum(math.prod(v["shape"]) for v in t.values()),
            "params_codec_decode": sum(math.prod(s) for s in dec.values()), "codec_dtype": "/".join(dts),
            "codec_file_bytes": hdr["size"]}


def main(argv):
    repo = argv[0] if argv else "fishaudio/drama-3"
    d = ROOT / "local" / "hf_cache" / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    out = d / "fish_codec.json"
    url = f"https://huggingface.co/{repo}/resolve/main/codec.pth"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_torch_ckpt.py"), url, str(out)], check=True)
    ex = codec_extra(json.loads(out.read_text()))
    rel = ROOT / "accel_dse" / "data" / "releases" / (repo.replace("/", "__") + ".json")
    s = json.loads(rel.read_text())
    s.update(ex)
    rel.write_text(json.dumps(s, indent=1, sort_keys=True))
    print(f"{repo}: codec params {ex['params_codec'] / 1e6:.1f} M, decode path {ex['params_codec_decode'] / 1e6:.1f} M "
          f"({len(ex['codec_tensors'])} tensors, {ex['codec_dtype']})")


if __name__ == "__main__":
    main(sys.argv[1:])
