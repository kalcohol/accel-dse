"""Build the compact pipeline-component data (text encoders, VAE decoders) used by core/pipeline.py.

Inputs: checkpoint headers fetched over HTTP into local/hf_cache/pipeline/ (safetensors headers read with HTTP
range requests by scripts/fetch_hf_release.st_header; the umT5 .pth with scripts/fetch_torch_ckpt.py).  Only tensor
names / shapes / dtypes are read — no weights are downloaded.

    python3 scripts/build_pipeline_data.py            # → accel_dse/data/pipeline/<key>.json

Each output keeps the tensors the evaluated part of the component uses (text encoder: the text transformer;
VAE: the decoder + post_quant_conv), the stored parameter count of the whole loaded checkpoint, and its dtype.
"""

from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "local", "hf_cache", "pipeline")
OUT = os.path.join(ROOT, "accel_dse", "data", "pipeline")

DT = {"F32": "fp32", "BF16": "bf16", "F16": "fp16", "float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}

# key: (header file, source (repo / sub or URL), used-tensor filter, dtype override, note)
SPECS = {
    "umt5-xxl": ("wan_umt5.json", "Wan-AI/Wan2.1-T2V-1.3B · models_t5_umt5-xxl-enc-bf16.pth",
                 lambda k: True, None, ""),
    "t5-v1_1-xxl.bf16": ("cog_te.json", "zai-org/CogVideoX-5b · text_encoder", lambda k: True, None, ""),
    "t5-v1_1-xxl.fp16": ("cog2_te.json", "zai-org/CogVideoX-2b · text_encoder", lambda k: True, None, ""),
    "t5-v1_1-xxl.fp32": ("ltx_te.json", "Lightricks/LTX-Video · text_encoder", lambda k: True, None, ""),
    "t5-v1_1-xxl.mochi": ("mochi_te.json", "genmo/mochi-1-preview · text_encoder", lambda k: True, None, ""),
    "t5-v1_1-xxl.deepfloyd": ("cog_te.json", "DeepFloyd/t5-v1_1-xxl · pytorch_model-0000{1,2}-of-00002.bin",
                              lambda k: True, "fp32",
                              "DeepFloyd 仅发布 .bin（无 safetensors 头）：形状取同构的 T5 v1.1 XXL 编码器"
                              "（CogVideoX text_encoder 头），dtype 按 .bin 总大小 19.05 GB = 4.76 B × 4 B 判定为 fp32"),
    "llava-llama3-8b": ("hy_te.json", "hunyuanvideo-community/HunyuanVideo · text_encoder", lambda k: True, None, ""),
    "clip-vit-l-text": ("hy_te2.json", "hunyuanvideo-community/HunyuanVideo · text_encoder_2", lambda k: True,
                        None, ""),
    "qwen3-vl-32b": ("h3_te.json", "MiniMaxAI/MiniMax-H3 · text_encoder",
                     lambda k: k.startswith("model.language_model."), None,
                     "视觉塔（0.60 B）与 lm_head（0.78 B）随 text_encoder 加载、计入存储，但文本编码不经过它们"),
    "wan-vae": ("wan_vae.json", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers · vae", None, None, ""),
    "cogvideox-vae.5b": ("cog5_vae.json", "zai-org/CogVideoX-5b · vae", None, None, ""),
    "cogvideox-vae.2b": ("cog2_vae.json", "zai-org/CogVideoX-2b · vae", None, None, ""),
    "hunyuan-vae": ("hy_vae.json", "hunyuanvideo-community/HunyuanVideo · vae", None, None, ""),
    "ltx-vae": ("ltx_vae.json", "Lightricks/LTX-Video · vae", None, None, ""),
    "mochi-vae": ("mochi_vae.json", "genmo/mochi-1-preview · vae", None, None, ""),
    "opensora-vae": ("os_vae.json", "hpcai-tech/OpenSora-VAE-v1.2", None, None, ""),
    "h3-vae": ("h3_vae.json", "MiniMaxAI/MiniMax-H3 · vae", None, None, ""),
    "h3-audio-vae": ("h3_avae.json", "MiniMaxAI/MiniMax-H3 · audio_vae", None, None, ""),
}


def _vae_used(k: str) -> bool:
    """Decoder half of a VAE (+ post_quant_conv); the encoder is not run for text-to-video."""
    return (k.startswith(("decoder.", "post_quant_conv")) or ".decoder." in k or "post_quant_conv" in k
            or k.startswith("decoder"))


def build(key: str) -> dict:
    fn, src, used, dt_over, note = SPECS[key]
    d = json.load(open(os.path.join(SRC, fn)))
    t = d["tensors"]
    used = used or _vae_used
    keep = {k: v["shape"] for k, v in t.items() if used(k)}
    dts = {DT.get(v["dtype"], v["dtype"]) for k, v in t.items()}
    if len(dts) != 1 and not dt_over:
        raise SystemExit(f"{key}: mixed dtypes {dts}")
    out = {"key": key, "source": src, "dtype": dt_over or dts.pop(),
           "params_stored": sum(math.prod(v["shape"]) for v in t.values()),
           "params_used": sum(math.prod(s) for s in keep.values()),
           "config": {k: v for k, v in (d.get("config") or {}).items()
                      if not (isinstance(v, list) and len(v) > 16) and not k.startswith("_")},
           "tensors": keep}
    if note:
        out["note"] = note
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for key in (sys.argv[1:] or SPECS):
        o = build(key)
        with open(os.path.join(OUT, key + ".json"), "w") as f:
            json.dump(o, f, separators=(",", ":"), sort_keys=True)
        print(f"{key:24s} {o['dtype']:5s} stored {o['params_stored'] / 1e9:7.3f} B  used {o['params_used'] / 1e9:7.3f} B"
              f"  {len(o['tensors'])} tensors")


if __name__ == "__main__":
    main()
