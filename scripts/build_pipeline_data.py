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
    # 0.116 image DiT pipelines (headers fetched like the others: text_encoder/ and vae/ safetensors of the repo)
    "qwen2.5-vl-7b": ("qi_te.json", "Qwen/Qwen-Image · text_encoder", lambda k: k.startswith("model."), None,
                      "视觉塔（0.68 B）与 lm_head 随 text_encoder 加载、计入存储，文生图不经过它们"),
    "qwen3-vl-8b": ("qi21_te.json", "Qwen/Qwen-Image-2.1-Turbo · text_encoder",
                    lambda k: k.startswith("model.language_model."), None,
                    "视觉塔与 lm_head 随 text_encoder 加载、计入存储，文生图不经过它们"),
    "qwen3-4b": ("fk_te.json", "black-forest-labs/FLUX.2-klein-4B · text_encoder", lambda k: True, None, ""),
    "qwen-image-vae": ("qi_vae.json", "Qwen/Qwen-Image · vae", None, None, ""),
    "qwen-image-2.1-vae": ("qi21_vae.json", "Qwen/Qwen-Image-2.1-Turbo · vae", None, None, ""),
    # 0.144 LingBot-Video: Qwen3-VL-4B text encoder shipped in the release (its VAE = the Wan2.1 VAE, header-identical)
    "qwen3-vl-4b.lingbot": ("lbv_te.json", "robbyant/lingbot-video-moe-dmd-30b-a3b · text_encoder",
                            lambda k: k.startswith("model.language_model."), None,
                            "视觉塔与 lm_head 随 text_encoder 加载、计入存储，文生视频不经过它们"),
    "flux2-vae": ("fk_vae.json", "black-forest-labs/FLUX.2-klein-4B · vae", None, "bf16",
                  "int64 张量为 BatchNorm 计数（num_batches_tracked），权重均为 bf16"),
    # 0.251 FLUX 3 Image Klein (headers of the byte-identical public mirror unsloth/flux-3-image-klein): the text encoder is
    # Qwen3-VL-4B-Instruct (header-identical to the LingBot one), the VAE is AutoencoderKLFlux2 (FLUX.2 shapes) in fp32
    "qwen3-vl-4b.flux3": ("f3_te.json", "black-forest-labs/flux-3-image-klein · text_encoder",
                          lambda k: k.startswith("model.language_model."), None,
                          "视觉塔与 lm_head 随 text_encoder 加载、计入存储，文生图不经过它们"),
    "flux3-vae": ("f3_vae.json", "black-forest-labs/flux-3-image-klein · vae", None, "fp32",
                  "AutoencoderKLFlux2（与 FLUX.2 VAE 同形状）按发布 fp32（config.json vae.dtype float32）；"
                  "int64 张量为 BatchNorm 计数"),
    # 0.257 MOSS-Image (OpenMOSS-Team/MOSS-Image-Base, public): text_encoder/ = Qwen3_5Model (Qwen3.5-9B language
    # model + vision tower, no lm_head / MTP), vae/ = AutoencoderKLFlux2 with the FLUX.2 small decoder, fp32
    "qwen3.5-9b.moss": ("moss_te.json", "OpenMOSS-Team/MOSS-Image-Base · text_encoder",
                        lambda k: k.startswith("language_model."), None,
                        "视觉塔随 text_encoder 加载、计入存储（Qwen3_5Model 无 lm_head / MTP），文生图不经过它"),
    "moss-vae": ("moss_vae.json", "OpenMOSS-Team/MOSS-Image-Base · vae", None, "fp32",
                 "AutoencoderKLFlux2 + FLUX.2 small decoder（decoder 96 / 192 / 384 / 384）按发布 fp32；"
                 "int64 张量为 BatchNorm 计数"),
}


def _vae_used(k: str) -> bool:
    """Decoder half of a VAE (+ post_quant_conv); the encoder is not run for text-to-video."""
    return (k.startswith(("decoder.", "post_quant_conv")) or ".decoder." in k or "post_quant_conv" in k
            or k.startswith("decoder"))


# 0.252: VAEs whose encoder runs in the modelled pipeline (FLUX 3 reference images are VAE-encoded) — their encoder
# half (+ quant_conv, latent BatchNorm) is kept as "enc_tensors"
ENCODE = {"flux3-vae"}


def _vae_enc(k: str) -> bool:
    return k.startswith(("encoder.", "quant_conv", "bn."))


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
    if key in ENCODE:
        out["enc_tensors"] = {k: v["shape"] for k, v in t.items() if _vae_enc(k)}
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
