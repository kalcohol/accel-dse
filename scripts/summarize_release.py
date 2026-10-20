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
import math
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
    ("vision", r"(vision|visual|vit\.|image_|mm_projector|audio|multi_modal_projector|patch_embed|aligner\.|(^|\.)merger\.|^vit_large_projector|^speech_embeddings\.|^linear_proj\.)"),
    ("mtp", r"(^mtp\.|\.mtp\.|nextn|shared_head|\.eh_proj|\.enorm|\.hnorm)"),
    ("embed", r"(embed_tokens|\bwte\b|tok_embeddings|word_embeddings|^embed\.|\.embed\.weight|ple_embedding|^rwkv7\.emb\.weight$|^backbone\.embeddings\.)"),
    ("lm_head", r"(lm_head|^output\.weight$|embed_out|^head\.weight$)"),
    ("norm", r"(norm\.|norm$|_norm|ln_|layernorm|\.ln\d)"),
    ("router", r"(\.gate\.weight$|\.router\.|gate\.e_score_correction|\.gate\.bias$|shared_expert_gate|\.gate\.tid2eid)"),
    ("shared_expert", r"(shared_expert|\.share_expert\.)"),
    ("expert", r"(\.experts\.|block_sparse_moe\.experts|routed_expert|\.moe\.(gate|up|down)_proj\.)"),
    ("mlp", r"(\.mixer\.(gate_proj|up_proj|down_proj|suv_gate|suv_up)|\.mlp_alpha$)"),   # 0.253 Dory / Nemotron-H MLP mixers
    ("attn", r"(self_attn|attention|\.attn\.|linear_attn|\.mixer\.|\.attn_|\.conv\.(in_proj|out_proj|conv)\.|^rwkv7\.blocks\.\d+\.att\.)"),  # 0.114 LFM2 short conv; 0.191 RWKV-7
    ("mlp", r"(\.mlp\.|feed_forward|\.ffn\.)"),
]


# Non-LLM releases (video DiT denoisers, protein language models): domain role tables.  Order matters.
ROLE_DOMAIN = {
    "video": [
        ("cond", r"(modulation|time_embedding|time_projection|norm\d\.linear|norm_out\.linear|text_embedding|text_proj|"
                 r"time_text_embed|time_embed|context_embedder|caption_proj|t_block|t_embedder|fps_embedder|y_embedder|"
                 r"adaln_proj|norm1_context\.linear|\.norm\.linear|scale_shift_table|token_refiner|_mod\.\d|stream_modulation|"
                 r"time_guidance_embed|text_embedder)"),  # 0.115 image DiT: Qwen-Image img_/txt_mod, FLUX.2 shared modulation
        ("norm", r"(norm_post_|norm_[qk]\.|norm_added_[qk]|[qk]_norm\.|norm\d?\.(weight|bias)$|norm_final|norm_out\.norm|"
                 r"\.norm\.(weight|bias)$|^final_norm_video\.)"),   # 0.257 MOSS-Image final norm
        ("attn", r"(self_attn|cross_attn|attn1|attn2|\.attn\.|\.proj\.[qkvo]\.)"),   # 0.257 MOSS-Image layers.N.proj.{q,k,v,o}
        # 0.144 MoE DiT (LingBot-Video): routed / shared experts and the router keep their LLM roles
        ("router", r"\.ffn\.router\."),
        ("shared_expert", r"\.ffn\.shared_experts\."),
        ("expert", r"\.ffn\.experts\."),
        ("mlp", r"(\.ffn\.|\.ff\.|ff_context|\.mlp\.|proj_mlp|single_transformer_blocks\.\d+\.proj_out|_mlp\.|\.proj\.ffn_)"),
        ("io", r"(^video_embedder|^final_linear_video|patch_embed|^proj_out|^head\.|x_embedder|^proj_in|audio_proj|final_layer\.linear|^img_in|^txt_in|^txt_norm)"),
    ],
    "protein": [
        ("buffer", r"(inv_freq$|position_ids$)"),
        ("pos_embed", r"position_embeddings"),
        ("embed", r"word_embeddings"),
        ("head_misc", r"contact_head"),
        ("lm_head", r"^lm_head\."),
        ("norm", r"(LayerNorm|layer_norm|emb_layer_norm)"),
        ("attn", r"\.attention\."),
        ("mlp", r"(\.intermediate\.|\.output\.dense)"),
    ],
    # 0.106 genomics encoders (Carbon-A / Generanno: Llama-style blocks, per-token strand heads score.{0,1})
    "genomics": [
        ("buffer", r"(inv_freq$|position_ids$)"),
        ("embed", r"embed_tokens"),
        ("lm_head", r"^score\.\d+\."),
        ("norm", r"(layernorm|\.norm\.)"),
        ("attn", r"self_attn"),
        ("mlp", r"\.mlp\."),
    ],
}
BLOCK_RE = re.compile(r"(blocks|layers|layer|transformer_blocks)\.(\d+)\.")


def domain_of(cfg: dict) -> str:
    # 0.115: image DiTs (Qwen-Image, FLUX.2) use the video role table (gen domain, 1-frame latent)
    cls = cfg.get("_class_name") or ""
    if cls in ("WanModel", "WanTransformer3DModel", "CogVideoXTransformer3DModel", "HunyuanVideoTransformer3DModel",
               "LTXVideoTransformer3DModel", "MochiTransformer3DModel", "MiniMaxH3Transformer3DModel",
               "QwenImageTransformer2DModel", "QwenImage21Transformer2DModel", "Flux2Transformer2DModel", "Flux3Transformer2DModel",
               "LingBotVideoTransformer3DModel", "MOSSImageTransformer") \
            or cfg.get("model_type") == "STDiT3":
        return "video"
    if cfg.get("model_type") == "esm":
        return "protein"
    if cfg.get("model_type") == "generanno":
        return "genomics"
    return "llm"


def classify_domain(name: str, domain: str) -> str:
    for role, pat in ROLE_DOMAIN[domain]:
        if re.search(pat, name):
            return role
    return "other"


def classify(name: str, n_layers: int | None = None) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if n_layers and m and int(m.group(1)) >= n_layers and not re.search(ROLE[0][1], name):
        return "mtp"
    for role, pat in ROLE:
        if re.search(pat, name):
            return role
    return "other"


def _whisper_role(name: str) -> str:
    """0.270 Whisper encoder-decoder (WhisperForConditionalGeneration): the audio encoder → vision (encoder side);
    decoder self-attention + cross-attention (encoder_attn) → attn, fc1 / fc2 → mlp, learned positions → pos_embed."""
    if name.startswith(("model.encoder.", "encoder.")):
        return "vision"
    if "embed_positions" in name:
        return "pos_embed"
    if "embed_tokens" in name:
        return "embed"
    if name.startswith("proj_out"):
        return "lm_head"
    if "layer_norm" in name:
        return "norm"
    if "self_attn." in name or "encoder_attn." in name:
        return "attn"
    if ".fc1." in name or ".fc2." in name:
        return "mlp"
    return "other"


def _step_audio_role(name: str, n_layers: int | None) -> str:
    """0.272 Step-Audio-2-mini (modeling_step_audio_2.py StepAudio2ForCausalLM): Whisper-style audio encoder
    (encoder.*) and its Adaptor (adapter.*) → vision; Qwen2Model (model.*) + untied lm_head → the LLM."""
    if name.startswith(("encoder.", "adapter.")):
        return "vision"
    return classify(name, n_layers)


def _omni_role(name: str, n_layers: int | None) -> str:
    """0.271 Qwen2.5-Omni (Qwen2_5OmniModel): the Thinker (thinker.model / thinker.lm_head) is the LLM, its audio tower
    and ViT → vision; the speech-output Talker (talker.*) and token2wav (DiT + BigVGAN, token2wav.*) are separate
    non-LLM roles (params_talker / params_token2wav), modelled by core/omni.py."""
    if name.startswith("talker."):
        return "talker"
    if name.startswith("token2wav."):
        return "token2wav"
    return classify(name, n_layers)


def _kimi_role(name: str, n_layers: int | None) -> str:
    """0.273 Kimi-Audio-7B-Instruct (modeling_moonshot_kimia.py MoonshotKimiaForCausalLM): Whisper-large-v3 encoder
    (whisper.*, from the repo's whisper-large-v3/ — scripts/fetch_pt_release.py --kimi) + vq_adaptor → vision; the
    28-layer Qwen2 trunk, the 6 mimo layers (audio branch from layer 21), both heads (lm_head text / mimo_output
    audio) → the LLM; audio detokenizer DiT (detok.*) → flow; BigVGAN (voc.*) → vocoder."""
    if name.startswith(("whisper.", "model.vq_adaptor.")):
        return "vision"
    if name.startswith("detok."):
        return "flow"
    if name.startswith("voc."):
        return "vocoder"
    if name == "mimo_output.weight":
        return "lm_head"
    if name.endswith(".rotary_emb.inv_freq"):      # RoPE buffer saved in the checkpoint (bf16), not a parameter
        return "buffer"
    return classify(name, n_layers)


def _moshi_role(name: str, n_layers: int | None) -> str:
    """0.274 Kyutai Moshi (moshi.models.lm.LMModel): temporal transformer (transformer.*, out_norm) + text_emb / 16
    audio emb tables + text_linear → the LLM; the depth transformer (depformer*, linears.*) → depth; Mimi codec
    (mimi.*, from moshi_audio.json) → codec."""
    if name.startswith("mimi."):
        return "codec"
    if name.startswith(("depformer", "linears.")):
        return "depth"
    if name.startswith(("emb.", "text_emb.")):
        return "embed"
    if name.startswith("text_linear."):
        return "lm_head"
    if ".self_attn." in name:
        return "attn"
    if ".gating." in name:
        return "mlp"
    return "norm" if name.endswith(".alpha") else "other"


def _firered_role(name: str, n_layers: int | None) -> str:
    """0.275 FireRedTTS3-Base (fireredtts3_base.py FireRedTTS3BaseCore): the Qwen3-1.7B backbone (backbone_llm.*, a
    Qwen3Model — no lm_head) → the LLM; the per-step speech head (DiT flow head, dit_head, patch encoder, stop head,
    speaker projections) → tts; the RedAE codec (redae.*, from firered_audio.json) → codec."""
    if name.startswith("redae."):
        return "codec"
    if name.startswith("backbone_llm."):
        return classify("model." + name[len("backbone_llm."):], n_layers)
    return "tts"


def _fish_role(name: str, n_layers: int | None) -> str:
    """0.276 Fish Audio Drama 3 (vllm-omni fish_omni_model.py / codebook_tower.py): the Qwen3.5-MoE backbone
    (model.language_model.*) → the LLM; the 13 codebook-MTP tower layers run on every backbone row with their own KV
    cache → LLM roles too (attention / dense MLP / norms; eh_proj with the attention); the 9 codebook heads and the
    candidate-row lm_head → lm_head; the frame-input codebook embeddings and the tower's per-layer code tables → embed
    (rows looked up).  The DAC codec ships as codec.pth (not read here)."""
    if name.startswith("audio_decoder"):
        if ".heads." in name:
            return "lm_head"
        if "embedding" in name:
            return "embed"
        if "feed_forward." in name:
            return "mlp"
        if name.endswith("norm.weight"):
            return "norm"
        return "attn"            # attention.wqkv / wo, eh_proj
    return classify(name, n_layers)


def _cosy_role(name: str, n_layers: int | None) -> str:
    """0.271 CosyVoice2 / Fun-CosyVoice3 (scripts/fetch_pt_release.py): the Qwen2.5-0.5B backbone is the LLM; its text
    embedding, the stored text lm_head (unused by the TTS loop) and the speech-token embedding are tables (embed role —
    looked up / stored); ``llm_decoder`` (hidden → speech tokens) is the decode head; flow.* / hift.* → flow / vocoder."""
    if name.startswith("flow."):
        return "flow"
    if name.startswith("hift."):
        return "vocoder"
    if name.startswith("llm_decoder."):
        return "lm_head"
    if name.startswith(("speech_embedding.", "llm_embedding.", "llm.model.lm_head.")):
        return "embed"
    return classify(name, n_layers)


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
    # 0.175: NVFP4 (ModelOpt quant_algo NVFP4 / compressed-tensors nvfp4-pack-quantized): e2m1 + e4m3 per 16 + fp32 per tensor
    _q = json.dumps(qc or {}).lower()
    nv4 = "nvfp4" in _q
    tensors = {}
    subs = sorted({k.split("/", 1)[0] for k in headers if "/" in k})
    for key, h in headers.items():
        pre = ""
        if subs and key.split("/", 1)[0] != subs[0] and "/" in key:
            pre = key.split("/", 1)[0] + "::"       # 2nd+ denoiser of a multi-expert release (Wan2.2 A14B)
        for name, m in h.items():
            name = pre + name if name != "__metadata__" else name
            if name != "__metadata__":
                n = 1
                for s in m["shape"]:
                    n *= int(s)
                tensors[name] = (m["dtype"], m["shape"], n, m["data_offsets"][1] - m["data_offsets"][0])
    ka = d / "kimi_audio.json"
    if ka.exists():       # 0.273 Kimi-Audio: whisper encoder + detokenizer + vocoder (outside the safetensors index)
        kj = json.loads(ka.read_text())
        for name, (dt, shape) in kj["tensors"].items():
            n = math.prod(shape)
            tensors[name] = (dt, shape, n, n * {"F32": 4, "F16": 2, "BF16": 2}[dt])
    mo = d / "moshi_audio.json"
    if mo.exists():       # 0.274 Moshi: Mimi codec safetensors (fp32), outside model.safetensors
        for name, (dt, shape) in json.loads(mo.read_text())["tensors"].items():
            n = math.prod(shape)
            tensors[name] = (dt, shape, n, n * {"F32": 4, "F16": 2, "BF16": 2}[dt])
    fr = d / "firered_audio.json"
    if fr.exists():       # 0.275 FireRedTTS3: RedAE codec safetensors (fp32), outside the base model.safetensors
        for name, (dt, shape) in json.loads(fr.read_text())["tensors"].items():
            n = math.prod(shape)
            tensors[name] = (dt, shape, n, n * {"F32": 4, "F16": 2, "BF16": 2}[dt])
    if cfg.get("model_type") == "fireredtts3":
        tc = cfg.get("backbone") or tc
    nl = tc.get("num_hidden_layers") or cfg.get("num_hidden_layers") or cfg.get("num_layers")
    domain = domain_of(cfg)
    weights: dict[str, dict] = {}   # base name -> {role, fmt, params, bytes}
    scale_bytes: dict[str, int] = {}
    buffers = 0
    for name, (dt, shape, n, nb) in tensors.items():
        is_blocks = name.endswith("_blocks")
        if dt == "I64":
            buffers += nb
            continue
        if not is_blocks and SCALE_RE.search(name) and _base(name) != name and not name.startswith(("mimi.", "redae.")):
            scale_bytes[_base(name)] = scale_bytes.get(_base(name), 0) + nb
            continue
        if is_blocks or (dt in ("U8", "I8") and (fp4 or name.endswith("weight_packed"))):
            fmt, params = ("nvfp4" if nv4 else "mxfp4"), n * 2
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
        role = (_whisper_role(name) if cfg.get("model_type") == "whisper" else
                _omni_role(name, nl) if cfg.get("model_type") == "qwen2_5_omni" else
                _step_audio_role(name, nl) if cfg.get("model_type") == "step_audio_2" else
                _moshi_role(name, nl) if cfg.get("model_type") == "moshi" else
                _firered_role(name, nl) if cfg.get("model_type") == "fireredtts3" else
                _fish_role(name, nl) if cfg.get("model_type") == "fish_omni" else
                _kimi_role(name, nl) if "MoonshotKimiaForCausalLM" in (cfg.get("architectures") or []) else
                _cosy_role(name, nl) if cfg.get("model_type") in ("cosyvoice2", "cosyvoice3") else
                classify(name, nl) if domain == "llm" else classify_domain(name, domain))
        if role == "buffer":
            buffers += nb
            continue
        weights[b] = {"role": role, "fmt": fmt, "params": params, "bytes": nb}
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
    llm_roles = [r for r in roles if r not in ("vision", "mtp", "buffer", "scale", "talker", "token2wav", "flow", "vocoder", "depth", "codec", "tts")]
    tot = lambda rs, k: sum(v[k] for r in rs for v in roles.get(r, {}).values())
    extra = {}
    for r in ("talker", "token2wav", "flow", "vocoder", "depth", "codec", "tts"):   # 0.271 speech output (Omni / CosyVoice; absent → no key)
        if r in roles:
            extra[f"params_{r}"] = tot([r], "params")
            extra[f"bytes_{r}"] = tot([r], "bytes")
    if "flow" in roles:                        # 0.271 CosyVoice: flow / HiFT tensor shapes drive core/cosyvoice.py
        extra["speech_tensors"] = {n: t[1] for n, t in tensors.items() if n.startswith(("flow.", "hift."))}
    if mo.exists():
        # 0.274 Moshi: depformer + Mimi tensor shapes drive core/moshi.py
        extra["speech_tensors"] = {n: list(t[1]) for n, t in tensors.items() if n.startswith(("depformer", "linears.", "mimi."))}
    if fr.exists():
        # 0.275 FireRedTTS3: speech head + RedAE tensor shapes drive core/fireredtts3.py
        extra["speech_tensors"] = {n: list(t[1]) for n, t in tensors.items() if not n.startswith("backbone_llm.")}
    if ka.exists():
        extra["speech_tensors"] = {n: list(t[1]) for n, t in tensors.items() if n.startswith(("detok.", "voc."))}
        extra["kimi_audio"] = kj["config"]
        extra["whisper_tensors"] = {n: list(t[1]) for n, t in tensors.items() if n.startswith("whisper.")}
    t2w = d / "token2wav.json"
    if t2w.exists():
        # 0.273 Step-Audio-2-mini: CosyVoice2-style token2wav shipped as .pt (scripts/fetch_pt_release.py --token2wav):
        # flow / HiFT params (fp32) + tensor shapes + yaml fields, outside params_llm / params_vision
        tw = json.loads(t2w.read_text())
        sz = lambda pre: sum(math.prod(v[1]) for k, v in tw["tensors"].items() if k.startswith(pre))
        extra.update(params_flow=sz("flow."), bytes_flow=4 * sz("flow."), params_vocoder=sz("hift."),
                     bytes_vocoder=4 * sz("hift."), speech_tensors={k: v[1] for k, v in tw["tensors"].items()},
                     token2wav=tw["config"])
    if domain != "llm":
        # tensor shapes outside the repeated blocks + of block 0 (io / conditioning dims come from the header)
        extra["domain"] = domain
        last: dict[str, int] = {}
        for n in tensors:
            m_ = BLOCK_RE.search(n)
            if m_ and "::" not in n:
                last[n[:m_.start(2)]] = max(last.get(n[:m_.start(2)], 0), int(m_.group(2)))
        # block 0 and the last block of every stack (the last block may differ: Mochi context_pre_only)
        extra["shapes"] = {n: t[1] for n, t in sorted(tensors.items()) if "::" not in n and
                           (not BLOCK_RE.search(n) or int(BLOCK_RE.search(n).group(2)) in
                            (0, last.get(n[:BLOCK_RE.search(n).start(2)], 0)))}
        if subs:
            extra["experts"] = {s_: sum(t[2] for n, t in tensors.items() if (n.startswith(s_ + "::") if i else "::" not in n))
                                for i, s_ in enumerate(subs)}
        src = d / "source.json"
        if src.exists():
            extra["source"] = json.loads(src.read_text())
    return {
        **extra,
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
