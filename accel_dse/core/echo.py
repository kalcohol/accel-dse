"""0.279 Index-Echo S2ST (IndexTeam/Index-Echo-S2ST-2B / -9B): speech → translated, voice-preserved speech in one
process (docs/MODEL.md §224.2).

Reference code 「来源」 (the HF release): modeling_dubbing.py, code/tts/pipeline.py (DubPipeline.extract / synth),
code/stlm/audio_model.py + stlm_api.py (ST LM), code/bridge/hidden2cv.py (Hidden2CVMapper), README "Default inference
settings"; CosyVoice3 = the catalog's Fun-CosyVoice3-0.5B-2512 (llm.pt / flow.pt / hift.pt byte-identical, LFS sha256).

Per request (one source clip, batch 1 — ``dub`` handles one clip; all components on one GPU, README "one CUDA GPU"):
  1. AuT encoder (Qwen3-Omni) + connector on the source audio → audio positions of the ST prompt;
  2. ST LM greedy decode: source transcript, newline, translation (``generate_one``, max_new_tokens 1024);
  3. one more full forward over prompt + generated text (``extract``: ``model.llm(inputs_embeds=…,
     output_hidden_states=True)`` — no KV reuse) → last-layer hidden states of the m translation tokens;
  4. Hidden2CVMapper (fp32): LN(h)·W_h + LN(e)·W_e + pos + lang → 3 bidirectional pre-LN Transformer layers (d_e, GELU
     FFN d_ff) → zero-init out Linear, residual on e (the CV3 text embedding of the m frontend tokens);
  5. CosyVoice3 LM (fp32, ``_patch_cv3_tf56``): prompt = sos + text emb(INSTRUCT + source transcript) + mapped m +
     task + prompt speech tokens of the source audio; samples speech tokens (min 2m, max min(20m, 1500));
  6. CosyVoice3 token2wav with stream=False: one flow pass (prompt + all speech tokens, CFM 10 Euler × CFG 2) + one
     HiFT pass over the generated mel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ir import Op

COSY3_ID = "fun-cosyvoice3-0.5b-2512"
TEXT_TOK_PER_S = 3.0      # text tokens per second of speech (transcript and translation) 「假设」 (~150 words / min × 1.2)
ST_PROMPT = 36            # chat template + instruction + empty think skeleton, tokens 「假设」 (instruction ≈ 22)
CV3_INSTRUCT = 8
ECHO_REASON = ("S2ST：CosyVoice3 前端的 speech_tokenizer_v3（源音频 → 提示语音 token）与 CampPlus 说话人向量（ONNX）未计入；"
               "文本 token / 语音秒按假设值（TEXT_TOK_PER_S）")          # 'You are a helpful assistant.<|endofprompt|>' tokens 「假设」


@dataclass(frozen=True)
class EchoSpec:
    tensors: tuple            # ((name, shape), …) mapper.* (bridge/mapper.safetensors, fp32)
    d_h: int
    d_e: int
    n_layer: int
    n_head: int
    d_ff: int
    max_len: int
    n_lang: int
    cosy_id: str = COSY3_ID
    max_new_tokens: int = 1024
    speech_max: int = 1500
    cap_per_text: int = 20
    min_per_text: int = 2
    label: str = ""

    @property
    def mapper_params(self) -> int:
        return sum(math.prod(s) for _, s in self.tensors)

    def speech_tokens(self, m: int, want: int) -> int:
        """Speech tokens generated for m frontend text tokens when the audio needs ``want`` (25 / s): clipped to
        [2m, min(20m, 1500)] (pipeline.synth min_len / max_len 「来源」)."""
        return max(self.min_per_text * m, min(want, self.cap_per_text * m, self.speech_max))


def echo_spec(cfg: dict, rel: dict | None) -> EchoSpec | None:
    if cfg.get("model_type") != "index_echo_s2st" or not rel:
        return None
    e = cfg.get("echo") or {}
    mc = e.get("mapper") or {}
    t = tuple((n[len("mapper."):], tuple(s)) for n, s in sorted((rel.get("echo_tensors") or {}).items())
              if n.startswith("mapper."))
    d_e = dict(t)["e_proj.weight"][0]
    return EchoSpec(t, mc["d_h"], d_e, mc["n_layer"], mc["n_head"], mc["d_ff"], mc["max_len"], mc.get("n_lang", 0),
                    speech_max=e.get("speech_max_tokens", 1500), cap_per_text=e.get("speech_cap_per_text", 20),
                    min_per_text=e.get("speech_min_per_text", 2), max_new_tokens=e.get("max_new_tokens", 1024),
                    label=f"Index-Echo S2ST（ST LM {cfg['text_config']['hidden_size']} → 映射器 → CosyVoice3）")


def mapper_ops(e: EchoSpec, m: int) -> list[Op]:
    """Hidden2CVMapper over m positions, fp32: one GEMM per weight matrix (Linear [out, in]; in_proj [3d, d]) with its
    bias / LayerNorm read alongside, full bidirectional attention per layer (src_key_padding_mask None), pos / lang
    rows looked up."""
    ops: list[Op] = []
    sh = dict(e.tensors)
    for n, s in e.tensors:
        if len(s) != 2 or n in ("pos", "lang_emb.weight"):
            continue
        mod = n.rsplit(".", 1)[0] if n.endswith(".weight") else n.replace("_weight", "")
        bias = sh.get(mod + ".bias") or sh.get(n.replace("_weight", "_bias")) or (0,)
        k, nn_ = s[1], s[0]
        ops.append(Op("echo.mapper." + n, "gemm", 0, m=m, k=k, n=nn_, role="speech", w_params=k * nn_ + bias[0],
                      w_bits=32.0, w_fmt="fp32", a_fmt="fp32", act_bytes=m * (k + nn_) * 4.0, vec=m * nn_ * 2.0,
                      stream=True))
    hd = e.d_e // e.n_head
    for li in range(e.n_layer):
        ops += [Op(f"echo.mapper.{li}.qk", "attn", 0, m=m, k=hd, n=m, count=e.n_head, a_fmt="fp32",
                   act_bytes=e.n_head * 2 * m * hd * 4.0),
                Op(f"echo.mapper.{li}.pv", "attn", 0, m=m, k=m, n=hd, count=e.n_head, a_fmt="fp32",
                   act_bytes=e.n_head * 2 * m * hd * 4.0),
                Op(f"echo.mapper.{li}.softmax", "vector", 0, vec=e.n_head * m * m * 6.0)]
    norms = sum(math.prod(s) for n, s in e.tensors if "norm" in n)
    ops.append(Op("echo.mapper.norms", "vector", 0, vec=m * 8.0 * (e.d_h + (2 * e.n_layer + 1) * e.d_e),
                  w_params=norms, w_bits=32.0, w_fmt="fp32"))
    ops.append(Op("echo.mapper.pos", "lookup", 0, kv_read=m * e.d_e * 4.0 + (e.d_e * 4.0 if e.n_lang else 0.0)))
    return ops
