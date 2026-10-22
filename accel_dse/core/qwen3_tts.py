"""0.338 Qwen3-TTS (12 Hz discrete multi-codebook LM; docs/MODEL.md §283).

Reference code 「来源」 github QwenLM/Qwen3-TTS modeling_qwen3_tts.py
(Qwen3TTSTalkerForConditionalGeneration.forward / .generate,
Qwen3TTSTalkerCodePredictorModelForConditionalGeneration) + speech_tokenizer
modeling_qwen3_tts_tokenizer_v2.py; HF Qwen/Qwen3-TTS-12Hz-*-{Base,CustomVoice,VoiceDesign}
config.json + safetensors headers.

Per 12.5 Hz frame (24 kHz / encode_downsample_rate 1920 「来源」):
  1. Talker (catalog LLM, bf16): one decode step with KV cache; ``codec_head`` draws the first
     codebook id (vocab 3072).  Input embed = sum of the 16 codebook embeddings (+ trailing text
     projection).  Text tokens use ``text_embedding`` (text_hidden) → ``text_projection`` MLP into
     talker hidden; codec path uses ``codec_embedding`` (talker hidden).
  2. Code predictor (bf16): AR generate of ``num_code_groups − 1`` further codes; each step is a
     5-layer Qwen3 decoder (GQA + qk_norm, SwiGLU) over the growing in-frame sequence, then a
     per-group ``lm_head[i]``.  Prefill seed = [talker_hidden, first_code_embed].
  3. Speech tokenizer decode (fp32, speech_tokenizer/*): RVQ codes → waveform at 24 kHz (non-DiT
     decoder; costed from released decoder.* tensor shapes + config).  Encoder used only for
     voice-clone prompt encode (Base) — storage + one-shot, not the per-frame path.
  4. Speaker encoder (Base only, bf16 CampPlus/Res2Net): once per clone request on the prompt mel.

Real-time ⇔ frame time ≤ 80 ms.  First audio ≈ a few frames + one tokenizer decode window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ir import Op

VEC_PER_OUT = 3.0
FRAME_HZ = 12.5          # 24000 / 1920 「来源」 speech_tokenizer encode_downsample_rate
FRAME_S = 1.0 / FRAME_HZ


@dataclass(frozen=True)
class Qwen3TtsSpec:
    # code predictor (talker.code_predictor.*)
    cp_h: int
    cp_layers: int
    cp_heads: int
    cp_kv: int
    cp_hd: int
    cp_inter: int
    cp_vocab: int
    n_groups: int                # num_code_groups (talker predicts group 0; predictor does the rest)
    talker_h: int
    talker_vocab: int
    # speech tokenizer decoder (fp32)
    tok_tensors: tuple           # ((name, shape), …) decoder.* only
    tok_h: int = 512
    tok_layers: int = 8
    tok_heads: int = 16
    tok_hd: int = 64
    tok_inter: int = 1024
    tok_window: int = 72
    sample_rate: int = 24000
    upsample: int = 1920         # samples per frame
    # speaker encoder (Base; 0 if CustomVoice / VoiceDesign)
    spk_params: int = 0
    text_proj_params: int = 0   # talker.text_projection (prompt); role other
    text_h: int = 2048           # text_embedding / text_projection input
    kind: str = "qwen3_tts"

    @property
    def cp_steps(self) -> int:
        return max(0, self.n_groups - 1)

    @property
    def frame_hz(self) -> float:
        return FRAME_HZ

    @property
    def frame_s(self) -> float:
        return FRAME_S

    @property
    def samples_per_frame(self) -> int:
        return self.upsample

    def _layer_params(self, h: int, heads: int, kv: int, hd: int, inter: int) -> int:
        # Qwen3: q/k/v/o + q_norm/k_norm + gated MLP + 2 RMSNorm (no attn bias) 「来源」
        return (heads * hd * h + 2 * kv * hd * h + heads * hd * h + heads * hd + kv * hd
                + 3 * h * inter + 2 * h)

    @property
    def code_predictor_params(self) -> int:
        # layers + final RMSNorm + (n_groups-1) embed tables (width = talker_h 「来源」) +
        # (n_groups-1) lm_heads (width = cp_h) + optional small_to_mtp_projection
        layer = self._layer_params(self.cp_h, self.cp_heads, self.cp_kv, self.cp_hd, self.cp_inter)
        emb = self.cp_steps * self.cp_vocab * self.talker_h
        heads = self.cp_steps * self.cp_vocab * self.cp_h
        proj = (self.talker_h * self.cp_h + self.cp_h) if self.talker_h != self.cp_h else 0
        return self.cp_layers * layer + self.cp_h + emb + heads + proj

    @property
    def tokenizer_dec_params(self) -> int:
        return sum(math.prod(s) for n, s in self.tok_tensors if n.startswith("decoder."))

    @property
    def stored_bytes(self) -> float:
        # predictor bf16 + tokenizer decoder fp32 + speaker bf16
        return (self.code_predictor_params * 2.0 + self.tokenizer_dec_params * 4.0
                + self.spk_params * 2.0)


def qwen3_tts_spec(cfg: dict, rel: dict | None = None) -> Qwen3TtsSpec | None:
    if cfg.get("model_type") != "qwen3_tts":
        return None
    tc = cfg.get("talker_config") or {}
    if not tc.get("hidden_size"):
        return None
    pc = tc.get("code_predictor_config") or {}
    if not pc.get("hidden_size"):
        return None
    st = (rel or {}).get("speech_tensors") or {}
    tok = tuple((n, tuple(v) if not isinstance(v, tuple) else v)
                for n, v in st.items() if n.startswith("decoder."))
    # prefer decoder_config dims when present (shared tokenizer)
    dc = ((rel or {}).get("tokenizer_config") or {}).get("decoder_config") or {}
    sc = cfg.get("speaker_encoder_config")
    spk = int((rel or {}).get("params_spk") or 0)
    if sc and not spk:
        # fallback: count from release roles if summarize put them under spk
        spk = 0
    return Qwen3TtsSpec(
        cp_h=pc["hidden_size"], cp_layers=pc["num_hidden_layers"],
        cp_heads=pc["num_attention_heads"], cp_kv=pc["num_key_value_heads"],
        cp_hd=pc.get("head_dim") or (pc["hidden_size"] // pc["num_attention_heads"]),
        cp_inter=pc["intermediate_size"], cp_vocab=pc["vocab_size"],
        n_groups=tc.get("num_code_groups") or pc.get("num_code_groups") or 16,
        talker_h=tc["hidden_size"], talker_vocab=tc["vocab_size"],
        tok_tensors=tok,
        tok_h=dc.get("hidden_size", 512), tok_layers=dc.get("num_hidden_layers", 8),
        tok_heads=dc.get("num_attention_heads", 16),
        tok_hd=dc.get("head_dim") or 64,
        tok_inter=dc.get("intermediate_size", 1024),
        tok_window=dc.get("sliding_window") or 72,
        sample_rate=((rel or {}).get("tokenizer_config") or {}).get("output_sample_rate") or 24000,
        upsample=((rel or {}).get("tokenizer_config") or {}).get("decode_upsample_rate") or 1920,
        spk_params=spk,
        text_proj_params=int((rel or {}).get("params_other") or
            sum(v.get("params", 0) for v in ((rel or {}).get("roles") or {}).get("other", {}).values())),
        text_h=tc.get("text_hidden_size") or tc["hidden_size"],
    )


def _g(name: str, m: int, k: int, n: int, bits: float, *, w: int | None = None, cnt: int = 1,
       fmt: str | None = None) -> Op:
    ab = bits / 8
    fmt = fmt or ("fp32" if bits == 32 else "bf16")
    return Op(name, "gemm", 0, m=m, k=k, n=n, count=cnt, role="speech",
              w_params=(k * n if w is None else w) * cnt, w_bits=bits, w_fmt=fmt, a_fmt=fmt,
              act_bytes=m * (k + n) * ab * cnt, vec=m * n * VEC_PER_OUT * cnt, stream=True)


def _attn(name: str, q: int, keys: int, hd: int, heads: int, bits: float, b: int) -> list[Op]:
    ab = bits / 8
    c = b * heads
    af = "fp32" if bits == 32 else "bf16"
    return [Op(name + ".qk", "attn", 0, m=q, k=hd, n=keys, count=c, a_fmt=af,
               act_bytes=c * (q + keys) * hd * ab, stream=True),
            Op(name + ".pv", "attn", 0, m=q, k=keys, n=hd, count=c, a_fmt=af,
               act_bytes=c * (q + keys) * hd * ab),
            Op(name + ".softmax", "vector", 0, vec=c * q * keys * 5.0)]


def code_predictor_ops(s: Qwen3TtsSpec, b: int) -> list[Op]:
    """One frame after talker drew group-0.

    HF ``code_predictor.generate(inputs_embeds=cat(past_hidden, last_id_hidden),
    max_new_tokens=num_code_groups-1)`` 「来源」: one **prefill** over the 2-row seed, then
    ``cp_steps`` **decode** steps (KV cache), each emitting one residual codebook id.
    Embeddings are talker_h wide; layers are cp_h — Identity or Linear(talker_h → cp_h).
    """
    h, hd, inter = s.cp_h, s.cp_hd, s.cp_inter
    kv_d = s.cp_kv * hd
    ops: list[Op] = []
    if s.talker_h != h:
        ops.append(_g("cp.small_to_mtp", b * 2, s.talker_h, h, 16, w=s.talker_h * h + h))

    def layer_block(tag: str, rows: int, keys: int) -> list[Op]:
        out: list[Op] = []
        for li in range(s.cp_layers):
            p = f"{tag}.{li}"
            out += [_g(f"{p}.q", b * rows, h, s.cp_heads * hd, 16),
                    _g(f"{p}.k", b * rows, h, kv_d, 16),
                    _g(f"{p}.v", b * rows, h, kv_d, 16),
                    _g(f"{p}.o", b * rows, s.cp_heads * hd, h, 16),
                    _g(f"{p}.gate", b * rows, h, inter, 16),
                    _g(f"{p}.up", b * rows, h, inter, 16),
                    _g(f"{p}.down", b * rows, inter, h, 16),
                    Op(f"{p}.norm", "vector", li, vec=b * rows * h * 10)]
            out += _attn(f"{p}.attn", rows, keys, hd, s.cp_heads, 16, b)
        return out

    # seed embeds: talker past_hidden + group-0 codec embed (2 rows) — looked up / carried in
    ops.append(Op("cp.seed_emb", "lookup", 0, kv_read=b * 2 * s.talker_h * 2.0, vec=b * 2 * s.talker_h))
    # prefill over 2 seed positions
    ops += layer_block("cp.prefill", 2, 2)
    ops.append(_g("cp.prefill.head", b, h, s.cp_vocab, 16))  # first of the cp_steps logits from prefill last row
    ops.append(Op("cp.prefill.sample", "vector", 0, vec=b * s.cp_vocab * 3, sfu=b * s.cp_vocab))
    # remaining cp_steps-1 decode steps (KV grows: after prefill cache=2, then 3..cp_steps+1)
    for k in range(1, s.cp_steps):
        ops.append(Op(f"cp.{k}.emb", "lookup", 0, kv_read=b * s.talker_h * 2.0, vec=b * s.talker_h))
        keys = 2 + k  # seed 2 + k prior generated codes
        ops += layer_block(f"cp.{k}", 1, keys)
        ops.append(_g(f"cp.{k}.head", b, h, s.cp_vocab, 16))
        ops.append(Op(f"cp.{k}.sample", "vector", 0, vec=b * s.cp_vocab * 3, sfu=b * s.cp_vocab))
    return ops



def tokenizer_decode_ops(s: Qwen3TtsSpec, b: int, frames: int = 1) -> list[Op]:
    """Decode ``frames`` of 16-group codes → waveform.  Prefers released decoder.* GEMMs; falls back to
    config-shaped transformer + upsample placeholder when headers are absent."""
    ops: list[Op] = []
    if s.tok_tensors:
        # one GEMM per weight at frame rate (Linear [out,in]; Conv / ConvTranspose approximated as GEMM over
        # positions = frames or samples/frame) 「近似」 for non-transformer weights
        for n, sh in s.tok_tensors:
            if not n.endswith(".weight") or len(sh) < 2:
                continue
            if "codebook" in n:
                ops.append(Op(n, "lookup", 0, kv_read=b * frames * sh[-1] * 4.0))
                continue
            if len(sh) == 2:
                # transformer / linear: rows = frames (sliding window decode) 「近似」
                ops.append(_g(n, b * frames, sh[1], sh[0], 32, w=math.prod(sh)))
            elif len(sh) == 3:
                # Conv1d [out, in/g, k]: positions ≈ frames or upsample 「近似」
                pos = frames if "upsample" not in n else max(1, frames * s.upsample // 64)
                ops.append(_g(n, b * pos, sh[1] * sh[2], sh[0], 32, w=math.prod(sh)))
            elif len(sh) == 4:
                pos = max(1, frames * s.upsample // 64)
                ops.append(_g(n, b * pos, math.prod(sh[1:]), sh[0], 32, w=math.prod(sh)))
        return ops
    # config-only fallback: 8-layer decoder transformer over ``frames`` + ISTFT-ish upsample
    h, hd, inter = s.tok_h, s.tok_hd, s.tok_inter
    kv_d = s.tok_heads * hd   # decoder config: num_key_value_heads == num_attention_heads
    for li in range(s.tok_layers):
        p = f"tok.{li}"
        ops += [_g(f"{p}.qkv", b * frames, h, h + 2 * kv_d, 32),
                _g(f"{p}.o", b * frames, h, h, 32),
                _g(f"{p}.gate_up", b * frames, h, 2 * inter, 32),
                _g(f"{p}.down", b * frames, inter, h, 32)]
        ops += _attn(f"{p}.attn", frames, min(frames, s.tok_window), hd, s.tok_heads, 32, b)
    ops.append(Op("tok.upsample", "vector", 0, vec=b * frames * s.upsample * 8))
    return ops


def speaker_encoder_ops(s: Qwen3TtsSpec, b: int = 1) -> list[Op]:
    """One-shot clone prompt (Base).  Weight traffic only when params known 「近似」."""
    if not s.spk_params:
        return []
    return [Op("spk.enc", "gemm", 0, m=b, k=s.spk_params, n=1, role="speech",
               w_params=s.spk_params, w_bits=16.0, w_fmt="bf16", a_fmt="bf16",
               act_bytes=b * s.spk_params * 2.0, stream=True)]


def code_predictor_weight_bytes(s: Qwen3TtsSpec) -> float:
    return s.code_predictor_params * 2.0


def tokenizer_weight_bytes(s: Qwen3TtsSpec) -> float:
    return s.tokenizer_dec_params * 4.0
