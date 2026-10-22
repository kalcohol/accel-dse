"""0.345/0.346 Qwen3-Omni (dual MoE + code2wav; docs/MODEL.md §290 / §291).

Reference 「来源」: HF Qwen/Qwen3-Omni-30B-A3B-Instruct config.json + safetensors headers;
transformers modeling_qwen3_omni_moe.py / configuration_qwen3_omni_moe.py;
vllm-omni code2wav (Qwen3OmniMoeCode2Wav).

Pipeline (Thinker = catalog LLM with AuT + ViT storage; this module costs speech out):
  1. Talker MoE (``talker.*``, bf16): 20-layer GQA MoE, h 1024, 16 heads × 128, kv 2, 128 experts top-6,
     moe_inter 384, shared_expert_intermediate_size 768; codec vocab 3072.  One decode step per 12.5 Hz frame
     (codec_head draws group-0).
  2. Code predictor (``talker.code_predictor.*``, bf16): AR generate of num_code_groups−1 further codes;
     5-layer Qwen3 decoder (GQA + qk_norm, SwiGLU) — same shape as Qwen3-TTS predictor.
  3. Code2wav (``code2wav.*``, bf16 as released): embed+mean RVQ → sliding-window pre_transformer (8L h1024,
     window 72) → upsampling_ratios [2,2] CausalTransConv+ConvNeXt → decoder (CausalConv → DecoderBlocks
     upsample_rates [8,5,4,3] + residual SnakeBeta) → waveform.  total_upsample = 1920; output 24 kHz
     「来源」 transformers / Qwen3-TTS same hop → FRAME_HZ = 12.5.

Gated on ``model_type == qwen3_omni_moe`` only — does not touch omni_spec (qwen2_5_omni) or qwen3_tts_spec.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from .ir import Op, Phase, Shard, build_rank_ops

VEC_PER_OUT = 3.0
SNAKE_VEC = 56.0
FRAME_HZ = 12.5          # 24000 / 1920 「来源」 total_upsample; same hop as Qwen3-TTS
FRAME_S = 1.0 / FRAME_HZ
SAMPLE_RATE = 24000      # 「来源」 transformers Qwen3-Omni audio output / README


@dataclass(frozen=True)
class Qwen3OmniSpec:
    # Talker MoE (talker.text_config)
    t_h: int
    t_layers: int
    t_heads: int
    t_kv: int
    t_hd: int
    t_moe_inter: int
    t_experts: int
    t_topk: int
    t_shared_inter: int
    t_vocab: int
    # Code predictor
    cp_h: int
    cp_layers: int
    cp_heads: int
    cp_kv: int
    cp_hd: int
    cp_inter: int
    cp_vocab: int
    n_groups: int
    # Code2wav
    c2_h: int
    c2_layers: int
    c2_heads: int
    c2_kv: int
    c2_inter: int
    c2_window: int
    c2_decoder_dim: int
    c2_codebook: int
    c2_quantizers: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    sample_rate: int = SAMPLE_RATE
    # release param counts (exact when headers present)
    talker_params: int = 0       # talker.* excl. code_predictor
    cp_params: int = 0
    code2wav_params: int = 0
    # 0.346 Talker ResizeMLP (text_projection / hidden_projection) 「来源」 modeling TalkerResizeMLP
    thinker_hidden: int = 2048
    proj_inter: int = 2048       # talker text_config.intermediate_size (not moe_inter)
    kind: str = "qwen3_omni"

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
    def total_upsample(self) -> int:
        n = 1
        for r in self.upsample_rates + self.upsampling_ratios:
            n *= r
        return n

    @property
    def samples_per_frame(self) -> int:
        return self.total_upsample

    def _cp_layer_params(self) -> int:
        h, heads, kv, hd, inter = self.cp_h, self.cp_heads, self.cp_kv, self.cp_hd, self.cp_inter
        return (heads * hd * h + 2 * kv * hd * h + heads * hd * h + heads * hd + kv * hd
                + 3 * h * inter + 2 * h)

    @property
    def code_predictor_params(self) -> int:
        if self.cp_params:
            return self.cp_params
        layer = self._cp_layer_params()
        emb = self.cp_steps * self.cp_vocab * self.t_h
        heads = self.cp_steps * self.cp_vocab * self.cp_h
        return self.cp_layers * layer + self.cp_h + emb + heads

    @property
    def stored_bytes(self) -> float:
        # all bf16 as released 「来源」 safetensors headers
        tp = self.talker_params or self._talker_params_cfg()
        return (tp + self.code_predictor_params + (self.code2wav_params or self._code2wav_params_cfg())) * 2.0

    def _talker_params_cfg(self) -> int:
        # GQA + MoE + shared + norms + embed/head 「近似」 when headers absent
        h, H, kv, hd = self.t_h, self.t_heads, self.t_kv, self.t_hd
        attn = H * hd * h + 2 * kv * hd * h + H * hd * h + H * hd + kv * hd + 2 * h
        exp = 3 * h * self.t_moe_inter * self.t_experts
        shared = (3 * h * self.t_shared_inter + h) if self.t_shared_inter else 0  # + gate
        router = h * self.t_experts
        return self.t_layers * (attn + exp + shared + router) + self.t_vocab * h * 2

    def _code2wav_params_cfg(self) -> int:
        h, L, inter = self.c2_h, self.c2_layers, self.c2_inter
        # pre_transformer: MHA (kv=heads) + SwiGLU + 2 RMSNorm + 2 layer_scale
        layer = 4 * h * h + 3 * h * inter + 4 * h
        emb = self.c2_codebook * self.c2_quantizers * h
        # upsample: TransConv h→h k=factor + ConvNeXt (dw k7 + 2 Linear)
        up = 0
        for f in self.upsampling_ratios:
            up += h * h * f + h * 7 + h * 4 * h + 4 * h * h + h
        # decoder: conv h→dec_dim k7 + blocks + final
        dec = h * self.c2_decoder_dim * 7 + self.c2_decoder_dim
        for i, r in enumerate(self.upsample_rates):
            ind = self.c2_decoder_dim // 2 ** i
            outd = self.c2_decoder_dim // 2 ** (i + 1)
            dec += 2 * ind  # snake α β
            dec += ind * outd * (2 * r)  # TransConv
            for _ in (1, 3, 9):
                dec += outd * outd * 7 + outd * outd  # residual unit
                dec += 2 * outd  # snake in residual
        outd = self.c2_decoder_dim // 2 ** len(self.upsample_rates)
        dec += 2 * outd + outd * 7  # final snake + conv→1
        return L * layer + emb + up + dec

    def talker_kv_bytes(self, ctx: int) -> float:
        return ctx * self.t_layers * 2 * self.t_kv * self.t_hd * 2.0


def qwen3_omni_spec(cfg: dict, rel: dict | None = None) -> Qwen3OmniSpec | None:
    """Build from public HF config; gated on ``qwen3_omni_moe`` only."""
    if cfg.get("model_type") != "qwen3_omni_moe":
        return None
    if not cfg.get("enable_audio_output", True):
        return None
    talk = cfg.get("talker_config") or {}
    tc = talk.get("text_config") or {}
    if not tc.get("hidden_size") or not tc.get("num_experts"):
        return None
    pc = talk.get("code_predictor_config") or {}
    if not pc.get("hidden_size"):
        return None
    c2 = cfg.get("code2wav_config") or {}
    if not c2.get("hidden_size") or not c2.get("upsample_rates"):
        return None
    hd = tc.get("head_dim") or (tc["hidden_size"] // tc["num_attention_heads"])
    cphd = pc.get("head_dim") or (pc["hidden_size"] // pc["num_attention_heads"])
    rates = tuple(c2["upsample_rates"])
    ratios = tuple(c2.get("upsampling_ratios") or (2, 2))
    tp = int((rel or {}).get("params_talker") or 0)
    cpp = int((rel or {}).get("params_depth") or 0)  # code_predictor → depth role
    c2p = int((rel or {}).get("params_code2wav") or 0)
    return Qwen3OmniSpec(
        t_h=tc["hidden_size"], t_layers=tc["num_hidden_layers"],
        t_heads=tc["num_attention_heads"], t_kv=tc.get("num_key_value_heads") or tc["num_attention_heads"],
        t_hd=hd, t_moe_inter=tc["moe_intermediate_size"],
        t_experts=tc["num_experts"], t_topk=tc["num_experts_per_tok"],
        t_shared_inter=int(tc.get("shared_expert_intermediate_size") or 0),
        t_vocab=tc["vocab_size"],
        cp_h=pc["hidden_size"], cp_layers=pc["num_hidden_layers"],
        cp_heads=pc["num_attention_heads"], cp_kv=pc.get("num_key_value_heads") or pc["num_attention_heads"],
        cp_hd=cphd, cp_inter=pc["intermediate_size"], cp_vocab=pc.get("vocab_size") or 2048,
        n_groups=talk.get("num_code_groups") or pc.get("num_code_groups") or 16,
        c2_h=c2["hidden_size"], c2_layers=c2["num_hidden_layers"],
        c2_heads=c2["num_attention_heads"], c2_kv=c2.get("num_key_value_heads") or c2["num_attention_heads"],
        c2_inter=c2["intermediate_size"], c2_window=c2.get("sliding_window") or 72,
        c2_decoder_dim=c2["decoder_dim"], c2_codebook=c2.get("codebook_size") or 2048,
        c2_quantizers=c2["num_quantizers"],
        upsample_rates=rates, upsampling_ratios=ratios,
        talker_params=tp, cp_params=cpp, code2wav_params=c2p,
        thinker_hidden=int(talk.get("thinker_hidden_size") or 2048),
        proj_inter=int(tc.get("intermediate_size") or 2048),
    )


def _g(name: str, m: int, k: int, n: int, bits: float = 16.0, *, w: int | None = None, cnt: int = 1) -> Op:
    ab = bits / 8
    fmt = "fp32" if bits == 32 else "bf16"
    return Op(name, "gemm", 0, m=m, k=k, n=n, count=cnt, role="speech",
              w_params=(k * n if w is None else w) * cnt, w_bits=bits, w_fmt=fmt, a_fmt=fmt,
              act_bytes=m * (k + n) * ab * cnt, vec=m * n * VEC_PER_OUT * cnt, stream=True)


def _attn(name: str, q: int, keys: int, hd: int, heads: int, b: int) -> list[Op]:
    c = b * heads
    return [Op(name + ".qk", "attn", 0, m=q, k=hd, n=keys, count=c, a_fmt="bf16",
               act_bytes=c * (q + keys) * hd * 2.0, stream=True),
            Op(name + ".pv", "attn", 0, m=q, k=keys, n=hd, count=c, a_fmt="bf16",
               act_bytes=c * (q + keys) * hd * 2.0),
            Op(name + ".softmax", "vector", 0, vec=c * q * keys * 5.0)]


@lru_cache(maxsize=4)
def _talker_model(s: Qwen3OmniSpec):
    """Nested MoE ModelSpec for talker decode ops via build_rank_ops 「来源」 _build_std MoE path."""
    from .model import ModelSpec, RoleFormat, _build_std
    bf = RoleFormat("bf16", 16.0)
    c = {
        "model_type": "qwen3_omni_moe_text",
        "hidden_size": s.t_h, "num_hidden_layers": s.t_layers,
        "num_attention_heads": s.t_heads, "num_key_value_heads": s.t_kv, "head_dim": s.t_hd,
        "num_experts": s.t_experts, "num_experts_per_tok": s.t_topk,
        "moe_intermediate_size": s.t_moe_inter,
        "intermediate_size": s.t_moe_inter * 2,
        "vocab_size": s.t_vocab, "tie_word_embeddings": True,
    }
    if s.t_shared_inter:
        c["shared_expert_intermediate_size"] = s.t_shared_inter
    layers, _mtp, notes, cov = _build_std(c)
    return ModelSpec(
        id="qwen3-omni-talker", hf_id="", arch="GQA MoE", hidden=s.t_h, vocab=s.t_vocab,
        layers=tuple(layers), tie_embeddings=True, coverage=cov, notes=tuple(notes),
        formats=(("attn", bf), ("expert", bf), ("shared_expert", bf), ("router", bf),
                 ("mlp", bf), ("embed", bf)),
        act_fmt="bf16", kv_fmt="bf16",
    )


def talker_projection_ops(s: Qwen3OmniSpec, b: int, n_tok: int) -> list[Op]:
    """0.346 Thinker→Talker ResizeMLP once per setup (text_projection) 「来源」 Qwen3OmniMoeTalkerResizeMLP:
    Linear(thinker_h → proj_inter) + act + Linear(proj_inter → talker_h); bf16.  Called on user/assistant
    thinker embeds before Talker generate — not per codec frame."""
    n = max(1, n_tok)
    return [_g("talker.proj.fc1", b * n, s.thinker_hidden, s.proj_inter, 16),
            Op("talker.proj.act", "vector", 0, vec=b * n * s.proj_inter * VEC_PER_OUT),
            _g("talker.proj.fc2", b * n, s.proj_inter, s.t_h, 16)]


def talker_ops(s: Qwen3OmniSpec, b: int, q: int, ctx: int) -> list[Op]:
    """One Talker pass: ``q`` new positions over ``ctx`` cached (bf16 K/V); MoE via build_rank_ops."""
    m = _talker_model(s)
    kind = "decode" if q == 1 else "prefill"
    ph = Phase(kind=kind, batch=b, q=q, ctx=ctx)
    ops: list[Op] = [Op("talker.codec_embed", "lookup", 0, kv_read=b * q * s.t_h * 2.0, vec=b * q * s.t_h)]
    for li in range(m.n_layers):
        ops += build_rank_ops(m, li, ph, Shard())
    # codec_head tied to codec_embedding 「来源」 _tied_weights_keys — logits compute, w_params=0
    ops += [Op("talker.final_norm", "vector", 0, vec=b * s.t_h * 8),
            _g("talker.codec_head", b, s.t_h, s.t_vocab, 16, w=0),
            Op("talker.sample", "vector", 0, vec=b * s.t_vocab * 8)]
    return ops


def talker_weight_bytes(s: Qwen3OmniSpec) -> float:
    return (s.talker_params or s._talker_params_cfg()) * 2.0


def code_predictor_ops(s: Qwen3OmniSpec, b: int) -> list[Op]:
    """One frame after talker drew group-0 — same AR shape as Qwen3-TTS 「来源」."""
    h, hd, inter = s.cp_h, s.cp_hd, s.cp_inter
    kv_d = s.cp_kv * hd
    ops: list[Op] = []

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
            out += _attn(f"{p}.attn", rows, keys, hd, s.cp_heads, b)
        return out

    ops.append(Op("cp.seed_emb", "lookup", 0, kv_read=b * 2 * s.t_h * 2.0, vec=b * 2 * s.t_h))
    ops += layer_block("cp.prefill", 2, 2)
    ops.append(_g("cp.prefill.head", b, h, s.cp_vocab, 16))
    ops.append(Op("cp.prefill.sample", "vector", 0, vec=b * s.cp_vocab * 3, sfu=b * s.cp_vocab))
    for k in range(1, s.cp_steps):
        ops.append(Op(f"cp.{k}.emb", "lookup", 0, kv_read=b * s.t_h * 2.0, vec=b * s.t_h))
        keys = 2 + k
        ops += layer_block(f"cp.{k}", 1, keys)
        ops.append(_g(f"cp.{k}.head", b, h, s.cp_vocab, 16))
        ops.append(Op(f"cp.{k}.sample", "vector", 0, vec=b * s.cp_vocab * 3, sfu=b * s.cp_vocab))
    return ops


def code2wav_ops(s: Qwen3OmniSpec, b: int, frames: int = 1) -> list[Op]:
    """Decode ``frames`` of 16-group codes → waveform (bf16 as released)."""
    h, hd, inter = s.c2_h, s.c2_h // s.c2_heads, s.c2_inter
    ops: list[Op] = []
    # embed + mean over quantizers
    ops.append(Op("c2w.embed", "lookup", 0,
                  kv_read=b * frames * s.c2_quantizers * h * 2.0,
                  vec=b * frames * s.c2_quantizers * h))
    # 0.346: streaming one new frame with past_key_values attends up to sliding_window 「来源」
    # Code2WavAttention.sliding_window + chunked_decode left_context; was min(frames, window)=1 → understated.
    keys = s.c2_window if frames == 1 else min(frames, s.c2_window)
    for li in range(s.c2_layers):
        p = f"c2w.pre.{li}"
        ops += [_g(f"{p}.qkv", b * frames, h, h + 2 * h, 16),  # MHA kv=heads
                _g(f"{p}.o", b * frames, h, h, 16),
                _g(f"{p}.gate_up", b * frames, h, 2 * inter, 16),
                _g(f"{p}.down", b * frames, inter, h, 16),
                Op(f"{p}.norm", "vector", li, vec=b * frames * h * 12)]
        ops += _attn(f"{p}.attn", frames, keys, hd, s.c2_heads, b)
    # upsampling_ratios
    L = frames
    for i, f in enumerate(s.upsampling_ratios):
        ops.append(_g(f"c2w.up.{i}.trans", b * L, h, h * f, 16, w=h * h * f))
        L *= f
        ops += [_g(f"c2w.up.{i}.pw1", b * L, h, 4 * h, 16),
                _g(f"c2w.up.{i}.pw2", b * L, 4 * h, h, 16),
                Op(f"c2w.up.{i}.dw", "vector", 0, vec=b * L * h * 7),
                Op(f"c2w.up.{i}.snake", "vector", 0, vec=b * L * h * SNAKE_VEC)]
    # decoder pre-conv
    ops.append(_g("c2w.dec.pre", b * L, h * 7, s.c2_decoder_dim, 16, w=h * s.c2_decoder_dim * 7))
    ch = s.c2_decoder_dim
    for i, r in enumerate(s.upsample_rates):
        out_ch = s.c2_decoder_dim // 2 ** (i + 1)
        ops.append(Op(f"c2w.dec.{i}.snake0", "vector", 0, vec=b * L * ch * SNAKE_VEC))
        ops.append(_g(f"c2w.dec.{i}.trans", b * L, ch, out_ch * (2 * r), 16, w=ch * out_ch * (2 * r)))
        L *= r
        ch = out_ch
        for j, dil in enumerate((1, 3, 9)):
            ops += [Op(f"c2w.dec.{i}.r{j}.snake", "vector", 0, vec=b * L * ch * SNAKE_VEC * 2),
                    _g(f"c2w.dec.{i}.r{j}.c1", b * L, ch * 7, ch, 16, w=ch * ch * 7),
                    _g(f"c2w.dec.{i}.r{j}.c2", b * L, ch, ch, 16, w=ch * ch)]
    ops += [Op("c2w.dec.snake_out", "vector", 0, vec=b * L * ch * SNAKE_VEC),
            _g("c2w.dec.post", b * L, ch * 7, 1, 16, w=ch * 7)]
    return ops


def code_predictor_weight_bytes(s: Qwen3OmniSpec) -> float:
    return s.code_predictor_params * 2.0


def code2wav_weight_bytes(s: Qwen3OmniSpec) -> float:
    return (s.code2wav_params or s._code2wav_params_cfg()) * 2.0
