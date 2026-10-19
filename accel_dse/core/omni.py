"""0.271 Qwen2.5-Omni speech output (docs/MODEL.md §216): Talker + token2wav (DiT flow matching + BigVGAN), streamed.

Reference code 「来源」: transformers models/qwen2_5_omni (modeling_qwen2_5_omni.py, configuration_qwen2_5_omni.py) and the
vLLM-Omni streaming wrapper vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_token2wav.py
(Qwen2_5OmniToken2WavModel.process_little_chunk / _process_chunk_for_50hz, "aligned with Qwen2Code2wav").

Pipeline per request (the Thinker = the catalog's LLM; this module costs what comes after it):
  1. Talker (``talker.*``, bf16): a 24-layer Qwen2-style decoder, h 896, 12 heads × 128 (q 1536), GQA kv 4, SwiGLU 18944,
     codec vocab 8448.  Input per position = codec embedding (8448 × 3584 table) + Thinker embedding / hidden state,
     projected by ``thinker_to_talker_proj`` (3584 → 896).  It prefills over the Thinker's prompt positions, then emits
     one codec token per step — 50 codec tokens per audio second (DiT ``repeats`` 2 → 100 mel frames / s; BigVGAN
     hop Π upsample_rates = 240 samples at 24 kHz 「来源」 README sf.write(…, samplerate=24000)).
  2. token2wav DiT (fp32 — the reference forces fp32 and falls back from flash attention): 22 blocks, dim 1024,
     16 heads × 64, AdaLayerNormZero (Linear d → 6d on the time embedding), GELU-tanh MLP ff_mult 2; block attention
     (block 24 frames; layer 10 also sees the next block, layers 0 / 20 the previous one; masked SDPA = dense scores);
     input = [noisy mel 80, ECAPA-TDNN speaker vector 128 from the reference mel, codec embedding 512 (repeated ×2),
     speaker embedding 192] → Linear 912 → 1024; norm_out (Linear d → 2d) + proj_out d → 80.  Sampler: RK4 over
     ``num_steps`` = 10 time points (sway −1) → 9 intervals × 4 = 36 DiT evaluations, each with CFG (cond + uncond rows
     in one batch, guidance 0.5).  The ECAPA encoder re-runs on the reference mel in every evaluation (input_embed).
  3. BigVGAN (fp32): conv_pre k7 (80 → 1536), 6 × [ConvTranspose1d (rates 5,3,2,2,2,2; kernels 11,7,4,4,4,4; channels
     halve) + 3 AMPBlocks (kernels 3 / 7 / 11, dilations 1,3,5; two convs per dilation; anti-aliased SnakeBeta before
     each conv)], SnakeBeta, conv_post k7 (no bias).
Streaming (vLLM-Omni): bs_mel 24, chunk = 3 · 24 = 72 mel frames (0.72 s), past cache 48, future cache 24 → DiT window
of chunk i = mel [72i − 48, 72(i+1) + 24) (first chunk 96 frames, steady 144); BigVGAN on the new 72 frames + 40 frames
of the previous mel (future_size 20 · 2); chunk 0 emits (72 − 20) · 240 samples; token2wav batch 1 only
("Only batch size = 1 is currently supported") → streams run their chunks one after another.
Every param formula below reproduces the released safetensors header exactly (talker 1,351,360,256; DiT 333,607,792;
BigVGAN 115,443,504).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .ir import Op

VEC_PER_OUT = 3.0
SNAKE_VEC = 56.0            # anti-aliased SnakeBeta per element: 2× FIR upsample (12 taps), sin² · 1/β, FIR down 「近似」


@dataclass(frozen=True)
class OmniSpec:
    # Talker
    t_h: int
    t_layers: int
    t_heads: int
    t_hd: int
    t_kv: int
    t_inter: int
    t_vocab: int
    t_emb: int
    # DiT
    d: int
    depth: int
    heads: int
    ff: int
    mel: int
    repeats: int
    codec_dim: int
    codec_n: int
    enc_dim: int
    enc_emb: int
    block: int
    enc_ch: tuple[int, ...]
    enc_k: tuple[int, ...]
    enc_dil: tuple[int, ...]
    enc_att: int
    enc_scale: int
    enc_se: int
    # BigVGAN
    v_ch: int
    v_rates: tuple[int, ...]
    v_kernels: tuple[int, ...]
    v_rb_k: tuple[int, ...]
    v_rb_dil: tuple[tuple[int, ...], ...]
    v_post_bias: bool
    rate: int = 24000           # README 「来源」
    steps: int = 10             # sample / fast_block_sample default time points (RK4 → 4 · (steps − 1) evaluations)
    guidance: float = 0.5
    bs_mel: int = 24            # vLLM-Omni streaming (factor 2)
    batched_chunk: int = 3
    future_size: int = 20
    ref_frames: int = 300       # reference mel frames of the speaker prompt (3 s) 「假设」

    # ---- rates / streaming geometry
    @property
    def hop(self) -> int:
        n = 1
        for r in self.v_rates:
            n *= r
        return n

    @property
    def mel_per_s(self) -> float:
        return self.rate / self.hop

    @property
    def codec_per_s(self) -> float:
        return self.mel_per_s / self.repeats

    @property
    def chunk(self) -> int:
        return self.bs_mel * self.batched_chunk

    @property
    def past(self) -> int:
        return self.bs_mel * self.repeats

    @property
    def future(self) -> int:
        return self.bs_mel

    @property
    def evals(self) -> int:
        return 4 * (self.steps - 1)

    @property
    def cfg_mult(self) -> int:
        return 2 if self.guidance >= 1e-5 else 1

    def dit_window(self, i: int) -> int:
        return (i + 1) * self.chunk + self.future - max(i * self.chunk - self.past, 0)

    def voc_frames(self, i: int) -> int:
        return self.chunk if i == 0 else self.chunk + 2 * self.future_size

    @property
    def first_codec(self) -> int:
        """Codec tokens the first DiT window needs (mel [0, chunk + future) / repeats)."""
        return (self.chunk + self.future) // self.repeats

    @property
    def first_audio_s(self) -> float:
        return (self.chunk - self.future_size) * self.hop / self.rate

    @property
    def chunk_s(self) -> float:
        return self.chunk / self.mel_per_s

    # ---- params (released header, exact)
    @property
    def talker_layer(self) -> int:
        h, q, kv = self.t_h, self.t_heads * self.t_hd, self.t_kv * self.t_hd
        return (h * q + q) + 2 * (h * kv + kv) + q * h + 3 * h * self.t_inter + 2 * h

    @property
    def talker_params(self) -> int:
        h = self.t_h
        return (self.t_layers * self.talker_layer + self.t_vocab * self.t_emb + h + (self.t_emb * h + h)
                + h * self.t_vocab)

    @property
    def talker_run_params(self) -> int:
        """Weights read per talker step (the codec embedding table is looked up: one row per token)."""
        return self.talker_params - self.t_vocab * self.t_emb

    @property
    def ecapa_params(self) -> int:
        c, k, dl = self.enc_ch, self.enc_k, self.enc_dil
        p = self.mel * c[0] * k[0] + c[0]
        for i in range(1, len(c) - 1):
            ci, co, s = c[i - 1], c[i], self.enc_scale
            w = co // s
            p += (ci * co + co) + (s - 1) * (w * w * k[i] + w) + (co * co + co)
            p += (co * self.enc_se + self.enc_se) + (self.enc_se * co + co)
        cm = sum(c[1:-1])
        p += cm * c[-1] * k[-1] + c[-1]
        p += (3 * c[-1] * self.enc_att + self.enc_att) + (self.enc_att * c[-1] + c[-1])
        p += 2 * c[-1] * self.enc_dim + self.enc_dim
        return p

    @property
    def dit_block(self) -> int:
        d = self.d
        return (6 * d * d + 6 * d) + 4 * (d * d + d) + (d * self.ff + self.ff) + (self.ff * d + d)

    @property
    def dit_params(self) -> int:
        d = self.d
        din = self.mel + self.enc_dim + self.codec_dim + self.enc_emb
        return (self.depth * self.dit_block + (256 * d + d) + (d * d + d) + (self.codec_n + 1) * self.codec_dim
                + (din * d + d) + (2 * d * d + 2 * d) + (d * self.mel + self.mel) + (d // self.heads) // 2
                + self.ecapa_params)

    def _voc_stages(self) -> list[tuple[int, int, int, int]]:
        """(C_in, C_out, kernel, rate) per upsample stage."""
        out, c = [], self.v_ch
        for r, k in zip(self.v_rates, self.v_kernels):
            out.append((c, c // 2, k, r))
            c //= 2
        return out

    @property
    def bigvgan_params(self) -> int:
        p = self.mel * self.v_ch * 7 + self.v_ch
        for ci, co, k, _ in self._voc_stages():
            p += ci * co * k + co
            for rk, dils in zip(self.v_rb_k, self.v_rb_dil):
                p += 2 * len(dils) * (co * co * rk + co) + 2 * len(dils) * 2 * co
        c = self._voc_stages()[-1][1]
        return p + 2 * c + c * 7 + (1 if self.v_post_bias else 0)

    @property
    def t2w_params(self) -> int:
        return self.dit_params + self.bigvgan_params

    @property
    def dit_run_params(self) -> int:
        """DiT weights read per evaluation (the codec table is looked up)."""
        return self.dit_params - (self.codec_n + 1) * self.codec_dim

    @property
    def stored_bytes(self) -> float:
        return self.talker_params * 2.0 + self.t2w_params * 4.0

    def talker_kv_bytes(self, ctx: int) -> float:
        return self.t_layers * ctx * 2 * self.t_kv * self.t_hd * 2.0


def omni_spec(cfg: dict) -> OmniSpec | None:
    if cfg.get("model_type") != "qwen2_5_omni" or not cfg.get("talker_config") or not cfg.get("token2wav_config"):
        return None
    tc, w = cfg["talker_config"], cfg["token2wav_config"]
    dc, bc = w["dit_config"], w["bigvgan_config"]
    # DiT block geometry: configuration_qwen2_5_omni.py defaults (block_size 24, look_ahead [10], look_backward [0, 20])
    return OmniSpec(
        t_h=tc["hidden_size"], t_layers=tc["num_hidden_layers"], t_heads=tc["num_attention_heads"],
        t_hd=tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"], t_kv=tc["num_key_value_heads"],
        t_inter=tc["intermediate_size"], t_vocab=tc["vocab_size"], t_emb=tc.get("embedding_size") or tc["hidden_size"],
        d=dc["dim"], depth=dc["depth"], heads=dc["heads"], ff=dc["dim"] * dc["ff_mult"], mel=dc["mel_dim"],
        repeats=dc["repeats"], codec_dim=dc["emb_dim"], codec_n=dc["num_embeds"], enc_dim=dc["enc_dim"],
        enc_emb=dc["enc_emb_dim"], block=dc.get("block_size", 24), enc_ch=tuple(dc["enc_channels"]),
        enc_k=tuple(dc["enc_kernel_sizes"]), enc_dil=tuple(dc["enc_dilations"]), enc_att=dc["enc_attention_channels"],
        enc_scale=dc["enc_res2net_scale"], enc_se=dc["enc_se_channels"],
        v_ch=bc["upsample_initial_channel"], v_rates=tuple(bc["upsample_rates"]),
        v_kernels=tuple(bc["upsample_kernel_sizes"]), v_rb_k=tuple(bc["resblock_kernel_sizes"]),
        v_rb_dil=tuple(tuple(x) for x in bc["resblock_dilation_sizes"]), v_post_bias=bool(bc.get("use_bias_at_final", True)))


# ---------------------------------------------------------------- ops
def _gemm(name: str, m: int, k: int, n: int, fmt: str, w: int | None = None, bias: int | None = None,
          role: str = "omni") -> Op:
    """GEMM whose weight read = the matrix (``w``, default k·n) + its bias (``bias``, default n; 0 = none)."""
    bits = 32.0 if fmt == "fp32" else 16.0
    ab = bits / 8
    wp = (k * n if w is None else w) + (n if bias is None else bias)
    return Op(name, "gemm", 0, m=m, k=k, n=n, role=role, w_params=wp, w_bits=bits, w_fmt=fmt,
              a_fmt=fmt, act_bytes=m * (k + n) * ab, vec=m * n * VEC_PER_OUT, stream=True)


def talker_ops(o: OmniSpec, b: int, q: int, ctx: int) -> list[Op]:
    """One Talker pass for ``b`` streams: ``q`` new positions each (prefill: the prompt; decode: 1) over ``ctx``
    cached positions (bf16 K/V)."""
    h, H, hd, kvd = o.t_h, o.t_heads, o.t_hd, o.t_kv * o.t_hd
    rows = b * q
    keys = ctx + q
    ops = [Op("talker.codec_embed", "lookup", 0, kv_read=rows * o.t_emb * 2.0, vec=rows * o.t_emb),
           _gemm("talker.thinker_proj", rows, o.t_emb, h, "bf16")]          # Linear with bias
    for li in range(o.t_layers):
        p = f"talker.{li}."
        ops += [_gemm(p + "qkv", rows, h, H * hd + 2 * kvd, "bf16"), _gemm(p + "o", rows, H * hd, h, "bf16", bias=0),
                _gemm(p + "gate_up", rows, h, 2 * o.t_inter, "bf16", bias=0),
                _gemm(p + "down", rows, o.t_inter, h, "bf16", bias=0),
                Op(p + "norm", "vector", li, vec=rows * h * 8),
                Op(p + "qk", "attn", li, m=q, k=hd, n=keys, count=b * H, causal=(0.5 if q > 1 and not ctx else 1.0),
                   kv_read=b * ctx * 2 * kvd * 2.0, kv_write=rows * 2 * kvd * 2.0, act_bytes=b * H * (q + keys) * hd * 2.0),
                Op(p + "pv", "attn", li, m=q, k=keys, n=hd, count=b * H, causal=(0.5 if q > 1 and not ctx else 1.0),
                   act_bytes=b * H * (q + keys) * hd * 2.0),
                Op(p + "softmax", "vector", li, vec=b * H * q * keys * 5.0)]
    ops += [Op("talker.final_norm", "vector", 0, vec=b * h * 8),
            _gemm("talker.codec_head", b, h, o.t_vocab, "bf16", bias=0),
            Op("talker.sample", "vector", 0, vec=b * o.t_vocab * 8)]
    return ops


def _ecapa_ops(o: OmniSpec, rows_b: int, F: int) -> list[Op]:
    """ECAPA-TDNN speaker encoder on the reference mel (``rows_b`` = CFG rows, F frames each)."""
    c, k = o.enc_ch, o.enc_k
    R = rows_b * F
    w = lambda x: x
    ops = [_gemm("t2w.dit.ecapa.tdnn0", R, o.mel * k[0], c[0], "fp32", w(o.mel * k[0] * c[0]))]
    for i in range(1, len(c) - 1):
        ci, co, s = c[i - 1], c[i], o.enc_scale
        wd = co // s
        ops += [_gemm(f"t2w.dit.ecapa.{i}.tdnn1", R, ci, co, "fp32", w(ci * co)),
                _gemm(f"t2w.dit.ecapa.{i}.res2", R * (s - 1), wd * k[i], wd, "fp32", w((s - 1) * wd * wd * k[i]),
                      w((s - 1) * wd)),
                _gemm(f"t2w.dit.ecapa.{i}.tdnn2", R, co, co, "fp32", w(co * co)),
                _gemm(f"t2w.dit.ecapa.{i}.se1", rows_b, co, o.enc_se, "fp32", w(co * o.enc_se)),
                _gemm(f"t2w.dit.ecapa.{i}.se2", rows_b, o.enc_se, co, "fp32", w(o.enc_se * co))]
    cm = sum(c[1:-1])
    ops += [_gemm("t2w.dit.ecapa.mfa", R, cm * k[-1], c[-1], "fp32", w(cm * k[-1] * c[-1])),
            _gemm("t2w.dit.ecapa.asp", R, 3 * c[-1], o.enc_att, "fp32", w(3 * c[-1] * o.enc_att)),
            _gemm("t2w.dit.ecapa.asp_out", R, o.enc_att, c[-1], "fp32", w(o.enc_att * c[-1])),
            _gemm("t2w.dit.ecapa.fc", rows_b, 2 * c[-1], o.enc_dim, "fp32", w(2 * c[-1] * o.enc_dim))]
    return ops


def dit_eval_ops(o: OmniSpec, T: int, wscale: float = 1.0) -> list[Op]:
    """One DiT evaluation on a T-frame window (batch 1; CFG doubles the rows).  ``wscale`` < 1: share of the weight
    bytes that still come from DRAM on a re-read evaluation (0.269 re-read pin, see omni_pin)."""
    B = o.cfg_mult
    R, d, hd = B * T, o.d, o.d // o.heads
    w = lambda x: x
    din = o.mel + o.enc_dim + o.codec_dim + o.enc_emb
    ops = [_gemm("t2w.dit.time0", B, 256, d, "fp32", w(256 * d)), _gemm("t2w.dit.time1", B, d, d, "fp32", w(d * d)),
           Op("t2w.dit.codec_embed", "lookup", 0, kv_read=B * (T // o.repeats) * o.codec_dim * 4.0, vec=R * o.codec_dim)]
    ops += _ecapa_ops(o, B, o.ref_frames)
    ops.append(_gemm("t2w.dit.input_proj", R, din, d, "fp32", w(din * d)))
    for li in range(o.depth):
        p = f"t2w.dit.{li}."
        ops += [_gemm(p + "ada", B, d, 6 * d, "fp32", w(6 * d * d)),
                _gemm(p + "qkv", R, d, 3 * d, "fp32", w(3 * d * d)), _gemm(p + "o", R, d, d, "fp32", w(d * d)),
                _gemm(p + "ff1", R, d, o.ff, "fp32", w(d * o.ff)), _gemm(p + "ff2", R, o.ff, d, "fp32", w(o.ff * d)),
                Op(p + "norm_mod", "vector", li, vec=R * d * 12),
                # masked SDPA over the whole window (block_diff mask): dense T × T scores 「来源」
                Op(p + "qk", "attn", li, m=T, k=hd, n=T, count=B * o.heads, a_fmt="fp32",
                   act_bytes=B * o.heads * 2 * T * hd * 4.0, stream=True),
                Op(p + "pv", "attn", li, m=T, k=T, n=hd, count=B * o.heads, a_fmt="fp32",
                   act_bytes=B * o.heads * 2 * T * hd * 4.0),
                Op(p + "softmax", "vector", li, vec=B * o.heads * T * T * 6.0)]
    ops += [_gemm("t2w.dit.norm_out", B, d, 2 * d, "fp32", w(2 * d * d)),
            _gemm("t2w.dit.proj_out", R, d, o.mel, "fp32", w(d * o.mel)),
            Op("t2w.dit.cfg_rk4", "vector", 0, vec=T * o.mel * 8)]
    if wscale != 1.0:
        ops = [replace(x, w_params=int(round(x.w_params * wscale))) if x.kind == "gemm" else x for x in ops]
    return ops


def bigvgan_ops(o: OmniSpec, M: int) -> list[Op]:
    """BigVGAN on M mel frames (batch 1): convolutions as implicit GEMMs; transposed convs as M_in × C_in × (C_out·k)."""
    ops = [_gemm("t2w.voc.conv_pre", M, o.mel * 7, o.v_ch, "fp32")]
    L = M
    for si, (ci, co, k, r) in enumerate(o._voc_stages()):
        ops.append(_gemm(f"t2w.voc.{si}.up", L, ci, co * k, "fp32", bias=co))
        L *= r
        for rk, dils in zip(o.v_rb_k, o.v_rb_dil):
            for j in range(2 * len(dils)):
                ops += [Op(f"t2w.voc.{si}.k{rk}.{j}.snake", "vector", si, vec=L * co * SNAKE_VEC, sfu=L * co * 2),
                        _gemm(f"t2w.voc.{si}.k{rk}.{j}.conv", L, co * rk, co, "fp32")]
        ops.append(Op(f"t2w.voc.{si}.avg", "vector", si, vec=L * co * len(o.v_rb_k)))
    c = o._voc_stages()[-1][1]
    ops += [Op("t2w.voc.act_post", "vector", 0, vec=L * c * SNAKE_VEC, sfu=L * c * 2),
            _gemm("t2w.voc.conv_post", L, c * 7, 1, "fp32", bias=1 if o.v_post_bias else 0)]
    return ops


def omni_pin(o: OmniSpec, sram_bytes: float, max_act: float = 0.0) -> float:
    """0.269 re-read rule on the DiT: its run weights are read ``evals`` (36) times per chunk (R B / B saved > the 2 B / B
    of recurrent state) → pinned first into the SRAM left after staging (memplan.plan: staging = max(2 MiB, 2 · max
    activation)).  Returns the DRAM miss share of a re-read evaluation's weight bytes."""
    W = o.dit_run_params * 4.0
    free = max(0.0, sram_bytes - max(2 * 2 ** 20, 2 * max_act))
    return 1.0 - min(W, free) / W


def chunk_flops(o: OmniSpec, T: int, M: int) -> dict:
    f = lambda ops: sum(x.flops for x in ops if x.kind in ("gemm", "attn"))
    return {"dit_eval": f(dit_eval_ops(o, T)), "dit_chunk": o.evals * f(dit_eval_ops(o, T)), "voc": f(bigvgan_ops(o, M))}
