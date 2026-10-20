"""0.271 CosyVoice2-0.5B / Fun-CosyVoice3-0.5B-2512 speech output (docs/MODEL.md §216.3).

Reference code 「来源」 (github FunAudioLLM/CosyVoice main): cosyvoice/cli/model.py (CosyVoice2Model / CosyVoice3Model
.tts / .token2wav), flow/flow.py (CausalMaskedDiffWithXvec / CausalMaskedDiffWithDiT .inference), flow/flow_matching.py
(CausalConditionalCFM.solve_euler), flow/decoder.py (CausalConditionalDecoder), flow/DiT/dit.py, hifigan/generator.py
(HiFTGenerator / CausalHiFTGenerator); cosyvoice2.yaml / cosyvoice3.yaml.  Params: the released llm.pt / flow.pt /
hift.pt state dicts (scripts/fetch_pt_release.py; fp32).

  LM: Qwen2.5-0.5B backbone (the catalog model) emits 25 speech tokens per audio second (token_frame_rate), head
      llm_decoder (hidden → 6564 / 6761 speech tokens).
  flow (per streamed chunk, batch 1 — ``assert token.shape[0] == 1``): re-run from scratch over [prompt tokens +
      every token generated so far (+ 3 look-ahead)] — CosyVoice2: UpsampleConformerEncoder (6 blocks at 25 Hz,
      ×2 upsample conv, 4 blocks at 50 Hz, rel-pos attention) + U-Net CausalConditionalDecoder (256 ch, ResNet1D +
      transformer blocks, attention 8 × 64); CosyVoice3: pre-look-ahead conv + DiT (22 blocks, dim 1024, 16 × 64,
      the Qwen2.5-Omni block shape).  CFM: 10 Euler steps (cosine schedule), CFG in one batch of 2
      (inference_cfg_rate 0.7); the estimator runs on all mel frames (prompt + generated, 50 / s); the output keeps the
      new frames.
  HiFT vocoder: NSF source + ResBlocks at ×8, ×40, ×120 the mel rate, conv_post → 18 channels → iSTFT (n_fft 16,
      hop 4) → 24 kHz.  CosyVoice2 vocodes the new mel (+ an 8-frame mel cache); CosyVoice3 re-vocodes the whole
      generated mel each chunk (hift_cache_dict 'mel' grows; speech_offset slices the new samples).
  Streaming hops (cli/model.py): token_hop_len 25 (+ pad of the prompt to a multiple of 25 on the first chunk), then
      × stream_scale_factor 2 per chunk up to 4 · 25 = 100.  ``self.token_hop_len`` is an instance attribute that is
      never reset, so on a long-lived model object every request after the first starts at 100 tokens — reported as
      the warm first-audio latency.
Ops are built from the released tensor shapes: one GEMM per weight at its module's rate (Linear [out, in]; Conv1d
[out, in / groups, k] per output position; CosyVoice2 ConvTranspose [in, out, k] per input position), its bias /
weight-norm gain folded into the weight read; attention cores added per block.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from .ir import Op

VEC_PER_OUT = 3.0


@dataclass(frozen=True)
class CosySpec:
    version: int
    tensors: tuple                 # ((name, shape), …) flow.* / hift.* (released .pt state dicts)
    token_rate: int
    mel_ratio: int
    hop0: int
    lookahead: int
    rate: int
    ups: tuple[int, ...]
    istft_nfft: int
    istft_hop: int
    flow_heads: int
    flow_hd: int
    cfg_rate: float
    steps: int = 10                 # flow.inference n_timesteps
    scale: int = 2                  # stream_scale_factor
    max_hop_mult: int = 4           # token_max_hop_len = 4 · token_hop_len
    mel_cache: int = 8              # CosyVoice2 hift mel cache frames
    prompt_tokens: int = 75         # zero-shot prompt speech (3 s at 25 Hz) 「假设」
    # 0.273 Step-Audio-2-mini token2wav (stream "cache"): fixed 25-token chunks through flow.inference_chunk — the
    # conformer and the DiT estimator keep K/V caches (the estimator's trimmed to prompt mel + 100 frames), so each
    # chunk runs only its new tokens / frames; version stays 2 (HiFT with ConvTranspose, 8-frame mel cache)
    stream: str = "recompute"
    label: str = ""
    est_keep: int = 100             # estimator att cache kept beyond the prompt mel (token2wav.stream) 「来源」

    @property
    def mel_rate(self) -> int:
        return self.token_rate * self.mel_ratio

    @property
    def cfg_mult(self) -> int:
        return 2 if self.cfg_rate > 0 else 1

    def _p(self, pre: str) -> int:
        return sum(math.prod(s) for n, s in self.tensors if n.startswith(pre))

    @property
    def flow_params(self) -> int:
        return self._p("flow.")

    @property
    def hift_params(self) -> int:
        return self._p("hift.")

    @property
    def stored_bytes(self) -> float:
        return (self.flow_params + self.hift_params) * 4.0

    def hops(self, n_tokens: int, warm: bool = False) -> list[tuple[int, int, bool]]:
        """(token_offset, hop, finalize) per token2wav call for an utterance of n_tokens (cli/model.py tts loop).
        0.273 stream "cache" (Step-Audio2 examples-vllm-stream.py): CHUNK_SIZE 25 + 3 look-ahead per call, no
        prompt padding, no growth; the remainder on the last call."""
        if self.stream == "cache":
            out, off = [], 0
            while n_tokens - off >= self.hop0 + self.lookahead:
                out.append((off, self.hop0, False))
                off += self.hop0
            out.append((off, n_tokens - off, True))
            return out
        pad = -(-self.prompt_tokens // self.hop0) * self.hop0 - self.prompt_tokens
        hop = self.hop0 * (self.max_hop_mult if warm else 1)
        out, off = [], 0
        while True:
            h = hop + pad if off == 0 else hop
            if n_tokens - off < h + self.lookahead:
                break
            out.append((off, h, False))
            off += h
            hop = min(self.hop0 * self.max_hop_mult, hop * self.scale)
        out.append((off, n_tokens - off, True))
        return out


def cosy_spec(cfg: dict, rel: dict | None) -> CosySpec | None:
    if cfg.get("model_type") == "step_audio_2" and rel and rel.get("token2wav") and rel.get("speech_tensors"):
        tw = rel["token2wav"]
        return CosySpec(2, tuple((n, tuple(s)) for n, s in rel["speech_tensors"].items()), tw["token_frame_rate"],
                        tw["token_mel_ratio"], tw["chunk_size"], tw["pre_lookahead_len"], tw["sample_rate"],
                        tuple(tw["upsample_rates"]), tw["istft_n_fft"], tw["istft_hop"], tw["num_heads"], tw["head_dim"],
                        tw["inference_cfg_rate"], mel_cache=tw["mel_cache"], stream="cache", label="Step-Audio-2 token2wav")
    if cfg.get("model_type") not in ("cosyvoice2", "cosyvoice3") or not rel or not rel.get("speech_tensors"):
        return None
    cv = cfg["cosyvoice"]
    return CosySpec(3 if cfg["model_type"] == "cosyvoice3" else 2,
                    tuple((n, tuple(s)) for n, s in rel["speech_tensors"].items()),
                    cv["token_frame_rate"], cv["token_mel_ratio"], cv["chunk_size"], cv["pre_lookahead_len"],
                    cv["sample_rate"], tuple(cv["upsample_rates"]), cv["istft_n_fft"], cv["istft_hop"],
                    cv["attention_heads_flow"], cv["attention_head_dim"], cv["inference_cfg_rate"])


def _gemms(c: CosySpec, prefix: str, rows_of) -> list[Op]:
    """One fp32 GEMM per weight tensor under ``prefix`` with rows from rows_of(name) (None = skip / lookup).  1-D
    tensors (biases, norms, Snake α, weight-norm gains) are read with their module: folded into the module's GEMM when
    one exists, else into a weight-read-only op."""
    ops: list[Op] = []
    used = 0
    mats = [(n, s) for n, s in c.tensors if n.startswith(prefix) and len(s) >= 2 and not n.endswith("original0")]
    for n, s in mats:
        rows = rows_of(n)
        if rows is None:
            continue
        numel = math.prod(s)
        mod = re.sub(r"\.(weight|parametrizations\.weight\.original1)$", "", n)
        side = sum(math.prod(t) for m, t in c.tensors if m in (mod + ".bias", mod + ".parametrizations.weight.original0"))
        if ".ups." in n and c.version == 2:                  # ConvTranspose1d [in, out, k]: per input position
            k, nn_ = s[0], s[1] * s[2]
        else:                                                # Linear [out, in] / Conv1d [out, in / g, k]
            k, nn_ = numel // s[0], s[0]
        ops.append(Op(n, "gemm", 0, m=rows, k=k, n=nn_, role="speech", w_params=numel + side, w_bits=32.0,
                      w_fmt="fp32", a_fmt="fp32", act_bytes=rows * (k + nn_) * 4.0, vec=rows * nn_ * VEC_PER_OUT,
                      stream=True))
        used += numel + side
    return ops


def _attn(name: str, T: int, keys: int, hd: int, cnt: int) -> list[Op]:
    return [Op(name + ".qk", "attn", 0, m=T, k=hd, n=keys, count=cnt, a_fmt="fp32", act_bytes=cnt * (T + keys) * hd * 4.0,
               stream=True),
            Op(name + ".pv", "attn", 0, m=T, k=keys, n=hd, count=cnt, a_fmt="fp32", act_bytes=cnt * (T + keys) * hd * 4.0),
            Op(name + ".softmax", "vector", 0, vec=cnt * T * keys * 6.0)]


def _rest(c: CosySpec, prefix: str, ops: list[Op], tag: str) -> list[Op]:
    """Weight bytes under ``prefix`` not carried by a GEMM (norms, Snake α / β, positional biases): one read-only op."""
    have = sum(o.w_params for o in ops if o.kind == "gemm")
    tot = c._p(prefix)
    left = tot - have
    return [Op(tag, "gemm", 0, m=1, k=left, n=1, role="speech", w_params=left, w_bits=32.0, w_fmt="fp32", a_fmt="fp32")] \
        if left > 0 else []


def flow_ops(c: CosySpec, n_tok: int) -> tuple[list[Op], list[Op]]:
    """(per-chunk ops, per-evaluation ops) of one flow.inference over n_tok tokens (prompt + generated)."""
    N, M, B = n_tok, n_tok * c.mel_ratio, c.cfg_mult
    est = "flow.decoder.estimator."
    if c.version == 2:
        def chunk_rows(n):
            if n.startswith(est) or n.startswith("flow.input_embedding"):
                return None
            if n.startswith("flow.spk_embed_affine_layer"):
                return 1
            T = M if (n.startswith("flow.encoder.up_") or n.startswith("flow.encoder_proj")) else N
            return 2 * T - 1 if ".linear_pos." in n else T

        def eval_rows(n):
            if not n.startswith(est):
                return None
            return B if (".time_mlp." in n or ".mlp.1." in n) else B * M
    else:
        def chunk_rows(n):
            if n.startswith(est) or n.startswith("flow.input_embedding"):
                return None
            return 1 if n.startswith("flow.spk_embed_affine_layer") else N

        def eval_rows(n):
            if not n.startswith(est):
                return None
            return B if (".time_embed." in n or "attn_norm.linear" in n or "norm_out.linear" in n) else B * M
    ch = _gemms(c, "flow.", chunk_rows)
    ch.append(Op("flow.input_embedding", "lookup", 0, kv_read=N * dict(c.tensors)["flow.input_embedding.weight"][1] * 4.0))
    ev = _gemms(c, est, eval_rows)
    hd = c.flow_hd
    if c.version == 2:
        names = [n for n, _ in c.tensors]
        for n, s in c.tensors:
            m_ = re.match(r"flow\.encoder\.(up_)?encoders\.(\d+)\.self_attn\.pos_bias_u$", n)
            if m_:
                T = M if m_.group(1) else N
                h = s[0]
                # rel-pos (espnet): q·kᵀ (matrix_ac) + q·posᵀ (matrix_bd over 2T − 1 relative positions)
                ch += _attn(n[:-len(".pos_bias_u")], T, T, s[1], h)
                ch.append(Op(n[:-len(".pos_bias_u")] + ".bd", "attn", 0, m=T, k=s[1], n=2 * T - 1, count=h, a_fmt="fp32",
                             act_bytes=h * (3 * T) * s[1] * 4.0))
        blocks = [n for n in names if n.startswith(est) and n.endswith("attn1.to_q.weight")]
    else:
        blocks = [n for n, _ in c.tensors if n.startswith(est) and n.endswith("attn.to_q.weight")]
    for n in blocks:
        q = dict(c.tensors)[n][0]
        ev += _attn(n[:-len(".to_q.weight")], M, M, hd, B * (q // hd))
    ev.append(Op("flow.euler_cfg", "vector", 0, vec=M * 80 * 6))
    ch += _rest(c, "flow.", ch + ev, "flow.vectors")
    return ch, ev


def flow_chunk_ops(c: CosySpec, N: int, enc_off: int, est_cache: int) -> tuple[list[Op], list[Op]]:
    """0.273 Step-Audio-2 token2wav, one flow.inference_chunk (Step-Audio2 cosyvoice2/flow/flow.py, upsample_encoder_v2.py
    forward_chunk, decoder_dit.py forward_chunk 「来源」): N new tokens (+3 look-ahead into the pre-look-ahead conv),
    M = 2N new mel frames.  Conformer: 6 blocks on N rows over enc_off + N keys (att cache of every token since
    setup_cache, never trimmed; espnet rel-pos: linear_pos / q·posᵀ over 2·keys − 1 positions), up conv, 4 blocks on
    M rows over 2·enc_off + M keys.  Estimator (DiT 16 × d512, 8 × 64, causal conv block k3 ×2, adaLN 9d): per Euler
    step B·M rows; attention over est_cache + M keys (K/V cache per step and block); t-embedding / adaLN on B rows."""
    M, B = N * c.mel_ratio, c.cfg_mult
    L = N + c.lookahead
    est = "flow.decoder.estimator."
    ko, ku = enc_off + N, 2 * enc_off + M

    def chunk_rows(n):
        if n.startswith(est) or n.startswith("flow.input_embedding"):
            return None
        if n.startswith("flow.spk_embed_affine_layer"):
            return 1
        if n.startswith("flow.encoder.pre_lookahead_layer"):
            return N
        up = n.startswith(("flow.encoder.up_", "flow.encoder_proj"))
        if ".linear_pos." in n:
            return 2 * (ku if up else ko) - 1
        return M if up else N

    def eval_rows(n):
        if not n.startswith(est):
            return None
        return B if (".t_embedder." in n or "adaLN_modulation" in n) else B * M
    ch = _gemms(c, "flow.", chunk_rows)
    ch.append(Op("flow.input_embedding", "lookup", 0, kv_read=L * dict(c.tensors)["flow.input_embedding.weight"][1] * 4.0))
    for n, s_ in c.tensors:
        m_ = re.match(r"flow\.encoder\.(up_)?encoders\.(\d+)\.self_attn\.pos_bias_u$", n)
        if m_:
            T, K = (M, ku) if m_.group(1) else (N, ko)
            h = s_[0]
            ch += _attn(n[:-len(".pos_bias_u")], T, K, s_[1], h)
            ch.append(Op(n[:-len(".pos_bias_u")] + ".bd", "attn", 0, m=T, k=s_[1], n=2 * K - 1, count=h, a_fmt="fp32",
                         act_bytes=h * (T + 2 * K) * s_[1] * 4.0))
    ev = _gemms(c, est, eval_rows)
    hd = c.flow_hd
    for n in [n for n, _ in c.tensors if n.startswith(est) and n.endswith("attn.to_q.weight")]:
        q = dict(c.tensors)[n][0]
        ev += _attn(n[:-len(".to_q.weight")], M, est_cache + M, hd, B * (q // hd))
    ev.append(Op("flow.euler_cfg", "vector", 0, vec=M * 80 * 6))
    ch += _rest(c, "flow.", ch + ev, "flow.vectors")
    return ch, ev


def hift_ops(c: CosySpec, M: int) -> list[Op]:
    cum = [math.prod(c.ups[:i + 1]) for i in range(len(c.ups))]
    nres = len([n for n, _ in c.tensors if re.match(r"hift\.resblocks\.\d+\.convs1\.0\.bias$", n)]) // len(c.ups)

    def rows(n):
        if n.startswith(("hift.f0_predictor", "hift.conv_pre")):
            return M
        m_ = re.match(r"hift\.ups\.(\d+)\.", n)
        if m_:
            i = int(m_.group(1))
            return M * (cum[i - 1] if i else 1) if c.version == 2 else M * cum[i]
        m_ = re.match(r"hift\.resblocks\.(\d+)\.", n)
        if m_:
            return M * cum[int(m_.group(1)) // nres]
        m_ = re.match(r"hift\.source_(downs|resblocks)\.(\d+)", n)
        if m_:
            return M * cum[int(m_.group(2))]
        if n.startswith("hift.conv_post"):
            return M * cum[-1]
        if n.startswith("hift.m_source"):
            return M * cum[-1] * c.istft_hop
        return M
    ops = _gemms(c, "hift.", rows)
    S, F, nf = M * cum[-1] * c.istft_hop, M * cum[-1], c.istft_nfft
    snake = sum(o.m * o.n * 10 for o in ops if ".convs" in o.name)     # Snake before every ResBlock conv
    ops += [Op("hift.snake", "vector", 0, vec=snake, sfu=snake / 5),
            Op("hift.sinegen", "vector", 0, vec=S * 9 * 12, sfu=S * 9),
            Op("hift.stft_istft", "vector", 0, vec=2 * F * (5.0 * nf * math.log2(nf) + 4 * nf))]
    return ops + _rest(c, "hift.", ops, "hift.vectors")


def cosy_pin(c: CosySpec, sram_bytes: float) -> float:
    """0.269 re-read rule on the flow estimator (read ``steps`` times per chunk): DRAM miss share of a re-read."""
    W = c._p("flow.decoder.estimator.") * 4.0
    free = max(0.0, sram_bytes - 2 * 2 ** 20)
    return 1.0 - min(W, free) / W


def scaled(ops: list[Op], k: float) -> list[Op]:
    return [replace(o, w_params=int(round(o.w_params * k))) if o.kind == "gemm" else o for o in ops]
