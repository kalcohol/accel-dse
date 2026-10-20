"""0.273 Kimi-Audio-7B-Instruct speech output (docs/MODEL.md §218).

Reference code 「来源」 (github MoonshotAI/Kimi-Audio master): kimia_infer/api/kimia.py (_generate_loop,
detokenize_audio), models/detokenizer/__init__.py (PrefixStreamingFlowMatchingDetokenizer.detokenize_streaming,
get_audio_detokenizer), semantic_fm_prefix_streaming.py (infer_chunk), flow_matching/model.py (DiT forward),
flow_matching/dit_block.py, flow_matching/ode_wrapper.py (update_incremental_state), flow_matching/scheduler.py
(torchdyn NeuralODE Euler), bigvgan_wrapper.py; HF moonshotai/Kimi-Audio-7B-Instruct config.json,
audio_detokenizer/config.yaml + model.pt, vocoder/config.json + model.pt (scripts/fetch_pt_release.py --kimi).

  LM (the catalog model): each step emits one text token (lm_head) and one audio token (mimo_output, 12.5 Hz semantic
      tokens, GLM-4-Voice codebook 16384); the first kimia_mimo_audiodelaytokens (6) audio slots are blanks.  Audio in
      real time needs 12.5 steps / s → LM RTF = 12.5 · TPOT.
  Detokenizer (batch 1, bf16 — the .pt is fp32, cast at load): tokens ×4 (repeat_interleave → 50 Hz mel frames);
      chunks of 30 tokens (the first too); each chunk carries the previous chunk's last 12 tokens (look_ahead) in
      front, so it runs (12 + 30) · 4 = 168 frames (first chunk 120) and keeps all but the last 48.  Flow matching:
      t_span = linspace(0, 1, ode_step = 30) through torchdyn's fixed-step Euler → 29 DiT evaluations, then one more
      evaluation at t = 0.999 over the kept frames to write their K/V (infer_chunk).  DiT: 16 blocks, d 2304, 18 × 128
      heads, MLP 4d, adaLN-Zero modulation 6d **per frame** (c = t-embedding + semantic-token embedding, (N, T, D)),
      RoPE, LayerNorms without affine; attention over the K/V of earlier kept frames (capped at max_kv_cache_tokens
      900 frames, newest kept) + the chunk.  The last chunk (is_final) keeps everything (no extra evaluation).
  BigVGAN (24 kHz, hop 480 = Π upsample_rates 5·2·2·2·2·3·2): runs on [held-back mel + new mel]; returns the first
      half of chunk 0 and holds the rest; later chunks return 2 × the held length (overlap-add with a Hamming
      window) — the frame bookkeeping below follows detokenize_streaming exactly.
  The released kimia.py generates every token first and then detokenizes; a streaming deployment that feeds the
  chunk API as tokens arrive 「假设」 gives first audio = T_ft + (6 + 30 − 1) · TPOT + chunk 0 (DiT + commit + BigVGAN).
Ops are built from the released tensor shapes (core/cosyvoice._gemms: one GEMM per weight, BigVGAN ConvTranspose per
input position, weight-norm gains folded); weights / activations in bf16.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .cosyvoice import _attn, _gemms, _rest
from .ir import Op


@dataclass(frozen=True)
class KimiSpec:
    tensors: tuple                 # ((name, shape), …) detok.* / voc.*
    token_rate: float = 12.5
    upsample: int = 4
    chunk: int = 30
    first_chunk: int = 30
    look_ahead: int = 12
    ode_steps: int = 30
    max_kv: int = 900
    delay: int = 6                 # kimia_mimo_audiodelaytokens
    depth: int = 16
    d: int = 2304
    heads: int = 18
    mel: int = 80
    hop: int = 480
    rate: int = 24000
    ups: tuple = (5, 2, 2, 2, 2, 3, 2)
    version: int = 2               # _gemms: ".ups." tensors are ConvTranspose1d [in, out, k]
    label: str = "Kimi-Audio detokenizer"

    def _p(self, pre: str) -> int:
        return sum(math.prod(s) for n, s in self.tensors if n.startswith(pre))

    @property
    def dit_params(self) -> int:
        return self._p("detok.")

    @property
    def voc_params(self) -> int:
        return self._p("voc.")

    @property
    def evals(self) -> int:
        """torchdyn NeuralODE(solver="euler") over t_span = linspace(0, 1, ode_steps): one evaluation per interval."""
        return self.ode_steps - 1

    @property
    def mel_rate(self) -> float:
        return self.token_rate * self.upsample

    @property
    def stored_bytes(self) -> float:
        return (self.dit_params + self.voc_params) * 2.0       # bf16 resident (cast at load)

    def kv_bytes(self, frames: int) -> float:
        return self.depth * 2 * frames * self.d * 2.0

    def plan(self, n_tok: int) -> list[dict]:
        """Per detokenize_streaming call: new tokens, DiT rows T, kept frames, DiT K/V frames before the call, BigVGAN
        input frames, audio frames returned (kimia.py detokenize_audio + PrefixStreamingFlowMatchingDetokenizer)."""
        la, up = self.look_ahead, self.upsample
        calls, off = [], 0
        sizes = [min(self.first_chunk, n_tok)]
        rest = n_tok - sizes[0]
        while rest > 0:
            sizes.append(min(self.chunk, rest))
            rest -= sizes[-1]
        cache, pre, out = 0, None, []
        for i, n in enumerate(sizes):
            fin = i == len(sizes) - 1
            T = ((la if i else 0) + n) * up
            keep = T if fin else T - la * up
            if pre is None:
                voc = keep
                ret = keep if fin else (keep * self.hop // 2) // self.hop     # first half of chunk 0
                pre_n = 0 if fin else keep - keep // 2
            else:
                voc = pre + keep
                ret = voc if fin else (2 * pre if voc > 2 * pre else voc // 2)
                pre_n = voc - ret
            out.append({"tokens": n, "T": T, "keep": keep, "kv": cache, "voc": voc, "ret": ret, "final": fin,
                        "commit": not fin})
            cache = min(self.max_kv, cache + keep)
            pre = pre_n
        return out


def kimi_spec(cfg: dict, rel: dict | None) -> KimiSpec | None:
    if "MoonshotKimiaForCausalLM" not in (cfg.get("architectures") or []) or not rel or not rel.get("kimi_audio"):
        return None
    k = rel["kimi_audio"]
    dit, voc = k["dit"], k["vocoder"]
    return KimiSpec(tuple((n, tuple(s)) for n, s in rel["speech_tensors"].items()), k["token_rate"], k["upsample"],
                    k["chunk"], k["first_chunk"], k["look_ahead"], k["ode_steps"], k["max_kv_frames"],
                    cfg.get("kimia_mimo_audiodelaytokens", 0), dit["depth"], dit["hidden_size"], dit["num_heads"],
                    voc["num_mels"], voc["hop_size"], voc["sampling_rate"], tuple(voc["upsample_rates"]))


def _bf16(ops: list[Op]) -> list[Op]:
    return [replace(o, w_bits=16.0, w_fmt="bf16", a_fmt="bf16", act_bytes=o.act_bytes / 2) if o.kind in ("gemm", "attn")
            else o for o in ops]


def dit_eval_ops(k: KimiSpec, T: int, kv: int) -> list[Op]:
    """One DiT evaluation on T frames over kv cached frames: every block GEMM (adaLN modulation included — c is per
    frame) on T rows, the t-embedder MLP on one row, the semantic-token embedding as a T-row lookup."""
    def rows(n):
        if n.startswith("detok.semantic_token_embedding"):
            return None
        return 1 if n.startswith("detok.t_embedder") else T
    ops = _gemms(k, "detok.", rows)
    hd = k.d // k.heads
    for li in range(k.depth):
        a = _attn(f"detok.blocks.{li}.attn", T, kv + T, hd, k.heads)
        # 0.289: the block's K / V of the kv cached frames (bf16, all heads) re-read by the evaluation 「推导」 — on the
        # attention op (≤ 0.288: speech.py passed Σ blocks as extra_dram = kv_bytes(kv); same bytes)
        a[0] = replace(a[0], kv_read=2 * kv * k.d * 2.0)
        ops += a
        ops.append(Op(f"detok.blocks.{li}.norm_mod_rope", "vector", li, vec=T * k.d * 16.0))
    ops.append(Op("detok.semantic_embed", "lookup", 0, kv_read=T * k.d * 2.0, vec=T * k.d * 2.0))
    ops.append(Op("detok.euler", "vector", 0, vec=T * k.mel * 3.0))
    return _bf16(ops)


def voc_ops(k: KimiSpec, M: int) -> list[Op]:
    """BigVGAN on M mel frames: conv_pre at the mel rate, ups.i (ConvTranspose) per input position, AMP resblocks
    (4 kernel sizes per stage, 2 × 3 convs) at the stage's output rate with an anti-aliased SnakeBeta before every
    conv (2× FIR up / down, 12 taps 「近似」 as core/omni.SNAKE_VEC), activation_post + conv_post at 24 kHz."""
    from .omni import SNAKE_VEC
    import re
    cum = [math.prod(k.ups[:i + 1]) for i in range(len(k.ups))]
    nk = len([n for n, _ in k.tensors if re.match(r"voc\.resblocks\.\d+\.convs1\.0\.bias$", n)]) // len(k.ups)

    def rows(n):
        if n.startswith("voc.conv_pre"):
            return M
        m_ = re.match(r"voc\.ups\.(\d+)\.", n)
        if m_:
            i = int(m_.group(1))
            return M * (cum[i - 1] if i else 1)
        m_ = re.match(r"voc\.resblocks\.(\d+)\.", n)
        if m_:
            return M * cum[int(m_.group(1)) // nk]
        return M * cum[-1]                       # conv_post
    ops = _gemms(k, "voc.", rows)
    vec = []
    for o in ops:
        if ".convs" in o.name or o.name.startswith("voc.conv_post"):
            vec.append(Op(o.name + ".snake", "vector", 0, vec=o.m * o.k / max(1, _ksize(k, o.name)) * SNAKE_VEC,
                          sfu=o.m * o.k / max(1, _ksize(k, o.name)) * 2))
    return _bf16(ops + _rest(k, "voc.", ops, "voc.snake_params") + vec)


def _ksize(k: KimiSpec, name: str) -> int:
    for n, s in k.tensors:
        if n == name:
            return s[-1]
    return 1


def kimi_pin(k: KimiSpec, sram_bytes: float) -> float:
    """0.269 re-read rule on the DiT (read 30 times per chunk): DRAM miss share of a re-read."""
    W = k.dit_params * 2.0
    free = max(0.0, sram_bytes - 2 * 2 ** 20)
    return 1.0 - min(W, free) / W


def scaled(ops: list[Op], s: float) -> list[Op]:
    return [replace(o, w_params=int(round(o.w_params * s))) if o.kind == "gemm" else o for o in ops]
