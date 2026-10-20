"""0.275 FireRedTTS3-Base zero-shot TTS (docs/MODEL.md §220.2).

Reference code 「来源」 github FireRedTeam/FireRedTTS3@7a1f3a7: fireredtts3/llm/fireredtts3_base.py
(Qwen3_1_7B_ConfigDict, FireRedTTS3BaseCore.generate / _flow_one_step / _backbone_one_step), llm/dit.py (DiT,
DiTBlock with conv, adaLN 9d, FinalLayer adaLN 2d), llm/patch_encoder.py + llm/modules.py (PatchEncoder: cls + patch,
plain DiTBlocks, RMSNorm FinalLayer), redae/redae.py (RedAE encoder / decoder: Qwen3 stacks, ISTFT head), core.py
(generate defaults n_timesteps 10, inference_cfg 2.0); HF FireRedTeam/FireRedTTS3 fireredtts3_base / redae headers.

One autoregressive step (``generate``) = one patch of 4 RedAE latents = 4 × 960 samples @ 24 kHz = 160 ms of audio:
  1. backbone: Qwen3-1.7B (28 × d2048, GQA 16/8, head 128, no lm_head), one position, KV cache — under bf16
     autocast (weights stored fp32; each call re-casts them 「推导」: modelled as the fp32 read only 「近似」);
  2. stop_head Linear(2048 → 1) on the new hidden state;
  3. dit_head Linear(2048 → 1024) over the last history_patches + 1 = 3 backbone outputs (re-run every step);
  4. _flow_one_step: ``n_timesteps`` = 10 Euler steps over t = 1 − cos(π/2 · linspace) ; inference_cfg 2.0 > 0
     doubles the batch (cond + zeroed-cond rows).  DiT input per row = [8 history latents + 4 noisy latents] ×
     (latent 64 ‖ backbone cond 1024 repeated per patch ‖ speaker 512) → in_proj → 11 DiTBlocks (adaLN 9d from the
     timestep, RMSNorm, full attention over the 12 positions, conv1d k3 ×2 with Mish, GELU-tanh MLP 3d) → adaLN
     FinalLayer → 64; only the 4 current latents are updated.  No autocast here → fp32 「来源」;
  5. patch_encoder: [cls + 4 latents] → in_proj → 8 plain DiTBlocks (MLP 4d) → RMSNorm + Linear(1024 → 2048) on all
     5 positions, cls row = the next backbone input embedding.
After the loop (per text segment; core.py splits long text and cross-fades) RedAE decodes every latent at once
(non-streaming): in_proj 64 → 2 × 896 (25 → 50 Hz), Qwen3 18 × d896 (GQA 14/2, head 128, SwiGLU 3584, sliding
window 64, no KV cache — one pass), ISTFT head Linear(896 → 1922) + irfft (n_fft 1920, hop 480), fp32.
Per request before the loop: RedAE encoder on the prompt audio (patches of 480 samples → Qwen3 18 × d896 → 4-layer
cls downsample over pairs → 64), the patch encoder on the prompt latents, spk_proj_llm / spk_proj_dit, and the
campplus speaker encoder (not costed: its checkpoint is a pickled .bin, not read).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ir import Op

VEC_PER_OUT = 3.0


@dataclass(frozen=True)
class FireRedSpec:
    tensors: tuple                  # ((name, shape), …) speech head + redae.*
    llm_h: int = 2048
    z: int = 64
    patch: int = 4
    hist_patches: int = 2
    spk: int = 512
    dit_d: int = 1024
    dit_depth: int = 11
    dit_heads: int = 16
    dit_mlp: float = 3.0
    pe_d: int = 1024
    pe_depth: int = 8
    pe_heads: int = 16
    pe_mlp: float = 4.0
    ae_h: int = 896
    ae_layers: int = 18
    ae_heads: int = 14
    ae_kv: int = 2
    ae_hd: int = 128
    ae_inter: int = 3584
    ae_window: int = 64
    ae_down_layers: int = 4
    up: int = 2                     # RedAE extra downsample / decoder upsample (25 ↔ 50 Hz)
    hop: int = 480
    rate: int = 24000
    flow_steps: int = 10
    cfg_scale: float = 2.0
    kind: str = "firered"
    w_bytes: float = 4.0            # everything stored and run fp32 (the backbone's autocast aside)

    def _p(self, *pre: str, skip: tuple = ()) -> int:
        return sum(math.prod(s) for n, s in self.tensors if n.startswith(pre) and not n.startswith(skip))

    @property
    def flow_params(self) -> int:
        return self._p("dit.")

    @property
    def agg_params(self) -> int:
        return self._p("patch_encoder.")

    @property
    def stop_params(self) -> int:
        """Small per-step heads: stop_head + dit_head."""
        return self._p("stop_head.", "dit_head.")

    @property
    def spk_params(self) -> int:
        return self._p("spk_proj_")

    @property
    def dec_params(self) -> int:
        return 0                    # non-streaming: RedAE decodes once per segment, not per step

    @property
    def redae_dec_params(self) -> int:
        return self._p("redae.decoder.", skip=("redae.decoder.qwen3.embed_tokens",))

    @property
    def redae_enc_params(self) -> int:
        return self._p("redae.encoder.", skip=("redae.encoder.qwen3.embed_tokens", "redae.encoder.downsample.qwen3.embed_tokens"))

    @property
    def redae_params(self) -> int:
        return self._p("redae.")

    @property
    def head_params(self) -> int:
        return self.flow_params + self.agg_params + self.stop_params + self.spk_params

    @property
    def run_params(self) -> int:
        """Weights read every step."""
        return self.flow_params + self.agg_params + self.stop_params

    @property
    def stored_params(self) -> int:
        return self.spk_params + self.redae_params

    @property
    def total_params(self) -> int:
        return self.head_params + self.redae_params

    @property
    def s_per_step(self) -> float:
        return self.patch * self.up * self.hop / self.rate

    @property
    def flow_rows(self) -> int:
        return (self.hist_patches + 1) * self.patch

    @property
    def cfg_mult(self) -> int:
        return 2 if self.cfg_scale > 0 else 1        # _flow_one_step: `if inference_cfg > 0` 「来源」

    @property
    def dec_kv_bytes(self) -> float:
        return 0.0

    @property
    def n_fft(self) -> int:
        return 4 * self.hop


def fire_spec(cfg: dict, rel: dict | None) -> FireRedSpec | None:
    if cfg.get("model_type") != "fireredtts3" or not rel or not rel.get("speech_tensors"):
        return None
    c, r, g = cfg["core"], cfg["redae"], cfg.get("generate") or {}
    return FireRedSpec(tuple((n, tuple(s)) for n, s in rel["speech_tensors"].items()), cfg["backbone"]["hidden_size"],
                       c["redae_dim"], c["patch_size"], c["num_history_patches"], c["spk_in_dim"],
                       c["dit_hidden_size"], c["dit_depth"], c["dit_num_heads"], float(c["dit_mlp_ratio"]),
                       c["patch_encoder_hidden_size"], c["patch_encoder_depth"], c["patch_encoder_num_heads"],
                       float(c["patch_encoder_mlp_ratio"]), r["dec_hidden_size"], r["dec_num_hidden_layers"],
                       r["dec_num_attention_heads"], r["dec_num_key_value_heads"], 128, r["dec_intermediate_size"],
                       r["dec_sliding_window"], r["enc_downsample_num_hidden_layers"], r["enc_extra_downsample_rate"],
                       r["audio_patch_size"], r["audio_sample_rate"], g.get("n_timesteps", 10),
                       g.get("inference_cfg", 2.0))


def _ops_kit(layer: int, replicated: int, pre: str, bias: bool = True):
    """``bias``: nn.Linear default (the speech head's Linears all carry one; the RedAE Qwen3 projections do not)."""
    ab = 4.0

    def gemm(name: str, m: int, k: int, n: int, w: int | None = None) -> Op:
        return Op(pre + name, "gemm", layer, m=m, k=k, n=n, role="tts", w_params=(k * n + (n if bias else 0)) if w is None else w,
                  w_bits=32.0, w_fmt="fp32", a_fmt="fp32", act_bytes=m * (k + n) * ab, replicated=replicated,
                  vec=m * n * VEC_PER_OUT)

    def attn(name: str, rows: int, keys: int, hd: int, cnt: int) -> list[Op]:
        return [Op(f"{pre}{name}.qk", "attn", layer, m=rows, k=hd, n=keys, count=cnt, a_fmt="fp32",
                   replicated=replicated, act_bytes=cnt * (rows + keys) * hd * ab),
                Op(f"{pre}{name}.pv", "attn", layer, m=rows, k=keys, n=hd, count=cnt, a_fmt="fp32",
                   replicated=replicated, act_bytes=cnt * (rows + keys) * hd * ab),
                Op(f"{pre}{name}.softmax", "vector", layer, vec=cnt * rows * keys * 5.0, replicated=replicated)]
    return gemm, attn


def fire_head_ops(t: FireRedSpec, b: int, layer: int, replicated: int = 1, af: str = "fp32") -> list[Op]:
    """Ops of one AR step's speech head for ``b`` requests (after the backbone's final norm)."""
    gemm, attn = _ops_kit(layer, replicated, "tts.")
    d, f = t.dit_d, int(t.dit_mlp * t.dit_d)
    ops = [gemm("stop_head", b, t.llm_h, 1, w=t.llm_h + 1),
           gemm("dit_head", b * (t.hist_patches + 1), t.llm_h, d, w=t.llm_h * d + d)]
    B = b * t.cfg_mult
    S = t.flow_rows
    R = B * S
    for s in range(t.flow_steps):
        p = f"flow{s}"
        ops += [gemm(f"{p}.t_mlp0", B, 256, d), gemm(f"{p}.t_mlp2", B, d, d),
                gemm(f"{p}.in_proj", R, t.z + d + t.spk, d)]
        for i in range(t.dit_depth):
            q = f"{p}.{i}"
            ops += [gemm(f"{q}.adaln", B, d, 9 * d), gemm(f"{q}.qkv", R, d, 3 * d), gemm(f"{q}.o", R, d, d),
                    gemm(f"{q}.conv1", R, 3 * d, d), gemm(f"{q}.conv2", R, 3 * d, d),
                    gemm(f"{q}.fc1", R, d, f), gemm(f"{q}.fc2", R, f, d),
                    Op(f"tts.{q}.norm_mod", "vector", layer, vec=R * (d * 18 + f * 8), sfu=R * (f + d),
                       replicated=replicated)]
            ops += attn(q, S, S, d // t.dit_heads, B * t.dit_heads)
        ops += [gemm(f"{p}.final_adaln", B, d, 2 * d), gemm(f"{p}.final", R, d, t.z),
                Op(f"tts.{p}.euler", "vector", layer, vec=b * t.patch * t.z * 8, replicated=replicated)]
    ops += patch_encoder_ops(t, b, layer, replicated)
    return ops


def patch_encoder_ops(t: FireRedSpec, n_patches: int, layer: int = 0, replicated: int = 1) -> list[Op]:
    gemm, attn = _ops_kit(layer, replicated, "tts.")
    d, f = t.pe_d, int(t.pe_mlp * t.pe_d)
    g = 1 + t.patch
    R = n_patches * g
    ops = [gemm("pe.in_proj", n_patches * t.patch, t.z, d)]
    for i in range(t.pe_depth):
        ops += [gemm(f"pe.{i}.qkv", R, d, 3 * d), gemm(f"pe.{i}.o", R, d, d), gemm(f"pe.{i}.fc1", R, d, f),
                gemm(f"pe.{i}.fc2", R, f, d),
                Op(f"tts.pe.{i}.norm", "vector", layer, vec=R * (d * 8 + f * 8), sfu=R * f, replicated=replicated)]
        ops += attn(f"pe.{i}", g, g, d // t.pe_heads, n_patches * t.pe_heads)
    ops += [gemm("pe.out", R, d, t.llm_h)]
    return ops


def _qwen_stack(t: FireRedSpec, gemm, attn, tag: str, rows: int, seq: int, keys: int, layers: int, groups: int) -> list[Op]:
    h, kvd, qd = t.ae_h, t.ae_kv * t.ae_hd, t.ae_heads * t.ae_hd
    ops: list[Op] = []
    for i in range(layers):
        ops += [gemm(f"{tag}.{i}.q", rows, h, qd), gemm(f"{tag}.{i}.kv", rows, h, 2 * kvd),
                gemm(f"{tag}.{i}.o", rows, qd, h), gemm(f"{tag}.{i}.gate_up", rows, h, 2 * t.ae_inter),
                gemm(f"{tag}.{i}.down", rows, t.ae_inter, h),
                Op(f"redae.{tag}.{i}.norm", "vector", 0, vec=rows * (h * 8 + t.ae_inter * 6), sfu=rows * t.ae_inter)]
        ops += attn(f"{tag}.{i}", seq, keys, t.ae_hd, groups * t.ae_heads)
    return ops


def redae_decode_ops(t: FireRedSpec, latents: int) -> list[Op]:
    """RedAE decoder over ``latents`` (25 Hz) at once → 2 × latents frames (50 Hz) → hop 480 samples each."""
    gemm, attn = _ops_kit(0, 1, "redae.", bias=False)
    M = latents * t.up
    W = min(t.ae_window, M)
    n = t.n_fft
    return ([gemm("dec.in_proj", latents, t.z, t.up * t.ae_h, w=t.z * t.up * t.ae_h + t.up * t.ae_h)]
            + _qwen_stack(t, gemm, attn, "dec", M, M, W, t.ae_layers, 1)
            + [gemm("dec.istft_out", M, t.ae_h, n + 2, w=t.ae_h * (n + 2) + n + 2),
               Op("redae.dec.irfft", "vector", 0, vec=M * (5.0 * n * math.log2(n) + 4 * n))])


def prompt_ops(t: FireRedSpec, prompt_audio_s: float) -> list[Op]:
    """Per request before the AR loop: RedAE encoder on the prompt audio, patch encoder on its latents, spk proj."""
    gemm, attn = _ops_kit(0, 1, "redae.", bias=False)
    lat_patches = max(1, math.ceil(prompt_audio_s * t.rate / (t.hop * t.up * t.patch)))
    E = lat_patches * t.patch * t.up                # 50 Hz frames (padded to a whole patch)
    h = t.ae_h
    ops = [gemm("enc.in0", E, t.hop, h, w=t.hop * h + h), gemm("enc.in1", E, h, h, w=h * h + h)]
    ops += _qwen_stack(t, gemm, attn, "enc", E, E, min(t.ae_window, E), t.ae_layers, 1)
    G = E // t.up
    ops += _qwen_stack(t, gemm, attn, "enc.down", G * (1 + t.up), 1 + t.up, 1 + t.up, t.ae_down_layers, G)
    ops += [gemm("enc.out", G, h, t.z, w=h * t.z + t.z)]
    ops += patch_encoder_ops(t, lat_patches)
    ops += [gemm("spk_proj_llm", 1, t.spk, t.llm_h, w=t.spk * t.llm_h + t.llm_h),
            gemm("spk_proj_dit", 1, t.spk, t.spk, w=t.spk * t.spk + t.spk)]
    return ops


def prompt_patches(t: FireRedSpec, prompt_audio_s: float) -> int:
    return max(1, math.ceil(prompt_audio_s * t.rate / (t.hop * t.up * t.patch)))


def fire_step_flops(t: FireRedSpec, b: int = 1) -> dict:
    ops = fire_head_ops(t, b, 0)
    f = lambda pre: sum(o.flops for o in ops if o.name.startswith(pre) and o.kind in ("gemm", "attn"))
    return {"flow": f("tts.flow"), "agg": f("tts.pe"), "vae": 0.0, "stop": f("tts.stop") + f("tts.dit_head"),
            "total": sum(o.flops for o in ops if o.kind in ("gemm", "attn"))}
