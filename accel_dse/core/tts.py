"""0.269 speech synthesis (TTS) decode head: Ming-3.0-tts-tiny (inclusionAI, ``lingv3_tts``).

Reference code 「来源」 (HF repo inclusionAI/Ming-3.0-tts-tiny): modeling_lingv3_tts.py ``generate_speech``, CFM.py,
dit.py, aggregator.py, modules.py (DiTBlock), vae_modules.py (Decoder.low_level_reconstruct), istft.py.

One autoregressive step = one LLM token whose last hidden state h drives
  1. ``stop_head`` Linear(h → 2) (stop probability, every step);
  2. ``flowloss.sample``: CFM Euler solver, ``flow_steps`` (default 10, EPSS timesteps) DiT evaluations; with
     ``cfg_scale`` ≠ 1 (default 2.0) DiT.forward_with_cfg doubles the batch (cond + zeroed-cond rows).  DiT input per
     row = [t-emb + c-emb, x-emb(latent history), x-emb(noisy patch)] = 1 + history + patch tokens (3 for the release),
     ``depth`` pre-RMSNorm blocks (biased q/k/v/out, GELU-tanh MLP mlp·d), RMSNorm + Linear(d → z) on every token;
     t_embedder (256 → d → d) and c_embedder (h → d) are re-run per evaluation (the reference does not cache them);
  3. ``linear_proj_audio`` Aggregator: [cls, x-emb(latent)] → ``agg_depth`` DiT blocks → Linear(d → h) = the next LLM
     input embedding (replaces the token embedding lookup);
  4. streaming AudioVAE decode of the new latent: fc1 (z → 896), linear ×patch upsample (4 frames), Qwen2 decoder
     (24 layers, GQA 14/2, head 64, SwiGLU 4864, sliding window 64, KV cache), ISTFT head Linear(896 → n_fft + 2),
     irfft (n_fft = 4·hop = 3528) + overlap-add.  44.1 kHz, hop 882 → one LLM step = 4 · 882 samples = 80 ms of audio.
Stored but not run for plain TTS: the AudioVAE encoder (+ its aggregator; voice-prompt latents only) and spk_head
(one Linear per request at prompt build) → storage only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ir import Op

VEC_PER_OUT = 3.0               # epilogue element-ops per GEMM output (bias / activation), as vision.py


@dataclass(frozen=True)
class TtsSpec:
    llm_h: int
    z: int                      # latent channels
    patch: int                  # latents per LLM step (tts_config.patch_size)
    history: int                # latent history tokens fed to the DiT
    flow_d: int
    flow_depth: int
    flow_heads: int
    flow_mlp: float
    agg_d: int
    agg_depth: int
    agg_heads: int
    agg_mlp: float
    dec_h: int
    dec_layers: int
    dec_heads: int
    dec_kv: int
    dec_inter: int
    dec_window: int
    up: int                     # AudioVAE patch_size (linear upsample frames per latent)
    hop: int                    # samples per decoder frame
    rate: int
    flow_steps: int = 10        # generate() default
    cfg_scale: float = 2.0      # generate() default
    enc_params: int = 0         # stored only (AudioVAE encoder + its aggregator)
    spk_params: int = 0         # stored only (spk_head)

    # ---- params (released safetensors header, exact)
    @staticmethod
    def _block(d: int, mlp: float) -> int:
        f = int(mlp * d)
        return 4 * (d * d + d) + (d * f + f) + (f * d + d) + 2 * d

    @property
    def flow_params(self) -> int:
        d = self.flow_d
        return (self.flow_depth * self._block(d, self.flow_mlp) + (256 * d + d) + (d * d + d) + (self.z * d + d)
                + (self.llm_h * d + d) + d + (d * self.z + self.z) + (d // self.flow_heads) // 2)

    @property
    def agg_params(self) -> int:
        d = self.agg_d
        return (self.agg_depth * self._block(d, self.agg_mlp) + d + (d * self.llm_h + self.llm_h) + (self.z * d + d)
                + d + (d // self.agg_heads) // 2)

    @property
    def stop_params(self) -> int:
        return 2 * self.llm_h + 2

    @property
    def n_fft(self) -> int:
        return 4 * self.hop

    @property
    def dec_params(self) -> int:
        h, hd = self.dec_h, self.dec_h // self.dec_heads
        kv = self.dec_kv * hd
        layer = (h * h + h) + 2 * (h * kv + kv) + h * h + 3 * h * self.dec_inter + 2 * h
        return (self.dec_layers * layer + 2 * h + (self.z * h + h) + (h * (self.n_fft + 2) + self.n_fft + 2)
                + self.n_fft)

    @property
    def run_params(self) -> int:
        """Weights read every step (flow DiT, aggregator, stop head, AudioVAE decoder)."""
        return self.flow_params + self.agg_params + self.stop_params + self.dec_params

    @property
    def llm_side_params(self) -> int:
        """Non-LLM tensors the release summary files under params_llm (no "audio" in the name)."""
        return self.flow_params + self.stop_params + self.spk_params

    @property
    def stored_params(self) -> int:
        return self.enc_params + self.spk_params

    @property
    def total_params(self) -> int:
        return self.run_params + self.stored_params

    @property
    def s_per_step(self) -> float:
        return self.patch * self.up * self.hop / self.rate

    @property
    def flow_rows(self) -> int:
        return 1 + self.history + self.patch

    @property
    def cfg_mult(self) -> int:
        return 1 if self.cfg_scale == 1.0 or self.cfg_scale < 1e-5 else 2

    @property
    def dec_kv_bytes(self) -> float:
        """Per-request AudioVAE decoder sliding-window K/V (bf16)."""
        return self.dec_layers * self.dec_window * 2 * self.dec_kv * (self.dec_h // self.dec_heads) * 2.0


def tts_spec(cfg: dict, headers: dict | None = None) -> TtsSpec | None:
    if cfg.get("model_type") != "lingv3_tts":
        return None
    lc, ac, tc = cfg["llm_config"], cfg["audio_config"], cfg["tts_config"]
    fc, gc = cfg["flow_config"], cfg["aggregator_config"]
    db = ac["dec_kwargs"]["backbone"]
    eb = ac["enc_kwargs"]["backbone"]
    # AudioVAE encoder (Qwen2 backbone + 1-layer-type aggregator + fc1/fc2/fc3 + norm + cls), stored only
    eh, ehd = eb["hidden_size"], eb["hidden_size"] // eb["num_attention_heads"]
    ekv = eb["num_key_value_heads"] * ehd
    el = (eh * eh + eh) + 2 * (eh * ekv + ekv) + eh * eh + 3 * eh * eb["intermediate_size"] + 2 * eh
    lat = ac["enc_kwargs"]["latent_dim"]
    enc = (eb["num_hidden_layers"] * el + 2 * eh                      # encoder backbone (+ embed_tokens row, norm)
           + 4 * el + 2 * eh                                           # aggregator backbone: 4 layers (header)
           + eh + ac["enc_kwargs"]["input_dim"] * eh + (eh * eh + eh) + (eh * 2 * lat + 2 * lat) + 2 * eh)
    sd = tc.get("speaker_embedding_dim", 192)
    return TtsSpec(
        llm_h=lc["hidden_size"], z=tc["latent_dim"], patch=tc["patch_size"], history=tc["history_size"],
        flow_d=fc["hidden_size"], flow_depth=fc["depth"], flow_heads=fc["num_heads"], flow_mlp=float(fc["mlp_ratio"]),
        agg_d=gc["hidden_size"], agg_depth=gc["depth"], agg_heads=gc["num_heads"], agg_mlp=float(gc["mlp_ratio"]),
        dec_h=db["hidden_size"], dec_layers=db["num_hidden_layers"], dec_heads=db["num_attention_heads"],
        dec_kv=db["num_key_value_heads"], dec_inter=db["intermediate_size"], dec_window=db.get("sliding_window") or 0,
        up=ac["patch_size"], hop=ac["dec_kwargs"]["output_dim"], rate=tc.get("sample_rate", ac["sample_rate"]),
        enc_params=enc, spk_params=sd * lc["hidden_size"] + lc["hidden_size"])


def tts_head_ops(t: TtsSpec, b: int, layer: int, replicated: int = 1, af: str = "bf16") -> list[Op]:
    """Ops of one TTS step for ``b`` requests (after the LLM's final norm)."""
    ab = 2.0

    def gemm(name: str, m: int, k: int, n: int) -> Op:
        return Op("tts." + name, "gemm", layer, m=m, k=k, n=n, role="tts", w_params=k * n, w_bits=16.0, w_fmt="bf16",
                  a_fmt=af, act_bytes=m * (k + n) * ab, replicated=replicated, vec=m * n * VEC_PER_OUT)

    def attn(name: str, rows: int, keys: int, hd: int, cnt: int, **kw) -> list[Op]:
        return [Op(f"tts.{name}.qk", "attn", layer, m=rows, k=hd, n=keys, count=cnt, replicated=replicated,
                   act_bytes=cnt * (rows + keys) * hd * ab, **kw),
                Op(f"tts.{name}.pv", "attn", layer, m=rows, k=keys, n=hd, count=cnt, replicated=replicated,
                   act_bytes=cnt * (rows + keys) * hd * ab),
                Op(f"tts.{name}.softmax", "vector", layer, vec=cnt * rows * keys * 5.0, replicated=replicated)]

    def blocks(tag: str, rows: int, seq: int, d: int, heads: int, mlp: float, depth: int) -> list[Op]:
        f = int(mlp * d)
        out: list[Op] = []
        for i in range(depth):
            out += [gemm(f"{tag}.{i}.qkv", rows, d, 3 * d), gemm(f"{tag}.{i}.o", rows, d, d),
                    gemm(f"{tag}.{i}.fc1", rows, d, f), gemm(f"{tag}.{i}.fc2", rows, f, d),
                    Op(f"tts.{tag}.{i}.norm", "vector", layer, vec=rows * d * 8, replicated=replicated)]
            out += attn(f"{tag}.{i}", seq, seq, d // heads, (rows // seq) * heads)
        return out

    ops = [gemm("stop_head", b, t.llm_h, 2)]
    B = b * t.cfg_mult
    R = B * t.flow_rows
    d = t.flow_d
    for s in range(t.flow_steps):
        p = f"flow{s}"
        ops += [gemm(f"{p}.t_mlp0", B, 256, d), gemm(f"{p}.t_mlp2", B, d, d), gemm(f"{p}.c_emb", B, t.llm_h, d),
                gemm(f"{p}.x_emb", B * (t.history + t.patch), t.z, d)]
        ops += blocks(p, R, t.flow_rows, d, t.flow_heads, t.flow_mlp, t.flow_depth)
        ops += [gemm(f"{p}.final", R, d, t.z),
                Op(f"tts.{p}.euler", "vector", layer, vec=b * t.patch * t.z * 8, replicated=replicated)]
    ga = 1 + t.patch
    ops += [gemm("agg.x_emb", b * t.patch, t.z, t.agg_d)]
    ops += blocks("agg", b * ga, ga, t.agg_d, t.agg_heads, t.agg_mlp, t.agg_depth)
    ops += [gemm("agg.final", b * ga, t.agg_d, t.llm_h)]
    # streaming AudioVAE decoder: patch latents → up frames each
    fr = b * t.patch * t.up
    h, hd = t.dec_h, t.dec_h // t.dec_heads
    kvd = t.dec_kv * hd
    W = t.dec_window or 64
    ops += [gemm("vae.fc1", b * t.patch, t.z, h), Op("tts.vae.upsample", "vector", layer, vec=fr * h * 2,
                                                       replicated=replicated)]
    for i in range(t.dec_layers):
        ops += [gemm(f"vae.{i}.qkv", fr, h, h + 2 * kvd), gemm(f"vae.{i}.o", fr, h, h),
                gemm(f"vae.{i}.gate_up", fr, h, 2 * t.dec_inter), gemm(f"vae.{i}.down", fr, t.dec_inter, h),
                Op(f"tts.vae.{i}.norm", "vector", layer, vec=fr * h * 8, replicated=replicated)]
        # sliding-window causal attention over the cached frames: each new frame sees ≤ W keys
        ops += attn(f"vae.{i}", t.patch * t.up, W, hd, b * t.dec_heads,
                    kv_read=b * W * 2 * kvd * ab, kv_write=fr * 2 * kvd * ab)
    n = t.n_fft
    ops += [gemm("vae.istft_out", fr, h, n + 2),
            Op("tts.vae.irfft", "vector", layer, vec=fr * (5.0 * n * math.log2(n) + 4 * n), replicated=replicated)]
    return ops


def tts_step_flops(t: TtsSpec, b: int = 1) -> dict:
    ops = tts_head_ops(t, b, 0)
    f = lambda pre: sum(o.flops for o in ops if o.name.startswith(pre) and o.kind in ("gemm", "attn"))
    return {"flow": f("tts.flow"), "agg": f("tts.agg"), "vae": f("tts.vae"), "stop": f("tts.stop"),
            "total": sum(o.flops for o in ops if o.kind in ("gemm", "attn"))}
