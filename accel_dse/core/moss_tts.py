"""0.346 MOSS-TTS Delay (docs/MODEL.md §291).

Reference 「来源」: HF OpenMOSS-Team/MOSS-TTS (and v1.5) config.json + safetensors headers;
modeling_moss_tts.py MossTTSDelayModel; README 「1s ≈ 12.5 tokens」.

Architecture (gated on ``model_type == moss_tts_delay`` only — not Moshi / qwen3_tts / Nano):
  * language_model = Qwen3 dense (catalog LLM via language_config) bf16 as released.
  * emb_ext[n_vq]: audio codebook embeddings (audio_vocab+1 × h), summed onto text embeds.
  * lm_heads[0]: text logits (vocab); lm_heads[1..n_vq]: audio logits (audio_vocab+1) — all on the
    same last_hidden_state per step (no depth transformer).
  * Delay pattern: n_vq staggered audio channels; first real audio after ~n_vq blank delay steps.
  * External audio tokenizer / vocoder is NOT in this checkpoint — waveform decode unmodeled.

Per 80 ms frame (12.5 Hz): one backbone decode step + (1 + n_vq) logit heads. Real-time ⇔ frame ≤ 80 ms.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import Op

VEC_PER_OUT = 3.0
FRAME_HZ = 12.5          # 「来源」 README: 1s ≈ 12.5 tokens; sampling_rate 24000
FRAME_S = 1.0 / FRAME_HZ
SAMPLE_RATE = 24000      # 「来源」 config.sampling_rate


@dataclass(frozen=True)
class MossTtsSpec:
    n_vq: int
    audio_vocab: int          # codebook size (heads use audio_vocab+1 for pad)
    hidden: int
    text_vocab: int
    delay_frames: int         # 「来源」 delay pattern fills n_vq before first audio eos path
    sample_rate: int = SAMPLE_RATE
    # exact header counts when present
    audio_embed_params: int = 0    # emb_ext.*
    audio_head_params: int = 0     # lm_heads.1..n_vq (excl. text head 0)
    kind: str = "moss_tts"

    @property
    def frame_hz(self) -> float:
        return FRAME_HZ

    @property
    def frame_s(self) -> float:
        return FRAME_S

    @property
    def head_vocab_audio(self) -> int:
        return self.audio_vocab + 1

    @property
    def stored_bytes(self) -> float:
        """Audio embeds + audio lm_heads (bf16); backbone counted on the LM card."""
        ae = self.audio_embed_params or self.n_vq * self.head_vocab_audio * self.hidden
        ah = self.audio_head_params or self.n_vq * self.head_vocab_audio * self.hidden
        return (ae + ah) * 2.0


def moss_tts_spec(cfg: dict, rel: dict | None = None) -> MossTtsSpec | None:
    """Build from public HF config; gated on ``moss_tts_delay`` only (Nano / Local stay out)."""
    if cfg.get("model_type") != "moss_tts_delay":
        return None
    lc = cfg.get("language_config") or {}
    if not lc.get("hidden_size") or not lc.get("num_hidden_layers"):
        return None
    n_vq = int(cfg.get("n_vq") or 0)
    av = int(cfg.get("audio_vocab_size") or 0)
    if n_vq < 1 or av < 1:
        return None
    h = int(lc["hidden_size"])
    # Audio emb_ext size is exact from dims 「来源」; params_depth = audio lm_heads (see _moss_tts_role).
    # Do not use params_embed — that role also holds language_model.embed_tokens.
    ae = n_vq * (av + 1) * h
    ah = int((rel or {}).get("params_depth") or ae)
    return MossTtsSpec(
        n_vq=n_vq, audio_vocab=av, hidden=h,
        text_vocab=int(lc.get("vocab_size") or 0),
        delay_frames=n_vq,
        sample_rate=int(cfg.get("sampling_rate") or SAMPLE_RATE),
        audio_embed_params=ae, audio_head_params=ah,
    )


def _g(name: str, m: int, k: int, n: int, bits: float = 16.0, *, w: int | None = None, cnt: int = 1) -> Op:
    ab = bits / 8
    fmt = "bf16"
    return Op(name, "gemm", 0, m=m, k=k, n=n, count=cnt, role="speech",
              w_params=(k * n if w is None else w) * cnt, w_bits=bits, w_fmt=fmt, a_fmt=fmt,
              act_bytes=m * (k + n) * ab * cnt, vec=m * n * VEC_PER_OUT * cnt, stream=True)


def moss_audio_head_ops(s: MossTtsSpec, b: int) -> list[Op]:
    """One step's audio channel heads (lm_heads.1..n_vq) + emb_ext lookups for the prior audio codes.
    Text head (lm_heads.0) is already in the catalog backbone decode (evaluate)."""
    ops: list[Op] = [
        # sum of n_vq audio embeds onto the residual stream (lookup traffic) 「来源」 get_input_embeddings
        Op("moss.emb_ext", "lookup", 0,
           kv_read=b * s.n_vq * s.hidden * 2.0, vec=b * s.n_vq * s.hidden),
    ]
    # n_vq audio logit heads share last_hidden (one row) 「来源」 forward: same hs for every head
    ops.append(_g("moss.audio_heads", b, s.hidden, s.head_vocab_audio, 16,
                  w=s.n_vq * s.hidden * s.head_vocab_audio, cnt=s.n_vq))
    ops.append(Op("moss.audio_sample", "vector", 0, vec=b * s.n_vq * s.head_vocab_audio * 3))
    return ops


def moss_audio_weight_bytes(s: MossTtsSpec) -> float:
    """Audio lm_heads only (bf16). emb_ext is already in Model.lookup_params on the LM card."""
    ah = s.audio_head_params or s.n_vq * s.head_vocab_audio * s.hidden
    return ah * 2.0
