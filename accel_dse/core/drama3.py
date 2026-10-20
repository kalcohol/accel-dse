"""0.276 Fish Audio Drama 3 TTS (docs/MODEL.md §221.3).

Reference code 「来源」 github vllm-project/vllm-omni@da572cd vllm_omni/model_executor/models/fish_omni/
(fish_omni_model.py, codebook_tower.py, pipeline.py; stage_input_processors/fish_omni.py) + HF fishaudio/drama-3
config.json / safetensors headers (README: 35B total, 3B active, FP8 MoE experts, rest bf16).

One forward = one sampled audio frame per request (Stage 0, ``fish_omni``):
  1. the input row ``[main, c0..c9]`` → main-token embedding + the backbone codebook's embedding, scaled
     1/√(1 + vq_backbone_codebooks) → Qwen3.5-MoE backbone (40 layers: 30 Gated-DeltaNet + 10 gated full attention,
     256 experts top-8 + shared expert, FP8 experts);
  2. the main token is drawn from the lm_head rows of end-of-speech + semantic codes 1..4095 only (vllm-omni keeps
     those 4096 rows — "cuts the per-frame head read from 1 GB to 16 MB");
  3. the codebook-MTP tower: 13 layers (4 shared + 9 per codebook), each mixes one code embedding into the hidden
     state (eh_proj; layers ≥ 1 fold the embedding half into a per-code table at load) and runs one gated-attention
     transformer block (GQA 16 / 2, head 256, SwiGLU 6144) whose causal attention covers the whole sequence — a paged
     KV cache with one row per backbone position; code k is drawn from codebook head k (9 × Linear 2048 → 4096).
  → modelled as 40 + 13 decoder layers on every row (prefill and decode) with the 9 heads as per-sequence rows
     「推导」: same FLOPs, weights and KV bytes; the serial sample between tower layers is not a separate latency 「近似」.
Stage 1 (``dac_decoder``, the Fish DAC of S2 Pro, 44.1 kHz, 2048 samples per frame): codec.pth (1.87 GB) is a torch
pickle not read here → not costed; streamed in chunks ramping 6, 8, 12, 16, 24, 32 then 64 frames with 32 frames of
left context 「来源」.  First audio = backbone prefill + 6 decode steps (+ the DAC's first chunk, not costed).
"""
from __future__ import annotations

from dataclasses import dataclass

CODEC_RATE = 44100      # fish_omni_model._CODEC_SAMPLE_RATE 「来源」
CODEC_HOP = 2048        # fish_omni_model._CODEC_HOP 「来源」
CHUNK_RAMP = (6, 8, 12, 16, 24, 32)     # stage_input_processors/fish_omni.DEFAULT_RAMP 「来源」
CHUNK = 64              # DEFAULT_CHUNK
LEFT_CONTEXT = 32       # DEFAULT_LEFT_CONTEXT
REF_S = 10.0            # voice-reference length (one prompt row per reference frame) 「假设」


@dataclass(frozen=True)
class FishSpec:
    n_codebooks: int = 10
    codebook_size: int = 4096
    shared_layers: int = 4
    tower_layers: int = 13
    dim: int = 2048
    n_head: int = 16
    n_kv: int = 2
    head_dim: int = 256
    inter: int = 6144
    candidates: int = 4096
    codec_params: int = 0       # not costed (codec.pth unread)
    kind: str = "fish"

    @property
    def frame_s(self) -> float:
        return CODEC_HOP / CODEC_RATE

    @property
    def frames_per_s(self) -> float:
        return CODEC_RATE / CODEC_HOP

    def ref_rows(self, ref_s: float) -> int:
        return max(0, -(-int(round(ref_s * CODEC_RATE)) // CODEC_HOP))


def is_fish(cfg: dict) -> bool:
    return (cfg or {}).get("model_type") == "fish_omni" and isinstance((cfg or {}).get("audio_decoder_config"), dict)


def n_candidates(cfg: dict) -> int:
    """fish_omni_model.main_candidate_ids: end of speech + semantic codes 1..end 「来源」."""
    return 1 + cfg["semantic_end_token_id"] - cfg["semantic_start_token_id"]


def fish_spec(cfg: dict) -> FishSpec | None:
    if not is_fish(cfg):
        return None
    a = cfg["audio_decoder_config"]
    q = a["num_codebooks"]
    return FishSpec(q, a["vocab_size"], a["shared_enrichment_layers"], a["shared_enrichment_layers"] + q - 1,
                    a["dim"], a["n_head"], a["n_local_heads"], a["head_dim"], a["intermediate_size"], n_candidates(cfg))


def tower_layers(fs: FishSpec, Layer, Linear, gqa, ffn_dense) -> list:
    """The 13 codebook-tower layers as decoder layers (codebook_tower.TowerLayer): eh_proj (layer 0: [e; h] 2d → d;
    layers ≥ 1: the h half d → d after the code-table fold 「来源」), qkv with the sigmoid output gate (q ‖ gate), o,
    SwiGLU; norms enorm / hnorm / input / q / k / post / final as misc."""
    d = fs.dim
    out = []
    for i in range(fs.tower_layers):
        lin, core = gqa(d, fs.n_head, fs.n_kv, fs.head_dim, out_gate=True)
        eh = Linear("eh_proj", (2 if i == 0 else 1) * d, d, "rep")
        ffn, fl = ffn_dense(d, fs.inter)
        out.append(Layer((eh,) + tuple(lin), core, ffn, fl, misc_params=5 * d + 2 * fs.head_dim))
    return out
