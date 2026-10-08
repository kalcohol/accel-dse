# accel-dse

> Analytical design-space exploration for inference accelerators (LLM · video-generation DiT · protein)

[中文](README.md) · [Model notes (Chinese)](docs/MODEL.md) · [Changelog](CHANGELOG.md) · [MIT](LICENSE)

**accel-dse** answers pre-silicon questions with a checkable analytical model: given a model (an LLM / VLM, or a video-generation DiT or protein language model; weights and dtypes as officially released), a datapath mapping, on-chip SRAM, a memory system (HBM / LPDDR) and a card count, which parallel layout and batch are best, what binds, and how sensitive the answer is to the assumptions. No third-party dependencies, Python ≥ 3.10.

## What it does

- **Models as released**: per-role parameter counts and storage dtypes (bf16 / block fp8 / MXFP4 / AWQ int4 …) from the HF `config.json` and safetensors headers (no weight download), within 0.5% of the released totals; official quantised releases are separate entries. Every model carries a three-axis label: provenance (official / mirror) × coverage (full / partial / architecture proxy, with a per-model list of what is approximated) × dtype; the catalogue is grouped vendor → family (a vendor's LLM, VLM, video and protein releases together).
- **Video generation and protein (0.41 – 0.43)**: the DiT denoiser of Wan2.1-T2V (1.3B / 14B) and CogVideoX (2b / 5b) — full 3D attention, 1–2 forwards per step with CFG — and the ESM-2 (650M / 3B) encoder are evaluable: clip latency, per-frame latency, frames/s/card, or batch latency, sequences/s/card, residues/s/card; Ulysses sequence parallelism (SP) and CFG parallelism (DP) are layout dimensions; text encoder and VAE were not modelled before 0.44. 0.42 adds Wan2.2-T2V-A14B (two experts, one active per step), HunyuanVideo (dual / single-stream, full 3D attention), LTX-Video 2B, Mochi 1, Open-Sora STDiT3 (factorized spatial / temporal attention, as released) and MiniMax-H3 (video + audio + text in one sequence). 0.43 adds protein structure prediction: ESMFold, AlphaFold 2 (official JAX parameters, model_1_ptm), OpenFold, Boltz-1 and Protenix — pair representation, triangle multiplication / attention, MSA row / column attention, outer-product mean, IPA and the diffusion module modelled from the checkpoint tensor shapes, with recycles / diffusion steps / samples adjustable (MSA / template search not modelled → coverage "partial"; ESMFold "full"). AlphaFold 3 weights are gated and stay "not yet on v2".
- **Whole video pipeline (0.44)**: text encoders (umT5 / T5-XXL / LLaVA-Llama-3 + CLIP / Qwen3-VL text tower) and VAE decoders (incl. HunyuanVideo tiling, Open-Sora per-frame SD-VAE, the H3 ViT decoder and audio VAE) are op graphs built from the released checkpoint tensor headers; their time and storage are counted by default (serial with the denoise loop, assumption — docs/MODEL.md §11.4) and all video models are coverage "full"; `workload.pipeline = false` / CLI `--dit-only` evaluates the DiT alone. New activation-dtype what-if (web 「激活 dtype」 / CLI `--act fp32`) prices the fp32 activations of the structure-model reference implementations.
- **Multi-card structure models and component placement (0.45)**: structure models support DAP (FastFold dynamic axial parallelism, on the layout's SP axis): pair / MSA / template grids split along a residue axis, with the all-gathers of triangle multiplication / pair bias / outer-product mean and the row ↔ column all-to-alls counted per kernel (AF2 7.8× on 8 cards; diffusion-heavy Protenix 1.8×). Video component placement `workload.placement`: resident / FSDP-sharded text encoder (Wan `--t5_fsdp`) / sequential offload (diffusers `enable_model_cpu_offload`, Wan `--offload_model`), default auto = the first that fits — Wan2.1-14B, Wan2.2-A14B and MiniMax-H3 now fit 64 GiB LPDDR (1–2.5 s host reload per request). Optional VAE tiled decode (`--vae-tiling`, CogVideoX / Mochi with the diffusers default tiles). Request TFLOP is now the useful FLOPs of the whole replica (was one rank on multi-card layouts).
- **0.46**: diffusion samples split over the DAP cards (Protenix, 5 samples: 8 cards 1.8× → 6.0×; exact, no extra communication); DiT weight FSDP (Wan `--dit_fsdp`, `workload.dit_fsdp` / CLI `--dit-fsdp`); text encoder on the host CPU (Wan `--t5_cpu`, `--te-cpu --host-TFLOPS`, host throughput is an assumption); optional Wan VAE tiling (diffusers parameters).
- **0.47**: multi-card tile-parallel VAE decode (MiniMax-H3 release's `parallel_tiling`, `--vae-parallel`; H3 SP8 decode 24.4 → 3.5 s); optional LTX VAE tiling; MiniMax-H3 video decode corrected to the released always-tiled path (256 px tiles); cross-request overlap (host encoder ∥ denoise, `--overlap`). Open-Sora tiling and structure-model pair TP have no reference implementation and are skipped.
- **0.47.1**: energy action counts (MAC / vector / SRAM / DRAM / link / card·s per token / frame / sequence) × a user-supplied energy-per-action table (`--pJ-mac … --idle-W`, API `energy`); no energy numbers ship with the tool.
- **0.48**: system-level cache (`--slc-mib`, pin / lru policies, off by default) and a two-tier interconnect (die-to-die inside a package `--package-cards --d2d-GBps` vs the cross-package network tier; collectives are tiered by the levels a group spans); energy gains SLC / D2D actions. Default results identical to 0.47.1.
- **0.49**: MoE expert-load skew (`--moe-skew` or a measured per-expert distribution `--moe-expert-load`; the busiest EP rank sets the time, energy counts conserved); resource / area budgets (SRAM / MAC / cards / power / area-proxy limits with user-entered densities; headroom reported, over-budget layouts flagged). Default results unchanged.
- **Mapping is a design variable**: output-stationary, weight-stationary (edge load / broadside load), OS + GEMV unit, reconfigurable; each op gets a MAC bound and an SRAM-feed bound; formats the chip does not support natively pay dequantisation on the vector unit.
- **Per-rank op graph**: TP / PP / attention DP / EP / ETP; a single card is the all-ones layout, there is no second code path.
- **Memory plan and schedule**: SRAM residency of weights / KV, staging, per-stage capacity; each stage is `max(MAC/FEED, VECTOR, DRAM, LINK) + SYNC`, with the binding term and the useful-MAC fraction reported.
- **Exact search**: the largest batch per layout under the TPOT SLO (branch and bound, checked against brute force), decode or prefill-inclusive goodput objective, a TTFT flag for DP prefill, and ranking stability under perturbed assumptions.
- **Validation**: parameters vs releases, trend bands for an H100-like configuration, comparison with GenZ — see the [model notes](docs/MODEL.md).

## Quick start

```bash
git clone https://github.com/kalcohol/accel-dse.git
cd accel-dse
python3 tests/run_tests.py          # self-test (zero deps; or python3 -m pytest tests/ -q)
python3 -m accel_dse serve          # web workbench → http://127.0.0.1:8765
```

Command line:

```bash
python3 -m accel_dse models                                   # catalogue with three-axis labels
python3 -m accel_dse eval --model qwen3-8b --best-batch       # one scenario (default 100T chip + LPDDR5X)
python3 -m accel_dse compare --model deepseek-v3 --cards 8 \
    --mem hbm3e_8s_12h24g_9200 --ctx 4096                     # best layout per mapping
python3 -m accel_dse search --model qwen3-32b --cards 8 --objective goodput --mem hbm3e_8s_12h24g_9200
python3 -m accel_dse stability --model qwen3-32b --cards 8 --mem hbm3e_8s_12h24g_9200
python3 -m accel_dse eval --model wan2.1-1.3b --steps 20 --sp 2   # video: clip / per-frame latency (480P, 81 frames)
python3 -m accel_dse eval --model esm2-650m --seq-len 1022    # protein: batch latency, sequences/s
python3 -m accel_dse eval --model boltz-1 --msa 1024 --samples 5   # structure prediction: trunk / diffusion / confidence, TFLOP per sequence
python3 -m accel_dse validate                                 # trend bands + GenZ comparison
```

Every subcommand accepts `--json`. The HTTP API (`/api/eval`, `/api/compare`, `/api/layouts`, `/api/stability`, `/api/sweep`, `/api/pareto`, `/api/memory`) takes the same scenario description and parses it strictly (unknown fields and NaN / Infinity are rejected).

## Layout

```
accel-dse/
├── accel_dse/
│   ├── core/          # scenario, model spec, op graph, mapping, memory plan, schedule, search, stability, validation
│   ├── api.py         # JSON API shared by the web UI and the CLI
│   ├── serve.py web/  # local web workbench
│   ├── cli.py
│   ├── mem_catalog.py # structured HBM / LPDDR catalogue with provenance tags
│   └── data/          # release summaries (releases/), model catalogue, GenZ reference
├── docs/MODEL.md      # method, formulas, validation and scope
├── docs/research/     # memory spec research and sources
├── scripts/           # release summary fetcher, GenZ reference generator
└── tests/
```

## Disclaimer

Hardware parameters (frequency, array geometry, SRAM port, DRAM efficiency, link bandwidth and sync latency, MAC efficiency) are adjustable inputs labelled as assumptions, not calibrated against silicon; use the results for relative comparison and trends, not as performance claims. Model parameters come from public releases and are not vendor performance claims.

## Feedback

Please report problems or modelling errors via [GitHub Issues](https://github.com/kalcohol/accel-dse/issues), with the command line or API request body to reproduce.

## License

[MIT](LICENSE) © 2026 accel-dse Project
