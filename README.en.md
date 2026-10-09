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
- **0.50**: prefill / decode disaggregation (`--pd`: separate prefill and decode pools with their own card counts and layouts, KV hand-off timed on the interconnect tier, TTFT / TPOT / goodput compared with colocated serving on the same cards; steady-state fluid model, first cut); three-tier interconnect — optional package die-to-die (off by default = monolithic die; when on, UCIe-A 48 GT/s by default, also UCIe-A 64G / UCIe-S 32G / BoW / NVLink-C2C, `--d2d --d2d-std`), in-node scale-up, cross-node network (`--node-cards --net-GBps`). Default results unchanged.
- **0.51**: PD queueing and tail latency (closed-form approximations, labelled 「假设」): at one Poisson arrival rate, TTFT p50 / p90 / p99, TPOT quantiles, worst inter-token gap and SLO goodput for PD, colocated prefill-first and colocated chunked prefill; continuous-batching decode via a Little's-law fixed point, KV vs pool-collective contention, PD energy (`--pd-load --pd-rate --pd-chunk`). Default results unchanged.
- **0.52**: PD request-length spread (`--pd-prompt-cv / --pd-out-cv` or a discrete `--pd-mix`: M/G/1 queueing, length-biased decode context, capacity over the mix), prefix-cache hit rate (`--pd-prefix-hit`: only the uncached part is prefilled, smaller KV hand-off), pool layout search (`--pd-search-layouts`); fixed the colocated chunked-prefill mean iteration time. All labelled 「假设」; default results unchanged.
- **0.53**: PD prefix-cache capacity + LRU eviction (`--pd-prefix-len / --pd-prefix-count / --pd-prefix-zipf`: Zipf working set, Che approximation; the hit rate follows from the free DRAM or `--pd-prefix-cache-GB`, `--pd-prefix-affinity` for prefix-aware routing; an explicit `--pd-prefix-hit` still overrides), heterogeneous pools (`--pd-prefill-chip / --pd-prefill-mem`: the prefill pool may use another chip / memory), decode batch in the layout search (`--pd-search-decode-batch`). All labelled 「假设」, off by default; default results unchanged.
- **0.54**: validation release. A request-level discrete-event simulator (DES) re-uses the same per-step costs to check the PD closed-form queueing model (V4 serving: 54-point grid, `accel-dse validate` / `scripts/v4_serving.py`; SLO-goodput error −8 % … 0 %, colocated TPOT tails optimistic; see MODEL.md §18.4). Fixed continuous-batching decode (birth–death instead of the Little fixed point), the prefill-first TPOT quantile and worst stall, and the TTFT quantile under a service-time mix (convolution); PD report numbers change accordingly. Optional `--pd-sim` attaches simulated tails. Default results unchanged.
- **0.55**: colocated tails and coverage. V4 is now multi-seed (30 points × 3 seeds, adding an MoE family, Qwen3-30B-A3B, and a TP4 family, Qwen3-32B); batched decode arrivals behind prefill stalls (batch-arrival chain + pair factor), a correlation-time TPOT window, chunked TTFT tails (quasi-static mixture + vacations) and the worst gap as the lifetime-maximum batch bring the colocated TPOT p90 median error from −14 … −18 % to about −4 % (MODEL.md §18.5). Optional decode KV-capacity policy `--pd-kv-policy wait|recompute` (default off). Default results unchanged.
- **0.60**: large card counts and a finer network model (「假设」, default results unchanged).
  - Cards per replica 64 → 8192 (API / UI / CLI), so multi-chip 1P–10P SKUs with hundreds to thousands of cards over many nodes can be evaluated. The batch cap scales with cards, and the PD search / queueing model is pruned and interpolated; a 1024-card layout ranking takes about 8 s.
  - Three-tier fat-tree (leaf / spine / core) with per-tier oversubscription, nodes per pod, and the share of traffic leaving the leaf / pod.
  - Concurrent physical ports per tier (`overlap = ports`), plus an exposed-time algorithm choice that accounts for compute overlap (`auto_overlap`).
  - Multi-hop PP hand-offs on ring / torus / full-mesh scale-up; PD KV contention can feed back into decode TPOT.
  - See MODEL.md §19.1–19.5.
- **0.59**: hardware side: cross-node network topology and collective algorithms (`fabric`, 「假设」, off by default). The cross-node tier supports fat-tree / leaf-spine with an oversubscription ratio and rail-optimized fabrics (PXN); the in-node scale-up supports switch / full mesh / ring / 2D torus. Each collective takes the cheapest of ring / tree (NCCL double binary tree) / hierarchical under α-β plus per-step latency, with optional in-network reduction (SHARP / NVLS-class vendor option). EP all-to-all over an oversubscribed fabric, PP hand-offs that cross leaves, and PD KV transfer contending with collectives are all costed by their traffic share. Checked by hand against the NCCL ring / tree formulas (MODEL.md §19). Default results unchanged.
- **0.58**: fixes 0.57's over-pessimistic colocated KV admission at CV1: the slot hold counts only prefill service, the colocated conditional wait is Gamma (c² follows service CV), and chunked preemption is scaled by (1 − ρ_pre). Colocated CV1 TTFT p90 goes from +23 / +40 % to +14 / +21 %, and chunked preemption count from +30…+60 % to −10…+4 %. New radix / partial prefix matching (`pd.prefix_tree`, 「假设」, off by default): a multi-level shared-prefix tree in a token-capacity LRU, with a per-node Che approximation, where partial hits save the matched tokens; validated against a RadixLRU DES (token hit within 0.004) (MODEL.md §18.8). Default results unchanged.
- **0.57**: KV-capacity model fixes. The admission wait is an M/G/c slot queue (Cosmetatos M/D/c blend; colocated requests also hold their slot during prefill); the saturation preemption rate accounts for the refill offset, and the restore share is iterated to a fixed point on the slowed slot chain. A DES bug is fixed: a sequence being restored did not count toward KV use, which caused most of the recompute "cascade" recorded in 0.56. Preemption-count error goes from +39…+190 % to −23…+60 %, and PD recompute CV1 TTFT p90 from −40 % to −21 % (MODEL.md §18.7). High-load V4 grid points use a 20000-request DES (`--n-high`). Default results unchanged.
- **0.56**: serving-model follow-ups. KV capacity is admitted in vLLM order (slot and KV reserved before prefill), so the admission wait counts toward TTFT and SLO goodput (`--pd-kv-admit`); preemptions enter the TPOT / worst-gap tails; new swap preemption (`--pd-kv-policy swap --pd-swap-GBps`, host link 50 GB/s by default, an assumption). Length bins go from 8 to 15 and the M/G/1 wait uses the exact Pollaczek–Khinchine law, so the worst TTFT p50 error under length CV drops from +47 % to ±18 % (MODEL.md §18.6). Strict API input typing (400 instead of 500). Per-pool static power in the PD energy (`--idle-W-prefill`). Default results unchanged.
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
