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
- **0.70.0**: validation round 4 — tbo reads weights per micro-batch, op-level check vs the DeepSeek decode trace, optional `Chip.split_instances` (MODEL.md §25.9).
- **0.69.0**: validation round 3 — continuous-batching proxy, MI300X / B200 reference hardware + 31 rows, optional per-layer overhead `layer_overhead_us` (MODEL.md §25.8).
- **0.68.0**: external validation round 2 — 34 published MoE / MLA / large-EP measurements, H800 reference hardware, `exec_overlap="tbo"` (MODEL.md §25.7).
- **0.67.0**: 0.66.0 + catalog coverage closures — all LLMs now complete (MODEL.md §2.1); previously complete models numerically unchanged.
- **0.66.0**: external validation — reference hardware (H100 / H200 / A100, datasheet peaks) against 100 published measurements (TensorRT-LLM, Databricks) with a held-out calibration study; optional `exec_overlap` intra-stage overlap modes (default unchanged). Finding: full intra-stage overlap is the dominant systematic error; no fitted efficiency factors (MODEL.md §25).
- **0.65.1**: V4 grid re-run with the 0.65 closed form and DES stability rule (resumable `--partial`); fixed atom truncation in the bulk-service laws (cap-2 TTFT tail was 19 % optimistic at length CV 1) (MODEL.md §24.8).
- **0.65.0**: exact greedy bulk-service prefill queue (closed-form PD SLO capacity within 3 % of the DES, prefill-first within 10 %; the fluid cap-2 model was +72 % optimistic), DES stability by drift / utilisation, SRAM / link / dequant counts of idle EP ranks, every integer output-stationary tile (MODEL.md §24).
- **0.64.0**: SLO-aware prefill batch cap (PD / prefill-first, with a stable-cap fallback), DES SLO search by scan + refine, 2-D GEMM blocking for activation spill, fractional (two-point) running batch in the closed form, weight dequant under uneven TP / EP and DRAM bytes of idle EP ranks (MODEL.md §23).
- **0.63.0**: external-review fixes (9 findings): per-head MLA decode absorption, video CFG dependency in the PP schedule, non-monotone SLO rate search, radix-cache path capacity, uneven TP without dropped heads, per-pair PP routes, work conservation for energy / TFLOP, stochastic DES speculative acceptance, thread-safe fabric log; new LLM prefill activation streaming / spill (MODEL.md §22).
- **0.62.1**: round-6 convergence audit, code and results unchanged: combinatorial fuzz (2400 scenarios) and PD identity fuzz found no new defects; stale MODEL.md numbers fixed (DeepSeek-V3 / Kimi-K2 fabric examples after the 0.61.1 bf16 combine, release count 72, ESMFold fp32 example).
- **0.62**: VLM vision encoders modelled per release (ViT + merger / projector; patch grid and image tokens from each release's image preprocessor; FLOPs match the transformers / released remote-code reference counts exactly): `serving.images` (default 0 = text only) and `image_w / image_h` (default 1024×1024, assumed); the encoder runs at prefill and image tokens join the prompt / KV; CLI `--images --image-size`. Pipeline stages are now split by cost by default (`pp_split = "cost"`: LM head / embedding / MTP and heterogeneous stacks included, candidates arbitrated by the full model, never slower than equal layer counts; `layers` keeps the old split): PP > 1 ticks drop 0.03 % – 50 % (structure models most). Also fixed: PD hand-off bytes with a cached prefix (sliding windows / recurrent state per layer kind), the DeepSeek-V4.1 aligner counted as vision, host-link energy of video offload reloads. See MODEL.md §19.14, §20, §21.
- **0.61.4**: round-4 audit fixes. Idle DP ranks no longer charged energy / TFLOP (e.g. ESMFold batch 1 · DP2 MAC per unit 194.65 → 97.32 TFLOP); PD no longer errors when the prefill layout does not divide the decode pool, and the budget checks the prefill pool; new `energy.pJ_bit_host` (swap host link, no default); mixed fp8 × int8 upcasts to bf16; Pareto extends past batch 512; warnings for context beyond the release limit and PP stage imbalance. See MODEL.md §19.13.
- **0.61.3**: round-3 audit fixes. Structure models checked against independent FLOP counts of the released code: AF2 / OpenFold template rows join the MSA (+0.4 %); Boltz-1 inference cache and checkpoint alias dedup (−11 % FLOPs, batch latency 13.05 → 7.94 s, 592.01 M params); Protenix pair bias per diffusion sample (+3.6 % FLOPs, 19.33 → 27.86 s); Open-Sora temporal VAE −1.6 %. Cold storage (embedding table etc.) no longer pinned in SRAM / SLC; PD energy counts preemption restores; budget scope note and a warning when no batch meets the SLO. See MODEL.md §19.12.
- **0.61.2**: round-2 audit fixes. The LTX-Video VAE decode was 2.5–3.3× low (up-block resnets now run at the block's output resolution, matching a diffusers FLOP count); ESMFold runs 4 trunk passes as the reference does (was 5, −20 % FLOPs); the TPOT-quantile correlation time no longer overflows at large occupancy; radix prefix-cache nodes that cannot fit take no capacity; the area budget decides on a lower bound; stale search / sweep results are flagged in the UI. See MODEL.md §19.11.
- **0.61.1**: round-1 audit fixes. DeepSeek-V4 compressed-layer decode read c× too little KV; prefill of windowed / top-k / compressed attention now charges the actual mean keys per query (DSA prefill −14 … −20 %), and the indexer is causal; sliding-window-only latent layers store only the window; linear attention reads/writes its conv state; fp8-activation releases send all-reduce / combine in bf16 (「假设」); serving sizes are bounded. See MODEL.md §19.10.
- **0.61**: performance and network detail (default results unchanged). Large-scale PD queueing reports 20–50 s → 3–4 s; layout comparison / stability run on the server's process pool (about 7 s at 1024 cards). Optional per-step latency for steps leaving the leaf / pod, NVLS-class all-gather / reduce-scatter, NCCL's second tree in the leaf cut, and LL / LL128 / Simple protocol efficiency (「假设」). See MODEL.md §19.6–19.9.
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
python3 -m accel_dse eval --model qwen3.8-27b --phase prefill --prompt 1024 \
    --images 2 --image-size 1280x720 --tp 2                   # VLM: 2 images per request, encoder + image tokens
python3 -m accel_dse eval --model esmfold --pp 4 --pp-split layers   # equal-layer PP split (default: cost-balanced)
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
