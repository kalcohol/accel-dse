# Changelog

本项目的重要变更记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)（1.0 之前次版本号可能包含不兼容变更）。
0.31.0 及更早版本以 `npu-inference-dse`（包名 `npu_dse`）发布。

## [0.158.0] — sizing: MAC cores + DRAM bandwidth; parallel test runner
- `sizing_report` adds `cores` (MAC array via engine count, R×C per engine and vector width held; area = ΔkMAC × `mm2_per_kmac`, from a density preset or user input only) and `dram_bw` (effective GB/s via `mem_eff`, same stack and capacity; no PHY area because PHYs come in discrete channels). SIZING setters now act on the Scenario. Default chip: LLM set (qwen3-8b+gpt-oss-20b) has no headroom on cores (TTFT is compute-bound) or bandwidth (TPOT is bandwidth-bound); esm2-650m cores 16→10 (12.9 mm² saved with n7_tpuv4i); wan2.1 bandwidth → 65 %.
- `scripts/run_tests.py`: stdlib parallel runner (one pytest per file, longest-first from `scripts/test_durations.json`, per-file timeout reported as TIMEOUT). Runs the same files as serial pytest. Full suite: 818 passed in 415 s wall on 8 cores (longest-first); serial run is ≈ 3490 s total. The 0.157 "stalled shard" was not a hang: that shard held 049+057+061_3 (≈ 14 min alone) while 8 shards competed for 8 cores. Main hotspot: test_core_057::test_refill_offset_and_cascade (296 s) — `queue_report` for recompute at load 0.85 takes ≈ 318 s (SLO-rate scans × cap selection × KV wrap); left for a separate bit-identical optimisation.
- No default changed; fingerprints unchanged. Tests: tests/test_core_158.py (5); test_core_157 helper updated to the Scenario-level setter.

## [0.157.0] — datapath sizing headroom
- Added `insight.sizing_report` + `POST /api/insight_sizing` + UI「缩减余量」table: per resource (dequant elems/cycle, vector lanes, SRAM MiB) the smallest value at which every model in the set keeps every domain metric within ±1 % (TIE) and still fits. Halving then 3-step geometric bisection; floor = current/1024; reported min is always a probed pass; `limited_by` names the model/metric that fails just below; SRAM gets `mm2_saved` when `mm2_per_mib_sram` is set.
- Default chip findings: LLM (qwen3-8b+gpt-oss-20b) dequant→21 %, lanes→46 %, SRAM→46 %; wan2.1 dequant→9 %, lanes→59 %, SRAM→25 %; esm2-650m dequant→3 %, lanes→16 %, SRAM→65 %.
- No default changed (sizing is advisory) → no fingerprint re-baseline. Tests: tests/test_core_157.py (3).

## [0.156.0]
### Added (standing invariants; no engine change, defaults / fingerprints unchanged)
- `val5/v0156/dpfuzz.py <repo> <seed> <N> [dp|res|dom|hard]`: randomized datapath / residency / non-LLM multi-card suite.
  - **dp** (quantized releases: AWQ, FP8, MXFP4, MLA FP8): dequant throughput ↑ never slower and ↓ never faster; vector lanes ×2 never slower; storage independent of compute knobs.
  - **res**: SRAM ×2 never slower, pinned bytes never fewer, weight reads never more; batch ↑ keeps stored weights, never pins more (staging grows), and weight reads per sequence never rise.
  - **dom** (video / image DiT, protein, genomics): TP / SP conserve total FLOPs; per-card stored weights never grow with cards; DP keeps protein / genomics latency and never raises diffusion latency (DP = CFG cond / uncond split); fits never flips off.
  - **hard**: forced mappings, SLC, DiT FSDP, SP and PP mixes; SRAM, lanes, frequency, engines, SLC bandwidth and DRAM package count must stay monotone.
- Results: 330 randomized cases → 0 violations. Two first-run flags were harness errors, not engine bugs: diffusion DP splits CFG branches, and weight reads per step rise with batch (more MoE experts touched; staging shrinks pinning), so the check is per sequence.
- `tests/test_core_156.py` (3) deterministic guards; the dequant check also asserts the path is exercised (1 elem/cycle ≫ slower).

## [0.155.0]
### Added (standing invariants; no engine change)
- `val5/v0155/ppfab.py <repo> <seed> <N> [pp|fab|edge]`: randomized PP / fabric invariant suite (12 LLMs incl. MoE, hybrid, MLA, tied-embedding; 3 memory types).
  - **pp**: contiguous layer partition; stored weights conserved (at most one extra tied-embedding table when the head lands on another stage, as memplan intends); KV conserved; max-stage DRAM need ≤ PP1; decode weight reads only drop, by at most the added SRAM; fits never flips off.
  - **fab**: auto ≤ every explicit algorithm on stage totals; link bandwidth ×2 never slower; α ×2 never faster; TP1 ⇒ zero link time.
  - **edge**: oversub ↑ never faster; channel cap ↑ never slower and never faster than uncapped; in-network reduce under auto never slower; PP-microbatch weight reads per tick ≥ PP1 − pipeline SRAM.
- Results: pp+fab 240 cases, edge 120 cases → 0 violations after three invariant-definition corrections (no mechanism bug): tied-head copy, per-stage SRAM pinning, α counted in `t_sync` not `t_link`.
- `tests/test_core_155.py` (2): deterministic guards for the same invariants.

## [0.154.0]
### Added
- **Optional TCO inputs** `Budget.card_usd`, `Budget.amort_years`, `Budget.pue` (validated: PUE ≥ 1, years > 0).
  - Insight reports `tco` per model ($/unit = CapEx `card_usd / (amort_years · thr/card)` + OpEx `J · PUE · $/kWh`) and knob `tco_gain` / `rank_tco`.
  - Knobs are assumed not to change the card price (noted; hardware cost stays in Δmm² / upgrade path). Missing $/kWh → CapEx only; missing PUE → IT-only OpEx (noted).
  - No silent defaults. `TCO_PRESETS` holds PUE only: Uptime 2024 industry average 1.56 and 2023 capacity-weighted 1.47 (「来源」). No published card price / amortisation figure adopted.
  - UI: 卡价 / 摊销 / PUE inputs, PUE 预设, insight 「每单位 TCO」 line.
### Checked (catalog hygiene, no fixes needed)
- 53 catalog repos resolve (3 GitHub-only `gh:` repos OK); 189 cited URLs incl. split-string constants resolve; 92 cached release configs match live HF architecture fields (12 diffusers repos have no root config, expected); coverage label vs reasons / param gap: 0 mismatches. Scripts in `val5/v0154/`.
### Tests
- `tests/test_core_154.py` (3); Playwright `val5/v0154/ui.py`: insight + beam path with energy + electricity + PUE presets and TCO, llm and gen.

## [0.153.0]
### Added
- **Absolute electricity cost for insight**: `Budget.usd_per_kWh` and `Budget.tdp_W_card`.
  - When `$/kWh` is set and J per unit is known, knobs get `d_usd` / `gain_per_usd` / `rank_per_usd` (Δ$ from ΔJ).
  - J source: energy table first; else constant-TDP × time 「假设」 (`tdp_W × tpot` for LLM, `tdp_W / thr` otherwise).
  - **No silent electricity default.** Optional labelled presets `ELECTRICITY_PRESETS`:
    - `eia_us_industrial_2024` / `eia_us_commercial_2024` (EIA 2024, 「来源」): 0.0813 / 0.1275 $/kWh.
    - `cn_36city_industrial_2024_09` (chinaprice.cn 0.62 CNY/kWh ÷ 7.1 FX, 「二手/假设」).
  - UI: 电价预设 / `$/kWh` / `TDP W/卡`; insight paints 「每美元」.
### Documented
- **Step-5 vision** remains `partial` after reviewing StepFun open weights + platform docs: overview 728→169 tokens;
  optional external 504² crops (81 tokens each); crop-count algorithm not published. Approx string updated.
- **AGENTS §73**: crisp product-default backlog table (frequency, SRAM, dtype, C2C/tp/pp, KV fabric, EP/MLA, …) with
  current tool values; no engine defaults changed.
### Tests
- `tests/test_core_153.py` (4).

## [0.152.0]
### Added
- **Cited energy presets** `energy.ENERGY_PRESETS`: `horowitz_45nm` and `tpuv4i_7nm`, analogous to density presets.
  - Every value carries a status (「来源」 / 「推导」 / 「假设」) and a source string.
  - Use via `energy.energy_preset` or the UI select 「能耗预设」. Fills only unset fields; user values (incl. 0) win.
  - `idle_W` / `idle_W_prefill` are never filled (static power is a product choice).
  - Horowitz ISSCC 2014 Fig 1.1.9 (45 nm): MAC / vec / SRAM / DRAM 「来源」; UCIe 0.25 pJ/bit and NVLink-C2C 1.3 pJ/bit 「来源」; net / host 「假设」.
  - TPUv4i ISCA 2021 Table 2 scales Horowitz → 7 nm (logic 2.6×, SRAM geom. mean of 1.3–2.4×, DRAM 6.3×) as 「推导」.
- **Insight `rank_per_J`**: absolute ΔJ accumulated across models (`hi.J − lo.J`); win-win (faster and ΔJ ≤ 0) sorts first as 1e30; `|d_J| < 1e-12` → 0. UI paints a 「每焦耳」 line when a preset or user table is set.
### Tests / review
- `tests/test_core_152.py` (5); `val5/v0152/hostile.py`: multi-domain insight/path randomized invariants (gen / protein / genomics / llm × 40), 0 violations.
- Playwright: energy preset fill + gen 「每焦耳」 + protein domain metrics + path.

## [0.151.0]
### Added
- **Insight / upgrade path multi-domain**: model sets may be llm, gen (video/image), protein or genomics. Metrics follow the domain — LLM keeps TTFT / TPOT / tok/s/card; gen uses clip/image latency and frames/s/card; protein uses batch latency and sequences/s/card; genomics uses batch latency and windows/s/card. A set must share one domain (mixed sets return an ApiError). Default model lists per domain are in `DEFAULT_MODELS_BY_DOMAIN`.
- **CSV / JSON export** of the insight ranking and the upgrade path, using the existing export envelope (`tool` / `schema` / `kind` / `columns` / `rows`; schema stays 1). UI buttons on the 设计洞察 tab.
- README.md / README.en.md: short 「设计洞察」 / Design insight usage blurb (no version bullets); HTTP API list includes `/api/insight` and `/api/insight_path`.

## [0.150.0]
### Fixed (hostile review of 0.146–0.149)
- `upgrade_path`: `max_cost` was ignored in area mode, so a path could run past the cap. It now applies in both modes, in the reported cost unit (mm² in area mode), for greedy and beam.
- `phy_area`: card-to-card link PHY was counted even when all cards sit in one package. Those cards talk over D2D, so the PHY was double counted with D2D. The C2C term now applies only when cards > dies per package. The `c2c_x2` insight knob is likewise offered only when the link is actually used.
- UI 「密度预设」:
  - Picking a preset overwrote densities the user had typed. Now it fills only empty fields and ones it filled earlier itself, and lists the user values it kept.
  - Choosing 「不用预设」 now clears the values the preset filled.
### Added
- Notes in the insight and upgrade-path responses (shown in the UI):
  - `pd.enabled` is evaluated colocated, not as PD.
  - `mm2_fixed` must exclude any PHY counted separately.
  - Scale-out NIC PHY is not in the proxy.
- Preset source notes: n28 kMAC is an int8 MAC (optimistic for bf16), and TPUv4i's "128 MB" is taken as MiB (≤ 5 % difference).
- Tests: test_core_150 (7).
- `val5/v0150/hostile.py`: randomized invariants for preset precedence, PHY additivity, no double terms, determinism, beam ≥ greedy, never worse than start, cost and area caps, and Δmm² signs.
- sysfuzz `area` family: every domain plus PD bases, additivity, monotonicity, budget-report consistency. On the old tree it reproduced the C2C bug (20 violations); on the fixed tree it finds 0.

## [0.149.0]
### Added
- **Cited per-node density presets** `budget.DENSITY_PRESETS`: n28_tpuv1, n7_tpuv4i and n5_derived.
  - Every value carries a status (「来源」 / 「推导」 / 「二手」 / 「假设」) and a source.
  - Use them via `budget.density_preset` or the UI select 「密度预设」. They fill only unset fields; user entries win.
  - SRAM and MAC density come from the TPUv1 (ISCA 2017) and TPUv4i (ISCA 2021) floorplans, as upper bounds. N5 is derived via the TSMC bitcell ratio (0.021/0.027) and the 1.8× logic-density claim.
  - UCIe-A module PHY 388.8 × 1043 µm comes from the Hot Chips 2023 UCIe tutorial.
  - 112G SerDes area comes from VLSI 2022 (secondary report). The HBM PHY figure is 「二手/假设」 and the LPDDR PHY figure is 「假设」.
- **Interface PHY area**: new budget fields `mm2_per_dq_hbm`, `mm2_per_dq_lpddr`, `mm2_per_d2d_unit` and `mm2_per_GBps_c2c`, plus `full_area` / `phy_area`.
  - They are counted in the budget area check and in insight Δmm² for DRAM / D2D / C2C knobs. This resolves the 0.146 open item on PHY area.
  - With no PHY density set, results are unchanged.
- **Beam search** for the upgrade path (`beam` 1..5; UI checkbox for width 3). Greedy is always computed too, and the beam result is returned only if its final gain is ≥ greedy's.
  - In a random check of 48 cases, beam was better in 11 and never worse.
### Fixed
- Insight knob `d2d_x2` was a no-op for catalog D2D grades, where bandwidth = units × per-unit. It now doubles `d2d_units`.

## [0.148.0]
### Added
- **Insight upgrade path**: `core.insight.upgrade_path`, `POST /api/insight_path` (`models`, `metric` ttft|tpot|thr, `steps` 1..10, `max_cost`) and a UI table 「升级路径」 in the 设计洞察 tab.
  - Greedy multi-step: each step re-runs every additive knob on the *current* config, so knob interactions are re-evaluated. Knobs include mixed-precision MAC (W4A8 / W4A16 / W8A16) and dequant ×4.
  - It takes the largest ln(geomean gain)/cost.
  - Cost is Δmm² from the user's budget densities when they are given; knobs with no die area are converted at 1 unit = 1 % of the start area 「假设」. The cap is `budget.die_mm2`.
  - Otherwise cost is the labelled relative-cost proxy `REL_COST` 「假设」; the SRAM / MAC knobs scale with their current size, and the cap is `max_cost`.
  - A step is accepted only if the geomean improves by > 1 %, no model is worse than at the start, a model that fit still fits, and the budget holds. So the path is never worse than the start, and the (cum cost, cum gain) chain is monotone.
- Tests: test_core_148 (7). A random path fuzz found 0 worse results.

## [0.147.0]
### Added
- **Mixed-precision native MAC as a design variable**: `FormatSupport.mixed` holds (W, A, rate) pairs. `gemm_exec` runs a listed pair natively (exec label `w*a`, e.g. `mxfp4*fp8`) with no weight dequant. An unlisted pair still dequantizes W to A (or both operands upcast to bf16), and that cost is still charged.
  - Presets `dtypes.MIXED_PRESETS`:
    - W4A8: mxfp4 / nvfp4 × fp8 and int4 × int8 at the FP8 rate. Sourced: PTX ISA 8.7 `tcgen05.mma kind::f8f6f4`.
    - W4A16 and W8A16 at the bf16 rate, labelled 「假设」 (the gain is only the removed dequant).
  - Validation: format names must be known, the rate must be finite and > 0, and malformed entries return an ApiError.
- Insight knobs `mixed_w4a8` / `mixed_w4a16` / `mixed_w8a16`. When the chip already has the pairs, the knob is inverted and reports their value.
- The catalog lists `chips[].mixed` and `mixed_presets`. The chip panel's dtype line shows the mixed pairs, with toggles W4A8 / W4A16 / W8A16.
### Unchanged
- Default chips have no mixed pairs. The `omit_default` field keeps scenario hashes unchanged (default 723894c17df5d7bb).

## [0.146.0]
### Added
- **NPU design-insight report**: `core/insight.py`, `POST /api/insight` and the UI tab 「设计洞察」.
  - For a chip config and a model set (default: qwen3-8b, gpt-oss-120b, mimo-v2.6-flash, step-5-preview, deepseek-v3.2, ling-3.0-flash-vl, i.e. sparse / indexer attention plus MXFP4 experts), it ranks hardware knobs by TTFT / TPOT / decode-throughput speed-up (geomean over models).
  - Knobs: SRAM, SLC, MAC rows, cores, array shape, DRAM rate and package count, native FP8 / MXFP4, dequant throughput, D2D, C2C.
  - Each knob is a one-variable probe of the existing `evaluate`.
  - Area per mm² uses only the user's budget densities. The power proxy uses the user's energy table, otherwise action counts per token.
  - New labelled assumption: native MXFP4 MAC = 2× FP8 rate 「假设」.
  - No model / engine change; fingerprints unchanged.

## [0.145.0]
### Added
- **MiMo-V2.6-Pro** (`XiaomiMiMo/MiMo-V2.6-Pro-RL`, https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL), `mimo_v2` builder:
  - 1,021,248,126,336 params + MTP ×3 (1,934,715,264), exact;
  - vision 1,033,761,792, exact;
  - MXFP4 experts, FP8 attention.
- **Ling-3.0-flash-VL** (`inclusionAI/Ling-3.0-flash-VL`, https://huggingface.co/inclusionAI/Ling-3.0-flash-VL), released 2026-09-04:
  - Verified to be a different architecture from Ling-3.1-flash: h 2560, 42 layers (7 MLA + 35 KDA), MLA without q-LoRA, 512 × 768 experts, 2 dense layers, no MTP.
  - 124,414,211,552 params, exact.
  - Vision: Qwen3-VL ViT + `linear_proj`, 434,248,944, exact.
- Catalog: 90 → 92 releases.
### Changed
- MiniCPM-V-4.7 image slicing follows the transformers `MiniCPMV4_6ImageProcessor`:
  - overview + up to 9 slices, patches / 16 tokens per view, plus the special tokens;
  - 1024×768 is now 329 tokens (was 64).
  - Coverage partial → full.
- MiMo ViT banded windows (±64 over the patch sequence, sink bias, 4 full blocks) are now modelled. Coverage of mimo-v2.6-flash goes from partial to full.
- `summarize_release`: `^linear_proj.` is classified as vision.
### Fixed
- Step-5 indexer cache writes are now one fp32 + fp8 summary per completed region; they were idx_dim bytes per token, 8/5× too many. Stored indexer bytes now count completed regions only.
  - Found by the 0.145 hostile review (randomized write = store invariants).
- Fingerprints unchanged.

## [0.144.0]
### Added
- Catalog refresh for the NPU product purpose (official Aug–Oct 2026 open-weight releases, public `config.json` + safetensors headers, as-released dtypes, exact parameter counts). Catalog 85 → 90 releases (72 → 77 listed).
  - **Step-5-Preview** (`stepfun-ai/Step-5-Preview`, https://huggingface.co/stepfun-ai/Step-5-Preview): new `step5` builder following the official `inference/model.py` + `kernel.py`. 69 SWA-512 layers and 23 DSA full layers: the indexer (16 × 256) scores a cached fp8 summary per completed 8-token region, and each KV group reads its top-512 regions plus the current one (≤ 4104 keys). Head-wise sigmoid gate, 352 experts top-8 + shared, MTP ×3. FP8 experts (bf16 in layers 88–90), FP8 dynamic activations. 598.48 B / 27.1 B active, exact.
  - **MiMo-V2.6-Flash** (`XiaomiMiMo/MiMo-V2.6-Flash-RL`, https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL): new `mimo_v2` builder. 39 SWA-128 layers (KV 8, attention sink) and 9 full layers (KV 4), qk 192 / v 128, 256 experts top-8 with no shared expert, 1 dense layer, MTP ×3. MXFP4 experts, FP8 attention / dense. 308.78 B, exact.
  - **Intern-S2-397B** (`internlm/Intern-S2-397B`, https://huggingface.co/internlm/Intern-S2-397B): Qwen3.5-MoE hybrid (GDN + full), bf16. 396.37 B + MTP 6.60 B, exact.
  - **MiniCPM-V-4.7-35B-A3B** (`openbmb/MiniCPM-V-4.7-35B-A3B`, https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B): Qwen3.5-MoE text, bf16. 34.66 B, exact. The config declares MTP but the release ships no MTP weights, so no draft module is modelled.
  - **LingBot-Video MoE-DMD 30B-A3B** (`robbyant/lingbot-video-moe-dmd-30b-a3b`, https://huggingface.co/robbyant/lingbot-video-moe-dmd-30b-a3b; source https://github.com/Robbyant/lingbot-video): new `lingbot` MoE DiT builder.
    - 48 single-stream blocks over [video ‖ text]: MHA with QK-norm, per-token AdaLN plus per-block table, sandwich norms.
    - FFN: 128 experts top-8 + shared expert; sigmoid grouped routing.
    - Workload: 8-step DMD, CFG 1, 832×480, 121 frames.
    - 30.085 B / 2.9 B active, exact.
- `ir._full_layer_ops`: MoE FFN for full-sequence (DiT) forwards. Router and shared-expert GEMMs run on every local token; routed experts run as grouped GEMMs over the rank's tokens (uniform-routing hit expectation, TP splits `d_expert`, no EP as in the reference). New `Layer.post_norm` charges the sandwich RMSNorms.
- Vision encoders, all equal to the release headers:
  - Intern-S2: Qwen ViT, 456.0 M.
  - Step-5: Perception Encoder 47 × 1536 + two stride-2 3×3 downsamplers + projector, 1.981 B; 169 tokens per 728² overview.
  - MiniCPM-V-4.7: SigLIP 27 × 1152 with the in-tower 2×2 merger after block 6 (`VisionSpec.insert`) + 2×2 merger, 553.0 M; one 448² view.
  - MiMo-V2.6: GQA ViT 28 × 1280 (`VisionSpec.kv_heads`) + merger, 728.6 M; the 261 M audio encoder is stored only (`VisionSpec.extra`).
  - Approximations become coverage reasons (`VisionSpec.approx`), so Step-5, MiniCPM-V-4.7 and MiMo-V2.6 are labelled `partial`: Step-5 crops are not counted, MiniCPM slices are not modelled, MiMo windowed ViT layers are charged as full attention. Their LLM side is exact.
- LingBot-Video pipeline: text encoder = the release's Qwen3-VL-4B (`qwen3-vl-4b.lingbot`, 1k tokens 「假设」), VAE = Wan2.1 VAE (header-identical, fp32). Coverage is `full`.
- `AttnCore.idx_sum`: indexer over cached region summaries. Only newly completed regions are compressed; the cache holds fp32 + fp8 per region; decode reads one fp8 summary per region.
### Fixed
- `summarize_release` roles:
  - vision: `merger.` (MiniCPM-V projector), `vit_large_projector`, `speech_embeddings`.
  - `share_expert` → shared_expert; `.moe.(gate|up|down)_proj` → expert.
  - Video: `.ffn.router / experts / shared_experts`, `text_embedder` → cond, `norm_post_` → norm.
  - Re-summarizing every cached release leaves all existing roles and parameter counts unchanged.
- Fingerprints unchanged (`0a954915c5fb003f` / `53ee9e08165795c6`).

## [0.143.0]
### Changed
- UI: the accumulator field no longer looks empty. The chip facts show `累加器 <total> KiB (默认 64 KiB/核)`; the field placeholder shows the derived default total, with a tooltip giving per-core × cores (an entered value is the chip total); the assumptions line shows the default breakdown. `api_catalog` chips carry `acc_total_kib` and `acc_kib_per_engine_default`.
- `moe_a2a = "node"` (opt-in, DeepEP normal kernels), mechanism from DeepEP's published H800 table (README 2025-06):
  - A remote token-expert pair lands on the same-local-rank GPU of the destination node and is forwarded over NVLink, so the scale-up tier carries (k − 1)/k of the remote pairs as well as the node-local ones (EP32 on 8-GPU nodes: 7/32 → 28/32 of the payload).
  - Node-mode NVLink runs at DeepEP's achieved 153–158 of the 200 GB/s nominal (`DEEPEP_NVL_EFF` = 0.78).
  - Internode RDMA: the published 43 / 58 / 51 GB/s (EP16 / 32 / 64) is at or above the 50 GB/s line rate under DeepEP's logical byte count, so no derating is applied 「假设」.
  - The SM share stays at DeepEP's default 20 / 132.
  - Per-pair mode and all defaults are unchanged.
### Added
- Per-shape kernel ceilings: `Chip.fmt_ceiling` keys `gemm:<fmt>@<N>x<K>` (dense) and `@g<N>x<K>` (grouped) are applied at the nearest published shape (`evaluate._shape_cap`). The validation variant `deepgemm_variant` uses the DeepGEMM README (2025-03, H800) FP8 table (e.g. MLA kv_b K 512: 590 TFLOPS, expert down K 2048: 1099) for DeepGEMM sources only (`DEEPGEMM_SRC`).
- `tests/test_core_143.py`.
### Evaluated, not adopted (defaults and fingerprints unchanged)
- Balanced-case analysis (`val5/v0143/phases.py`, deepseek-profile EP32 prefill). With node-mode all-to-all and TBO, the step is compute-bound in the model (attention 746 ms, experts 611 ms, dispatch 369 ms, combine 739 ms per step), so the DeepEP bandwidth corrections do not move it. The 1.30× over-prediction is compute. The per-shape DeepGEMM ceilings take the balanced rows from 1.38 / 1.30× to 1.23 / 1.16×. The rest is attention / SM share (open).
- Group table (mean |err|): default 27.5 % mean of groups; + DeepGEMM shapes 27.8 % (MoE decode 1.7 → 3.9 %); + DeepGEMM + `moe_a2a` node + sm 30.4 % (MoE prefill 8.9 → 35.2 %; 19.9 % even with the published 2.56× linear-placement imbalance on default-EPLB rows). Applying DeepGEMM shapes to all Hopper FP8 rows (TRT-LLM / vLLM do not run DeepGEMM) was mixed: v1 max 19.8 → 22.5 %. So `moe_a2a` stays opt-in and the default variant is unchanged.

## [0.142.0]
### Fixed
- `exec_overlap = "pipe"` (default) with `fabric.algo = "auto_overlap"` (default): the stage re-pick set the link / sync terms, but the flow-shop replay still ran every collective at its per-collective default pick. That inconsistent mix could undercut both consistent choices, so 4 → 8 D2D units was +0.009 % slower (sysfuzz 0.141 seed 41: Qwen3-Next TP8/EP8, 2-card packages, 4-unit point = hier α with ring bandwidth). The replay now uses the same picks (`fabric` pick override), and the combination is chosen on the pipe total itself (`evaluate._pipe_repick`): the stage-optimal and default picks, then coordinate descent. The chunked-prefill fused replay uses the decode stage's picks. `auto_overlap` ≤ `auto` holds by construction.
- Results change only for pipe + fabric-on scenarios whose re-pick differed from the default pick (they were optimistic before). Example: the seed-41 point at 4 units, 115.155 → 115.212 ms TTFT (equal to `auto`).
### Changed
- GPU validation: rows whose source simulates absolutely balanced routing (`eplb = "perfect"`: deepseek-ai/profile-data, SGLang "simulated perfect EPLB") run with `moe_rank_max = "off"`, since there is no random slowest-rank excess. MoE prefill 9.2 → 8.9 %; mean of group errors 27.6 → 27.5 %.
### Evaluated, not adopted
- Default-EPLB imbalance from a published statistic (TensorRT-LLM wide-EP EPLB README: DeepSeek-R1 EP32, linear placement, layer-averaged (max − mean)/mean = 1.564, i.e. hottest rank 2.56× mean; 0.115 with EPLB) as the slowest-rank skew on the `eplb = "default"` rows, with `moe_a2a` node + sm (20/132). Default rows go 1.62–1.69× → 0.80–0.92×, but the perfectly balanced rows stay 1.30–1.38× over: the node + sm all-to-all mechanism over-predicts the balanced case, which imbalance cannot explain. MoE prefill 23.7 % vs 8.9 %, so `moe_a2a` stays opt-in. The same skew on the per-pair default is 86.8 %.
### Added
- Fingerprints and scenario hashes are unchanged (no pinned scenario has fabric on under pipe). `test_core_060` now states its stage-rule hier pick at `exec_overlap = "stage"`; under pipe, the consistent replay keeps tree, which is 2.7 % faster than hier. It adds a pipe check that the result is never worse than any forced algorithm.
- `tests/test_core_142.py`; `val5/v0142/` (sysfuzz seeds 41 / 45 / 46: 0 violations; `eplb_try.py`; Playwright UI smoke).

## [0.141.0]
### Changed (defaults; fingerprints re-baselined, user-approved)
- GPU / validation default: `extval.DEFAULT = "default"` = the per-SM candidate path with ceil (`ceil_variant(candidate_variant())`); `scenario` / `predict` / `table` / `predict_cb` default to it, and `ext_validation.json` is regenerated. Group errors, old `catalog` → new `default` (gmean / mean |err|): v1 static 1.32/37.3 % → 1.06/20.1 %; v1 TTFT 1.81/87.2 % → 1.29/31.6 %; v1 max 1.28/33.0 % → 0.89/19.8 %; MoE prefill 0.94/8.8 % → 0.94/9.2 %; MoE decode 1.11/15.6 % → 0.98/1.7 %; MoE static 2.10/109.9 % → 1.82/82.0 %; MoE max (Mixtral) 1.71/77.8 % → 1.01/20.0 %; r3 lat 3.67/267.4 % → 1.39/38.8 %; r3 max 1.89/90.2 % → 1.18/32.9 %; r4 1.12/37.8 % → 0.95/19.6 %; all 204 rows 1.50/60.2 % → 1.04/23.1 %; mean of group errors 76.5 % → 27.6 %.
- Per-core accumulator and SRAM port. `acc_kib = None` now means 64 KiB × engines (an explicit value stays the chip total); new `acc_kib_per_engine` (exclusive with `acc_kib`). The default port is 4·(R+C)·2 B/cycle per core, × engines (was 4·(R+C·E)·2 for the chip), e.g. 56×56×16: 7616 → 14336 B/cycle. With `instance_sched = "auto"` more engines are never slower; forced `wide` / `ws_*` schedules can still be up to +11 % slower with more engines, because the ganged array fill (R + C·E) grows (definitional).
- Migration: `upgrade_legacy` converts saved chips with a numeric `acc_kib`, an `sram_port_Bpc` key and no `acc_kib_per_engine` (also `pd.prefill_chip`) to per-engine fields that reproduce the old hardware exactly.
- `serving.moe_rank_max` default `"off"` → `"expected"` (MoE groups better overall; catalog column: decode 15.6 → 14.5 %, prefill 8.8 → 9.2 %). `moe_a2a` node+sm stays opt-in: MoE prefill 8.8 → 52.7 % (mean of group errors 32.4 % vs 27.6 %).
- `fabric.algo` default `"auto"` → `"auto_overlap"` (per-stage re-pick; monotone in bandwidth, see 0.140 notes).
- PD: after `best_prefill`, the smallest power-of-two prefill batch that still meets the TTFT SLO and carries the same request rate is used. Default PD TTFT goes from 1298 to 658 ms at the same goodput.
- Re-baselined pins: fingerprints `_fp` stage `e2176af2610c144c` → `0a954915c5fb003f`, default `392692e6937f69ef` → `53ee9e08165795c6`; 125 `fpx` `d29ef7bf6936030d` / `c4e341987bf2461f` → `4690f724dabe9fa6` / `ea2456afee7c3a86`; scenario hashes qwen3-8b `917ab559e6aa1c07` → `723894c17df5d7bb`, qwen3.8-27b `a6463624c5748f5f` → `95f47a808a9c342d` (and the six other 088 hashes); coverage KPI mochi-1 `d134d7d69b76` → `07a12203b662`. Tests that exercise other mechanisms pin `moe_rank_max = "off"` / `algo = "auto"` explicitly.
### Added
- `tests/test_core_141.py`; `val5/v0141/` (`groups.py`, engine-adapted `fuzz.py`).
### Notes
- Fuzz (r75): engine fuzz is clean under auto / cta; sysfuzz seeds 41–43 found one +0.009 % point (2× D2D units, `exec_overlap = pipe`): the `auto_overlap` re-pick minimises stage time, not flow-shop time. Documented, not fixed.

## [0.140.0]
### Fixed
- PP split (`pp_split = "cost"`, default): the time proxy (stage rule, full DRAM streaming) proposes splits and the full model picks among them. A better split one layer away was never tried, and when the proxy found nothing better than equal counts the full model was skipped altogether. So more DRAM bandwidth could select a 2.7 % slower split (Qwen3-8B-FP8 PP2 prefill, LPDDR6 4→6 packages). `_pick_split` now hill-climbs each boundary ±1 layer under the full model (fits first, then the slowest stage; at most `PP_REFINE_STEPS` = 8 rounds), including in the equal-count case.
- Default results change for PP > 1 only, and only get faster: 48 of 340 catalog PP2 / PP4 points improve, by up to 11.8 % (AlphaFold2 PP4; Kimi-K2 / Qwen3.5-397B PP4 decode about −10 %). None gets slower. PP evaluation costs about 2.3× more. PP = 1, the fingerprints, and the validation group table are unchanged.
### Added
- `val5/v0140/sysfuzz.py` (reusable; `python3 sysfuzz.py <repo> <seed> <N> [eval|pd|search]`): a system-level fuzzer over TP / PP / EP / DP, D2D packages, node / net tiers, fabric, overlap modes, all domains (LLM, video, image, protein, genomics), PD pools (KV bandwidth, layer-wise KV, prefix cache, prefix hit), and batch search. Invariants: more link / D2D / net / DRAM / SRAM / clock / host bandwidth never lowers throughput or raises latency; fits and SLO-ok stay true; DRAM need is conserved (PP = 1); PD goodput does not drop and its SLO flags do not flip; `best_batch` is at least any feasible point and is monotone in resources.
- `tests/test_core_140.py`.
### Notes
- With `fabric.algo = "auto"`, each collective picks its algorithm by its own bw + α, so a hidden bandwidth term can make a higher-α algorithm win and the stage gets slightly slower (+0.08 %, net 50→100 GB/s). `auto_overlap` re-picks per stage and is monotone. Left as is.
- PD TTFT can rise with more resources because the prefill batch is the largest one inside the TTFT SLO; goodput never drops. This is by design.

## [0.139.0]
### Fixed
- Multi-core GEMM picker (`instance_sched = "auto"` / `"split"`): row and column splits make every core group convert its own copy of a non-native operand (e.g. AWQ int4 weights), but the picker compared array cycles only. A wider SRAM port could select a split whose replicated dequant made the op vector-bound: up to 7× slower with a 4× port. Each plan is now scored as max(array, conversion) cycles on the dequant / vector path. The catalog sweep is unchanged.
- `exec_overlap = "pipe"`: MoE dispatch / combine were credited against the shared expert's compute seconds, but they hide under the shared expert's wall time in the flow shop. When the shared expert was DMA-bound, a faster clock exposed more all-to-all (+0.3 % TTFT at 1.5× clock). The simulator now credits the window it replays. Results are identical when the window is compute-bound, and the catalog sweep is unchanged.
### Added
- `tests/test_core_139.py`: standing tests for both fixes, plus seeded randomized invariants: 4× SRAM / port / lanes / accumulator, 1.5× clock, more DRAM channels, 8× dequant, and 4× link are never slower; DRAM need is conserved; SRAM residency does not drop.
- MODEL §85: the NPU hostile-review scope, method and findings.
### Changed
- `test_core_127` GEMM monotonicity (count + 1, reconf ≤ fixed organisation) is now stated on the picker's objective, max(array, conversion) cycles, rather than on array cycles alone.
### Not changed
- Defaults, the GPU / validation default, fingerprints, and the validation group table (both fixes are NPU-only paths).

## [0.138.0]
### Added
- `extval.GEMM_CEILING` / `ceil_variant` (opt-in): published library-best dense-GEMM fractions per chip and format (A100 fp16/bf16 0.844, Hopper bf16/fp16 0.752, B200 bf16 0.689, B200 fp4 0.691; sources in the code). Under tile I/O they set the per-tile main-loop rate, as Hopper fp8 already did.
- `Chip.tile_io = "persistent"` (opt-in): tile I/O hidden under the neighbouring tile's main loop (persistent ping-pong kernels). Every group is worse than `dram`, and the DeepGEMM trace matches `dram`.
- MODEL §84: decomposition of candidate vs candidate + tile I/O, the full group table, and a re-read of the TRT-LLM 0.21 DeepSeek-R1 H200 source.
### Not changed
- Defaults, the GPU / validation default, and fingerprints.

## [0.137.0]
### Fixed
- `exec_overlap = "tbo"`: per-kernel / per-layer overhead (`kernel_us`, `layer_overhead_us`) runs on the compute stream, so the other micro-batch's all-to-all now hides it (`StageTime.t_kern`). It used to be added after the max, so link-bound TBO stages paid it in full. This was the per-SM candidate's only regression (MoE prefill 8.8 → 8.9 %, now 8.8 %).
### Notes
- With this fix the per-SM path (candidate, with or without tile I/O) worsens no validation group compared with the default (MODEL §83 has the table). The default is not switched in this version (user decision). Fingerprints are unchanged.

## [0.136.0]
### Fixed
- Opt-in tile I/O (0.135) was also charged to attention under `cta`: a per-tile prologue and C writeback to DRAM, which flash attention never does, on K / V bytes already counted. For 8B decode this made attention 4.5× too slow and caused the v1 max / r4 under-prediction. It now applies to GEMMs only. Candidate + tile I/O now beats the `wide` default in every group except MoE prefill (8.8 → 8.9 %).
### Added
- `Chip.kernel_vec` (opt-in) and `extval.kfloor_variant`: a per-kernel floor of 2.0 µs (the DeepSeek decode trace's 10th-percentile small-kernel duration) on every GEMM / attention / vector kernel. It replaces the per-layer lump, which double-counts vector work at large batch. It improves max-load and r4 but worsens TTFT and r3 latency, so it stays opt-in.
- MODEL §82: decomposition of the v1 TTFT / max-load rows.
### Not changed
- Defaults and fingerprints. The per-SM path still fails the rule by 0.1 point in MoE prefill.

## [0.135.0]
### Added
- `Chip.tile_io` (`""` default | `"dram"`, opt-in, with `instance_sched = "cta"`): each GEMM output tile also pays its pipeline prologue (one K-stage of A and B) and its C writeback at the SM's share of DRAM bandwidth. With a format ceiling, the main loop runs at the ceiling per tile. Against the DeepSeek prefill trace's 9 DeepGEMM shapes the mean error is 7 % (every shape 0.89–0.98×), against 33 % for the default `wide` and 34 % for `cta`. Small-K GEMMs (down, q_b) are no longer too fast.
- `extval.tileio_variant`; MODEL §81.
### Not changed (default-change rule)
- On top of `cta` GEMMs, tile I/O improves almost every validation group. Compared with the shipped `wide` default it worsens several groups, so the default is unchanged; fingerprints are unchanged. Per-SM attention scheduling (`attn_sched = "cta"`) also remains opt-in.

## [0.134.0]
### Added
- `serving.moe_a2a` (`pair` default | `node` | `auto`, opt-in; CLI `--moe-a2a`): node-deduplicating EP all-to-all, as in DeepEP's normal kernels. A token crosses the scale-out network once per destination node. Network bytes scale by N·(1 − C(E−E/N, k)/C(E, k))/k, which is 0.45 for DeepSeek-V3 on EP32 across 4 nodes. `auto` applies it in prefill only.
- `serving.moe_a2a_sm` (opt-in; CLI `--moe-a2a-sm`): the share of compute units the node-mode all-to-all kernels hold while overlapping prefill. `extval.DEEPEP_NORMAL_SM` = 20/132.
- `extval.a2a_variant`; MODEL §80, a per-op decomposition of the DeepSeek-V3 prefill against the public DeepSeek trace.
### Not changed (default-change rule)
- With both options on, the balanced-routing prefill rows match (1.00–1.06×), but the default-EPLB rows over-predict because routing imbalance is unmodelled, so the prefill group worsens (8.8 → 20.9 %). Defaults and fingerprints are unchanged.
- catalog_kernel's 78 % prefill error comes from serialising communication the sources overlap (TBO); `tbopf_variant` / `candidate_variant` already give 8.8–8.9 %. The r4 max-load error follows ISL in dense and MoE models alike (prefill utilisation, MODEL §39).

## [0.133.0]
### Added
- `serving.moe_rank_max` (`off` default | `expected`, opt-in; CLI `--moe-rank-max`): MoE routing with the expected per-step maximum over EP ranks of distinct experts read and token-expert pairs under random routing (normal order statistic). Against a Monte-Carlo routing simulation: 3.6 % / 4.5 % mean error vs 23 % / 22 % for the mean rank. Not the default: on the validation rows it helps decode/static/max-load under the catalog variant but worsens DeepSeek-V3 prefill and catalog_kernel decode. Fingerprints unchanged.
- `extval.rankmax_variant`; tests for the order statistic vs Monte-Carlo, default unchanged, validation and CLI.

## [0.132.0]
### Changed
- MoE routing with a measured `serving.moe_expert_load`: the busiest rank's distinct experts are now the exact occupancy Σ 1−(1−p_e)^T (p_e = k·load_e/Σload), maximised over EP ranks, instead of the uniform expectation. Against a Monte-Carlo routing simulation the mean error falls from 22 % to 1.6 %. Concentrated loads now read fewer experts (faster); a hot rank with spread-out load reads more (slower). Fingerprints and validation groups are unchanged.
### Added
- `serving.moe_skew_spread` (`popular` default | `uniform`, opt-in): for a scalar `moe_skew`, `uniform` spreads the hot rank's extra pairs over all its experts (p = k·s/E). Not the default because it is worse than the current rule for Zipf-shaped popularity.
- Tests for occupancy vs Monte-Carlo, unchanged scalar behaviour, the opt-in, and validation.

## [0.131.0]
### Fixed
- Energy report: a table that held only entries the single-scenario report cannot use (`idle_W_prefill`, which is for the PD prefill pool; `pJ_bit_host` without host-link traffic) produced `J_per_unit = 0` and `avg_W_per_card = 0`, which reads as zero energy. It now returns no total plus a note, as the PD energy helper already did, and the UI shows the note.
### Verified (no change)
- Headless browser (Chrome + Playwright, 1400 / 900 px):
  - all 72 selectable models load with no console errors or page errors;
  - for 11 models across every domain, eval / sweep / compare complete with no text overflow or overlapping KPI tiles;
  - the capacity-fix button, layout search, Pareto and the models / about tabs work;
  - a forced 400 dims the last result with a note, and the next valid eval clears it.
- Energy / power model:
  - idle-only power equals the table's W per card in every domain;
  - MAC / vector counts per token are invariant under TP / PP;
  - DRAM + SLC bytes are conserved when the cache absorbs traffic;
  - MoE skew moves no work and charges only the longer window;
  - bit entries are × 8 and MAC entries per bf16-equivalent MAC;
  - weight bytes per token fall about 1 / B;
  - the PD energy helper and the single report agree to 1e-12;
  - units per window = throughput × window.
  - These are pinned by tests/test_core_131.py.
  - PP re-reads weights per micro-batch (more DRAM / SRAM bytes per token under PP), which is real, not double counting.

## [0.130.0]
### Fixed (hostile review: frontend, newest catalog builders)
- `/api/sweep` returned 200 with a flat line for more knobs the model does not read. These now return 400 with the reason: `serving.ctx` / `serving.prompt` on fixed-workload models; `serving.prompt` in decode without PD; `serving.moe_skew` at EP = 1; and workload knobs that don't apply to the workload kind (`frames` on image DiTs and protein models, `msa` on sequence encoders, `samples` without diffusion). A sweep over every catalog model × 14 paths now has no flat 200 results. The UI no longer offers `moe_skew` on dense models, `spec_k` without MTP / draft, or `frames` on image DiTs; the catalog exposes `draft` for this.
- A rejected eval (400) left the previous result on screen looking current. It is now dimmed, with a note that it comes from the last valid input.
- Number formatting rounded before choosing the tier: 99.96 showed as "100.0", 9.999 as "10.00", and 9999.6 as "10000" while 10 000 showed as "10,000".
- Image DiTs (Qwen-Image, FLUX.2-klein) were labelled with video terms: "单段延迟" in sweep / Pareto / compare / layout columns and exports, "帧" in the energy table and the scenario line, and "帧/s/卡" in the capacity fix hint (genomics showed "decode 吞吐" there). They now read "单张延迟" / "张", and genomics shows "窗口/s/卡".
- A non-JSON error body (proxy 502 page) surfaced as "Unexpected token <". It now shows the HTTP status.
### Verified (no change)
- Newest builders against the official checkpoints:
  - Carbon-8B, Carbon-A-1.2B, FLUX.2-klein-4B, Qwen-Image and Qwen-Image-2.1-Turbo match exactly (index metadata / summed safetensors headers).
  - LFM2.5-24B-A2B is official `total_parameters` + 64 × 38 fp32 router `expert_bias` buffers, consistent with `total_size`.
  - Ling-3.1-flash main + MTP equals the snapshot, whose bytes equal the index `total_size`.
  - These counts are now pinned by tests.
- Stale-response handling: per-channel sequencing + abort is sound. The shared `mem` channel (form vs one-click) is latest-wins by design.

## [0.129.0]
### Fixed (hostile review: UI / API layer, DES seeding)
- HTTP: a malformed `Content-Length` (`-5`, `abc`, `1e3`) crashed the request handler, so the connection was dropped with no response. It is now a 400. Deeply nested JSON (a `RecursionError` in the parser) was a 500; it is now a 400.
- Worker pool: a search worker killed abruptly (OOM, signal) broke the process pool permanently, and every later compare / layouts / fit / stability request returned 500 until restart. The broken pool is now replaced and the request retried once.
- Validation: `chip.mac_eff` > 1 was accepted (sustained above peak). It must now be in (0, 1], matching the UI input.
- `/api/sweep`: sweeping a knob the model ignores returned a flat line with 200, which reads as "insensitive": `workload.*` on an LLM, `serving.images` / `image_w` / `image_h` without a vision tower, `serving.moe_skew` on a dense model, `serving.spec_k` without MTP / draft. These now return 400 with the reason.
- DES: `simulate(seed=None)` seeded arrivals from the clock (not reproducible) while the draft stream used "spec-None". Seeds must now be integers, and `compare(seeds=…)` rejects repeated seeds (identical runs reported zero noise).
### Verified (no change)
- 900 fuzzed scenario bodies over every scenario leaf × 7 routes gave no 500s. Scenario round-trip (eval → `scenario` → eval, under any `chip_preset`) is hash- and summary-identical. 72 concurrent mixed requests against the threaded server (with the process pool) match serial responses. DES output is identical across `PYTHONHASHSEED` values and unaffected by runs in between. `/api/fit` and `/api/sweep` bounds behave as intended (1–32 values, integer paths, oversized values → 400).
- Colocated ITL p99 interpolation: not offered. The model's probabilities already match the DES (0.128), so interpolating would only imitate the empirical quantile estimator; it is not a mechanism, and it would change the defaults.

## [0.128.0]
### Investigated (no closed-form change)
- Colocated prefill-first worst-case ITL (p99 over requests of each request's longest gap) read 8–26 % below the DES at load 0.4 on edge layouts. DES decomposition: the worst gaps are almost entirely prefill busy time (tp3: 172 of 177 ms; the decode-step part is ≈ 5 ms), so neither the batch-size step variance nor a larger post-burst step explains the gap. With one prompt length, busy durations take only the values n · prefill latency, so the p99 is an atom. The closed-form probabilities P(max ≥ n) match the DES (tp3 P(≥3) 0.0099 vs 0.0102; pp2 0.033 vs 0.034; pp3 P(≥4) 0.0085 vs 0.0100). The 0.99 level sits at an atom boundary, where a ±0.1-point shift in probability moves the quantile by a whole prefill. The bias is an artefact of how the quantile is defined, not a missing mechanism (MODEL §74).
### Changed
- The JSON export of compare / sweep / Pareto carries `schema: 1` (bumped on any key or meaning change).
### Verified (no change)
- Catalog labels: coverage reasons are present exactly for partial entries; dtype labels are well formed and show the expert format only when it differs from attention; offline entries cite https sources. The DeepSeek-V4.1-Flash `mlp` role (15 360 fp32 parameters) feeds no operator.
- Sensitivity: the block runs scenario swaps (overlap bracket, DRAM efficiency 0.5 / 0.9, vector lanes ÷4), not finite differences. `ratio` = alternative / base, and `hinge` = |ratio − 1| > 0.15 or a bottleneck flip. Sweep-sensitivity skip keys all refer to existing cases.
- Export schema: the sweep row and sensitivity block key sets are locked by tests.

## [0.127.0]
### Fixed (hostile review round 3, hardware side)
- Capacity fix (`/api/fit`): the fewest-cards search tried powers of two only, so a model that fits on 3 cards was offered 4 (Llama-3.3-70B on the default chip: TP3 holds it). It now also scans the card counts between the last failing and the first fitting power of two (≤ 32 counts) and returns `pow2_cards` alongside; the UI shows both when they differ.
### Verified (no change)
- GEMM mapping (all organisations, wide / auto / CTA scheduling, 5 weight formats, 6000 random shapes): never above the peak MAC rate or the SRAM-port weight bound, monotone in instance count, `reconf` ≤ every single organisation. The dtype execution matrix (native / dequant-W / upcast / downcast).
- Energy: useful MAC counts per token are invariant under PP / TP / DP for dense models and under DP·EP for MoE; extra counts under TP come only from work every TP rank really executes (e.g. the MoE router and the V4 compressor / indexer), and extra vector work under PP from weight dequantisation repeated per micro-batch, both by design.
- Faster hardware (SRAM, port, frequency, MAC efficiency, memory efficiency, link, SLC, GEMV unit) never gives a slower step (7 models × 2 chips × 2 batches).
- Capacity: per-stage need = weights + KV + state + reserve + activations + pipeline components. Video adds component activation peaks, FSDP gathered layers or the prefix cache on top, all accounted once.
- Pareto front = brute-force non-dominated set.
- The TP3 colocated ITL-max tail, rerun with 10 seeds: −26 % (band 30 %). The closed form runs 8–26 % low on colocated ITL max at load 0.4 on every edge layout.

## [0.126.0]
### Fixed (hostile review round 2)
- Compressed + sliding-window attention (DeepSeek-V4 layers with compress > 1 and a window): a decode query at context n was charged ⌈n/c⌉ + w keys even when that exceeds n (129 keys at n = 1), while prefill already used min(p, ·). `AttnCore.ctx_eff` now caps at n, so decode equals the prefill last-position count. Only short contexts (n ≲ w + n/c) change; default fingerprints are unchanged.
### Verified (no change)
- Attention key counts vs brute force; GQA under TP that does not divide the KV heads (storage and decode KV reads follow `gqa_local`); KV / recurrent-state bytes per family vs hand formulas (GQA, MLA, gpt-oss sliding layers, Qwen3-Next GDN state); hierarchical all-reduce / all-gather / all-to-all volumes vs the textbook ring / direct formulas; collective payloads; PP partitions (contiguous, non-empty, work-conserving, pp > layers rejected); DiT cross-attention shapes incl. sequence parallel; MLA absorbed decode / naive prefill.
- DES vs closed form on edge layouts (PP2, TP3, PP3, PP2·TP2, 1-card prefill pool): within the validation bands except one ITL-max tail (−33 % vs a 30 % band, noisy).

## [0.125.0]
### Fixed (hostile core review)
- Grouped (block-diagonal) linears: the per-rank GEMM kept the full N per group (N = n instead of n/g), so DeepSeek-V4 / V4-Pro / V4.1 `o_a` (8 groups) did 8× its work and weight reads, and the storage plan held 8× its bytes (V4-Flash batch-1 decode weight reads +69 %, V4-Pro +121 %; stored weights +7 %). `ir.groups_local`: whole groups per rank (⌈g/TP⌉), N split once TP > g. No validation data covers V4, so no validation result changes. Fingerprints rebaselined (stage c034bfe3e3a3afa2 → e2176af2610c144c, default e51c7ec17bbdb998 → 392692e6937f69ef); only the V4 rows move (without them the hash is unchanged, tested).
- Batch search bound: the branch-and-bound bound was E/step, but prefill throughput is B·S/step and video uses the host-overlap period, so the bound was too low and pruned the optimum (qwen3-8b 2048-token prefill: batch 124 instead of 112; 4-card layout ranking top-1 1.5 % below brute force; Wan te_cpu + overlap batch 5 instead of 4). The bound now uses thr(p)/p. Decode results are unchanged.
### Verified (no change)
- Per-op weight GEMM shape = weight params, and share / wshare ≤ 1 under TP / EP / SP splits for every catalog family (incl. LFM2, Ling, Carbon, image DiTs, structure models); batch-1 decode weight reads = active − embedding params (misc vectors ≤ 0.7 %); stored weights vs release bytes; batch-search monotonicity (P1 / P2) over 10 models × 3 layouts × 6 options.
- Public data searched for the open gaps (ROCm per-sequence decode overhead, small-M GEMM efficiency): nothing quantitative enough to adopt.

## [0.124.0]
### Validator audit
- Adversarial audit of the validators against brute-force references on small random cases: pdsim RadixLRU / LRUCache, DecodeReplica, ColocPrefillFirst, KVServer (Lindley), PrefillReplica batch cap, `_pct`, the overlap pipe flow shop (+ `feed_runs` / `stage_pipe_time` repeat shortcuts), vecsanity PUBLIC values, extval summary. No simulator bug found.
- Fixed: `extval.summary` crashed (`math.log`) on a zero / infinite ratio; such rows are now skipped and counted as `n_invalid`.
- Fixed: `data/ext_validation.json` was stale (generated by 0.66 with a hardcoded label); regenerated (29/700 rows moved ≤ 9.7 %, summary max-errors ±0.4 pt; no conclusion changes). `scripts/ext_validate.py` stamps `__version__`.
- Brute-force guards added as tests (test_core_124).
### Added
- `fabric.channels` + `fabric.channel_GBps` (opt-in, 「假设」): per-channel copy-engine cap on collective bandwidth per scale-up / network tier, min(link, channels × channel_GBps). CLI `--fabric-channels` / `--channel-gbps`, UI field. Defaults unchanged (fingerprints c034bfe3e3a3afa2 / e51c7ec17bbdb998).

## [0.123.0]

### Fixed

- 0.122 多轮会话模型「偏低最多 −18 %」的根因在仿真器而不是模型：`pdsim.RadixLRU.access` 收到比树浅的路径（多轮 prompt）时，按树深循环，`path[:k]` 在 k 超过路径长度后仍等于整条路径，叶节点按最深一层的大小入账（例 64 而不是 512 token）→ 缓存实际存的历史偏少、命中偏高。现在按路径自身深度停止（整条路径的调用逐位不变）。修正后对仿真：顺序会话 −3.9 … +0.6 %，各轮独立 −1.0 … +0.1 %。部分历史存活（按回访新近程度）实测 ≤ 0.4 % 的请求，未单独建模。

### Added

- PD DES 带多轮历史：会话流（顺序会话 / 各轮独立）、预热流在测量流之前且共享打开的会话、请求只插入到自己这一轮、只能匹配到上一轮。会话粘在一个副本上（按会话路由；闭式每副本 A / 副本数个会话，前缀感知路由时合并）。DES 与闭式 token 命中率相差 ≤ 0.01（dense8b，4 轮，64 / 256 / 32 会话）。移除「DES 不含轮次」警告。
- `pd.prefix_turn_prompt`（默认关）：serving.prompt := 树 + (T − 1)/2 ·（输出 + 用户 token），decode 上下文 := prompt + 输出 / 2。命令行 `--pd-turn-prompt`，工作台勾选项。

## [0.122.0]

### Added

- 多轮对话前缀（radix 树，默认关）：`pd.prefix_turns`（每会话轮数，1 = 关）、`pd.prefix_turn_user`（每轮新用户 token）、`pd.prefix_writeback`（完成后把 prompt + 输出写回树）、`pd.prefix_sessions`（同时进行的会话数；0 = 各轮独立到达）。第 t 轮 prompt = 上一轮 prompt + 输出 + 新用户消息，历史作为分支为 1 的层接在树叶下；不写回时第 t 轮只能命中上一轮的 prompt，写回时再多命中上一轮输出。命令行 `--pd-turns / --pd-turn-user / --pd-writeback / --pd-sessions`，工作台 PD 前缀树下方。
- 闭式两种：各轮独立（IRM，叶路径 = 会话；Che 逐节点、按触达份额）与顺序会话（Che 按请求时间：会话间隔几何分布，节点存活 = 间隔 ≤ τ，结束后滞留 τ；基础树节点按会话突发的 τ 膨胀）。对 RadixLRU 仿真：IRM −6 … +0.1 %，顺序会话 −18 … +3 %（偏保守，最大误差在会话多、写回、容量中等处）。PD DES 仍按不带历史的树仿真（给出警告）。

### Fixed

- CHANGELOG：0.116–0.120 的版本标题在发布时被下一版覆盖，已补回（新测试检查 0.100 起每版都有标题）。

## [0.121.0]

### Added

- PD + fabric：`fabric.kv_feedback_prefill`（默认关，需同时开 `kv_feedback`）。KV 流的 NIC 份额 u_kv 同样占用 prefill 池（发送端）的 NIC：prefill 池跨节点集合通信带宽 × (1 − u_kv)，prefill TTFT 与请求速率在同一不动点里重算（0.60 只回馈 decode 侧）。命令行 `--kv-feedback-prefill`，工作台 fabric 面板勾选项。例：DeepSeek-V3 PD（DP64·EP64，12.5 GB/s，oversub 4）TTFT 5800 → 5971 ms，goodput / 卡 2.59 → 2.51，u_kv 2.98 → 2.89 %。



## [0.120.0]

### Added

- 每 kernel 启动 / 调度开销 `chip.kernel_us`（0.92 起已有，默认 0 = 不计，结果不变）扩展到结构模型：融合注意力、三角乘 / OPM / PWA 各 1 次；分块三角注意力每块 3 次（QKᵀ、softmax、·V）。命令行 `--kernel-us`，工作台芯片参数里可填。

### Fixed

- 分块三角注意力（0.119，非默认）在单块 logits 放得下片上时 DRAM 记为 0，而融合内核要付 q / k / v / 输出的一遍流式 → 小块显得比融合快（OpenFold 512 残基 13.65 vs 13.76 s）。现在 logits 留在片上、q / k / v / 输出照付一遍：kernel_us = 0 时与融合相同，> 0 时更慢（0.53 µs：13.86 vs 13.77 s）。

### Audit

- 工作台 vs API 一致性（headless，30 个跨领域随机场景，经真实控件操作）：同一请求的哈希与 summary 一致，每个 KPI / 流水级 / 存储表数值在显示舍入内相等——0 问题。
- 0.100 以来未改动模块的随机不变量（queueing、lengths、mem_catalog、energy、fabric、vision）：无误；视觉分辨率到达 token 上限处 token 数可随尺寸非单调（Kimi / DeepSeek 参考 resize 逐轴取整），属参考行为。



## [0.119.0]

### Added

- 结构模型三角注意力内核 `workload.tri_attn`：`fused`（flash 式，默认，结果不变）| `materialized`（参考 PyTorch 实现整块写出 H·N³ logits）| `chunked`（`chunk_layer` 每次 `tri_chunk` 行，0 = 参考默认）| `reference`（发布代码在该 N 下的做法：ESMFold 不分块；OpenFold / AF2 chunk 4；Boltz-1 N > 384 → 128 否则 512；Protenix ≤ 1024 不分块、≤ 1536 512、≤ 2048 256、更长 128）。计算量不变；每块 logits 定激活峰值，超过 SRAM/2 时 logits 走 4 次 DRAM（写、softmax 读写、P·V 读）。推理无重算。命令行 `--tri-attn` / `--tri-chunk`，工作台蛋白质工作负载里可选。

### Fixed（工作台）

- 图像模型此前显示「视频生成 / 单段延迟 / 1 帧 / 帧/s」→「图像生成 / 单张延迟 / 张/小时」，隐藏帧数、SLO 改为单张。
- 基因组模型在对比 / 扫描 / 敏感度表头显示「序列/s」→「窗口/s」；能耗表「次数 / window」→「窗口」。
- Carbon 基础 DNA 语言模型加领域标签；页头领域列表补全（VLM / 图像 / 基因组）。



## [0.118.0]

### Fixed

- Open-Sora 1.2 潜帧：非 17 整数倍的末段此前补齐到 17 帧（5 潜帧）；参考 `VideoAutoencoderPipeline.get_latent_size` 为整块 5 个 + 末段 r 帧前补到 4 的倍数 → `⌈r/4⌉` 个（例：100 帧 30 → 29 潜帧）。默认 102 帧不变。
- 基因组窗口：窗口不长于重叠（默认 1024 token）时此前把重叠钳到 L − 1（每窗口只前进 1 token）；现按单窗口、无重叠计并警告。

### Added

- 结构模型：激活计划按融合（flash 式）三角注意力计（不物化 H·N³ logits）；未分块参考实现的单块 logits 超过每卡 DRAM 时给出警告（只告警，不改结果）。

### Audit

- MiniMax-H3 VAE 解码几何对照 diffusers `AutoencoderKLMiniMaxH3._decode` / `_split_tiles`：段数、5 + 2 潜帧重叠、补帧与 tile 数一致；Open-Sora 时间 + 空间 VAE 解码几何一致（除上面的末段潜帧）。
- 蛋白质路径：三角乘 / 三角注意力 / MSA 行列注意力 / OPM / PWA 闭式、recycle / 步数 / 样本线性、DAP（SP 2 / 4）总计算量守恒（5 个模型）、N³ 渐近、MSA 深度（列注意力 S²）凸、TP 拒绝——无误。基因组 16k / 32k 窗口正常。
- HF 新发布（2026-10-01 起、非 gated、公开配置）：只有已收录的 Carbon-A-1.2B；Prompteina（10-09，设计模型）只有 Lightning .ckpt、config.json 无结构、自编码器另发——未收录。



## [0.117.0]

### Fixed

- 视频分辨率取整：非 `vae_s · patch` 整数倍的宽高此前按向上取整的 patch 网格计；参考实现向下取整（diffusers WanPipeline 带警告取整，HunyuanVideo / LTX-Video / Mochi / CogVideoX 拒绝或在 latent `//` 与步长 p 的 patch 卷积处截断）→ 统一向下取整到 `vae_s · patch` 并警告（图像模型保持 0.116 规则）。
- Wan2.1 / 2.2 帧数：WanPipeline 把 `num_frames` 取为 `F // 4 · 4 + 1`（F ≡ 0 mod 4 时多 1 个潜帧）；新 `vae_frames = "wan"`。因果 VAE 模型报告的帧数改为解码器实际输出 `(lt − 1)·t + 1`。
- 发布默认工作负载不受影响；指纹 c034bfe3e3a3afa2 / e51c7ec17bbdb998 不变。

### Audit

- 10 个视频模型的文本编码器 + VAE 流水线自 0.44 起已齐全（coverage full），无需新增。
- VAE 3D（时间压缩）审计：各卷积族解码器几何（因果时间上采样 `e·t − (e−1)`、LTX 4×4 unpatchify）精确还原请求的帧数 / 宽高；VAE 配置的空间 / 时间压缩、潜通道数与目录及 DiT `in_channels` 全部一致——其他生成模型无 FLUX.2 式的 VAE 下采样倍数错误。



## [0.116.0]

### Added
- **Text-prefix K/V reuse across denoise steps** (Qwen-Image-2.1). The mechanism follows the diffusers reference:
  `QwenImage21Transformer2DModel` with `causal_condition` and `QwenImage21KVCache`, and `QwenImage21Pipeline` with
  `use_kv_cache=True` as its default.
  - With `causal_condition`, text tokens attend causally to text only, while image tokens attend to [text ‖ image]
    (a block-causal mask). The text prefix is modulated from t = 0, so its K/V do not depend on the timestep.
  - Step 1 extracts and writes the prefix K/V. Steps 2…n run only the image rows, with keys = cached prefix + image,
    and read the cache.
  - The denoise time is per_step · (t_extract + (steps − 1) · t_cached). The cache (layers × T × heads × (dk + dv))
    is resident per in-flight sequence.
  - The feature is model-declared through `NativeWorkload.cond_causal` / `prefix_kv`, both set from the release config.
    `workload.prefix_kv` = auto | on | off (default auto, which follows the model). `on` is refused, with a warning,
    for models without causal_condition. With steps = 1 there is no reuse.
  - The block-causal mask is now applied even when reuse is off. For Qwen-Image-2.1-Turbo the DiT work goes from
    3019.8 to 3002.1 TFLOP per image with reuse off, and to 2976.8 with reuse on (the default). Denoise time goes
    from 32.50 to 32.14 s.
- **Text-encoder and VAE stages for the image DiTs**, built from the released text_encoder/ and vae/ safetensors
  headers. Coverage is now **full** for all 3 image models.
  - Qwen-Image: Qwen2.5-VL-7B text tower (290 tokens) + the Qwen-Image VAE (Wan-VAE decoder).
  - Qwen-Image-2.1-Turbo: Qwen3-VL-8B text tower (270 tokens) + the Qwen-Image-2.1 VAE (new `qi21` family: ×16,
    5 up blocks, 1152-wide).
  - FLUX.2-klein-4B: Qwen3-4B (512 tokens, all 36 layers run) + the FLUX.2 VAE (SD-VAE-style 2-D decoder).
  - Causal VAE decoders skip `time_conv` on the first latent frame, as the reference does (feat_cache "Rep"). A
    1-frame image decode therefore runs none of it; video (T > 1) is unchanged.
  - FLUX.2 workload: vae_s 8 + DiT patch 2. This matches the pipeline's 2×2 packing and gives the same 64×64 tokens
    at 1024², so the VAE input size is now correct.
- **Image-mode resolution rule.** The latent is 2·⌊px / (2·vae_s)⌋, so sizes round *down* to a multiple of 2·vae_s,
  as `QwenImage` / `QwenImage21` / `Flux2` pipelines do; a warning is given. Video keeps its ceil rule.

### Audit
- Adversarial audit of image mode (`val5/v0116/audit.py`): no defects found beyond the resolution rule above.
  - Latent, grid and sequence length match the reference rule over 8 sizes × 3 models.
  - CFG doubles the DiT FLOP and the forward sequences; per-request FLOP is invariant to batch and PP.
  - Denoise time is linear in steps; with reuse it equals t_x + (s − 1)·t_c.
  - The extract / cached FLOP of Turbo matches the closed forms (< 0.2 %).
  - Per-request FLOP (with reuse) is conserved under TP / SP / PP / DP, including uneven splits.
  - CP is rejected for gen models by design; use SP.

### Tests / suite time
- The suite-time creep from 535 to 701 s was the critical-path module. `test_core_scenario_search` contains the 4
  top-k branch-and-bound searches (156.8 s alone, unchanged since 0.113). It started late, because it was not in
  `SLOW_FIRST`, and stretched to 325 s under 8-way contention.
  - Fix: split it into `test_core_scenario_search_topk1/2` (2 cases each, assertions unchanged).
  - The fixed-output half of `test_core_057::test_refill_offset_and_cascade` moves to `test_core_057b`.
  - Heavy modules are scheduled first. Results are unchanged.
- New tests: `test_core_116`.

## [0.115.0]

### Added
- Image DiTs in the gen domain. The domain grouping is image *and* video generation.
  - Before this release, the gen path already evaluated a 1-frame clip mechanically (causal (F − 1)/t + 1 = 1 latent
    frame), but it had no image mode.
  - Image mode: an image is one latent frame (frames 1, fps 0, temporal patch 1, no temporal attention).
    - A `workload.frames` override on an image model is ignored, with a warning.
    - The summary workload carries `image: true`, and the UI shows 图像.
    - The domain label is now 图像 / 视频生成（DiT）.
    - Video summaries are unchanged, because the key is only present for image models.
- New releases at released dtype (BF16). Parameter counts match the safetensors headers exactly.
  - **Qwen-Image-2.1-Turbo** (Qwen, 2026-10-09): 7,115,124,736 parameters. https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo
    - New `qwenimage21` builder: 32 single-stream blocks over [text ‖ image] with shared QKV / out / SwiGLU weights,
      one shared timestep modulation and no per-block AdaLN tables.
    - The VAE is ×16 with z = 64 and a 1×1 patch. Defaults: 2048×2048, 8 steps, CFG 1 (README).
  - **Qwen-Image** (Qwen, 2025-08): 20,430,401,088 parameters. https://huggingface.co/Qwen/Qwen-Image
    - New `qwenimage` builder: 60 dual-stream MMDiT blocks with per-stream AdaLN.
    - The VAE is ×8 with a 2×2 patch. Defaults: 1328×1328, 50 steps, CFG 2.
  - **FLUX.2-klein-4B** (Black Forest Labs, Apache-2.0, not gated): 3,875,544,576 parameters.
    https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
    - New `flux2` builder: 5 dual-stream blocks + 20 single-stream parallel blocks (fused QKV + MLP-in, fused out),
      with shared modulation.
    - Defaults: 1024×1024, 4 steps, no CFG, text padded to 512.
    - FLUX.1-dev / schnell and FLUX.2-dev are gated, so they are not added.
  - In all three, the text encoder and VAE are not modelled. The coverage is therefore partial, with the DiT trunk
    only, as for any gen release without a pipeline entry. Text is counted in the joint sequence: 256 tokens for
    Qwen (an assumption, since prompt length varies) and 512 for FLUX.2 (the pipeline pads to this length).
- `summarize_release`: image DiT classes use the video role table, with new patterns (`_mod.N`,
  `stream_modulation`, `_mlp.`, `img_in` / `txt_in`). The roles of the existing video releases are unchanged.
- Standing tests: `test_core_115`. The release inventory goes from 82 to 85.

### Audit
- Adversarial audit of the LFM2 short-conv and Ling KDA builders: no defects found.
  - Work is conserved (FLOPs, 1e-6) over TP / ETP / EP / DP / PP, including uneven 3 / 5.
  - KV equals the closed form: LFM has 10 GQA layers; Ling has 9 MLA layers at 1152 B per token per layer, with the
    latent replicated per TP rank.
  - Recurrent / conv state is independent of the sequence length and Σ over ranks equals the global value (ceil
    bound at TP 3): LFM is 30 × h × 2 columns; Ling is 63 × (64 × 128² + 3 · 8192 · 3), fp32.
  - Decode and prefill state traffic is 2 × the state.

## [0.114.0]

### Added
- Coverage refresh. Official releases since early October 2026 with a public, non-gated config.json, at released
  dtype. Parameter counts match the safetensors headers exactly.
  - **LFM2.5-24B-A2B** (Liquid AI, 2026-10-12, BF16) and its official **-FP8** (compressed-tensors block FP8; the
    short-conv layers stay BF16): 23,971,587,712 parameters.
    - New `lfm2` / `lfm2_moe` builder: 30 of 40 layers are gated short convs (in_proj 3h, depthwise kernel 3,
      B·x / C·y gates, out_proj), and 10 are GQA with q / k RMSNorm.
    - The first 2 layers are dense SwiGLU; the rest are MoE with 64 experts, top-4, and an expert bias.
    - The conv layer runs as a linear core with `lin = "conv"`: state is (k − 1) columns per channel, and the vector
      work is the conv plus 2 gate products per element.
  - **Ling-3.1-flash** (inclusionAI / Ant Group, 2026-10-13, FP8 block-128): 563,946,483,520 parameters, plus an
    8,174,197,248-parameter MTP layer.
    - `bailing_hybrid` goes through the hybrid builder in groups of 8: 7 KDA layers (full-rank f_proj) and 1 MLA
      layer with a head-wise sigmoid output gate.
    - The first 4 layers are dense; the rest are MoE with 512 experts, top-8, 1 shared expert and an expert bias.
- `summarize_release`: LFM2 `conv.*` tensors map to the attn role.
- Standing tests: `test_core_114`. The release inventory goes from 79 to 82 (64 LLM / VLM).

### Audit
- Config drift: 78 stored configs match current HF main on every shared key. boltz-1 has no config.json, and the 3
  GitHub-only releases were skipped.
- Weight drift: tensor bytes vs the current safetensors sizes are unchanged for 75 releases. 4 were flagged, all
  artifacts: ESMFold and Boltz ship .bin / .ckpt, and the Mistral repos ship consolidated + sharded copies.

### Not added
- Qwen-Image-2.1-Turbo (2026-10-09): an image DiT, and image generation is not a modelled domain (video DiT only).

## [0.113.0]

### Fixed
- dtype exec matrix: fp32 inputs on hardware without fp32 are narrowed to the widest native format. They are now
  labelled `downcast` instead of `upcast_both`. Only the label changes; the conversion cost and every result are
  unchanged.

### Added
- `accel-dse eval --scenario FILE` evaluates a full scenario JSON: a UI export, an API `scenario`, or a body with
  `scenario` (+ `chip_preset`). The result is identical to `api_eval`. Before this, the CLI could not reach 22 chip
  micro-architecture fields beyond the presets.

### Audit
- Hardware catalog (val5/v0113/hw_audit.py), 0 violations:
  - 1,186 LPDDR5/5X/6 + HBM3/3E/4/4E combinations: bandwidth = bus × MT/s, capacity = stacks × height × die (HBM)
    or units × GB, mem-id round-trip, defaults in their lists, LPDDR6 payload 256/288.
  - Reference GPUs: R·C·E·2·f = datasheet BF16 peak, FP8 = 2× (MI300X 2614.9 is datasheet rounding), HBM
    bandwidth within 1 % and capacity within 8 % (marketed GB vs GiB stacks), link = bidirectional / 2.
  - D2D (UCIe-A 48/64, UCIe-S 32, BoW-256/512): lanes × rate / 8; NVLink-C2C 450 GB/s per direction.
  - Power and area are user budgets (no catalog values).
- CLI vs API: every one of the 154 scenario flags reaches the scenario hash (PD flags under `--pd`). The gap found is
  closed by `--scenario` above.
- State round-trip: 120 random scenarios (chip / PD / D2D / fabric / link overrides). Re-evaluating the API's echoed
  scenario gives the same hash and summary. The UI keeps no URL or localStorage state.
- Standing tests: `test_core_113`.

### Performance (bit-identical)
- `ir.rank_shares` builds the unsharded op graph once instead of twice (layout search 134 → 126 s). 152 randomized
  API outputs are byte-identical.

## [0.112.0]

### Fixed
- Radix prefix cache (`pd.prefix_tree`). A leading chain of single-node levels (one shared system prompt) is touched
  by every request, so under LRU it is always resident once its path fits. The Che branch gave it h = 1 − e^{−T} < 1,
  which made H₁ fall (1.0 → 0.9999) as capacity grew and the exact path handed over to Che. Those levels are now
  pinned, and Che runs on the remaining capacity.

### Performance (results bit-identical)
- `overlap.stage_pipe_time` is memoised on its full input, and the `_Sim.feed` loop keeps its state in locals.
- pdqueue: the victim-wait `r_surv` sum skips zero terms, `surv` is memoised, and `_gamma_q` is LRU-cached.
- bulkq: the `_wait_hists` and `bulk_law` caches go from 128 / 256 to 1024 entries (they were evicting within a
  single PD report).
- A PD kv_cap report takes 242 → 152 s (swap) and 242 → 190 s (recompute), with an identical SHA-256 of the full
  report. 152 randomized `api_eval` / `api_sweep` outputs are byte-identical to 0.111.

### Audit
- Randomized adversarial audit (val5/v0112) of spec_k, PP microbatches, MoE skew cap / measured load, prefix scenarios,
  the dtype exec matrix and radix tree vs `RadixLRU`.
  - 0 violations apart from the radix pin above.
  - Che vs LRU: worst error 0.14 on one adversarial tree (two 500-token roots with tiny leaves, where the exact
    enumeration is over budget). This is a known Che approximation, recorded and not fitted.
  - fp32 × fp32 on hardware without fp32 is labelled `upcast_both`, although it is a forced down-cast. Only the label
    is affected; the cost is correct.
- Standing tests: `test_core_112`.

## [0.111.0]

### Added
- `pd.ttft_resid` (opt-in; `--pd-ttft-resid`; Web "decode 残余"), affecting the colocated prefill-first TTFT.
  - `full` (default, unchanged): every arrival waits half a decode step at the median and a full step at
    p90 / p99.
  - `busy`: the arrival waits only when an iteration is in flight. With prefill idle and decode non-empty,
    p = (1 − ρ)(1 − π₀) from the decode birth–death law, and R = Bern(p)·U(0, step) is convolved with the TTFT law.
  - Against the DES (60 cases, 6,000 requests each), `busy` improves mean 6.1 → 1.3 %, p50 8.5 → 4.7 % and p90
    10.9 → 2.0 %. It worsens big-model p50 (3.7 → 4.7 %) and CV 1 p99 (4.3 → 5.5 %), so it is not the default.
- DES vs analytical audit over Carbon LLMs (30 scenarios, 90 mode pairs). The errors are in line with the qwen3-8b
  reference (median 1–4 %), and genomics / protein / video encoders ignore PD with a warning.
- Standing tests: `test_core_111`.

### Investigated (not changed)
- r3 lat (vLLM 0.11.2 MI300X Llama-70B FP8 TP8, static batch). pred / meas falls from 1.0 at batch 1 to 0.46 at
  batch 128, both for ISL 128 and ISL 2048. The decode-step gap grows about linearly with batch (≈ 0.1–0.16 ms per
  sequence per step), while the KV, attention and link terms are small. This points to per-sequence host / sampling
  overhead or ROCm paged-attention efficiency. No public per-sequence figure exists, so nothing was modelled.
- v1 ttft (TRT-LLM 0.8, batch 1): short prompts give 0.59–0.79. CTA-tiled GEMMs (all GEMMs, or dense prefill
  only) were probed and rejected, because long prompts overshoot: v1 ttft 27.1 → 40.9 %.

## [0.110.0]

### Fixed
- Energy: `Chip.sfu_per_cycle` inflated the `vec` action count of SFU-bound ops (softmax / sigmoid-heavy folding
  and genomics models), because vector work was taken from the op's max(lane, SFU) time. Vector work is now the
  lane work. The SFU still sets the time.

### Added
- MODEL §56: a consolidated validation summary of all 10 groups for 4 variants (error, median and range, share
  within ±25 %), from public assumptions only.
- Energy / sweep audit over genomics, protein and Carbon LLM with the current defaults (`val5/v0110_audit.py`).
  Sweep rows equal eval (energy included), throughput is monotone, and time-only knobs leave counts unchanged.
  Replicated DAP ops are charged per rank, while useful work is conserved.
- Standing tests: `test_core_110`.

## [0.109.0]

### Changed
- extval refhw candidate (`candidate_variant`): prefill runs with two-batch overlap (`exec_overlap = "tbo"`) for
  the sources whose published prefill configuration overlaps EP dispatch / combine with compute (`TBO_PREFILL_SRC`:
  SGLang large-scale EP blog 2025-05, DeepSeek profile-data). Decode and every other source are unchanged.
  - moe pf: 48.2 → 8.5 %. The other nine groups are identical, `catalog` is unchanged, and the NPU fingerprints are
    unchanged (extval only).
  - The opt-in `tbopf_variant()` applies the same rule to any base variant.

### Fixed
- `/api/eval` (and every `result_dict` consumer) reported latency metrics that do not apply to the result as
  `0.0` ms, while `/api/sweep` rows reported them as null. The affected metrics:
  - decode: `ttft_ms`, `latency_ms`;
  - LLM prefill: `tpot_ms`, `latency_ms`;
  - encoders and video: `tpot_ms`, `ttft_ms`.

  They are now null in the API, as in sweep, so exported JSON / CSV of eval and sweep agree. The core
  `Result.summary()` and the fingerprints are unchanged.

### Added
- Randomized UI / API audit (`val5/v0109_api_audit*.py`):
  - Body fuzz over 7 POST routes plus malformed raw bodies: only 200 / 400, never 5xx, and JSON always finite.
  - Sweep rows, Pareto points and `/api/sensitivity` all equal `/api/eval`.
  - CSV export (`csvCell` / `expCSV`): 3,000 adversarial cells round-trip through a CSV parser.
- Standing tests: `test_core_109`.

### Investigated (not changed)
- moe static (one row, TRT-LLM DeepSeek-R1 H200 EP8, attention DP; predicts 1.82×). Decode is DRAM-bound on weight
  streaming. No documented mechanism (two-batch overlap, imbalance, padding) explains the gap without fitting, so
  it is left as is.
- Per-node deduplicated DeepEP dispatch is not modelled. It would raise the pf rows further.

## [0.108.0]

### Added
- `Scenario.kv_mem_eff` (opt-in, default unset): DRAM efficiency of the KV-cache read / write streams. When unset,
  KV bytes run at `mem_eff` and every number is bit-identical (fingerprints unchanged). Only time changes; bytes and
  energy do not. extval `kveff_variant()` uses `KV_STREAM_EFF` = 3000 / 3350, FlashMLA's published H800 figure
  (not fitted). It improves r4 18.1 → 16.5 % and v1 max 18.3 → 13.9 %, but worsens moe dec / moe max / r3 max /
  r3 lat / v1 ttft, so it is **not** a default.
- Randomized audit of VLM vision encoder and video DiT under TP / SP / PP / DP / CP, D2D tiers and SLC policies
  (`val5/v0108_audit.py`, 2 runs, ~1,000 cases). Results:
  - Vision work is conserved; tflop = batch × images × per-image tflop.
  - CP with images is rejected, and decode does no encode.
  - DiT work is conserved; CP is rejected for video with an explanation.
  - D2D is monotone in grade and units, and `d2d_enabled = false` is inert apart from its warning.
  - `pin_bw` is never slower than `pin` and is monotone in SLC size.
- Standing tests: `test_core_108`.

### Fixed
- A slower-than-DRAM SLC under `slc_policy = "lru"` made a larger SLC slower without any warning. The `pin` warning
  now covers `lru`, with policy-specific text.

### Investigated (not adopted)
- r4 long-sequence under-prediction:
  - Bisecting the batch to the true KV capacity: r4 got worse, 18.1 → 18.4 %, and DeepSeek long-OSL overshot to
    1.3–1.6×.
  - CTA-tiled prefill GEMMs: r4 got worse, 18.1 → 19.9 %, and MoE overshot. The prototype was removed.

## [0.107.0]

### Fixed
- `attn_mode = "bidir"` with context parallelism (`layout.cp > 1`) charged each CP rank the causal zigzag share
  (S²/(2·cp)) of the attention. Bidirectional work was lost as cp grew: cp 4 at 32K kept 68–78 % of it (Carbon-3B:
  590 → 442 TFLOP). Now each rank's ⌈S/cp⌉ queries see all S keys, so work is conserved to 1e-6 under CP, TP, PP
  and their combinations. Causal CP is unchanged.

### Added
- Genomics adversarial audit (`val5/v0107_audit.py`), 173 checks with 0 violations after the fix:
  - Carbon-A windows/s/card is monotone in frequency, MAC efficiency, memory efficiency, link bandwidth and SRAM
    (60 sweeps).
  - Window accounting holds for L = 1 … 65,536 (overlap < L, new bp = (L − overlap)·6, rates consistent).
  - Encoder work is conserved over TP / PP / SP / DP (30 points; CP is rejected).
  - bidir LLM prefill is conserved under TP / PP / CP (30 points).
  - Sweeps equal `api_eval`; Pareto fronts are non-dominated and sorted; sensitivity is finite (15 each).
- Standing tests: `test_core_107`.

## [0.106.0]

### Added
- Genomics model group (vendor "Hugging Face Bio", `HuggingFaceBio/*` releases fetched with
  `scripts/fetch_hf_release.py`). Parameter counts match the releases exactly for all four models.
  - `carbon-a-1.2b` (Carbon-A, Generanno), new domain `genomics`. It is a bidirectional token-classification
    encoder: 26 layers, hidden 2048, GQA 32 / 4 heads, head_dim 64, SwiGLU 5632, vocab 4128 (6-mer tokens), no KV
    cache. The output is two per-token strand heads `score.{0,1}`. Their release shape is hidden → 12, i.e. 6 bases
    × 2 classes (CDS / non-coding) per token, not hidden → 2. There is no vocabulary MLM head (new
    `ModelSpec.vocab_head`).
  - Carbon-A workload: 16,384-token windows (98,304 bp) that overlap by 1,024 tokens. The KPIs are windows/s/card,
    tokens/s/card and new bp/s/card (92,160 new bp per window).
  - Carbon-A weights and activations are BF16, the model card's inference setting (BF16 + FlashAttention-2). The
    safetensors are fp32.
  - `carbon-500m` / `carbon-3b` / `carbon-8b`: stock `LlamaForCausalLM` DNA LMs, released in BF16, evaluated as LLMs.
    They are appended after the domain releases, so the catalog order and fingerprints are unchanged.
- `Scenario.attn_mode` (`reference` | `bidir`; CLI `--attn-mode`; UI 注意力掩码 select). `bidir` is a what-if: every
  causal self-attention becomes non-causal (full S² scores, so qk / pv FLOPs are exactly 2× causal at ctx 0), and
  prefill writes no KV cache. It is prefill-only (decode raises) and is ignored, with a warning, for models that
  already run a full-sequence forward. The LLM prefill path now honours `AttnCore.causal`.
- UI / API: `genomics` in the domain list (`/api/health`), with its own badge, window / Mbp/s KPIs, window-length
  input and seq_len sweep.
- Tests: `test_core_106` (bidir = 2× causal attention FLOPs and no KV write; reference identical; dense causal
  fingerprints c034bfe3… / e51c7ec1… unchanged; Carbon params, dtypes, heads and workload; API).

## [0.105.0]

### Added
- `extval.SOURCE_ESTIMATOR` / `estimator()` / `est_variant()`: a batching estimator per source for the max-load
  rows, set from each source's published methodology (not fitted). Every max-load source (TRT-LLM 0.17 / 0.21 /
  0.21-b200 / 1.2.0 `trtllm-bench throughput`, and ROCm vLLM 0.11.2 offline throughput) is an in-flight /
  continuous-batching engine fed the whole dataset at once, so all of them get the continuous-batching proxy with
  the engine's default chunk budget and sequence cap:
  - TRT-LLM: `max_num_tokens` 8192.
  - vLLM `LLM` class on MI300X: `max_num_batched_tokens` 16384, `max_num_seqs` 1024 (`get_batch_defaults`).

  **Not adopted as the default.** On cand093, r3 max goes from 28.5 to 27.9 and v1 max from 18.3 to 18.1, but
  moe max goes from 12.9 to 18.4 and r4 from 18.1 to 24.5. The continuous-batching proxy lowers max-load
  predictions; that fixes H200 DeepSeek (2.44× → 1.19×) but deepens the r4 rows that were already under-predicted.
  `predict_cb` gains a `max_seqs` cap.
- Randomized audits:
  - CP: 40 points, dense and MoE (ep = tp·cp), cp 1/2/4, S 8K–131K; work is conserved to 1e-6.
  - Sweep vs eval: 20 sweeps, every row exactly equal to `api_eval`.
  - Pareto: 20 fronts, all non-dominated.
  - PD monotonicity (60 points): fluid capacity is monotone in KV bandwidth, cards, prefix hit and link
    bandwidth (0 violations). At a fixed absolute arrival rate, the queueing TTFT (mean / p90 / p99) never rises
    with a prefix hit, link ×2 or KV bandwidth ×4 (0 violations). The headline fluid `ttft_ms` can rise (38 cases),
    but only because faster prefill lets `best_prefill` pick a larger batch under the TTFT SLO; at an unchanged
    batch it never rises. Not a bug; now a standing test. See MODEL §51.

## [0.104.0]

### Added
- `Chip.attn_vec_overlap` (opt-in, default False, so the NPU fingerprint is unchanged). In the kernel and tbo
  modes, the flash-attention kernel's softmax / exp now overlaps its MMAs: the attention term becomes
  max(array, vector, KV), as in FlashAttention-3 (Shah et al. 2024: warpgroup ping-pong plus intra-warpgroup
  pipelining). Other vector kernels stay serial. Added `StageTime.t_vec_attn` / `attn_ovl`.
- `extval.fa3_variant()` turns this on for H100, H200, H800 and B200; A100 (FA2) and MI300X keep the serial
  softmax. **Not adopted as the refhw default:** the 10 groups are mixed.
  - catalog_kernel + fa3: r3 max goes from 36.2 to 43.6 and moe max gets worse; moe pf/dec and v1 max get better.
  - cand093 + fa3: r4 improves from 18.1 to 16.7 and v1 max from 18.3 to 14.6; r3 max worsens from 28.5 to 33.2.
  - The SFU rate re-evaluated with the overlap on (fa3+sfu vs fa3) is still mixed: moe pf and r4 get worse. It
    stays opt-in.
- A randomized invariant audit of the chunk MoE experts (149 points, 0 violations; `val5/v0104_moe_rand.py`).

## [0.103.0]

### Added
- `Chip.sfu_per_cycle` (opt-in, default None, so the NPU fingerprint is unchanged): transcendentals (`Op.sfu`)
  run on a separate SFU pipe next to the vector lanes, and an op takes the slower of the two. Counted per element:
  softmax 1 exp (LLM, DiT, protein, DSpark), SiLU activation 2 (exp + reciprocal), norms 1 rsqrt per token, and
  sampling 1 exp per logit.
- `RefHW.sfu_per_sm`: 16 for the NVIDIA references (CUDA C++ Programming Guide, native arithmetic throughput,
  cc 7.x–10.0). MI300X is left unset because AMD publishes no per-CU transcendental rate.
- `extval.sfu_variant()`. **Not adopted as the refhw default.** Under the default variant (catalog, stage
  overlap) all 10 groups are unchanged. Under the kernel and candidate variants the results are mixed: moe pf,
  moe dec / r4, and v1 max get worse.

### Changed
- A co-located chunk now reads the extra MoE experts its tokens touch, using the uniform-routing union of decode
  and chunk tokens (`evaluate.union_routing` / `extra_expert_bytes`). The pipe replay uses the merged expert GEMM
  (hit experts, rows per expert). This applies in every overlap mode. Example: qwen3-30b-a3b with a decode batch
  of 1 plus a 128-token chunk now streams ~all 128 experts, 3.5–9.7× the decode step (was 1.0–1.3×). Dense models
  are unchanged.

## [0.102.0]

### Changed
- Under `exec_overlap = "pipe"`, the co-located chunked-prefill iteration (`pdqueue._fused_step`, which the DES
  and `extval._fused` also use) is now replayed op by op by `evaluate.pipe_fused_time` / `overlap.merge_chunk`.
  This replaces the 0.101 bracket interpolation. Each GEMM merges into one GEMM of m_dec + round(f·m_pre) rows, so
  the weights are read once. The chunk's flash-attention step comes after the decode's and re-reads rr × the
  prompt's KV writes. The chunk's vector ops and collectives follow the decode op of the same name, scaled × f. A
  prefill-only GEMM (DeepSeek non-absorbed MLA) runs as the chunk's own GEMM at its real shape. The serial bound
  of the fused step now uses the merged ops' own array and vector seconds, because a chunk GEMM costs more than
  f × the prompt GEMM. The interpolation remains only as a fallback when the layer groups do not line up, which
  happened at 0 of 72 audit points. Results are memoised per stage pair, so DES runtime stays the same as 0.101.
  The NPU single-point fingerprint is unchanged (e51c7ec1); PD coloc_chunked numbers under pipe change.
- Added a warning when weight conversion (dequant) takes longer than a stage's DRAM time.

### Added
- `core/vecsanity.py` checks catalog presets against public datapoints (H100, A100, MI300X, TPU v4/v5p,
  Ascend 910) for the MAC:vector-lane ratio and int4 dequant headroom, with citations. Presets are not changed.

## [0.101.0]

### Fixed
- **Co-located chunked-prefill iterations ignored the overlap mode** (`pdqueue._fused_step`). The fused
  decode + chunk iteration always used the stage rule. Under the 0.100 pipe default, chunked co-location was
  therefore priced with full overlap: qwen3-8b TP2 gave 66.06 req/s under stage and the same 66.06 under pipe.
  It now follows the scenario's mode:
  - stage: unchanged, bit-identical;
  - kernel-serial modes: components add, as `extval._fused` did since 0.70; extval now delegates to this function,
    and its numbers are identical;
  - pipe: the fused op sequence is not replayed. The iteration keeps the decode step's position between the stage
    and serial bounds 「假设」, which is exact at f = 0.

  qwen3-8b: stage 66.06 > pipe 63.06 > serial 37.40 req/s. The NPU fingerprint is unchanged (e51c7ec1; it does not
  cover PD) and refhw is unchanged (10/10 groups identical).

### Validation (pipe default, val5/v101)
- **Feature combos:** 206 random scenarios over PD, CP, PP, TP / SP / EP, D2D, SLC pin / lru / pin_bw, fabric,
  video and protein. No pipe stage outside [stage, serial] and no errors.
- **Energy:** action counts are identical across overlap modes (rel 1e-9); only idle card-seconds (static energy)
  grow with the slower step.
- **Feature monotonicity:** link, D2D, SLC size / bandwidth, SRAM and mem_eff: 0 violations.
- **Resource monotonicity:** 3 × 300 random pipe-weighted points. 4 violations, all the known 0.95 artefact: the
  `mem` step lpddr6_16x96 → hbm3e_4s lowers capacity, so the model no longer fits.
- **Knees and the 0.95 constant table** re-run under the default: MODEL §47.

## [0.100.0]

### Changed — default numbers change (intentional)
- **`Scenario.exec_overlap` defaults to `"pipe"`** (hardware-derived overlap, MODEL §44/§45), replacing `"stage"`.
  Under stage, design conclusions such as the SRAM knee and the irrelevance of link bandwidth were artefacts of
  assuming full overlap. `stage` stays selectable (`--exec-overlap stage`, `"exec_overlap": "stage"`).
  Refhw validation is unchanged because extval variants set their overlap explicitly: all 10 groups are identical
  to 0.99.
- **Fingerprint (209 catalog points).** Default c034bfe3 → e51c7ec1. With `exec_overlap: "stage"` the numbers are
  still c034bfe3 (pinned in test_core_0100). Of the 209 points none is bit-identical, 5 change by more than 1 %,
  the median ratio is 1.000, the worst is −3.0 % (wan2.1-1.3b TP2), and no bottleneck flips.
- **Saved scenarios.** A partial dict without `exec_overlap` keeps its hash (e.g. `{"model": "qwen3-8b"}` →
  917ab559e6aa1c07) but now evaluates under pipe. Full dicts saved by the UI carry `"exec_overlap": "stage"` and
  reload as stage, so they reproduce their numbers exactly.
- **Sensitivity.** For a pipe base the panel shows stage (吞吐上界) and serial (吞吐下界). Sweep / Pareto get a
  new "全重叠（上界）" case.
- **Pipe speed.** Inside a run of identical tiles the simulator now skips ahead once the state repeats. It is
  checked only around powers of two, and short runs are replayed. Difference from the 0.99 replay is ≤ 4.3e-12.

### Correction to 0.99
- The 0.99 claim that the exact replay is bit-equal to brute force on 189 points was wrong. That comparison ran
  `python -c` from inside one worktree, so both sides imported the same code. Re-run from a neutral directory:
  the 0.98 extrapolation differed from the exact replay by at most 9.9e-5 (relative).

### Tests re-pinned to `exec_overlap="stage"` (they hand-calculate or pin stage numbers or identities)
- test_core_051, 052, 059, 060, 062_1, 077, 089, 096, 097, and coverage `_kpi`.
- 051 found that "chunked prefill moves no more DRAM per token than prefill-first" holds under stage but not under
  pipe (+5 %). Under pipe the chunked iterations are slower, the running decode batch is smaller, and weight reads
  are amortised over fewer tokens. The relation depends on the regime; it is not an invariant.
- 052: the chunk-share identity is exact only for additive step costs (stage); under pipe it is off by 1.8e-6.

## [0.99.0]

### Changed (opt-in `pipe` only; NPU default and fingerprint c034bfe3 unchanged)
- `exec_overlap = "pipe"` is now replayed exactly, with no extrapolation. A streaming flow-shop simulator advances
  repeated layers one at a time and jumps ahead only once the state, relative to the compute clock, repeats; that
  jump is exact for a deterministic flow shop. On 189 catalog points (base, TP2, SRAM 256 MiB) the result is
  bit-equal to brute-force replay. The worst evaluation takes 1.9 s (protenix, SRAM 256 MiB); DiT takes ≤ 0.3 s.
  Tile sizes are whole bytes, so the window sums are exact (replaces the 0.98 float tolerance).
- DMA tile = min(1 MiB, buffer / 2): double buffering needs two tiles. A larger tile used to serialise
  load and compute (q8 decode b1: 1.11× → 1.76× stage at a 2 MiB tile).
- Sensitivity panel / sweep / Pareto: the overlap alternatives are bracketed. The panel shows
  `pipe` first (现实估计), then `serial` (吞吐下界), and `stage` (吞吐上界) when the base is not stage. New
  `base_label` field. Sweep cases are "pipe（现实估计）", then "无重叠（下界）".

### Validation
- Knees from 0.97 under pipe (val5/v099/knees.log):
  - **Robust:** HBM decode clock (0.8 GHz), FP8 decode clock (1.5), batch (128), mac_eff (0.9).
  - **Shifts:** LPDDR prefill SRAM 64 → 128 MiB; D2D package 1 → 2; DeepSeek-V3 link 50 → 400 GB/s (the link
    curve is flat under stage); Wan frames best 33 → 17.
- Sensitivity of pipe to its constants (tile 256 KiB … 4 MiB, minimum 8 … 128 tiles; 7 cases × cross-op on / off):
  ≤ 0.7 % change (val5/v099/consts.log). The one large effect was tile > buffer / 2, which is now capped.

## [0.98.0]

### Added (opt-in; NPU default `exec_overlap = "stage"` and fingerprint c034bfe3 unchanged)
- `exec_overlap = "pipe"`: overlap derived from the hardware instead of all-or-nothing (core/overlap.py). The stage's
  op sequence is replayed on a DMA engine and a compute engine:
  - DMA runs ahead of compute by at most the SRAM left for streaming (≥ the plan's 2 MiB staging area);
  - ops are tiled (≥ 32 tiles, 1 MiB DMA tiles), so prologue and epilogue are exposed;
  - flash-attention fusion; vector ops depend on the op before them;
  - collectives sit on the compute chain while prefetch continues, and MoE dispatch / combine hide behind the
    shared-expert compute;
  - PP hand-offs are exposed and FSDP gathers are prefetched.

  The result is bracketed by `stage` and `serial` and is monotone in SRAM (tiling fixed by the op; float-tolerant
  buffer test).
- `Chip.pipe_cross_op` (default True; False = GPU kernel boundaries, no prefetch across ops).
- `Chip.dma_engines` / `dma_outstanding_KiB` / `dram_latency_ns`: DRAM bandwidth capped by Little's law
  (all three or none; every mode).
- `extval.pipe_variant(base, cross_op=False)`. Sensitivity panel: a `pipe` row (hardware-derived overlap) next to
  the serial one. CLI: `--exec-overlap pipe`.

### Validation (not adopted for refhw: not every group improves)
- Random bracket test over 187 stage points (5 models, 3 chips, PP / TP / EP / fabric / D2D, prefill / decode):
  never outside [stage, serial] before the clamp; median position 0.06 of the way from stage to serial, p90 0.37.
- SRAM sweeps 1 MiB … 1 GiB (3 models × HBM / LPDDR × 3 phases × cross-op on / off): 0 monotonicity violations.
- Refhw, 10 groups (mean |log|, %), catalog_kernel → catalog + pipe (kernel boundaries):
  moe pf 57.8 → 55.2, dec 33.4 → 33.5, static 30.0 → 29.7, max 17.1 → 17.2; r3 max 36.2 → 43.4, lat 118.4 → 118.4;
  r4 26.7 → 28.3; v1 static 24.0 → 24.8, ttft 50.0 → 48.6, max 20.4 → 18.7. Candidate + pipe vs candidate: 7 of 10
  improve (r4 18.1 → 16.6), but r3 max 28.5 → 33.0 and moe dec 5.4 → 5.7 worsen. Neither is adopted.
- If pipe were the NPU default: fingerprint c034bfe3 → f7963e41. 8 of 89 catalog points change by > 0.1 %,
  at most −3.0 % throughput.

## [0.97.0]

### Fixed
- `Chip.sram_port_Bpc_per_engine` (0.96) crashed every GEMM that took the engine-group split (`mapping._gemm_cost`
  rebuilt the sub-chip with both port fields set → ValueError). The sub-chip now clears the per-engine field. Default
  untouched; the 0.96 test never evaluated a scenario with the field set.

### Added
- Sweep / Pareto sensitivity (opt-in: `"sensitivity": true` on `/api/sweep` and `/api/pareto`; UI 敏感度复核 checkbox).
  Every point is re-run under the same alternatives as the eval panel (no overlap; DRAM efficiency 0.5 / 0.9; ¼ of the
  vector lanes), each applied to the swept point itself. Sweeps report, per alternative, the best values (ties within
  1 %), the knee (first value within 5 % of the best), discordant pairs (> 1 % both ways) and bottleneck flips, with a
  flag on every affected row. Pareto reports base points that leave the front and new ones that join it. Flags appear
  in the tables and in the CSV / JSON exports. An alternative that is swept itself is skipped.
- `/api/sensitivity`: the eval sensitivity block on its own. The UI now renders the main result first and fills the
  敏感度 panel after it. `/api/eval` still includes the block by default. The block now carries `scope` (PD pools not
  re-evaluated) and uses the domain unit in the UI.

### Validation
- FP8 attention against FlashAttention-3 (H100 SXM5): FP16 up to 740 TFLOPS, FP8 ≈ 1.2 PFLOPS, so ≈ 1.6× at hd 256.
  The H100-like preset gives 1.76× with no softmax overlap and 2.0× with full overlap at hd 256, and 1.61× / 2.0× at
  hd 128: the measured ratio falls inside the bracket. Absolute TFLOPS on this preset (959 bf16) exceed FA3's 740
  because the preset's mac_eff is not FA3-calibrated; that is a property of the preset, not of the FP8 path. FA3 fuses
  Q quantisation into RoPE (free); the model charges one vector op per Q element (a slight over-count).
- Sensitivity latency (cold, server): qwen3-8b eval 5.5 → 10 ms, DeepSeek-V3 prefill 15 → 37 ms, DiT 99 → 498 ms.
  Hence the separate endpoint.

## [0.96.0]

### Added (opt-in; NPU catalog fingerprint c034bfe3 unchanged)
- `Chip.attn_fmt = "kv"`: native KV-format attention. When the KV format is native, QK / PV run at its MAC rate with
  Q quantised on the vector path and P cast in the softmax epilogue (the FlashAttention-3 FP8 recipe); there is no
  per-step KV up-cast. Default "" = bf16 math with the KV up-cast. Exposed in the UI (注意力计算) and the CLI (`--attn-fmt`).
- `Chip.sram_port_Bpc_per_engine`: SRAM port width per engine, scaled by `engines`, so engine sweeps and die splits
  carry a per-die port. Exclusive with the chip-wide `sram_port_Bpc`. UI + `--sram-port-per-engine`.
- 敏感度 panel / `sensitivity` block in `/api/eval` (on by default; `"sensitivity": false` or `--no-sensitivity` turns
  it off): the same scenario under no overlap (serial), DRAM efficiency 0.5 / 0.9 and ¼ of the vector lanes, with the
  ratio, the bottleneck, flips, and the constants the result hinges on (|Δ| > 15 % or a bottleneck flip). The CLI prints
  one 敏感度 line when something hinges.

### Unchanged (documented)
- `mem_eff` stays 0.7 for both HBM and LPDDR. Public streaming measurements put both at about 80–95 % of peak:
  H100 HBM3 BabelStream Triad ≈ 87–89 % (Godoy et al. 2025); Grace LPDDR5X STREAM Triad 80–95 % (NVIDIA Grace
  benchmarking guide), up to 98 % (NVIDIA Grace blog), 90.6 % independent (GH200, 480 GB). That does not support a
  type-specific default. 0.7 is an inference-traffic effective value, not a STREAM value (MODEL §42).

## [0.95.0]

### Fixed
- `FormatSupport.dequant_elems_per_cycle` was validated and documented but never read: format conversion always ran on
  the vector lanes. It now sets the conversion throughput; the default (None = vector lanes) is bit-identical.

### Validation (NPU design space, MODEL §41)
- Randomised monotonicity check (≈ 890 pairs: SRAM / engines / clock / memory bandwidth / SLC size + bandwidth,
  stage / kernel / serial overlap, dense / MoE / linear-attention, decode / prefill): no violations. The only flags were
  memory steps that lose capacity (LPDDR6 256 GB → HBM3e 144 GB no longer fits).
- Every bottleneck flip in the SRAM / engines / partition / memory / clock / DRAM-efficiency sweeps is explained by a
  roofline crossing (§41).
- Conclusion sensitivity to 「假设」 constants is tabulated. The constants that drive conclusions are exec_overlap,
  mem_eff, mac_eff and vector_lanes (with FP8 KV); acc_kib, gemv_macs, noc_hop, sram_port (×4) and d2d units do not.

## [0.94.0]

### Added (all opt-in; NPU catalog and refhw defaults bit-identical)
- `Scenario.mem_cap_GB` (usable DRAM per card, GB) and `Scenario.kv_mem_frac` (KV / state share of post-weight free
  memory). `extval` "mem_rt": published per-GPU capacity (H200 141 GB, B200 180 GB vs 144 / 192 GiB stacks),
  TRT-LLM free_gpu_memory_fraction 0.9, vLLM gpu_memory_utilization 0.9.
- `Chip.fmt_ceiling` (sustained fraction of peak per exec format, or per `<kind>:<fmt>`). Hopper values from public
  kernel benchmarks: FP8 block-scaled GEMM 1550 / 1979 (DeepGEMM), MLA decode attention 660 / 989.5 (FlashMLA).
- `extval.candidate_variant()` now bundles these. Still not the default: it improves 9 of 10 validation groups (moe
  decode 33.4 → 5.4 %, r4 hold-out 26.7 → 18.1 %) but the single-row moe `static_tok_s_total` (trt-dsr1-h200) goes
  30 → 60 %.

### Validation
- DeepGEMM's public H800 table replayed through the GEMM model (`val5/deepgemm094.py`): `wide` gmean 1.27×, `cta` 1.61×
  — Hopper FP8 compute is over-rated without the ceiling; memory-bound shapes agree with DRAM efficiency 0.7.

## [0.93.0]

### Fixed
- External validation formats per checkpoint / hardware from public docs (not fitted): gpt-oss on Blackwell runs MoE as
  MXFP4 weights × MXFP8 activations (TRT-LLM gpt_oss README, W4A8 / UTCQMMA) — was bf16 activations with weight dequant;
  `nvidia/DeepSeek-R1-0528-NVFP4-v2` keeps FP8 attention projections (model card) — was all-NVFP4. r4 (hold-out) MAPE
  31.2 → 26.7 %; v1 / moe / r3 identical.

### Added
- Per-role GEMM activation formats (`act_<role>` in formats overrides, `ModelSpec.act_fmt_for`); catalog unchanged.
- Opt-in `Chip.attn_sched` (schedule for attention ops only, e.g. `"cta"`); default empty = bit-identical.
- `extval.TRACE_LAYER_US` = 62.3 µs/layer from the public DeepSeek profile-data decode trace, and
  `extval.candidate_variant()` (attention `cta` + that overhead). Not default: improves 9 of 10 validation groups
  (r4 26.7 → 19.2 %, r3 latency 118 → 33 %) but worsens the single-row moe `static_tok_s_total` group (30 → 63 %).

## [0.92.0]

### Added
- 参考硬件的 GPU kernel 模型（可选，见 MODEL §38）：`Chip.instance_sched = "cta"`（线程块 tile / wave 量化、split-K、epilogue，tile 集合取自 NVIDIA 矩阵乘用户指南）和 `Chip.kernel_us`（每个 kernel 的固定代价；参考值 0.53 µs 来自 NVIDIA CUDA Graphs 博客）。通过 `extval.gpu_variant` 选用。
- 采纳检验脚本与结果表（r4 TRT-LLM 1.2 作为留出测试集，不在上面拟合）。

### Changed
- 外部验证的满载吞吐行（max_tok_s_*）：放不放得下改为按 in-flight 批检查。prefill 激活按一个 chunk（8192 token），常驻请求的 KV 仍按 decode 点检查。moe 集误差 18.1 → 17.1 %，r4 32.6 → 31.2 %，v1 / r3 不变；静态批行不变。

### Validation
- cta 和 kernel_us **不**作为参考硬件默认：吞吐在 v1 / r3 / moe 上偏乐观 1.2–1.8×，误差变大。公开的 launch 下限（每层约 5 µs）解释不了约 100 µs / 层的软件开销残差。参考硬件保持 wide。
- NPU 目录芯片的默认数值逐位不变（209 个场景的指纹 c034bfe3 一致）；`kernel_us` 是 omit_default 字段，场景 hash 不变。

## [0.91.0]

### Added
- SLC 策略 `pin_bw`（可选；Web「SLC 策略」/ CLI `--slc-policy pin_bw`）：钉住方式与 pin 相同，但把包含式 SLC 当作第二条读路径，命中按 SLC / DRAM 带宽均衡分流，访存时间 = T / (B_dram + B_slc)。不比 pin、也不比没有 SLC 更慢，SLC 越大越不慢。默认 pin 不变（普通缓存的命中就由缓存服务）。
- 外部验证集 r4：TensorRT-LLM 1.2.0 perf-overview，B200 FP4 / H200 FP8，6 个模型，38 行，新指标 `max_tok_s_gpu`（满载每卡输出 tok/s，DEP = 注意力 DP + EP）。
- 诊断变体 `extval.sched_variant`（Chip.instance_sched = auto）。

### Fixed
- 外部验证的 FP8 / NVFP4 格式覆盖漏了路由专家（`expert`），MoE 专家 GEMM 仍按 bf16 计算。moe 集吞吐误差 21 → 18 %，r4 59 → 33 %。gpt-oss 行用发布的 MXFP4 权重 + FP8 KV。

### Validation
- 参考芯片的 `instance_sched = wide` 让大 prefill GEMM 利用率只有 12–24 %；`auto` 约 99 %，但吞吐会偏乐观 1.2–1.75×（缺 kernel 开销）。保留默认，详见 MODEL §37。
- MLPerf Inference v5.1 / v6.0 没有每次运行的 ISL / OSL，仍只作背景；GB200 / B300、405B、gpt-oss H200「FP8」不纳入（原因见 MODEL §37）。
- 默认数值与 0.90 逐位相同（209 个场景的指纹一致）。

## [0.90.0]

### Added
- Web「扫描 · Pareto」页：在 fabric 开启 + D2D 芯粒 + 算法不是 auto_overlap 时，提示带宽敏感性分析应改用 auto_overlap（扫描带宽 / α 类参数时措辞更强），并提供一键「切换为 auto_overlap」。
- 警告：SLC 带宽低于 DRAM 时，pin 策略读得反而更慢。

### Fixed
- 视频 / 蛋白模型开启 PD 时以前被静默接受：现在给出警告并按单一部署评估，API 不再附带 `pd: {error}` 块。
- 视频 / 蛋白模型设置 layout.cp 时，报错信息现在说明原因，并指向 sp（Ulysses / DAP）。

### Validation
- 视频 / 蛋白对抗：17 个模型、196 个随机场景（D2D / fabric / SLC / node_cards / exec_overlap / layer_overhead），不崩溃，输出有限，结果确定，能耗分项之和等于总量。
  - 结构模型的 MAC 计数随 DAP 线性增加，正好是 1 + f·(sp−1)，即真实的 single / atom 轨道重复计算。
  - 固定算法因为协议仍逐次选择，同样对带宽不单调；auto_overlap 单调。
- PD 扫描（0.89 未完成的部分，可续跑）：200 / 200 完成（148 个 prefill 池用 CP），0 崩溃、0 ValueError、CP 标签全部正确；33 个 prefill 容量 = 0 都是 prefill 池放不下（fits = False 且带明确警告）；72 个 coloc.ttft_ms / tpot_eff_ms = inf 都是合并部署放不下（API 序列化成 null）。没有发现新问题。
- 默认指纹：372 个点逐位不变。UI：local/ui_check_swhint.py 0 个问题。

## [0.89.0]

### Added
- Web：映射对比、参数扫描、Pareto 结果可以导出 CSV / JSON（「导出 CSV」「导出 JSON」按钮）。
  - CSV：UTF-8 带 BOM，Excel 可直接打开；中文表头带单位（如「吞吐（tok/s/卡）」「TPOT（ms）」「有效 MAC 利用率（%）」）；数值是原始值（不取整）；防公式注入。
  - JSON：工具 / 版本 / 时间、场景快照、列定义、逐行数据，以及原始 API 响应（映射对比含各映射的稳定性结果）。
  - 结果运行中隐藏导出按钮。

### Fixed
- CP 工作量 KPI 现在精确（步时不变）：
  - DSA top-k：前 top-k 个早期位置看到的键更少（0.88 在 131K 时偏高 0.8 %）。
  - LM 头：只有持有最后一块的卡才投影最后一个 token（0.88 偏高 (CP−1) 倍 LM 头工作量，短 prompt 时为 3e-5 – 4e-4）。

### Validation
- 对抗：242 个随机 CP 场景（43 个支持 CP 的模型 × PD / fabric / D2D / indexer / PP / 微批 / exec_overlap）与 CP1 对照，工作量完全相同；结果确定，hash 往返一致；能耗分项之和等于总量；无超线性加速；PD 下 prefill 池用 CP 布局正常。
- fabric `auto` 对链路带宽不单调：19–24 % 的 D2D + fabric 场景出现「链路 ÷4 反而快 ≤ 0.38 %」。这是已有设计（auto 逐次集合通信取 min(带宽 + α)，相当于运行时调优器的选法），现已写明幅度。`auto_overlap` 在同一批场景上 0 / 160 违例，常驻测试锁定这一点。
- PD 报告中合并部署放不下时 `coloc.ttft_ms = inf`，与 CP 无关；API 序列化成 null（HTTP 200）。
- UI：local/ui_check_export.py 0 个问题（下载文件逐值与 API 响应一致）。默认指纹不变。

## [0.88.0]

### Added
- 上下文并行 `layout.cp`（Web 布局「CP」/ CLI `--cp`），只用于 LLM prefill，可选，默认 1，指纹不变：环形注意力，按 zigzag 负载均衡。
  - 每张 CP 卡持有每条序列 1/CP 的 token 及其 KV 分片；每个注意力层把本卡的 KV 块在 CP 组内 all-gather（环形，组内跨度为 TP）；权重在 CP 卡间复制。MoE 要求 EP·ETP = TP·DP·CP。
  - 建模方式是精确等价，没有引入新的近似：一张 CP 卡等价于一个注意力 DP 副本，对每条序列做一次分块 prefill（q = ⌈S/CP⌉ 个新 token，前缀 c 满足 q·(c + q/2) = S²/(2·CP)，即最忙的 zigzag 卡的因果工作量）。
  - MLA：卡间交换的是 latent（576 B/token，而 K/V 是 82 KB），每张卡要对收到的块重做 kv_b 上投影，总量 = (CP+1)/2 倍，这是实际代价。DSA top-k 的前 2048 个位置使工作量 KPI 偏高 0.8 %（131K）/ 0.1 %（1M），步时不受影响。
  - 不支持：decode、线性 / 滑窗 / 压缩 KV 层、prefix_cached、images，都会给出明确报错。PD 时可以把 CP 放在 `pd.prefill_layout` 上。

### Fixed
- Web 服务：监听队列 5（socketserver 默认值）在 64 / 128 个并发请求的突发下重置了 39 % / 55 % 的连接（ECONNRESET）。现在监听队列为 1024、处理线程为 daemon；256 个并发请求全部成功。
- Web 服务：发了请求头却不发 body 的客户端会一直占着一个线程。现在 socket 超时 30 s，超时返回 408。

### Validation
- CP 手算：qwen3-8b 1M prefill CP8，环形 KV 0.338 s = 7 × 19.3 GB / 400 GB/s；每卡 KV 正好是 1/8（19.3 / 154.6 GB）；稠密模型各布局工作量守恒（误差 ≤ 1e-5，来自前缀取整）；MAC 能耗计数守恒。DeepSeek-V3.2 1M prefill：TP8 2820 s → CP8 410 s（0.87 的 key_split 是 425 s），每卡内存 181 → 132 GB。
- 场景 hash 向后兼容：10 个代表性场景在 0.70.1、0.80、0.86 与当前版本的 hash 完全相同；旧版本的 to_dict 与 to_dict(full) 都能在当前版本载入，且 hash 相同（golden 值写入常驻测试）。新版本的 full dict 带有新字段，旧版本按严格解析会拒绝，这是设计行为；最小 dict 仍能被旧版本载入。
- 并发：24 个线程混合 eval / PD / CP / 视频 / 蛋白 / compare / stability / layouts / 错误请求，结果与串行逐字节相同；fabric 状态本来就是 contextvar，并且按线程隔离。
- docs/MODEL.md 机械一致性检查：章节编号连续；19 个文件路径都存在；38 个代码引用都能解析（其余是 JSON 键或场景字段）；文中写出的常量值与代码一致。
- Web 没有扫描 / Pareto 导出功能（没有下载 / CSV / 剪贴板），因此不需要审计。
- 默认指纹：372 个点与 0.87 逐位相同。UI：local/ui_check_cp.py 0 个问题。

## [0.87.0]

### Added
- `Scenario.indexer`（Web「indexer 部署」/ CLI `--indexer`）：稀疏注意力 indexer（DeepSeek-V3.2 / V4、GLM-5 类 DSA）在注意力 TP 上的部署方式，可选，默认关。
  - `reference`（默认）= 按发布：V3.2 / GLM-5 在每个 TP rank 上重复计算，V4 按头切分。指纹不变。
  - `replicated` = 每个 TP rank 都算全部头。
  - `tp_split` = 头按 TP 列切分，fp32 打分矩阵做 all-reduce（V4 参考实现的做法）。
  - `key_split` = 缓存的 index 键按 TP 分片：各 rank 用全部头给本地键打分，取局部 top-k，all-gather TP·k 个（fp32 分数，int32 下标）候选再合并；结果精确，indexer 缓存按 TP 分摊。

  「按头切分 + 只汇集 top-k」不精确：分数是 Σ_h w_h·ReLU(q_h·k)，局部头的部分和上取 top-k 不等于全局 top-k。精确的头切分需要分数 all-reduce；只汇集 top-k 的精确做法是按键切分。`key_split` 只支持逐 token 的 DSA indexer；QSA 分块、k-pool、V4 压缩键会报错。
- 注意力 DP 副本空闲警告：每个微批的序列数小于 DP 时提示（长上下文单请求 prefill 用 DP 注意力没有收益）。
- CLI 补齐 11 个之前没有参数的场景字段：`--exec-overlap`、`--layer-overhead-us`、`--microbatches`、`--indexer`、`--clip-slo`、`--seq-slo`、`--fold-slo`、`--hop-d2d-us`、`--hop-link-us`、`--hop-net-us`、`--hop-net-tree-us`。README 说 API 接受与命令行相同的场景描述，此前这些字段只能经 API 设置。

### Validation
- 手算（DeepSeek-V3.2，8 卡 TP8，1M 提示 prefill）：reference 2820 s；tp_split 598 s，其中打分 all-reduce 587 s = 2.35e14 B / 400 GB/s；key_split 425 s，其中 top-k 汇集 18.3 s = 7.3e12 B / 400 GB/s，每 rank indexer 缓存少 7.4 GB（= 7/8 × 8.45 GB）。
- 工作量守恒：9 个带 indexer 的模型 × TP 1 / 2 / 4 / 8 × prefill / decode，4 个选项的有用 FLOP 完全相同；TP 1 时 4 个选项完全一致；reference 等于发布时的模式。
- DP 注意力（V3.2，1M prefill）：batch 1 时 7/8 的副本空闲（3056 s）；batch 8 时 DP8 每序列 406 s，且是唯一放得下的布局。
- 默认指纹：372 个点（全部模型 × 3 种布局 × 2 个阶段）与 0.86 逐位相同。
- UI 端到端（local/ui_check_indexer.py）：中文选项正确，请求带 indexer，结果随选项变化，默认值等价于不传。
- CLI 审计：命令行与 API 结果一致（hash 与 summary 相同）；15 种错误输入干净退出、没有 traceback（TP 3 与 microbatches 0 都是合法输入：不均匀 TP / 自动）。

## [0.86.0]

### Validation
- 组合对抗审计（val14/combo.py，960 个随机组合：PD + D2D 小芯片 + SLC（pin / lru）+ MoE 倾斜 + 投机解码 + 前缀缓存（prefix_cached / prefix_hit / radix 树）+ KV 策略（wait / recompute / swap）+ 异构 prefill 芯片）：fits ⇔ 各段 DRAM 需求 ≤ 容量、时间分量有限且 total ≥ 最大分量、能量按动作求和 = 总量且对 pJ 线性、关闭 D2D / SLC 时其能量为 0、倾斜 / 接受率 / 已缓存前缀 / pin SLC 单调、TP 上工作量守恒、PD 流体恒等式（req_s = min 容量，decode 容量 = 副本·tok/s / 输出长度，goodput = req_s·输出长度）、best_split ≥ 给定划分、排队指标有限、PD 能量按动作求和——0 违例、0 崩溃。
- 共置与 PD 的 1:1 极限：prefill 池与 decode 同布局时，PD 每卡 goodput ≤ 共置分时流体 goodput（35 个反例全部是 PD prefill 池用 TP1、共置的 prefill 跑在 decode 的 TP≥2 布局上，省去了集合通信；同布局后比值 0.50–0.99）。该界只在同布局下成立，已写成常驻测试。
- 数值极端（val14/extremes.py）：上下文到 1M（qwen3-8b、llama-3.1-8b、qwen3-next、deepseek-v3.2、minimax-m1）、1 → 1024 卡、batch 1 → 65536、主频 0.1 / 10 GHz、SRAM 1 MiB / 16 GiB：0 个非有限值、0 个 fits 不一致，TPOT 随上下文、TTFT 随提示长度单调；超过 max_position_embeddings 时有警告。被拒绝的输入都是正确的校验（稠密模型不支持 DP/EP；SRAM 必须 > 0）。
- 观察（非 bug）：deepseek-v3.2 在 8 卡 TP8 上 1M 提示 prefill 2820 s，主要是 lightning indexer：O(L²)，因果减半，fp8，并且按参考实现（`Linear`，不按 TP 切分）在每个 TP rank 上重复计算；手算 5.5e17 FLOP/rank，与模型一致。
- 没有发现 bug；数值输出与 0.85.0 逐位相同（仅新增测试）。

## [0.85.0]

### Fixed
- 紧 KV 闭式在「整数装箱」下凭空预测抢占（点 18 PD 最坏间隔 243 ms，DES 3.8 ms、0 次抢占）：recompute / swap 准入要求 used + S ≤ K，最多同时运行 n = min(B, ⌊K / S_min⌋) 条序列；若 n·(max(S + o) + e) ≤ K（e = 每步增长），任何到达过程都不可能抢占。连续（Gauss / Gamma）footprint 模型看不到这一点。现在该条件成立时 Rice 与饱和抢占率都置 0（`pdqueue.PACK_BOUND`，`kv_cap.pack_preempt_free`）。
- API：`/api/eval`、`/api/fit`、`/api/layouts`、`/api/compare`、`/api/stability`、`/api/pareto` 会静默接受未知的 body 顶层键（例如拼错的 `"objetive"` 会悄悄按默认目标运行），与 README 的「严格解析」不符。现按路由校验允许的键，未知键返回 400（UI / CLI 发送的键已核对）。
- UI：映射对比「任何布局都放不下」的结论和各面板的错误信息没有设置 `dataset.done`，改了卡数等之后不会标为过期（仍显示「1 卡下…放不下」）。现在这些分支同样会被标为过期。
- `domain.py` 文档字符串过时：视频管线的文本编码器 / VAE 解码早已建模（不属于 coverage 原因）。

### Validation
- 紧 KV 98 点网格（val12 / val13）：装箱界只在 9 个点-模式对上生效（点 18 / 36 / 39 的全部模式，这些点 DES 都是 0 次抢占），其余逐位不变；0 个变差。PD 最坏间隔 242.7 → 3.8 ms（DES 3.8）、85.5 → 2.7 ms（DES 2.7）。
- UI 端到端（Playwright，box）：ui_check / ui_check_catalog / ui_check_domain 均 0 问题。后两个脚本里的过时预期已更新：Wan、HunyuanVideo 已接入；clip_s 含固定的 TE / VAE 时间，因此按 denoise_s 检查 50→20 步比例。新增 local/ui_check_stale.py：完成后编辑、重新运行清除、运行中编辑（慢场景 compare 5.9 s 与稳定性）、「放不下」结论都正确标为过期。
- 目录覆盖标签：63 个模型，「完整」⇔ 无原因且参数与发布一致（偏差 0）；4 个蛋白模型「部分」的原因属实；10 个视频模型都建模了 TE + VAE；AlphaFold3「暂未接入」。
- 默认场景指纹不变。

## [0.84.0]

### Changed（紧 KV 闭式数值有意变化；默认场景指纹不变）
- 饱和抢占增长律默认改为 Gamma（`SAT_MODEL="gamma"`），并加入到达方差项：c²_eff = c²_life + (1 − c²_life)·ρ_K，ρ_K = E[k]·E[footprint]/K（`_arr_c2`，`SAT_ARR_M=1`）。依据：启动时刻为 Poisson 的运行集合，其离开过程接近 Poisson（Burke / M/G/∞），只有在 KV 很空时，门控准入才让它保持固定节拍。修正了 0.83 研究里纯寿命 c² 在高负载固定长度点低估抢占的问题（test_core_056 场景：抢占/请求 0.006 → 0.018，DES 0.024）。
- 恢复等待律默认 `VICTIM_WAIT="blend"`（0.83 研究），取 c²_eff。

### Validation
- 紧 KV 网格扩为 98 点（val12：原 42 点 + test_core_056 场景 + 55 个负载 0.85/0.9、多为固定长度的点；高负载小容量点多数在 DES 中不稳定，被排除），共 95 个 DES 稳定的点-模式对。
- 指数 m ∈ {1, 2, 4} 只在偶数点上选（m = 2 / 4 与 m 关闭都在偶数点有变差）→ 选 m = 1；奇数点（留出）0 个变差。
- 全部 95 对相对 0.83 默认（= 0.82）：0 个变差（容差 0.1 log + seed 散布）；抢占 log 误差均值 0.81 → 0.76，最坏间隔 p99 0.89/1.02 → 0.77（最大 5.8 → 4.2）。点 18 PD 只部分改善（243 ms，DES 3.8，原 522）。点 42 PD 与点 17 在 DES 中两个 seed 都漂移、不稳定，未计入（若计入点 42 PD，m = 1 略差：0.018 vs 0.020，DES 0.024）。

## [0.83.0]

### Research（未采用为默认；默认与 0.82 逐位一致）
- 紧 KV 饱和抢占高估（固定长度点 17/18 最坏间隔高 65–140×）：根因是饱和项把「两次离开之间的增长」当指数分布（无记忆），固定长度时 footprint 只在容量下方振荡、几乎不越线。现按 Gamma 增长律（`_sat_evict_gamma`，形状 = 1 / 寿命 CV²；寿命 CV² 合并输出长度散布与逐请求 TPOT 散布）：CV² = 1 退化为旧式，CV² → 0 时 → 0。`pdqueue.SAT_MODEL="gamma"` 内部开关（默认 "exp"）。
- 恢复等待律 `VICTIM_WAIT="blend"`（内部开关）：P(W > x) = (1 − c²)·resid + c²·Exp(λ_r)，两个极限（固定长度 / 无记忆）各自精确；c² ≥ 1 时与旧式逐位一致。

### Validation
- 全紧 KV 网格（42 点 × 3 模式 × 2 seed，DES 共用基线）：相对 0.82，没有一个点变差（容差 0.1 log + seed 散布）；抢占/请求 log 误差均值 0.70 → 0.46，最坏间隔 p99 log 误差均值 0.89 → 0.37（最大 5.8 → 2.9）；TPOT 不变。点 17/18（DES 无抢占）：PD 4 vs 3.8 ms、chunked 14 vs 13 ms、prefill_first 174 vs 175 ms / 312 vs 158 ms（原 969）。单独 resid 在 MoE CV1 点变差 4–6 处，按 c² < 1 硬切分仍 2 处 → 不采用。
- radix 精确解改为子集 DP（不按次序枚举，状态预算 100k）：与 0.82 枚举相差 ≤ 7e-14，回退 Che 次数少约 17 %；随机树对 RadixLRU 平均误差 0.005–0.013。仍回退的是「很多极小叶子」的树。
- **否决点**：test_core_056 的场景（dense8b、固定长度、负载 0.85、recompute 12 GB，不在网格内）DES 抢占/请求 0.024–0.055，exp 0.020，gamma 0.006；gamma 下最坏间隔 12.8 ms，低于不设 KV 上限时的 34 ms。高负载下 Poisson 到达本身就会让 footprint 越线，只看寿命 CV² 漏掉了这一项 → 不满足「处处更好」，默认保持 exp/exp。
- 默认场景指纹不变。

## [0.82.0] - 2026-10-11

radix 小树精确驻留；recompute / swap 被抢占者恢复等待的研究（未采用）。默认指纹不变。

- **radix 前缀树（数值修正，仅 `pd.prefix_tree`）**：节点很少时改用精确解。IRM 下 LRU 的稳态新近次序 = 按概率不放回抽取路径；按 token 计容量时保留能放下的最长新近前缀。存储路径 ≤ 4096、枚举步数 ≤ 50 000 时启用，否则仍用 Che。562 棵随机树对 RadixLRU：平均 |误差| 0.030–0.042 → 0.008–0.015，没有一棵变差；「长根 + 两个长子节点、只放得下一个」由偏 0.42 变为精确（Σp²）。按长度加权边界节点的另一修正更差，未采用。
- **被抢占者恢复等待（未采用）**：推导出按长度分布条件化的剩余寿命律（固定长度时 DES 40 ms 对旧闭式 740 ms），在 42 个随机紧 KV 点上最长间隔中位误差 0.50 → 0.28，但 MoE CV 1 等 9 个模式点变差，不满足「处处更好」。默认仍为 Exp(1/λ_r)（与 0.81 逐位一致，10 个 PD 报告哈希相同），新律留作内部开关 `pdqueue.VICTIM_WAIT = "resid"`。另发现固定长度下饱和抢占次数高估（MODEL §28）。
- 新增常驻测试 `tests/test_core_082.py`（4 项）。

## [0.81.0] - 2026-10-11

第 15 轮对抗性审计：前缀缓存、KV 策略、分块 prefill、投机解码、异构 prefill、PD 能耗对照 DES / 恒等式；MODEL.md 过时表述清理。默认指纹不变。

- **radix 前缀树的节点粒度（数值修正，仅 `pd.prefix_tree`）**：LRU 只存整节点，Che 按连续容量求解，缓存只放得下少数节点时高估命中（10 个 128-token 节点、308 token：0.24 → 0.20，与 RadixLRU 一致）。节点长度相同时取 ⌊C/L⌋·L；不同时减去被淘汰节点平均长度的一半。286 个随机树对 RadixLRU：平均 |误差| 0.029 → 0.013，p90 0.088 → 0.030。整前缀 LRU 的 Che 对 DES 60 / 60 一致。
- **DES 修正（PD before_prefill + KV 策略）**：KV 到达的请求原来在迭代中途记为开始 decode，首个间隔多算在途迭代的剩余时间：同一调度下，开一个从不绑定的 KV 上限，最长间隔 p99 就从 10.0 变为 14.3 ms。现在在下一个迭代边界开始，与关闭上限逐位一致。V4 数据不受影响。
- **构造函数接受芯片预设名**：`Scenario(chip="1P")` 原来在 evaluate 时 AttributeError，`PDConfig(prefill_chip="100T")` 原来被拒绝；现在都解析为预设（哈希不变），未知名称报 ValueError。
- **对照结论（无需修改）**：28 个随机 KV 策略 / 投机解码 / 异构 prefill / 分块点对 DES，除 seed 噪声大的 TTFT p99 外都在 V4 误差带内；两个短运行「DES 不稳定」点在 4 倍长度下稳定。PD 能耗：静态能耗按池 = W × 卡数 /(λ·out)，KV 交接字节只计一次，对能耗表线性。投机解码能耗按接受 token 计。
- **已知缺口（记录，未拟合）**：合并模式 recompute、KV 很紧时，闭式按「等一个离开」Exp(1/λ_r) 计被抢占者等待（740 ms），DES 平均 40 ms（被抢占者下一边界即可装回，vLLM 式）；最长间隔偏保守 5–8 倍（MODEL §27）。
- **MODEL.md**：标注 / 改正后续版本已补上的「不做」与过时常数（网络拓扑与网内归约、异构池、按池闲置功率、radix / KV 容量排队、`split_instances` → `instance_sched`、decode batch 搜索上限、池搜索副本规模、`workload.tmpl_msa`）；新增 §27。
- 新增常驻测试 `tests/test_core_081.py`（7 项）。

## [0.80.0] - 2026-10-11

PD 池布局搜索审计收尾与提速（结果逐位不变）；V4 服务验证改用长运行重跑。

- **PD 池布局搜索审计**：14 个随机场景（4 / 8 / 16 卡，dense / MoE，是否搜索 decode batch）全部通过：最佳流体 / SLO goodput 不劣于当前布局、排序正确、卡数守恒、重复运行一致。0.79 里「16 卡 > 10 min」不是单次报告耗时：单次报告最慢 109 s（11 路并行），10 分钟来自审计脚本为验证确定性跑两遍，且与 8 进程全量测试同时运行。
- **SLO 速率搜索提速（逐位一致）**：
  - `_slo_rate` 的稳定性二分只读 `stable`。PD 模式改走仅判稳定的路径：跳过 TTFT 分布与尾部计算；prefill 侧按各 cap 的闭式条件判稳定，不解 bulk 链。KV 链路判据只通过 u_coll 依赖所选 cap，且对 u_coll 单调，所有稳定 cap 给出相同结论时直接采用，否则回到完整路径。有 KV 容量上限时始终走完整路径。
  - PD 无 KV 上限时，若默认 cap 未满足 TPOT SLO，则不再逐个尝试其他 cap（decode 与 prefill cap 无关，任何 cap 都不可能满足）。
  - 11 个随机池布局场景的报告 JSON 逐字节一致，耗时降为 1/1.4–1/2.5（最慢 109 → 63 s）。V4 网格重算的闭式值与旧数据逐位一致。
- **V4 长运行重跑**：`accel_dse/data/v4_serving.json` 改为 4 倍长度（12000 / 80000 请求 × 3 seed，SLO goodput 二分每次 6000 请求）。DES 中位数移动 ≤ 1.3 %，原网格未被短仿真拉偏。seed 噪声降为约 1/2–1/4，30 点全部仍在误差带内，部分最大误差收窄（chunked TTFT p99 0.23 → 0.12、PD itl_max 0.23 → 0.15）。chunked SLO goodput 的最大误差变大（0.05 → 0.17，仍在 0.20 带内），旧值是噪声下的偏乐观估计。说明见 MODEL §18.4。
- 新增常驻测试 `tests/test_core_080.py`：仅判稳定路径与完整模式对各 λ 结论一致（含 KV 上限）；`_prefill_stable_caps` 非空当且仅当 `_prefill_server` 稳定；`_slo_rate` 结果与不走捷径时相等。
- 指纹：默认 fp / fpmc 与 0.79.0 一致。全量 436 / 436 通过（-j 8，271 s，无 pytest）。

## [0.79.0] - 2026-10-11

PD 尾部误差溯源；搜索 / 优化层与视频 / 蛋白质服务极端工况的对抗性审计（第 13 轮）。数值不变。

- **长 prompt、负载 0.9 的 TTFT p99「+36 %」不是解析误差**：该行（prefill batch 上限 1、D = 774 ms、λ = 1.163 /s、ρ = 0.90）是纯 M/D/1，解析的 p50 / p90 / p99（3110 / 9124 / 17730 ms）与 Erlang 精确 M/D/1 等待分布（80 位精度）逐毫秒一致。偏差来自 DES 的有限时长：10000 请求 × 3 种子的 p99 为 13.1 s，40000 请求升到 16.0 / 20.1 / 20.0 s（均值 18.7 s，向精确值收敛）。因此不做尾部「改进」（任何改动都会偏离精确值）。注意：高负载下 V4 / DES 误差带部分反映 DES 采样偏差，评估时需要更长的运行。
- **搜索 / 优化层对照穷举（随机小空间，未发现问题）**：300 次 batch 搜索与穷举逐一一致（164 次有可行 batch）；50 次布局 top-3（分支定界）与完整排序一致；结果确定（含进程池预取 = 串行）；最优从不劣于任一可行的 batch-1 点；4000 组随机点的 Pareto 前沿 = 精确非支配集（含 NaN、重复点）；PD 池布局搜索（7 个随机场景）：最佳流体 / SLO 不劣于当前布局、排序与卡数守恒、重复运行一致。
- **视频 / 蛋白质极端工况**（全部 17 个模型 × 帧数 1–257、分辨率 64²–3840×2160、步数 1–200、CFG、残基 1–16384、MSA 1–16384、recycle 1–20、样本 1–25、batch 1–65536、TP / SP / DP / PP 布局，共 452 次评估）：无异常 / NaN，延迟随帧数 / 分辨率 / 步数 / 残基 / MSA / batch 单调，容量随 batch 单调。
  - 新增警告：结构模型 MSA 行数超过发布实现上限（AF2 / OpenFold 508 聚类行、Boltz 4096 等）时提示发布代码会聚类 / 截断，按所填行数计（what-if）。原先静默。默认场景不受影响。
- 新增常驻测试 `tests/test_core_079.py`（5 项：M/D/1 精确分位数、batch 搜索 = 穷举、布局 top-k = 完整排序且不劣于可行点、Pareto 精确、视频 / 蛋白质极端与 MSA 警告）。
- 指纹：默认 fp / fpmc 与 0.78.0 一致。全量 433 / 433 通过（-j 8，345 s，无 pytest）。

## [0.78.0] - 2026-10-11

链路能耗改为物理上线字节；PD 服务侧（解析 vs DES）极端工况与 UI 对抗性审计（第 12 轮）。

- **链路能耗按物理上线字节（有意的数值变更，仅能耗动作计数）**：每次集合通信按所选算法与各层实际发送字节计入 `d2d` / 节点内 `link` / 跨节点 `net`（各自 `pJ_bit_*`）：环形 all-reduce 2(n−1)/n × 载荷、all-gather / reduce-scatter (n−1) × 分片、all-to-all (n−1)/n × 载荷、tree 2(n−1)/n、分层按各层体积、in-network 顶层每 rank 一个分片、协议开销 ÷ 效率、PP 交接 = 载荷。新增 `StageTime.wire_d2d / wire_link / wire_net`、API `link_GB.wire`。`link_bytes / d2d_bytes / net_bytes`（载荷口径，计时与 PD KV 份额）不变：时延 / 吞吐指纹 fp / fpmc 与 0.77.0 完全一致（75/75 × 2）。
  - 能耗计数变化（全部 LLM，0.77 → 0.78，MAC / DRAM 计数不变）：TP2 ×1（环形 2(n−1)/n = 1）；TP4 ×1.5；TP8 ×1.75；TP4 跨 2-die 封装 ×1.5（D2D = 载荷、节点内 = ½ 载荷，原为 ⅔ / ⅓）；EP16（每节点 4 卡）all-to-all ×15/16；TP1 不变。用户提供的 `pJ_bit_link` 等现按每个上线比特计。
- **修复：DES 稳定性判定漏判过载**。请求寿命（输出 512 token × ~50 ms ≈ 26 s）长于测量窗口时，800 个请求的到达跨度不足 2 个请求寿命，积压尚未显现：漂移检验（0.43 < 0.5）与（合并模式无）利用率检验均看不到，合并 prefill-first 在 1.84× 容量下仍判为稳定。新增必要条件：提供的 token 速率 ≤ decode 池最佳 token 速率（副本数 × max_k k·tokens/step(k)；合并模式再扣除 prefill 占用时间份额），超出即不稳定（`over_capacity`）。修复后 DES 与解析的稳定边界一致（合并 9.76 rps：9.5 稳定、10.8 不稳定）。
- PD vs DES 极端网格（负载 0.05–0.95 及 1.1× / 2× 过载、batch 1 / 4 / 1024、prompt 128–32k、输出 8–4096）：稳定性判断除上述过载外全部一致；无 NaN / 异常。batch 1024 时 decode 池放不下，按设计报告容量 0 并给出「容量不足」警告。长 prompt（16k–32k）在负载 0.9 的 TTFT 尾部误差 +67% / +103%（800 请求）主要是 DES 采样不足：10000 请求 × 3 种子收敛到 p50 +2%、p90 +11%、p99 +36%（解析尾部偏保守，未改）。输出极短（8 token）时合并模式 TPOT 误差大（TPOT 由少量受干扰的 token 决定），已记录。
- **修复（UI 陈旧状态）**：「最佳布局」搜索期间若场景被修改，结果不再覆盖新的修改（提示已过期）；稳定性面板在场景修改后标记为过期（原先只在渲染时打标签）。
- **UI 单位**：存储器容量选择框标为 GiB（容量按 2³⁰ 计，与详情一致）；通信分层提示改为按上线字节的份额并注明口径。
- 新增常驻测试 `tests/test_core_078.py`（5 项）；`test_core_065` 的链路计数核对改为 `wire_link`。全量 428 通过（-j 8、隐藏 pytest，331 s）。

## [0.77.0] - 2026-10-11

2026 年发布模型的锁定 + 硬件侧（映射 / 内存规划）对抗性审计（第 11 轮）。数值不变。

- 2026 年模型（GLM-5 / 5.2 / 5.3 / 5.3-Flash、DeepSeek-V4-Flash / V4-Pro / V4.1-Flash、Kimi-K3 / K2.7-Code、Qwen3.8-2.4T / 27B / Flash-Next）在 Hugging Face 均已公开：本地快照（2026-10-08 抓取）与线上 config.json 逐键一致，safetensors 索引 total_size 与张量数完全一致；标签准确（Kimi-K2.7-Code 标为 nearest：用户要求的「Kimi K2.7」无同名发布）。由常驻测试锁定。
- 硬件侧手算核对（均一致，未发现 bug）：DRAM 有效带宽 = 原始 × 0.7（LPDDR6 另 × 8/9 载荷）；解码 DRAM 时间 = 字节 / 有效带宽；SRAM / SLC 容量扫描单调、饱和时退化为纯计算；SLC pin 命中 = min(容量, 每步足迹)、t_slc = 命中 / slc_GBps，LRU 对大于缓存的循环扫描命中为 0；AWQ int4 = 4 + 20/128 bit、反量化 2 次向量操作 / 元素 / lane；FP8 检查点按 W8A8 原生；容量按 GiB；集合通信 α-β（t_sync = α × 次数、环形 2(n−1)/n）；D2D 分层（层内 RS/AG + 层间 AR 的 α 与字节份额）；MAC 计数。
- 待决（未改，已文档化的口径）：能耗中的链路字节按「集合通信载荷」计（MODEL.md：链路字节按载荷口径、按各层实际发送字节之比拆分），而非物理发送字节（环形 all-reduce 为 2(n−1)/n × 载荷，all-to-all 为 (n−1)/n）。改为物理口径会改变所有 TP > 1 场景的链路能耗，需要用户决定。
- 新增常驻测试 `tests/test_core_077.py`（7 项）。
- 指纹：默认 fp / fpmc 与 0.76.0 完全一致（75/75 × 2）。全量 423 通过（-j 8、隐藏 pytest，334 s）。

## [0.76.0] - 2026-10-11

对抗性审计（第 10 轮）：LLM / VLM 逐算子形状对官方配置与参考代码，解码 KV 读取与内存规划字节核对。**有意改变 3 个模型的数值**。

- 修复：softmax 注意力的 sigmoid 输出门。Qwen3-Next 的 `Qwen3NextAttention` 固定把 q_proj 做成 2× 宽（query + gate），配置里没有 `attn_output_gate` 键，原先漏掉（−100.7 M 参数、12 层全注意力的 q GEMM 少一半）；MiniMax-Text-01 / M1 的 softmax 层（`MiniMaxText01Attention`）只有 q/k/v/o，输出门属于闪电注意力层，原先多算（+503 M 参数）。现三者参数与发布的 safetensors 一致。默认场景：qwen3-next-80b-a3b TPOT +2.83 %、8k prefill TTFT +2.58 %；minimax-text-01 / m1-80k TPOT −1.05 %、TTFT −1.03 %。其余模型不变。
- 审计（与官方 config.json / modeling 代码逐项一致）：DeepSeek-V3 / R1 / V3.1（MLA 吸收解码、q_lora 1536、kv 512 + 64、前 3 层稠密）、V3.2（DSA 索引器 64×128、fp8 + scale 132 B/token、top-2048 及短上下文边界）、Kimi-K2 / K2.5（64 头、首层稠密、384 专家）、GLM-4.5 / 4.6 / Air、Qwen3 稠密 / MoE、Qwen3-Next / Qwen3.5（Gated DeltaNet 投影、fp32 状态 + conv 状态）、MiniMax（闪电注意力、每 8 层 1 层 softmax）、gpt-oss（128 滑窗与全注意力交替）、Mixtral 等；视觉塔 Qwen3.5（无 deepstack）、Kimi-K2.5 MoonViT。
- 字节核对：全部 58 个 LLM 的 memplan KV + 索引器 + 状态存储与 `cache_new_bytes` 完全一致；稠密模型的解码读取 = 整个缓存 + 新 token 一项；稀疏 / 压缩 / 跨层共享缓存模型（DSA、V4、GLM-5 等）读取 ≠ 存储，属设计行为。
- 新增常驻测试 `tests/test_core_076.py`：输出门、参数总量对发布值（全部 LLM，±0.1 %）、上述形状锁定、解码 KV 读取对 memplan、视觉塔配置。
- 指纹：fp / fpmc 仅 qwen3-next-80b-a3b、minimax-text-01、minimax-m1-80k 变化（有意），其余 72 个与 0.75.0 完全一致。全量 416 通过 / 0 失败（-j 8 349 s，隐藏 pytest）。

## [0.75.0] - 2026-10-11

对抗性审计（第 9 轮：蛋白 / 视频逐算子形状、硬件规格）+ PD 长度分布报告剖析。

- 修复：只承载 misc 参数的「参数层」（ESMFold 87/92、Boltz-1 144/145/147、Protenix 91/93/94/96）在 full 阶段走了 Transformer 块路径，凭空产生 pre-norm / ffn-norm / residual 向量操作和零尺寸注意力算子（TP > 1 时还会有 all-reduce）。现不产生任何算子（参数仍由 memplan 计入热集）。向量工作量 −4e-6（相对），时延逐位不变；仅这三个模型的向量能耗末位变化。
- 审计（与参考代码 / 配置逐算子核对，均一致）：OpenFold（模板 / extra-MSA / Evoformer：行列注意力 8×32、OPM c=32、三角乘 c_z=128、三角注意力 4×32、132 = 128 + 4 模板行）、ESMFold（ESM-2 3B、折叠主干 32 头、IPA 16+4·3 / 16+8·3+128）、Boltz-1（pairformer 16×24、扩散 24 块 16×48、原子窗口 32×128 4 头、PWA 8×32）、Protenix（AF3：PWA 8×8、5 个样本批处理、逐样本置信度）；视频 Wan 2.1、HunyuanVideo（双流 / 单流、调制 6× / 3×、融合 linear2）、LTX、Open-Sora STDiT3（空间 / 时间交替）、Mochi、CogVideoX 的帧 → token 换算；参考硬件 MI300X / B200（nvfp4 9 PF）/ H100 / H800 / H200 / A100 与内存目录（HBM3/3E/4、LPDDR5/5X/6）对数据手册。
- 剖析（PD + 长度 cv 报告）：热点为 `_tagged`（每块平均 3.8 项）与 `_wait_hists`（每段平均 6.4 bin）中约 5.8 亿次 dict 操作。各量都随 λ 变化，跨二分无可复用；稠密列表 / 切片向量化两种改写均逐位一致但不更快（段太短），已撤回。保留 `_wait_hists` 去掉逐键临时分配的微调（逐位一致，提速在噪声内）。纯 Python 下已到下限，进一步提速需改变数值的算法改动。
- 新增常驻测试 `tests/test_core_075.py`：参数层无算子、无零尺寸注意力；上述蛋白 / 视频参考形状锁定。
- 指纹：默认 fp / fpmc 与 0.74.0 完全一致（75/75 × 2）；PD 长度 cv 比较报告逐位一致。全量 412 通过 / 0 失败（-j 8 356 s，隐藏 pytest）。

## [0.74.0] - 2026-10-11

对抗性审计（第 8 轮）+ 测试运行器 + 测试提速。

- 修复：`serving.moe_skew` 未受物理上限约束（可在 EP = 2 时取 3.0，即最忙 rank 承担 150 % 的 token-专家对）。现按 EP·min(top_k, ⌈E/EP⌉)/top_k 封顶并给出警告；默认（skew = 1）及所有不超上限的取值不变。
- 修复：`/api/fit` 遇到非法布局（如 pp 超过层数、MoE 的 ep·etp ≠ tp·dp）时 ValueError 未被捕获 → HTTP 500；现返回 400。`serve.handle` 另把路由抛出的 ValueError 统一按 400（输入校验）处理。
- 测试运行器：模块 / 测试按精确名选择——`test_x`、`tests/test_x.py`（整个模块）、`test_x::test_y`（单个测试）、或测试函数全名；`-k WORD` 保留旧的子串匹配；无匹配时报错退出（之前 `test_api` 会顺带选中其他模块里的 `test_api_*`）。
- 提速：`bulkq._tagged` 首次访问路径不再构造 kd 列表（预建 (ΣF, k) 键行；同样的乘积、同样的累加顺序，结果逐位一致），带长度 cv 的 PD 报告约快 11 %（135 → 120 s）。慢测试的主体是带长度 cv 的 PD 排队报告本身（每份 ~2 min），已无可在不改结果的前提下省掉的重复计算（逐 λ 搜索内的模式评估已验证无重复）。
- 新增常驻测试 `tests/test_core_074.py`：投机解码（MTP 模型：每步期望 token、TPOT 定义、验证步不比单 token 便宜、接受率单调）、MoE 倾斜（不加速、单调、上限、有用工作量不变）、D2D（更快链路不更慢、每封装 1 die = 单片）、dtype 矩阵（存储字节随位宽有序、有用工作不变、非原生格式必有反量化）、VLM（图像 token、编码器、TTFT ≥ 等长文本）、API 输入随机变异只返回 200 / 400 且可序列化、运行器精确选择。
- 审计规模：fuzz7 1500 例 + API 变异 8000 次；除上述 2 处外 0 违例（spec_k 在无 MTP 模型上按文档忽略、视觉编码器只在 prefill 计、fp8_tensor 走原生 fp8、最忙 rank 激活随倾斜增长、非有限数经 api.clean 输出为 null，均属预期）。
- 指纹：默认 fp / fpmc 与 0.73.0 完全一致（75/75 × 2）。全量 408 通过 / 0 失败（-j 8 299 s，0.73 为 326 s）。

## [0.73.0] - 2026-10-11

对抗性审计（第 7 轮）：按四条验收标准（逐 rank 工作守恒、合法调度依赖、硬容量约束、搜索不劣于已知可行点）对此前无性质测试的区域做随机不变量检查。

- 修复：`lengths.lognormal_bins` 逐 bin 取整使长度分布均值漂移（如 out_len 64、cv 0.9 → 64.13），PD goodput ≠ req/s × out_len / 卡数。改为保持均值的整数化（Σ = round(N·mean)，floor 后按小数部分补 1，值保持非降）；仅当 mean·N < N（均值 < ~3 且 cv 大）时受 ≥ 1 下限所限。只影响设置了 prompt_cv / out_cv 的 PD / 排队结果（量级 ≤ 0.2 %），默认指纹不变。
- 新增常驻测试 `tests/test_core_073_audit.py`：能量计数（求和、逐动作线性、与能耗表无关、idle = 卡 × 窗口）、SLC（pin / lru 不变慢、不增 DRAM 字节、fits 不变）、SRAM / SLC 分区容量、fabric 集合通信（有限、份额 ∈ [0,1]、随载荷单调且至多线性、不低于最快链路理想界、oversub 单调）、PP 调度（step ≥ Σ 阶段、≥ 微批 × 瓶颈、层划分恰好一次）、视频 / 蛋白调度（编码 → 去噪 → 解码链、周期界、工作量对 batch / SP / DP 守恒）、搜索不劣于随机可行点与暴力 batch、PD 流体 / 能量恒等式、长度分布均值。
- 审计规模：约 1.3 万次随机检查（val7/fuzz*.py），除上述 1 处外 0 违例（另确认 idle_W_prefill 仅用于 PD prefill 池，属文档行为）。
- 指纹：默认 fp / fpmc 与 0.72.0 完全一致（75/75 × 2）。外部验证不受影响（refhw 未设长度 cv）。
- 测试：`-j 1` 顺序全量 403 通过 / 0 失败，测试耗时合计 3339 s（墙钟约 61 min；最慢 test_core_052 924 s、061_3 458 s、053 265 s）；`-j 8` 约 5.5 min。

## [0.72.0] - 2026-10-11

0.71 调度审计遗留项（建模说明 §26.1）。**有意改变默认数值**（仅 auto 芯片；参考硬件 GPU 仍为 wide，外部校核不变）。

### 变更（默认数值）
- `instance_sched="auto"` 也作用于单个 GEMM（count = 1）：候选为 wide、各核组大小 g | E、活跃组数减半，以及把 M 行或 N 列切到多组（行切分每组流过完整 K × N，列切分每组流过完整 M × K）。修正了 wide 阵列填充 R + C·E 随核数增长而让窄 N 的 GEMM 变慢、核越多越慢的问题。`"split"` 的 count = 1 语义不变。
- B > 1024 的 decode 链插值网格由 2^(1/8) 加密为 2^(1/12)（auto 下 step(k) 不够平滑，0.5 % 误差带需要）。
- 逐数变化（`/workspace/val5/headline.py`，0.71.0 → 0.72.0，3 芯片 × 目录模型 × 3 工作点 × 3 布局，1305 个场景）：451 个数值改变，瓶颈翻转 MAC → DRAM 21，MAC → LINK 9。

| chip | point | metric | n | min | median | max |
|---|---|---|---|---|---|---|
| 100T | decode b1 | throughput | 22 | +0.1% | +16.7% | +82.3% |
| 100T | decode b16 ctx32k | throughput | 25 | +0.1% | +16.7% | +114.3% |
| 100T | prefill b2 8k | throughput | 141 | +0.1% | +5.6% | +86.2% |
| 100T | prefill b2 8k | ttft | 116 | -36.3% | -4.8% | -1.0% |
| 1P | decode b1 | throughput | 21 | +0.0% | +16.5% | +154.3% |
| 1P | decode b16 ctx32k | throughput | 23 | +0.0% | +17.7% | +115.6% |
| 1P | prefill b2 8k | throughput | 32 | +0.0% | +22.2% | +131.7% |
| 1P | prefill b2 8k | ttft | 11 | -56.8% | -37.4% | -9.9% |
| H100-like | decode b1 | throughput | 20 | +0.0% | +3.9% | +40.0% |
| H100-like | decode b16 ctx32k | throughput | 20 | +0.0% | +3.9% | +40.2% |
| H100-like | prefill b2 8k | throughput | 20 | +0.0% | +3.9% | +40.1% |
- 指纹基线有意更新（单卡与多卡两版，75 / 75 改变）。

### 新增（默认关闭，默认数值与 hash 不变）
- `Chip.noc_topology`（"mesh" / "ring" / "crossbar"）与 `Chip.noc_hop_cycles`：跨 s 个核的调度每个 op 每个 sequencer 付一次 hop × 直径（crossbar 1、ring ⌊s/2⌋、mesh 2·(⌈√s⌉ − 1)）。
- `Chip.wave_overhead_cycles`：每个 op 付一次启动；≥ 2 个核组按锁步波次运行时每多一波付一次屏障（单一 sequencer —— wide 或单个活跃组 —— 背靠背流过，不付）。

### 审计
- `tests/test_core_071_audit.py` 扩展到 count = 1、随机 NoC 拓扑 / hop / 屏障：auto ≤ min(wide, split)、理想 MAC 下界、端口下界、核数翻倍不变慢，全部成立。第一版"每波都付开销"在核数单调上有 135 / 4000 个反例（单活跃组也付了每波开销），改为上述屏障语义后为 0。

### 测试 / 性能
- `tests/test_core_072.py`；test_core_071 的单 GEMM 断言按新语义更新（auto 严格快于 wide 的示例）。
- `tests/run_tests.py`：每个测试打印耗时；`-j N`（默认 min(8, CPU 数)）把测试模块放进各自的解释器并行跑，`-j 1` 为原来的进程内顺序执行。每个测试仍只跑一次，检查不变；全套墙钟时间见下。
- `bulkq._tagged`：按 (t, u 槽) 分小字典累加，键、和与首次插入顺序逐位不变（对捕获的调用逐项比对）。

## [0.71.0] - 2026-10-10

多核上的独立小 GEMM 调度成为 NPU 设计变量，默认 `"auto"`（建模说明 §26）。**有意改变默认数值。**

### 新增
- `Chip.instance_sched`：`"wide"` / `"split"` / `"auto"`。决定 count > 1 的独立 GEMM 实例（每请求 / 每头注意力、MLA 吸收后的每头乘积、每专家 GEMM）在 engine 上怎么排：auto 逐 op 取最优的核组大小 g | E（E/g 组，每组 R × (C·g)），外加行切分与“少用几组”；SRAM 端口是全片预算、按正在工作的组均分（空闲组让出份额），累加器按核分（g/E）；每组的实例背靠背流过，流水线填充每组一次。研究脚本 `scripts/instance_sched_study.py`，逐数对比脚本 `scripts/instance_sched_diff.py`。

### 变更（默认数值）
- 目录 NPU 芯片与 `Chip()` 默认改为 `"auto"`（≤ 0.70 的 wide 对多核芯片是建模错误）；`"wide"` 仍可选；参考硬件 GPU 显式 wide，外部校核表逐行不变；默认场景 hash 不变。
- 指纹基线有意更新（单卡与 TP8 / DP8·EP8 两版，75 / 75 改变）。逐数变化汇总：
changed numbers: 555 (models 74); bottleneck flips: 88 — MAC→DRAM 88

| chip | point | metric | n | min | median | max |
|---|---|---|---|---|---|---|
| 100T | decode b1 | throughput | 21 | +0.0% | +0.5% | +6.8% |
| 100T | decode b16 ctx32k | throughput | 27 | +0.0% | +0.6% | +6.8% |
| 100T | prefill b2 8k | throughput | 143 | +0.0% | +30.3% | +169.3% |
| 100T | prefill b2 8k | ttft | 116 | -62.9% | -25.7% | -7.1% |
| 1P | decode b1 | throughput | 11 | +0.0% | +1.7% | +28.6% |
| 1P | decode b16 ctx32k | throughput | 15 | +0.0% | +0.9% | +25.6% |
| 1P | prefill b2 8k | throughput | 82 | +0.0% | +54.2% | +773.5% |
| 1P | prefill b2 8k | ttft | 69 | -88.6% | -40.0% | -0.5% |
| H100-like | decode b1 | throughput | 9 | +0.0% | +0.0% | +1.0% |
| H100-like | decode b16 ctx32k | throughput | 10 | +0.0% | +0.0% | +1.1% |
| H100-like | prefill b2 8k | throughput | 31 | +0.0% | +6.5% | +209.8% |
| H100-like | prefill b2 8k | ttft | 21 | -67.7% | -8.3% | -2.8% |

- decode 中位数 < 2 %；prefill TTFT 中位数 −8.3 %（H100-like）/ −25.7 %（100T）/ −40.0 %（1P）；88 个场景瓶颈 MAC → DRAM（涉及 167 个数值）。

### 测试
- 手算 / 标定数值的测试钉在 wide（其数值按 wide 推导）：test_core_mapping、test_core_063 MLA 周期、test_core_062_1 §19 示例、test_core_064 PD SLO 带（auto 下 7.16 rps，wide 1.23）、test_core_coverage 的 0.65.1 指纹、V4 网格与 052/054–057 排队场景。auto 默认数值由 fp_0710 指纹基线覆盖。

### 审计（调度改动）
- 不变量扫描测试 `tests/test_core_071_audit.py`（600 组随机几何 / 形状 / 端口 / 累加器）：auto 不慢于 wide 和 split；不快于理想 MAC 下界（功守恒）；不快于全片 SRAM 端口下界；每核阵列与每核累加器不变时核数翻倍不变慢（端口按几何推导或总量固定两种）。
- 审计前的实现（每组固定 g/E 端口份额、每个实例各付一次填充）在核数翻倍时有 44 / 4000 个反例（≤ 3 % 到 1.7×，均在端口受限或小实例处）；修正后 0 个。目录 / 指纹 / 头条数值逐项不变（555 个变化数值与修正前完全相同）。
- count = 1 的单个 GEMM 仍只走 wide（本版范围）；wide 本身在核数翻倍时会因填充 (R + C·E) 变长而略慢，这是 wide 阵列的性质，不属于 auto。

## [0.70.1] - 2026-10-10

### 修复
- 0.70.0 给 Chip 加的 `split_instances` 字段会进入序列化，导致默认场景的 `Scenario.hash()` 改变（test_core_062 失败）；该字段改为 omit_default，hash 恢复为 0.61.4 起的值。

### 文档
- README.md / README.en.md 改成标准项目说明（功能、安装、用法、覆盖范围、文档链接），去掉逐版本条目；版本历史只记在 CHANGELOG（原 README 里的条目这里都已有）。新增 guard 测试 test_core_readme。

## [0.70.0] - 2026-10-10

外部校核第四轮（建模说明 §25.9）。新选项都关时，默认指纹不变。

### 变更
- `exec_overlap="tbo"`：每个微批各读一遍权重（双微批，t_dram 加一份 weights / BW）。只影响 tbo 模式。

### 新增
- `Chip.split_instances`（默认 False，「假设」）：批量 GEMM 实例分到各 engine 上，每个 engine 是一个 R × C 阵列；与整芯片宽阵列两者取便宜的。
- DeepSeek profile decode 行（MTP 按 spec_k = 1 处理，接受率用公开的 0.875）；extval 支持 spec_k / spec_accept、split_variant；`scripts/ext_opbreak.py` 与 DeepSeek decode trace 逐项对照；`scripts/ext_score_all.py`。

### 结论
- EP72 decode 1.27 → 1.19（tbo）；DeepSeek profile decode 0.93。宽阵列抽象下的注意力偏慢 2.8 ×，和 GEMM / 小 kernel / 通信偏快的部分互相抵消；split_instances 不在参考硬件上打开（否则要拟合 kernel 效率）。
- PD 端到端：没有公开的数值实测。

## [0.69.0] - 2026-10-10

外部校核第三轮（建模说明 §25.8）。默认指纹不变。

### 新增
- `Scenario.layer_overhead_us`（默认 0，关闭，「假设」，依赖 NPU / 运行时）：每层每步的固定时间，所有重叠模式下都暴露。
- 参考硬件 MI300X、B200；`data/ext_measurements_r3.json`（31 条：AMD ROCm vLLM MI300X、TensorRT-LLM B200 NVFP4）；指标 `static_latency_s`。
- extval：连续批代理 `predict_cb`（pdqueue 分块计划 + 合并步）；`scripts/ext_cb.py`。

### 结论
- 每层固定开销：只在 H100 短 prompt TTFT 上选 100 µs / 层，留出测试：非 H100 TTFT 误差 130 % → 22 %，MI300X 静态批延迟 227 % → 23 %。
- `moe_skew` 用公开 EPLB 统计核对（1.115 / 2.56）；更正 0.68：EP72 decode 的高估不是不均衡造成的。

## [0.68.0] - 2026-10-10

外部校核第二轮（MoE / MLA / 大规模 EP，建模说明 §25.7）。

### 新增
- `data/ext_measurements_moe.json`：34 条公开实测（逐条带 URL）：SGLang 96 × H100 大规模 EP（DeepSeek-V3 prefill EP32、decode EP72）、DeepSeek profile-data prefill（H800）、TensorRT-LLM DS-R1 8 × H200（EP8 + attention DP）、TensorRT-LLM 0.17 Mixtral-8x22B TP8（H200 / H100 / A100）。
- 参考硬件 H800-SXM（H100 die「假设」，NVLink 400 GB/s）。
- `Scenario.exec_overlap="tbo"`：计算按 kernel 串行，集合通信藏在计算后面（双微批重叠）；stage ≤ tbo ≤ kernel ≤ serial。默认不变。
- extval：多节点（node_cards / net）、dp / ep / etp 布局，指标 prefill_tok_s_node / decode_tok_s_node / static_tok_s_total。

### 结论
- 大规模 EP（显式双微批重叠）`stage` / `tbo` 误差 9 %；第一轮拟合出的 serial / kernel 组合留出误差 77 ~ 87 %。按软件文档选重叠模式的预定规则：1.03 / 15 %。继续不引入拟合效率系数。

## [0.67.0] - 2026-10-10

0.66.0 + 目录覆盖收口（分支 `cov`）：全部 LLM 覆盖为「完整」（71 完整 / 4 部分，含视频 / 蛋白质）。核对：0.65.1 时已完整的 50 个模型 evaluate 数值指纹（3 个服务点）不变；变化的只有收口的 15 个模型；外部校核表（`data/ext_validation.json`，含 3 个校核专用模型与目录 Llama-3.x）逐行不变。

覆盖度收口：按公开参考实现补齐「部分」模型，见建模说明 §2.1。

### 变更
- **DSA lightning indexer 按参考逐项建模**（DeepSeek-V3.2、GLM-5 / 5.2 / 5.3）：打分 GEMM 之后的 ReLU、按头加权求和、top-k 选择，k LayerNorm / RoPE；DeepSeek-V3.2 走参考的 fp8 路径（Hadamard 旋转、fp8 打分 GEMM、索引缓存 132 B / token）。GLM-5.2 / 5.3 的 57 个 `shared` 层不再带索引器（参数与发布由 +0.07 % 变为一致）；GLM-5 与 GLM-5.2 因此不再同结构，取消合并。
- **gpt-oss attention sink**：softmax 每行多 1 个 sink logit（openai/gpt-oss 参考 `sdpa`）。
- **线性注意力按参考 chunk 形式上阵列**（Gated DeltaNet / KDA chunk 64，MiniMax Lightning BLOCK 256，状态 fp32 均按参考）：块内 / 块间 GEMM 计入阵列，decode 递归向量计数按参考（delta 7·dk·dv，lightning 5·dk·dv）。
- **GLM-5.3-Flash 按 transformers glm5_next**：mHC 超连接混合计算（fn GEMM、Sinkhorn、collapse / mix，流水级间传 N 流）、HyperHead、DSA 索引器 key 池化（门控压缩、池打分、尾池、514 B / token 缓存）、KDA 低秩输出门（参数原多 1.05B，现与逐层 safetensors 头一致）。架构代理 → 完整。
- **DeepSeek-V4-Flash / -Pro 按官方 inference/model.py**：压缩器（重叠比 4 / 比 128）、自带压缩器的索引器（头按 TP 切分 + 分数 all-reduce）、attention sink、q / kv RMSNorm 与 o 逆 RoPE、mHC + 学习的末端 hc_head、MTP 为比 0 层；HC 参数原少算 ×4（−25 / −63 M）。架构代理 → 完整（tid2eid 18.6 MB 查表未计，压缩器 fp32 按 bf16，见 §2.1）。
- **DeepSeek-V4.1-Flash 按官方 inference/model.py + engram.py**：窗口 + top-k 压缩键（min(p, w) + min(k, ⌊p/r⌋)，精确）、压缩 KV / index key 只在 kv 源层计算与存储（其余层共享）、索引器只在索引源层运行（其余复用 top-k）、第 20 层候选块预选与后续层掩码、Engram 查表（fp8 行 + e8m0 scale，8.25 bit / 参数存储，按行分片读取 + all-reduce + wkv + 门控）、mHC 末端 pre_mix collapse、DSpark 块草稿（prefill 播种窗口，decode 每步一次 5 位置块 + Markov / 置信头；spec_k 上限为块大小）；视觉沿用 0.62。逐层参数与 safetensors 头一致。架构代理 → 完整。
- **Qwen3.8-Flash-Next 按 transformers qwen4_exp + SGLang qwen4_exp_mtp.py**：门控残差（低秩混合 + 注入，替代逐层 RMSNorm；末端 mixer、无最终 norm）、QSA 索引注意力（块均值池化打分、top-512 块 + 尾块，原始 key 缓存）、PLE n-gram 层（查表 + key / value 投影 + 门控 + 膨胀 depthwise conv 及其状态）、MTP 输入融合（每流 fc_hidden）。参数与发布逐项一致（原 +0.50 %）。架构代理 → 完整。
- 覆盖度：18 个模型收口为完整，目录中的 LLM 全部「完整」；剩余非完整 = 4 个结构预测「部分」（检索 / 松弛不在范围）；AlphaFold3 仍「暂未接入 v2」。
- 已知：`test_core_052::test_length_spread_effects` 在 548b9d5 上也超过 600 s（与本次改动无关，测试提速由主线处理）。
- 数值（另见 §2.1）：GLM-5.3-Flash / V4-Flash / V4-Pro decode TPOT +6 ~ +8 %、prefill 8k TTFT +23 ~ +31 %；DSA 模型 decode TPOT −9 ~ −13 %、长 prefill TTFT 最多 −49 %；线性注意力模型 prefill TTFT +3 ~ +17 %；V4.1-Flash / Qwen3.8-Flash-Next decode TPOT −23 / −31 %、prefill 8k TTFT +40 / +127 %、64k −23 / +34 %；已「完整」的 50 个模型默认结果不变（新测试钉住）。

## [0.66.0] - 2026-10-10

外部校核：用**公开发表的实测**（TensorRT-LLM 官方性能表、Databricks 推理工程博客）对照 accel-dse 的代价模型，并做留出法校准研究。只在本地，未发布。详见建模说明 §25。

### 性能（结果不变）
- `bulkq._tagged`（0.65 批服务排队的 tagged-request 分解，0.65.1 加原子后成为套件主耗时）：每个 ΣF 原子后的到达数只用到 pmf 前 ≤ b − 1 项，新 `_pois_head` 用同一递推只算这几项（逐位相同；全部下溢为 0 的情形直接返回）；同一块 F 的三个 ΣF 原子与 cdf 前缀和对所有 m0 共用；r ⊕ j 卷积改为稠密列表（同序同乘加）。核对：72 个目录模型 × 3 个服务点 evaluate 数值指纹 0.65.1 = 0.66.0；三份 disagg 报告（含 prompt_cv / out_cv）JSON 哈希逐位相同。
- 套件逐模块计时（box，`/workspace/prof66`）：test_core_052 1220 s → 302 s；逐模块总和 ≈ 2860 s → 1820 s（基线中 053 以后已部分含第一步提速，真实旧基线更大）。

### 新增
- `core/refhw.py` 参考硬件（仅校核用，不进 `CHIPS`）：H100-SXM、H200-SXM、A100-SXM-40GB / 80GB。峰值 FLOPS（稠密）、HBM 带宽 / 容量、NVLink 带宽取自厂商数据手册（逐条带链接）；其余（等效阵列几何、L2 作 SRAM、链路 α、映射 reconf、DRAM 效率）沿用模型的「假设」，**未按实测调**。
- `data/ext_measurements.json`：100 条实测（逐条带 URL，数值照抄）：TensorRT-LLM 0.8 静态批吞吐（H100 / H200 FP8、A100 FP16；LLaMA-2 7B / 70B、Mistral 7B；batch / TP / ISL / OSL 全给定）36 条、batch 1 首 token 延迟 18 条；TensorRT-LLM 0.21 满载吞吐（H100 / H200，Llama-3.1-8B TP1、Llama-3.3-70B TP2，FP8）44 条；Databricks A100-40GB Llama-2-70B TTFT 2 条。目录新增 `llama-2-7b`、`llama-2-70b`、`mistral-7b`（仅校核，unlisted）。
- `core/extval.py`（V6）+ `scripts/ext_validate.py` → `data/ext_validation.json`；`scripts/ext_calib.py`（可续跑网格）+ `scripts/ext_calib_report.py`（留出法报告）。
- `Scenario.exec_overlap`（「假设」，默认 `stage` = 0.65 行为，场景哈希不变）：`class`（GEMM ‖ 权重流量、attention ‖ KV 流量，两类与集合通信串行——逐 kernel 执行）、`kernel`（再加向量 kernel 串行）、`serial`（完全不重叠，上界）。`StageTime` 新增 `t_arr_attn`、`t_dram_kv`。

### 发现（未改默认）
- **级内全重叠是主要系统误差**：默认 `stage` 把整个流水级的计算、DRAM、链路取一个 max，对 GPU 实测吞吐偏乐观（几何均值 1.23×（满载）/ 1.32×（静态），峰值带宽时 1.50× / 1.60×），70B TP2 尤甚（TP all-reduce 被藏在计算后面）。`serial` / `kernel` 把满载吞吐误差从 32.5 % 降到 22.5 % / 22.6 %（目录 DRAM 效率），峰值带宽 + `serial` 14.9 %。
- **不引入拟合的效率系数**：在三种留出切分（按硬件、按模型、按数据源）上拟合 (DRAM 效率, MAC 效率)，留出误差相对未调参数只降 0 ~ 4 个百分点，`kernel` 模式按模型 / 按数据源切分反而变差（35.9 % → 41.6 %、39.5 % → 45.5 %）。系数不外推，保持纯数据手册峰值 + 目录默认效率。
- **短 prompt 的 TTFT 低估 2 ~ 5×**：128 token 时实测比模型多出约 0.11 ms / 层（H100 / H200）、0.2 ms / 层（A100）——kernel 启动 / 同步一类每层固定开销，模型没有；2048 token 时 7B 误差 −7 % ~ +4 %（H100 / H200）。
- TensorRT-LLM 0.8 上 Mistral-7B（GQA 8）与 LLaMA-2-7B（MHA）的吞吐几乎相同，模型因 KV 少 4× 预测 Mistral 快约 2×；「KV 不按 GQA 复用」假设能解释 0.8 的 Mistral，但会把 0.21（现代 kernel）的吞吐低估一半——只作为软件版本差异记录。

## [0.65.1] - 2026-10-10

V4 网格按 0.65 的闭式与 DES 稳定性判据重跑（顺序、逐点可续跑），并修正重跑中发现的批服务律原子截断。详见建模说明 §24.8。

### 修正
- **批墙钟律原子数**（`bulkq.MAX_ATOMS` 6 → 24、`PER_ATOMS` 12 → 48，`pdqueue.OTHER_ATOMS` 4 → 16、`OWN_ATOMS` 32 → 48）：0.65.0 把每请求墙钟律压到 12 个原子、批墙钟律压到 6 个，长度 CV 1 时服务时间的尾部被削平，嵌入链的等待（依赖服务律二阶以上的矩）偏小，上限 2 显得比上限 1 好。Qwen3-8B TP2 V4 load 0.3 CV 1（强制上限 2）：TTFT 均值 / p90 / p99 176.5 / 567.3 / 976.7 → 188.6 / 681.5 / 1150.4 ms，DES 703 / 1236（p90 误差 −19 % → −3 %）；上限 1 的均值 185 ms，所以闭式改选上限 1（与 DES 一致）。0.65.0 的 V4 重跑（修正前，18 点）PD TTFT p90 最差 −27 %、SLO goodput −27 %（乐观），修正后回到 0.57 水平（§24.8）。
- 指纹（0.65.0 → 0.65.1）：单节点 0 / 8136，多节点 0 / 5859，fabric 0 / 126758，fp63 0 / 5854；fp64 中 52 行上限 2 → 1（PD 28、prefill 优先 24），PD p90 −0.3 % ~ +25.1 %、p99 −0.1 % ~ +17.6 %、均值 0 ~ +7.8 %，prefill 优先 p90 −0.9 % ~ +24.4 %、SLO 速率 −15.2 % ~ 0，PD SLO 速率 / 最佳分配 −1.3 % ~ 0。

### 变更
- `v4_grid(partial_dir=, only=)`、`scripts/v4_serving.py --partial DIR --only N`：每点一个 JSON，已有的跳过（中断后续跑）；行里记 `elapsed_s`。
- `data/v4_serving.json` 重跑（0.65.1 闭式 + 绝对漂移稳定判据；load 0.85 每 seed 20000 请求，其余 3000，3 seed，单进程顺序）。30 点三种模式 DES 全部稳定；误差与 0.57 网格基本相同（落在容差内全部 100 %；PD TTFT p90 −6 % ~ +5 %（0.57：+6 %）、p99 −6 % ~ +6 %（+7 %），PD SLO goodput −2 % ~ +16 %（−1 % ~ +16 %），prefill 优先 SLO goodput −12 % ~ +6 %（−11 % ~ +5 %）），见 §24.8。

## [0.65.0] - 2026-10-10

收尾 0.64 的「未做」：prefill 批服务器改为精确的贪心批服务排队（M/G^[b]/1），DES 稳定性改为漂移 / 利用率判据，EP 空闲 rank 的 SRAM / 链路 / 反量化计数，输出驻留分块搜所有整数行块。新增性质测试 `tests/test_core_065.py`（10 项）。详见建模说明 §24。

### 变更
- **prefill 批服务器 = 贪心批服务排队**（`core/bulkq.py`，上限 ≥ 2）：空闲时取 min(队列, b) 个请求成批，墙钟 = 批内 TTFT(k, S_i) 的均值（与 DES 相同）。服务开始时刻的嵌入链精确求解（带状消元 + 几何尾），标记请求按年龄积分得到 TTFT 律（等待 + 自己批的墙钟 + KV 暴露，按 prompt 类型精确混合）。上限按精确均值选，分位数只对选中上限求。0.51–0.64 用流体 M/G/1（τ_b = T(b)/b，满批假设），部分负载下低估占用：Qwen3-Next-80B-A3B PD（SLO 80 / 100 ms）SLO 速率 2.564 → 1.591 req/s（DES 1.60，误差 −0.7 %，原来 +72 %）；负载 0.6 p90 TTFT 79.4 → 85.4 ms（DES 84.6）。
- **prefill 优先的忙时份额**用批服务器的真实忙时（部分批更长）：Qwen3-Next 稳定速率 2.803 → 2.606（**新发现**：0.64 在负载 0.95 报「cap 4 稳定」，DES 在 λ 2.75 时任何上限都不稳定）。
- **DES 稳定性**（`pdsim.stability`）：完成 + PD prefill 忙时 < 99.5 % + TTFT 与 TTFT 之后时间的窗口均值漂移 |d| ≤ 0.5（5 个窗口，预热丢弃）。`slo_rate` 与 V4 行用它（`stable_des`、`drift_*`）；原来「全部请求完成」对过载的有限运行也成立。
- **EP 空闲 rank**：SRAM 端口字节 / 反量化只计专家 GEMM，链路字节只计 EP dispatch / combine / expert all-reduce 份额（`StageResult.exp_acts`）。DP2·EP2 B1 decode 向量计数 MiniMax-M1 −8.4 %，Qwen3-30B-A3B-FP8 −49.3 %，Qwen3-235B-A22B-FP8 −48.7 %（fp8 专家权重反量化只算专家部分）。
- **输出驻留分块**搜所有整数 bm（O(√m) 个不同的 ⌈m/bm⌉），是整数块下的精确最优：视频 / 蛋白质 full 与部分 prefill 的 DRAM 计数 −1.6 % ~ −7.4 %。

- **校核**（§24.2）：prefill 受限 QN 强制上限 1 / 2 / 4 / 8 / 自动、CV 0 / 1，PD 闭式 SLO 容量对 DES −2.5 % ~ +1.1 %，p90 TTFT 在负载 ≤ 0.7 时 ≤ 3 %（负载 ≥ 0.85 闭式偏保守，有限长 DES 尾部偏低）；prefill 优先 SLO 容量 −3.7 % / +9.0 %（单 seed），p90 +3 % ~ +12 %（保守）。

### 指纹变化（0.64.0 → 0.65.0）
- 单节点 0 / 8136；fabric 0 / 126758。
- 多节点 12 / 5859：prefill 优先稳定速率 −0.3 % ~ −0.6 %，SLO 速率 +0.02 %。
- fp63 75 / 5854：DRAM（整数分块）−1.6 % ~ −7.4 %；空闲 EP 反量化 −8 % ~ −49 %；PD / prefill 优先速率见上。
- fp64（PD TTFT 80 行）：PD 上限 1 → 2 / 4 / 8 共 56 行，PD p90 −20 % ~ +36 %；prefill 优先 4 行由稳定变为不稳定（与 DES 一致），逐类见 §24.6。

### 未做
- 能耗按选中上限的满批 Result 计（部分批的能耗未分摊）。
- V4 网格（`scripts/v4_serving.py --slo`）未按新稳定判据重跑，`data/v4_serving.json` 仍为 0.57 的结果。
- 近饱和（负载 ≥ 0.85）DES 尾部需更长运行才能判定闭式偏差。
- 批墙钟律在 k > 8 时用 5 点 Gauss–Hermite（忽略偏度），8 以上非 2 的幂的 k 线性插值。

## [0.64.0] - 2026-10-09

收尾 0.63 的「未做」清单：SLO 感知的 prefill batch 上限、DES SLO 搜索、二维 GEMM 分块、闭式运行 batch 两点混合、不整除 TP / EP 的反量化、EP 空闲 DP rank 的 DRAM 字节。新增性质测试 `tests/test_core_064.py`（12 项）。详见建模说明 §23。

### 变更
- **SLO 感知的 prefill batch 上限**（PD / 合并 prefill 优先）：默认上限（平均 TTFT 最小）不满足 SLO 时，在满足 SLO 的上限中取平均 TTFT 最小者（`prefill.cap_rule = "slo"`）；可行集为各上限可行集之并，扫描 + 细化搜索不变。Qwen3-Next-80B-A3B（TP2·EP2 / TP1，B16，SLO 80 / 100 ms）PD SLO 速率 1.580 → 2.564 req/s，负载 0.6 时 p90 TTFT 86.4 → 79.4 ms（上限 1 → 2，均值 76.3 → 80.8 ms）。
- **prefill 优先的稳定上限回退**（新发现）：0.63 只按 prefill 队列选上限，可能让 decode 份额不稳定而其他上限稳定。现在默认上限不稳定时取稳定上限中平均 TTFT 最小者（`cap_rule = "stable"`）。上例负载 0.95 由「不稳定」变为 cap 4、p90 67.0 ms；SLO 速率 2.605 → 2.779。
- **DES `slo_rate`**：完成上限增长 + 二分，再在其下扫 8 点并细化（原来直接二分 SLO 判据）。
- **二维 GEMM 分块**：激活溢出在权重驻留 / 激活驻留 / 输出驻留三种嵌套中取最便宜者（`memplan.gemm_blocking`），流量 ≤ 0.63 且随 SRAM 不增。Qwen3-8B B4 4K prefill 溢出 16 / 64 / 256 MiB：398.6 / 149.8 / 101.5 → 305.6 / 143.8 / 96.6 GB。
- **闭式运行 batch 两点混合**：⌊n̄⌋ / ⌊n̄⌋+1 按小数部分混合（步时、残余、u_coll、能耗、分块 k_s），`running_batch` 报小数。prefill 优先 TTFT −3.1 % ~ +2.5 %。修掉了 0.63 分块模式在稳定上限处的错误不动点：Qwen3-30B-A3B TP2·EP2 分块 SLO 速率 0.715 → 0.177（0.18 以上 TPOT 都超 50 ms）。
- **反量化按平均 rank 的权重量**（`Op.wshare`）：qwen3-32b-awq decode 向量计数 TP3 / TP6 = TP1（原来 +2.4 % / +9.6 %）。
- **EP 空闲 DP rank 只计专家权重字节**（`dram.exp_dram / exp_slc`）：Qwen3-30B-A3B DP4·EP4 B1 decode DRAM / token 14.81 → 6.42 GB；DeepSeek-V3 DP8·EP8 B1 160.0 → 37.9 GB。

### 指纹变化（0.63.0 → 0.64.0）
- 单节点 0 / 1356；fabric 2 / 3150（DeepSeek-V3 PD TTFT −0.28 %）。
- 多节点 170 个值，都在两个 PD 报告：Qwen3-8B 合并分块 TTFT −5.9 % ~ −9.0 %；Qwen3-30B-A3B 分块 SLO 速率 0.715 → 0.177。
- fp63 381 / 840：DRAM 计数全部不升（prefill 最多 −50.8 %，DP·EP decode −12 % ~ −47 %），向量计数 TP3 / TP6 −0.5 % ~ −2.6 %；步时 / MAC 不变。
- fp64（PD TTFT 80 行）：418 个 TTFT 值变化，逐类见 §23.6。

### 未做
- 闭式上限 2 的 p90 比 DES 乐观；DES `complete` 是弱稳定性代理。
- 空闲 EP rank 的 SRAM / 链路 / 反量化计数。

## [0.63.0] - 2026-10-09

外部评审（external review）对 0.62.1 提出 9 条问题和 1 个覆盖缺口，本版全部修正。每条都先复现，再加永久的性质测试（work conservation、调度依赖合法性、容量硬约束、搜索结果不低于已知可行点）。详见建模说明 §22。

### 修正
- **[P1] MLA decode 吸收路径**：kv_b 原来建成一个宽 GEMM（DS-V3 (1, 512, 32768)），现按官方 model.py 改为逐头的 W_UK / W_UV 两组 GEMM（`attn.kv_b.uk` / `attn.kv_b.uv`）。参数、FLOPs、格式不变。100T kv_b 周期：OS 9472 → 40960，ws_broad 2215 → 3568。MLA 模型 decode 步时 +0.2 % ~ +11 %。
- **[P1] 视频 CFG 依赖**：同一请求的 cond / uncond 必须在同一去噪步内都完成。被同一请求连在一起的微批组 k > 1 时，每步按 max(mb, k + PP − 1) 个 tick 计。Wan2.1-1.3B B1 PP2 CFG2 1 步 3.255 → 4.883 s；PP > 1 的视频延迟 +30 % ~ +49 %，PP1 不变。
- **[P1] SLO 最大吞吐搜索不再假设单调**：先二分求稳定上限，再在 (0, λ_s] 上自上而下扫 24 点并二分细化，结果不低于任何已知可行点（含工作点）。Qwen3-Next-80B-A3B 复现例：1.724 → 2.892 req/s（2.834 可行）。
- **[P1] radix 前缀树容量依赖**：路径累计大小放得下才可缓存，子节点命中 ≤ 父节点。DES 的 RadixLRU 用同样的截断。两层 × 60 token、容量 100：[0.833, 0.833] → [1, 0]。
- **[P2] 不整除的 TP 不再丢头**：GQA 在「组完整」与「按头分」两种头完整的切分中取较便宜者；q / o / k / v 投影按头对齐。Qwen3-8B TP3 decode b8 35.89 → 36.42 ms（原来只覆盖 27 / 32 个头）；Phi-4 TP8 prefill +7.0 %。
- **[P2] PP 级间路由逐对评估**：每一对对应 rank 按各自跨越的层级计，取最慢的一对。PP2·TP64、8 卡 / 节点、4:1 收敛：1 → 4 ms。
- **[P2] 有效工作与最忙 rank 时间分开**：新增 `Op.share`，能耗 MAC / 向量计数与每请求 TFLOP 按有效工作计。以下几项只占时间，不计工作：不整除切分、专家补齐行、最后一个微批补齐、视觉编码器补齐。复制计算与每遍的权重反量化照常计。EP > 1 时没有序列的 DP rank 仍运行专家。例：ESM-2 650M B3 PP2 MAC / 序列 4.756e11 → 3.567e11（= PP1）；视觉 3 图 TP2 不再按 4 张计；Mixtral B3 decode 的 PP1 / PP2 / DP2·EP2 MAC / token 都是 40.4 G。
- **[P2] DES 投机解码**：每序列每步抽截断几何分布的接受数，均值等于闭式期望（原来四舍五入 1.7 → 2）。DS-V3 spec_k 1、a 0.7：DES / 闭式 TPOT 3.957 / 5.727 → 5.809 / 5.943 ms。
- **[P2] 线程安全**：fabric 的 `_LOG` / `_MULT` 与 pdsim.capture_ctx 的猴子补丁改为 contextvars（`fabric.log() / set_log() / set_mult()`，`disagg.CTX_SINK`）。

### 新增
- **LLM 激活流式 / 溢出**（覆盖缺口）：LLM 的 GEMM、lm_head、MTP 与 prefill 注意力使用全序列前向的流式模型：
  - staging ≤ SRAM/2；
  - GEMM 的分块按实例决定；
  - 注意力的前缀 KV 只读一次；
  - DRAM 容量加上超过 1 GiB 预留的激活工作集。

  Qwen3-8B B4 4K prefill 在 16 / 64 / 256 / 1024 MiB SRAM 下激活 DRAM 398.6 / 149.8 / 101.5 / 33.8 GB（原为 0）。近似：两端分块，不是最优二维分块。

### 指纹变化（0.62.1 → 0.63.0）
- 单节点 470 / 1356 项：MLA decode +0.23 % ~ +3.29 %（34）；视频 PP2 +37.4 % ~ +49.5 %（32）；prefill DRAM 需求 +0 % ~ +9.0 %（404），其中 fits 由是变否 2 项（gpt-oss-120b LPDDR5X 64 GiB B8 prefill，62.98 → 64.09 GiB）。prefill 步时不变。
- 多节点 34 / 985 项：视频 PP2·TP2 +30 % ~ +49 %；Phi-4 prefill TP8 +6.97 %、PP2·TP4 +5.55 %；PD 能耗计数，以及 Qwen3-8B 合并分块 decode 迭代 67.9 → 74.3 ms。
- fabric 2 / 3150 项：DeepSeek-V3 PD 中放不下的单卡 prefill 池 TTFT 3520 → 3871 ms。集合通信全部不变。
- 新指纹（能耗 / TFLOP / radix / PD 速率，840 项）：653 项变化，逐类原因见建模说明 §22.11。

### 未做（已写入文档）
- 让 prefill batch 上限的选择考虑 SLO（会改变全部 TTFT）。
- DES slo_rate 仍是二分（只用于校验）。
- 闭式 k̂ 取整。
- 最优二维 GEMM 分块。

### 测试
- `tests/test_core_063.py`（16 项）：MLA 算子与周期、CFG 依赖链与调度合法性性质、SLO ≥ 可行点、radix 容量性质、GQA 头完整、有效工作守恒性质（LLM / MoE / MLA / 混合 / 视频 / 蛋白质）、专家补齐、能耗随切分与补齐不变（含 MoE）、视觉补齐、PP 逐对路由、DES 抽样均值、fabric 线程安全、LLM 溢出随 SRAM 变化。

## [0.62.1] - 2026-10-09

第六轮收敛审计：0.62 改动回归复查、可选功能组合模糊测试、文档数值逐条复现。**代码与计算结果不变**（无新代码缺陷），只修正过时的文档数值并补测试。

### 审计
- 组合模糊测试 2400 个场景（SLC + PD + fabric + 前缀 / radix + KV 策略 + 倾斜 + 投机解码 + 图像 + 预算 + 能耗随机组合）：确定性、JSON 往返、输出有限、中性值恒等（images = 0、slc_mib = 0、fabric 关、spec_k = 0、PD 同布局、PP = 1 的切分方式、package_cards）、单调性、能耗守恒，以及 eval / sweep / search / Pareto / API 一致，全部通过；PD 恒等模糊测试 160 个场景无反例。
- 已知、不修（写入建模说明）：PP 按代价切分的候选代理只看 DRAM 字节，个别配置下带宽提高反而换到略慢的切分（1200 个场景中 1 例，+9.6 %，仍优于按层数均分，§21）；PD prefill batch 上限按平均 TTFT 选，p90 可能随负载在换挡处下降（§18.1）；Kimi-K2.5 图像缩放的 token 数不随分辨率严格单调，与参考实现 navit_resize_image 一致。

### 文档数值更正（建模说明）
- §19 例 A（DeepSeek-V3 DP64·EP64）、§19.4、§19.5（Kimi-K2 / DS256 / 端口）按 0.61.1 起 fp8 激活发布的 combine / all-reduce 按 bf16 计重算，原文仍是 0.59 / 0.60 的值：例 A prefill 关 fabric 715.9 ms（LINK 受限），leaf 2 · r = 4 为 2453 ms；Kimi-K2 decode 两层 r = 4 由 MAC 受限变为 LINK 受限 7.22 ms；§19.4 KV 反馈 TPOT 15.80 → 16.50 改为 23.25 → 23.94 ms；旧值保留在括号里。
- §2 / §9 V1 发布数 61 → 72（55 个 LLM / VLM + 10 个视频 + 7 个蛋白质）；§12 ESMFold fp32 激活例 6.3 → 12.5 s 改为 5.0 → 10.0 s（0.61.2 起主干 4 遍）；文首版本改为 0.62.x 并说明各节数值按所注版本。
- 其余数值例（§9 V0–V3、§11.4、§12、§14–§16、§18、§19 例 B–D、§19.1、§19.8、§19.9、§19.14、§20、README）逐条复现一致。

### 测试
- `tests/test_core_062_1.py`：发布数与参数误差、ESMFold fp32 例、fabric 文档例、可选功能中性值恒等、图像扫描与 eval 一致。

## [0.62.0] - 2026-10-09

第五轮：补上两个已知的覆盖缺口（VLM 视觉编码器；PP 按代价切分，设为默认），视频卸载的主机流量能耗，部分共享前缀字节核对，第四轮修正回归。

### 新增
- **VLM 视觉编码器**（建模说明 §20）：目录里 8 个 VLM 的视觉塔 + 合并 / 投影按发布 vision_config 与图像预处理器建模（patch、动态分辨率、空间合并、进入语言模型的 token、bf16 按发布）。7 个模型 × 3 种分辨率的 token 数与 FLOPs 与 transformers / 发布 remote code 的 FlopCounterMode 计数逐一精确一致，参数等于发布 params_vision。场景 `serving.images`（默认 0 = 只算文本，既有结果不变）、`serving.image_w / image_h`（默认 1024×1024「假设」）；编码器在 prefill 时运行（首级数据并行「假设」），图像 token 进入 prompt / KV / PD 长度；`summary.vision`；CLI `--images --image-size WxH`；Web 服务组输入（只在 VLM 显示）与「视觉编码器」卡片；扫描路径 serving.images / image_w / image_h。覆盖标签改为「VLM · 含视觉编码器」。
- **PP 按代价切分（默认）**（§21）：`pp_split = "cost"` 用逐层代价（含嵌入 / 输出头 / MTP / 级间传递）求使最慢级最快的连续切分，候选与按层数均分一起由完整模型裁决（节拍不劣、不会失去 fit）；`pp_split = "layers"` 保留旧切分；CLI `--pp-split`，Web 布局组「PP 切分」，CLI eval 打印各级层范围。

### 修正（§19.14）
- **PD 带缓存前缀的 KV 交接字节**按层类型计（滑窗 min(W, S − p)、递归状态整份、压缩条目按 ⌈·/c⌉），原为整请求 × (S − p)/S。例 S 8192 / p 4096：Qwen3.8-27B 346.9 → 425.3 MB，Kimi-K3 345.6 → 577.9 MB，GLM-5.3-Flash 134.0 → 210.3 MB；纯 GQA 不变。
- **DeepSeek-V4.1-Flash** 对齐器（73.41 M 参数）从 other 归入 vision：params_llm 748.568 → 748.495 B，params_vision 0.412 → 0.485 B，参数偏差 −0.051 % → −0.042 %（`scripts/summarize_release.py` 视觉正则补 aligner）。
- **视频卸载重载**的主机链路字节（每次重载全副本）按 `energy.pJ_bit_host` 计入能耗 `host`，未填时 `host_note`。例 Wan2.1-14B 单卡：69.0 GB / 段。
- **Kimi-K2.7-Code** 补进图像预处理表（与 K2.5 相同）。
- 关于页与建模说明 §2 的范围说明过时（视频 pipeline、结构模型已接入；能耗 / 面积是用户表驱动），已改写。

### 指纹变化（0.61.4 → 0.62.0）
- 默认场景（PP = 1）、images = 0 的所有结果、场景哈希：不变。
- 变化只在 PP > 1，全部是 cost 切分带来的节拍下降（0 个变慢、0 个 fits 变化；step 有变化的配置：单节点指纹 134 / 1356、多节点 168 / 985、fabric 715 / 3150），按模型（step_ms 变化，mem × mapping 或节点配置的范围）：
  - alphafold2：PP2·TP1 full -12.0…-7.3%；PP4·TP1 full -24.9%
  - boltz-1：PP2·TP1 full -35.5…-34.7%；PP4·TP1 full -49.7%
  - deepseek-v3：PP2·TP1 decode -3.2…-1.0%；PP2·TP2·DP2·EP4 decode -3.1%
  - deepseek-v3.2：PP2·TP1 decode -3.2…-1.6%；PP2·TP2·DP2·EP4 decode -3.0%
  - deepseek-v4-flash：PP2·TP1 decode -0.0%
  - deepseek-v4.1-flash：PP2·TP1 prefill -6.0…-5.7%；PP2·TP2·DP2·EP4 prefill -6.0%
  - esmfold：PP2·TP1 full -34.7…-34.3%；PP4·TP1 full -46.6%
  - glm-4.5-air：PP2·TP1 decode -1.2…-1.0%；PP2·TP2·DP2·EP4 decode -2.3%
  - glm-4.6：PP2·TP1 decode -2.1%；PP2·TP2·DP2·EP4 decode -2.1%
  - glm-5.2：PP2·TP1 decode -2.5%；PP2·TP2·DP2·EP4 decode -2.5%
  - glm-5.3：PP2·TP1 decode -3.6…-3.2%；PP2·TP2·DP2·EP4 decode -3.4%
  - glm-5.3-flash：PP2·TP1 decode -3.6…-3.1%；PP2·TP2·DP2·EP4 decode -3.5%
  - gpt-oss-120b：PP2·TP1 decode -9.1…-0.8%；PP2·TP2·DP2·EP4 decode -9.3%
  - gpt-oss-20b：PP2·TP1 decode -12.3…-2.1%；PP2·TP2·DP2·EP4 decode -12.5…-12.2%
  - llama-3.1-8b：PP2·TP1 decode -5.6…-2.4%；PP2·TP4 decode -4.7…-4.6%；PP4·TP2 decode -11.0…-11.0%
  - llama-3.2-1b：PP2·TP1 decode -12.6…-3.4%；PP2·TP4 decode -10.7…-10.7%；PP4·TP2 decode -36.2…-35.8%
  - llama-3.2-3b：PP2·TP1 decode -6.0…-3.7%；PP2·TP4 decode -6.4%；PP4·TP2 decode -20.4…-20.2%
  - llama-3.3-70b：PP2·TP1 decode -0.4…-0.0%；PP2·TP4 decode -0.3%；PP4·TP2 decode -0.9%
  - magistral-small：PP2·TP1 decode -0.7…-0.6%；PP2·TP4 decode -0.3%；PP4·TP2 decode -1.2%
  - openfold：PP2·TP1 full -3.6…-2.4%；PP4·TP1 full -0.5%
  - phi-4：PP2·TP1 decode -1.6…-1.5%；PP2·TP4 decode -0.7…-0.7%；PP4·TP2 decode -3.0%
  - protenix：PP2·TP1 full -37.9…-22.7%；PP4·TP1 full -50.4%
  - qwen2.5-72b：PP2·TP1 decode -0.9…-0.4%；PP2·TP4 decode -0.8%；PP4·TP2 decode -1.7%
  - qwen3-0.6b：PP2·TP1 decode -7.0…-6.1%；PP2·TP4 decode -5.7…-5.0%；PP4·TP2 decode -24.2…-23.6%
  - qwen3-1.7b：PP2·TP1 decode -9.9…-3.7%；PP2·TP4 decode -6.5%；PP4·TP2 decode -27.1…-26.8%
  - qwen3-235b-a22b：PP2·TP2·DP2·EP4 decode -0.1%
  - qwen3-235b-a22b-fp8：PP2·TP1 decode -2.0…-0.9%；PP2·TP2·DP2·EP4 decode -3.9%
  - qwen3-30b-a3b：PP2·TP1 decode -2.9…-0.7%；PP2·TP2·DP2·EP4 decode -3.7…-3.6%
  - qwen3-30b-a3b-fp8：PP2·TP1 decode -10.0…-3.3%；PP2·TP2·DP2·EP4 decode -10.0%
  - qwen3-32b：PP2·TP1 decode -1.5…-0.6%；PP2·TP4 decode -1.2%；PP4·TP2 decode -2.8%
  - qwen3-32b-awq：PP2·TP1 decode -5.5…-0.6%；PP2·TP4 decode -5.5…-5.3%；PP4·TP2 decode -14.7%
  - qwen3-32b-fp8：PP2·TP1 decode -2.9%；PP2·TP4 decode -2.9%；PP4·TP2 decode -9.7%
  - qwen3-4b：PP2·TP1 decode -4.9…-3.0%；PP2·TP4 decode -5.1%；PP4·TP2 decode -16.5…-16.4%
  - qwen3-8b：PP2·TP1 decode -4.9…-4.0%；PP2·TP4 decode -5.0%；PP4·TP2 decode -15.2…-15.1%
  - qwen3-8b-awq：PP2·TP1 decode -13.7…-4.0%；PP2·TP4 decode -9.8…-9.1%；PP4·TP2 decode -34.6…-34.2%
  - qwen3-8b-fp8：PP2·TP1 decode -8.9…-4.9%；PP2·TP4 decode -8.3…-7.8%；PP4·TP2 decode -22.6%
  - qwen3-next-80b-a3b：PP2·TP1 decode -3.1…-1.0%；PP2·TP2·DP2·EP4 decode -3.3…-3.3%
  - qwen3.8-27b：PP2·TP1 decode -3.4…-3.3%；PP2·TP4 decode -3.1…-3.0%；PP4·TP2 decode -11.5%
  - qwen3.8-flash-next：PP2·TP1 decode -3.9…-3.8%
  - seed-oss-36b：PP2·TP1 decode -1.2…-0.4%；PP2·TP4 decode -0.9%；PP4·TP2 decode -2.2%
  - fabric 指纹：deepseek-v3 PP2·TP4·DP4·EP16 −2.6 %、PP4·TP1·DP16·EP16 −11.0 %；llama-3.3-70b PP4·TP8 −0.2 %、PP8·TP4 −1.2 %；qwen3-32b PP8·TP2 −4.3 %（各 143 个拓扑 / 算法组合）。
  - 视频（Wan / CogVideoX / MiniMax-H3）只在 PP4 有 ≤ 0.1 % 的变化，指纹 PP2 无变化。
- 两个旧测试改为显式 `pp_split = "layers"`（MTP 只在末级、DeepSeek PP8 容量手算），一个 0.61.4 测试按新警告文本调整。

## [0.61.4] - 2026-10-09

第四轮审计：第三轮修正回归、PD prefill 池预算、swap 主机链路能耗、混合格式、VLM 视觉编码器、投机解码 / MTP、长上下文、batch > 1 视频 / 蛋白、Pareto、数值极端。

### 修正（见建模说明 §19.13）
- **DP 空闲 rank**：batch 不满 dp 时能耗与视频 TFLOP 不再按全部 rank 计。例 ESMFold batch 1 · DP2 每单位 MAC 194.65 → 97.32 TFLOP、1148 → 1078 J；Wan2.1-1.3B batch 1 · DP4 每请求 29440.6 → 28585.3 TFLOP。时间与容量不变。
- **PD**：prefill 布局卡数不整除 decode_cards 时不再报错；预算核对 prefill 池（`pd.budget`、`budget.ok_with_pd`）。
- **能耗**：新增 `energy.pJ_bit_host`（swap 主机链路，无默认值；未填时给出 `host_note`）。
- **混合格式**：fp8 × int8、fp16 × bf16 升到 bf16 执行（原来按同位宽原生）。发布模型不受影响。
- **Pareto**：默认 batch 扩到 512 以上（Qwen3-8B TP8 前沿最大 batch 384 → 3072）；过滤 NaN 点。
- 新警告：上下文超过发布 max_position_embeddings；PP 流水级按层数均分导致的不平衡（> 1.5×）。CLI 标注 VLM 视觉编码器未建模。

## [0.61.3] - 2026-10-09

第三轮审计：结构模型端到端 FLOPs 对独立计数（OpenFold / AF2 / Boltz-1 / Protenix 发布实现的 FlopCounterMode 计数）、Open-Sora VAE、DES 新场景不变量、PD 能耗与预算、前两轮修正的回归，以及芯片预设、SLC、dtype、场景哈希、CLI 与 README 示例。

### 修正（见建模说明 §19.12）
- **AF2 / OpenFold**：模板扭转角行并入 MSA 网格（参考实现拼接 T 行），扭转角嵌入按 T × N 行，OpenFold 全局列注意力 q 按行均值。默认 396 → 398 / 354 → 355 TFLOP，与参考计数相差 ≤ 0.02 %。
- **Boltz-1**：推理缓存（pair 条件、每层 pair 偏置投影、原子编码器特征只算一次），检查点别名张量去重。默认 296 → 262 TFLOP，批延迟 13.05 → 7.94 s；参数 606.38 M → 592.01 M。
- **Protenix**：扩散 transformer 的 pair 偏置按样本计（参考实现沿样本展开 z），原子编码器投影按原子 / pair 行。默认 419 → 434 TFLOP，批延迟 19.33 → 27.86 s。
- **Open-Sora VAE**：时间解码器 conv_blocks 按上采样前的分辨率。1158 → 1140 TFLOP。
- **冷存储**（查表 embedding、待机专家）不再占 SRAM / SLC 驻留；SLC pin 例 20.8 → 20.0 ms。
- **PD 能耗**计入 recompute 的重算 prefill 与 swap 的 DRAM 读写。
- PD 开启时预算报告注明范围；最佳 batch 无解时警告，CLI 显示 TPOT SLO；格式速率 int / float 哈希一致。

## [0.61.2] - 2026-10-09

第二轮对抗式审计：PD 排队 / DES 数学（小规模精确算例、Lindley 模拟、不变量），VAE 解码与结构模型端到端 FLOPs 对独立计数（torch FlopCounterMode 跑 diffusers / transformers 模块），能耗与预算，radix 前缀缓存，HBM 几何，非默认映射，MoE 倾斜与 Wan2.2 待机专家，视频 / 蛋白非默认布局，界面竞态（脚本化浏览器），HF 发布 config 现网复核。

### 修正（见建模说明 §19.11）
- **LTX-Video VAE 解码**：上采样块的 resnet 按块输出分辨率计（diffusers 先上采样再跑 resnet），原来少 2.5–3.3×。默认 15.4 → 50.5 TFLOP，与 diffusers 计数一致。
- **ESMFold 主干遍数** 5 → 4（参考实现 `num_recycles=None` 时循环 `max_recycles` 次）。512 残基 120.9 → 97.3 TFLOP，与 transformers 计数一致。
- **decode 占用相关时间 τ_int**：平均占用较大时 Poisson 方程求和抵消，τ 可到 1e80；改为均值以下从下往上累加。
- **radix 前缀缓存**：比缓存还长的节点及其子树不驻留、不占容量（Che 与 DES 两侧）。
- **面积预算**：缺密度时，已知项已超限就判为超限。
- **界面**：在途的搜索 / 扫描 / 对比 / Pareto 结果在场景改变后标为过期，并按发出请求时的模型渲染。

### 数值变化（默认指纹 1356 项中 116 项、多节点 985 项中 63 项、fabric 3150 项中 0 项；fits / 瓶颈无翻转）
- LTX-Video 整段延迟：单节点指纹 +3.9 … +8.5 %（单卡 +3.9 … +4.3 %；多卡时 VAE 不并行，占比更大），多节点 +7.6 … +14.6 %；DRAM 需求 +0.54 GiB（解码激活峰值 1.19 → 1.73 GiB）。
- ESMFold 批延迟 −19.9 … −20.0 %，每请求 FLOPs −19.2 %（384 残基 62.2 → 50.2 TFLOP）/ −19.5 %（512 残基 120.9 → 97.3）。
- PD 报告：TPOT p90 / p99 −0.0 … −0.7 %（平均占用大的 decode 池），其余不变。

## [0.61.1] - 2026-10-09

第一轮对抗式审计：手算复核稠密 / MoE / MLA / DSA / CSA / 线性注意力 / 视频 DiT / 结构模型 / PD 的 FLOPs、字节、KV、权重驻留与通信，跨模块（eval / sweep / layouts / CLI / API）一致性，默认关闭项等价，API 模糊测试（约 1.3 万个用例），以及 1440 / 1280 / 1024 / 820 px 宽度下的界面。

### 修正（影响默认数值，见建模说明 §19.10）
- **DeepSeek-V4 压缩层 decode 的 KV 读**被多除了一次压缩比（r = 4 层少 4×，r = 128 层少 128×），现在按完整条目计。
- **滑窗 / top-k / 压缩注意力的 prefill** 改按每个 query 实际 key 数的精确平均计（原按上限计，DSA prompt 4096 时多 33%，prompt ≤ 2048 时 2×）。
- **DSA / CSA 索引器的 prefill** 改为因果（原 top-k 与压缩层按整方阵计）。
- **纯滑窗 latent 层**只存窗口内的 KV；有前缀缓存的 prefill 中，纯滑窗层只读最后一个窗口。
- **线性注意力**每步读写 conv 状态（原只计存储）。
- **fp8 激活的发布**：all-reduce、MoE combine、PP 交接按 bf16 计（dispatch 仍按 fp8）「假设」。
- **serving 尺寸**上限 2²⁴，并拒绝布尔值。
- 界面：窄卡片的数值自动缩小字号，不再被截成「…」；SLC 策略选项文字缩短。

### 数值变化（默认指纹 1356 项中 197 项、多节点 985 项中 168 项、fabric 3150 项中 1432 项；fits / 瓶颈无翻转）
- prefill 步时（100T 默认芯片，batch 8，prompt 4096）：
  - DeepSeek-V3.2 −13.8 … −15.2 %（多节点 −15.2 … −19.5 %）
  - GLM-5.3 −10.4 … −11.5 %，GLM-5.2 −6.4 … −7.3 %
  - DeepSeek-V4.1-Flash −9.5 … −16.8 %，V4-Pro −5.6 … −5.7 %，V4-Flash −3.2 … −4.1 %
  - GLM-5.3-Flash −3.1 … −3.9 %，gpt-oss −0.03 … −0.04 %
- decode 步时：
  - DeepSeek-V4 系 +0.1 … +0.6 %（多节点 +0.2 … +2.8 %；KV 读 / 步在 1P、ctx 32k–128k 时 +19 … +98 %，计算受限点步时不变）
  - 线性注意力混合模型（Qwen3-Next、Qwen3.5/3.8、Kimi-K3、GLM-5.3-Flash）+0.03 … +0.3 %
- DRAM 需求：DeepSeek-V4.1-Flash / V4-Flash −0.02 … −0.42 %
- fabric 开、多节点、链路受限的 DeepSeek-V3 prefill：+2.3 … +26.9 %（20 项）；其余 1412 项只有链路字节变化。

## [0.61.0] - 2026-10-09

性能（纯 Python、零依赖），以及交换机层数时延、NVLS 类 allgather / reduce-scatter、NCCL 第二棵树、传输协议（「假设」，默认关）。默认结果（1356 项指纹）与 0.60.0 逐字节一致（与旧 fp471 基线的 26 项 MiniMax-H3 / LPDDR 差异在 0.60.0 中已存在），985 项多节点 / D2D / PD 指纹与 3150 项 fabric 开的指纹均 0 diff；大规模 PD 报告全 JSON 与 0.60 对照 0 diff（B ≤ 1024 的个别值有 ≤ 3e-16 的舍入差）。

### 性能
- PD 排队链：泊松链截断判据的 O(N²) `max(logp)` 改为滚动最大值；`_occ_tau` 泊松闭式、批到达内积用 `math.sumprod`；B ≥ 256 时链越过众数并低于峰值 1e-20 即停（丢弃质量 < 1e-17）；步长表只建一次。
- 评估：算子代价与 fabric 集合通信记忆化（1024 卡布局排名中命中 29× / 9×），空尾部算子跳过。
- 布局搜索：HTTP 服务下各候选的 b_max 区间在进程池中计算（排名、剪枝、评估数与串行相同）；稳定性的扰动用例分发到进程池；默认 worker 数 5 → min(8, CPU)。

| 操作（box 8 核；1P + HBM3e，8 卡 / 节点） | 0.60 | 0.61 单进程 | 0.61 服务（8 worker） |
|---|---|---|---|
| PD 排队报告 DeepSeek-V3 256 副本 B4096 | 21.2 s | 3.5 s | — |
| PD 排队报告 Kimi-K2 B8192 | 48.4 s | 3.5 s | — |
| PD 排队报告 8 副本 B256 | 16.3 s | 2.4 s | — |
| PD 排队报告 B2048 / B1024 | 18.1 / 6.4 s | 3.8 / 2.3 s | — |
| 布局排名 top-16，DeepSeek-V3 1024 卡 | 7.5 s | 6.2 s | 2.7 s |
| 映射对比 1024 卡（DeepSeek-V3 / Kimi-K2） | 27.6 / 28.4 s | 20.3 / 21.8 s | 7.2 / 5.3 s |
| 映射对比 DeepSeek-V3 4096 卡 | 43.1 s | 35.6 s | 8.3 s |
| 排名稳定性 1024 卡 | 30.9 s | 25.0 s | 5.5 s |
| PD + 排队 + 布局搜索 256 / 1024 卡 | 22.5 / 14.6 s | 16.5 / 8.6 s | — |

### 新增（「假设」，默认关）
- `fabric.hop_spine_us` / `hop_core_us`：网络步离开 leaf / pod 时的额外每步时延（ring 每步、tree 按出 leaf / pod 的层数、网内归约、all-to-all、PP、PD KV 的期望值）。
- `innet_reduce = net+link`：交换式 scale-up 上 allgather / reduce-scatter 的 NVLS 类候选 `innet`（字节不变，一次交换机往返代替 n − 1 步）；`fabric.collective` 支持 `reducescatter`。
- NCCL 第二棵树：跨 leaf 切边份额按两棵树各一半（m | n 且 n 偶数时不变）。
- `fabric.protocol` = off | auto | LL | LL128 | Simple：带宽效率 ½ / 120/128 / 1，每步时延按 NCCL tuner 协议比例；候选名带后缀（`ring/LL128`）。
- CLI：`--hop-spine-us --hop-core-us --fabric-protocol`；Web：跨 leaf / 跨 pod 每步附加、传输协议输入，算法列显示协议；扫描参数 `fabric.hop_spine_us`、`fabric.hop_core_us`。
- 建模说明 §19.6–19.9；测试 `tests/test_core_061.py`（14 项手算 / 等价性）。

### 数值例子（1P + HBM3e，三层：每 leaf 4、每 pod 32 节点，r₁ = r₂ = 2）
- DeepSeek-V3 PP4·DP256·EP256（1024 卡）decode b1024：5.075 ms；hop_spine/core 1/2 µs → +2.4%；protocol auto → +3.1%（all-to-all 选 LL128）；全开 +5.4%。prefill +0.02–0.04%。Kimi-K2 同布局 decode +2.4 / +3.2 / +5.6%。
- DeepSeek-V3 TP8·DP128·EP1024 decode：hop 1/2 µs → +5.4%；NVLS 类 allgather 把 logits allgather 7.2 → 4.2 µs（−0.05%）。

### 剩余缺口
- ECMP / 拥塞 / incast；协议的通道数与每通道带宽上限未建模；
- reduce-scatter 不在推理算子图中；KV 回馈只回灌 decode 池；
- PD + 布局搜索在 B ≤ 1024 时每个状态仍需一次精确评估（256 卡约 16 s）；
- 映射对比在服务中按映射并行，单个映射内部串行；未做支配候选剪枝（没有可证明的支配关系）。

## [0.60.0] - 2026-10-09

大规模卡数，以及三层 fat-tree、端口并发、按暴露时间选算法（「假设」，默认关）。默认结果（1356 项指纹）与 0.59.0 逐字节一致，985 项多节点 / D2D / PD 指纹也一致。另有 3150 项 fabric 开、默认参数的指纹，只在 PP > 1 且 scale-up 拓扑 ≠ switch 的 144 项上不同（新增的 PP 多跳，见下）。

### 新增
- **大规模卡数**：
  - 每副本卡数上限 64 → 8192（API / UI / CLI；`catalog.max_cards`），PD 池卡数上限 4096 → 65536。
  - 布局搜索 / 最佳 batch 的每副本 batch 上限 = max(4096, 64 × 卡数)（原来固定 4096，DP1024 时截断最优点）。
  - 容量修复建议（`/api/fit`）的候选卡数延伸到 8192。
  - PD 布局搜索：总卡数 > 64 时加入 128、256、… 的副本尺寸，各尺寸配额相同。
  - PD 排队：decode 生灭链在 B > 1024 时对 step(k) 做几何网格插值（误差 < 0.5%）；链长不再截断于 4096；SLO 切分在候选 > 32 时先粗后细（与穷举一致，1024 + 1024 卡 80 s → 15 s）。
  - Web：卡数输入上限 8192，大于 64 卡时默认布局为 TP ≤ 每节点卡数，其余 DP（MoE）/ PP（dense）。
- **三层 fat-tree**（`fabric.net_tiers = 3`，`pod_nodes`，`oversub_spine`）：
  - leaf / spine / core，网络层 β 除数 = max(1, r₁·f_leaf, r₁·r₂·f_pod)；r₂ = 1 时与两层逐位相同。
  - PP 交接跨 pod 为 r₁·r₂；PD KV 为 max(r₁(1 − m/n), r₁r₂(1 − P/n))。
  - 每 pod 节点数默认 = 每 leaf 节点数 × spine 下行端口数（radix·r₂/(1+r₂)）。
  - 报告每步跨节点流量，以及离开 leaf / pod 的份额（`fabric.net_traffic`，每行 `net_leaf_share` / `net_pod_share`）。
- **端口并发**（`fabric.overlap = ports`）：D2D / scale-up / NIC 是不同端口，可同时工作；同端口串行。级链路时间 = max(最忙端口的忙时之和, 最长的单次集合通信)，在 sum（0.59）与最忙端口之间。报告 `fabric.stages`（每级 sum / ports、各层忙时、窗口、α）。
- **按暴露时间选算法**（`fabric.algo = auto_overlap`）：每个流水级对各类集合通信的算法组合最小化 max(计算 / DRAM 窗口, 链路) + Σα（≤ 729 种穷举，否则坐标下降），不劣于 auto 或任何强制算法。
- **PP 多跳**：fabric 开时，节点内 ring / torus2d / full-mesh 上的 PP 交接按链路负载计（ring 8 封装、S = 2 → 4×；torus 4×2、S = 4 → 4×；full mesh n − 1）。
- **KV 回馈**（`fabric.kv_feedback`）：KV 流占 decode 池 NIC 的份额 u_kv 回灌 decode 池的网络集合通信（不动点），报告 `kv.fabric.feedback`（u_kv、回馈前后 TPOT）。
- CLI：`--net-tiers --pod-nodes --oversub-spine --fabric-overlap --kv-feedback`，`--fabric-algo auto_overlap`；输出流量分层、端口并发和 KV 回馈。Web：网络层数、每 pod 节点数、spine 收敛比、层间重叠、KV 回馈的输入，集合通信表新增「离开 leaf / pod」列。扫描参数 `fabric.oversub_spine`、`fabric.pod_nodes`。
- 建模说明 §19.1–19.5；测试 `tests/test_core_060.py`（13 项手算 / 性质）。

### 修正
- PD 排队：λ 极大时 1 − e^{−λ·lat} 舍入为 1 导致除零（大卡数的稳定速率二分会触发），现判为不稳定。

### 运行时间（box 单进程，测试套件同时在跑；1P + HBM3e，8 卡 / 节点，fabric 开）

| 操作 | DeepSeek-V3 256 卡 | DeepSeek-V3 1024 卡 | Kimi-K2 1024 卡 | DeepSeek-V3 4096 卡 |
|---|---|---|---|---|
| 单点评估（含 fabric 报告） | 0.02 s | 0.01 s | 0.01 s | 0.01 s |
| 评估 + 最佳 batch | 0.02 s | 0.03 s | 0.03 s | 0.05 s |
| 布局排名 top-16（271 / 451 / 451 / ~700 个布局） | 3.8 s | 8.6 s | 7.0 s | 13.6 s |
| 映射 × 布局对比 | 14.9 s | 29.2 s | 27.3 s | 46.8 s |
| 扫描（4 点） | < 0.01 s | < 0.01 s | < 0.01 s | < 0.01 s |
| 排名稳定性 | 15.0 s | 33.5 s | 31.5 s | — |
| PD + 排队 + 布局搜索 | 23.8 s | 21.0 s | 22.1 s | — |

### 数值例子（1P + HBM3e，8 卡 / 节点，网络 50 GB/s / 卡，leaf 由 radix 64 推导）
- **Kimi-K2 DP1024·EP1024（128 节点）**：
  - decode b8192：关 5.74 ms，两层 r = 1 / 2 / 4 都是 6.07 ms（link 被计算掩盖）；三层 pod 32、r₁/r₂ = 2/4 → 7.53 ms（LINK 绑定，−19% tok/s/卡）。
  - prefill b1024：关 560 ms，两层 r = 2 → 1085 ms，三层 2/2 → 1693 ms、2/4 → 3384 ms。76% 的跨节点字节离开 pod。
- **DeepSeek-V3 DP256·EP256（32 节点）**：pod 32 时与两层相同；pod 16、2/4 时 prefill b256 921 → 2181 ms，decode link 1.80 → 4.26 ms（仍被计算 5.85 ms 掩盖）。
- **端口并发 + auto_overlap**：
  - TP16 跨节点 prefill（pod 16，2/4）：616.5 → 584.1 ms（allreduce ring → hier）；
  - TP32 decode：tree → hier，89.64 → 89.17 ms；
  - TP8·DP32 decode：link 2.04 → 1.80 ms。
- **KV 回馈**：DeepSeek-V3 PD 960P + 64D，网络 12.5 GB/s，每 leaf 1 节点、r = 4：u_kv 4.5%，decode TPOT 15.80 → 16.50 ms。

### 剩余缺口
- 每步时延不随交换机层数增加；
- ECMP / 拥塞控制 / incast 未建模；
- 端口模型是稳态资源模型；
- KV 回馈只回灌 decode 池；
- 4096 卡以上的布局对比 / 稳定性为数十秒（未并行化）；
- PD 排队报告在大规模下仍需约 20 s。

## [0.59.0] - 2026-10-09

硬件侧：三层互连的跨节点网络拓扑与集合通信算法（`scenario.fabric`，「假设」，默认关）。默认结果（1356 项指纹）与 0.58.0 逐字节一致，另有 985 项多节点 / D2D / PD 指纹也一致。

### 新增
- **拓扑感知集合通信**（`core/fabric.py`，`fabric.enabled`）：每次 allreduce 在 ring（扁平多通道环）、tree（NCCL 式：节点内 chain，跨节点双二叉树）、hier（逐层 reduce-scatter → 顶层 allreduce → 逐层 all-gather）之间取最快（`fabric.algo = auto`，按 带宽时间 + α），也可指定。allgather 用 ring / hier，all-to-all 用直接交换。α = 每次集合通信的启动 α + 步数 × 每步时延，每步时延默认取 NCCL tuner 的 LL 协议默认值作数量级参考：节点内 0.6 µs、网络 ring 2.7 / tree 5.0 µs，D2D 0.1 µs「假设」。
- **跨节点拓扑**（`fabric.net_topology`）：fat-tree / leaf-spine（每 leaf m 个整节点，上行收敛比 `oversub`）或 rail-optimized（每 rail 交换机 m 个节点，跨 rail 流量走 PXN，计入节点内 scale-up 字节）。m 可指定（`leaf_nodes`），或由交换机端口数推出（`switch_radix` 64，下行端口 = radix·r/(1+r)）。离开 leaf 的流量份额 f 以 max(1, r·f) 变慢：
  - ring f = 1/m；
  - tree 按 NCCL `ncclGetBtree` 的跨 leaf 边计；
  - all-to-all f = (n − m)/(n − 1)；
  - PP 交接跨 leaf 为 r；
  - PD KV 为 r·(1 − m/n)。
- **节点内 scale-up 拓扑**：`link.topology` 原有字段（switch / ring）此前不参与计算，现在扩展为 switch / full_mesh / ring / torus2d（`fabric.torus_x`），只在 fabric 开时生效：
  - full_mesh：(n−1)/(k−1)；
  - ring：未绕满时为链 2×，× stride；
  - torus2d：按可用端口数；
  - all-to-all：按最短路径 / 维序路由的最大链路负载计（ring 8 卡 16/7，torus 4×4 为 32/15）。
- **网内归约**（`fabric.innet_reduce`，厂商选项「假设」）：net（SHARP / CollNet 类）、net+link（另含 NVLS 类，需交换式 scale-up），用于 allreduce 的 hier 顶层，每 rank 只收发一次。
- **争用**（「假设」）：并发组共享收敛上行（上述 f）。PD KV 传输与两池自身的集合通信共用 NIC，KV 可用带宽 × (1 − u_c)（`fabric.contention`），并报告集合通信被 KV 拖慢的倍数（不回灌）。PD 报告的 `kv.fabric` 给出 leaf 因子、u_coll、原始 / 有效 GB/s。
- 适用于 TP / EP / SP（DAP）/ DP / FSDP 集合通信、PP 交接、视频 VAE 并行与文本编码器分片的 all-gather，以及 PD KV 传输。
- **报告**：API `fabric`（每种集合通信的分层、所选算法、带宽 / 时延、其他候选，以及拓扑模型关时的步时间）；Web「拓扑感知集合通信」开关、各项输入和「集合通信」表；CLI `--fabric --oversub --leaf-nodes --net-topology --link-topology --fabric-algo --innet-reduce --switch-radix --torus-x`；扫描参数 `fabric.oversub`。
- `link.topology` 现在校验取值（非法值返回 400）。建模说明 §19；测试 `tests/test_core_059.py`（9 项）。

### 数值例子（fabric 开；1P + HBM3e，8 卡 / 节点，网络 50 GB/s / 卡）
- **DeepSeek-V3 DP64·EP64，8 节点，prefill b64**（关：707.7 ms，MAC 绑定）：
  - 每 leaf 2 节点：1:1 / 2:1 / 4:1 → 708.0 / 818.5 / 1636.1 ms（2:1 起 LINK 绑定）；
  - 每 leaf 1 节点：708.0 / 954.7 / 1908.6 ms；
  - 默认推导的 leaf（4 / 5 / 6 节点）：708.0 / 708.0 / 708.0 ms（link 476.9 → 545.1 ms）。
  - decode b512 一直是 7.73 ms（link 0.93 → 最多 3.73 ms，被计算掩盖）。
- **qwen3-32b TP16，2 节点，decode b64**：关 20.84 ms；ring / tree / hier = 22.92 / 22.43 / 21.85 ms，auto 选 hier。
- **单节点 TP2·PP4 prefill 的 link 时间**：switch 6.7 / ring 13.4 / torus 4×2 26.8 / full mesh 47.0 ms。
- **TP16 跨 2 节点 prefill，每 leaf 1 节点、4:1**：link 617 → 295 ms（开网内归约）。
- **PD KV**（qwen3-30b-a3b，4 节点 × 16 卡，每 leaf 1 节点）：2:1 / 4:1 → KV 传输 4.0 → 6.0 / 12.1 ms。

## [0.58.0] - 2026-10-09

修正 0.57 合并模式 KV 准入的 CV1 过保守，新增 radix / 部分前缀匹配（前缀树，「假设」，默认关）。默认（合并、非 PD）结果（1356 项指纹）与 0.57.0 逐字节一致；KV 策略和前缀树默认都关。

### 修正（只在 KV 策略开且容量受限时生效）
- **合并模式先准入后 prefill 的占槽时间 = prefill 服务时间**（`kv_cap.slot_hold_prefill_ms`）。0.57 取 T_pre = 未计等待的 TTFT 均值，里面包含 prefill 排队。DES 里准入发生在 prefill 开始的那个迭代边界，排队在准入之前，所以排队时间不占槽。现在 T_pre = TTFT 均值 − prefill 排队均值。CV1 下排队占 TTFT 的大头，0.57 的 Erlang C 放大因此过强。
- **合并模式的条件等待形状**（「假设」）：Gamma，c² = 1 + 0.7·min(2, c_s²)（`kv_cap.wait_c2`）。PD 仍为指数分布（c² = 1）。DES 实测合并条件等待 c²：prefill 优先 1.1 / 1.56（CV0 / CV1），分块 1.6 / 1.74；PD 0.6。0.57 试过固定 c² 1.8 更差，是因为当时的均值偏高，叠加后过保守。用正则化不完全 gamma `_gammq` 打表（512 点）。
- **分块模式的饱和抢占 × (1 − ρ_pre)**（`kv_cap.refill_tight_share`）：分块时 prefill 与 decode 同批，回填受 prefill 进度限制，只有 prefill 空闲的那部分时间才会被一次填满。分块抢占次数 +30 … +60 % → −10 … +4 %（4 seed × 8000；30000 请求长跑 CV1 −20 %）。

### 新增
- **radix / 部分前缀匹配**（`pd.prefix_tree`，「假设」，默认关）：前缀树按层给出 (tokens, 分支数, Zipf α)，例如 系统提示 512 tok × 4 → 文档 1536 tok × 200 → 对话 1024 tok × 100。每个请求在每层独立按 Zipf 选子节点。缓存是按 token 计容量的 LRU，命中时刷新从叶到根的整条路径，所以子节点驻留蕴含父节点驻留，各层共享同一份容量。闭式按节点用 Che 近似：h = 1 − e^{−pT}，T 由 Σ L_k · count · h = C_tokens 解出。匹配深度分布 P(深度 ≥ k) = Σ p·h，部分命中省掉已匹配的 token（cum_k，上限 S − 1）。prefill 的每个匹配深度单独作为一类请求进入排队混合；PD decode 池按深度持有率 q_dec 传输剩余 KV。与 `prefix_len` / `prefix_hit` 互斥；最多 6 层，叶路径 ≤ 1e9，超过 4096 组时按几何 p 分箱（比值 1.01）。
- DES：`pdsim.RadixLRU`（token 容量，刷新叶 → 根，按 LRU 淘汰），路径抽样只在开启前缀树时进行，所以其他场景的随机数流不变。`compare()` 输出 `prefix_level_hit`。
- 报告 `prefix_cache.tree`，每池 `token_hit` / `level_hit` / `depth_law` / `capacity_tokens`。CLI `--pd-prefix-tree 512:4:1,1536:200:1,1024:100:0`。Web 「前缀树」输入框和缓存表（各层命中 P(深度 ≥ k)、token 命中率、容量 tok）。
- 建模说明 §18.8；测试 `tests/test_core_058.py`。

### 误差
- **KV 策略**（dense8b 12 GB，before_prefill，L0.85，(闭式 − DES)/DES；4 seed × 8000 请求，0.57 → 0.58）：
  - `wait`：合并 CV1 TTFT p90 +23 / +40 % → +14 / +21 %（prefill 优先 / 分块），准入等待 W +4 / +9 %。CV0 p90 −22 / −6 % → −23 / −11 %。CV1 p99 −4 / +4 % → +12 / +16 %。
  - recompute / swap 合并 CV1 TTFT p90：−5 / +26 / +28 / +48 % → −13 / +2 / +15 / +24 %（prefill 优先 rec / 分块 rec / prefill 优先 swap / 分块 swap）。
  - 分块抢占次数 +30 … +60 % → −10 … +4 %。合并 CV1 TTFT p50 仍偏高 +60 … +92 %（中位数落在不等待的那部分，P(等待) 闭式低约 15 %，条件均值高约 15 %）。
- **长跑 DES**（4 seed × 30000 请求，CV1，L0.85）：8000 请求的 DES 从空系统起步，CV1 下低估稳态等待。以长跑为准，0.58 的合并 CV1 TTFT p90 误差为 wait +11 / +17 %，recompute −18 / −1 %，准入等待 +4 / +8 % 和 −23 / −9 %。p99 偏高 +5 … +47 %。分块抢占次数 −20 %。
- **PD recompute CV1**（长跑）：TTFT p90 −45 %，准入等待 −52 %，抢占次数 −30 %（8000 请求时 −21 / −26 / −18 %）。没找到有原则的修正。试过用碎片化后的有效 B 建模，B = 16 时 W 4380 vs DES 6777，不够，未采用。PD wait 长跑 p90 −16 %、p99 +2 %；swap p90 −9 %、p99 +11 %。
- **前缀树**（dense8b，树 512×4 → 1536×200 → 1024×100，3 seed × 3000，L0.6 / 0.85 × CV0 / 1 × 5 / 20 / 80 GB）：token 命中率闭式 vs DES 偏低 0.003 … 0.004（0.235 / 0.360 / 0.502），各层命中差 ≤ 0.016（5 GB 根层 0.945 vs 0.96），只看缓存时最大差 0.004。延迟误差：TTFT p90 −5 … +9 %，p99 −8 … +20 %（PD CV1），TPOT p90 −9 … +9 %；例外是 PD L0.85 80 GB：TPOT 均值 / p90 +12 / +18 %，最长间隔 +35 %（未查明）。

### 数值变化（§18.1 例子，load 0.8，KV 12 GB；KV 关时不变）
- `wait`：prefill 优先 TTFT 83 / 167 / 2091 → 83 / 167 / 2073 ms；分块 137 / 386 / 2377 → 135 / 362 / 2318。PD 不变（121 / 1138 / 3869）。
- `recompute`：prefill 优先 p99 1794 → 1776，分块 p99 2074 → 2013。`swap`：1785 → 1767，2063 → 2006。PD、TPOT、SLO goodput 不变（413 / 410 / 415 / 417）。

## [0.57.0] - 2026-10-09

KV 容量策略的准入等待与抢占模型重做，并修正 DES 里一个导致「连锁抢占」的记账错误；V4 网格高负载点用更长的 DES 重跑。默认（合并、非 PD）结果（1356 项指纹）与 0.56.0 逐字节一致；KV 策略默认仍关，所以 PD 报告只有开了 `pd.kv_policy` 且容量受限时数值才变。

### 修正
- **DES：恢复中的序列不占 KV / 槽位**（`pdsim._Replica`）。recompute / swap 的被抢占者开始重算 / 换入后，它的 KV 和槽位没有计入占用，所以 PD decode 副本在恢复期间仍会接收新的 KV 准入（PD 的准入在 prefill 完成时异步发生，不只在迭代边界）。恢复结束时批次溢出，刚恢复的请求（排在最后，最年轻）又被抢占：PD recompute CV1 L0.85 有 27 % 的抢占落在刚恢复的请求上，DES 准入等待 16.5 s、TTFT p90 49 s。vLLM 在调度重算时就分配 block，所以这是 DES 的错误，不是真实的连锁效应。现在恢复中的序列计入 `_kv_used()` 和槽位数（`restoring`），重复抢占降到 11 %，同一点的 DES 准入等待 4.4 s、TTFT p90 14.3 s、抢占 0.119 → 0.076 次/请求。合并模式只在迭代边界准入，基本不受影响（重复抢占 8 %）。0.56 报告里「recompute 连锁抢占，闭式 7.9 s vs DES 17.6 s」主要来自这个错误。
- CLI：PD 模式不稳定时，KV 行的表头误写「admission after_prefill」（取了第一个模式的 kv_cap，不稳定的模式没有 admit 字段）。

### 改进（只在 KV 策略开且容量受限时生效）
- **准入等待 = M/G/c 槽位队列**（「假设」）。均值：c_s² ≤ 1 时 W = c_s²·W_M/M/c + (1 − c_s²)·W_M/D/c，W_M/D/c 用 Cosmetatos 修正 ½·W_M/M/c / (1 + (1 − ρ)(c − 1)(√(4 + 5c) − 2)/(16ρc))；c_s² > 1 仍用 Lee–Longton。0.56 的 Lee–Longton 因子 (1 + c_s²)/2 不随槽位数 c 变化；c = 17、ρ ≈ 0.75 时 Cosmetatos 修正使定长输出的等待再低约 13 %（槽越多，近似等长的寿命越快排空队列）。条件等待仍取指数分布：DES 的条件等待 c² 在 PD 约 0.6、合并 1.6–2，试过按 c² 取 Gamma，两边都更差，未采用。
- **合并模式先准入后 prefill 时，prefill 期间也占槽**：槽位负载 a′ = a + λ_r·T_pre（T_pre = 该模式未计等待的 TTFT 均值），P(等待) 与 W 按 Erlang C 比例放大；a′ ≥ B 判不稳定（「KV 槽位不足：先准入后 prefill 时，prefill 期间也占着 KV 槽」）。`kv_cap.slot_hold_prefill_ms`。
- **饱和抢占率加「回填偏移」**：离开的请求释放的是 S + o_d，回填只用掉约 S，所以余量按 h → h + o_d − G 游走，而不是每次重新从 U(0, S) 抽。P(G − o_d > h) = 0.56 的项 × E_o[e^{−o/E[o]}]（G ~ Exp(E[o])，无记忆）：定长输出 e^{−1} ≈ 0.37，CV 1 约 0.47。`kv_cap.refill_offset_factor`。合并模式抢占次数偏高 +78 … +304 % → −23 … +60 %。
- **恢复占用的不动点**：恢复（重算 / 换入换出）占副本时间的比例 f，使每个槽的占用时间 × 1/(1 − f)，排队更多，饱和抢占更多，f 更大。在把服务率除以 1/(1 − f) 的槽位链上迭代 f ← ν(f)·T_stall（取最小不动点，发散 → 不稳定）。`_decode_birth_death` 返回 `pi_at(slow)`，可按减速因子重算稳态分布。`kv_cap.slot_slowdown`。
- Web 排队表注释、CLI 的 KV 行显示恢复占比 / 槽位占用倍数，以及合并模式的 prefill 期间占槽时间。

### 新增
- `v4_grid(n_high=, high_load=0.85)` / `scripts/v4_serving.py --n-high N --high-load L`：高负载点单独设 DES 请求数（SLO 二分仍用 n // 2），长任务先调度。
- 建模说明 §18.7；测试 `tests/test_core_057.py`。

### 误差
- **V4 网格**（30 点 × 3 seed；load 0.85 的点 20000 请求，其余 3000；(闭式 − DES)/DES）：PD TTFT p99 最差 +30 % → +7 %，落在容差内 97 → 100 %；PD TTFT p50 最差 −18 % → −13 %，落在容差内 93 → 100 %；PD / 分块最长间隔最差 +24 / +22 % → +11 / +11 %。load 0.85 点的 DES 噪声中位：PD TTFT p99 22 → 9 %，p90 14 → 4 %，TPOT 均值 7 → 3 %。长跑后显出来的真实偏差：合并 TPOT p90 最差 −11 → −15 %（prefill 优先）/ −12 %（分块），都在 tp4_32b L0.85 CV1；分块 TTFT p99 最差 −15 → −23 %（moe30b L0.85 CV0，该点噪声仍有 26 %）。SLO goodput 不变（二分的 DES 长度没变）。闭式没变（KV 关），所以这一节的变化全部来自 DES。
- **KV 策略**（dense8b 12 GB → B_kv 17，before_prefill，load 0.85，4 seed × 8000 请求，修正后的 DES；0.56 → 0.57）：
  - 抢占次数：+39 … +190 % → −23 … +60 %（PD swap −4 / −8 %，PD recompute −16 / −18 %，prefill 优先 −23 … +6 %，分块 +40 … +60 %）。
  - PD recompute CV1：TTFT p90 −40 % → −21 %，p99 −26 % → −5 %。合并 recompute CV1 TTFT p90 −32 / −26 % → −5 / +26 %。
  - `wait`：PD TTFT p90 +6 / +1 % → −6 / +1 %（CV0 / CV1），p99 +22 / +24 % → +8 / +23 %，准入等待 +12 / +1 % → −1 / 0 %；合并 CV0 p99 +12 % → −11 / −4 %，分块 CV0 p90 −21 % → −6 %。变差的：合并 CV1 TTFT p90 0 / −5 % → +23 / +40 %（prefill 期间占槽的修正在 CV1 过强）。
  - 0.56 写的「wait 折叠后 p90 +29 / +31 %、p99 最多 2×」主要来自 6 × 3000 的短跑 DES：同样的 0.56 闭式在 4 × 8000 上 PD p90 只差 +6 / +1 %。

### 数值变化（§18.1 例子，load 0.8，KV 12 GB；KV 关时不变）
- `wait`：PD TTFT 121 / 1410 / 4872 → 121 / 1138 / 3869 ms；prefill 优先 82 / 143 / 2845 → 83 / 167 / 2091；分块 133 / 317 / 2956 → 137 / 386 / 2377。TPOT、SLO goodput 不变。
- `recompute`：PD TTFT 114 / 1095 / 4374 → 114 / 906 / 3453 ms，TPOT 8.73 / 12.68 → 8.69 / 12.63 ms，最长间隔 461.9 → 12.8 ms（每请求抢占降到 1 % 以下，不再进入 p99）；prefill 优先 TTFT p99 2493 → 1794 ms。
- `swap`：PD TTFT → 114 / 871 / 3376 ms，最长间隔 389.2 → 12.8 ms。
- SLO goodput 不变（413 / 410 / 415 / 417）。

## [0.56.0] - 2026-10-09

API 输入校验收紧（400 而不是 500），KV 容量策略按 vLLM 的准入顺序计入 TTFT / SLO goodput，抢占进入 TPOT / 最长间隔尾部，新增 swap 抢占，长度档 8 → 15，PD 能耗可按池给静态功耗。默认（合并、非 PD）结果（1356 项指纹）与 0.55.0 逐字节一致；PD 报告里只有长度有 CV 时数值变化（见下），KV 策略默认仍关。

### 修正
- **`/api/eval` 等接口的 500**（AGENTS §25）：scenario 按 dataclass 类型注解严格校验，类型不符返回 400，并写明字段，例如 `Serving.batch: expected integer, got number 2.5`。整数字段接受整值浮点数（|x| ≤ 2⁵³，1e300 不再 OverflowError）。`scenario.chip` / `pd.prefill_chip` 可以直接写预置名（"1P"），未知名返回 400 并列出预置。`chip_preset`、sweep `path` 必须是字符串。模糊测试（每个字段换成错误类型 / 极值，7 个接口）全部 400。
- **PD 场景进布局搜索 / 稳定性 / fit 时的 500**：`search_layouts`、`ranking_stability`、`api_fit` 改布局时先回到合并部署（`Scenario.colocated()`），不再因 `pd.decode_cards` 与新布局不整除抛 ValueError。
- **长度档 8 → 15**（`lengths.BINS`）：8 个等概率档时 P(S ≥ s₅) 正好 = 0.5，TTFT p50 落在分布的跳变上，0.03 的概率差就让 p50 变几十 %。15 档时 0.5·N、0.9·N、0.99·N 都不是整数，三个分位数都不在档边界上。闭式和 DES 用同一套档，所以对拍仍然自洽。

- **M/G/1 等待分布改为精确的 Pollaczek–Khinchine 律**（`queueing._pk_surv`，用于 PD / prefill 优先的 TTFT 卷积、分块的准静态混合，以及报告里的 prefill 等待分位数）。等待 = 几何（ρ）个均衡剩余服务时间之和；离散服务律下剩余密度是阶梯函数，所以更新方程 g(x) = λ·Σ wᵢ[G(x) − G(x − τᵢ)] 可以在 800 点网格上推进：等待原子（1 − ρ）造成的台阶精确积分，其余部分用梯形格式；4·τ_max 以外接 Cramér–Lundberg 指数尾。均值与 P-K 公式差 < 1e-6（相对），M/D/1 与 Erlang 精确式一致，生存函数与 300 万样本的 Lindley 仿真差 ≤ 0.002。0.54 / 0.55 的 min(ρ, c·e^{−θx}) 只有尾部准确：服务时间跨 100×（CV 1）时，它给短等待（排在短请求后面）的概率太少，15 档后 PD TTFT p50 偏高 15–30 %。单一服务时间（定长）仍走精确 M/D/1，数值不变。
- `_mg1_busy` 改用平铺数组（结果相同，只是求和顺序不同），抵消 15 档带来的开销：CV 1 的 PD 报告 10.3 → 6.1 s（0.55：3.1 s）。

### 新增（默认关，或只在 KV 策略开时生效）
- **KV 准入顺序** `pd.kv_admit`：`before_prefill`（vLLM，默认）先占 KV 槽与显存再 prefill（PD：再拉取 KV），容量受限时准入等待计入 TTFT 与 SLO goodput。TTFT 分布由其 p50 / p90 / p99 重建，再卷积等待（原子 1 − p，Exp(W/p)）。`after_prefill` 保留 0.55 的顺序：等待单列，不计入。`kv_cap` 新增 `admit`、`in_ttft`、`p_wait`。DES 两种顺序都实现了：合并模式在 prefill 前准入，PD 在 KV 传输前准入，decode 侧保留到传输完成。
- **抢占进入尾部**（recompute / swap）：抢占率 = Rice 上穿 + 满副本准入时的饱和项 ν_sat = p·λ_r·E[(E[o]/S)(1 − e^{−S/E[o]})]「假设」。0.55 只有 Rice，比 DES 低约 10×，现在在 2× 以内。被抢占者的间隔 = 恢复时间 + Exp(1/λ_r)，按每请求抢占次数 p_v 混合进 TPOT 均值 / p90 / p99 与最长间隔。恢复占副本时间的比例 ≥ 1 时判不稳定。`kv_cap` 新增 `preempt_rice`、`preempt_sat`、`victim_gap_mean_ms`、`restore_share`。
- **swap 抢占** `pd.kv_policy = "swap"`，`pd.swap_GBps`（主机链路 GB/s / 卡）。不给时取 `workload.host_GBps` = 50 GB/s「假设」（PCIe 5.0 x16 有效带宽，0.46 起的已有假设）。换出 + 换入 = 2 ×（S̄ + ḡ）× KV 字节/token ÷（GB/s × 每副本卡数）。闭式和 DES 都实现了。`kv_cap` 新增 `swap_ms`、`swap_GBps_card`、`swap_source`。
- **PD 能耗按池给静态功耗**：`energy.idle_W_prefill`（PD prefill 池芯片每卡 W，不给 = `idle_W`，无默认值），用于异构 prefill 芯片。PD 模式的能耗新增 `tok_per_J`、`static_share`。只给 `idle_W_prefill` 时会写明另一池的静态能耗未计入。单芯片能耗报告不使用这个字段（也不把它列为缺项）。
- CLI：`--pd-kv-policy swap`、`--pd-kv-admit`、`--pd-swap-GBps`、`--idle-W-prefill`；PD 排队输出加能耗行（J/token、tok/J、静态占比）。Web：「KV 策略」加 swap，新增「KV 准入」「换出 GB/s / 卡」「静态 W / 卡 · PD prefill」；排队表的能耗列显示 tok/J。
- 建模说明 §18.6；测试 `tests/test_core_056.py`。

### 误差（V4 网格 30 点 × 3 seed × 3000 请求，(闭式 − DES)/DES；SLO goodput 符号相反，> 0 仍为保守）
- **TTFT p50**：prefill 优先最差 +47 % → −12 %，分块 +22 % → ±4 %，落在容差内的比例 87 % → 100 %（两种合并模式）。PD 最差 +18 % → −18 %（tp4 L0.85 CV1，该点 DES seed 噪声 28 %），中位 +1 % → 0 %。
- **TTFT p99**：prefill 优先最差 +19 % → +5 %，分块 +14 % → −15 %（dense8b L0.85 CV0.5 前缀）；PD 最差 +30 % 不变（moe30b L0.85 CV1，见下面「剩余缺口」）。
- **SLO goodput**：CV 1 的 12 个点 0.55 时闭式和 DES 都是 0（TTFT p90 落在最长档上，超过 SLO），现在为正值，所以统计点数 18 → 30。新加入的点上，零负载 TTFT p90 恰好略低于 SLO，速率对 TTFT 很敏感：PD 最差 +16 %（moe30b / tp4 CV1：闭式 52 / 19，DES 62 / 22 tok/s/卡，绝对值都很小），合并 −11 … +5 %。
- TPOT 均值 / p90、最长间隔与 0.55 基本相同（TPOT p90 中位 PD 0 %，合并 −4 / −5 %，最差 −11 %）。
- **KV 策略**（不在网格里，dense8b 12 GB → B_kv 17，6 seed × 3000）：load 0.6 时都接近。load 0.85：
  - `wait`：PD 折叠后 TTFT p90 偏保守（CV0 +29 %，CV1 +31 %，p99 最多 2×），准入等待 +34 … +44 %，TPOT ±1 %。
  - 抢占次数（recompute / swap）：0.55 低估约 10×，现在在 2× 以内（PD CV1 recompute 0.105 vs 0.100 次/请求）。
  - `recompute` CV1 L0.85：DES 出现连锁抢占（TTFT p90 17.6 s），闭式 7.9 s，明显偏乐观。

### 数值变化（§18.1 例子，load 0.8；0.55 → 0.56）
- 定长：不变。前缀（N = 20000，命中 / 未命中两类服务时间）：PD TTFT p90 129 → 128 ms；分块 TTFT 85 / 150 / 241 → 83 / 146 / 242 ms；其余不变。
- CV 0.5（15 档 + 精确 P-K）：
  - PD：TTFT 144 / 409 / 778 → 141 / 439 / 848 ms。
  - prefill 优先：TTFT 77 / 237 / 336 → 77 / 214 / 359 ms，TPOT 8.65 / 14.11 → 8.70 / 14.22 ms，最长间隔 666 → 722 ms，SLO goodput 399 → 398。
  - 分块：TTFT 145 / 358 / 624 → 138 / 370 / 674 ms，TPOT 8.63 / 14.14 → 8.68 / 14.24 ms，最长间隔 36.2 → 37.7 ms，SLO goodput 399 → 398。
- KV 12 GB（`kv_policy` 非默认）：
  - `wait`（默认 before_prefill）：PD TTFT 94 / 216 / 380 → 121 / 1410 / 4872 ms（准入等待 354 ms，P(等待) 24 %，现在计入 TTFT）；SLO goodput 413 不变（在 SLO 到达率处 TTFT 仍满足，TPOT 才是约束）。`after_prefill` 与 0.55 逐位相同。
  - `recompute`：PD TTFT → 114 / 1095 / 4374 ms，最长间隔 12.8 → 462 ms（被抢占者：重算 86 ms + 回队等待，每请求 0.025 次），TPOT 8.67 / 12.59 → 8.73 / 12.68 ms，SLO goodput 410 不变。
  - `swap`（新，50 GB/s/卡「假设」）：PD 最长间隔 389 ms（换出 + 换入 2 × 6.4 ms），TPOT 8.69 / 12.63 ms。

### 剩余缺口
- PD TTFT p99 在 CV1 L0.85 时 +27 … +30 %（tp4 / moe30b）。这不是闭式的偏差，是 DES 运行太短：n = 30000 时 tp4 DES 9161 ± 485 vs 闭式 9197（+0.4 %），moe30b 5939 ± 304 vs 6178（+4 %）；n = 3000 时，长尾在 τ_int 窗口里取样不足。网格仍按 3000 跑（30000 要多约 10× 时间）。
- KV：`wait` 折叠后 TTFT 偏保守；`recompute` 在高负载、CV1 下的连锁抢占没有建模；合并模式的 swap / recompute 抢占次数偏高（2–3×）。
- 档内展宽（连续长度）没做，用 15 档 + 精确 M/G/1 代替；CV 1 的 PD 报告 3.1 → 6.1 s。

## [0.55.0] - 2026-10-09

收敛 0.54 V4 找出的合并模式尾部误差，加 decode KV 容量策略（可选），把 V4 扩到 MoE 和多卡 TP。默认（合并、非 PD）结果（1356 项指纹）与 0.54.0 逐字节一致；PD 报告里的排队数值有变化，见下。

### 新增
- **KV 容量排队 / 抢占**（默认关）：`pd.kv_policy` = `wait`（为 prompt + 全部输出预留 KV，准入等待）/ `recompute`（只按当前上下文准入，超出时抢占最年轻的请求并重新 prefill），`pd.kv_capacity_GB`（不给则取剩余 DRAM）。只有 KV 槽位 < batch 时才生效。闭式（批链换槽位数、Little + Lee–Longton 准入等待、Rice 抢占率 + 重算停顿）与 DES 都实现了。每个模式输出 `kv_cap`；CLI `--pd-kv-policy --pd-kv-capacity-GB`；Web「KV 策略」「KV 容量 GB」。准入等待单独报告，不计入 TTFT / TPOT / SLO goodput。
- **V4 场景族**：`moe30b`（Qwen3-30B-A3B，TP2 · EP2）、`tp4_32b`（Qwen3-32B，每副本 TP4 跨 4 卡），各 6 点，与 dense8b（18 点）合计 30 点。`v4_scenario(..., family=)`、`V4_FAMILIES`。
- **多 seed 验证**：`pdsim.compare(seeds=…, slo_tol=…)` 对 DES 指标和 SLO 速率取平均，并报告 `noise`（seed 间标准差 / 均值）。网格默认 3 seed × 3000 请求，`scripts/v4_serving.py --seeds --jobs --families`（多进程）。摘要按场景族和全部给出，附 `noise_median`；`V4_BANDS` 加 TPOT 均值 ±10 %。
- `queueing.md1_cdf`（Erlang 精确 M/D/1 等待 CDF）、`queueing.mg1_mix_sum_quantile`（准静态混合 M/G/1 + 休假）；建模说明 §18.5；测试 `tests/test_core_055.py`。

### 修正（只影响 PD 报告，含其中的合并对照）
- **合并 prefill 优先**：prefill 忙期结束时一批请求同时进入 decode。birth–death 换成批到达链，并按配对系数 κ = 2∫P(O > t)²dt / E[O] 整形批分布，修正同批请求生命周期的相关（运行 batch 方差：链 6.99，DES 7.08；0.54 为 6.30）。
- **合并分块**：用更新—报酬过程算忙期内的有效成批程度（decode 在忙期内照常推进）；最后一个不满的块按实际比例计（0.54 按整块）。
- **TPOT p90 窗口因子**：常数 √(2/e) 改为按占用积分自相关时间 τ_int 的 OU 平均因子。
- **分块 TTFT 尾**：按运行 batch 环境做准静态混合 M/G/1，加纯 decode 迭代的休假。
- **最长间隔**（PD、分块）：从「看到的 batch 的 0.99 点」改为一生中最大 batch 的分位数（上穿率）。
- **M/D/1 等待分位数**：6 个服务时间以内用精确 CDF（原渐近式在 ρ ≈ 1 − q 附近偏保守 20–30 %）。
- 数值变化（§18.1 例子，load 0.8）：
  - PD：TTFT p50 102 → 94 ms，TPOT p90 14.06 → 14.70 ms，最长间隔 22.8 → 26.5 ms，SLO goodput 410 → 403 tok/s/卡（最佳切分仍为 2 + 6）。
  - prefill 优先：TPOT 7.45 / 11.14 → 8.22 / 13.35 ms，SLO goodput 442 → 411。
  - 分块：TTFT p99 330 → 377 ms，TPOT 7.43 / 11.09 → 8.20 / 13.38 ms，最长间隔 23.0 → 30.5 ms，SLO goodput 443 → 411。
  - CV 0.5 时 prefill 优先最长间隔 528 → 666 ms。

### 误差（30 点 × 3 seed，(闭式 − DES)/DES）
- TPOT p90 中位：PD 0 %，prefill 优先 −4 %，分块 −4 %（0.54：−5 / −18 / −14 %）；最差 −11 %（0.54：−43 %）。TPOT 均值全部在 ±6 % 内。
- SLO goodput：−5 % … +3 %。
- 最长间隔中位 −2 … +5 %，TTFT p90 全部在 ±10 % 内，前缀命中率绝对差 < 0.01。
- 没做：PD prefill 池内分块（FCFS 下没有 TTFT 收益，评估器也缺独立的块代价）、swap 抢占。剩余缺口（CV 下 TTFT p50 落在长度分档边界、CV 1 高负载 PD TTFT p99 +30 %、recompute 抢占次数低估、准入等待不进 SLO）见 MODEL.md §18.5。

## [0.54.0] - 2026-10-09

验证版本，不加新功能：新增请求级离散事件仿真（DES），用它量化 PD 闭式排队模型的误差（V4 服务验证），并修正 DES 揭示的几处闭式问题。默认（合并、非 PD）结果（1356 项指纹）与 0.53.0 逐字节一致。PD 报告里的排队数值有变化，见下。

### 新增
- `core/pdsim.py`：请求级 DES（「假设」）。复用闭式的逐步代价（prefill TTFT(b, S, p)、decode step(k)、分块融合迭代、KV 传输与争用缩放）；模拟泊松到达、FCFS、PD 两池（prefill 静态 batch 上限、每个 prefill 副本一条 KV 流、decode 连续批处理）、合并 prefill 优先与分块 prefill、离散长度分布、整前缀精确 LRU + Zipf（缓存预热，只在测量窗内计命中率）。提供 `simulate`、`slo_rate`（同一组随机数二分）、`compare`。
- V4 服务验证：`validation.v4_serving()`（实时子集，`accel-dse validate` 打印，约 15 s，`--no-v4` 跳过）、`validation.v4_grid()` 与 `scripts/v4_serving.py`（54 点网格：load 0.3 / 0.6 / 0.85 × CV 0 / 0.5 / 1 × 前缀 关 / 开 × PD / prefill 优先 / 分块）。结果存 `accel_dse/data/v4_serving.json`。原 V4（硅片，n/a）改名为 V5。
- 可选 `pd.simulate` / CLI `--pd-sim` / Web「DES 仿真尾部」（默认关）：PD 报告附 `pd.queue.sim.modes.{mode}`，包括 DES 的 TTFT / TPOT / 最长间隔分位数、命中率和相对闭式的误差；闭式数值不变。
- `queueing.mg1_sum_quantile`；建模说明 §18.4（误差表、偏乐观 / 偏保守的地方、数值变化）；测试 `tests/test_core_054.py`。

### 修正（只影响 PD 报告，含其中的合并对照）
- **decode 连续批处理**：Little 不动点 n̄ = λ·out·TPOT(⌈n̄⌉) 忽略了占用涨落，step(k) 随 k 变陡时偏乐观（§18.1 例子 load 0.8：平均 TPOT 6.49 ms，DES 9.05 ms）。改为 birth–death 处理器共享模型（9.21 ms）；TPOT 分位数与最长间隔按运行中的请求看到的 batch 分布计算。
- **prefill 优先 TPOT 分位数**原先重复计入停顿，改为平方和开根。
- **prefill 优先最长间隔**原先「TPOT + 一次 prefill」偏乐观（87 ms，DES 322 ms），改为一生中最长 prefill 忙期的 0.99 点（314 ms）；长度混合时按忙期时长卷积。
- **TTFT 分位数**：服务时间混合（长度 CV > 0、前缀命中 / 未命中）时，由等待分位数 + 时延分位数逐项相加（偏保守）改为 W ⊕ L_i 的正确分位数；只有一个服务时间取值时不变。
- 分块 prefill 的 decode 也走 birth–death，prefill 服务按看到的 batch 计。
- 数值变化（§18.1 例子，load 0.8）：
  - PD：TPOT 均值 / p90 6.49 / 9.00 → 9.21 / 14.06 ms，最长间隔 10.9 → 22.8 ms，SLO goodput 474 → 410 tok/s/卡（TTFT 不变）。
  - prefill 优先：TPOT 5.35 / 7.69 → 7.45 / 11.14 ms，最长间隔 83.5 → 314 ms，SLO goodput 506 → 442。
  - 分块：TTFT p90 177 → 209 ms，TPOT 5.35 / 7.39 → 7.43 / 11.09 ms，最长间隔 16.1 → 23.0 ms，SLO goodput 511 → 443。
  - CV 0.5 时 PD TTFT p90 539 → 409 ms，SLO goodput 316 → 404。

### 误差（54 点，(闭式 − DES)/DES）
- SLO goodput：三种模式均在 −8 % … 0 % 之间。
- TTFT p90：PD 全部在 ±20 % 内（中位 0）。
- TPOT p90：PD 中位 −5 %；合并模式系统性偏乐观（中位 −14 … −18 %，最差 −43 %）。
- 最长间隔：中位 −10 … −17 %。
- 前缀命中率：绝对差 < 0.01。
- 剩余缺口见 MODEL.md §18.4。

## [0.53.0] - 2026-10-09

前缀缓存容量 + LRU（Che）淘汰模型、异构 PD 池、布局搜索里的 decode batch。全部「假设」、默认关；默认结果（1356 项指纹）与 0.52.0 逐字节一致。

### Added
- `core/prefixcache.py`：Zipf 工作集 + 整前缀 LRU 的 Che 近似命中率（N ≤ 4096 精确，更大 N 对数分箱）；容量 = 剩余 DRAM（或 `pd.prefix_cache_GB`）/ 前缀足迹；随机路由 vs `pd.prefix_affinity`；PD prefill / decode / 合并各自独立命中率，decode 在 prefill 命中时也持有该前缀的条件概率进入 KV 交接。显式 `pd.prefix_hit` > 0 覆盖容量模型。
- 异构池：`pd.prefill_chip` / `pd.prefill_mem_id`（None = 与 decode 相同）；合并对照用 decode 池芯片 / 存储器。
- 布局搜索 `pd.search_decode_batch`：每种 decode 布局在 {B/2, B, 2B, 4B} 中取满足 TPOT SLO 的最高吞吐。
- CLI / Web 对应输入；响应 `pd.prefix_cache`、`pd.lengths.prefix_hit_source`、`pd.prefill.{chip,mem_id,hetero}`；建模说明 §18.3；测试 `tests/test_core_053.py`。

### Fixed
- `mdc_wait`：到达率贴着流体容量时 `c/service − λ` 舍入为 0 不再 ZeroDivisionError（标不稳定）。

### Changed
- `PDConfig` 新增 8 个字段（场景哈希随之变化）；默认值下 PD 报告与 0.52 相同。

### 不做 / 待定
- 部分前缀匹配、缓存预热 / 淘汰代价、队列 batch 对剩余 DRAM 的反馈、按芯片的闲置功率、PD prefill 池内分块、decode 端共享前缀 KV 读合并、长度感知调度；离散事件仿真对拍。

## [0.52.0] - 2026-10-09

PD 报告的三项补充：请求长度分布、前缀缓存命中率、池布局搜索。全部「假设」、默认关；默认结果（1356 项指纹）与 0.51.0 逐字节一致。

### Added
- 请求长度分布 `core/lengths.py`：`pd.prompt_cv` / `pd.out_cv`（对数正态，8 档等概率条件均值）或 `pd.length_mix`（离散联合分布 [权重, prompt, 输出]，≤ 16 行）。排队改为 M/G/1（`queueing.mg1`，离散服务分布的 P-K 均值 + Cramér–Lundberg 尾，单点即 M/D/1），每个 prompt 值单独求值；TTFT 分位用「自身 prefill + KV」的离散分位；decode 用 E[out]、长度偏置的上下文（`serving.ctx × ctx_ratio`）与 Allen–Cunneen (1 + c_s²)/2；流体容量（prefill / decode / KV 与合并对照）也按分布计算。`queueing.dquantile`、`mdc_wait(cs2=…)`。
- 前缀缓存：`serving.prefix_cached`（prefill 的 Phase(q = S − p, ctx = p)：只算新 token、读前缀 KV；吞吐仍按整个 prompt）与 `pd.prefix_hit`（每个 prompt 的命中比例），`pd.prefix_on_decode`（默认 true：KV 交接只传未缓存部分）。分块 prefill 块数与前缀重读随之变化。
- 池布局搜索 `pd.search_layouts`（opt-in）：两池每副本 1 / 2 / 4 / … / 64 卡的全部布局 × 同总卡数切分，流体 goodput 排序，部分布局对另算 SLO goodput（`layout_search`）。
- 每个模式的 `stable_rate_rps`；分块的 `chunk_share`；`pd.lengths` 摘要。
- CLI `--pd-prompt-cv --pd-out-cv --pd-mix --pd-prefix-hit --pd-prefix-not-on-decode --pd-search-layouts --prefix-cached`；Web PD 输入组「请求长度与前缀缓存」、单点页「池布局搜索」表；建模说明 §18.2；测试 `tests/test_core_052.py`。

### Fixed
- 合并 · 分块 prefill 的 decode 平均步时：0.51 用时间平均 ρ·T₁ + (1 − ρ)·T₀（偏向长迭代），改为每次迭代平均 T₀ / (1 − ρ + νT₀)，带块迭代占比 x = ν × 该值；块数用 ⌈S/C⌉。§18.1 例子里分块 TPOT 均值 9.4 → 5.4 ms、SLO goodput 433 → 511 tok/s/卡。PD 与 prefill 优先不变。

### Changed
- `Serving` 新增 `prefix_cached`，`PDConfig` 新增 6 个字段（场景哈希随之变化）；默认值下 PD 报告与 0.51 相同（分块修正除外）。

### 不做 / 待定
- 长度感知调度、缓存容量 / 淘汰 / 路由、decode 端共享前缀 KV 读合并、两池 batch 搜索、异构池、PD prefill 池内分块；离散事件仿真对拍。（0.53 已补容量 / LRU、异构池、布局搜索 decode batch）

## [0.51.0] - 2026-10-09

PD 报告从稳态流体模型往下深入一层：排队与尾延迟、连续批处理、分块 prefill、KV 与池内集合通信争用、PD 能耗。全部是解析近似「假设」，只在 `pd.enabled` 时计算，默认结果与 0.50.0 逐字节一致。

### Added
- `core/queueing.py`：M/D/1（P-K 均值、P(W>0) = ρ、Cramér–Lundberg 尾分位）、Erlang C（Allen–Cunneen ½）槽位等待、Poisson 分位；测试用 Lindley 仿真核对 p90 / p99。
- `core/pdqueue.py` → eval 响应 `pd.queue`：同一泊松到达率（`pd.load` × PD 流体容量，默认 0.8，或 `pd.rate_rps`）下比较 PD 分离、合并 · prefill 优先、合并 · 分块 prefill（`pd.chunk_tokens`，默认 512）的 TTFT p50 / p90 / p99、TPOT 均值 / p90 / p99（请求平均）、最长 token 间隔、端到端均值、SLO goodput（p90 TTFT 与 p90 TPOT 都满足的最大到达率 × out / 卡）。PD 的 prefill 池为带 batch 上限的 M/D/1（取平均 TTFT 最小的上限），decode 池连续批处理按 Little 不动点求运行 batch；prefill 优先模式计 decode 停顿；分块模式按级融合迭代（prefill 权重读与 decode 共用、前缀 KV 重读计入）。
- KV 与池内集合通信争用（没设 `pd.kv_GBps` 时）：KV 可用带宽扣除两池在该层的集合通信占用，KV 流量反过来拉长池的链路时间；`pd.kv_GBps` 视为专用通道。
- PD 的 SLO goodput 切分搜索（`pd_slo_splits`、`pd_slo_best_split`）。
- PD / 合并两种模式每输出 token 的动作计数与卡·秒（`pd.queue.energy`），给了能耗表时出 J / token。
- CLI `--pd-load --pd-rate --pd-chunk`、`--out-len`、`--ttft-slo`，`eval --pd` 输出排队表；Web PD 输入组加「负载 / 到达率 / 分块 token」，单点页加「排队与尾延迟」表；建模说明 §18.1；测试 `tests/test_core_051.py`。

### Changed
- 默认结果不变：1356 项指纹与 0.50.0 逐字节一致。`PDConfig` 新增 `load`、`rate_rps`、`chunk_tokens`（场景哈希随之变化）。`disagg_report` 新增可选 `energy`、`queue` 参数。

### 不做 / 待定
- 请求长度分布、抢占 / 换出 / KV 容量排队、前缀缓存、调度器开销、PD prefill 池内分块、异构池、池布局搜索；分位数逐项相加（偏保守），不做卷积或仿真。

## [0.50.0] - 2026-10-09

prefill / decode 分离（PD）作为可选服务模式；互连拆成三层（可选封装内 D2D、节点内 scale-up、跨节点网络）。全部默认关，默认结果与 0.49.0 逐字节一致。

### Added
- PD 分离 `core/disagg.py`（`scenario.pd`，默认关 = 合并服务）：decode 池用场景布局，prefill 池用 `pd.prefill_layout`，两池卡数 `pd.prefill_cards` / `pd.decode_cards`；prefill 池取满足 TTFT SLO 的最大 batch。请求率 = min(prefill 池、decode 池、KV 传输)，给出瓶颈与各池利用率、TTFT（prefill + 暴露的 KV 传输）、TPOT（纯 decode 步）、goodput / 卡，同总卡数的全部切分与最优切分，以及同卡数合并服务的对照（含有效 TPOT = TPOT / decode 占比）。KV 每请求 = 整模型 S token 的 KV + 索引键 + 状态；走跨节点网络（设了 `node_cards`）或节点内链路，`pd.kv_GBps` 可覆盖，`pd.kv_layerwise` 逐层流式只暴露掩盖不了的部分。Qwen3-8B 1P + HBM3E batch 32 prompt 8192：PD 2 + 6 卡 TPOT 42.2 ms（满足 50 ms SLO）569 tok/s/卡；合并 574 tok/s/卡但有效 TPOT 55.8 ms。只对 LLM / VLM。API `pd`，CLI `--pd --pd-prefill-* --pd-prefill-cards --pd-decode-cards --pd-kv-GBps --pd-layerwise`，Web 服务组「PD 分离」与单点页对照表。
- 三层互连：`d2d_enabled`（默认关 = 单片大 die）+ `package_cards` + D2D 档位 `d2d_std` / `d2d_units`；`link` 改称节点内 scale-up；新增跨节点 `net`（默认 50 GB/s、5 µs「假设」）与 `node_cards`（默认 0 = 单节点）。集合通信按组在封装 / 节点内的成员数逐级分层（all-reduce / all-gather 逐级、all-to-all 各层同时），PP 交接按边界选层；两层时与 0.48 公式逐位相同。能耗新增 `net`（`pJ_bit_net`），D2D + 节点内 + 跨节点 = 单层链路字节。每级 `link_GB` 增加 `scaleup`，`net` 改为跨节点字节。
- D2D 档位目录 `core/d2d_catalog.py`（公开数据，原始速率每方向每单元）：UCIe-A x64 @ 48 GT/s（384 GB/s，常用，D2D 开时默认 × 4 模块 = 1536 GB/s）、UCIe-A @ 64 GT/s（512）、UCIe-S x16 @ 32 GT/s（64）、BoW-256 / BoW-512（32 / 64 每 16 线 slice）、NVIDIA NVLink-C2C（450，厂商专有，参照）与自定义。48 / 64 GT/s 每 transfer 1 bit 与每 die 单元数标「假设」。API `/api/catalog` `d2d_standards`，CLI `accel-dse d2d`、`--d2d --d2d-std --d2d-units --node-cards --net-GBps --net-alpha-us --pJ-bit-net`，Web「三层互连」组（D2D 开关、档位选择、单元数、跨节点三项）；扫描新增 `d2d_units`、`node_cards`、`net.GBps`、`net.alpha_us`。
- 建模说明 §15 重写为三层互连并加 D2D 档位表（§15.1），新增 §18（PD）；测试 `tests/test_core_050.py`。

### Changed
- 默认结果不变：1356 项指纹与 0.49.0 逐字节一致。场景新增字段（`d2d_enabled`、`d2d_std`、`d2d_units`、`net`、`node_cards`、`pd`），场景哈希随之变化。
- D2D 默认带宽 2000 → 1536 GB/s（UCIe-A 48G × 4，只在 D2D 开时使用）。D2D 关时 `package_cards` 不生效并警告；0.48 / 0.49 的场景 JSON 中 `package_cards > 1` 自动视为 D2D 开，给了 `d2d.GBps` 视为自定义档位，照旧求值。
- Web「网络层」改称「节点内」；能耗表的链路项按三层拆分显示。

### 不做 / 待定
- PD：排队与到达波动、连续批处理动态与 chunked prefill、KV 与池内集合通信争用、前缀缓存、异构池、PD 能耗、池布局搜索。
- 互连：拓扑、拥塞 / 超额订阅、SHARP、NIC 的 PCIe 瓶颈、D2D flit / 协议效率（用自定义档位填有效带宽）。

## [0.49.0] - 2026-10-09

两个可选项：MoE 专家负载倾斜（工作负载旋钮）与资源 / 面积预算（设计约束，只报告余量）。默认关，默认结果与 0.48.0 逐字节一致。

### Added
- MoE 负载倾斜：`serving.moe_skew`（默认 1 = 均匀路由，[1, 64]，「假设」）或实测的每专家 token 分布 `serving.moe_expert_load`（按连续放置求每个 EP 宽度的倾斜，搜索中不同 EP 看到不同倾斜）。最忙 EP rank 决定级时间：其专家 GEMM 行数与 dispatch / combine / ETP 载荷按 skew × 平均计（上限 T·k 与每专家 T 行），命中专家数保持均匀期望。EP = 1 / 非 MoE 时忽略并警告。能耗计数取均匀路由下的每单位值（倾斜只搬工作），卡·秒按倾斜后的时间窗。100T + HBM3E，Qwen3-30B-A3B TP2·DP4·EP8：prefill 64 × 4096 TTFT 4608 → 倾斜 2 / 4：6048 / 8924 ms；decode batch 64 不变（行分块吸收），batch 2048：258.6 → 293.0 ms（倾斜 4）。summary 新增 `moe_skew`；CLI `--moe-skew --moe-expert-load`；Web「MoE 倾斜」；扫描 `serving.moe_skew`。
- 资源 / 面积预算 `core/budget.py`：请求体顶层 `budget`（不进 scenario）。限额：每卡 SRAM / SLC / MAC 单元 / 峰值 TFLOPS / DRAM 容量、每副本卡数、每卡平均功耗（用用户能耗表）、面积代理（每卡 / 每副本；SRAM、SLC、MAC 与固定面积的密度全部由用户填写，工具不内置工艺库）。报告余量与「未超 / 超出 / 无法判定」（缺密度，或功耗 / 面积只是下界且未超）。布局、映射对比、扫描每行加 `budget_ok`；超预算的布局排在后面，映射对比取预算内最优，容量检查的最少卡数受卡数预算限制。CLI `--budget-* --mm2-*`（search / compare 多一列）；Web「资源 / 面积预算」组与单点页预算表、布局表「超预算」标记。
- 建模说明 §16（MoE 倾斜）、§17（预算）；测试 `tests/test_core_049.py`。

### Changed
- 默认结果不变：1356 项指纹与 0.48.0（= 0.47.1）逐字节一致。Serving 新增字段，场景哈希随之变化。

### 不做 / 待定
- EPLB / 冗余专家、专家放置优化；工艺库、成本、面积与频率 / 功耗耦合。
- 三级互连与 SLC / D2D 默认数值仍待用户答复（0.48 的两个问题），本版未改动。

## [0.48.0] - 2026-10-09

硬件侧：系统级缓存（SLC）作为 SRAM 与 DRAM 之间的设计变量；两级互连（封装内 die-to-die 与跨封装网络）。两者默认关，默认结果与 0.47.1 逐字节一致。

### Added
- SLC（`chip.slc_mib`，默认 0 = 无；`slc_GBps` 默认 2000、`slc_policy` = `pin` | `lru`，全部「假设」）：命中字节 / `slc_GBps` 作为与 DRAM 并行的 `t_slc` 进入级时间，瓶颈可标 SLC；DRAM 只计未命中；不增加容量。`pin` 按 SRAM 同序钉住剩余热权重 → 专家 → KV / 状态，流式激活、FSDP gather、查表行、KV 写绕过；`lru` 按循环访问取全有或全无（SRAM 之外的工作集放得下才全部读命中）。视频 pipeline 组件不用 SLC。100T + LPDDR5X，Qwen3-8B decode batch 8：TPOT 104.2 → pin 4 / 8 / 16 GiB 81.7 / 59.2 / 20.8 ms；lru 16 GiB 放不下 → 不变，32 GiB → 20.0 ms。
- 两级互连：`link` 为跨封装 / 节点的网络层（默认不变 400 GB/s、3 µs），新增 `d2d`（同封装 die-to-die，默认 2000 GB/s、0.5 µs「假设」）与 `package_cards`（每封装卡数，默认 1 = 无 D2D 层）。卡按 TP → SP → DP → PP 编号，通信算子带步长 `comm_stride`（TP / ETP 1，SP / DAP 为 TP，EP 为 ETP，FSDP 为 TP），组在封装内的成员数 k 决定分层：all-reduce / all-gather 两级（NCCL 式）、all-to-all 按比例两路并行、PP 交接按流水级边界是否跨封装；k = 1 时与单层公式相同。VAE tile all-gather 与文本编码器分片 gather 同样分层。1P + HBM3E、网络 50 GB/s：Qwen3-8B prefill TP8 TTFT 676.8 ms（LINK）→ 每封装 4 卡 248.2 ms（MAC）。分层 all-reduce 的 α = 2α_d + α_n，小载荷 decode 部分跨封装时略慢（如实计）。
- 能耗：动作 `slc`（`pJ_bit_slc`）与 `d2d`（`pJ_bit_d2d`）；DRAM 只计 SLC 未命中，`link` 只计网络层份额（按每次集合通信两层发送字节之比拆分）。守恒：SLC + DRAM = 无 SLC 的 DRAM 字节，D2D + 网络 = 单层链路字节。
- 入口：API scenario `package_cards`、`d2d`、`chip.slc_*`，每级 `t_ms.slc`、`slc_GB`、`link_GB`（total / d2d / net）、`mem.slc_*`；扫描路径 `chip.slc_mib`、`chip.slc_GBps`、`package_cards`、`d2d.GBps`、`d2d.alpha_us`。CLI `--slc-mib --slc-GBps --slc-policy --package-cards --d2d-GBps --d2d-alpha-us --link-GBps --link-alpha-us --pJ-bit-slc --pJ-bit-d2d`。Web：芯片组 SLC 三项、并行组「每封装卡数 / D2D」、能耗组两项，级表 / 存储表 / 能耗表在启用时多出 SLC / D2D 列。
- 建模说明 §14（SLC）、§15（两级互连）；测试 `tests/test_core_048.py`（默认不变、闭式、映射到层、pin / lru、能耗拆分、校验 / API / CLI）。

### Changed
- 默认结果不变：1356 项指纹与 0.47.1 逐字节一致。场景新增字段（`d2d`、`package_cards`、芯片 `slc_*`），场景哈希随之变化。
- Web「链路 GB/s / 同步 α」改称「网络层 GB/s / 网络 α」。

### 不做 / 待定
- 第三层互连（封装内 D2D、节点内 scale-up、跨节点 scale-out）未分开——网络层一项代表封装以外的全部；拓扑、拥塞、SHARP 不建模。
- SLC 多卡共享、一致性、写回差异、bank 冲突不建模；默认 D2D / SLC 数值为占位「假设」。

## [0.47.1] - 2026-10-09

硬件侧第一小步：能耗动作计数（每动作能耗由用户提供，工具不内置数值）。

### Added
- `core/energy.py`：每输出单位（token / prompt token / 帧 / 序列）的动作计数——bf16 等效 MAC、向量操作、SRAM 端口字节、DRAM 字节、链路字节、卡·秒——全系统累计；视频含卡上的文本编码器与 VAE 解码（主机 CPU 编码器不计），VAE 多卡解码含 tile all-gather。能耗 = 计数 × 用户能耗表（`pJ_mac`、`pJ_vec`、`pJ_bit_sram`、`pJ_bit_dram`、`pJ_bit_link`、`idle_W`），空项不计并列出。API body 顶层 `energy`（不进 scenario、不改哈希），响应 `energy`；CLI `--pJ-mac … --idle-W`；Web「能耗」输入组与「动作计数与能耗」表。测试 `tests/test_core_energy.py`（计数守恒与闭式、线性、校验、API / CLI）。
- 计数器：`StageResult.sram_bytes`（SRAM 端口字节 = 供数周期 × 端口宽度），组件的 `acts`。

### Changed
- 无结果变化：1356 项指纹与 0.47.0 逐字节一致。

### 仍未建模
- 工艺推导的功耗、DVFS、SRAM 容量相关的访问能耗、DRAM 行激活 / 刷新、主机 / PUE、面积 / 成本。SLC、D2D 与网络两级互连尚未开始。

## [0.47.0] - 2026-10-09

视频 pipeline 收尾：VAE 多卡分块并行解码、LTX 分块、MiniMax-H3 解码按发布（总是分块）、跨请求重叠（主机编码器）。

### Added
- VAE 多卡分块并行解码 `workload.vae_parallel`（默认关；Web「VAE 多卡分块并行解码」，CLI `--vae-parallel`）：依据 MiniMax-H3 发布的 `vae_parallel_tiling`——每个分块调用内 rank r 解第 r、r + N、… 个 tile，再 all-gather 解码后的像素 tile；N = 副本的 PP·TP·SP 卡，算术不变，时间 = 最慢卡 + 每轮 all-gather，VAE 权重每卡一份。需分块解码（H3 / HunyuanVideo 总是分块，其余开 `vae_tiling`），否则忽略并警告。100T + HBM3E：H3 SP 2 / 4 / 8 解码 24.4 → 12.2 / 6.11 / 3.49 s；SP8 HunyuanVideo 221 → 34.0 s、Wan 分块 67.8 → 10.4 s。H3 以外为设计选项（xDiT DistVAE 只用于图像 VAE）。
- LTX-Video VAE 可选分块（diffusers `AutoencoderKLLTXVideo`：512 px tile / stride 448 → 潜空间 16 / 14）：默认 704 × 512 → 4 tile，×1.23，解码 0.72 → 0.89 s。
- 跨请求重叠 `workload.overlap`（默认关；CLI `--overlap`）：只在 `te_cpu` 时生效，主机编码下一请求与卡上去噪并行，稳态周期 = max(T_clip − T_text, T_text)，延迟不变、吞吐按周期计（结果新增 `period_s`）；组件同卡时按串行计并警告。
- 测试 `tests/test_core_047.py`。

### Changed
- MiniMax-H3 视频解码按发布更正：发布配置 `vae_decoder_tiling = 1`（tile 256 px、最小重叠 64 px；diffusers 文档写明默认分块、关掉会改变输出），时间上 5n + 2 潜帧 → n 段 × (5 + 2) 潜帧（原按 ⌈(T + 3)/5⌉ 段 × 5 潜帧、不分块）。默认 1344 × 768 124 帧：7 段 × 28 tile，解码 1741 → 1893 TFLOP、22.6 → 24.4 s，整段 1936 → 1937 s。1356 项指纹只有 minimax-h3 的 20 项变化（其余逐字节一致，LLM 不变）。
- 分块循环与 diffusers 一致：任一轴超过 tile 时两轴都按 stride 从 0 走（短轴多一条窄 tile）；默认分辨率下与 0.46 相同。
- `vae_tiling` 对没有参考分块的 VAE（Open-Sora）的警告改为「参考实现没有空间分块」。

### 不做 / 仍未建模
- Open-Sora VAE 分块：发布实现没有空间分块（只有逐帧 / 17 帧微批，已计），diffusers 未收录——不做。
- 结构模型 pair 的 TP：参考实现里没有，DAP + 样本分卡已覆盖多卡——维持不做。
- 不分块 VAE 的 patch 并行（halo 交换）；组件同卡时的跨请求重叠；AlphaFold 3 仍未接入。

## [0.46.0] - 2026-10-09

扩散样本分卡、DiT 权重 FSDP、文本编码器放主机 CPU、Wan VAE 分块。

### Added
- 结构模型扩散样本分卡 `workload.sample_split`（默认开；Web「DAP 时扩散样本分卡」，CLI `--no-sample-split` 关闭）：DAP > 1 且样本数 > 1 时，扩散模块（原子编码器 / 扩散 transformer / 原子解码器）的样本分到 DAP 各卡，每卡 ⌈S / D⌉ 条轨迹——样本独立、条件已在每卡，精确且无额外通信；块内 pair 网格工作仍按 DAP 切分。100T + HBM3E：Protenix（5 样本）DAP 8 10.65 → 3.22 s（相对单卡 1.8× → 6.0×）；Boltz-1 默认 1 个样本不变，5 个样本时 DAP 8 12.6 → 4.3 s。置信度头不分样本。参考实现无现成的多卡样本并行：作为布局设计选项给出。
- DiT 权重 FSDP `workload.dit_fsdp`（默认关；Web 复选框，CLI `--dit-fsdp`）：按 Wan `--dit_fsdp`，每级权重在 SP·DP 卡之间分片（w / g + 2 层预取），每次前向逐层 all-gather 活动权重（与计算重叠，每层 α）并把 gather 结果写入 DRAM；Wan2.2 空闲专家分片存储、不 gather；SP·DP = 1 或非视频模型时忽略并警告。例：Wan2.1-14B SP2 64 GiB LPDDR 每卡需求 57.8 GiB（auto 卸载）→ 45.1 GiB（常驻），延迟变化 < 0.1%。
- 文本编码器放主机 CPU `workload.te_cpu`（默认关；CLI `--te-cpu`）：按 Wan `--t5_cpu`，卡上不放编码器；编码时间 = 编码器 FLOPs / `workload.host_TFLOPS`（默认 2 TFLOPS「假设」，CLI `--host-TFLOPS`，请按实测填写）。例：Wan2.1-14B 单卡 64 GiB 常驻放得下（61.9 GiB，不再需要每请求重载），编码 4.8 s。
- Wan VAE 可选分块（`vae_tiling`，diffusers `AutoencoderKLWan` 的 256 px tile / stride 192；官方仓库不分块）：720P 28 tile，重叠 ×1.65，解码 41.3 → 67.8 s，激活峰值 3.81 → 0.27 GiB。
- Web：「放不下」面板增加「DiT 权重 FSDP」与「文本编码器放主机 CPU」修正；假设列表写明 FSDP / 主机 CPU / 样本分卡。测试 `tests/test_core_046.py`。

### Changed
- `Op.replicated` 可为分数（D > S 时空闲的样本槽）。默认场景（单卡、PP、DP、无 DAP）的 1356 项结果与 0.45 逐字节一致；变化只在 DAP > 1 且样本数 > 1 的结构模型，以及新的可选项。

### 仍未建模
- 置信度头的样本分卡；单个样本的扩散 transformer 的 token 维切分（Boltz-1 默认 1 个样本时 DAP 帮不上扩散部分）；pair 的 TP。
- 主机 CPU 编码器只有一个算力旋钮；VAE 多卡并行解码；跨请求的组件 / 去噪重叠；LTX / Open-Sora / H3 的可选 VAE 分块。
- AlphaFold 3 仍为「暂未接入 v2」（无公开权重）。

## [0.45.0] - 2026-10-09

结构模型多卡（DAP）、视频组件放置（分片 / 卸载）、可选 VAE 分块解码。

### Added
- 结构模型 DAP（FastFold 动态轴并行，`layout.sp` 即 DAP 度，Web 显示为「DAP」）：pair / 模板网格沿残基轴、MSA 网格沿行或列切到 D 张卡，权重复制；单一 / token / 原子轨道每卡重复（不重复计入请求 FLOPs）。逐核通信：三角乘法 all-gather 投影操作数，三角注意力 all-gather 偏置 + pair all-to-all 转置，MSA 行注意力 all-gather 偏置，列注意力两次 MSA all-to-all，外积均值 all-gather 右投影（AF3 类 MSA 模块先加一次 all-to-all），pair 加权平均 all-gather 权重，单一轨道注意力 all-gather 偏置。转置载荷用检查点里各模块输入宽度（`PairCore.width`：c_z / c_m / c_t）。pair 在途激活与跨 stage 传输按 ⌈N / D⌉ × N × c_z。100T + HBM3E、DAP 8：AF2 14.2 → 1.81 s（7.8×）、OpenFold 7.8×、ESMFold 7.1×、Boltz-1 3.6×、Protenix 1.8×（扩散 transformer 在重复的单一轨道上）。布局枚举 / 搜索 / 「放不下」修正都包含 DAP。
- 视频组件放置 `workload.placement`（Web「组件放置」、CLI `--placement`）：`resident`（0.44）、`shard`（文本编码器权重 FSDP 切到本副本全部卡，每卡 te_w / c + 2 层，逐层 all-gather 与编码计算重叠——Wan `--t5_fsdp`）、`offload`（组件与 DiT 分时占用显存，需求取最大值，每请求按 `workload.host_GBps`（默认 50 GB/s「假设」）重载权重——diffusers `enable_model_cpu_offload` / Wan `--offload_model`；Wan2.2 的空闲专家也停在主机）、`shard+offload`、`auto`（默认：第一个放得下的）。结果、警告、KPI 与假设列表写明所选放置与重载时间。
- 可选 VAE 分块解码 `workload.vae_tiling`（Web 复选框、CLI `--vae-tiling`，默认关）：CogVideoX（tile 240 × 360 px，重叠 1/6、1/5 → 480p 9 tile ×1.40）与 Mochi（256 px / stride 192 → 15 tile ×1.65）按 diffusers 默认参数；重叠重复计算与每 tile 权重重读计入，激活峰值按 tile。其余 VAE 开启时给出警告。
- 测试 `tests/test_core_dap_place.py`：DAP 有用 FLOPs 守恒与通信闭式、布局 / 延迟 / 显存、请求 FLOPs 与布局无关、放置（auto / 分片 / 卸载 / 重载闭式 / Wan2.2 空闲专家）、分块 tile 数与重叠。

### Changed
- 64 GiB LPDDR5X 上（默认 auto 放置）：Wan2.1-14B 单卡 / DP2 / SP2、Wan2.2-A14B PP2 / TP2、MiniMax-H3 全部布局从「放不下」变为「放得下」（卸载 +1.4–2.5 s / 请求，或 H3 PP2·TP2 分片 +0 s）。显式 `placement: resident` 与 0.44 完全一致；其余 1330 项结果（全部 LLM、蛋白质、能常驻的视频场景）逐字节不变。
- 请求 TFLOP（`tflop_per_request` / `dit_tflop_per_request`）改为整个副本的有用 FLOPs（每 rank 有用 FLOPs × TP·SP ÷ 每 rank 序列数），此前多卡布局只计一个 rank（例：Wan2.1-1.3B SP2 显示 14.2 PFLOP，实为 28.3）；按 SP 不切分的条件行（跨注意力 K/V、每序列行）记为重复。
- 结构模型的说明与 Web 工作负载说明改为「PP × DP × DAP」。

### Fixed
- 0.44 文档误称 Qwen3-VL 文本塔（66.7 GB = 62.1 GiB）单独超过 64 GiB；实际是与 DiT 同时常驻放不下。
- 两个组件在同一张卡时，激活超出量按两者最大值计（此前分别相加；现有场景中数值未变）。

### 仍未建模
- pair 的 TP；扩散样本的 DP 切分；DAP 无公开推理时延可逐项校验。
- DiT 权重 FSDP（Wan `--dit_fsdp`）、文本编码器在主机 CPU 上运行（`--t5_cpu`）、VAE 多卡并行解码、跨请求的组件 / 去噪重叠；Wan / LTX / Open-Sora / H3 的可选 VAE 分块。
- AlphaFold 3 仍为「暂未接入 v2」（无公开权重）。

## [0.44.0] - 2026-10-09

视频整条 pipeline：文本编码器与 VAE 解码的时间与存储计入评估；激活 dtype what-if。

### Added
- 视频 pipeline 组件（`accel_dse/core/pipeline.py`，数据 `accel_dse/data/pipeline/*.json` 由 `scripts/build_pipeline_data.py` 从发布检查点张量头生成，不下载权重）：文本编码器 umT5-XXL（Wan）、T5 v1.1 XXL（CogVideoX / LTX / Mochi / Open-Sora 的 DeepFloyd 版）、LLaVA-Llama-3-8B + CLIP-L 文本塔（HunyuanVideo）、Qwen3-VL 文本塔（H3）；VAE 解码器 Wan-VAE、CogVideoX、HunyuanVideo（diffusers 默认 tiling，720P 308 tile、重叠 ×2.64）、LTX、Mochi、Open-Sora v1.2（时间 VAE + 逐帧 SD-VAE）、H3 ViT 视频解码器与音频 VAE。卷积按隐式 GEMM（输入流量按真实张量），文本塔按补齐 token 数，提示数随 batch × CFG。定义与假设见 docs/MODEL.md §11.4。
- 默认计入：`T_clip = 文本编码 + 去噪 + 解码`，组件 FLOPs 计入每请求 TFLOP，组件权重（加载的整个检查点，dtype 按发布）计入首 / 末流水级卡的容量。`workload.pipeline = false`（Web 复选框、CLI `eval --dit-only`）只看 DiT，结果与 0.43 逐字节一致。
- 激活 dtype what-if：Web「激活 dtype」、CLI `eval --act fp32|bf16`（即 `formats_override [["act", …]]`），用于评估结构模型参考实现的 fp32 激活流量（例：ESMFold 100T + LPDDR5X 6.3 s → 12.5 s）。
- Web：单段延迟卡拆分「文本编码 + 去噪 + 解码」，吞吐卡显示组件 TFLOP，假设列表随模型 / pipeline / 激活 dtype 动态生成，容量面板显示组件权重并提供「只评估 DiT」修正；API 的级存储与 fit 返回 `pipe_w_GiB`。
- 测试：`tests/test_core_pipeline.py`（检查点大小、umT5 闭式 FLOPs、Qwen GQA / causal、各家 VAE 分辨率表与发布输出形状、Wan-VAE 手算、pipeline 开关等价与延迟加和、batch / DP / 容量、场景 / API / CLI、fp32 激活 what-if）。

### Changed
- 视频模型覆盖全部为「完整」（说明中列出组件与假设）。`validate` 的 Wan2.1 / Open-Sora 行改为整条 pipeline FLOPs（0.72 / 0.14，仍在带内）。
- 容量结论变化（64 GB LPDDR5X）：Wan2.1-14B（+umT5 11.4 GB，所有单卡 / DP / SP 布局）、Wan2.2-A14B 的 PP2 / TP2、MiniMax-H3（Qwen3-VL 66.7 GB 单独超过容量，组件不跨卡切分）从「放得下」变为「放不下」。HBM 上全部仍放得下。
- 延迟（100T + HBM3E、batch 1）：整段增加 0.6%（Wan2.1-14B）至 16.4%（Open-Sora 720p，逐帧 SD-VAE 1158 TFLOP）；HunyuanVideo 解码 221 s。LLM 与蛋白质结果不变。
- 隐式 GEMM 的 DRAM 流量按算子真实输入张量（`Op.in_elems`），映射的输出字节按 max(2, 激活字节)。

### Fixed
- Web：切换模型时 badge 停留在「只评估 DiT 主干」；pipeline 偏好跨模型保留（现随模型重置为默认，只保留 SLO）；结构模型的 badge 与假设文本（原显示编码器前向）；pair 模型 SLO 单位。

### 仍未建模
- 组件的多卡切分与并行 VAE 解码、文本编码器卸载、跨请求的组件 / 去噪重叠；CogVideoX / Mochi / H3 可选 VAE tiling 的重叠开销。
- 结构模型 pair 表示的 DAP 切分；fp32 激活 what-if 只计数据搬运，计算仍在 bf16 阵列。
- AlphaFold 3 仍为「暂未接入 v2」（无公开权重）。

## [0.43.0] - 2026-10-09

蛋白质结构预测接入 core v2：ESMFold、AlphaFold 2、OpenFold、Boltz-1、Protenix 可选择、可评估。

### Added
- 可评估的结构模型：ESMFold（facebook/esmfold_v1）、AlphaFold 2（官方 JAX 参数 `params_model_1_ptm`，CC BY 4.0）、OpenFold（`finetuning_ptm_2`）、Boltz-1（`boltz1_conf.ckpt`）、Protenix v0.5.0。参数取自检查点张量头（PyTorch zip 检查点只读 `data.pkl`；AF2 的 `.npz` 在发布 `.tar` 内按 tar / zip / `.npy` 头读取；HTTP range 请求，不下载权重），与发布逐项一致。dtype 按发布（fp32；ESMFold 的 ESM-2 为 fp16）。覆盖：ESMFold「完整」，其余「部分」（MSA / 模板检索与特征化、松弛未建模）。
- 结构模型算子：按模块名检测的三角乘法、三角注意力、MSA 行 / 列（含全局）注意力、外积均值、pair 加权平均、带 pair 偏置的注意力、IPA、模板点注意力、原子局部窗口注意力；每个权重 GEMM 按名字归到单一 / pair / MSA / 模板 / 原子网格，行数随残基数、MSA 行数、模板数、原子数变化。主干每遍重跑（recycle + 1 遍），扩散模块每步一次、样本成批，置信度头每样本一次。定义见 docs/MODEL.md §12。
- 工作负载新增 `msa`、`recycles`、`samples`（扩散步数沿用 `steps`）与结构模型的批延迟 SLO `fold_slo_s`（默认 120 s「假设」）；命令行 `eval --msa --recycles --samples`；扫描路径 `workload.msa / recycles / samples`。
- Web：结构模型的工作负载输入（MSA 行数、主干遍数、扩散步数 / 样本数、SLO s）、说明与指标卡（每序列 TFLOP、MSA / 遍数 / 扩散）；布局只显示 PP × DP。目录中离线条目显示原因。
- 校验：`validate` 增加 ESMFold 行（论文：V100 上 384 残基 14.2 s → 折算 V100 fp32 峰值的 28%）；新增 `tests/test_core_structure.py`（参数、AF2 JAX 参数与 OpenFold 逐层 GEMM 一致、核检测、闭式 FLOPs、recycle / 步数 / 样本线性缩放、PP × DP 布局、API）。

### Changed
- 结构模型只允许 PP × DP 布局（pair 表示的 DAP 切分未建模），TP / SP 报错并说明原因。
- 仍为「暂未接入 v2」：AlphaFold 3（权重需向 Google DeepMind 申请、条款禁止再分发，没有可公开核对的发布文件），目录中写明原因并指向 Protenix / Boltz-1。
- LLM、0.41 与 0.42 模型的结果不变（1296 项指纹逐字节一致）。

### Fixed
- Web：选择蛋白质模型（ESM-2）时工作负载说明的脚本错误（读取了视频专用字段）。

## [0.42.0] - 2026-10-09

其余 6 个视频生成（DiT）发布接入 core v2，可选择、可评估。

### Added
- 可评估的视频模型：Wan2.2-T2V-A14B、HunyuanVideo、LTX-Video（2B v0.9）、Mochi 1 preview、Open-Sora STDiT3（1.2）、MiniMax-H3。均从官方 config + safetensors 头（含最后一层，捕捉异构末层）建模，参数与发布偏差 ≤ 0.001%；dtype 按发布（Wan2.2 / LTX / Mochi / Open-Sora 为 fp32；HunyuanVideo bf16；H3 为 bf16、io 层 fp32）。覆盖均为「部分」：文本编码器、VAE（H3 另有音频 VAE）未建模，标签列出各自大小。
- 结构：Wan2.2 两个 14B 专家按噪声级切换，每步只算一个，另一个计入常驻存储（不计读流量）；HunyuanVideo 20 双流 + 40 单流块（单流块一个融合输入 GEMM、TP 下一次 all-reduce），全 3D 联合注意力；Mochi 非对称双流（末层文本只作 K/V）；Open-Sora 按发布的分解时空注意力（空间块在潜帧内、时间块沿时间轴）；H3 视频 + 立体声音频 + 文本打包为一条序列，AdaLN 分支（13.0B）按 README 预计算缓存、不计存储与读流量。Wan2.2 / HunyuanVideo 为全 3D 注意力（不做时空分解）。
- 工作负载：各发布的潜空间帧规则（因果 VAE 4× / 6× / 8×、Open-Sora 17 帧分块、H3 的 17n+5 补齐）、音频 token、guidance 蒸馏（CFG = 1，HunyuanVideo / H3 每步一次前向，可 what-if 为 2）。Web 工作负载说明与指标卡、命令行 `eval` 显示音频 token 与分解注意力。
- 校验：`validate` 增加 Open-Sora 行（README：H100 上 720p 4s 用时 130 s，DiT FLOPs 折算 H100 利用率 13%，若按全 3D 注意力会超过 100%）；新增测试覆盖参数 / dtype、token 规则、FLOPs 与闭式计数、TP / SP 守恒（含联合文本与音频行）、standby / 缓存存储。
- 发布头抓取支持多子目录仓库（`repo:sub1+sub2`）。

### Changed
- 仍为「暂未接入 v2」：蛋白质结构预测（ESMFold、AlphaFold3 / 2、Protenix、Boltz-1、OpenFold）——需要 pair 表示与三角更新等算子。LLM 与 0.41 模型的结果不变（1176 项指纹逐字节一致）。

## [0.41.0] - 2026-10-08

视频生成（DiT）与蛋白质模型接入 core v2，可选择、可评估。

### Added
- 可评估的视频生成模型：Wan2.1-T2V-14B / 1.3B、CogVideoX-5b / 2b（DiT 去噪主干，全 3D 注意力；CogVideoX 为文本 + 视频联合注意力）；蛋白质：ESM-2 3B / 650M（编码器一次前向）。均从官方 config + safetensors 头建模，参数与发布逐项一致；dtype 按发布（Wan2.1 / ESM-2 为 fp32，逐 GEMM 上转换为 bf16 并计入向量开销）。覆盖：视频「部分」（文本编码器与 VAE 未建模，标签列出其大小），ESM-2「完整」。
- 场景新增 `workload`（帧数、分辨率、去噪步数、CFG、蛋白质序列长度、单段 / 批延迟 SLO；0 = 发布默认），布局新增 SP（Ulysses 序列并行）；视频 / 蛋白质的 DP 切分前向 batch（含 CFG 并行）。
- 指标：视频单段延迟、每帧延迟、每去噪步时间、帧/s/卡、段/小时/卡、实时倍率；蛋白质批延迟、序列/s/卡、残基/s/卡。精确 batch / 布局搜索、映射对比、扫描、Pareto 与容量提示都支持这两个领域（goodput 只用于 LLM）。定义见 docs/MODEL.md §11。
- 全序列前向的激活流式模型（激活超出 SRAM 时分块进出 DRAM，注意力按 flash 式重读 K/V）；全序列注意力的映射可选 GEMM 方向（`Oᵀ = Vᵀ·Pᵀ`）。
- Web：模型选择器中这 6 个模型可选；工作负载输入、SP 输入，领域专用的指标卡与表头；LLM 专用控件（phase、ctx、TPOT/TTFT SLO、投机解码、KV what-if、goodput 目标）对视频 / 蛋白质隐藏。命令行 `eval` 支持 `--frames --height --width --steps --cfg --seq-len --sp`。
- 校验：视频 / 蛋白质的 V0 不变式、FLOPs 与独立计数对照、TP / SP 守恒、精确搜索对照暴力枚举；`validate` 增加一行视频合理性核对（Wan2.1 README 4090 用时折算）。

### Changed
- `/api/health` 的 domains 为 llm / vlm / gen / protein。其余视频 / 蛋白质条目（Wan2.2-A14B、HunyuanVideo、MiniMax-H3、LTX-Video、Mochi 1、Open-Sora、ESMFold、AlphaFold3 / 2、Protenix、Boltz-1、OpenFold）仍为「暂未接入 v2」。LLM 结果不变（266 项指纹逐字节一致）。

## [0.40.2] - 2026-10-08

### Changed
- 模型目录与选择器改为按厂商 → 系列分组，不再把 LLM 与 VLM 分成两组：同一厂商（以及同一系列）的文本版与多模态版放在一起，VLM 行带「VLM · 视觉编码器未建模」标记。同一厂商的不同品牌合并（阿里：Qwen · Wan；智谱：GLM · CogVideoX；Meta：Llama · ESM；字节跳动：Seed · Protenix）。
- 视频生成（DiT）条目列在各自厂商之下、可评估条目之后，标「视频生成 · 暂未接入 v2」；模型选择器中也以不可选项列出。

### Added
- 蛋白质模型回到目录（ESM-2 3B / 650M、ESMFold、AlphaFold3 / AlphaFold2、Protenix、Boltz-1、OpenFold），标「蛋白质 · 暂未接入 v2」，不能评估；维数取自原目录。
- `/api/models` 增加 `catalog`（完整目录的显示顺序：厂商 → 可评估系列 → 暂未接入的系列）；`offline` 条目增加 `evaluable`、`source`。模型目录页顶部给出各领域条目数。

## [0.40.1] - 2026-10-08

Web 工作台打磨与搜索提速。

### Changed
- 外存目录按 2026-10-08 审计修正：标签拆成规范状态 × 产品状态两轴；去掉 LPDDR5X x96（Apple 定制件，其他客户买不到）；LPCAMM2 9600 升为量产；SOCAMM2 256 GB / LPDDR5X x32 64 GB 降为送样；HBM4 12H×32Gb 改为「JEDEC 允许 · 无产品」（样品是 16H×24Gb）；LPDDR6 补 11733 档、12800 升为 JEDEC、x96 的 8/12 GB 标签统一；LPDDR5T = LPDDR5X-9600 别名；可选 LPDDR6 meta 模式（默认关，容量预留 1/16 为「假设」）；8/9 payload 保留。
- 卡数成为唯一的场景字段（每副本卡数 = PP·TP·DP）：映射对比与布局搜索都使用场景的卡数，不再各自有卡数输入；改卡数时按 TP 重新填充布局，「搜索当前卡数的最优布局并应用」一键给出最优布局与 batch。
- 默认场景可行：Web 默认 TPOT SLO 100 ms（LPDDR 类存储器）并开启自动 batch，打开即是满足 SLO 的结果（核心库的默认 SLO 仍为 50 ms）。
- 放不下时不再满屏红色：指标条隐藏，改为中文提示（需求 / 容量 / 权重）和一键修正——最少卡数及其最优布局与 batch、同类型更大容量的存储器、或可放下的最大 batch；TPOT 超 SLO、TTFT 不达标用琥珀色提示。
- 指标条精简为 5 项（TPOT、吞吐 / 卡、瓶颈与有效 MAC、goodput、DRAM 需求 / 容量），1100 px 以上固定一行。
- 映射对比表：次优布局并入布局列第二行，1280 / 1440 下不截断；放不下的映射显示「放不下」且不可点击；点击行同时应用映射、布局（含卡数）与 batch。布局搜索在 decode 目标下不显示空的 goodput / TTFT 列。
- 界面文字统一为中文（技术术语保留英文）：模型注释、结构描述、容量与投机解码警告、稳定性扰动名称、表头等。
- 扫描图的整数横轴不再显示小数。

- 模型目录与选择器按领域 → 厂商 → 系列分组（LLM / VLM / 图像与视频生成），去掉「用户指定 / 推荐补充」这类按来源的分组；名称统一为官方仓库名，系列内按尺寸从大到小，FP8 / AWQ 紧跟原版。
- 每个模型都显示覆盖度（完整 / 部分 / 架构代理，「完整」用低调样式），非「完整」模型附逐项的近似之处（悬停与模型目录可见），并说明覆盖度指建模覆盖而非模型好坏。
- VLM（Qwen3.5 / Qwen3.8-27B / Qwen3.8-Flash-Next / DeepSeek-V4.1-Flash / Kimi-K2.5 / Kimi-K3 / GLM-5.3-Flash）单独成组，注明只评估语言主干、视觉编码器未建模。
- 结构与 dtype 相同的发布合并为一条（DeepSeek-V3 / V3.1 / R1、Kimi-K2.5 / K2.7-Code、GLM-5 / 5.2、GLM-4.5 / 4.6、MiniMax-Text-01 / M1-80k）；Yi-1.5-34B、InternLM2.5-20B、InternLM3-8B、Qwen2.5-3B / 1.5B 不再单列。它们仍可按 id 评估并参与参数核对。
- 图像 / 视频生成（DiT）条目回到目录，单独成组并标「暂未接入 v2」（不可评估）。

### Performance
- 批量搜索的上界改为由所有已评估点构成的分段上界（仍只依赖 step(B) 不减），并且只在上界可能超过门槛时才继续二分 b_max。
- 稳定性检查以原 top-1 布局的精确得分为门槛做跨布局分支定界。
- HTTP 服务把映射对比（每种映射一个任务）与稳定性请求放到工作进程池并行（服务退出或被杀时工作进程随之退出）；Web 端并发请求 5 个稳定性结果并逐行填入。
- DeepSeek-V3 × 8 卡（HBM3E 288 GiB，SLO 100 ms）映射对比 + 5 项稳定性：0.40.0 单进程约 170 s → 0.40.1 单进程约 10–15 s，Web 服务（工作进程并行）冷启动约 4–5 s。

### Added
- `/api/fit`：容量检查与修正方案（见 docs/MODEL.md §7）。
- `/api/models` 增加 `domain` / `provider` / `family` / `coverage_reasons` / `vision_params_B` / `same_as`，以及 `offline`（暂未接入的生成模型）与 `unlisted`（未单列的发布及原因）；去掉 `section`。
- 测试：`/api/fit` 修正方案真实可行且卡数最少；稳定性的门槛搜索与完整搜索一致；目录分组、覆盖度与原因一致、合并条目确为同结构同 dtype。

## [0.40.0] - 2026-10-08

核心重写（core v2）。旧引擎、旧 Web 与旧命令行已移除，API 与 CLI 不向后兼容。

### Added
- 模型按发布建模：从 HF config 与 safetensors 头读取逐角色参数与存储 dtype（fp8 block、MXFP4、NVFP4、int4 AWQ 等），55 个发布与总量偏差 ≤0.5%；官方量化版为独立条目；三轴标签（来源 × 覆盖 × dtype），未逐项建模的结构标「架构代理」。
- 逐 rank 算子图：TP / PP / 注意力 DP / EP / ETP，单卡与多卡同一路径。
- 映射作为设计变量：OS、WS 边缘加载、WS 宽面广播、OS + GEMV、可重构；逐算子 MAC 界与 SRAM 供数界；芯片原生格式矩阵与反量化开销。
- 存储规划（SRAM 驻留、staging、逐 stage 容量）与调度（绑定瓶颈、有效 MAC 比例、α-β 集合通信、投机解码 / MTP）。
- 精确 batch 搜索（分支定界）、布局排名、TPOT–吞吐 Pareto、含 prefill 的 goodput 与 DP prefill TTFT 标记、排名稳定性。
- 校验套件：变形关系、参数对照、H100 类趋势区间、GenZ 对照（`accel-dse validate`）。
- Web 工作台重写：单点评估、映射对比（每种映射的最佳布局、瓶颈、有效 MAC、TTFT、稳定性）、布局搜索、扫描与 Pareto、模型目录；请求按序号丢弃过期响应。
- JSON API：严格解析（拒绝未知字段与 NaN / Infinity），扫描为受控替换。
- 命令行：`serve` / `models` / `eval` / `search` / `compare` / `stability` / `validate`。

### Removed
- 旧解析引擎及其命令（workbench、pareto、report 等）、示例输出与能耗 / 成本 stub。
- 视频与蛋白质领域暂时下线（见 docs/MODEL.md §10）。

## [0.31.1] - 2026-10-08

### Changed
- 项目改名 `accel-dse` 并发布到 GitHub；Python 包 `npu_dse` → `accel_dse`，命令行入口 `accel-dse`（无旧包名兼容层）。

### Added
- MIT License；`.gitignore`（生成物 `out/*` 默认忽略，保留文档引用的示例输出）。

## [0.31.0] - 2026-10-08

### Added
- MoE 专家切分 `moe_shard = tp_ep | ep_all`；MoE ep > 1 时注意力在 ep 组间按 DP 运行。
- PP decode 微批 `decode_mb`（默认 min(B, pp)），流水填满。
- goodput 计入 prefill：吞吐口径「摊销（分块混合 / 独占）| 上界」，`--goodput-mode` / `--prefill-mode` / `--out-len`。
- 投机解码 / MTP：`spec_k` / `spec_accept` / `spec_draft`；开启 MTP 时容量计入 MTP 模块。
- Web：⚙「投机解码 / MoE 切分 / PP 微批」，分解新增「草稿」与「PP 遍历 / 气泡」。

### Changed
- 工作台每个 decode 步计入 LM head（注意力 TP 时按词表并行）。
- 帕累托 LLM batch 搜索上限 4096 → 65536。

### Fixed
- all-to-all 字节按每 rank token 份额计（此前按整批，偏高）。
- tp > 1 且 ep > 1 的 MoE 布局不再把专家重复存储 tp 份。

## [0.30.0] - 2026-10-08

### Added
- 吞吐–交互性帕累托（`pareto` CLI、`POST /api/pareto`、Web「吞吐 / 交互」标签页）与 SLO goodput（KPI「SLO 吞吐 / 芯片」）。
- 视频 / 蛋白等效吞吐–时延曲线。

### Changed
- `capacity_GB` / `capacity_needed_GB` 统一为厂商标称 GB（2³⁰ B），十进制值见 `*_decimal`。
- 扫描行使用短标签（原始 id 在 CSV `raw_id`）；移动端存储配置改为底部抽屉。

### Fixed
- MLA 注意力权重按投影建模（DeepSeek-V3 ≈ 670.9B，此前偏大）；补齐前置稠密层、非共享 LM head、MLA KV 的 RoPE 部分。
- 旧 `lpddr6_{n}x24_*` 链接按总线位宽映射到 x96 / x48 封装。

## [0.29.0] - 2026-10-08

### Added
- 结构化存储目录（`mem_catalog.py`）：LPDDR5 / 5X / 6、SOCAMM2 / LPCAMM2、HBM3 / 3E / 4 / 4E；来源标签取最弱项；LPDDR6 payload 8/9。
- 暴露的通信同步时延 `t_sync = n_sync · α · (1 − overlap)`（α 默认 3 µs，假设）。
- `attn_parallel = tp | dp`；新场景预设 `edge-lpddr6-4x96`、`card-hbm3e-8x12h`、`server-socamm2-8`。

### Fixed
- LPDDR 几何：LPDDR5/5X 封装 x64（4×16），LPDDR6 封装 x96（4×24）；HBM 容量 = 层数 × die 密度；速率档绑定代际。
- KV 按 TP 切分（GQA 按 KV 头；MLA latent 每个 TP rank 全量）。
- ≤ 0.28 的封装 id 自动映射到最近的新配置。

## [0.28.1] - 2026-10-08

### Changed
- Web：下拉框短标签不截断；时间 / 字节单位自动缩放；抽屉打开时场景栏保持两行。

## [0.28.0] - 2026-10-08

### Changed
- Web 布局重构：吸顶场景栏、KPI 条、结果标签页、⚙ 高级参数抽屉、🔗 复制链接、宽屏分屏与移动端摘要。
- 测试的版本断言改为读取 `__version__`。

### Fixed
- 默认场景不再 OOM；芯片数步进器只给出引擎接受的取值。

## [0.27.0] - 2026-10-08

### Changed
- Web UI 简体中文化（技术术语保留英文）；修复表单 / 卡片 / 表格的文字重叠。

## [0.26.0] - 2026-10-08

### Added
- 可选（assumed）`non_gemm_overhead`（Softmax / RoPE / LN 粗开销）与 `dtype_mac_factors`（低精度 MAC 峰值因子），默认关闭。

### Fixed
- `export-csv` / `report` 默认输出改为相对仓库的 `out/`。

## [0.25.0] - 2026-10-08

### Added
- MetricsCard `scale_efficiency` / `speedup`（相对单卡）；报告新增 scale 章节。

## [0.24.0] - 2026-10-08

### Added
- 能耗 / 成本 assumed stub（`tdp_w`、`watts_per_tops`、`cost_per_card_usd` 等，默认关闭）。
- 场景预设（`list-presets`、`--preset`、`GET /api/presets`）。

## [0.23.0] - 2026-10-08

### Added
- CalibrationOverrides（`--calib`、API `calib`）；Web 假设值覆盖滑条；A|B 双卡对比。

## [0.22.0] - 2026-10-08

### Added
- 扫描结果导出 CSV / JSON；固定基线 Δ%；`?c=` 深链保存配置。

## [0.21.0] - 2026-10-08

### Added
- Web 对比 / 扫描面板与 `POST /api/sweep`（chips / package / compute / parallel / series，≤ 32 行）。

## [0.20.0] - 2026-10-08

### Added
- 视频 / 蛋白多卡 TP / PP / EP（ring / tree 集合通信、PP 气泡）。

## [0.19.0] - 2026-10-08

### Added
- 视频 / 蛋白多卡（chip_count → TP）。

### Fixed
- LPDDR 封装粒度改为 64-bit 器件。

## [0.18.1] - 2026-10-08

### Fixed
- `/api/eval` 接受 `series` / `series_id` 作为 `model_id` 别名。

## [0.18.0] - 2026-10-08

### Added
- `/api/eval` 支持视频 / 蛋白 MetricsCard。

## [0.17.0] - 2026-10-08

### Added
- 本地交互 Web 工作台 `serve`（stdlib，可选 FastAPI）。

## [0.16.0] - 2026-10-08

### Added
- 存储封装 / 算力层级目录（`list-packages`、`list-compute`）与对应扫描。

## [0.15.0] - 2026-10-08

### Added
- 公开 HF config / model card 模型维数目录（LLM / 视频 / 蛋白），`list-series --product`。

## [0.14.0] - 2026-10-07

### Added
- 自包含离线 HTML 报告 `report`。

## [0.13.0]

### Added
- `workbench-parallel`（tp×pp×ep 矩阵）、存储几何扫描、MetricsCard JSON / Markdown 导出。

## [0.12.0]

### Added
- 产品层 `WorkbenchConfig → MetricsCard`；`n_cores × tops_per_core`；模型系列包。

## [0.11.0]

### Added
- 跨域对照 `compare-domains`；`illustrative_large_dit`。

## [0.10.0]

### Added
- 视频 DiT / 蛋白示意负载与扫描。

## [0.9.0]

### Added
- MoE 专家并行与 all-to-all；MLA 压缩 KV；`export-csv`。

## [0.8.0]

### Added
- IB / RoCE KV fabric；轻量流水并行与 decode 气泡。

## [0.7.0]

### Added
- 张量并行 + C2C 集合通信；容量 OOM 标记。

## [0.6.0]

### Added
- batch 扫描；MoE 示意形状。

## [0.5.0]

### Changed
- SRAM：区分 staging 与 resident，resident 膝点；`--sram-policy`；独立 W/KV 量化。

## [0.4.0]

### Added
- 上下文扫描、SRAM 容量扫描、dtype（只缩字节）。

## [0.3.0]

### Added
- SKU 峰值模板 `sku_100t` / `sku_1p`；`scan-sku`。

## [0.2.0]

### Added
- 示意 27B 形状；HBM / LPDDR 预设；SRAM 三分区；带宽争用模式。

## [0.1.0]

### Added
- 初版：output-stationary PE 与 M=1 利用率公式；由 tiling 推导外部流量；TTFT / TPOT；toy 手算用例。
