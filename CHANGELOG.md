# Changelog

本项目的重要变更记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)（1.0 之前次版本号可能包含不兼容变更）。
0.31.0 及更早版本以 `npu-inference-dse`（包名 `npu_dse`）发布。




## [0.361.0] — 2026-10-24

### RWKV-8 Heron pile rc00（纯循环 + hybrid DeepEmbedAttention）

- **新增 2 个发布**，形状和 dtype 来自 `BlinkDL/rwkv-8-pile` 的 `.pth` 头（2026-10-24 读取），不是臆造。`param_check().rel_err == 0`，覆盖完整。
  - **rwkv8-pile**（`rwkv-8-pile-rc00-20250508.pth`，414 个张量，全部 bf16）：12 层、hidden 768、词表 50304、FFN 3072、head 64。时间混合与 RWKV-7 相同（WKV-7，fp32 状态 H×64×64）。DeepEmbed 是每层 `ffn.enn` `[50304, 768]`，只按当前 token 取一行。参数 631,289,856。
  - **rwkv8-pile-hybrid**（`rwkv-8-pile-rc00-hybrid-20250623.pth`）：同一时间混合和 FFN，没有 ln0。`emb` / `ffn.enn` / `qkv.ennk` / `qkv.emm` 是 fp16，GEMM 权重和 `lm_head` 是 bf16。另有 DeepEmbedAttention：缓存的是每层 fp16 的 kk/vv（各 32 维），每步对全部已缓存 token 做 `kkup` / `vvup`，乘 `ennk` / `emm` 行，`32·tanh(qk/512)` 后单头因果注意力。参数 1,252,796,928。
- **机制**（`rwkv_v8_rc00_demo.py` / `rwkv_v8_rc00_hybrid_demo.py`）：时间混合复用已有 `lin=rwkv7`。demo 的序列路径调用的是同一个 WKV-7 算子；prefill 仍按 RWKV-7 的 chunk 64 矩阵形式计，不另造一套递推。通道混合的 squared ReLU 沿用现有激活算子（与 RWKV-7 相同）。DeepEmbedAttention 是在现有 GEMM / 查表 / 因果注意力因子上补的一条 schedule：缓存字节按 demo 的 `torch.half`（2 字节），不跟 `kv_fmt` 假设走；参考实现是单卡，投影按复制计，缓存不按 TP 切分。
- **决策模型这条没有新权重**：
  - TypeSafe Jev 仍是托管闭源，没有公开 config / 权重。第三方蒸馏和 GGUF / MLX 不入库。
  - Microsoft-Decision-1 仍无权重，目录里继续是 Qwen3.5-9B 代理。
  - StartLux v1 / v1.1 和 ReaDy-Qwen3.8-27B 已在目录中。没有 v1.2 / v2。剩下的同仓库条目是 GGUF / MLX。
- **RWKV 没纳入的**：RWKV-7 G1k（2026-09-30）已是带 `config.json` 的最新一代，已在目录。更老的 G1j / Goose 不是最新。ROSA 只在玩具实验里，这两个检查点没有对应张量。

## [0.360.0] — 2026-10-24

### 语音目录：公开配置可直接映射的 ASR / 端到端旁支 (MODEL §360)

- **新增 15 个发布**，全部复用已有 builder，`param_check().rel_err == 0`，dtype 按发布文件，不新造形状：
  - **Whisper English**（`openai/whisper-{tiny,base,small,medium}.en`，fp32，词表 51864，比多语少一行）：encoder-decoder，`params()` 含编码器。
  - **Distil-Whisper**（`distil-whisper/distil-{small.en,medium.en,large-v2,large-v3,large-v3.5}`）：同一 Whisper 机制，解码层更少。权重 dtype 跟 safetensors header（medium.en / large-v2 的 config.torch_dtype 写成 float32，张量是 F16）：small.en、medium.en、large-v2、large-v3 为 fp16，large-v3.5 为 fp32。large-v3 与 large-v3.5 词表 51866、mel 128；large-v2 词表 51865、mel 80；small.en / medium.en 词表 51864。
  - **Qwen3-ASR-0.6B**（非 `-hf` 仓库，`qwen3_asr_audio_encoder`）：AuT 18 层 d896 + Qwen3-0.6B thinker，bf16，覆盖完整。
  - **Qwen2-Audio-7B**（base，与 Instruct 同形）：Whisper-v3 编码器 + Qwen2-7B，bf16，覆盖完整。
  - **Step-Audio-R1.1**：`step_audio_2` 编码器 + Qwen2-32B 形 LLM，bf16；发布里没有 token2wav，目录域为 ASR，覆盖完整。
  - **Qwen3-Omni-30B-A3B-Thinking / Captioner**：与 Instruct 相同的 Thinker（AuT + MoE + ViT），`enable_audio_output=false`，没有 Talker / code2wav。目录域 omni，语音 KPI 走 ASR。覆盖 partial（ViT 只计存储，与 Instruct 同一条原因）。
  - **Kimi-Audio-7B**（base）：与 Instruct 同形（28+6 层、Whisper 编码器、DiT、BigVGAN），bf16。目录域 omni，语音 KPI 为 kimi。覆盖 partial（GLM-4-Voice 分词器不在本仓库，与 Instruct 同一条原因）。
- **未纳入**（配置不公开、机制对不上现有 builder，或需要臆造维度）：
  - 已在 0.270–0.357 覆盖：Qwen2-Audio-Instruct、Qwen2.5-Omni 3B/7B、Qwen3-Omni Instruct、Qwen3-ASR-1.7B、CosyVoice2/3、Step-Audio-2-mini/Base/Think、Moshi moshiko/moshika、Kimi-Audio-7B-Instruct、Whisper tiny→large-v3/turbo。
  - Qwen-Audio / Qwen-Audio-Chat：`audio` 只有 `output_dim`，编码器维度不在 config。
  - Qwen3-ASR `*-hf`：`model_type` 是 `qwen3_asr_encoder`，对不上现有 AuT 门。
  - Step-Audio-R1、kyutai/pocket-tts：HF 401。
  - Step-Audio-EditX / TTS-3B / Chat / AQAA：主干是 step1 或 mmgpt，不是 `step_audio_2`。
  - Fun-Audio-Chat-8B：公开 config，但是 CRQ + invert tower，没有现成 schedule。
  - Fun-ASR-MLT-Nano-2512、SenseVoiceSmall：没有 transformers `config.json`。
  - CosyVoice-300M（v1）：yaml 公开，但是 TransformerLM + MaskedDiff，不是 CosyVoice2/3 builder。
  - kyutai/stt-2.6b-en、hibiki-1b / hibiki-2b：不是 Moshi 的 temporal + depth schedule。

## [0.359.0] — 2026-10-24

### DVFS 封顶后 HBM 器件功耗不再按卡功耗比例缩小 (HBM thermal)

- **Bug：** 功耗预算把工作点降频后，DSE / eval 用 `dram_dev_W × W_capped / W_nominal` 估算晶粒外 DRAM 器件功耗。HBM（及 LPDDR）静态是存储器时钟上的恒定瓦数，不随核心卡功耗下降。深封顶时该比例会把器件功耗压到静态以下，多出来的热被算进晶粒（`die_W` / `die_flux`）。卡级热流与 `pkg_W` 不变，所以 `thermal_binding` / `front_cap` 的功耗轴不受影响；变的是晶粒热流，以及「卡、晶两侧都超」才判热流超限时的可行性。
- **修复：** `dvfs_cap` 返回封顶时钟上能耗报告的 `dram_dev_W`（低于 `F_MIN` 的占空比只缩小动态部分，器件静态保持）。能耗 DSE 与 `/api/eval` 的热报告用这个数，不再按卡功耗比例缩放。
- **未改：** 冷却上限、功耗预算数值、`power_scope` 比较轴、三方 `thermal_binding`、前沿 `front_cap`。卡口径仍保留名义静态（0.315 复现）。

## [0.358.0] — 2026-10-23

### Whisper encoder-decoder 参数计数修复 (MODEL §358)

- **Bug 修复**: Whisper 编码器-解码器模型的 `params()` 之前只返回解码器参数（whisper-large-v3 报告 906M，实际应为 1.54B）
- **根本原因**: 编码器参数存储在 `vision_params` 中，但 `params()` 方法未包含编码器部分
- **修复内容**:
  - `Model.params()` 现在对编码器-解码器模型（`enc_ctx > 0` 且 `vision_params > 0`）包含编码器参数
  - `release_params` 对 Whisper 模型设置为 `params_llm + params_vision` 总和
  - 添加 test_core_358.py 验证修复并防止回归
- **影响模型**:
  - whisper-large-v3: `params()` 从 906M → 1.54B（+637M 编码器）✓
  - whisper-large-v3-turbo: `params()` 从 172M → 809M（+637M 编码器）✓
  - 其他 ASR 模型（Qwen2-Audio / Qwen3-ASR / Fun-ASR）不受影响（编码器已包含在 params_llm 中）
- **机制正确性**: 编码器是 Whisper 模型的核心组件（非可选视觉塔），应计入总参数
- **参数对齐**: `param_check()` 显示 `rel_err = 0.0`，与发布 safetensors header 完全对齐

## [0.357.0] — 2026-10-23

### 语音模型覆盖补充 (MODEL §357)

- **Whisper 系列补全**：OpenAI Whisper 全系列 ASR 模型（公开架构配置 + safetensors headers）
  - whisper-tiny（39M encoder + 38M decoder；4 层 encoder/decoder，384 维）
  - whisper-base（74M；6 层，512 维）
  - whisper-small（244M；12 层，768 维）
  - whisper-medium（769M；24 层，1024 维）
  - whisper-large（1.5B；32 层，1280 维，v1 原始版）
  - whisper-large-v2（1.5B；32 层，1280 维，v2 改进版）
  - 已有 whisper-large-v3 / -turbo 保持（0.270 已覆盖）
- **机制**：标准 Whisper encoder-decoder，cross-attention decode，30 秒固定窗口，FP32 as-released dtype
- **覆盖**：完整（all 6 size variants，encoder + decoder GEMMs + cross-attention + positional embeddings；参数总数对齐 HF headers）
- **调研**：Qwen-Omni / Qwen-Audio 系列（Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Qwen3-ASR, Qwen3-TTS）已在 0.270-0.346 完整覆盖；Kimi-Audio-7B（0.273）、CosyVoice2/3（0.271）、Step-Audio-2（0.272, 0.339）、Moshi（0.274, 0.339）均已存在；本轮专注 Whisper 官方全系列补齐

## [0.356.0] — 2026-10-23

### MiniMax M3 和 M2.7 (MODEL §356)

- **MiniMaxAI/MiniMax-M3**（non-Flash，native multimodal MoE ~426B / 23B active）：复用 M3.1-Flash block-level MSA 机制
  - 架构与 M3.1-Flash 一致：60 层，128 local experts / 4 per tok / 1 shared，block-level max-scored sparse attention（块大小 128，top-16 块，索引器 4 头 × 128 维）
  - 主要区别：无量化（BF16 存储），vision tower image_size 2016（vs M3.1-Flash 672）
  - 覆盖：**完整**（all GEMMs + block-level MSA + MoE + vision；params 426.4B 对齐 header）
- **MiniMaxAI/MiniMax-M2.7**（MoE ~229B / 29B active，fp8 量化）：纯 MoE 架构（无 sparse attention）
  - 62 层，256 local experts / 8 per tok / 0 shared，intermediate 1536，hidden 3072，GQA 48 heads / 8 KV（head_dim 128）
  - 量化：fp8（float8_e4m3fn，weight_block_size 128×128）主干，gate/lm_head 未量化（as-released）
  - MTP 模块：use_mtp=true，num_mtp_modules=3，mtp_transformer_layers=1（hybrid builder 路径，attn_type_list 全 1 = 全 full attention）
  - 覆盖：**完整**（all GEMMs + MoE + MTP；params 228.7B 对齐 header）

## [0.355.0] — 2026-10-23

### MiniMax M3.1-Flash 块级稀疏注意力 (MODEL §355)

- MiniMaxAI/MiniMax-M3.1-Flash（native multimodal MoE ~428B / 23B active）：block-level max-scored sparse attention（MSA）
- **机制新增**：`AttnCore.idx_score_type`（max | mean）区分块稀疏选择策略；max = 每块取索引器分数最大值，mean = 块级 mean pooling
- **block-level MSA**：sparse_block_size 128，sparse_topk_blocks 16（选 top-k 块），sparse_score_type max；索引器 4 头 × 128 维；ir.py 按块最大值计分 + 块级 top-k（≠ 现有 token-level 或 mean-pooled QSA）
- **MoE**：128 local experts / 4 per tok / 1 shared（前 3 层 dense），intermediate 3072 / dense_intermediate 12288 / shared_intermediate 3072
- **量化**：mxfp8 主干 + NVFP4 MoE experts（as-released dtype，硬件原生支持矩阵 + 反量化开销）
- Vision tower（ViT-like 1280/16 heads/32 layers, patch 14, image 672）+ multimodal projector 已建模（vision.py 路径存在）；DSpark draft head 存在但未建模（代码库无 clean hook）
- 覆盖：**完整**（all GEMMs + block-level MSA mechanism + MoE + vision；params 426.4B 对齐 header）

## [0.354.0] — 2026-10-23

### Validation and catalog review

- Full test suite green (1906 tests, 0 failures). Flaky list remains empty.
- 0.353 front_cap adversarial checks verified complete: power-budget near-ties correctly flagged across all cooling modes and power scopes.

### Catalog review: MiniMax M3.1 Flash

- Investigated MiniMax-M3.1-Flash (HF: MiniMaxAI/MiniMax-M3.1-Flash, 2026-10-19): 428B total / 23B active MoE, 1M context, mxfp8 quantization-aware training.
- Architecture requirements for clean integration: block-level sparse attention (MSA: `sparse_topk_blocks=16, sparse_block_size=128` — different from existing token-level `topk`), sparse indexer modeling (`sparse_num_index_heads=4, sparse_index_dim=128`), mxfp8 format verification from safetensors headers, Vision tower + multimodal handling.
- **Decision: deferred to 0.355.** M3.1 Flash requires significant new sparse-attention mechanisms (block-level selection, indexer cost) that do not fit a focused 0.354 release. Mechanism-first principle: verify block-sparse FLOPs and memory from first principles before catalog entry.

### Path

- 0.355: MiniMax M3.1 Flash clean builder (block-sparse MSA, mxfp8 verification, sparse indexer).

## [0.353.0] — 2026-10-23

### Adversarial front_cap near-ties (MODEL §298)

- 0.352 three-way `thermal_binding` / design-level 功耗预算 margin warn were correct, but `_front_thermal` only flagged flux / pkg ties — liquid+package front points sitting on the DVFS power_cap (~100% 占用) produced no `front_cap` warning.
- Front points now carry `cap_ratio` / `cap_margin*` on the same `power_scope` axis as DVFS; `near_ties` add `front_cap`; markdown table shows 预算占用. COOLING pins unchanged. No visual UI restyle (§293).

## [0.352.0] — 2026-10-23

### Adversarial card/die ratio persistence + UI (MODEL §297)

- thermal_report / DSE rows / report / chip+DSE UI still on card axis; liquid headroom ≈1.5% not feas ~140%.
- COOLING pins unchanged. `build_static` still copies live `app.js` (card-axis strings present).
- Pages branch (wt-pages @ 0.259) still pre-0.351 — rebuild from main is a separate track; not invented here.

### Power-budget three-way binding (mechanism)

- Design report adds `cap_margin_W` / `cap_headroom` / `cap_ratio` on the same axis DVFS uses (`power_scope`: card W vs package W).
- `thermal_binding` = max(卡级热流, 冷却封装功耗, 功耗预算). **liquid + package flips 热流 → 功耗预算** — changes which knob to turn.
- Limits / COOLING never retuned. No NPU absolute invent; no speech catalog churn.

### Path / deferred

- NPU: HBM4 still vendor ≈40% only — no absolute pJ. **No visual UI restyle** (no written user yes).

## [0.351.0] — 2026-10-23

### Adversarial card-axis margins across report / eval / DSE / static (MODEL §296)

- Re-verified liquid: design `flux_headroom` ≈1.5% (card), not ~+140% (feas min). COOLING pins unchanged.
- Gap: `/api/eval` already returned full `thermal_report`, but DSE rows dropped `card_flux_ratio` / `die_flux_ratio`, and the chip panel + DSE table showed feas `flux_ratio` as 「× 上限」— same class of bug as 0.350 for UI/static.

### Mechanism: persist + display card-axis ratios

- `thermal_report` exposes top-level `card_flux_ratio` / `die_flux_ratio` (limits[*] keep `card_ratio`/`die_ratio`).
- DSE `row.thermal` persists both; report relabels `flux_ratio` → 「可行性热流比 min(卡,晶)」; DIRECTION adds `pkg_margin_W` / `pkg_headroom` / `card_flux_ratio`.
- Web: chip thermal + DSE tooltip use **卡级** × 上限 (die note when skewed).

### Path / deferred

- NPU absolute search empty again (HBM4 ≈40% only; SPHBM4 interface; no public PCIe/CXL N5 mm²) → **no invent**.
- No speech catalog churn. **No visual UI restyle** (still no written user yes, §293).

## [0.350.0] — 2026-10-23

### Adversarial package-power / flux margins (MODEL §295)

- COOLING limits unchanged. Package margin / front_pkg arithmetic OK; no-cooling omits limits.
- **Bug:** `flux_headroom`, `thermal_binding`, front flux near-ties, and margins「热流」used `thermal.flux_ratio` = min(card, die) (feasibility: over only if both). Card-level `flux_margin_W_mm2` disagreed — e.g. liquid looked +140% expandable while card flux had ~1.5% left.

### Card-axis thermal margin consistency (mechanism)

- Margins / headroom / binding / front flux ties use **card** flux ÷ `COOLING.flux_W_mm2` (same axis as `flux_margin_W_mm2`).
- Report also shows `card_flux_ratio`, and when it differs, `die_flux_ratio` + `feas_flux_ratio` (feasibility min).
- Binding = tighter of card flux vs package power. Limits never retuned.

### Milestone path

- **Visual UI restyle:** no written user yes in NOTES/AGENTS/ui-audit → skipped (§293).
- Thinker∥Talker overlap: no sourced schedule → not invented. Prefer chip/report fix above.

## [0.349.0] — 2026-10-23

### Adversarial sidecar_bytes (MODEL §294)

- Moshi Mimi fp32 + depth bf16 and MOSS heads: `evaluate.fits` / memplan / PP-head / TP-full match `sidecar_bytes`; no `params()` double-count. Speech-card stacks stay off sidecar. DSE `fits` gate unchanged.

### NPU absolute search (empty) → thermal margin clarity

- Deep search again: HBM4 still vendor ≈40% only (no absolute pJ); SPHBM4 JEDEC interface only (Eliyan UMI/NuLink ≠ SPHBM4 PHY mm²); PCIe/CXL N5 controller area and HBM controller power still unpublished. **No invent.**
- **Mechanism:** design report adds package-power margin / headroom (parallel to flux 0.331) from the same `COOLING.pkg_W` 「来源」, `thermal_binding` (tighter of flux vs package), and front-point `pkg_within_tie` warnings — limits never retuned.

### Path / deferred

- NPU absolutes still pending. MOSS-Nano/Local; Thinker∥Talker pipeline. **No visual UI restyle** (ui-audit: user has not approved).

## [0.348.0] — 2026-10-23

### Adversarial sidecar_params (MODEL §293)

- No double-count with `params()` / param_err (sidecar still storage-only).
- PP: sidecar on head stage only; TP: full sidecar per rank (same pattern as `tts.stored_params`).
- MOSS audio heads: bf16 bytes correct. Moshi depth: bf16 correct; Mimi fp32 was missing from fits.

### `sidecar_bytes` — mixed-dtype same-card extras (mechanism)

- New `ModelSpec.sidecar_bytes`: exact resident bytes preferred by memplan over `sidecar_params * bits`.
- Moshi: `sidecar_bytes = MoshiSpec.stored_bytes` (depth bf16 + Mimi fp32). MOSS: heads × 2.
- `sidecar_params` / `sidecar_bits` kept for single-dtype introspection and backward path.

### Path choice / deferred

- **MOSS-Nano / Local:** still not clean (new arch / gated). **Thinker∥Talker overlap:** no sourced schedule. **NPU:** no new absolute. **No visual UI restyle.**

## [0.347.0] — 2026-10-23

### Adversarial MOSS-TTS Delay + Qwen3-Omni 0.346 fixes (MODEL §292)

- **MOSS first-audio (fixed).** Delay steps still run all `lm_heads` 「来源」; `first_audio` used `delay · TPOT` and understated. Now `delay · frame` (backbone + audio heads).
- **Qwen3-Omni sliding_window / text_projection:** rechecked across eval/speech — keys=72 for frames=1; projection 2048→2048→1024 once in prefill; dual-MoE/bf16/other omni·tts unchanged. No further code change.

### Same-card speech extras → `evaluate.fits` (mechanism)

- New `ModelSpec.sidecar_params` (+ `sidecar_bits`): resident weights outside `params_llm`, **not** in `params()` / param_err.
- `memplan` folds sidecar into head-stage `store_extra` (like `tts.stored_params`).
- Wired: MOSS-TTS audio lm_heads; Moshi depth (bf16). Mimi stays speech-KPI-only (fp32). Qwen3-Omni Talker stays speech-card (not sidecar).

### Path choice / deferred

- **MOSS-Nano:** public but new arch (GPT2 + local transformer, float32, `pytorch_model.bin`, separate tokenizer) — not cleanly additive; deferred.
- **MOSS-Local:** gated 401. **NPU:** no new public absolute. **No visual UI restyle.**

## [0.346.0] — 2026-10-23

### Adversarial Qwen3-Omni dual-MoE + code2wav (MODEL §291)

- **code2wav streaming KV (fixed).** `code2wav_ops(..., frames=1)` used `keys = min(frames, sliding_window) = 1`, understating Code2WavAttention's sliding window (config `sliding_window=72` / chunked_decode left_context). Now `keys = c2_window` when `frames==1`, else `min(frames, window)`. Shape / MoE routing / bf16 as released unchanged; other omni/tts builders untouched.
- **Talker setup projection (fixed).** Thinker→Talker `text_projection` ResizeMLP (2048→2048→1024) was missing from Talker prefill; added `talker_projection_ops` once per utterance into `_qwen3_omni_kpis` prefill. Not per-frame.
- **Joint Thinker/Talker scheduling:** existing `tot = first + max(audio*rtf, out_len*tpot)` kept — no new invent. NPU absolute not found this pass.

### MOSS-TTS Delay-8B (highest-value next)

- New builder `core/moss_tts.py` (kind `moss_tts`, domain tts). Gated on `model_type == moss_tts_delay` only (Nano/Local deferred). Qwen3 language_config backbone + `emb_ext[32]` + `lm_heads[1..32]` on the same card; 12.5 Hz; bf16 as released. External audio tokenizer/vocoder not in checkpoint → coverage partial.
- Catalog: `moss-tts` → OpenMOSS-Team/MOSS-TTS; SPEECH_MODELS / SPS_OUT 33→34. param_err 0.
- Speech KPI: frame = TPOT + audio heads; first ≈ ttft + 32·TPOT + frame. Waveform decode unmodeled.

### Freshness / UI

- Light freshness sweep (see NOTES). **No visual UI restyle.**

## [0.345.0] — 2026-10-23

### Added
- **Qwen3-Omni-30B-A3B** (`qwen3_omni_moe`): clean builder `core/qwen3_omni.py` — Thinker MoE (catalog) + Talker MoE + code predictor + code2wav from public HF config / safetensors headers (as-released bf16, sourced dims). AuT audio costed; ViT storage-only. Does not touch `omni_spec` / `qwen3_tts_spec`.

### Notes
- **NPU:** deep dig found no new shippable absolute (HBM4 all-in pJ/bit, SPHBM4 host PHY mm²/pJ, PCIe5/6×16+CXL N5 area, HBM controller power) — skip; keep prior Samsung HBM4 PHY.
- **UI:** no visual restyle (still waiting on user).
- Freshness light (A path only).

## [0.344.0] — 2026-10-23

### Changed
- **Adversarial of 0.343 functional UI** in `ui_smoke` / `ui_headless_smoke` + `test_core_344`: insight `.row` computed `display:flex` / `flex-wrap:wrap`; empty `.scroll > table` `padding-bottom: 0`; `#wl-kind` caption present on gen, cleared on LLM. Visual restyle still **not** applied.

### Pending / not applied
- **Qwen3-Omni-30B-A3B** (`qwen3_omni_moe`): public config dims confirmed (thinker MoE 128-exp top-8 h=2048 L48; talker MoE 128-exp top-6 h=1024 L20; code_predictor 5L/16 groups; **code2wav** 8L/1024/upsample 8·5·4·3 — not token2wav DiT/BigVGAN). `omni_spec` / `qwen3_tts_spec` correctly return None; no catalog entry. Needs a new omni mode — do not thin-wrap.
- **NPU:** still no shippable public absolute for PCIe/CXL controller mm², HBM controller at N7/N28, controller power, SPHBM4 host PHY (Samsung HC 2026 >8×4 mm already modelled for HBM4; blog %/HBM-side maps not used).
- OrcaCyber Nano: sha `1af1635`, gated auto, config 401.
- pocket-tts: sha `db04f0e`, gated auto, config 401.
- ui-audit visual harmony — waiting on the user (AGENTS §293).
- MOSS-TTS / Ming-3.0-audio-*.

## [0.343.0] — 2026-10-23

### Fixed
- **Functional UI only** (ported from `/workspace/ui-audit/ui.patch`, rebased onto f53ac7a; **not** the visual restyle):
  - `.row` / `.toolbar` flex layout so 设计洞察 controls sit in a wrapping row instead of one-per-line.
  - `paintDomain`: clear `#wl-kind` when the model has no workload presets (LLM / speech) so a prior gen/protein line does not linger.
  - `.scroll:has(> table:empty) { padding-bottom: 0 }` — empty result tables do not reserve scrollbar padding height.

### Pending / not applied
- **ui-audit visual harmony** (type scale, colors, segment buttons, KPI fonts, radii, chart sizing, etc.) — still waiting on the user; do not apply the rest of `ui.patch`.
- OrcaCyber Nano: sha `1af1635`, gated auto, config 401.
- pocket-tts: sha `db04f0e`, gated auto, config 401.
- NPU: no new public numbers this pass (skipped).
- Qwen3-Omni / MOSS-TTS / Ming-3.0-audio-*.

## [0.342.0] — 2026-10-23

### Changed
- Adversarial: `ming-omni-tts-0.5b` DiTAR (patch=4 / history=32 → 320 ms/step, Euler 10 × CFG 2, AudioVAE 44.1 kHz, flow0..flow9) held across eval / speech / SPS; 4× `s_per_step` vs Ming-3.0-tts-0.5B.
- Consistency audit (energy / thermal / TCO, 0.333–0.341): HBM3E 5.80 / HBM4 4.14 (=÷1.4) / LPDDR 3.23 presets, `MEM_PROXY` labels, `dram_pJ_field` selection, `energy_report` J∝pJ, `package_power` standby-vs-floor, LPDDR thermal knobs — all green; no code change.

### Pending / not applied
- **NPU design:** still no public numbers for PCIe/CXL controller area, HBM controller at N7/N28, controller power split, JEDEC SPHBM4 PHY mm², HBM4 absolute measured pJ/bit, or HBM NMC TOPS/W (Eliyan UMI / LPDDR5X-PIM / CXL MX1 marketing not used).
- OrcaCyber Nano: sha `1af1635` unchanged, gated auto, config 401.
- pocket-tts: sha `db04f0e` unchanged, gated auto, config 401.
- Qwen3-Omni / MOSS-TTS / Ming-3.0-audio-* (new arch).
- **UI patch not applied**.

## [0.341.0] — 2026-10-22

### Added
- `ming-omni-tts-0.5b` (inclusionAI/Ming-omni-tts-0.5B): same dense+DiTAR `tts_spec` as Ming-3.0-tts-0.5B; patch=4 / history=32 → 320 ms audio/step 「来源」; SPEECH 31 → 32.

### Changed
- Adversarial: Ming-3.0-tts-0.5B DiTAR (Euler 10 × CFG 2, AudioVAE 44.1 kHz, flow0..flow9) held across eval / speech / SPS.

### Pending / not applied
- **Qwen3-Omni** (`qwen3_omni_moe`): MoE thinker (128 experts) + MoE talker + `code_predictor` + `code2wav` (not token2wav) — new architecture, not a minimal omni/tts extension.
- **MOSS-TTS** (Nano / v1.5 / Local): `moss_tts_*` global+local GPT2 / delay patterns + separate audio tokenizer — new builder family.
- OrcaCyber Nano / pocket-tts: sha unchanged, gated auto, config 401.
- Ming-3.0-audio-{tiny,flash}: `bailing_moe_v3_audio` — new ASR arch.
- **UI patch not applied**.

## [0.340.0] — 2026-10-22

### Added
- `ming-3.0-tts-0.5b` (inclusionAI/Ming-3.0-tts-0.5B): dense Qwen2.5-0.5B + DiTAR (`ditar_config`) via existing `tts_spec` / TTS loop (FlowLoss Euler 10 × CFG 2 + Aggregator + AudioVAE) 「来源」 github inclusionAI/Ming-3.0-tts; SPEECH 30 → 31.
- summarize role map for dense+DiTAR: `flowloss` / `stop_head` / `spk_head` → `tts`.

### Changed
- Adversarial: 0.339 Omni-3B / Step-Base/Think / moshika + Qwen3-TTS prefill(2)+15 decode held across eval / speech / SPS.

### Pending / not applied
- pocket-tts: gated (config 401).
- OrcaCyber Nano 1.0: sha `1af1635` unchanged, gated auto, config 401.
- MOSS-TTS / Qwen3-Omni: new architectures — keep pending.
- NPU: no new public numbers.
- **UI patch not applied** (user still deciding).

## [0.339.0] — 2026-10-22

### Added
- Speech freshness (existing builders): `qwen2.5-omni-3b`, `step-audio-2-mini-base`, `step-audio-2-mini-think`, `moshika-7b` (SPEECH 26 → 30).
- Releases: Qwen2.5-Omni-3B, Step-Audio-2-mini-Base/Think (+ token2wav sidecars), moshika-pytorch-bf16 (`fetch_pt_release.py --moshi`).

### Changed
- Qwen3-TTS code predictor: HF `generate` = **prefill over 2 seed embeds** then `cp_steps` decode steps (was 15× decode(1) with growing KV — undercounted first step) 「来源」 modeling_qwen3_tts.

### Pending / not applied
- OrcaCyber Nano 1.0: sha `1af1635` unchanged, gated auto, config 401 (drafter GGUF only).
- NPU: no new public numbers (PCIe/CXL controller area, N7/N28 controller, controller power, SPHBM4 PHY, NMC) — skipped.
- Qwen3-Omni / MOSS-TTS / Ming-3.0-tts-0.5B / pocket-tts: need new architecture modes — keep pending.
- **UI patch not applied** (user still deciding).

## [0.338.0] — 2026-10-22

- **Qwen3-TTS builder** (kind `qwen3_tts`, domain tts). Discrete multi-codebook LM at 12.5 Hz: Talker (catalog Qwen3) draws the first codebook; code predictor AR-generates the remaining 15; speech_tokenizer decode (fp32) → 24 kHz. Reuses Moshi-style frame KPIs. **Released:** `qwen3-tts-{0.6b,1.7b}-{base,custom-voice}` + `qwen3-tts-1.7b-voice-design` (param_err 0). SPEECH_MODELS / SPS_OUT 21→26.
- **Freshness:** OrcaCyber Nano still gated (sha `1af1635`, config 401) — pending.
- **Adversarial:** Carbon-X eval/shapes/dtype OK (stock Qwen3, dna_config metadata only). OmniASR w2v as-released fp32 weight bits held across eval/speech; other ASR audio still bf16.

## [0.337.0] — 2026-10-22

- **Hotfix (0.336 suite on non-box hosts):** `test_core_336::test_docs_and_no_ui_patch` no longer requires `/workspace/ui-audit/ui.patch` (authoring-box only). It asserts the 0.336 CHANGELOG section documents that the UI patch was not applied.
- **OmniASR / w2v adversarial (0.336 follow-up).** Shapes, catalog dims, eval speech, DSE, and static paths check out for the four bezzam OmniASR LLM v2 ports. **As-released dtype:** audio-tower safetensors are fp32; `_w2v_ops` previously hard-coded bf16 weight bits (same as other ASR audio paths) and understated weight traffic. `AudioSpec` now carries `w_bits`/`w_fmt` (default bf16); `_omniasr_audio_spec` sets fp32 when `dtype`/`torch_dtype` is float32; `_w2v_ops` uses those fields. Other audio kinds unchanged.
- **Carbon-X.** `HuggingFaceBio/Carbon-X` — public `Qwen3ForCausalLM` + `dna_config` 6-mer tags, bf16 as released, 14.822B params (param_err 0). Added as `carbon-x` (same family as carbon-8b/3b/500m). Releases 197→198, listed+unlisted 205→206.
- **Speech pick (not added).** Qwen3-TTS (`qwen3_tts`: talker + codebooks + tokenizer-12Hz) has newly public usable config but needs a new TTS builder — documented pending (same posture as OmniASR CTC). No other TTS/omni with newly public usable dims this pass.
- **Freshness.** OrcaCyber Nano still gated (sha `1af1635`, config 401) — pending. Carbon-X added above.

## [0.336.0]

### OmniASR LLM v2 — Wav2Vec2-style audio builder in the asr domain (MODEL §281)
- **`omniasr_audio` builder** (kind `w2v`): 7× Conv1d feature extractor + grouped positional conv + bidirectional Transformer + `multi_modal_projector`, dims from public config / safetensors (bezzam/omniasr-llm-*-v2 HF ports of facebook/omniASR; official facebook repos ship `.pt` only).
- **Released:** `omniasr-llm-300m-v2` / `1b` / `3b` / `7b` (shared Llama text trunk 1.319 B fp32; audio tower 0.32 / 0.97 / 3.07 / 6.49 B). Frame rate 50 Hz; 10 s → 499 prompt audio tokens.
- **Speech insight / UI:** added to `SPEECH_MODELS` and `SPS_ASR` (17 → 21 models).
- **CTC siblings pending:** `omniasr_ctc` encoder builds, but no AR language backbone for the asr domain decode path — not catalogued.
- **Freshness:** OrcaCyber Nano sha `1af1635` unchanged, config 401 → still pending. Carbon-X (Qwen3 stock) noted, not added this pass.

### 0.335 LPDDR adversarial (light)
- Device-side 3.23 / PHY 1.3 / HBM 5.80 unchanged; `hbm_standby_W` / `hbm_device_static_W` still feed LPDDR through eval. Decode J ×0.970 vs Grace-4.0 default reconfirmed. No default change.

### Other
- **UI patch not applied** (`/workspace/ui-audit/ui.patch` present; user still deciding).

## [0.335.0]

### LPDDR device energy aligned with HBM 0.333 idle/active split (MODEL §280)
- **No Grace-style measured idle/active split exists** (subsystem peak 16 W / 500 GB/s only). Device-side active is taken from the same Micron Y52Q sheet already used for LPDDR standby.
- **`pJ_bit_dram_lpddr` 4.0 → 3.23** 「推导」 = Table 16 WT @ 9600 Mb/s x16: (IDD4R − IDD2N) × typ rails ÷ (9600 Mb/s × 16). IDD4RQ is datasheet typical 「近似」. Same structure as HBM `(total − idle) / BW`.
- **`pJ_bit_dram_lpddr6` 3.31 → 2.67** (÷ 1.21, Samsung ISSCC 2026 21 % claim kept).
- **Grace 4.0** kept as an all-in-at-peak cross-check in the source text (≈ 3.23 + PHY 1.3); not used as the per-bit default.
- **Impact** (qwen3-8b 1P tpuv4i_7nm): decode b64 LPDDR5X J/token ×0.970, $/M ×0.996; HBM path unchanged.
- **Test re-pins** (mechanical): test_core_311 J/token edge delta −10.2 % → −11.0 % / 63.3 → 58.1 mJ; test_core_306 wan2.1-1.3b energy-tie SRAM back to lat@16 (was J_tok≈19 under 4.0).

### 0.334 thermal knobs apply to LPDDR (consistency)
- `hbm_standby_W` / `hbm_device_static_W` now return LPDDR standby / full device static (were 0). Clocked IDD2N does not offset the area floor; full device static is out of `pkg_W_hot`. Cap / TCO `pkg_W` path unchanged in structure.

### Freshness
- Same HF new-row set as 0.333/0.334. OrcaCyber Nano: sha `1af1635` unchanged, gated auto, config 401 → pending.
- **OmniASR** (bezzam/omniasr-*-v2, 2026-10-22): public `config.json` + weights, but `model_type: omniasr` needs a new audio-encoder builder (not Whisper/AuT/SAN-M) → pending, not added.
- **UI patch not applied.**


## [0.334.0]

### Adversarial review of 0.333 HBM energy + package idle floor (MODEL §279)
- **No double-count (confirmed).** At the measured H200 streaming bandwidth, `hbm_static_W` (idle) + `pJ_bit_dram_hbm` × bits ≈ measured total within ~3 % — by construction of the idle / active split. Host PHY stays a separate line. Activation is inside the 5.80 pJ/bit term; there is no second activation charge.
- **Hot leakage used the wrong power (fixed).** 0.333 excluded only clocked `standby_W` from `pkg_W_hot`, so refresh (off-die, ~3 W on 8 × HBM3E) still rode the die-temperature factor `_hot_k()`. Floor offset correctly kept refresh (it overlaps the idle-floor measurement). One knob cannot serve both.
  - New `powercap.hbm_device_static_W` = full HBM device static (standby + refresh).
  - `package_power(..., device_static_W)` / `thermal_report(..., device_static_W)`: hot excludes `device_static_W`; floor still excludes only `standby_W`.
  - Plumbed through api + energy DSE rows. Cap / TCO `pkg_W` unchanged (hot is a margin flag only).
  - Example: 8 × HBM3E prefill, `pkg_W_hot` 874.5 → 866.7 W (−7.8 W); `pkg_W` identical.
- **Stale HBM4 2.46 labels (fixed).** Preset value has been 4.14 since 0.333; `MEM_PROXY["HBM4"]`, the energy.py 0.302 comment, and the web HBM / HBM4 field tooltips still said 2.46 / 3.44. Updated to 5.80 / 4.14. Not a styling change.
- **HBM4 ÷ 1.4 source reconfirmed.** Samsung HBM4 product page / 2026 shipping news and SK hynix (~40 % / 40%+a) still give only the relative claim — no public absolute pJ/bit. Keep ÷ 1.4; paper's 0.94–1.00× stays a sensitivity.
- **TCO / J/token deltas are mechanical (confirmed).** Overriding only `pJ_bit_dram_hbm` to 3.44 moves J by the DRAM share; standby lands in `hbm_static` / package idle. No fitting.
- **Highest-value remaining sourced item: skipped.** No new public absolute HBM4 (or other pending) number found beyond the vendor ratio already used.
- **Freshness:** same 6 rows as 0.333. OrcaCyber Nano retry: sha `1af1635` unchanged, gated auto, config 401, public siblings = drafter/mix GGUF only → still pending.
- **UI patch not applied.**


## [0.333.0]

### HBM energy reconciliation against Olgun et al., arXiv 2610.14244 (MICRO 2026) — MODEL §278
- **Split, not fitted.** The paper's released H200 data (github.com/CMU-SAFARI/HBM-Power, `case_studies/hbm3e_validation/data`; nvidia-smi GPU memory power of three H200s, 6 × HBM3E 8-high 24 Gb = 96 64-bit channels) gives idle (NOP + refresh) and total power under a streaming random-data read separately. They map one-to-one onto the model's static term and per-bit term. The ≈ 6.86 pJ/bit "all-in" figure (total ÷ bandwidth) is not used as a per-bit value.
- **`pJ_bit_dram_hbm` 3.44 → 5.80** 「推导」 = mean (total − idle) ÷ read bandwidth = 5.87 / 5.73 / 5.81 pJ/bit. It is device-side dynamic energy: activation at the streaming row locality, read data movement and I/O, with random data (const-0 data 5.09 on the same H200).
  - The old value was SK hynix's HBM2E 4.3 × 0.8 vendor figure.
  - Write ≈ 4.2 pJ/bit (const-0) is not split out 「近似」.
- **`pJ_bit_dram_hbm4` 2.457 → 4.144** (HBM3E ÷ 1.4, the vendor ratio is kept). The paper's own voltage-only projection (0.94–1.00× HBM3E per bit) is recorded as a sensitivity in the source text.
- **HBM standby: per 64-bit channel instead of per die.** The preset changes from `mW_hbm_die_standby` 48 mW/die (DRAMsim3 HBM2 IDD2N × 1.2 V) to the new field **`mW_hbm_ch_standby` = 376.8 mW/channel** 「推导」 = (38.42 W measured idle − 2.25 W refresh from this preset's 1.95 mW/Gb) ÷ 96.
  - Standby is set by the clocked channel interface, not by die count (the paper's Finding 6: it scales with data rate). So it is height-independent; HBM4 has 32 channels per stack, which gives 12.06 W per stack. This is within 2 % of the paper's own HBM4 idle extrapolation of 11.84 W (cross-check, not fitted).
  - `mW_hbm_die_standby` stays as a user input; the two fields sum when both are set.
  - Refresh is unchanged at 1.95 mW/Gb.
  - HBM static per card: hbm3e_4s 3.03 → 25.6 W; hbm3e_6s 4.55 → 38.4 W; hbm3e_8s_12h 9.10 → 52.7 W; hbm4_8s_12h 9.10 → 100.9 W.
- **Mechanism fix: clocked HBM standby no longer offsets the area-scaled idle floor.** `thermal.package_power(..., standby_W)`; new `powercap.hbm_standby_W(e)`, plumbed through `cap_power_fn` / `card_W_for_cap` / `thermal_report` / energy power-cap / TCO / energy DSE rows / api.
  - The floor (H100 71.8 W ÷ 814 mm², no CUDA context) is measured at idle clocks with the HBM in self-refresh. Refresh and SRAM / NoC leakage overlap it; clocked standby does not, and it is not die leakage either, so it is also excluded from the hot-leakage term.
  - ≤ 0.332 the standby was 48 mW/die, so the overlap was small. At 377 mW/channel the stacks filled the floor, and die area stopped affecting package power: test_core_327's custom-base-die decision flipped. With the fix it holds as in 0.332.
- **Impact** (qwen3-8b, 1P, tpuv4i_7nm, n5_cowos_s, no cap; old → new):

  | case | J/token | $/M | package W |
  |---|---|---|---|
  | decode b64 hbm3e_4s | 0.05411 → 0.07566 (×1.398) | 0.008388 → 0.009372 (×1.117) | 302.6 → 404.2 |
  | decode b1 hbm3e_4s | ×1.456 | 0.14938 → 0.16756 (×1.122) | |
  | prefill b8 hbm3e_4s | ×1.062 | ×1.032 | |
  | decode b64 hbm4_8s_12h | ×1.556 | ×1.100 | |
  | decode b64 hbm3e_6s (H200-like) | ×1.456 | ×1.113 | |

  - Energy share of $/M for decode b64 hbm3e_4s goes from 0.349 to 0.417.
  - Under a 400 W card cap with 8 × HBM4 12-high (test_core_324), the DVFS point drops from s 0.778 to 0.384 and $/M doubles (0.012257 → 0.024810).
- **Test re-pins (old → new in each file):**
  - 186: HBM per-bit pin; the HBM / LPDDR per-bit ratio replaces `jl > jh` (LPDDR 4.0 is an all-in subsystem figure, see §278.6).
  - 187, 211, 216 (`_t` keeps the per-die field at 48 mW), 217, 227, 228, 229, 233, 257, 302, 306, 307, 310, 313, 315 (now passes `standby_W`), 316 (card-scope capped clocks; air_oam front [1, 0, 15] → [4, 16]; package front [12] → [3, 0, 1]; 9 → 14 capped), 318 (carrier-yield flip leaves its range), 319, 324, 328, 330 (thin-margin rows 0 / 15 go over air, front [4, 16]), 331 (near-tie {0, 15} → {16}).
  - 317 and 327 are unchanged.
- **New `tests/test_core_333.py`** (7 tests).

### Freshness
- Same 6 rows as 0.332. OrcaCyber Nano retry: sha 1af1635 unchanged, config still 401 (gated; public metadata only).


## [0.332.0]

### Load-robust runtime guards (test infrastructure; no model default changed)
- **New `tests/workmeter.py`**: a deterministic work meter (stdlib `sys.monitoring`). Work = Python function starts in `accel_dse` code + calls issued from `accel_dse` code. Code outside the package is switched off per location (`DISABLE`).
  - The same request gives the same count whatever the load. Checks: disagg vllm_v0 148,629,143 calls in two runs; 210 under 8 busy-loop hogs on 8 cores gave 15/15 labels identical to the quiet run.
  - Overhead ≈ 1.6–1.8× CPU inside guarded blocks (start+call; start-only 1.2×, start+jump 2.3×).
- **`guard(bound_s, label)`** asserts calls ≤ `budget(bound_s, label)` = bound × the label's own quiet-box rate (calls / CPU s), i.e. quiet CPU ≤ bound: the old threshold.
  - Rates were measured 2026-10-22 by the new `scripts/workmeter_calibrate.py` and rounded down to 2 significant figures, so budgets are never looser.
  - `RATE_FLOOR` = 2.3 M/s (the lowest) applies to blocks under 0.05 s CPU or unlisted labels.
  - One global rate does not work: rates span 2.3–18.8 M/s.
- **Converted guards** (old → new):
  - 099: freeze-aware wall < 20 s → work ≤ 20 s-equivalent.
  - 209: freeze-aware wall < 2 s → work ≤ 2 s-equivalent.
  - 210: CPU < 20 / 80 s plus contention-scaled freeze-aware wall (0.304 / 0.316) → work ≤ 20 / 80 s-equivalent.
  - 237: CPU < 45 / 90 s, ×6 under -j → work ≤ 45 / 90 s-equivalent at any -j. This is stricter in parallel runs. The kernel work counts (1.5×) are kept.
  - 238: CPU < 60 s → work ≤ 60 s-equivalent.
  - 289: CPU < 20 s (60 s under -j) → work ≤ 20 s-equivalent at any -j. Kernel counts are kept.
  - A sleeping or blocking hang is caught by the runner's per-file timeout (1800 s).
- **088**: the burst / stalled-client checks are now server-side and deterministic. A `Handler` subclass records every 408 with its socket timeout; the burst fires none, and the stalled client gets exactly `[(408, 2)]`. The wall inference `dt < 20 + stall` was removed (a gross `dt < old_t + stall` bound stays), and the hung-server bound uses the freeze clock.
- **304**: the 210 guard check now uses deterministic work (the same burn counts equal after warm-up; an overrun trips; a cheap call passes).
- **311**: `_Steps(time.monotonic())`.
- **312** meta check: 238 must use `guard(60, …)` and no `process_time` (was: `process_time` present).
- New `tests/test_core_332.py` (8 tests).

### Decision timers monotonic
- `scripts/static_parity_check.py`, `ui_headless_smoke.py` and `ui_static_smoke.py` use `time.monotonic()` instead of `time.time()` for their timeout decisions; a wall-clock step was seen in 0.304.
- `core/validation.py` elapsed_s uses `time.monotonic()`.
- Print-only durations (fetch_hf_release, fuzz_sizing, ui_smoke, v4_serving) are left as they are.

### Audit (no change needed)
- No other wall/CPU mixing in the package: freeze_clock was fixed in 0.331. The Pyodide worker `ms` and shim `wallMs` are both `performance.now()` and labelled separately; the app.js progress tracker is `performance.now()`; the api pool has no timing.
- Gross subprocess / socket timeouts (087, 090, 129, 130, 184, 309, 310) have ≥ 10× slack and stay.


## [0.331.0]

### One repo, several release files; PixText-Teacher
- **Release key separate from hf_id.** New `catalog.RELEASE_KEYS` (model id → `"<repo>:<subfolder>"`) and `release_key()`.
  - The file is `releases/<repo /→__>_<subfolder /→__>.json`, using the existing `model.release_path`, which already mapped ":" → "_".
  - Every entry now carries `release`. Only the listed models differ from their hf_id, so all existing ids, files, aliases, labels, links, providers and the freshness org list are unchanged (regression test).
  - The hf_id of a keyed model stays the public repo (links, `provider_of`, freshness). Its alias is the release key, not the repo, so `ATH-MaaS/PixText` still resolves to MeanFlow.
  - `get_model` → `from_domain_release(..., release=)`.
- **Fetch / summarize.**
  - `scripts/fetch_hf_release.py` spec `repo:sub!` gives the subfolder its own file (`own_key`, `release_stem` = `release_path` stem); the cache dir and `source.json` carry `release_key`.
  - `summarize_release.py` reads it back (`_cache_key`) and accepts `repo:sub` items.
  - `release_repos.txt`: `ATH-MaaS/PixText:models/PixText-Teacher!`.
- **PixText-Teacher added** (`pixtext-teacher`, label "PixText · Teacher", release `ATH-MaaS__PixText_models__PixText-Teacher.json`).
  - Public config + safetensors header: 461 BF16 tensors, 951,979,840 params, 1,903,959,680 B. Matches the builder exactly, coverage full (no interval embedder).
  - Workload: the same reference case as MeanFlow (4096 + 392 tokens), 25 Euler steps, guidance 3.3 on (0.1, 1.0) → 25 + 22 = 47 forwards 「推导」 (sampling/flow.py `low < k/25 < high`).
  - Result: 540.45 TFLOP per request; default chip 6.056 s per request.

### Partial guidance (general workload field, not PixText-specific)
- **`NativeWorkload.cfg_interval = (lo, hi)`**, default `()` = every step guided, the old behaviour.
  - `domain.guided_steps(steps, interval)` counts steps k with lo < k/steps < hi (strict, as the PixText samplers do 「来源」; the guidance-interval idea is Kynkäänniemi et al. 2024).
  - `ResolvedWorkload.guided` (−1 = all).
  - The two-forward CFG path (`_cfg_asym`, 0.258) runs the negative branch on `guided` steps only: TFLOP = cond × steps + uncond × guided, and denoise = steps × tick + guided × tick_u. The asym dict gains `guided_steps` / `step_ms` / `guided_step_ms`.
  - Energy counts the negative-branch stages `guided_steps` times.
  - `workload.cfg_branches = same` (packed) and the PP > 1 fallback do not model the interval: they warn and count every step.
- **PixText no-glyph negative branch at its own shape.** `build_pixtext` sets `uncond = (prefix − crop tokens, ())`: 4292 tokens (main + items, no glyphs) vs 4488, through the existing asym path.
- **Defaults old → new (mechanism: release sampler code + README):**
  - pixtext-meanflow workload steps 7, cfg 1 → steps 4, cfg 2, cfg_interval (0.1, 1.0) (still 7 forwards).
  - Per forward: cond 11.84 TFLOP; uncond 11.84 → 11.19 TFLOP.
  - Per request: TFLOP 82.879 → 80.916 (−2.37%); default-chip latency 0.935 → 0.907 s.
  - MOSS-Image unchanged (4953.804 TFLOP per request, checked against 0.330 main).
- **Routing attention mask: not modelled (still dense 「近似」, +9.3% pairs on the conditional branch).**
  - Glyph rows attend only to their own item's slot, so cost depends on Σ nᵢ² over items. No workload field carries the per-item sizes, so modelling it would need a PixText-specific rule.
  - The no-glyph branch is fully dense under the mask, so it is exact now.

### Design report: front thermal sensitivity (no limit change)
- **Thermal section:**
  - new items `flux_margin_W_mm2` (limit − flux, 「推导」) and `flux_headroom` (limit / flux − 1 = the power increase at fixed area that reaches the limit);
  - new `front_thermal` block: every perf/W front point with flux, limit, margin in W/mm² and %, headroom and a within-tie flag;
  - near-tie warnings of kind `front_thermal`;
  - a Markdown table.
  - `compare_report` treats both new items as higher-is-better.
- **air_oam card front (qwen3-8b prefill tp):**
  - row 0: 0.8576 W/mm², margin 0.0024 W/mm² (0.28%);
  - row 15: 0.8581 W/mm², margin 0.0019 W/mm² (0.23%);
  - both are within the 1% tie and flagged. Row 1: 0.8509, 1.06%, not flagged.
- `COOLING["air_oam"]` stays 0.86 W/mm².

### Suite timing: freeze_clock never excuses executed CPU; test_core_316 deterministic
- **Symptom.** On msa1's 0.330 full suite (-j 8), `test_core_316::test_freeze_clock_excuses_only_non_execution` failed once (line 155, GIL case: `frozen == 0 and active == wall`); it passed on the box.
- **Mechanism (reproduced).** Under 8× CPU oversubscription (64 busy-loops on 8 cores) the 2 s GIL call stretches to ≈ 10 s of wall at ≈ 14% CPU, below the 20% stall threshold, so the sampler correctly sees a stall: the process could not run.
  - The real flaw: ≤ 0.330 excused the whole `gap − period`, including the CPU the process *did* execute inside the gap.
  - Result: active 0.24 s for 1.4 s of executed CPU (old) → active 1.61 s ≥ 1.41 s CPU (new).
  - A gap above the threshold during which the process ran ≥ 20% is not counted (unchanged, checked).
- **Fix, `scripts/freeze_clock.py` (old → new).** Excused per stall: `gap − period` → `gap − max(period, CPU in the gap)`, so active time ≥ process CPU time under any load. This matches the module's stated rule ("only time in which the process did not execute at all is excused").
  - Thresholds unchanged (0.05 s / 0.5 s / 20%, 「假设」).
  - Wall / CPU clocks are injectable; the decision moved to a pure `observe(w, c)`; `start=False` runs no thread; `t0` / `c0` baseline.
  - The self-check prints `cpu`, `act_all`, `cpu_all`.
- **test_core_316 (no check weakened).**
  - The three cases (SIGSTOP 2 s excused minus one period; sleeping hang charged in full; 2 s GIL call charged in full) now run on a fake clock with exact equalities.
  - New threshold / fix test: 0.49 s not a stall; 0.5 s is; exactly 20% CPU is running; 2 s at 15% CPU excuses 1.7 not 1.95; plus a 200-script random invariant active ≥ CPU and ≤ wall.
  - The real-subprocess check keeps the SIGSTOP and GIL runs with load-independent assertions: frozen ≥ 1.5 s for the stop, lifetime active ≥ lifetime CPU, active ≤ wall.
  - Wall-time equalities are no longer asserted on a real host. Starvation is a legitimate stall, so they cannot hold on a loaded host.

### Tests
- New `tests/test_core_331.py` (9).
- test_core_316: freeze-clock test split into 3 (deterministic cases, thresholds / fix, real-process invariants): 7 → 9 tests.
- test_core_330: MeanFlow workload steps 7, cfg 1 → 4, cfg 2 (+ cfg_interval); seqs 1 → 2; FLOP check now 4 × f(4488) + 3 × f(4292); the "no teacher" assertion → teacher present.
- test_core_062_1: release files 192 → 193; listed + unlisted 200 → 201 (+ pixtext-teacher).

### Freshness (since 2026-10-22, 110 orgs)
- Same list as 0.330: PixText (both checkpoints now catalogued), unsloth MOSS-Image GGUF / FP8 (third-party quants), TextVerifier-CJK-CLIP (classifier).
- **OrcaCyber Nano 1.0 still pending.** Same sha 1af1635 (lastModified 12:16 JST); public GGUF metadata is the dflash drafter's; config 401; no new orcarouter repos.


## [0.330.0]

### Adversarial review of the 0.329 verdict flips (no default value changes)
- **HBM3E = HBM3 bump map: 「假设」 → 「推导」 (value unchanged, 24 mm² per stack).**
  - JEDEC publishes no separate HBM3E standard (HBM3 = JESD238).
  - Synopsys' HBM3 PHY IP supports "JEDEC standard HBM3 DRAMs with data rates up to 9600 Mbps" on one 1024-bit, 16-channel micro-bump interface, i.e. HBM3E speeds on the HBM3 bump map 「来源」.
  - H200 (HBM3e) is Hopper; the GH100 whitepaper lists "6 HBM3 or HBM2e stacks, 12 512-bit Memory Controllers" 「来源」, and H200 uses the same die 「二手」.
  - AMD CDNA 3: "each stack of HBM memory is associated with 16 parallel channels", and MI300X HBM3 → MI325X HBM3E 「来源」.
  - No public measured HBM3E PHY area exists: B200 / MI325X die shots are unannotated, and Gaudi 3 is HBM2e.
  - Source text of `mm2_per_dq_hbm3` updated in all three presets.
- **Channels 16 / 32 「来源」.** The JEDEC JESD270-4 release (2025-04-16) "doubles the number of independent channels per stack, from 16 channels (HBM3) to 32"; AMD CDNA 3 confirms 16 per HBM3 / 3E stack.
- **Controller residual recorded (value unchanged).** Samsung's 5–10% quote (slide 9) uses a generic 6-stack xPU with no generation printed, in an HBM4 custom-base-die context 「推导」. If it meant an HBM3E xPU, HBM3E would be 7.15 mm² per stack instead of 3.575: range 3.575–7.15.
- **No double count 「推导」.**
  - Every die grows by exactly stacks/die × (8 + 3.575) / 0.77: +120.26 / +60.13 / +30.07 / +15.03 mm² for 8 / 4 / 2 / 1 stacks per die (checked mono, 2-die and 4-die at MAC×1 / ×2).
  - The shoreline uses stack edge lengths only (no PHY area).
  - D2D is only on multi-die and custom HBM4.
  - The DRAM port sits outside the 21% unknown (0.328). The ÷ 0.77 markup on the PHY is the proportional uncore convention, not a second PHY 「假设」.
- **Each 0.329 flip re-derived from the area increment alone:**
  - **Cost:** at D0 0.1, mono +$238.7 vs 4-die +$115.2. Yield is super-linear in area, so the crossover moves to ≈ 0.07.
  - **I/O:** tp8 / 900 GB/s needs the same lanes; only the edge grew. The new test compares an old-semantics budget with the new one on the same scenario.
  - **Air thermal:** 0.86 W/mm² is 700 W ÷ 814 mm² of the *whole* GH100 die (PHYs included), the same definition as `full_area`. Rows 0 / 15 of the air_oam card front sit at 0.858 vs 0.86 (0.2% margin): fragile, recorded.
  - **tco_sens:** thresholds follow the narrower mono / split margin.
  - **eval = DSE = report = static:** verified for 8 × HBM3E 774.462, 4 × HBM3E 631.215 and 8 × HBM4 894.722 mm² (static runs the same Python via pyodide).
- **Corrections to the 0.329 record (old → new):**
  - **MAC×2 TCO memory 4s → 8s mechanism.** The 0.329 test comment said "the 4-stack's thr edge over cost no longer wins". In fact the MAC×2 4-stack mono die grows 838.9 → 899.0 mm² and crosses the 858 mm² reticle, so only its split rows remain: 4s $/M 0.01164 → 0.01966 vs 8s 0.01641 → 0.01681. test_core_317 comment fixed.
  - **Unreported MAC×1 change.** The tp8 / 400 / D0 0.1 MAC×1 TCO memory moved 8s → 6s (0.00949 vs 0.00952, inside the 1% tie). The 0.329 CHANGELOG did not list it.
  - **PixText file sizes.** The 0.329 freshness note gave one size for both files. Actual: MeanFlow 1,907,943,360 B, Teacher 1,904,005,664 B.

### PixText-MeanFlow (ATH-MaaS/PixText, Apache-2.0, 2026-10-22 15:41 JST) in the gen domain
- **Release data.** The release has its own config format (`{"format_version", "model": {...}, "variant"}`, read via `summarize_release.domain_of`), plus the public safetensors header (HTTP Range).
  - 465 tensors, all BF16, 953,948,480 params, 1,907,896,960 B = the header, exactly.
  - Fetched as `ATH-MaaS/PixText:models/PixText-MeanFlow`. New role patterns: `^timestep_embedder.` / `^interval_embedder.` → cond; `^patch_projection.` / `^glyph_projection.` → io.
- **New builder `build_pixtext` (GroupJiT, from github.com/ATH-MaaS/PixText transformer.py 「来源」):**
  - Pixel space: 16×16×3 patches, `patch_projection` 768 → 1280; `glyph_projection` 256 → 1280. No VAE, no text encoder (README).
  - 32 blocks, each with: RMSNorm; fused qkv (+bias); RMS QK-norm; 3-axis RoPE 12 + 34 + 34 = head dim 80; proj; SwiGLU w12 1280 → 6826 and w3 3413 → 1280.
  - Block AdaLN (d → 6d), the timestep embedder and the interval embedder run **per token** (`conditioning_ids = arange(tokens)`). The interval embedder is called twice per token (f(r) − f(0)): the second call is compute, not storage.
  - Final layer: AdaLN per image + linear on the RGB rows.
- **Parameter check:** 953,948,480 = the header, coverage full.
- **Default workload = the release's own reference case 「来源」:**
  - main image 1024² (`main_target_area` 1048576), patch 16 → 4096 tokens;
  - edit items + glyphs = pass 1 of examples/pipeline_case: 12 crops, 196 tokens × 2 = 392 prefix tokens;
  - 7 forwards per request ("four MeanFlow updates use seven model forwards": 4 steps + 3 guidance-branch forwards inside guidance_interval (0.1, 1.0)), counted as steps 7, cfg 1.
- **Approximations 「近似」, sized on the reference case:**
  - Dense joint attention instead of the routing mask: 18,430,912 allowed pairs vs 4488² = 20,142,144, +9.3% pairs ≈ +2.6% FLOP.
  - Guidance forwards are counted at the conditional shape (4488 vs 4292 tokens) ≈ +1.9% FLOP.
  - Item RGB rows are counted with glyph_projection rows ≈ 0.005%.
- **Check:** 11.84 TFLOP per forward = the hand count; 82.9 TFLOP per request.
- **Catalog:** new provider ATH-MaaS（PixText） and family PixText. `_finish` gains `standalone=` / `fwd_note=` (opt-in; no other model affected).
- **Teacher pending.** PixText-Teacher (25 steps → 47 forwards 「推导」, no interval embedder, 951,979,840 params) has a fully read header, but one release file per repo (`release_path`) keeps it out this round.

### Tests
- New `tests/test_core_330.py` (9).
- test_core_317: comment corrected (mechanism above); assertions unchanged.
- test_core_062_1: release-file count 191 → 192 (llm 165 unchanged) and listed + unlisted 199 → 200 (+ pixtext-meanflow).

### Freshness (since 2026-10-22, 110 orgs)
- PixText added (above).
- unsloth MOSS-Image GGUF / FP8: third-party quants, not added.
- ATH-MaaS/TextVerifier-CJK-CLIP: classifier, out of scope.
- **OrcaCyber Nano 1.0 still pending.** lastModified 2026-10-22 12:16 JST (sha 1af1635); the public GGUF metadata is still the dflash drafter's (1.92 B); the main files are mixed quants; config.json 401.


## [0.329.0]

### Per-generation HBM host interface: HBM3 / HBM3E get their own PHY area and a channel-scaled controller
- **Mechanism.** Samsung's Hot Chips 2026 tutorial (S. Han, slide 11, read from the ServeTheHome slide image 「来源」) gives the HBM PHY footprint by generation: HBM2 6 × 1.2 mm, **HBM3 8 × 3 mm** (die photo cited "VLSI2022"), sHBM4 8 × 4 mm, cHBM4 D2D 8.5 × 1.5 mm, sHBM5 9.5 × 1.7 mm, with the footnote "PHY area is based on the bump map". Slide 10 draws the xPU-side HBM PHY mirroring the base-die one. A bump-map-limited footprint is set by the microbump pitch and the DQ / CA count, not by the logic node, so the host-side PHY equals the memory-side one, also for HBM3 (1x-nm DRAM base die) 「推导」. HBM3E keeps the HBM3 interface (1024 DQ, 16 channels) and is taken as the HBM3 bump map 「假设」.
- **PHY (old → new): HBM3 / HBM3E 1024 DQ × 0.015625 (the HBM4 per-DQ value) = 16 mm² → new field `mm2_per_dq_hbm3` = 8 × 3 / 1024 = 0.0234375 mm²/DQ = 24 mm² per stack** 「来源/推导」, in all three density presets (the PHY is bump-map-limited, so node-independent, like `mm2_per_dq_hbm`). HBM4 / 4E stay at 2048 DQ × 0.015625 = 32 mm² (8 × 4 mm). A budget without `mm2_per_dq_hbm3` falls back to `mm2_per_dq_hbm` (the ≤ 0.328 behaviour).
- **Controller (old → new): HBM3 / HBM3E none → 3.575 mm² per stack (n5_derived).** `mm2_per_hbm_mc` (7.15) is per HBM4 stack (32 channels, JESD270-4). A stack is scaled by its JEDEC channel count, 16 / 32 for HBM3 (JESD238) and HBM3E 「来源」 → 3.575 「推导」. NVIDIA GH100 has "12 512-bit memory controllers" for 6 HBM3 stacks 「来源」, i.e. controllers sized per channel group. HBM4E is taken at 32 channels 「假设」. Scope `MC_TYPES`: HBM4 / 4E → HBM3 / 3E / 4 / 4E. The generic `mm2_per_unit_mc` still wins unscaled; n7 / n28 still carry no controller.
- **Cross-check against the TPU floor plans used in 0.328 「推导」.** TPUv4i "HBMC & SerDes" ≤ 13.3 mm² per HBM2 port at N7 ≥ Samsung HBM2 PHY 7.2 + controller 8 ch × 7.15 / 32 = 1.79 at N5 (× 1.8 to N7 「假设」 = 3.2) = 10.4: consistent. Channel scaling fits Google's floor plans better than bus-width scaling. No public HBM3 TPU floor plan exists, so HBM3 itself is not checked. The secondary H100 die-shot estimates (≈ 24 vs ≈ 14 mm² per stack) conflict and are unsourced, so they are not used.
- **Report:** new `hbm_phy` item (mm² per stack, field and source, HBM std base only). `hbm_mc` now appears for HBM3E with "（HBM3E：× 16 / 32 通道，JEDEC 通道数「推导」）". **Notes:** the 0.328 HBM3 / 3E note "只用于 HBM4 / HBM4E" is retired. The energy-DSE PHY note no longer says "HBM3E 窄总线同密度". **UI:** new input "mm² / HBM3 DQ" (filled by every density preset); controller titles and preset note say HBM3 / 3E are scaled by 16 / 32 channels.

### Host IO: `mm2_host_io` (1.8% × 331 = 5.958 mm²) kept and reconciled with a modern PHY source
- VLSI 2022 5 nm DSP SerDes "for Ethernet and PCIe applications" (doi 10.1109/VLSITechnologyandCir46769.2022.9830304): quad 1806 × 825 µm = 0.3725 mm² per lane (abstract 「来源」) → x16 PHY 5.96 mm² 「推导」, equal to the current value within 0.04%. It is an upper bound for a 64 GT/s PCIe 6 / CXL 3 lane, since the source lane is 112 Gb/s. CXL uses the PCIe 6 PHY (Intel 3 VLSI 2024 transceiver, no area published).
- The PCIe / CXL controller is **not** counted: there is no public N5 figure. TPUv2 (NeuroMeter): "PCIe Link + PCIe Ctrl" 2% of < 611 mm² ≈ ≤ 12.2 mm² for both at 16 nm, Gen3 x16 「来源/推导」. Value and tag (二手) unchanged; source text extended.

### Effect (old → new, defaults)
- 1P qwen3-8b, 8 × HBM3E 9.2 Gb/s, n5_derived: std die 654.2 → 774.5 mm² (dram_phy 128 → 192, mem_ctrl 0 → 28.6, ÷ 0.77). J/token unchanged; $/M up. 1P RWKV prefill (4 × HBM3E) 571 → 631 mm²; its dram_bw × 2 what-if +83.1 → +143.2 mm² (still tight, not over reticle).
- **Decisions that change:**
  - 1P + 8 × HBM3E with tp8 / 900 GB/s scale-up at 100 G or 200 G lanes was unbuildable (I/O over the shoreline) and is now buildable; the I/O-over regime starts at 1200 GB/s. 400 GB/s at 50 G lanes is now feasible.
  - 8 × HBM3E prefill on air_oam: flux ≈ 0.99 W/mm² (over) → 0.850 (feasible, thin margin, limit 0.86).
  - Chiplet cost (n5_cowos_s, MAC×1): the mono / 4-die crossover moves from D0 ∈ (0.1, 0.2) to ≈ 0.07 /cm². At D0 0.1 the 4-die UCIe-A package ($2232.9) is now cheaper than mono ($2354.5).
  - tco_sens case 0: the D0 mono → split threshold moves 0.150 → 0.110 /cm². Wafer $ (19,395) and carrier yield (0.680) now also flip inside their sourced ranges. Best package flat $2013 → $2161, component $2116 → $2233 (both now 4-die).
  - The TCO-best memory at tp8 / 400 / D0 0.1 / MAC×2 moves 4-stack → 8-stack (both corners). The 16 pkg-vs-TCO disagreements are now 14 at tp8 + 2 at tp1 (was 16 at tp8).
  - energy_dse power-scope study, air_oam card front [4, 17] → [1, 0, 15].
- **Unchanged:** HBM4 / HBM4E (all paths; only `budget.provided` lists the new field), LPDDR / DDR, host IO.

### Tests
- New `tests/test_core_329.py` (9).
- Re-pinned with old → new comments: test_core_211, 213, 301, 310, 312, 313, 315, 316, 317, 318, 319, 328.

### Freshness
- Since 2026-10-22, new: ATH-MaaS/PixText (pixel-space text-editing image DiT, Apache-2.0, arXiv 2610.18284; hidden 1280, depth 32, 16 heads, patch 16; ≈ 0.95 B if bf16 「推导」). Pending: it needs a new domain builder.
- OrcaCyber Nano 1.0: still pending (GGUF metadata is still the dflash drafter's, config 401).

## [0.328.0]

### HBM4 / HBM4E memory-controller area now on by default (n5_derived, 7.15 mm² per stack)
- **The 0.327 blocker, resolved from public floor plans.** 0.327 kept the controller preset off because the 23% uncore share (TPUv1 control 2% + 21% unlabelled) might already hold controller logic. Three Google floor plans and one paper answer that 「来源」:
  - TPUv1 (Jouppi et al., ISCA 2017 Fig. 2): "DRAM port" 3% × 2 is drawn as its own block, inside the 10% I/O and outside the 21% unlabelled.
  - NeuroMeter (Tang, Li, Nai, Jouppi, Xie, HPCA 2021 industry track) §IV: TPUv1 "DRAM port 6.0%" and the "unknown components ~21%" are separate; TPUv2 (< 611 mm²): "HBM ports 9%" separate from "unknown components (which probably includes the inter-component interconnection) ~21%".
  - TPUv2 floor plan (Norrie et al., IEEE Micro 2021 Fig. 4): four "HBM port" blocks and no separate HBM controller block, while PCIe is split into "PCIe Link" and "PCIe ctrl" — so the HBM controller sits inside the port 「推导」; white areas "are filled with wiring".
  - TPUv4i floor plan (Jouppi et al., ISCA 2021 Fig. 6; Fig. 5 lists "HBM Controller (HBMC)"): two "HBMC & SerDes" blocks measure 6.6% of the die by pixels 「推导」 (scale check: CMEM measures 27.4% vs the caption's 28%) ≈ ≤ 13.3 mm² per stack of a < 400 mm² die; the on-chip interconnect is its own 20.5% block.
  - So in every Google design the memory port **with its controller** is outside the ~21% the model's uncore share copies; adding a controller term does not double count.
- **Value: 7.15 mm² per stack (low end of Samsung HC 2026's 5–10% × 858 mm² ÷ 6, 「二手/推导」).** Cross-check 「推导/假设」: TPUv2 9% × 611 ÷ 4 ≈ 13.7 mm² per HBM2 port, minus Samsung's HBM2 memory-side PHY 6 × 1.2 = 7.2 mm² (HC 2026 slide 11, 「二手」, host PHY ≈ memory-side PHY 「假设」) leaves ≤ 6.5 mm² of controller per HBM2 stack at 16 nm. TPUv4i (≤ 13.3 mm² per stack incl. PHY, 7 nm) leaves ≈ ≤ 6.1 mm²; × 4 channels (8 → 32 per stack) ÷ 1.8 (N7 → N5, n5_derived rule 「假设」) ≈ ≤ 13.6 mm² per HBM4 stack — inside Samsung's 7.15–14.3, not singling out a point. The low end is used; the high end stays as `mc_preset=samsung_hc2026_hi`.
- **Scope (old → new): `mm2_per_hbm_mc` applied to any HBM (0.327) → HBM4 / HBM4E only.** For HBM4 / 4E the PHY term (2048 DQ × 0.015625 = 32 mm²) is the generation's own Samsung figure (8 × 4 mm), so PHY + controller is the complete host interface. For HBM3 / 3E the PHY term is the same HBM4-derived per-DQ value (16 mm² per 1024-DQ stack) while Samsung's own HBM3 PHY is 8 × 3 = 24 mm²: the controller alone would be the smaller half of a correction. HBM3 / 3E stay without a controller until the PHY is made per-generation (pending), and carry a note saying so. LPDDR / DDR no longer get a note (the default fills the field).
- **Presets:** only `n5_derived` carries the default (the source's node class); `n7_tpuv4i` / `n28_tpuv1` do not (no node scaling invented). `budget.mc_preset` is now applied **before** `density_preset` so `samsung_hc2026_hi` overrides the n5_derived default (an explicit `mm2_per_hbm_mc` still wins). `mm2_per_hbm_mc: 0` reproduces 0.327.
- **UI:** the controller preset may replace a value the density preset filled; "不用预设" on it restores the density fill. Labels say HBM4 / HBM4E.
- **Report:** the `hbm_mc` item now also shows the density-preset default (source "面积密度预设 n5_derived：…"), HBM4 / HBM4E only.
- **Fixes:** `chiplet_example.budget(mc)` passed `mm2_per_hbm_mc` twice once the density preset carries it (TypeError) — merged. NeuroMeter citation in `_UNCORE` corrected: Tang et al., HPCA 2021 (was "Shao et al., MICRO 2021"); its 1.8% PCIe is NeuroMeter's **modelled** share, the published floor plan says 3.0% — value unchanged, recorded as pending.

### Effect (old → new, defaults; 1P qwen3-8b decode b64, 8 × HBM4 10 Gb/s, n5_derived, n5_cowos_s, accel_2026, D0 0.1)
- Std die 820.4 → 894.7 mm² (+ 8 × 7.15 ÷ 0.77); J/token unchanged; $/Mtok 9.7658 → 9.8969 m$. Custom base die 535.5 mm², 9.8678 m$ (unchanged) — **custom is now the cheaper option by default** (was std).
- Card-level thermal precedent area of a custom host 820 → 895 mm² (std-equivalent includes the controller ring, §270).
- DSE / hbm_base study (n5_cowos_l, air_oam): std MAC×1 mono 895 mm² > 858 reticle → verdict mono → only_split for qwen3-8b decode / prefill and qwen3-30b-a3b decode; custom stays mono. Std MAC×2 dies 1074 → 1149 mm² get one fewer auto memory edge (4 → 3, the existing fewest-edges rule on a longer perimeter), so their throughput moves (e.g. qwen3-8b decode 13758 → 13290 tok/s).
- Unchanged: HBM3 / HBM3E / LPDDR / DDR results (only `budget.provided` lists the new field, plus the HBM3E note); n7 / n28 presets; no-budget paths except the auto memory-edge proxy for HBM4 / 4E (which uses n5_derived).

### Tests
- New `tests/test_core_328.py` (8). Re-pinned with old → new values: `test_core_324` (std $/M 0.011879644 → 0.012256563, DSE base J 0.0377402 → 0.0377646, custom logic-die formula adds the controller) and `test_core_327` (off = `mm2_per_hbm_mc: 0`; n5_derived carries the field; no LPDDR note; std report shows the item).

### Freshness
- Sweep since 2026-10-21 (110 orgs): Ling-3.0-tiny-UBTree, Flint, Raon-1-mini, OvisOCR3, MOSS-Image are already catalogued / merged; TextVerifier-CJK-CLIP is a classifier (out of scope). Nothing new to add.
- OrcaCyber Nano 1.0: repo updated 2026-10-22 12:16 JST but the public GGUF metadata is still the drafter's (`dflash`), the main files are mixed quants without public metadata, config 401 — still pending.

## [0.327.0]

### HBM memory-controller area: a sourced preset (opt-in)
- **Why this item:** the AGENTS pending / unmodelled lists were checked against public numbers available now.
  - Organic-substrate SPHBM4 PHY pJ/bit and area: still no number. JEDEC and the trade press give none; Eliyan only claims "~2× better power efficiency" against an unnamed baseline.
  - The fewest-edges placement rule: no source with numbers (SoftHier / FlatAttention only parameterises 4-edge placement).
  - Base-die process energy: Samsung ISSCC 2026 15.6 gives SF4 and VDDQ 0.75 V, no pJ/bit split.
  - Near-memory compute: Hot Chips 2026 gave no TOPS or W.
  - The memory-controller area **has** a public figure. It also changes a design decision: whether a custom HBM base die beats the standard interface, and whether the std host still fits the reticle. Until now `mm2_per_unit_mc` had no preset, so std hosts carried no controller area and a custom base die took nothing off the host.
- **Source:**
  - S. Han (Samsung), Hot Chips 2026 tutorial "HBM Base Die: How HBM Will Evolve Using Advanced Logic Processes", 2026-08-23: "Memory controllers account for about 5-10% of the SoC area". Quoted by Maeil Business (MK) 2026-08-24; also TMTBreakout and Wccftech via cnyes 「二手」.
  - The talk's slide 9 reference xPU is 26 × 33 mm with 6 HBM stacks per xPU (ServeTheHome slide images) 「二手」.
  - Per stack = share × 858 mm² ÷ 6 = **7.15 / 14.3 mm²** 「推导」, taking the quote to refer to that xPU 「假设」.
- **What changed:**
  - New `Budget.mm2_per_hbm_mc`, the controller area per HBM stack. It applies to HBM memory only; on LPDDR / DDR / GDDR it is ignored with a note. The generic `mm2_per_unit_mc` wins when both are set.
  - `budget.mc_preset` = `samsung_hc2026_lo` | `samsung_hc2026_hi` (`budget.MC_PRESETS`), fills unset fields only. Wired into the API, the catalog (`mc_presets`), the UI ("HBM 控制器 mm² / 栈" input and "HBM 控制器预设" select), the design report (`hbm_mc` item) and `chiplet_example.py --mc-preset`.
  - Std hosts add stacks × the value (before the 23% uncore share); a custom base die drops it (0.324 rule).
  - The custom eval block reports `mc_mm2_per_stack` and `mc_reclaim_mm2`. When unset, it reports `mc_reclaim_range_mm2` and the note "内存控制器面积未计…". The design report flags the same.
- **Default unchanged** (no density preset carries it). The bottom-up area's 23% uncore is TPUv1 control 2% plus 21% unlabelled, and that unlabelled share may already hold controller logic. Adding the controller to every std die could therefore double count, and the source is a 2× range. Without the preset, all non-custom output is byte-identical to 0.326 (DSE std / custom / chiplet, evals, both studies).
- **Decision changes with the preset:**
  - **Eval**, 1P qwen3-8b decode b64, 8 × HBM4 10 Gb/s, n5_derived, n5_cowos_s, accel_2026, D0 0.1:

    | | std die / $/Mtok | custom die / $/Mtok |
    |---|---|---|
    | none | 820.4 mm² / 9.7658 m$ | 535.5 / 9.8678 |
    | lo | 894.7 / **9.8969** | 535.5 / 9.8678 |
    | hi | 969.0 / 10.0395 | 535.5 / 9.8678 |

    The cheaper option flips from std to custom at the 5% end.
  - **Energy DSE** (`--hbm-base-study`, n5_cowos_l, air_oam, cases 0 / 1 / 4):
    - The std MAC×1 verdict goes **mono → only_split** in all three cases under either preset: the std monolithic die is 895 / 969 mm², over the 858 mm² reticle.
    - Custom verdicts are unchanged.
    - Case 0 mono $/Mtok, std vs custom: 11.1181 vs 11.3158 (none), 11.2639 vs 11.3158 (lo), 11.4236 vs 11.3158 (hi).
    - Every split adds the controller once in total: Σ +74–75 mm² (lo) at N = 1 / 2 / 4.

### Consistency fixes (0.326 follow-up)
`hbm_base.NOT_MODELLED` and the custom shoreline note still said the organic SPHBM4 package was "未建模". It is modelled since 0.326 (`hbm_pkg=sphbm4`, exclusive with custom). The entry is removed and the note points to it. Only the custom HBM4 / HBM4E eval notes change.

### Freshness
- The HF sweep (2026-10-21 onwards, 110 orgs) found nothing new: OvisOCR3 (merged, 0.314), Flint (0.315), Raon-1-mini (0.320) and Ling-3.0-tiny-UBTree (0.311) are all handled.
- **OrcaCyber Nano 1.0 is still pending.** The public GGUF metadata is the DFlash drafter's (`architecture: dflash`, 1.92 B), config.json returns 401, and orcarouter has no newer repos.

### Tests
- New: `test_core_327` (8).

## [0.326.0]

### SPHBM4 on an organic substrate (HBM package option, opt-in)
- **Switch:** `Chip.hbm_pkg` = `si` (default: HBM on a silicon interposer, unchanged) | `sphbm4`. New module `core/hbm_pkg.py`. Also on the CLI (`--hbm-pkg`), in the UI (HBM 封装 select), in the design report (`hbm_pkg` item) and in `chiplet_example.py --hbm-pkg / --hbm-pkg-study`.
- **Sources:**
  - JEDEC JESD330-4 (v1.0 Jun 2026; PR 2026-07-13) 「来源」: same DRAM core dies as HBM4; 512 DQ (32 × 16-bit channels, 8 Quad Channels) with 4:1 serialisation, each channel "four times faster than the corresponding HBM4 channel"; bump pitch > 90 µm; channels ≤ 20 mm.
  - Rates 22.4 – 46.0 GT/s and 2.944 TB/s per stack at 46 GT/s (Tom's Hardware, Igor's Lab) 「二手」.
  - Eliyan (Electronic Design 2026-02-18) 「二手」: silicon "two per 25-mm die edge"; organic "three memory devices … along the same edge".
- **What changes when it applies:**
  - Shoreline per stack: 11 → **25 / 3 = 8.33 mm** (Eliyan's organic figure, used as published). Eliyan's own silicon figure is 12.5 mm against the model's 11 mm (Micron footprint), so the gain is 1.32×, not Eliyan's 1.5×.
  - The stacks leave the interposer: the DSE package-area limit (CoWoS-S / -L) and the 0.313 bounding box count only the logic dies (JEDEC: organic "rather than silicon substrates").
  - Stack bandwidth and capacity are unchanged: 512 DQ × 4× rate = the HBM4 stack at the same equivalent rate 「推导」.
- **Scope:** HBM4 only, at HBM4-equivalent rates of 5600 – 11500 MT/s (22.4 – 46.0 GT/s ÷ 4) 「推导」. HBM4 11700 / 13000 MT/s, HBM4E and HBM3E are ignored with a warning. Mutually exclusive with `hbm_base=custom` (SPHBM4 defines its own base die): validation error.
- **Not modelled: no public number, so HBM4 values are kept (`hbm_pkg.NOT_MODELLED`):**
  - SPHBM4 host-PHY area (Eliyan says only "often not more") and interface pJ/bit (Igor's Lab: "no reliable product measurements").
  - Organic-substrate $/mm² and flip-chip attach yield. Package $ keeps the stack footprint at the CoWoS rate, an upper bound; eval reports the most that dropping it could save (`pkg_usd_bound`, 8 × HBM4 on n5_cowos_s ≈ $549).
  - SPHBM4 $/GB (Tom's: the base die is "likely costlier").
  - Substrate size / layer limits, 20 mm routing feasibility, and SerDes latency.
- **Examples (eval: n5_derived, tpuv4i_7nm, accel_2026, n5_cowos_s, D0 0.1; si → sphbm4):**

  | Example | shoreline | memory edges | J/unit | $/Mtok (m$) |
  |---|---|---|---|---|
  | 1P 8×HBM4 qwen3-8b decode b64 | tight → tight | 4 → 3 | 0.037492 → 0.038588 | 9.7658 → 9.8150 |
  | 1P 12×HBM4 | **over → tight** | 4 → 4 | 0.040242 → 0.037966 | 13.5325 → 13.4305 |
  | H100-like 6×HBM4 | tight → tight | 4 → 3 | 0.036328 → 0.036688 | 8.0569 → 8.0730 |
  | H100-like 8×HBM4 | **over → tight** | 4 → 4 | 0.036587 → 0.036587 | 9.9735 → 9.9735 |

  - Throughput, die area and package $ are unchanged in every row.
  - The +2.9% / +1.0% J on the first and third rows is the 0.309 / 0.312 fewest-edges placement rule: narrower stacks fit on fewer edges, so DRAM gets fewer NoC attach edges. It is not the package. With `noc_mem_edges` pinned, J is byte-identical si vs sphbm4 (tested). eval reports this in `hbm_pkg.notes`.
- **Energy DSE (`--hbm-pkg-study`, cases 0 / 1 / 4, n5_cowos_l, air_oam, accel_2026):**
  - Interposer area per point drops by the stack footprints (8 × HBM4: e.g. 2942 → 820 mm² mono).
  - 12 × HBM4: mono and N = 2 rows go shoreline over → tight (buildable). Mono stays infeasible on the reticle (987 > 858 mm²), so the verdicts stay only_split.
  - 8 × HBM4 qwen3-8b decode MAC×1: verdict mono → split_J (J rule above), TCO verdict mono unchanged.

### Custom-HBM DSE speed (results byte-identical)
- 0.325 made the custom base-die DSE slower: 216 points 4.64 s cold / 2.49 s warm (std 3.4 / 2.1).
- Fixes, all exact:
  - `hbm_base.shore_floor_mm2` is memoised on (scenario, die): `full_area` and `std_equiv_delta` asked for the same floor, 180 bisections of which 66 were distinct.
  - `sized_layout` caches its die-independent inputs (memory units, the attach links of a 4-edge chip, I/O macros, D2D edges) per key-without-die.
  - After the first die shape, candidates whose memory or D2D is over by `_place`'s own test are skipped. Only a candidate with nothing over can be chosen there.
  - The water-fill in `_place` is an explicit scan with the same arithmetic and tie order.
- Now 3.5 – 3.8 s cold / 2.1 – 2.5 s warm (back to 0.324's 3.55 / 2.10 within run-to-run noise).
- Proof:
  - The full JSON of 4 DSEs (custom / std × 216-point and 126-point chiplet grids), 14 evals and both `chiplet_example` outputs is byte-identical to 0.325 (3,049,832 bytes each).
  - test_core_326 checks the scan and the pruning against the 0.325 form on 3000 random placements.

### Freshness
- The HF sweep on 2026-10-22 (110 orgs) matches 0.325: nothing addable.
- **OrcaCyber Nano 1.0 is still pending.** Retried the HF API (`expand` gguf / config / safetensors, tree, base repo, search) and third-party sources:
  - The public GGUF metadata is the DFlash2 drafter's (`architecture: dflash`, 1.92 B).
  - The base repo and config.json return 401.
  - localLLMrig states its KV heads / head_dim are *assumed* from Qwen3.5-27B because the config is gated.
  - The V100-LLM catalogue calls it "a Qwen3.8-27B type model" but gives only file sizes / sha256.
  - Heads, KV heads, head_dim, FFN, linear-attention dims and vocab remain unconfirmed.

### Tests
- New: `test_core_326` (10).
## [0.325.0]

### Custom HBM base die: adversarial review of 0.324 (fixes, sourced)
- **(a) Shoreline per stack: unchanged, but for the right reason.**
  - On a silicon interposer a short-reach D2D link needs the stack adjacent to the host, so the stack footprint (11 mm) sets the edge 「推导」. Eliyan: 2 stacks per 25 mm edge on silicon 「二手」.
  - The 0.324 reason ("D2D 8.5 mm ≥ PHY 8 mm") was wrong. The host-side UCIe link is only ≈ 2.7 mm of edge.
  - A closer pitch needs a longer-reach interface on an organic package (JEDEC SPHBM4 JESD330-4: 512 DQ, 4:1 serialisation, ≥ 90 µm bumps, 20 mm reach, "potentially increasing the total number of SPHBM stacks"; Eliyan 3 per 25 mm). That is a different package, so it is listed as not modelled (`ORGANIC_STACKS_PER_25MM` = (3, 2), reference only).
- **(b) Interface energy: no double count in 0.324, but the memory side was missing.**
  - Device `pJ_bit_dram_hbm4` 2.46 includes the base-die PHY. 0.324 scaled only the host PHY (0.29 × 0.55 = 0.1595).
  - Now one D2D link (`ucie-a-48`, 0.25 pJ/bit 「来源」 catalogue) replaces both PHY ends (host 0.29 Chae ISSCC 2023 「来源」; memory end taken symmetric 「推导」). It is split half host / half base die:
    - host `phy_pJ_per_bit` 0.1595 → **0.125**;
    - device delta (new `dram_dev_pJ_delta`) 0 → **−0.165**.
  - Interface ratio 0.25 / 0.58 = 0.43, inside Marvell's −45 … −70% (`VENDOR_IFACE_RATIO` (0.25, 0.55), cross-check only, 「二手」; Hot Chips 2025 "75% lower HBM & main die I/O power" is the upper bound).
  - The × 0.55 factor is gone. The same delta enters `pdqueue` (both branches).
- **(c) Area.**
  - The host D2D area was the 12.75 mm² base-die block (Samsung HC 2026). It is now derived from UCIe geometry: units = ⌈stack GB/s ÷ 384⌉ (HBM4 10 Gb/s → 7) × 0.3888 × 1.043 mm = **2.84 mm²/stack** 「推导」 (12.75 is kept as `base_die_d2d_mm2`, memory side, reference only).
  - Audit: shoreline, thermal, eval TCO, DSE cost / yield and package all read the custom die. NoC pitch (engine tile) and hops (engine grid) are correctly independent of the PHY.
  - New **shoreline floor**: a custom host die smaller than the edge the stacks need is held at that floor (`terms.shore_pad`, filler silicon) instead of being reported "over". This is the switch's own side effect; std is unaffected.
- **(d) Thermal: the 0.324 "smaller die flips to split" was an artifact.**
  - The card-level precedent (H100 700 W / 814 mm² = 0.86 W/mm² on air) is defined on a die that *includes* the HBM PHY / controller ring. Custom card flux is now judged on the std-equivalent area (`card_area_mm2` = die + `std_equiv_delta`). Die-heat flux stays on the real area.
  - Controller power: the std model has no separate controller energy term (none public; SC'25 lumps control + interface 8.5–13.5 pJ/bit), so there is nothing to move. Samsung HC 2026: controller moves to the base die, "thermal impact manageable". Direction: custom would run cooler. Listed, not modelled.
- **Old → new (custom; std byte-identical):**

  | Example | die mm² | shoreline | J/unit | $/Mtok | Note |
  |---|---|---|---|---|---|
  | 1P 8×HBM4 qwen3-8b decode b64, air | 620.4 → 535.5 (logic 517.5 + pad 18.0) | tight → tight | 0.036618 → 0.035282 | 0.010040 → 0.00987 | pkg $ 2145 → 2020 |
  | 1P prefill b8 | | | | 0.001452 → 0.00144 | thermal over → ok |
  | 1P qwen3-30b-a3b decode b64 | | | 0.050486 → 0.04836 | 0.015571 → 0.0153 | |
  | H100-like 6×HBM4, cap 400 W, liquid | 278.3 → 283.2 (logic 201 + pad) | **over → tight** | 0.03617 → 0.03411 | 0.008377 → 0.00829 | |
  | 1P 8×HBM4E, cap 300 W | 620 → 535 | | | 0.014224 → 0.01316 | thr 5849 → 6261 |
  | HBM3E | | | | unchanged | switch ignored |

- **Energy DSE (`--hbm-base-study`):** no thermal buildability flips any more. MAC×1 verdicts are mono, as std; 0.324 had the prefill thermal flip and a mono → only_split TCO verdict. At MAC×2 the custom mono die is 1074 → 771 mm², under the 858 mm² reticle, so mono becomes feasible (8b decode: thr 13758 → 15266, $/Mtok 8.69 → 7.96 m$).

### Near-memory compute / offload in the base die: not modelled (documented)
- Sources:
  - SK hynix StreamDQ (base-die dequantisation, "up to 5.15×"): no compute or energy figures.
  - Samsung phase 2: no figures.
  - AMMA arXiv 2604.26103 (academic, 96 TFLOPS per cube) states "PNM alone does not increase per-cube bandwidth".
- No vendor publishes base-die compute or energy, so nothing was added.
- Upper bound 「推导」: attention offload saves at most the interface energy of the KV bytes, i.e. KV-byte share × whole-link energy. On 1P qwen3-8b decode b64 custom that is ≈ 3.4% J at ctx 4K and ≈ 5.3% at 32K, before any base-die compute energy. This is recorded in `hbm_base.NOT_MODELLED`.

### Freshness
- The HF sweep on 2026-10-22 (110 orgs) matches 0.324: only text-to-image and classifier repos, none addable.
- **OrcaCyber Nano 1.0** (orcarouter, gated, licence qwen): pending, not added.
  - Public: `Qwen3_5ForConditionalGeneration`, 64 layers, hidden 5120, 48 linear + 16 full attention, ctx 262144, MTP. These match Qwen3.8-27B.
  - Not public: heads, KV heads, head_dim, FFN, linear-attention dims, vocab. config.json and the GGUF headers return 401 (gated); the base repo is not public; Thaki Cloud reports the base "officially unconfirmed".
  - If the base is confirmed, effective bits per (LLM + MTP) param, from file size 「推导」: Mix-13G 3.81, Mix-15.7G 4.60.
  - OrcaCyber Zero 1.0 / 1.5 and Ultra 1.0 are API-only with no config, so they cannot be added.

### Tests
- New: `test_core_325` (7).
- Re-pinned in `test_core_324`: custom PHY area 7 × 0.4055; die = logic + `shore_pad`; `phy_pJ_per_bit` 0.125; `dram_dev_pJ_delta` −0.165; CLI die 535.460; `test_hbm_base_study_no_thermal_artifact` (was the thermal flip).

## [0.324.0]

### Custom HBM4 / HBM4E base die (design option, opt-in)
- **What it models:** from HBM4 on, the base die is made on a logic process: SK hynix on TSMC N12, Samsung on SF4, Micron on in-house CMOS. Custom HBM4E and NVIDIA NVHBM base dies are on TSMC N3-class 「二手」. A *custom* base die moves the memory controller onto the memory and replaces the 2048-DQ host PHY with a narrow die-to-die link (Samsung Hot Chips 2026; Marvell custom HBM; NVHBM). New `Chip.hbm_base` = "std" (default, the JEDEC interface, ≤ 0.323) | "custom". Module `core/hbm_base.py` carries every number with its tag and source.
- **Numbers and labels:**
  - Host PHY per stack: 2048 DQ × 0.015625 = 32 mm² (std; Samsung HC 2026 "conventional PHY > 8 × 4 mm") → 8.5 × 1.5 = 12.75 mm² D2D (Samsung HC 2026 「二手」; host side taken equal to the memory side 「推导」).
  - Memory controller: off the host (the `mem_ctrl` term drops out when a density is set).
  - Host PHY pJ/bit: × 0.55. Marvell: interface power −45 … −70% "depending on scenarios" (EE Times part 2; Marvell PR "up to 70%") 「二手」. The low end is the default; the range (0.30, 0.55) is recorded.
  - Memory $/GB: × 1.10 on the sourced price. Custom premium "5% to 10% overhead tops" (EE Times part 1, analyst speculation) 「二手/假设」; range (1.05, 1.10). A user-entered price is taken as given.
  - Process: N12 / SF4 → N3P / N3 (reported, not used in a formula).
- **What it does not change (mechanism):**
  - The shoreline a stack needs stays 11 mm. The stack's footprint sets the edge, and the D2D block (8.5 mm) is no narrower than the PHY (8 mm) 「推导」: a custom die shrinks PHY *depth* and area. Shoreline therefore moves only through the smaller host die's shorter perimeter.
  - Stack bandwidth is unchanged. NVHBM's +30% comes with a faster interface spec, and Marvell's custom stack (2.048 TB/s) is slower; neither is a base-die property.
- **Listed, not modelled (no public figures):** near-memory compute / offload (Samsung phase 2); base-die process energy (N12 0.8 V → N3P 0.75 V); Marvell's "+33% stacks"; D2D hotspot (Samsung Heat Path Block, −35% peak temperature).
- **Scope:** HBM4 / HBM4E only. On HBM3E / LPDDR / GDDR the switch is ignored, with a warning. The std outputs are byte-identical to 0.323 (DSE JSON; eval adds the new `energy.phy_pJ_per_bit`). The eval block `hbm_base`, the DSE key `hbm_base`, the row flag `hbm_custom`, the DSE note and the design-report item appear only when custom.
- **Wiring:** the area path (`budget.phy_area` / `full_area`), energy (`energy.phy_pJ`, also used by pdqueue), price (`tco.mem_price` in eval TCO, the DSE `_tco_row` and the tornado `base_values` / `row_usd` / `input_range`) and `report._base`. Eval ≡ DSE row under custom. UI: chip panel select "HBM 基础晶粒". CLI: `--hbm-base {std,custom}`. Script: `scripts/chiplet_example.py --hbm-base custom` and `--hbm-base-study`.
- **Old → new on the examples** (eval, n5_derived, tpuv4i_7nm, accel_2026, n5_cowos_s, D0 0.1; std → custom):

  | Example | die mm² | shoreline | thr | J/unit | pkg $ | mem $ | $/Mtok | Note |
  |---|---|---|---|---|---|---|---|---|
  | 1P 8×HBM4 qwen3-8b decode b64, air_oam | 820.4 → 620.4 | tight → tight | 9609 → 9609 | 0.037492 → 0.036618 | 2502 → 2145 | 9378 → 10315 | 0.0097658 → 0.0100401 (+2.8%) | |
  | 1P 8×HBM4 qwen3-8b prefill b8 | (as above) | | 71192 → 71192 | 0.0071166 → 0.0071100 | | | 0.0014103 → 0.0014523 (+3.0%) | |
  | 1P 8×HBM4 qwen3-30b-a3b decode b64 | | | | 0.051874 → 0.050486 | | | 0.015136 → 0.015571 (+2.9%) | |
  | H100-like 6×HBM4 decode b64, cap 400 W, liquid | 428.3 → 278.3 | **tight → over** | 8662 → 8806 | | 1742 → 1575 | | 0.008164 → 0.008377 | The smaller host die no longer has the perimeter for 6 stacks: the area saving is bounded by the shoreline floor on a small host |
  | 1P 8×HBM4E decode b64, cap 300 W | | | 5299 → 5849 (+10.4%) | | | | 0.015128 → 0.014224 (−6.0%) | Under a binding cap the PHY energy saving buys clock |
  | 1P 8×HBM3E | | | | | | | unchanged | Switch ignored, warning |

- **Energy DSE example** (`chiplet_example.py --hbm-base-study`: cases 0 / 1 / 4 on 8 × HBM4, n5_cowos_l, accel_2026, air_oam, 700 W):
  - Every host die is 200 mm² (mono) / 100 (N=2) / 50 (N=4) smaller. That is 8 × 19.25 mm² PHY ÷ (1 − uncore 0.23), the existing area rule.
  - No shoreline status changes.
  - **Buildability flips on thermal:** qwen3-8b prefill MAC×1 mono / ×2簇 / ×4簇 and UCIe-A/S N=2, UCIe-A N=4 turn unbuildable on air. The same ~530 W die power on 620 instead of 820 mm²: flux 0.81 → 1.04 W/mm². The MAC×1 TCO verdict goes mono → only_split.
  - At MAC×2 (cap-bound) throughput rises 1.4 – 5.7%. $/Mtok goes up 3.5 – 4.8% at MAC×1 (the memory premium exceeds the die saving) and down 0.1 – 2.6% for MAC×2 mono.

### Energy DSE: exact reuse inside a search (CAP_MEMO)
- In a power-capped DSE every f_op < 1 point over the cap is capped from f_max (0.302 rule, `_run1`). Its DVFS bisection is therefore exactly the one its f_op = 1 twin already ran: same evaluate(s), table, cap, curve and package-power function. Prefill at 150 W, 216 points: 2844 evaluate calls, 1368 of them repeats.
- `energy_dse._memo` stores `evaluate(s)` and `dvfs_cap(...)` keyed on the frozen Scenario / EnergyTable, the cap, the curve and the package-power function's arguments. The store is scoped to one `energy_design_search` call and returns a deep copy of a cap dict. `energy_dse.CAP_MEMO` = False restores the ≤ 0.323 path.
- Results byte-identical (whole DSE JSON, on vs off) on 7 bodies: cap 150 / 300, prefill, MoE, mix phase, custom HBM4, include_unbuildable + card scope.
- CPython, 216 points: prefill cap 150 W 5.53 → 2.42 s, cap 150 W 3.88 → 2.58 s, MoE 4.85 → 2.94 s, mix 7.51 → 4.96 s. Pyodide (cold / warm): prefill cap 150 W 9.9 / 5.2 → 8.2 / 4.0 s; cap 150 W 6.9 / 5.0 → 5.8 / 4.1 s.
- **Not done:** skipping unbuildable points before evaluation. The DSE keeps every unbuildable row with its full metrics (thr, J, W, $, the why), whether or not `include_unbuildable` is set; the UI shows them, and the fronts and verdicts read the flags. Skipping them would change the result, so it is not "provably identical". Only exactly-known work is pruned.

### Freshness
- 110 orgs, 2026-10-22: the same set as 0.323's sweep (unsloth MOSS-Image GGUF / FP8 derivatives, an image-classification CLIP). Nothing to add.
- HBM sources dated up to 2026-10-19 (TrendForce: NVHBM on TSMC 3 nm) were used for the base-die numbers.

### Process
- 0.323's ext_validation.json stamp had been hand-edited (0.322.0 → 0.323.0). From 0.324 on, ext_validate is rerun after the version bump. This release's run changed only the stamp (0.323.0 → 0.324.0, script-generated); every validation number is unchanged.

### Tests
- tests/test_core_324.py (10):
  - std pinned to the 0.323 tree;
  - custom mechanisms (PHY 32 → 12.75 mm² per stack, die −200 mm², edges unchanged, pJ × 0.55, mem $ × 1.10, throughput unchanged);
  - area terms and controller removal;
  - price rules (user price as given);
  - HBM3E ignored with a warning;
  - eval ≡ DSE row under custom;
  - report item only when custom;
  - Chip validation, UI select and CLI flag;
  - the prefill thermal flip;
  - CAP_MEMO byte-identical with fewer evaluate calls, run-scoped.

## [0.323.0]

### Cross-module consistency audit (eval ↔ energy DSE ↔ report ↔ compare ↔ static build ↔ CLI)
- **Audit:** for seeded random designs (5 chips × 4 models × phase / batch / cooling / cap / cost / TCO preset / D0), compared eval vs the energy-DSE base row on J/unit, card W, package W, thermal status, buildable, shoreline status, memory edges, good-package $, $/M unit and the capped flag (tolerance 1e-9), and report items vs row fields, compare cells vs report items (value / tag / unit). Server and static Pages build answer through the same `serve.respond` (no worker pool on eval / DSE / report), so their bytes are equal by construction; the CLI prints `api.clean` of the same call. Four divergences found and fixed (eval aligned to the DSE / report convention, the more mechanistic path):
  1. **D0 unset:** eval TCO defaulted D0 to 0.1 /cm² while the DSE and report treat it as missing (die yield 1, flagged). Eval now does the same, with the note "未设 D0（tco.d0_per_cm2）：晶粒良率按 1 计…" and `pkg_src` "未设 D0：良率按 1 计（缺项）". 1P qwen3-8b decode b64, n5_cowos_s: `tco.pkg_usd` 1755.44 → 1615.51 (= DSE row).
  2. **Memory edges:** with auto edges eval resolved them on the preset's published die (H100-like 814 mm²) while the DSE sizes them on the budget-density die it reports. Eval now resolves them on the budget die whenever a budget is given (`hbm_mem_edges(scn, die_mm2)`). H100-like qwen3-8b decode b64, n5_derived: 2 → 3 edges; J/token 0.05587 → 0.05551, card W 155.87 → 154.87 (= DSE row).
  3. **Power cap source and operating point:** eval capped only at `energy.tdp_W` and ran thermal / TCO at the duty-cycle throttle; the DSE caps at `budget.power_W_card` (else `energy.tdp_W`) at the dfs clock. Eval now takes the same cap (`power_cap.cap_src` = "budget.power_W_card" / "energy.tdp_W"), keeps the duty-cycle headline (labelled conservative), and runs thermal (`operating_point` "capped") and TCO at the dfs point (`_capped_point`). H100-like qwen3-30b-a3b decode b1, tdp 150 W, liquid: TCO throughput 310.9 → 364.30 tok/s (duty ×0.853 → dfs ×0.99996, memory bound); thermal package W 173.1 → 150.0. The error without a cap now names both inputs: "power_cap 需要功耗上限：budget.power_W_card 或 energy.tdp_W…" (was "power_cap 需要 tdp_W…").
  4. **Bisection depth:** `dvfs_cap` ran 22 steps in eval / sensitivity and 12 in the DSE, so one capped design differed by up to 2⁻¹² of the clock range. One shared `powercap.CAP_ITERS` = 16 「推导」: (1 − F_MIN)/2¹⁶ ≈ 1.3e-5 of the clock, an order below the 2e-4 dfs ≡ duty tolerance (12 steps broke it). DSE capped clocks rise by ≤ 2.1e-4 (the 12-step clock is a prefix of the 16-step bisection); eval ones move by ≤ 1.3e-5.
- UI: the power-cap line names its source (功耗预算 / TDP), states that thermal / TCO use the dfs point, "未超 TDP" → "未超上限". CLI: `eval --body FILE` takes an /api/eval body as-is.

### Energy DSE speed (Pyodide)
- Measured on the static build (headless Chrome, worker `ms`): a 216-point DSE with every new axis took 3.6 – 6.6 s, but 15.3 s (cold) / 13.2 s (warm) once the power budget binds (150 W: 138 capped rows × 12 bisection evaluates).
- **Mapping memo keyed without the clock** (`mapping.FREQ_FREE_KEY`, default True): the GEMM mapping search counts cycles and no path in it reads `chip.freq_ghz` (only `gemm_seconds` / `vector_seconds` convert, with the caller's chip), so `gemm_cost` keys `_gemm_cost` on the chip at 1 GHz (`_cycle_chip`). Bit-identical results; the bisection and the f_scales axis reuse the nominal-clock search.
- **Layer IR memoised for one search** (`ir.rank_memo()`, entered by `energy_design_search`; `energy_dse.RANK_MEMO`): `build_rank_ops` is a pure function of (model, layer, phase, shard, glue), all frozen, and no chip field enters it, yet every bisection clock rebuilt it. Scoped to the call, so module flags toggled between runs are never served stale; callers get a fresh list.
- `energy_dse.hbm_stacks` / `_hbm_footprint` memoised (pure in the id string; the TCO flip scan re-parsed ids ~10⁵ times).
- Results bit-identical (the whole DSE JSON, both switches off vs on). CPython, 216 points: cap 150 W 10.19 → 3.32 s, cap 300 W 6.31 → 2.13 s, prefill cap 150 W 10.30 → 3.41 s, MoE cap 150 W 12.03 → 3.88 s. Pyodide (cold / warm): cap 150 W 15.3 / 13.2 → 6.9 / 5.0 s, prefill cap 150 W 15.4 / 11.4 → 9.9 / 5.2 s; unbinding 216-point DSEs 1.4 – 3.5 s. The remaining time is the overlap feed simulation (clock-dependent op times, already exact run-skipping). No pruning of dominated or unbuildable points: they are rows the UI shows (`include_unbuildable`), and the target was met without it.

### Freshness
- 110 orgs, 2026-10-21 / 22: Ling-3.0-tiny-UBTree, OvisOCR3, Flint, Raon already in data; 2026-10-22 only unsloth MOSS-Image GGUF / FP8 derivatives (MOSS-Image in the catalog since 0.257) and an image-classification CLIP. Nothing to add.

### Tests
- tests/test_core_323.py (7): eval ≡ DSE base row over 10 seeded designs; report / compare ≡ row over 3 seeded DSEs; wire / static ≡ API and CLI `--body --json` ≡ wire; cap source / iterations / budget-die edges; speedups bit-identical on a capped DSE; GEMM cycles independent of the clock over 40 seeded shapes; IR memo run-scoped.
- Re-pinned: test_core_189 (cap error text), test_core_316 (thermal at the capped point; 0.315 capped throughput now a lower bound within 3e-4), test_core_317 (TCO capped package W within the bisection band).

## [0.322.0]

### Parked items re-examined: Drama 3 c256 second-GPU wait, colocated vocoder c128 residual
- **Admission / capture mechanisms checked against sources and excluded** (sglang-omni @ 0029194, pinned sglang 0.5.21):
  - Decode CUDA graphs: sglang `arg_groups/memory_hook.py` sets the decode-graph `max_bs` to 256 on H20 / H200 (tp < 4), so c256 decode replays a graph. A capture limit is excluded 「来源」.
  - State pool: Drama 3 sets `disable_radix_cache` True and passes `max_running_requests` (`engine_builder.generation_defaults`). sglang then sizes the mamba / GDN state pool to exactly `max_running_requests` (`mem_cache/kv_cache_configurator.py`, branch "mamba_pool.from_max_running_requests"). The cookbook sets 256, so the state pool cannot cap admission below 256 「来源」.
  - Prefill graphs: a prefill above 16 requests (`first_frame_cuda_graph_max_requests`) or above 2048 tokens (`PREFILL_CUDA_GRAPH_MAX_TOKENS`) runs the first-frame step / prefill eagerly (`sglang_model.py`); it is not capped. The steady-state prefill batch at c256 is 10.8 requests × 148 rows ≈ 1,600 rows, inside both limits. Only the start burst is affected 「推导」.
  - Preprocessing: the cookbook requests carry pre-encoded `vq_codes`, so the GPU reference encode in `create_preprocessing_executor` does not run 「来源」.
  - Table 2 TTFA p95 / p50 at second GPU c256 is 2469 / 2146 = 1.15: a steady-state wait, not the start burst 「推导」.
- **Remaining sourced mechanism, not modelled:** Drama 3 runs `disable_overlap_schedule` True, so the scheduler's per-request CPU work (output processing, `Drama3FrameStreamer`, a tensor message per frame for the first `stride` frames) is serial with the GPU round. Its cost is not published.
  - New measured-only diagnostic `host_implied_us` in `extspeech.table` rows (1/meas − 1/pred per frame, µs) 「推导」.
  - Table 1 falls 393 → 194 µs (c8 → c64); Table 2 second GPU rises 77 → 113 µs (c128 → c256). No constant per-request host term fits both, so nothing is fitted. Predictions are unchanged.
- **Colocated vocoder c128 residual:** sourced candidate, documented, not adopted.
  - Drama 3's vocoder does not coalesce stream chunks (`can_batch_stream_chunks` False). Every incoming item is copied to the GPU (`build_stream_vocoder_chunk`: `codes.to(device, non_blocking=True)`).
  - The engine sends the first `stride` = 10 frames one item at a time, so each request makes 9 copy-only GPU submissions on the vocoder context on top of its 5 windows.
  - If each cost a switch pair like a window, the vocoder-side loss would grow by about +2.6 ms per request (≈ +33 %). Whether a small DMA copy from another context forces a compute time-slice switch is undocumented (Gilman & Walls 2021 O4 names transfer interference without a per-copy cost), so it is not a reading or a default.

### Static Pages (Pyodide) build: design report / compare
- Report and compare now use their own channels (`report`, `report_cmp`). Since both hit `/api/report`, the progress pill used to name both "report 设计结论报告"; it now keys `report_cmp` by channel (`CHAN_STAGE`, opt-in) and shows "compare 设计对比". Both stay on the slow lane (`/api/report`) and are non-blocking.
- A superseded reply no longer clears the newer request's status line (`if (!rr) return`).
- A reply that lands after its design search was replaced is discarded, not downloaded (`reportStale`, by `S.dseRes` identity).
- A scenario edit or a new design search drops in-flight report / compare requests (`dropReports` → `cancel`; the static build terminates the slow-lane worker). Finished report / compare statuses are stale-tagged (`invalidate` list; `staleTag(S.dseVer)`).
- `scripts/ui_static_smoke.py` step 2e: DSE search → report (progress pill, inputs enabled, downloaded) → compare double-click (superseded request killed or dropped, slow lane only) → mid-run scenario edit (dropped or stale-tagged; report stale-tagged).

### Freshness
- 110 orgs since 2026-10-21: Ling-3.0-tiny-UBTree, OvisOCR3, Flint and Raon are already in the data, fetched 2026-10-22 after their last change (10-21). Nothing new on 2026-10-22.

### Tests
- tests/test_core_322.py (4).

## [0.321.0]

### Parked item re-examined: colocated vocoder (Drama 3, vocoder on the engine's GPU)
- **Mechanism, now sourced:** sglang-omni @ 0029194 runs the TTS engine (process "pipeline") and the vocoder (process "vocoder") as two OS processes on GPU 0 (`models/fishaudio_drama3/config.py`), with CUDA MPS off by default (`config/schema.py` `mps: "off"`).
  - Without MPS, kernels of separate processes never run concurrently. The GPU time-slices between the contexts: NVIDIA "Improving GPU Utilization in Kubernetes"; Gilman & Walls, Performance 2021 (arXiv 2110.00459, §4.2).
  - "Loads add" was 「假设」 up to 0.320; it is now 「来源」.
- **New term:** each vocoder window, while the engine is busy, costs a context switch in and a switch out. The published gap between consecutive time slices is 145 µs (Gilman & Walls, measured on an RTX 3090, Ampere) 「二手」; it is applied to H200 as 「近似」.
  - `extspeech.SWITCH_S`; `predict(timeslice="switch")` is the default.
  - `timeslice="slice"` is an **upper bound**: each window holds a full ≈ 2 ms slice (same paper). Whether a context that runs out of work keeps the rest of its slice is not documented, so the bound is not a default.
  - `timeslice="none"` reproduces 0.320.
- **Same → second-GPU frames/s speed-up** (Table 2, ext default variant; `extspeech.colocated`, written by ext_validate):

  | c | measured | 0.320 (loads add) | 0.321 switch | slice upper bound |
  |---|---|---|---|---|
  | 128 | 1.696 | 1.343 | 1.421 | 1.615 |
  | 256 | 1.718 | 1.520 | 1.639 | 1.934 |

  - At c = 256 the measurement lies inside the published band. At c = 128 it lies above even the upper bound (1.62 vs 1.70), so a residual beyond time-slicing remains. Candidates are DAC kernel efficiency on GPU and memory-transfer interference (Gilman & Walls O4); there is no public trace. Documented, not fitted.
- **Rows old → new** (predicted / measured, default):
  - Table 2 same-GPU: c128 1.765 → 1.668, c256 2.127 → 1.973; second-GPU rows unchanged.
  - Table 1 gmean 1.376 → 1.352, c64 1.545 → 1.485, TTFA gmean 0.679 → 0.690.
  - Catalog variant: gmean 2.355 → 2.297.
- **Scope:** diagnostic only (`extspeech`). On an NPU the vocoder and the engine share one runtime and one context, so no time-slice term applies; the main speech model is unchanged.

### Parked item re-examined: Dory `lm_head.logit_scale` (fixed)
- 「来源」 vLLM PR #60149 `DoryForCausalLM`: `self.lm_head.logit_scale = nn.Parameter(torch.empty(vocab_size))`, and `compute_logits` returns `logits.float() * (logit_scale.float() / initializer_range)`. It is a learned per-vocab parameter plus one multiply per logit, not a quantisation scale.
- `summarize_release.NOT_SCALE_RE` keeps `lm_head.logit_scale`. The three Dory releases were re-summarised from their headers (`fetched` kept): params_llm 3,975,098,368 → 3,975,229,440 (+131,072), and bytes_llm +262,144 (the old "scale" role moves into lm_head).
- Template: `ModelSpec.logit_scale`. Dory: `io_misc_params` += vocab, and the sample vector op gets 4 instead of 3 ops per logit. Param checks stay exact (rel_err 0) for all three releases.
- Effect: dory1.0-4b on H100-like decode tok/s −0.0001 % at b1, −0.002 % at b64.

### Design comparison report (new)
- `report.compare_report(r, body, rows)` puts 2 – 3 DSE rows side by side; the first is the baseline A.
  - Each column is that design's own `design_report` item (value, tag, source, missing reason); nothing is recomputed.
  - Items are aligned by key, with Δ vs A for numbers and equality for everything else (◆ marks a difference).
  - Directed metrics (`DIRECTION`: $/M, perf/W, throughput, TPOT, J/unit, package W, flux, $/good package, …) get a winner, and a near tie when the gap to the runner-up is ≤ the DSE tie (1 %).
  - An item missing in only some designs is flagged (`missing_in`); its cell carries the reason and is never 0.
  - The same quantity with different evidence labels across designs (e.g. 推导 vs 假设) is flagged (`tag_differs`).
  - Energy shares show the union of each design's top shares (the row's own computed share).
- **$/M near ties** are mechanism checks: for each design vs A, the 0.320 pair check (one-at-a-time scans over the sourced ranges plus both joint corners). If a design is unpriced, the report says so instead of comparing.
- **Headline:** $/M, perf/W, throughput, TPOT, J/unit, package W and $/good package. Warnings cover the headline metrics plus the pair checks; each design's own warnings and missing items are listed too.
- **Exports:**
  - API: `POST /api/report` with `report_rows: [i, j(, k)]`, or `report_compare: true` (the report row vs its cheapest rival). Returns `{rows, hash, compare, markdown}`.
  - CLI: `scripts/design_report.py --compare 3,0[,1] | auto`.
  - UI: row input + 设计对比报告 button (downloads .md + .json).
- Example (`--example --compare 3,0,1`): A 32×128×128 $/M 0.008479; B +0.17 % and C +0.35 % (within the tie, A holds at both corners, no flips). Throughput and TPOT A ≈ B (0.01 %). The NoC / wire energy split differs most: wire share 0.091 vs 0.069, noc 0.048 vs 0.075.

### Freshness (2026-10-22 JST)
- 110 tracked orgs since 2026-10-21: nothing new. Raon-1-mini-preview-mlx and the Flint int4 / GGUF uploads are derivatives.
- Newest HF uploads overall (text generation, VLM, TTS since 2026-10-22 00:00 JST) are community fine-tunes, quantisations and conversions.

## [0.320.0]

### Design verdict report (new): one design, everything the tool knows, labels attached
- `core/report.py` `design_report(r, body, row=None)` builds the report from one `insight_dse` row (default: the cheapest $/M row, else the best perf/W, else the front head; `report_row` picks another).
- Sections: design, performance at SLO, energy, shoreline buildability, thermal / power, yield and cost per good package, TCO with tornado bars, joint adverse / favourable corners and skipped inputs.
- Every number is an item `{key, label, value, unit, tag, src}`. Tags are 来源 / 二手 / 推导 / 假设 / 近似 / 用户输入.
- **Never invented:** a value the run does not have is `None` with a `missing` reason. Examples: no TCO preset, an unpriced memory type (LPDDR6), no D0. TCO secondary items inherit the TCO-missing reason. The `missing` list collects them all.
- **Near-tie warnings** are mechanism checks, not heuristics:
  - the DSE compare `tco_flips` / `tco_corners` for this row's comparison;
  - `_pair_check` against the cheapest other priced design and against the best alternative memory of the same design (`scan_flip` over the TCO inputs plus both joint corners);
  - SLO, shoreline, flux and package-W margins within the DSE tie (1 %).
- Exports:
  - API: `POST /api/report` (`report_format` json | markdown | both, default both).
  - CLI: `scripts/design_report.py --body f.json | --example [--row i] [--md out.md] [--json out.json]`.
  - UI: a 设计结论报告 button next to the DSE run button downloads .md + .json.
- Example (`--example`: 1P qwen3-8b decode b64, dies 1 / 2, CoWoS-S, accel_2026, D0 0.1):
  - Row 3 (32×128×128, 4 × HBM3E) at $/M 0.008479.
  - Two warnings: mono → split at D0 0.182, and row 0 (128×64×64) is +0.17 % (within the tie, no flips).
  - Corners: adverse +400 %, favourable −44 %.

### Parked item re-examined: DeepGEMM masked grouped FP8 at 256+ rows (0.73)
- **Mechanism:** `extkern` priced `grouped_h800` (a DeepGEMM kernel) at the generic Hopper FP8 cap (1550 / 1979). Its scenario carried no `src`, so the DeepGEMM per-shape ceilings (`extval.DEEPGEMM_0325`, DeepGEMM README, H800, same (N, K) 4096×7168 / 7168×2048 and the g-variants) were never applied. Those ceilings have applied to every DeepGEMM validation row since 0.2xx.
- **Fix:** the grouped rows run under `deepgemm_variant` with a DeepGEMM stack src, the same `DEEPGEMM_SRC` scoping rule. Nothing is fitted.
- **Cross-check 「来源」:** DeepGEMM README @ 55ab91f (2025-03-25) masked-layout table (`ext_kernels_moe.json` `readme_masked_h800`, 6 rows). It matches the DeepSeek_Simulator CSV within 1.5 % (ratio 0.985–0.997), so the measured data are right.
- **Old → new** (model / measured, geometric mean):

  | rows / group | 0.319 | 0.320 |
  |---|---|---|
  | 256+ (all) | 0.729 | 0.86 (0.93 with the per-kernel floor) |
  | 256 | 0.844 | 0.85 |
  | 512 | 0.643 | 0.839 |
  | 1024 | 0.672 | 0.882 |
  | 2048 | 0.726 | 0.973 |
  | 4 – 128 | 1.03 | 1.03 (unchanged) |

- The remaining 0.84–0.88 at 256–1024 rows is masked-layout specific: the per-shape ceilings are dense / contiguous figures. It stays unfitted (AGENTS §247). The m = 2048 rows reach the ceiling (0.97).
- **Scope:** diagnostic only. `extkern` does not feed the cost model, and no evaluation, DSE or TCO number changes. `extkern.summary(deepgemm=False)` reproduces 0.319 (0.729). `ext_validation.json`: only `kernels.default` and the version stamp change.

### Fixed
- `energy_dse.D0_NOTE` still said package / bonding yield was not counted ("封装 / 键合良率未计"), which has been stale since 0.319. It now points to the cost item (carrier × attach × bond; not counted when no cost preset is selected).

### Freshness (2026-10-22 JST)
- **KRAFTON Raon-1-mini-preview** is now public (config.json 200, API gated false; it was HTTP 401 through 0.319). Added as `raon-1-mini-preview` (provider `krafton`, Apache-2.0):
  - **Model type** `raon1_mini`:
    - Layers: 25 = 18 RDA + 7 NoPE gated MLA. `rda_config.mla_layers` is 1-based; the release header puts the MLA tensors at layers 3 / 7 / … / 23 / 24.
    - RDA is the KDA layout: per-channel low-rank decay of rank head_dim 128, full-rank g_proj, short convs of kernel 4, plus `rda_w_proj` h → heads.
    - MLA: q_lora 576, kv_lora 512 + rope 64, 40 heads, output gate.
    - FFN: 1 dense layer (12032) + 24 MoE (128 experts top-6, sigmoid router with bias, 1 shared expert 1088).
  - **Raon_mHC:** 4 streams, two sites per layer, selector [2N + (N−1)² = 17, N·h] plus 21 scalars per site. The chart is closed-form (no Sinkhorn) and costed as one normalisation pass 「近似」. New `Layer.hc_rows`, used by `ir._hc_ops`. The final collapse is the stream mean.
  - SiTU-GLU is costed as SwiGLU 「近似」.
  - Routed experts are MXFP8 (E4M3, e8m0 per 32 → 8.25 bits); the rest is bf16. Default inputs are bf16 (W8A16; W8A8 on sm_100 per the modeling docstring is a what-if).
  - Params 28,783,218,602 = the release header exactly; active 4.317 B (vendor: 4.3 B).
- Other tracked orgs (110): nothing new. Newest HF uploads are derivatives only.

## [0.319.0]

### Package / bonding yield: a sourced per-component model replaces assembly yield 1.0 (default change)
- **Mechanism** (`energy_dse.asm_yield`, `package_cost`; `cost.asm_model`):
  - Y_asm = Y_carrier × y_attach^HBM stacks × y_bond^dies, with y_bond = microbump attach or hybrid bond (`cost.bond`).
  - **chip_first** (default, CoWoS): a failed carrier or attach scraps the known-good dies and the HBM stacks already on it. **chip_last** (`cost.flow`): the carrier is tested first, so only attach failures scrap them.
  - Rework fraction ρ (`cost.rework`, default 0). Scrap factor k = (1 − ρ(1 − Y_exposed)) / Y_exposed.
  - Per good package: good-die $ × k_kgd (`kgd_scrap_usd` column). HBM device $ × k_mem is applied in the TCO memory term (rows `mem_scrap_k`, `tco.mem_dev_usd`). LPDDR is off the carrier, so k_mem = 1.
  - Packaging $/mm² and test stay per shipped package: the Epoch / Silicon Analysts figures are top-down per package (interposer yield is inside them), so they are not divided by Y.
- **Calibration:**

  | Term | Value | Basis |
  |---|---|---|
  | CoWoS-S carrier | 0.985 | TSMC 98–99 % (OCP APAC 2026, via Silicon Analysts 2026-08-11) 「二手」 |
  | Microbump attach (per HBM stack and per die) | 0.998 | Epoch AI CoWoS-S packaging yield 95–99 % (mid 0.97); H100 = 1 die + 6 HBM sites → (0.97 / 0.985)^(1/7) 「推导」 |
  | CoWoS-L carrier | 0.81 | Epoch AI B200: packaging yield loss ≈ $1,000 on ≈ $3,800 of dies + HBM → Y ≈ 0.79 「推导」; ÷ 0.998^10 「推导」 |
  | Hybrid bond (per die) | 0.97 | ECTC D2W > 95 % (Gao et al. / Theil et al. 2020) and 96–99 % void-free (ECTC 2023) 「二手」 |
  | Scrap on failure | ρ = 0 | Epoch (KGD + HBM lost) 「二手」; rework 0 「假设」 |

  - Check: B200 (2 dies + 8 stacks, CoWoS-L) Y = 0.794; H100 (CoWoS-S) Y = 0.971.
  - Not used: the Chiplet Actuary defaults (SI bonding 0.95 per chip), because 0.95⁷ = 0.70 is inconsistent with Epoch's 95–99 % per CoWoS-S package.
- **Default change (old → new):** assembly yield 1.0 「假设」, nothing scrapped → component (CoWoS-S 0.985 / CoWoS-L 0.81 carrier, 0.998 attach, microbump, chip_first, ρ 0).
  - **Switch:** `cost: {"preset": …, "asm_model": "flat"}` reproduces 0.318 exactly. It was checked on all 7 chiplet cases plus 36 sweep runs × 2 TCO presets: 1,494 rows, usd_good_pkg / $/M / fronts identical.
  - CLI: `--asm flat` in `chiplet_example.py` and `tco_sens.py`. UI: the 封装良率 select.
- **Effect** (chiplet case 0, 1P qwen3-8b decode, 8 × HBM3E, CoWoS-L, D0 0.1):
  - Cheapest design usd_good_pkg $2,013 → $2,116.
  - $/Mtok 0.006594 → 0.007372 (+11.8 %), mostly HBM scrap (k_mem 1.257 × $3,360 of HBM).
  - Package-only break-even D0 vs the 654 mm² mono die: 4 dies 0.12 → 0.100, 2 dies 0.14 → 0.122. Chip-first scrap is proportional to the silicon $ at risk, which is largest for the big mono die.
- **Verdict changes old → new** (`scripts/tco_sens.py --asm-compare`):
  - n5_cowos_l, accel_2026:
    - CASES $/M mono/split: 0 / 14 changed.
    - SWEEP cheaper-by-$/package and by-$/M: 0 / 144.
    - Memory choice by thr / capex / $/M: 1 / 72, which is tp8/400 D0 0.30 MAC × 1, by $/M 8 stacks → 6 stacks. D0 0.3 lies outside the sourced D0 range.
  - n5_cowos_l server_2026, and n5_cowos_s with either preset: 0 changes.

### TCO sensitivity: joint corners
- `tco_sens.corner(base, side)`: every applicable input goes to its adverse ($/M-raising) or favourable end together. Low is adverse for life, carrier yield and attach yield. Memory goes to the end of each row's own type range.
- Two new inputs:
  - Carrier yield, per carrier (`CARRIER_RANGES`):
    - CoWoS-S 0.963 (Epoch low 0.95 ÷ 0.998^7 「推导」) – 0.99 (TSMC 98–99 % 「二手」).
    - CoWoS-L 0.663 (Epoch low 0.65 ÷ 0.998^10 「推导」) – 0.985 (the TSMC 2026 claim, which may cover the interposer only 「二手」).
  - Attach yield 0.995 (Epoch CoWoS-S low ÷ 0.985, 7th root 「推导」) – 1.0 (no attach loss 「假设」).
- Where it appears: `tco.sens.corners` (adverse / favourable $/M and Δ), compare entries `tco_corners` (verdict at each corner, DSE and `/api/tco_sens`), the corner line under the UI bar charts, and `scripts/tco_sens.py --study` (`corner_flips`).
- **Results** (component default), as flipped / total:

  | Study | Single input | Joint adverse | Joint favourable |
  |---|---|---|---|
  | accel_2026, cases | 3/7 (D0 0.150–0.155; 0.318: 0.162–0.167) | 3/7 → split | 0/7 |
  | accel_2026, sweep | 1/44 | 1/44 | 0/44 |
  | accel_2026, memory choice | 5/24 | 2/24 | 2/24 |
  | server_2026, cases | 3/7 (D0 0.186–0.194) | 3/7 | 0/7 |
  | server_2026, sweep | 1/44 | 1/44 | 0/44 |
  | server_2026, memory choice | 2/24 | 0/24 | 0/24 |

  - At accel_2026 both memory-choice corner flips are the D0 0.3 cells, which go to 8 stacks.
  - The tp8 MAC × 2 4-stack choice at D0 ≤ 0.2 holds at single inputs and at both corners.
  - New near tie: tp8/400 MAC × 1 at D0 0.05–0.2 is 8 stacks vs 6 stacks within single-input ranges (it was not under flat). It stays 8 stacks at both corners.
  - Case 0 cheapest design: adverse corner +283 %, favourable −52 %.

### Fixed
- `scan_flip` raised `ValueError` (min of an empty list) when the base value lay outside the scanned range and the label changed between the range end and the base. This crashed 0.318 DSE runs with `tco="server_2026"`, D0 0.3, tp 1 (all three HBM3E stack counts). The scan now bisects between the nearer end and the base, and marks the threshold `outside`.

### Freshness (2026-10-22 08:29 JST)
- `freshness_sweep.py`: 110 orgs, nothing new.
- HF newest: derivatives only.
- KRAFTON Raon-1-mini config.json still 401.

## [0.318.0]

### TCO sensitivity: which sourced input moves $/M most, and do the 0.317 verdicts flip
- **New `accel_dse/core/tco_sens.py`.** Each input is varied one at a time across its sourced range, with the others at the design's base values.
  - **Mechanism:** throughput, package power and the idle floor are fixed (none of these inputs changes them). Only cost terms move, through the same functions as before: `package_cost` (wafer $, D0 → Poisson yield, CoWoS $/mm²) and `tco_point`.
  - `row_usd(row, base, overrides)` reproduces the row's $/M exactly at the base (checked; a mismatch raises).
  - `tornado(row, base)` returns bars sorted by the larger |Δ $/M|.
  - `scan_flip(verdict, input, range)` grid-scans the range (33 points) and bisects every verdict change.
  - `compare_flips` does this for each mono / split $/M verdict of a DSE.
- **Ranges (`RANGES`, `tco.MEM_RANGES`), each end tagged:**

  | Input | Low | High | Basis |
  |---|---|---|---|
  | Wafer N5/N4 | $16,000 | $21,300 | Silicon Analysts band citing CSET / Morgan Stanley 「二手」 |
  | D0 | 0.05 /cm² | 0.20 /cm² | High: TSMC 2020 N5 pre-volume ≈ 0.2 「二手」; low: mature 「假设」 (no public mature N5 value) |
  | CoWoS $/mm² | × 0.909 | × 1.091 | B200 CoWoS-L $1,000–1,200 (Epoch AI 「二手」) ÷ $1,100; CoWoS-S borrows the relative band 「假设」 |
  | $/kWh | 0.0546 | 0.1385 | EIA state industrial minimum (NM, via DC Hub 「二手」) / EIA national commercial 「来源」 |
  | PUE | 1.09 | 1.58 | Google 2025 fleet TTM 「来源」 / Uptime 2021–2025 maximum (2023) 「来源」 |
  | Life | 4 years | 6 years | SemiAnalysis / Meta analysis 「二手」 / Alphabet 10-K 「来源」 |
  | Host share | 0 | $7,500 + 575 W | accel → server_2026 「推导」 |
  | Capital rate | 0 | 9.33 % | Meta analysis WACC 「二手」 |
  | HBM3E $/GB | 10.28 | 18.0 | GPUQuant contract low 2026-08-31 / Fubon via TrendForce 2026-08-13 「二手」 |
  | HBM4 $/GB | 15.56 | 32.0 | SK hynix mid-$500s per 36 GB stack (Seoul Economic Daily 2025-11-05 via Silicon Analysts) / Fubon 「二手」 |
  | LPDDR5X $/GB | 7.8 | 21 | LTA floor / ceiling (Wccftech 2026-05 「二手」) |

- **Where it appears:**
  - DSE with `tco`: `tco.sens` (tornado of the cheapest $/M row, or `tco_sens_row`) and per mono / split compare entry `tco_flips` (input, thresholds from → to).
  - eval `tco.sens`. Cost inputs are skipped when the user gives `pkg_usd`.
  - New route `POST /api/tco_sens` (insight_dse body plus optional `tco_sens_row`): the tornado, the design, the compare verdicts with their flips, and all ranges with sources.
  - CLI: `scripts/tco_sens.py` (`--case i` tornado, `--study` flip analysis).
  - UI: a bar chart under the DSE summary (with the flipping inputs and thresholds) and under the chip-panel TCO line.
- **Results (chiplet example, n5_cowos_l, accel_2026):**
  - **Biggest movers** (case 0 cheapest design, mono 8 × HBM3E):
    - Host share 0 → server: +134 %.
    - Capital rate 0 → 9.33 %: +20 %.
    - $/kWh: −13 / +19 %.
    - Memory $/GB: −17 / +1 %.
    - Life: +16 / −11 %.
    - PUE: −10 / +1 %.
    - D0: −1.4 / +4.5 %.
    - Wafer $: −0.3 / +1.2 %.
    - CoWoS: ±0.8 %.
  - Chip economics (wafer, D0, CoWoS) move $/Mtok by a few percent; facility, finance and host terms by 10–130 %.
  - **Mono / split verdicts (7 priced cases, all mono under $/M):**
    - **3 flip to split inside the D0 range**: case 0 at D0 ≈ 0.162, case 3 at ≈ 0.167, case 5 at ≈ 0.163 /cm², i.e. at pre-mature defect density.
    - No other input flips any case.
    - With server_2026 as the base, no case flips (the host share dilutes the package term).
    - The MAC × 2 entries are only_split (the mono die is unbuildable), which no cost input can change.
  - **Sweep cheaper option (44 cells with both sides buildable; D0 is the grid axis there, not scanned):** 1 flips. 4 × HBM3E, tp 1, D0 0.05, MAC × 2 goes mono → split at wafer ≥ $19,800 or $/kWh ≥ 0.1245.
  - **4- vs 8-stack choice:** the 0.317 result (tp 8, MAC × 2, D0 ≤ 0.2: $/M picks 4 stacks) **does not flip for any input in its range**. tp 1 always picks 8 stacks; tp 8 / 900 MAC × 1 always 6.
    - Flips occur only in the D0 0.3 cells, which lie outside the sourced D0 range. At tp 8 / 400 MAC × 2, 4s → 8s at wafer ≥ $18,390, memory ≤ $15.33/GB, $/kWh ≤ 0.060 or host ≥ 0.46.
- Limits: one input at a time (joint corners are not searched); package / bonding yield loss is still not modelled (asm_yield 1).

### Memory $/GB: three types newly priced (default change, old → new)
- **HBM3: none → $13.0/GB 「二手」**: Silicon Analysts 2025 contract, a model estimate on TrendForce's stated 2025 hike, low confidence; range 12.0–13.6.
- **HBM4E: none → $28/GB 「二手」**: UBS 2027 forecast $3.5/Gb (2026-06-22, via Sina). It is a forecast, not a transaction price. Range up to $40 「推导」 from Taiwanese analysts' 2027 HBM4 $4.16/Gb × the +20 % HBM4E premium.
- **LPDDR5: none → $10/GB 「二手」**: SemiAnalysis, 2026-05, via Wccftech. This is the pre-2Q26-increase level, and the source is loose about LPDDR5 vs 5X; range up to $19.8 「推导」 (× TrendForce 2Q26 +93–98 %).
- **LPDDR6 stays unpriced**: only a rumoured +20 % premium over LPDDR5X (Weibo tipster via Wccftech / eTeknix), no $/GB, so nothing is derived from it.
- Mechanism for the change: public, dated $/GB now exist for these types. Rows of these types now get a $/M instead of 「缺」. Rows of other types are unchanged.

### Freshness (2026-10-22 07:40 JST)
- No new release from the 110 tracked orgs.
- HF newest: derivatives only (Ornith-1.5-122B MLX / GGUF / NVFP4 quantisations of a catalogued model; Qwen3.8-27B SLQ quantisations; a MiniCPM-V-4.7 MNN conversion; third-party fine-tunes).
- Raon-1-mini-preview: still 401.
- Tests: new test_core_318 (9); test_core_317 now asserts only LPDDR6 unpriced.


## [0.317.0]

### Cost per unit of work (TCO, opt-in): $/M unit and unit/s/W per design
- **New `accel_dse/core/tco.py`.** `$/M unit = (annual capex charge + annual energy cost) ÷ units per year × 1e6`, per package at its operating point.
  - **Capex** = `usd_good_pkg` (0.310 chiplet economics: Σ good dies + packaging incl. HBM interposer footprint + test; **no memory devices**) + **memory GB × $/GB** (the only memory term) + `system_usd_per_card`.
  - Straight-line over `life_years`, or an annuity with `capital_rate`.
  - **Energy** = IT power × PUE × 8760 h × $/kWh.
    - IT power = util · (P_pkg + P_sys) + (1 − util) · (P_pkg,idle + P_sys).
    - P_pkg is the **complete package power** (`thermal.package_power`, the function the thermal check and the 0.316 power cap use) at the **capped** operating point. P_pkg,idle is the same function on the static alone.
  - **Units/year** = throughput at the scenario's batch × util × 3600 × 8760. SLO-violating points are already infeasible.
  - Also reports `units_per_s_per_W` (thr ÷ P_pkg, module input) and `units_per_s_per_facility_W`.
  - Cost basis is manufacturing cost (no vendor margin) 「假设」: the designer's view.
- **Defaults, every one labelled:**

  | Term | Value | Basis |
  |---|---|---|
  | PUE | 1.54 | Uptime Institute Global Data Center Survey 2025 weighted average, n = 681 「来源」 |
  | $/kWh | 0.0891 | EIA Electric Power Monthly Table 5.3, industrial, rolling 12 months to Jul 2026 「来源」 |
  | Life | 5 years | Amazon moved a server subset 6 → 5 years in 2025; Meta 5.5, Alphabet 6 (10-K 「来源」); range low end 「假设」 |
  | util / capital_rate | 1.0 / 0 | 「假设」 (Meta 24k-H100 analysis: 90 %, WACC 9.33 % 「二手」) |
  | HBM3E / HBM4 | $17.5 / $31.5 per GB | TrendForce citing Fubon, 2026-08-13 「二手」 |
  | LPDDR5X | $19.5 per GB | SemiAnalysis ~$10 base × TrendForce 2Q26 +93–98 % 「推导」 |
  | LPDDR5 / LPDDR6 / HBM3 / HBM4E | none | No recent public price: the row is unpriced and needs the user's figure, never 0 |
  | `server_2026` system | $7,500 and 575 W per card | SemiAnalysis H100 ($250k all-in − $190k server) ÷ 8 「推导」 (lower bound); DGX H100 10.2 kW max − 8 × 700 W, ÷ 8 「推导」 (nameplate upper bound) |

  - `accel_2026` (default) has no system terms. The user's values win: `mem_usd_per_GB` from the DSE first, then `tco.mem_usd_per_GB`.
- **Where it appears (opt-in; defaults unchanged, no `tco` key appears without it):**
  - `energy_design_search(..., tco=)` (needs `cost`): row `tco` (breakdown and shares), `front_tco` (thr ↑, $/M ↓), `best_tco` / `tco.best`, chiplet `verdict_tco` / `d_tco`, notes.
  - `api_eval` body `tco` (`pkg_usd`, or `cost` priced on the shoreline die with Poisson yield at `d0_per_cm2`, default 0.10).
  - `insight_dse` `tco`.
  - UI: chip panel 「TCO」 select plus a $/M line; DSE 「TCO 轴」 select, `$/M` and `/s/W` columns (◆ cheapest, ★ front), summary and verdicts.
  - `chiplet_example.py --tco accel_2026|server_2026`.
- **Adversarial checks** (test_core_317):
  - Memory is priced once: package $ is independent of the memory price; capex = package $ + GB × $/GB exactly; no memory key exists in COST_PRESETS; a DSE memory price is reused, not added.
  - Power is the complete package power once: = thermal pkg_W, = cap_pkg_W on capped rows, > card W. Energy is exactly IT W × PUE × h × $/kWh, with no separate DRAM, static or cooling term (PUE covers cooling).
  - The $/M front and best point contain only feasible and buildable rows. They exclude rows kept by `include_unbuildable` (decode, 8 × HBM3E, `shore_strict`: rows 0–2 stay on the main front, off the $/M front) and thermal-out rows (cooling preset).
  - Eval uses the capped operating point (package W = TDP 150 W, throughput × factor). `tco` without `cost` / `pkg_usd` is rejected.
- **Verdicts under $/Mtok** (`chiplet_example.py --tco`, n5_cowos_l, buildable only; same under `server_2026`):
  - **Mono / split: unchanged** in all 14 case × MAC combinations. Mono stays cheaper per token by +0.7 – +25.5 % (case 2, L32: D2D-bound split); `only_split` stays where the mono die is over the reticle.
  - **Cheaper option (0.312 sweep, 72 cells): 16 flip split → mono**, all at tp = 8.
    - At D0 0.2–0.3 the split package is cheaper to make (4 × HBM3E, D0 0.3: $1,865 vs $2,373), but its best split serves 2,100 vs 3,572 tok/s (−41 %), so $/Mtok is +55 %.
    - Package $ is only ~40 % of TCO, so the yield saving cannot pay for lost throughput.
  - **Memory choice (4 / 6 / 8 × HBM3E):**
    - By $/good package + memory, always 4 stacks.
    - By throughput vs by $/Mtok: they agree at tp 1 (8 stacks). They differ at tp 8 MAC × 2 for D0 ≤ 0.2 (6 of 8 cells): throughput picks 8 stacks, $/Mtok picks 4.
    - Mechanism: 8 stacks give +4 % throughput (3,728 vs 3,586) but force a split (the mono die with 8 PHYs is over the reticle) and double memory $ ($3,360 vs $1,680) → $/Mtok +39 %.
    - At tp 8 / 900 GB/s MAC × 1, 6 stacks win both.
- **Freshness (2026-10-22 JST):** no new release from tracked orgs. DeepSeek-V4.1-Flash-1017 (2026-10-17) is already catalogued; the new HF uploads are re-uploads, quantisations or fine-tunes. Raon-1-mini-preview: still 401.
- Tests: new test_core_317 (9).


## [0.316.0]

### Power cap in the same package scope as the thermal check; thermal front exclusion made visible; freeze-aware timing guards
- **The DVFS power cap now compares the complete package power**, the same number the thermal check judges. Both sides call one function, `thermal.cap_power_fn(die_mm2, dies, scope)` → `thermal.package_power`: card W + whole-card static floor (0.088 W/mm², less already-charged leakage) + VR / PDN loss (η 0.85), as defined in 0.315. So the two cannot diverge.
  - `powercap.dvfs_cap(..., pk)`: every comparison against the cap uses `pk(card W, static W(f), idle W)`. The duty-cycle branch inverts the same function with `thermal.card_W_for_cap` (bisection; `pk(c) ≥ c`, so the root lies in [0, cap]). It is infeasible when floor + loss alone exceed the cap.
  - `energy_report(..., pk)` / `_power_cap`: `scope`, `cap_card_W`, `pkg_needed_W`. DSE rows: `cap_pkg_W`; the reject reason reads 「封装口径」.
  - On a capped DSE row, `thermal.pkg_W` now equals `cap_pkg_W` (checked on all 42 capped example rows, max relative difference 0) and stays ≤ the budget. 0.315 capped the card W, so its capped "700 W" example rows sat at 896–900 W of package power.
  - **Switch:** `power_scope` ∈ {`package` (default), `card`}, on `energy_design_search`, the `eval` / `insight_dse` API and `chiplet_example.py --power-scope`. `card` reproduces 0.315 exactly: 0 differences on 14 case × cooling combinations. The UI checkbox is 「功耗预算按卡（0.315 口径）」.
  - Default changed for a mechanism reason: a shipped TDP is defined at the module input (OCP OAM r2.0 §8.4 「来源」, 0.315), so a budget compared to card W silently allowed static + VR on top. The what-if lever `lever_power_cap` stays card-level by design (a relative ±8 W probe).
- **Capped DSE examples, old (card) → new (package)**, `chiplet_example.py` cases 0–6 (126 rows, 700 W budget):

  | Cooling | Capped rows | Feasible | Front points | mono / split verdicts |
  |---|---|---|---|---|
  | none | 14 → 21 | 105 → 105 | 25 → 25 | unchanged |
  | air_oam | 14 → 21 | 83 → 94 | 26 → 24 | unchanged |

  - Cases 1 / 6 (qwen3-8b prefill tp, L0): 9 rows change, throughput × 0.758–0.936. bow-256 2- and 4-die are newly capped. With no cooling the front stays [12].
    - air_oam front [4, 17] → [12]: the MAC×2 2- / 4-die UCIe rows 12, 13, 15 and 16, thermally over at their 0.315 card-capped point (≈ 900 W package), become thermally feasible once the cap holds package W ≤ 700 W.
  - Case 3 (prefill L64): rows 11, 15 and 16 are newly capped (× 0.777 / 0.775 / 0.878). air_oam front [6, 16] → [6, 15].
  - Decode cases 0 / 2 / 4 / 5: unchanged (no row near the cap).
  - Unit tests pinned to the card scope, because they test other mechanisms at card-W budgets on 1200–3000 mm² dies:
    - test_core_302: under package scope 2× MAC @ 0.6 f (286 W card) is capped too and still beats 1× (56.8 k vs 50.9 k).
    - test_core_306: the package-scope front is 128 × 128 B only.
    - test_core_307: the package scope runs 2048 × 32 B at f × 0.52, 1.80 k tok/s (card 0.93, 3.19 k).
- **Thermal result in the DSE front, made explicit.** Since 0.313 a row over the selected cooling preset is unbuildable and leaves every front (like a shoreline-unbuildable row; `include_unbuildable` keeps it). With no cooling preset, thermal is reported only.
  - New: `front_thermal_out` marks rows that would be on the front without the cooling filter. The result adds `n_front_thermal_out` and `power_scope`, plus notes stating the rule and the cap scope.
  - Example: qwen3-8b prefill, card scope, air_oam → 1 point (row 12, flux 0.98 > 0.86 W/mm²); package scope → 0 (the cap already keeps it in).
  - UI: the DSE summary shows the cooling preset, 热超限 N 点 and how many were on the front, plus the cap scope. The table gains a 「热 / 封装 W」 column (✓ / ✗ + package W, flux in the tooltip) and the ✗热 front marker. The eval power-cap line shows the package W.
- **Freeze-aware timing guards** (`scripts/freeze_clock.py`). A sampler thread (50 ms) counts a wake-up gap ≥ 0.5 s as frozen only when the process burned < 20 % CPU over it 「假设」 (it did not execute at all: a host / VM freeze, SIGSTOP). Frozen time is subtracted; everything else counts 1:1.
  - A sleeping / blocking hang and a CPU-bound hang (including one C call holding the GIL) are still charged in full. Tested by SIGSTOP, sleep and GIL-holding child processes.
  - Used by the test_core_184 fuzz watchdog (SIGALRM re-arms for the remainder when frozen time ate the budget), test_core_099 (< 20 s), test_core_209 (< 2 s) and test_core_210 (`_timed` wall bound). The CPU bound in 210 is unchanged.
- **Freshness (2026-10-22 JST):** no new release from tracked orgs since 0.315. The sweep window (since 2026-10-21) lists only items already handled. Raon-1-mini-preview config.json still HTTP 401.
- Tests: new test_core_316 (7).


## [0.315.0]

### Complete package power for thermal / power buildability; suite-time check; Flint
- **Package power is now complete and in the shipped-TDP scope** (`thermal.package_power`). Judged package W =
  - modelled card W (energy_report, unchanged)
  - + **whole-card static floor, always on.** The floor is `STATIC_FLOOR_W_MM2` 0.088 W/mm² × logic-die area × VR_EFF (the delivered side), less the static already charged (SRAM / NoC-router leakage, HBM standby; same no-double-count rule as energy.py).
    - 0.088 is cross-checked by two vendors: H100 SXM idle 71.8 W 「二手」 ÷ 814 mm² = 0.088, and MI300X idle 131–133 W socket (AMD article, rocm-smi) 「来源」 ÷ 4 × 370 mm² IOD 「二手」 = 0.089 「推导」.
    - A table with `idle_W` / `idle_frac_tdp` uses that figure instead (board-level).
    - 0.314: static was included only with an idle figure.
  - + **VR / PDN loss**: delivered W × (1 / VR_EFF − 1), with **VR_EFF 0.85** 「近似」 (range 0.77–0.90).
    - Sources: ECCE 2024 48 V → 1 V single-stage VPD, 84.4 % full-load system efficiency (77.2 % at 0.65 V / 600 A) 「来源」; APEC 2024 48 → 1.8 V bus stage ≤ 95 % 「来源」 × a second stage.
    - Scope: OCP OAM r2.0 §8.4 defines TDP at the 44–59.5 V module input 「来源」, so a shipped TDP includes this loss.
    - A board-level idle figure carries no second loss.
- **Where each part sits:** HBM / DRAM devices are in the package but off the die (out of die heat); the host DRAM PHY is on the die (in die heat); VR loss is on the module (in package power, out of die heat). `die_W` = (delivered W − DRAM devices) ÷ dies.
- **Each flux precedent is judged in its own definition.** Old → new:
  - liquid card-level limit 1.70 → **0.96 W/mm²** (MI355X 1400 W ÷ 1464 mm² 「推导」).
  - HIR 2021 Table 1 thermal-budget rows become die-heat limits, `die_flux_W_mm2`: air 0.55, liquid 1.70 「来源」.
  - A point is buildable on flux when either precedent holds; `flux_ratio` = min of the two ratios.
  - Reason: 0.314 judged HIR's chip heat flux 1.70 against card-level power (VR and HBM included). Air is unchanged: 0.86 card-level decides.
- **Caveat narrowed.** 0.314's blanket "over is trustworthy, a marginal ok is not" is removed: the two omissions it covered (static, VR) are now in the power.
  - What remains is that the floor is measured at idle temperature (≈ 40 °C 「近似」).
  - `pkg_W_hot` applies the TSMC 3 nm leakage ratio 40 → 85 °C (× 3.21) to all static, an upper bound 「假设」. An "ok" row that is over under that bound gets `hot_ok` False and a note.
  - That bound is loose: it flags H100-like prefill at 627 W, while H100 ships at 700 W on air. So it is a flag and is not judged.
- **New report fields:** `card_W`, `power` (breakdown), `static_basis` (floor / idle / none / package), `static_add_W`, `vr_loss_W`, `pkg_W_hot`, `hot_ok`. `thermal_report(..., static_W, idle_W, basis="model" | "package")`: "package" judges a module-input power as given (shipped parts; every KNOWN_PART passes its preset).
- **Verdict effect (old → new)**, 80 chip × memory × phase cases (qwen3-8b, tpuv4i_7nm table without idle, 3 presets):
  - **air_oam / air_hgx: 8 → 12 over (4 flips ok → over each)**: 1P and 1P-lp4x64 prefill with 8 × HBM3E (card-level 0.78 → 0.99 W/mm²) and with HBM3 5-stack (0.85 → 1.08).
  - The 8 already over stay over (0.88–0.97 → 1.12–1.23).
  - liquid: 0 → 0 (highest die-heat ratio 0.72).
  - Package / card ratio: 1.26–2.35, median 1.38. The floor dominates on the ~100 W 100T chip.
  - With the H100 idle fraction in the table: 20 → 20 over, 0 flips.
  - H100-like: complete package power is 627 W prefill and 243 W decode, under its 700 W TDP. The floor path and the measured-idle path agree to < 1 %.
- **Suite wall time (0.314: 452 → 553 s) was load noise.** The unchanged 0.314 code re-run gives **446 s**, 1670 passed.
  - The 0.314 excess sat in 8 small files (test_core_059 / 087 / 144 / 269 / 273 / 276 / 295 / 296), each ~4.5 s → ~113 s in one concurrent window. In the rerun they take 3.9–4.9 s each.
  - The median per-file ratio against 0.313 is 1.00. Steal was 0.01 % over the rerun (/proc/stat, 5 s samples).
  - Cause: host-level VM freezes. A 2 s sampler stopped for 123 s and 47 s while /proc/stat advanced ≈ 3.6 CPU-s, and the virtio balloon moved ≈ 2 M pages each way. Steal and cgroup throttling were 0. Every concurrently running file gains the freeze length.
  - Fix for the real part: `scripts/test_durations.json` (stale since 0.238) is refreshed, so test_core_237 (≈ 120 s, ranked at 28 s) no longer starts last and sets the tail. This is a deliberate exception to the reset-durations step.
- **Freshness** (2026-10-22 JST):
  - + **Flint** (RecursiveSI, Apache-2.0, 2026-10-21 23:20 JST; coding-agent MoE) as model_type `flint`: 49 layers (2 dense + 47 MoE, 256 experts top-6 sigmoid + 1 ungated shared expert), 35 sliding-513 + 14 full NoPE layers, learned per-head sinks (`softmax_offset`), q / k norm, bf16.
    - 39.785 B, param_err 0. Active 2.82 B = the card's 2.5 B plus the input embedding.
    - New provider `recursive`.
  - Raon-1-mini-preview: still gated (HTTP 401).
- Tests: test_core_315.py (8). test_core_313 / 314 thermal asserts moved to 0.315 semantics (old values above). Releases 189 → 190 (llm 163 → 164); catalog total 197 → 198.

## [0.314.0]

### Thermal model audited against shipped parts; shore_strict evidence; OvisOCR3
- **Thermal: one flux definition on both sides.**
  - 0.313 judged the die-level flux ((card W − DRAM-device W) ÷ area) against 0.85 W/mm². That limit equals H100's *card-level* figure (700 W ÷ 814 mm² = 0.86), so the two sides used different definitions.
  - `flux_W_mm2` (the judged value) is now card-level: package W ÷ logic-die area, the only definition in which a shipped part's figure can be formed, since neither its HBM power nor its VR loss is published.
  - The die-level figure stays reported as `die_flux_W_mm2`. New fields: `flux_ratio`, `static_included`, `notes`, `flux_basis`.
  - Both figures are averages; hot spots are not modelled. A shipped part's average already carries its own hot-spot ratio 「近似」.
- **Thermal: what the modelled power contains.**
  - Dynamic MAC / vector / SRAM / SLC / NoC + wire / clock / DRAM device / DRAM PHY / D2D / link energy, SRAM and NoC-router leakage, and HBM standby.
  - Whole-card static (logic leakage, uncore) only when the energy table has `idle_W` or `idle_frac_tdp` × `tdp_W`. The chiplet example and its scans drop it. No VR loss is modelled.
  - Both omissions make the modelled power low against a shipped TDP, so "over" is robust and a marginal "ok" is not. The report now says so in `notes` when static is missing.
- **Thermal: limits re-sourced.** Rule: the larger of the highest shipped average and the HIR 2021 Table 1 thermal-budget figure. The HIR uniform-flux idealisations are no longer used. Old → new:
  - Air flux 0.85 → **0.86 W/mm²**. Basis: H100 / H200 SXM 700 W air (Dell XE9680: 8 × H200 700 W, air, 35 °C inlet 「来源」) ÷ 814 mm² 「推导」. HIR's 0.55 W/mm² thermal-budget figure is exceeded by shipped parts. 0.313's "0.55–0.85 at a 50 °C budget" misquoted HIR: 0.85 is its uniform-flux idealisation.
  - `air_oam` package 600 → **1000 W**: MI350X OAM air 1000 W TBP (AMD 「来源」); Gaudi 3 HL-325L OAM air 900 W (Hot Chips 2024 「来源」). The OCP OAM r2.0 600 W recommendation is exceeded by both.
  - `air_hgx` 1000 W: source upgraded to the NVIDIA Blackwell datasheet 「来源」.
  - `liquid` 1400 W: MI355X direct liquid 1400 W TBP (AMD 「来源」).
  - Liquid flux 2.5 → **1.70 W/mm²** (HIR Table 1 water cold plate, 50 °C budget, 4 cm² chip 「来源」). Shipped liquid parts reach ≤ 0.96 (MI355X on ≈ 1464 mm² IOD 「二手」).
- **Thermal: known parts.** KNOWN_PARTS now covers H100, H200, B200 air / liquid, MI300X, MI350X, MI355X, Gaudi 3 (area unpublished) and TPU v5p / v6e (no official TDP or area; third-party estimates not used). Card-level average flux, air: H100 / H200 0.86 > MI350X 0.68 > B200 ≥ 0.58 > MI300X 0.51; liquid: MI355X 0.96 > B200 ≥ 0.70.
- **Thermal: verdict effect (old → new)** over 80 chip × memory × phase cases (qwen3-8b, tpuv4i_7nm dynamic table, 3 presets each): **0 changes**.
  - Over air: the same 8 cases in both versions (1P and 1P-lp4x64 prefill with 1 / 2 / 4 × HBM3E and LPDDR5X 8 × 64; card-level 0.88–0.97 W/mm², 1.03–1.13× the H100 envelope). Over liquid: none.
  - With the H100-measured static floor charged (71.8 / 700 × 700 W 「二手」), the 1P prefill configurations over air go from 4 to 6 per chip (+ 8 × HBM3E 0.88, HBM3 5-stack 0.97).
- **shore_strict stays opt-in, now with evidence** (`budget.TIGHT_PRECEDENT`, shown as `precedent` on every "tight" shoreline report).
  - Every shipped HBM part found carries memory on at most two edges of a logic die: GV100 / GA100 / GH100 / B200, MI250X GCD, MI300X / MI350X IODs, TPU v5p (3 + 3), Trainium2 「二手」.
  - The only third-edge layout found is a patent claim: Google US 12,315,860 B2, claim 14 (12 stacks, some "along a third side of the second logic die") 「来源」.
  - Rubin Ultra's announced 4-die / 16-stack package places stacks on outer edges and was reportedly dropped 「二手」. No source excludes a third edge, so the default is unchanged.
  - The same patent cross-checks the 0.313 interposer bbox: a 650 mm² 5 nm die → about 2250 mm² interposer with surrounding HBM, against the model's 1P + 8 × HBM3E 654 → 2264 mm² (not fitted).
- **Freshness** (2026-10-22 JST):
  - + **OvisOCR3** (ATH-MaaS, Apache-2.0, 2026-10-21 11:36 JST): Qwen3.5-0.8B post-train, qwen3_5 VLM as released bf16, 0.752 B params.
    - It has the same config and header shapes as StartLux-Decision-v1-0.8B, except that it ships no MTP head and is all-bf16 where the Qwen base keeps a few fp32 norms. So it is **merged** (`MERGED["ovisocr3"] = "startlux-decision-v1-0.8b"`, the 0.251 / 0.288 precedent): resolvable by id and validated, but not listed.
    - Its preprocessor_config is identical to Qwen3.5's, so the 12-block ViT is built.
    - ATH-MaaS is added to the sweep's watched orgs.
  - KRAFTON Raon-1-mini-preview: config still gated (HTTP 401), so not added.
- Tests: test_core_314.py. test_core_313's thermal asserts moved to the 0.314 values (old values above). Release count 188 → 189 (llm 162 → 163); catalog total 196 → 197 (listed unchanged; + 1 merged).

## [0.313.0]

### Adversarial review of 0.310–0.312 (lane rate and lane edge, 4-sided memory, interposer box); thermal / power buildability
- **Review: how the scale-up lanes are counted.**
  - 0.312 counted every scale-up link in 100 Gb/s lanes at 0.285 mm (the PCIe5 lane edge). The only shipping 900 GB/s-per-direction link (NVLink5, B200) uses 18 links × 2 pairs × 200 Gb/s (224G PAM4; NVIDIA Blackwell whitepaper 「来源」). NVLink4 uses 100 G and NVLink3 uses 50 G (A100 / H100 whitepapers 「来源」).
  - A 224G lane is also wider: about 0.5 mm of die edge per duplex lane ("DSP/SERDES beachfront costs ~4 mm per Rx/Tx 8-lane port", OIF 448G Signaling for AI workshop, MaxLinear, April 2025 「来源」).
  - New defaults:
    - `Chip.io_su_lane_Gbps` 100 → 200 (the HBM3E-era lane). The H100-like presets are pinned to 100 (NVLink4).
    - `Chip.io_su_lane_mm` 0.285 → 0, meaning "by rate" (`budget.SU_LANE_MM`): 200 G → 0.5 mm 「来源」, ≤ 100 G → 0.285 mm 「假设」.
    - `io_su_lane_Gbps = 100` reproduces 0.312.
  - Per Tb/s the edge is similar (2.5 vs 2.85 mm), so the 0.312 verdicts hold. 1P + 8 × HBM3E at tp8 / 900 needs 16 + 36 lanes = 22.6 mm against 16.7 mm free (0.312: 16 + 72 lanes = 25.1 mm), so the monolithic die is still unbuildable, now on sourced numbers.
  - Lane-rate sensitivity (`chiplet_example.py --lane`): 100 G gives the same verdicts. 50 G also makes 8 × HBM3E tp8 / 400 monolithic unbuildable (64 lanes = 22.8 mm > 15.4 mm).
- **Review: 4-sided memory** (status "tight" counted as buildable since 0.310 「假设」). No shipping die carries memory on more than its two long edges (GH100, B200, MI300X IODs 「二手」).
  - The 6- and 8-stack monolithic verdicts rest on this assumption.
  - The new opt-in `shore_strict` (API, UI "岸线严格", `--strict`) counts tight as unbuildable. With it, 6- and 8-stack monolithic dies are unbuildable in every deployment (split only), the 7 example cases at MAC×1 become only_split, and the 4-stack verdicts are unchanged.
  - The default stays off: there is no source either way.
- **Review: interposer area.** 0.308–0.312 compared the sum of die and stack areas with the interposer limit. A stack row on 3–4 sides leaves the corner squares empty. The shoreline report now gives `footprint.bbox_mm2` (stack depth: HBM3/3E 11 mm, Micron 「来源」; HBM4 12.8 mm 「二手」), and the energy DSE checks single-die packages against max(sum, box).
  - Example: 1P + 8 × HBM3E is 2264 mm² against a sum of 1622. It is still under CoWoS-S 2700 mm², so no verdict changes.
- **Review: sources and labels.**
  - HBM3E stack 11 × 11 mm is now 「来源」 (Micron HBM3E product brief).
  - The host-side HBM PHY is narrower (HBM3 8 × 3 mm, HBM4 8 × 4 mm; Samsung via inlevel9 「二手」). The stack, not the PHY, sets the edge a stack occupies, because I/O cannot face a stack.
  - GH100's outline is not published. The H100-like 33 mm long edge follows from the 11 mm stack rule itself (noted as circular).
  - The 0.312 buildability count for tp1 was stale (see the 0.312 entry).
- **Review: UI.** The DSE "含不可建点" switch was sent only when dies were set. All buildability switches now go with every DSE.
- **Thermal / power buildability** (new `core/thermal.py`; reported always, judged only with `cooling`, so opt-in):
  - Package power = card power. Die heat = (card W − DRAM-device W) ÷ dies. Flux = die heat ÷ area.
  - Presets:
    - `air_oam`: 600 W (OCP OAM r2.0 「来源」), 0.85 W/mm² (air 0.55–0.85, IEEE HIR 2021 ch. 20 「来源」).
    - `air_hgx`: 1000 W (HGX B200 air 「二手」), 0.85 W/mm².
    - `liquid`: 1400 W (HGX B300 「二手」), 2.5 W/mm² (single-phase cold plate 1.7–2.5 「来源」).
  - Known part: H100 SXM 700 W on 814 mm² = 0.86 W/mm², at the air flux limit 「二手」.
  - Power delivery: the die current at 0.75 V 「假设」 is reported. It is not judged, because no public per-bump limit exists at current pitches.
  - Exposed in `/api/eval` `thermal` (+ the warning "不可建（热 / 功耗）"), each DSE row (`thermal`, `why_thermal`, `n_thermal_over`), the UI (chip panel "冷却" and `#chip-thermal`, DSE "冷却"), and `chiplet_example.py --cooling`.
  - Where it binds (tpuv4i_7nm energy, qwen3-8b prefill 4k):
    - 1P + 4 × HBM3E: 504 W on 571 mm² = 0.86 W/mm², over air and OK on liquid.
    - 1P + LPDDR5X 8 × 64: 0.93 W/mm², over air.
    - 1P + 8 × HBM3E: 0.75 W/mm², OK; the PHY area spreads the heat.
    - H100-like: 0.55 W/mm².
    - Decode stays at 0.1–0.25 W/mm².
- **Old → new** (0.312 → 0.313 defaults):
  - Shoreline status over 280 chip × memory × deployment cases (tp1 / 400, tp8 / 400, 450, 900): 0 changes.
  - Throughput (evaluate): 0 changes.
  - Chiplet break-even D0: identical to 0.312 (4 stacks 0.16 / 0.08; 6 stacks 0.16 / reticle; 8 stacks 0.12 at tp1 / 400, 0.13 at tp8 / 400, I/O-unbuildable at tp8 / 900).
- **Freshness** (2026-10-22):
  - HF sweep since 2026-10-21 (107 orgs): only Ling-3.0-tiny-UBTree (already in).
  - KRAFTON Raon-1-mini-preview (30B-A4B, Apache-2.0, 2026-10-21) has a gated config and custom `raon1_mini` code, so it is not added. KRAFTON is now watched by the sweep.
  - No public 224G die-edge figure beyond the OIF slide. 448G is still in OIF definition.
- Tests: test_core_313.py (6). test_core_312 pins `io_su_lane_Gbps = 100` (0.312 rule).

## [0.312.0]

### Shoreline feasibility as a first-class result; sized host + scale-up I/O edge; per-part scale-up link; 238 on CPU time
- **Buildability on every result**: `budget.shoreline_report(scn)` returns, per edge (W / E / N / S), length, memory units (whole HBM stacks / LPDDR channels), memory / D2D / I/O mm, used / free, role, plus overall `feasible` and `status` (ok / tight = memory beyond 2 edges / over / unknown) and the reasons. It is exposed as `Result.shoreline`, `/api/eval` `shoreline` (with the budget's die area when given; an infeasible eval adds the warning "不可建（岸线）：…"), and per energy-DSE row (`buildable`, `shoreline`, `shore`).
- **Fronts are buildable-only by default**: an infeasible row gets `why` and stays off every front / verdict. `include_unbuildable=True` (API key, UI checkbox "含不可建点", `chiplet_example.py --include-unbuildable`) keeps it (`why_shore`). Output has `n_unbuildable`.
- **Sized I/O edge** (`Chip.noc_mem_io_edges` default 1 → **-1 = sized**; 1 reproduces 0.311's whole reserved edge, 0 reproduces 0.310):
  - Host lanes: `io_host_lanes` = 16 (PCIe x16 per package, H100 datasheet 「来源」), split over the package's dies.
  - Scale-up lanes: Link.GBps × 8 ÷ `io_su_lane_Gbps` 100 「假设」. These apply only when the replica leaves the package. Lanes are counted in x4 macros, and a macro is never split across edges 「假设」.
  - Lane edge: 0.285 mm per duplex lane (`io_lane_mm` / `io_su_lane_mm`; Intel 32 Gb/s PCIe5 SerDes, CICC 2020, 285 µm/lane 「来源」). For 112G long-reach no public die-edge figure exists, so the same pitch is used 「假设」; XSR 0.05–0.14 mm/lane is the floor 「二手」.
  - Check: H100-like at tp8 with 450 GB/s needs 16 + 36 lanes (NVLink4 18 × 2 「二手」) = 14.8 mm.
- **Discrete fit and die shape (sized mode only)**: each edge holds whole units (HBM stack 11 mm; LPDDR channel 16 / 18.75 mm). A die without published geometry may be any rectangle of its area in the 26 × 33 mm exposure field with aspect ≤ 2 (`DIE_ASPECT_MAX` 「假设」; GH100 1.34, B200-class 1.2–1.4 「二手」), in 0.25 mm steps, squarest feasible first. D2D goes on the short or the long edges. Memory goes on the fewest non-D2D edges. I/O is placed first-fit into what is left.
- **Per-part scale-up link** (energy-DSE splits / NoC clusters): each die or cluster card gets Link.GBps / parts. ≤ 0.311 gave every part the whole link, i.e. K × the package's off-package bandwidth (and, now visible, K × the SerDes). Example: 1P + 8 × HBM3E prefill L64 at tp8 / 900, K = 4: 27.7k → 23.3k tok/s. At tp1 nothing changes.
- **Old → new** (840 cases as in 0.311; default 0.312 vs the 0.311 switch): 45 cases change, all up, 0 down.
  - 1P / 1P-lp4x64 + 8 × HBM3E / 8 × HBM4: memory edges 2 → 4. L16 / L32 throughput +100 % (attach links 10 → 20), J −42…−47 %.
  - 100T + 4 × HBM3E: 2 → 4 edges.
  - Buildability (70 chip × memory at tp1 / 400): 0.311 ok 42 / tight 13 / over 15; 0.312 ok 41 / tight 18 / over 11. *(Corrected in 0.313: shipped 0.312 gives ok 42 / tight 18 / over 10. 100T + 1 × HBM3E reshapes to 11.2 × 9.8 mm and holds its stack; the scan predates the shape freedom.)*
- **Chiplet verdicts, buildable only** (qwen3-8b decode, n5_cowos_l; `chiplet_example.py --sweep`). The split is cheaper from D0 (per cm²):
  - 4 × HBM3E: MAC×1 0.16, MAC×2 0.08 (tp1 / 400, tp8 / 400, tp8 / 900).
  - 6 ×: MAC×1 0.16; MAC×2 monolithic over the reticle (867–898 mm²), split only.
  - 8 ×: MAC×1 0.12 (tp1 / 400) or 0.13 (tp8 / 400). At tp8 / 900 monolithic is unbuildable (I/O 16 + 72 lanes = 25.1 mm > 16.7 mm left), split only. MAC×2 is over the reticle.
  - Example: 8 × tp1 at D0 0.1: monolithic $2013 (4 edges, buildable) < 4-die $2059.
  - 0.311 called 8-stack monolithic unbuildable (whole I/O edge). 0.312 restores "mono" for all 7 example cases at MAC×1 (as 0.310 had, now with the I/O checked).
- **UI**: chip panel has "I/O 岸线" (按通道 / 整边（0.311） / 不计（0.310）) and "主机 PCIe 通道", plus a per-edge shoreline table (`#chip-shore`). The DSE panel has the "含不可建点" switch, and its why column shows the shoreline reason.
- **test_core_238**: runtime bound measured in CPU time (`time.process_time`, as test_core_210), same 60 s limit, same checks.
- Freshness (2026-10-22): HF sweep since 2026-10-21 (107 orgs) found only Ling-3.0-tiny-UBTree (added in 0.311). UCIe 4.0 is not published (3.0 latest). JEDEC has no HBM4E standard (HBM4 JESD270-4A, SPHBM4 JESD330-4). Nothing new.
- Tests: test_core_312.py (7). test_core_311 pins `noc_mem_io_edges = 1`; test_core_213 / 214 expectations updated with 0.312 comments.

## [0.311.0]

### Audit of the 0.310 memory-edge default; host-I/O edge; per-edge interleave; robust headless smoke; Ling-3.0 + UBTree
- **Audit: the 0.310 "+49 %" at 100T + LPDDR5X 8 × 64, links 32 B/cycle is decode throughput per card**, 304.7 → 454.9 tok/s (qwen3-8b, batch 64). J/token went 70.05 → 62.91 mJ (−10.2 %), not +49 %.
  - Mechanism: 16 engines (R = C = 4), 1 GHz × 32 B links. Memory ingress links 2R = 8 → 2R + C = 12, so 256 → 384 GB/s. Decode moves from NoC-ingress-bound to DRAM-bound; 454.9 equals the L = 0 roofline.
  - Physical check: LPDDR 512 bits ÷ 18.75 pins/mm = 27.3 mm on a 10.39 mm proxy edge (107.9 mm², n5_derived 「近似」) needs 2.63 edges. So the 0.309 two-edge attach was not buildable, and a third edge is real (9.1 mm per edge balanced). What 0.310 got wrong was the other edges: it let memory take every edge not used by D2D, so nothing was left for host I/O (PCIe / NVLink-class). It also pooled all links of all memory edges as if every unit were spread evenly.
- **Host-I/O edge** (`Chip.noc_mem_io_edges`, default 1; 0 / 1): memory edges ≤ max(2, 4 − D2D edges − I/O edges). GH100 / B200 / MI300X keep non-HBM edges for D2D / PCIe / NVLink (0.212 note 「二手」); 1 edge 「假设」.
- **Published rectangle for H100-like**: long edge 33 mm (6 HBM sites = 3 per long side × 11 mm 「推导」), short edge 814 / 33 mm. Memory takes long edges first. Other dies stay square 「近似」.
- **Per-edge interleave** (`Chip.noc_mem_balance`, default "interleave"; "pooled" = 0.310): memory units (HBM stacks 11 mm; LPDDR x16 / LPDDR6 x24 channels at bits / 18.75 mm) are water-filled onto the memory edges, capped by edge length. Address interleave across all units makes the busiest edge the limit: links = min_e links_e · N / u_e (≤ pooled). `energy.noc.dram` reports `mem_load`, `mem_balance`, `mem_links`, `io_edges`.
- **Old → new** (840 cases: 7 chips × 14 memories × L ∈ {0, 16, 32, 64} × {qwen3-8b decode b64, qwen3-8b prefill, gpt-oss-120b decode b16}):
  - 0.310 → 0.311: 240 cases lower throughput, 0 higher. Default presets with default memory at L = 0: unchanged.
  - 100T + LPDDR5X 8 × 64, L32: 454.9 → 443.1 tok/s (load 11 / 11 / 10 channels, links 12 → 11.64), J 62.91 → 63.29 mJ. Net vs 0.309: throughput +45.4 %, J −9.7 %.
  - 1P / 1P-lp4x64 + 8 × HBM3E and 100T + 4 / 5 × HBM3E: 4 → 2 edges (the 4th edge is the I/O edge, 3 edges cannot hold the stacks either, so it is shoreline "over"); L16 / L32 throughput −58 %, back to 0.309.
  - One HBM stack (any chip): it sits on one edge, so attach links halve vs pooled (−50 % at L32).
  - H100-like + 8 × HBM3E: 3 edges (load 3 / 3 / 2), L32 836 → 743 tok/s. H100-like + 6 stacks stays at 2 edges (33 mm long edges).
  - 0.309 → 0.311: 99 lower, 108 higher.
- **Chiplet consequence** (`scripts/chiplet_example.py`, new `--io-edges {0,1}`): 1P + 8 × HBM3E monolithic (654 mm²) needs 88 mm of HBM edge but has 3 usable edges = 77 mm; 2 × 333 mm² dies (1 D2D + 1 I/O edge) have 37 mm for 44 mm. Only the 4-die split is buildable (memE 2, $2059 at D0 0.1). The 0.310 monolithic $2013 vs 4-die $2059 comparison assumed an unbuildable die.
- **Switches:** `noc_mem_io_edges = 0` + `noc_mem_balance = "pooled"` reproduces 0.310, except H100-like with ≥ 6 HBM stacks (published long edge). Adding `noc_mem_edges = 2` reproduces 0.309. Older tests that pin 0.308 / 0.309 / 0.310 numbers use these switches.
- **ui_headless_smoke robust**: every page wait goes through `wait_js` / `wait_sel` with a per-step timeout. A timeout names the step, the waited expression and the panel status text. The server's stderr goes to a temp log instead of an undrained PIPE (the likely hang under concurrent load). `STEP [elapsed]` prints live. A global watchdog (`--budget`, default 1800 s) ends a stuck run with the last step and the server state. Any browser-part exception becomes one ABORT error line. Every check is unchanged.
- **Ling-3.0 releases** (inclusionAI, public configs, fetched 2026-10-22): `ling-3.0-tiny` (7.89 B total / 1.38 B active) and `ling-3.0-flash` (124.4 B / 5.51 B), `bailing_hybrid` (KDA + MLA, the Ling-3.1-flash builder), full coverage, param_err 0. The 0.310 note "architecture not in catalog" was wrong.
- **UBTree draft** (`ling-3.0-tiny-ubtree` = Ling-3.0-tiny + inclusionAI/Ling-3.0-tiny-UBTree, 183,127,326 bf16 params exactly from the safetensors header): DFlash anchor block (16, 5 Qwen3 layers over target layers [4, 8, 12, 16, 22]) + top-64 candidates per row + bigram selector (rank 256, target-embedding MLP factorizer 1536 → 512 → 256) → pruned tree. `ModelSpec.draft_ubtree`. The target verifies 1 + nodes tokens (default 49 nodes as on the card; `serving.spec_draft_tokens` within [1 + k, 64]). Published τ 9.113 (mean of 6 card benchmarks, greedy). The SGLang code is unreleased, so the selector / tree ops and the non-causal block are 「假设」.
- Tests: test_core_311.py (6). test_core_308 / 309 / 310 pin their switches; test_core_213 / 214 / 062_1 expectations updated with 0.311 comments.

## [0.310.0]

### Memory-attach edges as the mechanism default; NoC-cluster axis; chiplet economics
- **Memory edges by shoreline occupancy everywhere** (`Chip.noc_mem_edges`: 0 = auto, the new default; 2 / 3 / 4 = fixed).
  - The Scenario resolves auto (`budget.mem_edges_auto`). So single eval, the energy DSE, the insight panels and the PD pools all use the same rule.
  - Rule: edges = ⌈interface edge / die side⌉, clamped to [2, 4 − D2D edges]. HBM needs stacks × 11 mm (0.309). New for LPDDR: bus bits ÷ 18.75 pins / mm (Credo Weaver brief, conventional LPDDR5X 0.18 Tb/s/mm at 9.6 GT/s 「二手/推导」). DDR / GDDR keep 2.
  - Die side comes from a preset's published area when the chip keeps its geometry (`PRESET_DIE_MM2`: H100-like 814 mm², NVIDIA 「来源」). Otherwise it comes from the n5_derived density proxy 「近似」. Energy-DSE rows use the search's own densities.
  - `energy.noc.dram` reports `mem_edges` / `mem_edges_auto`. Chip panel: 外存接入边 (自动 / 2 / 3 / 4).
  - Switch: `chip.noc_mem_edges = 2` reproduces 0.309 everywhere, including the chiplet DSE.
- **Old → new (auto vs 2):**
  - Default presets with their default memory are unchanged: 1P with 4 × HBM3E, H100-like (5 × HBM3 on 814 mm²), 100T with LPDDR5X 4 × 64.
  - 1P + 8 × HBM3E (4 edges): decode J −4.8 %; with 32 B/cycle links decode throughput +140 % (DRAM ingress 10 → 24 links) and J −48 %.
  - 100T + LPDDR5X 8 × 64 (3 edges): decode L32 throughput +49 %, J −10 %.
  - 100T + 4 × HBM3E (4 edges): decode L32 throughput +100 %.
  - The 0.309 chiplet example is unchanged in both views.
- **NoC clusters as an energy-DSE axis** (`clusters=[K…]`, API `clusters`, UI NoC 簇). Works with or without the dies axis. Example: prefill with 64 B/cycle links, 4 clusters give +54 % throughput on the same die.
  - Not offered in the chip panel: a cluster chip is K cards of one package (layout, memory and package are rewritten), which is the DSE's transform.
- **Chiplet economics** (`cost=` preset `n5_cowos_s` / `n5_cowos_l`, or a dict of overrides; API `cost`; UI 成本). Each row gets `usd_good_pkg` (with `si_usd`, `pkg_usd`, `test_usd`, `dpw`), and a `front_cost` Pareto front ranks throughput ↑, J ↓, cost ↓.
  - Formula: Σ good-die cost + packaging + test / assembly; HBM excluded. Each term is labelled:
    - Good die: N5 wafer $16,988 (CSET 「二手」) / DPW (π R²/A − π·2R/√(2A) 「来源」) / Y.
    - Packaging: CoWoS-S $750 per H100 and CoWoS-L $1,100 per B200 (Silicon Analysts 「二手」), per mm² of Σ dies + HBM 「推导」.
    - Test / assembly: $920, +22.5 % KGD for multi-die packages 「二手」.
  - 1P example (8 × HBM3E, CoWoS-L):

    | D0 | Monolithic | 2 dies | 4 dies |
    |---|---|---|---|
    | 0.1 | $2013 | $2097 (+4 %) | $2059 (+2 %) |
    | 0.2 | $2382 | — | $2100 (−12 %) |

    The break-even D0 is ≈ 0.12 /cm² (4 dies; 0.09–0.14 over KGD +15–30 %). MAC × 2 remains split-only (reticle).
- `scripts/chiplet_example.py --cost … --d0 …`.
- Freshness: new since 0.309 is inclusionAI/Ling-3.0-tiny-UBTree (2026-10-21, a 183 M tree-speculation draft head for Ling-3.0-tiny, whose bailing_hybrid target is not in the catalog). It was not added. UCIe 4.0 is still unreleased.

## [0.309.0]

### Fair monolithic-vs-split comparison in the chiplet DSE; self-contained chiplet example
- **On-die NoC clusters** (`Chip.noc_clusters = K`; hierarchical NoC). A monolithic die can now be partitioned like a K-die package: K clusters, each with its own mesh / ring over engines/K, joined by an on-die global tier.
  - Mechanism: the chip object is one cluster; the scenario has `package_cards = K` with `d2d_enabled`, and the cluster tier takes the place of the D2D tier. So TP / PP / EP traffic between clusters uses the existing collective model.
  - Cluster-tier bandwidth = the cluster's boundary attach links × link width × clock (the 0.308 edge rule; unlimited without a link width). α is the D2D α 「假设」.
  - Energy: each cluster-tier byte pays half a cluster NoC per end + 1 hop + 1 pitch for the boundary crossing (`pJ_bit_hop_noc` / `pJ_bit_mm_onchip`, Keckler 2011 「来源」); no D2D PHY energy (`energy.noc.d2d.on_die`).
  - Area, yield, reticle and shoreline use the whole die (`budget.whole_die`); no D2D PHY area. DRAM attach and hops are those of the whole die.
  - Precedent: A100 / H100 split L2 into two partitions, each serving the SMs of its directly connected GPCs (NVIDIA whitepapers 「来源」).
- **Memory attach on the HBM edges** (`Chip.noc_mem_edges` 2 / 3 / 4; `budget.hbm_mem_edges`). With 3 or 4 edges the NoC also attaches memory along the north / south rows, one router per column; memory links and DRAM hops follow 「推导」.
  - In the chiplet DSE the value is the number of die edges the HBM stacks occupy, ⌈stacks × 11 mm / side⌉, capped at 4 − D2D edges. Monolithic dies and split dies get the same rule.
  - This fixes the §253.3 gap: a "tight" HBM die used only two memory columns.
  - Default chip (2) and every non-chiplet path are unchanged.
- **DSE** (`energy_design_search(mono_clusters=True, mem_edges_shoreline=True)`; API `insight_dse`; UI 单片分簇):
  - With the dies axis on, each K > 1 in the axis adds a monolithic row with K clusters. New row fields: `clusters`, `mem_edges`.
  - `chiplet.compare` now compares the best split against the best monolithic row (flat or clustered), metric by metric (`mono_thr` / `mono_J`).
  - `verdict_flat` / `d_thr_flat` / `d_J_flat` keep the flat-only comparison. Setting both options to False reproduces 0.308 exactly.
- **Verdicts old → new** (1P, qwen3-8b / qwen3-30b-a3b, 8 × HBM3E, 700 W, MAC ×1; MODEL §254.2):

  | Case | 0.308 | 0.309 |
  |---|---|---|
  | Decode b64 | split_J −4.8 % J | mono (split −0.5 % thr, −0.8 % J) |
  | Decode, L = 32 | split_both +60 % / −31 % | mono |
  | Prefill, L = 64 | split_both +55 % / −4.6 % | mono (4-cluster mono within 1.1 %) |
  | MoE ep | split_J | mono |
  | MoE tp | split_J | mono |

  - Prefill at L = 0 stays mono. MAC ×2 stays only_split (reticle).
  - Splitting still wins on yield (KGD wafer area 1258 → 929 / 814 mm²) and on the reticle.
- **Script** `scripts/chiplet_example.py` (`--flat` for the 0.308 view, `--case N`). It reproduces MODEL §253.2 / §254.2 without box files.
- Freshness: model-release sweep unchanged vs 0.308. UCIe 3.0 is still the latest spec; 4.0 is targeted for end-2026 (not released).

## [0.308.0]

### Chiplet partitioning in the energy-aware DSE; D2D consistent with the NoC
- **Dies per package × D2D grade as search axes** (`energy_design_search(dies=, d2d_stds=, d2d_units=, die_split=, reticle_mm2=, pkg_mm2=, d0_per_cm2=)`; API `insight_dse`; UI 晶粒数 / D2D 接口 / 单元 / 切分 / 光罩 / 封装 / D0).
  - The package totals are split evenly over N dies: MACs, SRAM, SLC and memory stacks. Points whose engine or memory-unit counts do not divide by N are marked infeasible.
  - The split runs along TP, PP or EP (EP for MoE only). Cross-die traffic is that axis's existing collective on the D2D tier, so no new traffic model is added.
  - Package figures are per die × N (throughput, W); J/unit is unchanged. The power budget applies per package (cap ÷ N per die). `die_mm2` limits Σ die area.
  - Limits: each die ≤ reticle 858 mm² (26 × 33 「来源」; stitching not modelled 「近似」, default when the axis is on). Optional interposer limit: Σ dies + HBM footprints ≤ CoWoS-S 2700 / CoWoS-L 4719 mm² (TSMC 「来源」). Optional Poisson yield exp(−A·D0) with KGD (`kgd_mm2`, `front_kgd`; D0 reference N5 ≈ 0.10 /cm² 「二手」).
  - Per-design verdicts in `chiplet.compare`: only_split / split_thr / split_J / split_both / mono.
  - Example (1P, 8 × HBM3E, 700 W):
    - 2 × MAC (908 mm²) exceeds the reticle as one die; 2 or 4 dies make it feasible.
    - Decode b64 at 1 × MAC, 2 dies over UCIe-A: J/token −4.8 % (shorter NoC distances per die), throughput −0.5 %.
    - Prefill over BoW / UCIe-S: throughput −25 % / −15 % (D2D bandwidth); UCIe-A −3.4 %.
    - With NoC links of 32 B/cycle, DRAM ingress scales with die edge: 2 dies decode +60 % throughput, −31 % J.
    - Yield (D0 0.1): wafer area per good package 1258 → 814 mm² with 4 dies.
- **Per-grade D2D PHY footprint** (`d2d_catalog.D2D_PHY_GEOM`):
  - UCIe-A x64: 388.8 × 1043 µm; UCIe-S x16: 571.5 × 1540 µm (UCIe tutorial 「来源」).
  - BoW: one TX + one RX slice, 0.65 × 1.35 mm (BoW 2.0 §8.11 「推导」).
  - UCIe-S and BoW now use their own area. Old → new, 4 units: 1.62 → 3.52 / 3.51 mm². UCIe-A still uses `mm2_per_d2d_unit`; NVLink-C2C has none.
  - D2D shoreline check: units × edge ≤ D2D edges × side.
- **D2D edges:** a package of ≥ 3 dies now gives each die 2 inner D2D edges (was 1). The HBM shoreline loses an edge. Old → new: 4 dies × 4 stacks on a 300 mm² die goes from tight to over.
- **D2D through the die's NoC** (edge attach, the 0.305 rule):
  - Bandwidth: D2D ≤ attach links × link width × clock, with attach links = ⌈E/R⌉ per D2D edge (mesh) or 2 (ring). Applies only with a NoC link width set. Old → new, 2 × 16 engines, TP prefill: L = 32: −15.5 %; L = 128: −4.8 %.
  - Energy: D2D bytes pay NoC hops at both ends, (R + 1)/2 per end. Default on with edge attach. Old → new J/token, TP prefill: +0.29 % (2 dies), +0.85 % (4 dies).
- Freshness: UCIe 3.0 high-rate energy target 0.75 pJ/bit and "no change in bump locations" (SemiEngineering 2026-02-12 「二手」).
  - The footprint label moves from 「假设」 to 「二手」. The pJ/bit default is kept and 0.75 is noted as the upper reference; with 0.75, prefill J changes by ≤ 0.3 %.
  - UCIe 4.0 is not released. 107 orgs since 2026-10-21: none.
- Tests: tests/test_core_308.py (10).

## [0.307.0]

### NoC: router power before the power cap, router area counted once, local ports shared with the DRAM stream
- **Router static power is now priced before the power cap / DVFS** (fixes the 0.306 「近似」 of adding it after the cap). New energy fields `pJ_noc_router_cycle` and `mW_noc_router_leak` describe a reference router (64-bit, 5-port).
  - Each chip's routers scale from that reference by link width and port count. Charged only on mesh / ring with a link width set. No preset fills them; the DSE NoC-width axis fills them with Park/2.6 values, and user values win.
  - Clock floor: core rail, × V² · f under DVFS. Leakage: ∝ V, static under the cap. The energy table shows both rows.
  - Old → new, 1P decode DSE at 250 W: 2048 × 16² @ 32 B/cycle needs 268 W incl. 38 W of routers. 0.306 dropped it; 0.307 runs it DVFS-capped (f × 0.93, 3.19 k tok/s, 78.4 mJ). Uncapped points are unchanged and the front is unchanged.
- **Router area counted once** (fixes the 0.306 「假设」 of possible double counting with `uncore_frac`):
  - `uncore_frac` is TPUv1's control + unlabelled share. TPUv1 has no NoC: its blocks are joined by dedicated 256-byte paths (Jouppi ISCA 2017 「来源」).
  - So the share holds wiring and white space, not routers. NoC link wires stay inside it; routers are added once, after the share (`budget.full_area(noc_routers=True)`, `noc_router_mm2`).
  - Routers are not added when `mm2_fixed` is given, since that already covers the NoC.
  - Preset densities: no change. With `mm2_fixed`: −router mm² (old double count). MODEL §252.1.
- **Local (eject) port model** (`chip.noc_eject` "link", opt-in; needs a link width and XY routing). Each engine's local port is one link wide:
  - A multicast's per-engine ingest (s − 1)/s · payload / g bounds the op.
  - The DRAM stream shares those ports (D / E per engine) and, on a ring, the memory-stop links (D / 4). The mapping carries op-total busiest-link and local-port cycles; the stage DRAM time is ≥ (Σ op port load + DRAM share) / (L · f).
  - Default "off" is bit-identical to 0.306.
  - Effect (1P HBM3E, L = 64): mesh prefill −7 % / −15 % / −9 % / −5 % at E = 2 / 32 / 128 / 512; ring prefill −23 … −54 %; crossbar only at E ≤ 32 (DRAM share); decode within −2 % except rings at E = 2. MODEL §252.2.
- UI:
  - Chip panel: 本地端口 (off / link).
  - Energy inputs: NoC 路由器 pJ/周期 and NoC 路由器泄漏 mW, with router rows in the energy table.
- Tests: tests/test_core_307.py (8). test_core_306 now records the capped (no longer dropped) over-budget point.
- Freshness sweep (107 orgs, since 2026-10-21): no new releases.

## [0.306.0]

### NoC: DRAM-hop energy, chip-panel NoC controls, NoC link width as an energy-DSE axis
- **Default change (energy only): DRAM bytes now pay NoC hop energy** under `noc_mem_attach` "edge" (the 0.305 default).
  - Bytes travel from the two memory columns to the engines. The mean distance to the nearer column is booked on the `noc` (router) and `noc_wire` (pitch) actions.
  - Per byte: mesh = mean of min(c, C + 1 − c); ring = two stops; crossbar = 1 switch plus the same floor-plan wire. `Chip.noc_mem_hops()`. The report adds `noc.dram`.
  - Timing is unchanged. "none" restores ≤ 0.305.
  - Old → new J/token, tpuv4i_7nm preset, qwen3-8b:
    - 1P: decode b8 +15.8 %, decode b64 +14.2 %, prefill +0.8 %.
    - 100T: decode b64 +4.1 %.
  - Design conclusion: wan2.1-1.3b energy-tied SRAM sizing is now J/token-limited (min 16 → 19.0 MiB). MODEL §251.1.
- **UI (chip panel, 高级芯片参数):** NoC 拓扑, NoC 链路 B/周期, NoC 拥塞模型 (XY / 二分), DRAM 接入 (edge / none).
- **Energy DSE `noc_links` axis** (API `noc_links`, UI 「NoC 链路轴」). Each point is charged router area and width-dependent static power.
  - Source: Park et al. DAC 2012 router 227,230 µm², crossbar 26,840 µm², 5.6 mW clock floor, 76.7 mW / 16 leakage 「来源」.
  - Scaling 「推导/假设」: crossbar ∝ (ports·width)², rest ∝ ports·width. Node: user MAC density ÷ Horowitz MAC 「二手」; power ÷ 2.6.
  - The front now shows the trade-off. 1P decode, die 1300 mm², 300 W: 2048 × 16² @ 32 B/cycle (3.44 k tok/s, 77.9 mJ, NoC 291 mm² / 38 W) vs 128 × 64² @ 128 B/cycle (3.36 k, 61.3 mJ, NoC 149 mm² / 15 W). MODEL §251.2.
- **Row-link sharing (weight streaming vs activation all-gather) under XY:**
  - Mesh: proven never to exceed the modelled max(ingress link, busiest all-gather link), so it is exact. This corrects the 0.305 「近似」 wording.
  - Ring / crossbar port / mesh ejection port: bounded and documented, not modelled. MODEL §251.3.
- Tests: tests/test_core_306.py (6). test_core_300 pins `noc_mem_attach="none"` for its 0.300 records. test_core_223 adds the draft's DRAM-NoC hops.
- Freshness sweep (107 orgs, since 2026-10-21): no new releases.

## [0.305.0]

### NoC model consolidation (XY default, DRAM ingress through the NoC); LPDDR self-refresh / refresh vs temperature
- **Default change (only when `noc_link_Bpc` > 0): NoC routing bisection → xy.**
  - XY is the physical routing of a mesh (Tenstorrent X-then-Y).
  - Ring XY equals the bisection figure.
  - Crossbar XY is the port-ingress limit.
  - "bisection" stays selectable as the optimistic bound.
- **New: DRAM ingress through the NoC.** Chip `noc_mem_attach` ("edge" default, "none" = ≤ 0.304).
  - Memory tiles in two columns, one attach per engine row (Blackhole NoC 0 「来源」), cap delivered DRAM at 2·⌊√E⌋·L·f (ring 4·L·f).
  - Under XY, weight streaming (row links) and activation all-gathers (column links) use different bottleneck links; within-op sharing still adds.
- **Many-small-cores conclusion re-derived** (1P, HBM 3.3 TB/s):
  - At 64 B/cycle, decode is NoC-ingress bound below E ≈ 2048. Mesh E2048 is needed for full bandwidth; E512 gets 85 %; few big engines lose 81–91 %; rings are unusable. Best J: decode 2048 × 16² mesh, prefill 32 × 128² mesh.
  - At 256 B/cycle, E ≥ 128 is fully fed, and the few-big-engines prefill optimum returns (2 × 512²).
  - Design rule: L ≥ BW ÷ (2·⌊√E⌋·f). 100T / LPDDR is unaffected. MODEL §250.1–250.2.
- **LPDDR temperature**: energy `dram_temp_C` (0–95 °C; unset → `sram_temp_C` → 85 °C max as before).
  - Self-refresh is log-linear over the Micron Y52Q points (25/45/65 °C typical 0.77/1.23/2.84 mW, 85 °C max 9.99 mW per channel).
  - Refresh is 2× above 85 °C (JEDEC MR4).
  - 100T prefill background 0.85 → 0.50 W at 25 °C (J −0.7 %); 95 °C decode +2.6 %. UI: DRAM 温度 input. MODEL §250.3.
- Tests: tests/test_core_305.py (7); test_core_303 / 304 pin the models they describe (bisection, DRAM ingress off).
- Freshness sweep (107 orgs, since 2026-10-21): no new releases.

## [0.304.0]

### PD-queue LPDDR window; per-link (XY) NoC contention (opt-in); suite-time check; robust 210 guard
- **PD-queue LPDDR residency with an evaluation window (small default change).**
  - Each pool's representative iteration (largest-batch memoised evaluation of its phase) gives its DRAM busy fraction and gaps. The cards iterate ρ = u ÷ u_iter of the time, with the single-evaluation residency, and spend the rest in empty-queue self-refresh.
  - A single-phase point priced by the queue ledger now equals `energy_report` exactly.
  - Old → new: 100T LPDDR5X PD load 0.3: 0.42891 → 0.42747 J/token (−0.34 %); prefill pool 97 % self-refresh. HBM unchanged.
  - New `dram_residency` per pool in the queue energy. MODEL §249.1.
- **Per-link NoC contention, opt-in**: chip `noc_routing` ("bisection" default = 0.303, "xy").
  - XY prices the busiest link of the dataflow's dimension-ordered multicast / reduction trees. For an all-gather, mesh (R−1)/R of the payload on a column link, ring ½, crossbar port (s−1)/s, divided over the block's boundary links.
  - Cross-group copies add to the in-group multicasts on shared links.
  - API `noc_routing` on insight_noc / insight_dse. UI 拥塞模型 selects.
  - Effect (1P qwen3-8b prefill, 64 B/cycle): mid-size meshes / crossbars (E 8–128) lose 30–52 %; E 2048 mesh keeps full throughput with local plans; the 0.303 "mesh E 32–128" sweet spot disappears. MODEL §249.2.
- **Suite wall 0.303**: not a test-cost regression. The 8 longest concurrent files re-ran in 190 s vs 197 s on the 0.302 tree, at equal CPU-seconds; the 0.303 run met external load. MODEL §249.3.
- **test_core_210 guard**: request CPU time under the same thresholds, plus a wall check scaled by a measured contention factor. A compute regression is still caught; suite load no longer trips it.
- Runtime guards in test_core_209 / 210 / 238 and the harness's per-test times use time.perf_counter instead of time.time. The wall clock stepped by minutes mid-suite: 238 measured 318.9 s inside a file the runner timed at 78.5 s.
- Tests: tests/test_core_304.py (8).
- Freshness sweep (107 orgs, since 2026-10-21): no new releases.

## [0.303.0]

### NoC congestion (bisection bandwidth, opt-in); LPDDR power-down / self-refresh residency
- **NoC congestion (opt-in, no default change)**: new chip field `noc_link_Bpc` (B / cycle / link / direction; 0 = unlimited = ≤ 0.302).
  - The 0.299 dataflow NoC payloads (A-operand multicast, cross-group copies, split-K reduction) must cross the topology bisection: mesh ⌊√E⌋, ring 2, crossbar E/2 links. Each op takes ≥ payload / (links × L), overlapped with the array (max). Plan choice sees the NoC time; GemmCost gains `noc_cycles`, bound "noc".
  - Reference 64 B/cycle (Tenstorrent Blackhole, 512-bit flit per cycle per link 「来源」).
  - `/api/insight_noc` and `/api/insight_dse` take `noc_link_Bpc`; NoC insight rows gain `tok_s_free` / `noc_penalty`. UI: NoC 链路 B/周期 inputs and a NoC 拥塞损失 column.
  - Effect on the many-small-cores conclusion (1P qwen3-8b prefill, 64 B/cycle): few big engines lose 25–29 % (2 × 512² on any topology, 8 × 256² on ring / mesh); mesh E ≥ 128 and crossbar E ≥ 8 lose nothing; rings lose 25–60 %, worse with E. Best prefill by J becomes 8 × 256² crossbar; decode 128 × 64² ring. MODEL §248.1–248.2.
- **Default change (tpuv4i_7nm energy preset): LPDDR low-power residency**. New fields `mW_lpddr_ch_pd` 5.42 mW (Y52Q IDD2P 「来源」), `mW_lpddr_ch_sr` 9.99 mW (Y52Q power-down self refresh, 85 °C max 「来源」), `ns_lpddr_pd_gap` 15 ns and `ns_lpddr_sr_gap` 232.5 ns (JEDEC LPDDR5 tXP / tXSR + tSR 「二手」); all were unset before.
  - The LPDDR background is now the residency mix at the modelled DRAM utilisation: the active rank in standby, other ranks of a busy channel in power-down, channel-idle gaps ((1 − u)·window / layers) in self-refresh or power-down when they exceed the exit + minimum time. Refresh only outside self-refresh.
  - Old → new, 100T qwen3-8b J/token: LPDDR5X 4×16 GB decode b8 0.2135 → 0.1968 (−7.8 %; background 2.45 → 1.17 W/card), prefill −3.1 %; 4×32 GB decode −15.9 %; LPDDR6 decode −5.2 %. HBM unchanged. The PD queue ledger credits power-down only. MODEL §248.3–248.4.
  - `hbm_static` gains `powerdown_W`, `selfrefresh_W`, `residency`, `W_no_lowpower` (the 0.302 bound).
- Tests: tests/test_core_303.py (6); test_core_302 restated (the report's LPDDR background is now below the 0.302 bound).
- Freshness sweep (107 orgs, since 2026-10-21): no new releases.

## [0.302.0]

### Sourced HBM4 / LPDDR6 DRAM energy, LPDDR background power, DVFS operating point as a design-search axis; faster suite
- **Default change (tpuv4i_7nm energy preset)**: four new fields, all filled by the preset. User values still win, including 0.
  - `pJ_bit_dram_hbm4` unset → 2.46 pJ/bit (HBM4 / HBM4E; was the HBM3E 3.44). This is HBM3E ÷ 1.4: Samsung (HBM4 page / 2026-02 newsroom) and SK hynix (2025-09 newsroom) both claim ≈ 40 % better power efficiency than HBM3E (vendor claim 「推导」). HBM4E reuses it 「假设」.
  - `pJ_bit_dram_lpddr6` unset → 3.31 pJ/bit (LPDDR6; was the LPDDR5X 4.0). This is LPDDR5X ÷ 1.21: Kim et al. (Samsung), ISSCC 2026, 16 Gb 12.8 Gb/s LPDDR6, "21.0 % better energy efficiency than LPDDR5x" 「推导」.
  - `mW_lpddr_ch_standby` unset → 32.2 mW per x16 channel of 8 Gb: Micron Y52Q LPDDR5X data sheet, Table 16, IDD2N on all four rails 「来源」. Max values in non-power-down standby, so an upper side.
  - `mW_lpddr_Gb_refresh` unset → 0.765 mW/Gb: (IDD5 − IDD3N) × tRFCab 210 ns / tREFI 3.9 µs 「推导/二手」.
  - LPDDR background power is static, like HBM standby / refresh (§216). It is charged only when no idle figure is set, and its report key stays `hbm_static` (it now means DRAM background). LPDDR6 / LPDDR5 reuse the LPDDR5X figures 「假设」.
  - Effect, preset J/token, qwen3-8b:
    - 100T + LPDDR5X 4×16 GB: decode b8 0.182 → 0.214 (+17.6 %; +2.45 W/card); prefill +5.0 %.
    - 4×32 GB: decode +35 % (+4.9 W).
    - 1P + HBM4 4×12-high: decode b8 0.108 → 0.089 (−17.9 %).
    - 100T + LPDDR6: decode +3.4 % (cheaper bits, plus background).
    - HBM3E: unchanged.
  - CLI: `--pJ-bit-dram-hbm4` / `--pJ-bit-dram-lpddr6`. UI: two new energy inputs. The static label now reads "DRAM 后台".
  - The design-search notes no longer call HBM4 / LPDDR6 energy "borrowed". PHY energy and area are still the HBM3 / LPDDR5X proxies, labelled. MODEL §247.1–247.2.
- **DVFS operating point as a design-search axis**: `/api/insight_dse` `f_scales` (f / f_max, 0.15–1; default [1], so the 0.301 output is unchanged).
  - Each point is re-evaluated at freq_ghz·f. Core-rail energies are scaled × V(f)² and SRAM leakage × V(f), with V(f) from the chosen `curve`. This is the same mechanism as the 0.227 power cap (`powercap._point`).
  - A point still over the budget is capped from f_max, as before.
  - Rows gain `f_op` and `f_ghz`. UI: 频率 × input and a V–f curve select.
  - Example (1P qwen3-8b prefill, 300 W, RX 5700 XT V–f curve): 2× MACs at 0.6 f_max give 84.3 k tok/s at 3.36 mJ/token, vs 1× MACs at f_max throttled to 0.84 f, which give 59.1 k tok/s at 5.07 mJ/token. Under "dfs" (V flat) a lower clock never lowers J/token. MODEL §247.3.
- **Suite time (test-only)**: many PD-queue / DES tests read only operating-point outputs. For those, `tests/fast_queue.py` replaces the SLO-goodput rate bisection of `pdqueue.queue_report` with a −1 sentinel. That search took ≥ 90 % of these reports' time.
  - Applied to 43 test modules. Tests that read SLO outputs and the runtime-guard tests are excluded.
  - Equivalence was checked once for every wrapped module (`ACCEL_FASTQ_VERIFY=1`): each queue_report was run both ways and every non-SLO field compared; there were no differences.
  - test_core_049's compare-view budget check runs on qwen3-8b × 4 cards (same code path).
  - Production code is unchanged.
- Tests: tests/test_core_302.py (6); test_core_301 restated (LPDDR points now carry background power).
- Freshness sweep (107 orgs, since 2026-10-21): no new releases.

## [0.301.0]

### Energy-aware design search over compute + memory jointly; workload mix; SLO; memory $ (user input)
- `/api/insight_dse` gains memory-side axes:
  - DRAM: `mems` (catalog ids), or `mem_types` × `mem_count_scales` × `mem_caps`, snapped to catalog counts and capacities.
  - SLC: `slc_mibs`.
  - Defaults are the scenario's own memory and SLC, so 0.300 output is unchanged. The grid cap goes from 96 to 240 points.
- Per point:
  - Bandwidth, time and fit come from the cost model.
  - DRAM energy: per-kind pJ/bit + PHY + HBM standby / refresh. Sourced figures; HBM4 / HBM3 and LPDDR6 / LPDDR5 use the HBM3E and LPDDR5X figures as labelled approximations.
  - Die area adds the DRAM PHY (bus bits × mm2_per_dq_*) and the SLC.
  - HBM shoreline: "over" drops the point; "tight" is flagged.
- Optional user $/GB per memory type (`mem_usd_per_GB`, no built-in price) → memory $ per card, a `max_mem_usd` limit and a `front_usd` objective.
- Workload mix (`mix_out_tokens`): J/request = P·J_prefill + D·J_decode; card-s per request = P / thr_prefill + D / thr_decode (time-shared, each phase capped on its own).
- SLO limits: `slo_tpot_ms` / `slo_ttft_ms`, applied at the capped clock.
- Example (1P qwen3-8b decode b64, 800 mm² / 250 W): the front is half-MAC + HBM4 4 stacks or HBM3E 8 stacks, ≈ 4.8 k tok/s at 0.040 J/token. Under a 50 ms TPOT, LPDDR never makes the front. On 100T b8, HBM halves J/token vs LPDDR5X but needs the 3rd / 4th die edge. MODEL §246.
- UI: the DSE panel adds memory types / count / capacity, SLC, mix output tokens, TPOT / TTFT limits, $/GB and a memory $ limit. The table adds memory, GB/s, GB, SLC, TPOT, TTFT and $ columns.
  - Non-blocking: a scenario edit during a search cancels it (the static build terminates the slow-lane worker) and tags the panel. The progress pill is shown as before.
- ui_headless_smoke drives the memory + mix search and a mid-run edit.
- Tests: tests/test_core_301.py (7).
- Freshness sweep (107 orgs, since 2026-10-21): no new releases.

## [0.300.0]

### Energy presets include on-chip movement by default; energy-aware design search (Pareto)
- **Default change (energy presets)**: `horowitz_45nm` and `tpuv4i_7nm` now fill `pJ_bit_mm_onchip` and `pJ_bit_hop_noc`. Both were unset before.
  - horowitz_45nm: wire unset → 0.121 pJ/bit/mm (Keckler 2011, 40 nm 「来源」); NoC router unset → 0.416 pJ/bit/hop (Park DAC 2012 「推导」).
  - tpuv4i_7nm: wire unset → 0.078 pJ/bit/mm (Keckler 10 nm projection 「来源/假设」); NoC router unset → 0.160 pJ/bit/hop (÷ 2.6 「推导」).
  - Why: a preset J/token omitted every on-chip-movement cost (the SRAM / SLC distance term of 0.298, and the NoC traffic and its links of 0.299). Those are the costs that change with engine count, SRAM size and topology.
  - Effect on preset J/token: decode +0.6–1.2 %, prefill +4.7–7.6 % (1P / 100T, qwen3-8b; table in MODEL §245.1).
  - Conclusions that moved:
    - qwen3-8b 100T + LPDDR5X b8: 25 °C J-optimal SLC 32 → 16 GiB.
    - wan2.1-1.3b energy-tied SRAM sizing: no longer limited by J/token.
  - User values (incl. 0) still win. A user `slc_access_ref_mib` (√C) keeps the preset from filling the distance model, in the API and in the UI preset fill.
- **Energy-aware design search** (`core/energy_dse.py`, `/api/insight_dse`, slow lane):
  - Grid: MAC count × engine side × SRAM (× NoC topology).
  - Per point: throughput and fit from the cost model; J/unit from the energy table (MAC / SRAM / wire / NoC / DRAM / static); perf/W = 1 / J.
  - Budget: over-area points (budget `die_mm2` with densities) are dropped. Over-power points (`power_W_card` or `tdp_W`) run DVFS-capped (0.227 model, default) or are dropped (`power_mode: reject`).
  - Output: the Pareto front of throughput ↑ vs J/unit ↓, with differences ≤ 1 % counted as ties (`tie`); plus `front3` with area, and the best perf/W and best throughput points.
  - Example, 1P qwen3-8b under 800 mm² / 250 W:
    - Decode: the front is all half-MAC designs (−14 % J/token at equal throughput).
    - Prefill: power-capped full-MAC chips only match the half-MAC throughput.
- UI: an 「能耗感知设计搜索（Pareto）」 panel (`ins-dse-*`).
  - Controls: phase, over-power handling, MAC / SRAM multipliers, topology toggle.
  - Output: a scatter of J/unit vs throughput with the front polyline, and a table with the energy-breakdown shares.
- Tests: tests/test_core_300.py (10).
  - Files that pin the ≤ 0.299 preset (no on-chip movement) now say so explicitly: test_core_187 / 233 / 264 / 266 / 267 / 298 / 299.
- Freshness sweep (107 orgs, since 2026-10-21): no new releases.

## [0.299.0]

### NoC / on-chip data-movement energy from the dataflow (opt-in); energy breakdown; NoC design insight
- Why: 0.298 left NoC energy as a per-MAC constant, so it could not separate designs. It is now derived from what each GEMM schedule actually moves between engines.
- **Traffic** (`mapping.GemmCost.noc_hops` / `noc_pitch`, per op → `StageResult.noc_hops` / `noc_pitch` → energy actions `noc` / `noc_wire`):
  - Ganged ("wide") GEMM: the weight columns sit in each engine's local slice; the A row tile is multicast to the engines that hold output columns, min(E, ⌈N / cols⌉).
  - Core-group plans: a row split copies the K×N operand to its groups; a column split copies the M×K operand. Instances spread over groups keep their operands local 「假设」.
  - Split-K: the s fp32 partial m × n tiles are reduced over a tree of the s groups.
  - Topology (`chip.noc_topology`, existing design variable) per byte delivered to d blocks of u engines 「推导」:

    | Topology | Router traversals | Link length (pitches) |
    |---|---|---|
    | mesh (XY tree) | (d − 1)·√u | same as traversals |
    | ring | (d − 1)·u | same as traversals |
    | crossbar | d − 1 | d · ½√(d·u) |

  - The mesh tree follows Park et al. DAC 2012 (broadcast = k² crossbar + (k² − 1) link traversals).
  - Time is unchanged (NoC latency stays `noc_hop_cycles`).
- **Energy** (opt-in: `energy.pJ_bit_hop_noc`; None = no NoC term):
  - E = bits × hops × pJ_bit_hop_noc + bits × pitches × pJ_bit_mm_onchip × pitch_mm.
  - Router energy 「推导」, Park et al. DAC 2012 (45 nm 4×4 mesh, 1 mm links):
    - (427.3 mW − 76.7 mW leakage) / 653 Gb/s broadcast = 0.537 pJ/bit per router + link.
    - Minus Keckler 2011 wire 0.121 pJ/bit/mm → router 0.416 pJ/bit (lower bound).
    - ÷ 2.6 at 7 nm → `NOC_HOP_PJ_BIT_7NM` = 0.160.
  - Engine pitch = √(SRAM slice × 3 nm macro density + MACs / engine × `noc_mm2_per_kmac`).
    - Default MAC density 0.6875 mm² / 1024 MAC 「来源」: TPUv4i, 4 × 128² MXUs = 11 % of < 400 mm² (upper bound).
    - Uncore is left out, so the pitch is a lower bound.
  - The link wire is charged only with both `pJ_bit_hop_noc` and `pJ_bit_mm_onchip` set.
  - PD ledger uses the same terms (`energy.unit_pJ`). DVFS scales `pJ_bit_hop_noc` by V² (`CORE_ACTIONS["noc"]`).
- **Breakdown**: `energy_report` → `breakdown` {J, share, total, labels}.
  - Groups: MAC / 向量 / SRAM 阵列 / SLC 阵列 / 片上导线 / NoC 路由 / DRAM（含 PHY）/ 链路 / 时钟 / 静态; Σ = J_per_unit.
  - SRAM / SLC J is split into array and wire parts by their access scale.
  - UI: stacked bar + legend above the energy table (`energy-breakdown`), and NoC rows in the table.
- **Insight** `/api/insight_noc` (`core/noc_insight.py`; UI panel "NoC 与片上搬运能耗", slow lane in the static build):
  - The chip's MACs and SRAM are re-cut into E engines of r × r arrays and run on each topology, for decode and prefill. Each point gets time, J and the breakdown.
  - Also a central-SRAM placement priced analytically: every SRAM-port byte travels ½√E mesh hops, plus the whole-array wire.
  - Without NoC numbers in the table it fills the reference values (0.160 / 0.078) and says so.
- Design reading (tpuv4i_7nm + reference NoC, qwen3-8b, prompt 4096):
  - 1P prefill (mesh), NoC + link share and J / prompt token:

    | Engines | NoC + link share | J / prompt token |
    |---|---|---|
    | 2 | 2.0 % | — |
    | 8 | 3.9 % | 0.00726 |
    | 32 | 5.3 % | 0.00736 |
    | 128 | 9.5 % | 0.00784 |
    | 512 | 23 % | 0.00950 |
    | 2048 | 40 % | 0.01269 |

    Throughput gains only +1.3 % beyond E = 32. **Many small cores past ≈ 32–128 engines cost prefill energy for no speed.**
  - Decode NoC stays ≤ 2 % for every cut: decode engine count is decided by time, not NoC.
  - Crossbar (1P E32): +6 % prefill J vs mesh. Ring ≈ mesh at small E; its router energy is 1.7–7× mesh at E ≥ 512.
  - Central SRAM (1P E32): decode +24 % (0.125 → 0.156 J/token), prefill +24 %. Grows with E. **Keep SRAM sliced per engine.**
  - 100T and H100-like follow the same pattern.

### Freshness
- HF sweep since 2026-10-21 (107 orgs): empty. Nothing new.

### Default behaviour, old → new
- No default changes. The NoC term is opt-in (`pJ_bit_hop_noc`; no preset sets it). The cost model (time) is unchanged: 858 scan points identical. ext_validate: only the version stamp (0.298.0 → 0.299.0).
- Additive outputs:
  - `counts_per_unit.noc` / `noc_wire`; `energy_report.breakdown`; `noc` report block when set.
  - Energy table rows for NoC counts (energy "未提供" until set).
  - `_op_seconds` returns 10 values (+ NoC byte·hops / byte·pitches).
- `Result.summary()` is unchanged: components' NoC counts are energy-internal (`_pl_view`), and all fingerprints hold.

## [0.298.0]

### Energy: MAC = multiply + FP32 accumulate (presets); local SRAM access energy vs slice size
- Why these two (NOTES_0298): they are the energy terms that move NPU design conclusions. GPU-reference gap work is parked (needs a trace; AGENTS §248).
  - The tpuv4i_7nm split of the main design questions (qwen3-8b, prompt 4096): prefill is 68–74 % MAC; decode is DRAM + clock, MAC 2–10 %.
  - A NoC / inter-PE term (Eyeriss ratios) is ≈ a constant per-MAC factor on a systolic array, so it would not separate designs. Not added.
- **MAC energy** 「推导」: the presets priced a bf16 MAC as the 16-bit FP multiply alone (Horowitz ISSCC 2014: 1.1 pJ at 45 nm).
  - Matrix MACs accumulate in FP32 「来源」: Google Cloud TPU docs, "All multiplies take bfloat16 inputs, but all accumulations are performed in FP32 number format"; Cloud blog "BFloat16: the secret to high performance on Cloud TPUs".
  - So pJ_mac = 16b FP mult 1.1 + 32b FP add 0.9 = 2.0 pJ (45 nm, Horowitz). tpuv4i_7nm: 2.0 / 2.6 = 0.769 (TPUv4i logic scaling, as before).
  - `energy.PJ_MAC_MULT_ONLY` keeps the ≤ 0.297 values.
  - Effect (tpuv4i_7nm, qwen3-8b): prefill J/prompt-token +56–61 % (1P 0.00430 → 0.00691; 100T 0.00468 → 0.00730; H100-like 0.00451 → 0.00713); decode b8 +2–3 %, b128 +6–8 %.
  - Prefill card power rises by the same factor. Under a cap at the old prefill power, DVFS (rx5700xt_n7) runs at 0.85× clock → 0.85× prefill throughput on all three.
- **Local SRAM distance energy** 「推导/假设」: `sram_pJ_bit` = pJ_bit_sram + pJ_bit_mm_onchip × ½√((sram_mib / engines) · mm²/MiB).
  - Each engine streams its operands from its local slice; same ½√area mean distance as the 0.267 SLC model.
  - Wire coefficient: Keckler 2011 (same field). Density: Aoyagi 2025 3 nm HD macro (`slc_mm2_per_mib` / default), array-only → lower bound.
  - Only when `pJ_bit_mm_onchip` is set (no preset sets it). Report field `sram_access` (slice_mib, engines, wire_mm, scale, basis). PD-queue ledger uses the same pJ/bit.
  - Effect at 0.078 pJ/bit/mm: 1P (8 MiB slice) 0.687 mm, SRAM pJ/bit ×1.61; 100T (4 MiB) 0.486 mm; H100-like (3.1 MiB) 0.430 mm.
  - Design reading: on 1P, decode b8 J/token falls 64 → 512 MiB SRAM by 0.00267 J flat but only 0.00168 J with wire. Bigger SRAM is not free per access.
- **DVFS**: `powercap.CORE_ACTIONS` += `wire: pJ_bit_mm_onchip` (on-chip wires switch on the core rail, CV²).
- UI: pJ / MAC tooltip (FP32 accumulate); wire tooltip covers SRAM; energy table SRAM row shows slice / wire / scale.

### Freshness
- HF sweep since 2026-10-21 (107 orgs): empty. Nothing new.

### Default behaviour, old → new
- Preset `pJ_mac`: horowitz_45nm 1.1 → 2.0 (tag 来源 → 推导); tpuv4i_7nm 0.4231 → 0.7692. Every preset-based energy number moves by MAC count × Δ.
- With `pJ_bit_mm_onchip` set, SRAM traffic now pays the wire term (was flat pJ_bit_sram), and DVFS scales the wire coefficient by V² (was nominal).
- Cost model (time) unchanged: all 858 scan points identical to 0.297. ext_validate: only the version stamp (0.297.0 → 0.298.0).
- Tests re-stated by mechanism: test_core_152 (preset value / tag), test_core_233 (J/token 0.20190 → 0.22584 = old + MAC count × 0.9 / 2.6 pJ).

## [0.297.0]

### Grouped-GEMM efficiency at small per-expert row counts: checked against public kernel timings (`core/extkern.py`)
- The question (0.296 shared signature): on the GPU references the step grows with the tokens in the step faster than modelled.
  - Drama 3, batched prefill: throughput over-predicted at c64–256.
  - LFM2.5 target: marginal cost 24–78 µs/token measured vs 4.7–18 modelled.
- Is the cause grouped-GEMM (MoE) efficiency at small rows per expert? Both models are fine-grained MoEs: LFM2.5 has 64 experts top-4 (d 1536); Drama 3 has 256 experts top-8 (d 512) in 40 of its 53 layers.
- New diagnostic `core/extkern.py` + `data/ext_kernels_moe.json`, built only from public kernel timings 「来源」:
  - **`grouped_h800`**: DeepGEMM `m_grouped_gemm_fp8_fp8_bf16_nt_masked` on H800, kineto kernel time.
    - Source: DeepSeek_Simulator @ 7c69101, `results/H800_group_gemm.csv`; `python/test_decode_gemms.py` shows every group masked at m_per_group.
    - Coverage: 340 rows → 122 distinct (groups 1–18 × m 4–2048 × the two DeepSeek-V3 expert shapes, N × K = 4096 × 7168 and 7168 × 2048).
  - **`module_b200`**: TensorRT-LLM tech blog 24, "Per-Rank MoE Module Latency", TRTLLM-Gen column.
    - Setup: B200; DeepSeek-V3-style NVFP4 routed MoE at TP8 (hidden 7168, 256 experts, top-8; intermediate 2048 / 8 = 256 per rank 「推导」); 1–384 tokens.
    - The module covers router GEMM, TopK, alpha, FC1, FC2 and quantization.
- Model 「推导」: the extval default variant's chip with the same op pricing as `evaluate`. Grouped GEMM time = max(array time, operand bytes / achieved DRAM bandwidth).
  - Optionally plus the trace-sourced per-kernel floor: 1 kernel per grouped GEMM, 6 per module 「假设」 (one per listed stage).
  - The module uses uniform routing (`ir.moe_routing`) and NVFP4 at 4.5 bits (4 bits + one e4m3 scale per 16 「来源」).
- Results (never fitted), model / measured:

  | Case | Without floor | With floor | Note |
  |---|---|---|---|
  | H800 grouped, 4–128 rows per group | 1.03 (0.996–1.08 per bin) | 1.13 | measured 1.5–2.9 TB/s; bandwidth-bound |
  | H800 grouped, 256–2048 rows | 0.73 | — | the masked kernel runs below the model's FP8 ceiling; prefill regime, open |
  | B200 module, 1–384 tokens | — | 0.84–1.00 | — |

  - B200 module after every expert is hit (256 → 384 tokens): measured slope 0.115 µs/token vs model 0.005.
- Reading:
  - Small per-expert rows are already priced. On H800 the grouped GEMM stays bandwidth-bound and at measured bandwidth from 4 to 128 rows per expert, as the model has it.
  - The only missing piece is the B200 module's post-saturation slope (which also covers its token-proportional router / TopK / quantization kernels). Transferred to LFM2.5 as an upper bound 「近似」 (per token-expert pair, scaled by expert MACs, × top-4 × 40 layers), it is ≤ 3.95 µs/token per step. That is ≈ 20 % of the ≈ 19 µs/token missing at c128–256. For Drama 3 (top-8, d 512, 40 MoE layers) the same bound is 2.6 µs/token.
  - So the shared signature is mostly not grouped-GEMM efficiency. **No mechanism added; the cost model is unchanged.**
  - Remaining candidates (no public data): per-token non-GEMM kernels (activation quantization, routing / sort, finalize) and per-token host / scheduler work in SGLang.
- NPU designs (`npu_rows_curve`, catalog 1P / 100T / H100-like, LFM2.5 gate_up shape, 64 experts, FP8):
  - The array pads each expert's rows to its M granularity (16 rows for 1P / H100-like, 8 for 100T): array utilisation is 6–14 % at 1 row and > 95 % from 16–32 rows. This is the existing design variable (array rows × cols, via `gemm_cost`).
  - It costs time only when array time exceeds DRAM time. For weight-streaming MoE decode every design stays DRAM-bound up to 128 rows per expert (1P up to 512).
  - So no separate "MoE tile efficiency at small M" variable is needed for NPUs. It would matter for designs with a much higher FLOP : byte ratio or a coarser M tile; that is visible today by changing the array shape.

### Freshness
- HF sweep since 2026-10-21 (107 orgs): empty. Nothing new.

### Default behaviour, old → new
- The cost model is unchanged: all 858 points are identical to 0.296.
- ext_validate: every existing section is unchanged and the version stamp moved (0.296.0 → 0.297.0). New: the `kernels` section (summary, grouped_h800, module_b200 tables, npu_rows curves).

## [0.296.0]

### Drama 3: batched prefill in the steady state (`core/extspeech.py`)
- The issue (deferred in 0.295): the closed-loop cycle and TTFA priced the co-batched prefills serially, k·ttft₁ per batch and ttft₁ per request.
- Mechanism 「来源」: SGLang prefills every waiting request in one forward (PrefillAdder). `--prefill-max-requests` is unset, which means no request cap (sglang server_args docs); only `max_prefill_tokens` 16384 bounds the batch (⌊16384 / 148 rows⌋ = 110 requests). The 16-request first-frame CUDA-graph cap (`first_frame_cuda_graph_max_requests`, engine_builder.py / first_frame_graphs.py @ 0029194) only chooses graph or eager launch; it does not cap the batch.
- Model 「推导」/「近似」:
  - The 0.294 gate (p_win) and mean batch k = 1 + p_win·4a are kept (`_gate`).
  - The batch a request is in: n = 1 + Poisson(k − 1) 「近似」 (the others arrive independently), clipped at 110 (`_batch_dist`).
  - P(n) = the engine's batch-n prefill. Per request the GPU pays p̄ = E[P(n)/n]: it goes into the cycle (c·p̄) and the round stretch (T_round = tpot + a·(p̄ + vocoder)). A request waits for P̂ = E[P(n)].
  - TTFA = W_gate + P̂ + 10·T_round + W_voc + DAC(first window). At c = 1 it is identical to 0.295.
  - `predict(..., prefill="serial")` keeps the ≤ 0.295 pricing. Rows carry `pred_serial` / `ttfa_serial_ms`; `predict` returns `prefill_s_per_request` / `prefill_batch_mean`.
- Result (ext default variant, never fitted). Each cell is the predicted / measured ratio, serial → batched:

  | c | frames/s | TTFA p50 |
  |---|---|---|
  | 1 | 1.23 → 1.23 | 0.69 → 0.69 |
  | 8 | 1.38 → 1.38 | 0.67 → 0.66 |
  | 16 | 1.40 → 1.43 | 0.70 → 0.67 |
  | 32 | 1.37 → 1.47 | 0.75 → 0.67 |
  | 64 | 1.29 → 1.55 | 0.81 → 0.64 |
  | 128, same GPU | 1.26 → 1.77 | 0.89 → 0.58 |
  | 256, same GPU | 1.31 → 2.13 | 0.32 → 0.18 |
  | 128, second GPU | 0.91 → 1.40 | 1.28 → 0.75 |
  | 256, second GPU | 0.97 → 1.88 | 0.41 → 0.19 |

  - gmean frames/s ratio 1.323 → 1.376; TTFA 0.718 → 0.679. Catalog variant: 2.256 → 2.355, TTFA 0.447 → 0.420.
- Reading:
  - The serial pricing was compensating something missing. Once the prefill is priced as SGLang runs it, the engine alone (vocoder on a second GPU, so no vocoder in the cycle) runs 1.40× / 1.88× too fast at c = 128 / 256.
  - The TTFA ratio over c = 1 … 64 is now flat at 0.64–0.73. That is consistent with a uniform ≈ 1.4× slower decode round plus the c = 1 fixed ≈ 28 ms (host / IPC).
  - It is the same signature as the LFM2.5 target alone below: the real GPU step grows with the tokens in the step faster than the per-SM + per-kernel-floor path. It is not a TTS-specific mechanism.
  - c = 256 is still dominated by the unexplained pre-run wait (0.295).
- `burst` is unchanged: c = 256 wave 1 (777 / 727 ms) is now above the batched steady state (597 / 403 ms) and below the serial one. Tests re-stated by mechanism: wave 1 < ½ of the measured p50 and < the serial steady state.

### LFM2.5 DSpark c = 8 … 64 gap, located by mechanism (`extspec.token_curve`)
- Measured-implied steps 「推导」: invert the same closed loop on the measured tok/s (model ttft₁). Rows and `by_concurrency` carry `step_{base,spec}_{meas,pred}_ms` and `w_bw_meas_tbps` (modelled weight bytes / measured step).
- (i) A DSpark step costs what a target-alone step with the same token count (c·10 vs c) costs. Measured / interpolated base step at c = 1 … 16 (10 … 160 tokens): 1.15, 1.08, 1.03, 1.02, 0.91. Model: 1.13, 1.10, 1.09, 1.05, 1.01. So the verify / draft structure is right: draft cost, the Markov chain and per-step DSpark overhead do not explain the gap.
- (ii) The gap is the target's own token slope. Marginal target-alone cost in µs/token, measured vs model:

  | c range | Measured | Model |
  |---|---|---|
  | 8–16 | 227 | 132 |
  | 16–32 | 124 | 60 |
  | 32–64 | 78 | 18 |
  | 64–128 | 40 | 4.7 |
  | 128–256 | 24 | 4.7 |

  - The model flattens once the expert union saturates (every expert read from c ≈ 64): the step becomes weight-bound. The measured step keeps growing, and the implied weight bandwidth peaks at 2.6 TB/s (c = 32) and falls to 1.6 TB/s at c = 256.
  - Since speed-up ≈ τ · base_step(c) / spec_step(c·10 tokens), the model's flatter curve between 64 and 640 tokens gives the over-prediction: measured 20.9 / 11.8 = 1.77 vs model 8.23 / 6.32 = 1.30, i.e. ≈ 1.36× (the observed 23–31 % after prefill and draft).
- Checked and ruled out, with sources:
  - The overlap scheduler stays on for DSPARK on CUDA (sglang 488869c `speculative_hook._handle_dspark`: no overlap override).
  - The radix cache is off (protocol), so the hybrid-mamba `no_buffer` branch that disables overlap is not taken (`overrides._mamba_radix_cache_resolution`).
  - The README launch command sets no CUDA-graph or chunk flags.
- Not identified: the per-token cost at 64–2560 tokens. Candidate: FP8 fused-MoE kernel efficiency at small per-expert row counts (no B300-tuned config is sourced). No public trace exists, so it is not modelled.

### Freshness
- HF sweep since 2026-10-20 (107 orgs): only entries already seen (Ornith-1.5-122B siblings, unsloth re-uploads, LFM2.5-24B-A2B-DSpark-GGUF, a llama.cpp repack of a listed entry). The 2026-10-21 window is empty. Nothing new.

### Default behaviour, old → new
- The cost model is unchanged: all 858 points are identical to 0.295.
- ext_validate: the LLM metrics are unchanged and the version stamp moved (0.295.0 → 0.296.0). Intended diagnostic changes:
  - The speech `pred` / `ratio` / `shares` / `ttfa_pred_ms` / `ttfa_ratio` and summaries now use batched prefill (c ≥ 4 change; c = 1, 2 identical), plus the new `pred_serial` / `ttfa_serial_ms`.
  - spec rows gain the step fields, and the spec section gains `token_curve`. The spec predictions are unchanged.

## [0.295.0]

### Speculative decoding on a small-active MoE: what explains batch-1 DSpark 1.1× (model) vs 2.38× (B300)?
- New diagnostic `core/extspec.py` + `data/ext_measurements_spec.json`. It takes all 72 rows (8 datasets × concurrency 1 … 256) of the LFM2.5-24B-A2B-DSpark repo's `benchmarks/b300-fp8.csv` + `protocol.json` 「来源」. The setup there is FP8 target, SGLang 488869c + patch, block 9 / 10 draft tokens, temperature 0 and bf16 KV.
- Diagnostic only; nothing is fitted. The measured τ of each row is an input (`spec_accept_from_tau`), like the speech diagnostic's measured window schedule.
- Closed loop at concurrency c (mean field, as for speech) 「推导」: tok/s = c·OUT / (c·ttft₁ + OUT/τ·step(c)).
  - Hardware: the extval B200 reference chip stands in for the B300 「近似」. HGX B300 and B200 list the same dense FP8 rate and NVLink 5, and both have 8 TB/s HBM3e.
  - Prompt length: 256 tokens 「假设」 (not published).
- Mechanisms checked; mean speed-up over the 8 datasets, model vs measured:

  | c | Measured | Roofline (`catalog`) | Per-kernel floor (`default`) | Perfect locality (bound) | Expert union | Kernel share of target step |
  |---|---|---|---|---|---|---|
  | 1 | 2.38× | 1.13× | 2.59× | 4.15× | 7.75× | 0.74 |
  | 8 | 2.23× | 2.11× | 2.74× | 4.16× | 2.46× | 0.45 |
  | 32 | 2.69× | 3.67× | 3.52× | 3.74× | 1.14× | 0.30 |
  | 256 | 1.71× | 1.89× | 1.99× | 1.99× | 1.00× | 0.24 |

  - **(c) Hardware baseline: explains the gap.** The B300 baseline (314 tok/s at c = 1) is far from bandwidth-bound. With extval's trace-sourced per-kernel floor (MODEL §39 / §119; not fitted), 74 % of the modelled target step at c = 1 is per-kernel cost. Kernel cost does not grow with the 10 verified tokens, so the verify step costs 1.98× a plain step, not the 4.6× of the roofline. Speed-up 2.59× vs 2.38× measured (gmean over all 72 rows: 1.18× high).
  - **(a) Expert union: present, and consistent with the data.** 10 tokens touch ⌈64(1 − (60/64)^10)⌉ = 31 experts instead of 4 (7.75× the expert bytes at c = 1). The ratio falls to 1 by c = 128, where every expert is already read.
  - **(b) Expert locality: not supported.** No public LFM2 routing statistic exists, so only the upper bound is computed: the block routes like one token per sequence, and the union's expert bytes are removed at the step's achieved bandwidth 「近似」. It would give 4.15× at c = 1, far above 2.38×. Any strong locality would widen the gap, not close it.
- Chip-design reading: on a bandwidth-roofline accelerator with no per-kernel overhead (the catalog chips), block-verify speculation on a 64-expert top-4 MoE really is worth only ~1.1× at batch 1, because the verify reads 4.6× the bytes. Most of the GPU's 2.4× comes from amortising per-kernel cost. No cost-model change: the catalog chips keep ideal dispatch (0.284 decision).
- Residuals, not fitted: mid concurrency (c = 8 … 64) is predicted 23–31 % high, and the target-alone tok/s 1.38–1.88× high (as for Drama 3).
- `ext_validation.json` gains `spec[default|catalog]` = {summary, by_concurrency, rows}.

### Drama 3: synchronized start burst (TTFA at c = 256)
- `extspeech.burst(variant, c, vocoder)`, a closed-form schedule of c arrivals at t = 0 「推导」:
  - SGLang prefills every waiting request in one forward up to `max_prefill_tokens` 16384 (sglang server_args default 「来源」; Drama 3 context 16384, no chunked prefill), so batches hold ⌊16384 / 148 rows⌋ = 110 requests.
  - The 4-round prefill gate runs between batches. Each batch is prefilled together (P(n) at batch n, not n × batch-1).
  - First audio comes after 10 more decode rounds plus the first window. The batch's first windows reach the vocoder together and decode one at a time.
- Result:
  - c = 256 wave-1 mean TTFA is 777 ms (same GPU) / 727 ms (second GPU), below even the steady state (1,078 / 883 ms). Wave 1 is 256 of the 1,088 requests (23.5 %), so it cannot set the p50.
  - From the measurements alone: at c = 256 the p50 exceeds the time for 10 frames at the measured per-stream rate (c / frames/s) by 2.35 s (same GPU) and 1.53 s (second GPU). At c = 128 the excess is 97 / 54 ms.
  - So a wait of 1.5–2.3 s happens before a stream runs at the shared rate. The synchronized start rules itself out. The cookbook attributes the same-GPU case to a fully loaded vocoder queue, which would mean ≈ 3× the modelled per-window vocoder time (cf. the open colocated-throughput residual). The second-GPU case stays unexplained (an admission limit is suspected, not sourced). Steady-state TTFA is unchanged.
- Rows carry `ttfa_rate_ms` (measured-only) and `ttfa_burst_wave1_ms`.
- Found on the way, deferred: the steady-state TTFA / throughput cycle still costs co-batched prefills serially (k·ttft₁). At c = 256 that is 153 ms for k = 11.6, against a batched P(k) of a few tens of ms.

### Freshness
- HF sweep since 2026-10-20 (107 orgs): only entries already seen (Ornith-1.5-122B siblings, unsloth re-uploads, LFM2.5 DSpark GGUF). Nothing new.

### Default behaviour, old → new
- The cost model is unchanged: all 858 points (143 LLMs × 100T / 1P / H100-like × decode b8 / prefill) are identical to 0.294.
- ext_validate: the LLM metrics and the speech predictions are unchanged, and the version stamp moved (0.294.0 → 0.295.0). New: the `spec` section, plus the speech row fields `ttfa_rate_ms` / `ttfa_burst_wave1_ms`.

## [0.294.0]

### Drama 3 closed-loop TTFA (ext speech diagnostic, `core/extspeech.py`)
- Why this item: of the open AGENTS items it says the most about chip-design conclusions. Time to first audio under load is the speech SLO metric, and the 0.282 diagnostic predicted it 1.5–2.6× low with a gap that grew with concurrency (c = 1 … 64). That points to a missing serving mechanism, not a per-op cost error. The other candidates are blocked or small: the colocated-vocoder residual (measured 1.70× vs modelled 1.22×) needs a kernel trace that is not public (the vocoder decodes windows one at a time at batch 1, already modelled). Dory `logit_scale` is negligible. Suite-wall reduction does not change any modelled number.
- Mechanism 「来源」: SGLang `prefill_decode_interval` (Drama 3 engine_builder.py `DECODE_STEPS_BETWEEN_PREFILLS = 4` @ 0029194; sglang scheduler `_arm_prefill_decode_interval` / `_should_defer_prefill`). After a prefill batch the next 4 rounds defer prefill. After that, prefill runs first whenever a request waits.
- Model 「推导」 (closed loop at concurrency c, N frames per request, mean field):
  - a = (c − 1)/N: the other streams' arrivals per decode round.
  - T_round = tpot(c) + a·(ttft₁ + vocoder per request): each decode round stretched by the others' prefills and (same GPU, time-shared) their vocoder windows.
  - p_win = 4a/(4a + 1): the chance the prefill gate is shut. W_gate = p_win·1.5·T_round.
  - k = 1 + p_win·4a: requests in the same prefill batch, at serial cost k·ttft₁ 「近似」.
  - W_voc: Pollaczek–Khinchine wait of the first window behind the others' windows (one server) 「近似」.
  - TTFA = W_gate + k·ttft₁ + 10·T_round + W_voc + DAC(first window). At c = 1 it reduces to the 0.282 form, which is kept as `ttfa_noq_s` / `ttfa_noq_ms`.
  - With the vocoder on a second GPU, TTFA now keeps that GPU's first-window decode (0.282 dropped it).
- Result (ext default variant, Table 1 p50; never fitted). Predicted / measured TTFA: c1 0.69, c2 0.70, c4 0.73, c8 0.67, c16 0.70, c32 0.75, c64 0.81. Before (no queueing): 0.69 → 0.38. gmean 0.718.
  - The ratio is now flat. The remaining ≈ 0.7 matches the decode being ≈ 1.2–1.4× slower than modelled (frames/s ratio 1.23–1.40) plus a c = 1 fixed ≈ 28 ms (host / IPC hops, not modelled).
  - Table 2: c128 same GPU 608 vs 685 ms (0.89); second GPU 512 vs 401 (1.28). c256: 1078 vs 3403 and 883 vs 2146. The c = 256 rows are dominated by what is not modelled: the start burst (all c requests at t = 0; 1,088 requests = 4.25 per slot).
- `ext_validation.json` speech rows carry `ttfa_noq_ms` / `ttfa_ratio`; the summary carries `ttfa_gmean_ratio`. Throughput (frames/s) is unchanged.

### Freshness
- HF sweep since 2026-10-19 (107 orgs): the Ornith-1.5-122B siblings and unsloth re-uploads (as in 0.293), tencent Hy-Memory GGUF, nvidia Cosmos3 robotics, LiquidAI LFM2.5-24B-A2B-DSpark (+ GGUF).
- `freshness_sweep.EXTRA_ORGS` += 14 orgs. Speech: Zyphra, neuphonic, Marvis-AI, myshell-ai, parler-tts, hume-ai, KittenML. RWKV: fla-hub, recursal. Other: Tencent-Hunyuan, amazon, apple, ServiceNow-AI, RekaAI. None had a release in the window. No new decision-model release (perplexity-ai, startlux-models).
- **Added `lfm2.5-24b-a2b-fp8-dspark`**: the LFM2.5-24B-A2B FP8 target with the LiquidAI DSpark draft (repo created 2026-10-20 JST, lfm1.0), as in the README SGLang command (DSPARK, block 9, 10 draft tokens).
  - The draft (config.json + safetensors header 「来源」) has the SGLang native DSpark layout: 5 dense GQA layers (2048, 32/8 × 64, q/k norm, full attention), SwiGLU 6144, fc over 5 target layers, Markov head rank 256, confidence head, no own lm_head / embedding. bf16, 327 707 521 params (ours 327 707 520: the existing floor of extra / layers).
  - Published τ 5.16 (mean tokens / step over 8 datasets, c = 1, temperature 0, 1 × B300, FP8 target) → `spec_published` (i.i.d. acceptance 0.840 「推导」). `spec_accept` stays a lever.
  - Modelled b1 speed-up with k = 9 at that acceptance: 1.11× (1P), 1.17× (H100-like), 1.16× (100T). The README reports 2.38× (B300). On a bandwidth-bound roofline, the 10 verified tokens touch ≈ 7× the experts of one token (64 experts, top-4). The published baseline (324 tok/s at c = 1) is far from bandwidth-bound on a B300. Diagnostic only, not fitted.
  - `_attach_ready_dspark` reads `target_layer_ids` from `dflash_config` when the top-level key is absent (the LFM config has only the nested key).
  - Not added: the earlier LFM2.5 DSpark drafts (1.2B / 2.6B / VL-3B / 8B-A1B, 2026-08/09); their targets are not catalogued. Also not added: the DSpark GGUF.

### Default behaviour, old → new
- All 852 points of the 142 LLMs evaluated in 0.293 are identical: 100T / 1P / H100-like × decode b8 / prefill; TPOT, TTFT, bound, per-stage times and warnings. New: the 6 `lfm2.5-24b-a2b-fp8-dspark` points. With speculation off (the default) they equal `lfm2.5-24b-a2b-fp8`.
- ext_validate: LLM metrics unchanged and the version stamp moved (0.293.0 → 0.294.0). Speech: TTFA predictions as above; frames/s unchanged.

## [0.293.0]

### MoVA value-expert routing: release statistics, measured loads, residency class
- Release check 「来源」 (config.json, modeling_k2_horizon.py, model card, Jais 3 blog): no value-routing statistic or load figure is published. `router_aux_loss_coef` 0.001 is applied only to the FFN router logits (`output_router_logits` records `K2HorizonSparseMoeBlock`; default false). The `v_router` has a selection-only bias (aux-loss-free balancing) and no loss term. The W&B training-log link is a JavaScript app with no readable data. Uniform routing stays the default 「假设」.
- `serving.mova_expert_load` (64 values, measured tokens per value expert; API / CLI `--mova-expert-load`): touched value experts = the occupancy Σ_e 1 − (1 − p_e)^t, p_e = k·load_e/Σload (`ir.mova_hits`, the 0.132 rule; at least k, at most E). Pairs stay t·k. This is a separate router from the FFN one: `moe_expert_load` never reaches the value experts. On a model without MoVA the field is ignored with a warning. A uniform load gives the uniform hits exactly.
- Residency: does the 0.286 machinery apply? Not as a merged class: the routed-FFN class (`exp`) carries EP semantics (the idle-DP-rank expert reads in energy, the chunk union scaling, skew). Its principle does apply, so the value experts get their own class: `StageStorage.rt_w`, touched key `hot_rt`, `MemPlan.pinned_rt / slc_rt / sram_rt_hit / slc_frac_rt`, `memplan.rt_miss`.
  - Before (≤ 0.292), the value experts were ordinary hot bytes. The 0.192 pin rule ranked them as read every step, and every hot byte missed at one byte share, so on a big SRAM the 30 GB (fp32) / 15 GB (bf16) of value experts diluted the q / k / o / gate / shared-expert reads.
  - Now (0.192 rule, read share per byte) 「推导」: the every-step hot bytes go first, then the value experts and the routed FFN experts in order of read share (touched / stored bytes of the step; equal shares: value experts first). Each class misses at its own share. A measured value load uses the 0.286 A0 read-share curve (`dip_exp_hit`).
  - SLC: `pin` uses the same order. `dip` puts the off-SRAM value experts in the probabilistically read pool with the FFN experts, and the retained share applies to both 「近似」. `lru` is unchanged. The pipe replay (`overlap`) charges `attn.v_experts` at its own miss share.
- Design effect (jais-3-mova-36b-a4b-fp8, 100T + SRAM / SLC overrides, decode TPOT, 0.292 → 0.293) 「推导」:

  | Config | b1 | b8 |
  |---|---|---|
  | SRAM 4 GiB | 41.67 → 26.18 ms | 149.60 → 139.80 ms |
  | SRAM 8 GiB | 34.70 → 20.89 ms | 136.95 → 127.35 ms |
  | SLC 16 GiB pin | 20.65 → 15.32 ms | 111.45 → 103.57 ms |
  | SLC 16 GiB dip | 18.64 → 11.81 ms | 104.52 → 88.27 ms |

### Per-layer measured expert loads (AGENTS: carried since §236)
- Why this item: it has the most impact on chip-design conclusions of the open AGENTS items. Measured expert popularity is the input of the SRAM / SLC expert-residency study (0.261 / 0.263 / 0.286), the main on-chip memory sizing knob for MoE. With one vector, every layer had the same hot experts and the same skew. A heterogeneous MoE (Nemotron 3.5 Bolt, 0.289) could not take a measured load at all. The other open items are small (sigmoid out-gate, measured 0.2–0.65 % of TTFT on the MoVA model's softplus gate), cover one release (served ggml kernels: one catalogued GGUF model), or need data that is not public (host CPU, vocoder trace).
- `serving.moe_expert_load_layers` (one vector per MoE layer, in layer order, each with its layer's expert count; CLI `--moe-expert-load-layers FILE`) and `serving.mova_expert_load_layers` (one per MoVA layer). These are alternatives to the one-vector forms (and to `moe_skew`).
  - Each layer routes with its own skew and busiest-rank shape (`Phase.skew_l` / `v_pl`, `ir.layer_skew` / `layer_vp`; the one-vector rule applied per layer). Layers that are identical but load differently are no longer summed as one group (`evaluate._route_groups`).
  - Residency ranks every expert of the stage's layers by read probability (A0 across layers). Experts of different sizes (het MoE) are weighted by bytes (`dip_exp_hit(..., w)`).
  - The MTP layers and the co-located chunk's union routing use the layer-mean skew 「近似」. Energy action counts treat the per-layer FFN loads as the same EP-skew input as the one-vector form (counted at balance).
- One vector repeated per layer gives the one-vector result to 1e-12 (TPOT, energy counts). On Bolt, a one-vector load is still rejected; the error now names the per-layer field.

### Freshness
- HF sweep since 2026-10-20 (93 orgs): same 8 entries as the 0.292 sweep (Ornith-1.5-122B siblings, unsloth re-uploads); nothing new. Not added (as for Ornith): the Jais-3-MoVA GGUF / MLX siblings named on the model card.

### Default behaviour, old → new
- The 840 points of the other 140 evaluable LLMs are identical to 0.292 (100T / 1P / H100-like × decode b8 / prefill; TPOT, TTFT, bound, per-stage times, warnings). No default sets a measured load.
- Jais-3-MoVA only (residency class; DRAM bytes of prefill unchanged, its pipe replay now charges the value-expert GEMMs their own miss share). Times in ms.
  - Default (100T, b1, 4096): TPOT 115.16 → 114.92 (fp32), 48.53 → 48.29 (fp8); TTFT 973.34 → 973.47, 471.07 → 471.26.
  - Decode b8 TPOT, -fp8: 100T 162.06 → 161.91, 1P 9.360 → 9.325, H100-like 13.214 → 13.205.
  - Decode b8 TPOT, fp32: 426.64 → 426.51, 24.70 → 24.66, 34.78 → 34.77.
  - Prefill TTFT moves by ≤ 0.04 %.
- ext_validate: all metrics unchanged; only the version stamp moved (0.292.0 → 0.293.0).

## [0.292.0]

### Added: Jais-3-MoVA-36B-A4B (inception42 BF16 release, IFM FP8; Apache-2.0) — mixture-of-value-experts attention
- Mechanism 「来源」 modeling_k2_horizon.py (K2HorizonDecoderLayer / K2HorizonMoVAAttention, calc_router_weights, combine_routed_experts) and config.json:
  - Every sparse layer (not in `mlp_only_layers` 0–2; 45 of 48) replaces v_proj with `mova_num_experts` = 64 value experts, linear h → n_kv·hd = 2560 → 1024, no bias. Each token gets v = Σ over its top `mova_num_experts_per_tok` = 4 experts of w_e · silu(W_e x).
  - The `v_router` (h → 64, sigmoid) adds its bias only to the selection scores; the selected weights are renormalised and × `router_scaling_factor` 2.5.
  - `attention_gate_func: softplus`: every layer (also the dense ones) has `gate_proj` [32·128, h], and the attention output is multiplied by softplus(gate_proj x, β = ln 2) before o_proj.
- Model: `AttnCore.v_experts` / `v_topk` / `out_gate_fn`. The `attn_linears` of a MoVA layer are q (+ gate, 2 × width), k, v_router [h, 64], v_experts [h, 64·1024], o. `misc_params` adds the v_router bias. `active_params` counts 4 of the 64 value experts.
- IR (`ir._mova_v_ops`) 「推导」: the value experts use the MoE uniform-routing expectation over this rank's tokens. The rank runs one GEMM per touched expert (m_e rows), and only the touched experts' weights stream. With attention TP, every rank holds every expert's slice of its KV heads. Vector work: silu, routing weight and scatter-add (6 ops / element), plus the sigmoid + top-k selection (6 ops / expert). `attn_out_gate`: 5 ops + 2 SFU (exp, log) per element 「假设」. Decode b1 touches 4 experts, b8 touches 26, prefill touches all 64.
- Param check exact: 37 444 792 020 = release, for both repos. Active 5.95 B, embeddings and head included (4.67 B without).
- As released: the inception42 safetensors are fp32 (config `torch_dtype` bf16), so the weights are fp32. The IFM FP8 (`quant_method: fp8`, 128 × 128 blocks, dynamic activations) quantises only the routed FFN experts. Value experts, v_router, q/k/o/gate, shared expert, dense layers, embeddings and head stay bf16 (`ignored_layers`, 3 408 entries).
- Defaults (100T, lpddr5x_4x64_8533_16g, b1, 4096; TPOT / TTFT) 「推导」: jais-3-mova-36b-a4b 115.16 ms / 0.973 s (does not fit: 141.2 GiB per card); -fp8 48.53 ms / 0.471 s (fits).
- Release files 184 → 186 (llm 158 → 160). Family 「Jais 3」, provider Inception / MBZUAI IFM. The 0.291 test that MoVA was absent now points to test_core_292.

### Removed: the 0.291 per-run test report cache
- `tests/runcache.py`, its 4 tests in test_core_291, the per-run directory in `tests/run_tests.py` / `scripts/run_tests.py` and the 2-line `cached(disagg_report)` wrappers in test_core_053 / 058 / 061_3 / 061_4 / 086 / 088 / 101 / 231 / 234 are gone. The 9 test files are byte-identical to 0.290 (327b584), except 088 (below). Reason: 0 hits in two full runs (58 reports, no shared keys), so it was dead code to maintain. test_core_292 checks that nothing references it.

### Fixed: flaky test_core_088::test_server_burst_and_stalled_client
- Failure seen in the 0.291 suite: {200: 69, TimeoutError: 27}. The 96 burst clients had a fixed 120 s urlopen deadline. The test server evaluates in its handler threads (serialised on the GIL), so on a loaded box the last responses arrive later than 120 s while the server is healthy.
- New client `_post_progress` (raw socket, whole response read): it waits as long as the server keeps answering someone, and fails only after 60 s with no response anywhere (a hung server), or on any reset or refusal. The assertion is unchanged: all 96 must return 200. The 2 s read timeout is now set after the burst, only for the stalled-client check (set before it, a handler starved for more than 2 s could return 408 to a healthy request).
- Reproduction with a server slowed to 1.4 s per eval: old test FAIL {200: 78, TimeoutError: 18}, new test PASS (173 s). New `test_burst_client_still_fails_on_a_hung_server`: a listener that accepts and never answers raises TimeoutError, and a closed port raises a connection error.

### Test runner: longest-first start order
- `tests/run_tests.py` `order_longest_first`: modules start by recorded per-file time (`scripts/test_durations.json`; local updates stay out of commits). The fallback is the old SLOW_FIRST, then name order. Previously only the static SLOW_FIRST list was used, which lacked 049 / 222 / 231 / 234 / 235 / domain (the 0.291 run ended on test_core_domain). Every module still runs once, in its own process.

### Freshness
- HF sweep since 2026-10-20 (93 orgs): only the Ornith-1.5-122B siblings (FP8 listed; MLX / GGUF / NVFP4 not added) and unsloth re-uploads.

### Default behaviour, old → new
- All 840 points identical to 0.291 (140 evaluable LLMs × 100T / 1P / H100-like × decode b8 / prefill; TPOT, TTFT, bound, per-stage times, warnings).
- New entries: jais-3-mova-36b-a4b (fp32 as released; 100T b1 TPOT 115.16 ms, TTFT 0.973 s, no fit), jais-3-mova-36b-a4b-fp8 (48.53 ms, 0.471 s).
- ext_validate: all metrics unchanged; only the version stamp moved (0.291.0 → 0.292.0).

## [0.291.0]

### Added: pplx-decider decision models (perplexity-ai, Apache-2.0)
- Four checkpoints, listed in the decision grouping (`DECISION_IDS`): `pplx-decider-v1.2` (2026-10-15 JST), `pplx-decider-flash-v1.2` (10-16), `pplx-decider-v1.1-27b` (10-05), `pplx-decider-v1-27b` (10-02). Families: 「pplx-decider v1.2」 (v1.2 + Flash) and 「pplx-decider v1 / v1.1」. Provider: Perplexity（决策模型）.
- Backbone `Qwen3_5Model` as released (bf16). There is no lm_head: `readout.safetensors` is a bf16 [255, hidden] decision head 「来源」 (header fetched by `fetch_hf_release`). `ModelSpec.readout_rows` = 255; `head_params` = 255 · hidden, so the param check is exact (LLM 25.6259 B for the 27B checkpoints, 4.2064 B for Flash; rel_err 0). The tied-head dedup is skipped for decision releases.
- `decision_config.json` 「来源」: v1.1 / v1.2 / Flash `attention_mode: noncausal_full_attention` → the full-attention layers run non-causal (`causal=False`); the Gated-DeltaNet layers stay causal. v1 has no attention_mode → causal.
- v1.2 / Flash v1.2 `joint_config.json` (template `joint-question-inline-readout-v1`): one readout per question marker. New `Serving.questions` (1–4096, needs `labels`; CLI `--questions`): the readout GEMM has m = batch · questions; `decision()` reports `decisions_per_s` = batch · questions / latency and a 「联合请求」 note. `labels` above `readout_rows` (255, card: ≤ 255 options) is rejected.
- Image budget from the processor configs 「来源」: max pixels 1 638 400 (v1.2, Flash), 4 194 304 (v1.1), 262 144 (v1); a 2048² image is 1 600 tokens on v1.2 (card: "a 1.6 MP image uses about 1,600 sequence positions").
- Decision defaults (100T, b1, prompt 1050, 4 labels) 「推导」: v1.2 and v1.1 548.15 ms (1.824 decisions/s; v1.2 with 8 joint questions: 14.594/s at the same 548.15 ms), Flash v1.2 85.30 ms (11.723/s; ×8: 93.788/s), v1 545.81 ms (1.832/s). microsoft-decision-1 unchanged: 160.61 ms, 6.226/s. Non-causal vs causal at 16 384 tokens: v1.1 TTFT above v1 (full-attention cores lose the causal half). The card's GB200 fast-path median (96 ms) is a different chip and is not a calibration target 「假设」.

### Added: Jais 3 (inception42 BF16, IFM FP8; Apache-2.0) and Mellum2.1 (JetBrains, Apache-2.0)
- Jais 3 (`k2_horizon`, custom modeling_k2_horizon.py 「来源」): 375B-A23B, 32B, 7B, 3.7B, 0.9B bf16 (inception42, 2026-09-29 JST) and IFM FP8 for 375B-A23B / 32B / 7B / 3.7B (09-29 / 09-30). Sigmoid router + `moe_gate_bias` × `router_scaling_factor`, `num_shared_experts` shared experts of width `moe_intermediate_size`, dense `mlp_only_layers`, partial RoPE, group RMSNorm. `_build_std` adds the shared experts, the router bias and q/k norm when `query_key_norm`. Param checks exact (375B: 379.1672 B total, 26.669 B active).
- IFM FP8 as released: compressed-tensors W8A8 float, 128 × 128 block weights, dynamic per-group-128 activations → act fp8. The 375B-FP8 keeps attention, shared experts, routers and dense layers bf16 (W bf16/专家 fp8).
- Mellum2.1-12B-A2.5B-Thinking (2026-09-20 JST): transformers `mellum` = Qwen3-MoE attention (q_norm / k_norm) + Laguna-style layers, 3 : 1 sliding (1024) / full attention (21 / 7 of 28), 64 experts top-8, bf16. Param check exact (12.1499 B, 2.439 B active). The MTP head is not released.
- Defaults (100T, lpddr5x_4x64_8533_16g, b1, 4096; TPOT / TTFT) 「推导」: jais-3-375b-a23b 267.96 ms / 4.710 s (does not fit); -fp8 187.78 / 2.786 (does not fit); 32b 355.76 / 2.925 (does not fit); 32b-fp8 187.26 / 1.578; 7b 86.24 / 0.662; 7b-fp8 49.91 / 0.367; 3.7b 49.03 / 0.386; 3.7b-fp8 29.29 / 0.227; 0.9b 11.14 / 0.103; mellum2.1 23.35 / 0.208. pplx-decider-v1.2 / v1.1 257.17 ms / 2.165 s, v1 TTFT 2.131 s, Flash 38.00 ms / 0.347 s.
- Not added: Jais-3-MoVA (BF16 and FP8; routed V projections with a V-router and a softplus attention gate are not modelled, +21 % param gap), Jais 0.9B FP8 (no release), Uno adapters, GGUF / MLX, the pre-window K2-Horizon family, Mellum2 (2026-05), pplx PII-Tracer / embed / computer.
- Providers: Perplexity（决策模型）, Inception / MBZUAI IFM（Jais）, JetBrains（Mellum）. Release files 170 → 184 (llm 144 → 158).

### Test suite: per-run report cache (tests/runcache.py)
- `cached(fn)` wraps the test-level `disagg_report` calls of the PD-heavy files (053, 058, 061_3, 061_4, 086, 088, 101, 231, 234). Key = SHA-256 of (a) the code version: every file under `accel_dse/` (sources + data); (b) the live model state: every accel_dse module's upper-case data globals (`model_flags` / pin_legacy overrides, `REF_HW`, `SPLIT_K` …), the code location of every module-level function and class method (a monkeypatch changes it), every module-level ContextVar's effective value (e.g. the `capture_ctx` sink); (c) the function and its full arguments as canonical text (all dataclass fields, sorted sets / dicts; nothing hash-seed dependent). Exact perf memo tables are excluded. Unfingerprintable arguments or unpicklable results are not cached.
- In-process dict plus, when `tests/run_tests.py` / `scripts/run_tests.py` sets `ACCEL_DSE_TEST_RUNCACHE`, a per-run directory shared by the module processes (atomic rename, O_EXCL lock so a second process waits instead of recomputing; removed at the end of the run). A hit returns an unpickled copy. Test files only gained the 2-line wrapper; no assertion changed. test_core_291 covers copy semantics, flag / ContextVar / monkeypatch / argument / code-version keying, cross-process sharing and the unpicklable pass-through.
- Measured: 0 hits. The full run computed 58 cached reports, 0 shared across files, 0 reused in-file, CPU saved 0 s; the overhead is about 10 ms per call. The 0.290 profile's "duplicates" were not real: the 057 ↔ 058 report (≈ 83 s) runs under different model flags (057 is pinned to ≤ 0.192 + ≤ 0.171 attention orientation), and the 231 ↔ 234 one is internal to `pdsim.compare` / `capture_ctx`, whose live queue ctx (closures, id-keyed memos) cannot be shared across processes. 053 and 049 have no repeated reports.
- Suite wall alone (-j 8): 0.290 1468 s → 0.291 (pre-docs run, cache active) 1292 s; per-file times 053 292 → 280 s, 057 262 → 256 s, 234 212 → 200 s, 049 669 → 183 s. The difference is box noise (the 0.290 run had multi-minute stalls in 049 / 088 / 101), not the cache.

### Freshness
- HF sweep since 2026-10-19 (93 orgs): Ornith-1.5-122B MLX / GGUF / NVFP4 siblings (not added, same as the other Ornith NVFP4 / GGUF / MLX repos), unsloth re-uploads (Ornith GGUF, sam3.1), tencent Hy-Memory GGUF (Hy-Memory is listed), nvidia Cosmos3 robotics / world-model releases (outside the catalogued domains).
- `freshness_sweep`: IFM, JetBrains, perplexity-ai and inception42 are now provider orgs.

### Default behaviour, old → new
- All 756 points identical to 0.290 (126 evaluable LLMs × 100T / 1P / H100-like × decode b8 / prefill; TPOT, TTFT, bound, per-stage times, warnings).
- `decision()` gains a `questions` key (1 unless set); microsoft-decision-1 numbers unchanged. Scenarios serialise `serving.questions` only when it is not 1.
- New: the 14 listed entries above.
- ext_validate: only the version stamp changed (0.290.0 → 0.291.0).


## [0.290.0]

### Changed: IQ2_XXS dequant cost from the ggml reference dequant (Bolt-24GB)
- `dtypes.DEQUANT_OPS` / `dequant_ops`: vector element-ops per converted element. Every format keeps the 0.40 convention of 2 (unpack / convert + scale multiply). IQ2_XXS is now 3.65625, derived from ggml `dequantize_row_iq2_xxs` (ggml-quants.c) and `dequantize_iq2_xxs` (ggml-cuda dequantize.cuh, llama.cpp master 839a572a4c): + 1 sign per weight, + 4 / 8 (two table lookups, `iq2xxs_grid` 2 KiB and `ksigns_iq2xs` 128 B, plus two index ops per 8 weights), + 5 / 32 (sub-block scale d · (0.5 + s) · 0.25 per 32). The fp16 block scale is not charged, same as the nvfp4 / int4 scales. Q8_0 (qs · d) is the convention: 2.
- Block bytes, scales included, are unchanged and match the release: IQ2_XXS 66 B / 256 = 2.0625 b, Q8_0 34 B / 32 = 8.5 b (ggml-common.h).
- `gemm_cost`: `convert_x` / `convert_wx` carry the extra ops; the vector seconds and the per-pass weight-dequant seconds include them. A chip that lists (iq2_xxs, act) in `FormatSupport.mixed` decodes in the MAC datapath and pays no dequant. No public chip lists it.
- `vecsanity.format_headroom(chip, dram_GBps, fmt)`: dequant headroom of one weight format. IQ2_XXS is about 0.28× the int4 headroom: it streams 1.9× the elements per byte and costs 1.8× the ops per element.
- The dequant warning now names every weight format whose headroom is below int4 (for example 「iq2_xxs 余量 …×」). Models without such a format keep the old text.
- API: each stage reports `dequant_ms` and `dequant_bound` (VECTOR-bound with weight dequant ≥ half the vector time); `api_catalog` has `dequant_ops`. UI: the bound tag reads 「VECTOR · 反量化」 on such stages, and the assumptions list the per-format op counts. The coverage note for ggml formats states the ops (IQ2_XXS 3.66, Q8_0 2) and that the served fused kernels are not modelled per format.
- Effect on Bolt-24GB, decode b8: on the default chips the vector time rises but stays hidden, so TPOT is unchanged (100T 54.27 ms, 1P 5.616 ms). On 1P with 4096 vector lanes the stage is VECTOR-bound and dequant-bound, and TPOT goes from 10.38 to 16.11 ms.

### Performance: queue model hot paths (test suite)
- Profile (py-spy, cProfile) of the slowest files: test_core_053 / 057 / 234 / 049 spend most of their time in PD queue reports (`pdqueue._queue_weights`, `_victim_wait_law_raw.r_surv`, `bulkq.bulk_law`).
  - `_queue_weights` is memoised. About 25 % of its calls in a PD report repeat their arguments: 053 has 397 of 1 495 calls, 234 has 464 of 2 269. The caller still gets its own list.
  - The x ≥ 1 loops no longer test the index per element.
  - `bulk_law` splits the histogram update into first / middle / last ranges, so the exact +0.0 terms are dropped.
  - `r_surv` tests o > r instead of o − r > 0.0.
- Suite wall alone (-j 8): 1473 s (0.289) → 1468 s. That is within noise; one PD report took 124.9 s before and 116.7 s after, under the same concurrent load. The suite is still bounded by the PD-report files, and structural fixes (sharing reports across tests) are deferred.
- Results are bit-identical: 300 random `_queue_weights` cases, 3 `bulk_law` laws and 2 full `disagg_report`s (repr) match 0.289 exactly. All evaluable models × decode / prefill on 3 chips = 604 points, 0 differ.

### Freshness
- HF sweep since 2026-10-20 (92 orgs): only the Ornith 122B siblings (0.288) and unsloth re-uploads.
- `freshness_sweep.EXTRA_ORGS` += IFM, JetBrains, perplexity-ai. These orgs publish official releases that no earlier sweep watched: Jais 3, Mellum2.1, pplx-decider. They are not catalogued yet (AGENTS §240 deferred).

### Default behaviour, old → new
- Every default-chip result is unchanged (604 / 604 points, including Bolt-24GB TPOT 25.91 ms at 100T b1). What changed for Bolt-24GB: its vector / dequant time (100T b1: 5.02 → 6.62 ms, hidden behind DRAM), its warning text on chips where dequant exceeds DRAM (now names iq2_xxs headroom), and the coverage note.
- ext_validate: only the version stamp changed (0.289.0 → 0.290.0).


## [0.289.0]

### Added: the NVIDIA Nemotron 3.5 family (nvidia, OpenMDW-1.1)
- Read from config.json, the safetensors headers, preprocessor_config.json and the model cards; parameter counts are exact (param_err 0).
- **Bolt 74B-A9B** (repo created 2026-10-08 JST; model card release date 10/15). The card says it is a Star Elastic compression of Super-VL 120B-A12B. Architecture `nemotron_h_puzzle`: the same 88-layer layout (40 Mamba-2, 8 GQA, 40 latent-MoE), but each MoE layer has its own routed-expert count (240–496) and width (1536–2560) from `block_configs`, top-9 (parent: 512 experts, top-22), no MTP. Same RADIO ViT-H and processor (1024–13 312 patches). LLM 74 042 720 768 params, vision 842 360 326 (card: 74B total, 9B active).
  - Listed, as served: `nemotron-3.5-bolt-74b-a9b-24gb`, a ModelOpt MIXED_PRECISION checkpoint. Routed and shared experts are IQ2_XXS (ggml block, 66 B per 256 weights = 2.0625 b; the card calls it native 2-bit). Mamba in / out are NVFP4 W4A16. Attention q/k/v/o, the latent projections and lm_head are Q8_0 (34 B per 32 = 8.5 b). Embeddings, router, norms and the ViT are bf16. Every quantized role is weight-only, so GEMM inputs stay bf16. KV is bf16 and SSM state fp32, from config.json; the card's vLLM recipe chooses fp8 KV and a bf16 SSM cache.
  - Also listed: `-bf16` (same config and header shapes as the 24GB entry; listed separately because MERGED requires identical params and dtypes).
  - Defaults (100T, b1, ctx / prompt 4096): 24GB TPOT 25.91 ms, TTFT 0.817 s, fits; BF16 TPOT 96.03 ms, does not fit.
- **Lightning 30B-A3B** (repo created 2026-08-05 JST; predates every sweep window, never catalogued). `nemotron_h`: 52 layers, 128 experts top-6, no latent space, 1 MTP layer; text only. Listed: `-nvfp4` (MIXED_PRECISION: W4A16 NVFP4 experts / shared / lm_head, FP8 attention with fp8 inputs, fp8 KV); `-bf16` also listed. LLM 31 577 940 288, MTP 1 335 325 952. Defaults: NVFP4 TPOT 12.72 ms, TTFT 0.239 s; BF16 33.82 ms / 0.455 s.
- **Super-VL 120B-A12B BF16**: listed next to the NVFP4 entry (same architecture, different dtypes). Defaults: TPOT 129.28 ms, TTFT 1.827 s, does not fit.
- `summarize_release`: ggml block tensors (U8 `[out, in / QK, 66 | 34]` when the quantization config names IQ2_XXS / Q8_0) count QK weights per block. `dtypes`: formats `iq2_xxs` (2.0625 b) and `q8_0` (8.5 b), weight-only. Coverage note on those models: the dequant cost of the served b12x kernels is approximated as one vector conversion per element.
- Not added: Bolt GGUF and LiLiCorr / DSpark / DFlash drafts; Lightning Base; Nemotron 3.5 Embed / Rerank (retrieval models).

### Fixed: MoE routing with per-layer expert counts
- Each layer already built its own experts. Three places took the first MoE layer for the whole model:
  - Co-located chunk extra experts (`extra_expert_bytes`, the fused pipe merge). Now routed per expert shape: extra = exp_dram · Σ_g w_g (hit′_g − hit_g) / Σ_g w_g hit_g, with w_g the per-expert params of the stage's layers of that shape. Bolt, TP2·EP2 b4 + 1/32 of a 4096 prefill: the first layer's E (240) would under-count it.
  - `serving.moe_expert_load` now refuses models whose MoE layers differ in expert count (one vector cannot describe them; was silently applied to every layer).
  - The `moe_skew` physical cap takes the strictest layer (Bolt EP 64: 28.44, from the 240-expert layers).
- Every other model is uniform, so their results are unchanged (all-model compare below).

### Performance: pipe flow-shop state (protenix on a large-SRAM chip)
- protenix on SRAM 4 GiB + SLC 4 GiB pin did not finish in 700 s (0.288). Profile (cProfile, 240 s): > 99 % in `overlap._Sim.state` and its tuple build. With a buffer that large the window never evicts, so it holds every tile fed so far, and every layer repeat built a tuple of all of them (1 409 states in the first 240 s, ~0.5 M entries each; the whole evaluation takes 6 415).
- Now `state()` is an O(1) view (window list, head, end, clock); `_same` rejects on length and the cheap fields first and reads entries only when those match, with the same float expressions. The window is a head index into an append-only list (was `del win[:h]`); `shift` and compaction build new lists, so a view never sees a later change.
- Results are bit-identical: 600 random flow-shop cases against the 0.288 simulator kept verbatim in the test; all evaluable models × decode / prefill × {default, SRAM 4 GiB + SLC 4 GiB pin, SRAM 1 GiB + SLC 16 GiB pin} = 876 points, 872 identical, 0 differ. The other 4 (protenix and boltz-1, pin 4 GiB) hit the 60 s cap on 0.288 under load; boltz-1 matches its 0.288 value (4.532921773781235 s).
- Wall time (6 parallel workers, so slower than alone): protenix pin 4 GiB > 700 s → 3.2 s (2.4 s alone; step 8.7720 s); boltz-1 pin 4 GiB > 60 s → 1.4 s; protenix SRAM 1 GiB + SLC 16 GiB 50.4 → 1.1 s; boltz-1 32.1 → 0.9 s; alphafold2 / openfold pin 4 GiB 6.9 / 6.6 → 0.2 / 0.4 s.
- Sweep for other slow configs (batch 8; SRAM 16 GiB pin, and SRAM 4 GiB + SLC 64 GiB lru; every evaluable model): slowest 3.2 s (protenix). No other pathological case.
- Guard (test_core_289, like test_core_237): deterministic work of the protenix / boltz-1 pin cases (tiles fed, window entries compared, compare calls) ≤ 1.5 × their 0.289 counts, plus a CPU bound (20 s alone, 60 s under the parallel runner).

### Changed: Kimi-Audio cached-frame K/V on the DiT attention ops (AGENTS §238 deferred)
- The K/V of the cached frames is now `kv_read` on each block's attention op (2 · frames · d · 2 B), not `extra_dram` passed by speech.py. Same bytes, same roofline: all 68 speech KPI rows (`speech_kpis`, default and 100T + SLC 4 GiB pin, b1 / b4) are bit-identical.

### Default behaviour, old → new
- No change for any model already in the catalog: 872 / 876 points identical (above), the 4 others did not finish before; 68 / 68 speech KPI rows identical.
- New entries only (numbers above): the catalog lists 5 more models (146 → 151 listed; 176 listed + unlisted).
- ext_validate: only the version stamp changed (0.288.0 → 0.289.0).

### Freshness
- HF sweep since 2026-10-20 (89 orgs): only the Ornith 122B siblings (0.288) and unsloth re-uploads. Web check: nothing new; DeepSeek-V4.1-Flash-1017 is the 0.248 alias.
- The nvidia org listing showed the pre-window misses added above (Bolt, Lightning, Super-VL BF16).


## [0.288.0]

### Added: the Ornith family (ornith-ai, MIT)
- All public Ornith repos were read: config.json, safetensors headers, and the image and video preprocessor configs (all the same as Qwen3.5: 65 536–16 777 216 px, patch 16, merge 2; video 4 096–25 165 824).
- **Listed, as released:**
  - `ornith-1.5-122b-fp8`, `ornith-1.5-397b-fp8` and `ornith-1.5-35b-a3b-fp8`. These are official compressed-tensors W8A8 float: per-channel fp8 weights with dynamic per-token fp8 activations on the routed and shared experts and the full-attention projections. Linear attention, router, embeddings, head, MTP and ViT stay bf16.
  - The 122B-FP8 config is now public (it was a 404 in 0.287).
  - The catalog has no Qwen3.5 FP8 release, so these are new dtype points.
  - Parameter counts match exactly (param_err 0), coverage is full, and the vision towers are exact.
  - Defaults, TPOT at b8 / ctx 4096:
    - 35B-A3B-FP8 at TP1: 64.83 ms (bf16: 105.06 ms);
    - 122B-FP8 at TP2·EP2: 109.18 ms;
    - 397B-FP8 at TP8·EP8: 71.41 ms.
- **Merged (same config and header shapes as a listed entry; each keeps its own as-released dtypes):**
  - Ornith-1.5-9B → qwen3.5-9b; Ornith-1.5-35B-A3B → startlux-decision-v1-35b-a3b; Ornith-1.5-397B → qwen3.5-397b-a17b.
  - Ornith-1.0 9B / 35B / 397B are the 1.5 configs without the MTP head, merged into the same entries, the same way MAI-Gym-9B was.
  - Ornith-1.0 35B-FP8 / 397B-FP8 → the 1.5 FP8 entries.
  - The bf16 Ornith releases are all-bf16, where the Qwen base keeps a few fp32 norm / A_log tensors. For example, ornith-1.5-9b's TPOT is 8 ns below qwen3.5-9b's (92.45095 ms, b8, default chip).
- Not added: the NVFP4 (1.5 9B / 35B-A3B / 397B), GGUF, MLX and DFlash draft repos.

### Fixed: the memory-tier ledger beyond LLM stages (MODEL §233)
- The 0.287 per-class ledger now covers every domain:
  - **DiT / protein / genomics trunks** (`_evaluate_full`) now carry `StageResult.ledger`. That includes the cached-prefix (prefix_kv), mode-cache, CFG-asymmetric uncond and FSDP paths; the FSDP gather write counts as demand of its own.
  - The ledger closes on all 146 fuzz scenarios that evaluated (150 drawn), across 25 domain models × 7 SRAM/SLC configs × layouts × FSDP / placement / overlap. No hit.
- **Hit: components (text encoders, VAEs, vision / audio encoders, speech-card modules) dropped bytes.** `_component_time` counted only weights and activations.
  - **Lookup bytes were charged to no tier.** Examples: Qwen2.5-Omni Talker codec-embedding gathers (21.8 MB per b4 prefill), Kimi-Audio DiT, Drama 3 DAC, Moshi, CosyVoice / Index-Echo flow.
  - **K/V seconds sat outside the roofline.** The speech callers added them serially after it: Omni Talker K/V, CosyVoice / Index-Echo / Step-Audio flow chunk K/V, Kimi-Audio DiT cached-frame K/V.
  - Now every touched byte (weights, activations, K/V, state, lookups, plus `extra_dram` for bytes not in the op list) is DRAM traffic inside the component's roofline. Components get no on-chip credit, so SRAM = SLC = 0 and DRAM = demand.
  - Speech-card SRAM/SLC shares (omni_pin + omni_slc_plan; cosy_pin, kimi_pin) were checked by a property test (2000 random SRAM / SLC / batch): per re-read evaluation, SRAM + SLC + DRAM = 1, and the SLC holds at most its size. No hit.
- **Default behaviour, old → new:**
  - LLM / VLM / DiT / protein / genomics evaluations are unchanged: 143 evaluable models × decode / prefill × {default chip, SRAM 4 GiB + SLC 4 GiB pin} = 572 points, 570 identical; boltz-1 on the pin chip identical (step 4.5329 s, run separately). protenix on the pin chip did not finish in 700 s on either tree and was not compared (its default-chip points are identical).
  - Speech KPIs (`speech_kpis`, 10 s of audio) change as follows:

    | config | KPI | old | new |
    |---|---|---|---|
    | qwen2.5-omni-7b b4, 100T SLC 4 GiB pin | Talker step | 2.562 ms | 1.783 ms |
    | same | RTF | 0.4775 | 0.4382 |
    | same | first audio | 0.5832 s | 0.5451 s |
    | qwen2.5-omni-7b b1, same chip | RTF | 0.1749 | 0.1651 |
    | qwen2.5-omni-7b b4, default chip | first audio | 1.38140 s | 1.38081 s |
    | kimi-audio-7b-instruct b4, default | RTF | 0.96747 | 0.96766 |
    | drama-3 b4, default | RTF | 1.07719 | 1.07724 |
    | cosyvoice2-0.5b b4, default | total | 1.893255 s | 1.893252 s |
    | drama-3 100T b1 (diagnostic) | `dac.dram_act_GB` | 14.3114 | 14.3364 |
    | drama-3 100T b8 (diagnostic) | `dac.dram_act_GB` | 126.2905 | 126.4903 |
    | ext speech: drama-3 H200 stream c16 (default) | frames/s (ratio) | 1300.1 (1.402) | 1300.0 (1.401) |
    | ext speech: c64 (default) | frames/s | 2177.1 | 2177.0 |
    | ext speech T2: c256 voc-same (default / catalog) | frames/s | 3175.7 / 3990.5 | 3175.4 / 3990.0 |

    The Omni Talker step is DRAM-bound at small batch. Its K/V reads now overlap its weight streaming inside the roofline instead of being added after it. Kimi, Drama 3 and Moshi rise slightly: their lookup bytes are now charged. The Drama 3 DAC causal-conv history (`kv_read`) and codebook lookups are now DRAM traffic, so `_dac_window_dram`'s activation share is every non-weight touched byte (was `act` only; it must sum to the costed DRAM, test_core_278); ext speech Drama 3 predictions drop ≤ 0.02 % (all 7 changed rows ≤ 0.5 frames/s, ratios ≤ 0.001). 31 of the 68 speech KPI rows changed, at most −8 % (Omni + SLC). The rest change by less than 0.03 %.

### Fixed: design-insight panels tagged stale (AGENTS §236 deferred item)
- After a scenario edit, the insight, SLC-size, sizing and upgrade-path panels (`ins-status`, `ins-slc-status`, `ins-size-status`, `ins-path-status`) kept showing old results untagged. They are now in `invalidate()`'s list, mark `dataset.done` on success and on error, and carry `staleTag` for a result computed from an older scenario version. This is the same mechanism as the 0.285 speech panels, and the 0.286 structural guard now covers them.

### Freshness
- HF sweep since 2026-10-20 (89 orgs, now including ornith-ai): only the Ornith 122B siblings (FP8 added above; GGUF / MLX not modelled), unsloth/Ornith-1.5-122B-GGUF and unsloth/sam3.1 (re-uploads).
- Web check: NVIDIA Nemotron 3.5 Bolt 74B-A9B (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB, created 2026-10-07; BF16 repo 2026-10-09) is missing from the catalog. It predates the window, so it is recorded as a candidate (AGENTS §238).

## [0.287.0]

### Fixed: `local` symlink ignored; tracked-tree guard
- `.gitignore` had `local/`, which only matches a directory. The per-worktree `local` symlink (to the shared local data) is not a directory to git, so `git add -A` picked it up (0.286 WIP). Now `/local` (no trailing slash: directory or symlink at the root).
- test_core_287: the pattern has no trailing slash, `git check-ignore` reports `local`, and `git ls-files` has no `local` entry (skips without git).

### Fixed: memory-hierarchy ledger audit — bytes credited to two tiers or to none (MODEL §232)
- After the 0.286 expert-residency bug, every path that splits a step's bytes over SRAM / SLC / DRAM was audited adversarially. New ledger check: per demand class (weights, KV read / write, state, lookup, activations), SRAM-served + SLC-served + DRAM = touched bytes, every tier ≥ 0, no tier over capacity, nothing held on chip twice. The SRAM share is recomputed from the plan's resident bytes alone, independently of `step_dram_bytes`.
  - Property test: 4000 random plans (storages, batches, SRAM / SLC sizes, all four SLC policies, state buffers, reservations, measured expert loads, re-read weights, touched sums).
  - End to end: `StageResult.ledger` (storage, touched sums incl. `hot_rr`, SLC/DRAM bandwidth ratio) on every LLM stage; checked across policies, a state buffer, a measured load and re-read weights; plus a fuzzer over the catalog (5 200 random scenarios in the final rounds, 0 ledger / draft-capacity violations after the fixes; not in the suite).
- **Hits (fixed mechanism-first):**
  1. **Independent draft model planned on the whole chip.** The draft (0.222) runs on the head-stage cards, but its plan and the target's both used the full SRAM, state buffer and SLC: with SRAM 4 GiB the draft held 4091 MiB and the target 4076 MiB at once. Now the draft is planned first — its bytes are read k times per step, so the 0.192 rule (DRAM bytes saved per step per byte) ranks them first — and the head stage plans what it leaves (`memplan.plan(sram_resv, slc_resv, buf_resv)`, `memplan.held`). Staging is shared (the programs run one after the other) and sized by the larger: when the target's staging is larger, the draft is re-planned beside it, keeping its SLC share. With `lru`, if the target's off-SRAM set does not fit beside the draft, the step's sweep thrashes and the draft's SLC hits go too.
     - Consequence: the draft's pass now depends on the target's staging, which grows with the 1 + k verify tokens. On 1P (256 MiB SRAM), qwen3-32b + qwen3-0.6b, a k = 4 pass reads more DRAM than a k = 1 pass, and a topk 4 / 16-token tree slows the single pass from 0.854 to 0.863 ms. With uncontended SRAM a pass is still k-independent. test_core_223 / 224 now check exact per-pass invariance at SRAM 64 GiB, and monotonicity on 1P.
  2. **Co-located chunked prefill charged under the chunk's own plan.** `pdqueue._fused_step` / `_multi_bracket` took the prompt chunk's KV / state / activation DRAM bytes from the prefill-alone evaluation, whose plan has the chunk's KV on chip — on-chip bytes the decode batch already holds — and dropped the chunk's SLC hits from the iteration's SLC time. Now the chunk's touched bytes go through the decode plan (`evaluate.chunk_mem_bytes`); identical when neither plan holds KV / state on chip (all default chips).
  3. **Touched bytes of an empty class charged to no tier.** `hot_miss` / `exp_miss` / the KV and state miss shares returned 0 when the class had no planned storage, so those bytes vanished. Seen on image DiTs with a cached text prefix (`prefix_kv`): the prefix K/V is kept in the activation reserve, not as KV storage. Now miss share 1 (`memplan.kv_state_miss`).
  4. **Two-batch overlap (tbo) SLC time.** The second weight pass was charged to DRAM time but not SLC time; now `t_slc` includes it.
  5. A full SLC hit could leave −4e-15 B of DRAM after subtraction (wan2.2 fuzz case); clamped at 0.
- **Audited, no hit:** KV SRAM / SLC shares (pin, pin_bw, lru, dip); dense re-read pins (0.269); omni / speech components (separate cards or no SRAM credit; omni_slc_plan); TP / EP (per-rank plans, per-card SLC, busiest-rank expert share); PD disaggregation (new-KV bytes, free-KV clip, host-L2 hit_device + hit_host + miss = 1). FSDP pins against the full weights plus a gather write: conservative, left as is.
- **Default behaviour, old → new:** catalog chip presets have no SLC and small SRAM, so default results are unchanged: 142 models × decode / prefill give identical summaries, fingerprints unchanged. Changed (opt-in features on large on-chip memories):

  | config | old | new |
  |---|---|---|
  | draft qwen3-0.6b → qwen3-8b, 1P, b1 (TPOT) | 2.354 ms | 2.386 ms |
  | draft qwen3-0.6b → qwen3-8b, 100T, b8, k 3 (TPOT) | 71.468 ms | 71.659 ms |
  | draft, 100T SRAM 4 GiB, qwen3-8b b1 (TPOT) | 24.81 ms | 28.24 ms |
  | draft, 100T SLC 4 GiB pin, qwen3-8b b1 (TPOT) | 24.67 ms | 28.10 ms |
  | co-located chunk, qwen3-0.6b b64 ctx 4096, SRAM 4 GiB (iteration) | 171.49 ms | 173.58 ms |
  | same, SLC 4 GiB pin | 169.55 ms | 170.57 ms |
  | qwen-image-2.1-turbo cached step, b1 (t_dram; step compute-bound, unchanged) | 823.5 ms | 824.2 ms |
  | tbo + SLC (DRAM-bound, TPOT unchanged): t_slc | 4.29 ms | 8.59 ms |

  The chunk case: 470 MB of the chunk's KV had been charged 0 DRAM. The image case: 134 MB of K/V reads per cached step at b1 (1.07 GB at b8) were charged to no tier.

### Added: Ornith-1.5-122B (freshness)
- `ornith-1.5-122b` = ornith-ai/Ornith-1.5-122B (MIT; repo 2026-10-16, made public with its FP8 / GGUF / MLX siblings 2026-10-20). Qwen3.5-based post-train, `qwen3_5_moe` as released bf16: 48 layers (3 Gated-DeltaNet : 1 gated GQA), 256 experts top-8 + shared expert, hidden 3072, 1 MTP layer, Qwen3.5 ViT (27 blocks). Exact params from the safetensors headers (LLM 122.11 B, MTP 2.52 B, vision 451.3 M; param_err 0). Image / video processor sizes from its own preprocessor configs (same as Qwen3.5). New provider "Ornith AI" and family "Ornith-1.5"; ornith-ai is now swept.
  - The catalog has no Qwen3.5-122B-A10B, so this is a new size point, not a merge.
  - `vision.vision_spec` also accepts the transformers-5 name `qwen3_5_moe_vision` (no other release uses it).
  - TP4·EP4 on the default chip: decode b8 ctx 4096 TPOT 105.07 ms (DRAM-bound); prefill 4096 tokens TTFT 434.4 ms.
- Not added: Ornith-1.5-122B-FP8 ships no config.json / index yet (HTTP 404). The rest of the family (Ornith-1.0, 1.5 9B / 35B-A3B / 397B, 2026-06 / 08) predates the window; it was missed because ornith-ai was not swept (AGENTS §237).

### Freshness
- HF sweep since 2026-10-20 (88 orgs): unsloth/sam3.1 (community re-upload) and unsloth/Ornith-1.5-122B-GGUF, which led to the official ornith-ai release above.

## [0.286.0]

### Changed: UI wiring tests check parsed structure instead of exact source text (MODEL §231.1)
- Audit: 60 test files asserted exact app.js / index.html / pyodide-shim.js text (`"req('x', '/api/x', body(" in js`, a regex on a one-line catch block, a 1.5 kB character window around an id, `js.split(...)[:120]`). That breaks on harmless edits, which is what happened to test_core_276/277 in 0.285. Some checks were also silently weak: the 0.85 stale-flag regex matched nothing once a catch block was reflowed.
- New `tests/webstruct.py` (stdlib only):
  - a JS tokenizer that drops comments and whitespace, decodes string literals, and splits template literals into their text parts and `${…}` expressions;
  - function bodies, `req(chan, path, body)` call sites with their enclosing function, string and identifier sets;
  - object/array literals, shim route lists, catch blocks, element-id aliases and writes;
  - HTML ids, `<select>` options, and the text of a control's own `<label>` (via `html.parser`).
- Every exact-source assert in tests/ was converted (60 files): token-sequence matches (format-agnostic), parsed ids and options, function-scoped checks, and call-site structure. The 0.285 streams guard is now structural: one click-time `body(…)` snapshot bound once, before the request loop, spread into one-model requests. `scripts/ui_static_smoke.py` uses the same parser.
- New generalized guards (test_core_286) replace families of one-off pins:
  - every literal `$('id')` the UI looks up exists in index.html or is created by the UI (242 ids);
  - every status that `invalidate()` tags stale sets `dataset.done` on its success path and in every catch block that writes it;
  - every speech `kind` the evaluator emits has a `p.kind === …` KPI branch;
  - every endpoint the UI calls has a progress label.
- A lint test fails if a test reads UI source text and uses it other than through webstruct.
- The headless smoke now also checks the 0.285 snapshot live: it changes the model after the first streams row and requires every request to carry the click-time scenario, one model per request, and the stale tag on the panel.
- `scripts/ui_pin_mutations.py` (opt-in) applies edits to a scratch copy of the tree. The converted tests meet all 25 expectations: 8 format-only edits pass, 17 behaviour edits are caught. Against the same edits, the 0.285 tests failed 6 of the 8 format-only edits (false alarms). Of the 13 behaviour edits they cover they missed 3: an unflagged compare error, the fast-lane fallback, and a removed deprecation marker. They also missed an unflagged error in a reflowed catch block, and did not cover 4 more at all.

### Fixed: expert residency across SRAM and SLC; load-aware SRAM pin (MODEL §231.2)
- Bug (since 0.263): with `pin` / `pin_bw` and experts pinned in SRAM, the SLC expert hit share was computed relative to the off-SRAM experts, s/(E−p), but applied to every expert read. So the SLC was credited with more expert bytes than missed the SRAM: up to 2× when the SLC held part of the rest, and negative DRAM bytes (a division-by-zero crash) when it held all of it. This happened even under uniform load.
- Now the experts are ranked by read probability across both on-chip tiers. That is the 0.192 pin rule (DRAM bytes saved per step per byte) 「推导」.
  - The SRAM serves H(p/E) of expert reads; the SLC serves H((p+s)/E) − H(p/E); DRAM serves the rest.
  - H is the share of expert reads hitting the most-read resident byte fraction (the 0.261 `dip_exp_hit`, independent reference model 「近似」). Under uniform load, H(f) = f.
  - `dip` takes its off-SRAM experts as the less-read remainder after the SRAM pin (was: a load-neutral subset).
- Closes the open AGENTS item "load-aware `pin`" for on-chip SRAM: with `serving.moe_expert_load`, the SRAM-pinned experts are the most-read ones. New `MemPlan.sram_exp_hit` and `memplan.exp_miss`; the API's `mem` gets `sram_exp_hit` only when a load is given.
- `slc_frac_exp` / `slc_frac_exp_bytes` are now shares of all expert reads/bytes (were: of the off-SRAM experts).
- Why this item: it is a wrong number, not a missing refinement. It decides SLC sizing on SRAM-rich designs, and the candidates named for this release move conclusions far less:
  - MXFP8 activation scales only change the SRAM staging reserve (+3.1 % of one op's fp8 activation bytes; streams and collectives use the model act_fmt, which is bf16 for that release);
  - no other release has a quantized vision tower (now a data test);
  - there is no public kernel trace for the colocated vocoder (sglang-omni #581 only states that a DAC decode is ~30 launch-bound kernels).
- **Default behaviour, old → new:**
  - Catalog chip presets have no SLC, so default results are unchanged: 5 presets × 29 MoE LLMs × SLC 0.25–16 GiB give identical summaries, and fingerprints are unchanged.
  - Changed: designs whose SRAM pins experts and that have an SLC. On 100T with sram_mib 4 / 16 GiB and SLC 1 / 4 / 16 GiB, 87 of 261 points moved; none moved at SRAM 1 GiB. TPOT went up by 0 to +83 % (old results were optimistic), e.g.:

    | config | old TPOT | new TPOT |
    |---|---|---|
    | lfm2.5-24b-a2b, SRAM 16 / SLC 16 GiB | 3.24 ms | 5.92 ms |
    | hy-memory-30b-a3b-fp8, SRAM 16 / SLC 4 GiB | 2.82 ms | 4.34 ms |

    8 points that crashed with division by zero now evaluate.
  - A measured expert load now also helps the SRAM. qwen3-30b-a3b on 100T with 4 GiB SRAM and a Zipf-1 load 「假设」: the SRAM serves 33.6 % of expert reads (byte share ≈ 3.2 %), and TPOT goes 21.4 → 15.6 ms.

### Freshness
- HF sweep since 2026-10-20 (88 orgs): only unsloth/sam3.1 (a community re-upload, mask generation). Web check: Beam and Muse Spark Open are already catalogued; the 10-20 news items were published before the window. Nothing new and official in scope.

## [0.285.0]

### Fixed: static (GitHub Pages / Pyodide) build — every UI endpoint classified, insight panels stale-safe (MODEL §230.1)
- Audit of the features added since 0.270 on the static build: speech insight with the op-dispatch column, speech streams vs SLC, the LLM dispatch panel (`/api/insight_dispatch`), and `dispatch_sens` in eval. All of them already reach `accel_dse.serve.respond` in the Pyodide worker, and the slow-lane list named every insight route.
- Gaps found and fixed:
  - **Unclassified routes.** The shim only listed slow routes, and anything else silently ran on the fast lane. The shim now also lists the fast routes. A UI call in neither list fails a new test (`test_every_ui_call_classified_in_shim`). At runtime an unknown route still works: it goes to the abortable slow lane with a console warning.
  - **Mixed scenarios in speech streams.** The panel rebuilt the request body for every model row, so an edit mid-run mixed chips across rows. It now sends one scenario snapshot for all rows.
  - **Stale panels.** Speech insight and the LLM dispatch panel had no stale handling. They now record the scenario version when they start; a result that lands after an edit, or a panel already shown when an edit happens, gets "场景已改变，需重新运行" appended. Speech streams gets the same tag.
- New `scripts/ui_static_smoke.py` (opt-in; Playwright + Chrome + the Pyodide CDN; prints SKIP if the CDN is unreachable). It builds dist-static into a temp dir, serves it, and in headless Chromium:
  1. checks every endpoint app.js calls is classified in the shim, answered by the Pyodide worker, and sent to the expected lane;
  2. drives the eval walls note with `dispatch_sens`, the dispatch panel, speech insight with the dispatch column, and streams vs SLC;
  3. checks that a re-click kills or drops the running slow-lane request, that a scenario edit tags the panel stale, that the progress pill shows, and that inputs stay enabled while a request runs.
- `scripts/ui_headless_smoke.py` now runs it at the end.

### Changed: quantized vision tower runs and is stored at the release's formats (MODEL §230.2)
- Picked as this release's next highest-value open item. It is the only remaining one that breaks the as-released-dtypes rule, and it changes the encoder's MAC format on fp8-capable chips. Of the other open items:
  - the colocated vocoder, host CPU model and TTFA queueing each need an external measurement;
  - MXFP8 activation scales are ≤ 1/32 of activation bytes;
  - the CosyVoice speech tokenizer is 1.6 % of first audio.
- `summarize_release` writes `vision_fmt` (weight format, params, bytes and bits per tower-linear kind: attn / mlp / merger / patch) only when a vision weight is quantized. Of all releases, only Qwen3.8-Flash-Next-FP4-Local has one, and its JSON gained that key alone.
  - Per the safetensors headers 「来源」: `visual.blocks.*.attn.{qkv,proj}` are F8_E4M3 with U8 e8m0 per-32 scales, i.e. MXFP8, 8.25 bits. `mlp.linear_fc{1,2}` and `merger.linear_fc{1,2}` are U8-packed e2m1 with e4m3 per-16 scales, i.e. W4A16 NVFP4, 4.5 bits. `patch_embed` is bf16.
- `VisionSpec.wfmt` / `w_Bpp` (defaults: () = bf16 everywhere, and 2.0 bytes per param). `model._vision_formats` takes the GEMM input format from ModelOpt's per-role algorithms (`algo_by_role.vision` = [MXFP8, W4A16_NVFP4]): attention gets fp8 inputs (W8A8); MLP and merger keep bf16 inputs (weight-only).
- Stored tower weights now come from the release's vision bytes per param. The CLI VLM line says "按发布格式" instead of "bf16".
- **Default behaviour, old → new:**
  - Every other model: unchanged. Their default-scenario summaries are bit-identical, and every other VLM tower keeps `wfmt` () and 2.0 bytes per param.
  - Qwen3.8-Flash-Next-FP4-Local, measured on 100T with 1 image and a 2048-token prompt:

    | | old | new |
    |---|---|---|
    | tower weights | 0.898 GB | 0.327 GB |
    | vision encode | 64.1 ms | 57.9 ms (−10 %) |
    | TTFT | 648 ms | 642 ms |

### Freshness
- HF sweep since 2026-10-20 (88 orgs) and a web check: only unsloth/sam3.1 (re-upload) and community uploads. Nothing new and official in scope.

## [0.284.0]

### Decided: NPU presets keep ideal dispatch by default (MODEL §229.1; AGENTS §233 deferred item)
- Picked as this release's top open item: it is the only one that moves every small-batch decode number on every NPU preset. The other open items were measured as small:
  - the CosyVoice3 speech tokenizer is 5 ms of 319 ms first audio (1.6 %);
  - MXFP8 activation scales add ≤ 1/32 of activation bytes, in one release;
  - a quantized vision tower exists in one release (Qwen3.8-Flash-Next-FP4-Local).
- Evidence: all 41 catalog LLMs that fit on 100T + LPDDR5X 4×64-8533, decode at batch 1, ctx 1024:

  | dispatch | TPOT slowdown, median | worst | rows above +10 % |
  |---|---|---|---|
  | trace (1.5 µs per op 「二手」) | +1.7 % | +8 % (carbon-500m) | — |
  | host (20 µs per op 「二手」) | +23 % | — | 29 / 41 |

  - The per-op latency that degrades TPOT by 10 % (u_10) has a median of 8.9 µs, ranging from 1.9 to 60 µs.
- Decision: **no default change (old = new = ideal).** The sourced runtime with the lowest per-op cost (trace) moves the median by less than the model's other approximations. Host-launched dispatch is a software-stack choice, not a chip property. Making either the default for a hypothetical chip would be an assumption. Instead the sensitivity is now shown everywhere (below).

### Added: per-op dispatch sensitivity for every evaluation and for LLM decode (MODEL §229.2)
- `Result.dispatch_sens()` (via `core/dispatch_insight.step_sensitivity`): two engine runs, at ideal and at 1 µs per op. It reports:
  - dispatched ops per step;
  - u_10, the per-op latency at which the step slows by 10 %;
  - u_eq, the latency at which dispatch equals all other step time;
  - the slowdown at the trace / host defaults;
  - `exact` = single stage, one microbatch.
- The chip's own dispatch is stripped for the reference. It is returned by `/api/eval` as `dispatch_sens`, not in `summary()`, so fingerprints are unchanged. The UI walls note now reads "派发敏感度 …" and points to where the knob lives (芯片面板 › 高级芯片参数 › 派发 dispatch).
- `core/dispatch_insight.dispatch_sensitivity`, `/api/insight_dispatch` {models, batches, ctx}, and a new insight panel "LLM decode 派发敏感度". Per model × batch it shows:
  - TPOT when ideal and the bound;
  - ops per step and per token;
  - u_10 and u_eq;
  - TPOT at trace 1.5 µs, static 10 µs per step and host 20 µs, each evaluated by the engine.
  - Default set: qwen3-0.6b, llama-3.2-1b, qwen3-1.7b, qwen3-8b, llama-3.1-8b, qwen3-30b-a3b, gpt-oss-20b at batch 1 / 8 / 32.
- On 100T at batch 1:

  | model | ops per step | u_10 | slowdown at host 20 µs |
  |---|---|---|---|
  | qwen3-0.6b | 283 | 2.3 µs | ×1.87 |
  | qwen3-30b-a3b | 531 | 6.0 µs | ×1.33 |
  | gpt-oss-20b | 267 | 7.2 µs | ×1.28 |
  | qwen3-8b | 363 | 22 µs | ×1.09 |

  - u_10 grows with batch, because TPOT grows while the op count does not.
- The hardware panel already had the dispatch mode and µs (since 0.203 / 0.256, inside the "高级芯片参数" fold). It is unchanged apart from the new pointer in the walls note.

### Freshness
- HF sweep since 2026-10-20 (88 orgs) and a web check: only unsloth/sam3.1 (re-upload) and Tess-4.5-27B (community fine-tune). Nothing new and official in scope.

## [0.283.0]

### Changed: components pay the per-op dispatch / kernel cost (MODEL §228.1)
- Audit of the per-op dispatch design variable (`Chip.dispatch` / `dispatch_us`, 0.203: "static" 10 µs per step, "trace" 1.5 µs per op, "host" 20 µs per op; default "" = ideal). Every LLM / backbone stage already paid it, and so did speech decoders built from model layers (Drama 3 tower, Talker, depformer layers inside `evaluate`).
- It was **not** applied to components costed by `evaluate._component_time`: speech cards (flow, HiFT, DiT detokenizer, BigVGAN), codecs (DAC, Mimi), the Index-Echo mapper / tokenizer, and VAE / text encoders. `_sum_ops` counted the per-op cost, but the component's `StageTime` dropped it. The same applied to the GPU per-kernel cost `kernel_us`.
- Now a component adds that serial term like a stage: per op for trace / host, one program launch per execution for static, plus `kernel_us`. The result also carries `sync_s` / `dispatch_s` when non-zero.
- Behaviour change, old → new:
  - **Catalog chips: none.** Their dispatch is ideal and `kernel_us` is 0, so speech insight rows are bit-identical (checked over all 17 speech models on 100T).
  - **Changes only when a knob is set.** Example: the ext default variant (H200, decode `kernel_us` 3.03 µs) now pays 93 kernels per DAC window (+0.27 ms per window). Drama 3 speech diagnostic, predicted / measured:
    - gmean 1.343 → 1.323;
    - c = 64: 1.332 → 1.290;
    - Table 2, vocoder on the same GPU: 1.31 / 1.37 → 1.26 / 1.31.
- **The default stays ideal (no change).** Catalog NPU presets are design points, and the right per-op cost depends on the runtime (graph capture vs host launch). A global default would move every fingerprint and hide the lever. Instead, the sensitivity is reported per speech model (below).

### Added: op-dispatch sensitivity in the speech bottleneck insight (MODEL §228.2)
- `speech_bounds(..., dispatch=True)`, `/api/insight_speech` `dispatch` (default true), and a new UI column "op 调度敏感度" (checkbox). Dispatch is a serial term, so each stage is exactly linear in the per-op latency u: x_s(u) = x_s(0) + n_s·u. The engine measures n_s (dispatched ops per audio second) from two runs at 0 and 1 µs/op. Each row reports:
  - u_rt: the per-op latency where real time is lost;
  - u_eq: the latency where dispatch equals all the other time of the binding stage;
  - x at trace 1.5 / host 20 µs;
  - first-audio ms per µs/op.
- On 100T + LPDDR5X 4×64-8533 at batch 1, real time is lost at:

  | model | u_rt (µs/op) | dispatched ops per audio second |
  |---|---|---|
  | Drama 3 | 21 | 15.3 k |
  | Ming-TTS-tiny | 33 | 17.8 k |
  | Index-Echo-9B | 32 | 8.4 k |
  | FireRedTTS3 | 54 | 8.0 k |

  - Ming-TTS-tiny is already dispatch-dominated (more dispatch time than DRAM time) above 23 µs/op.
  - At host-style dispatch (20 µs), Drama 3 sits at x 0.98, against 0.67 when ideal.
  - ASR rows are insensitive (u_rt > 240 µs).
  - Rows not real time even when ideal (Moshi, Kimi-Audio, Step-Audio-2-mini, Qwen2.5-Omni) report u_rt = 0.

### Investigated: colocated-vocoder gap (MODEL §228.3, unresolved)
- Measured same → second GPU speed-up: 1.70× (128 streams) / 1.72× (256). Modelled: 1.17× / 1.21× before this release, 1.22× / 1.27× now (DAC kernel cost).
- SGLang-Omni @ 0029194 `vocoder.py` / `stream_windows.py`: window graphs are captured at lengths 10 / 20 / 32 / 64 / 96. Every window decodes padded to the next captured length, and only the final partial window pads, so the extra work is negligible. The decoder runs fp16 channels-last, which is cheaper than the fp32 we cost, so that goes the wrong way.
- The modelled vocoder is 6.4 ms per request. The measured split implies about 20 ms, 3.1× more. The unmodelled candidate is two-process GPU time-slicing without MPS (engine kernels wait for the vocoder's time slice). It cannot be checked without a kernel trace. Not fitted.

### Freshness
- HF sweep since 2026-10-20 (88 orgs) and a web check: only unsloth/sam3.1 (re-upload) and community fine-tunes. Nothing new and official in scope.

## [0.282.0]

### Added: Drama 3 on one H200 as an external speech measurement (MODEL §227.1)
- New `data/ext_measurements_speech.json`: the SGLang-Omni Drama 3 cookbook (also the HF model card's "Production Serving"), Tables 1–3 on one H200. Table 1: streaming frames/s and TTFA at concurrency 1–64. Table 2: concurrency 128 / 256 with the vocoder on the engine's GPU or on a second GPU. Table 3: non-streaming latency and RTF at concurrency 8.
- New `core/extspeech.py` (diagnostic; never fitted, never a cost-model default). It evaluates the engine on the H200 reference chip under an ext variant and replays the measured serving loop from sourced settings:
  - closed loop at concurrency c: c prefills (batch 1, serial) + N decode steps at batch c + the vocoder, all time-sharing one GPU;
  - N = 95 frames: 4.40 s of audio = Table 3 latency / RTF;
  - SGLang-Omni vocoder windows (`config.py` / `vocoder.py`): strides 10, 10, 12, 32, then 64, each re-decoding up to 32 earlier frames, one stream per decode.
- `scripts/ext_validate.py` writes a separate `speech` section (default and catalog variants). The LLM summary is unchanged.
- What explains the 0.276 "≈ 2.2× optimistic" gap (frames/s, predicted / measured):
  - catalog roofline (no GPU per-kernel cost): 3.13× at c = 1, 1.63× at c = 64;
  - with the ext default variant (per-kernel floor from the DeepSeek decode trace, 0.136 / 0.176; not refitted): 1.24× at c = 1, 1.33× at c = 64, geometric mean 1.34×.
  - At c = 1, per-kernel cost is 58 % of the 4.65 ms frame step (2.7 ms of fixed cost against 1.75 ms of DRAM time).
  - At c = 64, request prefill (≈ 150 rows per 4.4 s utterance) is a third of engine time.
- Table 2 isolates the rest. Vocoder on a second GPU: 0.91× / 0.97× at 128 / 256 streams. Vocoder on the same GPU: 1.31× / 1.37×. The measured same → second speed-up is 1.70×, the model gives 1.17×, so the colocated vocoder is the remaining gap (the codec on GPU is costed ~4× too cheap, or the two processes' time-slicing costs more than the sum).
- TTFA is under-predicted by 1.5–2× (HTTP / process hops, queueing and `prefill_decode_interval` are not modelled).

### Fixed: hyper-connection scales are parameters (MODEL §227.2)
- `summarize_release.NOT_SCALE_RE`: `hc_attn_scale` / `hc_ffn_scale` (3 per layer) and `hc_head_scale` (1) are learned parameters of the manifold hyper-connections, not quantization scales. The templates already counted them, so DeepSeek-V4-Flash / V4-Pro / V4.1-Flash and GLM-5.3-Flash are now exact (gaps of 259 / 367 / 240 / 270 parameters → 0; MTP still exact).
- Dory `lm_head.logit_scale` (131,072 values) is left as a scale: the template does not count it either, so switching would open a gap.

### Freshness
- HF sweep since 2026-10-20 (88 orgs) and a web check: nothing new and official in scope.

## [0.281.0]

### Changed: per-role activation formats for the remaining quantized releases (MODEL §226.1)
- Audit of every quantized release in the catalog (38) for GEMM input formats. ModelOpt NVFP4 / MIXED_PRECISION (0.254 / 0.280), `fp8` + dynamic / static activations, and compressed-tensors W8A8 float were already right. Three were not; up to 0.280 they used bf16 inputs everywhere:
  - **AMD Quark PTPC-FP8** (`muse-spark-open-fp8`): `global_quant_config.input_tensors` = fp8_e4m3, dynamic, per_channel → vLLM `QuarkW8A8Fp8` quantises inputs per token to fp8. Now act_attn / act_expert fp8 (the roles stored in fp8). bf16 mlp / lm_head keep bf16 inputs.
  - **AMD Quark MXFP4** (`muse-spark-open-mxfp4`): input fp4, group 32, e8m0, dynamic = W4A4 MXFP4 (native on MI355, emulated by dequantising elsewhere). Now act_expert mxfp4; `gemm_exec` picks native or bf16 from the chip's formats.
  - **Drama 3 `bsgemm-moe`**: block-FP8 routed experts with `activation_scheme` dynamic, served by vLLM block-FP8 MoE kernels → act_expert fp8.
- `model._quark_act_roles`: Quark layer / layer-type overrides and mixed W/A families (W4A8) return no activation formats; none of the catalog's Quark releases has them.
- Effect (4096-token prompt, TP8·EP8; TPOT unchanged):
  - B200: Muse FP8 TTFT 76.2 → 70.8 ms; Muse MXFP4 76.1 → 66.4 ms (native fp4).
  - 100T + LPDDR5X: Muse FP8 TTFT 1.102 → 1.026 s (TP8), 0.749 → 0.716 s (TP16). Muse MXFP4 unchanged (no fp4 MACs; W4A4 emulated as bf16 either way).
  - Drama 3, 100T: batch 1 unchanged (decode-bound); 32 clips first audio 1.647 → 1.630 s.

### Fixed: fused-expert scale suffix (MODEL §226.2)
- Audit of all 153 release JSONs (bits / weight per role and format) and the 93 cached header sets (names counted as weights that look like scales) for the 0.280 "csf" class of bug. One case: Muse-Spark-Open NVFP4 stores fused 3-D experts (`experts.gate_up_proj`) with `…_weight_scale`, `…_weight_scale_2` and `…_input_scale`. `_weight_scale_2` (fp32, one per expert) was counted as 46,080 parameters, and the per-16 e4m3 scales were attached to it by prefix match.
  - Role bytes and the per-role effective bits were already right (4.871 b), so no throughput number changes. The role table now shows experts as nvfp4 at 4.5 b + the bf16 experts of layers 0 / 61. LLM params 819,990,598,144 → 819,990,552,064 (param check −1.2e-6), and bytes_llm +184,320 (input scales now attached to their experts).
  - `SCALE_RE` gets `_(weight|input)_scale(_2)?$` as its first alternative, so the whole suffix is stripped. Re-summarising every other cached release changes nothing.
- The reverse case (real parameters counted as scales) is negligible: DeepSeek-V4 / GLM-5.3 `hc_*_scale` (3 values per layer) and Dory `lm_head.logit_scale` (131,072). Left as is.

### Freshness
- HF sweep since 2026-10-19: nothing new and official in scope (Hy-Memory GGUF, a MiniCPM OCR processor stub, SAM 3.1).

## [0.280.0]

### Added: nvidia Qwen3.8-Flash-Next-FP4-Local (MODEL §225.1)
- New catalog entry `qwen3.8-flash-next-fp4-local` (2026-10-08). It is NVIDIA's ModelOpt MIXED_PRECISION build of the catalogued Qwen3.8-Flash-Next, with dtypes as released. Family "Qwen3.8（NVIDIA 量化）", listing domain vlm. The config is identical to the base model apart from `ple_embedding_dtype` and the quantization config, and so are the image / video preprocessors.
- Formats, from the safetensors headers (375,487 tensors; LLM 176,943,899,520 parameters, exact; MTP and vision exact):
  - routed experts: NVFP4 with compressed scales ("csf": u8 fixed part + u32 exceptions), 4.26 bit / weight;
  - attention, linear attention and shared experts: MXFP8 (e4m3 + e8m0 per 32), 8.25 bit;
  - PLE n-gram table: NVFP4;
  - MTP experts and vision MLPs: W4A16 NVFP4;
  - gated-residual mixers, PLE projections, router and lm_head: BF16.
- `summarize_release`:
  - the csf scale tensors are scales, not parameters;
  - `attn_` / `mlp_hyper_connection` → role `other`. The base Qwen3.8-Flash-Next release is re-summarised; its numbers are unchanged because everything is BF16.
  - `_compact_qc` also handles quant_method `modelopt` + quant_algo MIXED_PRECISION, adds a csf suffix to the per-algorithm counts, and writes `algo_by_role`.
- Activation formats for ModelOpt MIXED_PRECISION now follow each role's per-layer algorithm (`model._w4a4_roles`):
  - "NVFP4" = W4A4, "W4A16_NVFP4" = weight-only, "FP8" / "MXFP8" = W8A8.
  - This also changes **Nemotron-3.5-Super-VL**: its FP8 attention / Mamba projections and lm_head now take fp8 inputs, and its NVFP4 experts take nvfp4 inputs; up to 0.279 every input was bf16. TTFT at a 4096-token prompt on 100T TP2·EP2 goes from 0.615 s to 0.516 s; TPOT is unchanged at 29.9 ms.
- Qwen4Exp gated-residual and PLE Linears carry role `other` (`Linear.role` is now honoured for attention-block Linears), so in the FP4 build they stay BF16 while attention is MXFP8. The PLE row lookup reads at the table's stored dtype.
- On 100T + LPDDR5X (64 GB cards), FP4-Local vs BF16:
  - TP2: fits (47.9 GiB), TPOT 24.3 ms; the BF16 model needs TP8.
  - TP4: TPOT 17.2 ms (the BF16 model does not fit).
  - TP8: 13.0 vs 20.6 ms.
- The catalog dtype label shows per-role activations (e.g. `A fp8/专家 nvfp4`).

### Added: Index-Echo in the speech bottleneck and streams-vs-SLC views (MODEL §225.2)
- `speech_insight.SPEECH_MODELS` and the UI's speech-output list now include `index-echo-s2st-2b` / `-9b`.
  - New stage `pipeline`: the whole serial, non-streaming S2ST pipeline, with x = total / output audio seconds.
  - The SLC estimate uses the re-read rate of the dominant decode stage.
- `speech._echo_kpis`:
  - b = serving.batch concurrent clips share the card. The ST / CV3 LM steps are batched; mapper, flow, HiFT and the speech tokenizer run per clip. b = 1 is the 0.279 result plus the new terms below.
  - `lm_only` is supported.
  - New `repin_s`: the ST and CV3 phases time-share SRAM / SLC, so each round re-loads both pinned sets once.
- Streams on 100T (no SLC → 8 GiB SLC): 2B 38 → 58, 9B 12 → 30.

### Added: CosyVoice3 frontend from its ONNX graphs (MODEL §225.3)
- `scripts/fetch_onnx_graph.py`: a stdlib protobuf walker over HTTP Range reads the graph structure (nodes, attributes, initializer shapes) and skips the weights. `frontend_ops` tracks each activation's time / frequency divisor through the convolution strides. `--release` writes the descriptors into the release key `onnx_frontend`.
- speech_tokenizer_v3 (242,009,608 F32):
  - two stride-2 convolutions → 12 blocks (q / k / v / o 1280², FSMN depthwise k31, full attention 20 × 64, GELU MLP 5120) → FSQ projection;
  - 124.8 GFLOP per 10 s;
  - costed on the card (`core/onnxfront.py`; onnxruntime CUDA EP in the reference frontend), 5.1 ms per 10 s clip on 100T.
- CampPlus (6,907,680 F32; CAM++): runs on the host CPU in the reference frontend (CPUExecutionProvider). Its 11.3 GFLOP per 10 s are reported and not charged to the card.
- Index-Echo coverage stays partial, with one reason: CampPlus on the host CPU and the CPU feature extraction.

## [0.279.0]

### Changed: test_core_237 phase-law guard (MODEL §224.1)
- `test_runtime_guard_phase_law_paths` now guards the deterministic work as its primary check: the call counts and summed result sizes of the eight phase-law kernels (`_phase_law`, `_phase_iters`, `_queue_weights`, `_queue_k_law`, `_decode_birth_death`, `_occ_tau`, `_mg1_busy`, `_slot_backlog`) per case, each ≤ 1.5 × the measured table + 10 (`_PHASE_WORK`).
- The CPU bounds stay as they were when run alone (45 s / 90 s). Under the parallel runner they get × 6 headroom, because measured `process_time` inflated 9.5 × under -j 8 for identical work (vllm_v0: 14.3 s alone, 136 s in-suite). A real slowdown in work is still caught by the counters, independent of load.

### Added: Index-Echo S2ST 2B / 9B (MODEL §224.2)
- New catalog entries `index-echo-s2st-2b` and `index-echo-s2st-9b` (bilibili IndexTeam). Speech → translated, voice-preserved speech in one process, listing domain omni.
- `scripts/fetch_pt_release.py --echo`: the ST LM shapes are read from the torch_dist `.metadata` (stub unpickler, no torch), and the AuT / connector / mapper shapes from the safetensors headers. Release JSONs:
  - 2B: LLM 1,881,825,088 (tied), audio 656,315,778, bridge 24,586,624;
  - 9B: LLM 8,953,803,264, audio 656,315,776, bridge 26,425,728.
  - All exact.
- Pipeline costed by `speech._echo_kpis` (kind `s2st`): AuT encoder (Qwen3-Omni audio tower) + connector → ST prefill → greedy decode (transcript + translation) → a second full forward for hidden states (no KV reuse) → Hidden2CVMapper (fp32, `core/echo.py`) → CosyVoice3 LM (fp32; speech tokens clipped to [2m, min(20m, 1500)]) → one flow pass + one HiFT pass (stream=False). CosyVoice3 reuses the catalog `fun-cosyvoice3-0.5b-2512`, whose files are byte-identical.
- On 100T + LPDDR5X, a 10 s clip → 10 s output:
  - 2B: 3.23 s, RTF 0.32, 11.3 GB, dominated by CV3 decode (1.83 s);
  - 9B: 7.30 s, RTF 0.73, 25.5 GB, dominated by ST decode (5.15 s).
- Coverage is partial: CosyVoice3's speech_tokenizer_v3 and CampPlus (ONNX) are not costed. Text tokens per second of speech (3.0) is an assumption.
- Web: an S2ST KPI block. Counts: releases 152 (llm 126), listed + unlisted 158.

### Added: Nemotron-3.5-Super-VL video (MODEL §224.3)
- `serving.videos` now works for `nemotron-3.5-super-vl-120b-a12b-nvfp4` (`vision._radio_video`).
  - Each frame follows the processor's video rule (`radio_video_patches`: ≈ 1024 patches with the aspect kept and even sides).
  - Frames are grouped in tubelets of 2, one tower pass each.
  - EVS pruning (video_pruning_rate 0.7) keeps max(one tubelet, ⌊t · h·w/4 · 0.3⌋) video tokens, plus `<img>` / `</img>` per tubelet.
- Example: 32 frames at 1280×720 → 16 tubelets of 24×42 patches, 1241 LLM tokens, 24.5 TFLOP of vision. At TP2·EP2 on 100T, TTFT goes from 0.615 s to 0.901 s at a 4096-token prompt.
- Coverage is still partial (one reason). The tubelet `video_embedder` (2 frames' patches) is costed as a single-frame patch embed, and cls tokens are still left out.
- The `frames_per_slice` vision field is now frames / slices.

## [0.278.0]

### Added: NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (MODEL §223.1–§223.3)
- New catalog entry `nemotron-3.5-super-vl-120b-a12b-nvfp4` (2026-10-12, as released: ModelOpt mixed precision, KV cache FP8). Family "Nemotron-3.5 Super VL"; the NVIDIA provider label is now "NVIDIA（Nemotron）".
- Release summary from the safetensors headers: LLM 120.669 B, MTP 2.942 B, vision 0.842 B, 74.99 GB.
  - Formats: experts and shared experts NVFP4 (group 16); Mamba in/out, q/k/v/o and lm_head FP8; embeddings, latent projections, router, conv and norms BF16.
  - `summarize_release._compact_qc`: the ModelOpt per-module lists (config.json is 8.7 MB) become counts per algorithm (`quantized_layers_by_algo`, `ignore_count`).
  - `_nemotron_h_omni_role`: `language_model.` prefix stripped; RADIO / projector → vision; latent projections → mlp; `backbone.norm_f` → norm.
- `model._build_nemotron_h` (Nemotron-H, one mixer per layer, from `layers_block_type`): 40 Mamba-2, 8 GQA (32 / 2 heads, head 128), 40 latent MoE (512 experts top-22, relu² non-gated, latent 1024, shared expert 5376). The one MTP module (attention + MoE) is one combined layer. Parameters match the release exactly (LLM and MTP).
- Mamba-2 (`model._mamba2`): SSD linear core, 128 heads × 64, state 128, B / C shared by 8 groups, conv kernel 4, chunk 128. In `ir`, the chunked prefill forms C·Bᵀ once per (chunk, group) instead of per head (`AttnCore.state_groups`); the other chunk GEMMs stay per head.
- Vision: new `radio` family in `core/vision.py`. C-RADIO ViT-H/16 (32 blocks, 1280, GELU MLP 5120, biased qkv) → final LayerNorm → 2×2 pixel shuffle → mlp1 (RMSNorm 5120 → 20480 → ReLU² → 4096).
  - Image size follows the processor's dynamic single-tile rule (`vision.radio_patches`, from `_compute_target_patches`). The patch budget is min((16384 − 4)·4, 13312), with a floor of 1024 patches and even sides. An image adds patches / 4 + 2 tokens (`<img>` … `</img>`).
  - The modelled tower + projector is 840.40 M. The remaining 1.97 M of the release (video_embedder + normalizer buffers) is stored only.
  - Coverage is partial: the 10 cls / register tokens are left out of the attention length (< 1 %), and the video path is not modelled.
- On 100T + LPDDR5X:
  - one card does not fit (67.8 GiB > 64);
  - at TP2·EP2: TTFT 0.615 s at a 4096-token prompt, 0.792 s with one 1024² image (1026 tokens, 8.3 TFLOP vision); TPOT 29.9 ms.

### Added: Drama 3 DAC DRAM split (MODEL §223.4)
- `speech_kpis` for drama-3 now reports `dac.dram_weights_GB` and `dac.dram_act_GB` (`speech._dac_window_dram`). They are the same op sums as the costed time.
- Per 10 s request on 100T: weights 5.9 GB at both b = 1 and b = 8; activations 14.3 GB at b = 1 and 126.3 GB at b = 8.
- SLC pinning of the DAC weights was **not** added. Activations dominate, and per GB the LM weights (read every step) are 27–64× more valuable to pin (10 s average to steady-state 64-frame windows). The stream lost at large SLC in 0.277 comes from DAC compute and activation traffic, not from weights.

## [0.277.0]

### Changed: faster streams-vs-SLC calculation, same results (MODEL §222.1)
- `speech_insight.speech_streams` output is byte-identical to 0.276 for every model except Drama 3, whose DAC is now costed (below). Checked by `cmp` of the full 15 models × 6 SLC JSON and by `tests/test_core_277.py`, which runs the 0.276 algorithm verbatim on moshi / Fun-CosyVoice3 / Kimi.
- Native full run (15 models × 6 SLC sizes): 128 s → 46 s on the box. Engine reruns went from 936 to 679. The savings come from:
  - `speech_kpis(..., lm_only=True)`. Omni, CosyVoice, Kimi, Moshi and Drama 3 return only the LM-card stage loads and the fit verdict, using the same expressions as the full KPIs. That skips speech-card timing and Moshi's own streams search, which the LM-card search never reads.
  - One memo of batch → verdict per SLC point, shared between the search and the b + 1 limit probe.
  - A near search: from the previous SLC size's answer, step ±1, ±2, ±4, then bisect. It is exact because x(b) is monotone, which was already assumed.
  - Memoised pure functions: `mapping._sub_chip` (core-group sub-chips), `evaluate._tts_step_flops`, and `speech._firered_fixed` (prompt encode + RedAE decode).
- UI: the "实时路数 / 卡 vs SLC" table sends one request per model and appends rows as they arrive. Status reads "计算中 i/n 模型". The run token is checked, so a newer run discards older replies (the existing per-channel stale discard is kept). In the static (Pyodide) build one model takes about 3–12 s in the slow-lane worker.

### Added: Fish Audio Drama 3 DAC decoder cost (MODEL §222.2)
- `scripts/fetch_fish_codec.py` reads codec.pth (1.87 GB torch zip pickle) with `fetch_torch_ckpt`: 1.5 MB fetched, no torch needed. It writes the decode-path shapes into the release JSON (`codec_tensors`, `params_codec` 391.4 M, `params_codec_decode` 184.5 M, F32). `summarize_release.py` picks up the same keys from `fish_codec.json`.
- `drama3.dac_ops(fs, new, ctx, hist, batch)` follows fish-speech `rvq.decode` + `modded_dac.Decoder`: codebook lookups → out_proj → 8-layer causal window transformer (dim 1024, window 128) → 2 × (ConvTranspose 2 + ConvNeXt) → conv7 1024 → 1536 → 4 DecoderBlocks (strides 8, 8, 4, 2, derived from the shapes) → conv7 → 1. That is 6.77 GFLOP per frame, matching an independent hand count (test). Windows follow the vllm-omni ramp of 6, 8, 12, 16, 24, 32, then 64 frames (`dac_windows`).
- `_fish_kpis`: the DAC is colocated with the LM and time-shared. The recipe runs both stages on one GPU, so RTF = LM RTF + Σ DAC(window, b) / audio. The b streams batch each window (vllm-omni `per_step_batch`). Decoding is stateful; the re-decode of left context is reported as `dac.redecode_s_per_request`. First audio adds DAC window 0. The 0.74 GB of fp32 decode weights count toward the card's memory fit.
- On 100T + LPDDR5X at batch 1:
  - DAC load 0.0106 (LM RTF 0.660 → 0.671);
  - first audio +5.0 ms;
  - streams per card 3 / 3 / 4 / 5 / 7 / 8 at SLC 0 / 0.5 / 1 / 2 / 4 / 8 GiB, against 3 / 4 / 4 / 5 / 8 / 9 in 0.276.
- Coverage: drama-3 goes from partial to full.

### Changed: test_core_237 runtime guard
- The primary guard now counts deterministic work per case (`_Pool.run` and `_chunk_fused` calls at ≤ 1.5 × the 0.277 counts), which does not depend on load.
- The CPU-second bounds are kept: 45 s / 90 s when run alone, × 2 when `scripts/run_tests.py` runs files in parallel (new env var `ACCEL_TEST_JOBS`).

## [0.276.0]

### Added: real-time streams per card vs SLC size (MODEL §221.1)
- `core/speech_insight.speech_streams(base, models, audio_s, slc_gib)` and `POST /api/insight_speech_streams` (keys `models`, `audio_s`, `slc_gib`). For each speech model and each SLC size (pin policy, default 0 / 0.5 / 1 / 2 / 4 / 8 GiB), the engine is rerun at increasing batch:
  - `streams_lm`: the largest batch whose non-speech-card stages stay real time (x ≤ 1) and fit. Found by a galloping search from the linear prediction, then bisection; capped at 1024.
  - `lm_limit`: why batch + 1 fails (realtime / memory / cap).
  - `streams_speech_card` (CosyVoice / Kimi / Step-Audio / Omni): the speech card's own real-time stream count. Speech cards = ⌈streams_lm / streams_speech_card⌉.
  - `streams_per_card` = streams_lm / (LM cards + speech cards).
  - `streams_lm_pred` = ⌊(1 − (x₁ − d)) / d⌋ with d = x₂ − x₁. This is the linear read-out from batch 1 and 2, valid only while the step stays weight-read bound. It is reported next to the engine result, never used in place of it.
- Insight tab, speech panel: "实时路数 / 卡 vs SLC" table. One row per model, one column per SLC size; hover shows the LM / speech-card split. By default it covers the speech-output models, with an "含 ASR 模型" checkbox to add the rest. Static build: routed via `serve.respond`, in the shim slow lane, with an unpacked-zip probe. `ui_headless_smoke` clicks it with SLC 0, 4.

### Added: FireRedTTS3 DiT SLC pin and bf16 re-cast option (MODEL §221.2)
- `_firered_kpis` gains a `dit_pin` block that reports the existing 0.269 re-read rule (memplan `rr_w` / `rr_reads`). The fp32 DiT is 1.17 GB, read 10× per step. The block reports:
  - SRAM-pinned and SLC-held bytes, the miss share and DRAM GB per step;
  - `slc_full_GiB`, the SLC needed to hold the whole DiT: 1.03 GiB on 100T.
  - Step time falls linearly with the pinned bytes (test). On 100T + LPDDR5X, TPOT goes 89.8 → 61.7 → 33.7 → 27.3 ms at SLC 0 / 0.5 / 1 / 2 GiB.
- `bf16_recast` option, not counted in TPOT. `_backbone_one_step` runs under `torch.autocast(bf16)` over fp32 weights, and the cast cache is dropped every step. That costs +4 B per non-embedding backbone param per step on an eager GPU path: +5.64 GB, +29.5 ms per step, RTF 0.57 → 0.75. UI: one line in the TTS KPI.

### Added: Fish Audio Drama 3 (MODEL §221.3)
- `fishaudio/drama-3` as `drama-3` (provider Fish Audio, family Drama 3, listing domain `tts`).
- Release headers: `summarize_release._fish_role`. The backbone and the 13-layer codebook-MTP tower are LLM roles; the 9 codebook heads are `lm_head`; code embeddings / tables are `embed`.
- The tower (vllm-omni `codebook_tower.py`) runs on every backbone row with its own causal KV cache. It is modelled as 13 extra decoder layers: GQA 16/2, head 256, sigmoid output gate, SwiGLU 6144, eh_proj. The 9 heads are per-sequence rows.
- The head reads only the 4096 candidate lm_head rows. The 252,288-row text embedding is looked up.
- Resident params = release − non-candidate lm_head rows − the folded eh_proj halves = 35,341,014,144 (gap 0).
- `speech._fish_kpis` (streaming): step = 1 frame of 46.4 ms. First audio = prefill (text + one row per reference frame) + 6 frames (the first chunk of the async-chunk ramp). The DAC decoder (codec.pth) is not costed.
- Added to `SPEECH_MODELS`, the UI badge / KPI line and the headless model list.

### Tests
- New `test_core_276` (streams search, Moshi curve, speech-card sizing, route / UI / shim, FireRed pin / recast) and `test_core_276b` (Drama 3).
- Listing count tests: 149 files / llm 123 / 155.

## [0.275.0]

### Added: speech bottleneck panel (MODEL §220.1)
- Insight tab, new "语音瓶颈" panel. It calls `POST /api/insight_speech` with the left-hand chip and memory, an audio-seconds input and a verify toggle. For each speech model the table shows:
  - binding stage, x and real time;
  - first audio and its dominant part;
  - the bandwidth or SLC needed for real time, with the verified x;
  - the per-stage x.
- Static (Pages) build: `/api/insight_speech` goes through `serve.respond` like every UI call (`test_static_routes` parses app.js). It is now in the shim's slow lane, together with `/api/insight_slc`, which was missing. The unpacked-zip probe also calls it.
- `ui_headless_smoke` clicks the panel and expects ≥ 10 rows.
- Fix: speech-insight rows take only the scenario's hardware / system fields. The UI model's serving (batch, draft model, spec), layout, format overrides, workload and PD no longer leak into speech rows; a draft model used to make Moshi look like x = 0.61. Verify adds the derived SLC on top of any SLC the chip already has.

### Added: FireRedTTS3-Base (MODEL §220.2)
- `FireRedTeam/FireRedTTS3` as `fireredtts3-base` (provider FireRedTeam, family FireRedTTS3, listing domain `tts`, coverage `full`).
- `fetch_pt_release.py --firered`:
  - The backbone config (Qwen3-1.7B: 28 × d2048, GQA 16/8, head 128, vocab 151,936, no lm_head) is a dict in `fireredtts3_base.py`, transcribed with its source.
  - `generate()` defaults: 10 Euler steps on a cosine schedule, CFG 2.0.
  - It reads the base and RedAE headers.
- Params exact (all fp32):
  - backbone 1,720,574,976;
  - speech head 400,057,921 (role `tts`: DiT 293.7 M, patch encoder 102.9 M, dit_head / stop / speaker projections);
  - RedAE 943,776,322 (role `codec`).
- `core/fireredtts3.py`:
  - Per AR step (4 latents = 160 ms of audio): backbone step, stop_head, dit_head over 3 conditions, then the DiT flow head. It runs 10 evaluations × CFG 2 over 12 positions: 11 blocks with adaLN 9d, attention, two k3 convs and a 3d MLP. The patch encoder follows, over cls + 4 positions.
  - RedAE decodes every latent once, after generation (non-streaming): Qwen3 18 × d896 with sliding window 64, then the ISTFT head.
  - Per request: RedAE encoder plus patch encoder on the prompt audio.
- The FireRedTTS3 path in `speech_kpis` (kind `tts`, label FireRedTTS3):
  - first audio = prompt encode + TTFT + steps · TPOT + RedAE decode, i.e. the whole segment;
  - `serving.audio_s` is the voice-prompt length (default 5 s 「假设」).
- 100T + LPDDR5X, 10 s out:
  - TPOT 89.8 ms per 160 ms step, so RTF 0.57: real time;
  - first audio 5.76 s (63 steps), because nothing streams;
  - the fp32 DiT (1.17 GB) is re-read 10 times per step (12.2 GB of head weights per step), the main SLC pin candidate.
- Shared code: `memplan` / `evaluate._tts_info` use the spec's weight bytes (`w_bytes`, fp32 = 4) instead of a fixed 2; `ir._head_ops` dispatches to `fire_head_ops`. Ming TTS is unchanged.
- UI: FireRedTTS3 badge and a non-streaming split in the TTS KPI. `ui_headless_smoke` also loads fireredtts3-base. The speech insight now includes fireredtts3-base.

### Added: Moshi per-layer overhead (MODEL §220.3)
- The existing `Scenario.layer_overhead_us` (0.69, off by default, so results are bit-identical) now also charges Moshi's serial layer passes per frame: 48 depformer one-row passes and 8 + 8 Mimi transformer layers. The temporal step already got it from `evaluate`.
- KPI fields: `layer_passes`, `layer_overhead_s`. At 10 µs per layer the frame grows by 0.96 ms (64 passes plus the temporal step's 32 layers): small next to the 6.8 ms of bandwidth excess.

### Changed
- Count tests: 148 release files, llm 122, 154 listed + unlisted.

## [0.274.0]

### Added: Kyutai Moshi (MODEL §219.1–219.2)
- `kyutai/moshiko-pytorch-bf16` as `moshiko-7b` (provider Kyutai, family Moshi, listing domain `omni`, coverage `full`).
- The HF repo ships no config. `scripts/fetch_pt_release.py --moshi` transcribes the `lm` / `mimi` blocks from moshi@e6a55d2 `loaders.py` (cited in `_source`) and reads the model.safetensors and Mimi tokenizer headers over HTTP Range. Tests check every transcribed number against the released tensor shapes.
- Params exact:
  - temporal transformer 6,973,366,272 (`params_llm`, bf16);
  - depth transformer 714,362,880 (`params_depth`, bf16);
  - Mimi 96,151,393 (`params_codec`, fp32).
- `core/moshi.py` costs each 80 ms frame:
  - Mimi encode: SEANet convs at 1920 / 480 / 96 / 16 / 2 positions, an 8-layer d512 transformer, a stride-2 conv and 8 codebook searches.
  - Depformer: 8 steps × 6 layers. `weights_per_step` gives each step its own weight slice, so every frame reads all 667 M GEMM weights once.
  - Mimi decode.
- `speech._moshi_kpis` (kind `moshi`):
  - frame time = Mimi encode + temporal step (KV ring buffer ≤ 3000 frames) + depformer + Mimi decode, serial on one card;
  - RTF = frame / 80 ms; response latency = 160 ms + frame;
  - also reports real-time streams and re-read bytes per frame.
- 100T + LPDDR5X at a full 3000-frame context:
  - frame 86.8 ms (temporal 78.1, depformer 7.0, Mimi 0.9 + 0.8), RTF 1.085;
  - 16.7 GB re-read per frame, which needs 208 GB/s effective against 191 available;
  - real time only while the KV ring buffer holds ≤ 518 frames (the first ~41 s of a conversation). At 100 frames, 5 streams fit real time, because weights are shared across the batch.
- UI: Moshi badge plus frame and bytes KPIs. `ui_headless_smoke` also loads moshiko-7b.

### Added: cross-model speech insight (MODEL §219.3)
- `core/speech_insight.py` and `POST /api/insight_speech` (`models`, `audio_s`, `verify`). For each catalog speech model at batch 1, every audio-clock stage is reported as busy seconds per audio second x. The output gives:
  - the binding stage;
  - the dominant first-audio component;
  - for x > 1: the effective DRAM bandwidth needed (BW · x), or the SLC needed to pin re-read weights: S = (x − 1) · BW / r, where r = tokens or frames per audio second.
- `verify` re-runs the KPIs with that SLC through the engine's own pin path.
- On 100T + LPDDR5X (191 GB/s effective), four models miss real time; all are bound by weight re-reads, and every predicted SLC verifies:

  | Model | Bound by | x | Fix: SLC | or bandwidth |
  |---|---|---|---|---|
  | Moshi | frame | 1.085 | 1.30 GB | 207 GB/s |
  | Kimi-Audio | LM | 1.19 | 2.92 GB | 228 GB/s |
  | Step-Audio-2-mini | LM | 1.85 | 6.5 GB | 354 GB/s |
  | Qwen2.5-Omni | speech card (Talker) | 1.03 | 0.11 GB | — |

- CosyVoice2/3, Ming TTS and all ASR models are real time.

### Changed
- `api_eval` attaches `speech` for Kimi and Moshi models too.
- Count tests: 147 release files, llm 121, 153 listed + unlisted.

### Deferred
- FireRedTTS3. Its configs describe only the DiT / patch encoder / RedAE. The backbone LLM, DiT sampling steps and CFG live in code (MODEL §219.4).
- drama-3, Index-Echo.

## [0.273.0]

### Added: UI surfaces the speech API blocks (MODEL §218.1)
- Omni: first-audio split (Thinker / Talker / DiT / vocoder, last stream), the 0.272 speech-card SLC block (`speech.slc`: hit shares, Talker / DiT seconds per audio second, DiT SRAM miss) and the audio-input block.
- CosyVoice / Step-Audio-2: first-audio split, per-chunk flow / HiFT times, estimator pin miss; titles use `speech.label`; cached streaming is described as such.
- ASR: the truncation note uses the cap in seconds (`asr.max_audio_s`, new in `evaluate._asr_info`); the e2e KPI shows encoder time and prompt audio tokens.
- `ui_headless_smoke` also loads qwen2.5-omni-7b, cosyvoice2-0.5b, kimi-audio-7b-instruct and step-audio-2-mini.

### Added: Step-Audio-2-mini speech output (MODEL §218.2)
- `scripts/fetch_pt_release.py --token2wav` reads `token2wav/flow.yaml` and the flow.pt / hift.pt state dicts over HTTP Range. Params exact: flow 155,802,352 and HiFT 20,821,295 (both fp32).
- `CosySpec.stream = "cache"`. Chunks are a fixed 25 tokens + 3 look-ahead through `flow.inference_chunk` (`cosyvoice.flow_chunk_ops`):
  - The conformer runs only the new tokens, over an att cache that is never trimmed.
  - The DiT estimator (16 × d512) runs only the new 50 frames, over prompt mel + ≤ 100 cached frames.
  - CFG 2, 10 Euler steps; HiFT on new mel + 8 cache frames.
- Listing domain becomes `omni` (speech in + out).
- 100T + LPDDR5X, 10 s out:
  - First audio 2.18 s, dominated by the LM (28 tokens × 74 ms).
  - LM RTF 1.85: the 7B LM cannot emit 25 tokens / s on this memory.
  - Speech card RTF 0.022, 44 real-time streams.

### Added: Kimi-Audio-7B-Instruct (MODEL §218.3)
- `moonshotai/Kimi-Audio-7B-Instruct` (listing domain `omni`, family Kimi-Audio, coverage `partial`).
- LLM: 28 Qwen2 layers + 6 mimo layers (the audio branch off layer 21), costed as one 34-layer stack, plus two 168448-row heads (`io_post` mimo_output, one row per sequence; `ir._head_ops` runs io_post heads of KV-cache models in prefill / decode). 9,735,124,992 params, exact.
- Audio in: the repo's whisper-large-v3 encoder (sequential 30 s padded windows) + 4-frame stacking + VQAdaptor at 12.5 positions / s. 668,178,432 params, exact. New `AudioSpec.stack`.
- Not costed (coverage reason): the GLM-4-Voice input tokenizer, which lives in another repo.
- Speech out (`core/kimi_audio.py`, `speech._kimi_kpis`, kind `kimi`):
  - Audio tokens at 12.5 Hz, after 6 blank delay steps.
  - Detokenizer DiT: 1,584,023,120 params, bf16 at run time, adaLN per frame. Each chunk is 30 tokens + 12 look-ahead (168 frames), runs 29 Euler evaluations plus 1 K/V commit, with a K/V cache of ≤ 900 frames.
  - BigVGAN: 241,118,777 params, run with exact held-back / overlap frame bookkeeping.
  - `scripts/fetch_pt_release.py --kimi` fetches these parts.
- 100T + LPDDR5X, 10 s out:
  - Streaming first audio 3.94 s; the released kimia.py, which generates everything first, gives 13.0 s.
  - The DiT is weight-bound: 0.48 s per 2.4 s chunk (30 reads × 3.1 GB). Speech-card RTF 0.24, 4 real-time streams.
  - LM RTF 1.19 (TPOT 95 ms).

### Notes
- Freshness sweep 2026-10-19/20: nothing new (only a GGUF repack).
- Deferred: Moshi, FireRedTTS3.


## [0.272.0]

### Added: Fun-ASR-Nano-2512 (MODEL §217.1)
- `FunAudioLLM/Fun-ASR-Nano-2512-hf` (transformers `fun_asr_nano`, bf16, listing domain `asr`, family Fun-ASR-Nano).
- New audio kind `sanm` (`vision._sanm_spec / _sanm_ops`):
  - Front end: Kaldi fbank, then LFR stacking of 7 frames at stride 6, giving 16.7 positions / s. Audio tokens = LFR frames.
  - Encoder: 70 SAN-M blocks (d512, FSMN depthwise k11, ReLU MLP), bidirectional over the whole utterance. All 70 layers run, including the 20 timestamp-prediction layers.
  - Projector 512 → 2048 → 1024, then 2 adaptor blocks.
  - Positions are capped at 2048 (≈ 122.9 s); longer audio is reported as `truncated_s`.
- Params exact vs the header: Qwen3-0.6B 596,049,920; tower 233,741,920. 30 s ≈ 0.27 TFLOP in the encoder.

### Added: Step-Audio-2-mini, audio in (MODEL §217.2)
- `stepfun-ai/Step-Audio-2-mini` (listing domain `asr`, family Step-Audio-2).
- Qwen2Model LLM with q/k/v bias and an untied 158720-token head: 7,663,326,720 params.
- Whisper-style encoder (learned positions 1500 × 1280; one pass per clip, ≤ 29.99 s) + AvgPool(2) + Adaptor (conv k3 s2 + GELU, then Linear 1280 → 2048 → 3584): 651,852,544 params. Both exact.
- Tokens follow the forward formula: 12.5 / s.
- `summarize_release._step_audio_role`.
- Speech output (token2wav: CosyVoice2 encoder + DiT estimator + HiFT, `.pt`) is not modelled; deferred.

### Added: Omni speech-card SLC pin (MODEL §217.3)
- `omni.omni_slc_plan`, `talker_weight_bytes` and `scale_weights`; `speech._omni_kpis` honours `chip.slc_mib` with `slc_policy` "pin".
  - DiT weights that miss SRAM are pinned first (50·b reads / s per byte), then the Talker (50 / s, 2.64 GB bf16).
  - Hits are served at `slc_GBps` on a parallel port.
  - New result block `speech.slc` (hit shares, Talker / DiT seconds per audio second).
  - No SLC → unchanged.
- Finding (100T + LPDDR5X, slc_GBps 2000 「假设」):
  - Batching alone never reaches real time, because token2wav runs at batch 1 and DiT time grows with b.
  - 1 GiB of SLC (DiT) gives 3 real-time streams.
  - 4 GiB (DiT + Talker) cuts the Talker step 14 → 1.7 ms: RTF 1.03 → 0.17, 9 streams.

### Fixed
- MODEL §216.2: the Talker weight bytes per codec token are 2.64 GB (1.32 B streamed params, bf16), not "1.33 GB".

### Freshness
- Sweep since 10-20: nothing new (88 orgs). HF trending created on or after 10-18: nothing.
- Deferred, with reasons in MODEL §217.4: fishaudio drama-3 (FishOmni hybrid MoE + 10-codebook decoder; no code in the repo), Index-Echo S2ST, FireRedTTS3, Kimi-Audio-7B, Step-Audio-2 token2wav.

## [0.271.0]

### Added — Qwen2.5-Omni-7B, speech in + speech out (MODEL §216.1–216.2)
- `Qwen/Qwen2.5-Omni-7B` (listing domain `omni`, family Qwen2.5-Omni, coverage partial). The params match the safetensors header exactly, component by component:
  - Thinker = Qwen2.5-7B (`_tc`: `thinker_config.text_config`, q/k/v bias): 7,615,616,512.
  - Audio tower: 639,647,232.
  - Talker: 1,351,360,256.
  - token2wav DiT: 333,607,792 (includes ECAPA 1,729,280).
  - BigVGAN: 115,443,504.
  - The Qwen2.5-VL ViT (676,550,144) is stored only. Image and video input are not modelled; this is the coverage reason.
- Audio tower (`vision._omni_audio_spec`): Whisper-v3 run on the valid frames only, in 200-frame chunks with attention inside each chunk, then AvgPool(2) and Linear 1280→3584. That gives 25 tokens/s, using the processor formula. There is no 30 s padding, but audio past 30 s is truncated (`AudioSpec.maxf`; `asr.truncated_s`).
- `scripts/summarize_release.py`: Omni roles `talker` / `token2wav` (`params_talker`, `params_token2wav`, kept outside `params_llm`).
- Speech output (`core/omni.py`, `ModelSpec.omni`):
  - Talker: 24 layers × h896, 12×128 heads, GQA kv 4, SwiGLU 18944, codec head 8448. It emits 50 codec tokens per audio second.
  - DiT (fp32): 22 × d1024, AdaLN-Zero, block-masked SDPA. RK4 over 10 time points gives 36 evaluations, each with CFG ×2. The ECAPA speaker encoder re-runs on the reference mel in every evaluation.
  - BigVGAN (fp32): 6 upsample stages ×240 with AMP blocks and anti-aliased SnakeBeta, 24 kHz.
  - Streaming follows vLLM-Omni: 72-mel chunks (0.72 s); DiT window 96 frames for the first chunk and 144 steady; vocoder on 72 / 112 frames; token2wav batch 1.
  - The DiT re-read share uses the 0.269 pin rule (`omni_pin`).

### Added — CosyVoice2-0.5B / Fun-CosyVoice3-0.5B-2512 (MODEL §216.3)
- `scripts/fetch_pt_release.py`: these releases ship torch `.pt` checkpoints only. Tensor names, shapes and dtypes are read from each checkpoint's pickled state dict over HTTP Range (zip central directory, then data.pkl), with no weights downloaded and no torch needed. The yaml fields go into `config.cosyvoice`.
- LM: the Qwen2.5-0.5B backbone, with vocab = the speech-token table / `llm_decoder` (6564 / 6761). The text embedding and the stored text lm_head are fp32 lookup tables. `params_llm` matches exactly, except the 6564-param `llm_decoder` bias of v2.
- `core/cosyvoice.py` (`ModelSpec.cosy`) builds one GEMM per released weight at its module's rate:
  - CosyVoice2 flow: Conformer encoder (25 Hz / 50 Hz, rel-pos attention) + U-Net estimator.
  - CosyVoice3 flow: DiT, the same block as the Omni DiT.
  - CFM: 10 Euler steps × CFG 2.
  - HiFT at ×8 / ×40 / ×120 the mel rate, then iSTFT.
  - Flow 112,549,360 / 332,257,120 and HiFT 20,821,295 / 20,779,887 are read exactly once per pass.
- Streaming follows cli/model.py:
  - Hops are 25 → 50 → 100 tokens (+3 look-ahead).
  - Every chunk re-runs the flow over the prompt plus all tokens generated so far.
  - CosyVoice3 re-vocodes all the mel generated so far; CosyVoice2 vocodes only the new mel (+ an 8-frame cache).
  - `token_hop_len` is never reset, so a long-lived instance starts later requests at a 100-token hop. This is reported as `first_audio_warm_s`.

### Added — speech-out KPIs
- `speech_kpis` kinds `omni` and `cosyvoice`: first-audio latency with its split, Talker step / DiT chunk / flow and HiFT per chunk, speech-card RTF for b streams, real-time streams per card, and output audio s / s per card (counting the speech card). Placement assumption 「假设」: one extra speech card per replica. `/api/eval` returns `speech` for these models; UI KPIs and badges.
- Finding (100T + LPDDR5X, 10 s out):
  - Omni is not real time even for one stream (speech-card RTF 1.03). Talker 50 × 14 ms = 0.70 s plus DiT 0.33 s per audio second. First audio 1.08 s, of which 0.67 s is the 48 Talker steps.
  - CosyVoice2 first audio is 0.22 s (0.78 s warm), RTF 0.012. CosyVoice3: 0.28 s (0.83 s warm), with the fp32 DiT weight-bound at 66 ms per chunk.

### Freshness
- Sweep of 88 orgs since 10-17: only DeepSeek-V4.1-Flash-1017 (an alias since 0.248) and repacks / GGUFs. Nothing new since 10-20.
- Pre-window speech gaps are deferred, with notes in MODEL §216.4: Fun-ASR-Nano-2512-hf (transformers config: SAN-M 70-layer encoder + adaptor + Qwen3-0.6B), fishaudio drama-3 (FishOmni, 40 GB + codec.pth), Index-Echo S2ST, FireRedTTS3.

## [0.270.0]

### Added — speech round: encoder-decoder ASR (MODEL §215.1–215.2)
- `openai/whisper-large-v3` / `-turbo` (fp16): Whisper as a real encoder-decoder. Decoder (`model._build_whisper`): MHA self-attention + cross-attention (`Layer.cross`) + GELU MLP, learned positions (lookup), tied head; `ModelSpec.enc_ctx` = 1500 encoder positions per 30 s window. New cross-attention decode path (`ir._cross_decode_ops`): q / o every step over all 1500 keys; cross K/V projected once at the first prefill pass from the encoder output and written to the cache, read by decode and chunked-prefill continuations; TP head split + all-reduce. Cross K/V is per-sequence fixed state in memplan (245.76 MB / request for v3, 30.72 MB for turbo) and in PD hand-off bytes.
- Whisper-v3 encoder (`vision._whisper_v3_spec`, `AudioSpec.pad / trunc / cross / pool / lin`): every pass padded to 30 s (3000 frames → 1500 positions); long-form = sequential windows; no audio positions in the decoder prompt (`AudioSpec.prompt_tokens`), encoder costed whenever `audio_s` > 0.
- `Qwen/Qwen2-Audio-7B-Instruct`: Whisper-v3 encoder + AvgPool1d(2) + Linear → Qwen-7B (text_config completed from transformers Qwen2Config defaults); prompt tokens per the processor formula (750 per 30 s), audio past 30 s truncated (reported).
- `Qwen/Qwen3-ASR-1.7B`: AuT encoder (`AudioSpec.kind = "aut"`: 100-frame chunks, Conv2d ×3 stride 2 → 13 positions / s, window attention of 104 positions, MLP projector) + Qwen3-1.7B (`thinker_config.text_config`).
- Params exact vs the safetensors headers for all four (decoder / LLM and encoder separately); `scripts/summarize_release.py` gained Whisper tensor roles. Listing domain `asr`, families Whisper / Qwen2-Audio / Qwen3-ASR.

### Added — audio-seconds serving convention (MODEL §215.3)
- `Result.asr` / summary `asr` (per phase): audio s, encoder positions, prompt audio tokens, windows, cross K/V MB per sequence, first-token latency + encoder time (prefill), decode RTF (decode), audio s / s (per replica and per card).
- `core/speech.py` `speech_kpis(scn, audio_out_s)`: end to end over both phases — ASR: first token, total = T_ft + out_len · TPOT, RTF, audio s / s / card; TTS (Ming-3.0-tts-tiny): first audio = T_ft + one step, RTF, output audio s / s / card. `/api/eval` returns it as `speech` (new body key `audio_out_s`, default 10 s); UI KPIs.
- Finding (100T + LPDDR5X, 30 s, 100 tokens): Whisper-v3 decode at B32 is cross-K/V bound (7.9 GB / step vs 1.8 GB weights; TPOT 9.7 → 56 ms); turbo cuts both 8× and is then encoder bound (779 of 798 ms first token at B32).

### Freshness
- Sweep since 10-20: nothing new (74 orgs). `freshness_sweep.EXTRA_ORGS` += 14 speech orgs (FunAudioLLM, funasr, kyutai, bosonai, SparkAudio, IndexTeam, fishaudio, SWivid, FireRedTeam, hexgrad, ResembleAI, nari-labs, canopylabs, sesame): nothing since 10-20 there either.

## [0.269.0]

### Added — speech synthesis: Ming-3.0-tts-tiny (MODEL §214.1)
- `inclusionAI/Ming-3.0-tts-tiny` (`lingv3_tts`, bf16): the Ming-3.0-asr decoder (BailingMoeV3 KDA/MLA MoE, `llm_config`) plus a TTS decode head that replaces the vocab head (`core/tts.py`, `ModelSpec.tts`, `ir._head_ops`). One decode step = one latent = 80 ms of 44.1 kHz audio: stop head, CFM flow DiT (16 × d1024) × 10 Euler steps × CFG 2 on 3 tokens, aggregator (8 × d1024) → next input embedding, streaming AudioVAE decoder (Qwen2 24 × d896, GQA 14/2, sliding window 64) + ISTFT head (n_fft 3528). Params exact vs the header component by component; total 8,981,004,468 = HF. AudioVAE encoder / spk_head stored only; lm_head stored, never read. AudioVAE decoder K/V ring as per-sequence state.
- Result / summary `tts`: `rtf`, `audio_s_per_s`, `streams_realtime`, per-step GFLOP split (flow 24.3 / VAE 2.9 vs LLM ≈ 2.8), head weight GB read per step (5.0) vs stored (1.34). Listing domain `tts`（语音合成）; UI badge + KPI.

### Added — re-read hot weight class (MODEL §214.2)
- `StageStorage.rr_w / rr_reads`: hot weights read R times per step (the flow DiT, R = flow_steps). By the 0.192 rule (DRAM bytes saved per step per byte) they are pinned before state (2 B/B) and ordinary weights in SRAM and in the `pin` SLC, and miss at their own share (`memplan.hot_miss`, `slc_rr`, `pinned_rr`; overlap modes too). No re-read class → identical to ≤ 0.268.
- Finding (100T + LPDDR5X, B1): the TTS head triples the step vs the same decoder alone (32.5 vs 11.7 ms); a 0.5 GiB SLC holding the 409 MB flow DiT cuts it to 13.5 ms (RTF 0.41 → 0.17).

### Fixed
- `/api/health` `domains` is now `catalog.DOMAINS` (the hard-coded list missed `asr` / `tts`); test guards health domains == DOMAINS ⊇ every listed model's domain.

### Freshness
- Sweep since 10-20: nothing new, no errors.

## [0.268.0]

### Deprecated — √C SLC access scaling (MODEL §213.1)
- `energy.slc_access_ref_mib` (0.266 √C) is deprecated: it extrapolates Horowitz's 8 KB–1 MB points to GiB SLCs and flips the J-optimal SLC to 0 GiB (§212.2). Behaviour unchanged for back-compat; `slc_access` now carries `deprecated: true` + `note` (`energy.SLC_SQRTC_DEPRECATION`), `insight.slc_energy_report` adds the note, UI field tagged 「已弃用」 and the SLC KPI line says so. Use `pJ_bit_mm_onchip` (distance model, §212).

### Added — Ming-3.0-asr-tiny (MODEL §213.2)
- `inclusionAI/Ming-3.0-asr-tiny` (2026-10-14, bf16): `ming_v3_asr` decoder = BailingMoeV3 (24 layers, groups of 4: 3 KDA + 1 MLA, 128 experts top-8 + 1 shared, first layer dense, MTP 1) via the Ling-3.x path — 7,893,392,800 + MTP 316,604,800 params, exact vs the safetensors header.
- Whisper audio encoder (`AudioSpec.kind = "whisper"`, MingAudioSpec / modeling_whisper_encoder.py 「来源」): 128-mel 16 kHz, 32 × d1280 / 20 heads, 3000-frame chunks each encoded separately, projector Conv1d k3 s2 + Linear → 25 tokens/s; 647,149,568 run params + 8,653,824 stored-only `linear_proj` (ignored on load) = released 655,803,392. Costed via `serving.audio_s` (≈ 2.29 TFLOP per 30 s chunk).
- Listing domain `asr`（语音识别）for audio-only releases; audio-only encoders (VisionSpec family `audio`) ignore `serving.images` with a warning.

### Freshness
- Sweep since 10-20: nothing new. inclusionAI / allenai / hpcai-tech rechecked by API: Ming-3.0-asr-tiny added (above); Ling-3.1-flash was already in the catalog (0.145); allenai HiRO-ACEv2 (10-12) is a climate emulator, out of scope; hpcai-tech nothing since 2025-03.

## [0.267.0]

### Added — distance-based SLC wire energy (MODEL §212.1)
- `energy.pJ_bit_mm_onchip` (+ optional `slc_mm2_per_mib`): SLC pJ/bit = `pJ_bit_slc + pJ_bit_mm_onchip × ½√(slc_mib · mm²/MiB)` (mean Manhattan distance from a centre port over a square array). Keckler et al. IEEE Micro 2011: 256 bit × 10 mm = 310 pJ at 40 nm (0.121 pJ/bit/mm), 200 pJ projected at 10 nm (0.078) 「来源/推导」; default density = Aoyagi 3 nm HD macro 35.5 Mbit/mm² (array only, distance lower bound). Mutually exclusive with 0.266 `slc_access_ref_mib`. Report `slc_access.wire_mm / array_mm2 / over_reticle`; insight curve `slc_pJ_bit / slc_wire_mm`; UI fields; PD ledger via the same helper. Unset = flat (fingerprints unchanged).
- qwen3-30b-a3b 100T + LPDDR5X B8 dip, 25 °C: 32 GiB stays J-optimal but the saving shrinks −35 % → −11.5 % (0.078 pJ/bit/mm, 44 mm); the 0.266 √C anchor at 1 MiB flips it to 0 GiB (45 nm small-cache extrapolation — too punitive).

### Freshness
- Sweep since 10-20: nothing new (inclusionAI / allenai / hpcai-tech errored in the sweep, rechecked by API: newest pre-window).

## [0.266.0]

### Added — SLC access energy vs capacity (√C wire scale) (MODEL §211.1)
- `energy.slc_access_ref_mib` (opt-in): effective `pJ_bit_slc × max(1, √(slc_mib / ref))`. Horowitz ISSCC 2014 8 KB / 32 KB / 1 MB ⇒ E ∝ √C 「推导」; large-cache H-tree dominance (CACTI / ISQED 2023). Unset = flat (= SRAM array 「假设」, fingerprints unchanged). UI「SLC 线缆锚点 MiB」; report `slc_access` {scale, pJ_bit, …}. Same in the PD ledger.

### Freshness
- Sweep since 10-20: nothing new.

## [0.265.0]

### Added — vector / SRAM-port datapath clock gating (MODEL §210.1)
- `Chip.clk_datapath_gate` (default False): gated share of P_clk uses `mean(u_mac, u_vec, u_sram_port)` instead of MAC util alone — closes §176 limit. Equal domain weight 「假设」. UI checkbox; energy row notes the basis.
- `energy.clock_util` helper; `clock_gating` now reports `vec_util` / `sram_port_util` / `util` / `basis`.

### Fixed — PD ledger SLC leak under idle (MODEL §210.2)
- `pdqueue._energy` now matches `energy_report` 0.263: SLC MiB always charged when the leak field is set; shared SRAM still idle-gated.

### Freshness
- Sweep since 10-20: nothing new.

## [0.264.0]

### Changed — 25/85 °C SLC energy in the default design-insight path (MODEL §209.1)
- `insight_report` / `POST /api/insight` attaches `slc_temp` whenever `uW_per_mib_sram_leak` is set: compact 25/85 °C J-opt / crossover / time-opt per model (grid `INSIGHT_SLC_GIB`), plus `recommend[].hot_cuts_J_opt`.
- Insight UI `#ins-slc-temp`: table + one-click「应用 25/85 °C J 最优」writes `#c-slc_mib`. No second click on「25/85 °C 对比」required for the headline trade-off.

### Freshness
- Sweep since 10-20: nothing new.

## [0.263.0]

### Changed — idle-aware incremental SLC leakage; 25/85 °C one-click compare (MODEL §208.1–208.2)
- With a measured idle, shared SRAM leak stays suppressed; **SLC MiB is always charged** (idle is assumed without this speculative SLC 「假设」). PD ledger matches. SLC energy insight now shows the hot-die crossover even when the energy panel has idle filled.
- `slc_energy_report(temps=(25,85))` / API `temps` / Insight「25/85 °C 对比」: side-by-side opt J / time / crossover.

### Changed — load-aware `pin` under a measured expert load (MODEL §208.3)
- pin / pin_bw keep the most-read experts (A0, same helper as dip). Uniform unchanged. qwen3-30b-a3b 16 GiB pin, Zipf-1: 108.7 → 80.1 ms.

### Freshness
- Sweep since 10-20: nothing new.

## [0.262.0]

### Added — J/token-optimal SLC size under die temperature (MODEL §207.1)
- `insight.slc_energy_report` / `POST /api/insight_slc` / Insight tab「SLC 能耗最优」: sweeps SLC GiB under `energy.sram_temp_C`, reports the J-optimal and time-optimal capacities and the hot-die crossover (J > no-SLC).
- qwen3-30b-a3b, 100T + LPDDR5X, B8, `dip`, no idle: at 25 °C opt J = 64 GiB; at 85 °C opt J = 0 GiB with crossover at 4 GiB, while time-opt stays 32 GiB.

### Notes — granite-switch deferred; freshness (MODEL §207.2–207.3)
- ibm-granite granite-switch-4.1: public config is not enough for an honest param_check (control-token LoRA adapters, serve path undocumented).
- Sweep since 10-19: nothing new.

## [0.261.0]

### Added — SRAM / SLC leakage vs die temperature (MODEL §206.1)
- `energy.sram_temp_C` (UI 能耗组「SRAM 结温 °C」): leakage × 4.742^((T − 25)/60) from the same 3 nm TSMC macro measured at 25 °C (15.5 µW) and 85 °C (73.5 µW) 「来源」, exponential in T 「推导」 (「近似」 outside 25–85 °C). Unset = 25 °C, so defaults are unchanged. The report adds `sram_leak_temp_C` and the 85 °C band `sram_leak_W_card_85C`; the PD ledger uses the same factor.
- qwen3-30b-a3b, 100T + LPDDR5X, B8, `dip`, 32 GiB SLC: J/token 0.1673 at 25 °C → 0.3875 at 85 °C (no SLC: 0.2572). On a hot die a GiB-scale SRAM SLC costs more energy than it saves.

### Changed — MiniMax-H3 token refiner: no RoPE now sourced (MODEL §206.2)
- diffusers `MiniMaxH3TokenRefinerBlock` ("No AdaLN and no rotary embedding", no mask). Numbers unchanged; label 「假设」 → 「来源」.

### Added — static-site guards (MODEL §206.3)
- `tests/test_static_routes.py`: every `/api` call in app.js is routed by `serve.respond` with its method; the shim's slow-lane routes exist; the package zip holds every tracked package file; the API answers from the unpacked zip alone.
- `scripts/build_static.py` packs every package file outside `web/` (was `*.py` / `*.json` only).

### Changed — `dip` SLC expert hits follow a measured expert load (MODEL §206.4)
- With `serving.moe_expert_load`, the retained experts are the most-read ones (A0 order, 「近似」): the read share served exceeds the byte share. New `MemPlan.slc_frac_exp_bytes` / API `mem.slc_frac_exp_bytes`. Uniform load unchanged.
- Example (Zipf-1 load 「假设」): qwen3-30b-a3b 32 GiB TPOT 62.2 → 34.4 ms.

### Freshness
- Sweep since 10-19: nothing new (Hy-Memory GGUF repack only).

## [0.260.0]

### Added — token-refiner attention for HunyuanVideo and MiniMax-H3 (MODEL §205.1)
- `ModelSpec.io_attn` takes an optional per-token vector field (`qknorm_rope` default, `qknorm`, `""`).
- HunyuanVideo: 24 × 128 heads, no QK-norm / RoPE 「来源」 (diffusers `HunyuanVideoIndividualTokenRefinerBlock`), full-length mask 「假设」. MiniMax-H3: 56 × 128, QK-norm 「推导」, RoPE not counted 「假设」.
- Negligible effect: hunyuanvideo 6908.7231 → 6908.7246 s, minimax-h3 1937.7807 → 1937.7929 s.

### Changed — SLC `dip`: every-step classes retained before routed experts (MODEL §205.2)
- `memplan.dip_split(K, C, E, e)`: hot weights, KV / state and activations (read every step) are kept first, routed experts get the rest (A0 frequency order, Aho–Denning–Ullman 1971, 「近似」); BIP churn ε·max(0, C + e − K) uses the per-step expert reads e. Dense models unchanged.
- New `MemPlan.slc_frac_exp` / API `mem.slc_frac_exp`.
- qwen3-30b-a3b, 100T + LPDDR5X, B8, TPOT ms at 4 / 16 / 32 GiB: 146.8 / 114.9 / 72.4 → 134.2 / 99.5 / 62.2 (pin 136.1 / 108.7 / 72.2).

### Added — UI control for `workload.cfg_branches` (MODEL §205.3)
- `CFG 负分支` select in the gen workload group, shown only for models with a declared negative branch (model info `workload.uncond`; MOSS-Image).

### Changed — SLC leakage charged; core-rail leakage ∝ V under DVFS (MODEL §205.4)
- The SLC's SRAM leakage is charged with the SRAM term (when no measured idle figure is given), in the main report and the PD ledger. qwen3-30b-a3b 32 GiB `dip`: J/token 0.1086 → 0.1673 (the SLC saving vs none shrinks from −58 % to −35 %, 7.57 W at the 25 °C lower bound).
- Power cap: SRAM / SLC / state-buffer leakage scales with V (AccelWattch, MICRO 2021, eq. 2 「来源」; linear = conservative 「近似」). The idle floor and HBM standby stay held 「假设」.

## [0.259.0]

### Added — online static version (GitHub Pages, accel_dse on Pyodide in the browser); non-blocking progress UI
- **Static build** (`scripts/build_static.py`, stdlib only) writes `dist-static/`:
  - the unchanged web UI;
  - `web-static/pyodide-shim.js`, a `fetch` wrapper for `/api/*`;
  - `web-static/pyodide-worker.js`, which runs Pyodide 0.29.5 (pinned, jsdelivr CDN, CPython 3.13) in Web Workers;
  - a deterministic `accel_dse-<hash>.zip` with the package `.py` files and `data/**/*.json`.
- **Same answers as the server.** Requests go through **`serve.respond(method, path, raw)`**, the server's own dispatch and JSON serialisation, now factored out of `Handler._api` and transport-free. Responses are byte-identical to `accel_dse serve`.
- **Two interpreters in the browser:**
  - *fast*: eval / sensitivity / memory / sweep / pareto / catalog;
  - *slow*: fit / compare / layouts / stability / insight*.
  - A capacity search never blocks evaluation. A stale running slow-lane search terminates its worker, and a fresh one starts.
- **Deploy:** `.github/workflows/pages.yml` builds and deploys to https://kalcohol.github.io/accel-dse/ on push to main (and on manual runs). One-time setting: Pages source = GitHub Actions.
- **Progress UI** (server and static modes). It never blocks the interface.
  - A thin animated bar at the top of the results area, plus a floating pill near the metric cards: 「计算中… 3.2 s · 阶段」.
  - Both appear only after 150 ms, use `pointer-events: none`, and never dim or disable inputs. Metric cards being recomputed get a subtle 「updating」 style.
  - On the first static load, the pill shows the Pyodide download / init / package phases.
  - Respects `prefers-reduced-motion`.
- **Edits during a calculation.** Any edit starts a new calculation and drops stale replies:
  - a scenario edit cancels the in-flight fit and sensitivity requests;
  - a fit reply that lands after an edit is discarded;
  - the search buttons (映射对比, 最佳布局, 设计洞察, 升级路径, 缩减余量) stay clickable, and clicking again restarts the search.
- **Fix: scrollbars covered text.** Overlay scrollbars (Chromium Fluent / macOS, Firefox GTK / macOS) take no layout space. On hover they appeared over the last line of a horizontally scrollable block, for example the notes just above 「本次评估的「假设」」.
  - Every `.scroll` container now keeps a 14 px bottom strip (`--sb-gap`) clear of text.
  - The parameter panel gets `scrollbar-gutter: stable` plus the same right padding.
  - Scrollbars are thin and muted.
  - Checked in Chromium (overlay on) and Firefox (GTK overlay on), on hover.
- `tests/test_static_build.py` checks `respond` and the bundle. `scripts/static_parity_check.py` (opt-in, Playwright) replays browser calls on the server and compares the bytes.

## [0.258.0]

### Changed — MOSS-Image CFG negative branch at its own shape; text-refiner attention core (MODEL §204.1–204.2)
- **Source 「来源」:** the release's `pipeline_moss_image.py` `__call__` defaults: `negative_prompt=''`, `uncond_skip_layers=(22,)` active at every step (`slg_start` 0 / `slg_stop` 1), and two transformer calls per step (cond, then uncond).
- **Negative branch** (`NativeWorkload.uncond = (10, (22,))`, new `workload.cfg_branches`, `ref` by default):
  - The empty negative prompt is chat-templated to 10 tokens 「推导」 (template + Qwen pre-tokenizer). It is encoded unpadded, in its own text-encoder call.
  - The uncond forward skips block 22 and runs on 4096 + 10 rows. Before, it ran 42 blocks over 4352 rows.
  - The two calls run in series, B sequences each, and each call streams the weights.
  - `cfg_branches = same` restores the pre-0.258 packed, symmetric forward.
  - Falls back with a warning for PP > 1, `serving.microbatches` ∉ {0, 2}, or `step_skip`.
- **Refiner attention core** (`ModelSpec.io_attn`, `ir._io_attn_ops`): the 2 text-refiner blocks' bidirectional SDPA over the text rows (24 × 128, QK-norm + RoPE) is now counted. Before, only their GEMMs were.
- **Numbers** (default chip, 1024²), `same` → `ref`:
  - latency 54.62 → 53.13 s (−2.7 %);
  - DiT 5130.5 → 4953.8 TFLOP per request (−3.4 %);
  - per step 1089.1 ms packed → 545.9 (cond) + 512.2 (uncond) ms;
  - text encoder 0.083 → 0.145 s (two per-prompt calls, each streaming the 9B tower);
  - J / image (tpuv4i_7nm) 1613 → 1558.
  - The refiner core adds < 0.01 s.
- **New result fields:** `Result.cfg_asym` (cond / uncond ms, TFLOP per sequence, shapes) and `domain_summary.cfg_asym`. The energy counts include both branches. CLI `--cfg-branches`.

### Added — SLC policy `dip`: thrash-resistant insertion (MODEL §204.3)
- **Source 「来源」:** Qureshi, Jaleel, Patt, Steely, Emer, ISCA 2007 (DIP / BIP), §4.1 Table 3.
  - On a cyclic sweep of T lines through a K-line cache, BIP keeps a hit rate of (K − 1 − ε·(T − K)) / T, with ε = 1/32 (the paper's setting). LRU gets 0.
  - DIP set dueling picks LRU when the working set fits.
- **`chip.slc_policy = "dip"`:**
  - If the off-SRAM working set fits, every read hits (as `lru`).
  - Otherwise each read class (weights, KV / state, activations) hits by the share (K − ε·(T − K)) / T 「推导」.
  - KV writes go to DRAM.
  - New field `MemPlan.slc_frac` (API `mem.slc_frac`). CLI `--slc-policy dip`; UI option 「dip 防抖替换」.
- **Why it matters for SLC sizing:**
  - `lru` (the cyclic-thrash bound) shows no gain until the whole working set fits.
  - A hardware SLC with thrash-resistant replacement gains in proportion to its size, close to software pinning.
  - Example: 100T + LPDDR5X 4×64, qwen3-8b decode B8, TPOT 104.2 ms without SLC. At SLC 1 / 4 / 8 / 12 / 16 / 32 GiB:

    | policy | TPOT (ms) at 1 / 4 / 8 / 12 / 16 / 32 GiB |
    |---|---|
    | pin | 98.5 / 81.7 / 59.2 / 36.7 / 14.3 / 11.7 |
    | lru | 104.2 / 104.2 / 104.2 / 104.2 / 104.2 / 11.7 |
    | dip | 101.6 / 84.2 / 61.1 / 37.9 / 14.7 / 11.7 (hit share 2 / 19 / 41 / 64 / 86 / 100 %) |

- Defaults are unchanged (`pin`).

### Freshness
- `scripts/freshness_sweep.py 2026-10-19`: nothing new across the 74 orgs. The MiniMaxAI retry found nothing either.
- HF trending (≥ 10-18): only a third-party repack of DeepSeek-V4.1-Flash-1017.

### Deferred
- A UI control for `workload.cfg_branches` (API / CLI only).
- Token-refiner attention cores of HunyuanVideo and MiniMax-H3 (same mechanism, not switched on).
- SLC `dip` with per-class reuse distances (MoE routed experts are not read every step; the whole working set is taken as one cyclic sweep 「近似」).

## [0.257.0]

### Added — freshness sweep script; MOSS-Image Base / SFT (MODEL §203.1–203.2)
- **`scripts/freshness_sweep.py`** (stdlib). It lists the models every tracked org created since a date: every Hugging Face org in `catalog.PROVIDERS` plus `EXTRA_ORGS`, 74 orgs in total. OpenMOSS-Team is now in the list.
  - Since 2026-10-17 the only finds are DeepSeek-V4.1-Flash-1017 (an alias since 0.248) and third-party repacks / GGUFs (unsloth and others).
  - mistralai Timestral (10-14) is a time-series model, out of domain.
- **MOSS-Image-Base / -SFT** (OpenMOSS-Team, diffusers, 2026-10-11), `moss-image-base` / `moss-image-sft`. SFT has the same architecture and is merged into Base.
  - **Architecture:** single-stream DiT `MOSSImageTransformer`, 42 blocks, d 3072, 24 × 128 heads, SwiGLU 8192 (fused gate_up), text refiner (2 blocks), shared modulation.
  - **Checks:** param_check exact (5,082,308,624, bf16 + the fp32 RoPE bands); coverage full.
  - **Workload:** 1024², 50 steps (shift 5 still gives 50 timesteps, 「推导」), CFG 3 = two forwards per step, text 256 tokens 「假设」.
    - The unconditional branch skips block 22 and uses an empty negative prompt. It is modelled at the conditional shape 「近似」 (+3–4 %).
    - `_situ` activation costed as SwiGLU 「近似」.
  - **Pipeline:**
    - Qwen3.5-9B text tower from the release headers, with 8 full-attention layers (3, 7, …, 31) and 24 Gated DeltaNet layers. GDN uses the LLM IR's chunked form (`pipeline._gdn_ops`), not softmax attention on every layer.
    - VAE = AutoencoderKLFlux2 with the FLUX.2 small decoder (new family `os_s384`, attention width 384), fp32 as released.
  - **Default chip:** 54.6 s per 1024² image (FLUX 3 Base 47.2 s), 5144 TFLOP per request.

### Changed — PD report energy is a per-pool ledger (MODEL §203.3, closes the two §177 approximations)
- `pdqueue._energy` takes each pool's own action counts (`pool_counts`; PD: decode = its iterations plus the KV receive / restore extras 「约定」, prefill = the rest). Three things follow:
  - **DRAM kind.** Each pool's DRAM bytes are priced with that pool's memory kind (pJ/bit + chip-side PHY). Before, everything used the decode pool's kind.
  - **Clock gating.** Each pool uses its own MAC utilisation, u_p = its MACs / its slot-cycles. Before, one utilisation was pooled over both pools.
  - **`J_by_pool`.** Every term is booked to the pool that spends it (decode / prefill: J / token, J by action, card-s, DRAM field, MAC utilisation). The pools sum exactly to `J_per_token`.
- **Identical chips, memory and gating:** unchanged to rounding. Every term is linear in the counts, so the pooled form was already exact there (tested).
- **Effect** on qwen3-8b, 1P tp2, 2P + 6D, B64, 4096 / 512, load 0.6, tpuv4i_7nm, J / token before → after:
  - prefill pool on LPDDR: .20480 → .20518;
  - decode 1P g .25 HBM / prefill 100T LPDDR: .56077 → .55749 (clock −4.4 %);
  - decode LPDDR / prefill HBM: .62410 → .62372;
  - decode g .25 / prefill g 1, both HBM: .16610 → .16284 (clock −12.7 %).
  - Pool utilisations: decode 0.021, prefill 0.433 (pooled 0.124).
  - Ledger, prefill on LPDDR: decode pool 0.163 J / token (DRAM 61 %), prefill pool 0.042 J / token (MAC 61 %).
- Colocated modes (one pool) and callers without `pool_counts` are unchanged.

### Deferred
- MOSS asymmetric CFG branches (the skipped block and the short negative prompt).
- MOSS refiner attention core.
- GitHub-only providers (`gh:` prefixes) are not in the Hugging Face sweep.

## [0.256.0]

### Changed — UI layout (knob panel and metric cards)
- **Knob panel grouped by concern** into collapsible sections: 模型 / 工作负载 / 投机解码 / 服务调度（主机开销 · PD 分离 · 排队）/ 卡数、并行与互连 / 芯片与存储器 / 资源 · 面积预算 / 能耗.
  - Each header shows a one-line summary (e.g. `decode · batch 自动 · ctx 4096`, `k 4 · 树 topk 2 · sampled`, `1 卡 · PP1 TP1 DP1`, `已填 3 项`). The open / closed state is kept in the browser; 「全部展开 / 全部折叠」 at the top.
  - Budget and energy (no built-in values, low-frequency) start collapsed.
  - Workload shape (phase, batch, ctx / prompt / output, SLOs, Dory loops / loop KV, VLM image / video inputs, auto-batch) now sits right below the model instead of after the interconnect knobs.
- **Conditional knobs.** The speculative-decoding section is LLM-only. Draft sampling, acceptance source, rollback and tree knobs appear once 投机 k ≥ 1; the tree acceptance model appears only for a tree (topk > 1 or tree tokens set).
- **No more 3-line captions.** The panel widens on wide screens (360 → 500 px). Knob grids auto-fit (≥ 140 px per control) instead of fixed 3 / 5 columns. Tags (假设 / what-if) stay on the caption line, and controls line up across a row.
- **Metric cards** use a responsive grid (≥ 230 px per card) instead of a fixed 5 columns. Long notes (machine balance, tree TPOT band) clamp to 3 lines; click the card to expand, or hover for the full text.
- All element ids and behaviour are unchanged. `scripts/ui_smoke.py` and `scripts/ui_headless_smoke.py` click 「全部展开」 first because budget and energy start collapsed.

### Added — UI controls for 0.255, nested chip fold
- **FLUX 3 模态流复用.** `workload.mode_cache` (0.255, SGLang StaticStreams) has a select in the workload → reference-image row, shown only for FLUX 3. Off = reference implementation, and off is not written to the scenario.
  - When it is active, a 「模态流复用」 metric card shows first-step → reused-step time, the reused rows, TFLOP per step and the cache size.
  - Muse Spark Open video inputs need no new control: 0.255 gave Muse a modelled video processor, so the existing VLM video inputs in 工作负载 now appear for it.
- **Nested 「高级芯片参数」 fold.** Inside 芯片与存储器, the main row keeps 频率 / SRAM MiB / MAC 效率. Everything else moves into a fold, collapsed by default: clock gating, state SRAM, SRAM ports, attention format, GEMV, accumulator, kernel launch, dispatch, dispatch µs and the SLC row.
  - Its header summarises dispatch / kernel launch / SLC / attention format.
  - 「全部展开」 opens nested folds too.
  - At 1440×900 the panel depth goes from 3.73 to 3.22 screens (llm-spec) and from 4.17 to 3.66 (flux3), with 53 / 57 visible controls instead of 66 / 69. There are still no 3-line captions or misaligned rows.

### Fixed
- The video DiT frame count (`w-frames`) was hidden for every video-generation model. It shared the `video-only` class with the VLM video inputs, so it was hidden whenever the model had no VLM video processor.
- The chip preset bar clipped its last preset (`H100-like-lp4x64`) at panel width. It now wraps.
## [0.255.0]

### Added — group-limited MoE routing load model (MODEL §202.1)
- **Source 「来源」.** transformers `DeepseekV3TopkRouter` / `BeamTopKRouter` and vLLM `grouped_topk` (`bailing_moe.py` `use_grouped_topk`). Each group is scored by the sum of its top-2 (bias-corrected) expert scores. The router keeps `topk_group` of the `n_group` groups, then takes top-k over the experts of those groups.
- **Which models.** `Ffn.groups = (n_group, topk_group)` is set from config when 1 < topk_group < n_group and E % n_group = 0. That covers DeepSeek-V3 / R1 / V3.1 / V3.2, Ling-3.1-flash, Ling-3.0-flash-VL and Beam-1.0, all at 8 / 4. Kimi and GLM use 1 / 1, so they are unrestricted and unchanged.
- **Mechanism 「推导」** (under the existing symmetric-routing 「假设」):
  - Each token picks g of the G contiguous groups uniformly, then k experts among their g·S experts. Expert marginals stay k/E, so mean hits and mean pairs are unchanged.
  - The per-rank spread is now exact for both EP layouts (ep ≥ G: a rank sits inside one group; ep < G: a rank holds G/ep groups). Distinct experts use within-group and cross-group pair terms; pairs are hypergeometric mixtures.
  - Monte-Carlo agreement for 256 / 8 / 4, T = 16, ep = 4 / 8 / 16: means within 0.5 % and sd within 1.5 %.
- **Default change (mechanism-justified).** The `serving.moe_rank_max = expected` per-rank maximum (default on) uses the grouped spread.
  - At 256 / 8 / 4, T = 16, EP 8, the expected max distinct experts per rank goes from 16.72 to 17.77 and the max pairs from 21.29 to 23.50. A Monte-Carlo run gives 17.99 / 24.4.
  - Ling-3.1-flash decode (batch 64, TP 8, EP 8, default chip): TPOT 313.59 → 319.19 ms.
  - DeepSeek-V3 and Beam decode are unchanged. Prefill moves by ≤ 0.03 %.
- **Node-deduplicated all-to-all.** `node_dedup_factor` accounts for groups: it is fewer destination nodes per token, as in the DeepSeek-V3 report §2.1.2 node-limited routing.
  - 256 / 8 / 4: 0.662 → 0.454 at 8 nodes, 0.452 → 0.366 at 4 nodes. Monte-Carlo gives 0.455 / 0.367.

### Added — FLUX 3 Base static mode-stream reuse (`workload.mode_cache`, MODEL §202.3)
- **Source 「来源」.** SGLang `multimodal_gen/runtime/models/dits/flux3_image.py` (`StaticStreams`, `_packed_blocks`) and `stages/model_specific_stages/flux3_image.py`.
  - Step 1 keeps the text and reference (image_cond) mode-block outputs and their single-stream (shift, scale, gate) modulations.
  - Steps 2…n run only the image mode block. `txt_in`, the reference embedder and the text / reference modulations are skipped.
- **When it applies.** SGLang enables it only when all of these hold: Base (no flow map), Ulysses SP 1, no CFG-parallel or sequential CFG, no compile / CUDA graph / layer-wise offload.
- **Knob.** `workload.mode_cache = on` (CLI `--mode-cache`). The default stays `off` = the BFL / diffusers reference, which recomputes every step. With `on`, the model ignores it with a warning on the distilled model, for SP > 1 and for PP > 1.
- **Cache.** 4 · (text + reference rows) · hidden per sequence.
- **Numbers** (Base, 1 reference, default chip): 97.26 → 90.17 TFLOP per reused step-sequence; latency 103.27 → 96.50 s.
- This closes the 0.252 deferred "reference mode-output cache".

### Changed — Muse Spark Open image / video resize now sourced (MODEL §202.2)
- **Source.** transformers main `models/muse_glimmer/image_processing_muse_glimmer.py` and `video_processing_muse_glimmer.py`. The auto map sends `muse_spark` to `MuseGlimmerImageProcessor` / `MuseGlimmerVideoProcessor`. The repo `processor_config.json` (amd Quark mirrors) gives image `max_image_tokens` 4096 and video `max_video_frame_tokens` 256, `num_frames` 600, fps 2.
- **Image resize.** `smart_resize`, verbatim: the integer 28-px grid closest in aspect among the floor / ceil of the ideal grid, under the token cap, with no rounding to the nearest multiple and no minimum.
  - LLM tokens = merged + `<|image_start|>` / `<|image_end|>`.
  - 1024² → 1008² px, 1298 tokens. Until 0.254 the assumed Qwen-style rounding gave 1036² px, 1369 tokens. At 1920×1080 both give 2691 merged tokens, plus 2 now.
- **New video path.** Frames are capped at 600 and rounded down to a multiple of 2. Each frame is resized to ≤ 256 merged tokens, and attention runs per temporal slice. Tokens = t·merged + t + 1.
- **Coverage.** The six Muse Spark entries go from `partial` to `full`.

### Changed — Step-5-Preview crops (MODEL §202.2)
- **Source.** No Step-5 processor exists in vLLM / SGLang / transformers.
  - transformers `step3p7` `Step3p7ImageProcessor._plan_patches` belongs to StepFun Step-3.7-Flash, which has an identical `vision_config` (PE 47 × 1536, 728 / 14), the same 169 / 81 token lengths and the same `image_token_id`.
  - Step-5's own `inference/validate_vision.py` loads its reference encoder through `configuration_step3p7`.
- **What is modelled.** That 728² overview + 504² sliding-window crop plan (max side 3024, 20 % snap-up), with per-crop `<patch_start>` / `<patch_end>`, row newlines, then `<im_start>` 169 `<im_end>`.
  - 1024×768 → 504 tokens (was 169, overview only).
- **Coverage.** Applying it to Step-5 is the remaining 「假设」, so Step-5 stays `partial` with that reason.

### Audit — "no public code" deferrals (directory-wide search of vLLM / SGLang / transformers / diffusers main, MODEL §202.4)
- Resolved: Muse resize / video, FLUX 3 reference reuse, Step-5 crops (as above).
- Still deferred: Dory TP > 1. vLLM PR #60149 (`models/dory.py`, open, updated 2026-10-19) still raises for TP ≠ 1, and Dory is in none of the three main trees.

### Freshness check
- HF orgs (75) plus trending, created on or after 2026-10-17: only DeepSeek-V4.1-Flash-1017 (already an alias since 0.248) and third-party repacks / fine-tunes. No additions.
- Noted for a later sweep: OpenMOSS-Team MOSS-Image (10-11) is outside this window, and its org is not in the list.

### Tests
- New `test_core_255`.
- Re-pins:
  - 144 (Step-5 1024×768 tokens 169 → 504);
  - 153 (Step-5 reason now names Step3p7ImageProcessor);
  - 174 (Muse vision no longer approximate);
  - coverage (Muse leaves VISION_PARTIAL).

## [0.254.0]

### Added — Reflection Beam-1.0 and DFlash2 drafts (MODEL §201)
| id | source | what | coverage |
|---|---|---|---|
| `beam-1.0-501b-a23b` / `-fp8` / `-nvfp4` | https://huggingface.co/reflection/Beam-1.0-501B-A23B (+ -FP8, -NVFP4) | Reflection Beam-1.0 501B-A23B MoE: 52 layers (3 dense + 49 MoE, 256 experts top-8, group-limited sigmoid router, no shared expert), GQA 48/8×128 with a per-head sigmoid output gate, 43 SWA-512 (RoPE) + 9 full NoPE layers; FP8 = compressed-tensors W8A8 (per-channel weights, dynamic per-token activations), NVFP4 = ModelOpt W4A4 on the experts only | full (param_check exact, 501,267,612,352) |
| `beam-1.0-501b-a23b-nvfp4-dflash2` | https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2 | Beam NVFP4 + its DFlash2 block draft (1.93B: 5 non-causal SWA-4096 layers, grouped dynamic convs, candidate selector) | full (draft exact) |
| `muse-spark-open-nvfp4-dflash2` | https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2 | Muse-Spark-Open NVFP4 + the nvidia DFlash2 draft (4.38B); closes the 0.253 deferral | partial (same vision as Muse NVFP4; draft exact) |

- **DFlash2 on the DSpark block path** (vLLM main `qwen3_dflash2.py` DFlashGroupedConv / CandidateSelector, `models/beam/nvidia/dflash2.py`, `spec_decode/dflash2/speculator.py` 「来源」):
  - The block is anchor + `spec_k` drafts. Any size up to the trained block (Beam 16, Muse 8) is allowed; larger values warn and are capped.
  - Block attention is non-causal. Context K/V come from one stacked GEMM into a sliding-window draft cache (KV rows are now written for windowed GQA drafts).
  - Each draft layer runs two `kernel_projection` GEMMs (h → 2·taps·groups) and the 2-tap convolutions on both sides of attention and of the MLP.
  - The tail runs the target lm_head on the draft rows, then top-16, the `hidden_projection` (h → 256), K × K × 256 edge scores over the gathered codebook rows, and the walk.
  - The codebooks are cold storage. The conv, selector and fc GEMMs are replicated on every TP rank.
- **Published acceptance.** `serving.spec_accept_src = published` uses the card τ:
  - Beam: 4.876 greedy / 4.67 sampled. The sampled run is T = 0.7, not T = 1, so it is labelled 「二手」.
  - Muse: 3.687 / 3.33.
  - Both are at 7 drafts and convert to an i.i.d. acceptance of 0.852 / 0.758 「推导」.

### Changed — as-released activation formats (mechanism: vLLM quantisation methods 「来源」)
- **compressed-tensors W8A8 float** (8-bit float `input_activations`: FP8_DYNAMIC per-token, FP8_BLOCK per-group-128) now runs fp8 activations; until 0.253 they were bf16.
  - Affects `lfm2.5-24b-a2b-fp8` (prefill, default scenario, TP 8: 33.19 → 24.38 ms; decode unchanged) and the new `beam-1.0-501b-a23b-fp8`.
- **ModelOpt NVFP4 (quant_algo `NVFP4` = W4A4)** now carries `act_expert = nvfp4` on its NVFP4 roles. Excluded modules (attention, dense layers, router, lm_head) keep bf16 inputs. Until 0.253 every role used bf16 inputs.
  - Affects `muse-spark-open-nvfp4` (+ -dspark), `step-5-preview-nvfp4` and the Beam NVFP4 entries.
  - Results change only on chips with a native NVFP4 MAC. The default 100T has none, so its numbers are unchanged.
  - `W4A16_NVFP4` (Dory) stays weight-only.

### Fixed
- MODEL §200.4 said vLLM had no code for the DFlash2 conv projector / selector. That was wrong: `qwen3_dflash2.py` has had both since PR #52816 (2026-08-21), and 0.253 had looked only at `qwen3_dflash.py`.

### Tests
- New test_core_254.
- Re-pins:
  - 062_1: releases 129 → 132, llm 105 → 108, catalog 133 → 138.
  - test_core_093: act_ formats are now allowed for W4A4 releases.
  - coverage: the Muse DFlash2 pair is vision-partial.

## [0.253.0]

### Added — freshness sweep from 2026-10-15 (MODEL §200)
| id | source | what | coverage |
|---|---|---|---|
| `dory1.0-4b` / `-mxfp8` / `-nvfp4` | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 (+ -MXFP8, -NVFP4) | NVIDIA Nemotron-Labs Dory: looped normalized-GPT 3.98B (8 input / 20 recurrent × loops / 8 output layers, SWA + full GQA with output gate); MXFP8 = W8A16, NVFP4 = W4A16 | full (param_check exact) |
| `mai-gym-9b` (merged → qwen3.5-9b) | https://huggingface.co/Tongyi-MAI/MAI-Gym-9B | GUI-agent Qwen3.5-9B fine-tune, no MTP head | full |

- **Looped LMs.**
  - `workload.loops` (0 = the release config, 2) and `workload.loop_kv` (`per_loop` default | `last_loop`). CLI: `--loops` / `--loop-kv`. UI: loop row, shown for Dory only.
  - The recurrent weights are stored once and re-read on every loop. `last_loop` keeps the one-loop KV size at any loop count.
  - The model warns for TP / PP > 1, because the reference is TP = 1 only.
- **DSpine sampled rounds.**
  - `serving.spec_sampling` = greedy (default) | sampled. A sampled round adds the eager fp32 `serial_rs_table` (K·(2RC + HR + KH) MACs per draft row) and the sample chain.
  - `serving.spec_accept_src` = manual (default, the 「假设」 lever) | published (the draft's card τ at that temperature → i.i.d. acceptance 「推导」).
  - CLI: `--spec-sampling` / `--spec-accept-src`. UI: two selects next to the speculative-decoding knobs.

### Changed
- DSpine resident draft tables now include the fp32 sampler copies (sglang_dspine `DSpineDraftSampler.__init__` 「来源」): 466.8 → 479.3 MB.
- `scripts/summarize_release.py`: `.mixer.{gate,up,down}_proj` / `suv_*` / `mlp_alpha` → the mlp role, and `backbone.embeddings` → embed. This affects the new Dory summaries only; existing releases were not re-summarised.

### Deferred
- nvidia Muse-Spark-Open DFlash2 draft: there is no public modeling code for its conv projector or path selector (design in MODEL §200.4).
- reflection Beam-1.0-501B-A23B: pre-window.

### Tests
New test_core_253. Re-pins: 062_1 (release / catalog counts), test_core_252 (DSpine table bytes), test_core_125 (looped models re-read the recurrent weights per loop). DeepSeek-V4.1-Flash-1017 was already aliased onto deepseek-v4.1-flash in 0.248, so it is not re-added.

## [0.252.0]

### Added — the 0.251 coverage gaps closed (MODEL §199)
- **FLUX 3 Image Klein (+ Base): reference-image edit / multi-reference.** Coverage partial → **full** (every header tensor modelled; param_check exact).
  - `workload.ref_images` (0 = T2I, ≤ 10 = BFL `MAX_REFERENCES`), `ref_width` / `ref_height` (0 = output size); CLI `--ref-images`, `--ref-size WxH`; UI reference row (FLUX 3 only).
  - Reference sizing follows the release rules (1 ref → ≤ 4 MiB-pixel canvas, ≥ 2 → each ≤ 1 MiB pixels, 16-px grid).
  - References are VAE-encoded once per request, one image per call (new `vae_encode` pipeline part, `enc_s`), then run through `mode_blocks.image_cond` and the joint single blocks **every step** — the BFL reference and diffusers have no reference KV cache 「来源」. Base CFG carries the references in both halves.
- **`qwen3-8b-dspine`** = Qwen/Qwen3-8B + tencent/Qwen3-8B-DSpine-b16 (DFlash block draft with a selected-prefix code bridge), on the DSpark block path: anchor row + 15 drafts (k fixed at 15), causal block attention, per-layer code-bridge GEMMs, top-K serial-table tail, resident codebook + tables (778 MB). Published τ 7.83 / 6.98 → a = 0.893 / 0.873 「推导」; `spec_accept` stays a lever. TP > 1 warns.

### Changed
- FLUX 3 mode blocks are per-mode self-attention (0.251: joint) and all blocks are parallel (one LN + modulation, one TP all-reduce 「推导」); modulation and time MLP per token (BFL reference). T2I latency +2.4 % (distilled 7.61 → 7.79 s; Base 46.45 → 47.23 s).
- `serving.spec_k` cap 8 → 16.

### Deferred
- FLUX 3 reference mode-output cache what-if (exact on Base only; design in MODEL §199.1).
- DSpine sampled-round `serial_rs_table` cost (greedy tail only).

### Tests
New test_core_252. Re-pins: 062_1 catalog total (+1), coverage list (FLUX 3 → full), test_core_251 FLUX 3.

## [0.251.0]

### Added — catalog freshness sweep (Hugging Face, through 2026-10-19; MODEL §198)
Every vendor org of the catalog was swept with `api/models?author=…&sort=lastModified` (44 orgs, including tencent, the unsloth mirrors, BlinkDL, Inferact and TypeSafe variants).
- **Window:** created on or after 2026-10-14, which is two days before the catalog's latest fetch on 2026-10-16, and not already in `releases/`.
- **Rules:** a model needs a public config.json; it runs as released, with its official dtypes; params are checked exactly against the safetensors headers.

| id | source | what | coverage |
|---|---|---|---|
| `hy-memory-30b-a3b` | https://huggingface.co/tencent/Hy-Memory-30B-A3B | Tencent Hy3 MoE (hy_v3, 30.06B / A3B, bf16), long-term-memory writer | full |
| `hy-memory-30b-a3b-fp8` | https://huggingface.co/tencent/Hy-Memory-30B-A3B-FP8 | official FP8: static W8A8 + fp8 KV | full |
| `step-5-preview-nvfp4` | https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4 | official NVFP4 of Step-5-Preview (config identical except quantization; 394.2 GB vs 641.4 GB) | partial (same vision as Step-5-Preview) |
| `flux-3-image-klein` | https://huggingface.co/black-forest-labs/flux-3-image-klein (gated; headers from https://huggingface.co/unsloth/flux-3-image-klein) | FLUX 3 image DiT 5.55B, distilled, 16 steps | partial |
| `flux-3-image-klein-base` | https://huggingface.co/black-forest-labs/flux-3-image-klein-base (gated; headers from https://huggingface.co/unsloth/flux-3-image-klein-base) | undistilled, 50 steps, true CFG | partial |
| `userrepgen-4b` (merged → startlux-decision-v1-4b) | https://huggingface.co/microsoft/UserRepGen-4B | Qwen3.5-4B fine-tune, same architecture, no MTP head (image processor: Qwen3.5 base 「假设」) | full |
| `flux-3-image-klein-pup` (merged → qwen3.5-9b) | https://huggingface.co/black-forest-labs/flux-3-image-klein-pup | FLUX 3 prompt upsampler, Qwen3.5-9B fine-tune, no MTP head | full |

- **Hy3 in `_build_std`, mechanism-first, exact params.**
  - The routed and shared experts are `expert_hidden_dim` (768) wide, as the header shows (experts.*.gate_proj is [768, 2048]); `moe_intermediate_size` (1536) is not the expert width.
  - `num_shared_experts` shared experts.
  - `mlp_layer_types` dense layer 0.
  - q/k RMSNorm.
  - fp32 `expert_bias` for the sigmoid router.
  - Before this change the template counted 58.23B against the release's 30.06B.
- **`_act_fmt`:** FP8 releases with `activation_scheme: static` (per-tensor input_scale) now run fp8 activations, like the dynamic ones. Only Hy-Memory-FP8 uses this.
- **`build_flux3`** (`Flux3Transformer2DModel`). There is no public modeling code yet, so the structure was read from the header:
  - 5 early per-modality blocks (txt / image / image_cond, each with its own q/k/v/out, QK-norm and SwiGLU), which is FLUX.2's dual-stream block plus a reference-image stream;
  - 28 shared single blocks (unfused);
  - per-modality modulation tables;
  - a flow-map target-time embedder (distilled checkpoint only).
  - Text-to-image is modelled. The image_cond stream (0.62B params) has no tokens at T2I, so it is stored and not run.
  - Pipeline: Qwen3-VL-4B text encoder (layers 4…32 concatenated = 20480 = context_embedder input) and the AutoencoderKLFlux2 VAE in fp32 (new pipeline keys `qwen3-vl-4b.flux3`, `flux3-vae`).
  - Text length is 320 tokens 「假设」 (pad_multiple 80, full upsampled caption).
  - The Base checkpoint has guidance 3.5 but no guidance embedder, so it uses true CFG (2 passes per step) 「推导」.
  - **Unmodelled fields:** the reference-image edit / multi-reference stream (`mode_blocks.image_cond`, `x_embedders.image_cond`, `*_modulations.image_cond`, `cond_in_channels`, `image_rope_scale`) and the prompt upsampler as a separate LLM decode.
- **Gated releases with a public byte-identical mirror** are read from the mirror. The check: every file's git LFS pointer oid matches; only the README differs. The release JSON keeps the official id and adds a `"mirror"` note.

### Changed
- **Muse Spark Open NVFP4 is now read, not derived** (0.175 推导 → 来源, via the public mirror https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-NVFP4).
  - hf_quant_config `exclude_modules` is layers.0., layers.61., self_attn and lm_head, so the experts of layers 0 and 61 stay bf16 (25.77B params).
  - Stored bytes rise from 495.97 to 533.01 GB (+7.5 %). Params are unchanged.
  - The expert effective bits rise from 4.50 to 4.87.
  - It still fits 8×B200 at TP8/EP8.
- Provider tencent now also covers the `tencent/` org. New families: Hy-Memory, FLUX 3, UserRepGen.

### Skipped (and why)
- mistralai Timestral-5M / 28M / 123M / 209M: time-series forecasting, a domain the catalog does not model.
- microsoft/scFM-dataselection: about 1000 single-cell Geneformer ablation checkpoints, not a model release in a modelled domain.
- tencent/Qwen3-8B-DSpine-b16: a DFlash block-diffusion draft for Qwen3-8B with a selected-prefix code bridge. The IR's draft path (DSpark) cannot express this mechanism-first, and a draft is not a standalone model. Deferred.
- GGUF-only repos, which have no config.json or safetensors index: openbmb MiniCPM-V-4.7-{35B-A3B,1B}-DSpark-gguf and unsloth GGUF repacks.
- unsloth FP8 / NVFP4 repacks that the vendor did not make.
- NVIDIA Nemotron-3.5 and Beam-1.0 (unsloth repacks): the vendor is not in the catalog.
- amd/Mixtral-8x7B w8a8: a third-party quant of a model not in the catalog.
- black-forest-labs/flux-3-image-klein-comfy: gated ComfyUI packaging of the same weights.
- DeepSeek-V4.1-Flash-1017: already added in 0.248.
- **API-only or no official open weights:** Jev (TypeSafe AI) and Microsoft-Decision-1 (still the 0.191 proxy). HF only has community "jev-style" fine-tunes and the openjev/* community reproduction. Recorded in AGENTS.

### Tests
- New test_core_251, with 4 tests:
  - every new entry loads, builds the IR and evaluates on the default chip with finite values;
  - Hy3 mechanisms;
  - FLUX 3 structure and steps;
  - merged-entry equality and Step-5 NVFP4 vs FP8.
- test_core_062_1: release counts 118 → 125 (llm 96 → 101), entries 121 → 128.
- test_core_coverage: + step-5-preview-nvfp4 (vision partial), + flux-3-image-klein(-base) (partial).
- test_core_175: NVFP4 bytes and bits now from the real headers.
- test_core_pipeline: + the two FLUX 3 pipelines; stored sizes `qwen3-vl-4b.flux3` 8.88 GB and `flux3-vae` 0.336 GB.
- test_core_062 (every catalog VLM has an encoder): UserRepGen-4B ships the Qwen3.5-4B vision tower but no preprocessor_config.json, so it uses the base-model processor (65536–16777216 px) 「假设」.


## [0.250.0]

### Added — the near-full lattice at U = 0, derived, **off by default** (MODEL §197)
- **The f = 0 boundary state 「推导」** (`_LATTICE0`, default `False`). The state f = 0 (q = 0, all slots full) is entered by an admission, whose prefill is in flight for one phase τ. It is left by a death, released one at a time.
  - An arrival at f = 0 opens a U > 0 excursion. The excursion takes the rest of the phase and returns with a fresh admission in flight.
  - Sojourn: x = (1 + (r_do − r_ch)m)/(r_do − (r_do − r_ch)λm), with m = (1 − e^{−λτ})/λ, r_do = c(B)a(B)/d and r_ch = n_B·T_d·k̄·a/(B·d)/τ.
  - Balance: π₀ = λx·π₁.
  - **Checked against DES A λ52** (`sd248/raw2_A52.npy`): λx is 1.32 (DES π₀/π₁ 1.30), and the in-flight share at f = 0 is 0.71 (DES 0.71).
- **The jump entry.** P(U > 0) = λπ₀S̄/(1 − ρ′), one M/G/1 busy period per entry. It replaces the diffusion interface 1/(θσ²), weighted by the slot-bound share β/d (the same share as `_SAT_ENTRY`). With the DES π₀ it gives 0.147, against DES 0.155.
- New diagnostic `x0`: the f = 0 sojourn, or `None` when the flag is off.

### Why it is off
With `_LATTICE0` on (`sd250/ana_l1.txt`, `xr_l1.txt`), neither goal is met:

| case | λ | DES | 0.249 (= 0.250 default) | lattice on |
|---|---|---|---|---|
| A | 50 / 52 / 53.5 / 54.3 | 143 / 341 / 734 / 1035 | 176 / 399 / 781 / 1152 | 145 / 294 / 646 / 1001 |
| D | 50 / 52 / 53.5 / 54.3 | 146 / 283 / 660 / 952 | 165 / 289 / 660 / 1006 | 153 / 231 / 556 / 887 |
| G | 44.5 / 46.5 / 47.5 / 48.3 | 182 / 457 / 797 / 1179 | 217 / 509 / 834 / 1281 | 193 / 367 / 658 / 1078 |

- **p90:** mean |err| is 10.7 % low, against 10.2 % high for 0.249. The goal was ~5 %.
- **p_sat at A λ52:** 0.138, against DES 0.155 (0.249: 0.188).
- **Crossings:** mean |err| is 0.260 %, above the 0.2 % goal (0.249: 0.152 %). The worst points are E mixed +1.38 % and F sglang +0.46 %.
  - Without the β/d weighting (`sd250/xr_l0.txt`), crossings were 0.411 % and E mixed +2.47 %.
- **What remains wrong** is the continuum below 0, not the boundary. Against DES (true-f, A λ52):
  - The modelled continuum is too wide and too deep: mean/sd 27.5/14.8 slots, against DES 24.8/13.1.
  - The deep log-slope is 0.11 per slot, against DES 0.29.
  - The deep σ_b² is about 2× the DES KM value (0.056–0.066 vs 0.026), and the deep drift is about 25 % weak.
  - As a result, the near-full level per slot is too low (0.0121 vs DES 0.013), and the correct jump entry from it under-shoots p_sat.

### Unchanged
- With the flag off, the code path is identical to 0.249 (atom z = 0). The KPI is unchanged: the A vllm slo_rate check gives 55.2329, the same as 0.249.
- sglang_mixed E is unchanged from 0.249: −4 to −13 % (1167 / 1437 / 1882 / 2979 vs DES 1219 / 1603 / 2163 / 3353). With the lattice on, it moves by the β/d share only.

### Tests
- New test_core_250: flag off by default, `x0` is `None` when off, and with it on λx falls in 1.1–1.6 and the in-flight share in 0.6–0.8, p_sat drops, and θ/base are unchanged.

### Deferred
- **The deep continuum.** The death Fano I_s is probably state-dependent: with Fano 1, the deep σ² would be 0.033 (≈ KM), but near full σ² would drop to 0.079 (KM 0.11–0.13). There is also a deep drift / pause term.
- Once the continuum matches, re-enable `_LATTICE0`.


## [0.249.0]

### Changed — state-dependent σ_b²(f) below U = 0 (MODEL §196)
- **Derived from the admission/death counting process 「推导」** (`_SIG2_F`, default on). Over times much longer than one admission, the decode-only time is t − τ_p·A(t). Each arrival therefore moves U by J(k) = d − n_B·T_d·k̄_f·a/B + τ_p·c(k)·a(k): its slot, minus the deaths its chunk iterations carry, plus the decode-only deaths its pause removes.
  - Near full J → S̄ = β + τ, which is the 0.244 1/(1 − ρ_p) batch-count amplification. Deep J → d.
  - The deaths around that mean replay clustered births (Fano I_s, §195).
  - Result: σ_b²(f) = λE[J²] + d·R(k)·I_s. The mean drift is unchanged.
- **A λ52 check.** σ_b² is 0.122 / 0.116 / 0.109 / 0.098 / 0.082 / 0.056 at f 0 / 12 / 24 / 36 / 48 / 60. DES true-f KM gives 0.131 / 0.109 / 0.086 / 0.071 / 0.049 / 0.026 at f 0 / 12 / 24–28 / 36 / 48 / 56. The 0.248 constant was 0.079.
- New diagnostic `sigma2_f`; `sigma2b` is now σ_b²(0).

### Validation
TTFT p90 (ms) against long-run DES (`sd247/lr_*`; `sd249/ana_s2.txt`):

| case | λ | DES | 0.248 | 0.249 |
|---|---|---|---|---|
| A | 50 / 52 / 53.5 / 54.3 | 143 / 341 / 734 / 1035 | 133 / 258 / 633 / 1003 | 176 / 399 / 781 / 1152 |
| D | 50 / 52 / 53.5 / 54.3 | 146 / 283 / 660 / 952 | 143 / 201 / 489 / 835 | 165 / 289 / 660 / 1006 |
| G | 44.5 / 46.5 / 47.5 / 48.3 | 182 / 457 / 797 / 1179 | 177 / 349 / 662 / 1106 | 217 / 509 / 834 / 1281 |

- Mean |err| goes from 13.9 % (all low) to 10.2 % (high, conservative).
- P(U > 0) at A λ52: 0.124 → 0.188, against DES 0.155. The gap is overshot rather than closed; see Deferred.

Crossings vs the 0.246 long-run DES table (`sd249/xr_s2.txt`): mean |err| 0.148 → 0.152 % (≤ 0.2 %).

| case | 0.248 → 0.249 | DES | err |
|---|---|---|---|
| A vllm / mixed | 55.34 → 55.23 | 55.28 | −0.09 % |
| A v0 | 55.53 → 55.42 | 55.45 | −0.05 % |
| D vllm | 55.53 → 55.42 | 55.45 | −0.05 % |
| D mixed | 55.25 → 55.13 | 55.13 | +0.01 % |
| F vllm | 43.88 → 43.75 | 43.75 | −0.01 % |
| F mixed | 43.66 → 43.53 | 43.64 | −0.26 % |
| G vllm | 49.11 → 49.00 | 49.02 | −0.05 % |
| G v0 | 49.47 → 49.35 | 49.42 | −0.15 % |
| E mixed | 18.62 → 18.73 | 18.59 | +0.77 % |
| A sglang | 55.01 → 54.90 | 55.05 | −0.27 % |
| D sglang | 54.99 → 54.88 | 54.95 | −0.13 % |
| F sglang | 42.77 → 42.64 | 42.66 | −0.04 % |

E (16k prompts, test_core_242), TTFT p90 (ms):

| profile | λ17.37 / 18 / 18.6 / 19.32 | DES | err |
|---|---|---|---|
| vllm | 1173 / 1432 / 1837 / 2710 | 1158 / 1436 / 1883 / 2748 | +1.3 / −0.3 / −2.4 / −1.4 % (0.248: +4 / +3 / 0 / 0) |
| sglang_mixed | 1167 / 1437 / 1882 / 2979 | 1219 / 1603 / 2163 / 3353 | −4.3 / −10.4 / −13.0 / −11.2 % (0.248: +3.7 / −3.8 / −8.4 / −6.1) |

The test_core_242 sglang_mixed band is back to "0.241 < model < DES".

### KPI (`sd249/kpi249.txt` vs `sd248/kpi248b.txt`)
24 of 49 rows changed, all SLO rates.

| rows | change |
|---|---|
| A / D / F / G | −0.11 to −0.13 rps (e.g. A vllm 55.34 → 55.23, F vllm 43.88 → 43.75) |
| C vllm / mixed | 18.29 → 18.49 |
| C sglang | 18.43 → 18.61 |
| C / E v0, E vllm | 18.74 → 18.78 |
| E sglang | 18.33 → 18.51 |
| E mixed | 18.62 → 18.73 |

### Tests
- New test_core_249: the σ²(f) shape (falls with f; near full above λS̄²; deep below 0.6× the near-full value); the flag off gives the 0.248 constant; p_sat rises; θ and the base are unchanged.
- test_core_239: E mixed pin 18.62 → 18.73.
- test_core_242: the sglang_mixed band goes back to 0.241 < model < DES.

### Deferred
- **Near-full lattice at U = 0.** DES has a pile at f = 0: P(q = 0, f = 0) = 0.0143, and per slot 0.0169 vs ≈ 0.013 at f 1–4.
  - Entry into U > 0 is a jump. p = λπ₀S̄/(1 − ρ′) with the DES π₀ gives 0.147, against DES 0.155.
  - The model's smooth law (0.0118 at f = 0) with the jump entry gives 0.107. The diffusion interface gives 0.188.
  - Needs a lattice treatment of the last slots. This is the remaining P(U > 0) / p90 residual.
- **σ_b² deep is high** (0.056 vs KM 0.026 at f ≈ 56–60).
- **(2) sglang_mixed E TTFT p90**, −4 to −13 %. Not attempted.
- E mixed crossing +0.77 %; E vllm λ18.65 bracket; idle k0; the 0.238 list.

## [0.248.0]

### Changed — slot-account floor law, derived in part (MODEL §195)
- **Age-structured death credit below U = 0 「推导」** (`_AGE_DEATHS`, default on). An admission adds Δ of slow-iteration wall time (fused: τ − n_B·T_d(B); SGLang non-mixed: τ). More births land in that time, so births cluster in token time with branching ratio η = λΔ, and the dying (oldest) request covaries less with the batch count.
  - Per-step deaths regress toward the mean batch m with w = 1 − (1 − g)(I_s − I₀)/(2 I_s), where I_s = 1/(1 − η)², I₀ = 1 + λ²p(1 − p)δ²/m_b, and g is the T_d(k) elasticity at m.
  - The death credit becomes c(k)·a(k) with a(k) = w + (1 − w)·m/k, and the chunk-iteration credit is scaled the same way. Above 0 nothing changes.
  - Checks: the token-time micro-model gives w 0.87 against the formula's 0.84–0.87, and its pieces I_s, I₀ and 1/(1 − g) match within 2 %. DES A λ52 per-step w ≈ 0.88; the model gives 0.867.
  - New `_slot_backlog(m_k=…)`; `w_age` is returned. The 0.247 w ≈ 0.65 was a misreading of the per-second blend.
- **Queue-ful entry law 「推导; 来源: DES」** (`_SAT_ENTRY`, default on). The U > 0 TTFT base becomes lat + β + (β/d)·(S̄ + E[S²]/(2S̄)); for A λ52 that is 79.5 → 173 ms.
  - U > 0 means q ≥ 1 prompts wait with the slots full. U enters that state by a whole S̄ (it is a lattice), and the admission in service is not counted in U.
  - DES P(U > 0.3 / 0.5 / 0.8 | U > 0) is 0.520 / 0.297 / 0.122, against 0.526 / 0.301 / 0.130 for S̄ + Exp(θ). DES wait − U is about 65 ms, which equals β + S_e.
  - DES TTFT | U > 0 p50 / p90 is 434 / 968 ms; at full weight the model gives 433 / 1009 (before: 328 / 905).
- **Slot-bound share β/d 「近似」.** The lattice entry needs f = 0. When the admission's own chunk iterations free its slot (β → 0), the queue-ful state is the budget FCFS queue at any f. The M/G/1 diffusion and T₀ already price that queue, so S̄ + S_e would double-count.
  - The suite caught this in test_core_242 (E vllm, 16k prompts, n_B 8.2, β 0.1 ms of d 22 ms).
  - The entry term is weighted by β/d: A 0.89, D ≈ 0.8, E 0.004.
  - E vllm TTFT p90 (ms):

    | λ | full entry term (before the fix) | shipped | DES (n4000×10) | flags off |
    |---|---|---|---|---|
    | 17.37 | 1346 | 1204 | 1158 | 1189 |
    | 18 | 1646 | 1481 | 1436 | 1469 |
    | 18.6 | 2045 | 1879 | 1883 | 1870 |
    | 19.32 | 2905 | 2736 | 2748 | 2730 |

    The test_core_242 vllm pin (±6 %) holds unchanged.
  - E sglang_mixed (β/d ≈ 0.5) TTFT p90 at λ17.37 / 18 / 18.6 / 19.32: 1167 / 1455 / 1897 / 2992 → 1264 / 1542 / 1982 / 3150 ms, against DES 1219 / 1603 / 2163 / 3353 (n4000×10 seeds). The error was −4 / −9 / −12 / −11 % and is now +3.7 / −3.8 / −8.4 / −6.1 %, so mean |err| goes 9.2 → 5.5 %. The test_core_242 band "0.241 < model < DES" becomes "0.241 < model and |err| < 10 %", because λ17.37 now sits 3.7 % above DES. Not re-checked with a long run.
- **Correction to 0.247.** The "T₀ near-full bulk-queue tail" was a probe artefact. The DES drops the in-flight iteration's prompts from `curs` before the iteration ends, so the probe over-counted free slots by 0.67 on average. With true f, U ≤ 0 arrivals have TTFT p90 115 ms and no tail; the 0.2–0.4 s band is the U ∈ (0, 0.3] s queue-ful state.

### Performance
- `_occ_tau`: the dot-product (`math.sumprod`) path is now used at any B when N·L > `FAST_NL_OCC` = 2000. Before, only B ≥ 256 used it. Same sums; rounding differs by ~1e-16.
- `_queue_weights`: the inner convolution is a slice update, bit-identical (checked on 16 (a, jn) pairs).
- test_core_237 guard cases, CPU seconds in the colocated paths, sequential and warm:

  | case | before | after |
  |---|---|---|
  | sglang_mixed | 19.4 | 14.9 |
  | sglang 6000 | 5.9 | 3.7 |
  | vllm_v0 6000 | 22.8 | 16.6 |

- The rt248 guard failure (vllm_v0 156.8 s CPU) was not a 0.248 regression. 056dd03 times the same case identically (cold 85.3 vs 86.0 s, warm 22.5 vs 22.8 s), and the file passes alone. rt248 ran the file at slot 64/223, beside the slow-first modules (wall 1143 s vs 1402–1861 s in earlier runs). The guard is unchanged.

### Added
- **DeepSeek-V4.1-Flash-1017** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, released 2026-10-17, HF sha 1bae0e10, MIT, fp8 + fp4 experts). Added as an alias and a `releases` note on `deepseek-v4.1-flash` in series_catalog.json.
  - Note: "config identical to V4.1-Flash; RLSF post-train update".
  - The sorted config.json diff is empty. The safetensors index total_size is 510286023000, equal to the stored bytes_total, so no separate release file was added.

### Validation — TTFT p90 (ms) vs long-run DES (`sd247/lr_*`; `sd248/ana_sb.txt`)

| case | λ | DES | 0.247 | 0.248 |
|---|---|---|---|---|
| A | 50 / 52 / 53.5 / 54.3 | 143 / 341 / 734 / 1035 | 124 / 181 / 545 / 937 | 133 / 258 / 633 / 1003 |
| D | 50 / 52 / 53.5 / 54.3 | 146 / 283 / 660 / 952 | 136 / 165 / 385 / 753 | 143 / 201 / 489 / 835 |
| G | 44.5 / 46.5 / 47.5 / 48.3 | 182 / 457 / 797 / 1179 | 165 / 260 / 560 / 1034 | 177 / 349 / 662 / 1106 |

The model is still low everywhere. The honest residual is P(U > 0): A λ52 gives 0.124 against DES 0.155, which is item (2), deferred.

Crossings vs the 0.246 long-run DES table (`sd248/xr_sb.txt`): mean |err| 0.27 % → 0.15 % (0.148 % over 13 rows).

| case | analytic (0.247 → 0.248) | DES | err |
|---|---|---|---|
| A vllm / mixed | 55.36 → 55.34 | 55.28 | +0.11 % |
| A v0 | 55.56 → 55.53 | 55.45 | +0.15 % |
| D vllm | 55.56 → 55.53 | 55.45 | +0.15 % |
| D mixed | 55.27 → 55.25 | 55.13 | +0.22 % |
| F vllm | 43.91 → 43.88 | 43.75 | +0.30 % |
| F mixed | 43.70 → 43.66 | 43.64 | +0.05 % |
| G vllm | 49.14 → 49.11 | 49.02 | +0.19 % |
| G v0 | 49.49 → 49.47 | 49.42 | +0.10 % |
| E mixed | 18.73 → 18.62 | 18.59 | +0.14 % |
| A sglang | 55.04 → 55.01 | 55.05 | −0.07 % |
| D sglang | 55.01 → 54.99 | 54.95 | +0.07 % |
| F sglang | 42.80 → 42.77 | 42.66 | +0.26 % |

### KPI (`sd248/kpi248b.txt` vs `sd246/kpi246_new.txt`)
24 of 49 rows changed, all SLO rates; operating-point TTFT/TPOT are unchanged.
- A/D/F/G: −0.02 to −0.06 rps.
- E vllm / v0 and C v0 are budget-bound (β/d ≈ 0), so they barely move.
- C vllm / mixed (−1.3 %) and the C/E SGLang non-mixed rows (−1.3 %) are slot-bound. SGLang non-mixed has no fused deaths, so β = d and the full entry term applies there.

| row | 0.247 → 0.248 |
|---|---|
| C vllm | 18.53 → 18.29 |
| C v0 | 18.75 → 18.74 |
| C sglang | 18.67 → 18.43 |
| C mixed | 18.53 → 18.29 |
| E vllm / v0 | 18.75 → 18.74 |
| E sglang | 18.58 → 18.33 |
| E mixed | 18.73 → 18.62 |

### Tests
- New test_core_248:
  - w within (0.80, 0.92), θ unchanged, p_sat up, and the flag-off value equal to 0.247.
  - The base equals lat + β + (β/d)(S̄ + S_e) with β/d in (0.8, 0.95) for A.
  - The 1017 alias resolves to the same ModelSpec, with no release file.
- test_core_239: E mixed pin 18.73 → 18.62.

### Deferred
- **(2) State-dependent σ²(f) below 0.** The true-f KM estimate (dt 77 ms) falls from 0.13 near full to 0.03 deep. A decomposition into arrival jumps and clustered deaths gives 0.088 / 0.052, the wrong shape. Plugging the KM μ, σ² straight into the floor gives p_sat 0.208 against DES 0.155, because sub-dt budget-queue transients inflate the coefficients. A coarse-grained derivation is needed.
- The large-U per-prompt cost: DES slope is 0.89·S̄ at U > 0.8 s.
- The β/d share is an interpolation. A derivation would come from the entry-state law P(f = 0 | entry).
- E vllm λ18.65 bracket; idle k0; the 0.238 list.

## [0.247.0]

### Investigated — slot-account TTFT p90 below ≈ 1 s (mechanism found; model unchanged, fix deferred)
- **Long-run DES TTFT p90, fused vllm** (`sd247/drv.sh`, n64000; A/D λ ≤ 53.5 seeds 1–2, the rest seed 1; `sd247/lr_{A,D,G}_vllm.txt`) against analytic 0.246:

  | case | λ | DES p90 (ms) | analytic | ratio |
  |---|---|---|---|---|
  | A | 50 / 52 / 53.5 / 54.3 | 143 / 341 / 734 / 1035 | 124 / 181 / 545 / 937 | 1.15 / 1.88 / 1.35 / 1.10 |
  | D | 50 / 52 / 53.5 / 54.3 | 146 / 283 / 660 / 952 | 136 / 165 / 385 / 753 | 1.07 / 1.71 / 1.71 / 1.26 |
  | G | 44.5 / 46.5 / 47.5 / 48.3 | 182 / 457 / 797 / 1179 | 165 / 260 / 560 / 1034 | 1.10 / 1.76 / 1.42 / 1.14 |

  The gap peaks where p90 is 0.2–0.6 s and closes toward the crossing (≈ 2 s), which is why the 0.246 SLO rates hold.
- **Floor law** (A λ52, `sd247/md.py` vs `sd246/probe8`). Below U = 0 the model's free-slot law has the DES mean and spread (f mean 24.6 / sd 12.4 vs 25.3 / 13.3). Its shape differs: the model peaks and falls ×2.2 toward f = 0, while the DES is flat near full. As a result p_sat is 0.114 vs DES 0.153.
- **Mechanism 1: age structure of the batch 「来源: DES」** (`sd247/probe11.py`, deaths per second by the running batch k, with the DES's own decode-only shares).

  | k | DES deaths/s | uniform-age (the model's c(k)) |
  |---|---|---|
  | 63 | 14.5 | 15.3 |
  | 50 | 13.3 | 14.6 |
  | 34.5 | 12.8 | 12.6 |
  | 14.7 | 9.4 | 8.0 |

  A full batch is young, because it was filled by a recent burst. A thin batch is old, its remaining requests left over from the busy period before. Deaths therefore regress toward the replay rate λ (≈ 0.65 weight on uniform-age at both ends). The model's death credit c(k) assumes uniform ages, which overstates the floor's restoring drift near U = 0: the Kramers–Moyal DES drift at f 0–8 is −0.02 to −0.05 U-s/s against the model's −0.09 to −0.08 (`sd247/probe9.py`).
  - The same probe shows a state-dependent variance below 0: 0.11–0.12 s near full, falling to 2λd² ≈ 0.03 deep (model constant 0.079).
- **Mechanism 2: the prefill bulk-queue tail near full** (`sd247/probe10.py`, DES arrival state of slow requests).
  - Requests with TTFT 0.2–0.4 s (5.8 %) arrive with a queue (q = 0 only 9 %) but with free slots (mean f 5.4). They wait 209 ms for the budget to reach them; only 36 % have the slots full. This is the plain-queue (T₀) branch in near-full, slow-iteration states, not the slot backlog.
  - Requests ≥ 0.4 s (8.6 %) are the backlog (f ≈ 0.7).
  - The model's backlog branch starts at base 80 ms, plus Exp(1/θ).
- **Hook check, not shipped.** `_FLOOR_W` (default 1.0) scales the floor drift, since the death credit is w·uniform-age + (1 − w)·λ replay, so μ → w·μ. w = 0.65 gives A λ52 p_sat 0.140 (DES ≈ 0.15) but p90 only 219 ms (DES 341), and moves the crossing's p90 1873 → 1827 ms. Mechanism 1 alone does not close the sub-1 s gap; both mechanisms need a derived form (deferred). w is a DES-read weight, not derived, so the default stays 1.0 and the KPI is unchanged.
- **Diagnostics.** `_slot_backlog` now returns the full `free_cdf`, the drift grid `mu_f` and `sigma2b`.

### KPI
- Unchanged: no default model change. The A vllm SLO rate is 55.363 as in 0.246.

### Tests
- New test_core_247: floor diagnostics (free_cdf monotone, μ(f) increasing, σ_b² > σ²), `_FLOOR_W` = 1.0 by default, and w < 1 scales μ(f) and raises p_sat with θ unchanged.

### Deferred
- **Derived floor law.**
  - Age-structured death credit (the regression of deaths toward λ replay, from the k-process correlation time vs the life L).
  - State-dependent σ²(f).
  - The T₀ prefill bulk-queue tail in near-full states.
  - Then re-validate p90 at λ 50–55 (A/D/G tables above) and the crossings.
- E vllm λ18.65 bracket: not run, because the DES slot went to the floor probes and the A/D/G sweep. 0.246 had λ18.45 seeds 1, 2 and 5 at 1606 / 1660 / 1758 ms.
- E mixed residual; idle k0; the 0.238 list.

## [0.246.0]

### Fixed — budget-chunked fused cases (D/F/E, A/G vllm_v0) move from the 0.243 block to the slot account
- **What the DES full-slot iteration contains** (`sd246/probe7.py`, D vllm λ55.4 / F vllm λ43.7; iterations with every slot held and a prompt waiting, by pieces and running decodes k):

  | kind | D share | D wall | F share | F wall |
  |---|---|---|---|---|
  | resumed middle chunk, k 63 | 42 % | 21.56 ms | 62 % | 22.20 ms |
  | tail + follower head, k 62 | 26 % | 21.95 | 22 % | 22.00 |
  | tail alone, k 63 | 16 % | 7.23 | 8 % | 5.70 |
  | fresh head chunk, k 63 | 16 % | 20.57 | 8 % | 20.57 |

- **What pack_at(B) omitted 「推导」.**
  - The partial prompt holds a slot, because max_num_seqs counts prefilling requests (§192). Its chunks therefore ride B − 1 decodes at budget C(B − 1), not B. A follower head joins the tail only in a slot that a death freed, so that two-prompt iteration runs B − 2 decodes. This is the new `k_top` argument of `_phase_iters` / `_phase_law`.
  - The follower starts at phase φ = the predecessor's tail. Its next chunk is a resumed one, which re-reads the head's KV (D 21.56 vs 20.57 ms fresh), and its own tail grows (D 126 → 255 tokens). pack_at puts followers at φ = 0 at k ≥ 1, because the phase chain costs fresh walls at every bin. The account needs one bin, so it runs the chain there.
  - Result: τ D 45.27 → 45.78 ms (DES 45.7–45.8), F 66.84 → 67.55 (DES 67.5). n_B D 2.43 → 2.39 (DES 2.35–2.38), F 3.29 → 3.27 (DES 3.26). E mixed τ 182.9 (DES 182.5). The tail-share probability stays pack_at's M/G/1 π: D 0.61, F 0.73 vs DES 0.62 / 0.74 「近似」.
- **k̄: deaths in the admission's chunk iterations 「推导」.** Those iterations carry k̄ = B − (prompts in flight) decodes, not B. By Little's law over chunk-iteration time, the number of prompts in flight is lat/τ. So β = d − n_B·T_d(B)·k̄/B, with the same form below 0. k̄: D 62.75, A 62.73 (DES running k in saturated iterations 62.7 / 62.8). 0.245 used B, overstating the chunk iterations' death credit by ≈ 2 %.
- **SLO rates (analytic vs long-run DES, 6 × n64000):**

  | case | DES | 0.245 | 0.246 |
  |---|---|---|---|
  | A vllm / mixed | 55.28 | 55.42 | 55.36 |
  | A vllm_v0 | 55.45 | 55.96 | 55.56 |
  | D vllm (= v0) | 55.45 | 55.96 | 55.56 |
  | D mixed | 55.13 | 55.41 | 55.27 |
  | F vllm | 43.75 | 43.89 | 43.91 |
  | F mixed | 43.64 | 43.73 | 43.70 |
  | G vllm | 49.02 | 49.18 | 49.14 |
  | G vllm_v0 | 49.42 | 49.37 | 49.49 |
  | E mixed | 18.59 | 18.68 | 18.73 |

  Mean |error| falls from 0.50 % to 0.27 %. The 0.245 block had mixed signs and was right only by compensation. Now every case is +0.1 to +0.4 %, except E mixed at +0.77 % (see Deferred). Every ranking still matches the DES. TTFT p90 at the DES crossing: A 1873, G 1807, D 1821, F 1710, D mixed 1768, F mixed 1886, G v0 1870 ms (DES ≈ 2000; 0.245 A 1792 / G 1742, block D 1402 / F 1844). p_sat D 0.498 / F 0.420 vs DES 0.517 / 0.474.
- The 0.243 clustered block is kept behind `_FUSED_BACKLOG_CHUNKED = False`, and k̄ behind `_KBAR`. `_slot_backlog` also returns `free_cdf`, the model's free-slot law below 0 (diagnostic).

### Checked — sub-1 s TTFT p90 in slot-account cases (not fixed)
- A vllm λ52 (`sd246/probe4/8.py`, n64000): DES p90 324–329 ms vs analytic 181. DES P(U > 0) is 0.153 vs model 0.114.
  - Where it agrees: U | U > 0 has mean 398 vs 1/θ 358 ms. At the boundary, P(U > 0)/density(0⁻) is 11.3 vs 11.5 slots.
  - Where it fails: the density below 0. The DES density is flat near 0 (×1.25 from f = 1 to 16, peak near f ≈ 28), while the model rises ×2.2 over the same span. The model therefore puts too much mass deep in the floor.
  - Below the crossing, p90 falls at the bottom of the backlog branch and is very sensitive to p_sat. The floor drift and variance below 0 are the open item.

### E vllm long-run DES
- Run in seed chunks (`sd246/lbs.py`, one (λ, seed) per process, ≈ 18 min and 5 GB each, results appended to `sd246/lbs_E_vllm.txt`). See Deferred for the finished points.

### The 0.243 clustered block: kept, for the non-packing path only
- `_slot_block_q` is still used where the account does not apply: fused profiles without budget packing (`chunk_profile="fixed"` / `pd.chunk_tokens`, legacy_512).
  - Example: a fixed C = 4096 profile in case A at its SLO rate 55.27 gets slot_block V 1.91 s, φ 0.625.
  - For legacy_512 in the KPI cases it is None anyway, since B·n_B ≥ out.
- `_FUSED_BACKLOG_CHUNKED = False` remains only as the hook that test_core_241 / test_core_245 use to check the block on its old cases (E mixed vs DES n4000, D vllm).
### KPI (19 of 49 rows changed; TTFT mean and TPOT unchanged everywhere)
- SLO rates:
  - A vllm / mixed 55.42 → 55.36.
  - A v0, D vllm and D v0 55.96 → 55.56.
  - D mixed 55.41 → 55.27.
  - F vllm / v0 43.89 → 43.91; F mixed 43.73 → 43.70.
  - G vllm / mixed 49.18 → 49.14; G v0 49.37 → 49.49.
  - C vllm / mixed 18.49 → 18.53; C v0, E vllm and E v0 18.82 → 18.75.
  - E mixed 18.68 → 18.73.
- Operating-point TTFT p99 falls about 1 %, because the account replaces the block at a small p_sat: A/B/D v0 95.25 → 94.16, D mixed 110.1 → 109.0, F 125.1 → 123.8, G v0 104.8 → 103.7, C 466.2 → 462.0, E mixed 389.2 → 384.9.

### Tests
- New test_core_246 (2 tests):
  - D/F account τ within 0.5 % and n_B within 2 % of the DES full-slot iteration, k̄ in (62.5, 63), no slot_block, and TTFT p90 at the DES crossing in (1.6, 2.3) s.
  - `_phase_iters` k_top identities.
- test_core_245: the β identity now carries k̄/B, and the budget-chunked block check runs behind `_FUSED_BACKLOG_CHUNKED = False`.
- test_core_243: E mixed is now on the account (p_sat in (0, 1), β > 0). The rate pin 18.59 ± 1.5 % holds at 18.73.
- test_core_241: the block-vs-DES check runs behind the flag.
- test_core_239: case E mixed SLO-rate pin 18.68 → 18.73 (moved pin; long-run DES 18.59).
- test_core_238: the vllm prompt-3000 case (cap 48, load 0.8) is budget-chunked, so it now goes through the account. TTFT p99 72.39 → 71.61 ms, mean 34.02 → 34.03; this is the intended account change. vl4096 / mx4096 / vl6000 p90 and p99 stay within the test's 3 % of the 0.238 DES (vl4096 72.14 / 110.61 vs DES 71.48 / 111.15).
- Runtime: the vl4096 `disagg_report` takes 19.6 s in 0.245 and 22.8 s now, standalone.
  - The profile shows no new hot loop. The account's chain at B − 1 is not in the top costs, and its λ-independent `_phase_iters` are now memoised on the pool across the SLO bisection.
  - The extra time comes from the decode fixed point being bisected more often as the SLO rate moves (pass_at 35 → 48 calls, same 29 `_coloc_chunked` calls).
  - In rt246 the case took 361.7 s against the 60 s guard because of memory pressure on the box: a 5–6 GB E vllm long-run DES seed was running beside the 8 suite workers, and the OOM killer fired during that suite. Run alone, test_core_238 passes with the three guarded cases at 71 s together.

### Deferred
- Sub-1 s TTFT p90: the free-slot density below 0 (see Checked), planned for 0.247.
- E mixed residual (+0.77 %): τ matches the DES, but the DES tail-share π is 0.89 vs the model's 0.84 (short run).
- Uniform +0.1 to +0.4 % over-rating: the account's p_sat at the crossing is low, e.g. F 0.420 vs DES 0.474.
- E vllm long-run DES: at λ18.45 only seeds 1, 2 and 5 finished (TTFT p90 1606 / 1660 / 1758 ms, all under 2 s, so the crossing is above 18.45; analytic 18.75). Seeds 3, 4 and 6 hit the 25-min per-seed timeout and λ18.65 was not run; the bracket is deferred.
- Idle k0; the 0.238 list.

## [0.245.0]

### Fixed — fused vLLM / SGLang-mixed over-rating in A and G: the 0.244 slot account carried over (`_slot_backlog` with n_B)
- **Source rule** (vLLM V1 `vllm/v1/core/sched/scheduler.py`, main, sha256 ed407c3d…). `schedule()` breaks its waiting loop at `len(running) >= max_num_running_reqs`, and `running` holds the partial prefills. So prefilling requests count in max_num_seqs, as in SGLang and the DES `ColocChunked._packed`.
- **Long-run DES state at arrivals** (n64000, at the long-run crossings; `sd245/probe*.py`).
  - Fraction of arrivals at P(U > 0): A 0.520, D 0.517, F 0.474 (A sglang 0.501).
  - U | U > 0 has mean ≈ 1.0 s in every case, and TTFT ≈ U + β + own latency.
  - So all fused cases share the backlog mechanism. 0.243's clustered block gave TTFT p90 at the DES crossing of A 1319, G 1303, D 1402, F 1844 ms against ≈ 2000. D/F came out right only because the block compensated; A/G were low, which caused the +1.7 / +2.0 % SLO-rate excess.
- **Fused account 「推导」.**
  - An admission's n_B chunk iterations at a full batch already give n_B·B·e/out deaths. Its decode-only wait is therefore β = d − n_B·T_d(B), with d = out·T_d(B)/(e·B) the U credit of a free slot. β ≤ 0 means no block.
  - Below 0 each arrival nets β + n_B·T_d(B)·f/B, because chunk iterations at batch B − f give only (B − f)/B of those deaths. Without this term D/F P(U > 0) comes out 0.21 / 0.14 instead of 0.52 / 0.47.
  - τ, n_B are the bulk queue's realised share at k = B − b_max, the fullest batch at which the budget's packing is slot-feasible. Admissions follow death clumps (the replayed birth cohorts), and a packed iteration needs b free slots. This gives A 39.5 ms (DES saturated 39.5) and G 48.1 (DES 48.0).
  - ρ_c/λ is not used for τ. Its bins at k = B carry pack_at's b = 1 tail-sharing share (36.1 ms in A, assuming a follower head that has no free slot), so it jumped with the decode fixed point: A λ56.6 → 56.8 went 39.5 → 36.2 ms and TTFT p90 7.7 → 0.75 s, which the test_core_243 monotone guard caught.
- **Scope: budget ≥ 2 whole prompts at a full batch (A/G, vllm and sglang_mixed).** A TTFT p90 at 55.28 is 1792 ms, G at 49.02 is 1742 ms (DES ≈ 2000). SLO rates:

  | case | DES (long run) | 0.244 | 0.245 |
  |---|---|---|---|
  | A vllm / mixed | 55.28 | 56.20 | 55.42 |
  | G vllm / mixed | 49.02 | 49.99 | 49.18 |

- **Why D/F (and E, A/G vllm_v0) stay on the 0.243 block.** The account is ≈ 30× sensitive to the admission share through ρ′ = λ(τ + β). In the budget-chunked regime τ is the full-batch b = 1 tail-shared chunk iteration (pack_at(B)), which is priced ≈ 1 % low: D 45.3 vs DES 45.8 ms, F 66.8 vs 67.5. That moves the SLO rate by +1.4 / +1.6 % (D ≈ 56.25 vs DES 55.45, F ≈ 44.45 vs 43.75), worse than the 0.243 block's +0.9 / +0.3 %.
  - With the DES τ, n_B the account reproduces D/F: p_sat 0.475 / 0.393, 1/θ 1.02 / 1.04 s, p90 1668 / 1525 ms. So the structure holds; the gap is the iteration pricing (deferred).

### Ranking (long-run DES vs analytic)

| case | long-run DES | analytic 0.245 |
|---|---|---|
| A | pf > v0 55.45 > vllm = mixed 55.28 > sglang 55.05 | pf 56.73 > v0 55.96 > vllm = mixed 55.42 > sglang 55.04 |
| G | v0 49.42 > vllm 49.02 | v0 49.37 > vllm 49.18 |
| D | vllm = v0 55.45 > mixed 55.13 > sglang 54.95 | 55.96 > 55.41 > 55.01 (same order, unchanged) |
| F | vllm 43.75 > mixed 43.64 > sglang 42.66 | 43.89 > 43.73 > 42.80 (same order, unchanged) |

The A order now matches the DES (0.244 flagged vllm > v0), and so does G.

### Checked — C/E against long-run DES (6 seeds × n64000)
- C sglang: 18.47 (analytic 18.67, +1.1 %).
- E sglang: 18.38 (18.58, +1.1 %).
- E sglang_mixed: 18.59 (18.68, +0.5 %); TTFT p90 at 18.6 ≈ 2015 ms (analytic 1935).
- E vllm did not finish in time (killed after 31 min without a bracket point); deferred.

### KPI (6 of 49 rows changed)
- SLO rate: A vllm and sglang_mixed 56.20 → 55.42; G vllm and sglang_mixed 49.99 → 49.18.
- Operating-point TTFT p99 −1.4 % in those rows and in B vllm / mixed (99.43 → 98.05; SLO unchanged 22.8): the backlog branch mixes in at small p_sat. TTFT mean and TPOT are unchanged.

### Tests
- test_core_245 (3 tests):
  - A/G account τ within 1 % of the DES saturated share, the β = d − n_B·T_d(B) identity, and TTFT p90 at the DES crossing within −20 / +15 %.
  - The budget-chunked case keeps the clustered block.
  - β ≤ 0 → no block; n_B = 0 reproduces the 0.244 account.
- test_core_243 re-pinned to the long-run DES (the n4000 references were transient):
  - A vllm 55.95 → 55.28, G 49.42 → 49.02.
  - E mixed SLO rate → 18.59 (±1.5 %); TTFT p90 at 18.6 → 2015 ms ± 15 %.

### Deferred
- Price the full-batch b = 1 tail-shared iteration (≈ 1 % low), then move D/F/E/vllm_v0 to the account.
- E vllm long-run DES.
- sglang TTFT p90 below 1 s.
- E mixed body; idle k0; the 0.238 list.

## [0.244.0]

### Fixed — SGLang non-mixed admission: prefilling requests count in max_running_requests (`_slot_backlog`)
- **Source rule** (`sd221/sgl232/scheduler.py`, `_get_new_batch_prefill_raw`). The admission loop stops at `len(can_run_list) >= get_num_allocatable_reqs(running_bs)` = max_running_requests − len(running_batch). So the prefilling requests count with the running batch, and `batch_is_full` clears only when a request leaves. The DES (`ColocChunked._packed`: running + curs < slots) applies the same rule.
- **Mechanism.** In a backlog every admission waits for a death. In the saturated cycle a death comes every β = out·T_d(B)/(e·B) of decode time (A: 512·4.262/64 = 34.1 ms), because the decodes pause in the prefill-only iterations. The admission server therefore costs S = τ + β (A: τ = 36.45 ms), and ρ′ = λ_c(τ̄ + β̄) gives λ_s = r_c/(τ + β) = 56.73, the old decode limit.
  - The blocking is the backlog itself. 0.243's iid atom used the time-average P(n ≥ B) = 0.74, which gave ρ_eff 0.86 and A TTFT p90 0.52 s vs DES 2 s.
- **Model 「推导; heavy-traffic diffusion 近似」.**
  - Slot account U = (τ + β)·q − β·f (q prompts waiting, f free slots).
  - Drift: −δ = −(1 − ρ′) above 0. Below 0 it is λβ̄ − (1 − ρ_p)·c(B − f), where c(k) = k·T_d(B)·e(k)/(B·T_d(k)·e(B)) is the decode credit rate. c is flat (c(48) = 0.91), so the soft floor is wide.
  - Variance: λE[S²] above 0 (the deaths replay the births). Below 0 it is λE[S²] + λβ̄², adding the deaths' own count noise (M/G/∞).
  - Density ∝ exp(∫2μ/σ²)/σ² on [−Bβ̄, ∞) gives p_sat = P(U > 0) and U | U > 0 ~ Exp(2δ/σ²).
  - TTFT = (1 − p_sat)·T₀ + p_sat·(lat + β + Exp). T₀ is the plain prefill-queue law, applied after the tail-sharing correlated rows.
  - That last step is why F moves now: the 0.243 iid atom went into the service mix, and the 0.237 correlated-row override then replaced it.
- **Floor check** (DES A λ53.5, n64000). P(U > 0) at arrivals is 0.282 vs 0.276 in the model, and the batch at arrival is mean 48, sd 14.3. The batch sd is far above Poisson(λH) because the prefill pauses amplify it. With births-only noise the model gives 0.229.
- **Long-run DES TTFT p90** (ms, 6 seeds × n64000) vs analytic:

  | λ | A DES | A ana | λ | F DES | F ana |
  |---|---|---|---|---|---|
  | 52 | 434 | 196 | 40.5 | 552 | 269 |
  | 53.5 | 867 | 675 | 41.5 | 978 | 634 |
  | 54.3 | 1267 | 1170 | 42.4 | 1657 | 1386 |
  | 55.06 | 2019 | 2075 | 43.06 | 2850 | 2583 |
  | 55.7 | 3344 | 3894 | 43.7 | 5772 | 6003 |

  It is good near the 2 s crossing. Below ≈1 s it under-predicts (the mixture body when p_sat is small; deferred). D equals A within se.

### Found — the n4000 DES rates of 0.243 are transient near saturation
- The relaxation time σ²/δ² is ≈ 83 s at A λ55.06, comparable to the run (≈ 72 s per replica). The window means drift, which is why the drift flag fired.
- Example: A vllm n4000 put the crossing at 55.95; long runs give 55.2 → 1874 ms and 55.95 → 3359 ms.
- The ranking was re-checked with long runs (`sd244/lbr.py`: 6 seeds × n64000, TTFT crossing, 1 % steps). prefill_first's TTFT stays ≈ 95 ms past λ_s (its decode backlog is not in TTFT), so it stays stability-limited (0.243 values).

  | case | long-run DES | analytic 0.244 |
  |---|---|---|
  | A | pf ≈56.7–57.4 > v0 55.45 > vllm = mixed 55.28 > sglang 55.05 | pf 56.73 > vllm = mixed 56.20 > v0 55.96 > sglang 55.04 |
  | D | pf > vllm = v0 55.45 > mixed 55.13 > sglang 54.95 | pf 56.73 > vllm = v0 55.96 > mixed 55.41 > sglang 55.01 |
  | F | pf ≈44.8–45.3 > vllm 43.75 > mixed 43.64 > sglang 42.66 | pf 44.83 > vllm 43.89 > mixed 43.73 > sglang 42.80 |
  | G | v0 49.42 > vllm 49.02 | vllm 49.99 > v0 49.37 |

- **sglang (non-fused):** analytic − DES is +0.0 / +0.1 / +0.3 % in A/D/F (0.243: +1.8 / +1.9 / +3.5 %). It is now last among the chunked engines in A/D/F, as in the DES.
- **Fused vs long-run DES:** D vllm +0.9 %, mixed +0.5 %; F vllm +0.3 %, mixed +0.2 %; A v0 +0.9 %, G v0 −0.1 %.
  - **Flagged for 0.245 (not fixed):** A vllm/mixed 56.20 vs 55.28 (+1.7 %) and G vllm 49.99 vs 49.02 (+2.0 %) are over the long-run DES.
  - So the A order of v0 vs vllm/mixed is inverted: DES v0 > vllm = mixed > sglang, analytic vllm = mixed > v0 > sglang. These rates are within ≈ 0.6 % (DES) and 0.9 % (analytic) of each other, a near-tie. G vllm vs v0 is likewise inverted.
  - The 0.243 test pins (A 55.95, G 49.42) were the transient n4000 references.

### KPI (7 of 49 rows changed, sglang only)
- SLO rate:

  | case | 0.243 | 0.244 |
  |---|---|---|
  | A | 56.73 | 55.04 |
  | C | 19.89 | 18.67 |
  | D | 56.69 | 55.01 |
  | E | 19.79 | 18.58 |
  | F | 44.36 | 42.80 |
  | G | 50.06 | 48.43 |
  | B (strict SLO) | 22.78 | 22.78 |

- Operating-point TTFT p99 is −1 % (A 79.35 → 78.5): the backlog branch (lat + β) mixes in with p_sat ≈ 1e-3. TTFT mean and TPOT are unchanged.
- C/E sglang are not DES-checked; there sglang falls from first among the chunked engines to between vllm and v0.

### Tests
- test_core_244 (3 tests): A monotone TTFT in λ, mechanism identities, and p_sat → 0 at low load; sglang SLO rates within 1.5 % of the long-run DES and below the fused engines in A/D/F; prefill_first/fused untouched.
- test_core_232: the "sglang with C ≥ prompt = prefill_first exactly" TTFT identity no longer holds by design (SGLang counts prefilling in max_running_requests, prefill_first does not). It is now TPOT/energy-identical with TTFT within 2 % (p99 79.35 → 78.50 ms, p_sat ≈ 1e-3): a changed premise, not a broken invariant.

### Deferred
- Fused A/G over the long-run DES (+1.7 / +2.0 %) and the A v0-vs-vllm near-tie (0.245).
- Re-pin the 0.243 tests to long-run DES references.
- sglang TTFT p90 below ≈1 s (body of the mixture at small p_sat).
- C/E sglang DES check.
- E mixed −7 … −12 %; idle k0; the 0.238 list.

## [0.243.0]

### Checked — 0.242 ranking shifts against the DES (A, D, F, G)
- **Method (`sd243/br.py`).** The DES SLO rate is found by λ bracketing in 1.5 % steps, 8 seeds × n4000, at the TTFT p90 2 s crossing (TPOT p90 ≪ 50 ms everywhere), with log interpolation. The p90 se is ≈ 10 %, so λ resolves to ≈ ±0.5 %.
  - prefill_first is limited by stability in the DES: 3/8 seeds are unstable at the analytic λ_s and 8/8 at +1.5 %. Its TTFT stays ≈ 0.12–0.24 s.
  - The DES drift flag fires on 4–7/8 seeds of fused and sglang runs near saturation, at TTFT p90 of only 1–2 s. That is window-mean noise, so the criterion stays TTFT-only, as for E.
- **SLO rates (rps):**

  | case | DES | analytic 0.242 | analytic 0.243 |
  |---|---|---|---|
  | A | pf ≈56.7–57.4 > vllm_v0 56.11 ≈ vllm 55.95 > sglang 55.71 | vllm 57.00 > pf = sglang 56.73 > v0 56.23 | pf = sglang 56.73 > vllm 56.20 > v0 55.96 |
  | D | pf ≈56.7–57.4 > vllm = v0 56.11 > mixed 55.77 ≈ sglang 55.65 | pf 56.73 > sglang 56.69 > vllm = v0 56.23 > mixed 56.01 | pf 56.73 > sglang 56.69 > vllm = v0 55.96 > mixed 55.41 |
  | F | pf ≈44.8–45.3 > vllm 43.96 ≈ mixed 43.82 > sglang 42.84 | pf 44.83 > sglang 44.36 > vllm = v0 44.18 ≈ mixed 44.17 | pf 44.83 > sglang 44.36 > vllm = v0 43.89 > mixed 43.73 |
  | G | vllm_v0 49.82 > vllm 49.42 | vllm 50.72 > v0 49.77 | vllm 49.99 > v0 49.37 |

- **What the DES confirms.**
  - prefill_first first in A/D/F (0.242 had vllm first in A: wrong; fixed).
  - Fused order in D/F (vllm ≥ mixed).
  - vllm vs vllm_v0 in A is a tie (DES 0.16 apart, analytic 0.24 apart; both inside ±0.5 %).
- **What it does not confirm.**
  - sglang (non-fused) above the fused profiles in A/D/F. The analytic is +1.8 / +1.9 / +3.5 % too optimistic there (mechanism below, deferred).
  - G vllm above vllm_v0: analytic +1.2 % / −0.9 % vs DES, a 0.8 % DES gap in the other direction.
- **Fused accuracy, analytic − DES.**
  - 0.242: A vllm +1.9 %, G vllm +2.6 %; the rest ≤ ±0.8 %.
  - 0.243: within ±1.4 % everywhere (A vllm +0.4, v0 −0.3; D vllm −0.3, mixed −0.6; F vllm −0.2, mixed −0.2; G vllm +1.2, v0 −0.9; E vllm +0.6, mixed +1.4).

### Fixed — false feasible window at the decode-capacity limit (fused decode fixed point)
- **Symptom.** Near λ_s the two fixed-point passes on the decode-seen batch oscillated: A vllm λ56.8 went k 57.2 → 63.2 → 52.3.
  - The TTFT law was non-monotone in λ: 56.0 → 16.7 s, 56.5 → 31 s, 56.8 → 0.14 s.
  - `_slo_rate`'s top-down scan took the false feasible window, so the SLO rate became the stability limit: A vllm 57.00 (DES 55.95), G vllm 50.72 (DES 49.42).
- **Fix.** When the passes oscillate (sign change, |Δ| > 0.5), bisect the fixed point k = seen(k) between the first two iterates. A decode-unstable pass inside the bracket makes the load unstable. Converged cases are untouched.
  - A vllm TTFT p90 is now monotone. SLO rate A 57.00 → 56.20, G 50.72 → 49.99.

### Changed — the slot block clusters on the full-batch bins (sglang_mixed residual)
- **Derivation 「推导」.** The block falls only on prompts served at a full batch: the quasi-static bins at k = B of the 0.242 queue-weighted law, deep in busy periods. 0.242 spread it iid over every bin (φ/B), which dilutes the full bins' load λ(S_B + V/B); the quasi-static mixture is convex in the bin load.
  - In the full regime one cycle is B admissions and out decode tokens.
  - The fill leaves J* busy-period decodes n_B tokens apart, with no block between them. It also leaves m = B − J* survivors of the starting batch, with residual lives uniform on [0, out − J*n_B]; all of them die before the cohort.
  - So m admissions wait V/m each (Σ = V = (out − B n_B)·T_d(B)), and the age pattern repeats every cycle.
  - m = 1 is the 0.241 lump. A vllm (k0 ≈ 52 of 64) gets small blocks at most admissions. With the 0.241 lump it was far too pessimistic (TTFT p90 8 s at λ54).
- **E sglang_mixed TTFT p90 (0.242 → 0.243 / DES):** 17.37 1125 → 1130 / 1219; 18.0 1400 → 1401 / 1603; 18.6 1856 → 1935 / 2163; 19.32 2992 → 3003 / 3353; 19.76 4595 → 4590 / 4559.
  - The residual narrows only at 18.6 (−14 → −10.5 %). It stays at ≈ −7 … −12 % elsewhere.
  - SLO rate 18.75 → 18.68 (DES 18.43). vllm E is unchanged (no block).

### Investigated, not shipped
- **Idle-time starting batch (scope 3).** k0 = k̄·E[(W − A)⁺]/W, where the current idle stretch (A ~ Exp of mean 1/(λT_d) decode iterations) holds no births. E mixed 18.6 gives k0 13.2 → 7.0, close to the DES's 7.6 for arrivals to an empty queue.
  - But the deterministic mean-field fill from the lower k0 under-predicts the queue batch. E got worse everywhere (mixed 18.6: 1935 → 1717; vllm 1833 → 1770; DES 2163 / 1883) and A moved to +1.3 %. Reverted and deferred.
- **sglang non-fused admission (deferred).** SGLang (and pdsim) count the prefilling request in max_running_requests. At a full batch a prompt waits for a death, and decodes advance only in decode-only iterations, so the wait is β = (out/e)·T_d(B)/B ≈ 34 ms in A. The analytic prefill-first machinery instead admits to decode after the prefill.
  - So the analytic TTFT stays ≈ 0.1 s up to the decode λ_s, while the DES p90 reaches 2 s at 55.7. P(n ≥ B) is 0.74 there.
  - An iid atom τ + β w.p. P(n ≥ B) raised the A/D TTFT p90 at 55.7 only from 97 to 520 ms (crossing ≈ 56.5 vs DES 55.7), and it misses F's tail-sharing rows.
  - It needs the queue-weighted / clustered treatment the fused path has, so it was not small and was reverted.

### KPI old → new (49 rows; 15 changed, SLO rates only; operating-point TTFT/TPOT unchanged)
- A vllm / sglang_mixed 57.00 → 56.20, vllm_v0 56.23 → 55.96. **Ranking:** prefill_first / sglang 56.73 now lead (0.242: vllm first).
- C vllm / sglang_mixed 18.32 → 18.49 (order unchanged).
- D vllm / vllm_v0 56.23 → 55.96, sglang_mixed 56.01 → 55.41 (order unchanged).
- E sglang_mixed 18.75 → 18.68 (order unchanged).
- F vllm / vllm_v0 44.18 → 43.89, sglang_mixed 44.17 → 43.73 (order unchanged).
- G vllm / sglang_mixed 50.72 → 49.99, vllm_v0 49.77 → 49.37. **Ranking:** prefill_first 50.08 / sglang 50.06 now lead.

### Tests
- `tests/test_core_243.py`:
  - A vllm TTFT p90 monotone in λ 54–57 (the bisection guard), and the A vllm SLO rate within 1.5 % of the DES 55.95.
  - G vllm within 1.5 % of the DES 49.42.
  - E sglang_mixed TTFT p90 at 18.6 between 0.242 and the DES, and the SLO rate in [18.43, 18.75).
- test_core_239: E sglang_mixed SLO rate 18.75 → 18.68.

### Deferred
- sglang non-fused admission block (A/D/F ranking).
- sglang_mixed E residual ≈ −7 … −12 %.
- Idle-time k0 together with a better fill law.
- G vllm vs vllm_v0 (+1.2 / −0.9 %).
- TPOT-mean birth-in-busy residual; joint phase–wait chain; the 0.238 list.

## [0.242.0]

### Fixed — near-saturation TTFT in fused paths: the share grows along a busy period (order effect), queue-weighted batch law
- **What the DES does.** With the slot cap lifted (sglang_mixed) and in vllm (cap inactive), the DES prompt queue **is work-conserving**. This is real scheduler behaviour, not a DES artefact.
  - A Lindley replay of the DES's own per-replica (arrival, chunk-server share) sequence reproduces the DES wait to ±2 %. vllm λ18.6 seed 2, wait mean/p90/p99 ms: DES 756/1926/3078 vs replay 744/1884/2986.
  - Decode-only iterations while a prompt waits take ≤ 0.1 % of busy time.
  - 0.241's "not work-conserving" was the slot cap alone.
- **The mechanism: order.** Shuffling the same shares (iid, same law) drops the wait p90 by ×1.10–1.30 (mean ×1.18 over 6 seeds, vllm 18.6).
  - Each finished prompt adds a decode, so k grows along a busy period: k_seen 7 → 27.5 and share 180.5 → 189.0 ms from arrival-queue 0 to ≥ 4.
  - The long waits sit behind the large-k shares, amplified by 1/(1 − ρ).
  - The analytic share mean was already right: ρ 0.857 vs DES 0.859, and an M/D/1 at the analytic share matches the shuffled replay per seed.
  - The 0.55 TTFT bins used the time-average batch law (mean k 11.6), whose shares are 177–191 ms.
- **Derivation 「推导」 (`pdqueue._queue_k_law`, `_queue_weights`).**
  - A share delays every prompt queued during it. Embedded M/D/1 departure chain: customer j of a busy period with x left behind gets weight (x_prev − 1)⁺ + a/2. The total is L_q + ρ/2 in closed form.
  - Its batch is the 0.241 deterministic-aging fill, k = min(j − 1, out/n_B) + k0·(1 − (j − 1)n_B/out)⁺ ≤ B, with k0 from the time-average law.
  - The 8 quasi-static TTFT bins are now equal-mass bins of this queue-weighted k law.
- **Slot block re-weighted the same way (`_slot_block_q`).** The 0.241 lump V hits only prompts deep in a busy period (index > J*), where the queue is long.
  - φ is now the queue weight past J* (sglang_mixed 18.6: 0.547) instead of the customer-weighted Borel tail (0.259).
  - 0.241's `_slot_block` is kept as the building block.
- **E sweep, TTFT p90 ms (0.241 → 0.242 vs DES n4000×10):**

  | λ | vllm 0.241 | 0.242 | DES | sglang_mixed 0.241 | 0.242 | DES |
  |---|---|---|---|---|---|---|
  | 15 | 702 | 717 | — | 679 | 699 | (≈740, 3 seeds) |
  | 17.37 | 1057 | 1180 | 1158 | 1008 | 1125 | 1219 |
  | 18.0 | 1219 | 1440 | 1436 | 1199 | 1400 | 1603 |
  | 18.6 | 1488 | 1833 | 1883 | 1461 | 1856 | 2163 |
  | 19.32 | 1992 | 2657 | 2748 | 2110 | 2992 | 3353 |
  | 19.76 | 2707 | 3695 | — | 3000 | 4595 | 4559 |

  - vllm is now within −3.3 … +1.9 %.
  - sglang_mixed is within −14 … +1 %. The rest is the iid lumped slot block (deferred).
  - SLO rate E: vllm 19.32 → 18.82 (DES 18.70); sglang_mixed 19.23 → 18.75 (DES 18.43).
- **Away from saturation.** Case C operating point (vllm, λ 3.47, DES n4000×5): TTFT mean 201.3 / p99 488.7 ms. Analytic 196.9 / 428.9 → 203.7 / 466.2 (p99 −12.2 % → −4.6 %).
- TPOT unchanged.

### KPI old → new (49 rows; 28 changed, operating-point TTFT within ±1 % except C fused)
- **TTFT at the operating point.** A/B/D/F/G fused move ≤ +0.9 % (legacy_512 +0.6–0.9 %).
  - C vllm / sglang_mixed: TTFT mean 196.9 → 203.7, p99 428.9 → 466.2 (DES-checked, see above).
  - C/E vllm_v0: 201.4 → 199.9, p99 403.4 → 398.2.
- **SLO rates (near saturation, fused only):**
  - A: vllm_v0 57.20 → 56.23, legacy_512 55.18 → 54.77. **Ranking:** vllm / sglang_mixed 57.0 now lead; vllm_v0 falls below prefill_first / sglang (56.73).
  - C: vllm / sglang_mixed 18.98 → 18.32, vllm_v0 19.32 → 18.82 (order unchanged: sglang 19.89 first).
  - D: vllm / vllm_v0 57.20 → 56.23, sglang_mixed 56.91 → 56.01. **Ranking:** prefill_first 56.73 / sglang 56.69 now ahead of the fused profiles.
  - E: vllm / vllm_v0 19.32 → 18.82, sglang_mixed 19.23 → 18.75 (order unchanged; matches the DES order).
  - F: vllm / vllm_v0 45.25 → 44.18, sglang_mixed 45.09 → 44.17. **Ranking:** prefill_first 44.83 > sglang 44.36 > fused.
  - G: vllm_v0 50.74 → 49.77 (now behind vllm / sglang_mixed 50.72), legacy_512 45.39 → 45.76.
  - The D/F/G SLO-rate shifts are not DES-checked; E is.

### Deferred
- sglang_mixed residual −8 … −14 % near saturation: the slot block is still an iid lump; blocks cluster in long busy periods.
- k0 at a busy-period start is taken from the time-average law; an idle-time law would be lower.
- TPOT-mean birth-in-busy residual.
- Joint phase–wait chain; the 0.238 list.

### Tests
- `tests/test_core_242.py`: `_queue_weights` closed form; `_queue_k_law` normalisation / light-load limit / instability; `_slot_block_q` ≥ the customer-weighted block; E vllm TTFT p90 within 6 % of the DES, sglang_mixed strictly between 0.241 and the DES (λ 17.37–19.32).
- test_core_239: E sglang_mixed SLO rate 19.23 → 18.75.
- test_core_233: case A (single chunk per prompt) pinned TTFT mean 42.4807 → 42.5230 ms (+0.1 %; a single-chunk fused share still grows with k — the decodes ride in the iteration — so the queue-weighted law applies there too); energy unchanged; the invariant (multi-chunk D 45.97 > single-chunk A) holds.

## [0.241.0]

### Fixed — near-saturation TTFT in fused paths: max_num_seqs blocking of the chunk server (partial, derived)
- **Hypothesis tested and refuted: queue–service correlation.** Instrumented DES (E sglang_mixed, seed 1): per-prompt chunk-server share 177.8 / 180.0 ms (cv 0.013 / 0.016) at λ15 / 18.6; vs queue length at arrival 176.9 → 182.0 ms (+2.7 %). But P-K with the DES's own service law gives a wait of 178 / 463 ms vs DES 193 / 635 ms: the DES prompt queue is not work-conserving.
- **Counterfactual DES (slots lifted to ∞, 3 seeds, n4000).**
  - sglang_mixed λ18.6: wait 743 → 524 ms, TTFT p90 2148 → 1597, TPOT p90 23.4 → 45.7. λ15: 191 → 183. ⇒ the max_num_seqs cap is the TTFT mechanism (it moves backlog from TPOT into TTFT; 0.240's "slot blocking is not the cause" was wrong — its *average* is small, but it is lumped).
  - vllm λ18.6: wait 642 vs 636, p90 1849 vs 1836 ⇒ cap inactive in vllm.
- **Mechanism 「推导」 (`pdqueue._slot_block`).** In a backlog every admission rides n_B fused iterations, so the busy period's decodes are n_B tokens apart and none dies before out tokens; the k0 decodes present at its start (PASTA: time-average batch) have uniform residual lives. k_j = j + k0·(1 − j·n_B/out)^+ reaches B after J* admissions; the next prompt then waits for the oldest decode, V = (out − B·n_B)·T_d(B) (≈ 232 ms in E), after which one prompt is admitted per death for B admissions with no further block ⇒ one V per B admissions beyond index J*. Busy-period index ~ Borel(ρ′) (deterministic share, M/D/1) with ρ′ = ρ + λφV/B (fixed point); φ = E[(N − J*)^+]/E[N]. Applied as a lumped vacation (τ + V w.p. φ/B) on the chunk-server share atoms (wait, not own latency). No block when B·n_B ≥ out (vllm E: 32·8.23 = 263 ≥ 256 — matches the counterfactual). Report key `slot_block` {V, p, J, phi}.
- **E sweep, TTFT p90 ms (0.240 → 0.241 vs DES n4000×10):**

  | λ | sglang_mixed 0.240 | 0.241 | DES | vllm 0.240 = 0.241 | DES |
  |---|---|---|---|---|---|
  | 15 | 677 | 679 | — | 702 | — |
  | 17.37 | 976 | 1008 | 1219 | 1057 | 1158 |
  | 18.0 | 1124 | 1199 | 1603 | 1219 | 1436 |
  | 18.6 | 1298 | 1461 | 2163 | 1488 | 1883 |
  | 19.32 | 1641 | 2110 | 3353 | 1992 | 2748 |
  | 19.76 | 1974 | 3000 | 4559 | 2707 | — |

  Every point moves toward the DES, none overshoots; λ15 unchanged. Remaining gap: the no-cap DES (p90 1597 at 18.6) is ~25 % above the no-block analytic (1298) in both profiles — a common non-cap mechanism (deferred).
- **TPOT unchanged** (0.240 cap still matches DES p90).

### KPI old → new (49 rows, 11 changed — SLO rates only; operating-point TTFT/TPOT unchanged)
- A vllm_v0 57.95 → 57.20 (still first).
- C vllm / sglang_mixed 20.12 → 18.98 — **ranking reverts**: sglang 19.89 > vllm_v0 19.32 > vllm / sglang_mixed 18.98 (0.240's C lead was the flagged TTFT-bound upper bound).
- D vllm / vllm_v0 57.95 → 57.20, sglang_mixed 57.58 → 56.91 (order unchanged).
- E sglang_mixed 19.79 → 19.23: now vllm 19.32 > sglang_mixed 19.23, the DES order (18.70 > 18.43); vllm unchanged (cap inactive). sglang (non-fused) 19.79 unchanged.
- F vllm / vllm_v0 46.01 → 45.25, sglang_mixed 45.88 → 45.09 (order unchanged).
- G vllm_v0 51.61 → 50.74 (still first, margin 0.02).

### Deferred
- Common ~25 % non-cap TTFT gap near saturation (both profiles; vllm entirely).
- TPOT mean birth-in-busy-period residual (needs the slot cap jointly; not cheap).
- Joint phase–wait chain; the 0.238 list.

### Tests
- `tests/test_core_241.py`: `_slot_block` reductions (no block when B·n_B ≥ out / unstable, J* fill, monotonicity), E sglang_mixed TTFT p90 strictly between 0.240 and DES at λ 17.37–19.32, blocking negligible at the operating point.

## [0.240.0]

### Fixed — fused TPOT quantiles near saturation: max_num_seqs caps a request's fused time
- **DES evidence (case E, n4000 × 10).** TPOT p90 stays flat at 23.3–23.4 ms (sglang_mixed) and 22.5–23.0 ms (vllm) from λ = 17.37 to 19.76. The 0.239 analytic TPOT p90 climbed 19.5 → 50 ms (sglang_mixed) and 19.3 → 52.7 ms (vllm).
  - Instrumented DES at λ = 18.6 (sglang_mixed): slots are full 25 % of the time. When running + prefilling = B (32), no prompt is admitted until a decode finishes.
- **Mechanism 「推导」.** A prompt is admitted only while running + prefilling < B. During one decode life a request therefore sees at most B admissions: the ≤ B − k free slots at its birth, plus one per death of the ≤ k requests older than it. Its decodes ride along at most B prompts' chunk iterations: t_on ≤ B·S_B, with n_on ≤ B·n_B fused tokens. Hence TPOT_req ≤ (B·S_B + (out − B·n_B)·T_d)/out, or S_B/n_B when B·n_B ≥ out.
  - Case E: B·n_B = 32 · 4 = 128 of 256 tokens, so the cap is ≈ (4·43 + 4·1.8)/8.
  - Any further load can only wait as prompts.
- **The cap.** Applied to the request TPOT p90/p99 of `_coloc_chunked_base`, and reported as `tpot_cap` / `tpot_capped`.
  - sglang_mixed 24.0 ms vs DES 23.3–23.4 (+2.6 %).
  - vllm 22.9–23.1 ms vs DES 22.9–23.0.
- **Effect on E.**
  - sglang_mixed is now TTFT-bound instead of TPOT-bound: SLO rate 19.76 → 19.79, goodput/card 632.4 → 633.3.
  - vllm unchanged at 19.32.

### Investigated, not fixed — near-saturation TTFT optimism and the SLO-rate gap (case E)
- **Status.** DES SLO rates (0.239) are vllm ≈ 18.70 and sglang_mixed ≈ 18.43; analytic is 19.32 / 19.79. The TPOT cap does not change this gap, because TTFT binds in the DES.
- **Slot blocking is not the TTFT cause.** Decode-only iterations while slots are full and prompts wait take only 2–3 % of the time, ≈ 7 ms per admission in the saturated cycle.
- **What the DES shows at λ = 18.6 (sglang_mixed).**
  - Prompt wait mean 635 ms; analytic TTFT mean 645 vs DES 848.
  - Chunk-server busy share 0.845 vs analytic 0.834.
  - Iteration time agrees: 43.0 vs 42.8 ms.
  - The batch seen by chunk iterations is 18.0 vs the analytic time-average 13.2: the batch grows while prompts queue.
  - The working hypothesis is queue–service correlation within prefill busy periods. Not derived yet.
- **TPOT mean is also optimistic away from the cap** (λ = 15: 5.0 vs DES 7.3).
  - Derived mechanism: a request starts decoding at its own prefill completion, i.e. inside a prefill busy period with probability ρ (DES 0.661 vs ρ 0.664). It spends the residual R = Σ over the N it leaves behind of busy periods in fused iterations: DES E[R] 0.647 s vs model L·E[BP] 0.68 s.
  - Together with Poisson arrivals during its decode-only time this gives TPOT 6.8 at λ = 15 (DES 7.3).
  - At λ = 18.6 the M/D/1 busy-period form overshoots (t_on 4.7 vs DES 3.4 s). The slot cap and the queue–service correlation interact there, so this is deferred too.
- **Tests:** `tests/test_core_240.py`. It checks the cap within 4 % of the DES TPOT p90 at λ = 18.0–19.76 for both profiles, and that the cap is inactive at E's operating point.
- **KPI A–G old → new:** 46 of 49 rows unchanged against 0.239; TTFT, TPOT mean/p99 at the operating points and J/token are unchanged everywhere (the cap is inactive at load 0.6). Only SLO-limited rates move, where the 0.239 TPOT p90 bound wrongly:
  - C vllm / sglang_mixed (C = 16382): SLO rate 19.88 → 20.12, goodput/card 636.1 → 643.8. **C ranking shift:** vllm / sglang_mixed 643.8 > sglang 636.5 > vllm_v0 622.2 (was sglang 636.5 ≳ vllm / sglang_mixed 636.1). The new rate is TTFT-bound, and near-saturation TTFT is known to be optimistic (E: 4–7 %), so this gain is an upper bound. Not DES-checked at C.
  - E sglang_mixed: 19.76 → 19.79 (632.4 → 633.3), now level with sglang (633.3) > vllm (618.3). The DES puts vllm ≈ 18.70 ahead of sglang_mixed ≈ 18.43 (0.239), so the E order of these three remains within model error.
- **Deferred:**
  - Near-saturation TTFT (queue–service correlation within busy periods).
  - The birth-in-busy-period TPOT mean together with the slot cap.
  - The joint phase–wait chain (0.239).
  - The 0.238 list.

## [0.239.0]

### Fixed — the 0.238 case-E KPI shift was a numerical artefact
- **Symptom.** E (16384/256, cap 48 GB) sglang_mixed SLO rate went 19.76 → 17.37 in 0.238. An analytic sweep of the fused TTFT law near saturation showed broken quantiles: at λ = 17.37 the mean was 505 ms but p50 was 12 295 ms and p90 = p99 = 12 393 ms. The quantiles were erratic in λ (λ = 18: p90 1372). vllm showed the same.
- **Cause.** The single-service (M/D/1) branch of the wait-correlated survival (`queueing._corr_surv`, 0.237) evaluated Erlang's alternating series out to 400 τ. Past ≈ 6 τ the series loses every digit at high ρ: at ρ = 0.95, P(W > 40 τ) comes out 0 or 1 against a true 0.0165. The 0.238 overflow fallback caught only the overflow itself. A garbage wait grid made the quantile search return its upper bracket.
- **Fix.** Use the exact series only within `MD1_EXACT_SPAN` (6) τ and the Cramér–Lundberg tail beyond, the convention `mg1_mix_sum_quantile` has used since 0.55 「推导: CL is the asymptotically exact M/G/1 tail」. After the fix the sweep is smooth and monotone. E SLO rates are back to sglang_mixed 19.76 (TPOT-bound) and vllm 19.32 (TTFT-bound), which restores the pre-0.238 E ranking.
- **DES check (n4000 × 10, E, TTFT p90 ms; DES TPOT p90 stays flat at 22.5–23.4 ms):**

  | λ (rps) | sglang_mixed DES (ana) | vllm DES (ana) |
  |---|---|---|
  | 17.37 | 1219 ± 40 (977) | 1158 ± 33 (1057) |
  | 18.0 | 1603 ± 70 (1124) | 1436 ± 54 (1219) |
  | 18.6 | 2163 ± 116 (1298) | 1883 ± 87 (1488) |
  | 19.32 | 3353 ± 244 (1641) | 2748 ± 178 (1992) |

  DES SLO rates (TTFT p90 = 2 s, interpolated) are sglang_mixed ≈ 18.43 and vllm ≈ 18.70. The DES ranks vllm ahead of sglang_mixed by ≈ 1.5 %, about 1–2 standard errors. The analytic model is 4–7 % optimistic for both near saturation, a pre-existing gap: its TPOT p90 climbs 19.5 → 50 ms between λ = 17.37 and 19.76, while the DES stays at 23 ms and its TTFT grows instead. Recorded, not fixed (see deferred).

### Investigated — the ≈ −5 % TTFT p99 residual (sglang 6000, sglang_mixed 6000, vllm 3000)
- **Ruled out.**
  - Batch-bin discretization: 32 bins instead of 8 move p99 by ≤ 0.2 %.
  - Decode-k growth within a burst: in the DES, the running batch at arrival rises only 3.8 → 5.1 from short to top-1 % waits.
- **Evidence for the mechanism (DES sglang 6000, n4000 × 3).**
  - A prompt's own prefill service rises with its own wait: mean 58.5 ms (wait < p50), 61.5, 79.2 (p90–p99), 86.7 ms (top 1 %). The topping correlation (0.237) explains only a few ms of that.
  - The top 1 % TTFT are 100 % followers.
  - The followers' phase mix depends on their wait. Below the median wait, 89 % are the first follower of a chain (head 2192 tokens, 2 iterations). In the top 5 % of waits only 6 % are, and the rest sit deeper in the chain (heads of 288 / 576 / 2480 / 2768 tokens, often 3 iterations, service 88.6 vs 78.7 ms).
  - The analytic model gives every follower with W > θ the stationary phase mix, so it misses the correlation between chain depth and wait.
- **Derived fix (deferred):** a joint (phase, W) Markov chain over successive prompts (the Lindley recursion with the 0.236 phase chain). This is a larger change with runtime cost, so it is documented here rather than shipped in this version.
- **Tests:** `tests/test_core_239.py`. It checks that the series failure exists and that the corrected tail matches the drifting-batch form at ρ = 0.95 within 2 %. It also checks that E's fused quantiles stay ordered and monotone in λ near saturation and that the E sglang_mixed SLO rate is 19.76.
- **KPI A–G old → new:** 48 of 49 rows are unchanged against 0.238. Only E sglang_mixed changes: SLO rate 17.37 → 19.76 and goodput/card 555.8 → 632.4, which is exactly the 0.237 value. Its TTFT p99 (388.4) and mean are unchanged, because the operating point (λ = 3.47) is far from saturation.
  - E ranking (chunked, analytic SLO goodput): sglang 633.3 > sglang_mixed 632.4 > vllm / vllm_v0 618.3, the pre-0.238 order. vllm stays at 19.32 rather than the 0.237 value of 19.44, because its 0.238 sweep was smooth and that move was real.
  - **E verdict against the DES (TTFT p90 = 2 s):** vllm ≈ 18.70, sglang_mixed ≈ 18.43, against analytic 19.32 / 19.76. The DES puts vllm slightly ahead, by ≈ 1.5 % or 1–2 standard errors. The analytic sglang_mixed rate is TPOT-bound because of the pre-existing near-saturation TPOT pessimism, so the order of these two profiles at E is within model error.
- **Deferred:**
  - The near-saturation fused regime: analytic TPOT p90 is pessimistic and TTFT p90 optimistic against the DES, and the SLO rate is 4–7 % optimistic at E.
  - The joint phase–wait chain for the −5 % p99 residual.
  - The 0.238 list.

## [0.238.0]

### Fixed — fused chunked TTFT tail (vLLM, sglang_mixed): wait correlation + followers at k ≥ 1
- **Mechanism (the 0.237 one, extended).** In the fused paths the running decodes take k tokens of the budget, C(k) = budget − k, so at k ≥ 1 even "multiples" (4096 at budget 2048 / 4096) are off-multiple and every prompt's tail shares its iteration with the next head.
  - Topping depends on the prompt's own wait: π(w) = 1 − e^{−λ(w + u + S′)}, where u is the decode-only iteration residual U(0, v_b) the arrival waits out (Fuhrmann–Cooper vacation; arrivals during it queue behind the prompt, so it counts for topping) 「推导 / 近似」.
  - Follower iff W > θ (predecessor's last-piece share) on the M/G/1 part of the wait only (an empty prompt queue makes a leader) 「近似」.
  - Per batch bin (8 quasi-static bins, 0.55): the bin's follower phase law (0.236 `_phase_law`, chain on) at the bin's running batch rounded to a multiple of 8, which gives the share atoms, the rows and θ; quasi-static mixture of the per-bin correlated survivals (`queueing.mg1_corr_mix_sum_quantile`, built on the 0.237 survival factored out as `_corr_surv`).
  - **Cheaper form (cost):** the phase chain at k ≥ 1 is used only for the 8 TTFT bins at k rounded to a multiple of 8 (C(k) moves < 1 % per step of 8); the birth–death mean step (`pack_at` over all k) keeps followers at φ = 0 as in 0.236. Runtime: each fused off-multiple case ≈ 12–14 s (guard < 60 s).
  - TTFT mean now uses the same bin atoms (followers at k ≥ 1 included), so mean and quantiles come from one law.
- **Parity (analytic vs DES, n4000 × 10 seeds, TTFT p90 / p99):**

  | Case (load) | DES p99 (± se) | 0.237 p90 / p99 | 0.238 p90 / p99 |
  |---|---|---|---|
  | vllm 4096 (0.85) | 111.15 ± 1.64 | −.028 / −.052 | +.010 / +.002 |
  | sglang_mixed 4096 (0.85) | 122.85 ± 2.01 | −.005 / −.082 | +.002 / +.006 |
  | vllm 6000 (0.85) | 149.51 ± 2.14 | −.029 / −.032 | −.018 / −.021 |
  | vllm 3000 (0.80) | 75.69 ± 1.29 | +.011 / −.044 | +.007 / −.044 |
  | sglang_mixed 6000 (0.85) | 153.12 ± 2.51 | +.003 / −.052 | −.000 / −.052 |

  p50 within ±0.004 everywhere. The SGLang prefill-only path (0.237) is untouched.
- **Findings.**
  - The big 0.237 residuals in the fused paths were at the "multiples": followers exist at every k ≥ 1 there, and 0.236 had cut the follower chain to k = 0 for cost.
  - vllm 3000 / sglang_mixed 6000: the correlation is real but its two parts cancel (at vllm 3000 λ = 7.68: topping correlation alone 75.0 ms, with the follower correlation 72.4, independent 72.4 / alone-latency 71.2). Their −0.044 / −0.052 p99 residual has the size of sglang 6000's −0.05, which 0.237 showed is not this mechanism either. Deferred, not tuned.
- **Fixed:** `_pk_surv` could index past its grid for x just below x_end (floating-point rounding); reached by the new path in test_core_052.
- **Fixed:** `_corr_surv` single-service wait: Erlang's M/D/1 series overflows beyond ≈ 170 τ (ρ ≈ 0.65 in test_core_088), so it falls back to the Cramér–Lundberg tail there. No change where the series is finite.
- **Runtime guard (test_core_237):** now measured in CPU time, because parallel suite load had inflated wall time to 66.95 s. The colocated phase-law paths are guarded at < 45 s per case (max 24.4 s alone, 30.0 s under suite load, vllm 4096 cv 0.5), and the whole report at < 90 s gross. The rest of that case is the disaggregated bulk law: ≈ 60 % of its runtime, 52.6 s wall in 0.237 and 57.6 s now.
- **Tests:** `tests/test_core_238.py` (reduction to the single-environment and the independent drifting-batch forms; Lindley Monte Carlo with a vacation residual; DES-referenced p90/p99 parity for vllm 4096, sglang_mixed 4096, vllm 6000 with the < 60 s guard; vllm 3000 unchanged).
- **KPI A–G old → new:** 34 of 49 rows unchanged; TPOT and J/token unchanged everywhere. TTFT changes only in fused profiles whose chunk is off-multiple at k ≥ 1:
  - A / B / D vllm_v0 (and D vllm, C = 2043): TTFT p99 92.19 → 95.25, mean 45.81 → 45.96. SLO rate unchanged.
  - D sglang_mixed: p99 103.7 → 110.1, rate unchanged.
  - C vllm / sglang_mixed (C = 16382): p99 410.9 → 428.9, rate unchanged.
  - C vllm_v0 and E vllm / vllm_v0 (C = 2046): p99 402.4 → 403.4, SLO rate 19.44 → 19.32 (goodput/card 622.2 → 618.3).
  - E sglang_mixed: p99 386.9 → 388.4, SLO rate 19.76 → 17.37 (goodput/card 632.4 → 555.8). Near the SLO-limited arrival rate the correlation adds more (follower and topped-latency mass sits on long waits).
  - F vllm / vllm_v0: p99 123.6 → 124.4. F sglang_mixed: 126.6 → 124.2.
  - G vllm_v0: p99 105.3 → 105.6.
  - **Ranking shift (E, chunked profiles, SLO goodput):** sglang 633.3 > vllm / vllm_v0 618.3 > sglang_mixed 555.8; previously sglang > sglang_mixed > vllm. Every other case keeps its ranking.
- **Deferred:** a DES check of the fused correlation at SLO-limited arrival rates (E sglang_mixed rate −12 %); the ≈ −0.05 p99 residual at long-prompt / 3000 fused cases (sglang 6000, sglang_mixed 6000, vllm 3000) — a wait-tail mechanism other than sharing; per-k (unrounded) phase chain in the mean step; mixed prompt classes per-class b (0.236).

## [0.237.0]

### Fixed — SGLang short-prompt TTFT p99 optimism (mechanism: wait correlation)
- **Diagnosis.**
  - The DES p99 is stable: n4000 × 10 seeds gives sglang 2500 at 69.87 ± 1.22 ms (standard error) and sglang 3000 at 79.66 ± 1.34, while p50/p90 agreed within ±0.01. The −0.15 / −0.12 errors were a real model gap.
  - Slot blocking is ruled out (0 blocked boundaries). The follower share agrees (DES 0.0305 vs analytic 0.031).
  - **Anatomy:** followers (prompts that start inside their predecessor's 4096-token iteration) are 3 % of requests but 88 % of the DES top 1 %. The follower median TTFT is 54.8 ms, about the leader p99.
  - The analytic convolved the wait W with the latency atoms independently.
- **Mechanism 1: topping depends on the prompt's own wait.** A prompt's last iteration is topped up iff someone arrived behind it before that iteration started, i.e. during its wait W and its first iterations S′ (FCFS).
  - Given W = w, π(w) = 1 − e^{−λ(w+S′)}.
  - Its mean over the M/G/1 wait law is exactly the 0.235 P-K LST π, E[e^{−λW}] = (1−ρ)/E[e^{−λS}]. The marginals and the TTFT mean are therefore unchanged; only the correlation is added.
  - Implemented as `queueing.mg1_corr_sum_quantile`, on the P-K wait law grid with suffix sums.
- **Mechanism 2: being a follower depends on the prompt's own wait.** A prompt starts inside its predecessor's last iteration iff that iteration had not started when it arrived. That holds when ≥ 1 prompt was waiting, or the one in service was still before its last iteration, i.e. W (share accounting) > θ, the predecessor's last-piece share.
  - θ is the topped classes' last-piece share, weighted by how often each is topped.
  - For deterministic shares this is exact. M/D/1 check: P(W > D) = P(N ≥ 2) = 0.028, vs DES follower share 0.0305.
  - Given W = w, a prompt uses its φ = 0 rows for w ≤ θ and its φ > 0 rows (stationary phase mix) for w > θ.
- **Scope:** the SGLang prefill-only path (`_coloc_prefill_first_base`, tail sharing active). Fused b = 1 environments (vLLM, sglang_mixed) are unchanged and deferred. Multiples of C are unchanged.
- **Parity (analytic vs DES, TTFT p90 / p99):**

  | Case (sglang) | 0.236 p90 / p99 | 0.237 p90 / p99 |
  |---|---|---|
  | 2500, load 0.85 (n4000 × 10) | +.005 / −.149 | +.011 / +.007 |
  | 3000, load 0.85 (n4000 × 10) | +.001 / −.118 | +.009 / −.001 |
  | 6000, load 0.85 (n4000 × 10) | +.007 / −.051 | +.008 / −.050 |
  | 4096, load 0.85 (multiple, n4000 × 10) | unchanged | +.005 / +.008 |
  | 2500, load 0.85 (n1500 × 3) | +.027 / −.166 | +.033 / −.013 |
  | 3000, load 0.85 (n1500 × 3) | +.019 / −.127 | +.027 / −.010 |
  | 6000, load 0.85 (n1500 × 3) | +.014 / −.050 | +.015 / −.050 |
  | 6000, load 0.6 (n1500 × 3) | −.011 / −.009 | +.012 / −.027 |

  p50 and TPOT are unchanged.
- **Tests (test_core_237).**
  - Reduction to the independent sum.
  - A Lindley Monte Carlo of exactly this model: p99 62.5 vs closed form 62.7 vs the independent convolution 57.0.
  - DES p99 parity, 6 seeds trimmed, within ±0.08.
  - The mean is unchanged.
  - A runtime guard: four phase-law cases (sglang_mixed 4096 at 48 GB, sglang 6000, vllm_v0 6000, vllm prompt CV 0.5) each under 60 s.

### Findings
- **vllm 900 at load 0.9, TTFT p99 is DES noise.** Per-seed p99 at n4000 × 10:
  - eight seeds are 26.2–28.1 ms; the analytic gives 28.0, +0.01 vs the median 27.8;
  - two seeds show metastable excursions: 46.8 and 605.1 ms.

  The earlier −0.57 was the mean over 3 seeds, one of them an excursion. This is not a model gap.

**Old → new (KPI A–G).** 48 of 49 rows are identical.
- F sglang TTFT p99 goes 127.1 → 124.8 ms; goodput/card stays 2839, TTFT mean 63.47 and J/token 0.2834 are unchanged.
- The SLO uses TTFT p90, so no ranking moves.

### Deferred to 0.238
- **Wait correlation for the fused b = 1 environments** (vLLM, sglang_mixed; mg1_mix_sum_quantile over the batch bins): vllm 3000 p99 −0.051, sglang_mixed 6000 p99 −0.056.
- **sglang 6000 p99 at −0.050** (n4000 × 10; follower phases of multi-iteration prompts carry a smaller overshoot): within band.
- **From 0.236:** per-class b, the phase chain at fused k ≥ 1 / length-bin mixtures, slot-bound b = 1 sharing, the itl_max lattice, and a full V4 grid re-run.

## [0.236.0]

### Changed (the 0.235 deferred packing mechanisms)
- **Head-start credit for the next prompt: the follower's phase chain** (`_phase_iters` / `_phase_law`, MODEL §183).
  - Under FCFS packing, prompts are packed into C-token iterations as one continuous stream. A prompt that starts inside its predecessor's topped-up last iteration begins at phase φ, the predecessor's tokens in that iteration. That φ fixes the prompt's own pieces, its tail, and the head it leaves for the next prompt. Example, 6000 tokens at C 4096: phases 0 → 1904 → 3808 → 1616 …
  - A leader (nobody waiting at the predecessor's last-iteration start) starts at φ = 0.
  - π(φ, i) uses the 0.235 Pollaczek–Khinchine LST rule with S′ = the time from its share start to its last iteration. It is solved jointly with the stationary phase law (phases exact for one prompt type, on a C/64 lattice for mixtures).
  - **Shares:** a piece in a full or topped-up iteration takes its tokens/C of the wall. The prompt that ends an iteration without a head takes the rest. Own latency runs from its share start to the end of its last iteration.
  - The φ = 0 atoms are the 0.235 rows, so multiples of C are unchanged.
  - **Used in:** the SGLang prefill-only tail split (`_coloc_prefill_first_base`) and the k = 0 bin of the fused b = 1 environments (`pack_at`). Report fields: `prefill.tail_share.follower` and `prefill.pack.follower_env`.
  - **Cost limits (「近似」):**
    - fused bins with k ≥ 1 keep followers at φ = 0 (the 0.235 rows), because C(k) and the walls change with k and every phase would cost fresh walls at every k (KPI case D took over 30 min before this limit);
    - with more than 2 prompt types (length bins), followers stay at φ = 0;
    - phases met by < 0.2 % of prompts fold into φ = 0;
    - mixtures use a C/16 lattice;
    - at most max(4, 32/types) latency-quantile atoms per type.

    test_core_052 had been hanging (71 s → over 22 min); with these limits its CLI test takes 34 s.
  - The longest iteration (itl) counts the leader phase plus phases met by ≥ 1 % of requests.
  - **DES check (share of prompt starts that are followers), analytic vs DES:**
    - SGLang 6000 at load 0.85: 0.149 vs 0.155;
    - SGLang 6000 at load 0.6: 0.103 vs 0.108.

    Before the k ≥ 1 cost limit the fused chain also matched: vllm 3000 at 0.8 gave 0.146 vs 0.155, and sglang_mixed 6000 at 0.85 gave 0.150 vs 0.156.
- **The bulk head moves work: the smaller later batch** (b ≥ 2, `pack_at`).
  - The (b+1)-th prompt's head of L = C − b·new tokens now goes into the head batch's wall: D_b is two-point. In 0.235 only the own-wall law O_b was two-point and D was kept.
  - The next full batch starts with the head-started prompt's remainder new − L and keeps (b − 1) whole prompts. Its own leftover, 2L, is capped below one more whole prompt (b completions 「近似」).
  - P(full batch after a head batch) = h, the head probability, assumed independent (「近似」). Fixed point with the bulk chain; O = D.
  - No head when the batch is slot-bound (k + b = max_num_seqs; pdsim admits none).
  - itl_max: the head wall at P(full)·h, and the remainder + head wall at P(full)·h².
- **Tests (test_core_236).**
  - Leader rows equal `_tail_rows`.
  - Follower credit: φ = 1904 leaves a 288-token head, so the overshoot is smaller than the leader's.
  - Multiples give a single phase.
  - DES follower share is within 0.03 (SGLang at loads 0.85 and 0.6).
  - No head when slot-bound.
  - Phase-law fuzzer (seed 236): phase law sums to 1, atom mass is conserved, latency ≥ share, follower ≤ π.
  - The 0.235 packing fuzzer and the adversarial parity set stay green.
- **Re-pinned:** test_core_233, case A coloc_chunked TTFT mean 42.4832 → 42.4807. The head's work moved into D_b; p50/p90 are unchanged and p99 goes 99.461 → 99.387.

### Findings
- **The 0.235 "+2–3 % p90 TTFT pessimism" was sampling noise** from n800 × 2 seeds. At n1500 × 3 seeds the 0.235 code already gave:
  - sglang 6000 at load 0.85: p90 +0.011;
  - at 0.6: −0.015;
  - sglang 2500 at 0.85: +0.015;
  - sglang 3000 at 0.85: +0.010.
- **Effect of 0.236 on that parity** (p90/p99, 0.235 → 0.236, n1500 × 3; measured before the fused k ≥ 1 cost limit, so the vllm 3000 and sglang_mixed rows are now back at 0.235 values):

  | Case | 0.235 | 0.236 |
  |---|---|---|
  | sglang 6000 / 0.85 | +.011 / −.055 | +.014 / −.050 |
  | sglang 6000 / 0.6 | −.015 / −.010 | −.011 / −.009 |
  | sglang 2500 / 0.85 | +.015 / −.172 | +.027 / −.166 |
  | sglang 3000 / 0.85 | +.010 / −.130 | +.019 / −.127 |
  | sglang_mixed 6000 / 0.85 | — | +.016 / −.054 |
  | vllm 3000 / 0.8 | +.022 / −.051 | +.022 / −.051 |
  | vllm 6000 / 0.85 | −.018 / −.029 | −.017 / −.028 |
  | vllm 700 / 0.9 | −.080 / +.008 | −.081 / +.001 |
  | vllm 5000 / 0.85 | +.001 / −.039 | +.001 / −.040 |
  | vllm 5000 / 0.95 | .000 / −.068 | .000 / −.069 |

  TPOT mean/p90 are unchanged (about −0.01 / −0.05). Everything is inside the V4 bands except the open residuals below.
- **Open residuals (pre-existing, not from the 0.236 mechanisms):**
  - SGLang short-prompt TTFT p99 is optimistic: 2500 at −0.166 and 3000 at −0.127 (load 0.85; DES seed noise about 0.06).
  - vllm 900 at load 0.9 has TTFT p99 −0.566, but the DES seed spread is 0.99 there: one seed has a long excursion.
  - itl_max lattice: vllm 5000 at 0.85 is −0.33 and at 0.95 is +0.13.

**Old → new (KPI, SLO goodput/card · TTFT mean/p99 ms · TPOT p99 ms):**
- **A:** vllm = sglang_mixed 3649 · 42.48/99.46 → 3648 · 42.48/99.39. The others are unchanged; the ranking is unchanged.
- **B:** vllm = sglang_mixed TTFT p99 99.43 → 99.36 (goodput/card 1460 unchanged).
- **C, D, E:** unchanged; ranking unchanged.
- **F (6000/512, 48 GB):**
  - sglang 2871 · 63.41/127.0 · 2.304 → 2839 · 63.47/127.1 · 2.308. The follower phases make the server share slightly larger: a follower's chunks start mid-iteration.
  - vllm / vllm_v0 2945 · 62.66 → 62.67 TTFT mean. The others are unchanged.
  - **Ranking:** sglang falls back below prefill-first (2839 < 2869; in 0.235 it was 2871 > 2869).
  - New F order: vllm = vllm_v0 2945 > sglang_mixed 2936 > prefill-first 2869 > sglang 2839 > legacy_512 2416 > PD 1130.
- **G (5000/512, HBM):**
  - vllm = sglang_mixed 3249 → 3246, from the head work moving into D_b.
  - **Ranking unchanged:** vllm_v0 3303 > vllm = sglang_mixed 3246 > prefill-first 3205 > sglang 3204 > legacy_512 2905 > PD 1383.

### Deferred to 0.237
- **Per-class b for mixed prompt types.** M/G^[b]/1 has a single batch cap. Per-class caps under FCFS token packing depend on the sequence of queued types: variable-capacity bin packing, with no embedded chain on the queue length. The DES covers it.
- **The phase chain for fused k ≥ 1 bins and for length-bin mixtures** (cost: fresh walls per phase per k).
- **Slot-bound b = 1 sharing** (k ≥ B − 1) is still modelled, because the itl_max k = B bin relies on it.
- **Open residuals:** SGLang short-prompt TTFT p99 (−0.13 … −0.17), vllm 900 at load 0.9 p99 (−0.57, DES p99 seed noise 0.99), and the itl_max lattice heuristic.
- **Full V4 grid re-run.**

## [0.235.0]

### Changed (the 0.234 packing gaps closed in the analytic; adversarial parity + fuzzer)
- **SGLang non-mixed tail sharing, analytic** (`_coloc_prefill_first_base` with `ctx["_tail_split"]`; sglang with `pack`).
  - Source (「来源」): SGLang `PrefillAdder` truncates the first request that does not fit; the chunked request is re-admitted first.
  - Model: at batch cap 1, each prompt type with 2·new > C becomes a two-point service. With probability 1 − π_i it runs alone. With probability π_i its last prefill-only iteration (τ tail tokens) is topped up with the next prompt's head; then the server share is Σ earlier chunks + τ/C·T_sh, and its own latency runs to the end of the shared iteration.
  - π_i comes from the 0.234 Pollaczek–Khinchine LST rule, now shared as `_tail_rows` / `_tail_pi` with the fused path.
  - The wait, busy-period batches, excess variance and itl busy duration use the expanded (share) law; TTFT uses (share, latency).
  - Check: π matches the DES (6000 tokens, C 4096, load 0.85: 0.156 vs measured 0.148).
  - Report field: `prefill.tail_share`.
- **vLLM bulk regime, the (b+1)-th prompt's head** (`pack_at`, b ≥ 2).
  - L = C − b·new. With probability h = P(more than b waiting | a full batch starts), taken from the bulk chain at service starts, the full batch also carries L tokens of the next prompt.
  - The b finishing prompts' own-wall law O_b becomes two-point. D is unchanged, because the work is token-conserving: the head leaves a later batch (「近似」).
  - itl_max takes the full-batch-plus-head wall when P(full batch)·h is met by ≥ 1 % of requests.
- **DES.** `ColocChunked._packed` now admits a prompt only while running + prefilling < slots (`max_num_seqs` counts the prefilling requests, vLLM V1 RUNNING / SGLang max_running_requests). Before this, the after_prefill admission path did not cap it.
- **Sources (unchanged from 0.234).** `max_num_partial_prefills` / `max_long_partial_prefills` do not exist in vLLM V1 main, and `long_prefill_token_threshold` defaults to 0 (off). The per-iteration invariants are therefore pieces + running decodes ≤ max_num_seqs, and at most one partial prompt (FCFS, 「推导」).
- **Tests (test_core_235).**
  - `_tail_rows` / `_tail_pi` units: multiples of C never share; one-iteration prompts have π below that of two-chunk prompts.
  - Cases D and E sglang are unchanged.
  - Adversarial non-multiple parity in the V4 bands for TTFT p50/p90/p99 and TPOT mean/p90: sglang 6000 (4096 + 1904) and 2500 (C/2 < new < C) at 0.85; vllm 3000 at budget 2048; vllm 700 (b = 2 + head) at 0.9; vllm 5000 (b = 3 + head) at 0.85. itl_max is a lattice atom and is not banded (MODEL §74).
  - Fuzzer (seed 235) over profile, prompt 300–9000, chunk budget 1024–6000, max_num_seqs 8/16/32 and load 0.7–0.95. Per iteration: tokens ≤ budget (including the decodes), running + pieces ≤ max_num_seqs, at most one partial prompt, decodes credited (vLLM) or paused (SGLang).

**Old → new.**
- **Cases A–E:** identical at reported precision (KPI table) for every profile and mode, so the rankings are unchanged. The only visible move is A coloc_chunked (vllm, b = 3 with leftover budget): TTFT mean 42.4804 → 42.4832 ms because the head probability is tiny; p50/p90/p99 and J/token are unchanged (the test_core_233 pin was updated).
- **New case F** (qwen3-8b 6000/512 B64, 48 GB card: sglang 4096 + 1904, vllm 3 chunks). sglang goes from 2856 · 61.28 / 120.5 · 2.306 · 0.2835 to 2871 · 63.41 / 127.0 · 2.304 · 0.2835 (SLO goodput/card · TTFT mean/p99 ms · TPOT p99 ms · J/token). Other profiles are unchanged.
  - **Ranking:** vllm = vllm_v0 2945 > sglang_mixed 2936 > prefill-first 2869 > sglang 2856 becomes vllm = vllm_v0 2945 > sglang_mixed 2936 > sglang 2871 > prefill-first 2869. sglang overtakes prefill-first by 0.07 %: the shared tail saves an iteration at the SLO rate, while the TTFT tail grows.
  - PD (1130) is still last of the colocated-or-PD options; legacy_512 is 2416.
- **New case G** (5000/512, HBM, vllm b = 3 + head): unchanged at load 0.6, because the head probability is ≈ 0. Ranking: vllm_v0 3303 > vllm = sglang_mixed 3249 > prefill-first 3205 > sglang 3204 > legacy_512 2905 > PD 1383.
- **Parity cost.** The sglang off-multiple TTFT p90/p99 error moves from −0.001/+0.038 to +0.024/+0.094 at load 0.6, and from +0.006/+0.062 to +0.037/+0.088 at load 0.85, inside the bands. The model now carries the shared prompt's longer latency but not the follower's head-start credit.

### Deferred to 0.236
- **Follower head-start latency credit** in tail sharing (the residual +2–3 % p90 pessimism for SGLang off-multiples).
- **itl_max lattice heuristic.** vllm 5000 at load 0.95 puts the 3-batch atom at +0.5; sglang off-multiples are at about +0.31.
- **Full V4 grid re-run.**


## [0.234.0]

### Changed (multi-prompt packing within one iteration budget; re-baselined)
- **Source (「来源」, `sd221/v234/SHA256SUMS`).**
  - vLLM V1 `Scheduler.schedule` (main): the running requests are scheduled first (decodes 1 token each, a running partial prefill min(left, budget)). Then the waiting queue is served FCFS `while token_budget > 0`, stopping at `len(running) ≥ max_num_seqs` (`max_num_active_reqs`). Each admitted request gets min(new tokens, budget left), so the last one is truncated (chunked).
  - `long_prefill_token_threshold` defaults to 0 (disabled) and is not modelled. `max_num_partial_prefills` / `max_long_partial_prefills` no longer exist in V1 (`config/scheduler.py`).
  - SGLang `PrefillAdder`: the chunked request is added first, then whole requests while they fit; the first one that does not fit is truncated to `rem_chunk_tokens` (`_select_prefill_admission`) and becomes the next chunked request. Page alignment is ignored (「近似」).
  - The two engines share one FCFS fill rule (`pdqueue.pack_fill`). At most one prompt is partial at any time (「推导」).
- **Which profiles pack.** `chunk_spec(...)["pack"]` is True for vllm, vllm_v0, sglang_mixed and sglang. legacy_512 and fixed keep one prompt per iteration.
- **Iteration cost.**
  - `_fused_multi` / `_multi_bracket` handle several prompt pieces in one iteration: compute, non-weight DRAM and link add, weights are read once, and sync is the max. This is exact for the stage and serial modes.
  - In pipe mode, the one-piece op-level replay is scaled by the bracket ratio (「近似」, exact for one piece).
  - `_packed_fused(pool, k, pieces)`: whole prompts of the same shape run as one batched prefill, the same way prefill-first batches them. A partial piece is its own chunk (`_chunk_res`).
- **Analytic (`_coloc_chunked_base.pack_at(k)`).** C(k) is the budget left by the k decodes; k = 0 gets the whole budget, as in the DES. b = min(⌊C/new⌋, B − k).
  - **b ≥ 2:** the chunk server is the greedy bulk queue M/G^[b]/1 (`core/bulkq`), with D_j = the packed iteration of k decodes plus j prompts. The per-request busy share, iterations per prompt and itl_max come from the bulk chain. The truncated head of a (b+1)-th prompt is not modelled (「近似」).
  - **b = 1 (tail sharing):** the last iteration (τ tail tokens) is topped up with the next prompt's head with probability π_i = 1 − (1 − ρ)/E[e^{−λS}]·e^{−λS′_i} (P-K LST; S′ = the iterations before the tail). This is ≈ ρ for multi-chunk prompts and ≈ ρ² for one-iteration prompts.
    - When shared, the server share is the full chunks plus τ/C of the shared iteration.
    - The prompt's own latency runs to the end of the shared iteration.
  - **TTFT:** environments with b ≥ 2 mix the per-environment bulk laws ⊕ U(0, decode step) (`_law_mix`); otherwise the 0.55 quasi-static M/G/1 mixture, with the two-point (share, latency) service.
  - **Report fields:** `prefill.pack` = b, π and mean batch per environment. `prefill.chunks` stays the per-prompt chunk plan; `prefill.iters_per_prompt` is the packed count.
- **DES (`ColocChunked._packed`).** The same FCFS fill, with slot admission as max_num_seqs. sglang (non-fused) now has a DES replica of its own: prefill-only packed iterations up to C tokens; the decodes pause.
- **Energy.** Unchanged. Action counts are per prompt token; the iteration time enters through the W identity.

**Parity** (DES n 1000, seeds 11/12, err = (ana − DES)/DES, ttft p50/p90/p99 · tpot mean/p90 · itl_max):

| Case | Errors |
|---|---|
| A vllm | .002/.023/.098 · .005/−.019 · .004 |
| A vllm, load 0.85 | .000/.028/.071 · .008/−.014 · .004 |
| C vllm | .000/−.029/−.093 · −.040/−.076 · .001 |
| D vllm | .000/.011/.068 · .006/−.018 · .004 |
| D sglang_mixed | .001/.002/.056 · .009/−.016 · .004 |
| E vllm | .000/.055/.099 · −.057/−.170 · .002 |
| E sglang_mixed | .000/.039/.053 · −.056/−.161 · .001 |

All are inside the V4 bands. E's tpot_p90 of −0.17 was already −0.17 in 0.233.

**Old → new, coloc_chunked** (qwen3-8b 1P TP 2, 2P + 6D, load 0.6). Cells are SLO goodput/card · TTFT mean / p99 ms · TPOT p99 ms · J/token:

| Case / profile | 0.233 | 0.234 |
|---|---|---|
| A vllm = sglang_mixed (b = 3, k = 0: 4) | 3658 · 42.06 / 82.93 · 2.338 · 0.2019 | 3649 · 42.48 / 99.46 · 2.338 · 0.2019 |
| A vllm_v0 (2043, tail sharing) | 3707 · 43.14 / 85.92 · 2.292 · 0.2031 | 3708 · 45.81 / 92.19 · 2.291 · 0.2031 |
| B vllm = sglang_mixed | 1459 · 42.05 / 82.9 | 1460 · 42.48 / 99.43 |
| B vllm_v0 | 1517 · 43.14 / 85.88 | 1517 · 45.8 / 92.17 |
| C vllm = sglang_mixed (16382 + tail) | 632.7 · 190.7 / 373.8 · 2.778 · 1.235 | 636.1 · 196.8 / 410.9 · 2.777 · 1.235 |
| D vllm = vllm_v0 (2043) | 3707 · 43.14 / 85.92 · 2.292 · 0.2031 | 3708 · 45.81 / 92.19 · 2.291 · 0.2031 |
| D sglang_mixed (4091) | 3700 · 42.55 / 84.29 · 2.314 · 0.2025 | 3685 · 48.04 / 103.7 · 2.313 · 0.2025 |
| E vllm = vllm_v0 (2046) | 615.7 · 193.7 / 382.1 · 2.717 · 1.229 | 622.2 · 201.4 / 402.4 · 2.717 · 1.229 |
| E sglang_mixed (4094) | 636.9 · 191.9 / 377.4 · 2.752 · 1.225 | 632.4 · 196.1 / 386.9 · 2.751 · 1.226 |

Unchanged: legacy_512 (one prompt per iteration), sglang (A: 4 whole prompts per C, already the bulk caps; D/E: new is a multiple of C, so there is no tail to share), prefill-first and PD.

- **Mechanism.** The prefills are compute-bound, so packing j prompts costs about j × one prompt; only the weight read is shared.
  - The first prompt of a packed iteration finishes later: the TTFT tail grows (A p99 +20 %, D sglang_mixed +23 %).
  - The capacity effect is small: ±0.5 % SLO goodput, except +1.1 % for E vllm (fewer near-empty tail iterations) and −0.7 % for E sglang_mixed.
- **Conclusion shifts.**
  - **E SLO goodput:** prefill-first 656.3 > sglang_mixed 636.9 > sglang 633.3 > vllm 615.7 becomes prefill-first 656.3 > sglang 633.3 > sglang_mixed 632.4 > vllm 622.2. sglang and sglang_mixed swap back (by 0.9); prefill-first stays first and vllm stays last.
  - **E energy:** sglang_mixed 1.226 = legacy_512 1.226 < sglang 1.228 < vllm 1.229 < prefill-first 1.236. Unchanged apart from the sglang_mixed / legacy_512 tie at 4 digits.
  - **Prefill-first's win in C and E is not closed by packing.** A 16384-token prompt fills the whole budget, so packing only shares the k-token tail. prefill-first batches two whole prompts without a token budget.
  - **C:** prefill-first 656.3 > sglang 636.5 > vllm 636.1; the gap to sglang narrows from 3.8 to 0.4.
  - **D:** the order vllm 3708 > sglang_mixed 3685 > prefill-first 3631 > sglang 3628 is unchanged.
  - **A:** vllm 3649 > prefill-first 3631 is unchanged; vllm_v0 3708 is the best colocated profile.
  - **PD vs colocated:** unchanged. PD loses in A, C, D and E. In B, PD 1501 beats prefill-first, vllm, sglang and legacy, but vllm_v0 has 1517 in both 0.233 and 0.234; that profile was not in the 0.233 table.

### Deferred to 0.235
- **SGLang non-mixed analytic tail sharing.** The DES models it. The analytic still uses the prefill-first bulk caps (b·new ≤ C) with no tail sharing, so there is a parity gap only when new mod C ≠ 0 and the prompt has more than one chunk (cases A–E are not affected).
- **The truncated (b+1)-th head in the bulk regime.**
- **Full V4 grid re-run.**


## [0.233.0]

### Changed (fused chunked-prefill iteration evaluated per chunk; re-baselined)
- **What changed.** `pdqueue._chunk_fused` now evaluates each fused chunk iteration (vllm / vllm_v0 / sglang_mixed / legacy_512 / fixed) as the engine runs it:
  - the chunk [q, e) is a prefill of e tokens with the first q cached (`_chunk_res`), carrying its own activation footprint and spill and the earlier tokens' KV re-read;
  - it is fused with the k running decodes (`_fused_step(dec, chunk, 1, 0)`), memoised per (k, S, p, q, e).
  - Up to 0.232 it was an f = C/new share of the whole prompt plus an averaged re-read term, which carried the whole prompt's activation spill into every chunk.
  - Used by the analytic mean step (Σ chunks per prompt; the longest chunk iteration sets the worst gap), the DES (`ColocChunked`, chunk [S − left, S − left + take)) and energy (`_chunk_counts`: the chunk evaluations' counts, each chunk's weight DRAM shared with the decode iteration; the separate re-read term is gone because it is inside each chunk's kv_read).
  - A bilinear token-grid interpolation of chunk boundaries was tried and dropped: it was 4–6 % off at GEMM tile boundaries. Chunks are evaluated exactly.
- **External-validation proxy.** The opt-in continuous-batching proxy (`extval._cb_at`) evaluates prompts longer than the chunk budget per chunk in the same way. The default external validation does not use it and is unchanged (stamp only).
  - The 7 TRT-LLM 1.2.0 rows with 32768-token prompts > 8192 change in prediction / measured: gpt-oss-120b 0.853 → 0.805, gpt-oss-20b 1.254 → 1.196, the others −0.001 … −0.005.
- **Invariants (test_core_233).**
  - A prompt that fits one chunk costs exactly its plain prefill, and the ≤ 0.232 formula coincides with the new one there.
  - Without activation spill, the summed chunk time and DRAM are ≥ the one-shot prefill. With spill (8192 and 16384 tokens on 1P), every chunk spills less, and summed DRAM can be below one-shot.
  - DES parity in the 2048-budget case D is inside the V4 bands. A fuzz (seed 233) over the fused profiles checks J/token × tok/s = W.

**Old → new, coloc_chunked** (0.232 cases, qwen3-8b 1P TP 2, 2P + 6D, load 0.6). Cells are SLO goodput/card · TTFT mean / p99 ms · TPOT p99 ms · J/token:

| Case / profile | 0.232 | 0.233 |
|---|---|---|
| A, B (one chunk per prompt), every profile | — | unchanged |
| C vllm = sglang_mixed (16382 + 2-token tail) | 633.2 · 190.4 / 373.2 · 2.776 · 1.235 | 632.7 · 190.7 / 373.8 · 2.778 · 1.235 |
| C legacy_512 | 479.2 · 243.9 / 519 · 2.852 · 1.238 | 486.5 · 242.9 / 516.4 · 2.842 · 1.226 |
| D vllm (2043) | 3711 · 43.1 / 85.79 · 2.291 · 0.2031 | 3707 · 43.14 / 85.92 · 2.292 · 0.2031 |
| D sglang_mixed (4091) | 3702 · 42.5 / 84.14 · 2.313 · 0.2025 | 3700 · 42.55 / 84.29 · 2.314 · 0.2025 |
| E vllm (2046) | 606.6 · 194.4 / 384.1 · 2.725 · 1.241 | 615.7 · 193.7 / 382.1 · 2.717 · 1.229 |
| E sglang_mixed (4094) | 624.3 · 192.6 / 379.2 · 2.759 · 1.237 | 636.9 · 191.9 / 377.4 · 2.752 · 1.225 |
| E legacy_512 | 479 · 243.9 / 519 · 2.852 · 1.238 | 486.5 · 242.9 / 516.4 · 2.842 · 1.226 |

The sglang profile (non-fused, already per chunk) and prefill-first are unchanged.

- **Conclusion shifts.**
  - **E SLO goodput:** prefill-first > sglang > sglang_mixed > vllm becomes prefill-first 656.3 > sglang_mixed 636.9 > sglang 633.3 > vllm 615.7. sglang and sglang_mixed swap; prefill-first stays first and vllm stays last.
  - **E energy:** sglang (1.228) is no longer the lowest. New order: sglang_mixed 1.225 < legacy_512 1.226 < sglang 1.228 < vllm 1.229 < prefill-first 1.236. Every chunked profile now avoids the 16384-token activation spill, so all are below prefill-first.
  - **D and A–C:** rankings unchanged. PD vs colocated unchanged (PD best only in B).

### Deferred to 0.234
- **Multi-prompt packing within one budget.** A bulk-service chunk server (M/G^[q]/1 with q(k) = ⌊C(k)/new⌋ plus the carried partial chunk), the per-k quasi-static mixture, a DES filler, and SGLang partial-prompt packing. Design is in NOTES_0233.
- **Full V4 grid re-run.**

## [0.232.0]

### Added (SGLang chunked-prefill profiles for the colocated chunked mode)
- **`pd.chunk_profile = "sglang"` (「来源」, non-fused).** This is SGLang's default, with `enable_mixed_chunk = False` (`arg_groups/fields/schedule.py`).
  - The scheduler runs a prefill batch whenever one can be formed ("Run prefill first if possible", `scheduler.get_next_batch_to_run`) and re-admits the chunked request first.
  - So a long prompt runs as consecutive prefill-only chunk iterations, and every running decode pauses until the prompt finishes.
  - **Model.** The prefill-first queue and decode machinery (analytic and DES `ColocPrefillFirst`) runs on a chunked view of the pool (`pdqueue._ChunkPool`). A prompt longer than C costs Σ_j prefill(tokens [p+jC, p+(j+1)C), earlier tokens as cached prefix). The engine model therefore carries each chunk's weight re-read, prefix-KV re-read and on-chip activations.
  - **Packing.** b prompts share one iteration only if b·new ≤ C.
  - **Energy.** The chunk iterations' own action counts are summed.
  - **Not modelled (「近似」).** A prompt's tail chunk is not packed together with the next prompt's head.
- **C = chunked_prefill_size by memory per card** (`arg_groups/memory_hook.py`, gpu_mem in MiB): < 20 GiB → 2048, < 35 → 4096, < 60 → 4096, < 90 → 8192, < 160 → 8192, otherwise 16384. `pd.chunk_budget` overrides it.
- **`"sglang_mixed"` (`--enable-mixed-chunk`, 「来源」).** PrefillAdder takes `running_bs` off `rem_chunk_tokens`, and `mix_with_running` fuses the decodes into the batch. This maps exactly onto the vLLM fused, decode-first rule with budget = chunked_prefill_size (test: identical to `vllm` at the same budget).
- **UI, CLI and smoke.** New profile options and a mode note for the non-fused profile ("SGLang 不混合 … decode 暂停"). UI smoke checks it.
- **Simulator parity** (sglang, C = 1024, 4096/512, load 0.6, n 1000, seeds 11/12). Analytic − DES: TTFT p50 −0.001, p90 +0.048, p99 +0.004; TPOT mean +0.004, p90 −0.028. Prefill-first on the same run: +0.054 / +0.010 / +0.004 / −0.028.
  - itl_max is −0.199. This is the single-prompt-length lattice atom of the busy-period p99 (MODEL §74); prefill-first shows −0.111.

**KPIs vs vllm and prefill-first** (qwen3-8b, 1P, TP 2, 2P + 6D, load 0.6, tpuv4i_7nm + idle_W 20; `sd221/kpi232.py`). SLO goodput is per card, TTFT is mean / p99 in ms, TPOT is p99 in ms:

| Case | Prefill-first | vllm | sglang | sglang_mixed |
|---|---|---|---|---|
| A 4096/512 B64, 288 GiB (C 16384) | 3631, 40.5 / 79.4, 2.339, 0.2027 | 3658, 42.1 / 82.9, 2.338, 0.2019 | = prefill-first | = vllm |
| B = A with SLO 300 / 2.5 ms (PD 1501) | 1458 | 1459 | 1458 | 1459 |
| C 16384/256 B32, 288 GiB | 656, 189.3 / 370, 2.786, 1.236 | 633, 190.4 / 373, 2.776, 1.235 | 637, = prefill-first per request | = vllm |
| D = A with cap 48 GB (vllm 2048 / sglang 4096) | 3631 | 3711, 43.1 / 85.8, 2.291, 0.2031 | 3628, = prefill-first | 3702, 42.5 / 84.1, 2.313, 0.2025 |
| E = C with cap 48 GB | 656, 189.3 / 370, 2.786, 1.236 | 607, 194.4 / 384, 2.725, 1.241 | 633, 190.3 / 373, 2.795, 1.228 | 624, 192.6 / 379, 2.759, 1.237 |

- **Conclusion shifts.**
  - **Whole prompt fits one chunk** (A, B, C on 288 GiB): SGLang is prefill-first per request. Its SLO-goodput loss (C: 637 vs 656; D: 3628 vs 3631) comes from packing: at higher λ prefill-first batches two prompts, which SGLang cannot fit into one chunk.
  - **Long prompts on smaller cards (E):** ranking by SLO goodput is prefill-first > sglang > sglang_mixed > vllm. The fused vLLM rule with a 2048 budget is now last.
  - **E energy:** sglang has the lowest J/token (1.228). The unchunked 16384-token prefill spills 53 GB of activations to DRAM; 4096-token chunks keep them on chip and pay only 4 × 7.4 GB of weight re-reads and 0.3–0.9 GB of KV re-reads.
  - **D:** vllm (3711) > sglang_mixed (3702) > prefill-first (3631) ≈ sglang (3628).
  - **PD vs colocated:** unchanged. Colocated is best in A and C–E; PD is best in B (1501).

### Deferred to 0.233
- **Fused-path activation spill.** `_fused_step` and the coloc_chunked energy scale the full prompt's non-weight DRAM (including activation spill) by the chunk share f. For long prompts this overstates vLLM chunk time and energy relative to the per-chunk engine evaluation used by `sglang`.
  - Mechanism fix: evaluate the chunk itself, as `_ChunkPool` does. This will move E's vllm figures.
- **Packing several prompts within one budget** (vllm fused path, and SGLang partial-prompt packing): needs a bulk-service chunk server.
- **Full V4 grid re-run** (still deferred).

## [0.231.0]

### Changed (colocated chunked prefill: engine profiles replace the 512-token 「假设」; default = vllm; re-baselined)
- New `pd.chunk_profile` (omit-default; the default scenario hash is unchanged) and `pd.chunk_budget`; CLI `--pd-chunk-profile`, `--pd-chunk-budget`; UI 「分块配置」/「token 预算」.
  - **`vllm` (default, 「来源」).** vLLM's iteration token budget `max_num_batched_tokens` includes the running decodes and is filled decode-first; the last pending prompt is chunked. So prompt tokens per iteration = budget − running batch k, computed per batch k (analytic: per-k chunk plans and chunk-iteration rate; DES: per iteration).
    - Budget by card memory, from vLLM `engine/arg_utils.py` (API server defaults): < 70 GiB → 2048, ≥ 70 GiB → 8192, ≥ 160 GiB → 16384. `chunk_budget` overrides it.
    - Budget ≤ batch is infeasible: the decodes fill the budget, so the mode is reported unstable with the reason. Decodes are never starved; in the DES an iteration with no budget left is decode-only.
  - **`vllm_v0` (「来源」).** The V0 docs default: budget 2048 regardless of memory.
  - **`legacy_512` (「假设」).** The ≤ 0.230 behaviour: 512 prompt tokens per iteration on top of the decodes.
  - **`fixed`.** `chunk_tokens` per iteration. A ≤ 0.230 scenario dict with `chunk_tokens` ≠ 512 and no profile is upgraded to `fixed`, and so is CLI `--pd-chunk` without a profile.
- **Why vllm is the default (mechanism).** The colocated chunked mode is the fused decode-first scheduler (Sarathi-Serve / vLLM). Its chunk size is an engine setting with a documented default, and the 512 figure was unsourced.
  - At 4096-token prompts, the 8192/16384 tiers run the whole prompt in one fused iteration. The 2048 tier runs it in 2–3 iterations.
  - The 512-token chunks cost 8 fused iterations, each re-reading the prompt's earlier KV.
- **Not modelled (「近似」).** Several prompts packed into one iteration: one prompt per chunk iteration, as before.
- **Validation.**
  - The V4 stored grid (`data/v4_serving.json`, 512-token runs) is pinned to `legacy_512` via `v4_scenario(chunk_profile=…)`. The live `validate` V4 subset now runs the vllm profile.
  - The coloc_chunked analytic-vs-DES errors stay within the same range as legacy_512 (load 0.6 CV 1: ttft_p90 −0.213 → −0.207; prefix: ttft_p99 +0.065 → +0.124, inside its band).
  - External validation (extval) uses its own chunk settings and is unaffected.

**Old → new KPIs, coloc_chunked, legacy_512 → vllm** (qwen3-8b, 1P, TP 2, 2P + 6D, load 0.6, tpuv4i_7nm + idle_W 20; `sd221/kpi231.py`):

| Case | C | TPOT p99 ms | TTFT mean / p99 ms | SLO goodput / card | J/token |
|---|---|---|---|---|---|
| A 4096/512 B64, 288 GiB (16384 tier) | 512 → 16379 | 2.357 → 2.338 | 55.2 / 118.7 → 42.1 / 82.9 | 3532 → 3658 | 0.2020 → 0.2019 |
| B = A with SLO TTFT 300 / TPOT 2.5 ms | 512 → 16379 | 2.356 → 2.337 | 55.2 / 118.7 → 42.1 / 82.9 | 1437 → 1459 | 0.2021 → 0.2020 |
| C 16384/256 B32 | 512 → 16382 | 2.852 → 2.776 | 243.9 / 519 → 190.4 / 373 | 479 → 633 | 1.238 → 1.235 |
| D = A with cap 48 GB (2048 tier) | 512 → 2043 | 2.357 → 2.291 | 55.2 / 118.7 → 43.1 / 85.8 | 3532 → 3711 | 0.2020 → 0.2031 |

- **Conclusion shifts.**
  - By SLO goodput/card, colocated chunked now ≥ colocated prefill-first in A (3658 vs 3631) and D (3711 vs 3631); it was below in all cases. In C it is still below (633 vs 656).
  - PD vs colocated: unchanged at the default. PD remains best under the tight SLO (B: PD 1501 vs chunked 1459) and colocated remains best under the loose SLO.
  - With `vllm_v0` (2048), B flips: chunked 1518 > PD 1501.
  - J/token: in D, coloc_chunked moves from between pd and prefill-first to the highest (0.2031 vs pf 0.2027): fewer, larger fused iterations re-read less KV but run at lower MAC utilisation per iteration. These are ±1 % effects.
- **Tests.** test_core_051 and 052 pin `legacy_512` / `fixed` for the 512-chunk mechanics. New test_core_231 covers the tiers, validation and upgrade, default vs legacy, infeasible budget, DES parity at a tiny budget (decodes not starved), and a fuzz (seed 231) over all profiles: chunk ≥ 0, decode-first budget split, TTFT monotone in prompt length, J/token × tok/s = W.

### Deferred
- **SGLang profile.** Main `chunked_prefill_size` is tiered by GPU memory (2048 / 4096 / 8192 / 16384; v0.3.6: 8192, ÷4 under 25 GB), with `enable_mixed_chunk = False` by default. Without mixing, a chunk iteration is prefill-only and pauses the decodes. It needs a new non-fused mode, close to prefill-first with a per-iteration prompt-token cap, with its own analytic, DES, energy and UI paths.
- **Full V4 grid re-run** under the vllm profile (hours).

## [0.230.0]

### Changed (PD report energy brought up to the main energy report; re-baselined)
- `pdqueue._energy` prices each operating point with the same terms as `energy.energy_report`. It now takes the pools' scenarios: PD = decode pool + prefill pool; colocated = the replica pool.
  - **Per-kind DRAM energy plus the chip-side PHY line** (`dram_phy`), like 0.186/0.188 in the main report. The DRAM bytes are priced with the first pool's memory kind; for heterogeneous PD memory this is 「近似」.
  - **`idle_frac_tdp × tdp_W` is now honoured.** Up to 0.229 the PD path read only an explicit `idle_W`, so the preset's H100 idle fraction was silently dropped.
  - **SRAM leakage** (state SRAM always; shared SRAM only without an idle figure) and **HBM standby/refresh** (only without an idle figure), per pool × card-seconds. Both count as static in `static_share`.
  - **Per-cycle clock term** (0.228) over every provisioned card-second, with the chip's gating g + (1 − g)·u (0.229). A provisioned card idles at serving clocks, which is the H100 "context, no work" state the preset value is measured in. u = MAC slot-cycles used ÷ clocked, pooled over all pools (「近似」, one utilisation). `clock_gating.mac_util` is reported.
- **Invariant (new test):** a single-phase operating point priced by `_energy` with its own card-seconds per unit equals `energy_report`. This holds to 1e-9 across idle / TDP / no-idle tables, g ∈ {1, 0.3}, decode/prefill, an LPDDR RWKV case with a state SRAM, and term by term. J/token × tok/s/card = W/card is fuzzed on the PD report (seed 230).

**PD J/token, old → new** (qwen3-8b, 1P, TP 2, PD 2 prefill + 6 decode cards, B 64, prompt 4096 / out 512, load 0.6, tpuv4i_7nm preset):

| Table | pd | coloc_prefill_first | coloc_chunked |
|---|---|---|---|
| + idle_W 20 | 0.1358 → 0.2004 | 0.1378 → 0.2027 | 0.1372 → 0.2020 (+48 %) |
| no idle | 0.1170 → 0.1902 | 0.1189 → 0.1925 | 0.1184 → 0.1918 (+63 %) |
| + tdp_W 700 | 0.1170 → 0.2492 | 0.1189 → 0.2514 | 0.1184 → 0.2508 (+113 %: the 71.8 W idle is now charged) |

- Conclusion shift: PD energy per token rises 1.5–2.1× and the static share is now non-zero without an explicit idle_W. The mode ranking is unchanged: pd < coloc_chunked < coloc_prefill_first. The clock term adds the same card-second cost to every mode, so the ~1–2 % gaps between modes shrink in relative terms.
- Performance, queueing and SLO results are unchanged (energy only).

### Deferred to 0.231
- **Chunked-prefill engine profiles.** The PD colocated `chunk_tokens` 512 「假设」 is unchanged. Planned design:
  - `vllm`: the budget `max_num_batched_tokens` 2048 includes the running decodes and is filled decode-first, so the chunk at batch k is 2048 − k. This needs per-batch chunk plans.
  - `sglang`: `chunked_prefill_size` 8192 (2048 below 25 GB). Mixed chunk is off by default, so a chunk iteration pauses the decodes; this is a non-fused iteration, i.e. a new mode rather than a parameter.
  - Deferred because 13 test files and the ext-validation V4 modes use `coloc_chunked`, so a default change needs its own re-baseline.
- Per-pool utilisation for the clock gating in heterogeneous PD; per-pool DRAM kind.

## [0.229.0]

### Added
- **Clock gating of idle MAC arrays** (design lever `Chip.clk_ungated_share` = g, in [0, 1]). The 0.228 per-cycle term becomes P_clk = MACs × f × `pJ_clk_mac_cycle` × (g + (1 − g) · u), where u is MAC utilisation over the window: useful MAC slot-cycles ÷ clocked slot-cycles.
  - The energy report adds `clock_gating` {ungated_share, mac_util, factor}.
  - UI: chip input "时钟未门控份额"; the energy row shows the utilisation and the gating when g < 1.
  - Sweep path `chip.clk_ungated_share`.
  - The power-cap re-evaluation keeps the lever. J/unit × units/s/card = W/card still holds (tested, fuzzed).
- **Sources for g:**
  - Eyeriss v1 (JSSC 2017 Fig. 16 / Table V, 65 nm; post-P&R on the measured AlexNet workloads; zero-ifmap cycles and unused PEs clock-gated) 「推导」: clock network 32.9 % of 332 mW (CONV1, 0 % zeros) = 109.2 mW, against 45.0 % of 236 mW (CONV5, 77.6 % zeros) = 106.2 mW. Clock power is almost flat (−2.7 %) while multiplier/adder power falls ×0.24. Solving 106.2 / 109.2 = g + (1 − g) · 0.24 gives **g ≈ 0.96**.
  - Eyeriss v2 (JETCAS 2019 Fig. 22) 「二手」: the clock-network share rises from about 21 % on a high-utilisation layer to about 55 % on a bandwidth-limited layer (sparse FC8). Clock power stays when the PEs stall.
  - Bose/Martonosi/Brooks (HC17 tutorial; POWER5-like, pre-silicon, fine-grained latch gating) 「推导」: maximum clock-power saving is 60 % at 20 % pipeline utilisation, so 0.4 = g + (1 − g) · 0.2 gives **g ≈ 0.25**. This is the aggressive-gating reference.

### Decision (default unchanged, g = 1.0; no re-baseline)
- The `tpuv4i_7nm` value of 0.1009 pJ is H100 power with a context and no work, i.e. at u ≈ 0. That is already the stalled floor of a GPU with its own gating; applying g < 1 on top would push stalled clock power below the measured floor.
- The one accelerator measurement with gating active (Eyeriss v1) shows clock power flat under gating (g ≈ 0.96).
- So the sources do not support lowering the default. Gating is shown as a lever: g = 0.25 means the design's clock tree is finely gated, a CPU-class design point not shown for NPUs.
- All 0.228 numbers are reproduced exactly at g = 1. Performance fingerprints are unchanged; the field is omit-default, so scenario hashes are stable.

### Lever results (1P, tpuv4i_7nm + idle 20 W; g 1.0 → 0.25)
- qwen3-8b decode b64 (u = 0.066): clock 52.9 → 15.8 W/card; 188.3 → 151.3 W; 47.8 → 38.4 mJ/token. Prefill b8 (u = 0.985): unchanged (311.1 → 310.5 W).
- qwen3-32b decode b128 (u = 0.145): 206.0 → 172.1 W.

| Cap throughput factor, dfs / V–f | 0.9 P | 0.7 P | 0.5 P |
|---|---|---|---|
| qwen3-8b decode b64, g 1 → 0.25 | 1.000/1.000 → 0.941/1.000 | 0.816/0.996 → 0.696/0.801 | 0.541/0.662 → 0.451/0.519 |
| qwen3-32b decode b128, g 1 → 0.25 | 1.000/1.000 → 0.932/1.000 | 0.794/1.000 → 0.694/0.876 | 0.531/0.698 → 0.456/0.576 |

- With gating, less of the clock power is frequency-proportional at a stalled point, so clock-only DFS sheds less: the 0.228 "decode sheds 10 % for free" result needs ungated clocks. Prefill is unchanged.
- RWKV-7 what-if: base 0.651 → 0.487 J/token at B1. `dram_bw_x2` d_J −6.2 % → −1.3 %; `two_die` −4.7 % → +0.1 %; `state_sram` −32.8 % → −29.6 % (ordering unchanged).
- Levers under a cap:
  - qwen3-8b at 120 W: the base still binds (134 W, dfs ×0.947); `works_capped` unchanged.
  - qwen3-32b at 150 W: the base no longer binds (146 W). `dram_bw_x2` gain ×1.03 / ×1.25 → ×1.14 / ×1.31, and it returns to `works_capped`.

### Deferred
- PD report energy (`pdqueue._energy`): clock term, SRAM leakage, HBM static.
- Chunked-prefill interference and re-sourcing the PD colocated 512-token chunk. Sources: vLLM `max_num_batched_tokens` 2048, decode-first, decodes counted in the budget; SGLang `chunked_prefill_size` 8192 (2048 below 25 GB), no mixing by default.
- Gating of the vector lanes and SRAM ports (u is MAC utilisation only).

## [0.228.0]

### Added (default energy change in the `tpuv4i_7nm` preset)
- **Per-cycle dynamic energy** `EnergyTable.pJ_clk_mac_cycle`: energy per MAC slot per core cycle, for the clock tree, always-clocked pipeline registers and control. It is paid on every cycle the card is clocked, busy or stalled: P_clk = MACs × f × pJ.
  - Gap it closes: before 0.228 every dynamic term was per action. With a memory-set op rate and fixed V, no modelled power fell with the clock, so under clock-only DFS decode needed s ≈ 0.36 to shed 10 %.
  - Under DVFS it is on the core rail: × V(f)², and f via the re-evaluation (`powercap.CORE_ACTIONS`). It is dynamic, not static: not in `static_share`, and duty cycling (clock gated in the off-period) does not save it, so dfs ≥ duty.
  - Energy report: `J_by_action.clock`, `counts_per_unit.clock` (MAC-slot·cycles per unit), `clock_W_card`. UI: input "pJ / MAC槽·周期" and an energy-table row with W/card. J/unit × units/s/card = W/card still holds (tested, fuzzed).
  - What-if `lever_power_cap`: dfs is now re-evaluated (`dvfs_cap`) when the term is set (it was the analytic duty ratio, exact only without the term); `s_dfs` added.
- **Preset value 0.1009 pJ** 「推导/二手」: H100 SXM idle power with a CUDA context (P0 clocks, no work) 121.7 W − without one 71.8 W = 49.9 W (the same measurement as `idle_frac_tdp`), ÷ 270,336 bf16 MAC slots (132 SM × 2048) ÷ 1.83 GHz (989.4 TFLOPS dense). Upper-side proxy (the Δ also holds the HBM P-state step and a GPU's non-tensor logic).
  - Cross-checks, not fitted: at full MAC load it is ≈ 19 % of MAC + clock power. That falls between a clock-gated 65 nm DNN accelerator's clocking at 5.5 % of active power (NSF PAR 10300339) and ungated CPUs, where clock distribution is 28 % and latches 46 % (Bose/Martonosi/Brooks, HC17 tutorial), or ≈ ½ of an Intel CPU (Brooks et al., IEEE Micro 2000). AccelWattch (MICRO'21) shows GV100 NANOSLEEP power rising with f (Fig. 2): clocked-idle power exists on GPUs. Its constant power, 32.5 W at f → 0, is static and not this term.
  - Not in `horowitz_45nm` (no 45 nm source).

### Changed (re-baselined, old → new; 1P, tpuv4i_7nm + idle 20 W unless noted)
- Clock power on 1P: 52.9 W/card at 1.0 f. This is 28 % of decode b64 J/token and 17 % of prefill b8.
- qwen3-8b decode b64: 135.5 → 188.3 W, 34.4 → 47.8 mJ/token. qwen3-8b prefill b8: 258 → 311 W. qwen3-32b decode b128: 153 → 206 W.
- RWKV-7 7.2B what-if (no idle; 0.211/0.217): base B1 0.432 → 0.651 J/token. state_sram at B314 4.46 → 5.59 mJ against base 6.50 → 8.32 mJ, so the ordering holds. **`dram_bw_x2` d_J +0.7 % → −6.3 %**: a faster step now saves clock energy per token. two_die +1.9 % → −4.7 %.
- Performance fingerprints unchanged (energy only).

**Power cap (0.227) re-checked: throughput factor dfs / V–f, old → new**

| Case | Cap 0.9 P | Cap 0.7 P | Cap 0.5 P |
|---|---|---|---|
| qwen3-8b decode b64 (DRAM) | 0.883 (s 0.36) / 0.976 → **1.000 (s 0.64) / 1.000** | 0.648 / 0.717 → 0.816 / 0.996 | 0.413 / 0.457 → 0.541 / 0.662 |
| qwen3-8b prefill b8 (MAC) | 0.892 / 0.962 → 0.893 / 0.962 | 0.675 / 0.874 → 0.679 / 0.877 | 0.458 / 0.765 → 0.465 / 0.771 |
| qwen3-32b decode b128 (DRAM) | 0.885 / 1.000 → **1.000 / 1.000** | 0.655 / 0.788 → 0.794 / 1.000 | 0.425 / 0.511 → 0.531 / 0.698 |

- Conclusion shift: memory-bound decode now sheds 10 % power by clock alone at ≈ 0.6 f with no throughput loss (0.227: s ≈ 0.36 and −12 %). Compute-bound prefill is unchanged: clock energy per unit is fixed there, so dfs ≈ duty.
- What-if under a cap:
  - qwen3-8b b64, cap 120 W: the base now binds itself (171 W; dfs ×0.852, V–f ×1.0). `dram_bw_x2` V–f gain ×1.07 → ×1.015; hbm_12h/16h dfs ×1.04/1.03 → ×1.05/1.045. `works_capped` unchanged (kv_fp8, ctx_half).
  - qwen3-32b b64, cap 150 W: the base now binds (181 W). **`dram_bw_x2` gain ×1.22 / ×1.35 → ×1.03 / ×1.25 and it leaves `works_capped`**; the HBM-height levers stay.

### Deferred
- Chunked-prefill prefill/decode interference in the main evaluation. The PD report already has `coloc_chunked` with `pd.chunk_tokens` 512 「假设」. Sources were collected for the next version: vLLM default `max_num_batched_tokens` 2048, with decode tokens counted in the budget and decode first (docs v0.7.3/v0.8.3; stable docs recommend > 8192 for throughput); SGLang `chunked_prefill_size` 8192 (2048 below 25 GB), `max_prefill_tokens` 16384, mixed chunk off by default.
- Per-cycle term in the PD report's energy (`pdqueue._energy`, which also lacks SRAM leakage and HBM static).
- An NPU-specific clock-power measurement.
- Clock gating of idle MAC arrays (would lower the stalled share).

## [0.227.0]

### Added
- **Power-capped operation by clock and voltage scaling** (`core/powercap.py`, opt-in). There is still no default cap; the TDP is the user's decision (AGENTS §73).
  - Each candidate point is the step re-evaluated at `chip.freq_ghz × s`. Only clocked work stretches (MAC, vector, SFU, SRAM port). DRAM, links and host keep their own clocks.
  - Core-rail action energies (mac, vec, sram, slc) scale with V(f)². DRAM, PHY, D2D, link, network and static power are held. Holding static power is conservative.
  - Bisection finds the highest clock that fits the cap. The lowest clock is 0.15 of nominal (A100 and H100 settable ranges, arXiv 2607.00819 Table 2, 「来源」). Below that, the 0.189 duty cycling applies on top.
  - Curves:
    - `dfs` (headline): clock only at nominal voltage. Its throughput equals the 0.189 duty-cycle model by construction; it adds the clock and the bound at the capped point.
    - `rx5700xt_n7` 「二手」: AMD RX 5700 XT default V–f curve (800 MHz 704 mV, 1409 MHz 819 mV, 2019 MHz 1185 mV; TSMC N7).
  - The energy report's `power_cap.dvfs` carries both curves. The CLI prints a line per curve. The UI energy panel shows clock, voltage, throughput and bound.
- **Per-lever cap binding in the balance what-if.** `/api/sensitivity` accepts `power_cap_W`; the UI sends the energy TDP when "按 TDP 封顶吞吐" is on.
  - The base row and every lever row carry `power_cap`: binds, W/card, dfs and V–f throughput factors, and the gain over the base under the same cap at the same batch.
  - New lists `works_capped` and `binds`.

### Fixed
- The 0.189 power cap counted SRAM leakage (0.212) and HBM standby/refresh (0.216) as dynamic energy that shrinks with the slowdown. They are now static.
  - Example: qwen3-8b decode b64, 1P, tpuv4i_7nm preset without idle, cap = 0.7 × P. Throughput ×0.700 → ×0.692 (3.09 W static).

### Unchanged
- Default fingerprints are identical, because the cap is opt-in.

## [0.226.0]

### Changed (default, re-baselined)
- `serving.spec_tree_accept_model` default changed from `indep` to **`chain`**.
  - `chain` is the guaranteed lower bound: the tree contains the same-depth top-1 chain.
  - `indep` overestimates the published EAGLE-1 chain/tree pairs by +49…+63 %, so it no longer sets the headline.
- Tree configurations now report TPOT and τ at the chain bound. Old→new at 1P, ctx 4096, a = 0.7, k3 / topk 4 / N16:

  | Model | Batch 1 TPOT (ms) | Batch 8 TPOT (ms) | Batch 32 TPOT (ms) |
  |---|---|---|---|
  | qwen3.8-27b | 4.946 → 7.716 | 6.729 → 10.498 | 14.621 → 22.809 |
  | qwen3-32b + qwen3-0.6b draft | 5.511 → 8.597 | — | 20.616 → 32.163 |
  | glm-5.3-flash | — | — | 34.282 → 53.482 |

- Default fingerprints are unchanged because no default preset or catalog scenario uses a tree (`spec_topk` > 1).

### Added
- Third acceptance model `rank`.
  - Source: Sequoia (arXiv 2402.12374) positional acceptance vector, App. E. Its first entry is p₁ = 0.7732.
  - Scaling 「推导」: rank j gets P_j · min(a/P₁, (1−a)/(1−P₁)).
  - The tree follows SGLang-style construction: topk expansion of the topk best leaves per step, then the N−1 best by score.
  - τ = 1 + Σ f(v) (Sequoia Prop. 3.4), held within [chain, indep].
  - Error against the EAGLE-1 pairs: **+0.8…+3.6 %**, not fitted. It is not the default because it rests on one vector.
  - CLI `--spec-tree-accept-model chain|rank|indep`.
- Every tree evaluation reports the band: `spec.tau_{chain,rank,indep}`, `spec.tpot_ms_{chain,rank,indep}`, and the check against the published pairs for all three models. The warning names the headline model.
- UI: new KPI card `#kpi-tree-band`. It shows the headline model, the TPOT band, τ for each model, and the upper-bound TPOT labelled as a secondary value.
- What-if row and `spec_tree_compare`:
  - The tree side now covers chain, rank and indep.
  - `headline` and `crossover` follow the scenario's model (default chain).
  - Per-model crossovers are reported.
  - The lever row itself is evaluated, and therefore ranked, at the headline model.

## [0.225.0]

### Added
- Published tree-drafting acceptance references (`accel_dse/data/spec_tree_refs.json`): six EAGLE-1 matched
  chain/tree pairs (arXiv 2401.15077v3 Table 5, MT-bench, T=0, tree `mc_sim_7b_63`: depth 5, topk 4, 26 tokens)
  and four unmatched points (EAGLE-2 Tables 1–2; SGLang PR #4247 EAGLE-3). `core/spec_refs.tree_check` reports the
  error of the acceptance model against them. Nothing is fitted.
- `serving.spec_tree_accept_model`: `indep` (default; the 0.224 「假设」 i.i.d.-sibling model, now labelled as the
  upper bound, +49…+63 % vs published tree τ) | `chain` (lower bound: the tree's top-1 path is the same-depth chain,
  −17…−19 %). CLI `--spec-tree-accept-model`.
- `Result.spec` / eval JSON `spec`: τ, the [chain, indep] band and the reference check; a warning on every tree
  evaluation states the band and the error.
- What-if row `spec_tree_tree` / `spec_tree_chain` (chain ↔ topk-4 tree of the same depth) with `spec_tree_compare`:
  chain, tree-indep and tree-chain TPOT / τ / memory at batch 1, 8 and 32, plus the batch intervals where the winner
  flips under each bound.
- UI: tree topk, tree tokens, tree acceptance model and state-tracking interval fields, with range checks and a client
  warning line. The what-if table shows the chain-vs-tree comparison (`#spec-tree-cmp`).

### Fixed
- Tree τ is capped at depth min(k, N − 1). 0.224 counted k depths even when the tree had fewer than k + 1 tokens.

### Unchanged
- Default fingerprints. `indep` remains the default, so no KPI changes unless N − 1 < k.

## [0.224.0]
- ReplaySSM, sourced from SGLang main (`gdn_replayssm_spec_decode.py`, `kda_replayssm_spec_decode.py`, gdn_backend,
  memory_pool):
  - GDN uses a circular ring. The history read per verify and the fold rate now follow SGLang's cursor rule
    (`_advance_gdn_spec_cursors_kernel`: early flush when write_pos + 2·q > L; on a flush step an exact sequential fold
    of write_pos records), evaluated at the expected acceptance E (`ir.replay_cycle`, 推导). This replaces the 0.223 假设
    h̄ = (L−q)/2.
  - Verify history cost is now output-only reconstruction, ∝ history·q·(dk+dv) (推导), instead of full state updates.
  - KDA: fold-every-commit (`replayssm_spec_fold`). The recurrent verify does not write state and stores the raw
    window; each commit replays the E accepted records. Ring entry is 4V+8K+4 B per head.
  - Ring length is a power of two ≥ max(16, 2q). SGLang asserts the power of two and its config enforces
    L ≥ 2·max_spec_len for GDN too, so the 0.223 假设 is now 来源. For q ≤ 8 nothing changes; for q = 9 L is 32
    (was 18).
  - Lightning / conv layers have no ReplaySSM path. They fall back to snapshot, with a warning.
- Radix-track lever `serving.mamba_track_interval` (CLI `--mamba-track-interval`; SGLang default 256 when tracking
  runs, 来源; 0 = off, the default). It adds a state copy at every boundary:
  - snapshot: scatter to the track slot;
  - GDN replay: materialize history up to the boundary;
  - KDA: write from the per-step fold.
  Whether a deployment tracks is 假设.
- Tree drafting (EAGLE-style): `serving.spec_topk` and `serving.spec_draft_tokens` (CLI `--spec-topk`,
  `--spec-draft-tokens`).
  - Verify covers N tree tokens. The tree attention mask is charged as ctx+N keys per query.
  - Hybrid targets use the snapshot path with N intermediate states. SGLang ReplaySSM is linear-chain only, so
    replay falls back to snapshot, with a warning.
  - MTP draft steps 2..k run topk rows per sequence; an independent draft runs them at batch·topk.
  - Acceptance 假设: per-depth a_t = 1−(1−a)^w with w = min(topk, (N−1)/k).
  - Bound check: N−1 ≤ topk + (k−1)·topk².
  - DSpark ignores topk, with a warning.
- Tests:
  - `tests/test_core_224.py`: cursor cycle, KDA fold, track, tree, and a seed-224 fuzzer over every spec mode with
    the J/token × tok/s = W invariant.
  - `test_core_223` ring_len(9): 18 → 32.
- KPI deltas: defaults unchanged (fingerprints and ext_validate identical). Replay-mode TPOT, 1P ctx 4096,
  0.223 → 0.224:

  | Case | Batch 1 | Batch 32 |
  |---|---|---|
  | ReaDy k=7 | 3.960 → 3.945 ms | 6.273 → 5.817 ms |
  | qwen3.8-27b k=1 | 9.876 → 9.810 ms | 14.112 → 12.585 ms |
  | qwen3.8-27b k=3 | 7.450 → 7.412 ms | 10.425 → 9.550 ms |

  Snapshot results and all memory figures are unchanged.

## [0.223.0]
- Independent draft energy. `Result.draft["acts"]` holds k × the draft pass's action counts, and `energy._counts`
  adds them. J/token and W/card now include the draft.
  - The invariant J/token × tok/s per card = W per card still holds; there is a new test,
    `test_core_217::test_main_eval_invariant_with_draft`.
  - qwen3-32b on 1P, batch 8, k=4, a=0.7:

    | Draft | J/token | W/card |
    |---|---|---|
    | qwen3-0.6b | 0.1314 → 0.1579 | 105.5 → 126.8 |
    | qwen3-8b | 0.1339 → 0.2479 | 64.8 → 119.9 |

- Speculative rollback strategy, `serving.spec_rollback` = `snapshot` (default, unchanged) | `replay`.
  - `replay` is SGLang ReplaySSM (`--enable-linear-replayssm-spec`; 来源 gdn_backend `_replayssm_target_verify`,
    memory_pool rings).
  - It has no q × intermediate-state scratch. It keeps a compact (d, k, g) ring of L = max(16, 2q) entries.
  - Per step, per sequence, per linear layer it costs:
    - one checkpoint read;
    - a replay of h̄ = (L−q)/2 ring entries (假设);
    - q ring writes;
    - a fold every (L−q)/E steps, where E is the expected number of accepted tokens.
  - Replay compute is (h̄ + E) extra recurrent token updates.
- New what-if row `spec_rollback_<alt>` on hybrid targets with speculation on. It carries `spec_rollback_compare`
  (both modes at batch 1 and batch 32), and the UI shows it.
- UI:
  - Draft-model dropdown (catalog LLMs). A vocab mismatch is flagged next to it; `/api/models` now carries `vocab`.
  - Spec rollback select.
  - The spec_k sweep is now offered whenever a draft model is set.
  - ui_smoke covers both new controls.
- Fix (0.222): static dispatch charged the independent draft twice. The ×2 now applies only to the target's own
  MTP / DSpark draft. A draft model's k passes carry their own dispatch, so the total is 1 + k programs.
- Tests: `tests/test_core_223.py` (9 tests, including a seed-223 fuzzer over model / draft / k / a / batch / mode),
  plus the static-dispatch ×2 unit test.
- KPI deltas: defaults unchanged. Preset fingerprints and ext_validate are identical. Only scenarios with a draft
  model set gain draft energy.

## [0.222.0]
- Spec-decode verify-step mechanism audit (MODEL §169). Three items were missing and sourced, so they are now modelled:
  - Hybrid-layer state snapshot (`ir.SPEC_STATE_SNAPSHOT`, 来源 SGLang memory_pool `intermediate_ssm_state_cache`
    + `update_mamba_state_after_mtp_verify`). A verify step with q = 1+k tokens now moves (q+3)× the recurrent
    state per sequence (was 2×). memplan reserves q × state per sequence as scratch (`stage_storage(spec_q=)`).
  - DSpark Markov head is sequential (`ir.DSPARK_SEQ_MARKOV`, 来源 SGLang `draft_markov_block`): B dependent
    head GEMMs plus B samples, replacing one batched GEMM.
  - Static dispatch: when a spec step is on the head stage, it is charged as two programs (draft + verify;
    来源 SGLang separate draft graph runner; NPU compiler doing the same is 假设).
  - Already correct, no change: k+1-token attention against the full KV (causal block), and the GEMM M-tile at
    M = k+1.
- ReaDy B200 like-for-like, before → after:

  | Metric | Before | After | Paper |
  |---|---|---|---|
  | Step ratio vs AR | 1.126 | 1.165 | 1.433 |
  | Speedup | 4.02× | 3.89× | 3.16× |

  The remainder is a residual (engine host/scheduling, not sourced). Nothing was fitted to the paper.
- New: generic independent draft model, `serving.draft_model` (CLI `--draft-model`).
  - Own weights and KV live on the target's head-stage cards, at the same TP/DP (假设).
  - Step = target verify + k × draft decode pass.
  - Tokens per step = (1−a^(k+1))/(1−a); acceptance is a 假设 lever.
  - Warnings for vocab mismatch, and for a target MTP that is stored but not run.
  - Limitation: draft energy is not yet in J/token.
- `tests/test_core_222.py`: mechanism tests plus a seed-222 fuzzer over target/draft/k/a/batch/tp.
- KPI deltas (1P, ctx 4096). Non-speculative presets and ext_validate are unchanged.

  | Case | TPOT before → after | DRAM need before → after |
  |---|---|---|
  | qwen3.8-27b k=1, b32 | 13.10 → 15.80 ms | 65.1 → 74.4 GiB |
  | ReaDy k=7, b32 | 5.31 → 8.39 ms | 69.8 → 107.2 GiB |
  | ReaDy k=7, b1 | 3.88 → 4.02 ms | |
  | DeepSeek-V4.1-flash k=7, b1 | 3.850 → 3.857 ms | |
  | Muse DSpark k=7, b1 | 18.89 → 19.01 ms | |

## [0.221.0]
- Default memory realism: every default now resolves to a shipping, sourced part, or is an explicit what-if.
  - LPDDR5 discrete x32 default capacity 6 GB → 8 GB per package (6 GB has had no product since 0.220; 8 GB = Samsung
    K3LKCKC0BM-MGCP, 来源). KPI: bandwidth unchanged, so TPOT and throughput are unchanged; capacity 4 × x32 24 → 32 GB
    (qwen3-8b, 100T, batch 8, ctx 8192: fits False → True; TPOT 345.149 ms and 23.18 tok/s/card unchanged).
  - H100-like preset memory `hbm3_5s_8h16g_5234` was tagged unspecified / no product because 5234 was not a catalog grade.
    It is the H100 SXM memory, so the HBM3 catalog gains rate 5234 (疑似 JEDEC · shipping; 来源/推导: H100 product page
    3.35 TB/s, Hopper whitepaper 5 stacks / 5120-bit / 2619 MHz DDR). No number changes; preset KPIs and hashes unchanged.
  - `mem_catalog.WHATIF_DEFAULTS` flags the two defaults that are what-ifs by construction (LPDDR6 x48, LPDDR6 LPCAMM2).
  - New test: no preset, legacy preset, System / scenario / UI default, or catalog type/form/width default may resolve to
    product status `none` unless flagged.
- Speculative decoding: new `qwen3.8-27b-ready` = Qwen3.8-27B target + StartLux ReaDy-Qwen3.8-27B DSpark draft (separate
  repo; both configs public 2026-10-17). The draft shares the target's embedding / lm_head (1.862 B params, matches the
  header). The recurrent-state reader is modelled at draft layer 4: fp32 state of target layers 30 / 62, gather + buffer +
  read, rank-8 query and value map, 864 → 128 → 5120. Full-attention draft layers now charge their K/V append.
  - The published acceptance length τ 4.5285 (arXiv 2610.20503, k 7, temperature 0) implies i.i.d. a = 0.828 (推导).
    It is exposed as `model.spec_published`; the `spec_accept` default stays 0.7 「假设」.
  - Modelled on 1P, batch 1: AR 64.1 → 258.1 tok/s (4.03×; paper B200 3.16×, residual = engine overhead not modelled).
  - Generic independent small-LM drafts are still not modelled (design in MODEL §168, deferred).
- Adversarial fuzzer on the separate-draft path (seed 221).
- Muse-Spark DSpark, qwen3.8-27b and every preset produce unchanged results.

## [0.220.0]
- LPDDR per-entry sourcing is complete: no entry is left on the research-doc citation.
  - The 24 remaining entries now cite Samsung product pages (K3LK4K40CM-JHCP, K3LKCKC0BM-MGCP, K3KL3L30EM-JUCV, K3KL9L90DM-MGCU), Micron datasheets (LPDDR5 MT62F512M64D4 / MT62F1G64D8; LPDDR5X Y5BP MT62F768M64D4 / MT62F1536M64D8) and the Micron LPDDR5X page (1γ 6/12/24 GB, 16 GB).
  - Five entries are 二手: four rest on a distributor list (glochip.com), and one, LPDDR5X x32 32 GB, is inferred (推导) from SOCAMM2 192 GB.
  - Entries with no product found are tagged 假设.
- Corrected availability tags (old → new):
  - LPDDR5X SOCAMM2 192 GB: sampling → shipping. SK hynix newsroom 2026-04-20 reports mass production; this reverts the 0.219 change, which relied on Micron's 2025-10-22 sampling release only.
  - SOCAMM2 48 / 64 / 96 / 128 GB: shipping → JEDEC-allowed, no product. No SOCAMM2 product source was found; Samsung lists "up to 256 GB", SK hynix 192 GB, Micron 192 / 256 GB.
  - LPDDR5 x32 3 GB, x32 6 GB, x64 3 GB: shipping → JEDEC-allowed, no product. No Samsung or Micron part was found.
  - Default capacities are unchanged: LPDDR5 x32 6 GB (now tagged no-product), x64 12 GB; SOCAMM2 192 GB.
- Other source updates:
  - CXMT LPDDR6 12.8 Gb/s and 16 GB now cite CXMT's own newsroom post (2026-09-07: 16 Gb dies, 1295-ball, 10667 with the SoC) instead of the TrendForce summary.
  - LPCAMM2 9600 stays 来源/假设: Micron's brief names a 2026 grade, and Samsung's LPCAMM2 page lists 8533 and up to 64 GB.
- Model catalog check (2026-10-17): nothing added.
  - RWKV: the newest release is RWKV7-G1k-20260930, already in the catalog.
  - StartLux: v1.1 is already in the catalog. The new ReaDy-Qwen3.8-27B is a speculative-decoding draft model, not a decision model, so it is not added (it would need a spec-decode path).
  - Microsoft-Decision-1 and Jev: no official public release.
- New adversarial fuzzer: what-if lever monotonicity (tests/test_core_220.py).

## [0.219.0]
- HBM: the two remaining 二手 entries are now 来源.
  - HBM3E 9.8 Gb/s: Samsung newsroom 2023-10-20 (Memory Tech Day), Shinebolt "9.8 Gbps per pin … samples … shipping". Rate tag corrected from custom·announced to custom·sampling.
  - HBM3 12-Hi 24 GB: SK hynix newsroom 2023-04-20. HBM3 8-Hi/12-Hi mass production: Samsung 2023-10-20.
- LPDDR: per-entry `LPDDR_SOURCES` / `lpddr_source()` replace the blanket per-type 二手.
  - Of 66 entries, 28 now cite JEDEC press releases (JESD209-5 6400; JESD209-5B 8533; JESD209-6 10.667 / 14.4 Gb/s) or vendor releases (Samsung, SK hynix, Micron, CXMT).
  - 14 configurations with no product are tagged 假设.
  - 24 keep the research-doc citation, labelled 二手 per entry (deferred).
- Corrections:
  - SOCAMM2 192 GB: jedec·shipping → jedec·sampling. Micron 2025-10-22 says "customer sampling … up to 192 GB"; no volume source was found. The default cap stays 192.
- Residual review: no new sourced mechanism was applied; see MODEL §166.

## [0.218.0]
- Samsung HBM4 24 GB: HBM4 densities (24, 32) → (16, 24, 32). New tags: (12, 16) custom·shipping, from the Samsung newsroom 2026-02-12 ("12-layer … 24 GB to 36 GB"); the 16 Gb die density is 推导 (Samsung doesn't state it), and it is outside the JESD270-4 PR density list (24/32 Gb). (8, 16) and (16, 16) are unspecified·none. Default density is still 24, so no default changed.
- SK hynix / Micron HBM4 re-checked against primary sources:
  - SK hynix: 12H 36 GB samples 2025-03; mass-production ready 2025-09-12 at > 10 Gb/s.
  - Micron: 12H 36 GB volume 2026-03-16 at > 11 Gb/s; 16H 48 GB samples.
  - Samsung: 11.7 Gb/s shipping, 13 Gb/s capable.
  - No tag changed.
- Provenance audit:
  - `HBM_CAP_SOURCES` values now start with a provenance tag and cover every product-claiming cap tag (HBM3 8/12H, HBM3E 8H, HBM4E added).
  - New `HBM_RATE_SOURCES` covers every catalog pin rate (HBM4E rates and 9.8 Gb/s HBM3E are 假设 / 二手).
  - LPDDR types get `src`; D2D standards get `src`.
  - New `hardware.CHIP_SOURCES` / `LINK_SOURCES`: chip geometries are 假设 design points; H100-like is 推导/来源.
  - Test: every catalog entry carries a valid tag and a note.
- What-if: the base row states its priced batch (`cost_batch`, `cost_basis` = "configured"). Lever rows are clamped so the priced batch never exceeds their own batch limit (no case violated it before the clamp). The UI shows the batch and its basis (临界 / 上限 / 配置) in the J/token and W columns.
- Deferred: the MI300X max-load and TRT-LLM 0.8 prefill residuals.

## [0.217.0]
- Fix: a what-if lever that cannot reach balance was costed at the configured batch (default 1). qwen3-8b showed 0.48 J/token at batch 1 next to its throughput at the batch limit. The energy accounting itself was consistent (J/token × tok/s/card = W/card held at batch 1). Unreachable levers are now costed at their own batch limit (`cost_basis` = "batch_limit"), and the base at the same batch capped at its own limit (`base_cost_batch`). New field `tok_s_card_at_cost_batch`.
- Reachable rows are unchanged: the RWKV state SRAM / DRAM×2 / two-die energy conclusions of 0.211–0.216 stand.
- tests/test_core_217.py: J/token × tok/s/card = W/card on the main eval (fuzzed models × chips × batch) and on every what-if row.

## [0.216.0]
- Sources re-verified for HBM capacity and status (JEDEC HBM3/HBM4 press releases; SK hynix, Samsung and Micron releases), recorded in `mem_catalog.HBM_CAP_SOURCES`. No cap tag changed. MODEL §162: the unsupported Micron HBM3E 12H "volume production" claim is corrected to "production-capable, in qualification" (来源).
- HBM standby + refresh static power: EnergyTable `mW_hbm_die_standby` / `mW_hbm_Gb_refresh` (tpuv4i_7nm preset: 48 mW/die, 1.95 mW/Gb, DRAMsim3 HBM2 IDD proxy, 二手/推导). Charged per stack × die only when no idle figure is set (no double counting, as with the 0.212 SRAM leakage).
- What-if: `hbm_W` / `d_hbm_W` per lever, shown in the power column. qwen3-8b 1P fp8: 12H +1.52 W, 16H +3.03 W, 8 stacks +3.03 W.
- tests/test_core_216.py.

## [0.215.0]
- balance_whatif: capacity lever `hbm_{H}h{D}g` (taller HBM stacks, same stack count / rate → same bandwidth), offered only when the base is kv_capacity-bound and no mem_cap override is set; the 2 largest catalog height×density options above the current one, with the catalog availability tag.
- Cost: JEDEC fixes the package height (720 µm HBM3/3E, 775 µm HBM4), so footprint and shoreline are unchanged (二手); DRAM pJ/bit is unchanged (假设); refresh/standby of the extra dies is not modelled.
- qwen3-8b 1P ctx 4096: 12H 144 GB batch limit 141→227 (fp8 KV 283→453); 16H 192 GB 247/479, now TPOT-SLO bound. The KV asymptote (28.8 / 57.0 FLOP/B) does not depend on capacity, so it never reaches chip balance (318).
- tests/test_core_215.py; test_core_208 key list extended.

## [0.214.0]
- **What-if lever "two-die package" (`two_die`, B200 approach)** on the existing multi-card + D2D model, with no new model. It has the same totals as DRAM ×2:
  - Each die has half the engines, SRAM and state SRAM.
  - Each die keeps the base memory (half of the 2× stacks), at half the capacity.
  - TP doubles across the dies, which sets the cross-die traffic: the existing TP collectives on the D2D tier.
  - D2D uses the scenario's grade and unit count: default UCIe-A x64 48 GT/s × 4, 0.25 pJ/bit. NV-HBI (B200, 10 TB/s) stays out of the selector as vendor-specific.
  - Costs come from the existing models: D2D PHY area per die (`phy_area`, UCIe x64 module 0.41 mm²), D2D energy (`pJ_bit_d2d`) and collective latency.
- **Shoreline:** a die in a multi-die package gives one edge to D2D, so the maximum is 3 sides; the typical budget stays 2 long sides.
- **What-if row fields:** `dies`, `d2d_phy_mm2`, `d2d_J_per_tok`, `coll_lat_ms`. The UI shows them as a per-die line under Δ area.
- **1P results** (n5_derived area including uncore; tpuv4i_7nm energy):

  | Model | Option | Area per die | Shoreline | TPOT at B = 1 | Critical batch (limit) | J/token at that batch |
  |---|---|---|---|---|---|---|
  | RWKV-7 7.2B | 1 die, 4 stacks | 571 mm² | ok (44 / 48 mm) | 4.13 ms | unreachable (state-bound) | — |
  | RWKV-7 7.2B | 1 die, 8 stacks | 654 mm² | tight (88 / 51 mm; 102 mm on 4 sides) | 2.07 ms | 758 (2048) | 5.63 mJ |
  | RWKV-7 7.2B | 2 dies × 4 stacks | 333 mm² | tight (44 / 36.5 mm; 54.8 mm on 3 sides) | 2.13 ms | 716 (2048) | 5.73 mJ |
  | qwen3-8b | 1 die, 4 stacks | 571 mm² | ok | 4.69 ms | unreachable (KV-bound) | — |
  | qwen3-8b | 1 die, 8 stacks | 654 mm² | tight | 2.35 ms | unreachable (limit 141) | — |
  | qwen3-8b | 2 dies × 4 stacks | 333 mm² | tight | 2.38 ms | unreachable (limit 139) | — |

  - The D2D cost is small: PHY 1.6 mm² per die, energy 0.002 mJ/token (< 0.1 %), collective latency 0.03 ms per step (TPOT +1.6–3 %).
  - Shoreline is per die. Splitting the same area raises the total edge only by √2: 2 × 36.5 = 73 mm on the long sides, still short of the 88 mm the 8 stacks need. A two-die package at this area therefore still uses the third side. It reaches two sides only with larger dies; B200's dies are reticle-class.

## [0.213.0]
- **Uncore area in the die proxy.** Default change, justified by mechanism: before this the proxy had no area for control, NoC, routing or host IO, so die and shoreline results were lower bounds.
  - New `Budget` fields:
    - `uncore_frac`: share of the *final* die, so die = (core + PHY + host IO + MC) / (1 − f).
    - `mm2_host_io`: fixed area of the host interface (PCIe x16 class, including its PHY).
    - `mm2_per_unit_mc`: memory controller per DRAM unit (HBM stack / package). No public figure, so no preset.
  - All three density presets now carry:
    - `uncore_frac` = 0.23 「来源/推导」. TPUv1 floor plan (Jouppi et al., ISCA 2017 Fig. 2): buffers 37 %, compute 30 %, I/O 10 %, control 2 %, leaving 21 % unlabelled. I/O is modelled separately, so uncore = 2 % + 21 %. NeuroMeter (MICRO 2021) also leaves ≈ 21 % of TPUv1 unknown. One design's share is used for every node 「假设」.
    - `mm2_host_io` = 6.0 mm² 「二手」. NeuroMeter's TPUv1 die-shot breakdown puts the PCIe interface at 1.8 % of 331 mm². It is not scaled by node because analog shrinks little 「假设」.
  - Not added when the user gives `mm2_fixed`, which already means "everything else". A note says so.
  - Budget panel inputs added for all three fields.
- **Re-evaluated RWKV-7 7.2B on 1P** (n5_derived, old → new):

  | Case | Die area | Shoreline (stacks × 11 mm vs edge on 2 sides / 4 sides) | Fits reticle? |
  |---|---|---|---|
  | 1P base | 434 → 571 mm² | 44 mm vs 41.7 → 47.8 mm: tight → **ok** | Yes |
  | DRAM ×2 | 498 → 654 mm² | 88 mm vs 51.2 / 102.3 mm: **tight** (needs the third and fourth side) | Yes; Δ area +64 → +83.1 mm² |
  | State SRAM 10.1 GiB | 7486 → 9729 mm² | — | No, 11.3× the reticle; Δ area +7052 → +9158 mm² |

  - The reticle flag and the shoreline check use the same die area. A test enforces this.
- Re-baselined: test_core_211 area bounds (state Δ 7052 → 9158; DRAM ×2 Δ 64 → 83.1). Performance fingerprints are unchanged, because the area proxy is not part of evaluation.

## [0.212.0]
- **HBM die-shoreline check** (`budget.shoreline_check`; in `budget_report.shoreline`, the budget panel and every what-if row).
  - Each stack needs ≥ 11 mm of die edge: the HBM4 base die is 11 × 12.8 mm (Samsung HC2026) 「二手」.
  - On a square die (the die limit if given, else the area proxy), 2 long sides are the typical budget. GH100, B200 and MI300X put HBM on the long sides and keep the rest for D2D / IO 「二手」. 4 sides are the geometric limit 「假设」.
  - Status is ok, tight (needs the third and fourth side) or over (needs a multi-die package). Known package maxima: B200 8 stacks on 2 dies, MI300X 8 「来源」. This is a warning, not a budget item.
- **SRAM leakage** (`EnergyTable.uW_per_mib_sram_leak`; the `tpuv4i_7nm` preset uses 230.7 µW/MiB).
  - Source: Aoyagi et al., IEEE JSSC 2025 / VLSI 2025, 3 nm HD 6T macro, 27.5 µW/Mbit at 25 °C 「来源，3 nm 代理」.
  - It is a lower bound: leakage at operating temperature is several times higher.
  - It is a static term `sram_leak`. The dedicated state buffer is always charged. The shared SRAM is charged only when no measured idle figure (idle_W / idle_frac_tdp) is set, because a measured card idle already includes its SRAM leakage, so nothing is counted twice. `static_share` includes it.
- **What-if:** each row adds a shoreline line under Δ area and Δ leakage next to W/card.
- **RWKV-7 7.2B on 1P:**
  - DRAM ×2 (8 HBM3E stacks) needs 88 mm of die edge. The area-proxy die (≈ 498 mm²) has 45 mm on its two long sides and 89 mm on all four, so it is tight. Even a full 858 mm² reticle die gives 59 mm on two sides, enough for 5 stacks. In practice 8 stacks means a multi-die package like B200.
  - The 1P base (4 stacks, 44 mm) is already slightly over two sides of its 434 mm² proxy die. The proxy has no fixed logic area, so the real die is larger.
  - The 10.1 GiB state SRAM adds ≥ 2.39 W of leakage at 25 °C, small next to ≈ 206 W of dynamic power. Its blocker is area (+7052 mm²), not leakage.

## [0.211.0]
- **Area and power of the dedicated state SRAM.**
  - What the existing accounting already covered:
    - Shared SRAM area: `budget.area_estimate` with `mm2_per_mib_sram` from the density presets.
    - SRAM access energy: `pJ_bit_sram` × SRAM-port bytes.
  - What changed:
    - `Chip.state_sram_mib` is now charged at the same macro density, as the `state_sram` area term.
    - The budget's `sram_mib` limit now counts on-die SRAM including the state buffer.
    - Access energy needed no change: state operands already go through the SRAM port whether or not the state is resident. Residency removes the DRAM read+write, so the DRAM energy line drops.
  - Not covered: SRAM leakage. Static power is `idle_W`, a product figure. Leakage is listed as not modelled rather than invented.
- **What-if lever cost columns** (`balance_whatif`; UI "机器平衡 what-if"):
  - Δ die area: `budget.full_area` = SRAM including the state buffer, plus DRAM PHY pins. Densities from `n5_derived` (SRAM 0.68 mm²/MiB 推导; HBM PHY 1/64 mm² per DQ 推导/二手).
  - Dynamic J/token and W/card from `tpuv4i_7nm`, without idle_W. Each lever is costed at its own critical batch, next to the base at the same batch.
  - A flag marks a die that would exceed the 858 mm² reticle (26 × 33 mm 来源).
  - `cost_basis` states which presets were used.
- **RWKV-7 7.2B on 1P** (fp32 state, as released):

  | Lever | Δ die area | J/token at its critical batch | Base chip at that batch |
  |---|---|---|---|
  | State SRAM 10.1 GiB | +7052 mm² (die ≈ 7486 mm², 8.7× the reticle; even the array-only N3E bound of 0.25 mm²/MiB gives ≈ 2590 mm²) | 4.34 mJ at B = 314 | 6.39 mJ |
  | DRAM bandwidth ×2 (HBM3E 4 → 8 stacks, same capacity) | +64 mm² of PHY | 5.63 mJ at B = 758 | 5.63 mJ (no extra bits per token) |
  | State bf16 (what-if only) | 0 mm² | 4.33 mJ at B = 1541 | 5.37 mJ |

  - Dynamic power at those batches: 204 W per card with the state SRAM, 242 W with 2× bandwidth.
  - For reaching balance, 2× DRAM bandwidth is about 110× cheaper in die area and fits on one die; the state buffer does not fit.

## [0.210.0]
- **Pipe-mode per-op KV / state split** (finishes STATE_SPLIT): overlap `items_of` now gives recurrent-state bytes their own `state` key, and `evaluate._kv_state_miss` weights each op by its own stream's miss share. These are the same shares as `memplan.step_dram_bytes`; the old pooled share returns when STATE_SPLIT is off. The step DRAM total is unchanged; only how it is spread across ops in pipe mode moves. Re-baselined: default (pipe) fingerprint 62fd08857bc4c055 → 99a72aef99a2e674; test_core_125 fpx({}) a148b9cd855d95fc → 9982d6b04e230874; test_core_171 Evo2 (Hyena) ttft 0.37786053 → 0.37785506.
- **RWKV-7 7.2B state size, reconciled.** Both earlier numbers are correct but measure different things:
  - 33.0 MiB = 34.6 MB per sequence is the *capacity* (fp32 state; what a buffer must hold).
  - ≈ 69.2 MB per sequence per step is the DRAM *traffic* when the state is not resident: it is read and written every step, so 2 × 34.6 MB.
  - Docs updated: the 0.208 entry, MODEL §155 and §156. The state SRAM what-if label now shows both: "每序列状态容量 33.0 MiB；未驻留时读+写 69.2 MB/步".
- **Rest of the adversarial review of 0.203–0.208** (`tests/test_core_210.py`). No new bugs found.
  - Multi-card PP (pp 2/3 × every dispatch mode × dense, MoE+state hybrid, RWKV): one stage per card; walls keys and shares sum to 1; the walls stage index is valid; stage ms ≤ step ms; exposed ms ≥ 0; no NaN; reachable ⇒ critical batch ≤ limit.
  - Preset memory table: the catalog `default_mem`, `hardware.default_mem` and the API-resolved `mem_id` agree for every preset, and `/api/memory` resolves each one.
  - Cross-route fuzz (eval / sensitivity / fit / sweep), seeded, with a per-request timing guard (20 s for eval, 80 s for multi-evaluation routes).
  - Huge-batch timing guard: a 100k-batch prefill or decode with pp=2 must finish in under 20 s. This guards against a return of the overlap replay bug fixed in 0.209.

## [0.209.0]
- **Recurrent-state on-chip residency as a design variable.**
  - New `Chip.state_sram_mib` (default 0): a dedicated state buffer, separate from the shared SRAM. `memplan.plan(state_buf=)` puts state in the buffer first. The remainder follows STATE_FIRST in the shared SRAM, then hot weights, experts and KV, and any SRAM left over takes more state. Partial residency is handled per byte.
  - `memplan.STATE_SPLIT` (bug fix): KV and state now each miss DRAM by their own resident share. Before this, one pooled miss fraction meant a hybrid model's pinned state was still charged DRAM traffic. `LEGACY_0191` turns it off.
  - The pipe-mode staging buffer no longer subtracts the dedicated buffer (`evaluate._shared_kv`). The per-op weight `f["kv"]` in pipe mode still uses the pooled fraction. That only affects how DRAM time is spread across ops; the step total is exact. 推导.
  - `critical_batch`: AI(B) is non-monotone under partial residency, so when the monotone search finds nothing, it probes the residency knees (k−1, k, k+1) and bisects the rising side. `ai_asymptote` is still the B→∞ (spilled) value; with a finite buffer, AI peaks at the knee.
  - What-if lever `state_sram` (when the saturation involves state): a buffer sized to hold N = the critical batch with unlimited state SRAM. The label reads "状态 SRAM X GiB（容纳 N 序列 × Y MiB）".
  - UI: a "状态 SRAM MiB" chip input. Stage memory now includes `state_sram_MiB`.
  - Answer: RWKV-7 7.2B (fp32 state, as released) on 1P reaches balance with a dedicated state SRAM of ≥ 10352 MiB ≈ 10.1 GiB. That is 315 sequences × 33.0 MiB; the critical batch is 315, below the 1664 limit. With the shared SRAM alone it needs ≈ 10.36 GiB. 推导 from the model's own byte counts; no tuning.
- **Adversarial review of 0.203–0.208 (partial; the full review and broader fuzzers move to 0.210).**
  - Bug: with the saturation unbounded, the UI's balance KPI always said "∞（KV 受限）", even for state, hybrid or weights saturation. It now uses `asymptote_by`.
  - Bug (performance, exact): in overlap `_Sim`, while a shared-expert window was open, its start was part of the state and moved in `shift`, so long tile runs never matched and were replayed tile by tile. A 100k-batch prefill (StartLux, pp=2) took 127 s and now takes 0.34 s, with an identical result. s0 is now absolute.
  - Fuzzers: `tests/test_core_209.py` covers edge inputs (batch 0/1, ctx 1 and 1M, TP=3, PP=3 with a MoE+state hybrid, no SLO, a tiny SLO, prefill labels) × every dispatch mode, plus a seeded random fuzz. It checks NaN, wall shares summing to 1, and reachable ⇒ critical batch ≤ limit. An API↔UI test confirms every balance / walls / what-if key the UI reads exists in the API response.
  - Ad-hoc fuzz: 6 seeds × 60 cases over eval / memory / sensitivity. No NaN and no exceptions; the only inf is `goodput.ttft_ms` for an infeasible prefill, which `clean` turns into null by design.
- Fingerprint re-baseline (STATE_SPLIT; hybrid models only): stage fp b47ec84b00fc0892 → cd25d689b7492f42; default (pipe) fp b4db8aea9e5680b6 → 62fd08857bc4c055; test_core_125 5b3efea75e12245d → dee4a1703e8fab85 and 8220364829b88de3 → a148b9cd855d95fc; test_core_171 Hyena ttft 0.37786546 → 0.37786053. StartLux / Qwen3-Next TPOT moves by −0.06 % to −0.63 %.

## [0.208.0]
- **Machine-balance what-if** (`api.balance_whatif`, part of the `/api/sensitivity` response, rendered under the sensitivity table as "机器平衡 what-if"). It reuses the sensitivity scan's case loop, now factored out as `api._eval_cases` and shared with `sensitivity_report`.
  - Each lever changes one variable:
    - KV fp8 (models whose saturation involves KV);
    - recurrent state bf16 (models with state);
    - ctx ÷ 2;
    - SRAM × 2;
    - DRAM bandwidth × 2 (twice the channels/stacks at the same capacity, via `mem_cap_GB`).
  - Each row shows chip FLOP/B, critical batch, batch limit (with what binds it), saturation intensity and TPOT at the configured batch.
  - A lever "works" when it makes balance reachable, or moves the asymptote or batch limit up, or TPOT down, by more than `WHATIF_GAIN = 10 %` (假设).
  - qwen3-8b on 1P (ctx 4096): KV fp8 (asymptote 28.8 → 57.1, limit 141 → 283) and ctx ÷ 2 work. DRAM × 2 halves TPOT (4.70 → 2.35 ms) but leaves balance unreachable. SRAM × 2 has no effect.
- **Saturation source** `balance.asymptote_by` ∈ kv / state / kv+state / weights / act: the per-sequence DRAM stream that grows most from batch 1 to 65536 (kv+state when both are ≥ 25 %).
  - Fixed-state models (RWKV-7) are labelled `state`. Their asymptote is independent of ctx, and the hint names the state levers (state bytes / SRAM-resident state), not context.
- **RWKV-7 G1k 7.2B** (fp32 state, as the reference kernels keep it):
  - The asymptote is 181 FLOP/B (state read + written every step, ≈ 69.2 MB per sequence per step of read + write traffic = 2 × the 34.6 MB (33.0 MiB) capacity; 0.210 wording). On 1P (318 FLOP/B) balance is **not reachable**; this is correct, because the fixed state is still a per-sequence DRAM stream.
  - It becomes reachable when the chip balance is below the asymptote: 2× DRAM gives B* = 758 ≤ 2048, and bf16 state (asymptote 331) gives B* = 1541 ≤ 2092.
  - New `formats_override` role `state` (what-if).
- **StartLux-Decision 35B-A3B** (hybrid GDN + full attention, MoE): `kv+state`, asymptote 29.8; every one of KV fp8, state bf16 and ctx ÷ 2 helps.
- **Decision readout** (prefill + labels): the critical batch now applies to a batch of decisions, limited by memory capacity and the TTFT SLO (`limited_by: "ttft_slo"`). StartLux 35B-A3B on 1P is DRAM-bound at batch 1 (38.7 FLOP/B); B* = 9 ≤ 332, so this is a hint.
  - The "limited by" label is now "memory capacity (KV / weights)", because weights not fitting also binds (e.g. 35B bf16 on 100T).
- Tests: `tests/test_core_208.py`; the UI smoke checks the what-if table. Fingerprints are unchanged; the NPU isolation test is kept.

## [0.207.0]
- **Default change (MI300X reference chip only): ROCm paged-attention KV-stream efficiency** (`extval.pa_rocm_eff`, variant flag `pa_rocm`; `default_0206` keeps the old default).
  - Source (independent of the validation README): ROCm/FlyDSL PR #825 (merged 2026-07-24). Production `pa_decode_ps_kernel`, fp8 per-token KV, block 16, head 128, gfx942 / MI308X, CUDA-graph wall time: bs32·ctx4096 34.9 µs, bs128·ctx4096 83.6 µs, bs128·ctx16384 216.3 µs, bs256·ctx8192 (8 q / 1 kv) 235.5 µs (来源).
  - KV bytes = bs·ctx·kv_heads·128·2·1 B (推导). Divided by 5.3 TB/s (二手), this gives efficiency 0.181 / 0.303 / 0.468 / 0.430. The curve is log-linear in KV bytes per layer call, clamped at the ends.
  - Applied to decode steps of rows on `PA_ROCM_HW = {"MI300X"}`, through `Scenario.kv_mem_eff` (two passes: the first pass gives the call size). Prefill (FA) is untouched. No NPU or NVIDIA path changes; the NPU isolation test is kept.
  - Transfer from MI308X (80 CU) to MI300X (304 CU) is 推导. PA is HBM-bound at these sizes; the 304-CU part may stream faster, so the curve is a conservative bound.
  - MI300X lat 1.02 / 23 % → 0.97 / 19 %; max-load 1.65 / 65 % → 1.49 / 49 % (per row 1.85/1.71/1.81/1.59 → 1.70/1.50/1.65/1.43). All other groups unchanged.
- Small-M FP8 GEMM on ROCm: source found, checked, **not applied**.
  - Source: aiter `a8w8_bpreshuffle_tuned_gemm.csv` (ROCm/aiter@bb6fc3c), gfx942 / 304 CU (MI300X), int8 a8w8 (same MFMA rate as fp8 on CDNA3), M = 64–40960, N×K = 1536×5120 and 5120×1280.
  - Kernel time is flat for M = 64–1024 (16.8 → 21.0 µs and 7.7 → 18.3 µs). Large M reaches 1000–1270 TFLOPS (0.38–0.49 of peak), consistent with the 0.179 MAF ceiling.
  - Model vs source: 0.37–0.86× at M ≤ 256 (a per-kernel floor at 7–8 MB weights; the source has no Llama-70B TP8 shapes), 0.8–1.3× at M ≥ 512.
  - A sourced small-M curve would lengthen the batch-1 step (already 40 % slow) and flatten the per-sequence slope, which is the wrong direction for the residual. The GEMM suspect is refuted for the slope.
- NVIDIA: no systematic optimistic batch slope.
  - Max-load gmean is H100 0.88 and H200 1.04. TRT-LLM static rows show no 3–4× slope error; the trend in batch is at most ≈ 1.3× across 56→896.
  - The sourced NVIDIA decode-attention equivalent (FlashMLA 3000/3350, `kveff_variant`, 0.108) already exists, so no NVIDIA change.
- Remaining MI300X max-load residual 1.43–1.70× (MODEL §154). Tests: `tests/test_core_207.py`; `test_core_206` pins its numbers to `default_0206`. Fingerprints unchanged.

## [0.206.0]
- Machine-balance verdict reworked around the **critical batch** (`Result.critical_batch`, `balance()`).
  - AI(B) is the model's own per-step flops ÷ DRAM bytes at each batch, so the KV reads that grow with batch are included. B* is the smallest batch with AI ≥ chip FLOP/B (at effective bandwidth). B_lim is the largest batch that fits the KV capacity and the TPOT SLO. Search: powers of two up to `BALANCE_BMAX = 65536` (假设), then bisection.
  - Warning only when balance cannot be reached within the limits:
    - (a) B* exists but is above B_lim (says whether KV capacity or the TPOT SLO binds);
    - (b) AI saturates below the chip balance (KV-bound) **and** the chip balance is above the public class top `BALANCE_CLASS_MAX = 918/1.638 = 560` FLOP/B (TPU v6e, 来源).
  - Otherwise there is a hint, not a warning: "raise batch to ≥ B*", or "KV-bound: the lever is KV bytes, not compute".
  - Default qwen3-8b at ctx 4096: 1P and H100-like are KV-bound (asymptote ≈ 28.7 FLOP/B; B_lim 141 / 113, KV capacity) and get a hint, no warning. The legacy `*-lp4x64` presets still warn.
  - At ctx 64: 1P B* = 396 (≤ B_lim 2816, reachable); 100T B* = 900 but the TPOT SLO allows 0, so it warns.
  - The bandwidth-idle side (prefill) keeps the 0.205 10× rule.
  - API `balance` adds `batch`, `critical_batch`, `batch_limit`, `limited_by`, `ai_asymptote`, `ai_at_limit`, `reachable`, `hint`. The UI KPI shows the critical batch, limit and asymptote.
- MI300X max-load residual (1.59–1.85×, MAE 65 %) broken down row by row (MODEL §153). No sourced fix exists, so there is **no default change** and scores are unchanged.
  - Prefill share is 2–32 %, all rows fit KV, and README flags were re-read (`--gpu-memory-utilization 0.9`, `--max-model-len 8192`, `--max-num-batched-tokens 131072`, one wave). Scheduling, chunked prefill and the measurement reading do not account for it.
  - The same README's latency table localises it: the measured decode-step slope in batch (b64→128) is 3.9× the model's, and in ctx (isl 128→2048 at b128) 3.2× the model's (effective KV stream ≈ 1.0 TB/s vs 3.1 modelled). At b1 the model is 40 % slow (dispatch), so the latency group's 21.6 % MAE hides opposite-sign errors.
  - The missing cost is per sequence (ROCm decode GEMM at mid M, paged attention with fp8 KV). Neither has a public absolute figure, so it is documented and pinned by a test, not fitted.
- New `tests/test_core_206.py`; the UI smoke checks the critical batch in the KPI and no balance warning on 1P. Fingerprints are unchanged.

## [0.205.0]
- Default memory pairing of the NPU presets changed (a mechanism-justified default change; 推导). The old pairing gave 1P 3840 FLOP/B and H100-like 3514 FLOP/B, against a public class range of about 200–560 FLOP/B.
  - Class (bf16 dense TFLOPS ÷ HBM TB/s):

    | Chip | FLOP/B | Source |
    |---|---|---|
    | H100 SXM | 295 | 来源 |
    | B200 | 281 | 来源 |
    | MI300X | 247 | 来源 |
    | TPU v5e | 229 | 来源 |
    | TPU v6e | 560 | 来源 |
    | Gaudi 3 | 496 (453 in a later whitepaper revision) | 来源 |
    | Ascend 910B | 200–250 | 二手 |
    | Ascend 910C | 250 | 二手 |

  - 1P: lpddr5x_4x64_8533_16g → hbm3e_4s_8h24g_9200 (4 stacks, 4.71 TB/s, 223 FLOP/B; 推导). Default qwen3-8b TPOT 80.96 → 4.69 ms; TTFT 126.06 → 60.2 ms.
  - H100-like: → hbm3_5s_8h16g_5234 (the H100 SXM memory; 286 FLOP/B; 来源). TPOT 82.09 → 6.69 ms; TTFT 284.62 → 65.47 ms.
  - 100T is unchanged (368 FLOP/B, already in the class range).
  - The defaults live in `hardware.PRESET_MEM` and are keyed by chip name. Chip objects and chip hashes are unchanged. `Scenario.mem_id = None` resolves through the preset; an explicit `mem_id` always wins. A preset name given as `scenario.chip` in the API brings its own pairing.
  - Legacy presets `1P-lp4x64` and `H100-like-lp4x64` keep the ≤ 0.204 pairing.
  - The fingerprints are 100T-based and unchanged.
- Machine-balance KPI and warning: `Result.balance()` and `balance` in `/api/eval`, plus a UI KPI. The warning fires when the chip's FLOP/B at effective bandwidth and the workload's arithmetic intensity differ by more than `BALANCE_WARN = 10` (假设), and names the idle side (compute or bandwidth). It notes that this is expected for small-batch decode.
- Tests: `tests/test_core_205.py`; UI smoke adds the preset switch and the balance checks. `test_core_204` pins the LPDDR pairing explicitly.

## [0.204.0]
- Web UI: new "每步时间分解 · Per-step time split" panel under the stage table, built from `walls` in the same `/api/eval` response (no extra request, so no race).
  - Exposed-share bar: compute / DRAM / collective BW / collective latency / dispatch / fixed overhead.
  - The wall is outlined and marked ▲; zh/en labels; an approximation note appears when `exact = false`.
  - The dispatch mode control (`c-dispatch`, `c-dispatch_us`) is in the chip panel.
- `scripts/ui_headless_smoke.py` now checks:
  - the bar shows a wall and the note is bilingual;
  - dispatch=host adds a dispatch segment;
  - a burst of five dispatch changes without waiting ends on the last choice, with no stale render;
  - the approximation note appears iff the overlap mode is not "stage".
- Naming: the walls key "static" is now `fixed_overhead` and `StageTime.t_static` is now `t_fixed` (0.203 API rename). It is the per-layer fixed overhead time, not static power; labels and docs say so.
- NPU preset sanity check (MODEL §151). 100T/1P qwen3-8b TP1 batch-1 decode at 82.0 / 81.0 ms is DRAM-limited and adds up:
  - 15.68 / 15.47 GB per step ÷ (LPDDR5X 273.1 GB/s × 0.7);
  - bytes = streamed weights 15.14 GB − SRAM-resident weights (64 / 256 MiB) + KV 0.604 GB;
  - 1P is 1.3 % faster only through its 192 MiB more SRAM; both presets use the same memory.
  - No bug found. Pinned in `tests/test_core_204.py`.

## [0.203.0]
- NPU dispatch model, new `Chip.dispatch` with optional `dispatch_us`. The default "" stays ideal (0 cost, ≤ 0.202 behaviour), so hashes and fingerprints don't change. Modes and sourced defaults:
  - `static` (one compiled program per step: launch plus token sync, once per step per stage): 10 µs per step. Source: JAX `jit_simple_dispatch` 9.1–10.1 µs, host path only (XLA commit 660c49a) 「二手」. This is a lower bound for TPU-style static graphs.
  - `trace` (device-replayed command stream: TT Metal Trace, ACLGraph replay): 1.5 µs per op. Source: tt-metal #20681 kernel preload "up to around 1.5us"; tracy op-to-op constant 1500 ns 「二手」.
  - `host` (host-launched per op): 20 µs per op. Source: vllm-ascend #13021, measured ~20 µs gaps on 910B 「二手」; the MindSpore ACLGraph doc's 20 µs is a worked example (「假设」 in the source).
  - No public figure was found for Gaudi recipe launch, Inferentia/Trainium NEFF launch, Groq or Cerebras. They use `static` with a user-supplied `dispatch_us` 「假设」.
- Counting: one op per GEMM, fused attention, stand-alone vector op or collective (`Op.launch` honoured). Dispatch is serial (`t_sync`, new `StageTime.t_disp`). CLI: `--dispatch`, `--dispatch-us`.
- The sensitivity panel (`/api/sensitivity`) gains three dispatch rows for NPU chips that are on the ideal default. This is the sensitivity scan.
- Per-step wall summary: `Result.walls()`, returned as `walls` in `/api/eval` and `eval --json`.
  - It splits the heaviest stage into compute / DRAM / collective bandwidth / collective latency / dispatch / static (`layer_overhead_us`, new `StageTime.t_static`) and names the wall.
  - The overlapped window is charged to the resource that sets it. The split is exact for `exec_overlap="stage"` and a dominant-resource reading otherwise (`exact` flag).
  - Not a duplicate: the existing stage `t_ms` gives raw per-resource times and `summary.bound` a single label; neither gives the exposed split or dispatch.
- Effect on decode TPOT, default qwen3-8b, ms (ideal / static / trace / host):

  | chip | TP | batch | ideal | static | trace | host | host Δ |
  |---|---|---|---|---|---|---|---|
  | 100T | 1 | 1 | 82.01 | 82.02 | 82.56 | 89.27 | +8.9% |
  | 100T | 4 | 1 | 20.47 | 20.48 | 21.12 | 29.19 | +42.6% |
  | 100T | 4 | 64 | 70.31 | 70.32 | 70.96 | 79.03 | +12.4% |
  | 1P | 1 | 1 | 80.96 | 80.97 | 81.50 | 88.22 | +9.0% |
  | 1P | 4 | 1 | 19.41 | 19.42 | 20.07 | 28.13 | +44.9% |

## [0.202.0]
- Standing leak audit of the GPU-specific mechanisms from 0.176–0.201 against the NPU design path (`tests/test_core_202.py`; register every new GPU-only mechanism there). Results:

  | Mechanism | Gate | Reaches NPU? | Verdict |
  |---|---|---|---|
  | Per-kernel latency `kernel_us`, `kernel_vec`; CUDA graph / eager gap (extval `TRACE_GAP_US`, `RUNTIME_BY_PHASE`) | Chip field (0 / off on NPU presets); gap is set only by extval for refhw | No | NPU dispatch = statically scheduled graph, no per-op launch 「假设」; users can set `kernel_us` / `layer_overhead_us` |
  | `gpu_glue` | Chip field | No | OK |
  | CTA tile/wave model, occupancy, DRAM floor, SMEM, L2 activations | `instance_sched="cta"`, `tile_io_Bpc` | No | OK (flag-flip test) |
  | Split-K / Stream-K (cta `mapping.SPLIT_K`) | cta only | No | OK |
  | Split-K over core groups (`evaluate.SPLIT_K`, 0.173) | none | Yes | Intended: an NPU dataflow option for output-stationary arrays 「推导」 |
  | **Wide wave refill (`WIDE_WAVES`, 0.193)** | `instance_sched=="wide"` | **Yes, a leak**: NPU auto→wide plans got the GPU "idle SMs take other row tiles" refill while the feed term still assumed a broadcast A tile (100T prefill −1.9 %) | **Fixed**: new `Chip.gpu` marker (refhw only), flag `mapping.GPU_WAVES_ONLY` (False = ≤ 0.201) |
  | TRT-LLM FP4 rates, achieved-peak netting (`NET_FULL`), `FMT_CEIL_REPLACE` | extval on refhw chips (`fmt_ceiling`, `gemm_ceil_mul`) | No | OK |
  | `idle_frac_tdp` (H100 whole-card idle) | `tpuv4i_7nm` energy preset, applied only with `tdp_W` | Yes | Kept: labelled 「二手」, and the report emits `idle_source_note` saying it is a GPU value applied to an NPU (0.189) |
  | Per-pin HBM/LPDDR PHY area and energy | energy / budget fields | Yes | Correct: memory-interface physics (Chae ISSCC 2023, Samsung PHYs), not GPU-specific |
  | Fused QKV, `W_BITS_ZERO_FIX` | none | Yes | Correct (0.201) |

  - Fingerprints: stage `b47ec84b00fc0892`, {} `b4db8aea9e5680b6`, fpx `5b3efea75e12245d` / `8220364829b88de3` (WIDE_WAVES gate).
  - Bug fixed: `W_BITS_ZERO_FIX=False` (the ≤ 0.196 branch) crashed when `w_bits` was None.
- MI300X all-reduce latency: new `Link.ar_lat="rocm"` replaces the CUDA 6.8 µs assumption.
  - Source 「来源」: vLLM PR #19744 kernel benchmark on 8 × MI300X, where the baseline is vLLM's ROCm custom all-reduce (used below 16 MB; QuickReduce is off by default). Measured: TP2 7.26 µs, TP4 7.18 µs, TP8 12.29 µs (smallest per TP).
  - Cross-checks: MK1 QuickReduce README on 8 × MI300X, RCCL at 32 KB = 17.15 / 18.17 / 16.22 µs (TP2/4/8); vLLM PR #19094 custom AR at TP2 = 8.2–11.6 µs.
  - TP3 and TP5–7 take the next listed TP 「假设」. Groups above 8 keep the NCCL Ring-LL model 「假设」. Flag `schedule.ROCM_AR` (False = ≤ 0.201).
  - Effect: MI300X latency ratio 1.105 → 1.024 (MAE 21.60 → 22.81 %); MI300X max-load MAE 67.24 → 65.02 %. Other groups unchanged. Nothing tuned.

## [0.201.0]
- NPU path vs the 0.196–0.200 GPU mechanisms (adversarial review):
  - Bug: the CUDA all-reduce latency (`CUSTOM_AR` 6.8 µs / NCCL Ring-LL) applied to every `switch` link, so NPU design points (default `Link(400, 3)`) and the cross-node `net` link silently got max(α, 6.8 µs) instead of their own `alpha_us`. Custom AR is an intra-NVLink-node CUDA kernel.
  - Fix: new `Link.ar_lat` ("" = own α, "gpu" = CUDA AR latency; omit-default, so hashes are stable); the 7 refhw GPU links are marked "gpu". Flag `schedule.GPU_AR_ONLY` (False = ≤ 0.200). Validation unchanged (all GPU rows use refhw links).
  - Consequence: every NPU fingerprint returns exactly to its 0.198 value (stage `6dca732e67c83a75`, fpx `4aa21d67471fd69e` / `a3548246a90a9ceb`; `{}` → `bc06528a82e19e2a`). The 0.200 test_core_081 finding (8 GB cap binding under the new α) was itself produced by this leak: with the NPU's own α, 8 GB is non-binding again. The test keeps the kv_blocked premise check, with 12 / 8 GB non-binding and 4 GB binding.
  - CLI: `--link-ar-lat`, `--d2d-ar-lat`, `--net-ar-lat`.
  - Genz reference points (`validation._h100`, a plain Link) also return to α = 3 µs, which matches GenZ's own α-β model.
  - DRAM floor, SMEM BW, occupancy fill, L2 activations: confined to `mapping._cta`, reached only with `tile_io_Bpc > 0` (GPU refhw chips); 100T / 1P have 0, verified flag-inert by test.
  - Fused QKV: IR-level and applied to NPUs too. Kept: concatenating the Q/K/V weights is a model/compiler transform, not GPU hardware, and work is conserved (test).
  - `W_BITS_ZERO_FIX`: general bug fix, correct on every chip.
- TP1 short-request gap (H200 mistral-7b 128: 4.05 vs 6.0 ms), gptSessionBenchmark v0.8.0 sources 「来源」:
  - Timed region = `session.generate()` + stream synchronize. It includes per-call buffer init/reshape, KV `addSequence`/`removeSequence`, decoder `newBatch` (sampling setup), the context engine pass, `decoderStepAsync` (sampling on last-token logits), and for output_len 1 one generation-loop iteration that only runs `prepareNextStep` + `shouldStopSync` (host sync) before stopping (no second engine pass), plus the output-id copy and final sync.
  - The model already charges the context pass, the last-token lm_head and sampling. The remaining items are host-side overheads with no sourced magnitude, so nothing is added; the gap stays 「假设」-attributed to them.
- Prefill lm_head: verified to project only the last token (m = batch). TRT-LLM v0.8.0 `gather_context_logits` defaults to False → `gather_last_token_logits` (models/modeling_utils.py) 「来源」. Test added.

## [0.200.0]
- test_core_081: the 2 % p99 gap in "non-binding KV cap = cap off" was a cap that binds, not a simulator bug.
  - Under the 0.199 all-reduce α, decode steps are slower and more KV stays resident, so the 8 GB cap refused 57 admissions (0 preemptions).
  - Proof with the same seed: 12 / 16 / 64 GB caps never block, and the result is bit-identical to cap off. The flat α at 8 GB never blocks either.
  - New `pdsim` counter `kv_blocked` in the simulate output. The test now checks 12 GB is non-binding and identical, and that 8 GB binds; the flat-α pin is removed.
  - test_spec_decode_energy_per_token: an exact float equality became isclose (31.999999999999996 vs 32.0).
- Correction to 0.199: the validation engines run TP all-reduce on a custom one/two-shot kernel, not NCCL (`schedule.CUSTOM_AR`).
  - Sources: TRT-LLM v0.8.0 plugin.py `use_custom_all_reduce = True`, not overridden by the 0.8 perf-doc builds 「来源」; vLLM `disable_custom_all_reduce = False`, with kernels derived from TRT-LLM's (PR #2192); SGLang uses the same kernel.
  - α = 6.8 µs: 8 × H100 at 2 KB with CUDA graphs (vLLM PR #33839) 「二手」. Using it for TP 2 / 4 and A100 is 「假设」. Groups over 8 GPUs keep the NCCL model.
- TRT-LLM 0.8 prefill, from the sources:
  - Builds: `--context_fmha enable` (TRT-LLM fused MHA, plugin default), no `--gemm_plugin` (TensorRT-native GEMM tactics), gptSessionBenchmark.
  - Attention is ≤ 7 % of prefill at ≤ 2048 tokens, so FMHA vs FlashAttention-2 can't explain a 20 % bias.
  - No published efficiency figure exists for the TensorRT-native GEMMs, so nothing was added.
- Default MAE % (0.199 → 0.200): static 22.89 → 23.04, TTFT 33.16 → 35.61, max-load 18.91 → 18.86, MoE 7.88 → 7.88, Mixtral 18.88 → 20.54, MI300X latency 24.15 → 21.60, MI300X max-load 63.94 → 67.24, B200 13.13 → 13.13.
- Fingerprints re-baselined: `_fp` be3e00dcc13fcdda / 7d5cdb465b922571, test_core_125 ba49c208f61b306c / 9bb4812eed9b116a.
## [0.199.0]
- Small-message all-reduce latency now follows NCCL's own cost model (`schedule.NCCL_AR_LAT`): Ring + LL = 6.6 + 2(n−1)·0.6 µs on NVLink (src/graph/tuning.cc v2.19.3 「来源」). That is 7.8 / 10.2 / 15.0 µs at TP 2 / 4 / 8; ≤ 0.198 it was a flat 3 µs.
  - Cross-check: nccl-tests issue #123 measured ≈ 21 µs on 8 × A100 「二手」.
  - 「假设」: engines run their collectives at NCCL latency (vLLM / SGLang custom all-reduce and the TRT-LLM plugin may be faster), and RCCL on MI300X uses the same table.
- Default MAE % (off → on): static 23.26 → 22.89, TTFT 36.86 → 33.16, max-load 18.72 → 18.91, MoE 7.88 → 7.88, Mixtral 21.32 → 18.88, MI300X latency 23.08 → 24.15 (bias 1.170 → 0.988), MI300X max-load 68.81 → 63.94, B200 13.13 → 13.13.
- Short-request TTFT, row by row:
  - (c) eager prefill launch gaps were already charged (2.0 µs floor + 2.40 µs eager gap per kernel, 0.176).
  - (b) the TRT-LLM 0.8 rows come from gptSessionBenchmark (C++, token ids, no tokenizer / HTTP), so (b) doesn't apply. The Databricks rows are server-measured, but there is no sourced overhead figure.
  - (a) explains ≈ 2 ms of the 6–58 ms TP8 gaps. The 2 ms TP1 gap (H200 mistral-7b 128) is still unexplained. Nothing was fitted.
- These tests are pinned to the flat α (`pin_module_flat_alpha`): schedule α-β formula, 077, 050 CLI, 059 bit-for-bit, and 081 DES. In 081, "non-binding KV cap = cap off" differed at p99 by 2 % under the new α; this looks like DES sampling sensitivity, not investigated.
- Fingerprints re-baselined: `_fp` 841b5c4e00c68d8c / 727e05f1eee3e647, test_core_125 3e753f36c5bcdb36 / fefb5b23fdf2782c. Legacy pins now also use the flat α.
## [0.198.0]
- Fused QKV: dense GQA layers emit one `attn.qkv` GEMM (`ir.FUSE_QKV`; False = ≤ 0.197 three GEMMs). Sources: vLLM QKVParallelLinear (models/llama.py qkv_proj), SGLang QKVParallelLinear, TensorRT-LLM Attention.qkv ColumnLinear / PyTorch-backend qkv_proj. FLOPs and weights are unchanged. MLA and grouped linears are untouched.
- Default MAE % (unfused → fused): static 22.48 → 23.26, TTFT 31.57 → 36.86, max-load 19.34 → 18.72, MoE 7.88 → 7.88, Mixtral 20.23 → 21.32, MI300X latency 21.86 → 23.08, MI300X max-load 68.32 → 68.81, B200 13.55 → 13.13.
- Re-baselined `_fp` fingerprints (6dca732e67c83a75 / bc06528a82e19e2a, were 5e72913c351849ac / eb634f2e8b513ffb) and test_core_125 fpx (4aa21d67471fd69e / a3548246a90a9ceb). Legacy pins now also run unfused QKV; per-projection shape and kernel-count tests are pinned unfused (`pin_module_unfused`). The IR share matcher treats attn.q/k/v/qkv as one work item, since fusion depends on per-shard head replication. test_core_139 gives the dequant knob 2e-4 slack: the split picker shows a +0.018 % per-op non-monotonicity on qwen3-8b-fp8 fused qkv.
- Prefill-heavy regressions traced: they are all in 128-token TTFT rows, which were already 1.17–1.96× too fast. 2048-token rows are unchanged by the L2 discount, which is applied per op over its own token count. The missing term is fixed short-request latency (per-request / TP-collective), still unsourced, so nothing was fitted.
## [0.197.0]
- GEMM DRAM floor: activation bytes count only beyond L2 (`CTA_L2_ACT`, L2 = refhw `sram_mib`). Before, the floor counted activations that memplan already keeps on chip (stage `dram["act"]` = 0), so they were counted twice. Example: H100 70B decode batch 256, attn.q 31.3 µs vs 28.6 µs of weights.
  - Reuse across tiles was never double-counted: the floor counts unique bytes.
  - The MI300X latency regression is not an L2 effect: small attn.k / attn.v kernels sit above their byte floor and kernels run one after another. Engines fuse QKV, but the IR keeps q, k and v as separate GEMMs. Not changed.
- Fix: `Op.w_bits` 0.0 (the default) now means "use the format's bits" (`W_BITS_ZERO_FIX`). Before, synthetic GEMMs, the extval ceiling bisection and attention ops got a free 0-byte B operand.
  - This, not throttling or the wave tail, was why the B200 2048³ occupancy point read 123 TF over measured: it is now 945 vs 1006.
- Default MAE % (0.196 → 0.197): static 22.22 → 22.48, TTFT 29.92 → 31.57, max-load 19.83 → 19.34, MoE 8.08 → 7.88, Mixtral 18.77 → 20.23, MI300X latency 21.74 → 21.86, MI300X max-load 68.16 → 68.32, B200 14.10 → 13.55.
- `default_occ` stays opt-in: at B200 2048³ its error is 61 TF vs 47 TF for the default, and validation is worse (TTFT 37.7 vs 31.6 %).
## [0.196.0]
- Fix: under tile I/O, GEMM array time is bounded below by the unique operand + output bytes / aggregate DRAM bandwidth (`mapping.CTA_DRAM_FLOOR`). ≤ 0.195 every tile-I/O mode could run below the roofline: the main loop was fed from the SRAM port, and occupancy divided the I/O by k. Fuzz: 690 / 788 / 684 of 1200 shapes were below it (worst 0.10×); now 0. The DeepSeek-V3 kv_b.uk bmm on H100 with occupancy goes 2.93 → 7.60 µs (= its byte floor).
- Default old → new MAE: static 22.76 → 22.22, TTFT 32.86 → 29.92, max-load 19.11 → 19.83, MoE 7.74 → 8.08, Mixtral 22.23 → 18.77, MI300X latency 21.07 → 21.74, B200 13.52 → 14.10.
- Occupancy: pipeline fill / drain is exposed once per SM: k prologues plus the last epilogue (`CTA_OCC_FILL`).
- SMEM-bound small tiles: `smem_Bpc_sm` is 128 B/clk/SM (`CTA_SMEM_BW`).
- `default_occ` is now built on the current default; the old one is kept as `default_occ_0195`. It stays opt-in: B200 2048³ 1129 vs 1006 measured (default 1034), and validation is worse.
- New roofline invariant fuzzer over all variants (tests/test_core_196).
## [0.195.0]
- tile_io "occupancy" (opt-in, `default_occ`): resident CTAs per SM from documented smem / register (TMEM) / thread limits vs CUTLASS sm80 default tile configs (3 stages × 128 B of K, 256 threads); a tile costs max(main, (main + I/O)/k). Not default: B200 2048³ 1034 → 1214 TF (meas 1006), v1 TTFT 32.9 → 42.1 %.
- B200 NVFP4: TRT-LLM rows use TRT-LLM's FP4 GEMM rates (PR #7943, CUTLASS default backend, 8192×8192×K: 2397–4535 TF), not the cuDNN library best (6220). B200 group 1.125 / 22.8 % → 1.008 / 13.5 %; 128/128 1.651 → 1.216, 2048/128 1.496 → 1.108. Other groups unchanged.
## [0.194.0]
- Small-GEMM check: the 0.193 B200 2048³ "1534 vs 1006" left out the default's existing per-kernel cost (2 µs floor + 1.03 µs gap) and tile I/O; with them it is 997. No new fixed-cost term: pipeline fill ≈ one DRAM latency (GH100 658.7 cycles, arXiv 2507.10789) is < 3 % at 2048³.
- Net GEMM ceiling now includes tile I/O + per-kernel cost at the source shape (`extval.NET_FULL`, bisection; ≤ 0.193 waves only, double-charging them). H100 4096³ 653 → 695 TF (meas 716), B200 2048³ 997 → 1034 (1006), 4096³ 1375 → 1446 (1532).
- Hopper `fmt_ceiling` merges instead of replacing (`FMT_CEIL_REPLACE`): bf16 / fp16 GEMM ceilings were silently dropped on H100 / H200 / H800.
- Re-score 0.193 → 0.194: static 22.66 → 22.76 %, TTFT 32.22 → 32.86, max-load 19.09 → 19.11, MoE 7.69 → 7.74, Mixtral 22.14 → 22.23, MI300X lat 20.98 → 21.07, MI300X max-load 66.79 → 68.21, B200 17.85 → 22.82 (NET_FULL; the padding had been compensating).
- MoE regression traced to decode rows: tile I/O "dram" on many-tile small GEMMs (MLA kv_b bmm 5.5 → 57 µs). Persistent tile I/O (sm90 CUTLASS) is a diagnostic variant (`default_tioauto`, TILE_IO_BY_HW); not default (v1 TTFT 1.38).
## [0.193.0]
- Reference GPUs: GEMMs = SM × CTA-tile wave quantisation (`instance_sched` "cta", NVIDIA tile set, tile I/O "dram") × sourced achieved-peak fraction applied multiplicatively (`Chip.gemm_ceil_mul`), net of the measured shape's own wave quantisation (`extval.CEIL_SHAPE`). Replaces the wide-gang N padding.
- `mapping.WIDE_WAVES` on by default (abstract wide chips no longer pad N); `Chip.wide_pad` keeps ≤ 0.192 padding for historical variants.
- cta split-K up to 512 (Stream-K) for tiny-output long-K GEMMs.
- Extval default old → new (geo / MAE): static 1.150/25.0 → 1.118/22.7; TTFT 1.307/32.8 → 1.322/32.2; max-load 0.985/21.5 → 0.955/19.1; MoE 0.962/7.0 → 0.943/7.7; Mixtral 1.291/29.6 → 1.213/22.1; MI300X lat 1.123/22.4 → 1.110/21.0; MI300X max-load 1.726/72.6 → 1.668/66.8; B200 1.061/17.5 → 1.065/17.9. `default_0192` keeps the old.
- Prefill marginal-cost invariant now holds on the default; fingerprints re-baselined (old reproduced under `LEGACY_0192`).
## [0.192.0]
- Prefill scaling found: the GPU "wide" gang pads an output dim to E·cols (H200 8448). Qwen3.5-9B at 4096 tokens has N = M = 4096 (48 % fill); at 8192 the 0.173 orientation swap fills the gang (97 %), so 4096 → 8192 cost 1.5×. `mapping.WIDE_WAVES` (opt-in) wave-quantises per-engine tiles instead: 2.0×, and the marginal-cost invariant holds. Default off: it reverses the §119 calibration (extval TTFT 1.31 → 1.44).
- SRAM pins recurrent state before weights (`memplan.STATE_FIRST`, 2 B traffic / B vs 1); hybrid / RWKV models are slightly faster; fingerprints re-baselined.
- AutoThink thought decode: `serving.think` / `think_tokens` (default 596 「二手」, ≤ 4096), CLI `--think [N]`; decision latency = readout + thought decode.
## [0.191.0]
- Decision workload (`serving.labels` 1..1024, prefill only; CLI `--labels`): one forward, label-row readout at the last position, no decode/KV write at ctx 0; reports decisions/s and latency per decision.
- StartLux-Decision v1 (0.8B/2B/4B/9B/27B/35B-A3B) and v1.1 (Instant/AutoThink) from live HF configs; Instant `jev_prompt_attention: bidirectional` → non-causal full-attention layers. Base Qwen/Qwen3.5-9B added.
- Microsoft-Decision-1 as an architecture proxy (Qwen3.5-9B config, coverage proxy, 推导). Jev not added (closed, no architecture).
- RWKV-7 G1k 1.5B/2.9B/7.2B/13.3B: new `rwkv7` family, exact param counts, fp32 H·64×64 state, context-invariant decode, chunk-64 prefill (paper equations).
## [0.190.0]
- UI smoke covers energy controls incl. a binding power cap; fixed preset idle fraction losing its citation after UI rounding, and share display rounding.
- CLI `eval --power-cap`.
## [0.189.0]
- Compound provenance tags accepted (core/provenance); LPDDR PHY 1.3 → 推导/二手.
- Energy report: static share, dynamic J, idle source note, >50 % static warning (UI).
- Opt-in TDP power cap (`power_cap`): throughput ×(TDP−P_static)/P_dyn when over TDP.
## [0.188.0]
### Added
- Separate chip-side DRAM PHY energy line (`pJ_bit_phy_hbm` 0.29 来源 / `pJ_bit_phy_lpddr` 1.3 二手).
- Static power from `idle_frac_tdp` × `tdp_W` (H100 measured idle 0.103 二手; TDP from the budget if unset).
- Prefill energy (`J_prefill`) in insight/sizing; LLM `energy_tie` constrains both phases.
## [0.187.0] - 2026-10-16
### 变更（默认值，tpuv4i_7nm 能耗预设）
- 一手来源与 0.186 的 PIM-AI 数值矛盾：一手数值约为它的一半，LPDDR/HBM 比值 1.16 而非 1.43。改为：
  - pJ_bit_dram_hbm：7.0「二手」→ 3.44「推导」（SK hynix IEDM 2023 15.6：HBM2E ≈ 4.3 pJ/bit，HBM3E 低 20%）。
  - pJ_bit_dram_lpddr：10.0「二手」→ 4.0「推导」（NVIDIA Grace：LPDDR5X 子系统约 16 W 提供 500 GB/s）。
  - 二者都是内存侧、峰值带宽口径，SoC 侧 PHY 不含（HBM3 host PHY ≈ 0.29 pJ/bit，Chae JSSC 2023）。交叉核对：O'Connor MICRO 2017 HBM2 3.97。
  - horowitz_45nm 预设不加按类型值（无 45 nm 时代的按类型来源）。
  - 2 nm LPDDR6 PHY 的 pJ/bit：0.184 所用来源（SemiAnalysis 对 ISSCC 2026 37.3 的报道）未给，未用。
- 效果（tpuv4i_7nm，默认芯片 + qwen3-8b）：DRAM 能耗 LPDDR5X 1.254 → 0.502 J/token，HBM3E 0.878 → 0.431。
### 新增：缩减余量的能耗暴露 / 能耗约束
- sizing_report / API / UI 每行给出最差模型在 min_ok 处的 DRAM 字节和 SRAM 字节倍数；有能耗表时也给 J/单位倍数。
- energy_tie=true（UI「能耗也不变差」）把「能耗 / 单位 ≤ 当前 × 1.01」加入可行条件。
- 例：wan2.1 SRAM 可缩到 25% 而延迟不变，但 DRAM 字节 ×1.64、能耗 ×1.033；加能耗约束后只能缩到 50%，受限于能耗。
- 选用的 UI 冒烟脚本已覆盖能耗约束。MODEL §134。

## [0.186.0] - 2026-10-16
### UI 冒烟（选用脚本）加方向断言
- scripts/ui_headless_smoke.py 现在检查渲染值确实变化且方向正确：固定 batch 1→16 TPOT 上升；2 卡 TP2 比 TP1 TPOT 低；DRAM 颗数减少则带宽、容量都降，恢复后带宽复原；prefill 显示 TTFT 卡。每次改动都等到 /api/eval 返回（不是任意 POST）。
- 首轮唯一告警「颗数 8→6 容量仍 128 GiB」查明是脚本误报：存储器改动先 POST /api/memory，脚本在那时读卡片，读到的是上一次渲染。UI 本身会在 /api/eval 返回后重渲染（96 GiB）。已修脚本，UI 无需改。终轮 0 错误。
### 新增：DRAM 能耗按存储类型区分
- EnergyTable 新增 pJ_bit_dram_hbm / pJ_bit_dram_lpddr（按场景 mem_id 选；未填回退 pJ_bit_dram）。报告给出 dram_pJ_field；只填按类型字段时 pJ_bit_dram 不再列为缺失；HBM 值不会给 LPDDR 定价。UI 能耗组、CLI（--pJ-bit-dram-hbm/-lpddr）、API 均可用。
- 预设 tpuv4i_7nm 新增（默认值变化：DRAM 能耗原先不分类型，一律 3.17 pJ/bit）：HBM 7、LPDDR 10 pJ/bit「二手」，二者都取自 Ortega et al.（UPMEM）PIM-AI arXiv 2411.17309 表 1，同一 2× 接口口径；交叉核对 O'Connor et al. MICRO 2017 HBM2 ≈ 3.97 pJ/bit（仅内存侧）。horowitz_45nm 预设不变。使用 tpuv4i_7nm 预设时，默认芯片 + qwen3-8b 的 DRAM 能耗：LPDDR5X 1.254 J/token（原 0.397），HBM3E 0.878（原 0.397）。MODEL §133。

## [0.185.0] - 2026-10-16
### 变更（默认值）
- 密度预设 mm2_per_dq_hbm：14/1024 = 0.0137「二手/假设」（H100 die-shot 估算，未核实）→ 8×4/2048 = 0.0156「推导/二手」：Samsung Hot Chips 2026（S. Han）HBM4 base die 上传统 PHY「超过 8×4 mm」对应 2048 DQ（SF4），由 Tom's Hardware / ServeTheHome 报道（原文已核对）；XPU 侧 PHY 取与内存侧相同（推导，下界）。HBM 的 dram_units / phy_area 面积 +14%。
- LPDDR5X N5/N4/N3 PHY 面积：仍无公开数字（Synopsys/Cadence 只给速率），保留 0.184 的 LPDDR6 SF2 值。
### 新增
- scripts/ui_headless_smoke.py（选用，不在 run_tests 内）：自起服务器，用 Playwright + Chrome 依次切换模型、阶段、芯片预设、映射、DRAM 数量/类型、密度预设、卡数/TP、D2D，每次都要求触发 API POST（<400）且 KPI 卡片重新出现；然后跑缩减余量（单项 + 联合）和升级路径。监听 pageerror、console error 和 5xx。首轮：31 步，0 错误，无需修复。

## [0.184.0] - 2026-10-16
### 修复（UI）
- 缩减余量面板：整数资源（核数、DRAM 封装/堆栈数）显示整数（旧 "4.00 个" / "16.0 核"）；已到 1 的整数资源显示「已是最小值 1」而不是「≤ 100% · 到搜索下限」。API 每行新增 `integer`、`unit_floor`。已用真实浏览器（Playwright + Chrome）跑过该面板，无 JS 错误。
### 新增
- scripts/fuzz_sizing.py：在随机小配置（1 核、1 封装、1×N 阵列、极小 SRAM/lanes/反量化，4 种密度预算）上跑 sizing 单项 / 联合 / upgrade_path / insight_report，每次调用有墙钟超时；检查 min_ok ≤ current、整数 ≥1、fail_at < min_ok、mm² ≥ 0、联合不含 dram_units。40 例 × 4 入口：0 发现，0 拒绝（单领域模型组）。测试套件内含 6 例冒烟。
### 变更（默认值）
- 密度预设 mm2_per_dq_lpddr：0.03「假设」→ 0.695/24 = 0.0290 mm²/DQ「推导/二手」（Samsung SF2 LPDDR6 PHY，ISSCC 2026 论文 37.3，经 SemiAnalysis：0.695 mm²，55.3 GB/s/mm² ⇒ 38.4 GB/s = 一个 x24 通道 @14.4 Gb/s）。dram_units / phy_area 的 LPDDR 面积 −3.5%（wan2.1 例 1.92 → 1.853 mm²）。MODEL §131。

## [0.183.0] - 2026-10-16
### 新增
- sizing 资源 `dram_units`（DRAM 封装 / 堆栈数，离散）：同型号、同速率、同 mem_eff，仅改数量，带宽与容量同减（容量不足即 fits 失败）。面积 = (n−n_min) × 单元位宽 × DRAM PHY 密度（HBM → mm2_per_dq_hbm 二手/假设；其他 → mm2_per_dq_lpddr 假设），按内存类型取，未设则不给 mm²。默认资源列表变更（旧 5 项 → 新 6 项）；联合缩减跳过它（与 dram_bw 同一条线，不可分离）。MODEL §130。
### 修复
- sizing 整数资源当前值为 1 时（engines=1、单封装）减半搜索死循环（0.5 四舍五入回 1）；现在到 1 即停。≤0.182 会挂起。
### 示例
- 默认芯片（LPDDR5X 4×x64，0.03 mm²/pin 假设）：qwen3-8b / esm2 不能少一个封装（TPOT/延迟 0.75）；wan2.1 可到 3 个，省 1.92 mm²。HBM3E 8 堆 + qwen3-8b：可到 3 堆，省 70 mm²（14 mm²/1024 pin 二手）。

## [0.182.0] — MAI models checked (not added); TRT-LLM 0.8 all-reduce paths recorded

- **MAI-Base-2.** Not added. The PDF table gives layers, hidden, heads, experts and latent MoE dims, but not head_dim, vocab, window or gating, and the totals do not reconcile with 2.8T. It is a closed, post-training model.
- **MAI-Code-1.1-Flash.** Not added. The card gives 137.9B / 6.8B, 3.3 bpw, 53 GB, 256K and a DFlash2 drafter; the llama.cpp arch is `maion-coder` (PR #29778). The hyperparameter values sit only in a GGUF behind the Azure ML registry's signed download; no public config was found.
- **TP all-reduce.** TRT-LLM v0.8.0 uses its custom all-reduce by default (one-shot below 500 kB, two-shot otherwise at TP8) and NCCL above 8 MB. No single-node 8 × H100 NCCL table at 1–64 MiB was found, so nothing was applied (MODEL §129).
- Scores unchanged.

## [0.181.0] — Continuous batching packs short prompts per prefill iteration; per-row TTFT / Mixtral residuals

- **Packing.** `_cb_at` packs ⌊C / isl⌋ prompts per prefill iteration (TRT-LLM `max_num_tokens` packs tokens from different sequences, 来源). ≤ 0.180 put one prompt per iteration, which capped the running batch at osl.
- **Running cap.** `predict_cb` now applies the source's engine cap by default (`SOURCE_ESTIMATOR`); TRT-LLM changes from none to 2048 (max_batch_size default). `PACK_PROMPTS[variant] = False` restores the old rule.
- **Default old → new** (speed ratio): v1 max-load 0.92 / 19 % → 0.99 / 21 %, Mixtral TP8 1.12 / 32 % → 1.29 / 30 %, B200 1.01 / 14 % → 1.06 / 17 %, MI300X max-load 1.76 / 76 % → 1.73 / 73 %. Other groups unchanged.
- **Per-row diagnosis** (MODEL §128).
  - v1 TTFT error scales with TP: all-reduce at prefill message sizes explains most of it, but only multi-node NCCL tables were found, so it is documented, not applied.
  - The Mixtral residual is uniform, about 1.24–1.36 per GPU; documented.
- New `test_core_181`; `ext_validation.json` regenerated.

## [0.180.0] — MI300X max-load: tokens/s definition settled by ratio test; host and attention mechanisms bounded

- vLLM 0.11.2 `benchmarks/throughput.py` prints total and output tokens/s (JSON `tokens_per_second` = total); the README column is unspecified.
- Cross-length ratio test: output reading sd(log) 0.068 vs total 0.22 → output kept. Rows tagged `tok_def: "output"`.
- Host scheduler cost (vLLM PRs #40459 / #46112, ≈ 1.9 µs/seq → ≈ 6 ms/step at 3200 seqs, ≈ 5 % of the step) and ROCm paged-attention (relative speedups only) were documented, not applied (MODEL §127).
- Scores unchanged (MI300X max-load 1.76 / 76 %); no default change. New `test_core_180`.

## [0.179.0] — MI300X max-load: one-wave batch + AMD MAF GEMM ceilings; ROCm source re-verified

- ROCm per-kernel source (MODEL §125) re-verified live, verbatim. The exact URL is now recorded; MI300X decode is kept.
- MI300X max-load rows now carry the README's Num Prompts = Max Num Seqs (3200 / 1500 / 2000 / 1500), and `predict` runs exactly that one-wave batch (old: best power of two, 4096 / 2048).
- The `rocm-vllm-0.11.2` estimator's chunk changed 16384 → 131072 (the README flag), and its cap now comes from the row.
- `GEMM_CEILING["MI300X"]` changed from peak (no figure) to the AMD MAF numbers: FP8 1223.6 / 2614.9 = 0.468, BF16 708 / 1307.4 = 0.542, FP16 654 / 1307.4 = 0.500.
- Default old → new (speed ratio): MI300X max-load 2.02 / 102 % → 1.76 / 76 %, MI300X lat 1.13 / 23 % → 1.12 / 22 %. Other groups unchanged.
- The residual (1.7–2.0×) is documented in MODEL §126. HBM (4.3 TB/s measured), RCCL (295–318 GB/s busbw) and KV capacity do not explain it.
- `ext_validation.json` regenerated; `test_core_105` updated; new `test_core_179`.

## [0.178.0] — Default = per-layer mechanism (per-kernel cost + glue); ROCm decode sourced; lump only as fallback

- **Default change** (mechanism-justified; MODEL §125).
  - The validation `default` replaces the flat 62.3 µs/layer lump with its mechanisms: the per-kernel floor plus the runtime launch gap (§122) and the GPU glue kernels (§123).
  - Old → new definition: `ceil(candidate)` with `layer_us` 62.3 becomes `rocm(glue(mech(default_lump)))`. The old default stays as the variant `default_lump`.
- **AMD.**
  - Decode uses a sourced ROCm HIP-graph per-kernel cost, `ROCM_KERNEL_US` 4.85 µs (public MI300X DeepSeek V4 Flash trace: ~1,200 launch-floor kernels ≈ 5.8 ms; 0.04 ms idle over 1,970 kernels).
  - Prefill keeps the 62.3 µs lump as a labelled 假设 fallback (`fallback_us`); there is no public ROCm eager trace.
  - `catalog_kernel_fixed` gets the same AMD treatment.
- **Old → new** (speed ratio, gmean / mean abs error):

| group | old | new |
|---|---|---|
| v1 static | 1.13 / 24 % | 1.15 / 25 % |
| TTFT | 1.33 / 35 % | 1.31 / 33 % |
| max-load | 0.92 / 20 % | 0.92 / 19 % |
| MoE | 0.97 / 8 % | 0.96 / 7 % |
| Mixtral | 1.14 / 33 % | 1.12 / 32 % |
| MI300X lat | 1.40 / 40 % | 1.13 / 23 % |
| MI300X max-load | 2.09 / 109 % | 2.02 / 102 % |
| B200 | 0.99 / 15 % | 1.01 / 14 % |

- `ext_validation.json` regenerated. Tests: `test_core_141` / `176` / `177` updated for the new definitions; new `test_core_178`.

## [0.177.0] — GPU glue kernels in the op list; ratio-convention note

- `Chip.gpu_glue` (opt-in) and `ir.glue_ops` add the GPU glue kernels. Each has its own bytes / work plus the 0.176 per-kernel fixed cost.
  - New kernels: activation quant casts, MLA q_a norm, decode split-KV combine, router top-k, router split-K reduce, shared-expert SwiGLU + cast, and sampling passes.
  - Collectives now count as GPU kernels (EP send + recv).
  - DeepSeek-V3 decode: 17.2 → 29.05 kernels/layer (trace 30.4).
- `catalog_kernel_fixed` changed from mech only to mech + glue. Old → new speed ratio: v1 static 1.23 → 1.21, TTFT 1.40 → 1.33, max-load 1.03 → 1.00, MoE 0.81 → 0.80, Mixtral 1.25 → 1.21, B200 1.11 → 1.07, MI300X unchanged. `ext_validation.json` regenerated.
- Not promoted to `catalog_kernel`, because it does not match or beat the flat 62.3 µs lump on every group. It loses on v1 static (30 % vs 29 %), B200 (13 % vs 12 %) and MI300X (glue not applied). Defaults unchanged.
- MODEL §123 (glue) and §124 (ratio direction conventions); §119 and §122 table labels corrected.

## [0.176.0] — Mechanism-scoped fixed cost for catalog_kernel; 128-token TTFT residual re-check

- The 62.3 µs/layer lump (deepseek-ai/profile-data decode) split by mechanism from the public decode + prefill traces: per kernel = small-kernel latency floor (2.0 µs, p10) + inter-kernel gap of the launch runtime (CUDA-graph decode 1.025 µs, eager prefill 2.404 µs). The rest of the lump is size-dependent work the model already charges (prefill small kernels p50 116 µs) → not repeated.
- New reported variant `catalog_kernel_fixed` (= `mech_variant("catalog_kernel")`, CUDA only; no ROCm trace → MI300X rows unchanged). `catalog_kernel` stays as the no-fixed-cost baseline. `data/ext_validation.json` regenerated with it.
- Bands old → new (catalog_kernel → catalog_kernel_fixed, gmean ratio): v1 static 1.31 → 1.23, v1 TTFT 1.75 → 1.40, v1 max-load 1.12 → 1.03, MoE large-EP 0.82 → 0.81, Mixtral TP8 1.34 → 1.25, MI300X 3.33 / 2.32 (unchanged), B200 NVFP4 1.26 → 1.11.
- Default unchanged (keeps the lump): default + mech would be worse on 6/8 groups because the model's op list has 17.2 kernels per DeepSeek decode layer vs 30.4 in the trace (casts, split-K reduce, sampling not listed).
- 128-token TTFT: residual after the fixed cost 1.8–2.1 ms (Hopper 7B), 3.7–4.7 ms (A100 7B), 7.3 / 15.0 ms (70B TP8). No sourced framework-time breakdown for TRT-LLM 0.8 GptSession → left open (MODEL §122).

## [0.175.0] — Muse Spark Open follow-ups: NVFP4, NaViT block windows, audio encoder, DSpark draft

- `muse-spark-open-nvfp4`（Meta 官方 NVFP4，仓库 gated → 推导）：release 格式支持 nvfp4（e2m1 + 每 16 个 e4m3 scale + 每张量 fp32，4.5 bit）；`summarize_release.py` 按 quantization_config 识别 NVFP4；专家 NVFP4、其余 bf16（假设，exclude 列表未公开）。bytes_llm 491.9 GB。
- 视觉：NaViT 37 个窗口层按 32×32 patch 块内注意力计（PR `_get_sparse_attention_perm`），13 层全注意力；进出各一次重排。
- 音频：PR `audio.py` AutoEncoder 前向（117.84M 参数，4 级 patchify 16·6·5·4 = 1920 样本 / token @ 24 kHz）；新增 `serving.audio_s`（默认 0，不序列化）。
- DSpark：`muse-spark-open-nvfp4-dspark` = NVFP4 目标 + Inferact/Muse-Spark-Open-DSpark（公开 config + safetensors 头，5,301,633,537 bf16 参数精确对上）；复用 dspark 模板，新增 `draft_kv`（上下文 K/V 投影宽度）。
- Re-baseline：仅 Muse 的图像 prefill（窗口层 attention 由全注意力上界改为块内）；其他模型不变。

## [0.174.0] — Meta Muse Spark Open (816B-A42B)

- 新增 `muse-spark-open`（BF16，由 FP8 header 推导）、`muse-spark-open-fp8`（amd PTPC-FP8）、`muse-spark-open-mxfp4`（amd Quark MXFP4）。形状来源：AMD 公开 config.json + vLLM PR #61879 @ 6cc692b5；Meta 仓库 gated。
- 参数核对（来源=release header）：819.99B，误差 −1.2e-6；experts 798.9B / attn 13.52B / latent 4.16B / embed+head 3.31B / vision 2.05B（含 audio/projector），与博客表一致；激活 42.8B（不含 embed/head）。
- NVFP4 官方版未列入：release role 格式不支持 NVFP4（假设/partial）。视觉塔（NaViT 50 层 + adapter，1.89B）已建模，37 个窗口层按全注意力计、缩放规则为假设；音频编码器仅计存储（coverage partial）。
- TPU v7 megakernel 仅作标注的 sanity 对比（无拟合），见 MODEL §120。默认值未改，无 re-baseline。

## [0.173.0] — Mapping-artifact sweep (wide orientation / instances / split-K), roofline-ratio invariant, 088 deflake
- Sweep 「推导」 (`/workspace/sweep173`): every GEMM / attention op of all listed models × prefill 8k / full 4k / decode b1 / b64 × {H100, B200, MI300X, A100, 100T, 1P}, comparing array time with FLOP time and the swapped orientation. Three artifacts of the 0.172 class:
  1. **Small N on "wide"** (GPU references): k / v projections (N 1024) ran 8–18×, routers (N 256–384) 37–74×, and hyper-connection mixers (N 4–24) 350–4700× their FLOP time, because one orientation was forced on a 32 × E·cols array. Fix: wide GEMMs take the cheaper of C / Cᵀ (cuBLAS row-major is exactly the swapped call). Conversion counts are preserved; packed-scale bits are dropped on the swapped side (≤ 10 %, feed only).
  2. **Instances serialised on "wide"**: MoE experts with few tokens (m 147–410, count 160–896) ran 4–9×. Chunked linear-attention blocks (64 × 128 × 64, count 8–16k) ran 66–132×. Long-sequence attention (count = batch × heads) ran 4.6× on B200. Fix: for count > 1, also cost the core-group (auto) plan, which spreads instances and splits each over the remaining groups (CUTLASS grouped GEMM; FA grid over batch × head × q-block).
  3. **Serial K on a tiny output** (all non-cta schedules): a 1 × 28672 × 24 hc mixer took 28672 cycles on one tile. That is about 5 ms per decode step for deepseek-v4-pro on 100T. Fix: split-K over core groups plus a vector-lane reduction (cuBLAS / CUTLASS split-K; Flash-Decoding for decode PV). Conversion work is unchanged.
- Flags: `evaluate.WIDE_GEMM_ORIENT` / `WIDE_GEMM_SPLIT` / `SPLIT_K` (default on) and `WIDE_GEMM_AUTO` (diagnostic, off). `model_flags(**LEGACY_0172)` reproduces 0.172.
- New standing invariant `tests/test_core_173.py`, run for 13 models (dense, GQA, MLA, MoE, hc, linear attention, Hyena, DiT, protein) × 4 phases × 9 chips: array ≤ max(3 · g · FLOP, operand bytes / BW) + 2 µs, where g is the geometric tile underfill of the better orientation (on wide, instances may share the gang). It fails with the fixes off.
- Not changed (documented):
  - The wide gang width itself: B200 has E·cols = 18944, so full-sequence DiT GEMMs with both sides < 18944 stay 2–6×. The `WIDE_GEMM_AUTO` diagnostic removes this but moves extval the wrong way (default Mixtral TP8 1.14 → 1.17, MI300X max-load 2.09 → 2.39). The wide underfill there stands in for real kernel efficiency, which is not otherwise modelled.
  - Small-M decode on output-stationary arrays: this is a real design effect (the os / os_vec / reconf choice).
- Default impact: abstract-chip defaults move ≤ 1.2 % (hc / MoE models; deepseek-v4-pro decode TPOT 208.11 → 205.66 ms). On the reference GPUs, prefill with small-N projections speeds up. evo2-7b 8k reference TTFT goes 0.41291 → 0.37423 s.
- External validation, as default / catalog_kernel geomean, 0.172 → 0.173: see MODEL §119.
- Over-predicting catalog_kernel groups (item 3, no fitting):
  - The variant has no fixed per-layer cost. Adding the sourced public-trace 62.3 µs/layer (`TRACE_LAYER_US`, §39) brings v1 static from 1.31 to 1.19, v1 TTFT from 1.75 to 1.36, and B200 NVFP4 from 1.26 to 1.05. GEMM ceilings add only about 0.01.
  - The residual sits on the 128-token TTFT rows (1.3–1.7×), which are fixed-cost dominated.
  - Before 0.172 the wide-array underfill masked this missing overhead.
- 088: the status line is now read until CRLF (a single recv can return a partial segment). Time is measured on the monotonic clock. The 20 s bound now moves only by the stall that a parallel 2 s control timer observes (unloaded: unchanged).
- Re-baselines:
  - Default fingerprints 0a954915c5fb003f / 53ee9e08165795c6 → 84df3be517a53ef4 / 1b755ac357ec6d2e, across 17 files.
  - 125 fpx 4690f724dabe9fa6 / ea2456afee7c3a86 → 048879e566bfde2c / 9b1f6dda6e1fd768.
  - 171 evo2 8k reference TTFT.
  - Pinned to 0.172 via `tests/pin_legacy.py`: the DES / queue operating-point tests (051 052 054 055 056 078 085 064-slo), the worked examples (062_1 §19, 098, 134), and the study bands (067 069 091 092 093).
  - `data/ext_validation.json` regenerated.

## [0.172.0] — Long-context attention: ~34× roofline was a mapping artifact (fixed)
- Root cause 「推导」: the GPU references run `instance_sched = "wide"`, which treats all SMs as one 32 × 8448 array. LLM prefill/decode attention kept its 0.40 orientation, so PV (m = q·g, k = ctx, n = head_dim = 128) used 128 of 8448 columns: 66× its FLOP time. QK + PV ran at ~34× ideal at every length (4k–131k). It was not a physical effect: no DRAM materialisation of the scores and no KV re-read term was involved (t_dram 0.70 s at 131k).
- 「来源」 FlashAttention-2 (arXiv 2307.08691) parallelises over (batch, head, query block), one CTA per SM, and reaches 73 % of peak on A100 and ~35 % on H100. FlashAttention-3 (arXiv 2407.08608) reaches up to 740 TFLOPS BF16 forward on H100 (75 %). Measured kernels therefore sit at 1.3–2.9× the FLOP bound, not 34×.
- Fix (default change): LLM attention now also costs the transposed product (Sᵀ = K·Qᵀ, Oᵀ = Vᵀ·Pᵀ) and keeps the cheaper one. This is the same freedom DiT/encoder attention (`orient`) has had since 0.40. GEMMs and the calibrated wide schedule are unchanged. `evaluate.ATTN_ORIENT` / `attn_orient(False)` reproduce ≤ 0.171.
- Now 「推导」: attention is 1.00–1.19× its FLOP time on every reference chip and preset at 4k–131k (new standing invariant `tests/test_core_172.py`, band [0.9, 3]). It is optimistic against FA3 (1.3×) and FA2-on-H100 (2.8×). The sourced `fmt_ceiling` "attn:bf16" stays opt-in.
- TTFT impact, old → new (H100-SXM reference, batch 1):

  | model | context | layout | TTFT before | TTFT after |
  |---|---|---|---|---|
  | llama-3.1-8b | 8k | 1 card | 0.81 s | 0.23 s |
  | llama-3.1-8b | 32k | 1 card | 10.4 s | 1.14 s |
  | llama-3.1-8b | 131k | 1 card | 156 s | 8.0 s |
  | evo2-7b | 8k | reference | 0.503 s | 0.413 s |
  | evo2-7b | 131k | 1 card | 30.4 s | 7.3 s |
  | evo2-7b | 1M | tp8 × cp8 | 25.9 s | 2.78 s |
  | evo2-7b | 1M | fused cp8 | 194 s | 9.0 s |

- External validation:
  - `default` (attention already on cta) is unchanged.
  - The headline `catalog_kernel` on the R4 hold-out (TRT-LLM 1.2.0) moves from 0.892 / 26.6 % to 1.016 / 16.8 %. The §39 "under-prediction tracks prefill share" pattern was largely this artifact.
  - v1 static 1.16 → 1.19, v1 TTFT 1.59 → 1.66, B200 NVFP4 1.16 → 1.24 (these are pred / measured).
- Re-baselines:
  - R4 bands in test_core_091 / 092 / 093.
  - The 0.171 evo2 8k reference TTFT goes from 0.50315 to 0.41291.
  - Pinned to ≤ 0.171 with `attn_orient(False)`: the 0.65.1 KPI hashes (37 LLMs changed) and the V4 DES-vs-queue bands (test_core_054 / 057), because the faster prefill moves the fixed-λ operating point.
  - `data/ext_validation.json` was regenerated.

## [0.171.0] — Evo 2 long-context prefill: Hyena context parallel, recurrence prefill, fused-FFT what-if
- Context parallel for StripedHyena 2. `layout.cp` now accepts Hyena layers.
  - 「来源」 Zymrael/savanna `use_cp_hyena`, from hyena.py and distributed_a2a.py. Each Hyena layer does an all-to-all `split_to_full` of x1, x2 and v, so every rank holds the full sequence for ch/cp channels and runs the FFT conv at 2S. It then does an all-to-all `full_to_split` of z back.
  - The featurizer short FIR takes (taps − 1) halo columns from the neighbouring rank (`ExchangeOverlappingRegionsCausal`).
  - The MHA layers use the existing 0.88 zigzag ring. Savanna's 40B 1M config uses `use_cp_hyena` with cp 2 / mp 64 (training).
- New `hyena_prefill` = reference | recurrence | fused (CLI `--hyena-prefill`). Default is reference.
  - recurrence 「来源」 vortex `prefill_via_direct_recurrence`: x1v is repeated to (b, ch, L, 16, 2) bf16, then a Python loop over L runs 14 elementwise kernels per token on the fp32 state, with no FFT conv. This lowers peak memory, but at 8k it moves more bytes than the FFT path (716 vs 628 GB) and adds 14·L kernel launches per LR layer.
  - fused (what-if 「假设」): one on-chip kernel per MR / LR FFT conv, with the modal state computed as a direct sum in the same pass and the filter built on chip (as vortex's tiled HCL build does). The rfft spectrum stays on chip while it fits one engine's activation share (SRAM/2/engines). Beyond that, a four-step out-of-core FFT adds one spectrum read and write per extra pass, forward and inverse 「推导」.
- Correction: the stock `compute_filter` materialises (ch, 16, L) fp32 twice. This is now counted in peak memory; 「来源」 the vortex HCL docstring says the intermediate "OOMs evo2_7b at L=131k". The 0.170 8k results are unchanged because the modal-FFT peak there is larger.
- Results (evo2-7b, H100-SXM reference hardware, batch 1):
  - 8k: reference 0.50 s, 628 GB of activation DRAM. fused 0.26 s, MAC-bound; its 51 GB of activation traffic is attention streaming. The fused spectrum fits on chip up to L ≈ 12k.
  - 1M bases:

    | cards | layout | reference | fused |
    |---|---|---|---|
    | 1 | — | 2140 GiB, no fit | 144 GiB, no fit (KV alone ≈ 84 GB) |
    | 8 | cp 8 | 278 GiB, no fit | 33 GiB, fits |
    | 64 | tp 8 × cp 8 | 35 GiB, fits, TTFT 25.9 s | — |

  - In those TTFTs, attention dominates the 1M MAC time. The array time for long-context MHA is about 25× its ideal FLOP time on this hardware model, and llama-3.1-8b at 131k shows the same ratio. That is a pre-existing attention-mapping issue, not new in this release.
- Tests: tests/test_core_171.py (4). test_core_170 now also checks the h-build peak.

## [0.170.0] — Evo 2 / StripedHyena 2 genomics models: hyena SE / MR / LR operators, FFT prefill, modal decode state
- New genomics models `evo2-40b`, `evo2-7b` (1M context) and `evo2-1b` (`evo2_1b_base`, 8192 context), from Arc Institute.
  - 「来源」 Shapes and dtypes come from the PyTorch zip checkpoints on HF `arcinstitute/evo2_*`, read over HTTP range (`scripts/fetch_torch_ckpt.py` now also takes a checkpoint split into `.part0|.part1` parts, as 40B ships). Hyper-parameters come from the inference configs in github.com/ArcInstitute/evo2 (`evo2/configs/evo2-{40b-1m,7b-1m,1b-8k}.yml`). `scripts/build_evo2_release.py` writes the release JSONs.
  - Parameters match the checkpoints exactly (41.118B / 6.582B / 1.108B). Excluded from the count: the stored `filter.t` time index and RoPE `inv_freq` buffers, and the duplicate copy of the tied unembedding that 40B stores.
- Operators 「来源」 Zymrael/vortex `engine.py` (prefill_style fft, as released). Every hyena block starts with an h → 3h projection and a 3-tap short FIR over the 3h channels, computes x1·v, and gates with x2.
  - SE: direct 7-tap depthwise FIR.
  - MR: 128-tap FIR, prefilled with an FFT convolution at 2L (`parallel_fir` takes the FFT path for filters ≥ 128 taps).
  - LR: modal long filter h = Σ R·p^t over 16 real modes, then rfft(h), fft(x1v), ·H, irfft.
  - LR prefill for generation (`prefill_via_modal_fft`): fft(p^t, 2L) per channel × mode, and ifft(X·S, 2L) per sequence × channel × mode, keeping only position L−1.
  - FFT FLOPs 「推导」: 5N·log2 N complex, half that for rfft/irfft. They are counted as vector work.
  - Each torch.fft call or elementwise kernel round-trips its fp32 / complex64 operands through DRAM (`Op.act_x`, streamed). The largest live LR tensor is 2 × b·ch·16·2L·8 B and counts toward capacity.
  - Decode: FIR caches of 2 / 6 / 127 bf16 columns, a 16-mode fp32 modal state, `step_fir` / `step_iir` arithmetic, and a KV cache only in the MHA layers (5 / 32 in 7B).
- Example (evo2-7b, H100-SXM reference hardware, batch 1):
  - Prefill 8192 bases: 0.50 s, DRAM-bound. Activation DRAM is 628 GB vs 31 GB of weights; the modal-FFT state extraction is ≈ 80 % of it (503 GB).
  - Prefill 1M bases does not fit one card: 2140 GiB needed.
  - Decode: 6.3 ms/token. The state is 13.5 MB per sequence and does not grow with context; KV is 80 KB/token.
- `AttnCore.state_bytes()` now computes linear / conv state in one place (memplan, ir); results for existing models are unchanged.
- Catalog: vendor Arc Institute (Evo 2), listed like Carbon, as causal DNA language models. Fingerprints of existing models are unchanged. Tests: tests/test_core_170.py (4).

## [0.169.0] — host L2 prefix tier on radix / multi-turn; card→host write traffic; UI
- The host L2 tier now also applies to the radix prefix tree (0.58) and multi-turn (0.122/0.123, including session mode).
  - Reachable token capacity: write_through max(C1, C2), write_back C1 + C2, with C2 = prefix_host_GB / footprint × path tokens. The Che per-node model runs at the device and the reachable capacity.
  - Host share = token hit at reach − token hit at device. Load cost reuses the 0.168 rule, using whole-path bytes with probability equal to the host token share (same mean bytes; 「假设」 for exposure).
- New: card→host backup writes. 「来源」 SGLang HiCache write policies.
  - Per request, as a fraction of the prefix: write_through backs up every newly computed token (1 − H); write_back demotes L1 evictions, and in steady state L1 admits every token it didn't hold (1 − H1) 「假设」.
  - Writes use the other PCIe direction (full duplex), so they get their own `host_write` capacity: GBps / (write bytes per card).
  - Shown in `cap_req_s` / `util`, and it can be the bottleneck. Write traffic doesn't add to TTFT (asynchronous).
- UI: PD panel inputs 主机 L2 GB / 副本 and L2 策略. The prefix-cache line shows device/host hits, load / exposed / TTFT increase and backup writes. Bottleneck and utilisation now include host read/write. `scripts/ui_smoke.py` extended; passes with 0 console errors.
- Example (dense8b validation tree, 20 GB device, 100 GB host, write_through): prefill token hit 0.360 → 0.526. Backup writes are 107 MB per card per request, capping a prefill replica at 466 req/s.
- Defaults are off, so results are bit-identical and fingerprints unchanged. Tests: tests/test_core_169.py (3).

## [0.168.0] — host-memory L2 tier for the prefix cache (HiCache-style)
- Sanity check of the 0.167 example (no code change). wan2.1-14b at 720P: 81 frames → latent 21×90×160 → 75600 tokens, d 5120, FFN 13824, 40 layers, CFG 2.
  - Per sequence and layer: linear 2.26e13 MAC, self-attention 2·L²·d = 5.85e13, cross-attention 4e11.
  - Per step: ×40 layers ×2 sequences ×2 = 1.304e16 FLOP. Model reports 13,047 TFLOP.
  - Default chip: 50176 MAC × 1 GHz = 100.35 TFLOPS bf16 → 130 s ideal. Model MAC time is 144.9 s (89.7 % efficiency); stage time 145.6 s. Not a units bug.
  - Order-of-magnitude check: TeaCache README, 14B on A800 ≈ 55 min / 50 steps ≈ 66 s/step at 312 TFLOPS peak.
- New mechanism (KV/memory hierarchy): opt-in `pd.prefix_host_GB` (host cache per replica) and `pd.prefix_host_policy` write_through | write_back, on the whole-prefix cache model (prefix_len > 0).
  - 「来源」 SGLang HiCache docs (hicache_design): GPU memory is L1, host memory L2. write_through backs up every device entry, so L1 ⊆ L2 and the union holds max(K1, K2) prefixes. write_back demotes entries on eviction, so the union holds K1 + K2. Prefill overlaps loading layer N+1's KV while layer N computes.
  - Model: union hit rate by Che over the reachable count 「假设」 LRU. Host-hit share = H(union) − H(K1). Host hits skip recompute but load their shard over each card's host link (pd.swap_GBps or workload.host_GBps, 「假设」 50 GB/s). Exposed time = max(t/L, t − TTFT·(L−1)/L), the same rule as the layer-wise PD hand-off.
  - Mean TTFT adds hit_host × exposed. A new `host` capacity entry, GBps / (hit_host · bytes per card) per prefill replica, can become the bottleneck. The colocated comparison uses the same rule.
  - Not modelled: the D2H write-through traffic (opposite PCIe direction), L3 storage, and the radix-tree and multi-turn modes (ignored with a warning).
- Example (qwen3-8b, TP2, 2048-token prefixes, Zipf 1, 100k prefixes, write_back 200 GB): prefill hit 0.572 → 0.601. Load per hit is 151 MB per card, 3.0 ms, exposed 0.08 ms.
- CLI --pd-prefix-host-GB / --pd-prefix-host-policy. Defaults are off, so results are bit-identical and fingerprints unchanged. Tests: tests/test_core_168.py (3).

## [0.167.0] — TeaCache-style step caching for video/image DiT
- New mechanism (outside VLM): skipped denoise steps. Opt-in Workload fields `step_skip` (skipped steps per request) and `step_cache` teacache | teacache_ret.
- 「来源」 ali-vilab/TeaCache, TeaCache4Wan2.1 `teacache_generate.py`:
  - A skipped forward still runs the patch, time and text embedders and the head. Only the block loop is replaced by `x += previous_residual`.
  - One residual is cached per CFG branch (`previous_residual_even` / `_odd`).
  - Steps always computed: teacache (ret_steps = 1·2, cutoff = steps·2 − 2) → first and last step; use_ret_steps (ret_steps = 5·2, cutoff = steps·2) → first 5.
  - The skip test reads only the timestep embedding (`modulated_inp = e / e0`), which is the same for cond and uncond, so both CFG branches skip the same steps.
- Model: denoise −= skip · per_step · (t_forward − t_skip), where t_skip = embedders + head + residual add (a vector op over tokens × hidden). Residual buffer = sequences on the rank × tokens/SP × hidden × act bytes (bf16 「假设」), added to stage 0 DRAM. FLOPs per request count skipped forwards at their reduced cost.
- Not applied, with a warning: PP > 1 (stage behaviour on a skip not modelled), use together with prefix_kv, and non-generation models. step_skip above steps − forced raises an error.
- Why `step_skip` is a user input: the skip count depends on the threshold and the per-model rescale coefficients applied to learned time embeddings, so it can't be derived without the weights. 推导 example from the TeaCache README (14B: 1829 s → 1234 s at thresh 0.14, end-to-end): about a third of the steps skipped. Not used as a default.
- CLI --step-skip / --step-cache. UI: 跳步数 / 跳步规则 inputs (values above the cap are flagged in the browser and not sent) and a 跳步缓存 KPI. `scripts/ui_smoke.py` extended; 0 console errors.
- Defaults are off, so results are bit-identical and fingerprints unchanged. Tests: tests/test_core_167.py (4).

## [0.166.0] — Kimi K2.5 / K2.7-Code video chunks; UI smoke fixes
- UI (Playwright smoke of the 0.165 video and host fields): switching to a VLM without a modelled video processor (e.g. GLM) while `videos` > 0 posted an eval that returned 400 (console error).
  - The video inputs are now `video-only`: shown only when the catalog's new `vision.video` flag is true, and `videos` is cleared otherwise.
  - Video-only requests no longer show an image KPI reading "0 张/请求 … 每卡 128 张". The TTFT token line now says 视觉（图像 + 视频） token when video is present.
  - The smoke script is `scripts/ui_smoke.py` (optional, needs Playwright; not part of the suite). After the fixes: 0 console errors, all KPIs render.
- New mechanism: Kimi video chunks, a different design from Qwen's.
  - 「来源」 moonshotai/Kimi-K2.5 and Kimi-K2.7-Code `preprocessor_config.json` media_proc_cfg: temporal_merge_kernel_size 4, in_patch_limit_each_frame 4096, in_patch_limit_video null / 655360, patch_limit_on_one_side 512.
  - 「来源」 Both repos' `config.json`: video_attn_type spatial_temporal, merge_type sd2_tpool.
  - 「来源」 kimi_k25_vision_processing.split_video_chunks + media_utils.navit_resize_video (frames cut into chunks of 4, last one shorter, per-frame patch limit) and modeling_kimi_k25 (cu_seqlens span t·h·w = joint attention over the chunk; tpool_patch_merger takes the mean over t before the merger). The two repos' vision code is identical; they differ only in cache helpers.
  - Effect: tokens per chunk = (h/28)·(w/28), independent of frames per chunk. The encoder runs one pass per chunk over 4·h·w patches, so attention is quadratic in the chunk. Example: kimi-k2.5, 32 frames at 1280×720 → 1204×700, 8 chunks × 1075 = 8600 tokens. The chunk pass costs 2.1× the FLOPs of 4 separate frames.
  - Kimi-K3 is not modelled (own kimi_k3 processor; no video_attn_type in its config) and still refuses `videos`.
  - `video_cap` doesn't apply to Kimi (reported as cap false). Chunk timestamp and media-marker text tokens are not counted.
- `ImageGrid.tframes` (default 1) and `VideoGrid.tail` added. Defaults are bit-identical; fingerprints unchanged.
- Tests: tests/test_core_166.py (4).

## [0.165.0] — VLM video input (Qwen3VLVideoProcessor models); video + host fields in the UI
- New mechanism: video clips per request. Opt-in Serving fields `videos` (0…16), `video_frames` (2…768, default 32 「假设」 = 16 s at the processor's 2 fps), `video_w`/`video_h` (default 1280×720 「假设」), `video_cap` (qwen-vl-utils per-frame cap, which transformers makes opt-in via cap_pixels_per_frame). Applies to the four models whose repos ship a Qwen3VLVideoProcessor: Qwen3.5-397B-A17B, Qwen3.8-27B, Qwen3.8-Flash-Next, Intern-S2-397B. Any other model raises an error when `videos` is set; a model with no vision tower gets an "ignored" warning.
- 「来源」 Each repo's video_preprocessor_config.json: size shortest_edge 4096 / longest_edge 25165824, patch 16, temporal_patch 2, merge 2. transformers qwen3_vl/video_processing_qwen3_vl.py: smart_resize bounds t_bar·h·w of the whole clip, factor = patch·merge = 32, grid_t = ⌈F/2⌉, fps 2, min/max frames 4/768, cap = max(min(768·32², longest/F), ⌊1.05·shortest⌋)·F. These are implemented verbatim. transformers vision_utils.get_vision_attention_seqlens (merge_temporal=False for Qwen3-VL) splits cu_seqlens per temporal slice, so the encoder runs one tower pass per slice: attention covers h·w patches, not the clip. The patch-embed Conv3d still spans 2 frames.
- Effect: tokens per video = grid_t · (ph·pw/4). They join prompt/ctx (KV, prefill length, PD) through the existing expansion marker. Encoder cost = images + videos·grid_t slice passes, summed in `vision.s`. `vision.video` reports slices, resized px, tokens, s and TFLOP. Example (qwen3.8-27b, 32 frames at 1280×720): 640×1152, 16 slices × 720 = 11520 tokens. Not counted: Qwen3-VL's per-slice "<t seconds>" timestamp text tokens (tokenizer-dependent, a few tokens per slice).
- Defaults are off, so results are bit-identical and fingerprints unchanged. Video CLI flags: --videos / --video-frames / --video-size / --video-cap. Sweep paths: serving.videos / serving.video_frames.
- UI: video inputs plus a 视频编码 KPI. The 0.164 host fields (host_step_us / host_seq_us / host_overlap / host_steps) are now editable, with a 主机开销 KPI from `summary.host`.
- Tests: tests/test_core_165.py (5).

## [0.164.0] — dram_bw sizing honours the DMA cap; opt-in host/runtime step overhead
- Fix (AGENTS §114): `dram_bw` sizing now reports the delivered bandwidth, min(memory effective GB/s, opt-in DMA issue limit), the same rule as `System.dram_GBps`. With the DMA limit binding, "current" was previously the memory's figure and overstated what is delivered.
- New mechanism (AGENTS §73 0.136 open gap: unmodelled host/scheduling time per step). Opt-in Scenario fields `host_step_us` (fixed per engine step), `host_seq_us` (per sequence on a DP rank), `host_overlap` serial|async, `host_steps` (multi-step scheduling, decode only). host = (step + seq × sequences/rank) µs, ÷ host_steps for decode. Serial: device + host; async: max(device, host). TPOT / TTFT / throughput follow, `Result.host` reports device / exposed / host_bound, and eval JSON carries it. LLM phases only.
- Grounding (qualitative, no adopted numbers): vLLM v0.6.0 blog (2024-09-05) — Llama-3-8B on 1×H100 in v0.5.3: 38 % GPU execution, 29 % scheduling + input preparation, 33 % HTTP server; fixes are multi-step scheduling and async output processing. WukLab (Srivatsa et al., 2024-09-10) — vLLM scheduling up to half of iteration time, grows with requests per batch; SGLang ≤ 18 %. Neither gives a portable µs figure (absolute values are figure-only and runtime-, CPU- and version-specific), so there is no default and no preset. All four fields default to off, which is bit-identical (fingerprints unchanged).
- CLI flags --host-step-us / --host-seq-us / --host-overlap / --host-steps (test_core_087 audit: every scenario field must have a CLI flag).
- Tests: tests/test_core_164.py (3).

## [0.163.0] — sizing adversarial review: lanes→dequant coupling fix
- Bug: the `lanes` sizing setter changed `vector_lanes` only. The dequant path defaults to the vector width (`formats.dequant_elems_per_cycle` None → lanes, mapping.py), so shrinking the vector unit also silently cut dequant throughput on the default chip. The lanes setter now pins the dequant rate to its current value. Latent on the sample sets: the vector width binds (TTFT) before dequant does (TPOT), so the minima, joint points and mm² are unchanged (qwen3-8b+gpt-oss-20b / wan2.1 / esm2-650m: 30.9 / 30.5 / 34.5 mm² at n7; qwen3-8b-awq lanes 1382, dequant 753). It would matter for any chip or model where dequant binds near the vector width.
- Standing guard: tests/test_core_163.py checks that every SIZING setter moves only its own resource, on every chip preset at ×0.5 and ×0.13.
- Hostile review `val5/v0163/sizefuzz.py`: 48 random cases (chip 100T / 1P / H100-like × 6 model sets across llm / gen / protein × density presets × unpriced modes). It independently re-checks that each single-resource min passes and its fail point fails, that the joint point holds and sits between single-resource min and current, setter isolation, and binding consistency: 0 violations.
- Known, not changed: with the opt-in DMA issue limit (`chip.dma_engines`), `dram_bw` reports the memory's effective GB/s rather than min(mem, DMA), so its "current" can overstate the delivered bandwidth.
- No default changed; fingerprints unchanged.

## [0.162.0] — priced vector lanes / dequant in sizing; safer unpriced handling
- New budget densities `mm2_per_klane` / `mm2_per_kdq` (per 1024 lanes / dequant elements per cycle), in every density preset, used by insight sizing only. They are not added to the die-area proxy, whose `mm2_fixed` already covers the vector unit, so area outputs and fingerprints are unchanged. No public NPU floorplan splits out the vector unit (TPUv4i ISCA 2021 Fig. 6 gives only CMEM 28 % and MXU 11 %). So each node's value = that node's MAC density × an operator-area ratio from Horowitz's 45 nm table (ISSCC 2014; area column via Sze et al. / MIT EEMS slides, 二手). Lane = FP32 mult + FP32 add = 11884 µm² vs bf16 MAC = FP16 mult + FP32 add = 5824 → ×2.0405. Dequant = FP16 mult + 8b add = 1676 → ×0.2878. Labelled 推导/二手, datapath-only lower bounds. n7: 1.403 mm²/klane, 0.198 mm²/kdq.
- DRAM bandwidth stays unpriced: sizing scales `mem_eff`, which is not a PHY change; PHY area per pin is already covered by the 0.149 PHY densities.
- Unpriced resources: new `unpriced` mode, default "last". Unknown area is no longer treated as free: unpriced back-off steps rank after every improving priced step. "free" keeps the 0.161 behaviour. Available in the API (`"unpriced"`) and the UI select. Reports `unpriced_mode` / `unpriced`.
- Fix: cores mm² converted R·C per core with /1000; densities are per 1024 MACs (as in `area_estimate`), now /1024 (cores mm² −2.4 %).
- Re-baseline, n7_tpuv4i joint priced mm² saved, 0.161 → 0.162 (default "last"): qwen3-8b+gpt-oss-20b 30.3 → 30.9; wan2.1 42.0 → 30.5 (42.5 with "free": the 0.161 figure came from giving back unpriced DRAM bandwidth instead of SRAM); esm2-650m 32.6 → 34.5. Single-resource mm² (esm2): lanes 4.11, dequant 0.67, SRAM 19.69, cores 12.63.
- Tests: tests/test_core_162.py (4); test_core_149 status set gains 推导/二手; test_core_158 / 161 use /1024.

## [0.161.0] — area-aware joint sizing; binding-resource summary
- Joint sizing, area-aware: when densities are set, a back-off step is ranked by the worst-case gain it recovers per mm² given back. A resource with no density costs 0 mm² and so ranks first among steps that help; its area is unknown, not free, and each joint row is marked `priced`. The 0.160 greedy path is always kept alongside a beam of 3. The final point maximises priced mm² saved (else total shrink Σlog(cur/v)) among probed passing points only (asserted). Exposes `search`, `greedy_mm2_saved` / `greedy_shrink`, `rank`, `evals`.
- Default chip, n7_tpuv4i densities, priced mm² saved 0.160 → 0.161: qwen3-8b+gpt-oss-20b 18.1 → 30.3, wan2.1 28.0 → 42.0, esm2-650m 17.4 → 32.6. The joint point keeps SRAM / cores at their single-resource minima and gives back dequant / lanes instead. Without densities the beam beats greedy on total shrink: 2.44 vs 2.34, 3.23 vs 2.64, 4.73 vs 4.34.
- `binding` in every sizing report and in the UI: resources with no single-resource headroom (and what limits them), the most over-provisioned resources, and the resources that only bind in combination (joint back-offs).
- No default changed; fingerprints unchanged. Tests: tests/test_core_161.py (3).

## [0.160.0] — joint sizing (interacting minima); robust test runner
- `sizing_report(..., joint=True)` / `{"joint": true}` / UI checkbox「联合缩减」. The single-resource minima interact: vector width and dequant share elementwise time, and less SRAM pushes traffic onto DRAM. So it first applies every minimum together and reports whether the ±1 % contract still holds (worst gain, which model/metric breaks). If not, it backs off one resource per step, picking the geometric step toward the current value with the best worst-case gain, until the contract holds; the result is a probed passing point. Default chip: applying all minima together fails for every sample set — esm2-650m ×0.744 (cores 10→13, lanes 581→2274, SRAM 41.5→51.5 MiB), wan2.1 ×0.871 (SRAM 16→32, BW 124→154 GB/s, dequant 317→1066), qwen3-8b+gpt-oss-20b ×0.983 (lanes 1643→2427, SRAM 29→43). So adding up the single-resource savings overstates the area an architect can actually cut. Greedy, not proven area-optimal. No default changed; fingerprints unchanged.
- `scripts/run_tests.py` robustness: the 0.159 silent death was the box's VM fork at 06:08:25 JST (dmesg: "crng reseeded due to virtual machine fork"), which killed the session-attached run (SIGKILL, which cannot be caught). The runner now writes `scripts/.run_status.json` (pid, done/pending, state) after every file. The next run warns when a previous run died without a summary, and `--resume` skips the files that already passed. SIGTERM/SIGHUP/SIGINT kill the children and print a summary with the pending files; any exception prints CRASHED plus the status. A child with no result line is reported with its exit code.
- Tests: tests/test_core_160.py (4).

## [0.159.0] — bit-identical queue_report speedup; stdlib-only parallel runner
- pdqueue: `_victim_wait_law` memoised on every input it reads (1596 calls / 428 distinct per recompute KV report; the returned law is read-only and its surv memo is pure); `_sat_evict_gamma` memoised on exact arguments (871 / 269); `_gamma_q` series drops `abs()` on quantities that are always positive. bulkq: `_chain` LRU 512 → 8192 (1205 calls / 923 distinct, so it was evicting).
- Exactness: full `capture_ctx` report (every float compared by repr) is IDENTICAL before and after for 5 cases (recompute 0.85 / 0.6 / cv0, swap 0.85, wait 0.85). Wall time: recompute 0.85 152 → 93 s, swap 152 → 96 s, rec 0.6 148 → 90 s, wait 65 → 57 s. Defaults and fingerprints unchanged.
- What's left in this path: `_gamma_q` evaluations inside the eviction quadrature, plus `bulkq._tagged` / `_stationary` / `_wait_hists` (already hand-tuned for bit-identity in 0.65–0.112). Going further would mean changing summation order or quadrature, which is not bit-identical, so it was not adopted.
- `scripts/run_tests.py` is now stdlib-only: each file runs through `tests/run_tests.py --one` (the project's zero-dependency harness) in parallel subprocesses, longest-first, with a per-file TIMEOUT. pytest is no longer needed anywhere.

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
