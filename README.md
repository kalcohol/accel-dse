# accel-dse

> 推理加速器设计空间探索（LLM · 视频生成 DiT · 蛋白质）· Analytical design-space exploration for inference accelerators

[English](README.en.md) · [建模说明](docs/MODEL.md) · [更新日志](CHANGELOG.md) · [MIT](LICENSE)

**accel-dse** 用可核对的解析模型回答流片前的问题：给定一个模型（LLM / VLM，或视频生成 DiT、蛋白质语言模型；按官方发布的权重与 dtype）、一种数据通路映射、片上 SRAM、存储器（HBM / LPDDR）和卡数，最好的并行布局与 batch 是什么，瓶颈在哪里，结论对假设有多敏感。零第三方依赖，Python ≥ 3.10。

## 它做什么

- **模型按发布建模**：从 HF `config.json` 与 safetensors 头（不下载权重）得到逐角色的参数量与存储 dtype（bf16 / fp8 block / MXFP4 / int4 AWQ 等），与发布总量偏差 ≤0.5%；官方量化版是独立条目。每个模型带三轴标签：来源（官方 / 镜像）× 覆盖（完整 / 部分 / 架构代理，附逐项的近似之处）× dtype；目录按厂商 → 系列排列（同一厂商的 LLM、VLM、视频与蛋白质模型在一起）。
- **视频生成与蛋白质（0.41 – 0.43）**：Wan2.1-T2V（1.3B / 14B）、CogVideoX（2b / 5b）的 DiT 去噪主干（全 3D 注意力，每步按 CFG 前向 1–2 次）与 ESM-2（650M / 3B）编码器可评估：单段延迟、每帧延迟、帧/s/卡，或批延迟、序列/s/卡、残基/s/卡；Ulysses 序列并行（SP）与 CFG 并行（DP）是布局维度；文本编码器与 VAE 在 0.44 前未建模。0.42 增加 Wan2.2-T2V-A14B（双专家，每步激活一个）、HunyuanVideo（双流 / 单流，全 3D 注意力）、LTX-Video 2B、Mochi 1、Open-Sora STDiT3（发布即为分解时空注意力）与 MiniMax-H3（视频 + 音频 + 文本联合序列）。0.43 接入蛋白质结构预测：ESMFold、AlphaFold 2（官方 JAX 参数 model_1_ptm）、OpenFold、Boltz-1、Protenix——pair 表示、三角乘法 / 三角注意力、MSA 行 / 列注意力、外积均值、IPA、扩散模块按检查点张量形状建模，recycle / 扩散步数 / 样本数可调（MSA / 模板检索未建模，覆盖「部分」；ESMFold「完整」）。AlphaFold 3 权重需申请，仍标「暂未接入 v2」。
- **视频整条 pipeline（0.44）**：文本编码器（umT5 / T5-XXL / LLaVA-Llama-3 + CLIP / Qwen3-VL 文本塔）与 VAE 解码器（含 HunyuanVideo tiling、Open-Sora 逐帧 SD-VAE、H3 ViT 解码器与音频 VAE）按发布检查点的张量头建成算子图，时间与存储默认计入（与去噪串行「假设」，见 docs/MODEL.md §11.4），视频覆盖全部为「完整」；`workload.pipeline = false` / CLI `--dit-only` 只看 DiT。另增激活 dtype what-if（Web「激活 dtype」/ CLI `--act fp32`），评估结构模型参考实现 fp32 激活的流量代价。
- **多卡结构模型与组件放置（0.45）**：结构模型支持 DAP（FastFold 动态轴并行，占用布局的 SP 维）：pair / MSA / 模板网格按残基轴切分，三角乘法 / 偏置 / 外积均值的 all-gather 与行 ↔ 列 all-to-all 逐核计入（AF2 8 卡 7.8×，扩散为主的 Protenix 1.8×）。视频组件放置 `workload.placement`：常驻 / 文本编码器 FSDP 分片（Wan `--t5_fsdp`）/ 顺序卸载（diffusers `enable_model_cpu_offload`、Wan `--offload_model`），默认 auto 取第一个放得下的——64 GiB LPDDR 上 Wan2.1-14B、Wan2.2-A14B、MiniMax-H3 可放下（每请求 1–2.5 s 主机重载）。可选 VAE 分块解码（`--vae-tiling`，CogVideoX / Mochi 按 diffusers 默认 tile）。请求 TFLOP 改为整个副本的有用 FLOPs（此前多卡布局按单 rank 计）。
- **映射是设计变量**：输出驻留（OS）、权重驻留（边缘加载 / 宽面广播）、OS + GEMV 单元、可重构，逐算子计算 MAC 界与 SRAM 供数界；芯片不原生支持的格式计入反量化开销。
- **逐 rank 算子图**：TP / PP / 注意力 DP / EP / ETP，单卡就是全 1 布局，没有第二条路径。
- **存储规划与调度**：权重 / KV 的 SRAM 驻留、staging、逐 stage 容量；每级 `max(MAC/FEED, VECTOR, DRAM, LINK) + SYNC`，绑定瓶颈与有效 MAC 比例直接给出。
- **精确搜索**：每个布局在 TPOT SLO 下的最大 batch（分支定界，已用暴力枚举核对），decode 或含 prefill 的 goodput 目标，DP prefill 的 TTFT 标记，排名在「假设」扰动下是否稳定。
- **校验**：参数对照发布、H100 类配置的趋势区间、与 GenZ 的对照，见 [建模说明](docs/MODEL.md)。

## 快速开始

```bash
git clone https://github.com/kalcohol/accel-dse.git
cd accel-dse
python3 tests/run_tests.py          # 自检（零依赖；也可 python3 -m pytest tests/ -q）
python3 -m accel_dse serve          # Web 工作台 → http://127.0.0.1:8765
```

命令行：

```bash
python3 -m accel_dse models                                   # 模型目录与三轴标签
python3 -m accel_dse eval --model qwen3-8b --best-batch       # 单点评估（默认 100T 芯片 + LPDDR5X）
python3 -m accel_dse compare --model deepseek-v3 --cards 8 \
    --mem hbm3e_8s_12h24g_9200 --ctx 4096                     # 每种映射的最佳布局
python3 -m accel_dse search --model qwen3-32b --cards 8 --objective goodput --mem hbm3e_8s_12h24g_9200
python3 -m accel_dse stability --model qwen3-32b --cards 8 --mem hbm3e_8s_12h24g_9200
python3 -m accel_dse eval --model wan2.1-1.3b --steps 20 --sp 2   # 视频：单段 / 每帧延迟（480P、81 帧）
python3 -m accel_dse eval --model esm2-650m --seq-len 1022    # 蛋白质：批延迟、序列/s
python3 -m accel_dse eval --model boltz-1 --msa 1024 --samples 5   # 结构预测：主干 / 扩散 / 置信度，TFLOP/序列
python3 -m accel_dse validate                                 # 趋势区间 + GenZ 对照
```

所有子命令都支持 `--json`。HTTP API（`/api/eval`、`/api/compare`、`/api/layouts`、`/api/stability`、`/api/sweep`、`/api/pareto`、`/api/memory`）接受与命令行相同的场景描述，严格解析（拒绝未知字段与 NaN / Infinity）。

## 目录结构

```
accel-dse/
├── accel_dse/
│   ├── core/          # 场景、模型规格、算子图、映射、存储规划、调度、搜索、稳定性、校验
│   ├── api.py         # JSON API（Web 与 CLI 共用）
│   ├── serve.py web/  # 本地 Web 工作台
│   ├── cli.py
│   ├── mem_catalog.py # HBM / LPDDR 结构化目录（带来源标签）
│   └── data/          # 发布摘要（releases/）、模型目录、GenZ 参考值
├── docs/MODEL.md      # 建模方法、公式、校验与范围
├── docs/research/     # 存储规格调研与来源
├── scripts/           # 发布摘要抓取、GenZ 参考值生成
└── tests/
```

## 免责声明

硬件参数（频率、阵列几何、SRAM 端口、DRAM 效率、链路带宽与同步时延、MAC 效率）都是标注为「假设」的可调输入，未经硅片标定；结果用于方案之间的相对比较与趋势判断，不是性能承诺。模型参数来自公开发布，不代表任何厂商的性能声明。

## 反馈

欢迎通过 [GitHub Issues](https://github.com/kalcohol/accel-dse/issues) 报告问题或指出建模错误，附上命令行或 API 请求体便于复现。

## License

[MIT](LICENSE) © 2026 accel-dse Project
