# accel-dse

> 推理加速器设计空间探索工作台 · Analytical DSE workbench for inference-only NPU / ASIC

[English](README.en.md) · [建模说明](docs/MODEL.md) · [更新日志](CHANGELOG.md) · [MIT](LICENSE)

**accel-dse** 是一个**可手算核对**的解析模型 + 本地 Web 工作台，用于推理专用 NPU / ASIC 的 **DSE（Design Space Exploration，设计空间探索）**：在流片前，把「算力 × 片上 SRAM × HBM/LPDDR × 多芯片并行 × 工作负载」放在同一张 MetricsCard 上比较，看清时延、吞吐、容量与带宽墙之间的取舍。所有中间量（FLOPs、DRAM 字节、利用率、集合通信字节）都可导出、可复算。零第三方依赖，Python ≥ 3.10。

## 建模范围

- **工作负载**：LLM（dense / MoE / MLA / GQA，内置数十个公开 HF config 维数）、视频 DiT（N_denoise × T² attention）、蛋白质（encoder + L² pair）。
- **单卡**：output-stationary PE 阵列（显式 M=1 decode 利用率）、SRAM 三分区（外部流量由 tiling + 容量推导，不需要手填 hit-rate）、HBM3/3E/4/4E 或 LPDDR5/5X/6（含 SOCAMM2 / LPCAMM2；带来源标签的结构化存储目录）。
- **多芯片**：TP / PP / EP（含 TP×EP 专家切分、注意力 DP）、C2C（chip-to-chip）集合通信 + 暴露同步时延、IB/RoCE KV fabric、PP decode 微批、投机解码 / MTP。
- **指标**：TTFT / TPOT（视频 TTFC、蛋白 time/seq）、compute vs memory 带宽墙分解、容量 / OOM、scale efficiency、吞吐–交互性 **Pareto** 前沿与 **SLO goodput**；可选的能耗 / 成本 stub（用户输入假设值）。

方法、公式与全部关键假设见 **[建模说明](docs/MODEL.md)**，欢迎逐条挑错。

## 快速开始

```bash
git clone https://github.com/kalcohol/accel-dse.git
cd accel-dse
python3 tests/run_tests.py          # 自检（零依赖；也可 python3 -m pytest tests/ -q）
python3 -m accel_dse serve          # 本地 Web 工作台
# 浏览器打开 http://127.0.0.1:8765
```

常用 CLI：

```bash
# 单个配置 → MetricsCard（8 芯片、每卡 HBM3E ×8 堆）
python3 -m accel_dse workbench --model deepseek-v3 --chips 8 --mem-type HBM3E --mem-count 8

# 吞吐–交互性 Pareto + SLO goodput（全部 TP×PP×EP 布局 × batch 至 KV 容量上限）
python3 -m accel_dse pareto --model qwen3-32b --chips 8 --mem-type HBM3E --mem-count 8 \
    --slo-ttft-ms 2000 --slo-tpot-ms 50 --csv out/pareto_qwen3-32b_8.csv

# 并行布局矩阵 / 目录 / 离线报告
python3 -m accel_dse workbench-parallel --model series/moe-active13b --chips 8
python3 -m accel_dse list-series --product      # 公开 HF 模型维数
python3 -m accel_dse list-packages              # 存储目录（HBM / LPDDR）
python3 -m accel_dse report --out out/report.html
python3 -m accel_dse --help                     # 全部子命令
```

可选：`pip install 'accel-dse[web]'` 使用 FastAPI + uvicorn 作为后端（默认 stdlib `http.server`）。

## 目录结构

```
accel-dse/
├── accel_dse/            # Python 包
│   ├── npu.py memory.py traffic.py evaluate.py   # 单卡引擎（PE / SRAM / DRAM / 流量）
│   ├── scaleup.py workloads.py                    # 多芯片 TP/PP/EP、视频 / 蛋白负载
│   ├── workbench.py pareto.py econ.py             # 产品层 MetricsCard、Pareto/goodput、能耗成本 stub
│   ├── mem_catalog.py package_ranges.py catalog.py series.py   # 存储 / 算力 / 模型目录
│   ├── serve.py web/                              # 本地 Web 工作台（API + 静态 UI）
│   ├── cli.py report.py                           # CLI、离线 HTML 报告
│   └── data/                                      # 模型维数目录（公开 HF config 烘焙）
├── docs/MODEL.md         # 建模方法、公式、假设与局限
├── docs/research/        # 存储规格调研与来源（JEDEC / 厂商资料）
├── examples/             # 手算核对（handcheck.md）、校准 / 能耗示例 JSON
├── scripts/              # 模型目录烘焙脚本
├── tests/                # 零依赖测试（run_tests.py）
└── out/                  # 示例输出（报告、CSV）
```

## 免责声明

本工具的所有数值均为**解析模型推算或明确标注的假设值**（UI / CSV 中标「假设 / assumed」），**不是实测硅片数据**，也未经硅后标定；频率、带宽效率、C2C / fabric 带宽、功耗与价格均为可调假设。请用于方案间的相对比较与趋势判断，不要当作产品规格引用。模型维数来自公开配置文件，不代表任何厂商对性能的声明。

## 反馈

欢迎通过 [GitHub Issues](https://github.com/kalcohol/accel-dse/issues) 报告问题、指出建模错误或提出需求。附上 `?c=` 分享链接（Web 顶栏「复制链接」）或 CLI 命令能帮助复现。

## License

[MIT](LICENSE) © 2026 accel-dse Project
