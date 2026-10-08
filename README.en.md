# accel-dse

> Analytical design-space exploration workbench for inference-only NPU / ASIC

[中文](README.md) · [Model notes](docs/MODEL.md) · [Changelog](CHANGELOG.md) · [MIT](LICENSE)

**accel-dse** is a **hand-checkable** analytical model plus a local web workbench for **DSE (Design Space Exploration)** of inference-only NPUs / ASICs. Before tape-out, it puts "compute × on-chip SRAM × HBM/LPDDR × multi-chip parallelism × workload" on a single MetricsCard so you can see the trade-offs between latency, throughput, capacity and bandwidth walls. Every intermediate quantity (FLOPs, DRAM bytes, utilization, collective bytes) can be exported and recomputed by hand. Zero third-party dependencies, Python ≥ 3.10.

## What it models

- **Workloads**: LLM (dense / MoE / MLA / GQA, with dimensions for dozens of public HF configs built in), video DiT (N_denoise × T² attention), protein (encoder + L² pair representation).
- **Single chip**: output-stationary PE array (explicit M=1 decode utilization), three-way SRAM partition (external traffic derived from tiling + capacity — no hand-entered hit rate), HBM3/3E/4/4E or LPDDR5/5X/6 (including SOCAMM2 / LPCAMM2; a structured memory catalog with provenance tags).
- **Multi-chip**: TP / PP / EP (including TP×EP expert sharding and attention DP), C2C (chip-to-chip) collectives plus exposed sync latency, IB/RoCE KV fabric, PP decode micro-batching, speculative decoding / MTP.
- **Metrics**: TTFT / TPOT (TTFC for video, time/seq for protein), compute vs memory bandwidth-wall breakdown, capacity / OOM, scale efficiency, throughput–interactivity **Pareto** front and **SLO goodput**; an optional energy / cost stub (user-supplied assumed values).

The method, formulas and all key assumptions are in the **[model notes](docs/MODEL.md)** (Chinese) — critique welcome, item by item.

## Quick start

```bash
git clone https://github.com/kalcohol/accel-dse.git
cd accel-dse
python3 tests/run_tests.py          # self-test (zero deps; or python3 -m pytest tests/ -q)
python3 -m accel_dse serve          # local web workbench
# open http://127.0.0.1:8765 in a browser
```

Common CLI commands:

```bash
# One configuration → MetricsCard (8 chips, 8 HBM3E stacks per chip)
python3 -m accel_dse workbench --model deepseek-v3 --chips 8 --mem-type HBM3E --mem-count 8

# Throughput–interactivity Pareto + SLO goodput (all TP×PP×EP layouts × batch up to KV capacity)
python3 -m accel_dse pareto --model qwen3-32b --chips 8 --mem-type HBM3E --mem-count 8 \
    --slo-ttft-ms 2000 --slo-tpot-ms 50 --csv out/pareto_qwen3-32b_8.csv

# Parallel-layout matrix / catalogs / offline report
python3 -m accel_dse workbench-parallel --model series/moe-active13b --chips 8
python3 -m accel_dse list-series --product      # public HF model dimensions
python3 -m accel_dse list-packages              # memory catalog (HBM / LPDDR)
python3 -m accel_dse report --out out/report.html
python3 -m accel_dse --help                     # all subcommands
```

Optional: `pip install 'accel-dse[web]'` to use FastAPI + uvicorn as the backend (default is stdlib `http.server`). The web UI is in Simplified Chinese, with technical terms kept in English.

## Project layout

```
accel-dse/
├── accel_dse/            # Python package
│   ├── npu.py memory.py traffic.py evaluate.py   # single-chip engine (PE / SRAM / DRAM / traffic)
│   ├── scaleup.py workloads.py                    # multi-chip TP/PP/EP, video / protein workloads
│   ├── workbench.py pareto.py econ.py             # product-level MetricsCard, Pareto/goodput, energy/cost stub
│   ├── mem_catalog.py package_ranges.py catalog.py series.py   # memory / compute / model catalogs
│   ├── serve.py web/                              # local web workbench (API + static UI)
│   ├── cli.py report.py                           # CLI, offline HTML report
│   └── data/                                      # model-dimension catalog (baked from public HF configs)
├── docs/MODEL.md         # modelling method, formulas, assumptions and limitations
├── docs/research/        # memory-spec research and sources (JEDEC / vendor material)
├── examples/             # hand-check walkthrough (handcheck.md), calibration / energy example JSON
├── scripts/              # model-catalog bake script
├── tests/                # zero-dependency tests (run_tests.py)
└── out/                  # example outputs (report, CSV)
```

## Disclaimer

All numbers in this tool are **analytical estimates or explicitly labelled assumptions** (marked "假设 / assumed" in the UI and CSV) — **not measured silicon**, and not calibrated against post-silicon data. Frequency, bandwidth efficiency, C2C / fabric bandwidth, power and price are all adjustable assumptions. Use the results for relative comparison between design points and for trend analysis; do not quote them as product specifications. Model dimensions come from public configuration files and do not represent any vendor's performance claims.

## Feedback

Bug reports, modelling corrections and feature requests are welcome via [GitHub Issues](https://github.com/kalcohol/accel-dse/issues). Including a `?c=` share link (the "复制链接" / copy-link button in the web top bar) or the CLI command helps reproduce the issue.

## License

[MIT](LICENSE) © 2026 accel-dse Project
