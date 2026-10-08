# MetricsCard — `illustrative_27B`

| field | value |
|---|---|
| chips | 8 |
| tp / pp / ep | 8 / 1 / 1 |
| mem | HBM3E 8×12H×24Gb @9200 · 9.42 TB/s 可用 · 288 GB · 厂商量产 (6594.6 GB/s eff) |
| mem raw / payload GB/s | 9420.8 / 9420.8 |
| mem tag | vendor_shipping |
| peak TOPS | 100.35 |
| domain | llm |
| TTFT_ms | 41.9199 |
| TPOT_ms | 4.5201 |
| TTFC_ms | 0.0000 |
| frames_per_s | 0.0000 |
| time_per_seq_ms | 0.0000 |
| pair_bytes | 0 |
| capacity_needed_GB | 8.4121 |
| capacity_GB | 288.0000 |
| wall | compute |
| util | 0.0149 |
| oom | False |
| t_compute_ms | 4.2801 |
| t_dram_ms | 1.0909 |
| t_c2c_ms | 0.0057 |
| t_sync_ms (exposed, α=3 µs × 80) | 0.2400 |
| attn_parallel / kv_replication | tp / 1 |
| moe_shard / decode_mb_eff (micro-batch) | - / 1 (1) |
| spec_k / accept / E[tok/step] | 0 / 0 / 1.0000 |
| TPOT_step_ms / t_draft_ms | 4.5201 / 0.0000 |
| t_fabric_ms | 0.0000 |
| t_bubble_ms | 0.0000 |
| bytes_W | 7183269888 |
| bytes_KV | 10506240 |
| bytes_coll | 2293760 |
| n_cores × tops/core | 16 × 6.272 |
| PE | 56x56×16 |
| SRAM_MiB | 64.0 |
| prompt / ctx / batch | 512 / 512 / 1 |
| scale_efficiency | 0.946904 |
| non_gemm_overhead | 0 |
| dtype_mac_factor | 1 |
| speedup | 7.57523 |
| t_single_primary_ms | 34.2411 |
| scale_metric | TPOT |

## Notes
- scale vs chips=1: t_single=34.2411 ms → speedup=7.5752× eff=0.9469 (chips=8)

## Assumptions
- illustrative/placeholder dense ~27B-class GQA (NOT a vendor checkpoint)
- chip_count=8 → tp=8, pp=1, ep=1 (default mapping)
- NPU cores×tops_per_core → n_engines=n_cores, near-square PE/core (16×6.25 T → 56x56×16) [assumed mapping]
- DRAM preset HBM HBM3E: 8×1024b (stack) × 9.2 GT/s × eff=0.7 | raw=9420.8 GB/s eff=6594.6 GB/s cap=309.2 GB (assumed) [assumed]
- exposed sync: α=3 µs per collective × (TP 2 all-reduce/layer, EP 2 all-to-all/layer, PP 1 send/stage) × (1−overlap 0) added outside max(compute, mem, c2c) [ASSUMED α; v0.29]
- LM head counted every decode step: 0.263 GB/card read + GEMM (vocab-parallel under attention TP; replicated head read in full per rank under attention DP; PP stages balanced around the head → amortised /pp) [v0.31, assumed]
- scale_efficiency=(t_single/t_multi)/chips on domain primary (TPOT); ideal 1.0; ignores host/NIC non-ideal; C2C/PP-bubble/EP already in t_multi [derived]

```
=== MetricsCard | llm | illustrative_27B | chips=8 tp/pp/ep=8/1/1 | HBM | peak=100.35 TOPS | mem_eff=6594.6 GB/s ===
  TTFT=41.9199 ms  TPOT=4.5201 ms/tok  wall=compute  util=0.0149  oom=False
  workload S=512/ctx=512/B=1
  breakdown: compute=4.2801 ms  dram=1.0909 ms  c2c=0.0057 ms  sync=0.2400 ms  fabric=0.0000 ms  bubble=0.0000 ms
  bytes: W=7183269888  KV=10506240  coll=2293760
  npu: cores=16 × 6.272 T/core → PE=56x56×16  SRAM=64.0 MiB
  scale: eff=0.9469  speedup=7.5752×  (metric=TPOT  t_single=34.2411 ms)
  notes: scale vs chips=1: t_single=34.2411 ms → speedup=7.5752× eff=0.9469 (chips=8)
  assumptions:
    - illustrative/placeholder dense ~27B-class GQA (NOT a vendor checkpoint)
    - chip_count=8 → tp=8, pp=1, ep=1 (default mapping)
    - NPU cores×tops_per_core → n_engines=n_cores, near-square PE/core (16×6.25 T → 56x56×16) [assumed mapping]
    - DRAM preset HBM HBM3E: 8×1024b (stack) × 9.2 GT/s × eff=0.7 | raw=9420.8 GB/s eff=6594.6 GB/s cap=309.2 GB (assumed) [assumed]
    - exposed sync: α=3 µs per collective × (TP 2 all-reduce/layer, EP 2 all-to-all/layer, PP 1 send/stage) × (1−overlap 0) added outside max(compute, mem, c2c) [ASSUMED α; v0.29]
    - LM head counted every decode step: 0.263 GB/card read + GEMM (vocab-parallel under attention TP; replicated head read in full per rank under attention DP; PP stages balanced around the head → amortised /pp) [v0.31, assumed]
    - scale_efficiency=(t_single/t_multi)/chips on domain primary (TPOT); ideal 1.0; ignores host/NIC non-ideal; C2C/PP-bubble/EP already in t_multi [derived]
```
