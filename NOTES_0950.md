# 0.95 (r29, /workspace/wt-r29, off r28 ba5e897 = 0.94.0)
NPU design-space adversarial review. Harness: val5/npu095/{sweeps,sweeps2,sens}.py
- sweeps (sram/engines/partition/mem/mem_eff/freq; D2D units/2-die; dtype matrix): all monotone; flips explained (MODEL §41).
- randomized monotonicity 3×400 (~890 checked): only capacity-losing mem steps (harness artifact).
- FOUND: FormatSupport.dequant_elems_per_cycle dead knob → honoured (default bit-identical, fp c034bfe3).
- sensitivity table: exec_overlap (highest), mem_eff, vector_lanes (fp8 KV gain 1.20→0.72), mac_eff drive conclusions; acc_kib/gemv/noc/port/d2d units don't.
- usability pitfall: explicit sram_port_Bpc absolute (doesn't scale with engines).
- not done: native fp8 attention option (would remove KV upcast dependence) — candidate for 0.96.
- suite 514/514 green (885 s); fp c034bfe3 unchanged. SHIP-READY (r29, not merged).
