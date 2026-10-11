# 0.94 (r28, /workspace/wt-r28, off main 3533eb9)
Goal: H200 DeepSeek over-prediction (trt-dsr1-h200 1.35x, r4 H200 DS up to 1.53x), principled; then re-eval candidate_variant, adopt if every group improves.
- Shipped 0.93: ff main→3533eb9, /workspace/accel-0930.bundle (sha b91d3ebe…), /workspace/ship-0930.txt (+AGENTS step).
- DeepGEMM public H800 table replay: wide 1.27x, cta 1.61x; memory side matches eff 0.7.
- Added opt-in: mem_cap_GB/kv_mem_frac (published 141/180 GB, TRT 0.9, vLLM 0.9), fmt_ceiling (Hopper fp8 1550/1979, attn:bf16 660/989.5).
- 0.21 blog config: max_num_tokens 1151, kv frac 0.8, batch 1024 fixed; 1.2.0: no MTP.
- candidate v094: 9/10 groups better (moe dec 5.4%, r4 18.1%), moe static trt-dsr1-h200 1.82x (30→60%) → NOT default. Fixes alone under wide worsen r4 → opt-in.
- Residual H200 DS decode 88 ms measured vs 42 ms model, no public cause found.
- suite r2: 510/511, only test_core_087 CLI-field coverage → added --mem-cap-gb/--kv-mem-frac; 087 + test_api green; fp c034bfe3 unchanged. SHIP-READY (r28, not merged).
