# 0.93 (r27, /workspace/wt-r27, off 573b22f)
Goals: r4 under-prediction causes row by row (public docs only, no fit on r4); no other set worse; ground ~100us/layer with public traces.
- format fixes (gpt-oss W4A8 Blackwell; DS-R1 NVFP4-v2 FP8 attention): r4 31.2→26.7 %, other sets identical → adopted. MTP: 1.2.0 configs have no speculative_config → ruled out. FP8 context FMHA: not documented as default → not modelled. Candidate (attn cta + 62.3 µs/layer from DeepSeek public trace): improves 9/10 groups (r4 19.2 %), worsens moe static (1 row trt-dsr1-h200 1.35→1.88) → NOT default. suite next
- suite 507/507 green (938 s); test_api re-run after version bump green; fp 209 c034bfe3e3a3afa2 unchanged. SHIP-READY.
