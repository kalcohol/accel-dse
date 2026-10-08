"""accel-dse core v2 — layered analytical model (0.40).

L0 scenario → L1 model / op-graph IR → L2 parallel transform → L3 mapping
→ L4 memory planning → L5 schedule → L6 serving → L7 search.
Single-card = tp=pp=ep=1; there is no second code path.
"""
