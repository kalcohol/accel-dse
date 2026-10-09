"""accel_dse — analytical design-space exploration for LLM inference accelerators.

Core v2 (0.40): models evaluated as released (HF configs + safetensors
headers), per-rank op-graph IR, datapath mapping as a design variable,
memory planning, schedule, exact search.  See docs/MODEL.md.
All hardware numbers are assumptions (「假设」), not silicon measurements.
"""

__version__ = "0.65.0"
