"""0.180: MI300X max-load token definition (output tok/s) and the documented residual (MODEL §127)."""
import math
import statistics as st
import sys
sys.path.insert(0, ".")
from accel_dse.core import extval as X

ROWS = [r for r in X.rows(X.DATA_R3) if r["hw"] == "MI300X" and r["metric"] == "max_tok_s_total"]


def test_rows_marked_output():
    assert len(ROWS) == 4 and all(r["tok_def"] == "output" for r in ROWS)


def test_output_interpretation_fits_better_across_lengths():
    lo, lt = [], []
    for r in ROWS:
        p = X.predict(r, "default")["pred"]
        lo.append(math.log(p / r["value"]))
        lt.append(math.log(p * (r["isl"] + r["osl"]) / r["osl"] / r["value"]))
    assert st.pstdev(lo) < 0.1 < 0.2 < st.pstdev(lt)
