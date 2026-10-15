"""0.116 suite-time split of test_core_057::test_refill_offset_and_cascade (fixed-output half, unchanged assertion):
a separate module runs in its own process, so the three DES replays no longer sit on one worker's critical path."""
import math

from tests.test_core_057 import _rep


def test_refill_offset_fixed_outputs():
    rep0, _ = _rep(0.85, 0.0, "recompute")
    k0 = rep0["queue"]["modes"]["pd"]["kv_cap"]
    assert abs(k0["refill_offset_factor"] - math.exp(-1)) < 0.02        # fixed outputs: e^{-1}
