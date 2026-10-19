"""0.249: state-dependent σ_b²(f) below U = 0 (MODEL §196).  Coarse-grained over ≫ one admission, each arrival moves U by
J(k) = d − n_B·T_d·k̄_f·a/B + τ_p·c(k)·a(k) (its slot, minus the deaths its chunk iterations carry, plus the decode-only
deaths its pause removes; near full J → β + τ = S̄, deep J → d), and the deaths around that mean are the replay of
births clustered in token time (Fano I_s): σ_b²(f) = λE[J²] + d·R(k)·I_s.  A λ52: 0.122 near full → 0.056 at f 60
(DES true-f KM 0.13 / 0.11 → 0.03; 0.248 constant 0.079)."""
import math

from accel_dse.core import pdqueue
from test_core_248 import _cp, _sb


def test_sigma2_f_shape_and_flag():
    cp, B = _cp()
    assert pdqueue._SIG2_F
    x = _sb(cp, B, m_k=40.0)
    s2 = [v for _, v in x["sigma2_f"]]
    assert all(b <= a + 1e-12 for a, b in zip(s2, s2[1:]))       # falls with the free slots
    assert 0.10 < s2[0] < 0.14 and s2[-1] < 0.6 * s2[0]
    lam, d = 13.0, x["d"]
    assert s2[0] > x["sigma2"] > lam * d * d                      # near full above λS̄² (+ death noise), floor λd²
    old = pdqueue._SIG2_F
    try:
        pdqueue._SIG2_F = False
        y = _sb(cp, B, m_k=40.0)
    finally:
        pdqueue._SIG2_F = old
    assert math.isclose(y["sigma2b"], y["sigma2"] + lam * d * d)
    assert all(math.isclose(v, y["sigma2b"]) for _, v in y["sigma2_f"])
    # larger noise near full → more mass near U = 0 → more backlog; above 0 and the base unchanged
    assert x["p_sat"] > y["p_sat"] and math.isclose(x["theta"], y["theta"]) and math.isclose(x["base"], y["base"])
