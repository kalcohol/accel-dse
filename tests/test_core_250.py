"""0.250: the near-full lattice at U = 0 (MODEL §197), derived but OFF by default (`_LATTICE0`).
f = 0 (q = 0, slots full) is entered by an admission (its prefill in flight for a phase τ) and left by a death; an
arrival there opens a U > 0 excursion that takes the rest of the phase and returns with a fresh admission in flight:
x = (1 + (r_do − r_ch)m)/(r_do − (r_do − r_ch)λm), m = (1 − e^{−λτ})/λ, π₀ = λx·π₁ (DES A λ52: π₀/π₁ 1.30 vs 1.32,
in-flight share at f = 0 0.71 vs 0.71), and P(U > 0) = λπ₀S̄/(1 − ρ′) (DES π₀ → 0.147 vs DES 0.155).
On: A λ52 p_sat 0.138 (DES 0.155; 0.249 0.188) but crossings mean |err| 0.26 % (> 0.2 %), so not the default."""
import math

from accel_dse.core import pdqueue
from test_core_248 import _cp, _sb


def test_lattice0_off_by_default_and_derivation():
    cp, B = _cp()
    assert pdqueue._LATTICE0 is False
    off = _sb(cp, B, m_k=40.0)
    assert off["x0"] is None
    old = pdqueue._LATTICE0
    try:
        pdqueue._LATTICE0 = True
        on = _sb(cp, B, m_k=40.0)
    finally:
        pdqueue._LATTICE0 = old
    lam, tau, d = 13.0, on["tau"], on["d"]
    m = (1 - math.exp(-lam * tau)) / lam
    x = on["x0"]
    assert x is not None and 0.05 < x < 0.2
    assert 1.1 < lam * x < 1.6                                   # the f = 0 pile (DES π₀/π₁ 1.30)
    share = (1 + lam * x) * m / x
    assert 0.6 < share < 0.8                                     # in-flight share at f = 0 (DES 0.71)
    # the jump entry replaces the diffusion interface: less backlog mass than 0.249's interface here
    assert 0 < on["p_sat"] < off["p_sat"]
    assert math.isclose(on["theta"], off["theta"]) and math.isclose(on["base"], off["base"])
    cdf = [c for _, c in on["free_cdf"]]
    assert cdf[0] > on["p_sat"] and all(b >= a for a, b in zip(cdf, cdf[1:]))
