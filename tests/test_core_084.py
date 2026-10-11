"""0.84: arrival-variance term in the saturation growth law; adopted defaults."""
import accel_dse.core.pdqueue as Q


def test_defaults_adopted():
    assert Q.SAT_MODEL == "gamma" and Q.VICTIM_WAIT == "blend" and Q.SAT_ARR_M == 1


def test_arr_c2_limits_and_monotone():
    assert Q._arr_c2(0.0, 1.0, 1) == 1.0                  # full KV: departures Poisson → exp law
    assert Q._arr_c2(0.3, 0.0, 1) == 0.3                  # empty KV: lifetime CV² only
    assert Q._arr_c2(1.0, 0.4, 1) == 1.0                  # memoryless stays memoryless
    v = [Q._arr_c2(0.1, r, 1) for r in (0.1, 0.3, 0.6, 0.9)]
    assert all(a < b for a, b in zip(v, v[1:]))
    assert abs(Q._arr_c2(0.2, 0.5, 1) - 0.6) < 1e-15
    assert Q._arr_c2(0.2, 1.7, 1) == 1.0                  # fill clamped
