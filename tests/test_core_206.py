"""0.206: critical batch for the machine-balance verdict; MI300X max-load residual localised (no default change)."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core import extval as X
from accel_dse.core.evaluate import BALANCE_CLASS_MAX, evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario


def _sc(n, ctx=4096, batch=1):
    s = Scenario(chip=CHIPS[n])
    return replace(s, serving=replace(s.serving, ctx=ctx, batch=batch))


def test_ai_rises_with_batch_and_saturates():
    r = evaluate(_sc("1P"))
    a = [r._ai_at(b)[0] for b in (1, 8, 64, 512, 4096)]
    assert all(x < y for x, y in zip(a, a[1:]))
    assert a[-1] / a[-2] < 1.1                     # KV reads grow with batch → saturation
    cb = r.critical_batch(r.balance()["chip_flop_per_byte"])
    assert cb["critical_batch"] is None and cb["ai_asymptote"] < 40 and cb["limited_by"] == "kv_capacity"


def test_balanced_preset_no_warning_but_hint():
    for n in ("1P", "H100-like"):
        b = evaluate(_sc(n)).balance()
        assert b["warning"] is None and b["hint"] and b["chip_flop_per_byte"] <= BALANCE_CLASS_MAX
    b = evaluate(_sc("1P", ctx=64)).balance()
    assert b["reachable"] and b["critical_batch"] > 1 and b["warning"] is None and "raise batch" in b["hint"]
    b2 = evaluate(_sc("1P", ctx=64, batch=b["critical_batch"])).balance()
    assert b2["hint"] is None and b2["warning"] is None and b2["workload_flop_per_byte"] >= b2["chip_flop_per_byte"] * 0.999


def test_critical_batch_is_the_crossing():
    r = evaluate(_sc("1P", ctx=64))
    bal = r.balance()["chip_flop_per_byte"]
    bs = r.critical_batch(bal)["critical_batch"]
    assert r._ai_at(bs)[0] >= bal > r._ai_at(bs - 1)[0]


def test_warns_only_when_unreachable():
    b = evaluate(_sc("1P-lp4x64")).balance()            # out-of-class balance, unreachable
    assert b["warning"] and "compute idle" in b["warning"]
    b = evaluate(_sc("100T", ctx=64)).balance()          # B* exists but the TPOT SLO caps the batch below it
    assert b["critical_batch"] and b["critical_batch"] > b["batch_limit"] and "TPOT SLO" in b["warning"]
    r = api.api_eval({"chip_preset": "1P"})
    assert {"critical_batch", "batch_limit", "limited_by", "ai_asymptote", "reachable", "hint"} <= set(r["balance"])


def test_ui_wiring():
    import pathlib
    root = pathlib.Path(api.__file__).parent
    assert "critical batch" in (root / "web/app.js").read_text()
    assert "critical batch shown" in (root.parent / "scripts/ui_headless_smoke.py").read_text()


def test_mi300x_residual_is_per_sequence_cost():
    """MODEL §153: the same README's latency table shows the model's decode-step slope in batch (and in ctx) is too
    shallow; the max-load over-prediction follows from it. Documented, not fitted."""
    rs = {(r["isl"], r["batch"]): r for r in X.rows(X.DATA_R3) if r["metric"] == "static_latency_s"}

    def step(isl, b):
        r = rs[(isl, b)]
        p = X.predict(r)
        return (r["value"] - p["ttft_s"]) / (r["osl"] - 1), p["tpot_s"]
    (m64, p64), (m128, p128) = step(128, 64), step(128, 128)
    assert (m128 - m64) > 3 * (p128 - p64)               # batch slope: measured ≈ 3.9× the model
    (k128, q128) = step(2048, 128)
    assert (k128 - m128) > 2.5 * (q128 - p128)           # ctx (KV) slope: measured ≈ 3.2× the model
    ml = [r for r in X.rows(X.DATA_R3) if r["hw"] == "MI300X" and r["metric"] == "max_tok_s_total"]
    assert all(1.5 < X.predict(r)["pred"] / r["value"] < 1.9 for r in ml)   # residual unchanged (no fix applied)
