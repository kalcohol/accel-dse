"""0.85: exact zero-preemption packing bound; strict top-level API keys; UI stale flags on verdict / error branches."""
import re
from dataclasses import replace
from pathlib import Path

import accel_dse.core.pdqueue as Q
from accel_dse import api
from accel_dse.api import ApiError
from accel_dse.core.disagg import disagg_report
from accel_dse.core.validation import v4_scenario


def test_pack_bound_unit():
    fixed = [(1.0, 4096, 512)]
    assert Q._pack_preempt_free(fixed, 15258, None, 1.0)            # ⌊15258/4096⌋ = 3, 3·(4608+1) ≤ K
    assert Q._pack_preempt_free(fixed, 4096 * 4 - 1, None, 1.0)      # 3 admitted, 3·4609 = 13827 ≤ 16383
    assert not Q._pack_preempt_free(fixed, 4096 * 3 + 100, None, 1.0)  # 3 admitted, 13827 > 12388
    assert not Q._pack_preempt_free(fixed, 4096 * 5, None, 1.0)       # 5 admitted, 5·4609 > 20480
    assert Q._pack_preempt_free(fixed, 4096 * 5, 4, 1.0)              # slot cap 4: 4·4609 ≤ 20480
    assert not Q._pack_preempt_free([(0.5, 100, 10), (0.5, 4000, 4000)], 9000, None, 1.0)   # short prompts pack many
    assert not Q._pack_preempt_free(fixed, 0, None, 1.0) and not Q._pack_preempt_free([], 1e6, None, 1.0)


def test_pack_bound_point18_zero_preemption():
    s = v4_scenario(0.3, 0.0, False, "moe30b")
    s = replace(s, pd=replace(s.pd, kv_policy="swap", kv_capacity_GB=1.5, kv_admit="before_prefill"))
    q = disagg_report(s)["queue"]["modes"]
    for m in ("pd", "coloc_prefill_first", "coloc_chunked"):
        c = q[m]["kv_cap"]
        assert c.get("pack_preempt_free") and c["preempt_per_req"] == 0.0, (m, c.get("preempt_per_req"))
    assert q["pd"]["itl_max_ms"] < 10.0          # DES 3.84 ms (0.84: 243 ms)


def test_api_rejects_unknown_top_level_keys():
    base = {"scenario": {"model": "qwen3-8b"}}
    for fn, extra in ((api.api_eval, {}), (api.api_fit, {}), (api.api_layouts, {}), (api.api_compare, {}),
                      (api.api_stability, {}), (api.api_pareto, {}), (api.api_sweep, {"path": "serving.batch", "values": [1]})):
        try:
            fn(base | extra | {"objetive": "goodput"})
            raise AssertionError(f"{fn.__name__} accepted an unknown key")
        except ApiError as e:
            assert "unknown keys" in str(e) and "objetive" in str(e), (fn.__name__, e)
    # the UI's body() keys stay accepted
    ui = {"chip_preset": "100T", "scenario": {"model": "qwen3-8b"}, "energy": {}, "budget": None}
    api.api_fit(dict(ui))
    api.api_eval(ui | {"best_batch": False, "goodput": False})


def test_ui_verdict_and_error_branches_are_stale_flagged():
    js = (Path(api.__file__).parent / "web" / "app.js").read_text()
    assert re.search(r"if \(none\) \{ \$\('cmp-status'\)\.dataset\.done = '1'; return; \}", js)
    for sid in ("cmp-status", "lay-status", "sw-status"):
        for m in re.finditer(r"catch \(e\) \{ \$\('" + sid + r"'\)\.textContent = '错误：' \+ e\.message;([^}]*)\}", js):
            assert f"$('{sid}').dataset.done = '1'" in m.group(1), sid


from tests.pin_legacy import pin_module  # noqa: E402  (0.173)
pin_module(globals())
