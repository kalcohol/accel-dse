"""0.100: exec_overlap = "pipe" is the default (catalog NPU chips); stage stays selectable; refhw unchanged."""
import hashlib, json
from accel_dse.core.catalog import entries
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario


def _fp(extra_top: dict) -> str:
    h = hashlib.sha256()
    for e in entries()[:60]:
        for extra in ({}, {"chip": {"slc_mib": 256}}, {"chip": {"slc_mib": 256, "slc_policy": "lru"}}, {"layout": {"tp": 2}}):
            try:
                r = evaluate(Scenario.from_dict({"model": e["id"], **extra_top, **extra}))
                h.update(json.dumps(r.summary(), sort_keys=True, default=str).encode())
            except Exception as ex:  # noqa: BLE001
                h.update(type(ex).__name__.encode())
    return h.hexdigest()[:16]


def test_default_is_pipe_and_stage_selectable():
    s = Scenario(model="qwen3-8b")
    assert s.exec_overlap == "pipe" and "exec_overlap" not in s.to_dict()
    st = Scenario.from_dict({"model": "qwen3-8b", "exec_overlap": "stage"})
    assert st.to_dict()["exec_overlap"] == "stage" and st.hash() != s.hash()
    assert s.hash() == "917ab559e6aa1c07"           # partial-dict hash unchanged (the scenario dict did not change)


def test_fingerprints():
    assert _fp({"exec_overlap": "stage"}) == "e2176af2610c144c"     # the 0.99 default numbers, explicitly stage
    assert _fp({}) == "392692e6937f69ef"                            # 0.100 default (pipe)


def test_refhw_unchanged():
    import accel_dse.core.extval as X
    r = X.rows(X.DATA)[0]
    for v in ("catalog", "catalog_kernel"):
        assert X.scenario(r, v, phase="decode", batch=8, ctx=1024, out_len=8).exec_overlap != "pipe"
