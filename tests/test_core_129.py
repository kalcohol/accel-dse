"""0.129 hostile review of the UI / API layer and DES seeding: malformed Content-Length / deep JSON answered (not
dropped / 500), a broken worker pool is replaced, mac_eff ≤ 1, inapplicable sweep knobs rejected, scenario
round-trip, concurrent requests equal serial ones, DES seed validation and determinism."""
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from accel_dse import api, serve
from accel_dse.core import pdsim as P
from accel_dse.core.validation import v4_scenario


def _post(path, body):
    return serve.handle("POST", path, json.dumps(body).encode())


def _raw(port, req: bytes) -> bytes:
    s = socket.create_connection(("127.0.0.1", port))
    s.settimeout(10)
    s.sendall(req)
    data = b""
    while True:
        c = s.recv(65536)
        if not c:
            break
        data += c
    return data


def test_http_bad_content_length_and_deep_json():
    srv = serve.Server(("127.0.0.1", 0), serve.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for cl in (b"-5", b"abc", b"1e3", "²".encode()):
            d = _raw(port, b"POST /api/fit HTTP/1.1\r\nHost: x\r\nContent-Length: " + cl + b"\r\nConnection: close\r\n\r\n{}")
            assert d.startswith(b"HTTP/1.0 400"), (cl, d[:80])
        ok = _raw(port, b"POST /api/fit HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
        assert ok.startswith(b"HTTP/1.0 200")
    finally:
        srv.shutdown()
        srv.server_close()
    deep = "[" * 20000 + "]" * 20000
    st, out = serve.handle("POST", "/api/eval", ('{"scenario": ' + deep + "}").encode()[:serve.MAX_BODY])
    assert st == 400 and "error" in out


def test_broken_pool_replaced():
    calls = []

    class Dead:
        def map(self, *a, **k):
            raise BrokenProcessPool("worker died")

        def shutdown(self, **k):
            calls.append("shutdown")
    old_pool, old_n = api._POOL, api._POOL_N
    api._POOL, api._POOL_N = Dead(), 1          # n = 1 → replacement runs in-process
    try:
        st, out = _post("/api/compare", {"cards": 2})
        assert st == 200 and out, out
        assert calls == ["shutdown"] and api._POOL is None
    finally:
        api._POOL, api._POOL_N = old_pool, old_n


def test_validation_gaps():
    st, out = _post("/api/eval", {"scenario": {"chip": {"mac_eff": 2}}})
    assert st == 400 and "mac_eff" in out["error"]
    assert _post("/api/eval", {"scenario": {"chip": {"mac_eff": 1}}})[0] == 200
    for path in ("workload.frames", "serving.images", "serving.spec_k", "serving.moe_skew"):
        st, out = _post("/api/sweep", {"path": path, "values": [1, 2]})
        assert st == 400 and out["error"].startswith(path), (path, out)
    st, out = _post("/api/sweep", {"path": "serving.moe_skew", "values": [1, 4],
                                   "scenario": {"model": "deepseek-v4-flash", "layout": {"tp": 8, "ep": 8}}})
    assert st == 200 and out["rows"][1]["tok_s"] < out["rows"][0]["tok_s"]


def test_scenario_round_trip():
    for body in ({}, {"chip_preset": "1P"}, {"chip_preset": "H100-like", "scenario": {"serving": {"batch": 16}}},
                 {"scenario": {"model": "deepseek-v4-flash", "layout": {"tp": 4, "ep": 4}, "mem_eff": 0.7}},
                 {"scenario": {"chip": {"freq_ghz": 1.3}, "exec_overlap": "stage", "pd": {"enabled": True}}}):
        st, o = _post("/api/eval", body)
        assert st == 200
        for pre in (None, "1P"):
            b2 = {"scenario": o["scenario"], **({"chip_preset": pre} if pre else {})}
            st2, o2 = _post("/api/eval", b2)
            assert st2 == 200 and o2["hash"] == o["hash"] and o2["summary"] == o["summary"]


def test_concurrent_equals_serial():
    reqs = [("/api/eval", {"scenario": {"serving": {"batch": b}}}) for b in (1, 16)]
    reqs += [("/api/compare", {"cards": 4}), ("/api/sweep", {"path": "serving.ctx", "values": [1024, 8192]}),
             ("/api/eval", {"scenario": {"serving": {"batch": 0}}})]
    serial = [json.dumps(_post(*r), sort_keys=True) for r in reqs]
    work = [i for i in range(len(reqs)) for _ in range(4)]
    with ThreadPoolExecutor(16) as ex:
        got = list(ex.map(lambda i: (i, json.dumps(_post(*reqs[i]), sort_keys=True)), work))
    assert all(g == serial[i] for i, g in got)


def test_des_seed_validation_and_determinism():
    rep, ctx = P.capture_ctx(v4_scenario(0.6))
    lam = rep["queue"]["lambda_rps"]
    for bad in (None, True, 1.5, "1"):
        try:
            P.simulate(ctx, lam, "pd", n_req=50, warmup=10, seed=bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    a = P.simulate(ctx, lam, "coloc_prefill_first", n_req=200, warmup=40, seed=3)
    P.simulate(ctx, lam, "pd", n_req=200, warmup=40, seed=9)          # unrelated run in between (no shared state)
    b = P.simulate(ctx, lam, "coloc_prefill_first", n_req=200, warmup=40, seed=3)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    try:
        P.compare(v4_scenario(0.6), n_req=50, warmup=10, seeds=(1, 1))
        raise AssertionError("duplicate seeds accepted")
    except ValueError:
        pass


# 0.302 suite time: the SLO-goodput rate searches are skipped — this module reads no SLO output (tests/fast_queue.py)
from tests.fast_queue import fast_module  # noqa: E402
fast_module(globals())
