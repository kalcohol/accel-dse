"""0.88: context parallel (Layout.cp, ring attention) for LLM prefill; scenario-hash back-compat; server under load."""
from __future__ import annotations

import json
import math
import socket
import threading
import time
import urllib.request

from accel_dse.core.catalog import get_model
from accel_dse.core.energy import EnergyTable, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.ir import Phase, Shard, _cdiv, build_rank_ops
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import PDConfig, Scenario, Serving

H = "hbm3e_8s_12h24g_9200"


def _pf(mid, L, S, b=1, **kw):
    return evaluate(Scenario(model=mid, layout=L, mem_id=H, node_cards=8, serving=Serving(phase="prefill", batch=b, prompt=S), **kw))


def _work(r):
    L = r.scenario.layout
    return sum(s.flops_u for s in r.stages) * L.tp * L.sp * L.dp * L.cp / r.scenario.serving.batch


def test_default_unchanged_and_label():
    assert Scenario(model="qwen3-8b").hash() == Scenario(model="qwen3-8b", layout=Layout(cp=1)).hash() == "723894c17df5d7bb"
    assert "cp" not in Scenario(model="qwen3-8b").to_dict()["layout"]
    assert Layout(tp=2, cp=4).label == "PP1·TP2·CP4" and Layout(tp=2, cp=4).cards == 8


def test_cp_work_conservation_dense_and_zigzag_equivalence():
    for S in (65536, 131072):
        w = [_work(_pf("qwen3-8b", L, S, b)) for L in (Layout(tp=8), Layout(cp=8), Layout(tp=2, cp=4)) for b in (1, 3)]
        assert max(w) / min(w) - 1 < 1e-5, w            # integer prefix rounding only
    # chunk phase: q·(c + q/2) = S² / (2·cp) — the busiest zigzag rank's causal attention work
    S, cp = 131072, 8
    q = _cdiv(S, cp); c = round(S * S / (2 * cp * q) - q / 2)
    assert abs(q * (c + q / 2) / (S * S / (2 * cp)) - 1) < 1e-9


def test_cp_mla_latent_upprojection_and_ring_bytes():
    m = get_model("deepseek-v3.2")
    S, cp = 131072, 8
    q = _cdiv(S, cp); c = round(S * S / (2 * cp * q) - q / 2)
    f = lambda ph: sum(o.flops for o in build_rank_ops(m, 3, ph, Shard()) if o.name == "attn.kv_b")
    # received latent blocks are up-projected on every rank (latent exchange, 576 B/token, not 82 KB of K/V):
    # Σ ranks (c + q) keys vs S → (cp + 1) / 2
    assert abs(f(Phase("prefill", cp, q=q, ctx=c)) / f(Phase("prefill", 1, q=S, ctx=0)) - (cp + 1) / 2) < 1e-3
    # ring: per layer per rank (cp − 1) blocks of q tokens (dense GQA: ⌈kv/TP⌉·(qk + v)·2 B), hand calc
    a = _pf("qwen3-8b", Layout(cp=8), 1048576)
    g = get_model("qwen3-8b").layers[0].core
    by = (1048576 // 8) * g.n_kv * (g.qk_dim + g.v_dim) * 2 * len(get_model("qwen3-8b").layers)   # payload per rank
    assert abs(a.stages[0].time.link_bytes / by - 1) < 1e-12
    assert abs(a.stages[0].time.t_link / (7 * by / 400e9) - 1) < 1e-3                 # cp − 1 blocks received


def test_cp_memory_and_long_prefill():
    a, b = _pf("qwen3-8b", Layout(tp=8), 1048576), _pf("qwen3-8b", Layout(cp=8), 1048576)
    assert a.fits and b.fits and abs(a.ttft / b.ttft - 1) < 0.01          # same work, compute-bound
    assert b.stages[0].time.t_link < a.stages[0].time.t_link / 4             # ring KV ≪ TP all-reduces
    kv1 = _pf("qwen3-8b", Layout(), 1048576).stages[0].mem.kv_total
    assert abs(b.stages[0].mem.kv_total / kv1 - 1 / 8) < 1e-12, (kv1, b.stages[0].mem.kv_total)   # own shard only
    d, c = _pf("deepseek-v3.2", Layout(tp=8, ep=8), 1048576), _pf("deepseek-v3.2", Layout(cp=8, ep=8), 1048576)
    assert c.fits and c.ttft < d.ttft / 5


def test_cp_rejections_and_moe_rule():
    for mid, kw in (("qwen3-8b", dict(phase="decode")), ("qwen3-next-80b-a3b", {}), ("gpt-oss-20b", {}),
                    ("qwen3-8b", dict(prefix_cached=1024))):
        L = Layout(cp=2, ep=2) if get_model(mid).is_moe else Layout(cp=2)
        try:
            evaluate(Scenario(model=mid, layout=L, serving=Serving(**{"phase": "prefill", "prompt": 4096, **kw})))
        except ValueError as e:
            assert "cp" in str(e)
        else:
            raise AssertionError((mid, kw))
    assert not Layout(cp=2).valid_for(False, full=True)
    assert Layout(tp=2, cp=4, ep=8).valid_for(True) and not Layout(tp=2, cp=4, ep=2).valid_for(True)


def test_cp_energy_and_pd_prefill_layout():
    T = EnergyTable(pJ_mac=0.5, pJ_vec=1.0, pJ_bit_sram=0.1, pJ_bit_dram=5.0, pJ_bit_link=10.0, idle_W=100.0,
                    pJ_bit_slc=1.0, pJ_bit_d2d=2.0, pJ_bit_net=20.0)
    e1, e8 = (energy_report(_pf("qwen3-8b", L, 65536, 2), T) for L in (Layout(tp=8), Layout(cp=8)))
    assert abs(e8["counts_per_unit"]["mac"] / e1["counts_per_unit"]["mac"] - 1) < 1e-5
    assert abs(sum(e8["J_by_action"].values()) - e8["J_per_unit"]) <= 1e-9 * e8["J_per_unit"]
    from accel_dse.core.disagg import disagg_report
    s = Scenario(model="qwen3-8b", mem_id=H, node_cards=8, serving=Serving(batch=16, prompt=32768, out_len=256),
                 pd=PDConfig(enabled=True, prefill_layout=Layout(cp=4), prefill_cards=4, decode_cards=4))
    r = disagg_report(s)
    assert r["prefill"]["layout"] == "PP1·TP1·CP4" and r["cap_req_s"]["prefill"] > 0


def test_scenario_hash_back_compat():
    """Hashes identical from 0.70.1 through 0.88 (val5/hash_compat.py); pinned here."""
    G = [({"model": "qwen3-8b"}, "723894c17df5d7bb"),
         ({"model": "deepseek-v3", "layout": {"tp": 8, "ep": 8}, "mem_id": H}, "7c782993c0592810"),
         ({"model": "qwen3-30b-a3b", "serving": {"phase": "prefill", "batch": 4, "prompt": 2048}, "layout": {"tp": 2, "ep": 2}}, "e28e246709c1e6e2"),
         ({"model": "wan2.1-1.3b", "layout": {"sp": 2}}, "b81420e735aad402"),
         ({"model": "qwen3-8b", "package_cards": 2, "d2d_enabled": True}, "48bf40d36a8b1682"),
         ({"model": "qwen3-8b", "pd": {"enabled": True, "prefill_cards": 2, "decode_cards": 2}}, "104bd23e73966053"),
         ({"model": "llama-3.1-8b", "fabric": {"enabled": True}}, "92a7383bf47ea4ae")]
    for d, h in G:
        s = Scenario.from_dict(d)
        assert s.hash() == h, (d, s.hash())
        assert Scenario.from_dict(s.to_dict(full=True)).hash() == h        # full dicts (UI) reload to the same hash


def _post_progress(port: int, body: bytes, done: list, quiet_s: float = 60.0, poll_s: float = 5.0):
    """0.292: POST /api/eval on a raw socket and read the whole response, waiting as long as the server keeps
    completing requests (``done`` grows).  Fails (TimeoutError) only after ``quiet_s`` with no response anywhere — a
    hung server — instead of a fixed per-request deadline that a loaded box (suite -j 8, evals serialised on the GIL)
    could exceed while the server is healthy.  Any reset / refusal still raises."""
    s = socket.create_connection(("127.0.0.1", port), timeout=quiet_s)
    try:
        s.sendall(b"POST /api/eval HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Connection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        s.settimeout(poll_s)
        got, seen, quiet = b"", len(done), 0.0
        while True:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                if len(done) > seen:                # the server answered someone meanwhile: still serving the queue
                    seen, quiet = len(done), 0.0
                    continue
                quiet += poll_s
                if quiet >= quiet_s:
                    raise TimeoutError(f"no server progress for {quiet_s:.0f} s") from None
                continue
            if not chunk:
                break
            got += chunk
        line = got.split(b"\r\n", 1)[0].split()
        return int(line[1]) if len(line) >= 2 and line[1].isdigit() else got[:40]
    finally:
        s.close()


def test_server_burst_and_stalled_client():
    from accel_dse import serve
    assert serve.Server.request_queue_size >= 128 and serve.Server.daemon_threads and serve.Handler.timeout > 0
    srv = serve.Server(("127.0.0.1", 0), serve.Handler)
    old_t = serve.Handler.timeout
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        port = srv.server_address[1]
        res, done = [], []
        body = json.dumps({"scenario": {"model": "qwen3-8b"}}).encode()

        def go():
            try:
                st = _post_progress(port, body, done)
            except Exception as e:      # noqa: BLE001 — any reset / refusal / hang is the failure under test
                st = type(e).__name__
            res.append(st)
            done.append(1)
        # 0.292: the burst runs with the production read timeout; the 2 s timeout is only for the stalled client below
        # (set before the burst, a handler starved > 2 s on a loaded box could 408 a healthy request)
        ts = [threading.Thread(target=go) for _ in range(96)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert res.count(200) == 96, {x: res.count(x) for x in set(res)}
        serve.Handler.timeout = 2
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(b"POST /api/eval HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{")
        # 0.173: read until the status line is complete (a single recv may return a partial segment under load) and
        # report what arrived; the bound checks the 2 s read timeout fired (not the 30 s default / a held thread)
        s.settimeout(60)
        t0 = time.monotonic()
        ctl = []                        # control: a 2 s sleep started at t0 absorbs the same machine stall
        ct = threading.Thread(target=lambda: (time.sleep(serve.Handler.timeout), ctl.append(time.monotonic() - t0)))
        ct.start()
        got = b""
        while b"\r\n" not in got:
            chunk = s.recv(256)
            if not chunk:
                break
            got += chunk
        dt = time.monotonic() - t0
        s.close()
        assert got.startswith(b"HTTP/1.0 408"), (got[:80], dt)
        ct.join()
        # unloaded: ctl ≈ 2 s → bound 20 s as before (proves the 2 s timeout fired, not the 30 s default); under a
        # whole-machine stall (suite -j 8) the bound moves by the stall the control timer saw, not by a fixed slack
        stall = max(0.0, ctl[0] - serve.Handler.timeout)
        assert dt < 20 + stall and dt < old_t + stall, (got[:40], dt, ctl[0])
    finally:
        serve.Handler.timeout = old_t
        srv.shutdown()
        srv.server_close()


def test_burst_client_still_fails_on_a_hung_server():
    """0.292: the progress-based client is not a weaker check — a server that accepts and never answers (and makes no
    progress for anyone) still fails it, and so does a reset."""
    lst = socket.socket()
    lst.bind(("127.0.0.1", 0))
    lst.listen(4)
    port = lst.getsockname()[1]
    held = []
    th = threading.Thread(target=lambda: held.append(lst.accept()), daemon=True)
    th.start()
    t0 = time.monotonic()
    try:
        _post_progress(port, b"{}", [], quiet_s=1.0, poll_s=0.25)
        raise AssertionError("hung server accepted")
    except TimeoutError:
        assert time.monotonic() - t0 < 30
    th.join(5)
    for c, _ in held:
        c.close()
    lst.close()
    try:                                              # nothing listening: refused, not a silent pass
        _post_progress(port, b"{}", [], quiet_s=1.0, poll_s=0.25)
        raise AssertionError("closed port accepted")
    except OSError as e:
        assert not isinstance(e, TimeoutError) or "progress" not in str(e)
