"""0.90: video / protein × new features (PD n/a, CP n/a, energy, D2D, SLC); sweep hint for auto_overlap."""
from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import replace
from pathlib import Path

from accel_dse.core.energy import EnergyTable, energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

H = "hbm3e_8s_12h24g_9200"
T = EnergyTable(pJ_mac=0.5, pJ_vec=1.0, pJ_bit_sram=0.1, pJ_bit_dram=5.0, pJ_bit_link=10.0, idle_W=100.0,
                pJ_bit_slc=1.0, pJ_bit_d2d=2.0, pJ_bit_net=20.0)


def _ev(d):
    return evaluate(Scenario.from_dict(d))


def test_pd_on_video_protein_warns_and_api_has_no_pd_block():
    for mid in ("wan2.1-1.3b", "esmfold"):
        r = _ev({"model": mid, "pd": {"enabled": True, "prefill_cards": 1, "decode_cards": 1}})
        assert any("PD 分离" in w and "无效" in w for w in r.warnings), r.warnings
        assert r.step == _ev({"model": mid}).step                         # ignored, not half-applied
    from accel_dse import serve
    srv = serve.Server(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for mid, want in (("wan2.1-1.3b", False), ("qwen3-8b", True)):
            req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/eval", data=json.dumps(
                {"scenario": {"model": mid, "pd": {"enabled": True, "prefill_cards": 1, "decode_cards": 1}}}).encode())
            with urllib.request.urlopen(req, timeout=120) as resp:
                assert ("pd" in json.loads(resp.read())) is want
    finally:
        srv.shutdown()
        srv.server_close()


def test_cp_on_video_protein_says_why():
    for mid, word in (("wan2.1-1.3b", "Ulysses"), ("esmfold", "DAP")):
        try:
            _ev({"model": mid, "layout": {"cp": 2}})
        except ValueError as e:
            assert "上下文并行" in str(e) and "sp" in str(e) and word in str(e), str(e)
        else:
            raise AssertionError(mid)


def test_slc_slower_than_dram_pin_warns():
    slow = _ev({"model": "esm2-650m", "mem_id": H, "chip": {"slc_mib": 512}})
    assert any("SLC 带宽" in w and "低于 DRAM" in w for w in slow.warnings)
    fast = _ev({"model": "esm2-650m", "mem_id": H, "chip": {"slc_mib": 512, "slc_GBps": 20000}})
    assert not any("SLC 带宽" in w for w in fast.warnings)
    assert not any("SLC 带宽" in w for w in _ev({"model": "esm2-650m", "mem_id": H}).warnings)   # default: no SLC
    # with the warning's premise removed, more SLC is never slower (serial overlap shows memory time)
    base = {"model": "esm2-650m", "mem_id": H, "layout": {"tp": 2}, "exec_overlap": "serial"}
    t = [_ev({**base, "chip": {"slc_mib": s, "slc_GBps": 20000}}).step for s in (0, 16, 64, 512)]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(t, t[1:])), t


def test_dap_replication_is_linear_in_sp_and_energy_sums():
    """Structure models: single / atom tracks replicated on every DAP rank — MAC count = 1 + f·(sp − 1) of 1 card."""
    for mid in ("protenix", "boltz-1"):
        e = {sp: energy_report(_ev({"model": mid, "layout": {"sp": sp}}), T) for sp in (1, 2, 4)}
        r2, r4 = (e[s]["counts_per_unit"]["mac"] / e[1]["counts_per_unit"]["mac"] for s in (2, 4))
        assert r2 > 1 and abs((r4 - 1) - 3 * (r2 - 1)) < 1e-9 * r4, (mid, r2, r4)
        for x in e.values():
            assert abs(sum(x["J_by_action"].values()) - x["J_per_unit"]) <= 1e-9 * x["J_per_unit"]
    for mid in ("wan2.1-1.3b", "esm2-650m"):   # non-structure: layout-invariant MAC count
        a = energy_report(_ev({"model": mid}), T)["counts_per_unit"]["mac"]
        b = energy_report(_ev({"model": mid, "layout": {"dp": 2, "sp": 2}}), T)["counts_per_unit"]["mac"]
        assert abs(b / a - 1) < 1e-3, (mid, b / a)


def test_fixed_algo_protocol_pick_vs_auto_overlap_monotone():
    """A fixed algorithm still picks its fastest protocol per collective (bw + α) — the same non-monotonicity as
    auto (esmfold DAP, D2D, tree: link ÷ 4 → −58 µs); auto_overlap is monotone."""
    d = {"model": "esmfold", "mem_id": H, "layout": {"dp": 2, "sp": 4}, "package_cards": 2, "d2d_enabled": True,
         "chip": {"slc_mib": 64, "slc_policy": "lru"}, "node_cards": 8, "exec_overlap": "tbo"}
    for algo, mono in (("tree", False), ("auto_overlap", True)):
        s = Scenario.from_dict({**d, "fabric": {"enabled": True, "algo": algo}})
        a = evaluate(s).step
        for sl in (replace(s, link=replace(s.link, GBps=s.link.GBps / 4)),
                   replace(s, d2d=replace(s.d2d, GBps=s.d2d.GBps / 4), d2d_std="custom")):
            b = evaluate(sl).step
            assert (b >= a * (1 - 1e-12)) is mono, (algo, a, b)


def test_sweep_hint_wiring():
    js, html = W.read(), W.read(W.INDEX_HTML)          # 0.286: parsed function bodies instead of character windows
    fn = W.functions(js)
    assert "sw-hint" in W.html_ids(html) and W.has(fn["swHint"], "f.algo !== 'auto_overlap'")
    assert W.has(fn["swHint"], "S.sc.d2d_enabled") and W.has(fn["swHint"], "id: 'sw-hint-fix'")
    assert W.has(fn["schedule"], "swHint();")       # every input change re-evaluates the hint
