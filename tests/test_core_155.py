"""0.155 standing invariants (mechanism guards; see val5/v0155/ppfab.py for the randomized suite):
PP partition / storage / KV conservation, DRAM-need and weight-read shifts bounded by SRAM; fabric auto ≤ explicit
(on stage totals — α lives in t_sync), bandwidth / α / oversub / channel monotonicity, in-network reduce never slower
under auto, TP1 ⇒ no link time."""
import dataclasses

from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import INNET_MODES
from accel_dse.core.scenario import Scenario


def _lay(s, **kv):
    for k, v in kv.items():
        s = s.replace("layout." + k, v)
    return s


def _tot(r):
    return sum(st.time.total for st in r.stages)


def test_pp_conservation_and_bounds():
    for m in ("qwen3-8b", "qwen3-30b-a3b", "lfm2.5-24b-a2b"):
        s = Scenario().replace("model", m).replace("serving.phase", "decode")
        r1 = evaluate(s)
        L = r1.model.n_layers
        for pp in (2, 4):
            rp = evaluate(_lay(s, pp=pp))
            cov = sorted(st.layers for st in rp.stages)
            assert cov[0][0] == 0 and cov[-1][1] == L and all(a[1] == b[0] for a, b in zip(cov, cov[1:]))
            w1 = sum(st.mem.stored_w for st in r1.stages)
            wp = sum(st.mem.stored_w for st in rp.stages)
            extra = r1.model.vocab * r1.model.hidden * 4 if r1.model.tie_embeddings else 0   # tied head copy
            assert w1 * (1 - 1e-9) <= wp <= w1 * (1 + 1e-9) + extra, (m, pp, w1, wp)
            k1 = sum(st.mem.kv_total for st in r1.stages)
            assert abs(sum(st.mem.kv_total for st in rp.stages) - k1) <= 1e-6 * max(k1, 1)
            assert max(st.mem.dram_need for st in rp.stages) <= max(st.mem.dram_need for st in r1.stages) * (1 + 1e-9)
            if rp.microbatches == 1:
                d1 = sum(st.dram["weights"] for st in r1.stages)
                dp = sum(st.dram["weights"] for st in rp.stages)
                assert dp <= d1 * (1 + 1e-9) and d1 - dp <= s.chip.sram_mib * 2**20 * (pp - 1) * (1 + 1e-9)


def test_fabric_auto_and_monotonicity():
    s = Scenario().replace("model", "qwen3-32b").replace("serving.phase", "decode").replace("fabric.enabled", True) \
        .replace("exec_overlap", "serial").replace("d2d_enabled", True).replace("package_cards", 2)
    s = _lay(s, tp=4)
    tot = {a: _tot(evaluate(s.replace("fabric.algo", a))) for a in ("auto", "ring", "tree", "hier")}
    assert tot["auto"] <= min(v for k, v in tot.items() if k != "auto") * (1 + 1e-12)
    sa = s.replace("fabric.algo", "auto")
    t0 = tot["auto"]
    assert _tot(evaluate(sa.replace("link.GBps", s.link.GBps * 2))) <= t0 * (1 + 1e-12)
    assert _tot(evaluate(sa.replace("link.alpha_us", s.link.alpha_us * 2))) >= t0 * (1 - 1e-12)
    n = sa.replace("node_cards", 2)
    tn = _tot(evaluate(n))
    assert _tot(evaluate(n.replace("fabric.oversub", 4.0))) >= tn * (1 - 1e-12)
    c4 = _tot(evaluate(n.replace("fabric", dataclasses.replace(n.fabric, channels=4, channel_GBps=25.0))))
    c16 = _tot(evaluate(n.replace("fabric", dataclasses.replace(n.fabric, channels=16, channel_GBps=25.0))))
    assert c16 <= c4 * (1 + 1e-12) and c16 >= tn * (1 - 1e-12)
    for mode in INNET_MODES[1:]:
        assert _tot(evaluate(n.replace("fabric.innet_reduce", mode))) <= tn * (1 + 1e-12)
    assert sum(st.time.t_link for st in evaluate(_lay(sa, tp=1).replace("package_cards", 1)).stages) == 0
