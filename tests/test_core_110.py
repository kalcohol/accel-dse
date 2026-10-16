"""0.110: energy accounting conservation (SFU is time-only); genomics / protein / Carbon sweep consistency."""
from accel_dse import api as A
from accel_dse.core.energy import energy_report
from accel_dse.core.evaluate import evaluate
from accel_dse.core.scenario import Scenario


def _c(sc):
    return energy_report(evaluate(Scenario.from_dict(sc)))["counts_per_unit"]


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-30)


def test_sfu_pipe_changes_time_not_energy_counts():
    for sc in ({"model": "protenix", "serving": {"batch": 2}, "workload": {"seq_len": 128}},
               {"model": "carbon-3b", "serving": {"phase": "prefill", "batch": 1, "prompt": 4096}},
               {"model": "esm2-650m"}):
        a, b = _c(sc), _c({**sc, "chip": {"sfu_per_cycle": 64}})
        for k in ("mac", "vec", "sram", "dram", "slc", "link", "net", "d2d"):
            assert _close(a[k], b[k]), (sc["model"], k, a[k], b[k])
    r0 = evaluate(Scenario.from_dict({"model": "protenix", "serving": {"batch": 2}, "workload": {"seq_len": 128}}))
    r1 = evaluate(Scenario.from_dict({"model": "protenix", "serving": {"batch": 2}, "workload": {"seq_len": 128}, "chip": {"sfu_per_cycle": 64}}))
    assert r1.latency > r0.latency          # SFU-bound softmax / sigmoid still cost time


def test_time_only_knobs_keep_counts():
    for sc in ({"model": "carbon-8b", "serving": {"phase": "decode", "batch": 8, "ctx": 8192}},
               {"model": "carbon-a-1.2b", "workload": {"seq_len": 8192}}, {"model": "esmfold", "workload": {"seq_len": 256}}):
        a = _c(sc)
        for extra in ({"exec_overlap": "kernel"}, {"kv_mem_eff": 0.9}, {"layer_overhead_us": 20.0}):
            b = _c({**sc, **extra})
            for k in ("mac", "vec", "dram", "sram", "link"):
                assert _close(a[k], b[k]), (sc["model"], extra, k)


def test_fold_useful_work_conserved_over_dap_and_replication_charged():
    for mid in ("boltz-1", "protenix", "esmfold"):
        sc = {"model": mid, "serving": {"batch": 1}, "workload": {"seq_len": 128}}
        u, mac = [], []
        for sp in (1, 2, 4):
            r = evaluate(Scenario.from_dict({**sc, "layout": {"sp": sp}}))
            u.append(sum(s.flops_u for s in r.stages) * sp)
            mac.append(energy_report(r)["counts_per_unit"]["mac"])
        assert _close(u[0], u[1], 1e-9) and _close(u[0], u[2], 1e-9), (mid, u)
        assert mac[0] <= mac[1] <= mac[2], (mid, mac)        # unsplit ops run on every DAP rank


def test_genomics_protein_sweep_rows_equal_eval_with_energy():
    for sc, path, vals in (({"model": "carbon-a-1.2b", "workload": {"seq_len": 8192}}, "serving.batch", [1, 4]),
                           ({"model": "esm2-650m"}, "chip.freq_ghz", [0.8, 1.6]),
                           ({"model": "carbon-3b", "serving": {"phase": "decode", "batch": 8, "ctx": 4096}}, "mem_eff", [0.5, 0.9])):
        rows = A.api_sweep({"scenario": sc, "path": path, "values": vals})["rows"]
        for v, row in zip(vals, rows):
            s2 = {**sc}
            if path == "serving.batch":
                s2["serving"] = {**sc.get("serving", {}), "batch": v}
            elif path == "chip.freq_ghz":
                s2["chip"] = {"freq_ghz": v}
            else:
                s2["mem_eff"] = v
            e = A.api_eval({"scenario": s2, "sensitivity": False})
            for k in ("tok_s_card", "tpot_ms", "ttft_ms", "latency_ms", "fits", "bound"):
                assert row[k] == e["summary"][k], (sc["model"], k)
            assert e["energy"]["counts_per_unit"] == energy_report(evaluate(Scenario.from_dict(s2)))["counts_per_unit"]


def test_fingerprints_unchanged():
    from tests.test_core_0100 import _fp
    assert _fp({"exec_overlap": "stage"}) == "be3e00dcc13fcdda"   # 0.200 re-baseline (CUSTOM_AR; was 841b5c4e00c68d8c) ← 0.199 re-baseline (NCCL_AR_LAT; was 6dca732e67c83a75) ← 0.198 re-baseline (FUSE_QKV; was 5e72913c351849ac) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 8410808fd4f1e578) ← 0.192 re-baseline (state pinned first; 0.173 84df3be517a53ef4) (was 0a954915c5fb003f)
    assert _fp({}) == "7d5cdb465b922571"   # 0.200 re-baseline (CUSTOM_AR; was 727e05f1eee3e647) ← 0.199 re-baseline (NCCL_AR_LAT; was bc06528a82e19e2a) ← 0.198 re-baseline (FUSE_QKV; was eb634f2e8b513ffb) ← 0.193 re-baseline (WIDE_WAVES, cta refhw; 0.192 b8ca44d92e7589cd) ← 0.192 re-baseline (state pinned first; 0.173 1b755ac357ec6d2e) (was 53ee9e08165795c6)
