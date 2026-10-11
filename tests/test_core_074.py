"""0.74 — adversarial round 8 (standing randomized invariants, deterministic seeds) + runner selection.

Areas: VLM vision-encoder path, speculative decoding (MTP), MoE skew / EP, D2D chiplet split, dtype matrix,
API input validation.  Bugs fixed: moe_skew above its physical cap EP·min(k, E/EP)/k was accepted; /api/fit let an
invalid layout escape as a ValueError (HTTP 500)."""
from __future__ import annotations

import json
import random
import sys
from dataclasses import replace
from pathlib import Path

from accel_dse import api, serve
from accel_dse.core.catalog import get_model
from accel_dse.core.dtypes import FORMATS
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.schedule import spec_expected_tokens

E = 1e-9


def _ev(s):
    try:
        return evaluate(s)
    except ValueError:
        return None


def _work(r):
    return sum(st.flops_u for st in r.stages)


def test_runner_exact_selection():
    sys.path.insert(0, str(Path(__file__).parent))
    import run_tests as R
    assert R._wants("test_api", "test_x", ["test_api"])                      # module, exact
    assert not R._wants("test_core_050", "test_api_cli_three_tier", ["test_api"])   # was a substring match
    assert R._wants("test_core_050", "test_api_cli_three_tier", ["test_core_050::test_api_cli_three_tier"])
    assert not R._wants("test_core_050", "test_other", ["test_core_050::test_api_cli_three_tier"])
    assert R._wants("test_core_052", "t", ["tests/test_core_052.py"])
    assert R._wants("test_core_050", "test_api_cli_three_tier", R._kflags(["-k", "api_cli"]))   # substring opt-in
    assert R._wants("test_core_050", "test_api_cli_three_tier", ["test_api_cli_three_tier"])     # exact test name


def test_moe_skew_physical_cap():
    for mid, ep in (("deepseek-v3", 2), ("qwen3-30b-a3b", 4), ("deepseek-v3", 64)):
        f = next(L.ffn for L in get_model(mid).layers if L.ffn.kind == "moe")
        cap = ep * min(f.top_k, -(-f.n_experts // ep)) / f.top_k
        r = evaluate(Scenario(model=mid, layout=Layout(dp=ep, ep=ep), serving=Serving(batch=8, moe_skew=3.0)))
        assert abs(r.moe_skew - min(3.0, cap)) < 1e-12, (mid, ep, r.moe_skew, cap)
        assert (r.moe_skew < 3.0) == any("物理上限" in w for w in r.warnings)
    r = evaluate(Scenario(model="deepseek-v3", layout=Layout(dp=2, ep=2), serving=Serving(batch=8, moe_skew=3.0)))
    r2 = evaluate(Scenario(model="deepseek-v3", layout=Layout(dp=2, ep=2), serving=Serving(batch=8, moe_skew=2.0)))
    assert r.step == r2.step                           # capped at EP = 2: identical to skew 2


def test_api_invalid_input_is_400_never_500():
    base = Scenario().to_dict(full=True)
    sc = {**base, "layout": {**base["layout"], "pp": 1000}}
    st, out = serve.handle("POST", "/api/fit", json.dumps({"scenario": sc}).encode())
    assert st == 400 and "pp" in out["error"]
    rnd = random.Random(74)
    leaves = []

    def walk(d, p=()):
        for k, v in d.items():
            walk(v, p + (k,)) if isinstance(d.get(k), dict) else leaves.append(p + (k,))
    walk(base)
    garb = [0, -1, -0.5, 1e18, "", "abc", None, [], {}, True, 2 ** 63, 0.5, 7]
    for _ in range(300):
        sc = json.loads(json.dumps(base))
        sc["model"] = rnd.choice(["qwen3-8b", "qwen3-30b-a3b", "qwen3.8-27b", "esm2-650m"])
        p = rnd.choice(leaves)
        d = sc
        for k in p[:-1]:
            d = d[k]
        d[p[-1]] = rnd.choice(garb)
        st, out = serve.handle("POST", rnd.choice(["/api/eval", "/api/memory", "/api/fit"]), json.dumps({"scenario": sc}).encode())
        assert st in (200, 400), (p, d[p[-1]], st)
        json.dumps(api.clean(out), allow_nan=False)    # what the server sends


def test_spec_skew_d2d_dtype_vlm_invariants():
    rnd = random.Random(8)
    seen = {}
    for _ in range(120):
        kind = rnd.choice(["spec", "skew", "d2d", "dtype", "vlm"])
        chip = CHIPS[rnd.choice(["100T", "1P", "H100-like"])]
        if kind == "spec":                    # MTP models only (others warn and ignore spec_k)
            k, a = rnd.choice([1, 2, 3]), rnd.choice([0.0, 0.3, 0.7, 1.0])
            sv = Serving(batch=rnd.choice([1, 8, 64]), spec_k=k, spec_accept=a)
            s = Scenario(model=rnd.choice(["deepseek-v3", "glm-4.5-air"]), chip=chip, layout=Layout(dp=8, ep=8), serving=sv, node_cards=8)
            r, r0 = _ev(s), _ev(replace(s, serving=replace(sv, spec_k=0)))
            if not r or not r0:
                continue
            assert abs(r.tokens_per_step - spec_expected_tokens(k, a)) <= E * r.tokens_per_step
            assert abs(r.tpot - r.step / r.tokens_per_step) <= E * r.tpot
            assert r.step >= r0.step * (1 - E)                    # verifying k drafts is never cheaper than 1 token
            if a == 0.0:
                assert r.tpot >= r0.tpot * (1 - E)
            r2 = _ev(replace(s, serving=replace(sv, spec_accept=min(1.0, a + 0.2))))
            assert r2 is None or r2.tpot <= r.tpot * (1 + E)
        elif kind == "skew":
            tp, dp = rnd.choice([1, 2]), rnd.choice([2, 4, 8])
            sk = rnd.choice([1.0, 1.2, 1.5, 3.0])
            sv = Serving(batch=rnd.choice([8, 64]), phase=rnd.choice(["decode", "prefill"]), moe_skew=sk)
            s = Scenario(model=rnd.choice(["qwen3-30b-a3b", "deepseek-v3", "glm-4.5-air"]), chip=chip,
                         layout=Layout(tp=tp, dp=dp, ep=tp * dp), serving=sv, node_cards=8)
            r, r1, rh = _ev(s), _ev(replace(s, serving=replace(sv, moe_skew=1.0))), _ev(replace(s, serving=replace(sv, moe_skew=sk + 0.5)))
            if not (r and r1 and rh):
                continue
            assert r1.step * (1 - E) <= r.step <= rh.step * (1 + E)        # skew never helps, monotone
            assert 1.0 - E <= r.moe_skew <= min(sk, s.layout.ep) + E
            assert abs(_work(r) / _work(r1) - 1) < 1e-6                    # useful work unchanged
            assert not (r.fits and not r1.fits)
        elif kind == "d2d":
            tp, pc = rnd.choice([2, 4, 8]), rnd.choice([2, 4])
            s = Scenario(model=rnd.choice(["qwen3-8b", "qwen3-32b"]), chip=chip, layout=Layout(tp=tp), node_cards=8,
                         serving=Serving(batch=rnd.choice([1, 16]), phase=rnd.choice(["decode", "prefill"])),
                         d2d_enabled=True, package_cards=pc, d2d_std="custom",
                         d2d=replace(Scenario().d2d, GBps=rnd.choice([50.0, 500.0])))
            r = _ev(s)
            if not r:
                continue
            rf, roff, r1 = _ev(replace(s, d2d=replace(s.d2d, GBps=s.d2d.GBps * 4))), _ev(replace(s, d2d_enabled=False)), _ev(replace(s, package_cards=1))
            assert rf.step <= r.step * (1 + E)                              # faster D2D never slower
            assert abs(r1.step - roff.step) <= E * roff.step                # one die per package = monolithic
            assert abs(_work(r) / _work(roff) - 1) < 1e-6
        elif kind == "dtype":
            mid = rnd.choice(["qwen3-8b", "llama-3.1-8b"])
            f1, f2 = rnd.sample([f for f in FORMATS if f != "fp32"], 2)
            s = Scenario(model=mid, chip=chip, serving=Serving(batch=rnd.choice([1, 16]), phase=rnd.choice(["decode", "prefill"])))
            roles = [k for k, _ in get_model(mid).formats]
            ra, rb = (_ev(replace(s, formats_override=tuple((k, f) for k in roles))) for f in (f1, f2))
            if not ra or not rb:
                continue
            wa, wb = (sum(st.mem.stored_w for st in x.stages) for x in (ra, rb))
            ba, bb = FORMATS[f1].bits, FORMATS[f2].bits
            assert not (ba < bb and wa > wb * (1 + 1e-6)) and not (ba > bb and wa < wb * (1 - 1e-6)), (f1, f2)
            assert abs(_work(ra) / _work(rb) - 1) < 1e-6
            nat = dict(chip.formats.rates)
            for f, x in ((f1, ra), (f2, rb)):
                if f.split("_")[0] not in nat and FORMATS[f].bits < 16:
                    assert sum(st.convert_elems for st in x.stages) > 0, (f, chip.name)   # non-native → dequant
        else:
            n = rnd.choice([1, 2, 4])
            sv = Serving(phase=rnd.choice(["prefill", "decode"]), images=n, image_w=rnd.choice([224, 1024, 2048]),
                         image_h=rnd.choice([224, 1024]), batch=rnd.choice([1, 4]))
            s = Scenario(model=rnd.choice(["glm-5.3-flash", "qwen3.8-27b", "kimi-k2.5"]), chip=chip, serving=sv,
                         layout=rnd.choice([Layout(), Layout(tp=2)]), node_cards=8)
            r, r0 = _ev(s), _ev(replace(s, serving=replace(sv, images=0)))
            if not r or not r0:
                continue
            v = r.vision
            assert v["image_tokens"] == n * v["tokens_per_image"] > 0
            assert not (r.fits and not r0.fits)
            if sv.phase == "prefill":
                assert v["s"] > 0 and v["tflop"] > 0 and r.ttft >= r0.ttft * (1 - E)
                rt = _ev(replace(s, serving=replace(sv, images=0, prompt=sv.prompt + v["image_tokens"])))
                assert rt is None or r.ttft >= rt.ttft * (1 - E)            # encoder + image tokens ≥ text-equivalent
                rm = _ev(replace(s, serving=replace(sv, images=2 * n)))
                assert rm is None or rm.ttft >= r.ttft * (1 - E)
        seen[kind] = seen.get(kind, 0) + 1
    assert all(seen.get(k, 0) >= 8 for k in ("spec", "skew", "d2d", "dtype", "vlm")), seen
