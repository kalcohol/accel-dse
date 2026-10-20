"""0.221: default memory realism — no default may resolve to a catalog configuration with no product unless it is
an explicit what-if (mem_catalog.WHATIF_DEFAULTS)."""
import re
from pathlib import Path

from accel_dse import mem_catalog as m
from accel_dse.core import hardware as hw
from accel_dse.core.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]


def _default_ids():
    ids = {"MEM_LEGACY_0204": hw.MEM_LEGACY_0204, "System.mem_id": hw.System(hw.CHIP_100T).mem_id}
    ids.update({f"PRESET_MEM[{k}]": v for k, v in hw.PRESET_MEM.items()})
    ids.update({f"default_mem({k})": hw.default_mem(c) for k, c in hw.CHIPS.items()})
    t = W.tokens(W.read())                           # 0.286: `|| 'lpddr…'` fallbacks as tokens (any quote style)
    fb = [t[i + 1][1] for i in range(len(t) - 1)
          if t[i] == ("punct", "||") and t[i + 1][0] == "str" and re.match(r"(?:lpddr|hbm)[0-9a-z_]+$", t[i + 1][1])]
    assert fb
    for i, x in enumerate(fb):
        ids[f"app.js fallback {i}"] = x
    page = " ".join(W.html_strings(W.read(W.INDEX_HTML)))      # 0.286: attribute values + text, not raw markup
    for x in re.findall(r"\b((?:lpddr5x?|lpddr6|hbm3e?|hbm4e?)_[0-9a-z_]+_[0-9.]+g?)\b", page):
        ids[f"index.html {x}"] = x
    return ids


def test_preset_defaults_have_products():
    bad = {k: (v, m.parse_mem_id(v).product_status) for k, v in _default_ids().items()
           if m.parse_mem_id(v).product_status == "none"}
    assert not bad, bad


def test_catalog_defaults_have_products_or_are_flagged_whatif():
    bad = []
    for t, cat in m.LPDDR_CATALOG.items():
        for fn, f in cat["forms"].items():
            for w in f["widths"]:
                s = m.make_spec(t, form=fn, width_bits=w)
                if s.product_status == "none" and (t, fn, w) not in m.WHATIF_DEFAULTS:
                    bad.append(s.id)
    for t in m.HBM_CATALOG:
        s = m.default_spec(t)
        if s.product_status == "none":
            bad.append(s.id)
    assert not bad, bad
    # flagged what-ifs really are what-ifs (no stale flags) and carry a 假设 reason
    for (t, fn, w), why in m.WHATIF_DEFAULTS.items():
        assert m.make_spec(t, form=fn, width_bits=w).product_status == "none", (t, fn, w)
        assert why.startswith("假设"), why


def test_scenario_default_memory_has_product():
    sc = Scenario(model="qwen3-8b", chip="H100-like")
    assert m.parse_mem_id(sc.mem_id).product_status != "none"


def test_defaults_0221_old_new():
    assert m.LPDDR_CATALOG["LPDDR5"]["forms"]["discrete"]["default_cap"][32] == 8          # was 6 (no product)
    assert m.make_spec("LPDDR5", width_bits=32).id == "lpddr5_4x32_6400_8g"
    h = m.parse_mem_id("hbm3_5s_8h16g_5234")
    assert h.component_status()["rate"] == {"spec": "jedec_likely", "product": "shipping"}
    assert abs(h.raw_GBps - 3349.76) < 0.01                                                 # unchanged number
    assert m.HBM_RATE_SOURCES[("HBM3", 5234)].startswith("来源/推导")


# --- separate DSpark draft + target: ReaDy-Qwen3.8-27B (both configs public, 2026-10-17) --------------------------
import random
from dataclasses import replace

from accel_dse import api
from accel_dse.core import catalog as C
from accel_dse.core import ir
from accel_dse.core.evaluate import evaluate
from accel_dse.core.model import spec_accept_from_tau
from accel_dse.core.schedule import spec_expected_tokens
from accel_dse.core.scenario import Layout, Serving
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

RD = "qwen3.8-27b-ready"


def test_ready_draft_structure_and_params():
    m, t = C.get_model(RD), C.get_model("qwen3.8-27b")
    assert m.draft == "dspark" and m.draft_block == 7 and m.draft_targets == 5 and m.draft_markov == 256
    assert m.draft_reader == (8, 2, 48, 128, 128, 128, 4) and m.spec_published == (4.5285, 7)
    assert len(m.mtp_layers) == 5 and all(L.core.window is None for L in m.mtp_layers)
    ours = sum(L.params(m.hidden) for L in m.mtp_layers) + m.mtp_extra_params * len(m.mtp_layers)
    assert abs(ours - 1857358337 - 4894720) < 10                 # draft + reader safetensors headers
    assert m.layers == t.layers and m.vocab == t.vocab           # the target is Qwen3.8-27B as released


def test_spec_accept_from_tau():
    assert abs(spec_accept_from_tau(4.5285, 7) - 0.8279) < 1e-3
    assert abs(spec_accept_from_tau(4.1739, 7) - 0.8010) < 1e-3
    for a in (0.0, 0.3, 0.7, 0.95):
        assert abs(spec_accept_from_tau(spec_expected_tokens(7, a), 7) - a) < 1e-6
    try:
        spec_accept_from_tau(9.0, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("tau > k + 1 must fail")


def test_reader_ops_and_bytes():
    m = C.get_model(RD)
    for b, tp in ((1, 1), (4, 2)):
        ph = ir.Phase("decode", b, 1 + 7, 4096)
        ops = ir.dspark_ops(m, ph, ir.Shard(tp=tp))
        g = [o for o in ops if o.name == "ready_state_gather"]
        assert len(g) == 1
        assert g[0].state_rw == 3 * b * 2 * (48 // tp) * 128 * 128 * 4          # fp32 state, gather + buffer + read
    assert not any(o.name.startswith("ready_") for o in ir.dspark_ops(m, ir.Phase("prefill", 1, 512, 0)))
    mu = C.get_model("muse-spark-open-nvfp4-dspark")
    assert not mu.draft_reader and not any(o.name.startswith("ready_") for o in
                                           ir.dspark_ops(mu, ir.Phase("decode", 1, 8, 4096)))


def test_ready_eval_vs_paper_mechanism():
    """Not a fit: the published τ sets the lever; the modelled speed-up is reported, ReaDy/DSpark-like ratio checked
    only for direction (reader costs < 1 % of the step)."""
    sv = {"batch": 1, "ctx": 4096}
    ar = api.api_eval({"chip_preset": "1P", "scenario": {"model": RD, "serving": sv}})
    sp = api.api_eval({"chip_preset": "1P", "scenario": {"model": RD, "serving": {**sv, "spec_k": 7, "spec_accept": 0.8279}}})
    assert sp["model"]["spec_published"]["accept_iid"] == 0.8279
    assert sp["summary"]["tok_s"] > 3 * ar["summary"]["tok_s"]
    m = C.get_model(RD)
    from accel_dse.core import hardware as hw
    from accel_dse.core.scenario import Scenario
    s = Scenario(model=RD, chip=hw.CHIP_1P, mem_id=hw.default_mem(hw.CHIP_1P), serving=Serving(batch=1, ctx=4096, spec_k=7,
                                                                                                spec_accept=0.8279))
    r1, r0 = evaluate(s, m), evaluate(s, replace(m, draft_reader=()))
    assert r0.step <= r1.step <= r0.step * 1.01


# --- adversarial fuzzer (0.221): the separate-draft (DSpark + reader) path ------------------------------------------
def test_draft_path_fuzz():
    rnd, n = random.Random(221), 0
    from accel_dse.core import hardware as hw
    from accel_dse.core.scenario import Scenario
    m = C.get_model(RD)
    for _ in range(6):
        b, ctx, tp = rnd.choice([1, 4, 32]), rnd.choice([512, 4096, 16384]), rnd.choice([1, 2, 4])
        k, a = rnd.randint(1, 8), rnd.choice([0.0, 0.4, 0.83, 1.0])
        chip = rnd.choice([hw.CHIP_1P, hw.CHIP_H100_LIKE])
        base = Scenario(model=RD, chip=chip, mem_id=hw.default_mem(chip), layout=Layout(tp=tp),
                        serving=Serving(batch=b, ctx=ctx, spec_k=k, spec_accept=a))
        r = evaluate(base, m)
        kk = min(k, m.draft_block)
        assert abs(r.tokens_per_step - spec_expected_tokens(kk, a)) <= 1e-9 * r.tokens_per_step, (b, ctx, tp, k, a)
        r_hi = evaluate(replace(base, serving=replace(base.serving, spec_accept=min(1.0, a + 0.1))), m)
        assert r_hi.step == r.step and r_hi.tpot <= r.tpot * (1 + 1e-12)       # acceptance is free, never slower
        r_off = evaluate(replace(base, serving=replace(base.serving, spec_k=0)), m)
        assert r.step >= r_off.step                                            # a draft pass + verify never shortens a step
        r_nr = evaluate(base, replace(m, draft_reader=()))
        assert r_nr.step <= r.step                                             # the reader only adds work
        n += 1
    assert n == 6
