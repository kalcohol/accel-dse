"""0.286: UI wiring checked through parsed structure (tests/webstruct.py), not exact source text.

* the tokenizer / parser itself: format edits leave every view unchanged, semantic edits change it;
* generalized guards that replace families of exact-string pins: every ``$('id')`` the UI looks up exists, every
  stale-tracked panel flags both its result and its error, every speech kind the evaluator emits has a UI block,
  every endpoint the UI calls has a progress label;
* a lint over tests/: UI source text (app.js / index.html / pyodide-shim.js) may only be consumed through webstruct, so
  new exact-source asserts cannot creep back in.
scripts/ui_pin_mutations.py (opt-in) applies format-only and behaviour-changing edits to a scratch tree and checks the
converted tests pass / fail accordingly; the headless smoke checks the streams snapshot live."""
import ast
import re
from pathlib import Path

from tests import webstruct as W

ROOT = Path(__file__).resolve().parents[1]
JS, HTML, SHIM = W.read(), W.read(W.INDEX_HTML), W.read(W.SHIM_JS)


def test_tokenizer_format_agnostic_and_semantic():
    a = "function f(x = {}) {\n  // c\n  const b0 = body({ slc_gib: slc });  /* d */\n  return req('k', '/api/k', { ...b0, models: [m[i]] });\n}"
    b = 'function f(x={}){const b0=body({slc_gib:slc,});return req("k","/api/k",{...b0,models:[m[i]],});}'
    assert W.tokens(a) == W.tokens(b)
    assert W.has(a, "req( 'k' , \"/api/k\"") and not W.has(a, "req('k', '/api/j'")
    assert not W.has(a, "models: [m[j]]") and W.count(b, "b0") == 2
    t = W.tokens("const r = /[,\\s]+/g, s = `a ${x ? `b${y}` : 'c'} d`; k = a / b / c;")
    assert ("re", "/[,\\s]+/g") in t and ("str", "a ") in t and ("str", " d") in t and ("id", "y") in t
    assert sum(x == ("punct", "/") for x in t) == 2
    assert W.functions(a)["f"] == W.functions(b)["f"]


def test_parser_covers_app_js():
    t = W.tokens(JS)
    opens = sum(x[0] == "punct" and x[1] in ("{", "${") for x in t)
    assert opens == sum(x == ("punct", "}") for x in t)                      # balanced: no mis-lexed regex / template
    fn = W.functions(JS)
    assert set(re.findall(r"^\s*(?:async\s+)?function\s+(\w+)\s*\(", JS, re.M)) <= set(fn)
    assert set(re.findall(r"^(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", JS, re.M)) <= set(fn)
    calls = W.req_calls(JS)
    assert len(calls) == len(re.findall(r"\breq\(\s*['\"]", JS)) and all(c["fn"] for c in calls)
    assert len(W.html_ids(HTML)) == len(re.findall(r"""\bid=(["'])[^"']+\1""", HTML))


def test_every_ui_element_lookup_exists():
    """Every literal ``$('id')`` in app.js names an element of index.html or one the UI creates (a missing id throws
    on first use — the failure mode exact-id pins tried to catch one id at a time)."""
    refs = W.element_refs(JS)
    assert len(refs) > 200
    missing = refs - W.html_ids(HTML) - W.js_created_ids(JS)
    assert not missing, sorted(missing)


def test_stale_tracked_panels_flag_errors_and_results():
    """invalidate() only tags a panel whose status carries dataset.done; so for each tracked status, the success path
    and every catch block that writes it must set dataset.done (0.85 compare / layouts / sweep, 0.285 speech panels)."""
    tracked = W.stale_tracked(JS)
    assert {"cmp-status", "lay-status", "sw-status", "ins-sp-status", "ins-sps-status", "ins-dp-status"} <= set(tracked)
    fns, spans = W.functions(JS), W.function_spans(JS)
    catches = W.blocks(JS, "catch (e)")
    for sid in tracked:
        writers = [n for n, b in fns.items() if sid in W.element_writes(b, "textContent", W.element_aliases(b))
                   and n != "invalidate"]
        assert writers, sid
        for n in writers:
            a, b = spans[n]
            inner = [(s, blk) for s, blk in catches if a <= s < b]
            al = W.element_aliases(fns[n])
            for _, blk in inner:
                if sid in W.element_writes(blk, "textContent", al):
                    assert sid in W.element_writes(blk, "dataset.done", al), (sid, n, "error not stale-flagged")
            outside = list(fns[n])
            for s, blk in inner:              # blank out catch bodies: the success path must flag on its own
                k = s - a
                outside[k:k + len(blk) + 5] = [("punct", ";")] * (len(blk) + 5)
            assert sid in W.element_writes(outside, "dataset.done", al), (sid, n, "result not stale-flagged")


def _speech_kinds() -> set:
    tree = ast.parse((ROOT / "accel_dse/core/speech.py").read_text())
    out = set()
    for d in ast.walk(tree):
        if isinstance(d, ast.Dict):
            for k, v in zip(d.keys, d.values):
                if isinstance(k, ast.Constant) and k.value == "kind" and isinstance(v, ast.Constant):
                    out.add(v.value)
    return out


def test_every_speech_kind_has_a_ui_block():
    """Generalizes the per-kind pins of 0.269–0.280 (``p.kind === 'moshi'`` …): every speech kind the evaluator can
    emit has a KPI branch in the renderer."""
    kinds = _speech_kinds()
    assert {"asr", "tts", "omni", "cosyvoice", "kimi", "moshi", "s2st"} <= kinds
    t = W.tokens(JS)
    ui = {t[i + 4][1] for i in range(len(t) - 4) if t[i:i + 4] == W.snippet("p.kind ===") and t[i + 4][0] == "str"}
    assert kinds <= ui, sorted(kinds - ui)


def test_every_endpoint_has_a_progress_label():
    stage = W.object_dict(W.literal(JS, "STAGE"))
    paths = {c["path"][len("/api/"):] for c in W.req_calls(JS)}
    assert paths <= set(stage), sorted(paths - set(stage))


_UI_FILES = ("app.js", "index.html", "pyodide-shim.js", "style.css")


def _reads_ui_source(node) -> bool:
    src = ast.unparse(node)
    return any(f in src for f in _UI_FILES) and re.search(r"\bread(_text)?\(", src) is not None \
        or (isinstance(node, ast.Call) and ast.unparse(node.func) in ("W.read",))


def test_tests_consume_ui_source_only_through_webstruct():
    """Lint: in tests/, text read from the UI sources is only passed to webstruct (W.*) functions — no ``"…" in js``,
    ``js.index(…)``, ``js.count(…)`` or regex over raw source.  Exceptions: style.css selector checks, the served page
    in test_api (behavioural HTTP check), and the built dist index.html (parsed with W)."""
    bad = []
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text())
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and _reads_ui_source(n.value):
                for tg in n.targets:
                    for e in ast.walk(tg):
                        if isinstance(e, ast.Name):
                            names.add(e.id)
        for n in ast.walk(tree):
            raw = (isinstance(n, ast.Name) and n.id in names and isinstance(n.ctx, ast.Load)) or \
                  (isinstance(n, (ast.Call, ast.Attribute)) and _reads_ui_source(n) and not isinstance(parents.get(n), ast.Assign)
                   and not (isinstance(n, ast.Call) and ast.unparse(n.func).startswith("W.")))
            if not raw:
                continue
            p, ok = parents.get(n), False
            while p is not None and not isinstance(p, ast.stmt):
                if isinstance(p, ast.Call) and ast.unparse(p.func).startswith("W."):
                    ok = True
                    break
                if isinstance(p, ast.Call) and isinstance(n, ast.Name) and p.func is n:
                    break
                p = parents.get(p)
            if ok or "style.css" in ast.unparse(parents.get(n, n)) or (f.name == "test_api.py"):
                continue
            if isinstance(n, ast.Attribute) and isinstance(parents.get(n), ast.Call) and parents[n].func is n:
                continue      # the read call itself; its result is checked where it is used
            bad.append(f"{f.name}:{n.lineno}: {ast.unparse(parents.get(n, n))[:100]}")
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------- (2) expert residency across SRAM / SLC
from accel_dse.core import memplan as M  # noqa: E402
from accel_dse.core.ir import hot_rank_probs  # noqa: E402

G = 2 ** 30
_ST = M.StageStorage(hot_w=1 * G, expert_w=10 * G, kv_per_seq=0.0, idx_per_seq=0.0, state_per_seq=0.0)
_T = {"hot": 1 * G, "exp": 10 * G, "kv_read": 0.0, "kv_write": 0.0, "state": 0.0, "lookup": 0.0}


def _zipf(n=64, k=8):
    return max(hot_rank_probs(tuple(1.0 / (i + 1) for i in range(n)), 1, k), key=sum)


def test_slc_pin_never_over_credits_experts():
    """≤ 0.285: with experts pinned in SRAM, the SLC expert hit share was s/(E−p) applied to every expert read → the
    SLC served more expert bytes than missed the SRAM (negative DRAM bytes).  Now SRAM + SLC + DRAM = the reads."""
    for sram, slc in ((6.1, 6.0), (6.1, 2.0), (3.1, 1.0), (1.5, 8.0)):
        mp = M.plan(_ST, 1, sram_bytes=sram * G, dram_cap=100 * G, max_act=0.0, slc_bytes=slc * G, slc_policy="pin")
        d = M.step_dram_bytes(mp, _ST, _T)
        p_share = mp.pinned_expert / _ST.expert_w
        assert d["exp_dram"] >= -1e-6 and d["weights"] >= -1e-6, (sram, slc, d)
        assert abs(d["exp_dram"] + d["exp_slc"] - _T["exp"] * (1 - p_share)) < 1e-3
        assert abs(d["exp_slc"] - _T["exp"] * mp.slc_expert / _ST.expert_w) < 1e-3      # uniform: byte share of all experts
    # the bug case: SRAM holds half the experts, the SLC the rest → no expert byte from DRAM
    mp = M.plan(_ST, 1, sram_bytes=6.1 * G, dram_cap=100 * G, max_act=0.0, slc_bytes=6 * G, slc_policy="pin")
    assert mp.pinned_expert > 0 and mp.pinned_expert + mp.slc_expert >= _ST.expert_w - 1
    assert abs(M.step_dram_bytes(mp, _ST, _T)["exp_dram"]) < 1e-3


def test_load_aware_sram_pin():
    """A measured expert load: the SRAM pins the most-read experts (0.192 rule ranks bytes by reads saved), so its read
    share exceeds its byte share; the SLC takes the next ones; the three tiers still sum to the expert reads."""
    pp, t = _zipf(), dict(_T, exp=3 * G)
    base = M.plan(_ST, 1, sram_bytes=3.1 * G, dram_cap=100 * G, max_act=0.0, exp_read=3 * G)
    aw = M.plan(_ST, 1, sram_bytes=3.1 * G, dram_cap=100 * G, max_act=0.0, exp_read=3 * G, exp_p=pp)
    f = aw.pinned_expert / _ST.expert_w
    assert base.sram_exp_hit == -1.0 and aw.pinned_expert == base.pinned_expert
    assert f < aw.sram_exp_hit <= 1.0 and abs(M.exp_miss(base, _ST) - (1 - f)) < 1e-12
    assert abs(aw.sram_exp_hit - M.dip_exp_hit(pp, 0.3, f)) < 1e-12
    for slc in (1.0, 4.0, 8.0):
        mp = M.plan(_ST, 1, sram_bytes=3.1 * G, dram_cap=100 * G, max_act=0.0, slc_bytes=slc * G, slc_policy="pin",
                    exp_read=3 * G, exp_p=pp)
        d = M.step_dram_bytes(mp, _ST, t)
        sram = t["exp"] * mp.sram_exp_hit
        assert abs(sram + d["exp_slc"] + d["exp_dram"] - t["exp"]) < 1e-3 and d["exp_dram"] >= -1e-6
        assert abs(mp.slc_frac_exp - (M.dip_exp_hit(pp, 0.3, (mp.pinned_expert + mp.slc_expert) / _ST.expert_w)
                                      - mp.sram_exp_hit)) < 1e-12
    # no expert in SRAM → exactly the 0.263 SLC rule (unchanged)
    mp = M.plan(_ST, 1, sram_bytes=1.0 * G, dram_cap=100 * G, max_act=0.0, slc_bytes=2 * G, slc_policy="pin",
                exp_read=3 * G, exp_p=pp)
    assert mp.pinned_expert == 0 and abs(mp.slc_frac_exp - M.dip_exp_hit(pp, 0.3, mp.slc_expert / _ST.expert_w)) < 1e-12


def test_dip_off_sram_remainder_consistent():
    pp, t = _zipf(), dict(_T, exp=3 * G)
    for sram in (1.0, 3.1):
        mp = M.plan(_ST, 1, sram_bytes=sram * G, dram_cap=100 * G, max_act=0.0, slc_bytes=2 * G, slc_policy="dip",
                    exp_read=3 * G, exp_p=pp)
        d = M.step_dram_bytes(mp, _ST, t)
        h1 = mp.sram_exp_hit
        assert d["exp_dram"] >= -1e-6 and abs(t["exp"] * h1 + d["exp_slc"] + d["exp_dram"] - t["exp"]) < 1e-3
        if mp.pinned_expert == 0:      # unchanged from 0.261 when the SRAM holds no expert
            assert abs(mp.slc_frac_exp - M.dip_exp_hit(pp, 0.3, mp.slc_frac_exp_bytes)) < 1e-12


def test_eval_load_aware_pin_end_to_end():
    from accel_dse import api
    load = [1.0 / (i + 1) for i in range(128)]
    sc = {"model": "qwen3-30b-a3b", "chip": {"name": "100T", "sram_mib": 4096}}
    a = api.api_eval({"scenario": sc, "sensitivity": False})
    b = api.api_eval({"scenario": {**sc, "serving": {"moe_expert_load": load}}, "sensitivity": False})
    ma, mb = a["stages"][-1]["mem"], b["stages"][-1]["mem"]
    assert "sram_exp_hit" not in ma and mb["residency"] == ma["residency"]
    assert ma["residency"] < 1 and 0 < mb["sram_exp_hit"] <= 1
    assert b["summary"]["tpot_ms"] <= a["summary"]["tpot_ms"] * 1.0001 or mb["sram_exp_hit"] >= ma["residency"]


def test_quantized_vision_tower_only_where_released():
    """(2) candidate check: every stored release's vision role is 16-bit except the one with vision_fmt, and
    vision_fmt exists exactly where a vision role is stored in a non-16-bit format."""
    import json
    seen = []
    for f in sorted((ROOT / "accel_dse/data/releases").glob("*.json")):
        d = json.loads(f.read_text())
        roles = d.get("roles")
        roles = ast.literal_eval(roles) if isinstance(roles, str) else roles
        vis = (roles or {}).get("vision") or {}
        q = {k for k, v in vis.items() if k not in ("bf16", "fp16", "fp32") and v.get("params", 0) > 0}
        assert bool(q) == bool(d.get("vision_fmt")), f.name
        if q:
            seen.append(f.stem)
    assert seen == ["nvidia__Qwen3.8-Flash-Next-FP4-Local"]
