"""0.286 audit of the structural UI checks (tests/webstruct.py users): apply format-only edits (must still PASS) and
behaviour-changing edits (must FAIL) to a scratch copy of the tracked tree, and run the guarding tests there.

Shows that the 0.286 conversion from exact-source asserts to parsed checks is format-agnostic and not weaker: every
behaviour edit below is caught.  The repo tree is never modified.  Opt-in (not part of run_tests; ~2 min).
Usage: python3 scripts/ui_pin_mutations.py [-v]   exit 0 = every expectation met."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS, HTML, SHIM = "accel_dse/web/app.js", "accel_dse/web/index.html", "web-static/pyodide-shim.js"
SR = "tests/test_static_routes.py"

# (label, expect, file, old, new, tests)   expect: "pass" = format-only edit, "fail" = behaviour change
MUTATIONS = [
    ("F1 streams request reflowed, double quotes, trailing comma", "pass", JS,
     "req('insight_speech_streams', '/api/insight_speech_streams', { ...b0, models: [models[i]] })",
     'req(\n        "insight_speech_streams",\n        "/api/insight_speech_streams",\n        { ...b0, models: [ models[i] ], },\n      )',
     ["test_core_276::test_api_route_ui_and_shim", "test_core_277::test_ui_progressive_and_model_lists",
      SR + "::test_insight_panels_stale_handling"]),
    ("F2 compare catch block reflowed over lines", "pass", JS,
     "} catch (e) { $('cmp-status').textContent = '错误：' + e.message; $('cmp-status').dataset.done = '1'; }",
     '} catch (e) {\n    $("cmp-status").textContent = "错误：" + e.message;\n    $("cmp-status").dataset.done = "1";\n  }',
     ["test_core_085::test_ui_verdict_and_error_branches_are_stale_flagged"]),
    ("F3 indexer handler spacing + comment", "pass", JS, "S.sc.indexer = e.target.value",
     "S.sc.indexer=/* select */e.target.value", ["test_core_087::test_ui_selector"]),
    ("F4 export schema spacing", "pass", JS, "const EXP_SCHEMA = 1;", "const  EXP_SCHEMA=1 ;",
     ["test_core_128::test_export_schema_stable", "test_core_151::test_export_wiring_and_schema"]),
    ("F5 html attribute quote style", "pass", HTML, 'id="ins-sp-run"', "id='ins-sp-run'",
     ["test_core_275::test_ui_panel_wired", "test_core_286"]),
    ("F6 shim route list quote style", "pass", SHIM, "'/api/insight_dispatch'", '"/api/insight_dispatch"',
     ["test_core_284::test_api_route_and_ui", SR + "::test_every_ui_call_classified_in_shim"]),
    ("F7 SLC temps trailing comma", "pass", JS, "payload.temps = [25, 85];", "payload.temps = [25, 85,];",
     ["test_core_263::test_api_temps_and_ui"]),
    ("F8 label gains a class, title reflowed", "pass", HTML, '<label title="已弃用（0.268）',
     '<label class="dep"\n        title="已弃用（0.268）', ["test_core_268::test_sqrtc_ui_marked"]),
    # behaviour changes — each must be caught
    ("S1 streams scenario re-read per row (the 0.285 bug)", "fail", JS,
     "{ ...b0, models: [models[i]] }", "{ ...body({ slc_gib: slc }), models: [models[i]] }",
     ["test_core_276::test_api_route_ui_and_shim"]),
    ("S1b same, 277 guard", "fail", JS, "{ ...b0, models: [models[i]] }", "{ ...body({ slc_gib: slc }), models: [models[i]] }",
     ["test_core_277::test_ui_progressive_and_model_lists"]),
    ("S2 compare error not stale-flagged", "fail", JS,
     "} catch (e) { $('cmp-status').textContent = '错误：' + e.message; $('cmp-status').dataset.done = '1'; }",
     "} catch (e) {\n    $('cmp-status').textContent = '错误：' + e.message;\n  }",
     ["test_core_085::test_ui_verdict_and_error_branches_are_stale_flagged"]),
    ("S3 indexer reads the wrong property", "fail", JS, "S.sc.indexer = e.target.value", "S.sc.indexer = e.target.checked",
     ["test_core_087::test_ui_selector"]),
    ("S4 export schema bumped silently", "fail", JS, "const EXP_SCHEMA = 1;", "const EXP_SCHEMA = 2;",
     ["test_core_128::test_export_schema_stable"]),
    ("S5 dispatch insight off the slow lane", "fail", SHIM, "'/api/insight_dispatch'", "'/api/insight_dispatch_x'",
     ["test_core_284::test_api_route_and_ui"]),
    ("S6 'static' label in the walls legend", "fail", JS, "const WALL_LBL = {", "const WALL_LBL = { static: ['静态', 'static'],",
     ["test_core_204::test_ui_wiring"]),
    ("S7 hint no longer refreshed on input change", "fail", JS, "cancel('sens'); swHint(); paintPanel();",
     "cancel('sens'); paintPanel();", ["test_core_090::test_sweep_hint_wiring"]),
    ("S8 latency label bypasses clipZh", "fail", JS, "const latL = full ? (gen ? `${clipZh(m)} s`",
     "const latL = full ? (gen ? `${m.label} s`", ["test_core_130::test_ui_static_fixes"]),
    ("S9 speech progress label changed", "fail", JS, "insight_speech: '语音瓶颈'", "insight_speech: '语音'",
     ["test_core_275::test_ui_panel_wired"]),
    ("S10 UI looks up an element the page lacks", "fail", JS, "const st = $('ins-sps-status');",
     "const st = $('ins-sps-status-x');", ["test_core_286::test_every_ui_element_lookup_exists"]),
    ("S11 dispatch error not stale-flagged", "fail", JS,
     "$('ins-dp-status').textContent = '错误：' + e.message;\n    $('ins-dp-status').dataset.done = '1';",
     "$('ins-dp-status').textContent = '错误：' + e.message;",
     ["test_core_286::test_stale_tracked_panels_flag_errors_and_results"]),
    ("S11b dispatch result not stale-flagged", "fail", JS,
     "行${staleTag(v0)}`;\n    $('ins-dp-status').dataset.done = '1';", "行${staleTag(v0)}`;",
     ["test_core_286::test_stale_tracked_panels_flag_errors_and_results"]),
    ("S12 unclassified routes sent to the fast lane", "fail", SHIM, "SLOW.has(route) || !FAST.has(route) ? slow : fast",
     "SLOW.has(route) ? slow : fast", [SR + "::test_every_ui_call_classified_in_shim"]),
    ("S13 snapshot dropped (per-row body)", "fail", JS, "const b0 = body({ slc_gib: slc });",
     "let b0 = body({ slc_gib: slc }); b0 = body({ slc_gib: slc });", ["test_core_276::test_api_route_ui_and_shim"]),
    ("S14 evaluator emits a speech kind the UI does not render", "fail", "accel_dse/core/speech.py",
     '"kind": "moshi", "label": "Moshi 全双工"', '"kind": "moshi_duplex", "label": "Moshi 全双工"',
     ["test_core_286::test_every_speech_kind_has_a_ui_block"]),
    ("S15 deprecated marker removed", "fail", HTML, '<label title="已弃用（0.268）：', '<label title="（0.268）：',
     ["test_core_268::test_sqrtc_ui_marked"]),
]


def _copy_tree(dst: Path) -> None:
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for f in filter(None, files.split("\0")):
        s = ROOT / f
        if s.is_file():
            (dst / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, dst / f)
    for f in ("tests/webstruct.py", "tests/test_core_286.py", "scripts/ui_pin_mutations.py"):   # before first commit
        if (ROOT / f).is_file():
            shutil.copy2(ROOT / f, dst / f)


def main(verbose: bool = False) -> int:
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _copy_tree(d)
        for label, expect, rel, old, new, tests in MUTATIONS:
            p = d / rel
            orig = p.read_text(encoding="utf-8")
            if orig.count(old) < 1:
                print(f"STALE {label}: anchor not found in {rel}")
                bad += 1
                continue
            p.write_text(orig.replace(old, new, 1), encoding="utf-8")
            try:
                r = subprocess.run([sys.executable, "tests/run_tests.py", *tests], cwd=d, capture_output=True,
                                   text=True, timeout=900)
            finally:
                p.write_text(orig, encoding="utf-8")
            got = "pass" if r.returncode == 0 else "fail"
            ok = got == expect
            bad += not ok
            print(f"{'ok  ' if ok else 'BAD '} {label}: expected {expect}, got {got}")
            if verbose or not ok:
                print("     " + (r.stdout.strip().splitlines() or [""])[-1])
    print(f"{len(MUTATIONS) - bad}/{len(MUTATIONS)} expectations met")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main("-v" in sys.argv))
