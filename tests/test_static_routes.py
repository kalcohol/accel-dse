"""0.261: static-site robustness (GitHub Pages build).  Every /api route the UI calls is served by serve.respond with
the method app.js uses; the shim's slow-lane list names real routes; and the package zip the Pyodide worker unpacks
is self-sufficient — the API answers from the unpacked zip alone (isolated interpreter, no repo tree on sys.path),
so no runtime file outside the package (docs/, scripts/, tests/) is needed and no package file is left out."""
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from accel_dse import serve

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "accel_dse" / "web" / "app.js").read_text(encoding="utf-8")
SHIM = (ROOT / "web-static" / "pyodide-shim.js").read_text(encoding="utf-8")


def _build_mod():
    spec = importlib.util.spec_from_file_location("build_static", ROOT / "scripts" / "build_static.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ui_calls() -> set[tuple[str, str]]:
    """(method, path) of every req(chan, '/api/…'[, body]) in app.js — GET when no body argument (app.js req())."""
    calls = set()
    for m in re.finditer(r"\breq\(\s*[^,]+,\s*'(/api/[^'?]*)'\s*([,)])", APP):
        calls.add(("POST" if m.group(2) == "," else "GET", m.group(1)))
    return calls


def test_ui_calls_parsed_and_no_other_api_literals():
    calls = ui_calls()
    assert len(calls) >= 15 and ("GET", "/api/health") in calls and ("POST", "/api/eval") in calls
    lits = set(re.findall(r"'(/api/[a-z_]+)", APP))
    assert lits == {p for _, p in calls}                     # no /api path reaches fetch some other way
    assert "fetch(" in APP and not re.search(r"fetch\(\s*['`]/api", APP)   # all API traffic goes through req()


def test_every_ui_call_is_routed_with_its_method():
    for method, path in sorted(ui_calls()):
        routes = serve.GET_ROUTES if method == "GET" else serve.POST_ROUTES
        assert path in routes, (method, path)
        st, payload = serve.respond(method, path, b"[1]" if method == "POST" else b"")
        assert st != 404 and payload, (method, path, st)
        if method == "GET":
            assert st == 200, (path, st)


def test_shim_slow_lane_names_real_routes():
    m = re.search(r"slowRoutes \|\| \[([^\]]*)\]", SHIM)
    slow = set(re.findall(r"'(/api/[a-z_]+)'", m.group(1)))
    assert slow and slow <= set(serve.POST_ROUTES)


def test_zip_has_every_tracked_package_file():
    m = _build_mod()
    with zipfile.ZipFile(__import__("io").BytesIO(m.package_zip())) as z:
        zn = set(z.namelist())
    try:
        out = subprocess.run(["git", "ls-files", "accel_dse"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        tracked = {l for l in out.stdout.split() if not l.startswith("accel_dse/web/")} if out.returncode == 0 else set()
    except (OSError, subprocess.SubprocessError):
        tracked = set()
    if not tracked:                                           # no git (sdist): fall back to the on-disk tree
        tracked = {str(p.relative_to(ROOT)) for p in (ROOT / "accel_dse").rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and p.suffix != ".pyc" and "web" not in p.relative_to(ROOT).parts[1:2]}
    assert tracked <= zn, sorted(tracked - zn)[:10]


_PROBE = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import accel_dse
assert accel_dse.__file__.startswith(sys.argv[1]), accel_dse.__file__
from accel_dse import serve
out = {}
for method, path, body in json.loads(sys.argv[2]):
    st, payload = serve.respond(method, path, body.encode())
    out[method + " " + path + " " + body[:40]] = [st, payload[:300].decode("utf-8", "replace")]
print(json.dumps(out))
'''

# cheap real requests that read the runtime data: catalogs, releases, drafts, pipeline components, memory catalog,
# spec-tree refs, ext measurements (via eval's validation notes) — a missing file is a 500 / error payload here
PROBES = [("GET", "/api/health", ""), ("GET", "/api/catalog", ""), ("GET", "/api/models", ""),
          ("POST", "/api/memory", '{"id": "lpddr5x_4x64_8533_16g"}'),
          ("POST", "/api/eval", "{}"),
          ("POST", "/api/eval", '{"scenario": {"model": "moss-image-base"}}'),
          ("POST", "/api/eval", '{"scenario": {"model": "qwen3-8b", "serving": {"spec_k": 3, "spec_topk": 4, '
                                '"spec_tree_accept_model": "rank", "draft_model": "qwen3-0.6b"}}}'),
          ("POST", "/api/sensitivity", "{}"),
          ("POST", "/api/insight_speech", '{"models": ["ming-3.0-tts-tiny"], "verify": false}'),   # 0.275
          ("POST", "/api/insight_speech_streams", '{"models": ["whisper-large-v3-turbo"], "slc_gib": [0]}')]   # 0.276


def test_unpacked_zip_answers_alone():
    m = _build_mod()
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(__import__("io").BytesIO(m.package_zip())) as z:
            z.extractall(d)
        r = subprocess.run([sys.executable, "-I", "-c", _PROBE, d, json.dumps(PROBES)], cwd=d,
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        res = json.loads(r.stdout)
    for k, (st, head) in res.items():
        assert st == 200, (k, st, head)
