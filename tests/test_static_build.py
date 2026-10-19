"""Static (Pyodide / GitHub Pages) build: serve.respond is the server's wire format, build_static output is complete."""
import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path

from accel_dse import api, serve

ROOT = Path(__file__).resolve().parents[1]


def _build_mod():
    spec = importlib.util.spec_from_file_location("build_static", ROOT / "scripts" / "build_static.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_respond_is_handle_plus_wire_json():
    for method, path, raw in [("GET", "/api/health", b""), ("POST", "/api/memory", b'{"id": "LPDDR5X"}'),
                              ("POST", "/api/eval", b"[1]"), ("GET", "/api/nope", b""), ("POST", "/api/eval?x=1", b"{")]:
        st, payload = serve.respond(method, path, raw)
        st2, out = serve.handle(method, path.split("?", 1)[0], raw)
        assert st == st2 and payload == json.dumps(api.clean(out), ensure_ascii=False, allow_nan=False).encode()


def test_build_static_bundle():
    m = _build_mod()
    with tempfile.TemporaryDirectory() as d:
        info = m.build(Path(d) / "dist", m.PYODIDE_VERSION)
        out = Path(info["out"])
        names = {p.name for p in out.iterdir()}
        assert {"index.html", "app.js", "style.css", "pyodide-shim.js", "pyodide-worker.js", "static-config.js",
                info["zip"]} <= names
        html = (out / "index.html").read_text(encoding="utf-8")
        assert html.index("pyodide-shim.js") < html.index('src="app.js"')
        cfg = (out / "static-config.js").read_text()
        assert f"/pyodide/v{m.PYODIDE_VERSION}/full/" in cfg and info["zip"] in cfg
        with zipfile.ZipFile(out / info["zip"]) as z:
            zn = set(z.namelist())
        assert "accel_dse/serve.py" in zn and "accel_dse/data/series_catalog.json" in zn
        assert not any("/web/" in n or "__pycache__" in n for n in zn)
        # every data file read at runtime ships (releases, drafts, pipeline, catalogs)
        data = {str(p.relative_to(ROOT)) for p in (ROOT / "accel_dse" / "data").rglob("*.json")}
        assert data <= zn
        assert m.package_zip() == m.package_zip()        # deterministic → stable content hash
