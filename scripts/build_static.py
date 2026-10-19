"""Build the static (GitHub Pages) web UI: the unchanged frontend + a fetch shim that answers /api/* with
accel_dse running on Pyodide in a Web Worker.  No server needed: `python3 -m http.server -d dist-static`.

    python3 scripts/build_static.py [--out dist-static] [--pyodide 0.29.5]

Output: index.html (app's, plus the shim), app.js / style.css (verbatim), pyodide-shim.js, pyodide-worker.js,
static-config.js, accel_dse-<hash>.zip (the package .py + data/*.json; web/ and CLI-only parts excluded).
Pyodide itself is loaded from the official jsdelivr CDN at a pinned version.  Stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "accel_dse"
WEB = PKG / "web"
STATIC_SRC = ROOT / "web-static"
PYODIDE_VERSION = "0.29.5"          # Python 3.13 (same minor as the server reference runs)
CDN = "https://cdn.jsdelivr.net/pyodide/v{v}/full/"


def package_zip() -> bytes:
    """accel_dse/**/*.py + data/**/*.json, deterministic (sorted, fixed timestamps) so the hash is stable."""
    files = sorted(p for p in PKG.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix in (".py", ".json") and WEB not in p.parents)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            zi = zipfile.ZipInfo(str(p.relative_to(ROOT)), date_time=(2020, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, p.read_bytes())
    return buf.getvalue()


def build(out: Path, pyodide: str) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    z = package_zip()
    zname = f"accel_dse-{hashlib.sha256(z).hexdigest()[:12]}.zip"
    (out / zname).write_bytes(z)
    for f in ("app.js", "style.css"):
        shutil.copy2(WEB / f, out / f)
    for f in ("pyodide-shim.js", "pyodide-worker.js"):
        shutil.copy2(STATIC_SRC / f, out / f)
    cfg = {"pyodideBase": CDN.format(v=pyodide), "packageZip": zname, "worker": "pyodide-worker.js"}
    (out / "static-config.js").write_text("self.ACCEL_STATIC = " + json.dumps(cfg) + ";\n", encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    tag = '<script src="app.js"></script>'
    if tag not in html:
        raise SystemExit("index.html: app.js script tag not found")
    html = html.replace(tag, '<script src="static-config.js"></script>\n<script src="pyodide-shim.js"></script>\n' + tag)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")
    return {"out": str(out), "zip": zname, "zip_bytes": len(z),
            "total_local_bytes": sum(p.stat().st_size for p in out.iterdir()), "pyodide": pyodide}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "dist-static"))
    ap.add_argument("--pyodide", default=PYODIDE_VERSION)
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.out), a.pyodide), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
