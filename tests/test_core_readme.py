"""Guard: version history lives only in CHANGELOG.md; the READMEs carry no per-version content."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readmes_have_no_version_bullets():
    for f in ("README.md", "README.en.md"):
        bad = [l for l in (ROOT / f).read_text(encoding="utf-8").splitlines()
               if re.match(r"\s*- \*\*v?\d+\.\d+", l) or re.search(r"[（(]v?0\.\d+(\.\d+)?\s*([–-]\s*0\.\d+)?[）)]", l)
               or re.search(r"\b(?:in|since|before|adds?) 0\.\d{2}\b|0\.\d{2}(?:\.\d+)? (?:增加|接入|起|前)", l)]
        assert not bad, (f, bad[:3])
