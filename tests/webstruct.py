"""0.286: structural (format-agnostic) views of the web UI sources for tests.

Tests used to assert exact app.js / index.html / pyodide-shim.js source strings (``"x = f(y)" in js``).  Those break on
harmless edits (whitespace, quote style, a comment, a renamed local), which trains people to re-pin the new text instead
of re-checking the guarded behaviour (0.285: test_core_276/277).  This module tokenizes the sources with the stdlib so a
test can assert *what the code does* instead:

* ``tokens(src)`` — JS tokens; comments/whitespace dropped, string literals decoded (quote style irrelevant), template
  literals split into their static parts (``str`` tokens) and their ``${…}`` expressions (ordinary tokens).
* ``has(src_or_tokens, snippet)`` — the snippet's token sequence occurs (whitespace, quotes, comments, trailing commas
  and line breaks do not matter; any change to identifiers, operators or literals does).
* ``count(src_or_tokens, snippet)`` — number of non-overlapping occurrences.
* ``functions(js)`` — ``{name: body_tokens}`` for ``function f(…) {…}``, ``async function``, and
  ``const f = (…) => {…}`` / ``const f = async (…) => {…}`` / ``const f = function (…) {…}``.
* ``strings(js)`` — every string-literal value and template static part (UI labels, routes, field keys).
* ``idents(js)`` — every identifier / property name.
* ``req_calls(js)`` — ``req(chan, path, body)`` calls: ``{"chan", "path", "fn", "body"}`` (body = arg tokens).
* ``literal(js, name)`` — tokens of the object/array literal bound by ``const name = {…}`` / ``[…]``.
* ``list_values(js, key)`` — string values of an ``X.key || [ … ]`` (shim route lists) or ``key = [ … ]`` literal.
* ``html_ids(html)``, ``html_options(html)``, ``html_attr_values(html, attr)`` — via ``html.parser``.
* ``between(toks, a, b)`` — token slice from the first occurrence of snippet ``a`` to the next ``b``.

Stdlib only (re, html.parser)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "accel_dse" / "web"
APP_JS = WEB / "app.js"
INDEX_HTML = WEB / "index.html"
SHIM_JS = ROOT / "web-static" / "pyodide-shim.js"

_PUNCT = sorted(""">>>= ... === !== **= <<= >>= >>> ?.( ?? ??= &&= ||= => == != <= >= && || ++ -- += -= *= /= %= &= |= ^= ** << >> ?.
{ } ( ) [ ] ; , < > + - * / % & | ^ ! ~ ? : = . @ #""".split(), key=len, reverse=True)
_KW_BEFORE_REGEX = {"return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw", "case", "do", "else",
                    "yield", "await"}
_ID = re.compile(r"[A-Za-z_$\u00a0-\uffff][\w$\u00a0-\uffff]*")
_NUM = re.compile(r"0[xX][\da-fA-F_]+n?|0[bB][01_]+n?|0[oO][0-7_]+n?|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?n?")
_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


class Tok(tuple):
    """(kind, value): kind ∈ id, num, str, punct, re."""
    __slots__ = ()

    def __new__(cls, kind, value):
        return tuple.__new__(cls, (kind, value))

    kind = property(lambda s: s[0])
    value = property(lambda s: s[1])


def _unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "u" and s[i + 2:i + 3] == "{":
                j = s.index("}", i)
                out.append(chr(int(s[i + 3:j], 16)))
                i = j + 1
                continue
            if n == "u":
                out.append(chr(int(s[i + 2:i + 6], 16)))
                i += 6
                continue
            if n == "x":
                out.append(chr(int(s[i + 2:i + 4], 16)))
                i += 4
                continue
            if n == "\n":
                i += 2
                continue
            out.append(_ESC.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _regex_ok(prev) -> bool:
    if prev is None:
        return True
    k, v = prev
    if k == "punct":
        return v not in (")", "]", "}")
    if k == "id":
        return v in _KW_BEFORE_REGEX
    return False


def _lex(src: str, i: int, out: list, stop_brace: bool) -> int:
    """Tokenize from i; when stop_brace, return at the '}' closing a template ${…} (index just past it)."""
    depth = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n\ufeff\u2028\u2029":
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            i = src.index("*/", i + 2) + 2
            continue
        prev = out[-1] if out else None
        if c in "'\"":
            j = i + 1
            while src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(Tok("str", _unescape(src[i + 1:j])))
            i = j + 1
            continue
        if c == "`":
            out.append(Tok("punct", "`"))
            j, buf = i + 1, []
            while True:
                ch = src[j]
                if ch == "\\":
                    buf.append(src[j:j + 2])
                    j += 2
                elif ch == "`":
                    out.append(Tok("str", _unescape("".join(buf))))
                    j += 1
                    break
                elif src.startswith("${", j):
                    out.append(Tok("str", _unescape("".join(buf))))
                    buf = []
                    out.append(Tok("punct", "${"))
                    j = _lex(src, j + 2, out, True)
                    out.append(Tok("punct", "}"))
                else:
                    buf.append(ch)
                    j += 1
            out.append(Tok("punct", "`"))
            i = j
            continue
        if c == "/" and _regex_ok(prev) and not src.startswith("/=", i) or (c == "/" and _regex_ok(prev) and prev and prev[1] in ("(", ",", "=")):
            j, cls = i + 1, False
            while True:
                ch = src[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "[":
                    cls = True
                elif ch == "]":
                    cls = False
                elif ch == "/" and not cls:
                    break
                elif ch == "\n":
                    raise ValueError(f"unterminated regex at {i}")
                j += 1
            m = re.compile(r"[a-z]*").match(src, j + 1)
            out.append(Tok("re", src[i:m.end()]))
            i = m.end()
            continue
        m = _NUM.match(src, i) if (c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit())) else None
        if m and m.end() > i:
            out.append(Tok("num", m.group().replace("_", "")))
            i = m.end()
            continue
        m = _ID.match(src, i)
        if m:
            out.append(Tok("id", m.group()))
            i = m.end()
            continue
        for p in _PUNCT:
            if src.startswith(p, i):
                if p == "?.(":           # optional call: emit ?. then (
                    p = "?."
                if stop_brace and p == "{":
                    depth += 1
                if stop_brace and p == "}":
                    if depth == 0:
                        return i + 1
                    depth -= 1
                out.append(Tok("punct", p))
                i += len(p)
                break
        else:
            raise ValueError(f"cannot tokenize {src[i:i + 20]!r} at {i}")
    if stop_brace:
        raise ValueError("unterminated template expression")
    return i


_CACHE: dict = {}


def tokens(src) -> list:
    if isinstance(src, list):
        return src
    key = hash(src)
    if key not in _CACHE:
        out: list = []
        _lex(src, 0, out, False)
        _CACHE[key] = _norm(out)
    return _CACHE[key]


def _norm(t: list) -> list:
    """Drop trailing commas before a closer and empty template parts at the edges are kept (they carry no text)."""
    return [x for k, x in enumerate(t) if not (x == ("punct", ",") and k + 1 < len(t) and t[k + 1][1] in (")", "]", "}")
                                              and t[k + 1][0] == "punct")]


def snippet(s: str) -> list:
    return tokens(s)


def _find(t: list, p: list, start: int = 0) -> int:
    if not p:
        return start
    first, n = p[0], len(p)
    for i in range(start, len(t) - n + 1):
        if t[i] == first and t[i:i + n] == p:
            return i
    return -1


def index(src, s: str, start: int = 0) -> int:
    """Token index of the first occurrence of snippet s at or after start (-1 if absent)."""
    return _find(tokens(src), snippet(s), start)


def has(src, s: str) -> bool:
    return _find(tokens(src), snippet(s)) >= 0


def count(src, s: str) -> int:
    t, p, c, i = tokens(src), snippet(s), 0, 0
    while True:
        i = _find(t, p, i)
        if i < 0:
            return c
        c += 1
        i += len(p)


def _match(t: list, i: int) -> int:
    """Index of the closer matching the opener at t[i] (t[i] is ( [ { or ${)."""
    pairs = {"(": ")", "[": "]", "{": "}", "${": "}"}
    stack = []
    for j in range(i, len(t)):
        k, v = t[j]
        if k != "punct":
            continue
        if v in pairs:
            stack.append(pairs[v])
        elif stack and v == stack[-1]:
            stack.pop()
            if not stack:
                return j
    raise ValueError("unbalanced")


def functions(js) -> dict:
    t = tokens(js)
    return {k: t[a:b] for k, (a, b) in function_spans(js).items()}


def function_spans(js) -> dict:
    """``{name: (start, end)}``: body token span (exclusive of the braces) in ``tokens(js)``."""
    key = ("spans", hash(js) if isinstance(js, str) else id(js))
    if key not in _CACHE:
        _CACHE[key] = _function_spans(tokens(js))
    return _CACHE[key]


def _function_spans(t: list) -> dict:
    out = {}
    for i in range(len(t) - 2):
        name = None
        if t[i] == ("id", "function") and t[i + 1][0] == "id":
            name, j = t[i + 1][1], i + 2
        elif t[i] in (("id", "const"), ("id", "let")) and t[i + 1][0] == "id" and t[i + 2] == ("punct", "="):
            j = i + 3
            if t[j] == ("id", "async"):
                j += 1
            if t[j] == ("id", "function"):
                j += 1
            if t[j] == ("punct", "("):
                k = _match(t, j)
                if t[k + 1] == ("punct", "=>") or t[k + 1] == ("punct", "{"):
                    name = t[i + 1][1]
            elif t[j][0] == "id" and t[j + 1] == ("punct", "=>"):
                name = t[i + 1][1]
        if name is None or name in out:
            continue
        while t[j] != ("punct", "{"):
            if t[j] == ("punct", "("):
                j = _match(t, j)
            j += 1
            if t[j - 1] == ("punct", "=>") and t[j] != ("punct", "{"):
                break
        if t[j] == ("punct", "{"):
            out[name] = (j + 1, _match(t, j))
        else:                       # expression-bodied arrow: up to the ; or , that ends the declaration
            k = j
            while k < len(t) and not (t[k][0] == "punct" and t[k][1] in (";", ",", ")", "]", "}")):
                k = _match(t, k) + 1 if t[k][0] == "punct" and t[k][1] in ("(", "[", "{", "${") else k + 1
            out[name] = (j, k)
    return out


def function_source(js: str, name: str) -> str:
    """Source text of a named function declaration (``function name(…) {…}``, optionally async), for running it
    (e.g. under node when available)."""
    m = re.search(r"(?:async\s+)?function\s+%s\s*\(" % re.escape(name), js)
    assert m, name
    k = m.end() - 1
    while True:
        k = js.index("{", k + 1)
        head: list = []
        try:
            _lex(js[m.start():k], 0, head, False)
        except (ValueError, IndexError):
            continue
        if sum(x == ("punct", "(") for x in head) == sum(x == ("punct", ")") for x in head) and head[-1] == ("punct", ")"):
            return js[m.start():_lex(js, k + 1, [], True)]


def props(toks, obj: str) -> set:
    """Property names read off ``obj`` (``obj.x`` / ``obj?.x``) in toks, obj not itself a property."""
    t = tokens(toks)
    return {t[i + 2][1] for i in range(1, len(t) - 2)
            if t[i] == ("id", obj) and t[i + 1][1] in (".", "?.") and t[i + 2][0] == "id" and t[i - 1][1] not in (".", "?.")}


_IDPATH = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*")


def mentions(js, s: str) -> bool:
    """Does the UI source use ``s``?  A bare name must be an identifier or an exact string literal (a field key or
    element id); a dotted path (``p.dac.x``) or code (``k: 'v'``, ``a === 'b'``) must occur as a token sequence; any
    other text (labels with spaces / CJK / hyphens) must occur inside a string literal or template part."""
    if re.fullmatch(r"[A-Za-z_$][\w$]*", s):
        return s in idents(js) or s in strings(js)
    if _IDPATH.fullmatch(s) or re.search(r"['\"`=(\[:]", s):
        return has(js, s)
    return any_string_contains(js, s)


def stale_tracked(js) -> list:
    """Element ids invalidate() tags stale (ids listed in its ``for (const id of [ … ])`` loops)."""
    fn, out = functions(js)["invalidate"], []
    for i in range(len(fn) - 6):
        if fn[i:i + 6] == snippet("for (const id of ["):
            out += [v for k, v in fn[i + 6:_match(fn, i + 5)] if k == "str"]
    return out


def catch_status_writes(js) -> list:
    """``(function, textContent ids, dataset.done ids)`` per ``catch (e) {…}`` block."""
    out = []
    for st, b in blocks(js, "catch (e)"):
        fn = enclosing(js, st)
        al = element_aliases(functions(js)[fn]) if fn else {}
        out.append((fn, element_writes(b, "textContent", al), element_writes(b, "dataset.done", al)))
    return out


def element_refs(js) -> set:
    """Ids looked up with ``$('id')`` / ``getElementById('id')`` (literal ids only)."""
    t, out = tokens(js), set()
    for i in range(len(t) - 3):
        if t[i] in (("id", "$"), ("id", "getElementById")) and t[i + 1] == ("punct", "(") and t[i + 2][0] == "str" \
                and t[i + 3] == ("punct", ")") and (t[i][1] == "$" or t[i - 1] == ("punct", ".")):
            out.add(t[i + 2][1])
    return out


def js_created_ids(js) -> set:
    """Ids the UI creates itself (``id: 'x'`` in an element-attribute object, or ``.id = 'x'``)."""
    t = tokens(js)
    return {t[i + 2][1] for i in range(len(t) - 2)
            if t[i] == ("id", "id") and t[i + 1][1] in (":", "=") and t[i + 2][0] == "str"}


def strings(js) -> set:
    return {v for k, v in tokens(js) if k == "str"}


def idents(js) -> set:
    return {v for k, v in tokens(js) if k == "id"}


def any_string_contains(js, frag: str) -> bool:
    return any(frag in s for s in strings(js))


def between(src, a: str, b: str) -> list:
    t = tokens(src)
    i = _find(t, snippet(a))
    assert i >= 0, a
    j = _find(t, snippet(b), i + 1)
    assert j >= 0, b
    return t[i:j]


def _args(t: list, i: int) -> list:
    """Split the argument list whose '(' is at t[i] into token lists."""
    end, args, cur, j = _match(t, i), [], [], i + 1
    while j < end:
        if t[j][0] == "punct" and t[j][1] in ("(", "[", "{", "${"):
            k = _match(t, j)
            cur.extend(t[j:k + 1])
            j = k + 1
            continue
        if t[j] == ("punct", ","):
            args.append(cur)
            cur = []
        else:
            cur.append(t[j])
        j += 1
    if cur:
        args.append(cur)
    return args


def calls(src, name: str) -> list:
    """Argument token lists of every call ``name(…)`` (``name`` may be dotted, e.g. ``S.sc.set``)."""
    t, p, out, i = tokens(src), snippet(name + "("), [], 0
    while True:
        i = _find(t, p, i)
        if i < 0:
            return out
        if i == 0 or t[i - 1] != ("punct", "."):
            out.append(_args(t, i + len(p) - 1))
        i += len(p)


def enclosing(js, idx: int) -> str | None:
    """Innermost named function whose body contains token index idx."""
    best = None
    for name, (a, b) in function_spans(js).items():
        if a <= idx < b and (best is None or b - a < best[1]):
            best = (name, b - a)
    return best and best[0]


def req_calls(js) -> list:
    """Every ``req(chan, path[, body])`` call site (not the definition).  chan is the string value, or the joined token
    text when it is an expression (e.g. ``'stab-' + row.mapping``)."""
    t, p, out, i = tokens(js), snippet("req("), [], 0
    while True:
        i = _find(t, p, i)
        if i < 0:
            return out
        if (i == 0 or t[i - 1] not in (("punct", "."), ("id", "function"))):
            a = _args(t, i + 1)
            if len(a) >= 2 and len(a[1]) == 1 and a[1][0][0] == "str":
                chan = a[0][0][1] if len(a[0]) == 1 and a[0][0][0] == "str" else " ".join(v for _, v in a[0])
                out.append({"chan": chan, "path": a[1][0][1], "fn": enclosing(js, i), "body": a[2] if len(a) > 2 else [],
                            "at": i})
        i += len(p)


def literal(js, name: str) -> list:
    t = tokens(js)
    for i in range(len(t) - 3):
        if t[i][1] in ("const", "let", "var") and t[i + 1] == ("id", name) and t[i + 2] == ("punct", "=") \
                and t[i + 3][1] in ("{", "["):
            return t[i + 4:_match(t, i + 3)]
    raise KeyError(name)


def list_values(src, key: str) -> set:
    """String values of ``….key || [ … ]`` (shim route lists, CFG fallbacks) or ``key = [ … ]``."""
    t = tokens(src)
    for i in range(len(t) - 2):
        if t[i] == ("id", key) and t[i + 1][1] in ("||", "??", "=") and t[i + 2] == ("punct", "["):
            return {v for k, v in t[i + 3:_match(t, i + 2)] if k == "str"}
    raise KeyError(key)


def object_keys(toks: list) -> list:
    """Top-level keys of an object literal body (tokens between the braces)."""
    keys, depth, expect = [], 0, True
    for k, v in toks:
        if k == "punct" and v in ("(", "[", "{", "${"):
            depth += 1
        elif k == "punct" and v in (")", "]", "}"):
            depth -= 1
        elif depth == 0 and k == "punct" and v == ",":
            expect = True
        elif depth == 0 and expect and k in ("id", "str", "num"):
            keys.append(v)
            expect = False
    return keys


def object_dict(toks: list) -> dict:
    """``{key: 'string'}`` entries of an object literal body whose value is a single string literal."""
    out, i = {}, 0
    while i + 2 < len(toks):
        if toks[i][0] in ("id", "str", "num") and toks[i + 1] == ("punct", ":") and toks[i + 2][0] == "str" \
                and (i + 3 == len(toks) or toks[i + 3][1] in (",", "}")):
            out[toks[i][1]] = toks[i + 2][1]
            i += 3
        else:
            i += 1
    return out


def snapshot_request(js, path: str) -> dict:
    """The UI's call to ``path`` sends a click-time scenario snapshot: its body is ``{ ...SNAP, … }`` where SNAP is
    bound exactly once in the calling function, from ``body(…)``, before the first loop that issues the request.
    Returns ``{"fn", "snap", "body"}``; raises AssertionError naming the broken condition."""
    cs = [c for c in req_calls(js) if c["path"] == path]
    assert len(cs) == 1, (path, [c["fn"] for c in cs])
    c = cs[0]
    b, fn = c["body"], functions(js)[c["fn"]]
    assert b[:2] == [("punct", "{"), ("punct", "...")] and b[2][0] == "id", (path, "body is not { ...snapshot, … }")
    snap = b[2][1]
    binds = [i for i in range(len(fn) - 1) if fn[i] == ("id", snap) and fn[i + 1] == ("punct", "=")]
    assert len(binds) == 1, (path, snap, "snapshot must be bound exactly once")
    assert fn[binds[0] + 2:binds[0] + 4] == [("id", "body"), ("punct", "(")], (path, snap, "snapshot is not body(…)")
    at = index(fn, f"req({c['chan']!r}")
    loop = max((i for i in range(at) if fn[i] in (("id", "for"), ("id", "while"))), default=-1)
    assert binds[0] < loop or loop < 0, (path, "snapshot taken inside the request loop")
    return {"fn": c["fn"], "snap": snap, "body": b}


def blocks(src, head: str) -> list:
    """``(start, body_tokens)`` for every ``head { … }`` (e.g. ``catch (e)``); start = token index of head."""
    t, p, out, i = tokens(src), snippet(head), [], 0
    while True:
        i = _find(t, p, i)
        if i < 0:
            return out
        j = i + len(p)
        if j < len(t) and t[j] == ("punct", "{"):
            out.append((i, t[j + 1:_match(t, j)]))
        i = j


def element_aliases(fn: list) -> dict:
    """``{alias: element_id}`` for ``const|let alias = $('id')`` inside a function body (also ``a = $('x'), b = …``)."""
    out = {}
    for i in range(len(fn) - 5):
        if fn[i][0] == "id" and fn[i + 1] == ("punct", "=") and fn[i + 2] == ("id", "$") and fn[i + 3] == ("punct", "(") \
                and fn[i + 4][0] == "str" and fn[i + 5] == ("punct", ")"):
            out[fn[i][1]] = fn[i + 4][1]
    return out


def element_writes(toks: list, prop: str, aliases: dict) -> set:
    """Element ids whose ``.prop`` (dotted, e.g. ``textContent`` or ``dataset.done``) is assigned in toks, through
    ``$('id').prop =`` or an alias from element_aliases."""
    p, out = snippet("." + prop + " ="), set()
    n = len(p)
    for i in range(len(toks) - n):
        if toks[i:i + n] != p:
            continue
        if toks[i - 1] == ("punct", ")") and toks[i - 2][0] == "str" and toks[i - 3] == ("punct", "(") \
                and toks[i - 4] == ("id", "$"):
            out.add(toks[i - 2][1])
        elif toks[i - 1][0] == "id" and toks[i - 1][1] in aliases and (i < 2 or toks[i - 2] != ("punct", ".")):
            out.add(aliases[toks[i - 1][1]])
    return out


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.options, self.attrs, self.text = [], [], [], []
        self.select_opts: dict = {}
        self._select = None
        self._labels: list = []           # open <label>s: [title, text parts, control ids]
        self.label_for: dict = {}         # control id → "title text" of its enclosing <label>

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.attrs.append((tag, a))
        if "id" in a:
            self.ids.append(a["id"])
        if tag == "label":
            self._labels.append([a.get("title") or "", [], []])
        elif self._labels and "id" in a:
            self._labels[-1][2].append(a["id"])
        if tag == "select":
            self._select = a.get("id")
            self.select_opts.setdefault(self._select, [])
        if tag == "option":
            self.options.append(a.get("value"))
            if self._select is not None:
                self.select_opts[self._select].append(a.get("value"))

    def handle_endtag(self, tag):
        if tag == "select":
            self._select = None
        if tag == "label" and self._labels:
            title, text, ctl = self._labels.pop()
            for c in ctl:
                self.label_for[c] = (title + " " + "".join(text)).strip()

    def handle_data(self, data):
        self.text.append(data)
        if self._labels:
            self._labels[-1][1].append(data)


def parse_html(html: str) -> _HTML:
    p = _HTML()
    p.feed(html)
    return p


def html_ids(html: str) -> set:
    return set(parse_html(html).ids)


def html_options(html: str) -> set:
    return set(parse_html(html).options)


def select_options(html: str, select_id: str) -> list:
    """Option values of ``<select id=select_id>`` in document order."""
    return parse_html(html).select_opts[select_id]


def label_text(html: str, control_id: str) -> str:
    """Title + visible text of the ``<label>`` that wraps the control ``control_id``."""
    return parse_html(html).label_for[control_id]


def html_attr_values(html: str, attr: str) -> set:
    return {a[attr] for _, a in parse_html(html).attrs if attr in a}


def html_strings(html: str) -> list:
    """Every attribute value and text node (titles, option values, labels) — the page's data, without markup."""
    p = parse_html(html)
    return [v for _, a in p.attrs for v in a.values() if v] + p.text


def html_text(html: str) -> str:
    return " ".join(parse_html(html).text)


def read(path: Path = APP_JS) -> str:
    return Path(path).read_text()
