"""Command line: serve | models | eval | search | compare | stability | validate."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, api


def _scenario_args(p: argparse.ArgumentParser, layout: bool = True) -> None:
    p.add_argument("--model", default="qwen3-8b", help="model id (see `models`)")
    p.add_argument("--chip", default="100T", help="chip preset: 100T | 1P | H100-like")
    p.add_argument("--mem", default=None, help="memory id, e.g. lpddr5x_4x64_8533_16g / hbm3e_8s_12h24g_9200")
    p.add_argument("--mem-eff", type=float, default=None, help="DRAM efficiency 「假设」 (default 0.7)")
    p.add_argument("--mapping", default="os", help="os | ws_edge | ws_broad | os_vec | reconf")
    p.add_argument("--phase", default="decode", choices=["decode", "prefill"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ctx", type=int, default=None, help="decode context length")
    p.add_argument("--prompt", type=int, default=None, help="prompt length (prefill / goodput)")
    p.add_argument("--spec-k", type=int, default=0)
    p.add_argument("--spec-accept", type=float, default=None)
    p.add_argument("--tpot-slo", type=float, default=None, help="TPOT SLO ms")
    if layout:
        for k in ("pp", "tp", "dp", "ep", "etp"):
            p.add_argument(f"--{k}", type=int, default=1)
    p.add_argument("--json", action="store_true", help="print raw JSON")


def _body(a: argparse.Namespace, layout: bool = True) -> dict:
    sv = {"phase": a.phase, "batch": a.batch, "spec_k": a.spec_k}
    for k, attr in (("ctx", "ctx"), ("prompt", "prompt"), ("spec_accept", "spec_accept"), ("tpot_slo_ms", "tpot_slo")):
        if getattr(a, attr) is not None:
            sv[k] = getattr(a, attr)
    sc = {"model": a.model, "mapping": a.mapping, "serving": sv}
    if a.mem:
        sc["mem_id"] = a.mem
    if a.mem_eff is not None:
        sc["mem_eff"] = a.mem_eff
    if layout:
        sc["layout"] = {k: getattr(a, k) for k in ("pp", "tp", "dp", "ep", "etp")}
    return {"chip_preset": a.chip, "scenario": sc}


def _table(rows: list[list], head: list[str]) -> None:
    cells = [head] + [[("" if c is None else f"{c:.3g}" if isinstance(c, float) else str(c)) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(head))]
    for i, r in enumerate(cells):
        print("  ".join(c.ljust(w[j]) for j, c in enumerate(r)))
        if i == 0:
            print("  ".join("-" * x for x in w))


def cmd_eval(a) -> dict:
    body = _body(a)
    body["best_batch"] = a.best_batch
    out = api.api_eval(body)
    if a.json:
        return out
    s = out["summary"]
    m = out["model"]
    print(f"{m['id']}  [{m['provenance']} · {m['coverage']} · {m['dtype']}]{'  「架构代理」' if m['proxy_badge'] else ''}")
    print(f"layout {s['layout']}  mapping {s['mapping']}  batch {s['batch']}  bound {s['bound']}  "
          f"array_util {s['array_util']:.1%}")
    if s["phase"] == "decode":
        print(f"TPOT {s['tpot_ms']:.2f} ms   {s['tok_s']:.1f} tok/s   {s['tok_s_card']:.1f} tok/s/card")
        if g := out.get("goodput"):
            print(f"goodput {g['tok_s_card']:.1f} tok/s/card   TTFT {g['ttft_ms']:.0f} ms "
                  f"{'OK' if g['ttft_ok'] else 'over SLO'}")
    else:
        print(f"TTFT {s['ttft_ms']:.1f} ms   {s['tok_s']:.0f} prompt tok/s")
    print(f"DRAM need {s['dram_need_GiB']:.1f} GiB / card   fits {s['fits']}")
    for w in s["warnings"]:
        print("  ! " + w)
    return {}


def cmd_search(a) -> dict:
    body = _body(a, layout=False)
    body.update(cards=a.cards, objective=a.objective)
    out = api.api_layouts(body)
    if a.json:
        return out
    _table([[r["layout"], r["batch"], r["tok_s_card"], r.get("goodput_card"), r["tpot_ms"], r["bound"],
             r["array_util"], r.get("ttft_ok")] for r in out["rows"]],
           ["layout", "batch", "tok/s/card", "goodput/card", "TPOT ms", "bound", "util", "TTFT ok"])
    return {}


def cmd_compare(a) -> dict:
    body = _body(a, layout=False)
    body.update(cards=a.cards, objective=a.objective)
    out = api.api_compare(body)
    if a.json:
        return out
    _table([[r["mapping"], r["layout"], r["batch"], r["tok_s_card"], r.get("goodput_card"), r["bound"],
             r["array_util"], r.get("ttft_ok")] for r in out["rows"]],
           ["mapping", "best layout", "batch", "tok/s/card", "goodput/card", "bound", "util", "TTFT ok"])
    return {}


def cmd_stability(a) -> dict:
    body = _body(a, layout=False)
    body.update(cards=a.cards, objective=a.objective, include_mapping=a.include_mapping)
    out = api.api_stability(body)
    if a.json:
        return out
    print(f"top-1 {out['base_top']}  agree {out['agree']}/{len(out['cases'])}  "
          f"{'稳定 stable' if out['stable'] else '不稳定 unstable'}")
    for c in out["cases"]:
        print("  ", c)
    return {}


def cmd_models(a) -> dict:
    out = api.api_models()
    if a.json:
        return out
    _table([[m["id"], m["provenance"], m["coverage"] + (" 「架构代理」" if m["proxy_badge"] else ""), m["dtype"],
             m["params_B"], m["active_B"]] for m in out["models"]],
           ["id", "provenance", "coverage", "dtype", "params B", "active B"])
    return {}


def cmd_validate(a) -> dict:
    from .core.validation import v2_trends, v3_genz
    v2, v3 = v2_trends(), v3_genz()
    if a.json:
        return {"v2": v2, "v3": v3}
    print("V2 trend bands (H100-like 「假设」)")
    _table([[r["check"], r["value"], f"[{r['lo']}, {r['hi']}]", r["ok"]] for r in v2], ["check", "value", "band", "ok"])
    print("\nV3 GenZ reference (Llama-3.1-8B decode)")
    _table([[r["mem_GBps"], r["batch"], r["ctx"], r["mapping"], r["tpot_ms"], r["ours_ms"], r["ratio"],
             r["comparable"]] for r in v3], ["mem GB/s", "batch", "ctx", "mapping", "GenZ ms", "ours ms", "ratio",
                                              "comparable"])
    return {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="accel-dse", description=f"accel_dse {__version__} (core v2)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="web UI + JSON API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    m = sub.add_parser("models", help="list models with provenance / coverage / dtype labels")
    m.add_argument("--json", action="store_true")
    e = sub.add_parser("eval", help="evaluate one scenario")
    _scenario_args(e)
    e.add_argument("--best-batch", action="store_true", help="largest batch meeting the TPOT SLO")
    for name, helptext in (("search", "rank layouts for N cards"), ("compare", "best layout per mapping"),
                           ("stability", "ranking stability under assumption perturbations")):
        x = sub.add_parser(name, help=helptext)
        _scenario_args(x, layout=False)
        x.add_argument("--cards", type=int, default=8)
        x.add_argument("--objective", default="decode", choices=["decode", "goodput"])
        if name == "stability":
            x.add_argument("--include-mapping", action="store_true")
    v = sub.add_parser("validate", help="V2 trend bands + V3 GenZ comparison")
    v.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .serve import run_server
        run_server(a.host, a.port)
        return 0
    fn = {"models": cmd_models, "eval": cmd_eval, "search": cmd_search, "compare": cmd_compare,
          "stability": cmd_stability, "validate": cmd_validate}[a.cmd]
    try:
        out = fn(a)
    except api.ApiError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if out:
        print(json.dumps(api.clean(out), ensure_ascii=False, indent=1, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
