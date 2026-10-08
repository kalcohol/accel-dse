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
    for k, hlp in (("frames", "video frames"), ("height", "video height px"), ("width", "video width px"),
                   ("steps", "denoise steps (structure models: diffusion steps)"), ("cfg", "forwards per step (2 = CFG)"),
                   ("seq-len", "protein residues"), ("msa", "MSA rows (structure models)"),
                   ("recycles", "trunk passes incl. the first (structure models)"),
                   ("samples", "diffusion samples per sequence (structure models)")):
        p.add_argument(f"--{k}", type=int, default=0, help=f"{hlp} (0 = release default; video / protein models)")
    if layout:
        for k in ("pp", "tp", "dp", "ep", "etp", "sp"):
            p.add_argument(f"--{k}", type=int, default=1)
    p.add_argument("--dit-only", action="store_true",
                   help="video: evaluate the DiT denoiser only (text encoder + VAE decode not counted; default counts them)")
    p.add_argument("--placement", default=None, choices=["auto", "resident", "shard", "offload", "shard+offload"],
                   help="video: where text encoder / VAE live (default auto: first that fits of resident → shard → "
                        "offload → shard+offload)")
    p.add_argument("--host-GBps", dest="host_GBps", type=float, default=None,
                   help="video offload: host → card bandwidth per card (default 50 GB/s 「假设」)")
    p.add_argument("--vae-tiling", action="store_true",
                   help="video: diffusers enable_tiling() decode (CogVideoX / Mochi / Wan / LTX; HunyuanVideo and "
                        "MiniMax-H3 always tile)")
    p.add_argument("--dit-fsdp", action="store_true",
                   help="video: DiT weights FSDP-sharded over each stage's SP·DP cards (Wan --dit_fsdp)")
    p.add_argument("--te-cpu", action="store_true", help="video: text encoder on the host CPU (Wan --t5_cpu)")
    p.add_argument("--host-TFLOPS", dest="host_TFLOPS", type=float, default=None,
                   help="--te-cpu: effective host CPU TFLOPS for the encoder (default 2 「假设」)")
    p.add_argument("--vae-parallel", action="store_true",
                   help="video: tiled VAE decode split over the replica's cards (MiniMax-H3 parallel_tiling)")
    p.add_argument("--overlap", action="store_true",
                   help="video, with --te-cpu: host encodes the next request while the cards denoise this one")
    p.add_argument("--no-sample-split", action="store_true",
                   help="structure models under DAP: replicate the diffusion samples on every DAP card (default: split)")
    p.add_argument("--act", default=None, choices=["fp32", "bf16"],
                   help="what-if activation dtype (video / protein models; labelled what-if)")
    for flag, dest, hlp in (("--pJ-mac", "pJ_mac", "per bf16-equivalent MAC"), ("--pJ-vec", "pJ_vec", "per vector op"),
                            ("--pJ-bit-sram", "pJ_bit_sram", "per SRAM-port bit"),
                            ("--pJ-bit-dram", "pJ_bit_dram", "per DRAM bit"),
                            ("--pJ-bit-link", "pJ_bit_link", "per link (network tier) bit"),
                            ("--idle-W", "idle_W", "W per card"),
                            ("--pJ-bit-slc", "pJ_bit_slc", "per SLC bit"), ("--pJ-bit-d2d", "pJ_bit_d2d", "per D2D bit")):
        p.add_argument(flag, dest=dest, type=float, default=None,
                       help=f"energy table (user-supplied, no default): {hlp}")
    p.add_argument("--slc-mib", dest="slc_mib", type=float, default=None,
                   help="system-level cache per card, MiB (0.48; default 0 = none) 「假设」")
    p.add_argument("--slc-GBps", dest="slc_GBps", type=float, default=None, help="SLC bandwidth (default 2000 「假设」)")
    p.add_argument("--slc-policy", dest="slc_policy", default=None, choices=["pin", "lru"],
                   help="SLC policy: pin (software-pinned like SRAM) | lru (all-or-nothing cyclic bound)")
    p.add_argument("--link-GBps", dest="link_GBps", type=float, default=None,
                   help="scale-up / network tier bandwidth per rank (default 400 「假设」)")
    p.add_argument("--link-alpha-us", dest="link_alpha_us", type=float, default=None, help="network tier α (default 3)")
    p.add_argument("--package-cards", dest="package_cards", type=int, default=None,
                   help="cards per package on the die-to-die tier (0.48; default 1 = no D2D tier)")
    p.add_argument("--d2d-GBps", dest="d2d_GBps", type=float, default=None, help="D2D bandwidth (default 2000 「假设」)")
    p.add_argument("--d2d-alpha-us", dest="d2d_alpha_us", type=float, default=None, help="D2D α (default 0.5 「假设」)")
    p.add_argument("--json", action="store_true", help="print raw JSON")


def _body(a: argparse.Namespace, layout: bool = True) -> dict:
    sv = {"phase": a.phase, "batch": a.batch, "spec_k": a.spec_k}
    for k, attr in (("ctx", "ctx"), ("prompt", "prompt"), ("spec_accept", "spec_accept"), ("tpot_slo_ms", "tpot_slo")):
        if getattr(a, attr) is not None:
            sv[k] = getattr(a, attr)
    sc = {"model": a.model, "mapping": a.mapping, "serving": sv}
    wl = {k: getattr(a, k) for k in ("frames", "height", "width", "steps", "cfg", "seq_len", "msa", "recycles",
                                     "samples") if getattr(a, k)}
    if getattr(a, "dit_only", False):
        wl["pipeline"] = False
    if getattr(a, "placement", None):
        wl["placement"] = a.placement
    if getattr(a, "host_GBps", None):
        wl["host_GBps"] = a.host_GBps
    if getattr(a, "vae_tiling", False):
        wl["vae_tiling"] = True
    if getattr(a, "dit_fsdp", False):
        wl["dit_fsdp"] = True
    if getattr(a, "te_cpu", False):
        wl["te_cpu"] = True
    if getattr(a, "host_TFLOPS", None):
        wl["host_TFLOPS"] = a.host_TFLOPS
    for k in ("vae_parallel", "overlap"):
        if getattr(a, k, False):
            wl[k] = True
    if getattr(a, "no_sample_split", False):
        wl["sample_split"] = False
    if wl:
        sc["workload"] = wl
    if getattr(a, "act", None):
        sc["formats_override"] = [["act", a.act]]
    if a.mem:
        sc["mem_id"] = a.mem
    if a.mem_eff is not None:
        sc["mem_eff"] = a.mem_eff
    if layout:
        sc["layout"] = {k: getattr(a, k) for k in ("pp", "tp", "dp", "ep", "etp", "sp")}
    chip = {k: getattr(a, k) for k in ("slc_mib", "slc_GBps", "slc_policy") if getattr(a, k, None) is not None}
    if chip:
        sc["chip"] = chip
    for tier, pre in (("link", "link"), ("d2d", "d2d")):
        lk = {k: getattr(a, f"{pre}_{k}") for k in ("GBps", "alpha_us") if getattr(a, f"{pre}_{k}", None) is not None}
        if lk:
            sc[tier] = lk
    if getattr(a, "package_cards", None) is not None:
        sc["package_cards"] = a.package_cards
    body = {"chip_preset": a.chip, "scenario": sc}
    en = {k: getattr(a, k) for k in ("pJ_mac", "pJ_vec", "pJ_bit_sram", "pJ_bit_dram", "pJ_bit_link", "idle_W",
                                     "pJ_bit_slc", "pJ_bit_d2d")
          if getattr(a, k, None) is not None}
    if en:
        body["energy"] = en
    return body


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
    print(f"{m['id']}  [{m['provenance']} · {m['coverage']} · {m['dtype']}{' · what-if' if m.get('what_if') else ''}]"
          f"{'  「架构代理」' if m['proxy_badge'] else ''}")
    print(f"layout {s['layout']}  mapping {s['mapping']}  batch {s['batch']}  bound {s['bound']}  "
          f"array_util {s['array_util']:.1%}")
    if g := s.get("gen"):
        w = g["workload"]
        if g["unit"] == "frame":
            print(f"video {w['width']}x{w['height']} {w['frames']} frames, {w['steps']} steps x CFG {w['cfg']}, "
                  f"{w['seq_tokens']} tokens/forward"
                  + (f" (incl. {w['audio_tokens']} audio)" if w.get("audio_tokens") else "")
                  + (" [factorized S/T attention]" if w.get("attention") == "factorized" else ""))
            print(f"clip {g['clip_s']:.1f} s   {g['s_per_frame']:.2f} s/frame   step {g['step_ms']:.0f} ms   "
                  f"{g['frames_per_s_card']:.3g} frames/s/card   SLO {'OK' if s['slo_ok'] else 'over'}")
            if pl := g.get("pipeline"):
                print(f"pipeline: text {pl['te_s']:.2f} s + denoise {g['denoise_s']:.1f} s + decode {pl['decode_s']:.1f} s"
                      + (f" + host reload {pl['load_s']:.2f} s" if pl.get("load_s") else "")
                      + f"  [placement {pl.get('place', 'resident')}"
                      + (f", TE sharded over {pl['te_cards']} cards" if pl.get("te_cards", 1) > 1 else "")
                      + (f", TE on host CPU @ {pl['host_TFLOPS']:g} TFLOPS" if pl.get("te_cpu") else "")
                      + (f", VAE decode over {pl['vae_par']} cards" if pl.get("vae_par", 1) > 1 else "")
                      + (f", overlapped period {pl['period_s']:.1f} s" if pl.get("overlap") else "") + "];  "
                      + ";  ".join(f"{p['label']} {p['s']:.2f} s {p['tflop']:.1f} TFLOP {p['stored_GB']:.2f} GB "
                                   f"{p['bound']}" for p in pl["parts"]))
            else:
                print("DiT only (text encoder / VAE decode not counted)")
        else:
            if w.get("recycles"):
                print(f"structure: MSA {w.get('msa') or w.get('xmsa') or 0} rows, {w['recycles']} trunk passes"
                      + (f", diffusion {w['diff_steps']} steps x {w['samples']} samples" if w.get("diff_steps") else "")
                      + f", {g['tflop_per_request']:.1f} TFLOP/sequence")
            print(f"protein {w['seq_len']} residues   batch {g['batch_ms']:.1f} ms   {g['seq_per_s_card']:.3g} seq/s/card"
                  f"   {g['residues_per_s_card']:.0f} residues/s/card   SLO {'OK' if s['slo_ok'] else 'over'}")
    elif s["phase"] == "decode":
        print(f"TPOT {s['tpot_ms']:.2f} ms   {s['tok_s']:.1f} tok/s   {s['tok_s_card']:.1f} tok/s/card")
        if g := out.get("goodput"):
            print(f"goodput {g['tok_s_card']:.1f} tok/s/card   TTFT {g['ttft_ms']:.0f} ms "
                  f"{'OK' if g['ttft_ok'] else 'over SLO'}")
    else:
        print(f"TTFT {s['ttft_ms']:.1f} ms   {s['tok_s']:.0f} prompt tok/s")
    print(f"DRAM need {s['dram_need_GiB']:.1f} GiB / card   fits {s['fits']}")
    if e := out.get("energy"):
        c = e["counts_per_unit"]
        print(f"actions / {e['unit']}: MAC {c['mac']:.3g}  vec {c['vec']:.3g}  SRAM {c['sram']:.3g} B  "
              f"DRAM {c['dram']:.3g} B  link {c['link']:.3g} B  card·s {c['idle_card_s']:.3g}"
              + (f"  SLC {c['slc']:.3g} B" if c.get("slc") else "") + (f"  D2D {c['d2d']:.3g} B" if c.get("d2d") else ""))
        if "J_per_unit" in e:
            print(f"energy {e['J_per_unit']:.4g} J / {e['unit']}  (avg {e['avg_W_per_card']:.0f} W/card; user-supplied "
                  f"table: {', '.join(e['provided'])}; missing: {', '.join(e['missing']) or '-'})  "
                  + "  ".join(f"{k} {v:.3g}" for k, v in e["J_by_action"].items()))
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
    print("\n暂未接入 v2（只列在目录中，不能评估；视频 Wan2.1 / CogVideoX 与蛋白质 ESM-2 已在上表中）:")
    _table([[o["id"], o["domain_label"], o["provider_label"], o["arch"]] for o in out["offline"]],
           ["id", "domain", "vendor", "arch"])
    return {}


def cmd_validate(a) -> dict:
    from .core.validation import v2_domain, v2_trends, v3_genz
    v2, v3, vd = v2_trends(), v3_genz(), v2_domain()
    if a.json:
        return {"v2": v2, "v3": v3, "v2_domain": vd}
    print("V2 trend bands (H100-like 「假设」)")
    _table([[r["check"], r["value"], f"[{r['lo']}, {r['hi']}]", r["ok"]] for r in v2], ["check", "value", "band", "ok"])
    print("\nV2 video / protein (workload + FLOP sanity, no hardware calibration)")
    _table([[r["check"], r["value"], f"[{r['lo']}, {r['hi']}]", r["ok"], r["note"]] for r in vd],
           ["check", "value", "band", "ok", "note"])
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
