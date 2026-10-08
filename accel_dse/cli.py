"""Command line: serve | models | eval | search | compare | stability | validate."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, api
from .core.d2d_catalog import D2D_DEFAULT_STD, D2D_DEFAULT_UNITS, D2D_STANDARDS


_BUDGET_FLAGS = (
    ("--budget-sram-mib", "b_sram_mib", "max SRAM MiB per card"), ("--budget-slc-mib", "b_slc_mib", "max SLC MiB per card"),
    ("--budget-macs", "b_macs", "max MAC units per card"), ("--budget-tflops", "b_tflops", "max peak bf16 TFLOPS per card"),
    ("--budget-dram-GiB", "b_dram_GiB", "max DRAM GiB per card"), ("--budget-cards", "b_cards", "max cards per replica"),
    ("--budget-power-W", "b_power_W_card", "max average W per card (needs the energy table)"),
    ("--budget-die-mm2", "b_die_mm2", "max die-area proxy per card, mm²"),
    ("--budget-system-mm2", "b_system_mm2", "max silicon-area proxy per replica, mm²"),
    ("--mm2-per-mib-sram", "b_mm2_per_mib_sram", "area density 「假设」"), ("--mm2-per-mib-slc", "b_mm2_per_mib_slc", "area density 「假设」"),
    ("--mm2-per-kmac", "b_mm2_per_kmac", "mm² per 1024 MACs 「假设」"), ("--mm2-fixed", "b_mm2_fixed", "fixed mm² per card 「假设」"))


def _load_list(v: str) -> list[float]:
    import os
    if os.path.exists(v):
        with open(v) as f:
            x = json.load(f)
        if not isinstance(x, list):
            raise SystemExit("--moe-expert-load file must hold a JSON list")
        return x
    return [float(t) for t in v.split(",") if t.strip()]


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
                            ("--pJ-bit-link", "pJ_bit_link", "per in-node scale-up link bit"),
                            ("--idle-W", "idle_W", "W per card"),
                            ("--pJ-bit-slc", "pJ_bit_slc", "per SLC bit"), ("--pJ-bit-d2d", "pJ_bit_d2d", "per D2D bit"),
                            ("--pJ-bit-net", "pJ_bit_net", "per cross-node network bit")):
        p.add_argument(flag, dest=dest, type=float, default=None,
                       help=f"energy table (user-supplied, no default): {hlp}")
    p.add_argument("--slc-mib", dest="slc_mib", type=float, default=None,
                   help="system-level cache per card, MiB (0.48; default 0 = none) 「假设」")
    p.add_argument("--slc-GBps", dest="slc_GBps", type=float, default=None, help="SLC bandwidth (default 2000 「假设」)")
    p.add_argument("--slc-policy", dest="slc_policy", default=None, choices=["pin", "lru"],
                   help="SLC policy: pin (software-pinned like SRAM) | lru (all-or-nothing cyclic bound)")
    p.add_argument("--link-GBps", dest="link_GBps", type=float, default=None,
                   help="in-node scale-up tier bandwidth per rank (default 400 「假设」)")
    p.add_argument("--link-alpha-us", dest="link_alpha_us", type=float, default=None, help="scale-up tier α (default 3)")
    p.add_argument("--d2d", dest="d2d_enabled", action="store_true", default=None,
                   help="chiplet stacking with a die-to-die tier (0.50; default off = monolithic die)")
    p.add_argument("--package-cards", dest="package_cards", type=int, default=None,
                   help="dies per package on the D2D tier (> 1 implies --d2d; default 1)")
    p.add_argument("--d2d-std", dest="d2d_std", default=None, choices=[*D2D_STANDARDS, "custom"],
                   help=f"D2D grade (default {D2D_DEFAULT_STD}; custom = --d2d-GBps)")
    p.add_argument("--d2d-units", dest="d2d_units", type=int, default=None,
                   help=f"D2D units per die (UCIe modules / BoW slices / links; default {D2D_DEFAULT_UNITS} 「假设」)")
    p.add_argument("--d2d-GBps", dest="d2d_GBps", type=float, default=None,
                   help="custom D2D bandwidth per die (implies --d2d-std custom)")
    p.add_argument("--d2d-alpha-us", dest="d2d_alpha_us", type=float, default=None, help="D2D α (default 0.5 「假设」)")
    p.add_argument("--node-cards", dest="node_cards", type=int, default=None,
                   help="cards per node for the cross-node tier (0.50; default 0 = one node, tier unused)")
    p.add_argument("--net-GBps", dest="net_GBps", type=float, default=None,
                   help="cross-node (IB / RoCE) bandwidth per card (default 50 = one 400 Gb/s NIC 「假设」)")
    p.add_argument("--net-alpha-us", dest="net_alpha_us", type=float, default=None, help="cross-node α (default 5 「假设」)")
    p.add_argument("--moe-skew", dest="moe_skew", type=float, default=None,
                   help="MoE: busiest EP rank's token load / mean (0.49; default 1 = uniform routing) 「假设」")
    p.add_argument("--moe-expert-load", dest="moe_expert_load", default=None,
                   help="MoE: measured tokens per routed expert — JSON file with a list, or comma-separated numbers "
                        "(skew derived per EP layout, contiguous expert placement)")
    p.add_argument("--pd", action="store_true", help="LLM: prefill / decode disaggregation report (0.50; decode pool = "
                   "the layout flags, prefill pool = --pd-prefill-*)")
    for k in ("pp", "tp", "dp", "ep", "etp"):
        p.add_argument(f"--pd-prefill-{k}", dest=f"pd_prefill_{k}", type=int, default=1, help=f"PD prefill layout {k}")
    p.add_argument("--pd-prefill-cards", dest="pd_prefill_cards", type=int, default=0, help="PD prefill pool cards (0 = one replica)")
    p.add_argument("--pd-decode-cards", dest="pd_decode_cards", type=int, default=0, help="PD decode pool cards (0 = one replica)")
    p.add_argument("--pd-kv-GBps", dest="pd_kv_GBps", type=float, default=None,
                   help="PD KV transfer GB/s per card (default: network link) 「假设」")
    p.add_argument("--pd-layerwise", action="store_true", help="PD: stream KV layer by layer during prefill")
    for flag, dest, hlp in _BUDGET_FLAGS:
        p.add_argument(flag, dest=dest, type=float, default=None, help=f"budget (0.49, user-supplied): {hlp}")
    p.add_argument("--json", action="store_true", help="print raw JSON")


def _body(a: argparse.Namespace, layout: bool = True) -> dict:
    sv = {"phase": a.phase, "batch": a.batch, "spec_k": a.spec_k}
    for k, attr in (("ctx", "ctx"), ("prompt", "prompt"), ("spec_accept", "spec_accept"), ("tpot_slo_ms", "tpot_slo")):
        if getattr(a, attr) is not None:
            sv[k] = getattr(a, attr)
    if getattr(a, "moe_skew", None) is not None:
        sv["moe_skew"] = a.moe_skew
    if getattr(a, "moe_expert_load", None):
        sv["moe_expert_load"] = _load_list(a.moe_expert_load)
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
    for tier, pre in (("link", "link"), ("d2d", "d2d"), ("net", "net")):
        lk = {k: getattr(a, f"{pre}_{k}") for k in ("GBps", "alpha_us") if getattr(a, f"{pre}_{k}", None) is not None}
        if lk:
            sc[tier] = lk
    for k in ("package_cards", "d2d_enabled", "d2d_std", "d2d_units", "node_cards"):
        if getattr(a, k, None) is not None:
            sc[k] = getattr(a, k)
    if sc.get("package_cards", 1) > 1 and "d2d_enabled" not in sc:
        sc["d2d_enabled"] = True
    if getattr(a, "pd", False):
        sc["pd"] = {"enabled": True, "prefill_layout": {k: getattr(a, f"pd_prefill_{k}") for k in ("pp", "tp", "dp", "ep", "etp")},
                    "prefill_cards": a.pd_prefill_cards, "decode_cards": a.pd_decode_cards,
                    "kv_GBps": a.pd_kv_GBps, "kv_layerwise": a.pd_layerwise}
    body = {"chip_preset": a.chip, "scenario": sc}
    en = {k: getattr(a, k) for k in ("pJ_mac", "pJ_vec", "pJ_bit_sram", "pJ_bit_dram", "pJ_bit_link", "idle_W",
                                     "pJ_bit_slc", "pJ_bit_d2d", "pJ_bit_net")
          if getattr(a, k, None) is not None}
    if en:
        body["energy"] = en
    bud = {d[2:]: (int(getattr(a, d)) if d == "b_cards" else getattr(a, d))
           for _, d, _ in _BUDGET_FLAGS if getattr(a, d, None) is not None}
    if bud:
        body["budget"] = bud
    return body


def _table(rows: list[list], head: list[str]) -> None:
    cells = [head] + [[("" if c is None else f"{c:.3g}" if isinstance(c, float) else str(c)) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(head))]
    for i, r in enumerate(cells):
        print("  ".join(c.ljust(w[j]) for j, c in enumerate(r)))
        if i == 0:
            print("  ".join("-" * x for x in w))


def _d2d_GBps(sc: dict) -> float:
    if sc["d2d_std"] == "custom":
        return sc["d2d"]["GBps"]
    return D2D_STANDARDS[sc["d2d_std"]]["per_unit_GBps"] * sc["d2d_units"]


def cmd_d2d(a) -> dict:
    out = {"standards": [{"id": k, **v} for k, v in D2D_STANDARDS.items()], "default_std": D2D_DEFAULT_STD,
           "default_units": D2D_DEFAULT_UNITS, "default_enabled": False}
    if a.json:
        return out
    print("D2D tier: off by default (monolithic die). --d2d turns it on; default grade "
          f"{D2D_DEFAULT_STD} x{D2D_DEFAULT_UNITS} units/die 「假设」")
    for k, v in D2D_STANDARDS.items():
        print(f"  {k:<11} {v['per_unit_GBps']:>6.0f} GB/s per {v['unit']:<14} {v['pJ_bit']:.2g} pJ/bit  "
              f"{'open' if v['open'] else 'vendor'}  {v['label']}")
    return {}


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
    sc = out["scenario"]
    if sc.get("d2d_enabled") or sc.get("node_cards"):
        d2d = (f"D2D {sc['package_cards']} dies/pkg @ {_d2d_GBps(sc):.0f} GB/s ({sc['d2d_std']}"
               + (f" x{sc['d2d_units']}" if sc["d2d_std"] != "custom" else "") + ")") if sc.get("d2d_enabled") \
            else "D2D off (monolithic)"
        net = (f"net {sc['net']['GBps']:.0f} GB/s across nodes of {sc['node_cards']} cards" if sc.get("node_cards")
               else "one node")
        print(f"interconnect: {d2d} | scale-up {sc['link']['GBps']:.0f} GB/s | {net}")
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
              + (f"  SLC {c['slc']:.3g} B" if c.get("slc") else "") + (f"  D2D {c['d2d']:.3g} B" if c.get("d2d") else "")
              + (f"  net {c['net']:.3g} B" if c.get("net") else ""))
        if "J_per_unit" in e:
            print(f"energy {e['J_per_unit']:.4g} J / {e['unit']}  (avg {e['avg_W_per_card']:.0f} W/card; user-supplied "
                  f"table: {', '.join(e['provided'])}; missing: {', '.join(e['missing']) or '-'})  "
                  + "  ".join(f"{k} {v:.3g}" for k, v in e["J_by_action"].items()))
    if b := out.get("budget"):
        mark = {True: "OK", False: "OVER", None: "?"}
        def val(i):
            return "—" if i["value"] is None else f"{i['value']:.4g}"
        print(f"budget {mark[b['ok']]}: " + "  ".join(
            f"{i['label']} {val(i)} / {i['limit']:g} {i['unit']} [{mark[i['ok']]}]" for i in b["items"]))
        for i in b["items"]:
            if i["note"]:
                print(f"  {i['label']}: {i['note']}")
    if (pd := out.get("pd")) and "error" not in pd:
        p_, d_, k_, c_ = pd["prefill"], pd["decode"], pd["kv"], pd["coloc"]
        print(f"PD: prefill {p_['cards']} cards ({p_['replicas']} × {p_['layout']}, batch {p_['batch']}, {p_['ttft_ms']:.0f} ms)  "
              f"decode {d_['cards']} cards ({d_['replicas']} × {d_['layout']}, batch {d_['batch']})  "
              f"KV {k_['bytes_per_req'] / 2**20:.1f} MiB/req, {k_['t_ms']:.2f} ms (exposed {k_['exposed_ms']:.2f})")
        print(f"    TTFT {pd['ttft_ms']:.0f} ms  TPOT {pd['tpot_ms']:.2f} ms  goodput {pd['goodput_per_card']:.1f} tok/s/card  "
              f"bottleneck {pd['bottleneck']}"
              + (f"  best split {pd['best_split']['prefill_cards']}P+{pd['best_split']['decode_cards']}D "
                 f"{pd['best_split']['goodput_per_card']:.1f}" if pd.get("best_split") else ""))
        print(f"    colocated {c_['cards']} cards: TTFT {c_['ttft_ms']:.0f} ms  TPOT {c_['tpot_ms']:.2f} ms "
              f"(effective {c_['tpot_eff_ms']:.2f})  goodput {c_['goodput_per_card']:.1f} tok/s/card")
        for w in pd["warnings"]:
            print("    ! " + w)
    elif pd:
        print("PD: " + pd["error"])
    for w in s["warnings"]:
        print("  ! " + w)
    return {}


def _bmark(r: dict) -> str:
    ok = r.get("budget_ok")
    return "OK" if ok is True else "?" if ok is None else "OVER " + ",".join(r.get("budget_violations", []))


def cmd_search(a) -> dict:
    body = _body(a, layout=False)
    body.update(cards=a.cards, objective=a.objective)
    out = api.api_layouts(body)
    if a.json:
        return out
    bud = "budget" in body
    _table([[r["layout"], r["batch"], r["tok_s_card"], r.get("goodput_card"), r["tpot_ms"], r["bound"],
             r["array_util"], r.get("ttft_ok")] + ([_bmark(r)] if bud else []) for r in out["rows"]],
           ["layout", "batch", "tok/s/card", "goodput/card", "TPOT ms", "bound", "util", "TTFT ok"]
           + (["budget"] if bud else []))
    return {}


def cmd_compare(a) -> dict:
    body = _body(a, layout=False)
    body.update(cards=a.cards, objective=a.objective)
    out = api.api_compare(body)
    if a.json:
        return out
    bud = "budget" in body
    _table([[r["mapping"], r["layout"], r["batch"], r["tok_s_card"], r.get("goodput_card"), r["bound"],
             r["array_util"], r.get("ttft_ok")] + ([_bmark(r)] if bud else []) for r in out["rows"]],
           ["mapping", "best layout", "batch", "tok/s/card", "goodput/card", "bound", "util", "TTFT ok"]
           + (["budget"] if bud else []))
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
    d = sub.add_parser("d2d", help="list the selectable die-to-die (D2D) grades (0.50)")
    d.add_argument("--json", action="store_true")
    v = sub.add_parser("validate", help="V2 trend bands + V3 GenZ comparison")
    v.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .serve import run_server
        run_server(a.host, a.port)
        return 0
    fn = {"models": cmd_models, "eval": cmd_eval, "search": cmd_search, "compare": cmd_compare,
          "stability": cmd_stability, "validate": cmd_validate, "d2d": cmd_d2d}[a.cmd]
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
