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
    p.add_argument("--ttft-slo", type=float, default=None, help="TTFT SLO ms")
    p.add_argument("--out-len", dest="out_len", type=int, default=None, help="output tokens per request (goodput / PD)")
    p.add_argument("--prefix-cached", dest="prefix_cached", type=int, default=None,
                   help="prefill: prompt tokens already in the KV cache (prefix-cache hit, 0.52)")
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
                            ("--idle-W-prefill", "idle_W_prefill", "W per card of the PD prefill pool (default --idle-W)"),
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
    p.add_argument("--fabric", action="store_true",
                   help="0.59 topology-aware collectives (ring / tree / hier α-β + per-step latency, fat-tree oversub, "
                        "rail, scale-up topology, in-network reduction, KV contention) 「假设」; off = 0.50 model")
    p.add_argument("--fabric-algo", dest="fabric_algo", default=None, choices=["auto", "ring", "tree", "hier"])
    p.add_argument("--net-topology", dest="net_topology", default=None, choices=["fat_tree", "rail"],
                   help="cross-node topology with --fabric (default fat_tree)")
    p.add_argument("--oversub", dest="oversub", type=float, default=None,
                   help="leaf uplink oversubscription r (down:up, default 1 = non-blocking) with --fabric 「假设」")
    p.add_argument("--leaf-nodes", dest="leaf_nodes", type=int, default=None,
                   help="nodes per leaf (fat_tree) / per rail switch (rail); default 0 = from --switch-radix")
    p.add_argument("--switch-radix", dest="switch_radix", type=int, default=None, help="leaf / rail switch ports (64)")
    p.add_argument("--link-topology", dest="link_topology", default=None, choices=["switch", "ring", "full_mesh", "torus2d"],
                   help="in-node scale-up topology with --fabric (default switch)")
    p.add_argument("--torus-x", dest="torus_x", type=int, default=None, help="X extent of a torus2d scale-up domain")
    p.add_argument("--innet-reduce", dest="innet_reduce", default=None, choices=["off", "net", "net+link"],
                   help="in-network reduction (SHARP / NVLS-class vendor option 「假设」) with --fabric")
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
    p.add_argument("--pd-load", dest="pd_load", type=float, default=None,
                   help="PD queueing: offered load as a fraction of the PD fluid capacity (0.51; default 0.8 「假设」)")
    p.add_argument("--pd-rate", dest="pd_rate", type=float, default=None,
                   help="PD queueing: absolute offered load, requests/s (overrides --pd-load)")
    p.add_argument("--pd-chunk", dest="pd_chunk", type=int, default=None,
                   help="colocated chunked-prefill token budget per iteration (comparison; default 512 「假设」)")
    p.add_argument("--pd-prompt-cv", dest="pd_prompt_cv", type=float, default=None,
                   help="PD: prompt-length coefficient of variation (lognormal, 8 bins; 0.52 「假设」)")
    p.add_argument("--pd-out-cv", dest="pd_out_cv", type=float, default=None,
                   help="PD: output-length coefficient of variation (lognormal, 8 bins)")
    p.add_argument("--pd-mix", dest="pd_mix", default=None,
                   help="PD: discrete length mix 'weight:prompt:out,…' (e.g. 0.7:1024:256,0.3:11264:1109)")
    p.add_argument("--pd-prefix-hit", dest="pd_prefix_hit", type=float, default=None,
                   help="PD: prefix-cache hit fraction of each prompt (0–0.99; skips that prefill, shrinks KV hand-off)")
    p.add_argument("--pd-prefix-not-on-decode", dest="pd_prefix_not_on_decode", action="store_true",
                   help="PD: the decode pool does not hold the cached prefix (transfer the full KV)")
    p.add_argument("--pd-search-layouts", dest="pd_search_layouts", action="store_true",
                   help="PD: also search the pools' layouts (not only the card split)")
    p.add_argument("--pd-prefill-chip", dest="pd_prefill_chip", default=None,
                   help="PD (0.53): chip preset of the prefill pool, 100T | 1P | H100-like (default: same as decode)")
    p.add_argument("--pd-prefill-mem", dest="pd_prefill_mem", default=None,
                   help="PD (0.53): memory id of the prefill pool (default: same as decode)")
    p.add_argument("--pd-prefix-len", dest="pd_prefix_len", type=int, default=None,
                   help="PD (0.53): shared-prefix length, tokens → prefix-cache capacity + LRU model (0 = off)")
    p.add_argument("--pd-prefix-tree", dest="pd_prefix_tree", default=None,
                   help="PD (0.58): radix prefix tree 'tokens:branching:zipf,…' root → leaf "
                        "(e.g. 1024:8:1,4096:100:1,1024:50:0) → partial prefix matching 「假设」; replaces --pd-prefix-len")
    p.add_argument("--pd-prefix-count", dest="pd_prefix_count", type=int, default=None,
                   help="PD (0.53): distinct prefixes in the working set (default 1000)")
    p.add_argument("--pd-prefix-zipf", dest="pd_prefix_zipf", type=float, default=None,
                   help="PD (0.53): Zipf popularity exponent of the prefixes (default 1.0; 0 = uniform)")
    p.add_argument("--pd-prefix-cache-GB", dest="pd_prefix_cache_GB", type=float, default=None,
                   help="PD (0.53): prefix-cache capacity per replica, GB (default: DRAM left after weights + active KV)")
    p.add_argument("--pd-prefix-affinity", dest="pd_prefix_affinity", action="store_true",
                   help="PD (0.53): prefix-aware routing (replicas partition the prefixes)")
    p.add_argument("--pd-search-decode-batch", dest="pd_search_decode_batch", action="store_true",
                   help="PD (0.53, with --pd-search-layouts): also search each decode layout's batch (B/2…4B, TPOT SLO)")
    p.add_argument("--pd-sim", dest="pd_sim", action="store_true",
                   help="PD (0.54): also run the request-level DES and print simulated tails vs the closed form (slower)")
    p.add_argument("--pd-kv-policy", dest="pd_kv_policy", choices=("off", "wait", "recompute", "swap"), default=None,
                   help="PD (0.55/0.56): decode KV capacity — wait (reserve prompt+output, admission waits) | recompute "
                        "(optimistic admission, preempt youngest + re-prefill) | swap (preempt youngest, KV out/in "
                        "over the host link); default off")
    p.add_argument("--pd-kv-admit", dest="pd_kv_admit", choices=("before_prefill", "after_prefill"), default=None,
                   help="PD (0.56): KV admission order — before_prefill (vLLM: slot + KV reserved before prefill / KV "
                        "pull, the wait is in TTFT and SLO goodput; default) | after_prefill (0.55: wait reported apart)")
    p.add_argument("--pd-swap-GBps", dest="pd_swap_GBps", type=float, default=None,
                   help="PD (0.56, --pd-kv-policy swap): host-link GB/s per card (default workload host_GBps 「假设」)")
    p.add_argument("--pd-kv-capacity-GB", dest="pd_kv_capacity_GB", type=float, default=None,
                   help="PD (0.55): KV capacity per decode replica, GB (default: DRAM left after weights)")
    for flag, dest, hlp in _BUDGET_FLAGS:
        p.add_argument(flag, dest=dest, type=float, default=None, help=f"budget (0.49, user-supplied): {hlp}")
    p.add_argument("--json", action="store_true", help="print raw JSON")


def _parse_mix(text: str) -> list:
    rows = []
    for part in text.split(","):
        bits = part.strip().split(":")
        if len(bits) != 3:
            raise SystemExit(f"--pd-mix: expected weight:prompt:out, got {part!r}")
        try:
            rows.append([float(bits[0]), int(bits[1]), int(bits[2])])
        except ValueError:
            raise SystemExit(f"--pd-mix: bad number in {part!r}") from None
    return rows


def _parse_tree(text: str) -> list:
    rows = []
    for part in text.split(","):
        bits = part.strip().split(":")
        if len(bits) != 3:
            raise SystemExit(f"--pd-prefix-tree: expected tokens:branching:zipf, got {part!r}")
        try:
            rows.append([int(bits[0]), int(bits[1]), float(bits[2])])
        except ValueError:
            raise SystemExit(f"--pd-prefix-tree: bad number in {part!r}") from None
    return rows


def _body(a: argparse.Namespace, layout: bool = True) -> dict:
    sv = {"phase": a.phase, "batch": a.batch, "spec_k": a.spec_k}
    for k, attr in (("ctx", "ctx"), ("prompt", "prompt"), ("spec_accept", "spec_accept"), ("tpot_slo_ms", "tpot_slo"),
                    ("ttft_slo_ms", "ttft_slo"), ("out_len", "out_len"), ("prefix_cached", "prefix_cached")):
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
    fab = {k: getattr(a, dest) for k, dest in (("algo", "fabric_algo"), ("net_topology", "net_topology"),
                                                ("oversub", "oversub"), ("leaf_nodes", "leaf_nodes"),
                                                ("switch_radix", "switch_radix"), ("torus_x", "torus_x"),
                                                ("innet_reduce", "innet_reduce")) if getattr(a, dest, None) is not None}
    if getattr(a, "fabric", False) or fab:
        sc["fabric"] = {"enabled": True, **fab}
    if getattr(a, "link_topology", None):
        sc.setdefault("link", {})["topology"] = a.link_topology
    if sc.get("package_cards", 1) > 1 and "d2d_enabled" not in sc:
        sc["d2d_enabled"] = True
    if getattr(a, "pd", False):
        sc["pd"] = {"enabled": True, "prefill_layout": {k: getattr(a, f"pd_prefill_{k}") for k in ("pp", "tp", "dp", "ep", "etp")},
                    "prefill_cards": a.pd_prefill_cards, "decode_cards": a.pd_decode_cards,
                    "kv_GBps": a.pd_kv_GBps, "kv_layerwise": a.pd_layerwise}
        for k, dest in (("load", "pd_load"), ("rate_rps", "pd_rate"), ("chunk_tokens", "pd_chunk"),
                        ("prompt_cv", "pd_prompt_cv"), ("out_cv", "pd_out_cv"), ("prefix_hit", "pd_prefix_hit"),
                        ("prefill_chip", "pd_prefill_chip"), ("prefill_mem_id", "pd_prefill_mem"),
                        ("prefix_len", "pd_prefix_len"), ("prefix_count", "pd_prefix_count"),
                        ("prefix_zipf", "pd_prefix_zipf"), ("prefix_cache_GB", "pd_prefix_cache_GB"),
                        ("kv_policy", "pd_kv_policy"), ("kv_capacity_GB", "pd_kv_capacity_GB"),
                        ("kv_admit", "pd_kv_admit"), ("swap_GBps", "pd_swap_GBps")):
            if getattr(a, dest, None) is not None:
                sc["pd"][k] = getattr(a, dest)
        if getattr(a, "pd_mix", None):
            sc["pd"]["length_mix"] = _parse_mix(a.pd_mix)
        if getattr(a, "pd_prefix_tree", None):
            sc["pd"]["prefix_tree"] = _parse_tree(a.pd_prefix_tree)
        if getattr(a, "pd_prefix_not_on_decode", False):
            sc["pd"]["prefix_on_decode"] = False
        if getattr(a, "pd_search_layouts", False):
            sc["pd"]["search_layouts"] = True
        if getattr(a, "pd_prefix_affinity", False):
            sc["pd"]["prefix_affinity"] = True
        if getattr(a, "pd_search_decode_batch", False):
            sc["pd"]["search_decode_batch"] = True
        if getattr(a, "pd_sim", False):
            sc["pd"]["simulate"] = True
    body = {"chip_preset": a.chip, "scenario": sc}
    en = {k: getattr(a, k) for k in ("pJ_mac", "pJ_vec", "pJ_bit_sram", "pJ_bit_dram", "pJ_bit_link", "idle_W",
                                     "pJ_bit_slc", "pJ_bit_d2d", "pJ_bit_net", "idle_W_prefill")
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
    if fb := out.get("fabric"):
        off = fb["step_ms_off"]
        print(f"fabric 「假设」: step {fb['step_ms']:.3f} ms" + (f" (model off {off:.3f} ms, {fb['step_ms'] / off - 1:+.1%})" if off else ""))
        print("  " + fb["basis"])
        tz = {"d2d": "D2D", "link": "scale-up", "net": "net"}
        for x in fb["rows"][:12]:
            lv = " · ".join(f"{tz[t]}×{n}" for t, n in x["levels"])
            alt = ", ".join(f"{k} {v['bw_us'] + v['alpha_us']:.2f}" for k, v in x["cands"].items() if k != x["algo"])
            print(f"  {x['kind']:<9} g={x['group']:<3} [{lv}] {x['bytes'] / 1024:9.1f} KiB ×{x['count']:<4} "
                  f"{x['algo'] + ('/' + x['top'] if x['top'] else ''):<10} bw {x['bw_us']:9.2f} µs  α {x['alpha_us']:7.2f} µs"
                  + (f"   (others: {alt})" if alt else ""))
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
        if p_.get("hetero"):
            print(f"    heterogeneous pools: prefill {p_['chip']} / {p_['mem_id']}  decode {d_['chip']} / {d_['mem_id']}")
        print(f"    TTFT {pd['ttft_ms']:.0f} ms  TPOT {pd['tpot_ms']:.2f} ms  goodput {pd['goodput_per_card']:.1f} tok/s/card  "
              f"bottleneck {pd['bottleneck']}"
              + (f"  best split {pd['best_split']['prefill_cards']}P+{pd['best_split']['decode_cards']}D "
                 f"{pd['best_split']['goodput_per_card']:.1f}" if pd.get("best_split") else ""))
        print(f"    colocated {c_['cards']} cards: TTFT {c_['ttft_ms']:.0f} ms  TPOT {c_['tpot_ms']:.2f} ms "
              f"(effective {c_['tpot_eff_ms']:.2f})  goodput {c_['goodput_per_card']:.1f} tok/s/card")
        for w in pd["warnings"]:
            print("    ! " + w)
        if (L := pd.get("lengths")) and not L["plain"]:
            print(f"  lengths 「假设」 ({L['source']}): mean prompt {L['mean_prompt']:.0f} (cv {L['prompt_cv_eff']:.2f})  "
                  f"mean out {L['mean_out']:.0f} (cv {L['out_cv_eff']:.2f})  decode ctx {L['decode_ctx']}"
                  + (f"  prefix hit {L['prefix_hit']:.0%}" + ("" if L["prefix_on_decode"] else " (full KV hand-off)")
                     if L["prefix_hit"] else ""))
        if (pc := pd.get("prefix_cache")) and pc.get("tree"):          # 0.58 radix tree
            print(f"  prefix cache 「假设」 radix tree LRU (Che per node): levels "
                  + " → ".join(f"{L_} tok × {n_} (Zipf {a_:g})" for L_, n_, a_ in pc["tree"])
                  + f", {'affinity routing' if pc['affinity'] else 'random routing'}")
            for k, nm in (("prefill", "PD prefill"), ("decode", "PD decode"), ("coloc", "colocated")):
                x = pc[k]
                print(f"    {nm:10s} {x['replicas']} replicas × {x['capacity_tokens']:.0f} tok ({x['source']})  token hit "
                      f"{x['token_hit']:.1%}  P(depth ≥ k) " + " / ".join(f"{v:.0%}" for v in x["level_hit"])
                      + (f"  decode holds {x['holds_given_prefill_hit']:.1%} of the matched tokens" if k == "decode" else ""))
        elif pc := pd.get("prefix_cache"):
            print(f"  prefix cache 「假设」 {pc['policy']}: {pc['prefix_count']} prefixes × {pc['prefix_len']} tok, "
                  f"Zipf {pc['zipf']:g}, {'affinity routing' if pc['affinity'] else 'random routing'}")
            for k, nm in (("prefill", "PD prefill"), ("decode", "PD decode"), ("coloc", "colocated")):
                x = pc[k]
                print(f"    {nm:10s} {x['replicas']} replicas × {x['K']} prefixes ({x['capacity_GB']:.1f} GB / "
                      f"{x['footprint_MB']:.1f} MB each, {x['source']})  hit {x['hit']:.1%}"
                      + (f"  holds | prefill hit {x['holds_given_prefill_hit']:.1%}" if k == "decode" else ""))
        if (q := pd.get("queue")) and "modes" in q:
            print(f"  queueing at {q['lambda_rps']:.3g} req/s"
                  + (f" ({q['load']:.0%} of PD capacity)" if q.get("load") else "") + "  「假设」 Poisson, " + ("M/D/1" if (pd.get("lengths") or {}).get("source", "fixed") == "fixed" and not pd.get("prefix_cache") else "M/G/1")
                  + ", Erlang C")
            names = {"pd": "PD", "coloc_prefill_first": "colocated prefill-first", "coloc_chunked":
                     f"colocated chunked ({q['chunk_tokens']} tok)"}
            for k, x in q["modes"].items():
                if not x["stable"]:
                    print(f"    {names[k]:<28} unstable: {x.get('why', '')}   SLO rate {x['slo_rate_rps']:.3g} req/s")
                    continue
                t = x["ttft_ms"]
                print(f"    {names[k]:<28} TTFT p50/p90/p99 {t['p50']:.0f}/{t['p90']:.0f}/{t['p99']:.0f} ms  "
                      f"TPOT mean/p90/p99 {x['tpot_mean_ms']:.1f}/{x['tpot_p90_ms']:.1f}/{x['tpot_p99_ms']:.1f} ms  "
                      f"max gap {x['itl_max_ms']:.0f} ms  SLO goodput {x['slo_goodput_per_card']:.1f} tok/s/card  "
                      f"stable ≤ {x['stable_rate_rps']:.3g} req/s")
            kvs = [(k, x["kv_cap"]) for k, x in q["modes"].items() if x.get("kv_cap")]
            if kvs:
                c0 = next((c for _, c in kvs if "admit" in c), kvs[0][1])   # an unstable mode carries no admit
                print(f"  decode KV capacity 「假设」 {c0['policy']}: {c0['capacity_tokens']} tok/replica "
                      f"({c0['source']}), mean running footprint {c0['footprint_tokens']:.0f} tok → "
                      f"{c0['slots_kv']} slots, admission {c0.get('admit', 'after_prefill')}"
                      + ((" (binds; admission wait included in TTFT and SLO goodput)" if c0.get("in_ttft") else
                          " (binds; admission wait reported here, not added to TTFT / TPOT / SLO goodput)")
                         if c0["binds"] else " (does not bind)"))
                for k, c in kvs:
                    if c["binds"] and q["modes"][k].get("stable"):
                        pol = c["policy"]
                        print(f"    {names[k]:<28} admission wait {c.get('slot_wait_mean_ms', 0):.0f} ms "
                              f"(P(wait) {c.get('p_wait', 0):.1%})"
                              + (f"  preemptions/req {c.get('preempt_per_req', 0):.3f}  re-prefill "
                                 f"{c.get('recompute_ms', 0):.0f} ms" if pol == "recompute" else "")
                              + (f"  preemptions/req {c.get('preempt_per_req', 0):.3f}  swap out+in "
                                 f"{2 * c.get('swap_ms', 0):.0f} ms @ {c.get('swap_GBps_card', 0):.0f} GB/s/card "
                                 f"({c.get('swap_source', '')})" if pol == "swap" else "")
                              + (f"  restore share {c['restore_share']:.1%} (slot hold x{c.get('slot_slowdown', 1):.3f})"
                                 if c.get("restore_share") else "")
                              + (f"  slot held through prefill {c['slot_hold_prefill_ms']:.0f} ms"
                                 if c.get("slot_hold_prefill_ms") else ""))
            for k, e in (q.get("energy") or {}).items():
                if e.get("J_per_token") is not None:
                    print(f"    {names[k]:<28} energy {e['J_per_token']:.4g} J/token  {e['tok_per_J'] or 0:.3g} tok/J"
                          + (f"  static {e['static_share']:.0%}" if e.get("static_share") else "")
                          + (f"  ({e['static_note']})" if e.get("static_note") else ""))
            if b := q.get("pd_slo_best_split"):
                print(f"    PD best split under SLO: {b['prefill_cards']}P+{b['decode_cards']}D "
                      f"{b['slo_goodput_per_card']:.1f} tok/s/card")
            if sim := q.get("sim"):
                print(f"  DES 「假设」 ({q.get('sim_basis', '')})")
                for k, sx in sim["modes"].items():
                    t, tp, it = sx["ttft_ms"], sx["tpot_ms"], sx["itl_max_ms"]
                    e = {kk: (v if v is not None else float("nan")) for kk, v in sx["err"].items()}
                    print(f"    DES {names[k]:<24} TTFT p50/p90/p99 {t['p50']:.0f}/{t['p90']:.0f}/{t['p99']:.0f} ms  "
                          f"TPOT mean/p90 {tp['mean']:.1f}/{tp['p90']:.1f} ms  max gap {it['p99']:.0f} ms"
                          f"  (err p90 TTFT {e.get('ttft_p90', float('nan')):+.0%}  TPOT {e.get('tpot_p90', float('nan')):+.0%}"
                          f"  gap {e.get('itl_max', float('nan')):+.0%})")
        elif q and q.get("error"):
            print("  queueing: " + q["error"])
        if ls := pd.get("layout_search"):
            print(f"  layout search 「假设」: {ls['candidates']} layouts{' (truncated)' if ls['truncated'] else ''}, "
                  f"{ls['pairs']} pairs on {ls['cards']} cards")
            for r in ls["rows"][:8]:
                slo = f"  SLO {r['slo_goodput_per_card']:.1f}" if "slo_goodput_per_card" in r else ""
                print(f"    P {r['prefill_layout']} ×{r['prefill_cards']}  D {r['decode_layout']} ×{r['decode_cards']}  "
                      f"{'batch '+str(r['decode_batch'])+'  ' if ls.get('decode_batch_searched') else ''}fluid {r['goodput_per_card']:.1f} tok/s/card ({r['bottleneck']}){slo}")
            if b := ls.get("best_slo"):
                print(f"    best under SLO: P {b['prefill_layout']} ×{b['prefill_cards']}  D {b['decode_layout']} "
                      f"×{b['decode_cards']}  {b['slo_goodput_per_card']:.1f} tok/s/card")
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
    from .core.validation import v2_domain, v2_trends, v3_genz, v4_serving
    v2, v3, vd = v2_trends(), v3_genz(), v2_domain()
    v4 = None if getattr(a, "no_v4", False) else v4_serving()
    if a.json:
        return {"v2": v2, "v3": v3, "v2_domain": vd, "v4": v4}
    print("V2 trend bands (H100-like 「假设」)")
    _table([[r["check"], r["value"], f"[{r['lo']}, {r['hi']}]", r["ok"]] for r in v2], ["check", "value", "band", "ok"])
    print("\nV2 video / protein (workload + FLOP sanity, no hardware calibration)")
    _table([[r["check"], r["value"], f"[{r['lo']}, {r['hi']}]", r["ok"], r["note"]] for r in vd],
           ["check", "value", "band", "ok", "note"])
    print("\nV3 GenZ reference (Llama-3.1-8B decode)")
    _table([[r["mem_GBps"], r["batch"], r["ctx"], r["mapping"], r["tpot_ms"], r["ours_ms"], r["ratio"],
             r["comparable"]] for r in v3], ["mem GB/s", "batch", "ctx", "mapping", "GenZ ms", "ours ms", "ratio",
                                              "comparable"])
    if v4 is not None:
        print("\nV4 serving: closed-form queueing vs request-level DES (Qwen3-8B 1P HBM3E TP2, PD 2+6; "
              "err = (closed − DES)/DES, > 0 = closed form pessimistic; full grid: scripts/v4_serving.py, "
              "accel_dse/data/v4_serving.json)")
        _table([[r["scenario"], r["mode"], r["metric"], r["sim"], r["ana"], f"{r['err']:+.0%}" if r["err"] == r["err"] else "n/a",
                 f"±{r['band']:.0%}", r["ok"]] for r in v4],
               ["scenario", "mode", "metric", "DES", "closed", "err", "band", "ok"])
    print("\nV5 silicon: n/a (no NPU exists — design guidance only)")
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
    v = sub.add_parser("validate", help="V2 trend bands + V3 GenZ comparison + V4 serving (closed form vs DES)")
    v.add_argument("--no-v4", dest="no_v4", action="store_true", help="skip the V4 serving DES subset (≈ 15 s)")
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
