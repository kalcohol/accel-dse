"""Self-contained HTML report from key workbench / compare scans.

No external CSS/JS deps — offline-readable. Absolute numbers are
assumed/uncalibrated unless derived from shape + tiling + capacities.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .scan import run_compare_domains
from .series import list_series
from .sku import SKU_100T
from .workbench import (
    DEFAULT_CHIP_SWEEP,
    MetricsCard,
    format_compute_sweep_table,
    format_mem_sweep_table,
    format_package_sweep_table,
    format_parallel_table,
    format_scan_table,
    workbench_compute_sweep,
    workbench_mem_sweep,
    workbench_package_sweep,
    workbench_parallel_matrix,
    workbench_scan,
)
from .package_ranges import (
    format_compute_table,
    format_package_table,
    list_compute_primaries,
    list_packages,
    package_counts,
)

HOW_TO_READ_WALLS = """
<p><strong>Wall</strong> = which time component dominates the roofline
<code>max(compute, dram, c2c, fabric) + sync</code> (then ÷(1−bubble)) for that
phase. v0.29: per-collective latency α (default 3 µs, assumed) is added
<em>outside</em> the max as exposed sync.</p>
<ul>
  <li><span class="tag compute">compute</span> — GEMM / attention cycles win
      (often HBM + decode M=1 util≈1/R, or DiT T² attn).</li>
  <li><span class="tag memory">memory</span> — external DRAM bytes / eff BW win
      (classic 27B decode weight-stream on LPDDR).</li>
  <li><span class="tag balanced">balanced</span> — compute ≈ memory (within ~5%).</li>
  <li><span class="tag c2c">c2c</span> — chip-to-chip collectives (TP all-reduce / EP A2A).</li>
  <li><span class="tag fabric">fabric</span> — remote KV IB/RoCE path.</li>
  <li><span class="tag sync">sync</span> — exposed collective latency
      (n_collectives × α × (1−overlap); TP 2 all-reduce/layer, EP 2 A2A/layer,
      PP 1 send/stage) — v0.29.</li>
  <li><span class="tag bubble">bubble</span> — pipeline-parallel idle
      (decode mb=1 → bubble=(pp−1)/pp).</li>
</ul>
<p>Read <em>breakdown</em> columns (t_compute / t_dram / t_c2c / …) alongside the
wall label. Staging SRAM (R=0) does <strong>not</strong> cut weight DRAM bytes;
only resident layers (R≥1) do. All absolute ms/GB/s are
<strong>assumed/uncalibrated</strong> unless noted derived.</p>
"""


@dataclass
class ReportBundle:
    """Collected scan outputs for HTML rendering."""

    series_rows: list[tuple[str, str, str, str, str]]
    chips_dense: list[MetricsCard]
    chips_moe: list[MetricsCard]
    parallel_dense: list[MetricsCard]
    parallel_moe: list[MetricsCard]
    mem_sweep: list[MetricsCard]
    mem_flip_note: str
    compare_rows: list[dict[str, Any]]
    generated_at: str
    version: str
    # v0.16 package / compute range DSE (assumed)
    package_table: str = ""
    package_sweep_lpddr: list = field(default_factory=list)
    package_sweep_hbm: list = field(default_factory=list)
    compute_table: str = ""
    compute_sweep_core: list = field(default_factory=list)
    compute_sweep_cluster: list = field(default_factory=list)
    package_flip_note: str = ""
    compute_flip_note: str = ""
    package_counts: dict = field(default_factory=dict)
    # v0.24 scenario presets × assumed energy / cost stub
    preset_rows: list = field(default_factory=list)  # [(domain_label, preset, card)]
    econ_example: dict = field(default_factory=dict)


def collect_report_data(
    *,
    chips_list: list[int] | tuple[int, ...] = DEFAULT_CHIP_SWEEP,
    parallel_chips: int = 8,
    n_cores: int = 16,
    tops_per_core: float = 6.25,
    prompt_len: int = 512,
    decode_seq_len: int = 512,
) -> ReportBundle:
    """Run (or recompute) key workbench + compare scans."""
    common = dict(
        mem_kind="HBM",
        n_cores=n_cores,
        tops_per_core=tops_per_core,
        prompt_len=prompt_len,
        decode_seq_len=decode_seq_len,
        batch=1,
        sram_mib=64.0,
    )
    chips_dense = workbench_scan(
        "series/dense-27b", chips_list=list(chips_list), **common
    )
    chips_moe = workbench_scan(
        "series/moe-active13b", chips_list=list(chips_list), **common
    )
    parallel_dense = workbench_parallel_matrix(
        "series/dense-27b", chips=parallel_chips, sort_by="TPOT", **common
    )
    parallel_moe = workbench_parallel_matrix(
        "series/moe-active13b", chips=parallel_chips, sort_by="TPOT", **common
    )
    mem_sweep = workbench_mem_sweep(
        "illustrative_27B",
        chips=1,
        mem_kind="HBM",
        vary="channels",
        channel_list=[1, 2, 4, 8],
        n_cores=n_cores,
        tops_per_core=tops_per_core,
        prompt_len=prompt_len,
        decode_seq_len=decode_seq_len,
    )
    flip_note = _mem_flip_note(mem_sweep)
    series_rows = []
    for e in list_series():
        shape_name = getattr(e.shape, "name", "?")
        series_rows.append((e.id, e.family, e.domain, shape_name, e.metadata))
    compare_rows = run_compare_domains(verbose=False, sku=SKU_100T)

    # v0.16: package + compute range primer (one real series + illustrative)
    # Prefer glm-5.3-flash if registered; else qwen3.8-27b; else illustrative_27B.
    from .series import SERIES_REGISTRY

    pkg_model = "illustrative_27B"
    for cand in ("glm-5.3-flash", "qwen3.8-27b", "series/dense-27b"):
        if cand in SERIES_REGISTRY:
            pkg_model = cand
            break
    pkg_lpddr = workbench_package_sweep(
        pkg_model, kind="LPDDR", chips=1, **common
    )
    pkg_hbm = workbench_package_sweep(
        pkg_model, kind="HBM", chips=1, **common
    )
    # Slim HBM table for report: stacks 2/4/8 × HBM3 / HBM3E / HBM4
    pkg_hbm_slim = [
        (p, c)
        for p, c in pkg_hbm
        if p.n_stacks in (2, 4, 8) and p.generation in ("HBM3", "HBM3E", "HBM4")
    ]
    # LPDDR slim (v0.29): LPDDR5X 2×x64 / 8×x64 across the rate band,
    # LPDDR6 2/4/6 × x96 @10667, SOCAMM2 8 modules
    pkg_lpddr_slim = [
        (p, c)
        for p, c in pkg_lpddr
        if (
            (p.generation == "LPDDR5X" and p.n_packages in (2, 8))
            or p.generation == "LPDDR6"
        )
    ]
    comp_core = workbench_compute_sweep(
        pkg_model, level="core", chips=1, mem_kind="LPDDR", **{
            k: v for k, v in common.items() if k not in ("n_cores", "tops_per_core", "mem_kind")
        }
    )
    comp_cluster = workbench_compute_sweep(
        pkg_model, level="cluster", chips=1, mem_kind="HBM", **{
            k: v for k, v in common.items() if k not in ("n_cores", "tops_per_core", "mem_kind")
        }
    )
    pkg_flip = _pair_flip_note(
        [(p.id, c) for p, c in pkg_lpddr_slim + pkg_hbm_slim],
        label="package",
    )
    comp_flip = _pair_flip_note(
        [(o.id, c) for o, c in comp_core + comp_cluster],
        label="compute",
    )
    counts = package_counts()

    # v0.24: scenario presets × EXAMPLE placeholder econ knobs (assumed stub)
    from .econ import EnergyCostKnobs, load_example_econ
    from .scenarios import evaluate_presets

    econ_example = load_example_econ()
    econ_kw = EnergyCostKnobs.from_dict(econ_example).apply_to_kwargs({})
    preset_rows: list = []
    for label, mid, extra in (
        ("llm illustrative_27B int8", "illustrative_27B", {"dtype": "int8",
         "prompt_len": prompt_len, "decode_seq_len": decode_seq_len}),
        ("video illustrative_dit_video", "illustrative_dit_video", {}),
        ("protein illustrative_protein_pair", "illustrative_protein_pair", {}),
    ):
        for pre, card in evaluate_presets(mid, **extra, **econ_kw):
            preset_rows.append((label, pre, card))

    return ReportBundle(
        series_rows=series_rows,
        chips_dense=chips_dense,
        chips_moe=chips_moe,
        parallel_dense=parallel_dense,
        parallel_moe=parallel_moe,
        mem_sweep=mem_sweep,
        mem_flip_note=flip_note,
        compare_rows=compare_rows,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
        or datetime.now().isoformat(timespec="seconds"),
        version=__version__,
        package_table=format_package_table(list_packages()),
        package_sweep_lpddr=pkg_lpddr_slim,
        package_sweep_hbm=pkg_hbm_slim,
        compute_table=format_compute_table(list_compute_primaries()),
        compute_sweep_core=comp_core,
        compute_sweep_cluster=comp_cluster,
        package_flip_note=pkg_flip,
        compute_flip_note=comp_flip,
        package_counts=counts,
        preset_rows=preset_rows,
        econ_example=econ_example,
    )


def _mem_flip_note(cards: list[MetricsCard]) -> str:
    if not cards:
        return "No mem-sweep rows."
    flips: list[str] = []
    prev = cards[0].wall
    for c in cards[1:]:
        if c.wall != prev:
            flips.append(
                f"wall {prev} → {c.wall} near eff≈{c.mem_eff_GBps:.0f} GB/s "
                f"(TPOT {c.TPOT_ms:.3f} ms)"
            )
            prev = c.wall
        else:
            prev = c.wall
    if not flips:
        walls = sorted({c.wall for c in cards})
        bw0, bw1 = cards[0].mem_eff_GBps, cards[-1].mem_eff_GBps
        return (
            f"No wall FLIP across channels sweep "
            f"({bw0:.0f}→{bw1:.0f} GB/s); walls seen: {', '.join(walls)}."
        )
    return "FLIP: " + "; ".join(flips) + "."



def _pair_flip_note(rows: list[tuple[str, MetricsCard]], *, label: str) -> str:
    if not rows:
        return f"No {label}-sweep rows."
    flips: list[str] = []
    prev = rows[0][1].wall
    for name, c in rows[1:]:
        if c.wall != prev:
            flips.append(
                f"wall {prev} → {c.wall} at {name} "
                f"(TPOT {c.TPOT_ms:.3f} ms, eff≈{c.mem_eff_GBps:.0f} GB/s)"
            )
            prev = c.wall
        else:
            prev = c.wall
    if not flips:
        walls = sorted({c.wall for _, c in rows})
        return (
            f"No wall FLIP across {label} sweep; walls seen: {', '.join(walls)}."
        )
    return "FLIP: " + "; ".join(flips)



def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _wall_span(wall: str) -> str:
    w = _esc(wall)
    cls = wall if wall in (
        "compute", "memory", "balanced", "c2c", "fabric", "bubble", "sync"
    ) else "other"
    return f'<span class="tag {cls}">{w}</span>'


def _scale_eff_table_html(cards: list[MetricsCard]) -> str:
    """Scale-efficiency vs chips (primary latency vs chips=1 baseline)."""
    headers = [
        "chips", "tp", "pp", "ep", "metric", "primary_ms",
        "t_single_ms", "speedup", "scale_eff", "wall", "oom",
    ]
    rows = []
    for c in cards:
        if c.domain == "video":
            primary = c.TTFC_ms
        elif c.domain == "protein":
            primary = c.time_per_seq_ms
        else:
            primary = c.TPOT_ms
        rows.append([
            c.chips, c.tp, c.pp, c.ep,
            c.scale_metric or "TPOT",
            f"{primary:.4f}",
            f"{c.t_single_primary_ms:.4f}",
            f"{c.speedup:.4f}",
            f"{c.scale_efficiency:.4f}",
            _wall_span(c.wall),
            c.oom,
        ])
    return _html_table(headers, rows)


def _cards_table_html(cards: list[MetricsCard], *, mode: str = "chips") -> str:
    if mode == "parallel":
        headers = [
            "rank", "chips", "tp", "pp", "ep", "TTFT_ms", "TPOT_ms",
            "wall", "t_c2c_ms", "t_sync_ms", "t_bubble_ms", "util", "oom",
        ]
        rows = []
        for i, c in enumerate(cards, 1):
            rows.append([
                i, c.chips, c.tp, c.pp, c.ep,
                f"{c.TTFT_ms:.4f}", f"{c.TPOT_ms:.4f}",
                _wall_span(c.wall),
                f"{c.t_c2c_ms:.4f}", f"{c.t_sync_ms:.4f}", f"{c.t_bubble_ms:.4f}",
                f"{c.util:.4f}", c.oom,
            ])
    else:
        headers = [
            "chips", "tp", "TTFT_ms", "TPOT_ms", "wall",
            "util", "peak_T", "mem_GBps", "scale_eff", "speedup", "oom",
        ]
        rows = []
        for c in cards:
            rows.append([
                c.chips, c.tp,
                f"{c.TTFT_ms:.4f}", f"{c.TPOT_ms:.4f}",
                _wall_span(c.wall),
                f"{c.util:.4f}", f"{c.peak_tops:.2f}",
                f"{c.mem_eff_GBps:.1f}",
                f"{c.scale_efficiency:.4f}", f"{c.speedup:.4f}",
                c.oom,
            ])
    return _html_table(headers, rows)


def _mem_table_html(cards: list[MetricsCard]) -> str:
    headers = [
        "mem", "eff_GBps", "TTFT_ms", "TPOT_ms", "wall",
        "t_comp_ms", "t_dram_ms", "flip", "chips",
    ]
    rows = []
    prev: str | None = None
    for c in cards:
        flip = ""
        if prev is not None and c.wall != prev:
            flip = "FLIP"
        prev = c.wall
        rows.append([
            c.mem_kind, f"{c.mem_eff_GBps:.1f}",
            f"{c.TTFT_ms:.4f}", f"{c.TPOT_ms:.4f}",
            _wall_span(c.wall),
            f"{c.t_compute_ms:.4f}", f"{c.t_dram_ms:.4f}",
            flip, c.chips,
        ])
    return _html_table(headers, rows)


def _compare_table_html(rows: list[dict[str, Any]]) -> str:
    headers = [
        "domain", "mem", "shape", "metric", "metric_ms",
        "wall", "dominant_cost", "notes",
    ]
    body = []
    for r in rows:
        body.append([
            r.get("domain", ""),
            r.get("mem", ""),
            r.get("shape", ""),
            r.get("metric_name", ""),
            f"{float(r.get('metric_ms', 0)):.4f}",
            _wall_span(str(r.get("wall", ""))),
            r.get("dominant_cost", ""),
            r.get("notes", ""),
        ])
    return _html_table(headers, body)


def _preset_econ_table_html(rows: list) -> str:
    headers = [
        "workload", "preset", "package", "compute", "chips", "primary", "ms",
        "wall", "oom", "est_power_W", "est_energy / unit (J)", "est_system_cost_usd",
        "est_usd_per_Mtok",
    ]
    body = []
    for label, pre, c in rows:
        if c.domain == "video":
            key, val, e, unit = "TTFC", c.TTFC_ms, c.est_energy_per_frame_J, "/frame"
        elif c.domain == "protein":
            key, val, e, unit = "t/seq", c.time_per_seq_ms, c.est_energy_per_seq_J, "/seq"
        else:
            key, val, e, unit = "TPOT", c.TPOT_ms, c.est_energy_per_token_J, "/token"
        body.append([
            label, pre.id, pre.package_id, pre.compute_id, c.chips, key,
            f"{val:.4f}", _wall_span(c.wall), "YES" if c.oom else "no",
            f"{c.est_power_W:.1f}", f"{e:.4g} {unit}",
            f"{c.est_system_cost_usd:,.0f}",
            f"{c.est_usd_per_Mtok:.4g}" if c.domain == "llm" else "n/a (LLM only)",
        ])
    return _html_table(headers, body)


def _series_table_html(rows: list[tuple[str, str, str, str, str]]) -> str:
    headers = ["id", "family", "domain", "shape", "metadata"]
    body = [[a, b, c, d, e] for a, b, c, d, e in rows]
    return _html_table(headers, body)


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_parts = []
    for row in rows:
        tds = []
        for cell in row:
            if isinstance(cell, str) and cell.startswith("<span"):
                tds.append(f"<td>{cell}</td>")
            else:
                tds.append(f"<td>{_esc(cell)}</td>")
        body_parts.append("<tr>" + "".join(tds) + "</tr>")
    return (
        f'<table>\n<thead><tr>{th}</tr></thead>\n'
        f'<tbody>\n' + "\n".join(body_parts) + "\n</tbody>\n</table>"
    )


_CSS = """
:root {
  --bg: #0f1419; --panel: #1a2332; --text: #e7ecf3; --muted: #9aa7b8;
  --accent: #5b9fd4; --border: #2a3a4e; --warn: #e6a23c;
  --compute: #3d9a6a; --memory: #c45c26; --balanced: #8a7a3a;
  --c2c: #6b7fd7; --fabric: #a855c8; --bubble: #b45309;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI,
    Roboto, "Noto Sans", "Helvetica Neue", Arial, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.45;
  font-size: 14px;
}
header {
  padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #1c2838, var(--bg));
}
header h1 { margin: 0 0 0.35rem; font-size: 1.45rem; }
header .meta { color: var(--muted); font-size: 0.9rem; }
.banner {
  margin: 1rem 1.5rem 0; padding: 0.85rem 1rem;
  background: #3a2a12; border: 1px solid var(--warn); border-radius: 8px;
  color: #f5d9a8;
}
.banner strong { color: #ffd27a; }
nav {
  display: flex; flex-wrap: wrap; gap: 0.5rem; padding: 0.75rem 1.5rem;
  border-bottom: 1px solid var(--border);
}
nav a {
  color: var(--accent); text-decoration: none; font-size: 0.85rem;
  padding: 0.2rem 0.5rem; border: 1px solid var(--border); border-radius: 4px;
}
nav a:hover { background: var(--panel); }
main { padding: 0.5rem 1.5rem 2.5rem; max-width: 1200px; }
section {
  margin-top: 1.5rem; padding: 1rem 1.1rem;
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
}
section h2 {
  margin: 0 0 0.75rem; font-size: 1.15rem;
  border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;
}
section h3 { margin: 1rem 0 0.5rem; font-size: 1rem; color: var(--muted); }
p, li { color: var(--text); }
code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.85em;
}
code { background: #0c1118; padding: 0.1em 0.35em; border-radius: 3px; }
pre.plain {
  background: #0c1118; padding: 0.75rem; border-radius: 6px;
  overflow-x: auto; white-space: pre; color: #c8d4e0;
  border: 1px solid var(--border);
}
.table-wrap { overflow-x: auto; }
table {
  border-collapse: collapse; width: 100%; font-size: 0.82rem;
  margin: 0.4rem 0 0.8rem;
}
th, td {
  border: 1px solid var(--border); padding: 0.35rem 0.5rem;
  text-align: left; vertical-align: top;
}
th { background: #121a26; color: var(--muted); font-weight: 600; }
tr:nth-child(even) td { background: rgba(0,0,0,0.15); }
.tag {
  display: inline-block; padding: 0.1rem 0.45rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 600; color: #fff;
}
.tag.compute { background: var(--compute); }
.tag.memory { background: var(--memory); }
.tag.balanced { background: var(--balanced); }
.tag.c2c { background: var(--c2c); }
.tag.fabric { background: var(--fabric); }
.tag.bubble { background: var(--bubble); }
.tag.sync { background: #0e7490; }
.tag.other { background: #555; }
.note { color: var(--muted); font-size: 0.9rem; }
footer {
  padding: 1rem 1.5rem 2rem; color: var(--muted); font-size: 0.8rem;
  border-top: 1px solid var(--border);
}
"""


def render_html(bundle: ReportBundle) -> str:
    """Build a self-contained HTML document string."""
    chips_dense_txt = format_scan_table(bundle.chips_dense)
    chips_moe_txt = format_scan_table(bundle.chips_moe)
    par_dense_txt = format_parallel_table(bundle.parallel_dense)
    par_moe_txt = format_parallel_table(bundle.parallel_moe)
    mem_txt = format_mem_sweep_table(bundle.mem_sweep)

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>',
        f"<title>accel_dse report v{_esc(bundle.version)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>accel_dse Metrics / wall report "
        f"<small>v{_esc(bundle.version)}</small></h1>",
        f'<div class="meta">Generated {_esc(bundle.generated_at)} · '
        f"offline self-contained HTML (inline CSS, no external deps)</div>",
        "</header>",
        '<div class="banner" id="assumptions">',
        "<strong>ASSUMPTIONS / UNCALIBRATED:</strong> Absolute TOPS, GB/s, "
        "frequency, C2C/fabric BW are assumed/uncalibrated; LLM/video/protein dims "
        "come from public HF configs where registered, else illustrative. Do not treat ms/GB as "
        "silicon-backed. See docs/MODEL.md.",
        "</div>",
        "<nav>",
        '<a href="#walls">How to read walls</a>',
        '<a href="#series">Model series</a>',
        '<a href="#packages">Package ranges</a>',
        '<a href="#compute">Compute ranges</a>',
        '<a href="#chips">Chips sweep</a>',
        '<a href="#parallel">Parallel matrix</a>',
        '<a href="#mem">Mem geometry</a>',
        '<a href="#compare">Cross-domain</a>',
        '<a href="#scale">Scale efficiency</a>',
        '<a href="#assumed-compute">Assumed compute extras</a>',
        '<a href="#econ">Presets × energy/cost (assumed)</a>',
        '<a href="#assumptions">Assumptions</a>',
        "</nav>",
        "<main>",
        '<section id="walls">',
        "<h2>How to read walls</h2>",
        HOW_TO_READ_WALLS,
        "</section>",
        '<section id="series">',
        "<h2>Model series list</h2>",
        '<p class="note">Real packs cite <code>hf:&lt;id&gt;</code> from public '
        "HF config.json / model cards; illustrative packs remain for handcheck. "
        "FLOPs/BW uncalibrated. CLI: "
        "<code>python3 -m accel_dse list-series</code> / "
        "<code>--product</code></p>",
        '<div class="table-wrap">',
        _series_table_html(bundle.series_rows),
        "</div>",
        "</section>",
        '<section id="packages">',
        "<h2>Memory catalog (LPDDR5/5X/6 · HBM3/3E/4/4E) — v0.29 structured</h2>",
        '<p class="note">Rebuilt from <code>docs/research/memory_specs_2026-10</code>; '
        "every rate / width / capacity carries a provenance tag and each config "
        "shows the <strong>weakest</strong> one (JEDEC &gt; 疑似 JEDEC &gt; 厂商量产 &gt; "
        "送样 &gt; 已发布 &gt; 推测). LPDDR5/5X: <strong>x64 package = 4×16-bit "
        "channels</strong> (x32 / vendor x96 also), 5500–10667 MT/s per generation. "
        "LPDDR6: <strong>x96 package = 4×24-bit channels (8×12-bit sub-ch)</strong>, "
        "10667/12800/14400 MT/s, usable BW = raw × <strong>8/9</strong> "
        "(BL24 metadata), kept separate from efficiency. SOCAMM2 / LPCAMM2: "
        "128-bit LPDDR5X modules. HBM: stacks × height × die density "
        "(HBM3/3E 1024-bit, HBM4/4E 2048-bit). Capacity GB = 2<sup>30</sup> B. "
        "BW = units × (width/8) × rate × payload × efficiency (efficiency "
        "<strong>assumed</strong> 0.70). CLI: "
        "<code>python3 -m accel_dse list-packages</code> · "
        "<code>workbench-scan --package-axis type|count|rate</code></p>",
        f"<p><strong>{_esc(bundle.package_flip_note or '')}</strong></p>",
        f"<p class=\"note\">Catalog counts: "
        f"{_esc(str(bundle.package_counts or {}))}</p>",
        "<h3>Catalog (peak GB/s derived)</h3>",
        '<pre class="plain">' + _esc(bundle.package_table or "") + "</pre>",
        "<h3>LPDDR sweep (LPDDR5X 2×x64 &amp; 8×x64 rate band + LPDDR6 2/4/6 × x96)</h3>",
        '<pre class="plain">'
        + _esc(format_package_sweep_table(bundle.package_sweep_lpddr or []))
        + "</pre>",
        "<h3>HBM sweep (stacks 2/4/8 × HBM3 / HBM3E / HBM4, 12-high)</h3>",
        '<pre class="plain">'
        + _esc(format_package_sweep_table(bundle.package_sweep_hbm or []))
        + "</pre>",
        "</section>",
        '<section id="compute">',
        "<h2>Compute hierarchy ranges — assumed DSE</h2>",
        '<p class="note">Single-core peak options 4 / 8 / 12 / 16 TOPS; '
        "cluster peaks 64 / 128 / 192 / 256 TOPS via n_cores × tops_per_core "
        "at assumed 1 GHz. Efficiency knobs (mac_efficiency / mem efficiency) "
        "remain adjustable placeholders. CLI: "
        "<code>python3 -m accel_dse list-compute</code> · "
        "<code>workbench-scan --sweep-compute</code></p>",
        f"<p><strong>{_esc(bundle.compute_flip_note or '')}</strong></p>",
        "<h3>Catalog primaries</h3>",
        '<pre class="plain">' + _esc(bundle.compute_table or "") + "</pre>",
        "<h3>Core tops sweep (4T–16T) @ LPDDR</h3>",
        '<pre class="plain">'
        + _esc(format_compute_sweep_table(bundle.compute_sweep_core or []))
        + "</pre>",
        "<h3>Cluster tops sweep (64T–256T) @ HBM</h3>",
        '<pre class="plain">'
        + _esc(format_compute_sweep_table(bundle.compute_sweep_cluster or []))
        + "</pre>",
        "</section>",
        '<section id="chips">',
        "<h2>Chips sweep table</h2>",
        "<p class=\"note\">Default mapping chip_count→tp (pp=ep=1), HBM, "
        "cores=16×6.25 T/core ≈ sku_100t. CLI: "
        "<code>python3 -m accel_dse workbench-scan --model series/dense-27b "
        "--chips-list 1 2 4 8</code></p>",
        "<h3>series/dense-27b</h3>",
        '<div class="table-wrap">',
        _cards_table_html(bundle.chips_dense),
        "</div>",
        "<pre class=\"plain\">" + _esc(chips_dense_txt) + "</pre>",
        "<h3>series/moe-active13b</h3>",
        '<div class="table-wrap">',
        _cards_table_html(bundle.chips_moe),
        "</div>",
        "<pre class=\"plain\">" + _esc(chips_moe_txt) + "</pre>",
        "</section>",
        '<section id="parallel">',
        "<h2>Parallel matrix — chips=8 dense + MoE</h2>",
        "<p class=\"note\">Enumerate tp×pp×ep=8 sorted by TPOT. Dense forces "
        "ep=1; MoE allows EP. CLI: "
        "<code>python3 -m accel_dse workbench-parallel --chips 8 "
        "--model series/dense-27b</code> (and series/moe-active13b).</p>",
        "<h3>Dense (series/dense-27b)</h3>",
        '<div class="table-wrap">',
        _cards_table_html(bundle.parallel_dense, mode="parallel"),
        "</div>",
        "<pre class=\"plain\">" + _esc(par_dense_txt) + "</pre>",
        "<h3>MoE (series/moe-active13b)</h3>",
        '<div class="table-wrap">',
        _cards_table_html(bundle.parallel_moe, mode="parallel"),
        "</div>",
        "<pre class=\"plain\">" + _esc(par_moe_txt) + "</pre>",
        "</section>",
        '<section id="mem">',
        "<h2>Mem geometry flip note</h2>",
        f"<p><strong>{_esc(bundle.mem_flip_note)}</strong></p>",
        "<p class=\"note\">HBM n_channels sweep {1,2,4,8} @ illustrative_27B "
        "chips=1. CLI: <code>python3 -m accel_dse workbench-scan --sweep-mem "
        "--mem hbm --mem-vary channels --chips 1</code></p>",
        '<div class="table-wrap">',
        _mem_table_html(bundle.mem_sweep),
        "</div>",
        "<pre class=\"plain\">" + _esc(mem_txt) + "</pre>",
        "</section>",
        '<section id="compare">',
        "<h2>Cross-domain compare snapshot</h2>",
        "<p class=\"note\">sku_100t × HBM/LPDDR: LLM TPOT / video TTFC / "
        "protein t/seq. CLI: "
        "<code>python3 -m accel_dse compare-domains</code></p>",
        '<div class="table-wrap">',
        _compare_table_html(bundle.compare_rows),
        "</div>",
        "</section>",
        '<section id="scale">',
        "<h2>Scale efficiency vs chips</h2>",
        '<p class="note"><strong>Definition:</strong> '
        '<code>speedup = t_single / t_multi</code>; '
        '<code>scale_efficiency = speedup / chip_count</code> '
        "(ideal <strong>1.0</strong>). Domain primary: LLM→TPOT, video→TTFC, "
        "protein→time_per_seq. Baseline = same WorkbenchConfig with "
        "<code>chips=tp=pp=ep=1</code>. "
        "<strong>Honesty:</strong> ignores host/NIC non-ideal; C2C / PP bubble / EP "
        "collectives are already inside <code>t_multi</code>. "
        "Default HBM + 400 GB/s C2C often stays near 1.0 (compute-bound); "
        "PP↑ or weak C2C drops efficiency (comm / bubble bound).</p>",
        "<h3>series/dense-27b — scale_efficiency vs chips</h3>",
        '<div class="table-wrap">',
        _scale_eff_table_html(bundle.chips_dense),
        "</div>",
        "<h3>series/moe-active13b — scale_efficiency vs chips</h3>",
        '<div class="table-wrap">',
        _scale_eff_table_html(bundle.chips_moe),
        "</div>",
        "</section>",
        '<section id="assumed-compute">',
        "<h2>Assumed compute extras (v0.26) — Softmax/RoPE/LN + dtype MAC</h2>",
        (
            '<p class="note"><strong>Opt-in; defaults preserve handcheck.</strong> '
            "Coarse non-GEMM overhead <code>non_gemm_overhead</code> (default <strong>0</strong>) "
            "adds a fraction of GEMM compute time for Softmax/RoPE/LN/misc: "
            "<code>t_compute' = t_compute × (1 + overhead)</code>. Optional finer "
            "<code>softmax_frac</code> / <code>rope_frac</code> / <code>norm_frac</code> "
            "sum into overhead when provided. "
            "<strong>Not cycle-accurate.</strong></p>"
        ),
        (
            '<p class="note">Optional <code>dtype_mac_factors</code> map '
            "(fp16/fp8/int8/int4; default all <strong>1.0</strong>) multiplies peak TOPS "
            "and divides GEMM compute time for the weight dtype. "
            "<strong>User/assumed — not silicon.</strong> Default bytes-only behavior "
            "unchanged. CLI: <code>--non-gemm-overhead</code> · "
            "<code>--dtype-mac-factor</code> JSON map. "
            "Web: Assumed compute extras. See docs/MODEL.md §4.</p>"
        ),
        "</section>",
        '<section id="econ">',
        "<h2>Scenario presets × energy / cost — ASSUMED stub (not silicon)</h2>",
        '<p class="note"><strong>Honesty:</strong> power is a single user knob '
        "(<code>tdp_w</code> per card, or <code>watts_per_tops × peak_tops</code>) × "
        "assumed <code>power_util</code> × chips — <strong>no PDK / JEDEC / vendor "
        "power model</strong>, cards only (no host / cooling / PUE). Cost = chips × "
        "(<code>cost_per_card_usd</code> + <code>mem_addon_usd</code>). LLM "
        "<code>$/MTok</code> = energy (<code>usd_per_kwh</code>) + capex amortized over "
        "<code>amortize_years × duty_cycle</code> of decode-only tokens. Engine defaults "
        "are <strong>OFF</strong>; the numbers below use the clearly-fake "
        "<strong>EXAMPLE placeholders</strong> from "
        "<code>examples/energy_cost.example.json</code>.</p>",
        "<p class=\"note\">EXAMPLE knobs: <code>"
        + _esc(", ".join(f"{k}={v}" for k, v in bundle.econ_example.items()
                         if not str(k).startswith("_") and k != "notes"))
        + "</code></p>",
        "<p class=\"note\">Presets (package + compute + chips; no power/price inside): "
        "<code>edge-lpddr-4x64</code>, <code>card-hbm-4stack</code>, "
        "<code>scaleup-8chip</code>. CLI: <code>python3 -m accel_dse list-presets</code> · "
        "<code>python3 -m accel_dse eval-presets --model illustrative_27B --econ "
        "examples/energy_cost.example.json</code> · "
        "<code>workbench --preset card-hbm-4stack --tdp-w … --cost-per-card …</code>. "
        "OOM rows: energy/cost not meaningful.</p>",
        "<p class=\"note\"><strong>Caveat:</strong> one flat EXAMPLE <code>tdp_w</code> is "
        "applied to every preset, so the edge row at 400 W/card is deliberately unrealistic — "
        "for cross-class comparison set <code>watts_per_tops</code> or per-preset "
        "<code>--tdp-w</code> yourself. Scale-up rows multiply power and cost by chips.</p>",
        '<div class="table-wrap">',
        _preset_econ_table_html(bundle.preset_rows or []),
        "</div>",
        "</section>",
        "</main>",
        "<footer>",
        f"accel_dse v{_esc(bundle.version)} · MetricsCard / wall report · "
        "Regenerate: <code>python3 -m accel_dse report --out out/report.html</code>",
        "</footer>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)


def write_report(
    out_path: str | Path,
    *,
    bundle: ReportBundle | None = None,
    verbose: bool = True,
) -> Path:
    """Collect scans (unless bundle given), write self-contained HTML."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bundle if bundle is not None else collect_report_data()
    html_text = render_html(data)
    path.write_text(html_text, encoding="utf-8")
    if verbose:
        print(f"wrote HTML report {path} ({path.stat().st_size} bytes)")
    return path
