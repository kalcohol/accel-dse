"""Energy / power + cost stubs for MetricsCard (v0.24) — ASSUMED, not silicon.

Honesty contract
----------------
* There is **no PDK / JEDEC / vendor power model** here. Power is a single
  user-supplied number per card (``tdp_w``) or ``watts_per_tops × peak_tops``,
  scaled by an assumed average-power factor ``power_util``.
* Code defaults are **OFF** (``None``): a MetricsCard only carries
  ``est_*`` energy / cost fields when the user sets knobs. No hidden wattage
  or price is baked into the engine.
* Example placeholder values (e.g. 400 W/card) live **only** in
  ``examples/energy_cost.example.json`` (mirrored at
  ``accel_dse/data/energy_cost.example.json`` for the web UI button) and are
  labeled EXAMPLE / placeholder.

Formulas (all ``est_*`` fields are assumed)::

    P_card_peak   = tdp_w                     (if set; wins)
                  = watts_per_tops × peak_tops (else, per-card peak TOPS)
    est_power_W   = chips × P_card_peak × power_util   (cards only; no host/cooling/PUE)

    LLM     est_energy_per_token_J = est_power_W × TPOT_s / batch     (decode step → B tokens)
    video   est_energy_per_frame_J = est_power_W × TTFC_s / (n_frames × batch)
    protein est_energy_per_seq_J   = est_power_W × time_per_seq_s / batch

    est_system_cost_usd = chips × (cost_per_card_usd + mem_addon_usd)

    LLM only (optional):
      est_tokens_per_s          = batch / TPOT_s                      (decode-only, derived)
      est_energy_usd_per_Mtok   = E_tok_J × 1e6 / 3.6e6 × usd_per_kwh
      est_capex_usd_per_Mtok    = system_cost / (tok/s × years × 365·86400 × duty) × 1e6
      est_usd_per_Mtok          = energy part + capex part (whichever knobs are set)

Defaults: ``power_util = 1.0`` (average = TDP; conservative upper bound —
lower it yourself), ``duty_cycle = 1.0``. Everything else ``None`` (off).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_POWER_UTIL = 1.0  # avg/peak power factor; 1.0 = TDP upper bound (assumed)
DEFAULT_DUTY_CYCLE = 1.0  # fraction of wall-clock the system serves (capex amortization)
SECONDS_PER_YEAR = 365.0 * 86400.0
J_PER_KWH = 3.6e6

ECON_FIELD_NAMES = (
    "tdp_w",
    "watts_per_tops",
    "power_util",
    "cost_per_card_usd",
    "mem_addon_usd",
    "usd_per_kwh",
    "amortize_years",
    "duty_cycle",
)

# Body / JSON aliases → canonical knob
_ALIASES = {
    "tdp": "tdp_w",
    "tdp_watts": "tdp_w",
    "card_tdp_w": "tdp_w",
    "w_per_tops": "watts_per_tops",
    "util_factor": "power_util",
    "power_util_factor": "power_util",
    "cost_per_card": "cost_per_card_usd",
    "card_cost_usd": "cost_per_card_usd",
    "mem_addon": "mem_addon_usd",
    "memory_addon_usd": "mem_addon_usd",
    "electricity_usd_per_kwh": "usd_per_kwh",
    "amortize_yrs": "amortize_years",
    "duty": "duty_cycle",
}

EXAMPLE_JSON_PATH = Path(__file__).resolve().parent / "data" / "energy_cost.example.json"


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    f = float(v)
    if f != f:  # NaN
        return None
    return f


@dataclass(frozen=True)
class EnergyCostKnobs:
    """Assumed energy / cost knobs. All ``None`` → stub off."""

    tdp_w: float | None = None
    watts_per_tops: float | None = None
    power_util: float | None = None
    cost_per_card_usd: float | None = None
    mem_addon_usd: float | None = None
    usd_per_kwh: float | None = None
    amortize_years: float | None = None
    duty_cycle: float | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "EnergyCostKnobs":
        if not d:
            return cls()
        if not isinstance(d, dict):
            raise TypeError("econ must be a JSON object")
        kw: dict[str, Any] = {}
        for k, v in d.items():
            if k.startswith("_"):
                continue  # comments / placeholders like "_note"
            key = _ALIASES.get(k, k)
            if key in ECON_FIELD_NAMES:
                kw[key] = _num(v)
            elif key in ("notes", "note"):
                kw["notes"] = str(v) if v is not None else None
        return cls(**kw)

    @classmethod
    def load_json(cls, path: str | Path) -> "EnergyCostKnobs":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"econ file must be a JSON object: {p}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = v
        return out

    def active_fields(self) -> list[str]:
        return [n for n in ECON_FIELD_NAMES if getattr(self, n) is not None]

    def apply_to_kwargs(self, kw: dict[str, Any]) -> dict[str, Any]:
        out = dict(kw)
        for n in ECON_FIELD_NAMES:
            v = getattr(self, n)
            if v is not None:
                out[n] = float(v)
        return out


def load_example_econ() -> dict[str, Any]:
    """Return EXAMPLE placeholder econ JSON (labeled fake) or {} if missing."""
    try:
        data = json.loads(EXAMPLE_JSON_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def econ_kwargs_from_obj(obj: Any) -> dict[str, Any]:
    """Pull econ knob attributes from a WorkbenchConfig-like object."""
    return {n: getattr(obj, n, None) for n in ECON_FIELD_NAMES}


def econ_configured(cfg: Any) -> bool:
    return any(
        getattr(cfg, n, None) is not None
        for n in ("tdp_w", "watts_per_tops", "cost_per_card_usd", "mem_addon_usd")
    )


def apply_energy_cost(card: Any, cfg: Any) -> Any:
    """Fill ``est_*`` fields on a MetricsCard in place (assumed stub). Returns card."""
    if not econ_configured(cfg):
        return card

    chips = max(int(getattr(card, "chips", 1) or 1), 1)
    batch = max(int(getattr(card, "batch", 1) or 1), 1)
    domain = str(getattr(card, "domain", "llm") or "llm")
    util = getattr(cfg, "power_util", None)
    util = DEFAULT_POWER_UTIL if util is None else float(util)
    if not (0.0 < util <= 1.5):
        raise ValueError(f"power_util must be in (0, 1.5], got {util}")
    duty = getattr(cfg, "duty_cycle", None)
    duty = DEFAULT_DUTY_CYCLE if duty is None else float(duty)
    if not (0.0 < duty <= 1.0):
        raise ValueError(f"duty_cycle must be in (0, 1], got {duty}")

    basis: list[str] = []
    notes: list[str] = []
    tdp = getattr(cfg, "tdp_w", None)
    wpt = getattr(cfg, "watts_per_tops", None)
    p_card_peak = 0.0
    if tdp is not None:
        if tdp < 0:
            raise ValueError("tdp_w must be >= 0")
        p_card_peak = float(tdp)
        basis.append(f"tdp_w={tdp:g} W/card")
        if wpt is not None:
            notes.append("tdp_w set → watts_per_tops ignored")
    elif wpt is not None:
        if wpt < 0:
            raise ValueError("watts_per_tops must be >= 0")
        p_card_peak = float(wpt) * float(card.peak_tops)
        basis.append(f"watts_per_tops={wpt:g} × peak {card.peak_tops:.4g} T = {p_card_peak:.4g} W/card")

    power_on = p_card_peak > 0
    if power_on:
        card.est_power_per_card_W = p_card_peak * util
        card.est_power_W = chips * p_card_peak * util
        basis.append(f"× power_util={util:g} × chips={chips} → {card.est_power_W:.4g} W")
        if domain == "video":
            ttfc_s = float(card.TTFC_ms or card.TTFT_ms) / 1e3
            n_fr = max(int(card.n_frames or 1), 1)
            card.est_energy_per_frame_J = card.est_power_W * ttfc_s / (n_fr * batch)
        elif domain == "protein":
            t_s = float(card.time_per_seq_ms or card.TTFT_ms) / 1e3
            card.est_energy_per_seq_J = card.est_power_W * t_s / batch
        else:
            tpot_s = float(card.TPOT_ms) / 1e3
            card.est_energy_per_token_J = card.est_power_W * tpot_s / batch
            card.est_energy_prefill_J = card.est_power_W * float(card.TTFT_ms) / 1e3

    cpc = getattr(cfg, "cost_per_card_usd", None)
    addon = getattr(cfg, "mem_addon_usd", None)
    if cpc is not None or addon is not None:
        per_card = float(cpc or 0.0) + float(addon or 0.0)
        card.est_system_cost_usd = chips * per_card
        basis.append(
            f"cost: chips={chips} × (card {float(cpc or 0):g} + mem add-on {float(addon or 0):g}) USD"
        )

    if domain == "llm" and card.TPOT_ms > 0:
        card.est_tokens_per_s = batch / (card.TPOT_ms / 1e3)
        kwh = getattr(cfg, "usd_per_kwh", None)
        years = getattr(cfg, "amortize_years", None)
        e_part = 0.0
        c_part = 0.0
        if kwh is not None and power_on:
            e_part = card.est_energy_per_token_J * 1e6 / J_PER_KWH * float(kwh)
            card.est_energy_usd_per_Mtok = e_part
        if years is not None and card.est_system_cost_usd > 0 and float(years) > 0:
            tok_life = card.est_tokens_per_s * float(years) * SECONDS_PER_YEAR * duty
            c_part = card.est_system_cost_usd / tok_life * 1e6
            card.est_capex_usd_per_Mtok = c_part
        if e_part or c_part:
            card.est_usd_per_Mtok = e_part + c_part
            notes.append(
                "$/MTok = decode-only tokens (no prefill, host, network, cooling, margin)"
            )
    elif domain != "llm" and (
        getattr(cfg, "usd_per_kwh", None) is not None
        or getattr(cfg, "amortize_years", None) is not None
    ):
        notes.append("$/MTok is LLM-only; usd_per_kwh / amortize_years ignored for " + domain)

    if card.oom:
        notes.append("OOM config — energy/cost numbers are not meaningful")

    card.econ_configured = True
    card.econ_basis = "; ".join(basis)
    card.assumptions = list(card.assumptions) + [
        "[assumed econ stub] " + card.econ_basis
        + " — user knobs, NOT silicon / PDK / JEDEC power; cards only (no host/cooling/PUE)"
    ] + [f"[assumed econ stub] {n}" for n in notes]
    return card
