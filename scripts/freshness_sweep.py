"""Catalog freshness sweep (0.257): list Hugging Face releases created on or after a date for every tracked org.

Usage:  python3 scripts/freshness_sweep.py 2026-10-17 [out.json]

The org list = every HF org in ``catalog.PROVIDERS`` + ``EXTRA_ORGS`` (orgs watched without a listed family yet, or
official mirrors used as byte sources).  Stdlib only; network access (huggingface.co public API) at run time only —
nothing in the test suite calls it.  Candidates still need the manual checks in AGENTS (official org, public config,
not a merge / alias of a listed entry, param_check exact).
"""
import json
import sys
import urllib.request

EXTRA_ORGS = (
    "OpenMOSS-Team",        # 0.257: MOSS-Image (diffusers DiT, 2026-10-11) — missing from the 0.251–0.255 sweeps
    "BlinkDL", "Inferact", "EvolutionaryScale", "typesafe-ai", "TypeSafeAI", "TypeSafe-AI", "google-deepmind",
    "NousResearch", "google", "ibm-granite", "allenai", "baidu", "PaddlePaddle", "rednote-hilab", "meituan-longcat",
    "ai-sage", "swiss-ai", "CohereLabs", "Motif-Technologies", "LGAI-EXAONE", "naver-hyperclovax", "kakaocorp",
    "upstage", "sarvamai", "stabilityai", "HiDream-ai", "Alpha-VLLM", "lmms-lab", "OpenGVLab", "Skywork",
    "jdopensource", "kuaishou", "Kwai-Kolors", "KwaiVGI", "ant-research",
    # 0.270 speech round: speech-model orgs (NOTES_0269 candidate list) — ASR / TTS / speech dialogue
    "FunAudioLLM", "funasr", "kyutai", "bosonai", "SparkAudio", "IndexTeam", "fishaudio", "SWivid", "FireRedTeam",
    "hexgrad", "ResembleAI", "nari-labs", "canopylabs", "sesame",
)


def sweep_orgs() -> list:
    from accel_dse.core.catalog import PROVIDERS
    orgs = []
    for _, _, prefixes in PROVIDERS:
        for p in prefixes:
            if p.startswith("gh:"):
                continue
            o = p.split("/")[0]
            if o and o not in orgs:
                orgs.append(o)
    for o in EXTRA_ORGS:
        if o not in orgs:
            orgs.append(o)
    return orgs


def list_org(org: str, limit: int = 100) -> list:
    url = f"https://huggingface.co/api/models?author={org}&sort=createdAt&direction=-1&limit={limit}&full=true"
    req = urllib.request.Request(url, headers={"User-Agent": "accel-dse-sweep"})
    rows = json.load(urllib.request.urlopen(req, timeout=60))
    return [{"id": m["id"], "createdAt": m.get("createdAt"), "lastModified": m.get("lastModified"),
             "gated": m.get("gated"), "pipeline": m.get("pipeline_tag"), "library": m.get("library_name")} for m in rows]


def main(argv):
    since = argv[1] if len(argv) > 1 else "2026-10-17"
    out = {}
    for o in sweep_orgs():
        try:
            rows = list_org(o)
        except Exception as e:      # noqa: BLE001 — report and continue
            print(o, "ERR", e)
            continue
        new = [m for m in rows if (m["createdAt"] or "") >= since]
        out[o] = new
        for m in new:
            print(m["createdAt"][:10], m["id"], "gated" if m["gated"] else "", m["pipeline"], m["library"])
    print("orgs", len(sweep_orgs()))
    if len(argv) > 2:
        json.dump(out, open(argv[2], "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv)
