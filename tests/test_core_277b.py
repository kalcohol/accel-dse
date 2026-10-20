"""0.277: Fish Audio Drama 3 DAC decoder (core/drama3.dac_ops) costed from codec.pth's shapes, colocated with the LM."""
import json
import math
from pathlib import Path

from accel_dse.core import drama3 as d3
from accel_dse.core.catalog import get_model
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario
from accel_dse.core.speech import speech_kpis
from accel_dse.core.speech_insight import _scn, _stages
from tests import webstruct as W  # noqa: E402  (0.286 structural UI checks)

ROOT = Path(__file__).resolve().parents[1]
REL = json.loads((ROOT / "accel_dse/data/releases/fishaudio__drama-3.json").read_text())
BASE = Scenario(model="qwen3-8b", chip=CHIPS["100T"], mem_id="lpddr5x_4x64_8533_16g")


def test_release_codec_shapes_and_rates():
    fs = get_model("drama-3").fish
    ct = REL["codec_tensors"]
    assert REL["codec_dtype"] == "F32" and REL["params_codec_decode"] == sum(math.prod(s) for s in ct.values())
    assert d3.dac_params(fs) == REL["params_codec_decode"] and REL["params_codec"] > REL["params_codec_decode"]
    assert not any(n.endswith(("causal_mask", "freqs_cis")) or n.startswith("encoder.") for n in ct)
    up, dec = d3.dac_rates(fs)
    assert up == (2, 2) and dec == (8, 8, 4, 2) and math.prod(up) * math.prod(dec) == d3.CODEC_HOP
    # every decode-path weight is read once per window: GEMM weights + side tensors + leftover vectors + codebooks
    ops = d3.dac_ops(fs, 64)
    cb = sum(math.prod(s) for n, s in ct.items() if n.endswith("codebook.weight") or ".in_proj." in n)
    assert sum(o.w_params for o in ops) + cb == d3.dac_params(fs)


def test_flops_per_frame_by_hand():
    """Independent count from the architecture (fish-speech rvq.decode + modded_dac.Decoder) 「推导」."""
    ct = REL["codec_tensors"]
    F = 64
    L = 8
    d, ff = 1024, 3072
    post = L * 2 * (3 * d * d + d * d + 3 * d * ff) * F + 2 * 2 * L * F * min(128, F) * d     # + QKᵀ, PV
    outp = 10 * 2 * 8 * d * F
    ups, T = 0, F
    for _ in range(2):                                   # ConvTranspose k = s = 2, then ConvNeXt at 2T
        ups += 2 * T * d * d * 2
        T *= 2
        ups += 2 * T * (7 * d + 2 * d * 4 * d)
    dec = 2 * T * 1024 * 7 * 1536
    c = 1536
    for s in (8, 8, 4, 2):
        dec += 2 * T * c * (c // 2) * 2 * s              # ConvTranspose k = 2s per input position
        T *= s
        c //= 2
        dec += 3 * 2 * T * c * c * (7 + 1)               # 3 residual units: conv7 + conv1
    dec += 2 * T * c * 7
    assert T == F * 2048 and c == 96 and len([n for n in ct if "post_module.layers." in n and n.endswith("wo.weight")]) == L
    got = d3.dac_gflop_per_frame(get_model("drama-3").fish, F)
    assert abs(got - (post + outp + ups + dec) / F / 1e9) < 1e-9, got
    assert 6.5 < got < 7.0


def test_windows_follow_the_ramp():
    w = d3.dac_windows(216)
    assert [n for n, _, _ in w] == [6, 8, 12, 16, 24, 32, 64, 54] and sum(n for n, _, _ in w) == 216
    assert [c for _, c, _ in w] == [0, 6, 14, 26, 32, 32, 32, 32]
    assert d3.dac_windows(4) == [(4, 0, 0)]


def test_kpis_colocated_dac_and_first_audio():
    k = speech_kpis(_scn(BASE, "drama-3", 10.0), 10.0)
    dac = k["dac"]
    assert k["dac_costed"] and dac["stateful"] and dac["colocated"] and dac["windows"] == [6, 8, 12, 16, 24, 32, 64, 54]
    assert dac["batch"] == 1 and dac["s"] == dac["s_batch1"] and abs(dac["load"] - sum(dac["s"]) / 10.0) < 1e-15
    assert abs(k["rtf"] - (k["lm_rtf"] + dac["load"])) < 1e-15 and 0 < dac["load"] < 0.05 * k["lm_rtf"]
    assert dac["redecode_s_per_request"] > dac["s_per_request_batch1"]       # left context re-decoded
    assert abs(k["first_audio_s"] - (k["ttft_s"] + 6 * k["tpot_s"] + dac["s_batch1"][0])) < 1e-12
    assert abs(dac["GB"] - 4 * REL["params_codec_decode"] / 1e9) < 1e-12
    st, _, _, _ = _stages(k, 10.0)
    assert st == {"lm": k["rtf"]}                       # no separate speech card: the DAC load is on the LM card
    lo = speech_kpis(_scn(BASE, "drama-3", 10.0), 10.0, lm_only=True)
    assert lo["lm_stages"] == {"lm": k["rtf"]} and lo["fits"] == k["fits"]
    # b streams batch each window: weights read once → less than b × the batch-1 time, more than once
    k8 = speech_kpis(_scn(BASE, "drama-3", 10.0).replace("serving.batch", 8), 10.0)
    assert k8["dac"]["batch"] == 8 and sum(k8["dac"]["s_batch1"]) < sum(k8["dac"]["s"]) < 8 * sum(k8["dac"]["s_batch1"])
    lo8 = speech_kpis(_scn(BASE, "drama-3", 10.0).replace("serving.batch", 8), 10.0, lm_only=True)
    assert lo8["lm_stages"] == {"lm": k8["rtf"]}
    assert get_model("drama-3").coverage == "full"


def test_codec_extra_and_summarize_hook():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_fish_codec import codec_extra
    hdr = {"size": 99, "tensors": {"decoder.model.0.conv.weight": {"dtype": "F32", "shape": [4, 2, 7]},
                                   "encoder.block.0.conv.weight": {"dtype": "F32", "shape": [2, 1, 7]},
                                   "quantizer.post_module.causal_mask": {"dtype": "BOOL", "shape": [8, 8]}}}
    ex = codec_extra(hdr)
    assert ex == {"codec_tensors": {"decoder.model.0.conv.weight": [4, 2, 7]}, "params_codec": 70,
                  "params_codec_decode": 56, "codec_dtype": "F32", "codec_file_bytes": 99}
    src = (ROOT / "scripts/summarize_release.py").read_text()
    assert 'fish_codec.json' in src and "codec_extra" in src


def test_ui_mentions_dac():
    js = W.read()                                      # 0.286: parsed
    assert W.any_string_contains(js, "DAC 首窗") and W.has(js, "p.dac.gflop_per_frame") and not W.any_string_contains(js, "（DAC 未计）")
