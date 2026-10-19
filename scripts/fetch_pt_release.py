#!/usr/bin/env python3
"""0.271: release metadata for FunAudioLLM CosyVoice2 / Fun-CosyVoice3 (torch .pt checkpoints, no safetensors).

Tensor names / shapes / dtypes come from each .pt's pickled state dict, read with HTTP Range requests (zip central
directory → data.pkl; no weights downloaded, no torch needed).  Writes local/hf_cache/<org__repo>/ config.json +
headers.json in the safetensors-header layout scripts/summarize_release.py reads:
  * llm.pt  → names as stored (``llm.model.model.*`` Qwen2.5-0.5B backbone, ``speech_embedding``, ``llm_decoder``, …)
  * flow.pt → ``flow.*`` ;  hift.pt → ``hift.*``
config.json = CosyVoice-BlankEN/config.json (the Qwen2 backbone, as ``llm_config``) + ``model_type`` cosyvoice2 /
cosyvoice3 + the fields of cosyvoice{2,3}.yaml the cost model uses (``cosyvoice``).
Usage: python3 scripts/fetch_pt_release.py FunAudioLLM/CosyVoice2-0.5B FunAudioLLM/Fun-CosyVoice3-0.5B-2512
"""
from __future__ import annotations

import collections
import io
import json
import pickle
import re
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "local" / "hf_cache"
HF = "https://huggingface.co"
UA = {"User-Agent": "accel-dse-release-fetch/0.271"}
DT = {"FloatStorage": ("F32", 4), "HalfStorage": ("F16", 2), "BFloat16Storage": ("BF16", 2)}


def _req(url: str, rng: tuple[int, int] | None = None, method: str = "GET"):
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h, method=method), timeout=60)


def _rng(url: str, a: int, b: int) -> bytes:
    with _req(url, (a, b)) as r:
        return r.read()


def _central(url: str) -> dict:
    with _req(url, method="HEAD") as r:
        n = int(r.headers["Content-Length"])
    tail = _rng(url, max(0, n - 65536), n - 1)
    i = tail.rfind(b"PK\x05\x06")
    eocd = tail[i:]
    cnt = struct.unpack("<H", eocd[10:12])[0]
    cd_size, cd_off = struct.unpack("<II", eocd[12:20])
    if cd_off == 0xFFFFFFFF or cnt == 0xFFFF:                 # zip64
        z = tail[tail.rfind(b"PK\x06\x06"):]
        cd_size, cd_off = struct.unpack("<QQ", z[40:56])
    cd = _rng(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        comp, csz, usz = struct.unpack("<HII", cd[p + 10:p + 12] + cd[p + 20:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nl].decode()
        extra, q = cd[p + 46 + nl:p + 46 + nl + el], 0
        while q < len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, k = extra[q + 4:q + 4 + hl], 0
                for flag in (usz, csz, off):
                    if flag == 0xFFFFFFFF:
                        v = struct.unpack("<Q", vals[k:k + 8])[0]
                        k += 8
                        if flag is usz:
                            usz = v
                        elif flag is csz:
                            csz = v
                        else:
                            off = v
            q += 4 + hl
        out[name] = (off, csz, comp)
        p += 46 + nl + el + cl
    return out


class _T:
    def __init__(self, st, shape):
        self.st, self.shape = st, tuple(shape)


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return lambda st, off, shape, stride, *a, **k: _T(st, shape)
        if name == "_rebuild_parameter":
            return lambda t, *a: t
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod.startswith("torch") and name.endswith("Storage"):
            return name
        return lambda *a, **k: None

    def persistent_load(self, pid):
        return pid[1] if isinstance(pid[1], str) else str(pid[1])


def pt_tensors(url: str) -> dict:
    cd = _central(url)
    off, csz, comp = cd[next(k for k in cd if k.endswith("data.pkl"))]
    nl, el = struct.unpack("<HH", _rng(url, off + 26, off + 29))
    data = _rng(url, off + 30 + nl + el, off + 30 + nl + el + csz - 1)
    if comp == 8:
        data = zlib.decompress(data, -15)
    res: dict = {}

    def walk(o, pre):
        if isinstance(o, _T):
            res[pre] = (o.st, o.shape)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{pre}.{k}" if pre else str(k))
    walk(_U(io.BytesIO(data)).load(), "")
    return res


def _yaml_fields(y: str) -> dict:
    g = lambda pat, cast=int: cast(re.search(pat, y).group(1))
    lst = lambda pat: [int(x) for x in re.search(pat, y).group(1).split(",")]
    flow_cls = re.search(r"flow: !new:cosyvoice\.flow\.flow\.(\w+)", y).group(1)
    return {"sample_rate": g(r"\nsample_rate: (\d+)"), "token_frame_rate": g(r"\ntoken_frame_rate: (\d+)"),
            "token_mel_ratio": g(r"\ntoken_mel_ratio: (\d+)"), "chunk_size": g(r"\nchunk_size: (\d+)"),
            "speech_token_size": g(r"speech_token_size: (\d+)"), "flow": flow_cls,
            "llm": re.search(r"llm: !new:cosyvoice\.llm\.llm\.(\w+)", y).group(1),
            "hift": re.search(r"hift: !new:cosyvoice\.hifigan\.generator\.(\w+)", y).group(1),
            "pre_lookahead_len": g(r"pre_lookahead_len: (\d+)"),
            "inference_cfg_rate": g(r"inference_cfg_rate: ([\d.]+)", float),
            "upsample_rates": lst(r"upsample_rates: \[([\d, ]+)\]"),
            "istft_n_fft": g(r"n_fft: (\d+)\s+hop_len"), "istft_hop": g(r"hop_len: (\d+)"),
            "base_channels": g(r"base_channels: (\d+)"), "nb_harmonics": g(r"nb_harmonics: (\d+)"),
            "attention_heads_flow": g(r"(?:num_heads|heads): (\d+)"), "attention_head_dim": g(r"(?:attention_head_dim|dim_head): (\d+)")}


def fetch(repo: str) -> None:
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{HF}/{repo}/resolve/main"
    tree = json.loads(urllib.request.urlopen(urllib.request.Request(f"{HF}/api/models/{repo}/tree/main", headers=UA),
                                             timeout=60).read())
    yname = next(x["path"] for x in tree if x["path"].startswith("cosyvoice") and x["path"].endswith(".yaml"))
    y = urllib.request.urlopen(urllib.request.Request(f"{base}/{yname}", headers=UA), timeout=60).read().decode()
    llm_cfg = json.loads(urllib.request.urlopen(urllib.request.Request(f"{base}/CosyVoice-BlankEN/config.json",
                                                                       headers=UA), timeout=60).read())
    headers = {}
    for f, pre in (("llm.pt", ""), ("flow.pt", "flow."), ("hift.pt", "hift.")):
        h, o = {}, 0
        for name, (st, shape) in pt_tensors(f"{base}/{f}").items():
            dt, nb = DT[st]
            n = 1
            for s in shape:
                n *= s
            h[pre + name] = {"dtype": dt, "shape": list(shape), "data_offsets": [o, o + n * nb]}
            o += n * nb
        headers[f] = h
    cv = _yaml_fields(y)
    cv["speech_vocab"] = headers["llm.pt"]["llm_decoder.weight"]["shape"][0]      # speech tokens + specials (header)
    cfg = {"model_type": "cosyvoice3" if "3" in yname else "cosyvoice2", "architectures": [cv["llm"]],
           "torch_dtype": "float32", "llm_config": llm_cfg, "cosyvoice": cv, "_source_yaml": yname}
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    (d / "headers.json").write_text(json.dumps(headers))
    print(repo, {k: len(v) for k, v in headers.items()})


if __name__ == "__main__":
    for r in sys.argv[1:]:
        fetch(r)
