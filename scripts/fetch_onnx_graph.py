#!/usr/bin/env python3
"""0.280: read an ONNX model's graph structure over HTTP Range — node list (op_type, inputs, outputs, int / ints /
float attributes) and initializer shapes — without downloading the weights (raw_data is skipped by length).

Stdlib only (no onnx / protobuf package): a minimal protobuf wire-format walker over onnx.proto field numbers
「来源」 (onnx/onnx.proto3: ModelProto.graph = 7; GraphProto node = 1, initializer = 5, input = 11, output = 12;
NodeProto input = 1, output = 2, name = 3, op_type = 4, attribute = 5; AttributeProto name = 1, f = 2, i = 3, t = 5,
floats = 7, ints = 8; TensorProto dims = 1, data_type = 2, name = 8, raw_data = 9).

    python3 scripts/fetch_onnx_graph.py <repo> <file.onnx> [...]
writes local/hf_cache/<repo>/onnx/<file>.json  {nodes: [[op, [in], [out], {attr: val}]], inits: {name: [dtype, dims]}}
    python3 scripts/fetch_onnx_graph.py --release <repo> <file.onnx> [...]
writes the cost descriptors (frontend_ops) into the release JSON key onnx_frontend
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "local" / "hf_cache"
UA = {"User-Agent": "accel-dse-onnx-graph/0.280"}
BLOCK = 1 << 20
DTYPE = {1: "F32", 2: "U8", 3: "I8", 5: "I16", 6: "I32", 7: "I64", 9: "BOOL", 10: "F16", 11: "F64", 16: "BF16"}


class Ranged:
    def __init__(self, url: str):
        self.url, self.blocks, self.reads = url, {}, 0

    def _block(self, k: int) -> bytes:
        if k not in self.blocks:
            req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={k * BLOCK}-{(k + 1) * BLOCK - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.blocks[k] = r.read()
            self.reads += 1
            if len(self.blocks) > 64:
                self.blocks.pop(next(iter(self.blocks)))
        return self.blocks[k]

    def byte(self, i: int) -> int:
        return self._block(i // BLOCK)[i % BLOCK]

    def read(self, i: int, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            b = self._block(i // BLOCK)
            o = i % BLOCK
            c = b[o:o + n]
            out += c
            i += len(c)
            n -= len(c)
        return bytes(out)


def varint(src, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = src.byte(i) if isinstance(src, Ranged) else src[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def fields(src, i: int, end: int, skip=()):
    """Yield (field, wire, value | (start, length)) over [i, end).  Length-delimited fields in ``skip`` are not read."""
    while i < end:
        k, i = varint(src, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = varint(src, i)
            yield f, w, v
        elif w == 2:
            n, i = varint(src, i)
            yield f, w, (i, n)
            i += n
        elif w == 5:
            yield f, w, (i, 4)
            i += 4
        elif w == 1:
            yield f, w, (i, 8)
            i += 8
        else:
            raise ValueError(f"wire type {w} at {i}")


def _bytes(src, span) -> bytes:
    i, n = span
    return src.read(i, n) if isinstance(src, Ranged) else src[i:i + n]


def _packed_ints(b: bytes) -> list[int]:
    out, i = [], 0
    while i < len(b):
        v, i = varint(b, i)
        out.append(v - (1 << 64) if v >= 1 << 63 else v)
    return out


def tensor(src, span) -> tuple[str, str, list[int]]:
    i, n = span
    dims, dt, name = [], 0, ""
    for f, w, v in fields(src, i, i + n):
        if f == 1:
            dims += [v] if w == 0 else _packed_ints(_bytes(src, v))
        elif f == 2:
            dt = v
        elif f == 8:
            name = _bytes(src, v).decode()
    return name, DTYPE.get(dt, str(dt)), dims


def node(src, span) -> list:
    import struct
    i, n = span
    ins, outs, op, attrs = [], [], "", {}
    for f, w, v in fields(src, i, i + n):
        if f == 1:
            ins.append(_bytes(src, v).decode())
        elif f == 2:
            outs.append(_bytes(src, v).decode())
        elif f == 4:
            op = _bytes(src, v).decode()
        elif f == 5:
            a = _bytes(src, v)
            an, val = "", None
            for g, wg, x in fields(a, 0, len(a)):
                if g == 1:
                    an = a[x[0]:x[0] + x[1]].decode()
                elif g == 2:
                    val = struct.unpack("<f", a[x[0]:x[0] + 4])[0]
                elif g == 3:
                    val = x - (1 << 64) if x >= 1 << 63 else x
                elif g == 5:
                    tn, tdt, td = tensor(a, x)
                    val = {"tensor": [tdt, td]}
                elif g == 8:
                    val = (val or []) + ([x - (1 << 64) if x >= 1 << 63 else x] if wg == 0 else _packed_ints(a[x[0]:x[0] + x[1]]))
            attrs[an] = val
    return [op, ins, outs, attrs]


def graph(url: str) -> dict:
    src = Ranged(url)
    g = None
    for f, w, v in fields(src, 0, 1 << 62):
        if f == 7:
            g = v
            break
    nodes, inits, gin, gout = [], {}, [], []
    gi, gn = g
    for f, w, v in fields(src, gi, gi + gn):
        if f == 1:
            nodes.append(node(src, v))
        elif f == 5:
            nm, dt, dims = tensor(src, v)
            inits[nm] = [dt, dims]
        elif f in (11, 12):
            b = _bytes(src, v)
            nm = next((b[x[0]:x[0] + x[1]].decode() for h, _, x in fields(b, 0, len(b)) if h == 1), "")
            (gin if f == 11 else gout).append(nm)
    return {"nodes": nodes, "inits": inits, "inputs": gin, "outputs": gout, "range_reads": src.reads}


def frontend_ops(g: dict, freq: int = 0) -> dict:
    """Compact cost descriptor of an audio-frontend graph, by walking the nodes in order and tracking each activation's
    time divisor dt (and frequency divisor df for 2-D convolutions) from the graph input 「推导」:
        Shape / Size   not an activation (the shape arithmetic and masks built from it are skipped)
        Conv           dt ×= time stride (last kernel axis), df ×= freq stride (2-D);  Reshape / Squeeze after a 2-D
                       stage → 1-D (frequency folded into channels)
        ReduceMean keepdims 0 over the last axis → one position (statistics pooling); keepdims 1 / AveragePool feed
                       an expand back to full length (CAM++ context / segment pooling) → passthrough 「近似」
        other ops      min dt of their activation inputs
    Entries (dt 0 = one position): ["gemm", k, n, df, dt, 2d] (Conv with group 1 or MatMul with an initializer weight; k = Cin·kernel),
    ["dw", channels, kernel, dt] (depthwise Conv), ["attn", D, heads, dt] (one per Softmax over activation MatMuls; D = the
    block's query width, heads = D / rotary table width when present)."""
    import math
    I = g["inits"]
    st: dict[str, tuple] = {g["inputs"][0]: (1, 1, bool(freq))}   # the feature input; lengths are not activations
    ops, rot, qw = [], None, {}
    for op, ins, outs, at in g["nodes"]:
        act = [st[x] for x in ins if x in st]
        if op in ("Shape", "Size"):
            continue                          # shape arithmetic is not an activation path
        if op == "Constant":
            v = at.get("value")
            if isinstance(v, dict) and "rotary_emb" in outs[0] and len(v["tensor"][1]) == 2:
                rot = v["tensor"][1][1]
            continue
        if not act:
            continue
        df = min(a[0] for a in act)
        dt = min(a[1] for a in act)
        two = any(a[2] for a in act)
        if op == "Conv":
            w = I[ins[1]][1]
            gr = at.get("group", 1)
            s_ = at.get("strides") or [1]
            if len(w) == 4:
                df, dt = df * s_[0], dt * s_[-1]
            else:
                dt *= s_[-1]
                two = False
            if gr > 1 and w[1] == 1:
                ops.append(["dw", w[0], math.prod(w[2:]), dt])
            else:
                ops.append(["gemm", w[1] * math.prod(w[2:]), w[0], df, dt, len(w) == 4])
        elif op == "MatMul" and len(ins) > 1 and ins[1] in I:
            k, n = I[ins[1]][1][-2:]
            ops.append(["gemm", k, n, 1, dt, False])
            if "/query/" in outs[0]:
                qw[outs[0].split("/attn/")[0]] = n
        elif op == "Softmax":
            blk = outs[0].split("/attn/")[0]
            D = qw.get(blk, 0)
            ops.append(["attn", D, D // rot if rot else 0, dt])
        elif op in ("Reshape", "Squeeze") and two and len(ins) and not any(x in I for x in ins[:1]):
            two = op == "Reshape" and False
        elif op == "ReduceMean" and at.get("keepdims", 1) == 0 and (at.get("axes") or [-1])[-1] == -1:
            dt = math.inf
        for o in outs:
            st[o] = (df, dt, two)
    params = sum(math.prod(d) for _, d in I.values())
    ops = [[0 if x == math.inf else x for x in o] for o in ops]      # dt 0 = one position (pooled statistics)
    return {"ops": ops, "params": params, "dtypes": sorted({d for d, _ in I.values()})}


# CosyVoice frontend (cosyvoice/cli/frontend.py, FunAudioLLM/CosyVoice main) 「来源」: speech tokenizer on
# CUDAExecutionProvider (when CUDA is present) with whisper.log_mel_spectrogram(n_mels=128) at 16 kHz (hop 160 → 100
# frames / s), asserted ≤ 30 s; CampPlus on CPUExecutionProvider with kaldi.fbank(num_mel_bins=80) (10 ms shift).
FRONTEND = {"speech_tokenizer_v3.onnx": {"role": "speech_tokenizer", "rate_hz": 100, "bins": 128, "freq": 0,
                                         "provider": "cuda", "max_s": 30},
            "campplus.onnx": {"role": "campplus", "rate_hz": 100, "bins": 80, "freq": 80, "provider": "cpu", "max_s": None}}


def into_release(repo: str, files: list[str]) -> None:
    """Write the cost descriptors under release key ``onnx_frontend`` (accel_dse/data/releases/<repo>.json)."""
    rp = ROOT / "accel_dse" / "data" / "releases" / (repo.replace("/", "__") + ".json")
    rel = json.loads(rp.read_text())
    fe = {}
    for fn in files:
        g = json.loads((CACHE / repo.replace("/", "__") / "onnx" / (fn + ".json")).read_text())
        meta = FRONTEND[fn]
        fe[meta["role"]] = {"file": fn, **meta, **frontend_ops(g, meta["freq"])}
    rel["onnx_frontend"] = fe
    rp.write_text(json.dumps(rel, indent=1, sort_keys=True))


def main(argv):
    if argv and argv[0] == "--release":
        return into_release(argv[1], argv[2:])
    repo, files = argv[0], argv[1:]
    out = CACHE / repo.replace("/", "__") / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    for fn in files:
        g = graph(f"https://huggingface.co/{repo}/resolve/main/{fn}")
        (out / (fn + ".json")).write_text(json.dumps(g))
        p = sum(__import__("math").prod(d) for _, d in g["inits"].values())
        print(f"{repo}/{fn}: {len(g['nodes'])} nodes, {len(g['inits'])} initializers, {p:,} elements, "
              f"{g['range_reads']} range reads")


if __name__ == "__main__":
    main(sys.argv[1:])
