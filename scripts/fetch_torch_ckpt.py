#!/usr/bin/env python3
"""Read tensor names / shapes / dtypes (and Lightning hyper-parameters) from a PyTorch zip checkpoint over HTTP
without downloading the weights (0.43).  Only the zip central directory and the small `data.pkl` entry are fetched
with HTTP range requests; the pickle is replayed by a restricted unpickler that never imports torch or executes any
callable from the file -- unknown globals become inert placeholder objects.  Output: the same header-JSON shape as
fetch_hf_release.py ({name: {"dtype", "shape"}}) so summarize_release.py can consume it.

usage: fetch_torch_ckpt.py URL OUT.json [--hparams OUT_hparams.json]
"""
from __future__ import annotations

import io
import json
import pickle
import sys
import urllib.request
import zipfile

UA = {"User-Agent": "accel-dse-release-fetch/0.43"}
DT = {"FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16", "DoubleStorage": "F64",
      "LongStorage": "I64", "IntStorage": "I32", "BoolStorage": "BOOL", "ByteStorage": "U8", "CharStorage": "I8",
      "ShortStorage": "I16"}
TDT = {"float32": "F32", "float": "F32", "float16": "F16", "half": "F16", "bfloat16": "BF16", "float64": "F64",
       "int64": "I64", "long": "I64", "int32": "I32", "bool": "BOOL", "uint8": "U8", "int8": "I8"}


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.fetched = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            a, b = i * self.block, min(self.size, (i + 1) * self.block) - 1
            req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.cache[i] = r.read()
            self.fetched += len(self.cache[i])
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, self.block)
            chunk = self._blk(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


class Inert:
    """Placeholder for any global the pickle references (classes, functions).  Calling it records the arguments."""
    def __init__(self, *a, **k):
        self.args, self.kw, self.state = a, k, None

    def __setstate__(self, s): self.state = s
    def __call__(self, *a, **k): return Inert(*a, **k)


class Tensor:
    def __init__(self, dtype, shape): self.dtype, self.shape = dtype, list(shape)


class DType:
    def __init__(self, name): self.name = name


def _rebuild(storage, offset, size, stride, *rest):
    return Tensor(storage[0], size)


def _rebuild_param(t, *rest):
    return t


class Unp(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_param
        if mod == "torch" and name in DT:
            return DT[name]
        if mod == "torch" and name in TDT:
            return DType(name)
        if mod == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if mod == "builtins" and name in ("set", "frozenset", "slice", "complex"):
            return {"set": set, "frozenset": frozenset, "slice": slice, "complex": complex}[name]
        return type(name, (Inert,), {})

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel); storage_type is a *Storage marker or torch.UntypedStorage
        st = pid[1]
        if isinstance(st, str):
            return (st, pid[2], pid[4])
        return ("U8", pid[2], pid[4])


def _walk_state(obj, prefix=""):
    """Find the tensor dict: plain state_dict, {'state_dict': ...}, {'model': ...}, {'ema': ...}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, Tensor):
                out[prefix + str(k)] = v
            elif isinstance(v, dict):
                out.update(_walk_state(v, prefix + str(k) + "."))
    return out


def _plain(o, depth=0):
    if depth > 12:
        return None
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _plain(v, depth + 1) for k, v in o.items() if not isinstance(v, Tensor)}
    if isinstance(o, (list, tuple)):
        return [_plain(v, depth + 1) for v in o]
    if isinstance(o, Inert):
        st = o.state if o.state is not None else (o.args or o.kw)
        return {"__class__": type(o).__name__, "value": _plain(st, depth + 1)}
    if isinstance(o, DType):
        return o.name
    return str(type(o).__name__)


def read(url: str):
    f = RangeFile(url)
    z = zipfile.ZipFile(f)
    pk = [n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
    if len(pk) != 1:
        raise SystemExit(f"expected one data.pkl, got {pk}")
    obj = Unp(io.BytesIO(z.read(pk[0]))).load()
    return obj, f


def main():
    url, out = sys.argv[1], sys.argv[2]
    obj, f = read(url)
    tensors = _walk_state(obj)
    hdr = {k: {"dtype": t.dtype, "shape": t.shape} for k, t in tensors.items()}
    json.dump({"url": url, "size": f.size, "tensors": hdr}, open(out, "w"), indent=0)
    print(f"{len(hdr)} tensors, file {f.size / 1e9:.2f} GB, fetched {f.fetched / 1e6:.1f} MB")
    if "--hparams" in sys.argv:
        hp = obj.get("hyper_parameters") if isinstance(obj, dict) else None
        json.dump(_plain(hp), open(sys.argv[sys.argv.index("--hparams") + 1], "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
