#!/usr/bin/env python3
"""Read array names / shapes / dtypes of one .npz member inside a (plain, uncompressed) remote .tar over HTTP range
requests — no weights are downloaded (tar headers + zip central directory + each .npy header only).

    python3 scripts/fetch_npz_header.py URL MEMBER OUT.json

Used for the AlphaFold 2 release (alphafold_params_2022-12-06.tar, CC BY 4.0)."""
from __future__ import annotations

import ast
import json
import struct
import sys
import urllib.request


def rng(url: str, a: int, n: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{a + n - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def tar_member(url: str, name: str) -> tuple[int, int]:
    off = 0
    while True:
        h = rng(url, off, 512)
        n = h[:100].rstrip(b"\0").decode()
        if not n:
            raise KeyError(name)
        size = int(h[124:136].strip(b"\0 ") or b"0", 8)
        if n == name:
            return off + 512, size
        off += 512 + (size + 511) // 512 * 512


def npz_shapes(url: str, base: int, size: int) -> dict:
    tail = rng(url, base + size - 65536, 65536)
    i = tail.rfind(b"PK\x05\x06")
    cd_n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    if cd_off == 0xFFFFFFFF or cd_n == 0xFFFF:                         # zip64
        j = tail.rfind(b"PK\x06\x06")
        cd_n, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32:j + 56])
    cd = rng(url, base + cd_off, cd_size)
    out, p = {}, 0
    for _ in range(cd_n):
        (method, csize, usize, nlen, xlen, clen) = struct.unpack("<H8xII HHH", cd[p + 10:p + 34])
        loff = struct.unpack("<I", cd[p + 42:p + 46])[0]
        fname = cd[p + 46:p + 46 + nlen].decode()
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if loff == 0xFFFFFFFF or usize == 0xFFFFFFFF:                 # zip64 extra field
            q = 0
            while q < len(extra):
                hid, hl = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals = list(struct.unpack(f"<{hl // 8}Q", extra[q + 4:q + 4 + hl // 8 * 8]))
                    if usize == 0xFFFFFFFF:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = vals.pop(0)
                    if loff == 0xFFFFFFFF:
                        loff = vals.pop(0)
                q += 4 + hl
        p += 46 + nlen + xlen + clen
        if method != 0:
            raise ValueError(f"{fname}: compressed member (method {method})")
        lh = rng(url, base + loff, 30 + 512)
        ln, lx = struct.unpack("<HH", lh[26:30])
        d = lh[30 + ln + lx:]
        if d[:6] != b"\x93NUMPY":
            raise ValueError(fname)
        hl = struct.unpack("<H", d[8:10])[0] if d[6] == 1 else struct.unpack("<I", d[8:12])[0]
        hs = 10 if d[6] == 1 else 12
        hdr = ast.literal_eval(d[hs:hs + hl].decode("latin1"))
        out[fname[:-4] if fname.endswith(".npy") else fname] = {"shape": list(hdr["shape"]), "dtype": hdr["descr"]}
    return out


if __name__ == "__main__":
    url, member, dst = sys.argv[1:4]
    base, size = tar_member(url, member)
    sh = npz_shapes(url, base, size)
    json.dump({"url": url, "member": member, "member_bytes": size, "tensors": sh}, open(dst, "w"), indent=0)
    print(len(sh), "arrays")
