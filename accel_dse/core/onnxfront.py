"""0.280 audio-frontend graphs read from the released ONNX files (scripts/fetch_onnx_graph.py → release key
``onnx_frontend``): CosyVoice3 speech_tokenizer_v3 (S3Tokenizer-style: 2 stride-2 convs → 12 blocks of q / k / v / o +
FSMN depthwise conv k31 + full attention + GELU MLP → FSQ projection) and CampPlus (CAM++ speaker embedding: 2-D conv
head → TDNN / dense CAM blocks → statistics pooling → 192-d embedding).

Each descriptor entry carries its own time divisor dt (from the graph's conv strides 「推导」), so the op list for a clip
of ``seconds`` is exact for the GEMM / conv / attention FLOPs of the graph; biases, norms and element-wise ops are
charged as the GEMMs' per-output vector work plus one op reading the remaining weights 「近似」.  Weights and activations fp32 (the ONNX initializers are
F32; onnxruntime runs them as stored)."""
from __future__ import annotations

import math

from .ir import Op


def positions(d: dict, seconds: float, dt: int) -> int:
    return 1 if not dt else max(1, math.ceil(d["rate_hz"] * seconds / dt - 1e-9))


def front_ops(d: dict, seconds: float, tag: str = "fe") -> list[Op]:
    ops: list[Op] = []
    wsum = 0
    for i, o in enumerate(d["ops"]):
        if o[0] == "gemm":
            _, k, n, df, dt, two = o
            m = positions(d, seconds, dt) * (max(1, math.ceil(d["freq"] / df - 1e-9)) if two else 1)
            wsum += k * n
            ops.append(Op(f"{tag}.gemm{i}", "gemm", 0, m=m, k=k, n=n, role="speech", w_params=k * n, w_bits=32.0,
                          w_fmt="fp32", a_fmt="fp32", act_bytes=m * (k + n) * 4.0, vec=m * n * 2.0, stream=True))
        elif o[0] == "dw":
            _, c, kw, dt = o
            t = positions(d, seconds, dt)
            wsum += c * kw
            ops.append(Op(f"{tag}.dw{i}", "vector", 0, vec=2.0 * c * kw * t, w_params=c * kw, w_bits=32.0, w_fmt="fp32",
                          act_bytes=2.0 * c * t * 4.0))
        elif o[0] == "attn":
            _, D, H, dt = o
            t = positions(d, seconds, dt)
            H = max(1, H)
            hd = D // H
            ops += [Op(f"{tag}.attn{i}.qk", "attn", 0, m=t, k=hd, n=t, count=H, a_fmt="fp32",
                       act_bytes=H * 2 * t * hd * 4.0),
                    Op(f"{tag}.attn{i}.pv", "attn", 0, m=t, k=t, n=hd, count=H, a_fmt="fp32",
                       act_bytes=H * 2 * t * hd * 4.0),
                    Op(f"{tag}.attn{i}.softmax", "vector", 0, vec=H * t * t * 6.0)]
    rest = d["params"] - wsum               # biases, norm / BN parameters: read once per pass (element-wise work rides on
    if rest > 0:                            # each GEMM's 2 vector ops / output 「近似」)
        ops.append(Op(f"{tag}.rest", "vector", 0, w_params=rest, w_bits=32.0, w_fmt="fp32"))
    return ops


def front_flops(d: dict, seconds: float) -> float:
    return sum(o.flops for o in front_ops(d, seconds) if o.kind in ("gemm", "attn"))
