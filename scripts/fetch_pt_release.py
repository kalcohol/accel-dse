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


def fetch_token2wav(repo: str, sub: str = "token2wav") -> None:
    """0.273 Step-Audio-2-mini: the safetensors release (LLM + audio encoder, scripts/fetch_hf_release.py) ships a
    CosyVoice2-style token2wav as .pt: ``<sub>/flow.yaml`` (CausalMaskedDiffWithXvec: UpsampleConformerEncoderV2 +
    CausalConditionalCFM over a DiT estimator) + flow.pt + hift.pt (flashcosyvoice HiFTGenerator defaults:
    upsample 8 / 5 / 3, iSTFT 16 / 4, 24 kHz — Step-Audio2 token2wav.py 「来源」).  Writes token2wav.json next to the
    safetensors headers: {"config": yaml fields, "tensors": {flow.* / hift.*: [dtype, shape]}}."""
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{HF}/{repo}/resolve/main/{sub}"
    y = urllib.request.urlopen(urllib.request.Request(f"{base}/flow.yaml", headers=UA), timeout=60).read().decode()
    g = lambda pat, cast=int: cast(re.search(pat, y).group(1))
    cfg = {"flow": re.search(r"flow: !new:[\w.]+\.(\w+)", y).group(1),
           "encoder": re.search(r"encoder: !new:[\w.]+\.(\w+)", y).group(1),
           "estimator": re.search(r"estimator: !new:[\w.]+\.(\w+)", y).group(1),
           "vocab_size": g(r"vocab_size: (\d+)"), "pre_lookahead_len": g(r"pre_lookahead_len: (\d+)"),
           "up_stride": g(r"up_stride: (\d+)"), "inference_cfg_rate": g(r"inference_cfg_rate: ([\d.]+)", float),
           "num_heads": g(r"estimator:[\s\S]*?num_heads: (\d+)"), "head_dim": g(r"head_dim: (\d+)"),
           # flashcosyvoice HiFTGenerator() defaults (token2wav.py builds it without arguments) 「来源」
           "sample_rate": 24000, "upsample_rates": [8, 5, 3], "istft_n_fft": 16, "istft_hop": 4,
           "token_frame_rate": 25, "token_mel_ratio": 2, "chunk_size": 25, "mel_cache": 8}
    tensors = {}
    for f, pre in (("flow.pt", "flow."), ("hift.pt", "hift.")):
        for name, (st, shape) in pt_tensors(f"{base}/{f}").items():
            tensors[pre + name] = [DT[st][0], list(shape)]
    (d / "token2wav.json").write_text(json.dumps({"config": cfg, "tensors": tensors}))
    print(repo, sub, len(tensors), cfg)


def fetch_kimi(repo: str) -> None:
    """0.273 Kimi-Audio-7B-Instruct: the safetensors index (scripts/fetch_hf_release.py) holds only the LLM (+ mimo
    audio branch, vq_adaptor).  The rest of the release: ``whisper-large-v3/`` (transformers WhisperForConditionalGeneration safetensors, model.encoder.*;
    Kimi-Audio runs its encoder only — kimia_infer/models/tokenizer/whisper_Lv3/whisper.py), ``audio_detokenizer/``
    (config.yaml + model.pt: flow-matching DiT, state_dict.speech_model.*) and ``vocoder/`` (config.json + model.pt:
    BigVGAN generator.*, weight norm).  Writes kimi_audio.json: {"config": {whisper, detokenizer yaml fields, vocoder,
    streaming constants of kimia_infer (「来源」 api/kimia.py detokenize_audio, models/detokenizer/__init__.py)},
    "tensors": {whisper.<encoder.*> / detok.* / voc.*: [dtype, shape]}}; weight norm (weight_v / weight_g) renamed to
    parametrizations.weight.original1 / original0 (the CosyVoice reader's convention)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fhr", str(ROOT / "scripts" / "fetch_hf_release.py"))
    fhr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fhr)
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{HF}/{repo}/resolve/main"
    get = lambda f: urllib.request.urlopen(urllib.request.Request(f"{base}/{f}", headers=UA), timeout=60).read().decode()
    wc = json.loads(get("whisper-large-v3/config.json"))
    vc = json.loads(get("vocoder/config.json"))
    y = get("audio_detokenizer/config.yaml")
    g = lambda pat, cast=int: cast(re.search(pat, y).group(1))
    cfg = {"whisper": {k: wc[k] for k in ("d_model", "encoder_layers", "encoder_attention_heads", "encoder_ffn_dim",
                                          "num_mel_bins", "max_source_positions")},
           "dit": {"depth": g(r"\n    depth: (\d+)"), "hidden_size": g(r"\n    hidden_size: (\d+)"),
                   "num_heads": g(r"\n    num_heads: (\d+)"), "mlp_ratio": g(r"mlp_ratio: ([\d.]+)", float),
                   "input_size": g(r"\n    input_size: (\d+)"), "semantic_vocab_size": g(r"semantic_vocab_size: (\d+)"),
                   "hz": g(r"hz: (\d+)")},
           "vocoder": {k: vc[k] for k in ("upsample_rates", "upsample_initial_channel", "resblock_kernel_sizes",
                                          "resblock_dilation_sizes", "hop_size", "sampling_rate", "num_mels")},
           # kimia_infer 「来源」: detokenize_audio chunk_size = first_chunk_size = 30 tokens, upsample_factor 4
           # (12.5 Hz tokens → 50 Hz mel), get_audio_detokenizer look_ahead_tokens 12, use_cfg False,
           # detokenize_streaming ode_step 30 (Euler), max_kv_cache_tokens 900 (StreamingSemanticFMWrapper default),
           # kimia_mimo_audiodelaytokens (config.json) blank audio steps before the first audio token
           "token_rate": 12.5, "upsample": 4, "chunk": 30, "first_chunk": 30, "look_ahead": 12, "ode_steps": 30,
           "cfg": False, "max_kv_frames": 900}
    tensors = {}
    for f, h in fhr.st_header_url(f"{base}/whisper-large-v3/model.safetensors").items():
        if f.startswith("model.encoder."):       # WhisperForConditionalGeneration layout; decoder not run
            tensors["whisper." + f[len("model."):]] = [h["dtype"], h["shape"]]

    def norm(n: str) -> str:
        return re.sub(r"\.weight_v$", ".parametrizations.weight.original1",
                      re.sub(r"\.weight_g$", ".parametrizations.weight.original0", n))
    for f, src, pre in (("audio_detokenizer/model.pt", "state_dict.speech_model.", "detok."),
                        ("vocoder/model.pt", "generator.", "voc.")):
        for name, (st, shape) in pt_tensors(f"{base}/{f}").items():
            if name.startswith(src):
                tensors[pre + norm(name[len(src):])] = [DT[st][0], list(shape)]
    (d / "kimi_audio.json").write_text(json.dumps({"config": cfg, "tensors": tensors}))
    print(repo, len(tensors), cfg)


MOSHI_SRC = "github kyutai-labs/moshi@e6a55d2 moshi/models/loaders.py (_seanet_kwargs, _quantizer_kwargs, _transformer_kwargs, _mimi_config, _lm_kwargs)"


def fetch_moshi(repo: str) -> None:
    """0.274 Kyutai Moshi (moshiko / moshika-pytorch-bf16): the HF repo has model.safetensors (LMModel: temporal
    transformer + depformer) and the Mimi codec safetensors but no config.json — the architecture lives in the moshi
    package's loaders.py defaults, transcribed here 「来源」 (MOSHI_SRC; checked against the header shapes by
    tests/test_core_274.py).  Writes local/hf_cache/<repo>/config.json (model_type "moshi", lm + mimi dicts) +
    headers.json (model.safetensors, the layout summarize_release.py reads) + moshi_audio.json (Mimi tensors:
    mimi.<name>: [dtype, shape])."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fhr", str(ROOT / "scripts" / "fetch_hf_release.py"))
    fhr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fhr)
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{HF}/{repo}/resolve/main"
    seanet = {"dimension": 512, "channels": 1, "causal": True, "n_filters": 64, "n_residual_layers": 1,
              "activation": "ELU", "compress": 2, "dilation_base": 2, "kernel_size": 7, "residual_kernel_size": 3,
              "last_kernel_size": 3, "ratios": [8, 6, 5, 4], "true_skip": True}
    mimi = {"sample_rate": 24000, "channels": 1, "frame_rate": 12.5, "seanet": seanet,
            "quantizer": {"dimension": 256, "n_q": 32, "bins": 2048, "input_dimension": 512, "output_dimension": 512},
            "transformer": {"d_model": 512, "num_heads": 8, "num_layers": 8, "causal": True, "layer_scale": 0.01,
                            "context": 250, "gating": "none", "norm": "layer_norm", "positional_embedding": "rope",
                            "dim_feedforward": 2048}}
    lm = {"dim": 4096, "text_card": 32000, "existing_text_padding_id": 3, "n_q": 16, "dep_q": 8, "card": 2048,
          "num_heads": 32, "num_layers": 32, "hidden_scale": 4.125, "causal": True, "context": 3000,
          "max_period": 10000, "gating": "silu", "norm": "rms_norm_f32", "positional_embedding": "rope",
          "depformer_dim": 1024, "depformer_dim_feedforward": int(4.125 * 1024), "depformer_num_heads": 16,
          "depformer_num_layers": 6, "depformer_multi_linear": True, "depformer_context": 8,
          "depformer_gating": "silu", "depformer_pos_emb": "none", "depformer_weights_per_step": True,
          "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]}
    cfg = {"model_type": "moshi", "architectures": ["MoshiLMModel"], "_source": MOSHI_SRC, "torch_dtype": "bfloat16",
           "lm": lm, "mimi": mimi, "mimi_file": "tokenizer-e351c8d8-checkpoint125.safetensors"}
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    (d / "headers.json").write_text(json.dumps({"model.safetensors": fhr.st_header_url(f"{base}/model.safetensors")}))
    t = {"mimi." + n: [h["dtype"], h["shape"]] for n, h in fhr.st_header_url(f"{base}/{cfg['mimi_file']}").items()
         if n != "__metadata__"}
    (d / "moshi_audio.json").write_text(json.dumps({"tensors": t}))
    print(repo, len(t))


FIRERED_SRC = ("github FireRedTeam/FireRedTTS3@7a1f3a7 fireredtts3/llm/fireredtts3_base.py (Qwen3_1_7B_ConfigDict, "
               "FireRedTTS3BaseCore.generate / _flow_one_step), llm/dit.py, llm/patch_encoder.py, llm/modules.py, "
               "redae/redae.py, core.py (n_timesteps 10, inference_cfg 2.0)")


def fetch_firered(repo: str) -> None:
    """0.275 FireRedTTS3-Base: the HF repo has fireredtts3_base/{config.json, model.safetensors} (DiT / patch-encoder
    fields only — the Qwen3-1.7B backbone config is a dict in the code) and redae/{config.json, model.safetensors}.
    Writes local/hf_cache/<repo>/config.json (model_type "fireredtts3": backbone = Qwen3_1_7B_ConfigDict transcribed
    from fireredtts3_base.py 「来源」 FIRERED_SRC, core = base config, redae = redae config, generate() defaults) +
    headers.json (the base model.safetensors) + firered_audio.json (RedAE tensors: redae.<name>: [dtype, shape]).
    The campplus speaker encoder (campp/campplus_voxceleb.bin, run once per request on the prompt audio) is not read."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fhr", str(ROOT / "scripts" / "fetch_hf_release.py"))
    fhr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fhr)
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    base = f"{HF}/{repo}/resolve/main"
    get = lambda f: json.loads(urllib.request.urlopen(f"{base}/{f}", timeout=60).read())
    backbone = {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "head_dim": 128, "hidden_act": "silu",
                "hidden_size": 2048, "intermediate_size": 6144, "max_position_embeddings": 40960,
                "max_window_layers": 28, "model_type": "qwen3", "num_attention_heads": 16, "num_hidden_layers": 28,
                "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_theta": 1000000, "sliding_window": None,
                "tie_word_embeddings": True, "use_sliding_window": False, "vocab_size": 151936}
    cfg = {"model_type": "fireredtts3", "architectures": ["FireRedTTS3BaseCore"], "_source": FIRERED_SRC,
           "torch_dtype": "float32", "backbone": backbone, "core": get("fireredtts3_base/config.json"),
           "redae": get("redae/config.json"),
           "generate": {"n_timesteps": 10, "inference_cfg": 2.0, "t_span": "1 - cos(linspace(0, 1, n + 1) · π / 2)",
                        "backbone_autocast": "bfloat16", "max_gen_steps": 400}}
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    (d / "headers.json").write_text(json.dumps({"model.safetensors": fhr.st_header_url(f"{base}/fireredtts3_base/model.safetensors")}))
    t = {"redae." + n: [h["dtype"], h["shape"]] for n, h in fhr.st_header_url(f"{base}/redae/model.safetensors").items()
         if n != "__metadata__"}
    (d / "firered_audio.json").write_text(json.dumps({"tensors": t}))
    print(repo, len(t))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--token2wav":         # 0.273: python3 scripts/fetch_pt_release.py --token2wav <repo>
        for r in args[1:]:
            fetch_token2wav(r)
    elif args and args[0] == "--firered":         # 0.275: python3 scripts/fetch_pt_release.py --firered <repo>
        for r in args[1:]:
            fetch_firered(r)
    elif args and args[0] == "--moshi":           # 0.274: python3 scripts/fetch_pt_release.py --moshi <repo>
        for r in args[1:]:
            fetch_moshi(r)
    elif args and args[0] == "--kimi":            # 0.273: python3 scripts/fetch_pt_release.py --kimi <repo>
        for r in args[1:]:
            fetch_kimi(r)
    else:
        for r in args:
            fetch(r)
