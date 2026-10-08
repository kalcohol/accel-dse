#!/usr/bin/env python3
"""Turn a PyTorch-checkpoint header (fetch_torch_ckpt.py output) into a release JSON for a structure model (0.43).

Same shape as summarize_release.py output where the builders need it: ``shapes`` (every non-stack tensor plus the
first and last block of each repeated stack), ``stacks`` (block counts), ``roles`` (params / bytes per storage dtype),
``params_llm`` / ``bytes_llm`` (all floating-point tensors the inference model loads; integer buffers carry no
params), ``config`` (official config values the builder cites) and ``source``.

usage: summarize_ckpt.py NAME   (NAME in RELEASES below; reads local/hf_cache/torch/<cache>.json)
"""
from __future__ import annotations

import datetime
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "accel_dse" / "data" / "releases"
CACHE = ROOT / "local" / "hf_cache" / "torch"
STACK = re.compile(r"^(.*?\.(?:blocks|layers|layer))\.(\d+)\.(.*)$")
BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I64": 8, "I32": 4, "BOOL": 1, "U8": 1, "I8": 1, "I16": 2}
FMT = {"F32": "fp32", "F16": "fp16", "BF16": "bf16", "F64": "fp64"}

RELEASES = {
    "esmfold": dict(
        repo="facebook/esmfold_v1", cache="esmfold", strip="",
        url="https://huggingface.co/facebook/esmfold_v1/resolve/main/pytorch_model.bin",
        config_url="https://huggingface.co/facebook/esmfold_v1/raw/main/config.json",
        role=lambda k: ("attn" if re.search(r"esm\.encoder\.layer\.\d+\.attention", k) else
                        "mlp" if re.search(r"esm\.encoder\.layer\.\d+\.(intermediate|output)", k) else
                        "embed" if k.startswith("esm.") else "pair")),
    "openfold": dict(
        repo="gh:aqlaboratory/openfold", cache="openfold_ptm2", strip="",
        url="https://openfold.s3.amazonaws.com/openfold_params/finetuning_ptm_2.pt",
        config={"source": "openfold/config.py model_config('finetuning_ptm') + data.predict",
                "c_m": 256, "c_z": 128, "c_e": 64, "c_t": 64, "c_s": 384, "no_blocks": 48, "extra_msa_blocks": 4,
                "template_blocks": 2, "max_msa_clusters": 512, "max_extra_msa": 1024, "max_templates": 4,
                "max_recycling_iters": 3, "no_ipa_blocks": 8, "c_hidden_msa_att": 32},
        role=lambda k: "pair"),
    "alphafold2": dict(
        repo="gh:google-deepmind/alphafold", cache="af2_model_1_ptm", npz=True,
        url="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar#params_model_1_ptm.npz",
        format="JAX/haiku .npz inside the release .tar (tar headers + zip central directory + .npy headers read with "
               "HTTP range requests; no weights downloaded); haiku names mapped to OpenFold-style names, "
               "[in, out] → [out, in], layer stacks unrolled",
        config={"source": "alphafold/model/config.py (model_1_ptm: CONFIG + CONFIG_DIFFS['model_1_ptm']) + "
                          "run_alphafold.py (monomer_ptm preset)",
                "c_m": 256, "c_z": 128, "c_e": 64, "c_t": 64, "c_s": 384, "no_blocks": 48, "extra_msa_blocks": 4,
                "template_blocks": 2, "max_msa_clusters": 512, "max_extra_msa": 5120, "max_templates": 4,
                "max_recycling_iters": 3, "no_ipa_blocks": 8, "c_hidden_msa_att": 32},
        role=lambda k: "pair"),
    "boltz1": dict(
        repo="boltz-community/boltz-1", cache="boltz1_conf", strip="state_dict.",
        url="https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
        hparams=("atom_s", "atom_z", "token_s", "token_z", "atoms_per_window_queries", "atoms_per_window_keys",
                 "confidence_imitate_trunk"),
        config={"source": "boltz v0.4.1 main.py (predict defaults) + checkpoint hyper_parameters",
                "recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1, "max_msa_seqs": 4096},
        role=lambda k: "pair"),
    "protenix": dict(
        repo="gh:bytedance/Protenix", cache="protenix_v0.5.0", strip="model.module.",
        url="https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.5.0.pt",
        config={"source": "Protenix v0.5.0 configs/configs_base.py + configs_data.py",
                "N_cycle": 4, "N_step": 200, "N_sample": 5, "msa_sample_cutoff_test": 2048, "template": False,
                "c_s": 384, "c_z": 128, "c_token": 768, "c_atom": 128, "c_atompair": 16, "pairformer_blocks": 48,
                "msa_blocks": 4, "diffusion_blocks": 24, "atom_blocks": 3, "confidence_blocks": 4},
        role=lambda k: "pair"),
}


# ------------------------------------------------------------------ AlphaFold 2 (JAX / haiku .npz → torch-style names)
_AF2_STACKS = (("evoformer/evoformer_iteration/", "evoformer.blocks."),
               ("evoformer/extra_msa_stack/", "extra_msa_stack.blocks."),
               ("evoformer/template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state/",
                "template_pair_stack.blocks."))
_AF2_MOD = [  # module path (haiku) → torch-style module name (OpenFold naming, so structure.py's tables apply)
    ("msa_row_attention_with_pair_bias/attention", "msa_att_row.mha"), ("msa_row_attention_with_pair_bias", "msa_att_row"),
    ("msa_column_global_attention/attention", "msa_att_col.global_attention"),
    ("msa_column_global_attention", "msa_att_col"),
    ("msa_column_attention/attention", "msa_att_col._msa_att.mha"), ("msa_column_attention", "msa_att_col._msa_att"),
    ("triangle_attention_starting_node/attention", "tri_att_start.mha"), ("triangle_attention_starting_node", "tri_att_start"),
    ("triangle_attention_ending_node/attention", "tri_att_end.mha"), ("triangle_attention_ending_node", "tri_att_end"),
    ("triangle_multiplication_outgoing", "tri_mul_out"), ("triangle_multiplication_incoming", "tri_mul_in"),
    ("outer_product_mean", "outer_product_mean"), ("msa_transition", "msa_transition"), ("pair_transition", "pair_transition"),
    ("evoformer/template_embedding/single_template_embedding/embedding2d", "template_pair_embedder.linear"),
    ("evoformer/template_embedding/single_template_embedding/output_layer_norm", "template_pair_stack.layer_norm"),
    ("evoformer/template_embedding/attention", "template_pointwise_att.mha"),
    ("evoformer/preprocess_1d", "input_embedder.linear_tf_m"), ("evoformer/preprocess_msa", "input_embedder.linear_msa_m"),
    ("evoformer/left_single", "input_embedder.linear_tf_z_i"), ("evoformer/right_single", "input_embedder.linear_tf_z_j"),
    ("evoformer/pair_activiations", "input_embedder.linear_relpos"),
    ("evoformer/prev_pos_linear", "recycling_embedder.linear"),
    ("evoformer/prev_msa_first_row_norm", "recycling_embedder.layer_norm_m"),
    ("evoformer/prev_pair_norm", "recycling_embedder.layer_norm_z"),
    ("evoformer/extra_msa_activations", "extra_msa_embedder.linear"),
    ("evoformer/template_single_embedding", "template_angle_embedder.linear_1"),
    ("evoformer/template_projection", "template_angle_embedder.linear_2"),
    ("evoformer/single_activations", "evoformer.linear"),
    ("distogram_head/half_logits", "aux_heads.distogram.linear"),
    ("experimentally_resolved_head/logits", "aux_heads.experimentally_resolved.linear"),
    ("masked_msa_head/logits", "aux_heads.masked_msa.linear"),
    ("predicted_aligned_error_head/logits", "aux_heads.tm.linear"),
    ("predicted_lddt_head", "aux_heads.plddt"),
    ("structure_module/fold_iteration/invariant_point_attention/q_scalar", "structure_module.ipa.linear_q"),
    ("structure_module/fold_iteration/invariant_point_attention/kv_scalar", "structure_module.ipa.linear_kv"),
    ("structure_module/fold_iteration/invariant_point_attention/q_point_local", "structure_module.ipa.linear_q_points"),
    ("structure_module/fold_iteration/invariant_point_attention/kv_point_local", "structure_module.ipa.linear_kv_points"),
    ("structure_module/fold_iteration/invariant_point_attention/attention_2d", "structure_module.ipa.linear_b"),
    ("structure_module/fold_iteration/invariant_point_attention/output_projection", "structure_module.ipa.linear_out"),
    ("structure_module/fold_iteration/invariant_point_attention", "structure_module.ipa"),
    ("structure_module/fold_iteration", "structure_module"), ("structure_module", "structure_module"),
]
_AF2_LEAF = {"query_w": "linear_q", "key_w": "linear_k", "value_w": "linear_v", "gating_w": "linear_g",
             "output_w": "linear_o", "feat_2d_weights": "linear_z", "left_projection": "linear_a_p",
             "right_projection": "linear_b_p", "left_gate": "linear_a_g", "right_gate": "linear_b_g",
             "output_projection": "linear_z", "gating_linear": "linear_g", "transition1": "linear_1",
             "transition2": "linear_2"}

_AF2_RENAME = ((r"outer_product_mean\.linear_a_p\.", "outer_product_mean.linear_1."),
               (r"outer_product_mean\.linear_b_p\.", "outer_product_mean.linear_2."),
               (r"outer_product_mean\.linear_o\.", "outer_product_mean.linear_out."),
               (r"(tri_att_(start|end))\.linear_z\.", r"\1.linear."))


def _af2_torch_shape(param: str, shp: list) -> list:
    """haiku [in, (h, d)] / [(h, d), out] → torch [out, in] (2-D weights); 1-D params unchanged."""
    if param in ("output_w",) and len(shp) >= 3:          # [h, d, out] or opm [c, c, out]
        return [shp[-1], math.prod(shp[:-1])]
    if param in ("weights", "query_w", "key_w", "value_w", "gating_w") and len(shp) >= 2:
        return [math.prod(shp[1:]), shp[0]]
    if param == "feat_2d_weights":                        # [c_z, h]
        return [shp[1], shp[0]]
    return list(shp)


def af2_tensors(raw: dict) -> dict:
    out = {}
    for key, v in raw["tensors"].items():
        path = key.removeprefix("alphafold/alphafold_iteration/")
        mod, param = path.split("//")
        shp = list(v["shape"])
        stack = next(((a, b) for a, b in _AF2_STACKS if mod.startswith(a)), None)
        n = 0
        if stack:
            mod = mod[len(stack[0]):]
            n, shp = shp[0], shp[1:]
        sub = mod.split("/")
        leaf = None
        if stack and param in ("weights", "bias") and sub[-1] in _AF2_LEAF:   # e.g. tri_mul …/left_projection//weights
            leaf, mod = _AF2_LEAF[sub[-1]], "/".join(sub[:-1])
        tmod = next((t + mod[len(h):].replace("/", ".") for h, t in _AF2_MOD if mod == h or mod.startswith(h + "/")),
                    None)
        if tmod is None:
            if not stack:
                raise KeyError(key)
            tmod = mod.replace("/", ".")
        if stack:
            for h, t in _AF2_MOD:                         # module names inside the stack
                if mod == h or mod.startswith(h + "/"):
                    tmod = t + mod[len(h):].replace("/", ".")
                    break
        if param in ("weights", "query_w", "key_w", "value_w", "gating_w", "output_w", "feat_2d_weights"):
            pname = (leaf or _AF2_LEAF.get(param, "")) + ".weight" if (leaf or param != "weights") else "weight"
        elif param in ("bias", "gating_b", "output_b"):
            pname = (leaf or {"gating_b": "linear_g", "output_b": "linear_o"}.get(param, "")) + ".bias" \
                if (leaf or param != "bias") else "bias"
        else:                                             # scale / offset / trainable_point_weights / …
            pname = param
        name = f"{tmod}.{pname}".replace("..", ".").strip(".")
        for a, b in _AF2_RENAME:
            name = re.sub(a, b, name)
        tshape = _af2_torch_shape(param, shp)
        assert math.prod(tshape) == math.prod(shp), key
        if stack:
            for i in range(n):
                out[f"{stack[1]}{i}.{name}"] = {"shape": tshape, "dtype": "F32"}
        else:
            assert name not in out, (key, name)
            out[name] = {"shape": tshape, "dtype": "F32"}
    return out


def main():
    name = sys.argv[1]
    r = RELEASES[name]
    if r.get("npz"):
        raw = json.loads((ROOT / "local" / "hf_cache" / "npz" / f"{r['cache']}.json").read_text())
        ten = af2_tensors(raw)
        raw["size"] = raw["member_bytes"]
    else:
        raw = json.loads((CACHE / f"{r['cache']}.json").read_text())
        ten = {k[len(r["strip"]):] if k.startswith(r["strip"]) else k: v for k, v in raw["tensors"].items()}
    stacks: dict[str, int] = {}
    for k in ten:
        m = STACK.match(k)
        if m:
            stacks[m.group(1)] = max(stacks.get(m.group(1), 0), int(m.group(2)) + 1)
    shapes, roles = {}, {}
    params = nbytes = 0
    for k, v in ten.items():
        n = math.prod(v["shape"]) if v["shape"] else 1
        m = STACK.match(k)
        if not m or int(m.group(2)) in (0, stacks[m.group(1)] - 1):
            shapes[k] = v["shape"]
        if v["dtype"] not in FMT:
            roles.setdefault("buffer", {}).setdefault(v["dtype"].lower(), {"params": 0, "bytes": 0})
            roles["buffer"][v["dtype"].lower()]["bytes"] += n * BYTES[v["dtype"]]
            continue
        role = r["role"](k)
        e = roles.setdefault(role, {}).setdefault(FMT[v["dtype"]], {"params": 0, "bytes": 0})
        e["params"] += n
        e["bytes"] += n * BYTES[v["dtype"]]
        params += n
        nbytes += n * BYTES[v["dtype"]]
    cfg = dict(r.get("config", {}))
    if r.get("config_url"):
        with urllib.request.urlopen(r["config_url"], timeout=60) as f:
            cfg = json.loads(f.read())
    if r.get("hparams"):
        hp = json.loads((CACHE / f"{r['cache']}_hp.json").read_text())
        cfg["hyper_parameters"] = {k: hp.get(k) for k in r["hparams"]}
    out = {"repo": r["repo"], "domain": "protein", "fetched": datetime.date.today().isoformat(),
           "n_tensors": len(ten), "params_llm": params, "bytes_llm": nbytes, "params_mtp": 0, "params_vision": 0,
           "config": cfg, "roles": roles, "stacks": stacks, "shapes": shapes,
           "source": {"url": r["url"], "file_bytes": raw["size"], "format": r.get("format", "pytorch zip checkpoint "
                      "(data.pkl only, read with HTTP range requests; no weights downloaded)")}}
    p = OUT / (r["repo"].replace(":", "_").replace("/", "__") + ".json")
    p.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(p.name, f"{params / 1e6:.2f}M params", {k: list(v) for k, v in roles.items()}, stacks)


if __name__ == "__main__":
    main()
