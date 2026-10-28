"""L1d — protein structure models on core v2 (0.43): ESMFold, OpenFold, Boltz-1, Protenix.

Built from the released checkpoint headers (tensor names + shapes, read without downloading the weights; see
scripts/fetch_torch_ckpt.py / summarize_ckpt.py).  Every 2-D weight of the checkpoint becomes one ``Linear`` whose
GEMM rows are the grid its module runs on (residues, N² pairs, S·N MSA rows, T·N² template pairs, atoms, local atom
pairs); everything else (norms, biases, scalar tables) is storage-only ``misc``.  The parameter total therefore equals
the checkpoint total by construction, and the row assignment is the modelling claim (tested).

Activation × activation work comes from the module types found in each block (``PairCore``): triangle
multiplicative updates (outgoing / incoming), triangle attention (start / end), MSA row attention with pair bias,
MSA column attention (global for AF2's extra-MSA stack), outer-product mean, pair-weighted averaging (AF3 MSA module),
sequence attention with pair bias (single track, diffusion transformer, IPA), and windowed atom attention.

Per request: the trunk runs ``recycles`` passes, the diffusion module ``steps`` × ``samples`` (samples batched; the
pair conditioning is shared across samples), the confidence head once per sample, output heads once.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from .model import AttnCore, Ffn, Layer, Linear, ModelSpec, PairCore, RoleFormat

# ------------------------------------------------------------------ rows of a weight GEMM (first match wins)
_ATOM_PAIR = re.compile(r"(pair_bias_attn\.proj_z|attention_pair_bias\.linear_nobias_z|embed_atompair|"
                        r"p_mlp|small_mlp|linear_no_bias_(d|invd|v)\b)")
# 0.61.3: projections that feed the atom-pair blocks but run on a coarser grid before the broadcast — the atom
# conditioning c → pair (Boltz c_to_p_trans_q/k, Protenix linear_no_bias_cl/cm: one row per atom) and the token pair
# z → atom pair (Boltz z_to_p_trans, Protenix linear_no_bias_z: one row per token pair); 0.43–0.61.2 ran them on the
# 32 × 128 atom-pair windows (up to 128× too many rows)
_ATOM_FROM_C = re.compile(r"(c_to_p_trans|linear_no_bias_(cl|cm)\b)")
_ATOM_FROM_Z = re.compile(r"(z_to_p_trans|linear_no_bias_z\b)")
_PAIR = re.compile(r"(tri_mul|tri_att|pair_transition|transition_z|z_transition|outer_product_mean(_msa)?\.(linear_out|proj_o)|"
                   r"relpe|rel_pos|relative_position|linear_relpos|recycling_embedder\.linear|z_recycle|z_cycle|"
                   r"zinit|z_init|token_bond|distogram|linear_no_bias_pae|linear_no_bias_pde|aux_heads\.tm|"
                   r"pairwise_conditioner|diffusion_conditioning\.linear_no_bias_z|linear_no_bias_d\b|"
                   r"linear_no_bias_d_wo|sequence_to_pair\.o_proj|pair_to_sequence|trunk2sm_z|mlp_pair|ptm_head|"
                   r"template_pointwise_att\.mha\.linear_(q|o)|template_embedder\.linear_no_bias_u|"
                   r"(msa_att_row|pair_weighted_averaging)\.(linear_z|proj_z|linear_no_bias_z)|"
                   r"(attention_pair_bias|attention|pair_bias_attn)\.(linear_nobias_z|proj_z)|ipa\.linear_b)")
_TMPL = re.compile(r"(template_pair_stack|template_pair_embedder|template_pointwise_att\.mha\.linear_(k|v)|"
                   r"template_embedder\.linear_no_bias_(z|a)|template_embedder\.pairformer)")
_MSA = re.compile(r"(msa_att_row|msa_att_col|msa_transition|transition_m\b|outer_product_mean(_msa)?\.(linear_[12]|proj_[ab])|"
                  r"pair_weighted_averaging|linear_msa_m|msa_proj|msa_module\.linear_no_bias_m|masked_msa|"
                  r"extra_msa_embedder)")


def rows_of(name: str, stack: str) -> str:
    atom = "atom_attention_" in name or "atom_encoder" in name
    if atom:
        if _ATOM_FROM_Z.search(name):
            return "pair"
        if _ATOM_FROM_C.search(name):
            return "atom"
        return "apair" if _ATOM_PAIR.search(name) else "atom"
    if "global_attention.linear_q" in name:     # 0.61.3: global column attention — q from the row-mean (N rows)
        return "res"
    if "template_angle_embedder" in name:      # 0.61.3: one row per template residue (T × N)
        return "tres"
    if _TMPL.search(name):
        return "tmpl"
    if _PAIR.search(name):
        return "pair"
    if _MSA.search(name):
        return "xmsa" if ("extra_msa" in name) else "msa"
    return "res"


# ------------------------------------------------------------------ module cores of one block
def _out(sh: dict, *names: str) -> int:
    for n in names:
        if n in sh:
            return sh[n][0]
    return 0


def _in(sh: dict, *names: str) -> int:
    for n in names:
        if n in sh and len(sh[n]) >= 2:
            return math.prod(sh[n][1:])
    return 0


def cores_of(sh: dict, prefix: str, ctx: dict) -> tuple[PairCore, ...]:
    """Module cores of the block whose tensors start with ``prefix`` (detected from the module names)."""
    keys = [k[len(prefix):] for k in sh if k.startswith(prefix)]
    mods = sorted({m.group(0) for k in keys for m in [re.match(
        r"(.*?(tri_mul_out|tri_mul_in|tri_att_start|tri_att_end|msa_att_row|msa_att_col|outer_product_mean(_msa)?|"
        r"(msa_)?pair_weighted_averaging|attention_pair_bias|pair_bias_attn|seq_attention|ipa|template_pointwise_att|"
        r"(?<![a-z_])attention(?=\.proj_)))", k)] if m})
    out = []
    xm = "extra_msa" in prefix
    for mod in mods:
        p = prefix + mod + "."
        leaf = mod.rsplit(".", 1)[-1]
        if leaf.startswith("tri_mul"):
            c = _out(sh, p + "linear_a_p.weight") or _out(sh, p + "p_in.weight") // 2
            out.append(PairCore("trimul", dim=c, over=ctx.get("grid", "pair"),
                                width=_in(sh, p + "linear_a_p.weight", p + "p_in.weight")))
        elif leaf.startswith("tri_att"):
            h = _out(sh, p + "linear.weight")
            out.append(PairCore("tri_att", heads=h, dim=_out(sh, p + "mha.linear_q.weight") // h,
                                over=ctx.get("grid", "pair"), width=_in(sh, p + "mha.linear_q.weight")))
        elif leaf == "msa_att_row":
            h = _out(sh, p + "linear_z.weight")
            out.append(PairCore("row_att", heads=h, dim=_out(sh, p + "mha.linear_q.weight") // h,
                                over="xmsa" if xm else "msa", width=_in(sh, p + "mha.linear_q.weight")))
        elif leaf == "msa_att_col":
            if p + "global_attention.linear_q.weight" in sh:
                c = _out(sh, p + "global_attention.linear_k.weight")
                out.append(PairCore("col_att", heads=_out(sh, p + "global_attention.linear_q.weight") // c, dim=c,
                                    over="xmsa" if xm else "msa", glob=True,
                                    width=_in(sh, p + "global_attention.linear_q.weight")))
            else:
                q = _out(sh, p + "_msa_att.mha.linear_q.weight", p + "mha.linear_q.weight")
                c = ctx.get("c_hidden_msa_att", 32)
                out.append(PairCore("col_att", heads=q // c, dim=c, over="xmsa" if xm else "msa",
                                    width=_in(sh, p + "_msa_att.mha.linear_q.weight", p + "mha.linear_q.weight")))
        elif leaf in ("outer_product_mean", "outer_product_mean_msa"):
            out.append(PairCore("opm", dim=_out(sh, p + "linear_1.weight", p + "proj_a.weight"),
                                dim2=_out(sh, p + "linear_2.weight", p + "proj_b.weight"), over="xmsa" if xm else "msa",
                                width=_in(sh, p + "linear_1.weight", p + "proj_a.weight")))
        elif leaf in ("pair_weighted_averaging", "msa_pair_weighted_averaging"):
            h = _out(sh, p + "linear_no_bias_z.weight", p + "proj_z.weight")
            out.append(PairCore("pwa", heads=h,
                                dim=_out(sh, p + "linear_no_bias_mv.weight", p + "proj_m.weight") // h,
                                width=_in(sh, p + "linear_no_bias_mv.weight", p + "proj_m.weight")))
        elif leaf in ("attention_pair_bias", "pair_bias_attn", "attention"):
            h = _out(sh, p + "linear_nobias_z.weight", p + "proj_z.1.weight", p + "proj_z.weight")
            q = _out(sh, p + "attention.linear_q.weight", p + "proj_q.weight")
            if ctx.get("atom"):
                out.append(PairCore("local_att", heads=h, dim=q // h, win=ctx["win"]))
            else:
                out.append(PairCore("seq_att", heads=h, dim=q // h))
        elif leaf == "seq_attention":            # ESMFold trunk: 1024-wide, bias heads from pair_to_sequence
            h = _out(sh, prefix + "pair_to_sequence.linear.weight")
            out.append(PairCore("seq_att", heads=h, dim=_out(sh, p + "o_proj.weight") // h))
        elif leaf == "ipa":                      # invariant point attention: scalar + point qk, scalar + point + pair v
            h = _out(sh, p + "linear_b.weight")
            sq = _out(sh, p + "linear_q.weight") // h
            pq = _out(sh, p + "linear_q_points.weight") // h          # points × 3
            pkv = _out(sh, p + "linear_kv_points.weight") // h        # (qk + v points) × 3
            cz = sh[p + "linear_b.weight"][1]
            out.append(PairCore("seq_att", heads=h, dim=sq + pq, v_dim=sq + (pkv - pq) + cz))
        elif leaf == "template_pointwise_att":
            q = _out(sh, p + "mha.linear_q.weight")
            out.append(PairCore("pt_att", heads=4, dim=q // 4))
    return tuple(out)


# ------------------------------------------------------------------ groups → layers
@dataclass(frozen=True)
class Group:
    prefix: str             # tensor prefix (stack: '<stack>.' with block index after it)
    stack: str              # display label
    repeat: str = ""        # "" once | recycle | diff
    iters: int = 1
    samples: str = ""
    blocks: bool = False    # True: prefix + '<i>.' blocks (count from release ``stacks``)
    atom: bool = False      # atom transformer (windowed attention)
    grid: str = "pair"      # grid of the pair cores (template stacks: tmpl)
    skip: bool = False      # stored but not run at inference (template embedder disabled)


def _block_layer(sh: dict, pre: str, g: Group, ctx: dict, claimed: set) -> Layer:
    lins, misc, mine = [], 0, {}
    for k, shp in sh.items():
        if not k.startswith(pre) or k in claimed:
            continue
        claimed.add(k)
        mine[k] = shp
        n = math.prod(shp) if shp else 1
        two_d = len(shp) >= 2 and k.endswith(".weight") and shp[0] > 1 and math.prod(shp[1:]) > 1
        if two_d and not g.skip and not k.endswith("embedding.weight") and "recycle_disto" not in k \
                and "pairwise_positional_embedding" not in k and not k.startswith("embedding."):
            short = k[len(pre):-len(".weight")]
            rows = "tmpl" if g.grid == "tmpl" else rows_of(k, g.stack)
            lins.append(Linear(short, math.prod(shp[1:]), shp[0], "col", rows=rows, role="pair"))
        else:
            misc += n
    cores = () if g.skip else cores_of(mine, pre, {**ctx, "atom": g.atom, "grid": g.grid})
    return Layer((), AttnCore("none"), Ffn("none"), (), misc, pair_linears=tuple(lins), pair_cores=cores,
                 repeat=g.repeat, iters=g.iters, samples=g.samples, misc_role="pair", stack=g.stack)


def layers_from(rel: dict, groups: list[Group], ctx: dict, strip: str = "") -> tuple[list[Layer], set]:
    sh = rel["shapes"]
    stacks = rel.get("stacks", {})
    claimed: set = set()
    layers: list[Layer] = []
    for g in groups:
        if g.blocks:
            name = g.prefix.rstrip(".")
            n = stacks[name]
            first = _block_layer(sh, f"{g.prefix}0.", g, ctx, claimed)
            last = first
            if n > 1:
                last = _block_layer(sh, f"{g.prefix}{n - 1}.", g, ctx, claimed)
                if last != first:       # heterogeneous last block (AF3 MSA module: no MSA update in the last block)
                    ctx.setdefault("_hetero", []).append(name)
            layers += [first] * (n - 1) + [last]
            # mark the middle blocks' tensors as claimed (not in shapes; counted through the replication)
        else:
            layers.append(_block_layer(sh, g.prefix, g, ctx, claimed))
    left = [k for k in sh if k not in claimed and not any(k.startswith(p) for p in ctx.get("not_modelled", ()))]
    if left:
        raise ValueError(f"unassigned tensors: {left[:8]}")
    return layers, claimed


# ------------------------------------------------------------------ builders
def _formats(rel: dict) -> tuple[tuple[str, RoleFormat], ...]:
    out = {}
    for role, v in rel.get("roles", {}).items():
        if role == "buffer":
            continue
        fmt, _ = max(v.items(), key=lambda kv: kv[1]["params"])
        p = sum(x["params"] for x in v.values())
        b = sum(x["bytes"] for x in v.values())
        if p:
            out[role] = RoleFormat(fmt if fmt in ("fp32", "bf16", "fp16") else "bf16", 8.0 * b / p)
    for role in ("attn", "mlp", "io", "embed"):
        out.setdefault(role, out["pair"])
    return tuple(sorted(out.items()))


_PREC = {"esmfold": "参考实现（HF EsmForProteinFolding）默认 fp32 计算",
         "openfold": "参考实现 fp32（matmul tf32）",
         "alphafold2": "参考实现 JAX 单体模型 fp32（monomer 配置不开 bfloat16）",
         "boltz1": "参考实现 precision=32（fp32）",
         "protenix": "参考实现以 bf16 为默认 dtype"}


def _spec(model_id, hf_id, rel, arch, hidden, layers, wl, notes, reasons, key, extra_params=0) -> ModelSpec:
    fm = _formats(rel)
    spec = ModelSpec(id=model_id, hf_id=hf_id, arch=arch, hidden=hidden, vocab=0, layers=tuple(layers), formats=fm,
                     act_fmt="bf16", kv_fmt="bf16", release_params=rel["params_llm"], release_bytes=rel["bytes_llm"],
                     quantized_release=False, domain="protein", kv_cache=False, io_misc_params=extra_params,
                     final_norm_params=0, adaln=False, workload=wl)
    notes = list(notes) + [
        "发布权重为 fp32（4 B / 参数）" + ("，ESM-2 语言模型部分为 fp16" if key == "esmfold" else "") +
        f"；激活按 bf16「假设」（{_PREC[key]}）——fp32 激活可用激活 dtype what-if（UI「激活 dtype」/ CLI --act fp32）评估："
        "激活存储、DRAM 流式与 SRAM 端口流量按 4 B，计算仍在 bf16 阵列上并计转换开销；芯片无 fp32 MAC → 逐 GEMM 转换为 bf16，开销计入向量单元",
        "多卡：DAP（FastFold 动态轴并行，布局的 SP 维）把 pair / MSA / 模板网格沿一个残基轴切到 DAP 张卡，"
        "单一 / 原子轨道（结构模块、扩散 transformer、原子注意力）每卡重复计算；通信按 FastFold DAP 内核（三角乘法 / "
        "外积均值的 all-gather、pair 偏置 all-gather、行 ↔ 列 all-to-all）计，见 docs/MODEL.md §12；"
        "DAP > 1 时扩散样本默认分到各卡（每卡 ⌈样本 / DAP⌉ 条，精确、无额外通信；workload.sample_split = false 时每卡重复）；"
        "pair 的 TP 未建模（布局 PP × DP × DAP，DP = 多条序列并行）"]
    pc = (spec.release_params - spec.params()) / spec.release_params
    if abs(pc) > 0.001:
        reasons = list(reasons) + [f"参数与发布相差 {pc:+.2%}"]
    return replace(spec, notes=tuple(notes), coverage_reasons=tuple(reasons),
                   coverage="partial" if reasons else "full")


def _apply_defaults(wl, **kw):
    return replace(wl, **{k: v for k, v in kw.items() if not getattr(wl, k)})


def _need_shape(sh: dict, name: str, shape: list) -> None:
    if list(sh.get(name) or []) != shape:
        raise ValueError(f"esmfold {name} shape is not {shape}")


def _lin_of(layer: Layer, name: str) -> Linear:
    found = [ln for ln in layer.pair_linears if ln.name == name]
    if len(found) != 1:
        raise ValueError(f"esmfold {name} is missing")
    return found[0]


def _expect_lin(ln: Linear, k: int, n: int, rows: str) -> None:
    if (ln.k, ln.n, ln.rows) != (k, n, rows):
        raise ValueError(f"esmfold {ln.name} shape is not [{k}, {n}] on {rows}")


def _require_esmfold_lm(c: dict, sh: dict) -> tuple[int, int, int, int, int]:
    """Published ESMFold language tower (transformers modeling_esm.py + modeling_esmfold.py).

    The encoder is the ESM-2 stack (pre-LN, erf-GELU, query scale, RoPE). Its width is
    ``hidden_size``. The folding trunk's ``sequence_state_dim`` is a different width.
    """
    if c.get("model_type") != "esm":
        raise ValueError("esmfold model_type is not esm")
    if c.get("is_folding_model") is not True:
        raise ValueError("esmfold is_folding_model is not true")
    if c.get("hidden_act") != "gelu":
        raise ValueError("esmfold hidden_act is not gelu")
    if c.get("position_embedding_type") != "rotary":
        raise ValueError("esmfold position_embedding_type is not rotary")
    if c.get("emb_layer_norm_before") is not False:
        raise ValueError("esmfold emb_layer_norm_before is not false")
    if c.get("attention_probs_dropout_prob") not in (0, 0.0) or c.get("hidden_dropout_prob") not in (0, 0.0):
        raise ValueError("esmfold dropout is not 0")
    ec = c.get("esmfold_config") or {}
    if ec.get("esm_type") != "esm2_3B":
        raise ValueError("esmfold esm_type is not esm2_3B")
    if ec.get("bypass_lm"):
        raise ValueError("esmfold bypass_lm is not false")
    if ec.get("use_esm_attn_map"):
        raise ValueError("esmfold use_esm_attn_map is not false")
    if ec.get("fp16_esm"):
        raise ValueError("esmfold fp16_esm is not false")
    trunk = ec.get("trunk") or {}
    d = int(c["hidden_size"])
    H = int(c["num_attention_heads"])
    f = int(c["intermediate_size"])
    L = int(c["num_hidden_layers"])
    cs = int(trunk.get("sequence_state_dim") or 0)
    if d <= 0 or H <= 0 or f <= 0 or L <= 0 or d % H or cs <= 0:
        raise ValueError("esmfold language width does not divide")
    _need_shape(sh, "esm.encoder.emb_layer_norm_after.weight", [d])
    _need_shape(sh, "esm.encoder.layer.0.attention.LayerNorm.weight", [d])
    _need_shape(sh, "esm.encoder.layer.0.LayerNorm.weight", [d])
    _need_shape(sh, f"esm.encoder.layer.{L - 1}.attention.LayerNorm.weight", [d])
    _need_shape(sh, f"esm.encoder.layer.{L - 1}.LayerNorm.weight", [d])
    _need_shape(sh, "esm.encoder.layer.0.intermediate.dense.weight", [f, d])
    _need_shape(sh, "esm_s_mlp.0.weight", [d])
    _need_shape(sh, "esm_s_mlp.1.weight", [cs, d])
    _need_shape(sh, "esm_s_mlp.3.weight", [cs, cs])
    if any("q_norm" in n or "k_norm" in n for n in sh):
        raise ValueError("esmfold has q_norm")
    if any(n.startswith("lm_head.dense") or n.startswith("lm_head.decoder") or n.startswith("lm_head.layer_norm")
           for n in sh):
        raise ValueError("esmfold has an ESM-2 MLM head")
    return d, H, f, L, cs


def _require_esmfold_fold(c: dict, sh: dict, stacks: dict, cs: int) -> tuple[int, ...]:
    """Published trunk, structure module, recycle norms, and lddt head.

    Returns pairwise width, structure sequence width, structure pair width, resnet width,
    structure-module iterations, lddt hidden, sequence heads, pair heads, and the IPA
    output-projection K (heads * (pair + ipa_dim + v_points * 4)).
    """
    ec = c.get("esmfold_config") or {}
    trunk = ec.get("trunk") or {}
    sm = trunk.get("structure_module") or {}
    cz = int(trunk.get("pairwise_state_dim") or 0)
    shw = int(trunk.get("sequence_head_width") or 0)
    phw = int(trunk.get("pairwise_head_width") or 0)
    nblocks = int(trunk.get("num_blocks") or 0)
    if trunk.get("dropout") not in (0, 0.0) or trunk.get("layer_drop") not in (0, 0.0):
        raise ValueError("esmfold trunk dropout is not 0")
    if sm.get("dropout_rate") != 0.1:
        raise ValueError("esmfold structure dropout_rate is not 0.1")
    if int(sm.get("num_transition_layers") or 0) != 1:
        raise ValueError("esmfold num_transition_layers is not 1")
    if int(sm.get("num_resnet_blocks") or 0) != 2:
        raise ValueError("esmfold num_resnet_blocks is not 2")
    if int(sm.get("num_angles") or 0) != 7:
        raise ValueError("esmfold num_angles is not 7")
    sm_s = int(sm.get("sequence_dim") or 0)
    sm_z = int(sm.get("pairwise_dim") or 0)
    resnet = int(sm.get("resnet_dim") or 0)
    ipa_dim = int(sm.get("ipa_dim") or 0)
    n_qk = int(sm.get("num_qk_points") or 0)
    n_v = int(sm.get("num_v_points") or 0)
    n_heads = int(sm.get("num_heads_ipa") or 0)
    sm_blocks = int(sm.get("num_blocks") or 0)
    hid = int(ec.get("lddt_head_hid_dim") or 0)
    if cs <= 0 or cz <= 0 or shw <= 0 or phw <= 0 or cs % shw or cz % phw:
        raise ValueError("esmfold trunk width does not divide")
    if nblocks < 1 or nblocks != stacks.get("trunk.blocks"):
        raise ValueError("esmfold trunk block count does not match")
    if stacks.get("trunk.structure_module.angle_resnet.layers") != 2:
        raise ValueError("esmfold angle resnet stack is not 2")
    if stacks.get("trunk.structure_module.transition.layers") != 1:
        raise ValueError("esmfold transition stack is not 1")
    if sm_s <= 0 or sm_z <= 0 or resnet <= 0 or sm_blocks < 1 or hid <= 0:
        raise ValueError("esmfold structure width is missing")
    if ipa_dim <= 0 or n_qk <= 0 or n_v <= 0 or n_heads <= 0:
        raise ValueError("esmfold IPA width is missing")
    seq_heads = cs // shw
    pair_heads = cz // phw
    for i in (0, nblocks - 1):
        p = f"trunk.blocks.{i}."
        _need_shape(sh, p + "layernorm_1.weight", [cs])
        _need_shape(sh, p + "mlp_seq.mlp.0.weight", [cs])
        _need_shape(sh, p + "mlp_seq.mlp.1.weight", [4 * cs, cs])
        _need_shape(sh, p + "mlp_pair.mlp.0.weight", [cz])
        _need_shape(sh, p + "mlp_pair.mlp.1.weight", [4 * cz, cz])
        _need_shape(sh, p + "pair_to_sequence.layernorm.weight", [cz])
        _need_shape(sh, p + "sequence_to_pair.layernorm.weight", [cs])
        for side in ("in", "out"):
            _need_shape(sh, p + f"tri_mul_in.layer_norm_{side}.weight", [cz])
            _need_shape(sh, p + f"tri_mul_out.layer_norm_{side}.weight", [cz])
        _need_shape(sh, p + "tri_att_start.layer_norm.weight", [cz])
        _need_shape(sh, p + "tri_att_end.layer_norm.weight", [cz])
    _need_shape(sh, "trunk.recycle_s_norm.weight", [cs])
    _need_shape(sh, "trunk.recycle_z_norm.weight", [cz])
    _need_shape(sh, "trunk.structure_module.layer_norm_s.weight", [sm_s])
    _need_shape(sh, "trunk.structure_module.layer_norm_z.weight", [sm_z])
    _need_shape(sh, "trunk.structure_module.layer_norm_ipa.weight", [sm_s])
    _need_shape(sh, "trunk.structure_module.transition.layer_norm.weight", [sm_s])
    _need_shape(sh, "trunk.structure_module.linear_in.weight", [sm_s, sm_s])
    _need_shape(sh, "lddt_head.0.weight", [sm_s])
    _need_shape(sh, "lddt_head.1.weight", [hid, sm_s])
    _need_shape(sh, "lddt_head.2.weight", [hid, hid])
    _need_shape(sh, "lddt_head.3.weight", [37 * 50, hid])
    if any(n.startswith("lddt_head.4") for n in sh):
        raise ValueError("esmfold lddt head has a fourth linear")
    ipa_k = n_heads * (sm_z + ipa_dim + n_v * 4)
    return cz, sm_s, sm_z, resnet, sm_blocks, hid, seq_heads, pair_heads, ipa_k


def build_esmfold(model_id: str, hf_id: str, rel: dict, wl) -> ModelSpec:
    from .domain import _gelu_ffn, _mha
    from .ir import ESMFOLD_SM_LINEARS, ESMFOLD_TRUNK_LINEARS
    c, sh = rel["config"], rel["shapes"]
    d, H, f, L, cs = _require_esmfold_lm(c, sh)
    cz, sm_s, sm_z, resnet, sm_blocks, hid, seq_heads, pair_heads, ipa_k = _require_esmfold_fold(
        c, sh, rel["stacks"], cs)
    t = c["esmfold_config"]["trunk"]
    alin, core = _mha(d, H, rope=d // H)
    # Same attention as the ESM-2 encoder: RoPE, query scaled by head_dim**-0.5, no QK-norm.
    core = replace(core, qk_norm=False, q_scale=True)
    ffn, flin = _gelu_ffn(d, f)
    # misc: q/k/v/o biases, 2 layer norms, FFN biases + the rotary inv_freq buffer stored (fp32) in the checkpoint
    esm_layer = Layer(alin, core, ffn, flin, 4 * d + 2 * d + f + d + 2 * d + (d // H) // 2, stack="esm",
                      vec_kind="esm2", vec_width=d)
    groups = [Group("trunk.blocks.", "trunk", "recycle", blocks=True),
              Group("trunk.structure_module.", "structure", "recycle", iters=t["structure_module"]["num_blocks"]),
              Group("trunk.", "recycle", "recycle"),
              Group("esm_s_", "esm_proj"), Group("embedding.", "esm_proj"),
              Group("distogram_head.", "heads"), Group("ptm_head.", "heads"), Group("lddt_head.", "heads"),
              Group("lm_head.", "heads"),
              Group("esm.", "esm_io", skip=True)]
    sh2 = {k: v for k, v in sh.items() if not k.startswith("esm.encoder.layer.") and k not in ("af2_to_esm",)
           and not k.endswith("position_ids")}
    pl, _ = layers_from({**rel, "shapes": sh2}, groups, {})
    tagged = []
    for layer in pl:
        names = [ln.name for ln in layer.pair_linears]
        if names == ["mlp.1", "mlp.3"] and layer.stack == "esm_proj":
            if layer.pair_linears[0].k != d or layer.pair_linears[0].n != cs or layer.pair_linears[1].k != cs \
                    or layer.pair_linears[1].n != cs:
                raise ValueError("esmfold projection GEMM shape is not language width to trunk width")
            tagged.append(replace(layer, vec_kind="esm_proj", vec_width=d))
        elif any(n.startswith("mlp.") for n in names) and layer.stack == "esm_proj":
            raise ValueError("esmfold esm_s_mlp GEMMs are not the published pair")
        elif layer.stack == "trunk":
            if {ln.name for ln in layer.pair_linears} != ESMFOLD_TRUNK_LINEARS:
                raise ValueError("esmfold trunk block linears are not the published set")
            _expect_lin(_lin_of(layer, "pair_to_sequence.linear"), cz, seq_heads, "pair")
            _expect_lin(_lin_of(layer, "seq_attention.proj"), cs, 3 * cs, "res")
            _expect_lin(_lin_of(layer, "mlp_seq.mlp.1"), cs, 4 * cs, "res")
            _expect_lin(_lin_of(layer, "mlp_seq.mlp.3"), 4 * cs, cs, "res")
            _expect_lin(_lin_of(layer, "sequence_to_pair.proj"), cs, cz, "res")
            for side in ("in", "out"):
                _expect_lin(_lin_of(layer, f"tri_mul_{side}.linear_a_g"), cz, cz, "pair")
                _expect_lin(_lin_of(layer, f"tri_mul_{side}.linear_z"), cz, cz, "pair")
            _expect_lin(_lin_of(layer, "tri_att_start.linear"), cz, pair_heads, "pair")
            _expect_lin(_lin_of(layer, "tri_att_end.linear"), cz, pair_heads, "pair")
            _expect_lin(_lin_of(layer, "mlp_pair.mlp.1"), cz, 4 * cz, "pair")
            _expect_lin(_lin_of(layer, "mlp_pair.mlp.3"), 4 * cz, cz, "pair")
            tagged.append(replace(layer, vec_kind="esm_trunk"))
        elif layer.stack == "structure":
            by_name = {ln.name: ln for ln in layer.pair_linears}
            if "linear_in" not in by_name or set(by_name) - {"linear_in"} != ESMFOLD_SM_LINEARS:
                raise ValueError("esmfold structure-module linears are not the published set")
            if layer.iters != sm_blocks or layer.repeat != "recycle":
                raise ValueError("esmfold structure-module repeat is not the published loop")
            lin_in = by_name["linear_in"]
            _expect_lin(lin_in, sm_s, sm_s, "res")
            _expect_lin(by_name["ipa.linear_out"], ipa_k, sm_s, "res")
            for nm in ("transition.layers.0.linear_1", "transition.layers.0.linear_2",
                       "transition.layers.0.linear_3"):
                _expect_lin(by_name[nm], sm_s, sm_s, "res")
            _expect_lin(by_name["angle_resnet.linear_initial"], sm_s, resnet, "res")
            _expect_lin(by_name["angle_resnet.linear_in"], sm_s, resnet, "res")
            for blk in (0, 1):
                _expect_lin(by_name[f"angle_resnet.layers.{blk}.linear_1"], resnet, resnet, "res")
                _expect_lin(by_name[f"angle_resnet.layers.{blk}.linear_2"], resnet, resnet, "res")
            _expect_lin(by_name["angle_resnet.linear_out"], resnet, 14, "res")
            # linear_in runs once per structure-module call. The block loop does not repeat it.
            # Its bias stays in the loop layer's misc so the parameter total is unchanged.
            pre = replace(layer, pair_linears=(lin_in,), pair_cores=(), misc_params=0,
                          vec_kind="esm_sm_pre", vec_width=sm_z, iters=1)
            rest = tuple(ln for ln in layer.pair_linears if ln.name != "linear_in")
            tagged.append(pre)
            tagged.append(replace(layer, pair_linears=rest, vec_kind="esm_sm"))
        elif layer.stack == "recycle":
            if names != ["trunk2sm_s", "trunk2sm_z"]:
                raise ValueError("esmfold recycle projections are not trunk2sm_s then trunk2sm_z")
            _expect_lin(layer.pair_linears[0], cs, sm_s, "res")
            _expect_lin(layer.pair_linears[1], cz, sm_z, "pair")
            tagged.append(replace(layer, vec_kind="esm_recycle"))
        elif layer.stack == "heads" and names == ["1", "2", "3"]:
            if layer.iters != 1 or layer.repeat != "":
                raise ValueError("esmfold lddt head is not a single pass")
            _expect_lin(layer.pair_linears[0], sm_s, hid, "res")
            _expect_lin(layer.pair_linears[1], hid, hid, "res")
            _expect_lin(layer.pair_linears[2], hid, 37 * 50, "res")
            tagged.append(replace(layer, vec_kind="esm_lddt"))
        else:
            tagged.append(layer)
    pl = tagged
    kinds = {}
    for layer in pl:
        kinds[layer.vec_kind] = kinds.get(layer.vec_kind, 0) + 1
    if kinds.get("esm_trunk") != 48 or kinds.get("esm_sm") != 1 or kinds.get("esm_sm_pre") != 1 \
            or kinds.get("esm_recycle") != 1 or kinds.get("esm_lddt") != 1 or kinds.get("esm_proj") != 1:
        raise ValueError("esmfold vector layers are not the published set")
    max_pos = c.get("max_position_embeddings", 1026)
    # 0.61.2: num_recycles=None runs the loop max_recycles times in total (the +1 for the first pass applies only to an
    # explicit num_recycles; esm/esmfold/v1/trunk.py and transformers EsmFoldingTrunk) — 0.43–0.61.1 ran 5 passes
    wl = _apply_defaults(wl, recycles=t["max_recycles"], pair_dim=t["pairwise_state_dim"], max_seq=max_pos - 4,
                         special_tokens=2)
    notes = [f"ESM-2 3B 语言模型（{L} 层，fp16 发布，宽 {d}）每请求一次前向（序列 + <cls>/<eos>），其 37 层隐状态加权求和后进入折叠主干",
             f"折叠主干 {rel['stacks']['trunk.blocks']} 块（序列 {cs} 维 + pair {t['pairwise_state_dim']} 维；三角乘法 ×2、三角注意力 ×2、"
             "带 pair 偏置的序列注意力），每次 recycle 全部重跑；默认主干前向 "
             f"{t['max_recycles']} 次（esm / HF 参考实现 num_recycles=None → 循环 max_recycles = {t['max_recycles']} 次，"
             "含首次前向；显式传 num_recycles = k 时才是 k + 1 次）",
             f"结构模块（IPA，{t['structure_module']['num_blocks']} 次迭代共享权重）每次 recycle 后运行；IPA 按 qk = 标量 + 点坐标、"
             "v = 标量 + 点 + pair 值的注意力计，刚体帧更新 / 扭转角 / 原子坐标重建的向量运算未计",
             "无 MSA、无模板（单序列）；chunk_size 128 只降低峰值显存，不改变计算量",
             "0.439：语言层的 LayerNorm 与 erf-GELU 按 ESM-2 宽度计，不用折叠主干的 hidden。"
             "最后一层之后的 emb_layer_norm_after 计一次。esm_s_mlp 在去掉 <cls>/<eos> 的残基行上计 LayerNorm 和 ReLU。"
             "没有 ESM-2 的 MLM 头。token_dropout 不单列。"
             "0.440：每个主干块计 11 次 LayerNorm 和 2 次 ReLU。recycle 的两条 LayerNorm 每次 recycle 计一次，不并进 trunk2sm。"
             "结构模块的 layer_norm_s / layer_norm_z 和 linear_in 每次 recycle 只跑一遍；循环体内的 LayerNorm 和 ReLU 按 num_blocks 次。"
             "lddt 头的 LayerNorm 和三层线性作用在 num_blocks 份残基状态上，只在最后一次 recycle 之后跑一遍，序列里没有 ReLU。"
             "逐线性 .ew 尾项不减。sigmoid 门、IPA 的 softplus、扭转角的 clamp、esm_s_combine 的 softmax 不单列"]
    return _spec(model_id, hf_id, rel, "ESMFold（ESM-2 3B + 折叠主干 + 结构模块）", t["sequence_state_dim"],
                 [esm_layer] * (L - 1) + [replace(esm_layer, emb_ln=True)] + pl, wl, notes, [], "esmfold")


def _af2_groups(c: dict) -> list[Group]:
    return [Group("template_pair_stack.blocks.", "template", "recycle", blocks=True, grid="tmpl"),
            Group("extra_msa_stack.blocks.", "extra_msa", "recycle", blocks=True),
            Group("evoformer.blocks.", "evoformer", "recycle", blocks=True),
            Group("structure_module.", "structure", "recycle", iters=c["no_ipa_blocks"]),
            Group("aux_heads.", "heads"),
            Group("", "embed", "recycle")]


def _need_vec(sh: dict, who: str, name: str, width: int) -> None:
    if list(sh.get(name) or []) != [width]:
        raise ValueError(f"{who} {name} shape is not [{width}]")


def _lin_norm(layer: Layer, name: str, who: str) -> Linear:
    from .ir import _strip_core
    found = [ln for ln in layer.pair_linears if _strip_core(ln.name) == name]
    if len(found) != 1:
        raise ValueError(f"{who} {name} is missing")
    return found[0]


def _expect_of(ln: Linear, k: int, n: int, rows: str, who: str) -> None:
    if (ln.k, ln.n, ln.rows) != (k, n, rows):
        raise ValueError(f"{who} {ln.name} shape is not [{k}, {n}] on {rows}")


def _tag_af2_vectors(pl: list[Layer], sh: dict, c: dict, stacks: dict, who: str) -> list[Layer]:
    """Bill published LayerNorm and ReLU on OpenFold and AlphaFold 2. Mask-mul stays in the .ew tail.

    OpenFold multiplies some transition outputs by a mask. The AF2 JAX Transition expands the mask and
    does not multiply it onto the output, so this pass does not bill mask-mul for either model.
    """
    from .ir import (
        ALPHAFOLD2_HEAD_LINEARS, ALPHAFOLD2_SM_LINEARS, ESMFOLD_SM_LINEARS, OPENFOLD_EMBED_LINEARS,
        OPENFOLD_EVO_LINEARS, OPENFOLD_EXTRA_LINEARS, OPENFOLD_HEAD_LINEARS, OPENFOLD_TMPL_LINEARS, _strip_core,
    )
    cm, cz, cs, ct, ce = int(c["c_m"]), int(c["c_z"]), int(c["c_s"]), int(c["c_t"]), int(c["c_e"])
    if cm <= 0 or cz <= 0 or cs <= 0 or ct <= 0 or ce <= 0:
        raise ValueError(f"{who} channel width is not positive")
    if c.get("no_blocks") != 48 or stacks.get("evoformer.blocks") != 48:
        raise ValueError(f"{who} evoformer blocks are not 48")
    if c.get("extra_msa_blocks") != 4 or stacks.get("extra_msa_stack.blocks") != 4:
        raise ValueError(f"{who} extra msa blocks are not 4")
    if c.get("template_blocks") != 2 or stacks.get("template_pair_stack.blocks") != 2:
        raise ValueError(f"{who} template blocks are not 2")
    if c.get("no_ipa_blocks") != 8:
        raise ValueError(f"{who} ipa blocks are not 8")
    of = who == "openfold"
    if of:
        pre_name = "linear_in"
        sm_set, head_set = ESMFOLD_SM_LINEARS, OPENFOLD_HEAD_LINEARS
        plddt = ("plddt.linear_1", "plddt.linear_2", "plddt.linear_3")
        angle = ("angle_resnet.linear_initial", "angle_resnet.linear_in",
                 "angle_resnet.layers.0.linear_1", "angle_resnet.layers.0.linear_2",
                 "angle_resnet.layers.1.linear_1", "angle_resnet.layers.1.linear_2",
                 "angle_resnet.linear_out")
        trans = ("transition.layers.0.linear_1", "transition.layers.0.linear_2", "transition.layers.0.linear_3")
        bb = "bb_update.linear"
    else:
        pre_name = "initial_projection"
        sm_set, head_set = ALPHAFOLD2_SM_LINEARS, ALPHAFOLD2_HEAD_LINEARS
        plddt = ("plddt.act_0", "plddt.act_1", "plddt.logits")
        angle = ("rigid_sidechain.input_projection_1", "rigid_sidechain.input_projection",
                 "rigid_sidechain.resblock1", "rigid_sidechain.resblock2",
                 "rigid_sidechain.resblock1_1", "rigid_sidechain.resblock2_1",
                 "rigid_sidechain.unnormalized_angles")
        trans = ("transition", "transition_1", "transition_2")
        bb = "affine_update"

    def evo_norms(msa_w: int, h_in: int, h_out: int, col: str) -> list[tuple[str, int]]:
        if of:
            return [
                ("core.msa_transition.layer_norm.weight", msa_w),
                ("core.outer_product_mean.layer_norm.weight", msa_w),
                ("core.pair_transition.layer_norm.weight", cz),
                ("core.tri_att_end.layer_norm.weight", cz),
                ("core.tri_att_start.layer_norm.weight", cz),
                ("core.tri_mul_in.layer_norm_in.weight", cz),
                ("core.tri_mul_in.layer_norm_out.weight", h_in),
                ("core.tri_mul_out.layer_norm_in.weight", cz),
                ("core.tri_mul_out.layer_norm_out.weight", h_out),
                (col, msa_w),
                ("msa_att_row.layer_norm_m.weight", msa_w),
                ("msa_att_row.layer_norm_z.weight", cz),
            ]
        return [
            ("msa_transition.input_layer_norm.scale", msa_w),
            ("outer_product_mean.layer_norm_input.scale", msa_w),
            ("pair_transition.input_layer_norm.scale", cz),
            ("tri_att_end.query_norm.scale", cz),
            ("tri_att_start.query_norm.scale", cz),
            ("tri_mul_in.layer_norm_input.scale", cz),
            ("tri_mul_in.center_layer_norm.scale", h_in),
            ("tri_mul_out.layer_norm_input.scale", cz),
            ("tri_mul_out.center_layer_norm.scale", h_out),
            (col, msa_w),
            ("msa_att_row.query_norm.scale", msa_w),
            ("msa_att_row.feat_2d_norm.scale", cz),
        ]

    def tmpl_norms(h_in: int, h_out: int) -> list[tuple[str, int]]:
        if of:
            return [
                ("pair_transition.layer_norm.weight", ct),
                ("tri_att_end.layer_norm.weight", ct),
                ("tri_att_start.layer_norm.weight", ct),
                ("tri_mul_in.layer_norm_in.weight", ct),
                ("tri_mul_in.layer_norm_out.weight", h_in),
                ("tri_mul_out.layer_norm_in.weight", ct),
                ("tri_mul_out.layer_norm_out.weight", h_out),
            ]
        return [
            ("pair_transition.input_layer_norm.scale", ct),
            ("tri_att_end.query_norm.scale", ct),
            ("tri_att_start.query_norm.scale", ct),
            ("tri_mul_in.layer_norm_input.scale", ct),
            ("tri_mul_in.center_layer_norm.scale", h_in),
            ("tri_mul_out.layer_norm_input.scale", ct),
            ("tri_mul_out.center_layer_norm.scale", h_out),
        ]

    def hidden(layer: Layer, side: str) -> int:
        z = _lin_norm(layer, f"tri_mul_{side}.linear_z", who)
        proj = _lin_norm(layer, f"tri_mul_{side}.linear_a_p", who)
        if z.k != proj.n:
            raise ValueError(f"{who} triangle multiply hidden width is not the projection width")
        return z.k

    def check_block_norms(prefix: str, last: int, specs_at) -> None:
        for idx in (0, last):
            for suffix, width in specs_at(idx):
                _need_vec(sh, who, f"{prefix}{idx}.{suffix}", width)

    evo_layers = [L for L in pl if L.stack == "evoformer"]
    extra_layers = [L for L in pl if L.stack == "extra_msa"]
    tmpl_layers = [L for L in pl if L.stack == "template"]
    if not evo_layers or not extra_layers or not tmpl_layers:
        raise ValueError(f"{who} vector layers are not the published set")

    def evo_at(idx: int):
        layer = evo_layers[0] if idx == 0 else evo_layers[-1]
        col = "msa_att_col._msa_att.layer_norm_m.weight" if of else "msa_att_col._msa_att.query_norm.scale"
        return evo_norms(cm, hidden(layer, "in"), hidden(layer, "out"), col)

    def extra_at(idx: int):
        layer = extra_layers[0] if idx == 0 else extra_layers[-1]
        col = "msa_att_col.layer_norm_m.weight" if of else "msa_att_col.query_norm.scale"
        return evo_norms(ce, hidden(layer, "in"), hidden(layer, "out"), col)

    def tmpl_at(idx: int):
        layer = tmpl_layers[0] if idx == 0 else tmpl_layers[-1]
        return tmpl_norms(hidden(layer, "in"), hidden(layer, "out"))

    check_block_norms("evoformer.blocks.", 47, evo_at)
    check_block_norms("extra_msa_stack.blocks.", 3, extra_at)
    check_block_norms("template_pair_stack.blocks.", 1, tmpl_at)
    if of:
        _need_vec(sh, who, "template_pair_stack.layer_norm.weight", ct)
        _need_vec(sh, who, "recycling_embedder.layer_norm_m.weight", cm)
        _need_vec(sh, who, "recycling_embedder.layer_norm_z.weight", cz)
        _need_vec(sh, who, "structure_module.layer_norm_s.weight", cs)
        _need_vec(sh, who, "structure_module.layer_norm_z.weight", cz)
        _need_vec(sh, who, "structure_module.layer_norm_ipa.weight", cs)
        _need_vec(sh, who, "structure_module.transition.layer_norm.weight", cs)
        _need_vec(sh, who, "aux_heads.plddt.layer_norm.weight", cs)
    else:
        _need_vec(sh, who, "template_pair_stack.layer_norm.scale", ct)
        _need_vec(sh, who, "recycling_embedder.layer_norm_m.scale", cm)
        _need_vec(sh, who, "recycling_embedder.layer_norm_z.scale", cz)
        _need_vec(sh, who, "structure_module.single_layer_norm.scale", cs)
        _need_vec(sh, who, "structure_module.pair_layer_norm.scale", cz)
        _need_vec(sh, who, "structure_module.attention_layer_norm.scale", cs)
        _need_vec(sh, who, "structure_module.transition_layer_norm.scale", cs)
        _need_vec(sh, who, "aux_heads.plddt.input_layer_norm.scale", cs)

    tagged: list[Layer] = []
    for layer in pl:
        normed = {_strip_core(ln.name) for ln in layer.pair_linears}
        if layer.stack == "template":
            if normed != OPENFOLD_TMPL_LINEARS or layer.repeat != "recycle":
                raise ValueError(f"{who} block linears are not the published set")
            _expect_of(_lin_norm(layer, "pair_transition.linear_1", who), ct, 2 * ct, "tmpl", who)
            for side in ("start", "end"):
                att = _lin_norm(layer, f"tri_att_{side}.linear", who)
                if (att.k, att.n, att.rows) != (ct, 4, "tmpl"):
                    raise ValueError(f"{who} triangle attention heads are not 4")
            tagged.append(replace(layer, vec_kind="of_block"))
        elif layer.stack in ("evoformer", "extra_msa"):
            want = OPENFOLD_EVO_LINEARS if layer.stack == "evoformer" else OPENFOLD_EXTRA_LINEARS
            msa_w = cm if layer.stack == "evoformer" else ce
            grid = "msa" if layer.stack == "evoformer" else "xmsa"
            if normed != want or layer.repeat != "recycle":
                raise ValueError(f"{who} block linears are not the published set")
            _expect_of(_lin_norm(layer, "msa_transition.linear_1", who), msa_w, 4 * msa_w, grid, who)
            _expect_of(_lin_norm(layer, "pair_transition.linear_1", who), cz, 4 * cz, "pair", who)
            _expect_of(_lin_norm(layer, "outer_product_mean.linear_1", who), msa_w, 32, grid, who)
            for side in ("start", "end"):
                att = _lin_norm(layer, f"tri_att_{side}.linear", who)
                if (att.k, att.n, att.rows) != (cz, 4, "pair"):
                    raise ValueError(f"{who} triangle attention heads are not 4")
            if layer.stack == "extra_msa":
                gate = _lin_norm(layer, "msa_att_col.global_attention.linear_g", who)
                query = _lin_norm(layer, "msa_att_col.global_attention.linear_q", who)
                if gate.rows != "xmsa" or gate.k != ce or query.rows != "res":
                    raise ValueError(f"{who} extra msa column norm is not on the msa grid")
            else:
                gate = _lin_norm(layer, "msa_att_col._msa_att.mha.linear_g", who)
                if gate.rows != "msa" or gate.k != cm:
                    raise ValueError(f"{who} msa column norm is not on the msa grid")
            row_z = _lin_norm(layer, "msa_att_row.linear_z", who)
            row_g = _lin_norm(layer, "msa_att_row.mha.linear_g", who)
            if row_z.rows != "pair" or row_z.k != cz or row_g.rows != grid or row_g.k != msa_w:
                raise ValueError(f"{who} msa row norm grids do not match the published forward")
            tagged.append(replace(layer, vec_kind="of_block"))
        elif layer.stack == "structure":
            by = {ln.name: ln for ln in layer.pair_linears}
            if layer.iters != 8 or layer.repeat != "recycle" or pre_name not in by \
                    or set(by) - {pre_name} != sm_set:
                raise ValueError(f"{who} structure-module loop linears are not the published set")
            _expect_of(by[pre_name], cs, cs, "res", who)
            _expect_of(by[angle[0]], cs, 128, "res", who)
            _expect_of(by[angle[1]], cs, 128, "res", who)
            for name in angle[2:6]:
                _expect_of(by[name], 128, 128, "res", who)
            _expect_of(by[angle[6]], 128, 14, "res", who)
            for name in trans:
                _expect_of(by[name], cs, cs, "res", who)
            _expect_of(by[bb], cs, 6, "res", who)
            ipa_out = by["ipa.linear_out"]
            if ipa_out.n != cs or ipa_out.rows != "res":
                raise ValueError(f"{who} ipa.linear_out shape is not the single width")
            # The input projection runs once per structure-module call. Its bias stays in the loop misc.
            pre = replace(layer, pair_linears=(by[pre_name],), pair_cores=(), misc_params=0,
                          vec_kind="of_sm_pre", vec_width=cz, iters=1)
            rest = tuple(ln for ln in layer.pair_linears if ln.name != pre_name)
            tagged.append(pre)
            tagged.append(replace(layer, pair_linears=rest, vec_kind="of_sm"))
        elif layer.stack == "heads":
            plddt_names = [ln.name for ln in layer.pair_linears if ln.name.startswith("plddt.")]
            if len(plddt_names) > 3:
                raise ValueError(f"{who} plddt head has a fourth linear")
            if layer.iters != 1 or layer.repeat != "" or {ln.name for ln in layer.pair_linears} != head_set:
                raise ValueError(f"{who} heads linears are not the published set")
            by = {ln.name: ln for ln in layer.pair_linears}
            _expect_of(by[plddt[0]], cs, 128, "res", who)
            _expect_of(by[plddt[1]], 128, 128, "res", who)
            _expect_of(by[plddt[2]], 128, 50, "res", who)
            tagged.append(replace(layer, vec_kind="of_heads"))
        elif layer.stack == "embed":
            if layer.repeat != "recycle" or layer.iters != 1 \
                    or {ln.name for ln in layer.pair_linears} != OPENFOLD_EMBED_LINEARS:
                raise ValueError(f"{who} embed linears are not the published set")
            by = {ln.name: ln for ln in layer.pair_linears}
            _expect_of(by["evoformer.linear"], cm, cs, "res", who)
            _expect_of(by["recycling_embedder.linear"], 15, cz, "pair", who)
            _expect_of(by["template_angle_embedder.linear_1"], 57, cm, "tres", who)
            _expect_of(by["template_angle_embedder.linear_2"], cm, cm, "tres", who)
            _expect_of(by["template_pair_embedder.linear"], 88, ct, "tmpl", who)
            _expect_of(by["template_pointwise_att.mha.linear_k"], ct, ct, "tmpl", who)
            tagged.append(replace(layer, vec_kind="of_embed"))
        else:
            raise ValueError(f"{who} vector layers are not the published set")
    kinds: dict[str, int] = {}
    for layer in tagged:
        kinds[layer.vec_kind] = kinds.get(layer.vec_kind, 0) + 1
    if kinds.get("of_block") != 54 or kinds.get("of_sm") != 1 or kinds.get("of_sm_pre") != 1 \
            or kinds.get("of_heads") != 1 or kinds.get("of_embed") != 1:
        raise ValueError(f"{who} vector layers are not the published set")
    return tagged


def build_alphafold2(model_id: str, hf_id: str, rel: dict, wl) -> ModelSpec:
    """AlphaFold 2 monomer (model_1_ptm), from the official JAX parameter release (names mapped to OpenFold-style
    names by scripts/summarize_ckpt.py, so the same row / core tables apply)."""
    c = rel["config"]
    pl, _ = layers_from(rel, _af2_groups(c), {"c_hidden_msa_att": c["c_hidden_msa_att"]})
    pl = _tag_af2_vectors(pl, rel["shapes"], c, rel["stacks"], "alphafold2")
    msa = c["max_msa_clusters"] - c["max_templates"]
    wl = _apply_defaults(wl, msa=msa, xmsa=c["max_extra_msa"], templates=c["max_templates"], tmpl_msa=True,
                         recycles=c["max_recycling_iters"] + 1, pair_dim=c["c_z"])
    notes = [f"官方 JAX 参数 params_model_1_ptm（alphafold_params_2022-12-06.tar，CC BY 4.0）：Evoformer "
             f"{rel['stacks']['evoformer.blocks']} 块（MSA {c['c_m']} 维 / pair {c['c_z']} 维）+ extra MSA 栈 "
             f"{rel['stacks']['extra_msa_stack.blocks']} 块（全局列注意力）+ 模板 pair 栈 "
             f"{rel['stacks']['template_pair_stack.blocks']} 块 + 结构模块（IPA × {c['no_ipa_blocks']}，循环共享权重）",
             f"默认（model_1_ptm）：MSA 聚类 {msa} 行（512 − 模板数）、extra MSA {c['max_extra_msa']} 行、模板 "
             f"{c['max_templates']} 个；num_recycle {c['max_recycling_iters']} → 主干 {c['max_recycling_iters'] + 1} 遍；"
             "MSA 行数按上限计「假设」（浅 MSA 更快）",
             "monomer 预设跑 5 个模型（model_1…5_ptm）各一次：这里评估单个模型的一次预测；5 模型 ≈ 5 倍（model_3–5 "
             "无模板，略少）",
             "模板扭转角嵌入每个模板残基一行（T × N 行），其输出拼接到 MSA 表示上、随 MSA 跑完 Evoformer 全部块（MSA 网格 = 聚类行 + T）；IPA 几何运算同 ESMFold 的近似；subbatch / "
             "全局分块只降低峰值显存",
             "0.441：Evoformer 和 extra MSA 每块计 12 次 LayerNorm 和 2 次 ReLU，模板 pair 块计 7 次 LayerNorm 和 1 次 ReLU。"
             "结构模块的 single / pair LayerNorm 和 initial_projection 每次 recycle 只跑一遍，IPA 循环仍是 8 次。"
             "pLDDT 的 LayerNorm 和两次 ReLU 只作用在最终 single 上，不乘 IPA 次数，也不乘 recycle。"
             "recycling 的第一条 MSA 行和 pair、模板角的 ReLU、模板栈最终 LayerNorm 每次 recycle 计一次。"
             "逐线性 .ew 不减。mask 乘、sigmoid 门、IPA 的 softplus、扭转角 clamp 不单列"]
    reasons = ["只评估神经网络推理：MSA / 模板检索（jackhmmer / HHblits 数据库搜索，CPU）与特征化未建模；AMBER 松弛未建模"]
    return _spec(model_id, hf_id, rel, "AlphaFold 2（Evoformer + 结构模块，官方 JAX 参数）", c["c_s"], pl, wl, notes,
                 reasons, "alphafold2")


def build_openfold(model_id: str, hf_id: str, rel: dict, wl) -> ModelSpec:
    c = rel["config"]
    groups = [Group("template_pair_stack.blocks.", "template", "recycle", blocks=True, grid="tmpl"),
              Group("extra_msa_stack.blocks.", "extra_msa", "recycle", blocks=True),
              Group("evoformer.blocks.", "evoformer", "recycle", blocks=True),
              Group("structure_module.", "structure", "recycle", iters=c["no_ipa_blocks"]),
              Group("aux_heads.", "heads"),
              Group("", "embed", "recycle")]
    pl, _ = layers_from(rel, groups, {"c_hidden_msa_att": c["c_hidden_msa_att"]})
    pl = _tag_af2_vectors(pl, rel["shapes"], c, rel["stacks"], "openfold")
    wl = _apply_defaults(wl, msa=c["max_msa_clusters"], xmsa=c["max_extra_msa"], templates=c["max_templates"], tmpl_msa=True,
                         recycles=c["max_recycling_iters"] + 1, pair_dim=c["c_z"])
    notes = [f"AF2 架构（OpenFold 复现权重 finetuning_ptm_2）：Evoformer {rel['stacks']['evoformer.blocks']} 块（MSA {c['c_m']} 维 / "
             f"pair {c['c_z']} 维）+ extra MSA 栈 {rel['stacks']['extra_msa_stack.blocks']} 块（全局列注意力）+ 模板 pair 栈 "
             f"{rel['stacks']['template_pair_stack.blocks']} 块 + 结构模块（IPA × {c['no_ipa_blocks']}，循环共享权重）",
             f"默认：MSA 聚类 {c['max_msa_clusters']} 行、extra MSA {c['max_extra_msa']} 行、模板 {c['max_templates']} 个"
             f"（data.predict）；recycle {c['max_recycling_iters']} 次 → 主干 {c['max_recycling_iters'] + 1} 遍（嵌入、模板、"
             "extra MSA、Evoformer、结构模块每遍都重跑）；MSA 行数按上限计「假设」（浅 MSA 更快）",
             "模板扭转角嵌入每个模板残基一行（T × N 行），其输出拼接到 MSA 表示上、随 MSA 跑完 Evoformer 全部块（MSA 网格 = 聚类行 + T）；IPA 几何运算同 ESMFold 的近似",
             "0.441：Evoformer 和 extra MSA 每块计 12 次 LayerNorm 和 2 次 ReLU，模板 pair 块计 7 次 LayerNorm 和 1 次 ReLU。"
             "结构模块的 layer_norm_s / layer_norm_z 和 linear_in 每次 recycle 只跑一遍，IPA 循环仍是 8 次。"
             "pLDDT 的 LayerNorm 和两次 ReLU 只作用在最终 single 上，不乘 IPA 次数，也不乘 recycle。"
             "recycling 的第一条 MSA 行和 pair、模板角的 ReLU、模板栈最终 LayerNorm 每次 recycle 计一次。"
             "逐线性 .ew 不减。mask 乘、sigmoid 门、IPA 的 softplus、扭转角 clamp 不单列",
             "0.459：MSATransition 与 PairTransition 在 linear_2 之后各乘一次 mask。"
             "发布配置 model._mask_trans 是 False：Evoformer 的 MSA 和所有 pair 过渡先造全 1 再乘；"
             "extra MSA 的 MSA 过渡传入真实 msa_mask。不记造 mask。宽度是 linear_2 的 N，不是 ReLU 的隐藏宽。"
             "sigmoid 门、模板嵌入 t * t_mask、OPM 的除以 clamp、IPA 的 softplus、扭转角 clamp 仍不单列"]
    reasons = ["只评估神经网络推理：MSA / 模板检索（jackhmmer / HHblits 数据库搜索，CPU）与特征化未建模；AMBER 松弛未建模"]
    return _spec(model_id, hf_id, rel, "AF2 架构（Evoformer + 结构模块，OpenFold 权重）", c["c_s"], pl, wl, notes, reasons,
                 "openfold")


def _af3_groups(pfx: dict) -> list[Group]:
    """AF3-family groups: atom encoders (windowed atom attention), trunk stacks (per recycle), diffusion module
    (per step, token / atom grids × samples), confidence (once per sample), heads (once)."""
    out = []
    for prefix, stack, rep, smp, atom in pfx["stacks"]:
        out.append(Group(prefix, stack, rep, samples=smp, blocks=True, atom=atom))
    for prefix, stack, rep, smp, skip in pfx["rest"]:
        out.append(Group(prefix, stack, rep, samples=smp, skip=skip))
    return out


_BOLTZ_CACHED = re.compile(r"(pairwise_conditioner|pair_bias_attn\.proj_z|atom_attention_encoder\.(embed_atom_features|"
                           r"embed_atompair_|s_to_c_trans|z_to_p_trans|c_to_p_trans_[qk]|p_mlp))")


def _hoist_step_cache(layers: list[Layer], cached: re.Pattern) -> list[Layer]:
    """0.61.3: the diffusion module's step-invariant work (Boltz-1 ``use_inference_model_cache``, on by default in
    ``boltz predict``: pair conditioning, every layer's pair-bias projection, the atom encoder's pair / conditioning
    features) runs on the first sampler step only and is reused afterwards.  Those weight GEMMs move out of the
    per-step diffusion layers into one ``diff_cache`` layer that runs once per request (independent of the
    samples, which only enter after the cache).  Parameters are unchanged (the linears move, nothing is added)."""
    once, out, first = [], [], None
    for L in layers:
        moved = [l for l in L.pair_linears if L.repeat == "diff" and cached.search(l.name)]
        if not moved:
            out.append(L)
            continue
        first = len(out) if first is None else first
        once += [replace(l, name=f"{L.stack}.{l.name}") for l in moved]
        out.append(replace(L, pair_linears=tuple(l for l in L.pair_linears if l not in moved)))
    if first is None:
        return layers
    return out[:first] + [Layer((), AttnCore("none"), Ffn("none"), (), 0, pair_linears=tuple(once), misc_role="pair",
                                stack="diff_cache")] + out[first:]


def _tag_boltz_vectors(pl: list[Layer], sh: dict, c: dict, stacks: dict) -> list[Layer]:
    """Bill published LayerNorm and Transition SiLU on Boltz-1. Sigmoid gates stay in the .ew tail.

    Diffusion AdaLN and ConditionedTransitionBlock are not the plain LayerNorm / SiLU recipe.
    """
    from .ir import (
        BOLTZ_CONF_LINEARS, BOLTZ_DIFF_LINEARS, BOLTZ_MSA_LINEARS, BOLTZ_MSA_REST, BOLTZ_PAIR_LINEARS,
        _bz_check_cache, _bz_check_conf, _bz_check_diff, _bz_check_msa, _bz_check_pair, _bz_check_recycle,
    )
    who = "boltz-1"
    hp = c["hyper_parameters"]
    if int(hp["token_s"]) != 384 or int(hp["token_z"]) != 128:
        raise ValueError(f"{who} token width is not the published pairformer")
    if stacks.get("pairformer_module.layers") != 48 or stacks.get("confidence_module.pairformer_module.layers") != 48:
        raise ValueError(f"{who} pairformer blocks are not 48")
    if stacks.get("msa_module.layers") != 4 or stacks.get("confidence_module.msa_module.layers") != 4:
        raise ValueError(f"{who} msa blocks are not 4")

    def block_norms(prefix: str, last: int, specs: tuple[tuple[str, int], ...]) -> None:
        for idx in (0, last):
            for suffix, width in specs:
                _need_vec(sh, who, f"{prefix}{idx}.{suffix}", width)

    pair_norms = (
        ("attention.norm_s.weight", 384),
        ("attention.proj_z.0.weight", 128),
        ("transition_s.norm.weight", 384),
        ("transition_z.norm.weight", 128),
        ("tri_att_start.layer_norm.weight", 128),
        ("tri_att_end.layer_norm.weight", 128),
        ("tri_mul_in.norm_in.weight", 128),
        ("tri_mul_in.norm_out.weight", 128),
        ("tri_mul_out.norm_in.weight", 128),
        ("tri_mul_out.norm_out.weight", 128),
    )
    msa_norms = (
        ("msa_transition.norm.weight", 64),
        ("outer_product_mean.norm.weight", 64),
        ("pair_weighted_averaging.norm_m.weight", 64),
        ("pair_weighted_averaging.norm_z.weight", 128),
        ("z_transition.norm.weight", 128),
        ("tri_att_start.layer_norm.weight", 128),
        ("tri_att_end.layer_norm.weight", 128),
        ("tri_mul_in.norm_in.weight", 128),
        ("tri_mul_in.norm_out.weight", 128),
        ("tri_mul_out.norm_in.weight", 128),
        ("tri_mul_out.norm_out.weight", 128),
    )
    block_norms("pairformer_module.layers.", 47, pair_norms)
    block_norms("confidence_module.pairformer_module.layers.", 47, pair_norms)
    block_norms("msa_module.layers.", 3, msa_norms)
    block_norms("confidence_module.msa_module.layers.", 3, msa_norms)
    for name, width in (
        ("s_norm.weight", 384),
        ("z_norm.weight", 128),
        ("confidence_module.s_norm.weight", 384),
        ("confidence_module.z_norm.weight", 128),
        ("confidence_module.final_s_norm.weight", 384),
        ("confidence_module.final_z_norm.weight", 128),
        ("confidence_module.s_diffusion_norm.weight", 768),
        ("structure_module.score_model.pairwise_conditioner.dim_pairwise_init_proj.0.weight", 256),
        ("structure_module.score_model.pairwise_conditioner.transitions.0.norm.weight", 128),
        ("structure_module.score_model.pairwise_conditioner.transitions.1.norm.weight", 128),
        ("structure_module.score_model.single_conditioner.norm_single.weight", 839),
        ("structure_module.score_model.single_conditioner.norm_fourier.weight", 256),
        ("structure_module.score_model.single_conditioner.transitions.0.norm.weight", 768),
        ("structure_module.score_model.single_conditioner.transitions.1.norm.weight", 768),
        ("structure_module.score_model.s_to_a_linear.0.weight", 768),
        ("structure_module.score_model.a_norm.weight", 768),
        ("structure_module.out_token_feat_update.norm_next.weight", 768),
        ("structure_module.out_token_feat_update.norm_fourier.weight", 256),
    ):
        _need_vec(sh, who, name, width)

    tagged: list[Layer] = []
    for layer in pl:
        names = {ln.name for ln in layer.pair_linears}
        if layer.stack in ("pairformer", "confidence"):
            if not names:
                tagged.append(layer)
            elif names == BOLTZ_PAIR_LINEARS:
                _bz_check_pair(layer)
                tagged.append(replace(layer, vec_kind="bz_pair"))
            elif layer.stack == "confidence" and names == BOLTZ_CONF_LINEARS:
                _bz_check_conf(layer)
                tagged.append(replace(layer, vec_kind="bz_conf"))
            else:
                raise ValueError(f"{who} pairformer linears are not the published set")
        elif layer.stack in ("msa", "conf_msa"):
            if names == BOLTZ_MSA_LINEARS:
                _bz_check_msa(layer)
                tagged.append(replace(layer, vec_kind="bz_msa"))
            elif layer.stack == "msa" and names == BOLTZ_MSA_REST:
                tagged.append(layer)
            elif not names:
                tagged.append(layer)
            else:
                raise ValueError(f"{who} msa linears are not the published set")
        elif layer.stack == "recycle":
            if not names:
                tagged.append(layer)
            else:
                _bz_check_recycle(layer)
                tagged.append(replace(layer, vec_kind="bz_recycle"))
        elif layer.stack == "diff_cache":
            _bz_check_cache(layer)
            tagged.append(replace(layer, vec_kind="bz_cache"))
        elif layer.stack == "diffusion" and "score_model.single_conditioner.single_embed" in names:
            _bz_check_diff(layer)
            tagged.append(replace(layer, vec_kind="bz_diff"))
        else:
            tagged.append(layer)
    kinds: dict[str, int] = {}
    recycle_shapes = []
    for layer in tagged:
        if layer.vec_kind:
            kinds[layer.vec_kind] = kinds.get(layer.vec_kind, 0) + 1
        if layer.vec_kind == "bz_recycle":
            lin = layer.pair_linears[0]
            recycle_shapes.append((lin.k, lin.n, lin.rows))
    if kinds.get("bz_pair") != 96 or kinds.get("bz_msa") != 8 or kinds.get("bz_recycle") != 2 \
            or kinds.get("bz_conf") != 1 or kinds.get("bz_cache") != 1 or kinds.get("bz_diff") != 1:
        raise ValueError(f"{who} vector layers are not the published set")
    if sorted(recycle_shapes) != [(128, 128, "pair"), (384, 384, "res")]:
        raise ValueError(f"{who} recycle projection shape is not the published pair")
    return tagged


def build_boltz1(model_id: str, hf_id: str, rel: dict, wl) -> ModelSpec:
    c = rel["config"]
    hp = c["hyper_parameters"]
    win = (hp["atoms_per_window_queries"], hp["atoms_per_window_keys"])
    ie = "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers."
    sm = "structure_module.score_model."
    pfx = {"stacks": [(ie, "atom_embed", "", "", True),
                      ("msa_module.layers.", "msa", "recycle", "", False),
                      ("pairformer_module.layers.", "pairformer", "recycle", "", False),
                      (sm + "atom_attention_encoder.atom_encoder.diffusion_transformer.layers.", "diff_atom_enc", "diff", "tok", True),
                      (sm + "token_transformer.layers.", "diffusion", "diff", "tok", False),
                      (sm + "atom_attention_decoder.atom_decoder.diffusion_transformer.layers.", "diff_atom_dec", "diff", "tok", True),
                      ("confidence_module." + ie, "conf_atom", "", "all", True),
                      ("confidence_module.msa_module.layers.", "conf_msa", "", "all", False),
                      ("confidence_module.pairformer_module.layers.", "confidence", "", "all", False)],
           "rest": [("input_embedder.", "embed", "", "", False),
                    ("s_recycle.", "recycle", "recycle", "", False), ("z_recycle.", "recycle", "recycle", "", False),
                    ("s_norm.", "recycle", "recycle", "", False), ("z_norm.", "recycle", "recycle", "", False),
                    ("msa_module.", "msa", "recycle", "", False),
                    ("pairformer_module.", "pairformer", "recycle", "", False),
                    ("structure_module.", "diffusion", "diff", "tok", False),
                    ("confidence_module.", "confidence", "", "all", False),
                    ("distogram_module.", "heads", "", "", False),
                    ("", "embed", "", "", False)]}
    pl, _ = layers_from(rel, _af3_groups(pfx), {"win": win})
    pl = _hoist_step_cache(pl, _BOLTZ_CACHED)
    pl = _tag_boltz_vectors(pl, rel["shapes"], c, rel["stacks"])
    wl = _apply_defaults(wl, msa=c["max_msa_seqs"], recycles=c["recycling_steps"] + 1, diff_steps=c["sampling_steps"],
                         samples=c["diffusion_samples"], pair_dim=hp["token_z"])
    notes = [f"AF3 类架构：MSA 模块 {rel['stacks']['msa_module.layers']} 块（pair 加权平均）+ Pairformer "
             f"{rel['stacks']['pairformer_module.layers']} 块（每次 recycle 重跑）+ 扩散模块（token transformer "
             f"{rel['stacks'][sm + 'token_transformer.layers']} 层 + 原子编码 / 解码各 3 层，窗口 {win[0]}×{win[1]} 原子）",
             f"默认（boltz v0.4.1 predict）：recycling_steps {c['recycling_steps']}（主干 {c['recycling_steps'] + 1} 遍）、"
             f"sampling_steps {c['sampling_steps']}、diffusion_samples {c['diffusion_samples']}、MSA ≤ {c['max_msa_seqs']} 行"
             "（按上限计「假设」）",
             f"置信度模块模仿主干（confidence_imitate_trunk：自带 MSA 模块 + {rel['stacks']['confidence_module.pairformer_module.layers']} "
             "块 Pairformer），每个样本跑一遍",
             "boltz predict 默认开启推理缓存（use_inference_model_cache）：pair 条件、每层 pair 偏置投影与原子编码器的原子对 / "
             "条件特征只在第一步计算、之后各步复用（diff_cache，每请求一次）；参考实现的 k / v 投影在重叠的 128 原子键窗口上算"
             "（4 倍行）、键窗口用稠密索引矩阵收集（各约占每步 1.4 %），这里按每原子一行计、收集不计（每步比参考计数少约 3 %）；原子数 = 残基 × "
             f"{wl.atoms_per_res:g}「假设」（仅蛋白质重原子）；原子级输入 / 输出投影按原子行计（部分实为 token 行，量很小）",
             "0.442：Pairformer 每块计 10 次 LayerNorm，外加 single / pair Transition 的 SiLU 和一次相乘。"
             "MSA 块再加 msa Transition、OPM 和 PWA 的 LayerNorm，一共 11 次。"
             "主干 recycle 的 s_norm / z_norm 每次 recycle 计一次。置信度 imitate_trunk 的 s_norm / z_norm 在按样本展开之前计一次，"
             "s_diffusion_norm、final_s_norm、final_z_norm 在展开之后计一次。"
             "扩散的 plain Transition（single / pairwise conditioner）和 a_norm、s_to_a 的 LayerNorm、norm_single、"
             "两次 Fourier LayerNorm、norm_next 按发布前向计。AdaLN、扩散 SwiGLU、sigmoid 门不单列。"
             "逐线性 .ew 不减。mask 乘、OPM 的 mask 归一化除法不单列"]
    reasons = ["只评估神经网络推理：MSA 检索（MMseqs2 服务器）与特征化 / 分子预处理未建模"]
    return _spec(model_id, hf_id, rel, "Boltz-1（AF3 类：MSA 模块 + Pairformer + 扩散）", hp["token_s"], pl, wl, notes,
                 reasons, "boltz1")


def _px_affine(sh: dict, name: str, width: int) -> None:
    _need_vec(sh, "protenix", name + ".weight", width)
    _need_vec(sh, "protenix", name + ".bias", width)


def _px_scale(sh: dict, name: str, width: int) -> None:
    """create_offset=False LayerNorm: scale only."""
    _need_vec(sh, "protenix", name + ".weight", width)
    if name + ".bias" in sh:
        raise ValueError(f"protenix {name}.bias is not in the published LayerNorm")


def _px_adaln_stays(sh: dict, name: str, width: int) -> None:
    """AdaptiveLayerNorm is sigmoid(scale) * norm(a) + bias. Do not bill it as LayerNorm."""
    scale = name + ".layernorm_a.weight"
    if scale in sh:
        raise ValueError(f"protenix {scale} is not the published scale-free AdaLN")
    _px_scale(sh, name + ".layernorm_s", width)


def _tag_protenix_vectors(pl: list[Layer], sh: dict, c: dict, stacks: dict) -> list[Layer]:
    """Bill published LayerNorm, Transition SiLU, and ReLU. AdaLN and diffusion SwiGLU stay out.

    TemplateEmbedder.forward returns 0 in v0.5.0, so its stored norms are not executed.
    """
    from .ir import (
        PROTENIX_ATOM_DIT_LINEARS, PROTENIX_CONF_LINEARS, PROTENIX_DIFF_LINEARS, PROTENIX_DIT_LINEARS,
        PROTENIX_EMBED_LINEARS, PROTENIX_MSA_LAST_LINEARS, PROTENIX_MSA_LINEARS, PROTENIX_MSA_REST,
        PROTENIX_PAIR_LINEARS, _px_check_conf, _px_check_diff, _px_check_dit, _px_check_embed,
        _px_check_msa, _px_check_pair, _px_check_recycle,
    )
    who = "protenix"
    if int(c["c_s"]) != 384 or int(c["c_z"]) != 128 or int(c["c_token"]) != 768:
        raise ValueError(f"{who} channel width is not the published pairformer")
    if int(c["c_atom"]) != 128 or int(c["c_atompair"]) != 16:
        raise ValueError(f"{who} atom width is not the published encoder")
    if c.get("template") is not False:
        raise ValueError(f"{who} template flag is not the published disabled setting")
    if c.get("pairformer_blocks") != 48 or stacks.get("pairformer_stack.blocks") != 48:
        raise ValueError(f"{who} pairformer blocks are not 48")
    if c.get("msa_blocks") != 4 or stacks.get("msa_module.blocks") != 4:
        raise ValueError(f"{who} msa blocks are not 4")
    if c.get("confidence_blocks") != 4 or stacks.get("confidence_head.pairformer_stack.blocks") != 4:
        raise ValueError(f"{who} confidence blocks are not 4")
    if c.get("diffusion_blocks") != 24 or stacks.get("diffusion_module.diffusion_transformer.blocks") != 24:
        raise ValueError(f"{who} diffusion blocks are not 24")
    atom_stacks = (
        "input_embedder.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks",
        "diffusion_module.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks",
        "diffusion_module.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks",
    )
    if c.get("atom_blocks") != 3 or any(stacks.get(name) != 3 for name in atom_stacks):
        raise ValueError(f"{who} atom transformer blocks are not 3")
    for layer in pl:
        for core in layer.pair_cores:
            if core.kind == "local_att" and core.win != (32, 128):
                raise ValueError(f"{who} atom window is not 32 x 128")

    def pair_block(prefix: str, last: int) -> None:
        for idx in (0, last):
            base = f"{prefix}{idx}."
            for suffix, width in (
                ("attention_pair_bias.layernorm_a", 384),
                ("attention_pair_bias.layernorm_z", 128),
                ("single_transition.layernorm1", 384),
                ("pair_transition.layernorm1", 128),
                ("tri_att_start.layer_norm", 128),
                ("tri_att_end.layer_norm", 128),
                ("tri_mul_in.layer_norm_in", 128),
                ("tri_mul_in.layer_norm_out", 128),
                ("tri_mul_out.layer_norm_in", 128),
                ("tri_mul_out.layer_norm_out", 128),
            ):
                _px_affine(sh, base + suffix, width)

    pair_block("pairformer_stack.blocks.", 47)
    pair_block("confidence_head.pairformer_stack.blocks.", 3)
    msa0 = "msa_module.blocks.0."
    for suffix, width in (
        ("msa_stack.msa_pair_weighted_averaging.layernorm_m", 64),
        ("msa_stack.msa_pair_weighted_averaging.layernorm_z", 128),
        ("msa_stack.transition_m.layernorm1", 64),
        ("outer_product_mean_msa.layer_norm", 64),
        ("pair_stack.pair_transition.layernorm1", 128),
        ("pair_stack.tri_att_start.layer_norm", 128),
        ("pair_stack.tri_att_end.layer_norm", 128),
        ("pair_stack.tri_mul_in.layer_norm_in", 128),
        ("pair_stack.tri_mul_in.layer_norm_out", 128),
        ("pair_stack.tri_mul_out.layer_norm_in", 128),
        ("pair_stack.tri_mul_out.layer_norm_out", 128),
    ):
        _px_affine(sh, msa0 + suffix, width)
    msa3 = "msa_module.blocks.3."
    for suffix, width in (
        ("outer_product_mean_msa.layer_norm", 64),
        ("pair_stack.pair_transition.layernorm1", 128),
        ("pair_stack.tri_att_start.layer_norm", 128),
        ("pair_stack.tri_att_end.layer_norm", 128),
        ("pair_stack.tri_mul_in.layer_norm_in", 128),
        ("pair_stack.tri_mul_in.layer_norm_out", 128),
        ("pair_stack.tri_mul_out.layer_norm_in", 128),
        ("pair_stack.tri_mul_out.layer_norm_out", 128),
    ):
        _px_affine(sh, msa3 + suffix, width)
    if any(k.startswith("msa_module.blocks.3.msa_stack.") for k in sh):
        raise ValueError(f"{who} last msa block still has an MSA stack")
    _px_affine(sh, "layernorm_z_cycle", 128)
    _px_affine(sh, "layernorm_s", 384)
    for name, width in (
        ("confidence_head.input_strunk_ln", 384),
        ("confidence_head.pae_ln", 128),
        ("confidence_head.pde_ln", 128),
        ("confidence_head.plddt_ln", 384),
        ("confidence_head.resolved_ln", 384),
    ):
        _px_affine(sh, name, width)
    for name, width in (
        ("diffusion_module.diffusion_conditioning.layernorm_z", 256),
        ("diffusion_module.diffusion_conditioning.layernorm_s", 833),
        ("diffusion_module.diffusion_conditioning.layernorm_n", 256),
        ("diffusion_module.atom_attention_encoder.layernorm_s", 384),
        ("diffusion_module.atom_attention_encoder.layernorm_z", 128),
        ("diffusion_module.layernorm_s", 384),
        ("diffusion_module.layernorm_a", 768),
        ("diffusion_module.atom_attention_decoder.layernorm_q", 128),
    ):
        _px_scale(sh, name, width)
    for name, width in (
        ("diffusion_module.diffusion_conditioning.transition_z1.layernorm1", 128),
        ("diffusion_module.diffusion_conditioning.transition_z2.layernorm1", 128),
        ("diffusion_module.diffusion_conditioning.transition_s1.layernorm1", 384),
        ("diffusion_module.diffusion_conditioning.transition_s2.layernorm1", 384),
    ):
        _px_affine(sh, name, width)
    _px_dit_norm(sh, "diffusion_module.diffusion_transformer.blocks.", 23, 128, 384)
    for prefix in atom_stacks:
        _px_dit_norm(sh, prefix + ".", 2, 16, 128)

    tagged: list[Layer] = []
    for layer in pl:
        names = {ln.name for ln in layer.pair_linears}
        if layer.stack in ("pairformer", "confidence"):
            if not names:
                tagged.append(layer)
            elif names == PROTENIX_PAIR_LINEARS:
                _px_check_pair(layer)
                tagged.append(replace(layer, vec_kind="px_pair"))
            elif layer.stack == "confidence" and names == PROTENIX_CONF_LINEARS:
                _px_check_conf(layer)
                tagged.append(replace(layer, vec_kind="px_conf"))
            else:
                raise ValueError(f"{who} pairformer linears are not the published set")
        elif layer.stack == "msa":
            if names == PROTENIX_MSA_LINEARS:
                _px_check_msa(layer, False)
                tagged.append(replace(layer, vec_kind="px_msa"))
            elif names == PROTENIX_MSA_LAST_LINEARS:
                _px_check_msa(layer, True)
                tagged.append(replace(layer, vec_kind="px_msa_last"))
            elif names == PROTENIX_MSA_REST or not names:
                tagged.append(layer)
            else:
                raise ValueError(f"{who} msa linears are not the published set")
        elif layer.stack == "recycle":
            if not names:
                tagged.append(layer)
            else:
                _px_check_recycle(layer)
                tagged.append(replace(layer, vec_kind="px_recycle"))
        elif layer.stack == "embed":
            if names == PROTENIX_EMBED_LINEARS:
                _px_check_embed(layer)
                tagged.append(replace(layer, vec_kind="px_embed"))
            else:
                tagged.append(layer)
        elif layer.stack == "diffusion":
            if names == PROTENIX_DIFF_LINEARS:
                _px_check_diff(layer)
                tagged.append(replace(layer, vec_kind="px_diff"))
            elif names == PROTENIX_DIT_LINEARS:
                _px_check_dit(layer)
                tagged.append(replace(layer, vec_kind="px_dit"))
            else:
                raise ValueError(f"{who} diffusion linears are not the published set")
        elif layer.stack in ("atom_embed", "diff_atom_enc", "diff_atom_dec"):
            if names == PROTENIX_ATOM_DIT_LINEARS:
                _px_check_dit(layer)
                tagged.append(replace(layer, vec_kind="px_dit"))
            else:
                raise ValueError(f"{who} atom transformer linears are not the published set")
        else:
            tagged.append(layer)
    kinds: dict[str, int] = {}
    recycle_shapes = []
    for layer in tagged:
        if layer.vec_kind:
            kinds[layer.vec_kind] = kinds.get(layer.vec_kind, 0) + 1
        if layer.vec_kind == "px_recycle":
            lin = layer.pair_linears[0]
            recycle_shapes.append((lin.k, lin.n, lin.rows))
    if kinds != {
        "px_pair": 52, "px_msa": 3, "px_msa_last": 1, "px_recycle": 2,
        "px_embed": 1, "px_diff": 1, "px_conf": 1, "px_dit": 33,
    }:
        raise ValueError(f"{who} vector layers are not the published set")
    if sorted(recycle_shapes) != [(128, 128, "pair"), (384, 384, "res")]:
        raise ValueError(f"{who} recycle projection shape is not the published pair")
    return tagged


def _px_dit_norm(sh: dict, prefix: str, last: int, c_z: int, c_s: int) -> None:
    for idx in (0, last):
        base = f"{prefix}{idx}."
        _px_scale(sh, base + "attention_pair_bias.layernorm_z", c_z)
        _px_adaln_stays(sh, base + "attention_pair_bias.layernorm_a", c_s)
        _px_adaln_stays(sh, base + "conditioned_transition_block.adaln", c_s)
        if c_z == 16:
            _px_adaln_stays(sh, base + "attention_pair_bias.layernorm_kv", c_s)


def build_protenix(model_id: str, hf_id: str, rel: dict, wl) -> ModelSpec:
    c = rel["config"]
    win = (32, 128)
    ie = "input_embedder.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks."
    dm = "diffusion_module."
    pfx = {"stacks": [(ie, "atom_embed", "", "", True),
                      ("msa_module.blocks.", "msa", "recycle", "", False),
                      ("pairformer_stack.blocks.", "pairformer", "recycle", "", False),
                      (dm + "atom_attention_encoder.atom_transformer.diffusion_transformer.blocks.", "diff_atom_enc", "diff", "tok", True),
                      (dm + "diffusion_transformer.blocks.", "diffusion", "diff", "tok", False),
                      (dm + "atom_attention_decoder.atom_transformer.diffusion_transformer.blocks.", "diff_atom_dec", "diff", "tok", True),
                      ("confidence_head.pairformer_stack.blocks.", "confidence", "", "all", False)],
           "rest": [("input_embedder.", "embed", "", "", False),
                    ("linear_no_bias_z_cycle.", "recycle", "recycle", "", False),
                    ("layernorm_z_cycle.", "recycle", "recycle", "", False),
                    ("linear_no_bias_s.", "recycle", "recycle", "", False),
                    ("layernorm_s.", "recycle", "recycle", "", False),
                    ("template_embedder.", "template", "", "", True),
                    ("msa_module.", "msa", "recycle", "", False),
                    ("pairformer_stack.", "pairformer", "recycle", "", False),
                    (dm, "diffusion", "diff", "tok", False),
                    ("confidence_head.", "confidence", "", "all", False),
                    ("distogram_head.", "heads", "", "", False),
                    ("", "embed", "", "", False)]}
    pl, _ = layers_from(rel, _af3_groups(pfx), {"win": win})
    # 0.61.3: Protenix expands the pair conditioning over the N_sample axis before the diffusion transformer
    # (DiffusionModule.f_forward: expand_at_dim(z_pair, dim=-4, n=N_sample)), so each block's pair-bias projection and
    # the atom encoder's z → atom-pair projection run once per sample (Boltz-1 projects first and repeats after)
    pl = [replace(L, pair_linears=tuple(replace(l, rows="spair") if l.rows == "pair" and l.name.endswith(
              ("attention_pair_bias.linear_nobias_z", "atom_attention_encoder.linear_no_bias_z")) else l
              for l in L.pair_linears)) if L.repeat == "diff" else L for L in pl]
    pl = _tag_protenix_vectors(pl, rel["shapes"], c, rel["stacks"])
    wl = _apply_defaults(wl, msa=c["msa_sample_cutoff_test"], recycles=c["N_cycle"], diff_steps=c["N_step"],
                         samples=c["N_sample"], pair_dim=c["c_z"])
    notes = [f"AF3 复现（v0.5.0）：MSA 模块 {c['msa_blocks']} 块 + Pairformer {c['pairformer_blocks']} 块（每个 cycle 重跑）+ "
             f"扩散模块（diffusion transformer {c['diffusion_blocks']} 层，{c['c_token']} 维 + 原子编码 / 解码各 "
             f"{c['atom_blocks']} 层，窗口 32×128 原子）+ 置信度头（Pairformer {c['confidence_blocks']} 块，每个样本一遍）",
             f"默认（configs_base / configs_data）：N_cycle {c['N_cycle']}、N_step {c['N_step']}、N_sample {c['N_sample']}"
             f"（样本批量并行）、MSA 采样 {c['msa_sample_cutoff_test']} 行（按上限计「假设」）；模板关闭（template_embedder "
             "权重只计存储）",
             "扩散模块按参考实现每步重算 pair 条件（v0.5.0 无跨步缓存），并把 pair 条件沿样本轴展开：每层 pair 偏置投影与原子编码器的 "
             "z → 原子对投影每个样本各算一次；原子数 = 残基 × "
             f"{wl.atoms_per_res:g}「假设」（仅蛋白质重原子）；原子级输入 / 输出投影按原子行计（部分实为 token 行；每步比参考计数多约 0.6 %）",
             "0.443：Pairformer 每块计 10 次 LayerNorm，外加 single / pair Transition 的 SiLU 和一次相乘。"
             "MSA 前 3 块再加 PWA、msa Transition 和 OPM 的 LayerNorm，一共 11 次；最后一块没有 MSAStack，是 8 次。"
             "recycle 的 layernorm_s / layernorm_z_cycle 每次 cycle 计一次。置信度 input_strunk_ln 在按样本展开之前计一次，"
             "pae / pde 在 pair 网格上按样本计，plddt / resolved 在原子行上按样本计。"
             "扩散条件化的 plain LayerNorm 和 Transition、原子编码器的 LayerNorm 和 ReLU、解码器 layernorm_q、"
             "以及扩散 transformer 的 pair-bias layernorm_z 按发布前向计。AdaLN、扩散 SwiGLU、sigmoid 门不单列。"
             "逐线性 .ew 不减。mask 乘、OPM 的 mask 归一化除法、input_strunk 前面的 clamp 不单列"]
    reasons = ["只评估神经网络推理：MSA 检索与特征化 / 分子预处理未建模"]
    return _spec(model_id, hf_id, rel, "Protenix（AF3 复现：MSA 模块 + Pairformer + 扩散）", c["c_s"], pl, wl, notes,
                 reasons, "protenix")


BUILDERS = {"esmfold": build_esmfold, "openfold": build_openfold, "alphafold2": build_alphafold2, "boltz1": build_boltz1, "protenix": build_protenix}
