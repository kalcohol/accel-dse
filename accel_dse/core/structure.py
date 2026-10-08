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
_ATOM_PAIR = re.compile(r"(pair_bias_attn\.proj_z|attention_pair_bias\.linear_nobias_z|embed_atompair|c_to_p_trans|"
                        r"p_mlp|small_mlp|z_to_p_trans|linear_no_bias_(d|invd|v|cl|cm|z)\b)")
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
        return "apair" if _ATOM_PAIR.search(name) else "atom"
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
            out.append(PairCore("trimul", dim=c, over=ctx.get("grid", "pair")))
        elif leaf.startswith("tri_att"):
            h = _out(sh, p + "linear.weight")
            out.append(PairCore("tri_att", heads=h, dim=_out(sh, p + "mha.linear_q.weight") // h,
                                over=ctx.get("grid", "pair")))
        elif leaf == "msa_att_row":
            h = _out(sh, p + "linear_z.weight")
            out.append(PairCore("row_att", heads=h, dim=_out(sh, p + "mha.linear_q.weight") // h,
                                over="xmsa" if xm else "msa"))
        elif leaf == "msa_att_col":
            if p + "global_attention.linear_q.weight" in sh:
                c = _out(sh, p + "global_attention.linear_k.weight")
                out.append(PairCore("col_att", heads=_out(sh, p + "global_attention.linear_q.weight") // c, dim=c,
                                    over="xmsa" if xm else "msa", glob=True))
            else:
                q = _out(sh, p + "_msa_att.mha.linear_q.weight", p + "mha.linear_q.weight")
                c = ctx.get("c_hidden_msa_att", 32)
                out.append(PairCore("col_att", heads=q // c, dim=c, over="xmsa" if xm else "msa"))
        elif leaf in ("outer_product_mean", "outer_product_mean_msa"):
            out.append(PairCore("opm", dim=_out(sh, p + "linear_1.weight", p + "proj_a.weight"),
                                dim2=_out(sh, p + "linear_2.weight", p + "proj_b.weight"), over="xmsa" if xm else "msa"))
        elif leaf in ("pair_weighted_averaging", "msa_pair_weighted_averaging"):
            h = _out(sh, p + "linear_no_bias_z.weight", p + "proj_z.weight")
            out.append(PairCore("pwa", heads=h,
                                dim=_out(sh, p + "linear_no_bias_mv.weight", p + "proj_m.weight") // h))
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
        f"；激活按 bf16「假设」（{_PREC[key]}；fp32 激活的双倍流量未建模）；芯片无 fp32 MAC → 逐 GEMM 转换为 bf16，开销计入向量单元",
        "TP / SP（pair 表示的 DAP 切分）未建模：布局只取 PP × DP（DP = 多条序列并行）"]
    pc = (spec.release_params - spec.params()) / spec.release_params
    if abs(pc) > 0.001:
        reasons = list(reasons) + [f"参数与发布相差 {pc:+.2%}"]
    return replace(spec, notes=tuple(notes), coverage_reasons=tuple(reasons),
                   coverage="partial" if reasons else "full")


def _apply_defaults(wl, **kw):
    return replace(wl, **{k: v for k, v in kw.items() if not getattr(wl, k)})


def build_esmfold(model_id: str, hf_id: str, rel: dict, wl) -> ModelSpec:
    from .domain import _gelu_ffn, _mha
    c, sh = rel["config"], rel["shapes"]
    d, H, f, L = c["hidden_size"], c["num_attention_heads"], c["intermediate_size"], c["num_hidden_layers"]
    t = c["esmfold_config"]["trunk"]
    alin, core = _mha(d, H, rope=d // H)
    ffn, flin = _gelu_ffn(d, f)
    # misc: q/k/v/o biases, 2 layer norms, FFN biases + the rotary inv_freq buffer stored (fp32) in the checkpoint
    esm_layer = Layer(alin, core, ffn, flin, 4 * d + 2 * d + f + d + 2 * d + (d // H) // 2, stack="esm")
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
    max_pos = c.get("max_position_embeddings", 1026)
    wl = _apply_defaults(wl, recycles=t["max_recycles"] + 1, pair_dim=t["pairwise_state_dim"], max_seq=max_pos - 4,
                         special_tokens=2)
    notes = [f"ESM-2 3B 语言模型（{L} 层，fp16 发布）每请求一次前向（序列 + <cls>/<eos>），其 37 层隐状态加权求和后进入折叠主干",
             f"折叠主干 {rel['stacks']['trunk.blocks']} 块（序列 1024 维 + pair {t['pairwise_state_dim']} 维；三角乘法 ×2、三角注意力 ×2、"
             "带 pair 偏置的序列注意力），每次 recycle 全部重跑；默认主干前向 "
             f"{t['max_recycles'] + 1} 次（HF / esm 参考实现 num_recycles=None → max_recycles={t['max_recycles']}，"
             "首次前向不计入 recycle）",
             f"结构模块（IPA，{t['structure_module']['num_blocks']} 次迭代共享权重）每次 recycle 后运行；IPA 按 qk = 标量 + 点坐标、"
             "v = 标量 + 点 + pair 值的注意力计，刚体帧更新 / 扭转角 / 原子坐标重建的向量运算未计",
             "无 MSA、无模板（单序列）；chunk_size 128 只降低峰值显存，不改变计算量"]
    return _spec(model_id, hf_id, rel, "ESMFold（ESM-2 3B + 折叠主干 + 结构模块）", t["sequence_state_dim"],
                 [esm_layer] * L + pl, wl, notes, [], "esmfold")


def _af2_groups(c: dict) -> list[Group]:
    return [Group("template_pair_stack.blocks.", "template", "recycle", blocks=True, grid="tmpl"),
            Group("extra_msa_stack.blocks.", "extra_msa", "recycle", blocks=True),
            Group("evoformer.blocks.", "evoformer", "recycle", blocks=True),
            Group("structure_module.", "structure", "recycle", iters=c["no_ipa_blocks"]),
            Group("aux_heads.", "heads"),
            Group("", "embed", "recycle")]


def build_alphafold2(model_id: str, hf_id: str, rel: dict, wl) -> ModelSpec:
    """AlphaFold 2 monomer (model_1_ptm), from the official JAX parameter release (names mapped to OpenFold-style
    names by scripts/summarize_ckpt.py, so the same row / core tables apply)."""
    c = rel["config"]
    pl, _ = layers_from(rel, _af2_groups(c), {"c_hidden_msa_att": c["c_hidden_msa_att"]})
    msa = c["max_msa_clusters"] - c["max_templates"]
    wl = _apply_defaults(wl, msa=msa, xmsa=c["max_extra_msa"], templates=c["max_templates"],
                         recycles=c["max_recycling_iters"] + 1, pair_dim=c["c_z"])
    notes = [f"官方 JAX 参数 params_model_1_ptm（alphafold_params_2022-12-06.tar，CC BY 4.0）：Evoformer "
             f"{rel['stacks']['evoformer.blocks']} 块（MSA {c['c_m']} 维 / pair {c['c_z']} 维）+ extra MSA 栈 "
             f"{rel['stacks']['extra_msa_stack.blocks']} 块（全局列注意力）+ 模板 pair 栈 "
             f"{rel['stacks']['template_pair_stack.blocks']} 块 + 结构模块（IPA × {c['no_ipa_blocks']}，共享权重）",
             f"默认（model_1_ptm）：MSA 聚类 {msa} 行（512 − 模板数）、extra MSA {c['max_extra_msa']} 行、模板 "
             f"{c['max_templates']} 个；num_recycle {c['max_recycling_iters']} → 主干 {c['max_recycling_iters'] + 1} 遍；"
             "MSA 行数按上限计「假设」（浅 MSA 更快）",
             "monomer 预设跑 5 个模型（model_1…5_ptm）各一次：这里评估单个模型的一次预测；5 模型 ≈ 5 倍（model_3–5 "
             "无模板，略少）",
             "模板扭转角嵌入按每残基一行计（实际 T × N 行，量很小）；IPA 几何运算同 ESMFold 的近似；subbatch / "
             "全局分块只降低峰值显存"]
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
    wl = _apply_defaults(wl, msa=c["max_msa_clusters"], xmsa=c["max_extra_msa"], templates=c["max_templates"],
                         recycles=c["max_recycling_iters"] + 1, pair_dim=c["c_z"])
    notes = [f"AF2 架构（OpenFold 复现权重 finetuning_ptm_2）：Evoformer {rel['stacks']['evoformer.blocks']} 块（MSA {c['c_m']} 维 / "
             f"pair {c['c_z']} 维）+ extra MSA 栈 {rel['stacks']['extra_msa_stack.blocks']} 块（全局列注意力）+ 模板 pair 栈 "
             f"{rel['stacks']['template_pair_stack.blocks']} 块 + 结构模块（IPA × {c['no_ipa_blocks']}）",
             f"默认：MSA 聚类 {c['max_msa_clusters']} 行、extra MSA {c['max_extra_msa']} 行、模板 {c['max_templates']} 个"
             f"（data.predict）；recycle {c['max_recycling_iters']} 次 → 主干 {c['max_recycling_iters'] + 1} 遍（嵌入、模板、"
             "extra MSA、Evoformer、结构模块每遍都重跑）；MSA 行数按上限计「假设」（浅 MSA 更快）",
             "模板扭转角嵌入按每残基一行计（实际 T × N 行，量很小）；IPA 几何运算同 ESMFold 的近似"]
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
             "扩散模块按参考算法每步计算 pair 条件与每层 pair 偏置（与样本无关、可跨步缓存，未作为优化建模）；原子数 = 残基 × "
             f"{wl.atoms_per_res:g}「假设」（仅蛋白质重原子）；原子级输入 / 输出投影按原子行计（部分实为 token 行，量很小）"]
    reasons = ["只评估神经网络推理：MSA 检索（MMseqs2 服务器）与特征化 / 分子预处理未建模"]
    return _spec(model_id, hf_id, rel, "Boltz-1（AF3 类：MSA 模块 + Pairformer + 扩散）", hp["token_s"], pl, wl, notes,
                 reasons, "boltz1")


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
    wl = _apply_defaults(wl, msa=c["msa_sample_cutoff_test"], recycles=c["N_cycle"], diff_steps=c["N_step"],
                         samples=c["N_sample"], pair_dim=c["c_z"])
    notes = [f"AF3 复现（v0.5.0）：MSA 模块 {c['msa_blocks']} 块 + Pairformer {c['pairformer_blocks']} 块（每个 cycle 重跑）+ "
             f"扩散模块（diffusion transformer {c['diffusion_blocks']} 层，{c['c_token']} 维 + 原子编码 / 解码各 "
             f"{c['atom_blocks']} 层，窗口 32×128 原子）+ 置信度头（Pairformer {c['confidence_blocks']} 块，每个样本一遍）",
             f"默认（configs_base / configs_data）：N_cycle {c['N_cycle']}、N_step {c['N_step']}、N_sample {c['N_sample']}"
             f"（样本批量并行）、MSA 采样 {c['msa_sample_cutoff_test']} 行（按上限计「假设」）；模板关闭（template_embedder "
             "权重只计存储）",
             "扩散模块按参考算法每步计算 pair 条件与每层 pair 偏置（与样本无关、可跨步缓存，未作为优化建模）；原子数 = 残基 × "
             f"{wl.atoms_per_res:g}「假设」（仅蛋白质重原子）；原子级输入 / 输出投影按原子行计（部分实为 token 行，量很小）"]
    reasons = ["只评估神经网络推理：MSA 检索与特征化 / 分子预处理未建模"]
    return _spec(model_id, hf_id, rel, "Protenix（AF3 复现：MSA 模块 + Pairformer + 扩散）", c["c_s"], pl, wl, notes,
                 reasons, "protenix")


BUILDERS = {"esmfold": build_esmfold, "openfold": build_openfold, "alphafold2": build_alphafold2, "boltz1": build_boltz1, "protenix": build_protenix}
