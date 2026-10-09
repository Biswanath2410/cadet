#!/usr/bin/env python3

import os
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# TODO: after a few more samples, check whether permissive calls need a separate cutoff.
# Basic helpers

def clean_token(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "na", "null"}:
        return np.nan
    return s

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default

def safe_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}

def split_semicolon_list(x) -> List[str]:
    if pd.isna(x):
        return []
    vals = []
    for part in str(x).replace(",", ";").split(";"):
        part = clean_token(part)
        if pd.notna(part):
            vals.append(str(part))
    return vals

def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = clean_token(item)
        if pd.isna(item):
            continue
        if item not in seen:
            seen.add(item)
            out.append(str(item))
    return out

def is_ensembl_gene(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).startswith("ENSG")

def choose_readable(primary, fallback=np.nan):
    primary = clean_token(primary)
    fallback = clean_token(fallback)
    if pd.notna(primary) and not is_ensembl_gene(primary):
        return primary
    if pd.notna(fallback):
        return fallback
    return primary

def first_existing(df: pd.DataFrame, names: List[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)

def first_non_null(vals):
    for v in vals:
        v = clean_token(v)
        if pd.notna(v):
            return v
    return np.nan

def numeric(series: pd.Series, default=np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)

def weighted_vote(df: pd.DataFrame, name_col: str, weight_col: str) -> pd.DataFrame:
    if name_col not in df.columns or weight_col not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=[name_col, "n_support", "weighted_support"])
    tmp = df[[name_col, weight_col]].copy()
    tmp[name_col] = tmp[name_col].map(clean_token)
    tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(0.0)
    tmp = tmp.dropna(subset=[name_col])
    if len(tmp) == 0:
        return pd.DataFrame(columns=[name_col, "n_support", "weighted_support"])
    return (tmp.groupby(name_col, dropna=True) .agg(n_support=(name_col, "size"), weighted_support=(weight_col, "sum")) .reset_index().sort_values(["weighted_support", "n_support", name_col], ascending=[False, False, True]).reset_index(drop=True))

# Loading and column cleanup

def optional_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_module_membership(input_dir: str) -> Tuple[pd.DataFrame, str]:
    module_dir = os.path.join(input_dir, "module_consolidation")
    preferred = os.path.join(module_dir, "significant_blocks_with_module_ids.csv")
    fallback = os.path.join(module_dir, "significant_peaks_with_module_ids.csv")
    if os.path.exists(preferred):
        return pd.read_csv(preferred), preferred
    if os.path.exists(fallback):
        return pd.read_csv(fallback), fallback
    return pd.DataFrame(), preferred

def load_block_annotation(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Layer 6 block annotation: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise RuntimeError("Layer 6 block_functional_annotation.csv is empty.")
    if "module_id" not in df.columns:
        raise ValueError("block_functional_annotation.csv must contain module_id")
    out = df.copy()
    if "peak_blk_id" not in out.columns:
        out["peak_blk_id"] = np.arange(len(out), dtype=int)
    # label cleanup from different layer 6 versions
    out["blk_primary_label"] = first_existing(out,["primary_blk_label", "upgraded_peak_block_class", "representative_upgraded_peak_class", "module_label"],"unclassified_significant_contact").astype(str)

    # coordinates / structural flags
    for c in ["chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j"]:
        if c not in out.columns:
            out[c] = np.nan
    out["near_junction"] = first_existing(out, ["near_junction", "any_near_junction"], False).map(safe_bool)
    out["near_discordant_join"] = first_existing(out, ["near_discordant_join", "any_near_discordant_join"], False).map(safe_bool)
    out["is_interchromosomal"] = first_existing(out, ["is_interchromosomal"], False).map(safe_bool)
    out["segment_relation"] = first_existing(out, ["segment_relation"], "unknown").astype(str)

    # contact-level evidence carried forward
    out["max_obs_over_exp_raw"] = numeric(first_existing(out, ["max_obs_over_exp_raw", "obs_over_exp_raw", "rep_obs_over_exp_raw"], 1.0), 1.0)
    out["max_local_oe"] = numeric(first_existing(out, ["max_local_oe", "local_obs_over_mean"], 1.0), 1.0)
    out["max_discord"] = numeric(first_existing(out, ["max_discord", "discord"], 0.0), 0.0)
    out["max_discord_pos"] = out["max_discord"].clip(lower=0.0)

    # gene and oncogene names/distances
    for prefix in ["i", "j"]:
        out[f"near_gene_name_{prefix}"] = first_existing(out, [f"near_gene_name_{prefix}", f"anc_{prefix}_nearest_gene"], np.nan)
        out[f"near_gene_dist_{prefix}"] = numeric(first_existing(out, [f"near_gene_dist_{prefix}", f"anc_{prefix}_nearest_gene_distance_bp"], np.nan), np.nan)
        out[f"near_onco_name_{prefix}"] = first_existing(out, [f"near_onco_name_{prefix}", f"anc_{prefix}_nearest_oncogene"], np.nan)
        out[f"near_onco_dist_{prefix}"] = numeric(first_existing(out, [f"near_onco_dist_{prefix}", f"anc_{prefix}_nearest_oncogene_distance_bp"], np.nan), np.nan)

    out["best_blk_gene"] = first_existing(out, ["best_blk_gene", "cand_gene_name", "anc_i_nearest_gene"], np.nan)
    out["best_blk_onco"] = first_existing(out, ["best_blk_onco", "cand_onco_name", "anc_i_nearest_oncogene"], np.nan)

    # anchor role flags
    out["anc_i_enh_main"] = first_existing(out, ["anc_i_has_moderate_enh", "anc_i_is_enhancer_like_main", "anc_i_enhancer_like"], False).map(safe_bool)
    out["anc_j_enh_main"] = first_existing(out, ["anc_j_has_moderate_enh", "anc_j_is_enhancer_like_main", "anc_j_enhancer_like"], False).map(safe_bool)
    out["anc_i_enh_any"] = first_existing(out, ["anc_i_has_any_enh", "anc_i_enhancer_like"], False).map(safe_bool)
    out["anc_j_enh_any"] = first_existing(out, ["anc_j_has_any_enh", "anc_j_enhancer_like"], False).map(safe_bool)
    out["anc_i_onco_like"] = first_existing(out, ["anc_i_onco_like", "anc_i_is_oncogene_like"], False).map(safe_bool)
    out["anc_j_onco_like"] = first_existing(out, ["anc_j_onco_like", "anc_j_is_oncogene_like"], False).map(safe_bool)
    out["anc_i_gene_like"] = first_existing(out, ["anc_i_gene_like", "anc_i_is_gene_like"], False).map(safe_bool)
    out["anc_j_gene_like"] = first_existing(out, ["anc_j_gene_like", "anc_j_is_gene_like"], False).map(safe_bool)
    out["anc_i_prom_like"] = first_existing(out, ["anc_i_promoter_like_strict", "anc_i_is_promoter_like"], False).map(safe_bool)
    out["anc_j_prom_like"] = first_existing(out, ["anc_j_promoter_like_strict", "anc_j_is_promoter_like"], False).map(safe_bool)

    # enhancer support around each anchor
    out["anc_i_enh_supp_tier"] = first_existing(out, ["anc_i_enh_supp_tier"], "unknown").astype(str)
    out["anc_j_enh_supp_tier"] = first_existing(out, ["anc_j_enh_supp_tier"], "unknown").astype(str)
    out["rel_enh_burden_tier"] = first_existing(out, ["rel_enh_burden_tier"], "unknown").astype(str)
    out["n_local_enh_like_i"] = numeric(first_existing(out, ["n_local_enh_like_i"], 0), 0)
    out["n_local_enh_like_j"] = numeric(first_existing(out, ["n_local_enh_like_j"], 0), 0)
    out["n_local_enh_like_total"] = numeric(first_existing(out, ["n_local_enh_like_total"], out["n_local_enh_like_i"] + out["n_local_enh_like_j"]), 0)
    out["best_anchor_enh_per10kb"] = numeric(first_existing(out, ["best_anchor_enh_per10kb"], np.nan), np.nan)

    # keep strict and permissive logic separate here
    out["pair_enh_gene"] = first_existing(out, ["pair_enh_gene"], False).map(safe_bool)
    out["pair_enh_onco"] = first_existing(out, ["pair_enh_onco"], False).map(safe_bool)
    out["pair_enh_onco_perm"] = first_existing(out, ["pair_enh_onco_perm"], False).map(safe_bool)
    out["pair_prom_disrupt"] = first_existing(out, ["pair_prom_disrupt"], False).map(safe_bool)

    # local regulatory context flags
    out["any_local_onco"] = first_existing(out, ["any_local_onco"], False).map(safe_bool)
    out["any_local_enh"] = first_existing(out, ["any_local_enh"], out["n_local_enh_like_total"] > 0).map(safe_bool)
    out["any_local_prom"] = first_existing(out, ["any_local_prom"], False).map(safe_bool)
    out["any_local_ctcf"] = first_existing(out, ["any_local_ctcf"], False).map(safe_bool)

    return out

def load_modules(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Layer 6 prioritized modules: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise RuntimeError("Layer 6 prioritized_functional_modules.csv is empty.")
    if "module_id" not in df.columns:
        raise ValueError("prioritized_functional_modules.csv must contain module_id")

    out = df.copy()
    if "func_rank" in out.columns:
        out["func_rank"] = pd.to_numeric(out["func_rank"], errors="coerce")
        missing_rank = out["func_rank"].isna()
        if missing_rank.any():
            out.loc[missing_rank, "func_rank"] = np.arange(1, int(missing_rank.sum()) + 1)
        out["func_rank"] = out["func_rank"].astype(int)
    else:
        out["func_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["module_label"] = first_existing(out, ["module_label", "dominant_label_disp", "module_label_disp"], "unknown").astype(str)
    out["module_type_disp"] = first_existing(out, ["module_type_disp", "grouping_type_disp"], "unknown").astype(str)
    out["module_label_disp"] = first_existing(out, ["module_label_disp", "dominant_label_disp"], out["module_label"]).astype(str)
    out["func_recommendation"] = first_existing(out, ["func_recommendation"], "unresolved_functional_context").astype(str)
    out["priority"] = numeric(first_existing(out, ["priority"], 0.0), 0.0)
    out["func_priority"] = numeric(first_existing(out, ["func_priority"], 0.0), 0.0)
    out["combined_priority"] = numeric(first_existing(out, ["combined_priority"], out["priority"]), 0.0)
    out["n_member_blk_func"] = safe_numeric_col(out, ["n_member_blk_func", "n_member_blk", "n_member_peaks"], 0).astype(int)
    out["n_pair_enh_gene"] = safe_numeric_col(out, ["n_pair_enh_gene"], 0).astype(int)
    out["n_pair_enh_onco"] = safe_numeric_col(out, ["n_pair_enh_onco"], 0).astype(int)
    out["n_pair_enh_onco_perm"] = safe_numeric_col(out, ["n_pair_enh_onco_perm"], 0).astype(int)
    out["n_pair_prom_disrupt"] = safe_numeric_col(out, ["n_pair_prom_disrupt"], 0).astype(int)
    out["max_discord_func"] = numeric(first_existing(out, ["max_discord_func"], 0.0), 0.0)
    out["max_obs_over_exp_raw_func"] = numeric(first_existing(out, ["max_obs_over_exp_raw_func", "max_obs_over_exp_raw"], 1.0), 1.0)
    out["n_uniq_cand_oncos"] = safe_numeric_col(out, ["n_uniq_cand_oncos"], 0).astype(int)
    out["module_oncos"] = first_existing(out, ["module_oncos"], "").astype(str)
    out["n_uniq_local_oncos"] = safe_numeric_col(out, ["n_uniq_local_oncos"], 0).astype(int)
    out["n_enh_anchors"] = safe_numeric_col(out, ["n_enh_anchors"], 0).astype(int)
    return out

def safe_numeric_col(df: pd.DataFrame, names: List[str], default) -> pd.Series:
    return numeric(first_existing(df, names, default), default)

# Block-level interpretation

def side_label(i: bool, j: bool) -> str:
    if i and j:
        return "both"
    if i:
        return "i"
    if j:
        return "j"
    return "none"

def add_side_labels(row: pd.Series) -> Dict[str, str]:
    return {"enh_anchor_side_main": side_label(safe_bool(row.get("anc_i_enh_main")), safe_bool(row.get("anc_j_enh_main"))),
        "enh_anchor_side_any": side_label(safe_bool(row.get("anc_i_enh_any")), safe_bool(row.get("anc_j_enh_any"))),
        "gene_anchor_side": side_label(safe_bool(row.get("anc_i_gene_like")), safe_bool(row.get("anc_j_gene_like"))),
        "onco_anchor_side": side_label(safe_bool(row.get("anc_i_onco_like")), safe_bool(row.get("anc_j_onco_like"))),
        "prom_anchor_side": side_label(safe_bool(row.get("anc_i_prom_like")), safe_bool(row.get("anc_j_prom_like")))}

def interpret_block_logic(row: pd.Series) -> str:
    strict_onco = safe_bool(row.get("pair_enh_onco", False))
    perm_onco = safe_bool(row.get("pair_enh_onco_perm", False))
    enh_gene = safe_bool(row.get("pair_enh_gene", False))
    prom = safe_bool(row.get("pair_prom_disrupt", False))

    if strict_onco:
        return "strict_enhancer_oncogene_block"
    if prom:
        return "promoter_disruption_block"
    if perm_onco:
        return "permissive_enhancer_oncogene_context_block"
    if enh_gene:
        return "enhancer_gene_block"
    if safe_bool(row.get("any_local_onco")) or safe_bool(row.get("any_local_enh")) or safe_bool(row.get("any_local_prom")) or safe_bool(row.get("any_local_ctcf")):
        return "local_regulatory_context_block"
    return "unresolved_block"

def choose_best_target(row: pd.Series, target_type: str) -> Tuple[object, object, object]:
    candidates = []
    if target_type == "oncogene":
        for s in ["i", "j"]:
            name = choose_readable(row.get(f"near_onco_name_{s}", np.nan), row.get("best_blk_onco", np.nan))
            dist = safe_float(row.get(f"near_onco_dist_{s}", np.nan), np.inf)
            is_like = safe_bool(row.get(f"anc_{s}_onco_like", False))
            if pd.notna(name):
                candidates.append((0 if is_like else 1, dist, str(name), s))
    else:
        for s in ["i", "j"]:
            name = choose_readable(row.get(f"near_gene_name_{s}", np.nan), row.get("best_blk_gene", np.nan))
            dist = safe_float(row.get(f"near_gene_dist_{s}", np.nan), np.inf)
            is_like = safe_bool(row.get(f"anc_{s}_gene_like", False))
            if pd.notna(name):
                candidates.append((0 if is_like else 1, dist, str(name), s))
    if not candidates:
        return np.nan, np.nan, np.nan
    candidates = sorted(candidates, key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][2], candidates[0][1], candidates[0][3]

def pick_opposite_oncogene(row: pd.Series, strict: bool = True) -> Tuple[object, object, object]:
    enh_i = safe_bool(row.get("anc_i_enh_main" if strict else "anc_i_enh_any"))
    enh_j = safe_bool(row.get("anc_j_enh_main" if strict else "anc_j_enh_any"))
    onco_i = safe_bool(row.get("anc_i_onco_like"))
    onco_j = safe_bool(row.get("anc_j_onco_like"))

    candidates = []
    if enh_i and onco_j:
        candidates.append((safe_float(row.get("near_onco_dist_j", np.nan), np.inf), choose_readable(row.get("near_onco_name_j", np.nan), row.get("best_blk_onco", np.nan)), "j"))
    if enh_j and onco_i:
        candidates.append((safe_float(row.get("near_onco_dist_i", np.nan), np.inf), choose_readable(row.get("near_onco_name_i", np.nan), row.get("best_blk_onco", np.nan)), "i"))
    candidates = [(d, n, s) for d, n, s in candidates if pd.notna(n)]
    if candidates:
        candidates = sorted(candidates, key=lambda x: (x[0], str(x[1])))
        return candidates[0][1], candidates[0][0], candidates[0][2]
    return choose_best_target(row, "oncogene")

def determine_block_targets(row: pd.Series) -> Dict[str, object]:
    logic = interpret_block_logic(row)
    gene_name, gene_dist, gene_side = choose_best_target(row, "gene")

    if logic == "strict_enhancer_oncogene_block":
        onco_name, onco_dist, onco_side = pick_opposite_oncogene(row, strict=True)
    elif logic == "permissive_enhancer_oncogene_context_block":
        onco_name, onco_dist, onco_side = pick_opposite_oncogene(row, strict=False)
    else:
        onco_name, onco_dist, onco_side = choose_best_target(row, "oncogene")

    if logic in {"strict_enhancer_oncogene_block", "permissive_enhancer_oncogene_context_block"}:
        preferred_type, preferred_name, preferred_side, preferred_dist = "oncogene", onco_name, onco_side, onco_dist
    elif logic in {"enhancer_gene_block", "promoter_disruption_block"}:
        preferred_type, preferred_name, preferred_side, preferred_dist = "gene_or_promoter", gene_name, gene_side, gene_dist
    else:
        preferred_type = "context"
        preferred_name = first_non_null([onco_name, gene_name])
        preferred_side = first_non_null([onco_side, gene_side])
        preferred_dist = first_non_null([onco_dist, gene_dist])

    return {"blk_logic_label": logic, "cand_gene_name": gene_name, "cand_gene_dist": gene_dist, "cand_gene_side": gene_side, "cand_onco_name": onco_name, "cand_onco_dist": onco_dist,  "cand_onco_side": onco_side, "preferred_target_type": preferred_type, "preferred_target_name": preferred_name,  "preferred_target_side": preferred_side, "preferred_target_dist": preferred_dist}

def build_block_evidence(block_func_df: pd.DataFrame, modules_df: pd.DataFrame) -> pd.DataFrame:
    blk = block_func_df.copy()
    merge_cols = [
        "module_id", "func_rank", "module_label", "module_type_disp", "module_label_disp",
        "priority", "func_priority", "combined_priority", "func_recommendation",
        "n_pair_enh_onco", "n_pair_enh_onco_perm", "n_pair_enh_gene", "n_pair_prom_disrupt",
        "n_uniq_cand_oncos", "module_oncos",
    ]
    merge_cols = [c for c in merge_cols if c in modules_df.columns]
    blk = blk.merge(modules_df[merge_cols], on="module_id", how="left", suffixes=("", "_module"))

    side_df = pd.DataFrame([add_side_labels(row) for _, row in blk.iterrows()])
    target_df = pd.DataFrame([determine_block_targets(row) for _, row in blk.iterrows()])
    blk = pd.concat([blk.reset_index(drop=True), side_df, target_df], axis=1)

    oe = np.log2(np.maximum(pd.to_numeric(blk["max_obs_over_exp_raw"], errors="coerce").fillna(1.0), 1.0) + 1.0)
    disc = pd.to_numeric(blk["max_discord_pos"], errors="coerce").fillna(0.0)
    strict_bonus = blk["blk_logic_label"].eq("strict_enhancer_oncogene_block").astype(float) * 0.50
    prom_bonus = blk["blk_logic_label"].eq("promoter_disruption_block").astype(float) * 0.25
    blk["blk_supp_weight"] = oe * (1.0 + disc + strict_bonus + prom_bonus)

    return blk

# Module-level summaries

def target_support_table(block_evidence_df: pd.DataFrame, logic_set: set, target_col: str, output_prefix: str) -> pd.DataFrame:
    rows = []
    for module_id, grp in block_evidence_df.groupby("module_id", sort=False):
        use = grp[grp["blk_logic_label"].isin(logic_set)].copy()
        vote = weighted_vote(use, target_col, "blk_supp_weight")
        if len(vote) == 0:
            rows.append({
                "module_id": module_id, f"top_{output_prefix}": np.nan, f"top_{output_prefix}_n_supp": 0,
                f"top_{output_prefix}_wsupp": 0.0, f"{output_prefix}_list": ""
            })
        else:
            rows.append({"module_id": module_id, f"top_{output_prefix}": vote.iloc[0][target_col],
                f"top_{output_prefix}_n_supp": int(vote.iloc[0]["n_support"]),
                f"top_{output_prefix}_wsupp": float(vote.iloc[0]["weighted_support"]),
                f"{output_prefix}_list": ";".join(vote[target_col].astype(str).tolist())
            })
    return pd.DataFrame(rows)

def pick_evidence_type(mod: pd.Series, grp: pd.DataFrame) -> str:
    n_strict = safe_int(mod.get("n_pair_enh_onco", grp["blk_logic_label"].eq("strict_enhancer_oncogene_block").sum()))
    n_perm = safe_int(mod.get("n_pair_enh_onco_perm", grp["blk_logic_label"].eq("permissive_enhancer_oncogene_context_block").sum()))
    n_gene = safe_int(mod.get("n_pair_enh_gene", grp["blk_logic_label"].eq("enhancer_gene_block").sum()))
    n_prom = safe_int(mod.get("n_pair_prom_disrupt", grp["blk_logic_label"].eq("promoter_disruption_block").sum()))
    rec = str(mod.get("func_recommendation", ""))

    if n_strict > 0:
        return "strict_enhancer_oncogene"
    if n_prom > 0 and "promoter" in rec:
        return "promoter_disruption"
    if n_perm > 0:
        return "permissive_enhancer_oncogene_context"
    if n_gene > 0:
        return "enhancer_gene_context"
    if grp["blk_logic_label"].isin(["local_regulatory_context_block"]).any():
        return "local_regulatory_context"
    return "weak_or_unresolved"

def summarize_modules(block_evidence_df: pd.DataFrame, modules_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_mod = {m: g.copy() for m, g in block_evidence_df.groupby("module_id", sort=False)}

    for _, mod in modules_df.sort_values("func_rank").iterrows():
        module_id = mod["module_id"]
        grp = by_mod.get(module_id, pd.DataFrame())
        if len(grp) > 0:
            top_block = grp.sort_values(["blk_supp_weight", "max_obs_over_exp_raw", "max_discord_pos"], ascending=[False, False, False]).iloc[0]
        else:
            top_block = pd.Series(dtype=object)

        main_ev = pick_evidence_type(mod, grp) if len(grp) > 0 else "weak_or_unresolved"

        rows.append({"module_id": module_id, "func_rank": safe_int(mod.get("func_rank", np.nan), 9999),
            "module_type_disp": mod.get("module_type_disp", "unknown"),
            "module_label_disp": mod.get("module_label_disp", mod.get("module_label", "unknown")),
            "func_recommendation": mod.get("func_recommendation", "unresolved_functional_context"), "main_evidence_type": main_ev,
            "n_member_blk_func": safe_int(mod.get("n_member_blk_func", len(grp)), len(grp)),
            "n_pair_enh_gene": safe_int(mod.get("n_pair_enh_gene", 0)), "n_pair_enh_onco_strict": safe_int(mod.get("n_pair_enh_onco", 0)),
            "n_pair_enh_onco_perm": safe_int(mod.get("n_pair_enh_onco_perm", 0)),
            "n_pair_prom_disrupt": safe_int(mod.get("n_pair_prom_disrupt", 0)),
            "n_uniq_cand_oncos": safe_int(mod.get("n_uniq_cand_oncos", 0)),
            "module_oncos": mod.get("module_oncos", ""),
            "priority": safe_float(mod.get("priority", np.nan), np.nan),
            "func_priority": safe_float(mod.get("func_priority", np.nan), np.nan),
            "combined_priority": safe_float(mod.get("combined_priority", np.nan), np.nan),
            "max_discord_func": safe_float(mod.get("max_discord_func", np.nan), np.nan),
            "max_obs_over_exp_raw_func": safe_float(mod.get("max_obs_over_exp_raw_func", np.nan), np.nan),
            "top_blk_id": top_block.get("peak_blk_id", np.nan),
            "top_blk_label": top_block.get("blk_primary_label", np.nan),
            "top_blk_logic": top_block.get("blk_logic_label", np.nan),
            "top_blk_obs_over_exp_raw": safe_float(top_block.get("max_obs_over_exp_raw", np.nan), np.nan),
            "top_blk_discord": safe_float(top_block.get("max_discord", np.nan), np.nan),
            "top_blk_enh_tier": top_block.get("rel_enh_burden_tier", np.nan)})
    return pd.DataFrame(rows)

def clean_list_str(x) -> str:
    vals = unique_preserve_order(split_semicolon_list(x))
    non_ens = [v for v in vals if not is_ensembl_gene(v)]
    ens = [v for v in vals if is_ensembl_gene(v)]
    return ";".join(non_ens + ens)

def add_target_columns(module_summary: pd.DataFrame, gene_support: pd.DataFrame, onco_support: pd.DataFrame, onco_perm_support: pd.DataFrame) -> pd.DataFrame:
    out = module_summary.copy()
    out = out.merge(gene_support, on="module_id", how="left")
    out = out.merge(onco_support, on="module_id", how="left")
    out = out.merge(onco_perm_support, on="module_id", how="left", suffixes=("", "_permissive"))

    for col in ["gene_list", "onco_list", "onco_perm_list", "module_oncos"]:
        if col in out.columns:
            out[f"{col}_clean"] = out[col].map(clean_list_str)

    # if the strict vote is empty, keep the module-level candidate list as context
    def final_target(row):
        ev = str(row.get("main_evidence_type", ""))
        strict_onco = clean_token(row.get("top_onco", np.nan))
        perm_onco = clean_token(row.get("top_onco_perm", np.nan))
        gene = clean_token(row.get("top_gene", np.nan))
        mod_onco = split_semicolon_list(row.get("module_oncos", ""))
        mod_onco = mod_onco[0] if mod_onco else np.nan

        if ev == "strict_enhancer_oncogene" and pd.notna(strict_onco):
            return strict_onco
        if ev == "permissive_enhancer_oncogene_context" and pd.notna(perm_onco):
            return perm_onco
        if ev in {"enhancer_gene_context", "promoter_disruption"} and pd.notna(gene):
            return gene
        return first_non_null([strict_onco, perm_onco, mod_onco, gene])

    out["final_target_name"] = out.apply(final_target, axis=1)
    return out

def add_interpretation_text(module_df: pd.DataFrame) -> pd.DataFrame:
    out = module_df.copy()
    texts = []
    for _, row in out.iterrows():
        mid = row.get("module_id")
        rank = row.get("func_rank")
        ev = str(row.get("main_evidence_type", "weak_or_unresolved"))
        rec = str(row.get("func_recommendation", "unresolved"))
        target = clean_token(row.get("final_target_name", np.nan))
        onco_names = clean_list_str(row.get("module_oncos", ""))
        n_strict = safe_int(row.get("n_pair_enh_onco_strict", 0))
        n_perm = safe_int(row.get("n_pair_enh_onco_perm", 0))
        n_gene = safe_int(row.get("n_pair_enh_gene", 0))
        n_prom = safe_int(row.get("n_pair_prom_disrupt", 0))
        n_blocks = safe_int(row.get("n_member_blk_func", 0))
        disc = safe_float(row.get("max_discord_func", np.nan), np.nan)
        comb = safe_float(row.get("combined_priority", np.nan), np.nan)

        parts = [
            f"Module {mid} ranks {rank} by Layer 6 functional prioritization.",
            f"It is classified as {rec} with main evidence type {ev}.",
            f"The module contains {n_blocks} supporting contact blocks; strict enhancer-oncogene blocks={n_strict}, permissive enhancer-oncogene-context blocks={n_perm}, enhancer-gene blocks={n_gene}, promoter-disruption blocks={n_prom}.",
        ]
        if pd.notna(target):
            parts.append(f"Final nominated target/context gene is {target}.")
        if onco_names:
            parts.append(f"Candidate oncogene context: {onco_names}.")
        if np.isfinite(disc):
            parts.append(f"Maximum distance-discordance score is {disc:.3f}.")
        if np.isfinite(comb):
            parts.append(f"Combined functional priority score is {comb:.3f}.")
        texts.append(" ".join(parts))
    out["interpretation_summary"] = texts
    return out

def make_clean_reports(block_evidence: pd.DataFrame, module_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    block_cols = ["module_id", "func_rank", "peak_blk_id", "blk_primary_label", "blk_logic_label",
        "enh_anchor_side_main", "enh_anchor_side_any", "gene_anchor_side", "onco_anchor_side", "prom_anchor_side","cand_gene_name", "cand_gene_dist", "cand_gene_side",
        "cand_onco_name", "cand_onco_dist", "cand_onco_side",
        "preferred_target_type", "preferred_target_name", "preferred_target_side", "preferred_target_dist",
        "rel_enh_burden_tier", "best_anchor_enh_per10kb",
        "max_obs_over_exp_raw", "max_discord", "near_junction", "near_discordant_join", "blk_supp_weight"]
    block_cols = [c for c in block_cols if c in block_evidence.columns]
    block_report = block_evidence[block_cols].copy()

    module_cols = ["module_id", "func_rank", "module_type_disp", "module_label_disp", "func_recommendation", "main_evidence_type",
        "final_target_name", "top_onco", "top_onco_n_supp", "top_onco_wsupp",
        "top_onco_perm", "top_onco_perm_n_supp", "top_gene", "top_gene_n_supp",
        "module_oncos", "n_uniq_cand_oncos", "n_member_blk_func",
        "n_pair_enh_onco_strict", "n_pair_enh_onco_perm", "n_pair_enh_gene", "n_pair_prom_disrupt",
        "priority", "func_priority", "combined_priority", "max_discord_func",
        "top_blk_id", "top_blk_logic", "interpretation_summary"]
    module_cols = [c for c in module_cols if c in module_summary.columns]
    module_report = module_summary[module_cols].copy()

    for df in [block_report, module_report]:
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].replace("", np.nan).fillna("NA")
    return block_report, module_report

def write_narratives(module_summary: pd.DataFrame, outpath: str, top_n: int) -> None:
    use = module_summary.sort_values("func_rank").head(top_n)
    with open(outpath, "w") as f:
        for _, row in use.iterrows():
            f.write(f"Module {row['module_id']} (functional rank {row['func_rank']})\n")
            f.write(f"Grouping/type: {row.get('module_type_disp', 'NA')}\n")
            f.write(f"Dominant contact label: {row.get('module_label_disp', 'NA')}\n")
            f.write(f"Recommendation: {row.get('func_recommendation', 'NA')}\n")
            f.write(f"Main evidence type: {row.get('main_evidence_type', 'NA')}\n")
            f.write(f"Final target/context gene: {row.get('final_target_name', 'NA')}\n")
            f.write(f"Strict enhancer-oncogene blocks: {row.get('n_pair_enh_onco_strict', 0)}\n")
            f.write(f"Permissive enhancer-oncogene-context blocks: {row.get('n_pair_enh_onco_perm', 0)}\n")
            f.write(f"Enhancer-gene blocks: {row.get('n_pair_enh_gene', 0)}\n")
            f.write(f"Promoter-disruption blocks: {row.get('n_pair_prom_disrupt', 0)}\n")
            f.write(f"Candidate oncogene context: {row.get('module_oncos', 'NA')}\n")
            f.write(f"Summary: {row.get('interpretation_summary', '')}\n")
            f.write("\n" + "=" * 80 + "\n\n")

# Main

def main() -> None:
    ap = argparse.ArgumentParser(description="CADET Layer 7: module interpretation")
    ap.add_argument("--input_dir", required=True, help="cadet_final_results folder")
    ap.add_argument("--top_n", type=int, default=20, help="Top modules to report")
    args = ap.parse_args()

    input_dir = args.input_dir
    layer6_dir = os.path.join(input_dir, "layer6_functional_annotation")
    outdir = os.path.join(input_dir, "layer7_module_interpretation")
    os.makedirs(outdir, exist_ok=True)
    block_func_csv = os.path.join(layer6_dir, "block_functional_annotation.csv")
    modules_csv = os.path.join(layer6_dir, "prioritized_functional_modules.csv")

    print("Loading finalized Layer 6 outputs ...")
    membership_df, membership_path = load_module_membership(input_dir)
    block_func_df = load_block_annotation(block_func_csv)
    modules_df = load_modules(modules_csv)

    print("Building block-level evidence table ...")
    block_evidence = build_block_evidence(block_func_df, modules_df)

    print("Summarizing target support ...")
    gene_support = target_support_table(block_evidence, {"enhancer_gene_block", "promoter_disruption_block"}, "cand_gene_name", "gene")
    oncogene_support = target_support_table(block_evidence, {"strict_enhancer_oncogene_block"}, "cand_onco_name", "onco")
    oncogene_perm_support = target_support_table(block_evidence, {"permissive_enhancer_oncogene_context_block"}, "cand_onco_name", "onco_perm")

    print("Building module interpretation table ...")
    module_summary = summarize_modules(block_evidence, modules_df)
    module_summary = add_target_columns(module_summary, gene_support, oncogene_support, oncogene_perm_support)
    module_summary = add_interpretation_text(module_summary)
    module_summary = module_summary.sort_values(["func_rank", "combined_priority"], ascending=[True, False]).reset_index(drop=True)

    block_report, module_report = make_clean_reports(block_evidence, module_summary)

    print("Writing outputs ...")
    block_evidence.to_csv(os.path.join(outdir, "layer7_block_evidence_all_modules.csv"), index=False)
    top_ids = set(module_summary.head(args.top_n)["module_id"].tolist())
    top_block_cols = [c for c in block_report.columns if c in block_evidence.columns]
    block_evidence.loc[block_evidence["module_id"].isin(top_ids), top_block_cols].to_csv(
        os.path.join(outdir, "layer7_block_evidence_top_modules.csv"), index=False)

    gene_support.to_csv(os.path.join(outdir, "layer7_module_gene_support_table.csv"), index=False)
    oncogene_support.to_csv(os.path.join(outdir, "layer7_module_oncogene_support_table.csv"), index=False)
    oncogene_perm_support.to_csv(os.path.join(outdir, "layer7_module_oncogene_permissive_support_table.csv"), index=False)
    module_summary.to_csv(os.path.join(outdir, "layer7_module_interpretation_summary.csv"), index=False)
    block_report.to_csv(os.path.join(outdir, "layer7_block_evidence_clean_report.csv"), index=False)
    module_report.to_csv(os.path.join(outdir, "layer7_module_summary_clean_report.csv"), index=False)
    write_narratives(module_summary, os.path.join(outdir, "layer7_module_narratives.txt"), args.top_n)

    params = {"layer7_version": "publication_interpretation_v1", "input_dir": input_dir, "module_membership_file_used": membership_path, "module_membership_rows_loaded": int(len(membership_df)), "block_functional_annotation": block_func_csv,
        "prioritized_functional_modules": modules_csv, "purpose": "interpret Layer 6 annotations",
        "block_logic_labels": ["strict_enhancer_oncogene_block","permissive_enhancer_oncogene_context_block","enhancer_gene_block", "promoter_disruption_block",
        "local_regulatory_context_block", "unresolved_block"],"blk_supp_weight": "log2(max_obs_over_exp_raw + 1) * (1 + max(positive distance-discordance,0) + strict/promoter evidence bonus)","target_nomination": "strict enhancer-oncogene evidence first, then permissive context"}
    with open(os.path.join(outdir, "layer7_parameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("\nLayer 7 interpretation completed.")
    print(f"Block evidence rows: {len(block_evidence)}")
    print(f"Module summary rows: {len(module_summary)}")
    print(f"Outputs written to: {outdir}")
    show_cols = ["module_id", "func_rank", "main_evidence_type", "final_target_name", "func_recommendation",
                 "n_pair_enh_onco_strict", "n_pair_enh_onco_perm", "n_pair_enh_gene",
                 "n_pair_prom_disrupt", "module_oncos"]
    show_cols = [c for c in show_cols if c in module_summary.columns]
    print("\nTop interpreted modules:")
    print(module_summary[show_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
