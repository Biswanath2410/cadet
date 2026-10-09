#!/usr/bin/env python3

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication figure style (matched to finalized CADET Layers 1-5)
PUB_FONTSIZE = 22
PUB_TICK_FONTSIZE = 22
PUB_LEGEND_FONTSIZE = 18
PUB_DPI = 300
PUB_FIGSIZE_RECT = (12, 6)

# Label cleanup for current and older CADET outputs

# older labels -> current names
LABEL_ALIASES = {"discordant_join_associated_block": "discordant_join_associated_contact",
    "anchor_recurrent_block": "anchor_recurrent_contact",
    "focal_contact_candidate_block": "focal_contact_candidate",
    "junction_neighbor_block": "junction_neighbor_contact",
    "unclassified_significant_block": "unclassified_significant_contact",
    "discordant_join_block": "discordant_join_associated_contact",
    "discordant_join_peak": "discordant_join_associated_contact",
    "interchromosomal_rewiring_block": "discordant_join_associated_contact",
    "strong_rewiring_block": "discordant_join_associated_contact",
    "distal_proximity_block": "focal_contact_candidate",
    "candidate_loop_like_block": "focal_contact_candidate",
    "candidate_loop_like_peak": "focal_contact_candidate",
    "stripe_or_anchor_recurrent_block": "anchor_recurrent_contact",
    "stripe_or_anchor_recurrent_peak": "anchor_recurrent_contact",
    "junction_neighbor_peak": "junction_neighbor_contact",
    "ambiguous_block": "unclassified_significant_contact",
    "ambiguous_peak": "unclassified_significant_contact"}

LABEL_DISPLAY = {
    'discordant_join_associated_contact': 'Discordant-join associated', 'anchor_recurrent_contact': 'Anchor-recurrent',
    'focal_contact_candidate': 'Focal contact candidate', 'junction_neighbor_contact': 'Junction-neighbor',
    'unclassified_significant_contact': 'Unclassified significant'}

# older module types -> current names
MODULE_TYPE_ALIASES = {"discordant_join_neighborhood_module": "discordant_join_associated_module"}

MODULE_TYPE_DISPLAY = {'discordant_join_associated_module': 'Discordant-join module', 'shared_anchor_recurrent_module': 'Shared-anchor recurrent module',
    'compact_pairspace_module': 'Compact pair-space module', 'singleton_contact_module': 'Singleton contact module'}

# helpers

def normalize_chr(x) -> str:
    s = str(x).strip()
    if s == "nan":
        return s
    return s if s.startswith("chr") else "chr" + s

def normalize_gene_symbol(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().upper()

def normalize_label(x) -> str:
    s = str(x) if pd.notna(x) else "unclassified_significant_contact"
    return LABEL_ALIASES.get(s, s if s in LABEL_DISPLAY else "unclassified_significant_contact")

def normalize_module_type(x) -> str:
    s = str(x) if pd.notna(x) else "singleton_contact_module"
    return MODULE_TYPE_ALIASES.get(s, s if s in MODULE_TYPE_DISPLAY else "singleton_contact_module")

def safe_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}

def first_existing(df: pd.DataFrame, names: List[str], default=np.nan) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)

def numeric(series: pd.Series, default=np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)

def midpoint(start, end) -> float:
    return (float(start) + float(end)) / 2.0

def parse_gtf_attributes(attr_string: str) -> Dict[str, str]:
    attrs = {}
    for m in re.finditer(r'(\S+)\s+"([^"]+)"', str(attr_string)):
        attrs[m.group(1)] = m.group(2)
    return attrs

def overlap_mask(chrom_df: Optional[pd.DataFrame], start: int, end: int) -> np.ndarray:
    if chrom_df is None or len(chrom_df) == 0:
        return np.zeros(0, dtype=bool)
    return (chrom_df["end"].values >= start) & (chrom_df["start"].values <= end)

def find_overlapping_rows(chrom_df: Optional[pd.DataFrame], start: int, end: int) -> pd.DataFrame:
    if chrom_df is None or len(chrom_df) == 0:
        return pd.DataFrame(columns=[])
    return chrom_df.loc[overlap_mask(chrom_df, start, end)].copy()

def split_by_chrom(df: pd.DataFrame, chrom_col="chrom") -> Dict[str, pd.DataFrame]:
    out = {}
    if len(df) == 0:
        return out
    for chrom, grp in df.groupby(chrom_col, sort=False):
        out[str(chrom)] = grp.sort_values(["start", "end"]).reset_index(drop=True)
    return out

def savefig(fig, png_path: str, pdf_path: Optional[str] = None) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(png_path, dpi=PUB_DPI, bbox_inches="tight")
    if pdf_path:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def minmax_scale(values) -> pd.Series:
    x = pd.Series(values, dtype="float64")
    out = pd.Series(np.zeros(len(x)), index=x.index, dtype="float64")
    finite = np.isfinite(x.values)
    if finite.sum() == 0:
        return out
    xmin = float(np.nanmin(x.values[finite]))
    xmax = float(np.nanmax(x.values[finite]))
    if xmax <= xmin:
        out.loc[finite] = 1.0
    else:
        out.loc[finite] = (x.loc[finite] - xmin) / (xmax - xmin)
    return out

def split_semicolon_values(series: pd.Series) -> List[str]:
    vals = []
    for x in series.dropna().astype(str):
        for token in x.split(";"):
            token = token.strip()
            if token and token.lower() != "nan" and token != ".":
                vals.append(token)
    return sorted(set(vals))

def enhancer_tier_from_rank(rank: float) -> str:
    if rank >= 3:
        return "strong"
    if rank >= 2:
        return "moderate"
    if rank >= 1:
        return "weak"
    return "none"

def quantile_tier(value: float, q50: float, q75: float, q90: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "none"
    if value >= q90:
        return "relative_high"
    if value >= q75:
        return "relative_moderate"
    if value >= q50:
        return "relative_low"
    return "relative_background"

# Reference loaders

def load_oncokb_genes(tsv_file: str) -> pd.DataFrame:
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"Missing OncoKB file: {tsv_file}")
    df = pd.read_csv(tsv_file, sep="\t")
    if "Hugo Symbol" not in df.columns:
        raise ValueError("OncoKB file must contain column 'Hugo Symbol'.")
    use_cols = ["Hugo Symbol"] + (["Gene Type"] if "Gene Type" in df.columns else [])
    out = df[use_cols].copy()
    out["gene_name"] = out["Hugo Symbol"].map(normalize_gene_symbol)
    if "Gene Type" not in out.columns:
        out["Gene Type"] = "UNKNOWN"
    return out.dropna(subset=["gene_name"]).drop_duplicates("gene_name")[["gene_name", "Gene Type"]].reset_index(drop=True)

def load_gencode_genes(gtf_file: str) -> pd.DataFrame:
    if not os.path.exists(gtf_file):
        raise FileNotFoundError(f"Missing GENCODE GTF: {gtf_file}")
    rows = []
    with open(gtf_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = parts
            if feature != "gene":
                continue
            attr_map = parse_gtf_attributes(attrs)
            gene_name = attr_map.get("gene_name", np.nan)
            if pd.isna(gene_name):
                continue
            start_i = int(start)
            end_i = int(end)
            tss = start_i if strand == "+" else end_i
            rows.append({
                "chrom": normalize_chr(chrom), "start": start_i, "end": end_i, "strand": strand, "tss": int(tss),
                "gene_id": attr_map.get("gene_id", np.nan), "gene_name": normalize_gene_symbol(gene_name),
                "gene_type": attr_map.get("gene_type", attr_map.get("gene_biotype", np.nan))
            })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError(f"No gene rows parsed from {gtf_file}")
    return out.drop_duplicates(["chrom", "start", "end", "gene_name"]).reset_index(drop=True)

def load_ccres(bed_file: str) -> pd.DataFrame:
    if not os.path.exists(bed_file):
        raise FileNotFoundError(f"Missing cCRE BED: {bed_file}")
    rows = []
    with open(bed_file, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = normalize_chr(parts[0])
            start = int(parts[1]) + 1  # BED 0-based start to 1-based inclusive
            end = int(parts[2])
            ccre_id = parts[3] if len(parts) > 3 else "."
            ccre_accession = parts[4] if len(parts) > 4 else "."
            annotation = parts[5] if len(parts) > 5 else (parts[3] if len(parts) > 3 else "")
            ann = str(annotation)
            is_promoter = "PLS" in ann
            is_enhancer = ("pELS" in ann) or ("dELS" in ann) or ("ELS" in ann)
            is_ctcf = "CTCF" in ann
            if is_promoter:
                ccre_class = "promoter_like"
            elif is_enhancer:
                ccre_class = "enhancer_like"
            elif is_ctcf:
                ccre_class = "ctcf_like"
            else:
                ccre_class = "other"
            rows.append({
                "chrom": chrom, "start": start, "end": end, "ccre_id": ccre_id, "ccre_accession": ccre_accession, "annotation": annotation,
                "ccre_class": ccre_class, "is_promoter_like": bool(is_promoter), "is_enhancer_like": bool(is_enhancer), "is_ctcf_like": bool(is_ctcf)
            })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError(f"No cCRE rows parsed from {bed_file}")
    return out.drop_duplicates(["chrom", "start", "end", "ccre_id"]).reset_index(drop=True)

# CADET input loaders

def load_block_table(input_dir: str) -> pd.DataFrame:
    candidates = [os.path.join(input_dir, "module_consolidation", "significant_blocks_with_module_ids.csv"),
        os.path.join(input_dir, "module_consolidation", "significant_peaks_with_module_ids.csv"),
        os.path.join(input_dir, "upgraded_annotation", "merged_peak_blocks_annotated.csv")]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("Could not find Layer 6 block input. Tried: " + "; ".join(candidates))
    df = pd.read_csv(path)
    if len(df) == 0:
        raise RuntimeError(f"Block table is empty: {path}")

    out = df.copy()
    if "peak_blk_id" not in out.columns:
        out["peak_blk_id"] = np.arange(len(out), dtype=int)
    if "module_id" not in out.columns:
        out["module_id"] = -1

    required_coords = ["chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j"]
    missing = [c for c in required_coords if c not in out.columns]
    if missing:
        raise ValueError(f"Block table missing required anchor-coordinate columns: {missing}")

    raw_label = first_existing(out,["primary_blk_label", "module_label", "layer3_blk_label", "upgraded_peak_block_class", "representative_upgraded_peak_class", "representative_peak_class", "upgraded_peak_class"],
        "unclassified_significant_contact")
    out["blk_label_raw"] = raw_label.astype(str)
    out["blk_label"] = out["blk_label_raw"].map(normalize_label)
    out["blk_label_disp"] = out["blk_label"].map(LABEL_DISPLAY)

    # normalize evidence/geometry columns used downstream
    out["max_obs_over_exp_raw"] = numeric(first_existing(out, ["max_obs_over_exp_raw", "obs_over_exp_raw", "rep_obs_over_exp_raw", "mean_obs_over_exp_raw"], 1.0), 1.0)
    out["mean_obs_over_exp_raw"] = numeric(first_existing(out, ["mean_obs_over_exp_raw", "max_obs_over_exp_raw"], 1.0), 1.0)
    out["max_local_oe"] = numeric(first_existing(out, ["max_local_oe", "local_obs_over_mean", "mean_local_obs_over_mean"], 1.0), 1.0)
    out["discord"] = numeric(first_existing(out, ["max_discord", "discord", "mean_discord"], np.nan))
    out["max_discord"] = numeric(first_existing(out, ["max_discord", "discord"], np.nan))
    out["mean_discord"] = numeric(first_existing(out, ["mean_discord", "discord"], np.nan))
    out["n_blk_neighbors_w5"] = numeric(first_existing(out, ["n_blk_neighbors_w5", "n_sig_neighbors_w5", "max_neighbor_count"], 0), 0).astype(int)
    out["near_junction"] = first_existing(out, ["near_junction", "any_near_junction"], False).map(safe_bool)
    out["near_discordant_join"] = first_existing(out, ["near_discordant_join", "any_near_discordant_join"], False).map(safe_bool)
    out["is_interchromosomal"] = first_existing(out, ["is_interchromosomal"], False).map(safe_bool)
    out["struct_ctx"] = first_existing(out, ["struct_blk_ctx", "struct_ctx"], "unknown").astype(str)
    out["junction_ctx"] = first_existing(out, ["junction_blk_ctx", "junction_ctx"], "unknown").astype(str)
    out["morphology_class"] = first_existing(out, ["morphology_blk_class", "morphology_class"], "unknown").astype(str)
    return out

def load_module_summary(input_dir: str) -> pd.DataFrame:
    p = os.path.join(input_dir, "module_consolidation", "module_summary.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing module summary: {p}")
    df = pd.read_csv(p)
    if len(df) == 0:
        raise RuntimeError("module_summary.csv is empty")
    out = df.copy()
    if "module_id" not in out.columns:
        raise ValueError("module_summary.csv must contain module_id")
    out["module_type"] = first_existing(out, ["module_type"], "singleton_contact_module").map(normalize_module_type)
    out["module_type_disp"] = out["module_type"].map(MODULE_TYPE_DISPLAY)
    out["module_label"] = first_existing(out, ["module_label", "primary_module_label", "rep_label", "top_peak_class"], "unclassified_significant_contact").map(normalize_label)
    out["module_label_disp"] = out["module_label"].map(LABEL_DISPLAY)
    out["n_member_blk"] = numeric(first_existing(out, ["n_member_blk", "n_member_peaks", "module_size"], 1), 1).astype(int)
    out["max_obs_over_exp_raw"] = numeric(first_existing(out, ["max_obs_over_exp_raw", "rep_obs_over_exp_raw", "mean_obs_over_exp_raw"], 1.0), 1.0)
    out["max_local_oe"] = numeric(first_existing(out, ["max_local_oe"], 1.0), 1.0)
    out["max_discord"] = numeric(first_existing(out, ["max_discord"], np.nan))
    out["any_near_discordant_join"] = first_existing(out, ["any_near_discordant_join"], False).map(safe_bool)
    out["any_near_junction"] = first_existing(out, ["any_near_junction"], False).map(safe_bool)
    return out

def load_module_priority(input_dir: str) -> pd.DataFrame:
    p = os.path.join(input_dir, "module_prioritization", "module_priority_scores.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing module priority scores: {p}")
    df = pd.read_csv(p)
    if len(df) == 0:
        raise RuntimeError("module_priority_scores.csv is empty")
    out = df.copy()
    if "module_id" not in out.columns:
        raise ValueError("module_priority_scores.csv must contain module_id")
    for c, default in {"priority": 0.0, "rank_agg": np.nan, "weighted_priority": np.nan,
                       "struct_supp": np.nan, "anchor_recur_supp": np.nan,
                       "focal_contact_supp": np.nan, "rank_overall": np.nan}.items():
        if c not in out.columns:
            out[c] = default
    if "priority_tier" not in out.columns:
        out["priority_tier"] = "unknown"
    if "recommendation" not in out.columns:
        out["recommendation"] = "not_available"
    return out

def build_module_table(module_summary: pd.DataFrame, module_priority: pd.DataFrame) -> pd.DataFrame:
    pr = module_priority.copy()
    drop_cols = [c for c in module_summary.columns if c in pr.columns and c != "module_id"]
    pr = pr.drop(columns=drop_cols, errors="ignore")
    out = module_summary.merge(pr, on="module_id", how="left")
    out["priority"] = numeric(first_existing(out, ["priority"], 0.0), 0.0)
    out["priority_tier"] = first_existing(out, ["priority_tier"], "unknown").astype(str)
    out["recommendation"] = first_existing(out, ["recommendation"], "not_available").astype(str)
    return out

#  helpers for annot

def nearest_gene_at(chrom_gene_df: Optional[pd.DataFrame], pos: float) -> Dict[str, object]:
    if chrom_gene_df is None or len(chrom_gene_df) == 0:
        return {"nearest_gene_name": np.nan, "nearest_gene_id": np.nan, "nearest_gene_type": np.nan, "nearest_gene_distance_bp": np.nan}
    d = np.abs(chrom_gene_df["tss"].values.astype(float) - float(pos))
    idx = int(np.argmin(d))
    row = chrom_gene_df.iloc[idx]
    return {"nearest_gene_name": row["gene_name"], "nearest_gene_id": row["gene_id"], "nearest_gene_type": row["gene_type"], "nearest_gene_distance_bp": float(d[idx])}

def nearest_oncogene_at(chrom_gene_df: Optional[pd.DataFrame], oncogene_set: set, pos: float) -> Dict[str, object]:
    if chrom_gene_df is None or len(chrom_gene_df) == 0:
        return {"nearest_oncogene_name": np.nan, "nearest_oncogene_distance_bp": np.nan}
    onc = chrom_gene_df[chrom_gene_df["gene_name"].isin(oncogene_set)].copy()
    if len(onc) == 0:
        return {"nearest_oncogene_name": np.nan, "nearest_oncogene_distance_bp": np.nan}
    d = np.abs(onc["tss"].values.astype(float) - float(pos))
    idx = int(np.argmin(d))
    row = onc.iloc[idx]
    return {"nearest_oncogene_name": row["gene_name"], "nearest_oncogene_distance_bp": float(d[idx])}

def enhancer_support_tier(count: int, weak_min: int, moderate_min: int, strong_min: int) -> str:
    if count >= strong_min:
        return "strong"
    if count >= moderate_min:
        return "moderate"
    if count >= weak_min:
        return "weak"
    return "none"

def tier_rank(tier: str) -> int:
    return {"none": 0, "weak": 1, "moderate": 2, "strong": 3}.get(str(tier), 0)

def annotate_anchor(prefix: str, block_df: pd.DataFrame, genes_by_chr: Dict[str, pd.DataFrame], ccres_by_chr: Dict[str, pd.DataFrame], oncogene_set: set, local_window_bp: int) -> pd.DataFrame:
    rows = []
    for _, row in block_df.iterrows():
        chrom = normalize_chr(row[f"chrom_{prefix}"])
        start = int(row[f"start_{prefix}"])
        end = int(row[f"end_{prefix}"])
        mid = midpoint(start, end)
        gene_chr = genes_by_chr.get(chrom)
        ccre_chr = ccres_by_chr.get(chrom)
        overlapping_genes = find_overlapping_rows(gene_chr, start, end)
        overlapping_ccres = find_overlapping_rows(ccre_chr, start, end)
        local_start = max(1, int(mid - local_window_bp))
        local_end = int(mid + local_window_bp)
        local_genes = find_overlapping_rows(gene_chr, local_start, local_end)
        local_ccres = find_overlapping_rows(ccre_chr, local_start, local_end)
        ng = nearest_gene_at(gene_chr, mid)
        no = nearest_oncogene_at(gene_chr, oncogene_set, mid)
        overlap_gene_names = sorted(overlapping_genes["gene_name"].dropna().astype(str).unique().tolist()) if len(overlapping_genes) else []
        overlap_oncogenes = sorted([g for g in overlap_gene_names if g in oncogene_set])
        local_gene_names = sorted(local_genes["gene_name"].dropna().astype(str).unique().tolist()) if len(local_genes) else []
        local_oncogenes = sorted([g for g in local_gene_names if g in oncogene_set])
        rows.append({"peak_blk_id": row["peak_blk_id"], f"{prefix}_midpoint": mid, f"overlap_gene_names_{prefix}": ";".join(overlap_gene_names),
            f"n_overlap_genes_{prefix}": int(len(overlap_gene_names)), f"overlap_onco_names_{prefix}": ";".join(overlap_oncogenes),
            f"n_overlap_oncos_{prefix}": int(len(overlap_oncogenes)), f"near_gene_name_{prefix}": ng["nearest_gene_name"],
            f"near_gene_id_{prefix}": ng["nearest_gene_id"], f"near_gene_type_{prefix}": ng["nearest_gene_type"],
            f"near_gene_dist_{prefix}": ng["nearest_gene_distance_bp"], f"near_onco_name_{prefix}": no["nearest_oncogene_name"],
            f"near_onco_dist_{prefix}": no["nearest_oncogene_distance_bp"], f"n_overlap_ccres_{prefix}": int(len(overlapping_ccres)),
            f"n_overlap_prom_like_{prefix}": int(overlapping_ccres["is_promoter_like"].sum()) if len(overlapping_ccres) else 0,
            f"n_overlap_enh_like_{prefix}": int(overlapping_ccres["is_enhancer_like"].sum()) if len(overlapping_ccres) else 0,
            f"n_overlap_ctcf_like_{prefix}": int(overlapping_ccres["is_ctcf_like"].sum()) if len(overlapping_ccres) else 0,
            f"local_window_bp_{prefix}": int(local_window_bp), f"local_gene_names_{prefix}": ";".join(local_gene_names),
            f"n_local_genes_{prefix}": int(len(local_gene_names)), f"local_onco_names_{prefix}": ";".join(local_oncogenes),
            f"n_local_oncos_{prefix}": int(len(local_oncogenes)), f"n_local_ccres_{prefix}": int(len(local_ccres)),
            f"n_local_prom_like_{prefix}": int(local_ccres["is_promoter_like"].sum()) if len(local_ccres) else 0,
            f"n_local_enh_like_{prefix}": int(local_ccres["is_enhancer_like"].sum()) if len(local_ccres) else 0,
            f"n_local_ctcf_like_{prefix}": int(local_ccres["is_ctcf_like"].sum()) if len(local_ccres) else 0})
    return pd.DataFrame(rows)

def annotate_blocks(block_df: pd.DataFrame, genes_df: pd.DataFrame, ccres_df: pd.DataFrame, oncokb_df: pd.DataFrame, local_window_bp: int) -> pd.DataFrame:
    genes_by_chr = split_by_chrom(genes_df)
    ccres_by_chr = split_by_chrom(ccres_df)
    oncogene_set = set(oncokb_df["gene_name"].dropna().astype(str).tolist())
    left = annotate_anchor("i", block_df, genes_by_chr, ccres_by_chr, oncogene_set, local_window_bp)
    right = annotate_anchor("j", block_df, genes_by_chr, ccres_by_chr, oncogene_set, local_window_bp)
    out = block_df.merge(left, on="peak_blk_id", how="left").merge(right, on="peak_blk_id", how="left")
    # anchor-level totals
    for key in ["local_enh_like", "local_prom_like", "local_ctcf_like", "local_oncos", "overlap_oncos"]:
        ci = f"n_{key}_i"
        cj = f"n_{key}_j"
        if ci in out.columns and cj in out.columns:
            out[f"n_{key}_total"] = numeric(out[ci], 0) + numeric(out[cj], 0)
    out["any_local_onco"] = (numeric(out["n_local_oncos_i"], 0) > 0) | (numeric(out["n_local_oncos_j"], 0) > 0)
    out["any_local_enh"] = (numeric(out["n_local_enh_like_i"], 0) > 0) | (numeric(out["n_local_enh_like_j"], 0) > 0)
    out["any_local_prom"] = (numeric(out["n_local_prom_like_i"], 0) > 0) | (numeric(out["n_local_prom_like_j"], 0) > 0)
    out["any_local_ctcf"] = (numeric(out["n_local_ctcf_like_i"], 0) > 0) | (numeric(out["n_local_ctcf_like_j"], 0) > 0)
    # nearest-gene summaries
    gene_i = numeric(out["near_gene_dist_i"], np.nan)
    gene_j = numeric(out["near_gene_dist_j"], np.nan)
    out["min_near_gene_dist"] = np.nanmin(np.vstack([gene_i.values, gene_j.values]), axis=0)
    onco_i = numeric(out["near_onco_dist_i"], np.nan)
    onco_j = numeric(out["near_onco_dist_j"], np.nan)
    tmp = np.vstack([np.where(pd.isna(onco_i), np.inf, onco_i), np.where(pd.isna(onco_j), np.inf, onco_j)])
    out["min_near_onco_dist"] = np.nanmin(tmp, axis=0)
    out.loc[np.isinf(out["min_near_onco_dist"]), "min_near_onco_dist"] = np.nan
    def best_by_dist(row, name_i, dist_i, name_j, dist_j):
        cand = []
        if pd.notna(row.get(name_i)) and pd.notna(row.get(dist_i)):
            cand.append((float(row[dist_i]), row[name_i]))
        if pd.notna(row.get(name_j)) and pd.notna(row.get(dist_j)):
            cand.append((float(row[dist_j]), row[name_j]))
        if not cand:
            return np.nan
        return sorted(cand, key=lambda x: x[0])[0][1]
    out["best_blk_gene"] = out.apply(lambda r: best_by_dist(r, "near_gene_name_i", "near_gene_dist_i", "near_gene_name_j", "near_gene_dist_j"), axis=1)
    out["best_blk_onco"] = out.apply(lambda r: best_by_dist(r, "near_onco_name_i", "near_onco_dist_i", "near_onco_name_j", "near_onco_dist_j"), axis=1)
    return out

def assign_anchor_roles(df: pd.DataFrame, gene_dist_bp: int, oncogene_dist_bp: int, enhancer_weak_min: int, enhancer_moderate_min: int, enhancer_strong_min: int, min_promoters: int) -> pd.DataFrame:
    out = df.copy()
    for p in ["i", "j"]:
        nearest_gene = numeric(out[f"near_gene_dist_{p}"], np.nan)
        nearest_onco = numeric(out[f"near_onco_dist_{p}"], np.nan)
        local_enh = numeric(out[f"n_local_enh_like_{p}"], 0).astype(int)
        overlap_enh = numeric(out[f"n_overlap_enh_like_{p}"], 0).astype(int)
        # local count drives the tier; direct overlap can still mark weak support
        tier_counts = np.maximum(local_enh.values, overlap_enh.values)
        out[f"anc_{p}_enh_supp_tier"] = [enhancer_support_tier(int(x), enhancer_weak_min, enhancer_moderate_min, enhancer_strong_min) for x in tier_counts]
        out[f"anc_{p}_has_any_enh"] = out[f"anc_{p}_enh_supp_tier"].map(lambda x: tier_rank(x) >= 1)
        out[f"anc_{p}_has_moderate_enh"] = out[f"anc_{p}_enh_supp_tier"].map(lambda x: tier_rank(x) >= 2)
        out[f"anc_{p}_has_strong_enh"] = out[f"anc_{p}_enh_supp_tier"].map(lambda x: tier_rank(x) >= 3)
        out[f"anc_{p}_gene_like"] = (numeric(out[f"n_overlap_genes_{p}"], 0) > 0) | (nearest_gene <= gene_dist_bp) | (numeric(out[f"n_overlap_prom_like_{p}"], 0) > 0)
        out[f"anc_{p}_onco_like"] = (numeric(out[f"n_overlap_oncos_{p}"], 0) > 0) | (nearest_onco <= oncogene_dist_bp)
        out[f"anc_{p}_prom_like"] = (numeric(out[f"n_overlap_prom_like_{p}"], 0) > 0) | (numeric(out[f"n_local_prom_like_{p}"], 0).astype(int) >= min_promoters)
    out["best_enh_supp_tier"] = [max([a, b], key=tier_rank) for a, b in zip(out["anc_i_enh_supp_tier"], out["anc_j_enh_supp_tier"])]
    out["best_enh_supp_rank"] = out["best_enh_supp_tier"].map(tier_rank)
    return out

def add_pair_flags(df: pd.DataFrame, struct_disc_min: float = 1.0, onco_disc_min: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    disc = numeric(first_existing(out, ["max_discord", "discord"], 0.0), 0.0).clip(lower=0.0)
    label = out["blk_label"].map(normalize_label)
    structural_context = (
        label.eq("discordant_join_associated_contact") |
        out["near_discordant_join"].map(safe_bool) |
        out["is_interchromosomal"].map(safe_bool) |
        out["struct_ctx"].astype(str).str.contains("rewiring|interchrom|discordant", case=False, na=False) |
        (disc >= struct_disc_min)
    )
    anchor_rec_ctx = label.eq("anchor_recurrent_contact") | out["morphology_class"].astype(str).str.contains("anchor", case=False, na=False)
    out["has_struct_supp"] = structural_context
    out["has_recur_supp"] = anchor_rec_ctx

    # main calls use moderate/strong enhancer support; permissive columns keep weak support too
    enh_i_any = out["anc_i_has_any_enh"]
    enh_j_any = out["anc_j_has_any_enh"]
    enh_i_mod = out["anc_i_has_moderate_enh"]
    enh_j_mod = out["anc_j_has_moderate_enh"]

    out["pair_enh_gene_perm"] = ((enh_i_any & out["anc_j_gene_like"]) | (enh_j_any & out["anc_i_gene_like"])) & (structural_context | anchor_rec_ctx)
    out["pair_enh_gene"] = ((enh_i_mod & out["anc_j_gene_like"]) | (enh_j_mod & out["anc_i_gene_like"])) & (structural_context | anchor_rec_ctx)
    out["pair_enh_onco_perm"] = ((enh_i_any & out["anc_j_onco_like"]) | (enh_j_any & out["anc_i_onco_like"])) & (structural_context | anchor_rec_ctx | (disc >= onco_disc_min))
    out["pair_enh_onco"] = ((enh_i_mod & out["anc_j_onco_like"]) | (enh_j_mod & out["anc_i_onco_like"])) & (structural_context | (disc >= onco_disc_min))
    out["pair_strong_enh_onco"] = ((out["anc_i_has_strong_enh"] & out["anc_j_onco_like"]) | (out["anc_j_has_strong_enh"] & out["anc_i_onco_like"])) & structural_context
    out["pair_prom_disrupt"] = (out["anc_i_prom_like"] | out["anc_j_prom_like"]) & (out["near_junction"].map(safe_bool) | out["near_discordant_join"].map(safe_bool) | (disc >= struct_disc_min))
    return out

# Module spans and summaries

def compute_module_spans(block_func: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module_id, grp in block_func.groupby("module_id", sort=False):
        anchors = []
        for _, r in grp.iterrows():
            anchors.append((normalize_chr(r["chrom_i"]), int(r["start_i"]), int(r["end_i"])))
            anchors.append((normalize_chr(r["chrom_j"]), int(r["start_j"]), int(r["end_j"])))
        adf = pd.DataFrame(anchors, columns=["chrom", "start", "end"]).drop_duplicates()
        for chrom, cg in adf.groupby("chrom", sort=False):
            rows.append({"module_id": int(module_id), "chrom": chrom, "module_span_start": int(cg["start"].min()), "module_span_end": int(cg["end"].max()), "n_anchor_intervals": int(len(cg))})
    return pd.DataFrame(rows)

def annotate_module_spans(module_spans: pd.DataFrame, genes_df: pd.DataFrame, ccres_df: pd.DataFrame, oncokb_df: pd.DataFrame) -> pd.DataFrame:
    genes_by_chr = split_by_chrom(genes_df)
    ccres_by_chr = split_by_chrom(ccres_df)
    oncogene_set = set(oncokb_df["gene_name"].dropna().astype(str).tolist())
    rows = []
    for _, row in module_spans.iterrows():
        chrom = normalize_chr(row["chrom"])
        start = int(row["module_span_start"])
        end = int(row["module_span_end"])
        og = find_overlapping_rows(genes_by_chr.get(chrom), start, end)
        oc = find_overlapping_rows(ccres_by_chr.get(chrom), start, end)
        genes = sorted(og["gene_name"].dropna().astype(str).unique().tolist()) if len(og) else []
        oncos = sorted([g for g in genes if g in oncogene_set])
        rows.append({"module_id": int(row["module_id"]), "chrom": chrom, "module_span_start": start, "module_span_end": end,
            "module_span_width_bp": int(end - start + 1), "module_gene_names": ";".join(genes), "n_module_genes": int(len(genes)),
            "module_onco_names": ";".join(oncos), "n_module_oncos": int(len(oncos)), "n_module_ccres": int(len(oc)),
            "n_module_prom_like": int(oc["is_promoter_like"].sum()) if len(oc) else 0,
            "n_module_enh_like": int(oc["is_enhancer_like"].sum()) if len(oc) else 0,
            "n_module_ctcf_like": int(oc["is_ctcf_like"].sum()) if len(oc) else 0})
    return pd.DataFrame(rows)

def summarize_module(module_table: pd.DataFrame, block_func: pd.DataFrame, module_span_ann: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module_id, grp in block_func.groupby("module_id", sort=False):
        span_grp = module_span_ann[module_span_ann["module_id"] == module_id].copy()
        n_blocks = int(len(grp))
        max_oe = float(numeric(grp["max_obs_over_exp_raw"], 1.0).max())
        max_disc = float(numeric(grp["max_discord"], 0.0).clip(lower=0.0).max())
        max_local_enh = int(numeric(grp["n_local_enh_like_total"], 0).max()) if "n_local_enh_like_total" in grp else 0
        mean_local_enh = float(numeric(grp["n_local_enh_like_total"], 0).mean()) if "n_local_enh_like_total" in grp else 0.0
        min_onco = numeric(grp["min_near_onco_dist"], np.nan).dropna()
        min_onco_dist = float(min_onco.min()) if len(min_onco) else np.nan

        n_enh_gene = int(grp["pair_enh_gene"].fillna(False).sum())
        n_enh_gene_perm = int(grp["pair_enh_gene_perm"].fillna(False).sum())
        n_enh_onco = int(grp["pair_enh_onco"].fillna(False).sum())
        n_enh_onco_perm = int(grp["pair_enh_onco_perm"].fillna(False).sum())
        n_strong_enh_onco = int(grp["pair_strong_enh_onco"].fillna(False).sum())
        n_prom_disrupt = int(grp["pair_prom_disrupt"].fillna(False).sum())
        n_struct = int(grp["has_struct_supp"].fillna(False).sum())
        n_anchor_recurrent = int(grp["has_recur_supp"].fillna(False).sum())
        n_blocks_weak_enh = int((grp["best_enh_supp_rank"] >= 1).sum())
        n_blocks_mod_enh = int((grp["best_enh_supp_rank"] >= 2).sum())
        n_blocks_strong_enh = int((grp["best_enh_supp_rank"] >= 3).sum())
        n_local_onco = int(grp["any_local_onco"].fillna(False).sum())
        n_local_prom = int(grp["any_local_prom"].fillna(False).sum())
        n_local_ctcf = int(grp["any_local_ctcf"].fillna(False).sum())

        # count unique features so repeated anchors do not dominate the module summary
        unique_local_genes = sorted(set(split_semicolon_values(grp["local_gene_names_i"]) + split_semicolon_values(grp["local_gene_names_j"])))
        uniq_local_onco = sorted(set(split_semicolon_values(grp["local_onco_names_i"]) + split_semicolon_values(grp["local_onco_names_j"])))
        unique_best_genes = sorted(set(grp["best_blk_gene"].dropna().astype(str).tolist()))
        uniq_best_onco = sorted(set(grp["best_blk_onco"].dropna().astype(str).tolist()))
        uniq_cand_onco = sorted(set(uniq_local_onco + uniq_best_onco))
        enh_anchor_keys = []
        for _, r in grp.iterrows():
            for side in ["i", "j"]:
                if tier_rank(r.get(f"anc_{side}_enh_supp_tier", "none")) >= 2:
                    enh_anchor_keys.append(f"{normalize_chr(r[f'chrom_{side}'])}:{int(r[f'start_{side}'])}-{int(r[f'end_{side}'])}")
        uniq_enh_anchors = sorted(set(enh_anchor_keys))

        frac_enh_gene = n_enh_gene / max(n_blocks, 1)
        frac_enh_onco = n_enh_onco / max(n_blocks, 1)
        frac_prom = n_prom_disrupt / max(n_blocks, 1)
        size_factor = min(np.log2(n_blocks + 1) / 2.5, 1.0)
        onco_bonus = min(len(uniq_cand_onco) / 3.0, 1.0)
        enh_anchor_bonus = min(len(uniq_enh_anchors) / 4.0, 1.0)

        # Functional score is support/prioritization, not statistical significance.
        # Capping repeated-block fractions keeps recurrent modules from inflating too much.
        score = 0.0
        score += min(np.log10(max_oe + 1.0) / 1.5, 1.0)
        score += min(np.log10(max(max_disc, 0.0) + 1.0) * 2.0, 1.8)
        score += 1.4 * min(frac_enh_gene * size_factor, 1.0)
        score += 3.0 * min(frac_enh_onco * size_factor, 1.0)
        score += 1.2 * min(frac_prom * size_factor, 1.0)
        score += 0.45 * min(n_blocks_mod_enh / max(n_blocks, 1), 1.0)
        score += 0.25 * min(n_blocks_weak_enh / max(n_blocks, 1), 1.0)
        score += 0.35 * min(n_local_onco / max(n_blocks, 1), 1.0)
        score += 0.30 * onco_bonus
        score += 0.25 * enh_anchor_bonus
        score += 0.20 * min(n_local_prom / max(n_blocks, 1), 1.0)
        score += 0.15 * min(n_local_ctcf / max(n_blocks, 1), 1.0)
        if pd.notna(min_onco_dist):
            if min_onco_dist <= 5000:
                score += 1.0
            elif min_onco_dist <= 15000:
                score += 0.55

        if n_strong_enh_onco > 0 and n_struct > 0:
            rec = "high_priority_candidate_enhancer_oncogene_contact"
        elif n_enh_onco > 0 and n_struct > 0:
            rec = "candidate_enhancer_oncogene_contact"
        elif n_enh_onco_perm > 0 and n_struct > 0:
            rec = "exploratory_enhancer_oncogene_contact"
        elif n_enh_gene > 0 and n_struct > 0:
            rec = "candidate_enhancer_gene_contact"
        elif n_prom_disrupt > 0:
            rec = "candidate_promoter_disruption_contact"
        elif n_local_onco > 0 and n_blocks_mod_enh > 0 and (n_struct > 0 or n_anchor_recurrent > 0):
            rec = "regulatory_support_candidate"
        elif n_local_ctcf > 0 and n_struct > 0:
            rec = "architectural_regulatory_candidate"
        else:
            rec = "general_functional_context_candidate"

        rows.append({"module_id": int(module_id), "n_member_blk_func": n_blocks,
            "module_blk_labels": ";".join(sorted(set(grp["blk_label"].dropna().astype(str).tolist()))),
            "module_best_blk_genes": ";".join(unique_best_genes), "module_best_blk_oncos": ";".join(uniq_best_onco),
            "module_local_genes": ";".join(unique_local_genes), "module_local_oncos": ";".join(uniq_local_onco),
            "module_oncos": ";".join(uniq_cand_onco), "n_uniq_local_genes": int(len(unique_local_genes)),
            "n_uniq_local_oncos": int(len(uniq_local_onco)), "n_uniq_cand_oncos": int(len(uniq_cand_onco)),
            "n_enh_anchors": int(len(uniq_enh_anchors)),
            "enh_anchor_ids": ";".join(uniq_enh_anchors), "n_blk_any_enh_supp": n_blocks_weak_enh,
            "n_blk_moderate_enh_supp": n_blocks_mod_enh, "n_blk_strong_enh_supp": n_blocks_strong_enh,
            "n_blk_local_onco": n_local_onco, "n_blk_local_prom": n_local_prom, "n_blk_local_ctcf": n_local_ctcf,
            "max_local_enh_per_blk": max_local_enh, "mean_local_enh_per_blk": mean_local_enh, "n_pair_enh_gene": n_enh_gene,
            "n_pair_enh_gene_perm": n_enh_gene_perm, "n_pair_enh_onco": n_enh_onco, "n_pair_enh_onco_perm": n_enh_onco_perm,
            "n_pair_strong_enh_onco": n_strong_enh_onco, "n_pair_prom_disrupt": n_prom_disrupt, "frac_enh_gene": float(frac_enh_gene),
            "frac_enh_onco": float(frac_enh_onco), "frac_prom_disrupt": float(frac_prom),
            "n_blk_struct_supp": n_struct, "n_blk_recur_supp": n_anchor_recurrent,
            "max_obs_over_exp_raw_func": max_oe, "max_discord_func": max_disc,
            "min_near_onco_dist": min_onco_dist, "n_span_records": int(len(span_grp)),
            "n_module_genes_span": int(span_grp["n_module_genes"].sum()) if len(span_grp) else 0,
            "n_module_oncos_span": int(span_grp["n_module_oncos"].sum()) if len(span_grp) else 0,
            "n_module_enh_like_span": int(span_grp["n_module_enh_like"].sum()) if len(span_grp) else 0,
            "n_module_prom_like_span": int(span_grp["n_module_prom_like"].sum()) if len(span_grp) else 0,
            "n_module_ctcf_like_span": int(span_grp["n_module_ctcf_like"].sum()) if len(span_grp) else 0,
            "func_priority": float(score), "func_recommendation": rec})
    func = pd.DataFrame(rows)
    out = module_table.merge(func, on="module_id", how="left")
    max_func = float(out["func_priority"].max()) if len(out) else 1.0
    if not np.isfinite(max_func) or max_func <= 0:
        max_func = 1.0
    out["func_priority_scaled"] = out["func_priority"].fillna(0).astype(float) / max_func
    out["combined_priority"] = 0.60 * out["priority"].fillna(0).astype(float) + 0.40 * out["func_priority_scaled"].fillna(0).astype(float)
    out = out.sort_values(["combined_priority", "func_priority", "priority"], ascending=[False, False, False]).reset_index(drop=True)
    out["func_rank"] = np.arange(1, len(out) + 1)
    return out

# Enhancer-burden QC

def add_burden_qc(block_func: pd.DataFrame, local_window_bp: int) -> pd.DataFrame:
    out = block_func.copy()
    window_kb_per_anchor = max((2.0 * float(local_window_bp)) / 1000.0, 1e-9)
    out["anc_i_local_enh_per10kb"] = numeric(out["n_local_enh_like_i"], 0) / window_kb_per_anchor * 10.0
    out["anc_j_local_enh_per10kb"] = numeric(out["n_local_enh_like_j"], 0) / window_kb_per_anchor * 10.0
    out["best_anchor_enh_per10kb"] = np.maximum(out["anc_i_local_enh_per10kb"], out["anc_j_local_enh_per10kb"])
    out["total_anchor_enh_per10kb"] = out["anc_i_local_enh_per10kb"] + out["anc_j_local_enh_per10kb"]
    vals = out["best_anchor_enh_per10kb"].astype(float)
    q50 = float(vals.quantile(0.50)) if len(vals) else 0.0
    q75 = float(vals.quantile(0.75)) if len(vals) else 0.0
    q90 = float(vals.quantile(0.90)) if len(vals) else 0.0
    out["rel_enh_burden_tier"] = [quantile_tier(v, q50, q75, q90) for v in vals]
    out["strong_enh_only_ctx"] = ((out["best_enh_supp_rank"] >= 3) & (~out["any_local_onco"].fillna(False)) & (~out["pair_enh_onco"].fillna(False)) &
        (~out["pair_prom_disrupt"].fillna(False)))
    return out

def summarize_burden_qc(block_func: pd.DataFrame) -> pd.DataFrame:
    cols = ["n_local_enh_like_i", "n_local_enh_like_j", "n_local_enh_like_total",
        "anc_i_local_enh_per10kb", "anc_j_local_enh_per10kb",
        "best_anchor_enh_per10kb", "total_anchor_enh_per10kb"]
    rows = []
    for c in cols:
        if c not in block_func.columns:
            continue
        x = numeric(block_func[c], np.nan).dropna()
        if len(x) == 0:
            continue
        rows.append({
            "metric": c, "n": int(len(x)), "min": float(x.min()), "q10": float(x.quantile(0.10)), "median": float(x.median()),
            "q75": float(x.quantile(0.75)), "q90": float(x.quantile(0.90)), "max": float(x.max())
        })
    return pd.DataFrame(rows)

# Plot helpers
# NOTE: no fixed hex colors here, default matplotlib colors are fine for QC plots.

def plot_top_modules_bar(df: pd.DataFrame, outpath: str, top_n: int = 20) -> None:
    use = df.head(min(top_n, len(df))).copy()
    if len(use) == 0:
        return
    # Horizontal ranking is easier to read and matches finalized CADET summary plots.
    use = use.iloc[::-1].copy()
    fig_h = max(6.0, 0.45 * len(use) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(use))
    ax.barh(y, use["combined_priority"].fillna(0).values)
    ax.set_yticks(y)
    ax.set_yticklabels([f"M{m}" for m in use["module_id"]], fontsize=PUB_TICK_FONTSIZE)
    ax.set_xlabel("Combined priority score", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Module", fontsize=PUB_FONTSIZE)
    ax.set_title("Functionally prioritized CADET contact modules", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PUB_TICK_FONTSIZE)
    savefig(fig, outpath)

def plot_tier_counts(block_func: pd.DataFrame, outpath: str) -> None:
    vals = block_func["best_enh_supp_tier"].value_counts().reindex(["none", "weak", "moderate", "strong"]).fillna(0)
    labels = ["None", "Weak", "Moderate", "Strong"]
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    y = np.arange(len(vals))
    ax.barh(y, vals.values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PUB_TICK_FONTSIZE)
    ax.set_xlabel("Number of contact blocks", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Best enhancer-support tier per block", fontsize=PUB_FONTSIZE)
    ax.set_title("Graded enhancer support", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PUB_TICK_FONTSIZE)
    savefig(fig, outpath)

def plot_burden_hist(block_func: pd.DataFrame, outpath: str) -> None:
    if "best_anchor_enh_per10kb" not in block_func.columns:
        return
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.hist(block_func["best_anchor_enh_per10kb"].dropna().astype(float), bins=min(20, max(5, len(block_func))))
    ax.set_xlabel("Best-anchor enhancer-like cCRE density per 10 kb", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Number of contact blocks", fontsize=PUB_FONTSIZE)
    ax.set_title("Enhancer-burden QC", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    savefig(fig, outpath)

def plot_relative_tiers(block_func: pd.DataFrame, outpath: str) -> None:
    if "rel_enh_burden_tier" not in block_func.columns:
        return
    order = ["relative_background", "relative_low", "relative_moderate", "relative_high"]
    vals = block_func["rel_enh_burden_tier"].value_counts().reindex(order).fillna(0)
    labels = ["Background", "Low", "Moderate", "High"]
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    y = np.arange(len(vals))
    ax.barh(y, vals.values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PUB_TICK_FONTSIZE)
    ax.set_xlabel("Number of contact blocks", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Relative enhancer-burden tier", fontsize=PUB_FONTSIZE)
    ax.set_title("Relative enhancer-burden QC", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PUB_TICK_FONTSIZE)
    savefig(fig, outpath)

def plot_onco_vs_enh(df: pd.DataFrame, outpath: str) -> None:
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.scatter(df["max_local_enh_per_blk"].fillna(0), df["n_blk_local_onco"].fillna(0), s=50, alpha=0.8, edgecolors="black", linewidths=0.4)
    ax.set_xlabel("Maximum local enhancer-like cCRE count per block", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Blocks with local oncogene support", fontsize=PUB_FONTSIZE)
    ax.set_title("Local enhancer burden vs oncogene support", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    savefig(fig, outpath)

def plot_score_vs_discordance(df: pd.DataFrame, outpath: str) -> None:
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.scatter(df["max_discord_func"].fillna(0).clip(lower=0), df["func_priority"].fillna(0), s=50, alpha=0.8, edgecolors="black", linewidths=0.4)
    ax.set_xlabel("Maximum positive distance-discordance score", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Functional priority score", fontsize=PUB_FONTSIZE)
    ax.set_title("Structural discordance vs functional support", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    savefig(fig, outpath)

def plot_pair_logic_summary(df: pd.DataFrame, outpath: str, top_n: int = 20) -> None:
    top = df.head(min(top_n, len(df))).copy()
    if len(top) == 0:
        return
    top = top.iloc[::-1].copy()
    y = np.arange(len(top))
    height = 0.22
    fig_h = max(6.0, 0.5 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(y - height, top["frac_enh_gene"].fillna(0), height=height, label="Enhancer-gene")
    ax.barh(y, top["frac_enh_onco"].fillna(0), height=height, label="Enhancer-oncogene")
    ax.barh(y + height, top["frac_prom_disrupt"].fillna(0), height=height, label="Promoter disruption")
    ax.set_yticks(y)
    ax.set_yticklabels([f"M{m}" for m in top["module_id"]], fontsize=PUB_TICK_FONTSIZE)
    ax.set_xlabel("Fraction of module blocks", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Module", fontsize=PUB_FONTSIZE)
    ax.set_title("Pairwise regulatory logic across prioritized modules", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    savefig(fig, outpath)

def plot_recommendation_bar(df: pd.DataFrame, outpath: str) -> None:
    vals = df["func_recommendation"].value_counts()
    if len(vals) == 0:
        return
    vals = vals.iloc[::-1]
    fig_h = max(6.0, 0.65 * len(vals) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(vals))
    ax.barh(y, vals.values)
    ax.set_yticks(y)
    ax.set_yticklabels(vals.index.astype(str), fontsize=PUB_TICK_FONTSIZE)
    ax.set_xlabel("Number of modules", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Functional recommendation", fontsize=PUB_FONTSIZE)
    ax.set_title("Functional recommendation classes", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PUB_TICK_FONTSIZE)
    savefig(fig, outpath)

# Main

def main() -> None:
    ap = argparse.ArgumentParser(description="CADET Layer 6: functional annotation")
    ap.add_argument("--input_dir", required=True, help="cadet_final_results folder")
    ap.add_argument("--oncokb_tsv", required=True, help="OncoKB cancerGeneList.tsv")
    ap.add_argument("--gencode_gtf", required=True, help="GENCODE GTF (GRCh38)")
    ap.add_argument("--ccres_bed", required=True, help="ENCODE cCRE BED (GRCh38)")
    ap.add_argument("--local_window_bp", type=int, default=25000)
    ap.add_argument("--gene_dist_bp", type=int, default=5000)
    ap.add_argument("--oncogene_dist_bp", type=int, default=15000)
    ap.add_argument("--enhancer_weak_min", type=int, default=1)
    ap.add_argument("--enhancer_moderate_min", type=int, default=3)
    ap.add_argument("--enhancer_strong_min", type=int, default=5)
    ap.add_argument("--min_promoters", type=int, default=1)
    ap.add_argument("--struct_disc_min", type=float, default=1.0)
    ap.add_argument("--plot_top_n", type=int, default=20)
    args = ap.parse_args()

    input_dir = args.input_dir
    outdir = os.path.join(input_dir, "layer6_functional_annotation")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    print("Loading finalized CADET module/block inputs ...")
    blocks = load_block_table(input_dir)
    module_summary = load_module_summary(input_dir)
    module_priority = load_module_priority(input_dir)
    module_table = build_module_table(module_summary, module_priority)

    print("Loading functional references ...")
    oncokb = load_oncokb_genes(args.oncokb_tsv)
    genes = load_gencode_genes(args.gencode_gtf)
    ccres = load_ccres(args.ccres_bed)

    print(f"Contact blocks loaded: {len(blocks)}")
    print(f"Modules loaded: {len(module_table)}")
    print(f"OncoKB genes loaded: {len(oncokb)}")
    print(f"GENCODE genes loaded: {len(genes)}")
    print(f"cCREs loaded: {len(ccres)}")

    print("Annotating block anchors ...")
    block_func = annotate_blocks(blocks, genes, ccres, oncokb, local_window_bp=args.local_window_bp)
    block_func = assign_anchor_roles(block_func, gene_dist_bp=args.gene_dist_bp, oncogene_dist_bp=args.oncogene_dist_bp,
        enhancer_weak_min=args.enhancer_weak_min, enhancer_moderate_min=args.enhancer_moderate_min, enhancer_strong_min=args.enhancer_strong_min,min_promoters=args.min_promoters)
    block_func = add_pair_flags(block_func, struct_disc_min=args.struct_disc_min, onco_disc_min=args.struct_disc_min)
    block_func = add_burden_qc(block_func, local_window_bp=args.local_window_bp)

    print("Building descriptive module spans ...")
    module_spans = compute_module_spans(block_func)
    module_span_ann = annotate_module_spans(module_spans, genes, ccres, oncokb)

    print("Summarizing module-level functional support ...")
    prioritized = summarize_module(module_table, block_func, module_span_ann)

    block_func.to_csv(os.path.join(outdir, "block_functional_annotation.csv"), index=False)
    module_spans.to_csv(os.path.join(outdir, "module_spans_descriptive.csv"), index=False)
    module_span_ann.to_csv(os.path.join(outdir, "module_span_functional_annotation_descriptive.csv"), index=False)
    prioritized.to_csv(os.path.join(outdir, "prioritized_functional_modules.csv"), index=False)
    summarize_burden_qc(block_func).to_csv(os.path.join(outdir, "enhancer_burden_qc_summary.csv"), index=False)
    burden_cols = ["peak_blk_id", "module_id", "blk_label", "chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j",
     "best_blk_gene", "best_blk_onco", "n_local_enh_like_total", "best_anchor_enh_per10kb",
     "rel_enh_burden_tier", "pair_enh_onco", "strong_enh_only_ctx"]
    burden_cols = [c for c in burden_cols if c in block_func.columns]
    block_func.sort_values("best_anchor_enh_per10kb", ascending=False).head(50)[burden_cols].to_csv(
        os.path.join(outdir, "top_enhancer_burden_blocks.csv"), index=False)
    block_func.loc[block_func["strong_enh_only_ctx"], burden_cols].to_csv(
        os.path.join(outdir, "strong_enhancer_only_context_blocks.csv"), index=False)

    params = {"layer6_version": "publication_functional_annotation",
        "input_dir": input_dir,
        "oncokb_tsv": args.oncokb_tsv,
        "gencode_gtf": args.gencode_gtf,
        "ccres_bed": args.ccres_bed,
        "important_genome_build_note": "The GTF and cCRE/enhancer BED must match the CADET coordinate genome build.",
        "local_window_bp": args.local_window_bp,
        "gene_dist_bp": args.gene_dist_bp,
        "oncogene_dist_bp": args.oncogene_dist_bp,
        "enhancer_tiers": {"weak_min": args.enhancer_weak_min,
            "moderate_min": args.enhancer_moderate_min,
            "strong_min": args.enhancer_strong_min,
            "note": "calls use moderate or strong support; permissive columns use any support"},
        "enhancer_burden_qc": {
            "density_metric": "enhancer-like cCRE count per 10 kb around each anchor",
            "strong_enhancer_only_context_blocks": "strong enhancer support but no oncogene or promoter logic"},
        "structural_discordance_min": args.struct_disc_min,
        "purpose": "functional annotation of modules"
    }
    with open(os.path.join(outdir, "layer6_parameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("Making plots ...")
    plot_top_modules_bar(prioritized, os.path.join(plot_dir, "top_functional_modules_barplot.png"), top_n=args.plot_top_n)
    plot_tier_counts(block_func, os.path.join(plot_dir, "block_enhancer_support_tiers.png"))
    plot_burden_hist(block_func, os.path.join(plot_dir, "enhancer_burden_density_histogram.png"))
    plot_relative_tiers(block_func, os.path.join(plot_dir, "relative_enhancer_burden_tiers.png"))
    plot_onco_vs_enh(prioritized, os.path.join(plot_dir, "local_oncogene_vs_local_enhancer_support.png"))
    plot_score_vs_discordance(prioritized, os.path.join(plot_dir, "discordance_vs_functional_score.png"))
    plot_pair_logic_summary(prioritized, os.path.join(plot_dir, "pair_logic_summary_top_modules.png"), top_n=args.plot_top_n)
    plot_recommendation_bar(prioritized, os.path.join(plot_dir, "functional_recommendation_summary.png"))

    print("\nLayer 6 functional annotation completed.")
    print(f"Block annotations: {len(block_func)}")
    print(f"Module spans: {len(module_spans)}")
    print(f"Module span annotations: {len(module_span_ann)}")
    print(f"Prioritized modules: {len(prioritized)}")
    show_cols = ["func_rank", "module_id", "module_type_disp", "module_label_disp", "priority",
        "func_priority", "combined_priority", "n_pair_enh_onco", "n_pair_enh_onco_perm",
        "n_pair_enh_gene", "n_pair_prom_disrupt", "n_blk_moderate_enh_supp",
        "n_blk_local_onco", "n_uniq_cand_oncos", "module_oncos",
        "max_discord_func", "func_recommendation"]
    show_cols = [c for c in show_cols if c in prioritized.columns]
    print("\nTop functionally prioritized modules:")
    print(prioritized[show_cols].head(10).to_string(index=False))
    print(f"\nOutputs written to: {outdir}")

if __name__ == "__main__":
    main()
