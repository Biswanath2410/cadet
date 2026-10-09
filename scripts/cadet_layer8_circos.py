#!/usr/bin/env python3

import argparse
import os
import math
import json
import re
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pycirclize import Circos


def norm_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s if s and s.lower() not in {"nan", "none", "null"} else np.nan

def norm_chr(x):
    s = str(x).strip()
    return s if s.startswith("chr") else "chr" + s

def safe_int(x, default=None):
    if pd.isna(x):
        return default
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default

def safe_float(x, default=np.nan):
    if pd.isna(x):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

ENSEMBL_RE = re.compile(r"^(ENS[A-Z]*G|ENS[A-Z]*T|ENSG|ENST|ENSMUSG|ENSMUST)\d+", re.IGNORECASE)

def is_ensembl_like(x) -> bool:
    s = norm_text(x)
    if pd.isna(s):
        return False
    return bool(ENSEMBL_RE.match(str(s)))

def clean_label_value(x) -> Optional[str]:
    s = norm_text(x)
    if pd.isna(s):
        return None
    s = str(s).strip()
    # Split semicolon/comma lists and keep first readable symbol.
    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    for p in parts:
        if not is_ensembl_like(p) and p.lower() not in {"nan", "none", "null"}:
            return p
    return None

def first_readable_label(row: pd.Series, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in row.index:
            val = clean_label_value(row.get(c))
            if val:
                return val
    return None

def _valid_side(x) -> Optional[str]:
    s = norm_text(x)
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    if s in {"i", "anchor_i", "left", "bin1"}:
        return "i"
    if s in {"j", "anchor_j", "right", "bin2"}:
        return "j"
    return None

def label_side_from_text(row: pd.Series, label: str) -> Optional[str]:
    if not label:
        return None
    lab = str(label).strip()
    i_cols = ["candidate_gene_name_i", "candidate_oncogene_name_i",
        "near_gene_name_i", "near_onco_name_i",
        "overlap_gene_names_i", "overlap_onco_names_i",
        "local_gene_names_i", "local_onco_names_i"]
    j_cols = ["candidate_gene_name_j", "candidate_oncogene_name_j",
        "near_gene_name_j", "near_onco_name_j",
        "overlap_gene_names_j", "overlap_onco_names_j","local_gene_names_j", "local_onco_names_j"]
    def contains(cols):
        for c in cols:
            if c in row.index and pd.notna(row.get(c)):
                parts = [p.strip() for p in re.split(r"[;,]", str(row.get(c))) if p.strip()]
                if lab in parts:
                    return True
        return False
    if contains(i_cols) and not contains(j_cols):
        return "i"
    if contains(j_cols) and not contains(i_cols):
        return "j"
    return None

def _logic_from_row(row: pd.Series) -> str:
    for col in ["blk_logic_label", "peak_logic_label", "evidence_type", "main_evidence_type", "func_recommendation"]:
        if col in row.index:
            v = norm_text(row.get(col))
            if pd.notna(v):
                return str(v)
    return ""

def _pick_label(row: pd.Series, candidates: List[Tuple[str, Optional[str]]]) -> Tuple[Optional[str], Optional[str]]:
    for label_col, side_col in candidates:
        if label_col not in row.index:
            continue
        label = clean_label_value(row.get(label_col))
        if not label:
            continue
        side = _valid_side(row.get(side_col)) if side_col and side_col in row.index else None
        if side is None:
            side = label_side_from_text(row, label)
        return label, side
    return None, None

def _readable_list(x) -> Optional[str]:
    return clean_label_value(x)

def _side_gene(row: pd.Series, side: Optional[str]) -> Optional[str]:
    side = _valid_side(side)
    if side not in {"i", "j"}:
        return None
    cols = [f"candidate_gene_name_{side}",
        f"preferred_target_name_{side}",
        f"overlap_gene_names_{side}",
        f"near_gene_name_{side}",
        f"local_gene_names_{side}",
        f"best_block_gene_{side}"]
    # Most files do not have side-suffixed candidate columns, so also use unsuffixed
    # candidate/preferred fields only when their side field matches the requested side.
    if _valid_side(row.get("cand_gene_side")) == side:
        cols.insert(0, "cand_gene_name")
    if _valid_side(row.get("preferred_target_side")) == side:
        cols.insert(1, "preferred_target_name")
    if _valid_side(row.get("gene_anchor_side")) == side:
        cols.insert(2, "top_gene")
    for c in cols:
        if c in row.index:
            lab = _readable_list(row.get(c))
            if lab:
                return lab
    return None

def _side_oncogene(row: pd.Series, side: Optional[str]) -> Optional[str]:
    side = _valid_side(side)
    if side not in {"i", "j"}:
        return None
    cols = [f"candidate_oncogene_name_{side}",f"overlap_onco_names_{side}",f"near_onco_name_{side}",
        f"local_onco_names_{side}",f"best_block_oncogene_{side}"]
    if _valid_side(row.get("cand_onco_side")) == side:
        cols.insert(0, "cand_onco_name")
    if _valid_side(row.get("onco_anchor_side")) == side:
        cols.insert(1, "top_onco")
        cols.insert(2, "module_oncos")
    for c in cols:
        if c in row.index:
            lab = _readable_list(row.get(c))
            if lab:
                return lab
    return None

def plot_label_and_side(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    logic = _logic_from_row(row).lower()

    # 1. Strict/permissive enhancer-oncogene: show oncogene, placed on oncogene side.
    if "enhancer_oncogene" in logic:
        side = _valid_side(row.get("cand_onco_side")) or _valid_side(row.get("onco_anchor_side"))
        label = _side_oncogene(row, side)
        if label:
            return label, side
        label, side2 = _pick_label(row, [("cand_onco_name", "cand_onco_side"),
            ("top_onco", "onco_anchor_side"),
            ("module_oncos", "onco_anchor_side")])
        return label, side2

    # 2. Enhancer-gene or promoter-disruption: show gene/promoter target, not oncogene fallback.
    if "enhancer_gene" in logic or "promoter" in logic:
        side = (_valid_side(row.get("cand_gene_side")) or
                _valid_side(row.get("preferred_target_side")) or
                _valid_side(row.get("gene_anchor_side")) or
                _valid_side(row.get("prom_anchor_side")))
        label = _side_gene(row, side)
        if label:
            return label, side
        for s in ["i", "j"]:
            label = _side_gene(row, s)
            if label:
                return label, s
        return None, None

    # 3. Local context / unresolved: intentionally suppress labels to avoid MYC everywhere.
    if "local" in logic or "context" in logic or "unresolved" in logic or "weak" in logic:
        return None, None

    # 4. Fallback for older files: prefer readable target, but still do not use oncogene as universal fallback.
    label, side = _pick_label(row, [("preferred_target_name", "preferred_target_side"),
        ("cand_gene_name", "cand_gene_side"),
        ("final_target_name", "preferred_target_side")])
    return label, side

SEGMENT_COLORS = ["#3A7CA5", "#D9534F", "#5CB85C", "#F0AD4E", "#9C6ADE",
    "#17A2B8", "#E83E8C", "#6C757D", "#FF8C42", "#2C7DA0",
    "#84A98C", "#52489C", "#C9ADA7", "#A4036F", "#048BA8",
    "#16DB93", "#F29E4C", "#0F4C5C", "#9E2A2B", "#7A9CC6"]

# layer 7 / layer 8 evidence classes -- publication palette
# magenta = strict enhancer-oncogene  (paper-eye-grabbing)
# blue    = permissive / enhancer-gene context
# orange  = promoter disruption
# grey    = local / unresolved
EVIDENCE_COLORS = {'strict_enhancer_oncogene': '#B5179E', 'strict_enhancer_oncogene_block': '#B5179E', 'permissive_enhancer_oncogene_context': '#4361EE',
    'permissive_enhancer_oncogene_context_block': '#4361EE', 'enhancer_gene_context': '#4895EF', 'enhancer_gene_block': '#4895EF',
    'promoter_disruption_context': '#F8961E', 'promoter_disruption_block': '#F8961E', 'local_regulatory_context': '#8D99AE',
    'local_regulatory_context_block': '#8D99AE', 'unresolved': '#CED4DA', 'unresolved_block': '#CED4DA',
    'enhancer_oncogene': '#B5179E', 'enhancer_oncogene_peak': '#B5179E', 'enhancer_gene': '#4895EF',
    'enhancer_gene_peak': '#4895EF', 'promoter_disruption': '#F8961E', 'promoter_disruption_peak': '#F8961E',
    'mixed_support_peak': '#4361EE', 'local_support_only_peak': '#8D99AE', 'weak_or_unresolved': '#CED4DA'}

EVIDENCE_DISPLAY = {'strict_enhancer_oncogene': 'Strict enhancer-oncogene', 'permissive_enhancer_oncogene_context': 'Permissive/context',
    'enhancer_gene_context': 'Enhancer-gene context', 'promoter_disruption_context': 'Promoter disruption',
    'local_regulatory_context': 'Local regulatory context', 'unresolved': 'Unresolved/context only'}

def segment_color(seg_idx: int) -> str:
    return SEGMENT_COLORS[int(seg_idx) % len(SEGMENT_COLORS)]

def canonical_evidence_key(row: pd.Series, color_by: str = "block") -> str:
    if color_by == "module":
        cols = ["main_evidence_type", "func_recommendation", "module_label", "blk_logic_label", "peak_logic_label", "evidence_type"]
    elif color_by == "auto":
        cols = ["blk_logic_label", "main_evidence_type", "peak_logic_label", "evidence_type", "func_recommendation", "module_label"]
    else:  # default: color connectors by the actual block/link evidence
        cols = ["blk_logic_label", "peak_logic_label", "evidence_type", "main_evidence_type", "func_recommendation", "module_label"]
    for col in cols:
        if col in row.index:
            key = norm_text(row.get(col))
            if pd.notna(key):
                key = str(key)
                if key in EVIDENCE_COLORS:
                    return key
                if "enhancer_oncogene" in key and "permissive" not in key:
                    return "strict_enhancer_oncogene"
                if "permissive" in key:
                    return "permissive_enhancer_oncogene_context"
                if "enhancer_gene" in key:
                    return "enhancer_gene_context"
                if "promoter" in key:
                    return "promoter_disruption_context"
                if "local" in key or "context" in key:
                    return "local_regulatory_context"
    return "unresolved"

def evidence_color(row: pd.Series, color_by: str = "block") -> str:
    return EVIDENCE_COLORS.get(canonical_evidence_key(row, color_by=color_by), "#999999")

def load_path_bins(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = ["path_bin", "seg_idx", "chrom", "genomic_bp", "strand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"path_bins.tsv missing required columns: {missing}")

    df["chrom"] = df["chrom"].map(norm_chr)
    df["path_bin"] = df["path_bin"].astype(int)
    df["seg_idx"] = df["seg_idx"].astype(int)
    df["genomic_bp"] = df["genomic_bp"].astype(int)

    diffs = (
        df.sort_values(["seg_idx", "genomic_bp"]) .groupby("seg_idx")["genomic_bp"] .diff() .dropna()
    )
    positive_diffs = diffs[diffs > 0]
    inferred_res = int(positive_diffs.mode().iloc[0]) if len(positive_diffs) else 1

    if "segment_start_bp" not in df.columns:
        df["segment_start_bp"] = df.groupby("seg_idx")["genomic_bp"].transform("min")
    if "segment_end_bp" not in df.columns:
        df["segment_end_bp"] = df.groupby("seg_idx")["genomic_bp"].transform("max") + inferred_res - 1

    df["segment_start_bp"] = df["segment_start_bp"].astype(int)
    df["segment_end_bp"] = df["segment_end_bp"].astype(int)
    return df

def load_layer3_annotation(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Newer Layer 3/4 block files may use representative_bin1/2. Keep bin1/bin2 if available.
    required = ["bin1", "bin2", "chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j", "seg_idx_i", "seg_idx_j"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Layer3 annotation missing columns: {missing}")
    return df

def load_peak_table(layer7_peak_csv: Optional[str], module_csv: Optional[str], part2_csv: Optional[str]) -> pd.DataFrame:
    if layer7_peak_csv and os.path.exists(layer7_peak_csv):
        peak = pd.read_csv(layer7_peak_csv)
        # Layer 7 block evidence already contains module fields, but merge summary fields if requested.
        if module_csv and os.path.exists(module_csv) and "module_id" in peak.columns:
            mod = pd.read_csv(module_csv)
            keep = [c for c in [
                "module_id", "func_rank", "module_type_disp", "module_label_disp",
                "func_recommendation", "main_evidence_type", "combined_priority",
                "final_target_name", "top_onco", "top_gene", "module_oncos"
            ] if c in mod.columns]
            if "module_id" in keep:
                overlap = [c for c in keep if c in peak.columns and c != "module_id"]
                peak = peak.drop(columns=overlap, errors="ignore").merge( mod[keep].drop_duplicates("module_id"), on="module_id", how="left" )
        return peak

    if part2_csv and os.path.exists(part2_csv):
        return pd.read_csv(part2_csv)

    raise FileNotFoundError("No usable peak table found. Provide Layer7 block evidence or --part2_csv.")

def format_bp_mb(x: int, digits: int = 2) -> str:
    try:
        x = int(x)
    except Exception:
        return str(x)
    return f"{x/1_000_000:.{digits}f}"

def make_seg_label(chrom: str, start_bp: int, end_bp: int, strand: str, path_bin_start: int, path_bin_end: int) -> str:
    chrom = norm_chr(chrom)
    strand = str(strand) if str(strand) in {"+", "-"} else "+"
    return f"{chrom}:{format_bp_mb(start_bp)}-{format_bp_mb(end_bp)} Mb ({strand})"

def format_genomic_tick(v: float, seg: Dict[str, Any]) -> str:
    seg_start = int(seg["segment_start_bp"])
    seg_end = int(seg["segment_end_bp"])
    strand = str(seg.get("strand", "+"))
    if strand == "-":
        genomic = seg_end - int(v)
    else:
        genomic = seg_start + int(v)
    return format_bp_mb(genomic, digits=2)

def build_segment_table(path_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg_idx, grp in path_df.groupby("seg_idx", sort=True):
        grp = grp.sort_values("path_bin")
        chrom = norm_chr(grp["chrom"].iloc[0])
        strand = str(grp["strand"].iloc[0])
        seg_start = safe_int(grp["segment_start_bp"].iloc[0], safe_int(grp["genomic_bp"].min(), 0))
        seg_end = safe_int(grp["segment_end_bp"].iloc[0], safe_int(grp["genomic_bp"].max(), 0))
        if seg_end < seg_start:
            seg_start, seg_end = seg_end, seg_start
        seg_len = max(int(seg_end - seg_start + 1), 1)
        path_bin_start = int(grp["path_bin"].min())
        path_bin_end = int(grp["path_bin"].max())
        # Internal sector name must be unique because duplicated ecDNA segments can
        # share identical genomic coordinates. The displayed sector label below is
        # the real genomic interval, not S1/S2.
        sector_name = f"seg{int(seg_idx):02d}|{chrom}:{seg_start}-{seg_end}|{strand}|bins{path_bin_start}-{path_bin_end}"
        sector_label = make_seg_label(chrom, seg_start, seg_end, strand, path_bin_start, path_bin_end)
        rows.append({"seg_idx": int(seg_idx),"sector_name": sector_name,"sector_label": sector_label,"chrom": chrom,
            "strand": strand,"segment_start_bp": int(seg_start),"segment_end_bp": int(seg_end),"segment_len_bp": int(seg_len),"path_bin_start": path_bin_start,"path_bin_end": path_bin_end})
    return pd.DataFrame(rows).sort_values("seg_idx").reset_index(drop=True)

def genomic_to_relative(start: int, end: int, seg: Dict[str, Any]) -> Tuple[int, int]:
    seg_start = int(seg["segment_start_bp"])
    seg_end = int(seg["segment_end_bp"])
    seg_len = int(seg["segment_len_bp"])
    start = max(min(int(start), seg_end), seg_start)
    end = max(min(int(end), seg_end), seg_start)
    if end < start:
        start, end = end, start

    if str(seg.get("strand", "+")) == "-":
        rel_start = seg_end - end
        rel_end = seg_end - start
    else:
        rel_start = start - seg_start
        rel_end = end - seg_start

    rel_start = max(0, min(seg_len - 1, int(rel_start)))
    rel_end = max(rel_start + 1, min(seg_len, int(rel_end)))
    return rel_start, rel_end

def harmonize_peak_bins(peaks: pd.DataFrame) -> pd.DataFrame:
    out = peaks.copy()
    # Layer7 block table has representative_bin1/2 and bin1/bin2. Prefer representative if present.
    if "bin1" not in out.columns and "rep_bin1" in out.columns:
        out["bin1"] = out["rep_bin1"]
    if "bin2" not in out.columns and "rep_bin2" in out.columns:
        out["bin2"] = out["rep_bin2"]
    if "bin1" in out.columns and "rep_bin1" in out.columns:
        out["bin1"] = out["rep_bin1"].fillna(out["bin1"])
    if "bin2" in out.columns and "rep_bin2" in out.columns:
        out["bin2"] = out["rep_bin2"].fillna(out["bin2"])
    if "bin1" in out.columns:
        out["bin1"] = pd.to_numeric(out["bin1"], errors="coerce").astype("Int64")
    if "bin2" in out.columns:
        out["bin2"] = pd.to_numeric(out["bin2"], errors="coerce").astype("Int64")
    return out

def attach_coordinates(peaks: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    peaks = harmonize_peak_bins(peaks)
    # If the peak table already contains the required coordinate columns, use them directly.
    needed = ["chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j"]
    seg_cols_ok = ("seg_idx_i" in peaks.columns and "seg_idx_j" in peaks.columns) or ("seg_idx_i_rep" in peaks.columns and "seg_idx_j_rep" in peaks.columns)
    if all(c in peaks.columns for c in needed) and seg_cols_ok:
        out = peaks.copy()
        if "seg_idx_i" not in out.columns and "seg_idx_i_rep" in out.columns:
            out["seg_idx_i"] = out["seg_idx_i_rep"]
        if "seg_idx_j" not in out.columns and "seg_idx_j_rep" in out.columns:
            out["seg_idx_j"] = out["seg_idx_j_rep"]
        return out

    ann_use = ann[[ "bin1", "bin2", "chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j", "seg_idx_i", "seg_idx_j" ]].drop_duplicates(["bin1", "bin2"])

    out = peaks.merge(ann_use, on=["bin1", "bin2"], how="left")
    missing = out["seg_idx_i"].isna() if "seg_idx_i" in out.columns else pd.Series(True, index=out.index)
    if missing.any():
        rev = ann_use.rename(columns={"bin1": "bin2", "bin2": "bin1",
            "chrom_i": "chrom_j", "start_i": "start_j", "end_i": "end_j", "seg_idx_i": "seg_idx_j",
            "chrom_j": "chrom_i", "start_j": "start_i", "end_j": "end_i", "seg_idx_j": "seg_idx_i"})
        fill = peaks.loc[missing, ["bin1", "bin2"]].merge(rev, on=["bin1", "bin2"], how="left")
        for c in ["chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j", "seg_idx_i", "seg_idx_j"]:
            if c in out.columns and c in fill.columns:
                out.loc[missing, c] = fill[c].values
    return out

def add_sector_coords(peaks: pd.DataFrame, seg_df: pd.DataFrame) -> pd.DataFrame:
    seg_map = seg_df.set_index("seg_idx").to_dict(orient="index")
    rows = []
    for _, row in peaks.iterrows():
        d = row.to_dict()
        si = safe_int(row.get("seg_idx_i"))
        sj = safe_int(row.get("seg_idx_j"))
        if si not in seg_map or sj not in seg_map:
            continue
        segi = seg_map[si]
        segj = seg_map[sj]
        ri0, ri1 = genomic_to_relative(safe_int(row.get("start_i"), 0), safe_int(row.get("end_i"), 0), segi)
        rj0, rj1 = genomic_to_relative(safe_int(row.get("start_j"), 0), safe_int(row.get("end_j"), 0), segj)
        d.update({"sector_i": segi["sector_name"], "rel_start_i": ri0, "rel_end_i": ri1,
                  "sector_j": segj["sector_name"], "rel_start_j": rj0, "rel_end_j": rj1})
        rows.append(d)
    return pd.DataFrame(rows)

def infer_score_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["blk_supp_weight", "combined_priority", "func_priority", "priority",
        "peak_support_weight", "rank_agg_priority", "cadet_score",
        "z_emp", "z_score", "log2_oe", "max_obs_over_exp_raw", "obs_over_exp_raw", "local_fc_mean"]:
        if c in df.columns:
            return c
    return None

def select_links(df: pd.DataFrame, top_n: int, top_modules: int, score_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if score_col is None:
        score_col = infer_score_col(out)
    if score_col and score_col in out.columns:
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce")

    if "module_id" in out.columns and "func_rank" in out.columns and top_modules > 0:
        out["func_rank"] = pd.to_numeric(out["func_rank"], errors="coerce")
        use = out[out["func_rank"].notna() & (out["func_rank"] <= top_modules)].copy()
        if len(use) > 0:
            sort_cols = ["func_rank"]
            asc = [True]
            if score_col and score_col in use.columns:
                sort_cols.append(score_col); asc.append(False)
            if "blk_supp_weight" in use.columns and "blk_supp_weight" not in sort_cols:
                sort_cols.append("blk_supp_weight"); asc.append(False)
            use = use.sort_values(sort_cols, ascending=asc)
            reps = use.groupby("module_id", sort=False, as_index=False).head(1)
            out = reps.copy()

    if score_col and score_col in out.columns:
        out = out.sort_values(score_col, ascending=False)
    elif "func_rank" in out.columns:
        out = out.sort_values("func_rank", ascending=True)

    if top_n > 0:
        out = out.head(top_n)
    return out.reset_index(drop=True)

def draw_circos(rep: pd.DataFrame,seg_df: pd.DataFrame,png_path: str, pdf_path: str, title: str, label_top: int,
    coord_label_mode: str = "range", min_sector_label_bp: int = 0, coord_label_fontsize: float = 18.0,
    tick_label_min_bp: int = 400000,color_by: str = "block",chrom_label_fontsize: float = 20.0,gene_label_fontsize: float = 15.0,legend_fontsize: float = 18.0, figure_size: float = 14.0) -> None:
    sectors = {r["sector_name"]: int(r["segment_len_bp"]) for _, r in seg_df.iterrows()}
    circos = Circos(sectors, space=2.5)
    sector_map = {s.name: s for s in circos.sectors}

    # Presentation-only threshold:
    # chromosome names on sectors shorter than this are moved outside the ring.
    # This does not change sector size, coordinates, links, or biological results.
    SHORT_SECTOR_BP = 100000

    for _, seg in seg_df.iterrows():
        sector = sector_map[seg["sector_name"]]
        seg_len = int(seg["segment_len_bp"])
        seg_idx = int(seg["seg_idx"])

        # Colored ecDNA path segment. Sector gaps themselves mark path breakpoints.
        outer = sector.add_track((96, 100))
        outer.axis(fc=segment_color(seg_idx), ec="white", lw=1.0)

        chrom_label = str(seg.get("chrom", ""))

        if seg_len >= SHORT_SECTOR_BP:
            try:
                outer.text(chrom_label,x=seg_len / 2,r=98,size=chrom_label_fontsize,color="white",adjust_rotation=True,)
            except Exception:
                outer.text(chrom_label,size=chrom_label_fontsize,color="white",r=98,)
        else:
            chr_radii = [106, 111, 116]
            r_chr = chr_radii[seg_idx % len(chr_radii)]
            try:
                sector.text(chrom_label,x=seg_len / 2, r=r_chr, size=chrom_label_fontsize, color="black", adjust_rotation=True, orientation="horizontal",)
            except Exception:
                sector.text(chrom_label,r=r_chr,size=chrom_label_fontsize, color="black", adjust_rotation=True, orientation="horizontal",)

        # Compact genomic coordinate range outside each sector.
        # Keep these farther out than short-sector chromosome names 
        if coord_label_mode in {"range", "both"} and seg_len >= int(min_sector_label_bp):
            label = (f"{format_bp_mb(seg['segment_start_bp'], digits=1)}–"f"{format_bp_mb(seg['segment_end_bp'], digits=1)} Mb")
            coord_radii = [123, 129, 135, 141]
            r_coord = coord_radii[seg_idx % len(coord_radii)]
            try:
                sector.text(label,x=seg_len / 2,r=r_coord,size=coord_label_fontsize, color="black", adjust_rotation=True, orientation="horizontal",)
            except Exception:
                sector.text(label,r=r_coord,size=coord_label_fontsize, color="black", adjust_rotation=True, orientation="horizontal",)

        # Optional sparse genomic Mb ticks for long sectors.
        if coord_label_mode in {"ticks", "both"} and seg_len >= int(tick_label_min_bp):
            tick = max(250000, int(math.ceil(seg_len / 3 / 50000.0) * 50000))
            try:
                outer.xticks_by_interval( tick, label_formatter=lambda v, seg=seg: format_genomic_tick(v, seg), label_size=coord_label_fontsize, )
            except Exception:
                pass

        inner = sector.add_track((92, 95))
        inner.axis(fc="#F2F4F7", ec="none")

    # Draw finalized links only; strength affects display weight but not selection.
    rep_sorted = rep.assign(
        _draw_weight=rep.apply(
            lambda r: safe_float( r.get( "blk_supp_weight", r.get("combined_priority", r.get("max_obs_over_exp_raw", 1)), ), 1.0, ),
            axis=1,
        )
    ).sort_values("_draw_weight", ascending=True)

    for _, row in rep_sorted.iterrows():
        color = evidence_color(row, color_by=color_by)
        score = float(row["_draw_weight"])
        weight_norm = min(max(score, 0.0), 10.0) / 10.0
        alpha = 0.40 + 0.50 * weight_norm
        lw = 0.9 + 1.6 * weight_norm
        circos.link(( row["sector_i"], int(row["rel_start_i"]), int(max(row["rel_end_i"], row["rel_start_i"] + 1)), ),
            ( row["sector_j"], int(row["rel_start_j"]), int(max(row["rel_end_j"], row["rel_start_j"] + 1)), ),
            color=color, alpha=alpha, lw=lw)

    # Readable target labels only.
    if label_top > 0:
        rep = rep.copy()
        tmp = rep.apply(lambda r: plot_label_and_side(r), axis=1)
        rep["_plot_label"] = [x[0] for x in tmp]
        rep["_plot_label_side"] = [x[1] for x in tmp]
        lab = rep.dropna(subset=["_plot_label"]).head(label_top)
        used_label_pairs = set()

        # Plotting-only collision control: retain the highest-priority readable
        used_sector_positions: Dict[str, List[float]] = {}

        for _, row in lab.iterrows():
            side = _valid_side(row.get("_plot_label_side")) or "j"
            if side == "i":
                sector_name = row.get("sector_i")
                x = (
                    safe_int(row.get("rel_start_i", 0), 0)
                    + safe_int(row.get("rel_end_i", 1), 1)
                ) / 2
            else:
                sector_name = row.get("sector_j", row.get("sector_i"))
                x = (
                    safe_int(row.get("rel_start_j", row.get("rel_start_i", 0)), 0)
                    + safe_int(row.get("rel_end_j", row.get("rel_end_i", 1)), 1)
                ) / 2

            if pd.isna(sector_name) or sector_name not in sector_map:
                continue

            # Suppress only visually overlapping text labels within the same sector.
            # Chords, selected links, coordinates, evidence classes, and tables are untouched.
            seg_len_for_label = float(sectors.get(sector_name, 1))
            x_norm = float(x) / max(seg_len_for_label, 1.0)
            occupied = used_sector_positions.setdefault(str(sector_name), [])
            if any(abs(x_norm - prev) < 0.055 for prev in occupied):
                continue

            key = (sector_name, row["_plot_label"])
            if key in used_label_pairs:
                continue

            sector = sector_map[sector_name]
            track = sector.add_track((67, 76))
            track.axis(fc="white", ec="none")

            label_str = str(row["_plot_label"])
            onco_cols = [ "target_oncogene", "is_oncogene", "target_oncogene_symbol", "target_oncokb", ]
            is_onco = False
            for c in onco_cols:
                v = row.get(c) if c in row.index else None
                if isinstance(v, str) and v.strip().lower() not in {"", "nan", "none", "false", "0"}:
                    is_onco = True
                    break
                if (
                    isinstance(v, (int, float, np.integer, np.floating))
                    and not pd.isna(v)
                    and float(v) > 0
                ):
                    is_onco = True
                    break

            color = "#B5179E" if is_onco else "#1A1A1A"
            track.text( label_str, x=x, size=gene_label_fontsize, orientation="horizontal", color=color, )
            used_label_pairs.add(key)
            occupied.append(x_norm)

    fig = circos.plotfig()
    fig.set_size_inches(figure_size, figure_size)
    fig.suptitle(title, fontsize=22, y=0.98)

    legend_items = [Line2D([0], [0], color="#B5179E", lw=2.2, label="Strict enhancer-oncogene"),
        Line2D([0], [0], color="#4361EE", lw=2.2, label="Permissive / context"),
        Line2D([0], [0], color="#4895EF", lw=2.2, label="Enhancer-gene context"),
        Line2D([0], [0], color="#F8961E", lw=2.2, label="Promoter disruption"),
        Line2D([0], [0], color="#8D99AE", lw=2.2, label="Local / context only")]
    fig.legend(handles=legend_items,loc="lower center",bbox_to_anchor=(0.5, -0.065),
        ncol=3,frameon=False,fontsize=legend_fontsize,handlelength=2.4,columnspacing=1.8)

    # Render once, then save the identical figure to both publication formats.
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="CADET circos plot")
    ap.add_argument("--input_dir", default="cadet_SI_final_output_", help="cadet_final_results folder")
    ap.add_argument("--path_bins", default=None)
    ap.add_argument("--layer3_csv", default=None)
    ap.add_argument("--layer7_peak_csv", default=None, help="Layer 7 block evidence")
    ap.add_argument("--module_csv", default=None, help="Layer 7 module summary")
    ap.add_argument("--part2_csv", default=None, help="Fallback peak table")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--top_n", type=int, default=50, help="Max links; 0 = all")
    ap.add_argument("--top_modules", type=int, default=20, help="Links from the top N modules; 0 = all")
    ap.add_argument("--label_top", type=int, default=8, help="Max gene labels")
    ap.add_argument("--coord_label_mode", choices=["range", "ticks", "both", "none"], default="range", help="Mb coordinates: one range per sector, ticks, both or none")
    ap.add_argument("--min_sector_label_bp", type=int, default=0, help="Min sector length (bp) for a coordinate label")
    ap.add_argument("--coord_label_fontsize", type=float, default=18.0, help="Coordinate label font size")
    ap.add_argument("--chrom_label_fontsize", type=float, default=20.0, help="Chromosome label font size")
    ap.add_argument("--gene_label_fontsize", type=float, default=15.0, help="Gene label font size")
    ap.add_argument("--legend_fontsize", type=float, default=18.0, help="Legend font size.")
    ap.add_argument("--figure_size", type=float, default=14.0, help="Figure size (inches)")
    ap.add_argument("--tick_label_min_bp", type=int, default=400000, help="Min sector length (bp) for Mb ticks")
    ap.add_argument("--score_col", default=None)
    ap.add_argument("--color_by", choices=["block", "module", "auto"], default="block", help="Color links by block or module evidence")
    ap.add_argument("--title", default="Interpreted ecDNA contact modules")
    args = ap.parse_args()

    input_dir = args.input_dir
    path_bins = args.path_bins or os.path.join(input_dir, "path_bins.tsv")
    layer3_csv = args.layer3_csv or os.path.join(input_dir, "upgraded_annotation", "significant_interactions_upgraded_annotated.csv")
    layer7_peak_csv = args.layer7_peak_csv or first_existing([
        os.path.join(input_dir, "layer7_module_interpretation", "layer7_block_evidence_all_modules.csv"),
        os.path.join(input_dir, "layer7_module_interpretation", "layer7_block_evidence_top_modules.csv"),
        os.path.join(input_dir, "layer7_module_interpretation", "layer7_peak_evidence_all_modules.csv"),
    ])
    module_csv = args.module_csv or os.path.join(input_dir, "layer7_module_interpretation", "layer7_module_summary_clean_report.csv")
    outdir = args.outdir or os.path.join(input_dir, "layer8b_ecDNA_circos_publication")
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(path_bins):
        raise FileNotFoundError(f"Missing path bins: {path_bins}")
    if not os.path.exists(layer3_csv):
        raise FileNotFoundError(f"Missing Layer3 annotation: {layer3_csv}")

    print("Loading path bins and annotations...")
    path_df = load_path_bins(path_bins)
    seg_df = build_segment_table(path_df)
    ann = load_layer3_annotation(layer3_csv)

    print("Loading interpreted block/link table...")
    peaks = load_peak_table(layer7_peak_csv, module_csv if os.path.exists(module_csv) else None, args.part2_csv)
    peaks = harmonize_peak_bins(peaks)
    if "bin1" not in peaks.columns or "bin2" not in peaks.columns:
        raise ValueError("Peak/block table must contain bin1 and bin2 or representative_bin1 and representative_bin2 columns.")

    print("Mapping links to ecDNA path sectors...")
    peaks = attach_coordinates(peaks, ann)
    mapped = add_sector_coords(peaks, seg_df)
    if len(mapped) == 0:
        raise ValueError("No links could be mapped to ecDNA path sectors.")

    rep = select_links(mapped, top_n=args.top_n, top_modules=args.top_modules, score_col=args.score_col)
    if len(rep) == 0:
        raise ValueError("No representative links selected for plotting.")

    tmp = rep.apply(lambda r: plot_label_and_side(r), axis=1)
    rep["plot_label_no_ens"] = [x[0] for x in tmp]
    rep["plot_label_side"] = [x[1] for x in tmp]
    rep["evidence_color_key"] = rep.apply(lambda r: canonical_evidence_key(r, color_by=args.color_by), axis=1)

    seg_out = os.path.join(outdir, "ecDNA_circos_segment_sectors.csv")
    link_out = os.path.join(outdir, "ecDNA_circos_plotted_links.csv")
    png_out = os.path.join(outdir, "ecDNA_circos.png")
    pdf_out = os.path.join(outdir, "ecDNA_circos.pdf")

    seg_df.to_csv(seg_out, index=False)
    rep.to_csv(link_out, index=False)

    print("Drawing circos plot...")
    draw_circos(rep, seg_df, png_out, pdf_out, title=args.title, label_top=args.label_top,
                coord_label_mode=args.coord_label_mode, min_sector_label_bp=args.min_sector_label_bp,
                coord_label_fontsize=args.coord_label_fontsize, tick_label_min_bp=args.tick_label_min_bp,
                color_by=args.color_by, chrom_label_fontsize=args.chrom_label_fontsize,
                gene_label_fontsize=args.gene_label_fontsize, legend_fontsize=args.legend_fontsize,
                figure_size=args.figure_size)

    params = vars(args).copy()
    params.update({
        "resolved_path_bins": path_bins, "resolved_layer3_csv": layer3_csv, "resolved_layer7_peak_csv": layer7_peak_csv,
        "resolved_module_csv": module_csv if os.path.exists(module_csv) else None, "n_segments": int(len(seg_df)), "n_input_links": int(len(peaks)),
        "n_mapped_links": int(len(mapped)), "n_plotted_links": int(len(rep)),
        "gene_label_rule": "oncogene for enhancer-oncogene links, gene for enhancer-gene/promoter links",
        "evidence_color_scheme": EVIDENCE_DISPLAY,
        "notes": "sectors are ecDNA path segments; duplicated segments are separate sectors"
    })
    with open(os.path.join(outdir, "ecDNA_circos_publication_parameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("Done.")
    print(f"Output PNG: {png_out}")
    print(f"Output PDF: {pdf_out}")
    print(f"Links:      {link_out}")

if __name__ == "__main__":
    main()
