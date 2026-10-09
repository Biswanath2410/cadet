#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd
from layer3_diagnostics import (plot_class_barplot,plot_ref_vs_ecdna,plot_discordance_vs_oe, plot_neighbors_vs_oe, plot_anchor_recurrence,plot_block_recurrence, plot_class_heatmap, plot_block_heatmap,plot_hub_vs_oe,)

# utilities

def safe_log10(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    return np.log10(np.maximum(x, eps))

def safe_divide(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a / np.maximum(b, eps)

def symmetrize_from_upper(mat_upper):
    mat_upper = np.asarray(mat_upper, dtype=float)
    return mat_upper + mat_upper.T - np.diag(np.diag(mat_upper))

# input tables

def load_sig_table(sig_csv):
    df = pd.read_csv(sig_csv)

    required = ["bin1", "bin2","distance","count_raw", "mu_raw","count_ice", "mu_ice", "obs_over_exp_raw", "obs_over_exp_ice", "log2_oe_raw", "log2_oe_ice", "local_obs_over_mean", "local_obs_over_q75", "passes_local", "peak_class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in significant_interactions.csv: {missing}")

    return df

def load_latent_pairs(latent_pairs_tsv):
    df = pd.read_csv(latent_pairs_tsv, sep="\t")

    required = ["path_bin_i", "path_bin_j","chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j", "seg_idx_i", "seg_idx_j", "ref_id_i", "ref_id_j", "path_distance_bins", "path_dist"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in latent_pairs.tsv: {missing}")

    return df

def add_latent_info(sig_df, lat_df):
    lat_use = lat_df[[ "path_bin_i", "path_bin_j", "chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j", "seg_idx_i", "seg_idx_j", "ref_id_i", "ref_id_j", "path_distance_bins", "path_dist"]].drop_duplicates().copy()

    out = sig_df.merge(lat_use,left_on=["bin1", "bin2"],right_on=["path_bin_i", "path_bin_j"],how="left")

    missing = out["chrom_i"].isna().sum()
    if missing > 0:
        raise ValueError(f"{missing} significant peaks could not be matched with latent_pairs.tsv. " "Check if bin1/bin2 correspond to path_bin_i/path_bin_j.")

    out = out.drop(columns=["path_bin_i", "path_bin_j"])
    return out
def load_path_bins(path_bins_tsv):
    df = pd.read_csv(path_bins_tsv, sep="\t")
    required = ["path_bin", "chrom", "genomic_bp", "seg_idx", "strand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in path_bins.tsv: {missing}")
    return df

#def load_observed_matrix(matrix_txt):
#    return np.loadtxt(matrix_txt, delimiter="\t")

# path/junction helpers

def find_segment_bounds(path_df):
    path_df = path_df.sort_values("path_bin").copy()

    seg_ranges = (path_df.groupby("seg_idx", as_index=False).agg(seg_start_bin=("path_bin", "min"), seg_end_bin=("path_bin", "max")).sort_values("seg_start_bin").reset_index(drop=True))

    junction_bins = set()
    junction_pairs = []

    for i in range(len(seg_ranges) - 1):
        left_end = int(seg_ranges.loc[i, "seg_end_bin"])
        right_start = int(seg_ranges.loc[i + 1, "seg_start_bin"])
        junction_bins.add(left_end)
        junction_bins.add(right_start)
        junction_pairs.append((left_end, right_start))

    return seg_ranges, sorted(junction_bins), junction_pairs

def find_discordant_joins(path_df, large_gap_bp=100000, continuity_tol_bp=10000):
    path_df = path_df.sort_values("path_bin").copy()

    if "strand" not in path_df.columns:
        raise ValueError("path_bins.tsv must contain a strand column for discordant join inference.")

    seg_info = (path_df.groupby("seg_idx", as_index=False).agg(chrom=("chrom", "first"), start_bp=("genomic_bp", "min"), end_bp=("genomic_bp", "max"), start_bin=("path_bin", "min"), end_bin=("path_bin", "max"),strand=("strand", "first")).sort_values("start_bin").reset_index(drop=True))

    discordant_bins = set()
    discordant_pairs = []
    rows = []

    for i in range(len(seg_info) - 1):
        left = seg_info.loc[i]
        right = seg_info.loc[i + 1]

        same_chr = str(left["chrom"]) == str(right["chrom"])
        left_strand = str(left["strand"])
        right_strand = str(right["strand"])

        if left_strand == "+":
            left_exit_bp = float(left["end_bp"])
        else:
            left_exit_bp = float(left["start_bp"])

        if right_strand == "+":
            right_entry_bp = float(right["start_bp"])
        else:
            right_entry_bp = float(right["end_bp"])

        ref_gap = abs(right_entry_bp - left_exit_bp)

        strand_flip = left_strand != right_strand
        continuity_break = (not same_chr) or strand_flip or (ref_gap > continuity_tol_bp)
        large_jump = (not same_chr) or (ref_gap > large_gap_bp)

        is_discordant = continuity_break

        if is_discordant:
            left_end = int(left["end_bin"])
            right_start = int(right["start_bin"])
            discordant_bins.add(left_end)
            discordant_bins.add(right_start)
            discordant_pairs.append((left_end, right_start))

        rows.append({"left_seg_idx": int(left["seg_idx"]), "right_seg_idx": int(right["seg_idx"]), "left_chr": str(left["chrom"]), "right_chr": str(right["chrom"]), "left_strand": left_strand, "right_strand": right_strand, "left_exit_bp": left_exit_bp, "right_entry_bp": right_entry_bp, "ref_gap_bp": ref_gap, "same_chr": same_chr, "strand_flip": strand_flip, "large_jump": large_jump, "is_discordant": is_discordant, "left_end_bin": int(left["end_bin"]), "right_start_bin": int(right["start_bin"]) })

    join_df = pd.DataFrame(rows)
    return seg_info, sorted(discordant_bins), discordant_pairs, join_df

def is_near_junction(bin_i, bin_j, junction_bins, proximity_bins=2):
    for jb in junction_bins:
        if abs(bin_i - jb) <= proximity_bins or abs(bin_j - jb) <= proximity_bins:
            return True
    return False

def add_structural_features(sig_df, junction_bins, discordant_join_bins=None, junction_pad=2):
    out = sig_df.copy()
    if len(out) == 0:
        return out

    if discordant_join_bins is None:
        discordant_join_bins = []

    required = [ "bin1", "bin2", "chrom_i", "chrom_j", "start_i", "end_i", "start_j", "end_j", "seg_idx_i", "seg_idx_j", "path_dist"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Cannot add structural features; missing columns: {missing}")

    # reference midpoints
    mid_i = (out["start_i"].astype(float) + out["end_i"].astype(float)) / 2.0
    mid_j = (out["start_j"].astype(float) + out["end_j"].astype(float)) / 2.0

    same_chr = out["chrom_i"].astype(str).values == out["chrom_j"].astype(str).values
    out["is_interchromosomal"] = ~same_chr
    out["chr_relation"] = np.where(same_chr, "same_chr", "interchromosomal")

    # ref distance only makes sense on the same chromosome
    out["ref_dist"] = np.where(same_chr, np.abs(mid_i - mid_j), np.nan)
    out["ecDNA_dist"] = out["path_dist"].astype(float)

    out["ref_over_ecDNA_ratio"] = np.where(same_chr,safe_divide(out["ref_dist"].values + 1.0, out["ecDNA_dist"].values + 1.0),np.nan)

    out["discord"] = np.where(same_chr, np.log10(out["ref_dist"].values + 1.0) - np.log10(out["ecDNA_dist"].values + 1.0),np.nan)

    seg_i = out["seg_idx_i"].astype(int).values
    seg_j = out["seg_idx_j"].astype(int).values
    seg_diff = np.abs(seg_i - seg_j)
    out["segment_relation"] = np.where(seg_diff == 0,"same_segment",
        np.where(seg_diff == 1, "adjacent_segment", "nonadjacent_segment"))

    out["near_junction"] = [is_near_junction(int(b1), int(b2), junction_bins, junction_pad)
        for b1, b2 in zip(out["bin1"], out["bin2"])]

    out["near_discordant_join"] = [is_near_junction(int(b1), int(b2), discordant_join_bins, junction_pad)
        for b1, b2 in zip(out["bin1"], out["bin2"])]

    # local neighborhood counts for later morphology/anchor-reuse 
    bins1 = out["bin1"].astype(int).values
    bins2 = out["bin2"].astype(int).values
    n = len(out)
    n_neighbors = np.zeros(n, dtype=int)
    for i in range(n):
        close_same = (np.abs(bins1 - bins1[i]) <= 5) & (np.abs(bins2 - bins2[i]) <= 5)
        close_swap = (np.abs(bins1 - bins2[i]) <= 5) & (np.abs(bins2 - bins1[i]) <= 5)
        n_neighbors[i] = int(np.sum(close_same | close_swap) - 1)

    out["n_sig_neighbors_w5"] = n_neighbors
    out["n_same_anchor_peaks_w5"] = count_anchor_neighbors(out, window_bins=5)
    out["is_hub_associated"] = out["n_sig_neighbors_w5"] >= 8

    return out

def count_anchor_neighbors(sig_df, window_bins=5):
    bins1 = sig_df["bin1"].values.astype(int)
    bins2 = sig_df["bin2"].values.astype(int)
    n = len(sig_df)

    neighbors = np.zeros(n, dtype=int)

    for i in range(n):
        bi1 = bins1[i]
        bi2 = bins2[i]
        c1 = np.sum((np.abs(bins1 - bi1) <= window_bins) | (np.abs(bins2 - bi1) <= window_bins)) - 1
        c2 = np.sum((np.abs(bins1 - bi2) <= window_bins) | (np.abs(bins2 - bi2) <= window_bins)) - 1

        neighbors[i] = int(max(c1, c2))

    return neighbors

# merge nearby significant pixels

def merge_pixels(sig_df, radius_bin1=2, radius_bin2=2):
    df = sig_df.copy().reset_index(drop=True)
    n = len(df)

    if n == 0:
        df["peak_blk_id"] = pd.Series(dtype=int)
        return df

    # Coordinates of all significant pixels
    bin1 = df["bin1"].values.astype(int)
    bin2 = df["bin2"].values.astype(int)

    # For every significant pixel, store which other pixels are connected to it.
    connections = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):

            # Normal orientation:
            # (bin1_i, bin2_i) compared with (bin1_j, bin2_j)
            bin1_close = abs(bin1[i] - bin1[j]) <= radius_bin1
            bin2_close = abs(bin2[i] - bin2[j]) <= radius_bin2

            same_orientation = bin1_close and bin2_close

            # Swapped orientation : the contact map is symmetric:
            
            swapped_bin1_close = abs(bin1[i] - bin2[j]) <= radius_bin1
            swapped_bin2_close = abs(bin2[i] - bin1[j]) <= radius_bin2

            swapped_orientation = swapped_bin1_close and swapped_bin2_close

            # If they are close, connect them in both directions.
            if same_orientation or swapped_orientation:
                connections[i].append(j)
                connections[j].append(i)

    peak_block_id = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)

    block_number = 0

    for start_pixel in range(n):

        # Already assigned through an earlier connected group
        if visited[start_pixel]:
            continue

        # found a new peak block
        block_number += 1

        # Start exploring from this pixel
        to_visit = [start_pixel]
        visited[start_pixel] = True

        while to_visit:
            current_pixel = to_visit.pop()

            # Assign this pixel to the current peak block
            peak_block_id[current_pixel] = block_number

            # Look at all significant pixels connected to it
            for neighbor_pixel in connections[current_pixel]:

                if not visited[neighbor_pixel]:
                    visited[neighbor_pixel] = True
                    to_visit.append(neighbor_pixel)

    df["peak_blk_id"] = peak_block_id

    return df

def block_neighbor_counts(block_df, window_bins=5):
    n = len(block_df)
    neighbors = np.zeros(n, dtype=int)

    c1 = block_df["blk_bin1_center"].values.astype(float)
    c2 = block_df["blk_bin2_center"].values.astype(float)

    for i in range(n):
        count = 0
        for j in range(n):
            if i == j:
                continue

            same_orientation = (abs(c1[i] - c1[j]) <= window_bins and abs(c2[i] - c2[j]) <= window_bins)
            swapped_orientation = (abs(c1[i] - c2[j]) <= window_bins and abs(c2[i] - c1[j]) <= window_bins)
            if same_orientation or swapped_orientation:
                count += 1
        neighbors[i] = count

    return neighbors

def block_anchor_recurrence(block_df, window_bins=5):
    n = len(block_df)
    recurrence = np.zeros(n, dtype=int)
    c1 = block_df["blk_bin1_center"].values.astype(float)
    c2 = block_df["blk_bin2_center"].values.astype(float)

    for i in range(n):
        ai = (c1[i], c2[i])
        count = 0
        for j in range(n):
            if i == j:
                continue
            aj = (c1[j], c2[j])
            same_anchor = (abs(ai[0] - aj[0]) <= window_bins or abs(ai[0] - aj[1]) <= window_bins or abs(ai[1] - aj[0]) <= window_bins or abs(ai[1] - aj[1]) <= window_bins)
            if same_anchor:
                count += 1
        recurrence[i] = count
    return recurrence

def summarize_blocks(ann_df):
    if len(ann_df) == 0:
        return pd.DataFrame()

    rows = []

    for block_id, grp in ann_df.groupby("peak_blk_id", sort=True):
        grp = grp.copy().reset_index(drop=True)

        rep = grp.sort_values(["obs_over_exp_raw", "local_obs_over_mean", "obs_over_exp_ice"], ascending=False).iloc[0]

        rows.append({"peak_blk_id": int(block_id),"n_pixels": int(len(grp)),
                     "blk_bin1_min": int(grp["bin1"].min()), "blk_bin1_max": int(grp["bin1"].max()), 
                     "blk_bin2_min": int(grp["bin2"].min()), "blk_bin2_max": int(grp["bin2"].max()), 
                     "blk_bin1_center": float(grp["bin1"].mean()), "blk_bin2_center": float(grp["bin2"].mean()), "rep_bin1": int(rep["bin1"]), "rep_bin2": int(rep["bin2"]), "chrom_i": str(rep["chrom_i"]), "start_i": int(rep["start_i"]), "end_i": int(rep["end_i"]), "chrom_j": str(rep["chrom_j"]), "start_j": int(rep["start_j"]), "end_j": int(rep["end_j"]), "chr_relation": str(rep["chr_relation"]), "is_interchromosomal": bool(rep["is_interchromosomal"]), "segment_relation": str(rep["segment_relation"]), "seg_idx_i_rep": int(rep["seg_idx_i"]), "seg_idx_j_rep": int(rep["seg_idx_j"]), "ref_id_i_rep": int(rep["ref_id_i"]), "ref_id_j_rep": int(rep["ref_id_j"]), 
                     "ref_dist": float(rep["ref_dist"]) if pd.notna(rep["ref_dist"]) else np.nan, "ecDNA_dist": float(rep["ecDNA_dist"]), 
                     "ref_over_ecDNA_ratio": float(rep["ref_over_ecDNA_ratio"]) if pd.notna(rep["ref_over_ecDNA_ratio"]) else np.nan,
            
            "discord": float(rep["discord"]) if pd.notna(rep["discord"]) else np.nan, # keep value for quick checking
            # block-level summaries
"max_discord": float(grp["discord"].max(skipna=True)) if grp["discord"].notna().any() else np.nan,
"mean_discord": float(grp["discord"].mean(skipna=True)) if grp["discord"].notna().any() else np.nan,
"near_junction": bool(grp["near_junction"].any()), "near_discordant_join": bool(grp["near_discordant_join"].any()), "max_obs_over_exp_raw": float(grp["obs_over_exp_raw"].max()), "mean_obs_over_exp_raw": float(grp["obs_over_exp_raw"].mean()), "max_obs_over_exp_ice": float(grp["obs_over_exp_ice"].max()), "mean_obs_over_exp_ice": float(grp["obs_over_exp_ice"].mean()), "max_local_oe": float(grp["local_obs_over_mean"].max()), "mean_local_obs_over_mean": float(grp["local_obs_over_mean"].mean()), "any_passes_local": bool(grp["passes_local"].any()), "representative_peak_class": str(rep["peak_class"]), "representative_upgraded_peak_class": str(rep["upgraded_peak_class"])})

    block_df = pd.DataFrame(rows)
    block_df["block_width_bins"] = block_df["blk_bin2_max"] - block_df["blk_bin2_min"] + 1
    block_df["block_height_bins"] = block_df["blk_bin1_max"] - block_df["blk_bin1_min"] + 1
    block_df["block_long_axis_bins"] = np.maximum(block_df["block_width_bins"], block_df["block_height_bins"])
    block_df["block_short_axis_bins"] = np.maximum(1, np.minimum(block_df["block_width_bins"], block_df["block_height_bins"]))
    block_df["block_aspect_ratio"] = block_df["block_long_axis_bins"] / block_df["block_short_axis_bins"]
    block_df["n_blk_neighbors_w5"] = block_neighbor_counts(block_df, window_bins=5)
    block_df["n_same_anchor_blk_w5"] = block_anchor_recurrence(block_df, window_bins=5)
    # candidate flag for Layer 4 shared-anchor modules. intentionally looser
    # (>= 2) than the anchor_recurrent_contact label rule 
    block_df["is_anchor_recur_blk"] = block_df["n_same_anchor_blk_w5"] >= 2
    block_df["is_hub_associated_block"] = block_df["n_blk_neighbors_w5"] >= 6

    return block_df

def classify_peak_blocks(block_df, sv_discordance_min=1.0, sv_raw_oe_min=2.2, sv_local_oe_min=1.8, rw_disc_strong=1.2, rw_disc_moderate=1.0, rw_raw_oe_min=2.0, rw_local_oe_min=1.5, loop_discordance_max=0.2,loop_raw_oe_min=1.8, loop_local_oe_min=1.8, loop_neighbor_max=4, loop_anchor_neighbor_max=2, stripe_span_min_bins=12, stripe_aspect_min=4.0, compact_block_max_bins=6):
    out = block_df.copy()
    if len(out) == 0:
        return out

    primary_labels = []
    ranks = []
    morphology = []
    structural_contexts = []
    junction_contexts = []
    confidence = []

    is_focal_list = []
    is_compact_list = []
    anchor_rec_flags = []
    stripe_flags = []
    is_enriched_list = []

    for _, row in out.iterrows():
        is_interchr = bool(row.get("is_interchromosomal", False))
        near_junction = bool(row.get("near_junction", False))
        near_discordant_join = bool(row.get("near_discordant_join", False))
        seg_rel = str(row.get("segment_relation", "unknown"))

        disc = row.get("discord", np.nan)
        disc = float(disc) if pd.notna(disc) else np.nan

        n_nb = int(row.get("n_blk_neighbors_w5", 0))
        n_anchor = int(row.get("n_same_anchor_blk_w5", 0))
        local_ok = bool(row.get("any_passes_local", False))
        oe_raw = float(row.get("max_obs_over_exp_raw", 0.0))
        local_fc = float(row.get("max_local_oe", 0.0))

        width = float(row.get("block_width_bins", 1.0))
        height = float(row.get("block_height_bins", 1.0))
        long_axis = float(row.get("block_long_axis_bins", max(width, height)))
        aspect = float(row.get("block_aspect_ratio", max(width, height) / max(min(width, height), 1.0)))

        is_enriched = (oe_raw >= loop_raw_oe_min) or (local_fc >= loop_local_oe_min)
        is_focal = bool(local_ok and oe_raw >= loop_raw_oe_min and local_fc >= loop_local_oe_min)
        is_compact = bool(width <= compact_block_max_bins and height <= compact_block_max_bins)
        is_anchor_recurrent = bool(n_anchor > loop_anchor_neighbor_max)
        is_shape_stripe = bool(long_axis >= stripe_span_min_bins and aspect >= stripe_aspect_min)
        is_broad_or_hub = bool(n_nb >= loop_neighbor_max)

        is_focal_list.append(is_focal)
        is_compact_list.append(is_compact)
        anchor_rec_flags.append(is_anchor_recurrent)
        stripe_flags.append(is_shape_stripe)
        is_enriched_list.append(is_enriched)

        # morphology only; no chromosome/segment rule here
        if is_focal and is_compact and not is_anchor_recurrent and not is_shape_stripe:
            morph = "compact_focal_contact"
        elif is_shape_stripe:
            morph = "stripe_like_contact"
        elif is_anchor_recurrent:
            morph = "anchor_recurrent_contact"
        elif is_broad_or_hub:
            morph = "hub_or_broad_contact"
        elif is_enriched:
            morph = "enriched_unassigned_contact"
        else:
            morph = "weak_significant_contact"

        # structural context
        if is_interchr:
            struct = "interchromosomal_context"
        elif pd.notna(disc) and disc >= rw_disc_strong:
            struct = "same_chr_strong_rewiring_context"
        elif pd.notna(disc) and disc >= rw_disc_moderate:
            struct = "same_chr_distal_proximity_context"
        elif seg_rel == "same_segment":
            struct = "same_segment_context"
        elif seg_rel == "adjacent_segment":
            struct = "adjacent_segment_context"
        else:
            struct = "unassigned_structural_context"

        # junction context
        if near_discordant_join:
            jctx = "near_discordant_path_join"
        elif near_junction:
            jctx = "near_segment_junction"
        else:
            jctx = "not_junction_near"

        # primary label for plots/tables; keep the individual flags above
        if near_discordant_join and is_enriched:
            label = "discordant_join_associated_contact"
            rank = 1
        elif is_anchor_recurrent or is_shape_stripe:
            label = "anchor_recurrent_contact"
            rank = 2
        elif is_focal and is_compact:
            label = "focal_contact_candidate"
            rank = 3
        elif near_junction and is_enriched:
            label = "junction_neighbor_contact"
            rank = 4
        else:
            label = "unclassified_significant_contact"
            rank = 9

        if label in {"discordant_join_associated_contact", "focal_contact_candidate"} and is_focal:
            conf = "higher_confidence"
        elif is_enriched:
            conf = "moderate_confidence"
        else:
            conf = "low_confidence"

        morphology.append(morph)
        structural_contexts.append(struct)
        junction_contexts.append(jctx)
        primary_labels.append(label)
        ranks.append(rank)
        confidence.append(conf)

    out["is_focal_contact_cand"] = is_focal_list
    out["is_compact_block"] = is_compact_list
    out["is_anchor_recur_v4"] = anchor_rec_flags
    out["is_shape_stripe_v4"] = stripe_flags
    out["is_enriched_contact_v4"] = is_enriched_list

    out["morphology_blk_class"] = morphology
    out["struct_blk_ctx"] = structural_contexts
    out["junction_blk_ctx"] = junction_contexts
    out["confidence_blk_class"] = confidence

    out["primary_blk_label"] = primary_labels
    out["upgraded_peak_block_class"] = primary_labels
    out["block_class_rank"] = ranks

    return out

def classify_peaks(sig_df, sv_discordance_min=1.0, sv_raw_oe_min=2.2, sv_local_oe_min=1.8, rw_disc_strong=1.2, rw_disc_moderate=1.0, rw_raw_oe_min=2.0, rw_local_oe_min=1.5, loop_discordance_max=0.2,loop_raw_oe_min=1.8, loop_local_oe_min=1.8, loop_neighbor_max=8, loop_anchor_neighbor_max=2):
    out = sig_df.copy()
    if len(out) == 0:
        return out

    primary_labels = []
    ranks = []
    morphology = []
    structural_contexts = []
    junction_contexts = []
    confidence = []

    is_focal_list = []
    anchor_rec_flags = []
    is_enriched_list = []

    for _, row in out.iterrows():
        is_interchr = bool(row.get("is_interchromosomal", False))
        near_junction = bool(row.get("near_junction", False))
        near_discordant_join = bool(row.get("near_discordant_join", False))
        seg_rel = str(row.get("segment_relation", "unknown"))

        disc = row.get("discord", np.nan)
        disc = float(disc) if pd.notna(disc) else np.nan

        n_nb = int(row.get("n_sig_neighbors_w5", 0))
        n_anchor = int(row.get("n_same_anchor_peaks_w5", 0))
        local_ok = bool(row.get("passes_local", False))
        oe_raw = float(row.get("obs_over_exp_raw", 0.0))
        local_fc = float(row.get("local_obs_over_mean", 0.0))

        is_enriched = (oe_raw >= loop_raw_oe_min) or (local_fc >= loop_local_oe_min)
        is_focal = bool(local_ok and oe_raw >= loop_raw_oe_min and local_fc >= loop_local_oe_min)
        is_anchor_recurrent = bool(n_anchor > loop_anchor_neighbor_max)
        is_broad_or_hub = bool(n_nb >= loop_neighbor_max)

        is_focal_list.append(is_focal)
        anchor_rec_flags.append(is_anchor_recurrent)
        is_enriched_list.append(is_enriched)

        # morphology only; no same-chromosome/same-segment rule
        if is_focal and not is_anchor_recurrent:
            morph = "focal_contact_pixel"
        elif is_anchor_recurrent:
            morph = "anchor_recurrent_pixel"
        elif is_broad_or_hub:
            morph = "hub_or_broad_pixel"
        elif is_enriched:
            morph = "enriched_unassigned_pixel"
        else:
            morph = "weak_significant_pixel"

        # structural context
        if is_interchr:
            struct = "interchromosomal_context"
        elif pd.notna(disc) and disc >= rw_disc_strong:
            struct = "same_chr_strong_rewiring_context"
        elif pd.notna(disc) and disc >= rw_disc_moderate:
            struct = "same_chr_distal_proximity_context"
        elif seg_rel == "same_segment":
            struct = "same_segment_context"
        elif seg_rel == "adjacent_segment":
            struct = "adjacent_segment_context"
        else:
            struct = "unassigned_structural_context"

        # junction context
        if near_discordant_join:
            jctx = "near_discordant_path_join"
        elif near_junction:
            jctx = "near_segment_junction"
        else:
            jctx = "not_junction_near"

        # primary label for plots/tables
        if near_discordant_join and is_enriched:
            label = "discordant_join_associated_contact"
            rank = 1
        elif is_anchor_recurrent:
            label = "anchor_recurrent_contact"
            rank = 2
        elif is_focal:
            label = "focal_contact_candidate"
            rank = 3
        elif near_junction and is_enriched:
            label = "junction_neighbor_contact"
            rank = 4
        else:
            label = "unclassified_significant_contact"
            rank = 9

        if label in {"discordant_join_associated_contact", "focal_contact_candidate"} and is_focal:
            conf = "higher_confidence"
        elif is_enriched:
            conf = "moderate_confidence"
        else:
            conf = "low_confidence"

        morphology.append(morph)
        structural_contexts.append(struct)
        junction_contexts.append(jctx)
        primary_labels.append(label)
        ranks.append(rank)
        confidence.append(conf)

    out["is_focal_contact_cand"] = is_focal_list
    out["is_anchor_recur_v4"] = anchor_rec_flags
    out["is_enriched_contact_v4"] = is_enriched_list

    out["morphology_class"] = morphology
    out["struct_ctx"] = structural_contexts
    out["junction_ctx"] = junction_contexts
    out["confidence_class"] = confidence

    out["primary_label"] = primary_labels
    out["upgraded_peak_class"] = primary_labels
    out["class_rank"] = ranks

    return out

def summarize_classes(df, class_col="upgraded_peak_class"):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[class_col, "n"])
    if class_col not in df.columns:
        return pd.DataFrame(columns=[class_col, "n"])
    out = (df[class_col].fillna("unclassified_significant_contact").astype(str).value_counts(dropna=False).rename_axis(class_col)
        .reset_index(name="n"))
    return out

# main runner

def main():
    # inputs
    parser = argparse.ArgumentParser(description="CADET Layer 3: contact blocks")
    parser.add_argument("--input_dir", default="cadet_SI_final_output_",
                        help="Layer 3 input folder")
    parser.add_argument("--outdir", default=None,
                        help="Output folder")
    parser.add_argument("--matrix", default=None,
                        help="Latent matrix (default: in input_dir)")
    parser.add_argument("--junction_pad", type=int, default=2)
    parser.add_argument("--large_gap_bp", type=float, default=100000)
    parser.add_argument("--continuity_tol_bp", type=float, default=10000)
    parser.add_argument("--merge_radius_bin1", type=int, default=2,
                        help="Merge radius for bin1")
    parser.add_argument("--merge_radius_bin2", type=int, default=2,
                        help="Merge radius for bin2")
    parser.add_argument("--fig_width", type=float, default=10.0)
    parser.add_argument("--fig_height", type=float, default=10.0)
    parser.add_argument("--draw_centers", action="store_true", default=True,
                        help="Mark block centers")
    parser.add_argument("--no_draw_centers", action="store_false", dest="draw_centers")
    parser.add_argument("--min_box_size", type=float, default=5.0)
    parser.add_argument("--legend_fontsize", type=float, default=18.0)
    parser.add_argument("--loop_anchor_neighbor_max", type=int, default=2,
                        help="Max same-anchor neighbors before a block is anchor-recurrent")
    parser.add_argument("--compact_block_max_bins", type=int, default=6,
                        help="Max size (bins) of a focal block")
    parser.add_argument("--stripe_span_min_bins", type=int, default=12,
                        help="Min span (bins) of a stripe")
    parser.add_argument("--stripe_aspect_min", type=float, default=4.0,
                        help="Min aspect ratio of a stripe")
    args = parser.parse_args()

    input_dir = args.input_dir
    sig_csv = os.path.join(input_dir, "significant_interactions.csv")
    latent_pairs_tsv = os.path.join(input_dir, "latent_pairs.tsv")
    path_bins_tsv = os.path.join(input_dir, "path_bins.tsv")
    latent_matrix_txt = args.matrix or os.path.join(input_dir, "latent_deconvolved_matrix_symmetric_display.txt")
    outdir = args.outdir or os.path.join(input_dir, "upgraded_annotation")
    plot_dir = os.path.join(outdir, "plots")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # thresholds from current CADET runs
    # TODO: revisit these if Part 2 peak density changes a lot across samples
    junction_pad = args.junction_pad
    sv_discordance_min = 1.0
    sv_raw_oe_min = 2.2
    sv_local_oe_min = 1.8

    rw_disc_strong = 1.2
    rw_disc_moderate = 1.0
    rw_raw_oe_min = 2.0

    loop_discordance_max = 0.2
    loop_raw_oe_min = 1.8
    loop_local_oe_min = 1.8
    loop_neighbor_max = 8
    # merge settings
    # NOTE: small radius; this is meant to merge adjacent pixels, not broad domains
    merge_radius_bin1 = args.merge_radius_bin1
    merge_radius_bin2 = args.merge_radius_bin2

    # load inputs
    sig_df = load_sig_table(sig_csv)
    lat_df = load_latent_pairs(latent_pairs_tsv)
    sig_df = add_latent_info(sig_df, lat_df)

    path_df = load_path_bins(path_bins_tsv)
    #observed_matrix = load_observed_matrix(latent_matrix_txt)
    observed_matrix = np.loadtxt(latent_matrix_txt, delimiter="\t")
    seg_ranges, junction_bins, junction_pairs = find_segment_bounds(path_df)
    seg_info, discordant_join_bins, discordant_join_pairs, join_df = find_discordant_joins(
    path_df,
    large_gap_bp=args.large_gap_bp,
    continuity_tol_bp=args.continuity_tol_bp
    )
    pd.DataFrame({"junction_bin": junction_bins}).to_csv(os.path.join(outdir, "junction_bins.csv"), index=False)

    pd.DataFrame({"discordant_join_bin": discordant_join_bins}).to_csv(os.path.join(outdir, "discordant_join_bins.csv"), index=False)

    pd.DataFrame(discordant_join_pairs, columns=["left_bin", "right_bin"]).to_csv(os.path.join(outdir, "discordant_join_pairs.csv"), index=False)
    join_df.to_csv(os.path.join(outdir, "discordant_join_annotation.csv"), index=False)

    # annotate pixels
    ann = add_structural_features(
    sig_df=sig_df,
    junction_bins=junction_bins,
    discordant_join_bins=discordant_join_bins,
    junction_pad=junction_pad
    )
    #    sig_df=sig_df,
    #    junction_bins=junction_bins,
    #    junction_pad=junction_pad
    #)
    ann = classify_peaks(ann, sv_discordance_min=sv_discordance_min, sv_raw_oe_min=sv_raw_oe_min,
    sv_local_oe_min=sv_local_oe_min, rw_disc_strong=rw_disc_strong,
    rw_disc_moderate=rw_disc_moderate, rw_raw_oe_min=rw_raw_oe_min,
    loop_discordance_max=loop_discordance_max,loop_raw_oe_min=loop_raw_oe_min,loop_local_oe_min=loop_local_oe_min,
    loop_neighbor_max=loop_neighbor_max,loop_anchor_neighbor_max=args.loop_anchor_neighbor_max)
    # merge significant pixels into event-level blocks
    merged_pixels = merge_pixels(ann,radius_bin1=merge_radius_bin1,radius_bin2=merge_radius_bin2)

    block_df = summarize_blocks(merged_pixels)

    block_df = classify_peak_blocks(block_df,sv_discordance_min=sv_discordance_min,sv_raw_oe_min=sv_raw_oe_min, sv_local_oe_min=sv_local_oe_min,rw_disc_strong=rw_disc_strong, rw_disc_moderate=rw_disc_moderate, rw_raw_oe_min=rw_raw_oe_min, loop_discordance_max=loop_discordance_max, loop_raw_oe_min=loop_raw_oe_min, loop_local_oe_min=loop_local_oe_min, loop_neighbor_max=4, loop_anchor_neighbor_max=args.loop_anchor_neighbor_max, stripe_span_min_bins=args.stripe_span_min_bins, stripe_aspect_min=args.stripe_aspect_min, compact_block_max_bins=args.compact_block_max_bins)
    
    block_class_summary = summarize_classes(block_df.rename(columns={"upgraded_peak_block_class": "upgraded_peak_class"}),
        class_col="upgraded_peak_class")

    class_summary = summarize_classes(ann, class_col="upgraded_peak_class") #class_col="upgraded_peak_block_class"
    hub_summary = pd.DataFrame({"n_total_peaks": [len(ann)],
                                   "n_hub_associated": [int(ann["is_hub_associated"].sum())],
                                   "frac_hub_associated": [float(ann["is_hub_associated"].mean())]})

    # save tables
    ann.to_csv(os.path.join(outdir, "significant_interactions_upgraded_annotated.csv"), index=False)
    class_summary.to_csv(os.path.join(outdir, "upgraded_class_summary.csv"), index=False)
    hub_summary.to_csv(os.path.join(outdir, "hub_summary.csv"), index=False)
    seg_ranges.to_csv(os.path.join(outdir, "segment_ranges.csv"), index=False)
    seg_info.to_csv(os.path.join(outdir, "segment_info.csv"), index=False)
    merged_pixels.to_csv(os.path.join(outdir, "significant_pixels_with_peak_blocks.csv"), index=False)
    # Layer 4 reads the bin/label/enrichment columns, Layer 6 and the circos plot the anchor coordinates
    block_table_cols = ["peak_blk_id", "n_pixels",
    "blk_bin1_min", "blk_bin1_max", "blk_bin2_min", "blk_bin2_max",
    "blk_bin1_center", "blk_bin2_center", "rep_bin1", "rep_bin2",
    "chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j", "seg_idx_i_rep", "seg_idx_j_rep",
    "primary_blk_label", "morphology_blk_class", "struct_blk_ctx",
    "junction_blk_ctx", "confidence_blk_class", "segment_relation", "is_interchromosomal",
     "max_obs_over_exp_raw", "mean_obs_over_exp_raw", "max_local_oe",
    "max_discord", "mean_discord",
    "near_junction", "near_discordant_join",
    "n_blk_neighbors_w5", "n_same_anchor_blk_w5", "is_anchor_recur_blk"]
    block_df[block_table_cols].to_csv(os.path.join(outdir, "merged_peak_blocks_annotated.csv"), index=False)
    block_class_summary.to_csv(os.path.join(outdir, "merged_peak_block_class_summary.csv"), index=False)
    with open(os.path.join(outdir, "annotation_parameters.json"), "w") as f:
        json.dump({"junction_proximity_bins": junction_pad,
        "large_gap_bp": args.large_gap_bp,
        "continuity_tol_bp": args.continuity_tol_bp,
        "sv_discordance_min": sv_discordance_min,
        "sv_raw_oe_min": sv_raw_oe_min,
        "sv_local_oe_min": sv_local_oe_min,
        "rewiring_discordance_strong": rw_disc_strong,
        "rewiring_discordance_moderate": rw_disc_moderate,
        "rewiring_raw_oe_min": rw_raw_oe_min,
        "loop_discordance_max": loop_discordance_max,
        "loop_raw_oe_min": loop_raw_oe_min,
        "loop_local_oe_min": loop_local_oe_min,
        "loop_neighbor_max": loop_neighbor_max,
        "merge_radius_bin1": merge_radius_bin1,
        "merge_radius_bin2": merge_radius_bin2,
        "loop_anchor_neighbor_max": args.loop_anchor_neighbor_max,
        "stripe_span_min_bins": args.stripe_span_min_bins,
        "stripe_aspect_min": args.stripe_aspect_min,
        "compact_block_max_bins": args.compact_block_max_bins,
        "layer3_design": "v4_feature_annotation_conservative_primary_labels"
        }, f, indent=2)

    plot_class_barplot(class_summary, os.path.join(plot_dir, "upgraded_class_summary.png"), class_col="upgraded_peak_class")

    plot_ref_vs_ecdna(ann, os.path.join(plot_dir, "ref_vs_ecdna_distance.png"))

    plot_discordance_vs_oe(ann,os.path.join(plot_dir, "discordance_vs_raw_enrichment.png"))

    plot_neighbors_vs_oe(ann,os.path.join(plot_dir, "neighbors_vs_raw_enrichment.png"))

    plot_anchor_recurrence(ann,os.path.join(plot_dir, "same_anchor_recurrence_vs_enrichment.png"))

    plot_block_recurrence(block_df, os.path.join(plot_dir, "block_same_anchor_recurrence_vs_enrichment.png"))

    plot_class_heatmap(matrix=observed_matrix,df=ann,outpath=os.path.join(plot_dir, "significant_heatmap_upgraded_classes.png"),title="Primary contact labels",path_df=path_df)

    plot_block_heatmap(matrix=observed_matrix, pixel_df=merged_pixels, block_df=block_df, outpath=os.path.join(plot_dir, "merged_peak_blocks_on_heatmap.png"), title="Primary labels for merged contact blocks", draw_pixels=False,draw_boxes=True,draw_centers=args.draw_centers,figsize=(args.fig_width, args.fig_height), legend_fontsize=args.legend_fontsize, min_box_size=args.min_box_size,path_df=path_df)
    
    plot_block_heatmap(matrix=observed_matrix,pixel_df=merged_pixels, block_df=block_df, outpath=os.path.join(plot_dir, "merged_peak_blocks_on_heatmap.pdf"), title="Primary labels for merged contact blocks", draw_pixels=False, draw_boxes=True, draw_centers=args.draw_centers, figsize=(args.fig_width, args.fig_height), legend_fontsize=args.legend_fontsize, min_box_size=args.min_box_size, path_df=path_df)
    plot_hub_vs_oe(ann,os.path.join(plot_dir, "hub_flag_vs_enrichment.png"))
  
    print("Upgraded annotation completed.")
    print(f"Annotated peaks: {len(ann)}")
    print("\nClass summary:")
    print(class_summary.to_string(index=False))
    print(f"\nOutputs written to: {outdir}")
    print(f"\nMerged peak blocks: {len(block_df)}")
    print("\nMerged block class summary:")
    print(block_class_summary.to_string(index=False))

if __name__ == "__main__":
    main()
