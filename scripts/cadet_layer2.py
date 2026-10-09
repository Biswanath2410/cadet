#!/usr/bin/env python3

import os
import json
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from iced import normalization as iced_normalization

warnings.filterwarnings("ignore", category=RuntimeWarning)

PUB_DPI = 300
PUB_FIGSIZE_SQUARE = (10, 10)
PUB_FIGSIZE_RECT = (10, 6)
PUB_FONTSIZE = 22
PUB_TICK_FONTSIZE = 22
PUB_CBAR_FONTSIZE = 22
PUB_LEGEND_FONTSIZE = 18

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_square_matrix(path):
    mat = np.loadtxt(path, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix must be square: {path}")
    return mat

def symmetrize(mat):
    return (np.asarray(mat, dtype=float) + np.asarray(mat, dtype=float).T) / 2.0

def safe_divide(a, b, eps=1e-12):
    return np.asarray(a, dtype=float) / np.maximum(np.asarray(b, dtype=float), eps)

def robust_scale_iqr(x, min_scale=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return min_scale
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < min_scale:
        scale = min_scale
    return float(scale)

def robust_center_scale(x, trim_lower=0.025, trim_upper=0.975, min_scale=0.05, scale_method="iqr"):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, min_scale, 0
    lo, hi = np.quantile(x, [trim_lower, trim_upper])
    bg = x[(x >= lo) & (x <= hi)]
    if bg.size == 0:
        bg = x
    center = float(np.median(bg))
    if scale_method == "iqr":
        scale = robust_scale_iqr(bg, min_scale=min_scale)
    elif scale_method == "mad":
        mad = np.median(np.abs(bg - np.median(bg)))
        scale = max(float(1.4826 * mad), min_scale)
    else:
        raise ValueError("scale_method must be iqr or mad")
    return center, scale, int(bg.size)

def ice_obs_exp(X_raw, B_raw):
    if iced_normalization is None:
        raise ImportError("--use_shared_ice requires the iced package.")
    if X_raw.shape != B_raw.shape:
        raise ValueError("X_raw and B_raw must have same shape for shared ICE normalization.")

    X_work = np.asarray(X_raw, dtype=float).copy()
    B_work = np.asarray(B_raw, dtype=float).copy()
    X_work[~np.isfinite(X_work)] = 0.0
    B_work[~np.isfinite(B_work)] = 0.0
    X_work[X_work < 0] = 0.0
    B_work[B_work < 0] = 0.0

    X_ice, bias = iced_normalization.ICE_normalization(X_work, output_bias=True)
    bias = np.asarray(bias, dtype=float)
    bias[~np.isfinite(bias)] = 1.0
    bias[bias <= 0] = 1.0

    B_ice = B_work / np.outer(bias, bias)
    X_ice[~np.isfinite(X_ice)] = 0.0
    B_ice[~np.isfinite(B_ice)] = 0.0

    return symmetrize(X_ice), symmetrize(B_ice), bias

# upper triangle only, Part 3 expects bin1/bin2 pairs from here
def matrices_to_pairs(X, B, res, eps=1.0):
    if X.shape != B.shape:
        raise ValueError("X and B must have same shape")
    n = X.shape[0]
    i, j = np.triu_indices(n, k=1)
    delta = np.abs(i - j)
    dist_bin = np.minimum(delta, n - delta)
    obs = X[i, j].astype(float)
    exp = B[i, j].astype(float)
    oe = (obs + eps) / (exp + eps)
    return pd.DataFrame({"bin1": i.astype(int), "bin2": j.astype(int), "count": obs, "mu_hat": exp, "dist_bin": dist_bin.astype(int), "dist": (dist_bin * res).astype(float),
                         "obs_over_exp": oe, "log2_oe": np.log2(oe)})

def select_eligible_pairs(df, min_test_dist_bins=2, min_mu=1e-8):
    out = df.copy()
    out = out[out["dist_bin"] >= min_test_dist_bins].copy()
    out = out[np.isfinite(out["count"]) & np.isfinite(out["mu_hat"]) & np.isfinite(out["log2_oe"])].copy()
    out = out[out["mu_hat"] > min_mu].copy()
    return out.reset_index(drop=True)

def assign_distance_strata(df, short_max_bins=10, mid_max_bins=30):
    out = df.copy()
    cond_short = out["dist_bin"] <= short_max_bins
    cond_mid = (out["dist_bin"] > short_max_bins) & (out["dist_bin"] <= mid_max_bins)
    out["distance_stratum"] = np.select([cond_short, cond_mid], ["short", "mid"], default="long")
    return out

def mask_sister_offsets(df, n, sister_offset=None, sister_tol=0):
    out = df.copy()
    if sister_offset is None:
        out["is_sister_masked"] = False
        return out
    raw_delta = np.abs(out["bin2"].values - out["bin1"].values)
    circ_delta = np.minimum(raw_delta, n - raw_delta)
    target = min(int(sister_offset), n - int(sister_offset))
    masked = np.abs(circ_delta - target) <= int(sister_tol)
    out["is_sister_masked"] = masked
    return out[~masked].reset_index(drop=True)

def compute_oe_matrix(X, B, eps=1.0):
    return safe_divide(X + eps, B + eps)

# local focality: skip the inner square around the candidate 
def local_window_stats(matrix, i, j, outer_radius=4, inner_radius=1, min_bg_pixels=8, circular=False):
    n = matrix.shape[0]
    if circular:
        rows = np.mod(np.arange(i - outer_radius, i + outer_radius + 1), n)
        cols = np.mod(np.arange(j - outer_radius, j + outer_radius + 1), n)
        wrap_affected = bool((i - outer_radius < 0) or (i + outer_radius >= n) or (j - outer_radius < 0) or (j + outer_radius >= n))
    else:
        r0, r1 = max(0, i - outer_radius), min(n, i + outer_radius + 1)
        c0, c1 = max(0, j - outer_radius), min(n, j + outer_radius + 1)
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        wrap_affected = bool((i - outer_radius < 0) or (i + outer_radius >= n) or (j - outer_radius < 0) or (j + outer_radius >= n))
    sub = matrix[np.ix_(rows, cols)].astype(float)
    rr = rows[:, None]
    cc = cols[None, :]
    center_mask = (np.abs(rr - i) <= inner_radius) & (np.abs(cc - j) <= inner_radius)
    bg = sub[~center_mask]
    bg = bg[np.isfinite(bg)]
    center_value = float(matrix[i, j])
    if len(bg) < min_bg_pixels:
        return {"local_mean": np.nan, "local_q75": np.nan, "local_n": int(len(bg)), "is_local_max": False, "wrap_affected": wrap_affected}
    return {"local_mean": float(np.mean(bg)),"local_q75": float(np.quantile(bg, 0.75)),"local_n": int(len(bg)), "is_local_max": bool(center_value >= np.nanmax(bg)),"wrap_affected": wrap_affected}

def add_local_focality(df, oe_matrix, outer_radius=4, inner_radius=1, min_bg_pixels=8, circular=False):
    out = df.copy()
    rows = []
    for _, row in out.iterrows():
        rows.append(local_window_stats(oe_matrix, int(row["bin1"]), int(row["bin2"]), outer_radius, inner_radius, min_bg_pixels, circular=circular))
    s = pd.DataFrame(rows)
    out["local_oe_mean"] = s["local_mean"].values
    out["local_oe_q75"] = s["local_q75"].values
    out["local_bg_n"] = s["local_n"].values
    out["is_local_max"] = s["is_local_max"].values
    out["local_wrap_affected"] = s["wrap_affected"].values
    out["local_fc_mean"] = safe_divide(out["obs_over_exp"], out["local_oe_mean"])
    out["local_fc_q75"] = safe_divide(out["obs_over_exp"], out["local_oe_q75"])
    return out

# fit background per circular distance, fall back to short/mid/long when sparse
def fit_background(df, trim_lower=0.025, trim_upper=0.975, min_pairs=50, min_scale=0.05, scale_method="iqr"):
    out = df.copy()
    out["bg_center"] = np.nan
    out["bg_scale"] = np.nan
    out["bg_n_used"] = 0
    out["bg_source"] = ""
    summary_rows = []

    stratum_params = {}
    for stratum, grp in out.groupby("distance_stratum", sort=False):
        center, scale, n_used = robust_center_scale(grp["log2_oe"].values, trim_lower, trim_upper, min_scale, scale_method)
        stratum_params[stratum] = (center, scale, n_used)

    for dist, grp in out.groupby("dist_bin", sort=True):
        idx = grp.index.values
        if len(grp) >= min_pairs:
            center, scale, n_used = robust_center_scale(grp["log2_oe"].values, trim_lower, trim_upper, min_scale, scale_method)
            source = f"dist_{dist}"
        else:
            stratum = grp["distance_stratum"].iloc[0]
            center, scale, n_used = stratum_params[stratum]
            source = f"stratum_{stratum}"
        out.loc[idx, "bg_center"] = center
        out.loc[idx, "bg_scale"] = scale
        out.loc[idx, "bg_n_used"] = n_used
        out.loc[idx, "bg_source"] = source
        summary_rows.append({"dist_bin": int(dist), "n_pairs": int(len(grp)), "bg_center": center, "bg_scale": scale, "bg_n_used": int(n_used), "bg_source": source})
    return out, pd.DataFrame(summary_rows)

def add_empirical_pvalues(df, min_scale=0.05):
    out = df.copy()
    scale = np.maximum(out["bg_scale"].values.astype(float), min_scale)
    out["z_emp"] = (out["log2_oe"].values - out["bg_center"].values) / scale
    out["p_value"] = 1.0 - norm.cdf(out["z_emp"].values)
    out["p_value"] = np.clip(out["p_value"].values, 0.0, 1.0)
    out["q_value"] = np.nan
    out["passes_fdr"] = False
    out["in_fdr_universe"] = False
    return out

def fdr_universe_mask(df, fdr_universe="focal", local_fc_threshold=1.2, require_local_max=True, count_floor=1.0):
    if fdr_universe == "eligible":
        return np.ones(len(df), dtype=bool)
    if fdr_universe != "focal":
        raise ValueError("fdr_universe must be 'focal' or 'eligible'")
    mask = (np.isfinite(df["local_fc_mean"].values) & (df["local_fc_mean"].values >= local_fc_threshold) & (df["count"].values >= count_floor))
    if require_local_max:
        mask = mask & (df["is_local_max"].values == True)
    return mask

def bh_fdr_to_universe(df, universe_mask, fdr_alpha=0.05):
    out = df.copy()
    universe_mask = np.asarray(universe_mask, dtype=bool)
    valid = universe_mask & np.isfinite(out["p_value"].values)
    out.loc[valid, "in_fdr_universe"] = True
    if valid.sum() > 0:
        q = multipletests(out.loc[valid, "p_value"].values, alpha=fdr_alpha, method="fdr_bh")[1]
        out.loc[valid, "q_value"] = q
        out.loc[valid, "passes_fdr"] = out.loc[valid, "q_value"] <= fdr_alpha
    return out

# final reported peaks: FDR pass + focality + per-stratum/top-k pruning
def build_candidates(df, candidate_topk_total=50, cand_frac=0.10, cand_min=10, local_fc_threshold=1.2, require_local_max=True, count_floor=1.0, require_global_fdr=True):
    out = df.copy()
    focal = np.isfinite(out["local_fc_mean"]) & (out["local_fc_mean"] >= local_fc_threshold) & (out["count"] >= count_floor)
    if require_local_max:
        focal = focal & (out["is_local_max"] == True)
    if require_global_fdr:
        focal = focal & (out["passes_fdr"] == True)
    out = out[focal].copy()
    if len(out) == 0:
        return out
    keep_idx = []
    for stratum, grp in out.groupby("distance_stratum", sort=False):
        n_keep = max(cand_min, int(math.ceil(cand_frac * len(grp))))
        n_keep = min(n_keep, len(grp))
        grp2 = grp.sort_values(["q_value", "z_emp", "log2_oe", "local_fc_mean"], ascending=[True, False, False, False]).head(n_keep)
        keep_idx.extend(grp2.index.tolist())
    out = out.loc[sorted(set(keep_idx))].copy()
    out = out.sort_values(["q_value", "z_emp", "log2_oe", "local_fc_mean"], ascending=[True, False, False, False]).copy()
    if candidate_topk_total is not None and int(candidate_topk_total) > 0:
        out = out.head(int(candidate_topk_total)).copy()
    out["cand_rank"] = np.arange(1, len(out) + 1)
    out["is_cand"] = True
    out["is_significant"] = True
    return out.reset_index(drop=True)

def call_part2(X, B, res, eps=1.0, min_test_dist_bins=2, short_max_bins=10, mid_max_bins=30, outer_radius=4, inner_radius=1, min_bg_pixels=8, circular_local_window=False, trim_lower=0.025, trim_upper=0.975, min_pairs=50, min_scale=0.05, scale_method="iqr", candidate_topk_total=50, cand_frac=0.10, cand_min=10, local_fc_threshold=1.2, require_local_max=True, count_floor=1.0, fdr_alpha=0.05, exclude_sister_offset=None, sister_tolerance=0, require_global_fdr=True, fdr_universe="focal"):
    X = symmetrize(X)
    B = symmetrize(B)
    n = X.shape[0]
    df_all = matrices_to_pairs(X, B, res=res, eps=eps)
    df_eligible = select_eligible_pairs(df_all, min_test_dist_bins=min_test_dist_bins)
    df_eligible = assign_distance_strata(df_eligible, short_max_bins=short_max_bins, mid_max_bins=mid_max_bins)
    before_sister = len(df_eligible)
    df_eligible = mask_sister_offsets(df_eligible, n=n, sister_offset=exclude_sister_offset, sister_tol=sister_tolerance)
    n_sister_masked = before_sister - len(df_eligible)
    oe = compute_oe_matrix(X, B, eps=eps)
    df_eligible = add_local_focality(df_eligible, oe, outer_radius, inner_radius, min_bg_pixels, circular=circular_local_window)
    df_eligible, bg_summary = fit_background(df_eligible, trim_lower, trim_upper, min_pairs, min_scale, scale_method)
    df_tested_all = add_empirical_pvalues(df_eligible, min_scale=min_scale)
    fdr_mask = fdr_universe_mask(df_tested_all, fdr_universe=fdr_universe, local_fc_threshold=local_fc_threshold, require_local_max=require_local_max, count_floor=count_floor)
    df_tested_all = bh_fdr_to_universe(df_tested_all, fdr_mask, fdr_alpha=fdr_alpha)
    sig_df = build_candidates(df_tested_all, candidate_topk_total, cand_frac, cand_min, local_fc_threshold, require_local_max, count_floor, require_global_fdr)
    run_summary = pd.DataFrame([{"n_all_pairs": int(len(df_all)), "n_eligible_pairs": int(len(df_eligible)), "n_in_fdr_universe": int(df_tested_all["in_fdr_universe"].sum()),
        "n_fdr_pass": int(df_tested_all["passes_fdr"].sum()), "n_candidates_final": int(len(sig_df)), "fdr_universe": str(fdr_universe),
        "fdr_alpha": float(fdr_alpha), "local_fc_threshold": float(local_fc_threshold),"cand_topk_total": int(candidate_topk_total),
        "require_global_fdr": bool(require_global_fdr)}])
    return {"df_all": df_all, "df_eligible_tested_all": df_tested_all, "background_summary": bg_summary, "sig_df": sig_df, "run_summary": run_summary}

def _empty_plot(outpath, message):
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=PUB_FONTSIZE)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def _pctl(vals, pct, default=1.0):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default
    v = np.nanpercentile(vals, pct)
    return float(v) if np.isfinite(v) and v > 0 else default

def load_path_plot_layout(path_bins_file, res):
    if path_bins_file is None:
        return None, None

    df = pd.read_csv(path_bins_file, sep="\t")
    required = {"path_bin", "seg_idx", "chrom", "genomic_bp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"path_bins.tsv is missing required columns: {sorted(missing)}")
        
    df = df.sort_values("path_bin").reset_index(drop=True)
    boundaries = [0]
    intervals = []

    for seg_idx, grp in df.groupby("seg_idx", sort=False):
        start_path = int(grp["path_bin"].min())
        end_path = int(grp["path_bin"].max()) + 1

        chrom = str(grp.iloc[0]["chrom"])
        start_bp = int(grp["genomic_bp"].min())
        end_bp = int(grp["genomic_bp"].max()) + res
        intervals.append((chrom, start_bp, end_bp))

        if boundaries[-1] != start_path:
            boundaries.append(start_path)

        if boundaries[-1] != end_path:
            boundaries.append(end_path)

    return boundaries, intervals

def add_path_annotations(ax, boundaries, intervals):
    if boundaries is None or intervals is None:
        return

    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, linestyle="--", linewidth=1, color="black", alpha=0.7)
        ax.axhline(b - 0.5, linestyle="--", linewidth=1, color="black", alpha=0.7)

    ax.set_yticks(boundaries)
    ax.set_yticklabels([str(x) for x in boundaries], fontsize=PUB_TICK_FONTSIZE)

    centers = [(boundaries[i] + boundaries[i + 1] - 1) / 2 for i in range(len(intervals))]
    labels = [f"{chrom}:{start/1e6:.2f}-{end/1e6:.2f} Mb" for chrom, start, end in intervals]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=PUB_TICK_FONTSIZE)

def plot_matrix_heatmap(matrix, outpath, title, cmap="YlOrRd", log10=True, boundaries=None,intervals=None, cbar_label=None):
    mat = np.asarray(matrix, dtype=float)
    data = np.ma.log10(np.ma.masked_less_equal(mat, 0)) if log10 else np.ma.masked_invalid(mat)
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_SQUARE)
    im = ax.matshow(data, cmap=cmap)
    ax.set_title(title, fontsize=PUB_FONTSIZE)
    ax.set_xlabel("Path bin", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Path bin", fontsize=PUB_FONTSIZE)
    add_path_annotations(ax, boundaries, intervals)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=PUB_CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_peak_heatmap(matrix, sig_df, outpath, title,boundaries=None,intervals=None):
    mat = np.asarray(matrix, dtype=float)
    data = np.ma.log10(np.ma.masked_less_equal(mat, 0))
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_SQUARE)
    im = ax.matshow(data, cmap="YlOrRd")
    if sig_df is not None and len(sig_df) > 0:
        ax.scatter(sig_df["bin2"], sig_df["bin1"], s=44, facecolors="none", edgecolors="blue", linewidths=1.2)
    ax.set_title(title, fontsize=PUB_FONTSIZE)
    ax.set_xlabel("Path bin", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Path bin", fontsize=PUB_FONTSIZE)
    add_path_annotations(ax, boundaries, intervals)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(X_{ij})$", fontsize=PUB_CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_log2oe_heatmap(log2oe, sig_df, outpath, boundaries=None,intervals=None):
    mat = np.asarray(log2oe, dtype=float)
    vmax = _pctl(np.abs(mat[np.isfinite(mat)]), 99, default=1.0)
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_SQUARE)
    im = ax.matshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    if sig_df is not None and len(sig_df) > 0:
        ax.scatter(sig_df["bin2"], sig_df["bin1"], s=44, facecolors="none", edgecolors="black", linewidths=1.2)
    ax.set_title(r"Latent $\log_2$(O/E) with final CADET peaks", fontsize=PUB_FONTSIZE)
    ax.set_xlabel("Path bin", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Path bin", fontsize=PUB_FONTSIZE)
    add_path_annotations(ax, boundaries, intervals)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_2$(O/E)", fontsize=PUB_CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_score_histogram(df, outpath):
    vals = df["log2_oe"].values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        _empty_plot(outpath, r"No eligible $\log_2$(O/E) values")
        return
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.hist(vals, bins=80)
    ax.set_xlabel(r"$\log_2$(O/E)", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Number of eligible pairs", fontsize=PUB_FONTSIZE)
    ax.set_title(r"Eligible-pair $\log_2$(O/E) distribution", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_qq_pvalues(df, outpath):
    p = df["p_value"].values.astype(float)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if p.size == 0:
        _empty_plot(outpath, "No valid p-values")
        return
    p = np.sort(p)
    exp = -np.log10((np.arange(1, p.size + 1) - 0.5) / p.size)
    obs = -np.log10(p)
    m = max(float(np.nanmax(exp)), float(np.nanmax(obs)))
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.scatter(exp, obs, s=8, alpha=0.6)
    ax.plot([0, m], [0, m], linestyle="--", linewidth=1)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=PUB_FONTSIZE)
    ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=PUB_FONTSIZE)
    ax.set_title("Empirical p-value Q–Q plot", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_z_vs_distance(df, sig_df, outpath):
    base = df[np.isfinite(df["dist"]) & (df["dist"] > 0) & np.isfinite(df["z_emp"])].copy()
    if base.empty:
        _empty_plot(outpath, "No valid z-scores")
        return
    sample = base.sample(n=min(len(base), 100000), random_state=1) if len(base) > 100000 else base
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.scatter(sample["dist"] / 1000.0, sample["z_emp"], s=3, alpha=0.20, label="Eligible pairs")
    if sig_df is not None and len(sig_df) > 0:
        ax.scatter(sig_df["dist"] / 1000.0, sig_df["z_emp"], s=36, marker="x", label="Final peaks")
    ax.set_xscale("log")
    ax.set_xlabel("ecDNA circular distance (kb)", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Empirical z-score", fontsize=PUB_FONTSIZE)
    ax.set_title("Distance-aware z-scores", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(fontsize=PUB_LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def summarize_by_distance(df):
    return df.groupby("dist_bin", sort=True).agg(dist=("dist", "first"),n_tested=("bin1", "size"),n_fdr=("passes_fdr", "sum")).reset_index()

def plot_sig_by_distance(df, sig_df, outpath, n_bins=25, min_tested_per_bin=30):
    tmp = df[np.isfinite(df["dist"]) & (df["dist"] > 0)].copy()
    if tmp.empty:
        _empty_plot(outpath, "No distance summary data")
        return
    keys = set()
    if sig_df is not None and len(sig_df) > 0:
        keys = set(zip(sig_df["bin1"].astype(int), sig_df["bin2"].astype(int)))
    tmp["is_final"] = [(int(a), int(b)) in keys for a, b in zip(tmp["bin1"], tmp["bin2"])]
    dmin, dmax = tmp["dist"].min(), tmp["dist"].max()
    if dmin == dmax:
        _empty_plot(outpath, "Only one tested distance")
        return
    tmp["distance_group"] = pd.cut(tmp["dist"], np.logspace(np.log10(dmin), np.log10(dmax), n_bins + 1), include_lowest=True)
    g = tmp.groupby("distance_group", observed=False).agg(
        n_tested=("bin1", "size"), n_fdr=("passes_fdr", "sum"),
        n_final=("is_final", "sum"), d_left=("dist", "min"), d_right=("dist", "max")
    ).reset_index(drop=True)
    g = g[g["n_tested"] >= min_tested_per_bin].copy()
    if g.empty:
        _empty_plot(outpath, "No distance bins passed minimum tested threshold")
        return
    g["dist_mid"] = np.sqrt(g["d_left"] * g["d_right"])
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.plot(g["dist_mid"] / 1000.0, g["n_fdr"] / g["n_tested"], marker="o", linewidth=2, label="FDR-significant")
    ax.plot(g["dist_mid"] / 1000.0, g["n_final"] / g["n_tested"], marker="o", linewidth=2, label="Final peaks")
    ax.set_xscale("log")
    ax.set_xlabel("ecDNA circular distance (kb)", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Fraction of eligible pairs", fontsize=PUB_FONTSIZE)
    ax.set_title("Significant interactions by distance", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(fontsize=PUB_LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_local_enrichment(sig_df, outpath, local_fc_threshold):
    if sig_df is None or len(sig_df) == 0:
        _empty_plot(outpath, "No final significant peaks")
        return
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ax.scatter(np.maximum(sig_df["obs_over_exp"], 1e-8), np.maximum(sig_df["local_fc_mean"], 1e-8), s=32, alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(local_fc_threshold, linestyle="--", linewidth=1)
    ax.set_xlabel("Global O/E", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Local O/E fold change", fontsize=PUB_FONTSIZE)
    ax.set_title("Global versus local enrichment of final peaks", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def make_part2_plots(X, B, result, outdir, local_fc_threshold, boundaries=None, intervals=None):
    plot_dir = os.path.join(outdir, "plots")
    ensure_dir(plot_dir)
    Xs = symmetrize(X)
    Bs = symmetrize(B)
    sig_df = result["sig_df"]
    df = result["df_eligible_tested_all"]
    oe = compute_oe_matrix(Xs, Bs, eps=1.0)
    log2oe = np.log2(np.maximum(oe, 1e-12))
    plot_matrix_heatmap(Xs, os.path.join(plot_dir, "latent_X_heatmap.png"), "Latent contact matrix", boundaries=boundaries, intervals=intervals, cbar_label=r"$\log_{10}(X_{ij})$")
    plot_matrix_heatmap(Bs, os.path.join(plot_dir, "expected_lambda_heatmap.png"), "Expected contact matrix", boundaries=boundaries, intervals=intervals, cbar_label=r"$\log_{10}(\lambda_{ij})$")
    plot_peak_heatmap(Xs, sig_df, os.path.join(plot_dir, "latent_X_with_final_peaks.png"), "Latent contact matrix with final CADET peaks", boundaries=boundaries,intervals=intervals)
    plot_log2oe_heatmap(log2oe, sig_df, os.path.join(plot_dir, "log2OE_with_final_peaks.png"),boundaries=boundaries,intervals=intervals)
    plot_score_histogram(df, os.path.join(plot_dir, "eligible_log2OE_histogram.png"))
    plot_qq_pvalues(df, os.path.join(plot_dir, "empirical_pvalue_QQ.png"))
    plot_z_vs_distance(df, sig_df, os.path.join(plot_dir, "z_score_vs_distance.png"))
    plot_sig_by_distance(df, sig_df, os.path.join(plot_dir, "significance_by_distance.png"))
    plot_local_enrichment(sig_df, os.path.join(plot_dir, "local_enrichment_final_peaks.png"), local_fc_threshold)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--path_bins",default=None,help="Layer 1 path_bins.tsv")
    ap.add_argument("--res", required=True, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--min_test_dist_bins", type=int, default=2)
    ap.add_argument("--short_max_bins", type=int, default=10)
    ap.add_argument("--mid_max_bins", type=int, default=30)
    ap.add_argument("--outer_radius", type=int, default=4)
    ap.add_argument("--inner_radius", type=int, default=1)
    ap.add_argument("--min_bg_pixels", type=int, default=8)
    ap.add_argument("--circular_local_window", action="store_true", help="Wrap the local window around the circle")
    ap.add_argument("--trim_lower", type=float, default=0.025)
    ap.add_argument("--trim_upper", type=float, default=0.975)
    ap.add_argument("--min_pairs", type=int, default=50)
    ap.add_argument("--min_scale", type=float, default=0.05)
    ap.add_argument("--scale_method", choices=["iqr", "mad"], default="iqr")
    ap.add_argument("--candidate_topk_total", type=int, default=50)
    ap.add_argument("--cand_frac", type=float, default=0.10)
    ap.add_argument("--cand_min", type=int, default=10)
    ap.add_argument("--local_fc_threshold", type=float, default=1.2)
    ap.add_argument("--count_floor", type=float, default=1.0)
    ap.add_argument("--no_local_max", action="store_true")
    ap.add_argument("--fdr_alpha", type=float, default=0.05)
    ap.add_argument("--exclude_sister_offset", type=int, default=None)
    ap.add_argument("--sister_tolerance", type=int, default=0)
    ap.add_argument("--no_require_global_fdr", action="store_true", help="Do not require FDR for final calls")
    ap.add_argument("--fdr_universe", choices=["focal", "eligible"], default="focal", help="FDR over focal candidates or all eligible pairs")
    ap.add_argument("--use_shared_ice", action="store_true", help="Apply ICE bias from X to both X and B")
    ap.add_argument("--save_ice_matrices", action="store_true", help="Save ICE-normalized matrices")
    ap.add_argument("--no_plots", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    path_boundaries, path_intervals = load_path_plot_layout(args.path_bins,args.res)
    X_raw = load_square_matrix(args.x)
    B_raw = load_square_matrix(args.b)

    if args.use_shared_ice:
        X, B, shared_bias = ice_obs_exp(X_raw, B_raw)
        np.savetxt(os.path.join(args.out, "shared_ice_bias.txt"), shared_bias, fmt="%.10g")
        if args.save_ice_matrices:
            np.savetxt(os.path.join(args.out, "latent_X_shared_ICE_symmetric.txt"), X, fmt="%.10g")
            np.savetxt(os.path.join(args.out, "lambda_shared_ICE_symmetric.txt"), B, fmt="%.10g")
    else:
        X = symmetrize(X_raw)
        B = symmetrize(B_raw)

    res = call_part2(
        X, B, res=args.res, eps=args.eps, min_test_dist_bins=args.min_test_dist_bins,
        short_max_bins=args.short_max_bins, mid_max_bins=args.mid_max_bins,
        outer_radius=args.outer_radius, inner_radius=args.inner_radius, min_bg_pixels=args.min_bg_pixels,
        circular_local_window=args.circular_local_window,
        trim_lower=args.trim_lower, trim_upper=args.trim_upper,
        min_pairs=args.min_pairs, min_scale=args.min_scale, scale_method=args.scale_method,
        candidate_topk_total=args.candidate_topk_total, cand_frac=args.cand_frac,
        cand_min=args.cand_min, local_fc_threshold=args.local_fc_threshold,
        require_local_max=(not args.no_local_max), count_floor=args.count_floor, fdr_alpha=args.fdr_alpha,
        exclude_sister_offset=args.exclude_sister_offset, sister_tolerance=args.sister_tolerance,
        require_global_fdr=(not args.no_require_global_fdr), fdr_universe=args.fdr_universe)
    res["df_all"].to_csv(os.path.join(args.out, "all_pairs.csv"), index=False)
    res["df_eligible_tested_all"].sort_values(["q_value", "z_emp"], ascending=[True, False]).to_csv(os.path.join(args.out, "eligible_pairs_tested_with_focal_fdr.csv"), index=False)
    res["background_summary"].to_csv(os.path.join(args.out, "background_summary.csv"), index=False)
    res["sig_df"].to_csv(os.path.join(args.out, "final_significant_interactions.csv"), index=False)
    res["run_summary"].to_csv(os.path.join(args.out, "run_summary.csv"), index=False)
    summarize_by_distance(res["df_eligible_tested_all"]).to_csv(os.path.join(args.out, "significance_by_distance.csv"), index=False)
    if not args.no_plots:
        make_part2_plots(X, B, res, args.out, args.local_fc_threshold, boundaries=path_boundaries, intervals=path_intervals)
    meta = vars(args).copy()
    meta["normalization_used_for_calling"] = "shared_ICE" if args.use_shared_ice else "raw_part1_matrices"
    with open(os.path.join(args.out, "part2_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(res["run_summary"].to_string(index=False))
    print(f"Saved final peaks to: {os.path.join(args.out, 'final_significant_interactions.csv')}")

if __name__ == "__main__":
    main()
