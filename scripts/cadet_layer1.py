#!/usr/bin/env python3
"""CADET Layer 1: deconvolve collapsed ecDNA Hi-C into copy-specific contacts.
Y[u, v] ~ Poisson(sum over family(u, v) of lambda_ij), with
lambda_ij = pi_i * pi_j * alpha_ij * f(d_ij). f(d) is fitted backbone from clean
single-copy pairs.
"""

import os
import json
import math
import warnings
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd

from layer1_diagnostics import (plot_ec3d_style, plot_centered, plot_diff, plot_backbone_fit,
    plot_alpha_dist, plot_convergence, plot_x_vs_lambda, plot_hist, plot_oe_spread, plot_family_metrics,
    recollapse_metrics, family_ambiguity, oe_spread, top1pct_overlap, alpha_bound_summary)

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_MAD_THRESH = 3.0

def upper_pair(i, j):
    return (i, j) if i <= j else (j, i)

def sym(mat):
    m = np.asarray(mat, dtype=float)
    return m + m.T - np.diag(np.diag(m))

def circ_dist(i, j, n):
    d = abs(i - j)
    return min(d, n - d)

def robust_z(x):
    x   = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.zeros_like(x) if mad == 0 else 0.6745 * (x - med) / mad

def safe_corr(a, b):
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def seg_to_bins(start_bp, end_bp, res):
    if end_bp < start_bp:
        raise ValueError(f"Invalid interval [{start_bp}, {end_bp}]")
    return start_bp // res, end_bp // res

def build_path_bin_table(intrvl, sign, res):
    rows = []
    bins = [0]
    segment_ranges = []

    for chrom, start_bp, end_bp in intrvl:
        start_bin, end_bin = seg_to_bins(start_bp, end_bp, res)
        n_bins = end_bin - start_bin
        segment_ranges.append((start_bin, end_bin, n_bins))
        bins.append(bins[-1] + n_bins)

    path_bin = 0

    for seg_idx in range(len(intrvl)):
        chrom, start_bp, end_bp = intrvl[seg_idx]
        strand = sign[seg_idx]
        start_bin, end_bin, n_bins = segment_ranges[seg_idx]

        if n_bins <= 0:
            continue

        if strand == "+":
            offsets = range(n_bins)
        else:
            offsets = range(n_bins - 1, -1, -1)

        for offset in offsets:
            genomic_bp = (start_bin + offset) * res
            row = {"path_bin": path_bin, "seg_idx": seg_idx, "chrom": chrom, "genomic_bp": genomic_bp,
                   "ref_bin_key": (chrom, genomic_bp), "strand": strand, "pos_in_segment": offset,
                   "segment_start_bp": start_bp, "segment_end_bp": end_bp}
            rows.append(row)
            path_bin += 1

    path_df = pd.DataFrame(rows).reset_index(drop=True)
    return path_df, bins

def build_refbin_index(path_df):
    ref_bin_keys = path_df["ref_bin_key"].tolist()
    unique_keys = list(dict.fromkeys(ref_bin_keys))

    key_to_id = {}
    for ref_id, key in enumerate(unique_keys):
        key_to_id[key] = ref_id

    ref_rows = []
    for ref_id, key in enumerate(unique_keys):
        chrom, genomic_bp = key
        ref_rows.append({"ref_id": ref_id, "chrom": chrom, "genomic_bp": genomic_bp})
    ref_df = pd.DataFrame(ref_rows)

    path_df = path_df.copy()
    path_df["ref_id"] = path_df["ref_bin_key"].map(key_to_id)

    members = defaultdict(list)
    for _, row in path_df.iterrows():
        ref_id = int(row["ref_id"])
        path_bin = int(row["path_bin"])
        members[ref_id].append(path_bin)

    multiplicity_map = {}
    for ref_id, path_bins in members.items():
        multiplicity_map[ref_id] = len(path_bins)
    ref_df["multiplicity"] = ref_df["ref_id"].map(multiplicity_map)

    return path_df, ref_df, members

def find_breakpoints(path_df):
    n = path_df.shape[0]
    bp_mask = np.zeros(n, dtype=bool)
    seg_ids = path_df["seg_idx"].values.astype(int)

    for i in range(n):
        l = (i - 1) % n; r = (i + 1) % n
        if seg_ids[l] != seg_ids[i] or seg_ids[r] != seg_ids[i]:
            bp_mask[i] = True

    return bp_mask

def _straw(hic_file, ch1, bp1, ch2, bp2, res):
    import hicstraw  
    result = hicstraw.straw(
        "observed", "NONE", hic_file,
        f"{ch1}:{bp1}:{bp1+res}", f"{ch2}:{bp2}:{bp2+res}", "BP", res)
    c = 0.0
    for rec in result:
        x, y = int(rec.binX), int(rec.binY)
        if ch1 == ch2 and bp1 == bp2:
            if x == bp1 and y == bp1: c += rec.counts
        else:
            if (x == bp1 and y == bp2) or (x == bp2 and y == bp1):
                c += rec.counts
    return c

def fetch_ref_matrix(hic_file, ref_df, res):
    import hicstraw

    hic_chroms = {str(c.name) for c in hicstraw.HiCFile(hic_file).getChromosomes()}

    def match_hic_chrom(ch):
        ch = str(ch).strip()

        candidates = [ch]

        if ch.startswith("chr"):
            candidates.append(ch[3:])
        else:
            candidates.append("chr" + ch)

        if ch in {"chrM", "chrMT", "M", "MT"}:
            candidates.extend(["M", "MT", "chrM", "chrMT"])

        for cand in candidates:
            if cand in hic_chroms:
                return cand

        raise ValueError(
            f"Chromosome {ch} not found in .hic file. "
            f"Tried {candidates}. "
            f"Available examples: {sorted(hic_chroms)[:30]}"
        )

    chrom_map = {
        ch: match_hic_chrom(ch)
        for ch in sorted(ref_df["chrom"].astype(str).unique())
    }

    changed = {k: v for k, v in chrom_map.items() if k != v}
    if changed:
        print("  Chromosome mapping for .hic fetch:")
        for k, v in changed.items():
            print(f"    {k} -> {v}")

    n = ref_df.shape[0]
    Y = np.zeros((n, n), dtype=float)

    for u in range(n):
        ch_u = chrom_map[str(ref_df.loc[u, "chrom"])]
        bp_u = int(ref_df.loc[u, "genomic_bp"])

        for v in range(u, n):
            ch_v = chrom_map[str(ref_df.loc[v, "chrom"])]
            bp_v = int(ref_df.loc[v, "genomic_bp"])

            c = _straw(
                hic_file,
                ch_u, bp_u,
                ch_v, bp_v,
                res
            )
            Y[u, v] = Y[v, u] = c

    return Y

def detect_bad_bins(matrix, mad_thresh=DEFAULT_MAD_THRESH, dominance_thresh=0.50):
    rs   = matrix.sum(axis=1).astype(float)
    nzf  = (matrix > 0).mean(axis=1).astype(float)
    rmax = matrix.max(axis=1).astype(float)
    dom  = np.divide(rmax, np.maximum(rs, 1e-12),
                     out=np.zeros_like(rmax), where=rs > 0)

    zrs = robust_z(rs); znz = robust_z(nzf)
    bad = (np.abs(zrs) > mad_thresh) | (np.abs(znz) > mad_thresh) | (dom > dominance_thresh)

    qc = pd.DataFrame({"bin": np.arange(matrix.shape[0], dtype=int), "row_sum": rs, "row_sum_z": zrs,
                          "nonzero_frac": nzf, "nonzero_frac_z": znz, "dominance": dom,
                          "flagged": bad})
    return qc, qc.loc[qc["flagged"], "bin"].astype(int).tolist()

def _family(ref_u, ref_v, members):
    path_bins_u = members[ref_u]
    path_bins_v = members[ref_v]
    family = []
    seen = set()

    if ref_u == ref_v:
        for a_idx in range(len(path_bins_u)):
            for b_idx in range(a_idx, len(path_bins_u)):
                pair = upper_pair(path_bins_u[a_idx], path_bins_u[b_idx])
                if pair not in seen:
                    family.append(pair)
                    seen.add(pair)
    else:
        for path_bin_u in path_bins_u:
            for path_bin_v in path_bins_v:
                pair = upper_pair(path_bin_u, path_bin_v)
                if pair not in seen:
                    family.append(pair)
                    seen.add(pair)

    return sorted(family)

def build_families(ref_df, members):
    n_ref = ref_df.shape[0]
    families = {}

    for ref_u in range(n_ref):
        for ref_v in range(ref_u, n_ref):
            families[(ref_u, ref_v)] = _family(ref_u, ref_v, members)

    return families

def _pava_decreasing(y, w=None):
    y = np.asarray(y, dtype=float); n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    blocks = []
    for i in range(n):
        blocks.append([i, i, y[i], w[i]])
        while len(blocks) >= 2 and blocks[-2][2] < blocks[-1][2]:
            b2 = blocks.pop(); b1 = blocks.pop()
            nw = b1[3] + b2[3]
            blocks.append([b1[0], b2[1], (b1[2]*b1[3] + b2[2]*b2[3]) / nw, nw])
    yhat = np.zeros(n)
    for s, e, lv, _ in blocks: yhat[s:e+1] = lv
    return yhat

def fit_backbone_decay(dist_bp, sum_x, sum_w, out_prefix=None):
    df = pd.DataFrame({"d": np.asarray(dist_bp, float), "sx": np.asarray(sum_x,   float),
                          "sw": np.asarray(sum_w,   float)})
    df = df[np.isfinite(df["d"]) & np.isfinite(df["sx"]) & np.isfinite(df["sw"])]
    df = df[(df["d"] > 0) & (df["sx"] >= 0) & (df["sw"] > 0)].sort_values("d")
    if df.shape[0] < 2:
        raise ValueError("Not enough distance groups to fit backbone.")

    x      = df["d"].values.astype(float)
    mu_raw = (df["sx"] / df["sw"]).values.astype(float)
    w      = df["sw"].values.astype(float)
    eps    = 1e-8
    y_fit  = np.power(2.0, _pava_decreasing(np.log2(np.maximum(mu_raw, eps)), w=w))

    if out_prefix is not None:
        plot_backbone_fit(x, np.maximum(mu_raw, eps), np.maximum(y_fit, eps),
                          f"{out_prefix}_backbone_fit.png",
                          "Initial monotone backbone fit")

    return {"dist": x, "raw_mean": mu_raw, "fitted_mean": y_fit}

def _interp_decay(d_bp, decay, zero_proxy=1.0):
    if d_bp <= 0: return float(zero_proxy)
    x, y = decay["dist"], decay["fitted_mean"]
    if d_bp <= x[0]:  return float(y[0])
    if d_bp >= x[-1]: return float(y[-1])
    idx  = np.searchsorted(x, d_bp)
    x0, x1, y0, y1 = x[idx-1], x[idx], y[idx-1], y[idx]
    lx0, lx1 = math.log2(x0), math.log2(x1)
    if lx1 == lx0: return float(y0)
    t  = (math.log2(d_bp) - lx0) / (lx1 - lx0)
    ly = math.log2(max(y0, 1e-12)) + t * (math.log2(max(y1, 1e-12)) - math.log2(max(y0, 1e-12)))
    return float(2 ** ly)

def build_backbone_matrix(n, res, decay, zero_proxy=1.0):
    """Upper-triangle f(d_ij) over path bins, using circular distance.

    Each path bin is one copy.
    """
    bb = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            bb[i, j] = _interp_decay(circ_dist(i, j, n) * res, decay, zero_proxy)
    return bb

def init_alpha(n):
    a = np.ones((n, n), dtype=float)
    for i in range(n): a[i, :i] = 0.0
    return a

def build_lambda(backbone, alpha):
    lam = np.array(backbone, dtype=float, copy=True)
    nz  = backbone > 0
    lam[nz] = backbone[nz] * alpha[nz]
    return lam

def build_effective_lambda(backbone, alpha, pi=None):
    lam = build_lambda(backbone, alpha)
    if pi is None:
        return lam
    pi = np.asarray(pi, dtype=float)
    lam = lam * (pi[:, None] * pi[None, :])
    lam[build_lambda(backbone, alpha) == 0] = 0.0
    return lam

def initial_fit_pairs(path_df, ref_df, Y, res, start_diag=2, trim_upper_q=0.95, trim_lower_q=0.01, bad_ref_set=None, exclude_breakpoints=True):
    """(distance_bp, count) pairs used to fit f(d) directly.
    did not take bins with copy number > 1, bins next to a junction.
    """
    mult = ref_df.set_index("ref_id")["multiplicity"].to_dict()
    bad_set  = set(bad_ref_set) if bad_ref_set is not None else set()
    bp_mask  = find_breakpoints(path_df) if exclude_breakpoints else np.zeros(path_df.shape[0], dtype=bool)

    rows = []; n = path_df.shape[0]
    for i in range(n):
        if bp_mask[i]: continue
        ri = int(path_df.loc[i, "ref_id"])
        if mult[ri] != 1 or ri in bad_set: continue

        for j in range(i + 1, n):
            if bp_mask[j]: continue
            rj = int(path_df.loc[j, "ref_id"])
            if mult[rj] != 1 or rj in bad_set: continue

            db = circ_dist(i, j, n)
            if db < start_diag: continue

            y = float(Y[ri, rj])
            if not np.isfinite(y) or y <= 0: continue

            rows.append({"dist": db * res, "count": y})

    df = pd.DataFrame(rows)
    if df.empty:
        return np.array([]), np.array([])

    med = (df.groupby("dist")["count"].median()
             .rename("median_count").reset_index())
    df  = df.merge(med, on="dist", how="left")
    df["rough_oe"] = df["count"] / np.maximum(df["median_count"], 1e-8)

    keep = []
    for _, grp in df.groupby("dist", sort=True):
        if grp.shape[0] < 10:
            keep.extend(grp.index.tolist()); continue
        lo = grp["rough_oe"].quantile(trim_lower_q)
        hi = grp["rough_oe"].quantile(trim_upper_q)
        g  = grp[(grp["rough_oe"] >= lo) & (grp["rough_oe"] <= hi)]
        keep.extend((g if g.shape[0] >= 5 else grp).index.tolist())

    df_keep = df.loc[sorted(set(keep))]
    return df_keep["dist"].values.astype(float), df_keep["count"].values.astype(float)

def fit_initial_decay(dist_bp, counts, out_prefix=None):
    if len(dist_bp) < 10:
        raise ValueError("Too few clean pairs for initial backbone fit. "
            "Check mad_thresh, trim parameters, or start_diag.")

    fit_data = pd.DataFrame({"d": dist_bp, "c": counts})
    grouped = fit_data.groupby("d", as_index=False).agg(
        sx=("c", "sum"),
        sw=("c", "size"),
    )
    grouped = grouped.sort_values("d")

    distances = grouped["d"].values
    count_sums = grouped["sx"].values
    n_pairs = grouped["sw"].values

    return fit_backbone_decay(distances,count_sums,n_pairs,out_prefix=out_prefix)

def calibrate_zero_proxy(Y, ref_df, trim_low=0.1, trim_high=0.9, fallback=1.0):
    mult = ref_df.set_index("ref_id")["multiplicity"].to_dict()
    diag_vals = []
    for u in range(Y.shape[0]):
        if mult.get(u, 1) != 1:
            continue
        v = float(Y[u, u])
        if np.isfinite(v) and v > 0:
            diag_vals.append(v)
    if len(diag_vals) < 5:
        return float(fallback), {"source": "fallback", "n_nondup_diag": len(diag_vals),
                                 "value": float(fallback)}
    arr = np.asarray(diag_vals, dtype=float)
    lo  = np.quantile(arr, trim_low)
    hi  = np.quantile(arr, trim_high)
    kept = arr[(arr >= lo) & (arr <= hi)]
    zp = float(np.median(kept)) if kept.size >= 3 else float(np.median(arr))
    return zp, {"source": "empirical_nondup_diagonal", "n_nondup_diag": len(diag_vals),"trim_window": [float(lo), float(hi)], "n_after_trim": int(kept.size), "value": zp,
                "raw_median": float(np.median(arr)), "raw_mean": float(np.mean(arr))}

def build_same_ref_mask(path_df):
    n = path_df.shape[0]
    rids = path_df["ref_id"].values.astype(int)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if rids[i] == rids[j]:
                mask[i, j] = True
    return mask

def compute_copy_anchors(Y, path_df, backbone, window_bins=50, eps=1e-3):
    """Per copy weight pi for path bins of duplicated reference bins.
    For each copy, sum Y against single copy neighbors and divide by the summed backbone expectation.
    """
    n_path = len(path_df)
    ref_of = path_df["ref_id"].values.astype(int)

    multiplicity = defaultdict(int)
    members = defaultdict(list)
    for path_bin, ref_id in enumerate(ref_of):
        ref_id = int(ref_id)
        multiplicity[ref_id] += 1
        members[ref_id].append(path_bin)

    pi = np.ones(n_path, dtype=float)
    window = int(window_bins)

    for ref_id, copy_path_bins in members.items():
        if multiplicity[ref_id] <= 1:
            continue

        scores = np.zeros(len(copy_path_bins))
        informative = np.zeros(len(copy_path_bins), dtype=bool)

        for copy_index, path_bin in enumerate(copy_path_bins):
            observed_sum = 0.0
            expected_sum = 0.0

            for offset in range(-window, window + 1):
                if offset == 0:
                    continue

                partner_path_bin = (path_bin + offset) % n_path
                partner_ref_id = int(ref_of[partner_path_bin])

                # Anchor only against uniquely represented reference bins.
                if multiplicity[partner_ref_id] != 1:
                    continue

                if path_bin <= partner_path_bin:
                    expected = float(backbone[path_bin, partner_path_bin])
                else:
                    expected = float(backbone[partner_path_bin, path_bin])

                if expected <= 0 or not np.isfinite(expected):
                    continue

                observed_sum += float(Y[ref_id, partner_ref_id])
                expected_sum += expected

            if expected_sum > 0:
                scores[copy_index] = observed_sum / expected_sum
                informative[copy_index] = True

        n_informative = int(informative.sum())
        if n_informative == 0:
            continue
        mean_info_score = scores[informative].mean()

        if mean_info_score > 0:
            for copy_index, path_bin in enumerate(copy_path_bins):
                if informative[copy_index]:
                    normalized_score = scores[copy_index] / mean_info_score
                    pi[path_bin] = max(normalized_score, eps)
        else:
            for copy_index, path_bin in enumerate(copy_path_bins):
                if informative[copy_index]:
                    pi[path_bin] = eps

    return pi

def e_step(Y, families, lam, min_lam=1e-12, pi=None):
    n_path = lam.shape[0]
    X_exp = np.zeros((n_path, n_path), dtype=float)
    touched = np.zeros((n_path, n_path), dtype=bool)
    n_ref = Y.shape[0]
    use_pi = pi is not None

    for u in range(n_ref):
        for v in range(u, n_ref):
            y = float(Y[u, v])
            fam = families[(u, v)]
            if not fam:
                continue

            for i, j in fam:
                touched[i, j] = True

            if y <= 0:
                continue

            if use_pi:
                weights = np.array([
                    max(lam[i, j] * pi[i] * pi[j], min_lam)
                    for i, j in fam
                ])
            else:
                weights = np.array([
                    max(lam[i, j], min_lam)
                    for i, j in fam
                ])

            total_weight = weights.sum()
            if total_weight > 0 and np.isfinite(total_weight):
                weights = weights / total_weight
            else:
                weights = np.ones(len(fam)) / len(fam)

            for (i, j), value in zip(fam, y * weights):
                X_exp[i, j] += value

    return X_exp, touched

def m_step_fixed_backbone(X_exp_upper, backbone_upper, n_path, tau=1.0, alpha_min=0.1, alpha_max=10.0, same_ref_mask=None, tau_between_copy=None, pi=None):
    alpha_new = np.zeros_like(X_exp_upper, dtype=float)
    use_split = (same_ref_mask is not None) and (tau_between_copy is not None)
    use_pi = pi is not None

    for i in range(n_path):
        for j in range(i, n_path):
            bb = float(backbone_upper[i, j])

            if use_pi:
                mu = bb * float(pi[i]) * float(pi[j])
            else:
                mu = bb

            xij = float(X_exp_upper[i, j])
            t = tau_between_copy if (use_split and same_ref_mask[i, j]) else tau

            if mu > 0 and np.isfinite(mu):
                alpha_hat = (xij + t * mu) / ((t + 1.0) * mu)
            else:
                alpha_hat = 1.0

            alpha_new[i, j] = max(alpha_min, min(alpha_max, alpha_hat))

    alpha_new = _norm_alpha_masked(
        alpha_new,
        exclude_mask=same_ref_mask
    )
    return build_lambda(backbone_upper, alpha_new), alpha_new

def _norm_alpha_masked(alpha, exclude_mask=None):
    n = alpha.shape[0]
    idx  = np.triu_indices(n)
    vals = alpha[idx]
    if exclude_mask is not None:
        ex = exclude_mask[idx]
        vals = vals[~ex]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0: return alpha
    gm = np.exp(np.mean(np.log(vals)))
    if gm > 0 and np.isfinite(gm): alpha = alpha / gm
    return alpha

def _loglik(Y, families, lam, min_mu=1e-12, pi=None):
    n_ref = Y.shape[0]
    ll = 0.0
    use_pi = pi is not None

    for u in range(n_ref):
        for v in range(u, n_ref):
            fam = families[(u, v)]
            if not fam:
                continue
            if use_pi:
                mu = max(
                    sum(max(pi[i] * pi[j] * lam[i, j], min_mu) for i, j in fam),
                    min_mu,
                )
            else:
                mu = max(
                    sum(max(lam[i, j], min_mu) for i, j in fam),
                    min_mu,
                )

            y = float(Y[u, v])
            ll += y * math.log(mu) - mu

    return ll

def run_em(Y, families, n_path, backbone_upper, lam_init, alpha_init, outdir, max_iter=100, tol=1e-6, tau=1.0, alpha_min=0.1, alpha_max=10.0,
    same_ref_mask=None, tau_between_copy=None, pi=None, verbose=True):
    lam = lam_init.copy()
    alpha = alpha_init.copy()
    hist = []
    prev_ll = None
    prev_X = None
    prev_alpha = None

    for it in range(1, max_iter + 1):
        X_exp, _ = e_step(Y, families, lam, pi=pi)

        lam_new, alpha_new = m_step_fixed_backbone(X_exp, backbone_upper, n_path, tau, alpha_min, alpha_max, same_ref_mask=same_ref_mask,
        tau_between_copy=tau_between_copy, pi=pi)

        ll = _loglik(Y, families, lam_new, pi=pi)
        rel_ll = abs(ll - prev_ll) / max(abs(prev_ll), 1.0) if prev_ll is not None else np.nan
        x_corr = safe_corr(X_exp, prev_X)
        al_corr = safe_corr(alpha_new, prev_alpha)

        av = alpha_new[np.triu_indices(n_path)]
        av = av[np.isfinite(av) & (av > 0)]
        hist.append({"iter": it, "loglik": ll, "rel_ll_change": rel_ll, "x_corr_prev": x_corr, "alpha_corr_prev": al_corr, "alpha_median": float(np.median(av)) if av.size > 0 else np.nan,
                    "alpha_mean": float(np.mean(av)) if av.size > 0 else np.nan,
                    "alpha_p95": float(np.percentile(av, 95)) if av.size > 0 else np.nan})

        if verbose:
            print(f"  [EM] {it:03d}  ll={ll:.2f}  rel_ll={rel_ll:.2e}  " f"x_corr={x_corr:.4f}  alpha_corr={al_corr:.4f}  "
                  f"alpha_med={hist[-1]['alpha_median']:.4f}")

        plot_alpha_dist(av, os.path.join(outdir, f"iter_{it:03d}_alpha.png"), title=rf"$\alpha_{{ij}}$ (iteration {it})")

        lam = lam_new
        alpha = alpha_new
        prev_ll = ll
        prev_X = X_exp.copy()
        prev_alpha = alpha_new.copy()

        stop_ll = np.isfinite(rel_ll) and rel_ll < tol
        stop_x = np.isfinite(x_corr) and x_corr > 0.999
        stop_alpha = np.isfinite(al_corr) and al_corr > 0.999

        if stop_ll and stop_x and stop_alpha:
            if verbose:
                print(f"  [EM] converged at iter {it}")
            break

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(outdir, "em_history.tsv"), sep="\t", index=False)

    X_final, touched_final = e_step(Y, families, lam, pi=pi)
    lam_eff = build_effective_lambda(backbone_upper, alpha, pi)
    return lam_eff, alpha, X_final, touched_final, hist_df

def collapse_to_ref(X, path_df, ref_df):
    n_ref = ref_df.shape[0]
    collapsed = np.zeros((n_ref, n_ref), dtype=float)
    ref_ids = path_df["ref_id"].values.astype(int)

    for path_i in range(len(ref_ids)):
        ref_i = ref_ids[path_i]
        for path_j in range(path_i, len(ref_ids)):
            ref_j = ref_ids[path_j]
            ref_u, ref_v = upper_pair(ref_i, ref_j)
            collapsed[ref_u, ref_v] += X[path_i, path_j]

    return sym(collapsed)

def save_tsv(df, path): 
    df.to_csv(path, sep="\t", index=False)

def save_path_bins(path_df, path, res):
    df = path_df.copy(); df["genomic_end_bp"] = df["genomic_bp"] + res; save_tsv(df, path)

def save_ref_bins(ref_df, path, res):
    df = ref_df.copy(); df["genomic_end_bp"] = df["genomic_bp"] + res; save_tsv(df, path)

def save_families(family_df, path): save_tsv(family_df, path)

def save_latent_pairs(X_upper, bb_upper, alpha_upper, lam_upper, path_df, ref_df, family_df, path, res):
    n = X_upper.shape[0]
    mult = ref_df.set_index("ref_id")["multiplicity"].to_dict()
    eps  = 1e-8

    fam_lu = {}
    for _, row in family_df.iterrows():
        fam_lu[(int(row["ref_u"]), int(row["ref_v"]))] = row

    rows = []
    for i in range(n):
        ri = int(path_df.loc[i, "ref_id"])
        for j in range(i, n):
            rj  = int(path_df.loc[j, "ref_id"])
            db  = circ_dist(i, j, n)
            uu, vv = upper_pair(ri, rj)
            fm  = fam_lu.get((uu, vv), {})
            rows.append({"path_bin_i": i, "path_bin_j": j,
            "chrom_i": path_df.loc[i, "chrom"],
            "start_i": int(path_df.loc[i, "genomic_bp"]),
            "end_i": int(path_df.loc[i, "genomic_bp"]) + res,
            "chrom_j": path_df.loc[j, "chrom"],
            "start_j":  int(path_df.loc[j, "genomic_bp"]),
            "end_j":  int(path_df.loc[j, "genomic_bp"]) + res,
            "seg_idx_i":  int(path_df.loc[i, "seg_idx"]), "seg_idx_j": int(path_df.loc[j, "seg_idx"]), "ref_id_i": ri, "ref_id_j": rj,
            "multiplicity_i":  int(mult.get(ri, 1)), "multiplicity_j": int(mult.get(rj, 1)),
            "max_multiplicity": int(max(mult.get(ri, 1), mult.get(rj, 1))),
            "latent_X": float(X_upper[i, j]), "backbone_mu": float(bb_upper[i, j]),
            "alpha_ij": float(alpha_upper[i, j]), "lambda_ij": float(lam_upper[i, j]),
            "log2_OE": math.log2((X_upper[i,j]+eps)/(lam_upper[i,j]+eps)),
            "dist_bins": int(db), "dist_bp": int(db * res),
            "is_diagonal": bool(db == 0),
            "family_size": int(fm.get("family_size", 0)),
            "family_entropy_norm":float(fm.get("family_entropy_norm", np.nan)),
            "dominant_pair_frac": float(fm.get("dominant_pair_fraction", np.nan))})
    save_tsv(pd.DataFrame(rows), path)

def norm_chr(chrom):
    chrom = str(chrom)
    if chrom.startswith("chr"):
        return chrom
    return "chr" + chrom

def load_bed(bed_file):
    df = pd.read_csv(bed_file, sep="\t", header=0)

    cleaned_columns = []
    for column in df.columns:
        cleaned = str(column).strip().lower().lstrip("#")
        cleaned_columns.append(cleaned)
    df.columns = cleaned_columns

    column_map = {}
    for column in df.columns:
        if column in ["chr", "chrom", "chromosome"]:
            column_map[column] = "chr"
        elif column == "start":
            column_map[column] = "start"
        elif column == "end":
            column_map[column] = "end"
        elif column in ["orientation", "strand", "sign"]:
            column_map[column] = "orientation"

    df = df.rename(columns=column_map)

    required_columns = ["chr", "start", "end", "orientation"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}'. Found: {list(df.columns)}")

    df = df[required_columns].copy()
    df["chr"] = df["chr"].astype(str).str.strip().apply(norm_chr)
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(int)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(int)
    df["orientation"] = df["orientation"].astype(str).str.strip()

    valid_orientation = df["orientation"].isin(["+", "-"])
    df = df[valid_orientation].copy()
    if df.empty:
        raise ValueError("No valid rows after filtering orientation.")

    intervals = df[["chr", "start", "end"]].values.tolist()
    orientations = df["orientation"].tolist()
    return intervals, orientations

def parse_args():
    p = argparse.ArgumentParser(description="CADET Layer 1: copy-aware deconvolution")
    p.add_argument("--bed", required=True)
    p.add_argument("--hic", required=False, default=None,
                   help="Input .hic file")
    p.add_argument("--y_cached", required=False, default=None,
                   help="Saved observed_Y.txt instead of --hic")
    p.add_argument("--res", type=int, required=True)
    p.add_argument("--out", default="cadet_part1_output")
    p.add_argument("--max_iter", type=int,   default=100)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--tau", type=float, default=1.0,
                   help="Shrinkage of alpha toward 1 (default 1).")
    p.add_argument("--alpha_min", type=float, default=0.1,
                   help="Lower bound for alpha (default 0.1).")
    p.add_argument("--alpha_max", type=float, default=10.0,
                   help="Upper bound for alpha (default 10).")
    p.add_argument("--mad_thresh", type=float, default=DEFAULT_MAD_THRESH,
                   help="MAD cutoff for reference-bin QC (default 3).")
    p.add_argument("--trim_upper", type=float, default=0.95,
                   help="Upper O/E quantile for the backbone fit")
    p.add_argument("--trim_lower", type=float, default=0.01,
                   help="Lower O/E quantile for the backbone fit")
    p.add_argument("--start_diag", type=int,   default=2,
                   help="First diagonal used for the backbone fit")
    p.add_argument("--tau_between_copy", type=float, default=10.0,
                   help="Alpha shrinkage between copies")
    p.add_argument("--anchor_window_bins", type=int, default=50,
                   help="Half-window (bins) for copy anchors; 0 = off")
    p.add_argument("--anchor_eps", type=float, default=1e-3,
                   help="Minimum copy-anchor weight (default 1e-3).")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "em_iters"), exist_ok=True)
    em_dir = os.path.join(outdir, "em_iters")

    intrvl, sign = load_bed(args.bed)
    hic = args.hic; res = args.res

    print("=" * 60)
    print("CADET Part 1 — ecDNA Hi-C deconvolution")
    print("=" * 60)
    print(f"  Resolution : {res} bp")
    print("  Backbone   : fixed (fitted once from clean pairs; "
          "not updated during EM)")
    print("  Alpha norm : geometric mean = 1 (identifiability)")

    path_df, bins = build_path_bin_table(intrvl, sign, res)
    if path_df.shape[0] == 0:
        raise ValueError("No path bins at this resolution.")

    path_df, ref_df, members = build_refbin_index(path_df)

    has_dup = bool((ref_df["multiplicity"] > 1).any())
    branch = "duplicated" if has_dup else "nonduplicated"
    n_path = path_df.shape[0]

    print(f"\n  Path bins : {n_path}")
    print(f" Reference bins : {ref_df.shape[0]}")
    print(f" Branch : {branch}")

    save_path_bins(path_df, os.path.join(outdir, "path_bins.tsv"), res)
    save_ref_bins(ref_df, os.path.join(outdir, "reference_bins.tsv"), res)

    if getattr(args, "y_cached", None):
        print(f"\n  Reusing cached Y from {args.y_cached}")
        Y = np.loadtxt(args.y_cached, delimiter="\t")
        expected_shape = (ref_df.shape[0], ref_df.shape[0])
        if Y.shape != expected_shape:
            raise ValueError(f"Cached Y shape {Y.shape} != expected {expected_shape} "f"for the given --bed and --res.")
    else:
        if not hic:
            raise ValueError("Either --hic or --y_cached must be provided.")
        if has_dup:
            print("\n  Fetching collapsed reference matrix Y ...")
        else:
            print("\n  Fetching nonduplicated direct reference matrix Y ...")
        Y = fetch_ref_matrix(hic, ref_df, res)

    np.savetxt(os.path.join(outdir, "observed_Y.txt"), Y, delimiter="\t")
    obs_matrix = sym(Y)

    qc, flagged_ref = detect_bad_bins(obs_matrix, args.mad_thresh)
    qc.to_csv(os.path.join(outdir, "bad_bin_qc.csv"), index=False)
    flagged_ref_set  = set(flagged_ref)
    flagged_path_set = {int(i) for i in range(n_path) if int(path_df.loc[i, "ref_id"]) in flagged_ref_set}
    print(f" Flagged {len(flagged_ref)} bad ref-bins " f"({len(flagged_path_set)} path bins) — excluded from backbone fit")

    families = build_families(ref_df, members)

    print("\n  Fitting initial backbone from clean unique pairs ...")
    d0, c0 = initial_fit_pairs( path_df, ref_df, Y, res, start_diag=args.start_diag,
        trim_upper_q=args.trim_upper, trim_lower_q=args.trim_lower,bad_ref_set=flagged_ref_set,
        exclude_breakpoints=True)

    if len(d0) < 10:
        raise ValueError(f"Only {len(d0)} clean pairs found for initial fit. " "Try relaxing --mad_thresh, --trim_upper/lower, or --start_diag.")

    print(f" Using {len(d0)} clean pairs for initial backbone fit")
    init_decay = fit_initial_decay(d0, c0, out_prefix=os.path.join(outdir, "init"))

    # f(0) is estimated empirically from non-duplicated reference diagonals.
    zero_proxy, zp_info = calibrate_zero_proxy(Y, ref_df)
    print(f" zero_proxy (empirical, non-dup diagonal): {zero_proxy:.4g}  "
          f"[n={zp_info.get('n_nondup_diag', 0)}, source={zp_info['source']}]")

    backbone = build_backbone_matrix(n_path, res, init_decay, zero_proxy)

    # same-ref, different-copy cells are weakly identified from Y[u,u].
    # keep stronger shrinkage here.
    same_ref_mask = build_same_ref_mask(path_df)
    n_same_ref = int(same_ref_mask.sum())
    print(f"Between-copy pairs at same ref: {n_same_ref}  "
          f"(stronger shrinkage: tau_between_copy={args.tau_between_copy})")

    # Default model: same-ref duplicated pairs keep the same circular-distance
    # backbone f(d_circ). Their weak identifiability is handled by stronger
    # alpha shrinkage (tau_between_copy), not by a second backbone branch.

    alpha0 = init_alpha(n_path)
    lam0 = build_lambda(backbone, alpha0)

    if args.anchor_window_bins > 0 and has_dup:
        pi = compute_copy_anchors(Y, path_df, backbone,window_bins=args.anchor_window_bins,eps=args.anchor_eps)
        multiplicity_map = ref_df.set_index("ref_id")["multiplicity"].to_dict()
        duplicated_path_flags = []
        for path_bin in range(len(path_df)):
            ref_id = int(path_df.loc[path_bin, "ref_id"])
            is_duplicated = multiplicity_map[ref_id] > 1
            duplicated_path_flags.append(is_duplicated)
        dup_path_mask = np.array(duplicated_path_flags, dtype=bool)
        pi_dup = pi[dup_path_mask]
        print(f"  copy-anchor pi:  n_dup_path={int(dup_path_mask.sum())}  "
              f"window=+/-{args.anchor_window_bins} bins  "
              f"pi[dup] p5/median/p95 = "
              f"{np.quantile(pi_dup, 0.05):.3g} / "
              f"{np.median(pi_dup):.3g} / "
              f"{np.quantile(pi_dup, 0.95):.3g}")
    else:
        pi = None
        print(f"  copy-anchor pi:  DISABLED " f"(anchor_window_bins={args.anchor_window_bins}, has_dup={has_dup})")

    print("\n  Running EM deconvolution on the full observed matrix Y")
    lam_f, alpha_f, X_f, touched, hist_df = run_em(Y, families, n_path, backbone, lam0, alpha0, em_dir,max_iter=args.max_iter, tol=args.tol,
        tau=args.tau, alpha_min=args.alpha_min, alpha_max=args.alpha_max,same_ref_mask=same_ref_mask,tau_between_copy=args.tau_between_copy,
        pi=pi,verbose=True)

    upper_triangle_mask = np.triu(np.ones_like(X_f, dtype=bool), k=0)
    untouched_upper = (~touched) & upper_triangle_mask
    n_untouched = int(untouched_upper.sum())

    # outputs used by Part 2
    X_sym = sym(X_f)
    lam_sym = sym(lam_f)
    bb_sym = sym(backbone)
    alpha_sym = sym(alpha_f)

    if pi is not None:
        np.savetxt(os.path.join(outdir, "copy_anchor_pi.txt"), pi, delimiter="\t")
    np.savetxt(os.path.join(outdir, "latent_X_upper.txt"),     X_f,     delimiter="\t")
    np.savetxt(os.path.join(outdir, "latent_X_symmetric.txt"), X_sym,          delimiter="\t")
    # effective lambda; Part 2 uses this for O/E and enrichment testing.
    np.savetxt(os.path.join(outdir, "lambda_upper.txt"),       lam_f,   delimiter="\t")
    np.savetxt(os.path.join(outdir, "lambda_symmetric.txt"),   lam_sym,        delimiter="\t")
    np.savetxt(os.path.join(outdir, "backbone_upper.txt"),     backbone,       delimiter="\t")
    np.savetxt(os.path.join(outdir, "alpha_upper.txt"),        alpha_f, delimiter="\t")

    ab_upper = build_lambda(backbone, alpha_f)
    np.savetxt(os.path.join(outdir, "alpha_backbone_upper.txt"),ab_upper, delimiter="\t")
    np.savetxt(os.path.join(outdir, "alpha_backbone_symmetric.txt"),sym(ab_upper), delimiter="\t")

    # *_raw files are now identical to the ones above; kept for older Part 2 scripts
    np.savetxt(os.path.join(outdir, "latent_X_upper_raw.txt"),   X_f,     delimiter="\t")
    np.savetxt(os.path.join(outdir, "lambda_upper_raw.txt"),     lam_f,   delimiter="\t")
    np.savetxt(os.path.join(outdir, "alpha_upper_raw.txt"),      alpha_f, delimiter="\t")
    np.savetxt(os.path.join(outdir, "touched_mask_upper.txt"),(touched & upper_triangle_mask).astype(int), delimiter="\t", fmt="%d")

    eps = 1e-8
    OE  = np.full_like(X_f, np.nan, dtype=float)
    for i in range(X_f.shape[0]):
        for j in range(i, X_f.shape[1]):
            xij  = X_f[i, j]
            lmij = lam_f[i, j]
            if np.isfinite(xij) and np.isfinite(lmij):
                OE[i, j] = math.log2((xij + eps) / (lmij + eps))
    OE_sym = sym(OE)
    np.savetxt(os.path.join(outdir, "log2_oe_upper.txt"),     OE,     delimiter="\t")
    np.savetxt(os.path.join(outdir, "log2_oe_symmetric.txt"), OE_sym, delimiter="\t")

    Y_hat_full = collapse_to_ref(X_f, path_df, ref_df)
    rc = recollapse_metrics(Y, Y_hat_full)
    rc.to_csv(os.path.join(outdir, "recollapse_metrics.csv"), index=False)
    np.savetxt(os.path.join(outdir, "recollapsed_Y.txt"), Y_hat_full, delimiter="\t")

    # family ambiguity. lam_f already has pi folded in, so don't pass pi again.
    family_df = family_ambiguity(families, Y, lam_f)
    save_families(family_df, os.path.join(outdir, "families.tsv"))

    save_latent_pairs(X_f, backbone, alpha_f, lam_f, path_df, ref_df, family_df, os.path.join(outdir, "latent_pairs.tsv"), res)

    ab_sum = alpha_bound_summary(alpha_f, args.alpha_min, args.alpha_max)
    ab_sum.to_csv(os.path.join(outdir, "diagnostics_alpha.tsv"), sep="\t", index=False)

    dw = oe_spread(X_f, lam_f, backbone, res)
    dw.to_csv(os.path.join(outdir, "diagnostics_distancewise_spread.tsv"), sep="\t", index=False)

    ov = top1pct_overlap(X_f, lam_f, backbone)
    ov.to_csv(os.path.join(outdir, "diagnostics_top1pct_overlap.tsv"), sep="\t", index=False)

    # ICE for display only (not used for statistics)
    try:
        from iced import normalization  # lazy import
        X_for_ice = np.where(np.isfinite(X_sym), X_sym, 0.0)
        X_ice = normalization.ICE_normalization(X_for_ice.copy())
        np.savetxt(os.path.join(outdir, "latent_X_ICE_display.txt"), X_ice, delimiter="\t")
    except Exception as e:
        print(f"  ICE normalization skipped: {e}"); X_ice = None

    # Meta for Part 2
    meta = { "branch":  branch,
        "resolution": res,
        "n_path_bins": int(n_path),
        "n_ref_bins":  int(ref_df.shape[0]),
        "flagged_ref_bins": sorted(flagged_ref),
        "flagged_path_bins": sorted(flagged_path_set),
        "outdir": outdir,
        "backbone_mode": "fixed",
        "alpha_norm": "geometric_mean_eq_1 (excluding between-copy-at-same-ref)",
        "lambda_model": ("per_copy, anchored: " "λ_ij = π_i · π_j · α_ij · f(d_ij). " "When pi is disabled, reduces to α_ij · f(d_ij)."),
        "zero_proxy": zp_info,
        "betweencopy_mode": "circular_distance_backbone_with_stronger_alpha_shrinkage",
        "tau_between_copy": float(args.tau_between_copy),
        "anchor_window_bins": int(args.anchor_window_bins),
        "anchor_eps": float(args.anchor_eps),
        "anchor_enabled": bool(pi is not None),
        "n_same_ref_pairs": int(n_same_ref),
        "n_untouched_latents": int(n_untouched),
        "primary_outputs": {"latent_X_symmetric": "Full matrix — all cells populated",
            "lambda_symmetric":   "Full matrix — all cells populated"}}
    with open(os.path.join(outdir, "part1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    params = {"hic": hic, "resolution": res, "branch": branch,
        "max_iter": args.max_iter, "tol": args.tol, "tau": args.tau,
        "alpha_min": args.alpha_min, "alpha_max": args.alpha_max,
        "mad_thresh": args.mad_thresh, "start_diag": args.start_diag,
        "trim_upper": args.trim_upper, "trim_lower": args.trim_lower,
        "backbone_mode": "fixed — backbone never updated during EM",
        "lambda_model":  ("per-copy, anchored:  lambda_eff_ij = pi_i * pi_j * alpha_ij * f(d_ij) "
            "when anchors are enabled (--anchor_window_bins > 0). When anchors are "
            "disabled (pi = 1 everywhere), reduces to lambda_ij = alpha_ij * f(d_ij). "
            "Multiplicity enters only via |family(u,v)| = m_u * m_v."),
        "anchor_enabled": bool(pi is not None),
        "alpha_norm":    "geometric mean = 1 (not arithmetic mean)"}
    with open(os.path.join(outdir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    plot_ec3d_style(obs_matrix, os.path.join(outdir, "observed_Y.png"),res, title="Observed reference contact matrix", axis_label="Reference bins",
        cbar_label=r"$\log_{10}(Y_{uv})$")
    plot_ec3d_style(X_sym, os.path.join(outdir, "latent_X.png"), res, Mod_bins=bins, intrvl=intrvl, title="Latent deconvolved contact matrix", axis_label="Path bins",
        cbar_label=r"$\log_{10}(X_{ij})$")
    lam_title = "Expected contact matrix"
    bb_title = "Circular-polymer backbone"
    plot_ec3d_style(lam_sym, os.path.join(outdir, "lambda.png"),res, Mod_bins=bins, intrvl=intrvl, title=lam_title, axis_label="Path bins",
        cbar_label=r"$\log_{10}(\lambda_{ij})$")
    plot_ec3d_style(bb_sym, os.path.join(outdir, "backbone.png"), res, Mod_bins=bins, intrvl=intrvl, title=bb_title, axis_label="Path bins",
        cbar_label=r"$\log_{10}(f(d_{ij}))$")
    plot_centered(np.log2(np.maximum(alpha_sym, 1e-8)), os.path.join(outdir, "log2_alpha.png"), res, Mod_bins=bins, intrvl=intrvl,title=r"$\log_2(\alpha_{ij})$",
        cbar_label=r"$\log_2(\alpha_{ij})$")
    plot_centered(OE_sym,os.path.join(outdir, "log2_oe.png"),res, Mod_bins=bins, intrvl=intrvl, title="Latent observed/expected enrichment",
        cbar_label=r"$\log_2(O/E)$")
    plot_diff(Y_hat_full, Y,os.path.join(outdir, "recollapsed_minus_observed.png"), res, title="Re-collapsed minus observed contacts",
        cbar_label=r"$\hat{Y}_{uv}-Y_{uv}$")
    plot_x_vs_lambda(X_f, lam_f, os.path.join(outdir, "diagnostics_X_vs_lambda.png"))
    plot_convergence(hist_df, os.path.join(outdir, "em_convergence.png"))
    plot_oe_spread(dw,os.path.join(outdir, "diagnostics_distancewise_spread.png"))
    plot_family_metrics(family_df, os.path.join(outdir, "family_entropy.png"),os.path.join(outdir, "family_size.png"))
    plot_hist(np.log2((X_f[np.triu_indices_from(X_f, k=1)] + eps) / (lam_f[np.triu_indices_from(lam_f, k=1)] + eps)),
        os.path.join(outdir, "diagnostics_log2_oe_hist.png"),"Latent observed/expected distribution",r"$\log_2(X_{ij}/\lambda_{ij})$")

    if X_ice is not None:
        plot_ec3d_style(X_ice, os.path.join(outdir, "latent_X_ICE_display.png"), res, Mod_bins=bins, intrvl=intrvl, title="ICE-normalized latent contact matrix",
        axis_label="Path bins", cbar_label=r"$\log_{10}(X^{ICE}_{ij})$")

    print("\n Recollapse metrics (should be close to 0):")
    print(rc.to_string(index=False))
    print("\n Alpha diagnostics:")
    print(ab_sum.to_string(index=False))
    print("\n Top-1% backbone vs lambda overlap:")
    print(ov.to_string(index=False))
    print("\n Key outputs for Part 2:")
    print(f" {outdir}/latent_X_symmetric.txt")
    print(f" {outdir}/lambda_symmetric.txt")
    print(f" {outdir}/part1_meta.json")
    print("\n Done.")

if __name__ == "__main__":
    main()
