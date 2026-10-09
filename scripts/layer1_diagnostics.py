#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB_SQ = (10, 10)
PUB_RECT = (10, 6)
PUB_FS = 22
PUB_LEGEND_FS = 18
PUB_DPI = 300
PUB_LW = 0.4

def circ_dist(i, j, n):
    d = abs(i - j)
    return min(d, n - d)

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

def _seg_labels(intrvl, Mod_bins, res):
    min_seg = 8 * res
    xlbl, ylbl = [], [''] * len(Mod_bins)
    for idx, (ch, s, e) in enumerate(intrvl):
        if e - s >= min_seg:
            sl = f"{s/1e6:.1f}".rstrip('0').rstrip('.') if s % 1e6 else f"{int(s/1e6)}"
            el = f"{e/1e6:.1f}".rstrip('0').rstrip('.') if e % 1e6 else f"{int(e/1e6)}"
            xlbl.append(f"{ch}:{sl}-{el}Mb")
            ylbl[idx] = str(Mod_bins[idx])
        else:
            xlbl.append('')
    if intrvl[-1][2] - intrvl[-1][1] >= min_seg:
        ylbl[-1] = str(Mod_bins[-1])
    return xlbl, ylbl

def _pub_ax(ax, res, title="", xlabel="", ylabel=None):
    if ylabel is None:
        ylabel = f"Bins ({res//1000} kb)"
    ax.set_title(title, fontsize=PUB_FS)
    ax.set_xlabel(xlabel, fontsize=PUB_FS)
    ax.set_ylabel(ylabel, fontsize=PUB_FS)
    ax.tick_params(axis='both', which='major', labelsize=PUB_FS)

def _seg_borders(ax, Mod_bins, intrvl, res):
    if Mod_bins is None or intrvl is None:
        return
    xlbl, ylbl = _seg_labels(intrvl, Mod_bins, res)
    borders = np.array(Mod_bins, dtype=float)
    mids = (borders[:-1] + borders[1:]) / 2.0
    ax.set_yticks(Mod_bins); ax.set_yticklabels(ylbl, fontsize=PUB_FS)
    ax.set_xticks(Mod_bins, minor=True)
    ax.set_xticks(mids); ax.set_xticklabels(xlbl, rotation=45, ha='left', fontsize=PUB_FS)
    ax.xaxis.grid(True, which='minor', color='black', linestyle='--', linewidth=PUB_LW)
    ax.yaxis.grid(True, which='minor', color='black', linestyle='--', linewidth=PUB_LW)
    for be in Mod_bins[1:]:
        ax.axhline(be, color='black', linestyle='--', linewidth=PUB_LW, clip_on=False)
        ax.axvline(be, color='black', linestyle='--', linewidth=PUB_LW, clip_on=False)

def plot_ec3d_style(matNT, path, res, Mod_bins=None, intrvl=None, title=None, min_segment_size=50_000, fontsize=PUB_FS, axis_label="Path bins", cbar_label=r"$\log_{10}(\mathrm{value})$"):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.matshow(np.log10(matNT), cmap='YlOrRd')

    xticklbl = []
    if intrvl is not None:
        for (chr_, start, end) in intrvl:
            if end - start >= min_segment_size:
                s = (f"{start/1e6:.1f}".rstrip('0').rstrip('.')
                     if start % 1e6 != 0 else f"{int(start/1e6)}")
                e = (f"{end/1e6:.1f}".rstrip('0').rstrip('.')
                     if end % 1e6 != 0 else f"{int(end/1e6)}")
                xticklbl.append(f"{chr_}:{s}-{e}Mb")
            else:
                xticklbl.append('')

    if Mod_bins is not None:
        ax.set_yticks(Mod_bins[0:], minor=False)
        yticklbl = [''] * len(Mod_bins)
        if intrvl is not None:
            for i, (chr_, start, end) in enumerate(intrvl):
                if end - start >= min_segment_size:
                    yticklbl[i] = str(Mod_bins[i])
            last_sz = intrvl[-1][2] - intrvl[-1][1]
            if last_sz >= min_segment_size:
                yticklbl[-1] = str(Mod_bins[-1])
        ax.set_yticklabels(yticklbl, fontsize=fontsize)

        borders = np.append(0, Mod_bins[1:])
        mids = (borders[:-1] + borders[1:]) / 2
        ax.set_xticks(Mod_bins, minor=True)
        ax.set_xticks(mids, minor=False)
        ax.set_xticklabels(xticklbl, rotation=45, ha='left',
                           fontsize=fontsize, minor=False)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(cbar_label, fontsize=fontsize)
    ax.set_ylabel(f"{axis_label} ({res//1000} kb)", fontsize=fontsize)
    ax.set_xlabel("")
    ax.xaxis.grid(True, which='minor', color='black', linestyle='--',
                  linewidth=0.4)
    ax.yaxis.grid(True, which='minor', color='black', linestyle='--',
                  linewidth=0.4)

    if Mod_bins is not None:
        for be in Mod_bins[1:]:
            ax.axhline(y=be, color='black', linestyle='--',
                       linewidth=0.4, clip_on=False)
            ax.axvline(x=be, color='black', linestyle='--',
                       linewidth=0.4, clip_on=False)

    if title is not None:
        ax.set_title(title, fontsize=fontsize, pad=14)

    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=PUB_DPI)
    plt.close(fig)

def plot_heatmap(data, path, res, Mod_bins=None, intrvl=None,
                 title="", cmap="YlOrRd", pmax=99.5,
                 log="log10", floor_percentile=50):
    data = np.asarray(data, dtype=float)
    fig, ax = plt.subplots(figsize=PUB_SQ)
    pm = np.ma.masked_invalid(data)
    if log == "log10":
        pm = np.ma.log10(pm + 1.0)
    elif log == "log2":
        pm = np.ma.masked_where(pm <= 0, pm)
        pm = np.ma.log2(pm)
    elif log == "linear":
        pass
    else:
        raise ValueError(f"Unknown log scale: {log!r}")
    fv = np.asarray(pm.compressed(), dtype=float)
    fv = fv[np.isfinite(fv)]
    vmin = float(np.percentile(fv, floor_percentile)) if fv.size > 0 else None
    vmax = float(np.percentile(fv, pmax)) if fv.size > 0 else None
    if vmin is not None and vmax is not None and vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax.matshow(pm, cmap=cmap, vmin=vmin, vmax=vmax)
    _pub_ax(ax, res, title=title)
    _seg_borders(ax, Mod_bins, intrvl, res)
    if Mod_bins is None:
        ax.set_xlabel(f"Bins ({res//1000} kb)", fontsize=PUB_FS)
    cbar_label = {"log10": r"$\log_{10}(\mathrm{value}+1)$", "log2":  r"$\log_2(\mathrm{value})$", "linear": "Value"}[log]
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=PUB_FS)
    cb.set_label(cbar_label, fontsize=PUB_FS)
    plt.tight_layout(); fig.savefig(path, bbox_inches='tight', dpi=PUB_DPI); plt.close(fig)

def plot_centered(data, path, res, Mod_bins=None, intrvl=None,title="", cmap="coolwarm", pmax=98, cbar_label="Value"):
    mat = np.ma.masked_invalid(np.array(data, dtype=float, copy=True))
    fv  = np.asarray(mat.compressed(), dtype=float)
    vmax = float(np.percentile(np.abs(fv), pmax)) if fv.size > 0 else 1.0
    if not np.isfinite(vmax) or vmax <= 0: vmax = 1.0
    fig, ax = plt.subplots(figsize=PUB_SQ)
    im = ax.matshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
    _pub_ax(ax, res, title=title)
    _seg_borders(ax, Mod_bins, intrvl, res)
    if Mod_bins is None:
        ax.set_xlabel(f"Bins ({res//1000} kb)", fontsize=PUB_FS)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=PUB_FS)
    cb.set_label(cbar_label, fontsize=PUB_FS)
    plt.tight_layout(); plt.savefig(path, dpi=PUB_DPI, bbox_inches='tight'); plt.close(fig)

def plot_diff(a, b, path, res, title="", cbar_label="Contact difference"):
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    vmax = float(np.nanpercentile(np.abs(diff), 99))
    if not np.isfinite(vmax) or vmax <= 0: vmax = 1.0
    fig, ax = plt.subplots(figsize=PUB_SQ)
    im = ax.matshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    _pub_ax(ax, res, title=title, xlabel=f"Bins ({res//1000} kb)", ylabel=f"Bins ({res//1000} kb)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=PUB_FS)
    cb.set_label(cbar_label, fontsize=PUB_FS)
    plt.tight_layout(); plt.savefig(path, dpi=PUB_DPI, bbox_inches='tight'); plt.close(fig)

def plot_backbone_fit(dist_bp, raw, fit, path, title):
    fig, ax = plt.subplots(figsize=PUB_RECT)
    ax.loglog(dist_bp, raw, "o", ms=4, label="Observed mean")
    ax.loglog(dist_bp, fit, "-", lw=2, label="Monotone backbone f(d)")
    ax.set_xlabel("ecDNA circular distance (bp)", fontsize=PUB_FS)
    ax.set_ylabel("Mean contact", fontsize=PUB_FS)
    ax.set_title(title, fontsize=PUB_FS)
    ax.tick_params(axis='both', which='major', labelsize=PUB_FS)
    ax.legend(fontsize=PUB_LEGEND_FS)
    plt.tight_layout(); plt.savefig(path, dpi=PUB_DPI, bbox_inches='tight'); plt.close(fig)

def plot_alpha_dist(av, path, title=r"$\alpha_{ij}$ distribution"):
    av = np.asarray(av, dtype=float); av = av[np.isfinite(av) & (av > 0)]
    if av.size == 0: return
    fig, ax = plt.subplots(figsize=PUB_RECT)
    ax.hist(np.log2(av), bins=100)
    ax.set_xlabel(r"$\log_2(\alpha_{ij})$", fontsize=PUB_FS)
    ax.set_ylabel("Count", fontsize=PUB_FS)
    ax.set_title(title, fontsize=PUB_FS)
    ax.tick_params(axis='both', which='major', labelsize=PUB_FS)
    plt.tight_layout(); plt.savefig(path, dpi=PUB_DPI, bbox_inches='tight'); plt.close(fig)

def plot_convergence(hist_df, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(hist_df["iter"], hist_df["loglik"], marker="o")
    axes[0].set_xlabel("Iteration", fontsize=PUB_FS)
    axes[0].set_ylabel("Log-likelihood", fontsize=PUB_FS)
    axes[0].set_title("Observed-data log-likelihood", fontsize=PUB_FS)

    axes[1].plot(hist_df["iter"], hist_df["x_corr_prev"], marker="o")
    axes[1].set_xlabel("Iteration", fontsize=PUB_FS)
    axes[1].set_ylabel(r"$\mathrm{corr}(X_t, X_{t-1})$", fontsize=PUB_FS)
    axes[1].set_title("Latent allocation stability", fontsize=PUB_FS)

    axes[2].plot(hist_df["iter"], hist_df["alpha_corr_prev"], marker="o")
    axes[2].set_xlabel("Iteration", fontsize=PUB_FS)
    axes[2].set_ylabel(r"$\mathrm{corr}(\alpha_t, \alpha_{t-1})$", fontsize=PUB_FS)
    axes[2].set_title("Alpha stability", fontsize=PUB_FS)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=PUB_FS)

    plt.tight_layout(); plt.savefig(path, dpi=PUB_DPI, bbox_inches='tight'); plt.close(fig)

def plot_x_vs_lambda(X_upper, lam_upper, path):
    eps = 1e-8
    idx = np.triu_indices_from(X_upper, k=1)
    x = X_upper[idx].astype(float)
    y = lam_upper[idx].astype(float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if x.size == 0: return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=6, alpha=0.3, rasterized=True)
    mn = min(x.min(), y.min()); mx = max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], "--", linewidth=1.5, color="red")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Latent deconvolved $X_{ij}$", fontsize=PUB_FS)
    ax.set_ylabel(r"Expected $\lambda_{ij}$", fontsize=PUB_FS)
    ax.set_title("Observed versus expected latent contacts", fontsize=PUB_FS)
    ax.tick_params(axis='both', which='major', labelsize=PUB_FS)
    plt.tight_layout(); plt.savefig(path, dpi=PUB_DPI, bbox_inches='tight'); plt.close(fig)

def plot_hist(vals, path, title, xlabel):
    vals = np.asarray(vals, dtype=float); vals = vals[np.isfinite(vals)]
    if vals.size == 0: return
    fig, ax = plt.subplots(figsize=PUB_RECT)
    ax.hist(vals, bins=80)
    ax.set_xlabel(xlabel, fontsize=PUB_FS)
    ax.set_ylabel("Count", fontsize=PUB_FS)
    ax.set_title(title,  fontsize=PUB_FS)
    ax.tick_params(axis='both', which='major', labelsize=PUB_FS)
    plt.tight_layout(); plt.savefig(path, dpi=PUB_DPI, bbox_inches='tight'); plt.close(fig)

def plot_oe_spread(dist_df, path):
    if dist_df is None or dist_df.shape[0] == 0: return
    fig, ax = plt.subplots(figsize=PUB_RECT)
    ax.plot(dist_df["dist"], dist_df["iqr_oe_lambda"], marker="o", label=r"IQR $\log_2(X/\lambda)$")
    ax.plot(dist_df["dist"], dist_df["iqr_oe_backbone"],marker="o", label=r"IQR $\log_2(X/\mathrm{backbone})$")
    ax.set_xscale("log")
    ax.set_xlabel("ecDNA circular distance (bp)", fontsize=PUB_FS)
    ax.set_ylabel(r"IQR of $\log_2(O/E)$", fontsize=PUB_FS)
    ax.set_title("Distance-wise O/E spread",      fontsize=PUB_FS)
    ax.tick_params(axis='both', which='major', labelsize=PUB_FS)
    ax.legend(fontsize=PUB_LEGEND_FS)
    plt.tight_layout(); plt.savefig(path, dpi=PUB_DPI, bbox_inches='tight'); plt.close(fig)

def plot_family_metrics(family_df, path_entropy, path_size):
    fig, ax = plt.subplots(figsize=PUB_RECT)
    vals = family_df["family_entropy_norm"].dropna().values
    if vals.size > 0:
        ax.hist(vals, bins=60)
    ax.set_xlabel("Normalized family entropy", fontsize=PUB_FS)
    ax.set_ylabel("Count",                     fontsize=PUB_FS)
    ax.set_title("Allocation ambiguity profile\n" "(0 = deterministic, 1 = uniform)", fontsize=PUB_FS)
    ax.tick_params(axis='both', which='major', labelsize=PUB_FS)
    plt.tight_layout(); plt.savefig(path_entropy, dpi=PUB_DPI, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PUB_RECT)
    ax.hist(family_df["family_size"].values, bins=30, edgecolor="black")
    ax.set_xlabel("Family size", fontsize=PUB_FS)
    ax.set_ylabel("Count", fontsize=PUB_FS)
    ax.set_title("Family size distribution", fontsize=PUB_FS)
    ax.tick_params(axis='both', which='major', labelsize=PUB_FS)
    plt.tight_layout(); plt.savefig(path_size, dpi=PUB_DPI, bbox_inches='tight')
    plt.close(fig)

def recollapse_metrics(Y_obs, Y_hat):
    idx = np.triu_indices_from(Y_obs)
    obs, hat = Y_obs[idx], Y_hat[idx]; diff = hat - obs
    corr = safe_corr(obs, hat)
    return pd.DataFrame([{"mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "pearson": corr,
        "max_abs_err": float(np.max(np.abs(diff))),
        "sum_abs_err": float(np.sum(np.abs(diff)))}])

def family_ambiguity(families, Y, lam, pi=None):
    rows = []
    use_pi = pi is not None
    for (u, v), fam in families.items():
        y_obs = float(Y[u, v])
        if not fam:
            rows.append({"ref_u": u, "ref_v": v, "observed_count": y_obs, "family_size": 0, "family_entropy": np.nan, "family_entropy_norm": np.nan, "family_pmax": np.nan,"dominant_pair": "", "dominant_pair_fraction": np.nan})
            continue

        if use_pi:
            w = np.array([max(pi[i] * pi[j] * lam[i, j], 1e-12)
                          for i, j in fam], dtype=float)
        else:
            w = np.array([max(lam[i, j], 1e-12) for i, j in fam], dtype=float)
        w /= w.sum()
        entropy = -np.sum(w * np.log(w))
        enorm   = entropy / math.log(len(fam)) if len(fam) > 1 else 0.0
        dom_idx = int(np.argmax(w))
        rows.append({"ref_u": u, "ref_v": v, "observed_count": y_obs, "family_size": len(fam),"family_entropy": float(entropy),
            "family_entropy_norm": float(enorm),"family_pmax": float(w[dom_idx]), "dominant_pair": f"{fam[dom_idx][0]}-{fam[dom_idx][1]}", "dominant_pair_fraction": float(w[dom_idx])})
    return pd.DataFrame(rows)

def oe_spread(X_upper, lam_upper, bb_upper, res):
    eps = 1e-8; n = X_upper.shape[0]; rows = []
    for i in range(n):
        for j in range(i + 1, n):
            x    = float(X_upper[i, j])
            lam  = float(lam_upper[i, j])
            bb   = float(bb_upper[i, j])
            if not (np.isfinite(x) and np.isfinite(lam) and np.isfinite(bb)):
                continue  # skip NaN-masked (unmeasurable) cells
            d_bp = circ_dist(i, j, n) * res
            rows.append({"dist": d_bp, "log2_oe_lambda": math.log2((x + eps) / (lam + eps)),
                            "log2_oe_backbone": math.log2((x + eps) / (bb + eps))})
    df  = pd.DataFrame(rows)
    out = (df.groupby("dist", as_index=False)
             .agg(n_pairs=("log2_oe_lambda","size"),
                  iqr_oe_lambda  =("log2_oe_lambda",
                                   lambda z: float(np.percentile(z,75)-np.percentile(z,25))),
                  iqr_oe_backbone=("log2_oe_backbone",
                                   lambda z: float(np.percentile(z,75)-np.percentile(z,25))))
             .sort_values("dist").reset_index(drop=True))
    return out

def top1pct_overlap(X_upper, lam_upper, bb_upper):
    eps = 1e-8; n = X_upper.shape[0]; rows = []
    for i in range(n):
        for j in range(i + 1, n):
            x = float(X_upper[i, j])
            lam = float(lam_upper[i, j])
            bb = float(bb_upper[i, j])
            if not (np.isfinite(x) and np.isfinite(lam) and np.isfinite(bb)):
                continue  # skip NaN-masked (unmeasurable) cells
            rows.append({"i": i, "j": j, "sl": math.log2((x + eps) / (lam + eps)),
                            "sb": math.log2((x + eps) / (bb  + eps))})
    df = pd.DataFrame(rows)
    k = max(1, int(math.ceil(len(df) * 0.01)))
    top_l = set(df.nlargest(k, "sl")[["i","j"]].itertuples(index=False, name=None))
    top_b = set(df.nlargest(k, "sb")[["i","j"]].itertuples(index=False, name=None))
    ov = len(top_l & top_b); un = len(top_l | top_b)
    return pd.DataFrame([{"n_total": len(df), "top_k": k,  "n_overlap": ov,
                           "jaccard": float(ov/un) if un > 0 else np.nan}])

def alpha_bound_summary(alpha_upper, alpha_min, alpha_max, tol=1e-6):
    vals = alpha_upper[np.triu_indices_from(alpha_upper)]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return pd.DataFrame([{}])
    return pd.DataFrame([{"alpha_min_param": alpha_min, "alpha_max_param": alpha_max, "alpha_median": float(np.median(vals)),
        "alpha_mean": float(np.mean(vals)), "alpha_p95": float(np.percentile(vals, 95)), "frac_at_min": float(np.mean(np.abs(vals - alpha_min) <= tol)),
        "frac_at_max": float(np.mean(np.abs(vals - alpha_max) <= tol))}])
