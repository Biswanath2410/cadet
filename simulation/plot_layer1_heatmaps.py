"""Heatmaps of Layer 1 truth vs deconvolution."""
import os,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_cadet_on_sim import import_cadet, run_cadet_mode


def load_and_run(simdir, cd):
    Y = np.loadtxt(os.path.join(simdir, "Y_observed.txt"), delimiter="\t")
    X = np.loadtxt(os.path.join(simdir, "X_true.txt"), delimiter="\t")
    path = pd.read_csv(os.path.join(simdir, "path_bins.tsv"), sep="\t")
    pi = np.loadtxt(os.path.join(simdir, "pi_true.txt"), delimiter="\t")
    rr = {m: run_cadet_mode(Y, path, mode=m, anchor_window_bins=50, cd=cd) for m in ["cadet_pi_on", "cadet_pi_off", "naive"]}

    return Y, X, pi, path, rr


def _symmetrize(M):
    M = np.asarray(M, dtype=float)
    return np.triu(M, 1) + np.triu(M, 1).T + np.diag(np.diag(M))


def ec3d_heatmap(ax, data, title, log="log10",cmap="YlOrRd", floor_percentile=50, pmax=99.5):

    data = _symmetrize(data)
    pm = np.ma.masked_invalid(data)

    if log == "log10":
        pm = np.ma.log10(pm + 1.0)
    elif log == "log2":
        pm = np.ma.masked_where(pm <= 0, pm)
        pm = np.ma.log2(pm)

    fv = np.asarray(pm.compressed(), dtype=float)
    fv = fv[np.isfinite(fv)]

    vmin = float(np.percentile(fv, floor_percentile)) if fv.size > 0 else None
    vmax = float(np.percentile(fv, pmax)) if fv.size > 0 else None

    if vmin is not None and vmax is not None and vmax <= vmin:
        vmax = vmin + 1e-6

    im = ax.matshow(pm, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def ec3d_centered(ax, data, title, cmap="coolwarm", pmax=99):

    mat = np.ma.masked_invalid(np.array(_symmetrize(data), dtype=float, copy=True))
    fv = np.asarray(mat.compressed(), dtype=float)
    vmax = (float(np.percentile(np.abs(fv), pmax)) if fv.size > 0 else 1.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = ax.matshow(mat,cmap=cmap,vmin=-vmax,vmax=vmax)

    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main(sim_root,out):
    specs=[("B_one_dup_symmetric_seed0","Symmetric duplicated copies"),
           ("C_one_dup_asymmetric_seed0","Asymmetric duplicated copies")]
    cd = import_cadet()
    fig, axs = plt.subplots(7, 2, figsize=(10, 28))
    for col, (name, head) in enumerate(specs):
        simdir = os.path.join(sim_root, name)
        Y, X, pi_true, path, rr = load_and_run(simdir, cd)

        X_pi_on = rr["cadet_pi_on"]["X_hat"]
        X_pi_off = rr["cadet_pi_off"]["X_hat"]
        X_naive = rr["naive"]["X_hat"]
        # Row 1: observed collapsed reference Y
        ec3d_heatmap(axs[0, col],Y,f"{head}: observed Y")
        # Row 2: latent truth X
        ec3d_heatmap(axs[1, col],X,"Latent X truth")
        # Row 3: CADET with pi
        ec3d_heatmap(axs[2, col],X_pi_on, "CADET π on")
        # Row 4: CADET without pi
        ec3d_heatmap(axs[3, col],X_pi_off,"CADET π off")
        # Row 5: backbone-only
        ec3d_heatmap(axs[4, col],X_naive,"Backbone only")
        # Row 6: residual of CADET pi-on relative to truth
        residual = ( np.log2(np.asarray(X_pi_on, dtype=float) + 1.0) - np.log2(np.asarray(X, dtype=float) + 1.0) )

        ec3d_centered( axs[5, col],residual,"Residual: log2(CADET π on + 1) − log2(truth + 1)")
        # Row 7: pi truth versus estimated pi
        # duplicated path bins only
        ax = axs[6, col]
        path_out = rr["cadet_pi_on"]["path_df"]
        ref_counts = path_out.groupby("ref_id").size().to_dict()

        dup_bins = [i for i, ref_id in enumerate(path_out["ref_id"].astype(int)) if ref_counts[int(ref_id)] > 1]

        if dup_bins:
            x = np.asarray(pi_true[dup_bins], dtype=float)
            y = np.asarray(rr["cadet_pi_on"]["pi_hat"][dup_bins], dtype=float)

            ax.scatter(x, y, s=24, alpha=0.8)
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            ax.plot([lo, hi],[lo, hi],"--",linewidth=1)

            corr = (np.corrcoef(x, y)[0, 1] if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0 else np.nan)

            rmse = np.sqrt(np.mean((y - x) ** 2))

            ax.text(0.04,0.96,f"Pearson r = {corr:.3f}\n" f"RMSE = {rmse:.3f}\n" f"n = {len(x)} duplicated path bins", transform=ax.transAxes, va="top", fontsize=8)

        ax.set_xlabel("True π")
        ax.set_ylabel("Estimated π")
        ax.set_title("Copy-anchor π recovery",fontsize=9,pad=3)
    fig.suptitle("CADET Layer 1: copy-resolved contact recovery",fontsize=15,y=.995)
    fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--sim_root",required=True); p.add_argument("--out",default="layer1_heatmaps.png")
    a=p.parse_args(); main(a.sim_root,a.out)
