#!/usr/bin/env python3
"""Extra figures: heatmaps for the two-duplication scenarios (D, E) and
seed-wise diagnostics (pi, alpha, dominant-pair accuracy, recollapse error).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_cadet_on_sim import import_cadet, run_cadet_mode


LABELS = {
    "B_one_dup_symmetric": "1 duplication,\nnear-equal copy activity",
    "C_one_dup_asymmetric": "1 duplication,\nunequal copy activity",
    "D_two_dup_symmetric": "2 duplications,\nnear-equal copy activity",
    "E_two_dup_asymmetric": "2 duplications,\nunequal copy activity",
}

DUP_SCENARIOS = [
    "B_one_dup_symmetric",
    "C_one_dup_asymmetric",
    "D_two_dup_symmetric",
    "E_two_dup_asymmetric",
]

MODES = ["cadet_pi_on", "cadet_pi_off", "naive"]
MODE_LABEL = {
    "cadet_pi_on": "CADET π on",
    "cadet_pi_off": "CADET π off",
    "naive": "Backbone only",
}



def _symmetrize(M):
    M = np.asarray(M, dtype=float)
    return np.triu(M, 1) + np.triu(M, 1).T + np.diag(np.diag(M))


def ec3d_heatmap(ax, data, title, log="log10",
                 cmap="YlOrRd", floor_percentile=50, pmax=99.5):
    data = _symmetrize(data)
    pm = np.ma.masked_invalid(data)

    if log == "log10":
        pm = np.ma.log10(pm + 1.0)
    elif log == "log2":
        pm = np.ma.masked_where(pm <= 0, pm)
        pm = np.ma.log2(pm)

    fv = np.asarray(pm.compressed(), dtype=float)
    fv = fv[np.isfinite(fv)]

    vmin = float(np.percentile(fv, floor_percentile)) if fv.size else None
    vmax = float(np.percentile(fv, pmax)) if fv.size else None
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
    vmax = float(np.percentile(np.abs(fv), pmax)) if fv.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = ax.matshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def load_and_run(simdir, cd):
    Y = np.loadtxt(os.path.join(simdir, "Y_observed.txt"), delimiter="\t")
    X = np.loadtxt(os.path.join(simdir, "X_true.txt"), delimiter="\t")
    path = pd.read_csv(os.path.join(simdir, "path_bins.tsv"), sep="\t")
    pi_true = np.loadtxt(os.path.join(simdir, "pi_true.txt"), delimiter="\t")

    rr = {
        mode: run_cadet_mode(
            Y, path, mode=mode, anchor_window_bins=50, cd=cd
        )
        for mode in MODES
    }
    return Y, X, pi_true, path, rr


def plot_complex_heatmaps(sim_root, out):
    """Same layout as plot_layer1_heatmaps.py, for scenarios D and E."""
    specs = [
        ("D_two_dup_symmetric_seed0", "Two duplications, near-equal copy activity"),
        ("E_two_dup_asymmetric_seed0", "Two duplications, unequal copy activity"),
    ]

    cd = import_cadet()
    fig, axs = plt.subplots(7, 2, figsize=(10, 28))

    for col, (name, head) in enumerate(specs):
        simdir = os.path.join(sim_root, name)
        if not os.path.isdir(simdir):
            raise FileNotFoundError(f"Missing simulated directory: {simdir}")

        Y, X, pi_true, path, rr = load_and_run(simdir, cd)

        X_on = rr["cadet_pi_on"]["X_hat"]
        X_off = rr["cadet_pi_off"]["X_hat"]
        X_naive = rr["naive"]["X_hat"]

        ec3d_heatmap(axs[0, col], Y, f"{head}: observed Y")
        ec3d_heatmap(axs[1, col], X, "Latent X truth")
        ec3d_heatmap(axs[2, col], X_on, "CADET π on")
        ec3d_heatmap(axs[3, col], X_off, "CADET π off")
        ec3d_heatmap(axs[4, col], X_naive, "Backbone only")

        residual = (
            np.log2(np.asarray(X_on, dtype=float) + 1.0)
            - np.log2(np.asarray(X, dtype=float) + 1.0)
        )
        ec3d_centered(
            axs[5, col],
            residual,
            "Residual: log2(CADET π on + 1) − log2(truth + 1)",
        )

        ax = axs[6, col]
        path_out = rr["cadet_pi_on"]["path_df"]
        ref_counts = path_out.groupby("ref_id").size().to_dict()
        dup_bins = [
            i for i, ref_id in enumerate(path_out["ref_id"].astype(int))
            if ref_counts[int(ref_id)] > 1
        ]

        if dup_bins:
            x = np.asarray(pi_true[dup_bins], dtype=float)
            y = np.asarray(rr["cadet_pi_on"]["pi_hat"][dup_bins], dtype=float)
            ax.scatter(x, y, s=24, alpha=0.8)

            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1)

            corr = (
                np.corrcoef(x, y)[0, 1]
                if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0
                else np.nan
            )
            rmse = np.sqrt(np.mean((y - x) ** 2))
            ax.text(
                0.04, 0.96,
                f"Pearson r = {corr:.3f}\n"
                f"RMSE = {rmse:.3f}\n"
                f"n = {len(x)} duplicated path bins",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

        ax.set_xlabel("True π")
        ax.set_ylabel("Estimated π")
        ax.set_title("Copy-anchor π recovery", fontsize=9, pad=3)

    fig.suptitle(
        "CADET Layer 1: copy-resolved recovery in two-duplication simulations",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")



# Seed-wise aggregate diagnostics


def _mean_sd(df, scenario, mode, metric):
    vals = pd.to_numeric(
        df.loc[
            (df["scenario"] == scenario) & (df["mode"] == mode),
            metric,
        ],
        errors="coerce",
    ).dropna()
    mean = float(vals.mean()) if len(vals) else np.nan
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, sd


def _bars(ax, df, metric, modes, ylabel, title, ylim=None):
    xs = np.arange(len(DUP_SCENARIOS))
    width = 0.24 if len(modes) == 3 else 0.34

    for k, mode in enumerate(modes):
        means, sds = [], []
        for scenario in DUP_SCENARIOS:
            mean, sd = _mean_sd(df, scenario, mode, metric)
            means.append(mean)
            sds.append(sd)

        offset = (k - (len(modes) - 1) / 2) * width
        ax.bar(
            xs + offset,
            means,
            width,
            yerr=sds,
            capsize=2,
            label=MODE_LABEL[mode],
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [LABELS[s] for s in DUP_SCENARIOS],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_seedwise_diagnostics(bench, out):
    df = pd.read_csv(bench, sep="\t")

    required = {
        "scenario", "seed", "mode",
        "pi_pearson_dup", "pi_rmse_dup",
        "alpha_r_ambig_highsupp",
        "alpha_auroc_ambig",
        "dominant_pair_acc",
        "recollapse_err",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")

    n_seeds = int(df["seed"].nunique())
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    # pi is meaningful only when copy-anchor estimation is enabled.
    _bars(
        axs[0, 0], df, "pi_pearson_dup", ["cadet_pi_on"],
        "Pearson r", "A  π recovery across seeds", (0, 1.02),
    )
    _bars(
        axs[0, 1], df, "pi_rmse_dup", ["cadet_pi_on"],
        "RMSE", "B  π estimation error", None,
    )

    _bars(
        axs[0, 2], df, "alpha_r_ambig_highsupp",
        ["cadet_pi_on", "cadet_pi_off"],
        "Pearson r",
        "C  α recovery in high-support ambiguous families",
        (0, 1.02),
    )

    _bars(
        axs[1, 0], df, "alpha_auroc_ambig",
        ["cadet_pi_on", "cadet_pi_off"],
        "AUROC",
        "D  Ranking enriched ambiguous families",
        (0.5, 1.02),
    )

    _bars(
        axs[1, 1], df, "dominant_pair_acc",
        MODES,
        "Accuracy",
        "E  Dominant copy-pair recovery",
        (0, 1.02),
    )

    # Count conservation is best displayed on log10 scale because errors are
    # at floating-point precision.
    ax = axs[1, 2]
    vals = pd.to_numeric(df["recollapse_err"], errors="coerce")
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals):
        logv = np.log10(vals)
        ax.hist(logv, bins=20)
        ax.axvline(logv.max(), linestyle="--", linewidth=1)
        ax.text(
            0.04, 0.96,
            f"max = {10**logv.max():.2e}\n"
            f"n = {len(logv)} runs",
            transform=ax.transAxes,
            va="top",
        )
    ax.set_xlabel("log10 max |collapse(X̂) − Y|")
    ax.set_ylabel("Runs")
    ax.set_title("F  Recollapse error", loc="left", fontweight="bold")

    handles, labels = axs[1, 1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        f"CADET Layer 1 additional simulation diagnostics "
        f"(n={n_seeds} seeds per scenario)",
        fontsize=15,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bench",
        required=True,
        help="bench_layer1/all_runs.tsv",
    )
    p.add_argument(
        "--sim_root",
        required=True,
        help="bench_layer1/sims",
    )
    p.add_argument(
        "--out_heatmaps",
        default="layer1_heatmaps_complex.png",
    )
    p.add_argument(
        "--out_diagnostics",
        default="layer1_seedwise_diagnostics.png",
    )
    args = p.parse_args()

    plot_complex_heatmaps(args.sim_root, args.out_heatmaps)
    plot_seedwise_diagnostics(args.bench, args.out_diagnostics)


if __name__ == "__main__":
    main()
