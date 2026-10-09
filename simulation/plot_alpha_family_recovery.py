"""Plot family-level alpha recovery from all_runs.tsv."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "A_no_dup": "No duplication",
    "B_one_dup_symmetric": "1 duplication, near-equal copy activity",
    "C_one_dup_asymmetric": "1 duplication, unequal copy activity",
    "D_two_dup_symmetric": "2 duplications, near-equal copy activity",
    "E_two_dup_asymmetric": "2 duplications, unequal copy activity",
}
MODES = ["cadet_pi_on", "cadet_pi_off"]
MODE_LABEL = {"cadet_pi_on": "CADET pi on", "cadet_pi_off": "CADET pi off"}

def bars(ax, df, metric, scenarios, title, ylabel, ylim=None):
    width = 0.34
    xs = np.arange(len(scenarios))
    for k, mode in enumerate(MODES):
        means, sds = [], []
        for scenario in scenarios:
            mask = (df["scenario"] == scenario) & (df["mode"] == mode)
            vals = pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()
            means.append(float(vals.mean()) if len(vals) else np.nan)
            sds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        offset = (k - (len(MODES)-1)/2) * width
        ax.bar(xs + offset, means, width, yerr=sds, capsize=2, label=MODE_LABEL[mode])
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(s, s) for s in scenarios], rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)

def main(bench, out):
    df = pd.read_csv(bench, sep="\t")
    required = {"scenario", "mode", "alpha_r_all",
                "alpha_r_ambig", "alpha_rmse_ambig"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "This plot expects benchmark_layer1.py all_runs.tsv. "
            f"Missing columns: {sorted(missing)}"
        )
    scenarios = list(dict.fromkeys(df["scenario"].astype(str).tolist()))
    dup = [s for s in scenarios if s != "A_no_dup"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    bars(axs[0], df, "alpha_r_all", scenarios,
         "A  Family-level alpha recovery", "Pearson r", (0, 1.02))
    bars(axs[1], df, "alpha_r_ambig", dup,
         "B  Alpha recovery in ambiguous families", "Pearson r", (0, 1.02))
    bars(axs[2], df, "alpha_rmse_ambig", dup,
         "C  Ambiguous-family alpha error", "RMSE", (0, None))

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("CADET Layer 1: family-level alpha recovery", fontsize=14, y=1.08)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True,
                        help="Per-run benchmark table: bench_layer1/all_runs.tsv")
    parser.add_argument("--out", default="layer1_alpha_family_recovery.png")
    args = parser.parse_args()
    main(args.bench, args.out)
