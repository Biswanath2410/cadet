"""Summary figure for the Layer 1 simulation."""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_cadet_on_sim import import_cadet, run_cadet_mode

LABELS={
 "A_no_dup":"No duplication",
 "B_one_dup_symmetric":"1 duplication, near-equal copy activity",
 "C_one_dup_asymmetric":"1 duplication, unequal copy activity",
 "D_two_dup_symmetric":"2 duplications, near-equal copy activity",
 "E_two_dup_asymmetric":"2 duplications, unequal copy activity",
}
MODES=["cadet_pi_on","cadet_pi_off","naive"]
MODE_LABEL={"cadet_pi_on":"CADET π on","cadet_pi_off":"CADET π off","naive":"Backbone only"}


def panel_bars(ax, df, metric, scenarios, title, ylabel, ylim=None):
    width=0.24; xs=np.arange(len(scenarios))
    for k,mode in enumerate(MODES):
        means=[]; sds=[]
        for s in scenarios:
            v=df.loc[(df["scenario"]==s)&(df["mode"]==mode),metric].dropna().astype(float)
            means.append(v.mean() if len(v) else np.nan)
            sds.append(v.std(ddof=1) if len(v)>1 else 0.0)
        ax.bar(xs+(k-1)*width,means,width,yerr=sds,capsize=2,label=MODE_LABEL[mode])
    ax.set_xticks(xs); ax.set_xticklabels([LABELS.get(s,s) for s in scenarios],rotation=28,ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title,loc="left",fontweight="bold")
    if ylim: ax.set_ylim(*ylim)


def main(bench, stress, sim, out):
    df=pd.read_csv(bench,sep="\t")
    st=pd.read_csv(stress,sep="\t")
    n_seeds=int(df["seed"].nunique())
    scenarios=list(dict.fromkeys(df["scenario"].tolist()))
    dup_scen=[s for s in scenarios if s!="A_no_dup"]

    Y=np.loadtxt(os.path.join(sim,"Y_observed.txt"),delimiter="\t")
    path=pd.read_csv(os.path.join(sim,"path_bins.tsv"),sep="\t")
    pi_true=np.loadtxt(os.path.join(sim,"pi_true.txt"),delimiter="\t")
    cd=import_cadet()
    r=run_cadet_mode(Y,path,mode="cadet_pi_on",anchor_window_bins=50,cd=cd)
    ref_counts=r["path_df"].groupby("ref_id").size().to_dict()
    dup_bins=[i for i,x in enumerate(r["path_df"].ref_id.astype(int)) if ref_counts[int(x)]>1]

    fig,axs=plt.subplots(2,3,figsize=(16,9))
    panel_bars(axs[0,0],df,"x_pearson_log",scenarios,
               f"A  Latent X recovery\n(n={n_seeds} {'seed' if n_seeds == 1 else 'seeds'} per scenario)","Pearson r, log1p X",(0,1.02))
    panel_bars(axs[0,1],df,"dup_x_pearson_log",dup_scen,
               "B  Recovery within ambiguous duplicated families","Pearson r, log1p X",(0,1.02))
    panel_bars(axs[0,2],df,"family_l1_mean",dup_scen,
               "C  Family allocation error","Mean L1 error",None)
    panel_bars(axs[1,0],df,"dominant_pair_acc",dup_scen,
               "D  Dominant copy-pair recovery","Accuracy",(0,1.02))

    ax=axs[1,1]

    if dup_bins:
        x = np.asarray(pi_true[dup_bins], dtype=float)
        y = np.asarray(r["pi_hat"][dup_bins], dtype=float)
        ax.scatter(x, y, s=32, alpha=.8)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", lw=1)
        corr = (np.corrcoef(x, y)[0,1] if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0 else np.nan)
        rmse = np.sqrt(np.mean((y - x) ** 2))

        if len(x) >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = np.nan, np.nan

        ax.text(.04, .96,f"Pearson r = {corr:.3f}\n"f"RMSE = {rmse:.3f}\n"f"slope = {slope:.3f}\n"f"n = {len(x)} duplicated path bins", transform=ax.transAxes, va="top")

    ax.set_xlabel("True π")
    ax.set_ylabel("Estimated π")
    ax.set_title("E  Copy-anchor π recovery\n(representative asymmetric replicate)",loc="left",fontweight="bold")

    ax=axs[1,2]
    order=[m for m in MODES if m in set(st["mode"])]
    vals=[]; errs=[]
    for m in order:
        v=st.loc[st["mode"]==m,"hat_share_copy_a"].astype(float)
        vals.append(v.mean()); errs.append(v.std(ddof=1))
        
    x = np.arange(len(order))
    mode_colors = {"cadet_pi_on": "tab:blue","cadet_pi_off": "tab:orange","naive": "tab:green",}
    ax.bar(x,vals,yerr=errs,capsize=3,color=[mode_colors[m] for m in order])

    truth = float(st["true_share_copy_a"].mean())
    ax.axhline(truth,ls="--",lw=1.2,color="gray",label=f"Truth = {truth:.3f}")
    
    ax.set_xticks(x); ax.set_xticklabels([MODE_LABEL[m] for m in order],rotation=20,ha="right")
    ax.set_ylim(0,1); ax.set_ylabel("Allocated share to high-π copy")
    nstress=int(st["seed"].nunique())
    ax.set_title(f"F  π copy-allocation test\n(n={nstress} {'seed' if nstress == 1 else 'seeds'})",loc="left",fontweight="bold")
    ax.legend(frameon=False,fontsize=8)

    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.01))
    fig.suptitle("CADET Layer 1 simulation validation",fontsize=15,y=1.04)
    fig.tight_layout()
    fig.savefig(out,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bench",required=True)
    p.add_argument("--stress",required=True)
    p.add_argument("--sim",required=True,help="Representative asymmetric simulation directory")
    p.add_argument("--out",default="layer1_simulation_summary.png")
    a=p.parse_args(); main(a.bench,a.stress,a.sim,a.out)
