"""Does pi move counts to the right copy?

Two copies of one reference bin sit on opposite sides of the circle and the
held-out targets are equally far from both, so the backbone alone gives 50:50.
Alpha is 1 everywhere, so only pi_true makes the copies differ. Pi is
estimated from a separate local window and scored on the held-out targets.
"""
import os
import argparse
import tempfile
import numpy as np
import pandas as pd
from run_cadet_on_sim import import_cadet, ensure_ref_bin_key

cd = import_cadet()
# Skip diagnostic plots.
cd.plot_alpha_dist = lambda *args, **kwargs: None


def cycle_dist(i, j, n):
    d = abs(int(i)-int(j)); return min(d, n-d)


def make_stress(seed=0, n_path=40, depth_scale=8.0, resolution_bp=5000):
    rng = np.random.default_rng(seed)
    copy_a, copy_b = 0, n_path//2
    heldout = [n_path//4, 3*n_path//4]

    # One duplicated ref family; every other path bin is a unique ref.
    path_to_ref = {}
    dup_ref = 0
    path_to_ref[copy_a] = dup_ref
    path_to_ref[copy_b] = dup_ref
    next_ref = 1
    for p in range(n_path):
        if p in (copy_a, copy_b):
            continue
        path_to_ref[p] = next_ref; next_ref += 1

    # Arithmetic-mean-1 duplicated-family truth.
    pi_true = np.ones(n_path, float)
    pi_true[copy_a] = 1.6
    pi_true[copy_b] = 0.4
    alpha_true = np.ones((n_path, n_path), float)  # only pi differs between copies

    f = np.zeros(n_path//2 + 1, float)
    f[0] = 100.0
    for d in range(1, len(f)):
        f[d] = 200.0 * (d ** -1.1)

    lam = np.zeros((n_path, n_path), float)
    X = np.zeros_like(lam)
    for i in range(n_path):
        for j in range(i, n_path):
            d = cycle_dist(i, j, n_path)
            mu = pi_true[i]*pi_true[j]*alpha_true[i,j]*f[d]*depth_scale
            lam[i,j] = mu
            X[i,j] = rng.poisson(mu)

    n_ref = max(path_to_ref.values()) + 1
    Y = np.zeros((n_ref, n_ref), float)
    for i in range(n_path):
        for j in range(i, n_path):
            u, v = path_to_ref[i], path_to_ref[j]
            if u > v: u, v = v, u
            Y[u,v] += X[i,j]

    path_df = pd.DataFrame({"path_bin": np.arange(n_path), "seg_idx": 0, "chrom": "chrSim",
        "genomic_bp": [path_to_ref[i]*resolution_bp for i in range(n_path)],
        "strand": "+", "pos_in_segment": np.arange(n_path),
        "segment_start_bp": 0, "segment_end_bp": n_path*resolution_bp,
        "genomic_end_bp": [path_to_ref[i]*5000+5000 for i in range(n_path)]})
    path_df = ensure_ref_bin_key(path_df)
    return Y, path_df, X, lam, pi_true, alpha_true, path_to_ref, heldout, copy_a, copy_b


def one_seed(seed, anchor_window_bins=6):
    Y, path_df, X_true, lam_true, pi_true, alpha_true, path_to_ref, heldout, ca, cb = make_stress(seed)
    path_df, ref_df, members = cd.build_refbin_index(path_df)
    families = cd.build_families(ref_df, members)
    n = len(path_df)

    # Use the exact known backbone here so the test focuses on pi rather
    # than conflating copy-anchor estimation with backbone-fitting error.
    f = np.zeros(n//2+1, float); f[0]=100.0
    for d in range(1,len(f)): f[d]=200.0*(d**-1.1)*8.0
    backbone = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i,n): backbone[i,j]=f[cycle_dist(i,j,n)]
    same_ref_mask = cd.build_same_ref_mask(path_df)
    pi_hat = cd.compute_copy_anchors(Y, path_df, backbone,
                                     window_bins=anchor_window_bins, eps=1e-3)
    alpha0 = cd.init_alpha(n); lam0 = cd.build_lambda(backbone, alpha0)

    rows=[]
    for mode in ["cadet_pi_on", "cadet_pi_off", "naive"]:
        if mode == "naive":
            X_hat,_=cd.e_step(Y,families,lam0,pi=None)
            ah=np.ones((n,n)); ph=np.ones(n)
        else:
            em_pi = pi_hat if mode=="cadet_pi_on" else None
            with tempfile.TemporaryDirectory(prefix="cadet_pi_stress_") as td:
                _, ah, X_hat, _, _ = cd.run_em(
                    Y, families, n, backbone, lam0, alpha0, td,
                    max_iter=100, tol=1e-6, tau=1.0,
                    alpha_min=0.1, alpha_max=10.0,
                    same_ref_mask=same_ref_mask, tau_between_copy=10.0,
                    pi=em_pi, verbose=False)
            ph = pi_hat if em_pi is not None else np.ones(n)

        for target in heldout:
            ia,ja=sorted((ca,target)); ib,jb=sorted((cb,target))
            ta=float(X_true[ia,ja]); tb=float(X_true[ib,jb])
            ha=float(X_hat[ia,ja]); hb=float(X_hat[ib,jb])
            true_share=ta/(ta+tb) if ta+tb>0 else np.nan
            hat_share=ha/(ha+hb) if ha+hb>0 else np.nan
            rows.append({ "seed":seed,"mode":mode,"target":target,
                "backbone_distance_copy_a":cycle_dist(ca,target,n),
                "backbone_distance_copy_b":cycle_dist(cb,target,n),
                "true_share_copy_a":true_share,"hat_share_copy_a":hat_share,
                "abs_share_error":abs(hat_share-true_share),
                "pi_true_copy_a":pi_true[ca],"pi_true_copy_b":pi_true[cb],
                "pi_hat_copy_a":ph[ca],"pi_hat_copy_b":ph[cb],
                "alpha_hat_copy_a_target":ah[ia,ja],
                "alpha_hat_copy_b_target":ah[ib,jb]})
    return rows


def run(out="pi_stress_layer1.tsv", n_seeds=50, anchor_window_bins=6):
    rows=[]
    for seed in range(n_seeds): rows.extend(one_seed(seed, anchor_window_bins))
    df=pd.DataFrame(rows)
    tied=(df["backbone_distance_copy_a"]==df["backbone_distance_copy_b"])
    if not bool(tied.all()): raise AssertionError("Test pairs do not have equal backbone values")
    df.to_csv(out,sep="\t",index=False)
    summary=(df.groupby("mode",sort=False)
             .agg(n=("hat_share_copy_a","size"),
                  true_share_mean=("true_share_copy_a","mean"),
                  hat_share_mean=("hat_share_copy_a","mean"),
                  hat_share_sd=("hat_share_copy_a","std"),
                  abs_share_error_mean=("abs_share_error","mean"),
                  pi_hat_copy_a_mean=("pi_hat_copy_a","mean"),
                  pi_hat_copy_b_mean=("pi_hat_copy_b","mean"))
             .reset_index())
    root,ext=os.path.splitext(out); sout=root+"_summary.tsv"
    summary.to_csv(sout,sep="\t",index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out} and {sout}")
    return df,summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="pi_stress_layer1.tsv")
    p.add_argument("--n_seeds",type=int,default=50)
    p.add_argument("--anchor_window_bins",type=int,default=6,
                   help="Purpose-specific local training window; held-out targets lie outside it")
    a=p.parse_args(); run(a.out,a.n_seeds,a.anchor_window_bins)
