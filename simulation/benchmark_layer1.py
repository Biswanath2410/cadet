"""Layer 1 simulation benchmark: all scenarios x seeds x modes."""
import os
import time
import argparse
import numpy as np
import pandas as pd

from simulate_ecdna_hic import SimConfig, simulate, cycle_dist
from run_cadet_on_sim import import_cadet, run_cadet_mode

cd = import_cadet()

# n_path >= 120 so the +/-50-bin anchor window never reaches the same partner twice.
SCENARIOS = [
    dict(name="A_no_dup", n_path=120, n_dup_regions=0, dup_size=0,
         multiplicity=2, n_peaks=18, depth_scale=1.0,
         pi_evenness=50.0, n_tads=2, peak_min_dist_bins=4),
    dict(name="B_one_dup_symmetric", n_path=120, n_dup_regions=1, dup_size=12,
         multiplicity=2, n_peaks=18, depth_scale=1.0,
         pi_evenness=50.0, n_tads=2, peak_min_dist_bins=4),
    dict(name="C_one_dup_asymmetric", n_path=120, n_dup_regions=1, dup_size=12,
         multiplicity=2, n_peaks=18, depth_scale=1.0,
         pi_evenness=1.5, n_tads=2, peak_min_dist_bins=4),
    dict(name="D_two_dup_symmetric", n_path=160, n_dup_regions=2, dup_size=15,
         multiplicity=2, n_peaks=24, depth_scale=1.0,
         pi_evenness=50.0, n_tads=3, peak_min_dist_bins=4),
    dict(name="E_two_dup_asymmetric", n_path=160, n_dup_regions=2, dup_size=15,
         multiplicity=2, n_peaks=24, depth_scale=1.0,
         pi_evenness=1.5, n_tads=3, peak_min_dist_bins=4),
]
MODES = ["cadet_pi_on", "cadet_pi_off", "naive"]


def safe_corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_family_members(path_to_ref):
    ref_to_paths = {}
    for p, r in path_to_ref.items():
        ref_to_paths.setdefault(int(r), []).append(int(p))
    refs = sorted(ref_to_paths)
    families = {}
    for ai, u in enumerate(refs):
        for v in refs[ai:]:
            mem = set()
            for i in ref_to_paths[u]:
                for j in ref_to_paths[v]:
                    a, b = (i, j) if i <= j else (j, i)
                    mem.add((a, b))
            families[(u, v)] = sorted(mem)
    return families


def alpha_geometric_mean(alpha, path_to_ref):
    vals = []
    n = alpha.shape[0]
    for i in range(n):
        for j in range(i, n):
            if i < j and int(path_to_ref[i]) == int(path_to_ref[j]):
                continue
            a = float(alpha[i, j])
            if np.isfinite(a) and a > 0:
                vals.append(a)
    return float(np.exp(np.mean(np.log(vals)))) if vals else np.nan


def pi_family_maxdev(pi, path_to_ref):
    ref_to_bins = {}
    for p, r in path_to_ref.items():
        ref_to_bins.setdefault(int(r), []).append(int(p))
    deviations = []
    for bins in ref_to_bins.values():
        if len(bins) > 1:
            deviations.append(abs(float(np.mean(pi[bins])) - 1.0))
    return float(max(deviations)) if deviations else 0.0


def duplicated_pair_mask(path_to_ref, n):
    families = build_family_members(path_to_ref)
    mask = np.zeros((n, n), dtype=bool)
    for members in families.values():
        if len(members) <= 1:
            continue
        for i, j in members:
            if i < j:
                mask[i, j] = True
    return mask


def allocation_metrics(X_true, X_hat, path_to_ref):
    l1s = []
    dominant_info = []
    for members in build_family_members(path_to_ref).values():
        if len(members) <= 1:
            continue
        t = np.asarray([X_true[i, j] for i, j in members], float)
        h = np.asarray([X_hat[i, j] for i, j in members], float)
        if t.sum() <= 0 or h.sum() <= 0:
            continue
        pt = t / t.sum(); ph = h / h.sum()
        l1s.append(float(np.sum(np.abs(pt - ph))))
        correct = int(np.argmax(pt)) == int(np.argmax(ph))
        # Require at least 5 percentage points separation from next-best member.
        if len(pt) >= 2:
            ordered = np.sort(pt)
            informative = (ordered[-1] - ordered[-2]) >= 0.05
        else:
            informative = False
        if informative:
            dominant_info.append(float(correct))
    return {
        "family_l1_mean": float(np.mean(l1s)) if l1s else np.nan,
        "dominant_pair_acc": float(np.mean(dominant_info)) if dominant_info else np.nan,
    }


def _binary_auc(y_true, score):
    """AUROC for binary labels, with average ranks for ties (no sklearn)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]; s = s[ok]
    n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan

    # AUROC via pairwise rank statistic, with average ranks for ties.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    k = 0
    while k < len(s):
        j = k + 1
        while j < len(s) and s[order[j]] == s[order[k]]:
            j += 1
        avg_rank = 0.5 * ((k + 1) + j)
        ranks[order[k:j]] = avg_rank
        k = j
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def alpha_family_records(alpha_true, alpha_hat, path_to_ref, family_support=None):
    """Median true and estimated alpha per reference family (different-reference families only:
    same-reference families are skipped because tau_between_copy lets their members differ).
    Returns records (true, estimated, ambiguous, support) and the largest within-family truth range."""
    records = []
    truth_ranges = []
    for (u, v), members in build_family_members(path_to_ref).items():
        if u == v:
            continue
        tv = np.asarray([alpha_true[i, j] for i, j in members], float)
        hv = np.asarray([alpha_hat[i, j] for i, j in members], float)
        ok = np.isfinite(tv) & np.isfinite(hv) & (tv > 0) & (hv > 0)
        if not ok.any():
            continue
        tv = tv[ok]; hv = hv[ok]
        truth_ranges.append(float(np.max(tv) - np.min(tv)))
        support = np.nan
        if family_support is not None:
            support = float(family_support.get((int(u), int(v)), np.nan))
        records.append((float(np.median(tv)), float(np.median(hv)), len(members) > 1, support))
    return records, (float(max(truth_ranges)) if truth_ranges else 0.0)


def alpha_family_metrics(alpha_true, alpha_hat, path_to_ref, family_support=None,
                         enriched_threshold=1.5):
    """Alpha recovery per reference family: correlation over all families and over ambiguous
    (duplicated) families, error on ambiguous families, correlation on the top-support quartile,
    and how well enriched families (true alpha > 1.5) are ranked (AUROC)."""
    records, _ = alpha_family_records(alpha_true, alpha_hat, path_to_ref, family_support)
    amb = [r for r in records if r[2]]
    out = {
        "alpha_r_all": safe_corr([r[0] for r in records], [r[1] for r in records]),
        "alpha_r_ambig": safe_corr([r[0] for r in amb], [r[1] for r in amb]),
        "alpha_rmse_ambig": (float(np.sqrt(np.mean([(r[0] - r[1]) ** 2 for r in amb])))
                                        if amb else np.nan),
    }
    supported = [r for r in records if np.isfinite(r[3])]
    if supported:
        q75 = float(np.quantile([r[3] for r in supported], 0.75))
        high_amb = [r for r in supported if r[3] >= q75 and r[2]]
        out["alpha_r_ambig_highsupp"] = safe_corr([r[0] for r in high_amb],
                                                                      [r[1] for r in high_amb])
    else:
        out["alpha_r_ambig_highsupp"] = np.nan
    out["alpha_auroc_ambig"] = (
        _binary_auc([r[0] > enriched_threshold for r in amb], [r[1] for r in amb]) if amb else np.nan)
    return out


def backbone_true_matrix(f_true, n, depth_scale):
    B = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i, n):
            d = cycle_dist(i, j, n)
            B[i, j] = float(f_true[d]) * float(depth_scale)
    return B


def recovery_metrics(sim, result):
    X_true = sim["X_true"]; X_hat = result["X_hat"]
    alpha_true = sim["alpha_true"]; alpha_hat = result["alpha_hat"]
    pi_true = sim["pi_true"]; pi_hat = result["pi_hat"]
    path_to_ref = sim["path_to_ref"]
    n = X_true.shape[0]
    tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    out = {}
    xt = X_true[tri]; xh = X_hat[tri]
    valid = np.isfinite(xt) & np.isfinite(xh) & ((xt > 0) | (xh > 0))
    out["x_pearson_log"] = safe_corr(np.log1p(xt[valid]), np.log1p(xh[valid]))

    dmask = duplicated_pair_mask(path_to_ref, n)
    dt = X_true[dmask]; dh = X_hat[dmask]
    dv = np.isfinite(dt) & np.isfinite(dh) & ((dt > 0) | (dh > 0))
    out["dup_x_pearson_log"] = (safe_corr(np.log1p(dt[dv]), np.log1p(dh[dv]))
                                if dv.sum() >= 3 else np.nan)

    out.update(allocation_metrics(X_true, X_hat, path_to_ref))

    # Score alpha by collapsed reference family. Layer 1 does not have separate
    # observations for different latent members of the same family.
    B_true_full = backbone_true_matrix(sim["f_true"], n, sim["config"].depth_scale)
    family_support = {}
    for fam, members in build_family_members(path_to_ref).items():
        support = 0.0
        for i, j in members:
            support += float(pi_true[i] * pi_true[j] * B_true_full[i, j])
        family_support[fam] = support
    out.update(alpha_family_metrics(alpha_true, alpha_hat, path_to_ref,
                                    family_support=family_support))

    ref_counts = {}
    for p, r in path_to_ref.items():
        ref_counts[int(r)] = ref_counts.get(int(r), 0) + 1
    dup_bins = [p for p, r in path_to_ref.items() if ref_counts[int(r)] > 1]
    if len(dup_bins) >= 3:
        pt = pi_true[dup_bins]; ph = pi_hat[dup_bins]
        out["pi_pearson_dup"] = safe_corr(pt, ph)
        out["pi_rmse_dup"] = float(np.sqrt(np.mean((pt-ph)**2)))
    else:
        out["pi_pearson_dup"] = np.nan
        out["pi_rmse_dup"] = np.nan
    out["recollapse_err"] = float(result["recollapse_err"])
    return out


def validate_truth(sim):
    gm = alpha_geometric_mean(sim["alpha_true"], sim["path_to_ref"])
    if not np.isclose(gm, 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"alpha_true geometric mean != 1: {gm}")
    dev = pi_family_maxdev(sim["pi_true"], sim["path_to_ref"])
    if dev > 1e-12:
        raise AssertionError(f"pi_true copy-family arithmetic mean != 1; max dev={dev}")
    # Every ordinary reference family must have one shared alpha truth value.
    _, max_range = alpha_family_records(sim["alpha_true"], sim["alpha_true"], sim["path_to_ref"])
    if max_range > 1e-12:
        raise AssertionError(f"alpha_true is not family-common; max within-family range={max_range}")


def run_sweep(out_root="bench_layer1", n_seeds=20, seed_start=0,
              anchor_window_bins=50):
    os.makedirs(out_root, exist_ok=True)
    rows = []
    seeds = list(range(seed_start, seed_start + n_seeds))
    total = len(SCENARIOS) * len(seeds) * len(MODES)
    done = 0
    t0 = time.time()
    for scen in SCENARIOS:
        for seed in seeds:
            cfg = SimConfig(
                name=f"{scen['name']}_seed{seed}",
                outdir=os.path.join(out_root, "sims"), seed=seed,
                **{k:v for k,v in scen.items() if k != "name"},
            )
            sim = simulate(cfg)
            sim["config"] = cfg
            validate_truth(sim)
            for mode in MODES:
                result = run_cadet_mode(
                    sim["Y"], sim["path_df"], mode=mode, cd=cd,
                    res=cfg.resolution_bp,
                    alpha_min=cfg.fit_alpha_min, alpha_max=cfg.fit_alpha_max,
                    anchor_window_bins=anchor_window_bins,
                )
                m = {"scenario": scen["name"], "seed": seed, "mode": mode}
                m.update(recovery_metrics(sim, result))
                rows.append(m)
                done += 1
                print(f"[{done:3d}/{total}] {scen['name']:24s} seed={seed:2d} "
                      f"{mode:13s} Xlog={m['x_pearson_log']:.3f} "
                      f"dupXlog={m['dup_x_pearson_log']:.3f} "
                      f"recollapse={m['recollapse_err']:.1e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "all_runs.tsv"), sep="\t", index=False)
    metric_cols = [c for c in df.columns if c not in ("scenario", "seed", "mode")]
    agg = df.groupby(["scenario", "mode"], sort=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns.to_flat_index()]
    agg.to_csv(os.path.join(out_root, "summary_by_scenario_mode.tsv"), sep="\t", index=False)

    print(f"Finished {len(df)} Layer-1 runs in {time.time()-t0:.1f}s")
    return df, agg


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CADET Layer 1 simulation benchmark")
    p.add_argument("--out_root", default="bench_layer1")
    p.add_argument("--n_seeds", type=int, default=20)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--anchor_window_bins", type=int, default=50,
                   help="Default 50, as in the pipeline")
    args = p.parse_args()
    run_sweep(args.out_root, args.n_seeds, args.seed_start, args.anchor_window_bins)
