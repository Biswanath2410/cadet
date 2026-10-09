"""Simulate ecDNA Hi-C with the Layer 1 model.
"""
import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List



@dataclass
class SimConfig:
    """Simulator settings (saved to sim_config.json)."""
    # Path layout
    n_path:        int = 60
    n_dup_regions: int = 1
    dup_size:      int = 10            # length of each duplicated block
    multiplicity:  int = 2             # copies per duplicated block (>=2)

    # Polymer backbone f(d) — power-law with explicit f(0)
    f0:                 float = 100.0
    backbone_alpha:     float = -1.1   # decay exponent (negative)
    backbone_beta:      float = 200.0  # scale: f(d>0) = beta * d^backbone_alpha

    # Per-pair enrichment alpha — focal "loop" peaks
    n_peaks:            int   = 20
    peak_alpha_min:     float = 3.0
    peak_alpha_max:     float = 8.0
    peak_min_dist_bins: int   = 3      # minimum circular distance for focal features

    # TAD-like blocks (within-domain α elevation)
    n_tads:             int   = 2
    tad_size_min:       int   = 6
    tad_size_max:       int   = 12
    tad_alpha:          float = 2.0    # α multiplier for within-TAD short-range pairs
    tad_max_d:          int   = 4      # only inflate within this cycle-distance

    # Copy activity (pi) within each duplicated family.
    # Low pi_evenness -> one copy dominates; high -> all copies near 1.
    pi_evenness: float = 2.0
    pi_force_asymmetry:         bool  = False   # if True, force one copy >> others
    pi_asymmetry_ratio:         float = 5.0     # used only if pi_force_asymmetry

    # Shared simulation / inference settings
    resolution_bp:      int   = 5000   # must match the Layer 1 bin size
    fit_alpha_min:      float = 0.1    # Layer 1 alpha lower bound
    fit_alpha_max:      float = 10.0   # Layer 1 alpha upper bound

    # Sampling
    depth_scale:        float = 1.0    # multiplies every lambda before Poisson
    noise_model:        str   = "poisson"   # "poisson" or "negbin"
    nb_theta:           float = 10.0   # NB dispersion (smaller = more overdispersed)
    add_poisson_noise:  bool  = True   # legacy alias: if False, X = lambda

    # Reproducibility
    seed:               int   = 0

    # Output
    name:               str   = "sim"
    outdir:             str   = "sim_out"

    # Older parameter names, still accepted.
    asymmetric_copy_load: bool  = False
    asym_copy_alpha:      float = 3.0
    asym_copy_radius:     int   = 2



def cycle_dist(i, j, n):
    d = abs(int(i) - int(j))
    return min(d, n - d)


def build_path_layout(cfg: SimConfig):
    """Map path bins to reference bins.

    Each duplicated region has dup_size bins and `multiplicity` copies,
    placed roughly opposite each other on the circle.
    """
    n = cfg.n_path
    path_to_ref = {}
    next_ref = 0

    dup_positions = []
    for r in range(cfg.n_dup_regions):
        copy_starts = [int(round(n * (r/(cfg.n_dup_regions * cfg.multiplicity)
                                       + c/cfg.multiplicity))) % n
                       for c in range(cfg.multiplicity)]
        dup_positions.append(copy_starts)

    occupied = [False] * n
    ref_for_block = []
    for r in range(cfg.n_dup_regions):
        block_ref_ids = list(range(next_ref, next_ref + cfg.dup_size))
        next_ref += cfg.dup_size
        ref_for_block.append(block_ref_ids)
        for start in dup_positions[r]:
            for k in range(cfg.dup_size):
                pos = (start + k) % n
                if occupied[pos]:
                    raise ValueError(
                        f"Duplicated regions overlap at path bin {pos}. "
                        "Reduce n_dup_regions, dup_size, or multiplicity, "
                        "or enlarge n_path.")
                path_to_ref[pos] = block_ref_ids[k]
                occupied[pos] = True

    for pos in range(n):
        if not occupied[pos]:
            path_to_ref[pos] = next_ref
            next_ref += 1

    return path_to_ref




def canonicalize_ref_ids(path_to_ref):
    """Renumber reference bins in order of first appearance along the path,
    as CADET does, so Y uses the same family IDs as CADET.
    """
    old_to_new = {}
    next_id = 0
    out = {}
    for pbin in sorted(path_to_ref):
        old_id = int(path_to_ref[pbin])
        if old_id not in old_to_new:
            old_to_new[old_id] = next_id
            next_id += 1
        out[int(pbin)] = int(old_to_new[old_id])
    return out

def build_backbone(cfg: SimConfig):
    """f(d) by circular distance: power law with a separate f(0)."""
    n = cfg.n_path
    max_d = n // 2
    f = np.zeros(max_d + 1)
    f[0] = cfg.f0
    for d in range(1, max_d + 1):
        f[d] = cfg.backbone_beta * (d ** cfg.backbone_alpha)
    return f


def sample_pi_true(cfg: SimConfig, path_to_ref, rng):
    """Draw pi_true for duplicated bins.

    Copy weights are random (or fixed if pi_force_asymmetry is set) and
    scaled to mean 1 within each family, as in compute_copy_anchors().
    Unique bins get pi = 1.
    """
    n = cfg.n_path
    ref_to_bins = {}
    for p, r in path_to_ref.items():
        ref_to_bins.setdefault(r, []).append(p)

    pi = np.ones(n)
    for r, bins in ref_to_bins.items():
        if len(bins) <= 1:
            continue
        m = len(bins)
        if cfg.pi_force_asymmetry:
            # one copy at ratio, others at 1
            w = np.ones(m)
            w[0] = cfg.pi_asymmetry_ratio
        else:
            w = rng.dirichlet(np.full(m, cfg.pi_evenness)) * m
        # Mean 1 within the family, as in CADET's copy anchors.
        w = np.clip(w, 1e-6, None)
        mean_w = float(np.mean(w))
        w = w / mean_w
        for b, val in zip(sorted(bins), w):
            pi[b] = float(val)

    # Legacy compatibility: if asymmetric_copy_load was set but pi
    # parameters were left at defaults, enforce force_asymmetry.
    if (cfg.asymmetric_copy_load and not cfg.pi_force_asymmetry
            and cfg.pi_evenness >= 1.5):
        for r, bins in ref_to_bins.items():
            if len(bins) <= 1: continue
            m = len(bins)
            w = np.ones(m)
            w[0] = max(2.0, cfg.asym_copy_alpha)
            w = np.clip(w, 1e-6, None)
            mean_w = float(np.mean(w))
            w = w / mean_w
            for b, val in zip(sorted(bins), w):
                pi[b] = float(val)

    return pi


def build_true_alpha(cfg: SimConfig, path_to_ref, rng):
    """One alpha per reference family, shared by all its path pairs.

    TAD and focal features are placed at path positions, but the enrichment
    goes to the whole reference family, since Y cannot tell members apart.
    Returns alpha_true (upper-triangle n x n) and a table of seeded families.
    """
    n = cfg.n_path
    alpha_true = np.ones((n, n), dtype=float)
    peaks: List[dict] = []

    # Reference family -> all compatible upper-triangle path-pairs.
    ref_to_bins = {}
    for pbin, rid in path_to_ref.items():
        ref_to_bins.setdefault(int(rid), []).append(int(pbin))
    refs = sorted(ref_to_bins)
    family_members = {}
    pair_to_family = {}
    for ai, uref in enumerate(refs):
        for vref in refs[ai:]:
            members = set()
            for i in ref_to_bins[uref]:
                for j in ref_to_bins[vref]:
                    a, b = (i, j) if i <= j else (j, i)
                    members.add((a, b))
            members = sorted(members)
            family_members[(uref, vref)] = members
            for pair in members:
                pair_to_family[pair] = (uref, vref)

    family_alpha = {fam: 1.0 for fam in family_members}

    def assign_family(seed_i, seed_j, value, feature_type):
        a, b = (int(seed_i), int(seed_j)) if seed_i <= seed_j else (int(seed_j), int(seed_i))
        fam = pair_to_family[(a, b)]
        old = family_alpha[fam]
        if value > old:
            family_alpha[fam] = float(value)
        peaks.append({
            "bin_i": a, "bin_j": b,
            "ref_u": int(fam[0]), "ref_v": int(fam[1]),
            "alpha": float(family_alpha[fam]),
            "type": feature_type,
            "family_size": int(len(family_members[fam])),
        })

    # ---- TAD-like family enrichments, seeded from local path geometry ----
    occupied_tad = np.zeros(n, dtype=bool)
    n_tads_placed = 0
    for _t in range(cfg.n_tads):
        placed = False
        for _ in range(100):
            ts = int(rng.integers(cfg.tad_size_min, cfg.tad_size_max + 1))
            start = int(rng.integers(0, n))
            idx = [(start + k) % n for k in range(ts)]
            if occupied_tad[idx].any():
                continue
            occupied_tad[idx] = True
            seeded = set()
            for ii in range(ts):
                for jj in range(ii + 1, ts):
                    bi, bj = idx[ii], idx[jj]
                    if cycle_dist(bi, bj, n) <= cfg.tad_max_d:
                        a, b = (bi, bj) if bi <= bj else (bj, bi)
                        fam = pair_to_family[(a, b)]
                        if fam not in seeded:
                            assign_family(a, b, cfg.tad_alpha, "tad_family")
                            seeded.add(fam)
            placed = True
            n_tads_placed += 1
            break
        if not placed:
            raise RuntimeError(
                f"Could place only {n_tads_placed} of {cfg.n_tads} requested TAD regions "
                "after 100 attempts per region. Change the layout or TAD settings."
            )

    # ---- Focal family enrichments ----
    tries = 0
    n_loops_done = 0
    used_families = {pair_to_family[(r["bin_i"], r["bin_j"])] for r in peaks}
    while n_loops_done < cfg.n_peaks and tries < cfg.n_peaks * 500:
        tries += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if cycle_dist(a, b, n) < cfg.peak_min_dist_bins:
            continue
        fam = pair_to_family[(a, b)]
        # Keep focal truth at the family level and avoid reusing a family.
        if fam in used_families:
            continue
        val = float(rng.uniform(cfg.peak_alpha_min, cfg.peak_alpha_max))
        assign_family(a, b, val, "loop_family")
        used_families.add(fam)
        n_loops_done += 1

    if n_loops_done != cfg.n_peaks:
        raise RuntimeError(
            f"Could place only {n_loops_done} of {cfg.n_peaks} requested focal families "
            f"after {tries} attempts. Change the layout or focal-feature settings."
        )

    # Materialize family-common alpha on every compatible path-pair.
    for fam, members in family_members.items():
        val = float(family_alpha[fam])
        for i, j in members:
            alpha_true[i, j] = val

    return alpha_true, peaks


def normalize_alpha(alpha_true, path_to_ref):
    """Scale alpha to geometric mean 1 over the same pairs as Layer 1
    (same-reference between-copy pairs excluded).
    """
    alpha = np.asarray(alpha_true, dtype=float).copy()
    n = alpha.shape[0]
    vals = []
    for i in range(n):
        for j in range(i, n):
            # Same-reference between-copy pairs are left out, as in Layer 1.
            if i < j and int(path_to_ref[i]) == int(path_to_ref[j]):
                continue
            a = float(alpha[i, j])
            if np.isfinite(a) and a > 0:
                vals.append(a)
    if vals:
        gm = float(np.exp(np.mean(np.log(np.asarray(vals, dtype=float)))))
        if np.isfinite(gm) and gm > 0:
            alpha = alpha / gm
    return alpha


def compute_true_pi(X_true, path_to_ref):
    """Descriptive pi from X_true, for diagnostics only.

    pi_true.txt holds the pi used to generate the data.
    """
    n_path = X_true.shape[0]
    ref_to_bins = {}
    for p, r in path_to_ref.items():
        ref_to_bins.setdefault(r, []).append(p)
    mult = {r: len(v) for r, v in ref_to_bins.items()}

    pi = np.ones(n_path)
    for r, bins in ref_to_bins.items():
        if mult[r] <= 1: continue
        scores = []
        for b in bins:
            num = 0.0
            for q, r_q in path_to_ref.items():
                if mult[r_q] != 1 or q == b: continue
                u, v = (b, q) if b < q else (q, b)
                num += X_true[u, v]
            scores.append(num)
        if max(scores) == 0: continue
        m = len(scores)
        s_sum = sum(scores)
        if s_sum > 0:
            normed = [m * s / s_sum for s in scores]
            for b, s in zip(bins, normed):
                pi[b] = s
    return pi


def simulate(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)

    # 1) Layout
    path_to_ref = build_path_layout(cfg)
    # Use the ref_id order build_refbin_index() will rebuild; must happen before Y.
    path_to_ref = canonicalize_ref_ids(path_to_ref)
    n_path = cfg.n_path

    # 2) Backbone
    f_arr = build_backbone(cfg)

    # 3) pi_true
    pi_true = sample_pi_true(cfg, path_to_ref, rng)

    # 4) Family-level alpha_true with TAD/focal enrichments, normalized as in CADET
    alpha_true, peaks = build_true_alpha(cfg, path_to_ref, rng)
    alpha_true = normalize_alpha(alpha_true, path_to_ref)

    # Truth must stay inside the alpha range Layer 1 can fit.
    finite_alpha = alpha_true[np.isfinite(alpha_true) & (alpha_true > 0)]
    alpha_truth_min = float(finite_alpha.min())
    alpha_truth_max = float(finite_alpha.max())
    if alpha_truth_min < cfg.fit_alpha_min - 1e-12 or alpha_truth_max > cfg.fit_alpha_max + 1e-12:
        raise ValueError(
            "Normalized alpha truth falls outside the fitted Layer 1 range: "
            f"truth=[{alpha_truth_min:.6g}, {alpha_truth_max:.6g}], "
            f"allowed=[{cfg.fit_alpha_min:.6g}, {cfg.fit_alpha_max:.6g}]."
        )

    # Keep the truth table consistent with the normalized alpha matrix.
    for rec in peaks:
        i, j = int(rec["bin_i"]), int(rec["bin_j"])
        u, v = (i, j) if i <= j else (j, i)
        rec["alpha"] = float(alpha_true[u, v])

    # 5) lambda_true and X
    #     lambda_ij = pi_i * pi_j * alpha_F(i,j) * f(d_ij) * depth
    X_true = np.zeros((n_path, n_path), dtype=float)
    lam_true = np.zeros((n_path, n_path), dtype=float)
    for i in range(n_path):
        for j in range(i, n_path):
            d = cycle_dist(i, j, n_path)
            lam = (pi_true[i] * pi_true[j]
                   * alpha_true[i, j] * f_arr[d] * cfg.depth_scale)
            lam_true[i, j] = lam
            if not cfg.add_poisson_noise:
                X_true[i, j] = float(lam)
            elif cfg.noise_model == "negbin":
                # NB parameterization: mean=lam, dispersion=theta
                #   p = theta / (theta + lam),  n = theta
                theta = max(cfg.nb_theta, 1e-6)
                p = theta / (theta + lam) if lam > 0 else 1.0
                X_true[i, j] = float(rng.negative_binomial(theta, p))
            else:
                X_true[i, j] = float(rng.poisson(lam))

    # 6) Collapse to Y
    n_ref = 1 + max(path_to_ref.values())
    Y = np.zeros((n_ref, n_ref), dtype=float)
    for i in range(n_path):
        for j in range(i, n_path):
            u, v = path_to_ref[i], path_to_ref[j]
            if u <= v: Y[u, v] += X_true[i, j]
            else:      Y[v, u] += X_true[i, j]

    # 7) Descriptive pi (diagnostics only)
    pi_descriptive = compute_true_pi(X_true, path_to_ref)

    # 8) path_df / ref_df in the format CADET expects
    path_df = pd.DataFrame({
        "path_bin":   np.arange(n_path),
        "seg_idx":    [0] * n_path,
        "chrom":      ["chrSim"] * n_path,
        "genomic_bp": [path_to_ref[i] * cfg.resolution_bp for i in range(n_path)],
        "ref_bin_key":[("chrSim", path_to_ref[i] * cfg.resolution_bp) for i in range(n_path)],
        "strand":     ["+"] * n_path,
        "pos_in_segment":  list(range(n_path)),
        "segment_start_bp":[0] * n_path,
        "segment_end_bp":  [n_path * cfg.resolution_bp] * n_path,
        "genomic_end_bp":  [path_to_ref[i] * cfg.resolution_bp + cfg.resolution_bp for i in range(n_path)],
    })

    # 9) Save everything
    outdir = os.path.join(cfg.outdir, cfg.name)
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(os.path.join(outdir, "Y_observed.txt"),     Y,              delimiter="\t")
    np.savetxt(os.path.join(outdir, "X_true.txt"),         X_true,         delimiter="\t")
    np.savetxt(os.path.join(outdir, "lambda_true.txt"),    lam_true,       delimiter="\t")
    np.savetxt(os.path.join(outdir, "alpha_true.txt"),     alpha_true,     delimiter="\t")
    np.savetxt(os.path.join(outdir, "pi_true.txt"),        pi_true,        delimiter="\t")
    np.savetxt(os.path.join(outdir, "pi_descriptive.txt"), pi_descriptive, delimiter="\t")
    np.savetxt(os.path.join(outdir, "f_true.txt"),         f_arr,          delimiter="\t")
    # No tuples in the TSV; ref_bin_key is rebuilt from chrom/genomic_bp on load.
    path_df.drop(columns=["ref_bin_key"]).to_csv(
        os.path.join(outdir, "path_bins.tsv"), sep="\t", index=False
    )
    # Family-level alpha truth.
    ref_to_bins = {}
    for pbin, rid in path_to_ref.items():
        ref_to_bins.setdefault(int(rid), []).append(int(pbin))
    fam_rows = []
    refs = sorted(ref_to_bins)
    for ai, uref in enumerate(refs):
        for vref in refs[ai:]:
            members = set()
            for i in ref_to_bins[uref]:
                for j in ref_to_bins[vref]:
                    a, b = (i, j) if i <= j else (j, i)
                    members.add((a, b))
            members = sorted(members)
            vals = [float(alpha_true[i, j]) for i, j in members]
            fam_rows.append({
                "ref_u": int(uref), "ref_v": int(vref),
                "n_members": int(len(members)),
                "is_ambiguous": bool(len(members) > 1),
                "is_same_ref": bool(uref == vref),
                "alpha_true_family": float(np.median(vals)),
                "alpha_true_within_family_range": float(max(vals)-min(vals)),
            })
    pd.DataFrame(fam_rows).to_csv(
        os.path.join(outdir, "alpha_family_truth.tsv"), sep="\t", index=False
    )

    features_df = pd.DataFrame(peaks)
    features_df.to_csv(os.path.join(outdir, "true_features.tsv"), sep="\t", index=False)
    # Combined file kept for older scripts; not the focal-loop truth.
    features_df.to_csv(os.path.join(outdir, "true_peaks.tsv"), sep="\t", index=False)
    if len(features_df):
        loop_df = features_df.loc[features_df["type"] == "loop_family"].copy()
        tad_df = features_df.loc[features_df["type"] == "tad_family"].copy()
    else:
        loop_df = pd.DataFrame(columns=["bin_i", "bin_j", "alpha", "type"])
        tad_df = pd.DataFrame(columns=["bin_i", "bin_j", "alpha", "type"])
    loop_df.to_csv(os.path.join(outdir, "true_loop_peaks.tsv"), sep="\t", index=False)
    tad_df.to_csv(os.path.join(outdir, "true_tad_blocks.tsv"), sep="\t", index=False)
    with open(os.path.join(outdir, "sim_config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    return {
        "Y": Y, "X_true": X_true, "lambda_true": lam_true,
        "alpha_true": alpha_true,
        "pi_true": pi_true, "pi_descriptive": pi_descriptive,
        "f_true": f_arr, "path_df": path_df,
        "path_to_ref": path_to_ref, "peaks": peaks, "outdir": outdir,
        "alpha_truth_min": alpha_truth_min, "alpha_truth_max": alpha_truth_max,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    for fld in SimConfig.__dataclass_fields__.values():
        kw = {"default": fld.default, "type": fld.type}
        if fld.type is bool:
            kw["action"] = "store_true" if not fld.default else "store_false"
            kw.pop("type"); kw.pop("default")
        p.add_argument(f"--{fld.name}", **kw)
    args = p.parse_args()
    cfg = SimConfig(**{k: v for k, v in vars(args).items()})
    res = simulate(cfg)
    print(f"Done. Wrote outputs to {res['outdir']}/")
    print(f"  n_path={cfg.n_path}  n_ref={len(set(res['path_to_ref'].values()))}")
    print(f"  n_focal_families_planted={sum(r['type']=='loop_family' for r in res['peaks'])}")
    print(f"  Y total reads = {res['Y'].sum():.0f}")
    print(f"  pi_true range: [{res['pi_true'].min():.3f}, {res['pi_true'].max():.3f}]")
    print(f"  alpha_true range: [{res['alpha_truth_min']:.3f}, {res['alpha_truth_max']:.3f}] "
          f"within fitted range [{cfg.fit_alpha_min:.1f}, {cfg.fit_alpha_max:.1f}]")
