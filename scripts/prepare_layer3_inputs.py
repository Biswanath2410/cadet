#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd

def load_matrix(path):
    m = np.loadtxt(path, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"Matrix must be square: {path}")
    return m

def standardize_path_bins(path_bins_tsv, res):
    df = pd.read_csv(path_bins_tsv, sep="\t")
    if "path_bin" not in df.columns:
        if "path_idx" in df.columns:
            df = df.rename(columns={"path_idx": "path_bin"})
        else:
            df.insert(0, "path_bin", np.arange(len(df), dtype=int))

    required_col = ["path_bin", "chrom"]
    for c in required_col:
        if c not in df.columns:
            raise ValueError(f"path_bins.tsv missing required column: {c}")

    if "genomic_bp" not in df.columns:
        if "start" in df.columns:
            df["genomic_bp"] = df["start"].astype(int)
        elif "start_bp" in df.columns:
            df["genomic_bp"] = df["start_bp"].astype(int)
        else:
            raise ValueError("path_bins.tsv needs genomic_bp, start, or start_bp")

    if "seg_idx" not in df.columns:
        df["seg_idx"] = 0
    if "strand" not in df.columns:
        df["strand"] = "+"
    if "ref_id" not in df.columns:
        df["ref_id"] = pd.factorize(list(zip(df["chrom"].astype(str), df["genomic_bp"].astype(int))))[0]

    out = df[["path_bin", "chrom", "genomic_bp", "seg_idx", "strand", "ref_id"]].copy()
    out["path_bin"] = out["path_bin"].astype(int)
    out["genomic_bp"] = out["genomic_bp"].astype(int)
    out["seg_idx"] = out["seg_idx"].astype(int)
    out["ref_id"] = out["ref_id"].astype(int)
    out = out.sort_values("path_bin").reset_index(drop=True)

    expected = np.arange(len(out), dtype=int)
    if not np.array_equal(out["path_bin"].to_numpy(), expected):
        raise ValueError("path_bin must be contiguous from 0 to n-1 for Layer 3 compatibility")
    return out

def build_latent_pairs(path_df, res):
    n = len(path_df)
    rows = []
    chrom = path_df["chrom"].astype(str).to_numpy()
    bp = path_df["genomic_bp"].astype(int).to_numpy()
    seg = path_df["seg_idx"].astype(int).to_numpy()
    ref = path_df["ref_id"].astype(int).to_numpy()

    for i in range(n):
        for j in range(i + 1, n):
            d = min(j - i, n - (j - i))
            rows.append({"path_bin_i": i, "path_bin_j": j, "chrom_i": chrom[i], "start_i": int(bp[i]), "end_i": int(bp[i] + res), "chrom_j": chrom[j], "start_j": int(bp[j]), "end_j": int(bp[j] + res), "seg_idx_i": int(seg[i]), "seg_idx_j": int(seg[j]), "ref_id_i": int(ref[i]), "ref_id_j": int(ref[j]), "path_distance_bins": int(d), "path_dist": float(d * res),})
    return pd.DataFrame(rows)

def convert_part2_table(part2_csv, local_fc_threshold):
    df = pd.read_csv(part2_csv)
    if len(df) == 0:
        raise ValueError("Part 2 final_significant_interactions.csv is empty")

    required = ["bin1", "bin2", "count", "mu_hat", "obs_over_exp", "log2_oe"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Part 2 file missing required columns: {missing}")

    out = pd.DataFrame()
    out["bin1"] = df["bin1"].astype(int)
    out["bin2"] = df["bin2"].astype(int)

    if "dist_bin" in df.columns:
        out["distance"] = df["dist_bin"].astype(int)
    elif "dist" in df.columns:
        out["distance"] = df["dist"].astype(float)
    else:
        out["distance"] = np.abs(out["bin2"] - out["bin1"]).astype(int)

    out["count_raw"] = df["count"].astype(float)
    out["mu_raw"] = df["mu_hat"].astype(float)
    out["count_ice"] = df["count"].astype(float)
    out["mu_ice"] = df["mu_hat"].astype(float)
    out["obs_over_exp_raw"] = df["obs_over_exp"].astype(float)
    out["obs_over_exp_ice"] = df["obs_over_exp"].astype(float)
    out["log2_oe_raw"] = df["log2_oe"].astype(float)
    out["log2_oe_ice"] = df["log2_oe"].astype(float)

    # Layer 3 expects local_obs_over_mean/q75 to behave like local fold-enrichment.
    if "local_fc_mean" in df.columns:
        out["local_obs_over_mean"] = df["local_fc_mean"].astype(float)
    elif "local_oe_mean" in df.columns:
        out["local_obs_over_mean"] = out["obs_over_exp_raw"] / np.maximum(df["local_oe_mean"].astype(float), 1e-12)
    else:
        out["local_obs_over_mean"] = np.nan

    if "local_fc_q75" in df.columns:
        out["local_obs_over_q75"] = df["local_fc_q75"].astype(float)
    elif "local_oe_q75" in df.columns:
        out["local_obs_over_q75"] = out["obs_over_exp_raw"] / np.maximum(df["local_oe_q75"].astype(float), 1e-12)
    else:
        out["local_obs_over_q75"] = np.nan

    if "is_local_max" in df.columns:
        is_local = df["is_local_max"].astype(bool)
    else:
        is_local = pd.Series(True, index=df.index)
    out["passes_local"] = (out["local_obs_over_mean"] >= local_fc_threshold) & is_local
 
    for c in ["z_emp", "p_value", "q_value", "cand_rank", "distance_stratum", "local_wrap_affected", "is_sister_masked"]:
        if c in df.columns:
            out[c] = df[c]

    out["peak_class"] = "latent_focal_SI"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part2_csv", required=True, help="Layer 2 final_significant_interactions.csv")
    ap.add_argument("--path_bins", required=True, help="Part 1 path_bins.tsv")
    ap.add_argument("--latent_matrix", required=True, help="Part 1 latent_X_symmetric.txt")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--res", type=int, default=5000)
    ap.add_argument("--local_fc_threshold", type=float, default=1.2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    path_df = standardize_path_bins(args.path_bins, args.res)
    X = load_matrix(args.latent_matrix)
    if X.shape[0] != len(path_df):
        raise ValueError(f"latent matrix shape {X.shape} does not match path_bins n={len(path_df)}")

    sig = convert_part2_table(args.part2_csv, args.local_fc_threshold)
    max_bin = int(max(sig["bin1"].max(), sig["bin2"].max()))
    if max_bin >= len(path_df):
        raise ValueError(f"Part 2 bin index {max_bin} exceeds path_bins length {len(path_df)}")

    latent_pairs = build_latent_pairs(path_df, args.res)

    sig.to_csv(os.path.join(args.outdir, "significant_interactions.csv"), index=False)
    latent_pairs.to_csv(os.path.join(args.outdir, "latent_pairs.tsv"), sep="\t", index=False)
    path_df[["path_bin", "chrom", "genomic_bp", "seg_idx", "strand"]].to_csv(os.path.join(args.outdir, "path_bins.tsv"), sep="\t", index=False)
    np.savetxt(os.path.join(args.outdir, "latent_deconvolved_matrix_symmetric_display.txt"), X, delimiter="\t", fmt="%.10g")

    manifest = {"part2_csv": args.part2_csv,"path_bins": args.path_bins,"latent_matrix": args.latent_matrix, "res": args.res, "n_path_bins": int(len(path_df)), "n_significant": int(len(sig)), "n_latent_pairs": int(len(latent_pairs)),}
    pd.Series(manifest).to_json(os.path.join(args.outdir, "layer3_adapter_manifest.json"), indent=2)

    print("Layer 3-compatible inputs written to:", args.outdir)
    print("  significant_interactions.csv", sig.shape)
    print("  latent_pairs.tsv", latent_pairs.shape)
    print("  path_bins.tsv", path_df.shape)
    print("  latent_deconvolved_matrix_symmetric_display.txt", X.shape)

if __name__ == "__main__":
    main()
