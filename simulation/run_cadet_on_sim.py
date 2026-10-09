"""Run Layer 1 (../scripts/cadet_layer1.py) on one simulated dataset.
"""
import os
import json
import sys
import tempfile
import importlib.util
import types
import numpy as np
import pandas as pd

# layer1_diagnostics imports iced on some systems; simulation does not use it.
if "iced" not in sys.modules:
    sys.modules["iced"] = types.ModuleType("iced")
    sys.modules["iced"].normalization = types.ModuleType("normalization")
    sys.modules["iced"].normalization.ICE_normalization = lambda x: x


def import_cadet(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    path = path or os.path.join(here, "..", "scripts", "cadet_layer1.py")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Layer 1 script not found: {path}")
    script_dir = os.path.dirname(path)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("cadet_layer1", path)
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)
    return cd


def ensure_ref_bin_key(path_df):
    """Rebuild the (chrom, genomic_bp) ref_bin_key tuples after reading path_bins.tsv."""
    df = path_df.copy()
    required = {"chrom", "genomic_bp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"path_df missing required columns: {sorted(missing)}")
    bp = pd.to_numeric(df["genomic_bp"], errors="raise").astype(np.int64)
    df["genomic_bp"] = bp
    df["ref_bin_key"] = list(zip(df["chrom"].astype(str), bp.astype(int)))
    return df


def fit_backbone_exact(cd, Y, path_df, ref_df, res=5000,
                       start_diag=2, trim_upper=0.95, trim_lower=0.01):
    """Backbone fit as in Layer 1, without reading a .hic file."""
    qc, flagged = cd.detect_bad_bins(cd.sym(Y), cd.DEFAULT_MAD_THRESH)
    flagged_set = set(flagged)
    d0, c0 = cd.initial_fit_pairs(
        path_df, ref_df, Y, res,
        start_diag=start_diag,
        trim_upper_q=trim_upper,
        trim_lower_q=trim_lower,
        bad_ref_set=flagged_set,
        exclude_breakpoints=True,
    )
    if len(d0) < 10:
        raise ValueError(f"Only {len(d0)} clean pairs found for initial backbone fit")
    decay = cd.fit_initial_decay(d0, c0)
    return decay, flagged_set, qc


def run_cadet_mode(Y, path_df, mode="cadet_pi_on", res=5000,
                   max_iter=100, tol=1e-6, tau=1.0,
                   alpha_min=0.1, alpha_max=10.0,
                   anchor_window_bins=50, anchor_eps=1e-3,
                   tau_between_copy=10.0, cd=None):
    """Run Layer 1 with pi, without pi, or backbone only (naive).

    Defaults are the Layer 1 command-line defaults.
    """
    if cd is None:
        cd = import_cadet()
    # Skip per-iteration diagnostic plots (no effect on the numbers).
    cd.plot_alpha_dist = lambda *args, **kwargs: None
    if mode not in {"cadet_pi_on", "cadet_pi_off", "naive"}:
        raise ValueError(f"Unknown mode: {mode}")

    path_df = ensure_ref_bin_key(path_df)
    path_df, ref_df, members = cd.build_refbin_index(path_df)
    n_path = int(path_df.shape[0])

    # The simulator stores genomic_bp = simulator_ref_id * resolution. After
    # CADET rebuilds ref_id, both numberings must still agree exactly.
    expected_ref = (pd.to_numeric(path_df["genomic_bp"], errors="raise") // int(res)).astype(int)
    actual_ref = pd.to_numeric(path_df["ref_id"], errors="raise").astype(int)
    if not np.array_equal(expected_ref.to_numpy(), actual_ref.to_numpy()):
        bad = np.flatnonzero(expected_ref.to_numpy() != actual_ref.to_numpy())
        show = bad[:5].tolist()
        raise AssertionError(
            "Simulation reference-family numbering does not match CADET. "
            f"First mismatched path bins: {show}"
        )
    families = cd.build_families(ref_df, members)

    decay, flagged_set, qc = fit_backbone_exact(cd, Y, path_df, ref_df, res=res)
    zero_proxy, zp_info = cd.calibrate_zero_proxy(Y, ref_df)
    if not np.isfinite(zero_proxy) or zero_proxy <= 0:
        raise ValueError(f"Invalid zero_proxy from Layer 1: {zero_proxy}")

    backbone = cd.build_backbone_matrix(n_path, res, decay, zero_proxy)
    same_ref_mask = cd.build_same_ref_mask(path_df)

    has_dup = bool((ref_df["multiplicity"] > 1).any())
    if mode == "cadet_pi_on" and has_dup and anchor_window_bins > 0:
        pi = cd.compute_copy_anchors(
            Y, path_df, backbone, window_bins=anchor_window_bins, eps=anchor_eps
        )
    else:
        pi = None

    alpha0 = cd.init_alpha(n_path)
    lam0 = cd.build_lambda(backbone, alpha0)

    if mode == "naive":
        X_hat, touched = cd.e_step(Y, families, lam0, pi=None)
        alpha_hat = np.ones((n_path, n_path), dtype=float)
        pi_hat = np.ones(n_path, dtype=float)
        lambda_hat = cd.build_effective_lambda(backbone, alpha_hat, pi_hat)
        hist_df = pd.DataFrame()
    else:
        em_pi = pi if mode == "cadet_pi_on" else None
        with tempfile.TemporaryDirectory(prefix="cadet_layer1_sim_em_") as emdir:
            lambda_hat, alpha_hat, X_hat, touched, hist_df = cd.run_em(
                Y, families, n_path, backbone, lam0, alpha0, emdir,
                max_iter=max_iter, tol=tol,
                tau=tau, alpha_min=alpha_min, alpha_max=alpha_max,
                same_ref_mask=same_ref_mask,
                tau_between_copy=tau_between_copy,
                pi=em_pi, verbose=False,
            )
        pi_hat = pi if pi is not None else np.ones(n_path, dtype=float)

    # Consistency checks.
    rebuilt_lambda = cd.build_effective_lambda(backbone, alpha_hat, pi_hat)
    if not np.allclose(lambda_hat, rebuilt_lambda, rtol=1e-10, atol=1e-10):
        raise AssertionError("lambda_hat is inconsistent with pi*pi*alpha*backbone")

    recollapsed = cd.collapse_to_ref(X_hat, path_df, ref_df)
    y_sym = cd.sym(Y)
    recollapse_err = float(np.max(np.abs(recollapsed - y_sym)))
    if recollapse_err > 1e-7:
        raise AssertionError(
            f"Layer 1 recollapse failed: max abs error={recollapse_err:.3e}"
        )

    return {
        "mode": mode,
        "X_hat": X_hat,
        "alpha_hat": alpha_hat,
        "pi_hat": pi_hat,
        "lambda_hat": lambda_hat,
        "backbone": backbone,
        "decay": decay,
        "zero_proxy": zero_proxy,
        "zero_proxy_info": zp_info,
        "n_path": n_path,
        "families": families,
        "ref_df": ref_df,
        "path_df": path_df,
        "same_ref_mask": same_ref_mask,
        "flagged_set": flagged_set,
        "bad_bin_qc": qc,
        "em_history": hist_df,
        "recollapse_err": recollapse_err,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run CADET Layer 1 on one simulated dataset")
    p.add_argument("--simdir", required=True)
    p.add_argument("--mode", default="cadet_pi_on",
                   choices=["cadet_pi_on", "cadet_pi_off", "naive"])
    p.add_argument("--res", type=int, default=None, help="Optional check; must match sim_config.json")
    p.add_argument("--anchor_window_bins", type=int, default=50)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    Y = np.loadtxt(os.path.join(args.simdir, "Y_observed.txt"), delimiter="\t")
    path_df = pd.read_csv(os.path.join(args.simdir, "path_bins.tsv"), sep="\t")
    config_path = os.path.join(args.simdir, "sim_config.json")
    with open(config_path) as fh:
        sim_cfg = json.load(fh)
    sim_res = int(sim_cfg.get("resolution_bp", 5000))
    if args.res is not None and int(args.res) != sim_res:
        raise ValueError(
            f"--res {args.res} does not match simulation resolution {sim_res}. "
            "Use the simulation resolution rather than overriding it."
        )
    result = run_cadet_mode(
        Y, path_df, mode=args.mode, res=sim_res,
        anchor_window_bins=args.anchor_window_bins,
    )
    outdir = args.out or os.path.join(args.simdir, f"layer1_{args.mode}")
    os.makedirs(outdir, exist_ok=True)
    for key in ["X_hat", "alpha_hat", "pi_hat", "lambda_hat", "backbone"]:
        np.savetxt(os.path.join(outdir, f"{key}.txt"), result[key], delimiter="\t")
    result["em_history"].to_csv(os.path.join(outdir, "em_history.tsv"), sep="\t", index=False)
    print(f"Wrote {outdir}")
    print(f"recollapse_err={result['recollapse_err']:.3e}")
