#!/usr/bin/env python3
"""
CADET layer 8: target recurrence checks
This is the last downstream check for the Layer 6/7 module calls. It mostly makes the cleaned module table, evidence summaries,
and a simple target-label shuffle for recurrence sanity checks.

Note: the empirical p-values are useful for ranking/checking repeated targets,
but with small module counts they should not be read as biological validation.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None


#  figure settings
PUB_FONTSIZE = 22
PUB_TICK_FONTSIZE = 22
PUB_LEGEND_FONTSIZE = 18
PUB_DPI = 300
PUB_FIGSIZE_RECT = (12, 6)
PUB_FIGSIZE_SQUARE = (10, 10)

# constants / display settings

EVIDENCE_ORDER = ["strict_enhancer_oncogene",
    "permissive_enhancer_oncogene_context",
    "enhancer_gene_context",
    "promoter_disruption_context",
    "local_regulatory_context",
    "unresolved"]

EVIDENCE_DISPLAY = {
    "strict_enhancer_oncogene": "Strict enhancer–oncogene",
    "permissive_enhancer_oncogene_context": "Permissive enhancer–oncogene context",
    "enhancer_gene_context": "Enhancer–gene context",
    "promoter_disruption_context": "Promoter-disruption context",
    "local_regulatory_context": "Local regulatory context",
    "unresolved": "Unresolved/context only"}


EVIDENCE_COLORS = {"strict_enhancer_oncogene": "tab:red",
    "permissive_enhancer_oncogene_context": "tab:purple",
    "enhancer_gene_context": "tab:blue",
    "promoter_disruption_context": "tab:orange",
    "local_regulatory_context": "tab:gray",
    "unresolved": "lightgray"}


# small helpers

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s if s and s.lower() not in {"nan", "none", "null"} else np.nan


def is_readable_target(x) -> bool:
    x = normalize_text(x)
    if pd.isna(x):
        return False
    return not str(x).startswith("ENSG")


def first_existing(df: pd.DataFrame, names: List[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def numeric_series(series: pd.Series, default=np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def bh_fdr(pvals) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if finite.sum() == 0:
        return out
    pv = pvals[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / (np.arange(1, len(ranked) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[finite] = restored
    return out


def cliffs_delta(x, y):
    x = np.asarray(pd.Series(x).dropna().astype(float).values, dtype=float)
    y = np.asarray(pd.Series(y).dropna().astype(float).values, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


def empirical_pvalue(obs, null_vals, greater=True) -> float:
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if len(null_vals) == 0 or not np.isfinite(obs):
        return np.nan
    if greater:
        return (1.0 + np.sum(null_vals >= obs)) / (len(null_vals) + 1.0)
    return (1.0 + np.sum(null_vals <= obs)) / (len(null_vals) + 1.0)


def percentile_rank(values, higher_is_better=True, neutral=0.5) -> pd.Series:
    x = pd.Series(values, dtype="float64")
    finite = np.isfinite(x.values)
    out = pd.Series(neutral, index=x.index, dtype="float64")
    if finite.sum() == 0:
        return out
    ranks = x.loc[finite].rank(method="average", pct=True, ascending=True)
    out.loc[finite] = ranks if higher_is_better else 1.0 - ranks
    return out


def split_semicolon_values(x) -> List[str]:
    x = normalize_text(x)
    if pd.isna(x):
        return []
    vals = []
    for part in str(x).replace(",", ";").split(";"):
        part = part.strip()
        if part and part.lower() not in {"nan", "none"}:
            vals.append(part)
    return sorted(set(vals))



def savefig(fig, png_path: str, pdf_path: str = None) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(png_path, dpi=PUB_DPI, bbox_inches="tight")
    if pdf_path:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# evidence class handling

def canonical_evidence_type(x, recommendation=None) -> str:
    s = normalize_text(x)
    r = normalize_text(recommendation)
    text = " ".join([str(v) for v in [s, r] if pd.notna(v)]).lower()

    if "strict_enhancer_oncogene" in text:
        return "strict_enhancer_oncogene"
    if "high_priority_candidate_enhancer_oncogene" in text:
        return "strict_enhancer_oncogene"
    if "high_confidence_candidate_enhancer_oncogene" in text:
        return "strict_enhancer_oncogene"
    if "permissive_enhancer_oncogene" in text:
        return "permissive_enhancer_oncogene_context"
    if "regulatory_support_candidate" in text and "myc" in text:
        return "permissive_enhancer_oncogene_context"
    if "enhancer_gene" in text:
        return "enhancer_gene_context"
    if "promoter_disruption" in text:
        return "promoter_disruption_context"
    if "local_regulatory" in text or "general_functional_context" in text or "regulatory_support" in text:
        return "local_regulatory_context"
    if "weak" in text or "unresolved" in text:
        return "unresolved"
    return "unresolved"


# loaders

def load_layer6_modules(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing Layer 6 prioritized module file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "module_id" not in df.columns:
        raise ValueError("Layer 6 prioritized file must contain module_id")

    out = df.copy()
    out["module_id"] = numeric_series(out["module_id"], -1).astype(int)

    if "func_rank" in out.columns:
        out["func_rank"] = numeric_series(out["func_rank"], np.nan)
    else:
        out = out.sort_values("combined_priority" if "combined_priority" in out.columns else "module_id", ascending=False)
        out["func_rank"] = np.arange(1, len(out) + 1, dtype=int)

    # accept both old and current Layer 6 column names
    out["combined_priority"] = numeric_series(
        first_existing(out, ["combined_priority", "priority"], 0.0), 0.0
    )
    out["func_priority"] = numeric_series(
        first_existing(out, ["func_priority"], 0.0), 0.0
    )
    out["priority"] = numeric_series(first_existing(out, ["priority"], out["combined_priority"]), 0.0)
    out["func_recommendation"] = first_existing(out, ["func_recommendation"], "unresolved").astype(str)
    out["module_label"] = first_existing(out, ["module_label", "dominant_contact_label", "module_label_disp"], "unclassified").astype(str)
    out["module_type_disp"] = first_existing(out, ["module_type_disp", "grouping_type_disp"], "module").astype(str)
    out["module_label_disp"] = first_existing(out, ["module_label_disp", "dominant_label_disp"], out["module_label"]).astype(str)

    # logic counts and fractions
    out["n_pair_enh_onco_strict"] = numeric_series(
        first_existing(out, ["n_pair_enh_onco", "n_pair_enh_onco_strict"], 0), 0
    )
    out["n_pair_enh_onco_perm"] = numeric_series(
        first_existing(out, ["n_pair_enh_onco_perm"], 0), 0
    )
    out["n_pair_enh_gene"] = numeric_series(first_existing(out, ["n_pair_enh_gene"], 0), 0)
    out["n_pair_prom_disrupt"] = numeric_series(first_existing(out, ["n_pair_prom_disrupt"], 0), 0)
    out["n_member_blk_func"] = numeric_series(
        first_existing(out, ["n_member_blk_func", "n_member_blk", "n_supporting_blocks"], 1), 1
    ).clip(lower=1)

    out["adj_frac_enh_onco_strict"] = out["n_pair_enh_onco_strict"] / out["n_member_blk_func"]
    out["adj_frac_enh_onco_perm"] = out["n_pair_enh_onco_perm"] / out["n_member_blk_func"]
    out["adj_frac_enh_gene"] = out["n_pair_enh_gene"] / out["n_member_blk_func"]
    out["adj_frac_prom_disrupt"] = out["n_pair_prom_disrupt"] / out["n_member_blk_func"]

    # old downstream scripts still look for this name
    out["adj_frac_enh_onco"] = out["adj_frac_enh_onco_strict"]

    out["max_discord_func"] = numeric_series(
        first_existing(out, ["max_discord_func", "max_discord"], 0.0), 0.0
    ).clip(lower=0.0)
    out["max_obs_over_exp_raw_func"] = numeric_series(
        first_existing(out, ["max_obs_over_exp_raw_func", "max_obs_over_exp_raw"], np.nan), np.nan
    )
    out["n_uniq_cand_oncos"] = numeric_series(
        first_existing(out, ["n_uniq_cand_oncos"], 0), 0
    )
    out["module_oncos"] = first_existing(
        out, ["module_oncos", "module_oncos_clean"], ""
    ).astype(str)

    return out


def load_layer7_summary(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing Layer 7 module summary file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "module_id" not in df.columns:
        raise ValueError("Layer 7 module summary must contain module_id")

    out = df.copy()
    out["module_id"] = numeric_series(out["module_id"], -1).astype(int)
    out["main_evidence_type"] = [
        canonical_evidence_type(e, r)
        for e, r in zip(
            first_existing(out, ["main_evidence_type"], "unresolved"),
            first_existing(out, ["func_recommendation"], "")
        )
    ]
    out["final_target_name"] = first_existing(out, ["final_target_name", "final_target", "top_onco"], np.nan).map(normalize_text)
    out["func_rank"] = numeric_series(first_existing(out, ["func_rank"], np.nan), np.nan)
    out["func_recommendation"] = first_existing(out, ["func_recommendation"], "unresolved").astype(str)
    out["combined_priority"] = numeric_series(first_existing(out, ["combined_priority", "priority"], 0.0), 0.0)
    out["top_onco"] = first_existing(out, ["top_onco"], np.nan).map(normalize_text)
    out["top_onco_perm"] = first_existing(out, ["top_onco_perm"], np.nan).map(normalize_text)
    out["top_gene"] = first_existing(out, ["top_gene"], np.nan).map(normalize_text)
    out["module_oncos"] = first_existing(
        out, ["module_oncos", "module_oncos_clean"], ""
    ).astype(str)

    return out


def build_module_table(layer6_df: pd.DataFrame, layer7_df: pd.DataFrame) -> pd.DataFrame:
    l6 = layer6_df.copy()
    l7_cols = [
        "module_id", "main_evidence_type", "final_target_name", "func_rank",
        "func_recommendation", "top_onco", "top_onco_perm",
        "top_gene", "module_oncos", "combined_priority"
    ]
    l7_cols = [c for c in l7_cols if c in layer7_df.columns]
    l7 = layer7_df[l7_cols].copy()

    # keep Layer 7 interpretation as the final call when names overlap
    l6 = l6.drop(columns=[c for c in l7_cols if c in l6.columns and c != "module_id"], errors="ignore")
    out = l6.merge(l7, on="module_id", how="left")

    out["main_evidence_type"] = [
        canonical_evidence_type(e, r)
        for e, r in zip(
            first_existing(out, ["main_evidence_type"], "unresolved"),
            first_existing(out, ["func_recommendation"], "")
        )
    ]
    out["main_evidence_type_disp"] = out["main_evidence_type"].map(EVIDENCE_DISPLAY).fillna(out["main_evidence_type"])
    out["final_target_name"] = first_existing(out, ["final_target_name", "top_onco", "top_onco_perm", "top_gene"], np.nan).map(normalize_text)

    # evidence composition from the earlier logic counts
    out["has_strict_enh_onco"] = out["n_pair_enh_onco_strict"] > 0
    out["has_perm_enh_onco_ctx"] = out["n_pair_enh_onco_perm"] > 0
    out["has_enh_gene_ctx"] = out["n_pair_enh_gene"] > 0
    out["has_prom_disrupt_ctx"] = out["n_pair_prom_disrupt"] > 0

    # descriptive rank aggregate
    rank_components = { "rank_combined_priority": percentile_rank(out["combined_priority"]),
        "rank_func_priority": percentile_rank(out["func_priority"]),
        "rank_strict_frac": percentile_rank(out["adj_frac_enh_onco_strict"]),
        "rank_perm_frac": percentile_rank(out["adj_frac_enh_onco_perm"]),
        "rank_enh_gene_frac": percentile_rank(out["adj_frac_enh_gene"]),
        "rank_prom_disrupt_frac": percentile_rank(out["adj_frac_prom_disrupt"]),
        "rank_positive_discord": percentile_rank(out["max_discord_func"]),
        "rank_module_supp": percentile_rank(np.log2(np.maximum(out["n_member_blk_func"], 1)))}
    for k, v in rank_components.items():
        out[k] = v

    out["rank_agg_priority"] = pd.DataFrame(rank_components).mean(axis=1)
    out["rewiring_strength"] = out["rank_agg_priority"]

    # compact evidence string for tables
    comps = []
    for _, row in out.iterrows():
        parts = []
        if row["has_strict_enh_onco"]:
            parts.append("strict_enh_onco")
        if row["has_perm_enh_onco_ctx"]:
            parts.append("permissive_enh_onco")
        if row["has_enh_gene_ctx"]:
            parts.append("enh_gene")
        if row["has_prom_disrupt_ctx"]:
            parts.append("promoter_disruption")
        if not parts:
            parts.append("local_context")
        comps.append(";".join(parts))
    out["evidence_composition"] = comps

    return out.sort_values("func_rank", na_position="last").reset_index(drop=True)


# evidence summaries and descriptive tests

def compute_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ev, grp in df.groupby("main_evidence_type", sort=False):
        rew = pd.to_numeric(grp["rewiring_strength"], errors="coerce").dropna()
        disc = pd.to_numeric(grp["max_discord_func"], errors="coerce").dropna()
        rows.append({"main_evidence_type": ev, "main_evidence_type_disp": EVIDENCE_DISPLAY.get(ev, ev),
            "n_modules": int(len(grp)), "n_strict_enh_onco_mods": int((grp["n_pair_enh_onco_strict"] > 0).sum()),
            "n_perm_enh_onco_mods": int((grp["n_pair_enh_onco_perm"] > 0).sum()),
            "n_enh_gene_mods": int((grp["n_pair_enh_gene"] > 0).sum()),
            "n_prom_disrupt_mods": int((grp["n_pair_prom_disrupt"] > 0).sum()),
            "rewiring_mean": float(rew.mean()) if len(rew) else np.nan,
            "rewiring_median": float(rew.median()) if len(rew) else np.nan,
            "rewiring_sd": float(rew.std(ddof=1)) if len(rew) > 1 else np.nan,
            "discord_mean": float(disc.mean()) if len(disc) else np.nan,
            "discord_median": float(disc.median()) if len(disc) else np.nan,
            "discord_sd": float(disc.std(ddof=1)) if len(disc) > 1 else np.nan,
            "targets": ";".join(sorted(set([str(x) for x in grp["final_target_name"].dropna()])))})
    return pd.DataFrame(rows).sort_values(["n_modules", "rewiring_median"], ascending=[False, False]).reset_index(drop=True)


def compute_pairwise_stats(df: pd.DataFrame, value_col: str, min_n: int = 2) -> pd.DataFrame:
    counts = df["main_evidence_type"].value_counts().to_dict()
    classes = [ev for ev in EVIDENCE_ORDER if counts.get(ev, 0) >= min_n]
    rows = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = classes[i], classes[j]
            xa = pd.to_numeric(df.loc[df["main_evidence_type"] == a, value_col], errors="coerce").dropna().values
            xb = pd.to_numeric(df.loc[df["main_evidence_type"] == b, value_col], errors="coerce").dropna().values
            if len(xa) < min_n or len(xb) < min_n or mannwhitneyu is None:
                p = np.nan
                u = np.nan
                delta = np.nan
            else:
                try:
                    u, p = mannwhitneyu(xa, xb, alternative="two-sided")
                except Exception:
                    u, p = np.nan, np.nan
                delta = cliffs_delta(xa, xb)
            rows.append({"group_a": a,"group_b": b,
                "group_a_disp": EVIDENCE_DISPLAY.get(a, a),
                "group_b_disp": EVIDENCE_DISPLAY.get(b, b),
                "value_col": value_col,"n_a": int(len(xa)),"n_b": int(len(xb)),
                "mean_a": float(np.mean(xa)) if len(xa) else np.nan,
                "mean_b": float(np.mean(xb)) if len(xb) else np.nan,
                "median_a": float(np.median(xa)) if len(xa) else np.nan,
                "median_b": float(np.median(xb)) if len(xb) else np.nan,
                "u_stat": u,"p_value": p,"cliffs_delta": delta,
                "interpretation_note": "Descriptive only; small module counts make formal p-values unstable."})
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_bh"] = bh_fdr(out["p_value"].values)
    return out


# target recurrence and nulls

def observed_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    use = df.copy()
    use["final_target_name"] = use["final_target_name"].map(normalize_text)
    use = use.dropna(subset=["final_target_name"]).copy()
    if len(use) == 0:
        return pd.DataFrame()

    def count_if(series, target):
        return int((series == target).sum())

    rows = []
    for target, grp in use.groupby("final_target_name", dropna=True):
        evidence_types = sorted(set(grp["main_evidence_type"].astype(str)))
        rows.append({"target_name": target,
            "is_readable_target": is_readable_target(target),
            "n_modules": int(grp["module_id"].nunique()),
            "weighted_target": float(grp["rewiring_strength"].sum()),
            "best_rank": int(pd.to_numeric(grp["func_rank"], errors="coerce").min()),
            "module_ids": ";".join(map(str, sorted(set(grp["module_id"].astype(int))))),
            "evidence_types": ";".join(evidence_types),
            "n_strict_enh_onco_mods": count_if(grp["main_evidence_type"], "strict_enhancer_oncogene"),
            "n_perm_ctx_mods": count_if(grp["main_evidence_type"], "permissive_enhancer_oncogene_context"),
            "n_enh_gene_mods": count_if(grp["main_evidence_type"], "enhancer_gene_context"),
            "n_prom_disrupt_mods": count_if(grp["main_evidence_type"], "promoter_disruption_context"),
            "n_local_ctx_mods": count_if(grp["main_evidence_type"], "local_regulatory_context"),
            "sum_strict_enh_onco_blk": int(pd.to_numeric(grp["n_pair_enh_onco_strict"], errors="coerce").fillna(0).sum()),
            "sum_perm_enh_onco_blk": int(pd.to_numeric(grp["n_pair_enh_onco_perm"], errors="coerce").fillna(0).sum()),
            "sum_enh_gene_blk": int(pd.to_numeric(grp["n_pair_enh_gene"], errors="coerce").fillna(0).sum())})
    out = pd.DataFrame(rows)
    return out.sort_values(["weighted_target", "n_modules", "best_rank"], ascending=[False, False, True]).reset_index(drop=True)


def _target_score_table(tmp: pd.DataFrame, target_col: str) -> pd.DataFrame:
    use = tmp.dropna(subset=[target_col]).copy()
    if len(use) == 0:
        return pd.DataFrame(columns=["target_name", "weighted_target"])
    out = (
        use.groupby(target_col, dropna=True)["rewiring_strength"]
        .sum()
        .reset_index()
        .rename(columns={target_col: "target_name", "rewiring_strength": "weighted_target"})
    )
    # downstream code expects these columns even when no targets are found
    if "weighted_target" not in out.columns:
        out["weighted_target"] = 0.0
    return out[["target_name", "weighted_target"]]


def module_null_once(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    tmp = df.copy()
    tmp["final_target_name"] = tmp["final_target_name"].map(normalize_text)
    tmp["shuffled_target"] = tmp["final_target_name"].copy()
    for ev, grp in tmp.groupby("main_evidence_type", sort=False):
        vals = grp["final_target_name"].values.copy().astype(object)
        rng.shuffle(vals)
        tmp.loc[grp.index, "shuffled_target"] = vals
    return _target_score_table(tmp, "shuffled_target")


def global_null_once(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    tmp = df.copy()
    vals = tmp["final_target_name"].map(normalize_text).values.copy().astype(object)
    rng.shuffle(vals)
    tmp["shuffled_target"] = vals
    return _target_score_table(tmp, "shuffled_target")


def run_null_simulations(df: pd.DataFrame, n_iter=1000, seed=7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mod_rows = []
    glob_rows = []
    for it in range(n_iter):
        m = module_null_once(df, rng)
        m["iteration"] = it + 1
        mod_rows.append(m)
        g = global_null_once(df, rng)
        g["iteration"] = it + 1
        glob_rows.append(g)
    module_aware_null = pd.concat(mod_rows, ignore_index=True) if mod_rows else pd.DataFrame()
    global_shuffle_null = pd.concat(glob_rows, ignore_index=True) if glob_rows else pd.DataFrame()
    return module_aware_null, global_shuffle_null


def _ensure_score_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Build a target-score table with stable columns."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["target_name", "weighted_target"])
    out = df.copy()
    if "target_name" not in out.columns:
        out["target_name"] = np.nan
    if "weighted_target" not in out.columns:
        if "rewiring_strength" in out.columns:
            out["weighted_target"] = pd.to_numeric(out["rewiring_strength"], errors="coerce").fillna(0.0)
        else:
            out["weighted_target"] = 0.0
    out["weighted_target"] = pd.to_numeric(out["weighted_target"], errors="coerce").fillna(0.0)
    return out


def summarize_targets(observed_df: pd.DataFrame, module_null_df: pd.DataFrame, global_null_df: pd.DataFrame) -> pd.DataFrame:
    module_null_df = _ensure_score_cols(module_null_df)
    global_null_df = _ensure_score_cols(global_null_df)
    if len(observed_df) == 0:
        return pd.DataFrame()
    rows = []
    for _, obs_row in observed_df.iterrows():
        target = obs_row["target_name"]
        obs = float(obs_row["weighted_target"])
        mod_null = module_null_df.loc[module_null_df["target_name"] == target, "weighted_target"].values if len(module_null_df) else np.array([])
        glob_null = global_null_df.loc[global_null_df["target_name"] == target, "weighted_target"].values if len(global_null_df) else np.array([])
        rows.append({
            **obs_row.to_dict(),
            "modaware_null_mean": float(np.mean(mod_null)) if len(mod_null) else np.nan,
            "modaware_null_sd": float(np.std(mod_null, ddof=1)) if len(mod_null) > 1 else np.nan,
            "shuffle_null_mean": float(np.mean(glob_null)) if len(glob_null) else np.nan,
            "shuffle_null_sd": float(np.std(glob_null, ddof=1)) if len(glob_null) > 1 else np.nan,
            "p_emp_modaware": empirical_pvalue(obs, mod_null, greater=True),
            "p_emp_shuffle": empirical_pvalue(obs, glob_null, greater=True),
        })
    out = pd.DataFrame(rows)
    out["fdr_modaware"] = bh_fdr(out["p_emp_modaware"].values)
    out["fdr_shuffle"] = bh_fdr(out["p_emp_shuffle"].values)
    out["delta_vs_modaware_mean"] = out["weighted_target"] - out["modaware_null_mean"]
    out["delta_vs_shuffle_mean"] = out["weighted_target"] - out["shuffle_null_mean"]
    out["recur_supp_note"] = np.where(
        out["n_strict_enh_onco_mods"] > 0,
        "contains_strict_enhancer_oncogene_module",
        np.where(out["n_perm_ctx_mods"] > 0, "permissive_or_context_recurrence", "context_only")
    )
    return out.sort_values(["weighted_target", "n_modules", "best_rank"], ascending=[False, False, True]).reset_index(drop=True)


def build_candidate_tables(target_validation_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(target_validation_df) == 0:
        return target_validation_df.copy(), target_validation_df.copy()
    # avoid over-calling; this is a ranked candidate list, not validation
    cand = target_validation_df[
        (target_validation_df["n_modules"] >= 1) &
        (
            (pd.to_numeric(target_validation_df["fdr_modaware"], errors="coerce") <= 0.25) |
            (pd.to_numeric(target_validation_df["fdr_shuffle"], errors="coerce") <= 0.25) |
            (target_validation_df["n_strict_enh_onco_mods"] > 0)
        )
    ].copy()
    readable = cand[cand["is_readable_target"]].copy()
    return cand, readable


# plots

def evidence_order_present(df: pd.DataFrame) -> List[str]:
    present = set(df["main_evidence_type"].astype(str))
    return [e for e in EVIDENCE_ORDER if e in present]

def plot_rewiring_vs_disc(df: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    for ev in evidence_order_present(df):
        grp = df[df["main_evidence_type"] == ev]
        ax.scatter(grp["max_discord_func"],grp["rewiring_strength"],
            s=70, alpha=0.85, edgecolors="black", linewidths=0.4,
            color=EVIDENCE_COLORS.get(ev, "gray"),
            label=EVIDENCE_DISPLAY.get(ev, ev))
    ax.set_xlabel("Max positive distance-discordance score", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Rank-aggregate module evidence score", fontsize=PUB_FONTSIZE)
    ax.set_title("Module evidence score vs structural discordance", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE)
    savefig(fig, out_png, out_pdf)

def plot_box_by_evidence(df: pd.DataFrame, value_col: str, ylabel: str, title: str, out_png: str, out_pdf: str) -> None:
    order = evidence_order_present(df)
    if not order:
        return
    groups = [pd.to_numeric(df.loc[df["main_evidence_type"] == ev, value_col], errors="coerce").dropna().values for ev in order]
    labels = [EVIDENCE_DISPLAY.get(ev, ev) for ev in order]
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False, vert=False)
    for patch, ev in zip(bp["boxes"], order):
        patch.set_facecolor(EVIDENCE_COLORS.get(ev, "gray"))
        patch.set_alpha(0.55)
    rng = np.random.default_rng(13)
    for i, ev in enumerate(order, start=1):
        vals = pd.to_numeric(df.loc[df["main_evidence_type"] == ev, value_col], errors="coerce").dropna().values
        if len(vals):
            ax.scatter(vals, rng.normal(i, 0.045, size=len(vals)), s=28, alpha=0.75,
                       color=EVIDENCE_COLORS.get(ev, "gray"), edgecolors="black", linewidths=0.25)
    ax.set_xlabel(ylabel, fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Evidence type", fontsize=PUB_FONTSIZE)
    ax.set_title(title, fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    savefig(fig, out_png, out_pdf)

def plot_recurrence_vs_nulls(target_validation_df: pd.DataFrame, out_png: str, out_pdf: str, top_n=20) -> None:
    df = target_validation_df.head(top_n).copy()
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, 0.55 * len(df) + 2)))
    y = np.arange(len(df))
    h = 0.24
    ax.barh(y + h, df["weighted_target"], height=h, label="Observed")
    ax.barh(y, df["modaware_null_mean"].fillna(0), height=h, label="Within-evidence shuffle")
    ax.barh(y - h, df["shuffle_null_mean"].fillna(0), height=h, label="Global shuffle")
    ax.set_yticks(y)
    ax.set_yticklabels(df["target_name"], fontsize=PUB_TICK_FONTSIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Summed module evidence score", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Final target/context gene", fontsize=PUB_FONTSIZE)
    ax.set_title(f"Observed target recurrence vs shuffle nulls (top {len(df)})", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE)
    savefig(fig, out_png, out_pdf)

def plot_distance_ecdf(df: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    for ev in evidence_order_present(df):
        vals = pd.to_numeric(df.loc[df["main_evidence_type"] == ev, "max_discord_func"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", linewidth=2.3, color=EVIDENCE_COLORS.get(ev, "gray"), label=EVIDENCE_DISPLAY.get(ev, ev))
    ax.set_xlabel("Max positive distance-discordance score", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("ECDF", fontsize=PUB_FONTSIZE)
    ax.set_title("Distance-discordance distribution by evidence type", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE)
    savefig(fig, out_png, out_pdf)

def plot_evidence_bar(df: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    count_cols = [
        ("n_pair_enh_onco_strict", "Strict enh–onco"),
        ("n_pair_enh_onco_perm", "Permissive enh–onco"),
        ("n_pair_enh_gene", "Enh–gene"),
        ("n_pair_prom_disrupt", "Promoter disruption"),
    ]
    use = df.sort_values("func_rank").copy()
    fig, ax = plt.subplots(figsize=(12, max(6, 0.48 * len(use) + 2)))
    y = np.arange(len(use))
    left = np.zeros(len(use))
    for col, label in count_cols:
        vals = pd.to_numeric(use[col], errors="coerce").fillna(0).values
        ax.barh(y, vals, left=left, label=label)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"M{m}" for m in use["module_id"]], fontsize=PUB_TICK_FONTSIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Number of supporting blocks", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Module", fontsize=PUB_FONTSIZE)
    ax.set_title("Evidence composition by module", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig, out_png, out_pdf)

def plot_validation_panel(df: pd.DataFrame, target_validation_df: pd.DataFrame, out_png: str, out_pdf: str, top_n_targets=10) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.ravel()
    panel_fs = 18
    panel_tick_fs = 16
    panel_legend_fs = 14

    for ev in evidence_order_present(df):
        grp = df[df["main_evidence_type"] == ev]
        ax1.scatter(grp["max_discord_func"], grp["rewiring_strength"], s=45, alpha=0.8, color=EVIDENCE_COLORS.get(ev, "gray"),
                    label=EVIDENCE_DISPLAY.get(ev, ev))
    ax1.set_xlabel("Max positive distance-discordance score", fontsize=panel_fs)
    ax1.set_ylabel("Rank-aggregate module evidence score", fontsize=panel_fs)
    ax1.set_title("A. Module evidence vs discordance", fontsize=panel_fs)
    ax1.tick_params(axis="both", labelsize=panel_tick_fs)

    order = evidence_order_present(df)
    groups = [pd.to_numeric(df.loc[df["main_evidence_type"] == ev, "rewiring_strength"], errors="coerce").dropna().values for ev in order]
    if groups:
        labels = [EVIDENCE_DISPLAY.get(ev, ev) for ev in order]
        bp = ax2.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=False, vert=False)
        for patch, ev in zip(bp["boxes"], order):
            patch.set_facecolor(EVIDENCE_COLORS.get(ev, "gray"))
            patch.set_alpha(0.55)
    ax2.set_xlabel("Rank-aggregate module evidence score", fontsize=panel_fs)
    ax2.set_ylabel("Evidence type", fontsize=panel_fs)
    ax2.set_title("B. Evidence score by class", fontsize=panel_fs)
    ax2.tick_params(axis="both", labelsize=panel_tick_fs)

    for ev in order:
        vals = pd.to_numeric(df.loc[df["main_evidence_type"] == ev, "max_discord_func"], errors="coerce").dropna().values
        if len(vals):
            xs = np.sort(vals)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax3.step(xs, ys, where="post", linewidth=2.2, color=EVIDENCE_COLORS.get(ev, "gray"),
                     label=EVIDENCE_DISPLAY.get(ev, ev))
    ax3.set_xlabel("Max positive distance-discordance score", fontsize=panel_fs)
    ax3.set_ylabel("ECDF", fontsize=panel_fs)
    ax3.set_title("C. Distance-discordance ECDF", fontsize=panel_fs)
    ax3.tick_params(axis="both", labelsize=panel_tick_fs)
    ax3.legend(frameon=False, fontsize=panel_legend_fs)

    top = target_validation_df.head(top_n_targets).copy()
    y = np.arange(len(top))
    h = 0.24
    if len(top):
        ax4.barh(y + h, top["weighted_target"], height=h, label="Observed")
        ax4.barh(y, top["modaware_null_mean"].fillna(0), height=h, label="Within-evidence")
        ax4.barh(y - h, top["shuffle_null_mean"].fillna(0), height=h, label="Global")
        ax4.set_yticks(y)
        ax4.set_yticklabels(top["target_name"])
        ax4.invert_yaxis()
    ax4.set_xlabel("Summed module evidence score", fontsize=panel_fs)
    ax4.set_ylabel("Target/context gene", fontsize=panel_fs)
    ax4.set_title("D. Target recurrence vs nulls", fontsize=panel_fs)
    ax4.tick_params(axis="both", labelsize=panel_tick_fs)
    ax4.legend(frameon=False, fontsize=panel_legend_fs)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)),
                   frameon=False, fontsize=panel_legend_fs)
    fig.tight_layout(rect=[0, 0, 1, 0.94], pad=1.5)
    fig.savefig(out_png, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(description="CADET Layer 8: target recurrence")
    ap.add_argument("--input_dir", required=True, help="cadet_final_results folder")
    ap.add_argument("--n_iter", type=int, default=1000, help="Shuffles for the null")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--top_n_targets", type=int, default=20, help="Targets in the plot")
    ap.add_argument("--pairwise_min_n", type=int, default=2, help="Min modules per group for pairwise tests")
    args = ap.parse_args()

    input_dir = args.input_dir
    layer6_dir = os.path.join(input_dir, "layer6_functional_annotation")
    layer7_dir = os.path.join(input_dir, "layer7_module_interpretation")
    layer6_csv = os.path.join(layer6_dir, "prioritized_functional_modules.csv")
    layer7_csv = os.path.join(layer7_dir, "layer7_module_summary_clean_report.csv")

    outdir = os.path.join(input_dir, "layer8_validation_publication")
    plot_dir = os.path.join(outdir, "plots")
    table_dir = os.path.join(outdir, "tables")
    ensure_dir(outdir)
    ensure_dir(plot_dir)
    ensure_dir(table_dir)

    print("Loading Layer 6/7 files ...")
    layer6_df = load_layer6_modules(layer6_csv)
    layer7_df = load_layer7_summary(layer7_csv)

    print("Building Layer 8 validated/interpreted module table ...")
    validated_df = build_module_table(layer6_df, layer7_df)

    print("Computing observed target recurrence ...")
    observed_targets_df = observed_recurrence(validated_df)

    print("Running target-label null simulations ...")
    module_nulls, global_nulls = run_null_simulations(validated_df, n_iter=args.n_iter, seed=args.seed)

    print("Summarizing target recurrence support ...")
    target_validation_df = summarize_targets(observed_targets_df, module_nulls, global_nulls)
    cand_all, cand_readable = build_candidate_tables(target_validation_df)

    print("Computing evidence-class summaries ...")
    group_summary_df = compute_group_summary(validated_df)
    rewiring_stats_df = compute_pairwise_stats(validated_df, "rewiring_strength", min_n=args.pairwise_min_n)
    discordance_stats_df = compute_pairwise_stats(validated_df, "max_discord_func", min_n=args.pairwise_min_n)

    print("Writing tables ...")
    # layer-6 columns not used by layer 8 stay in prioritized_functional_modules.csv
    layer6_keep = ["module_id", "module_type_disp", "module_label", "module_label_disp", "n_member_blk", "max_obs_over_exp_raw",
        "max_discord", "priority", "n_member_blk_func", "n_uniq_cand_oncos", "n_pair_enh_gene", "n_pair_enh_onco",
        "n_pair_enh_onco_perm", "n_pair_prom_disrupt", "max_obs_over_exp_raw_func", "max_discord_func", "func_priority",
        "func_rank", "func_recommendation", "module_oncos", "combined_priority", "n_pair_enh_onco_strict",
        "adj_frac_enh_onco_strict", "adj_frac_enh_onco_perm", "adj_frac_enh_gene", "adj_frac_prom_disrupt",
        "adj_frac_enh_onco"]
    out_cols = [c for c in validated_df.columns if c not in layer6_df.columns or c in layer6_keep]
    validated_df[out_cols].to_csv(os.path.join(table_dir, "validated_modules_publication.csv"), index=False)
    observed_targets_df.to_csv(os.path.join(table_dir, "observed_target_recurrence_publication.csv"), index=False)
    module_nulls.to_csv(os.path.join(table_dir, "module_aware_null_target_scores_publication.csv"), index=False)
    global_nulls.to_csv(os.path.join(table_dir, "global_shuffle_null_target_scores_publication.csv"), index=False)
    target_validation_df.to_csv(os.path.join(table_dir, "target_recurrence_validation_publication.csv"), index=False)
    cand_all.to_csv(os.path.join(table_dir, "candidate_recurrent_targets_all.csv"), index=False)
    cand_readable.to_csv(os.path.join(table_dir, "candidate_recurrent_targets_readable.csv"), index=False)
    group_summary_df.to_csv(os.path.join(table_dir, "evidence_group_summary_publication.csv"), index=False)
    rewiring_stats_df.to_csv(os.path.join(table_dir, "pairwise_module_evidence_stats_publication.csv"), index=False)
    discordance_stats_df.to_csv(os.path.join(table_dir, "pairwise_discordance_stats_publication.csv"), index=False)

    print("Making plots ...")
    plot_rewiring_vs_disc(validated_df,os.path.join(plot_dir, "module_evidence_vs_discordance_publication.png"),os.path.join(plot_dir, "module_evidence_vs_discordance_publication.pdf"),
    )
    plot_box_by_evidence(validated_df,"rewiring_strength","Rank-aggregate module evidence score","Module evidence score by evidence type",
    os.path.join(plot_dir, "module_evidence_score_by_class_publication.png"),os.path.join(plot_dir, "module_evidence_score_by_class_publication.pdf"))
    plot_box_by_evidence(validated_df,"max_discord_func","Max positive distance-discordance score","Structural discordance by evidence type",
        os.path.join(plot_dir, "discordance_by_evidence_class_publication.png"),os.path.join(plot_dir, "discordance_by_evidence_class_publication.pdf"))
    plot_recurrence_vs_nulls(target_validation_df,
        os.path.join(plot_dir, "target_recurrence_vs_nulls_publication.png"),os.path.join(plot_dir, "target_recurrence_vs_nulls_publication.pdf"),top_n=args.top_n_targets)
    plot_distance_ecdf(validated_df,os.path.join(plot_dir, "distance_discordance_ecdf_publication.png"),os.path.join(plot_dir, "distance_discordance_ecdf_publication.pdf"))
    plot_evidence_bar(validated_df,os.path.join(plot_dir, "module_evidence_composition_publication.png"),os.path.join(plot_dir, "module_evidence_composition_publication.pdf"))
    plot_validation_panel(validated_df,target_validation_df,os.path.join(plot_dir, "validation_panel_publication.png"),
        os.path.join(plot_dir, "validation_panel_publication.pdf"),top_n_targets=min(10, args.top_n_targets))

    params = {"layer8_version": "publication_validation_target_recurrence",
        "input_dir": input_dir,
        "layer6_csv": layer6_csv,
        "layer7_csv": layer7_csv,
        "n_iter": args.n_iter,
        "seed": args.seed,
        "top_n_targets": args.top_n_targets,
        "pairwise_min_n": args.pairwise_min_n,
        "canonical_evidence_order": EVIDENCE_ORDER,
        "notes": ["module evidence score = rank aggregate of Layer 6/7 features",
            "target recurrence = summed module evidence score per target",
            "nulls shuffle targets within evidence classes and globally"]}
    with open(os.path.join(outdir, "layer8_validation_publication_parameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("\nLayer 8 publication validation completed.")
    print(f"Validated/interpreted modules: {len(validated_df)}")
    print(f"Observed targets: {len(observed_targets_df)}")
    print(f"Module-aware null rows: {len(module_nulls)}")
    print(f"Global-shuffle null rows: {len(global_nulls)}")
    print(f"Candidate recurrent targets (all): {len(cand_all)}")
    print(f"Candidate recurrent targets (readable): {len(cand_readable)}")
    print(f"Evidence group rows: {len(group_summary_df)}")
    print(f"Outputs written to: {outdir}")
    if len(target_validation_df):
        show_cols = ["target_name", "n_modules", "weighted_target", "best_rank",
            "evidence_types", "n_strict_enh_onco_mods",
            "n_perm_ctx_mods", "p_emp_modaware",
            "p_emp_shuffle", "recur_supp_note"]
        print("\nTop target recurrence summary:")
        print(target_validation_df[show_cols].head(args.top_n_targets).to_string(index=False))
if __name__ == "__main__":
    main()
