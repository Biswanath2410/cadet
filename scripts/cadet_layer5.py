#!/usr/bin/env python3

import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication figure style (matched to final CADET Layers 1-4)
PUB_FONTSIZE = 22
PUB_TICK_FONTSIZE = 22
PUB_LEGEND_FONTSIZE = 18
PUB_DPI = 300
PUB_FIGSIZE_RECT = (12, 6)
PUB_FIGSIZE_SQUARE = (10, 10)

# Label handling: keep this compatible with Layer 4 and a few older runs

# older labels -> current names
LABEL_ALIASES = {
    "discordant_join_associated_block": "discordant_join_associated_contact",
    "anchor_recurrent_block": "anchor_recurrent_contact",
    "focal_contact_candidate_block": "focal_contact_candidate",
    "junction_neighbor_block": "junction_neighbor_contact",
    "unclassified_significant_block": "unclassified_significant_contact",
    "discordant_join_block": "discordant_join_associated_contact",
    "discordant_join_peak": "discordant_join_associated_contact",
    "interchromosomal_rewiring_block": "discordant_join_associated_contact",
    "strong_rewiring_block": "discordant_join_associated_contact",
    "distal_proximity_block": "focal_contact_candidate",
    "candidate_loop_like_block": "focal_contact_candidate",
    "candidate_loop_like_peak": "focal_contact_candidate",
    "stripe_or_anchor_recurrent_block": "anchor_recurrent_contact",
    "stripe_or_anchor_recurrent_peak": "anchor_recurrent_contact",
    "junction_neighbor_peak": "junction_neighbor_contact",
    "ambiguous_block": "unclassified_significant_contact",
    "ambiguous_peak": "unclassified_significant_contact",
}

# older module types -> current names
MODULE_TYPE_ALIASES = {"discordant_join_neighborhood_module": "discordant_join_associated_module"}

LABEL_DISPLAY = {"discordant_join_associated_contact": "Discordant-join associated",
    "anchor_recurrent_contact": "Anchor-recurrent",
    "focal_contact_candidate": "Focal contact candidate",
    "junction_neighbor_contact": "Junction-neighbor",
    "unclassified_significant_contact": "Unclassified significant"}

MODULE_TYPE_DISPLAY = {"discordant_join_associated_module": "Discordant-join module",
    "shared_anchor_recurrent_module": "Shared-anchor recurrent module",
    "compact_pairspace_module": "Compact pair-space module",
    "singleton_contact_module": "Singleton contact module"}

LABEL_COLOR = {"discordant_join_associated_contact": "red",
    "anchor_recurrent_contact": "purple",
    "focal_contact_candidate": "green",
    "junction_neighbor_contact": "deepskyblue",
    "unclassified_significant_contact": "dimgray"}

MODULE_TYPE_COLOR = {"discordant_join_associated_module": "red",
    "shared_anchor_recurrent_module": "purple",
    "compact_pairspace_module": "green",
    "singleton_contact_module": "dimgray"}

# Heuristic priors. These rank modules only
LABEL_PRIOR = {"discordant_join_associated_contact": 1.00,
    "anchor_recurrent_contact": 0.75,
    "focal_contact_candidate": 0.65,
    "junction_neighbor_contact": 0.45,
    "unclassified_significant_contact": 0.25}

MODULE_TYPE_PRIOR = {"discordant_join_associated_module": 1.00,
    "shared_anchor_recurrent_module": 0.75,
    "compact_pairspace_module": 0.60,
    "singleton_contact_module": 0.35}

DEFAULT_WEIGHTS = {"context_prior_score": 0.25,
    "enrichment": 0.25,
    "local_enrichment": 0.15,
    "discord": 0.15,
    "module_supp": 0.12,
    "anchor_recur": 0.08}


def normalize_label(x) -> str:
    s = str(x) if pd.notna(x) else "unclassified_significant_contact"
    return LABEL_ALIASES.get(s, s if s in LABEL_PRIOR else "unclassified_significant_contact")

def normalize_module_type(x) -> str:
    s = str(x) if pd.notna(x) else "singleton_contact_module"
    return MODULE_TYPE_ALIASES.get(s, s if s in MODULE_TYPE_PRIOR else "singleton_contact_module")

def first_existing(df: pd.DataFrame, names: List[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)

def as_numeric(series: pd.Series, default=np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)

def minmax_scale(values) -> pd.Series:
    x = pd.Series(values, dtype="float64")
    out = pd.Series(np.zeros(len(x)), index=x.index, dtype="float64")
    finite = np.isfinite(x.values)
    if finite.sum() == 0:
        return out
    xmin = float(np.nanmin(x.values[finite]))
    xmax = float(np.nanmax(x.values[finite]))
    if xmax <= xmin:
        out.loc[finite] = 1.0
        return out
    out.loc[finite] = (x.loc[finite] - xmin) / (xmax - xmin)
    return out

def percentile_rank(values, higher_is_better=True) -> pd.Series:
    x = pd.Series(values, dtype="float64")
    finite = np.isfinite(x.values)
    out = pd.Series(np.zeros(len(x)), index=x.index, dtype="float64")
    if finite.sum() == 0:
        return out
    # pct=True gives 1/n..1.0 for ascending ranks.
    ranks = x.loc[finite].rank(method="average", pct=True, ascending=True)
    if higher_is_better:
        out.loc[finite] = ranks
    else:
        out.loc[finite] = 1.0 - ranks
    return out

def savefig(fig, png_path: str, pdf_path: str = None) -> None:
    # Figure-export settings only; no analytical values are changed here.
    fig.tight_layout(pad=1.2)
    fig.savefig(png_path, dpi=PUB_DPI, bbox_inches="tight")
    if pdf_path:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

#  cleanup

def load_module_summary(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing Layer 4 module summary: {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise RuntimeError("Layer 4 module_summary.csv is empty.")

    if "module_id" not in df.columns:
        raise ValueError("module_summary.csv must contain module_id")

    out = df.copy()

    # Normalize labels/types from current or older Layer 4 outputs.
    label_raw = first_existing(out, ["module_label", "primary_module_label", "top_peak_class", "rep_label"], "unclassified_significant_contact")
    type_raw = first_existing(out, ["module_type"], "singleton_contact_module")
    out["module_label_raw"] = label_raw.astype(str)
    out["module_label"] = out["module_label_raw"].map(normalize_label)
    out["module_label_disp"] = out["module_label"].map(LABEL_DISPLAY)
    out["module_type_raw"] = type_raw.astype(str)
    out["module_type"] = out["module_type_raw"].map(normalize_module_type)
    out["module_type_disp"] = out["module_type"].map(MODULE_TYPE_DISPLAY)
    # Names used in the output tables/plots.
    out["grouping_type_disp"] = out["module_type_disp"]
    out["dominant_label_disp"] = out["module_label_disp"]
    # Support/geometry fields.
    out["n_member_blk"] = as_numeric(first_existing(out, ["n_member_blk", "n_member_peaks", "module_size"], 1), 1).astype(int)
    out["n_member_peaks"] = out["n_member_blk"]  # compatibility alias

    out["bin1_min"] = as_numeric(first_existing(out, ["bin1_min"], np.nan))
    out["bin1_max"] = as_numeric(first_existing(out, ["bin1_max"], np.nan))
    out["bin2_min"] = as_numeric(first_existing(out, ["bin2_min"], np.nan))
    out["bin2_max"] = as_numeric(first_existing(out, ["bin2_max"], np.nan))
    out["module_span_bin1"] = (out["bin1_max"] - out["bin1_min"] + 1).replace([np.inf, -np.inf], np.nan)
    out["module_span_bin2"] = (out["bin2_max"] - out["bin2_min"] + 1).replace([np.inf, -np.inf], np.nan)
    out["module_span_bin1"] = out["module_span_bin1"].fillna(1.0).clip(lower=1.0)
    out["module_span_bin2"] = out["module_span_bin2"].fillna(1.0).clip(lower=1.0)
    out["module_area_bins"] = (out["module_span_bin1"] * out["module_span_bin2"]).clip(lower=1.0)
    out["module_density"] = out["n_member_blk"] / out["module_area_bins"]

    out["partner_span_bins"] = as_numeric(first_existing(out, ["partner_span_bins"], np.nan))
    out["partner_span_bins"] = out["partner_span_bins"].fillna(np.maximum(out["module_span_bin1"], out["module_span_bin2"]))

    # Evidence columns used for ranking.
    out["max_obs_over_exp_raw"] = as_numeric(first_existing(out, ["max_obs_over_exp_raw", "rep_obs_over_exp_raw", "mean_obs_over_exp_raw"], 1.0), 1.0)
    out["mean_obs_over_exp_raw"] = as_numeric(first_existing(out, ["mean_obs_over_exp_raw", "max_obs_over_exp_raw"], 1.0), 1.0)
    out["max_local_oe"] = as_numeric(first_existing(out, ["max_local_oe"], np.nan))
    out["max_local_oe"] = out["max_local_oe"].fillna(1.0)
    out["max_discord"] = as_numeric(first_existing(out, ["max_discord"], np.nan))
    out["mean_discord"] = as_numeric(first_existing(out, ["mean_discord"], np.nan))
    out["max_neighbor_count"] = as_numeric(first_existing(out, ["max_neighbor_count", "n_blk_neighbors_w5"], 0), 0)

    out["module_anchor_bin"] = as_numeric(first_existing(out, ["module_anchor_bin"], np.nan))
    out["has_recur_anchor"] = out["module_anchor_bin"].notna() | out["module_type"].eq("shared_anchor_recurrent_module") | out["module_label"].eq("anchor_recurrent_contact")
    out["any_near_discordant_join"] = first_existing(out, ["any_near_discordant_join", "near_discordant_join"], False).astype(str).str.lower().isin(["true", "1", "yes", "y"])
    out["any_near_junction"] = first_existing(out, ["any_near_junction", "near_junction"], False).astype(str).str.lower().isin(["true", "1", "yes", "y"])

    # Representative coordinates for downstream compatibility.
    out["rep_bin1"] = as_numeric(first_existing(out, ["rep_bin1", "top_peak_bin1", "module_center_bin1"], np.nan))
    out["rep_bin2"] = as_numeric(first_existing(out, ["rep_bin2", "top_peak_bin2", "module_center_bin2"], np.nan))
    out["top_peak_bin1"] = out["rep_bin1"]
    out["top_peak_bin2"] = out["rep_bin2"]
    out["top_peak_class"] = out["module_label"]

    return out

# scoring

def add_priority_scores(mod: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = mod.copy()

    out["label_prior_score"] = out["module_label"].map(LABEL_PRIOR).fillna(0.25)
    out["module_type_prior_score"] = out["module_type"].map(MODULE_TYPE_PRIOR).fillna(0.35)
    out["context_prior_score"] = 0.60 * out["label_prior_score"] + 0.40 * out["module_type_prior_score"]

    out["distance_discordance_available"] = out["max_discord"].notna()
    out["max_distance_discordance_score_for_scoring"] = out["max_discord"].fillna(0.0).clip(lower=0.0)

    out["enrichment"] = minmax_scale(np.log2(np.maximum(out["max_obs_over_exp_raw"].astype(float), 1.0)))
    out["local_enrichment"] = minmax_scale(np.log2(np.maximum(out["max_local_oe"].astype(float), 1.0)))
    out["discord"] = minmax_scale(out["max_distance_discordance_score_for_scoring"])
    out["module_supp"] = minmax_scale(np.log2(np.maximum(out["n_member_blk"].astype(float), 1.0)))
    out["anchor_recur"] = minmax_scale(np.log2(np.maximum(out["partner_span_bins"].astype(float), 1.0)))
    # Avoid giving a full partner-span boost to non-anchor modules.
    out.loc[~out["has_recur_anchor"], "anchor_recur"] *= 0.35

    # Weighted heuristic score.
    out["weighted_priority"] = ( weights["context_prior_score"] * out["context_prior_score"] + weights["enrichment"] * out["enrichment"] + weights["local_enrichment"] * out["local_enrichment"] + weights["discord"] * out["discord"] + weights["module_supp"] * out["module_supp"] +
        weights["anchor_recur"] * out["anchor_recur"])

    # Rank aggregation is less sensitive to the exact weights.
    rank_components = { "rank_context_prior": percentile_rank(out["context_prior_score"]), "rank_enrichment": percentile_rank(np.log2(np.maximum(out["max_obs_over_exp_raw"], 1.0))), "rank_local_enrichment": percentile_rank(np.log2(np.maximum(out["max_local_oe"], 1.0))), "rank_discordance": percentile_rank(out["max_distance_discordance_score_for_scoring"]), "rank_module_supp": percentile_rank(np.log2(np.maximum(out["n_member_blk"], 1.0))), "rank_anchor_recurrence": percentile_rank(out["anchor_recur"])}
    for k, v in rank_components.items():
        out[k] = v
    out["rank_agg"] = pd.DataFrame(rank_components).mean(axis=1)

    # Evidence-axis scores used for shortlists/interpretation.
    out["struct_supp"] = (0.45 * out["context_prior_score"] + 0.25 * out["discord"] + 0.20 * out["enrichment"] + 0.10 * out["local_enrichment"])
    out["anchor_recur_supp"] = (0.45 * out["anchor_recur"] + 0.25 * out["module_supp"] + 0.20 * out["enrichment"] + 0.10 * out["context_prior_score"])
    out["focal_contact_supp"] = (0.40 * out["local_enrichment"] + 0.35 * out["enrichment"] + 0.15 * out["context_prior_score"] + 0.10 * (1.0 - out["module_supp"].clip(0, 1)))

    # Main score for sorting; rank aggregate gets the larger share.
    out["priority"] = 0.60 * out["rank_agg"] + 0.40 * out["weighted_priority"]

    out["high_discordance_flag"] = out["max_distance_discordance_score_for_scoring"] >= 1.0
    out["high_enrichment_flag"] = out["max_obs_over_exp_raw"] >= 2.0

    out["priority_tier"] = pd.cut(out["priority"], bins=[-np.inf, 0.35, 0.55, 0.75, np.inf], labels=["low", "moderate", "high", "top"], ordered=True)

    recommendations = []
    for _, row in out.iterrows():
        tier = str(row["priority_tier"])
        mtype = str(row["module_type"])
        label = str(row["module_label"])

        is_shared_anchor = mtype == "shared_anchor_recurrent_module"
        is_discordant_module = mtype == "discordant_join_associated_module"
        is_discordant_label = label == "discordant_join_associated_contact"
        is_anchor_label = label == "anchor_recurrent_contact"
        is_focal_module = mtype == "compact_pairspace_module"
        is_focal_label = label == "focal_contact_candidate"

        if tier in {"high", "top"} and is_shared_anchor and is_discordant_label:
            recommendations.append("prioritize_discordant_join_supported_recurrent_module")

        elif tier in {"high", "top"} and (is_discordant_module or is_discordant_label):
            recommendations.append("prioritize_discordant_join_module")

        elif tier in {"high", "top"} and (is_shared_anchor or is_anchor_label):
            recommendations.append("prioritize_anchor_recurrent_module")

        elif tier in {"high", "top"} and (is_focal_module or is_focal_label):
            recommendations.append("prioritize_focal_contact_module")

        elif tier in {"high", "top"}:
            recommendations.append("prioritize_high_scoring_contact_module")

        elif tier == "moderate":
            recommendations.append("secondary_candidate")

        else:
            recommendations.append("lower_priority_or_context_dependent")

    out["recommendation"] = recommendations

    out = out.sort_values(["priority", "rank_agg", "weighted_priority", "max_obs_over_exp_raw", "n_member_blk"],ascending=[False, False, False, False, False]).reset_index(drop=True)
    out["rank_overall"] = np.arange(1, len(out) + 1, dtype=int)
    return out

# shortlists and summaries

def build_shortlists(scored: pd.DataFrame, top_n: int) -> Dict[str, pd.DataFrame]:
    shortlists = {"top10_overall": scored.head(min(10, len(scored))).copy(),
        "top_overall": scored.head(min(top_n, len(scored))).copy(),
        "top_high_enrichment_modules": scored.sort_values(["max_obs_over_exp_raw", "priority"], ascending=False).head(top_n).copy()}
    shortlists["top_discordant_join_modules"] = scored[(scored["module_type"] == "discordant_join_associated_module") |
        (scored["module_label"] == "discordant_join_associated_contact") |
        (scored["any_near_discordant_join"])].head(top_n).copy()
    shortlists["top_anchor_recurrent_modules"] = scored[(scored["module_type"] == "shared_anchor_recurrent_module") |
        (scored["module_label"] == "anchor_recurrent_contact") |(scored["has_recur_anchor"])].head(top_n).copy()
    shortlists["top_focal_contact_candidates"] = scored[(scored["module_type"] == "compact_pairspace_module") |
        (scored["module_label"] == "focal_contact_candidate")].head(top_n).copy()
    shortlists["top_high_discordance_modules"] = scored[scored["high_discordance_flag"]].head(top_n).copy()
    return shortlists

def summarize_priority_old(scored: pd.DataFrame) -> pd.DataFrame:
    return (scored.groupby(["priority_tier", "module_type_disp", "module_label_disp"], observed=False) .size() .reset_index(name="n").sort_values(["priority_tier", "n"], ascending=[True, False]).reset_index(drop=True))

def summarize_priority(scored: pd.DataFrame) -> pd.DataFrame:
    return (scored.groupby( ["priority_tier", "grouping_type_disp", "dominant_label_disp"], observed=False ) .size().reset_index(name="n").sort_values(["priority_tier", "n"], ascending=[True, False]).reset_index(drop=True))
# plots

def plot_priority_bar(scored: pd.DataFrame, out_png: str, out_pdf: str, top_n: int = 25) -> None:
    df = scored.head(min(top_n, len(scored))).copy()
    if len(df) == 0:
        return

    # Categorical module summary: use horizontal bars for CADET publication figures.
    fig_h = max(6.0, 0.48 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(df))
    colors = [MODULE_TYPE_COLOR.get(t, "dimgray") for t in df["module_type"]]
    ax.barh(y, df["priority"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([f"M{m}" for m in df["module_id"]], fontsize=PUB_TICK_FONTSIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Priority score", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Module", fontsize=PUB_FONTSIZE)
    ax.set_title("Prioritized ecDNA contact modules", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PUB_TICK_FONTSIZE)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=c, label=MODULE_TYPE_DISPLAY[k])
        for k, c in MODULE_TYPE_COLOR.items()]
    ax.legend(handles=handles, frameon=False, fontsize=PUB_LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    savefig(fig, out_png, out_pdf)

def plot_scatter_by_type(scored: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str, out_png: str, out_pdf: str, logx: bool = False) -> None:
    if len(scored) == 0:
        return
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    for typ, grp in scored.groupby("module_type", sort=False):
        ax.scatter(
            grp[xcol], grp[ycol], s=65, alpha=0.85,
            edgecolors="black", linewidths=0.4,
            color=MODULE_TYPE_COLOR.get(typ, "dimgray"),
            label=MODULE_TYPE_DISPLAY.get(typ, typ),
        )
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=PUB_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PUB_FONTSIZE)
    ax.set_title(title, fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE)
    savefig(fig, out_png, out_pdf)

def plot_score_heatmap(scored: pd.DataFrame, out_png: str, out_pdf: str, top_n: int = 30) -> None:
    cols = ['context_prior_score', 'enrichment', 'local_enrichment', 'discord', 'module_supp', 'anchor_recur', 'rank_agg', 'weighted_priority']
    df = scored.head(min(top_n, len(scored))).copy()
    if len(df) == 0:
        return
    mat = df[cols].astype(float).values

    # This is an evidence-component matrix, not an ecDNA path contact heatmap;
    # therefore genomic interval/breakpoint annotations are not applicable here.
    fig_h = max(8.0, 0.48 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Scaled score", fontsize=PUB_FONTSIZE)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([f"M{m}" for m in df["module_id"]], fontsize=PUB_TICK_FONTSIZE)
    ax.set_xticks(np.arange(len(cols)))
    component_labels = ["Context prior", "Enrichment","Local enrichment", "Discordance", "Module support", "Anchor recurrence", "Rank aggregate", "Weighted priority"]
    ax.set_xticklabels(component_labels, rotation=35, ha="right", fontsize=PUB_TICK_FONTSIZE)
    ax.set_xlabel("Evidence component", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Module", fontsize=PUB_FONTSIZE)
    ax.set_title(f"Evidence components for top {len(df)} modules", fontsize=PUB_FONTSIZE)
    savefig(fig, out_png, out_pdf)

def plot_tier_bar(scored: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    if len(scored) == 0:
        return
    tab = scored.groupby(["priority_tier", "module_type"], observed=False).size().reset_index(name="n")
    if len(tab) == 0:
        return
    tiers = ["low", "moderate", "high", "top"]
    types = [t for t in MODULE_TYPE_DISPLAY if t in set(scored["module_type"])]

    # Categorical count summary: horizontal stacked bars by publication convention.
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    left = np.zeros(len(tiers))
    for typ in types:
        vals = []
        for tier in tiers:
            v = tab[(tab["priority_tier"].astype(str) == tier) & (tab["module_type"] == typ)]["n"]
            vals.append(int(v.iloc[0]) if len(v) else 0)
        ax.barh( tiers, vals, left=left, color=MODULE_TYPE_COLOR.get(typ, "dimgray"), label=MODULE_TYPE_DISPLAY.get(typ, typ), )
        left += np.array(vals)
    ax.set_xlabel("Number of modules", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Priority tier", fontsize=PUB_FONTSIZE)
    ax.set_title("Priority tiers by module type", fontsize=PUB_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    savefig(fig, out_png, out_pdf)

# main

def main() -> None:
    ap = argparse.ArgumentParser(description="CADET Layer 5: module priority")
    ap.add_argument("--input_dir", required=True, help="cadet_final_results folder")
    ap.add_argument("--top_n", type=int, default=15, help="Modules per shortlist")
    ap.add_argument("--plot_top_n", type=int, default=25, help="Modules in plots")
    ap.add_argument("--weights_json", default="", help="Score weights (JSON string or file)")
    args = ap.parse_args()

    weights = DEFAULT_WEIGHTS.copy()
    if args.weights_json:
        if os.path.exists(args.weights_json):
            with open(args.weights_json) as f:
                custom = json.load(f)
        else:
            custom = json.loads(args.weights_json)
        weights.update({k: float(v) for k, v in custom.items() if k in weights})
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Score weights must sum to a positive value.")
    weights = {k: v / total for k, v in weights.items()}

    input_dir = args.input_dir
    module_dir = os.path.join(input_dir, "module_consolidation")
    module_summary_csv = os.path.join(module_dir, "module_summary.csv")
    outdir = os.path.join(input_dir, "module_prioritization")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    modules = load_module_summary(module_summary_csv)
    scored = add_priority_scores(modules, weights)
    shortlists = build_shortlists(scored, top_n=args.top_n)
    tier_summary = summarize_priority(scored)

    # Main outputs.
    output_cols = ['rank_overall', 'module_id', 'module_type', 'module_label', 'priority', 'priority_tier', 'recommendation', 'rank_agg', 'weighted_priority', 'struct_supp', 'anchor_recur_supp', 'focal_contact_supp', 'enrichment', 'local_enrichment', 'discord', 'module_supp', 'anchor_recur', 'max_obs_over_exp_raw', 'max_local_oe', 'max_discord', 'n_member_blk', 'partner_span_bins', 'has_recur_anchor', 'any_near_discordant_join']
    output_cols = [c for c in output_cols if c in scored.columns]
    scored[output_cols].to_csv(os.path.join(outdir, "module_priority_scores.csv"), index=False)
    tier_summary.to_csv(os.path.join(outdir, "module_priority_tier_summary.csv"), index=False)
    for name, df in shortlists.items():
        df[output_cols].to_csv(os.path.join(outdir, f"{name}.csv"), index=False)

    # Compatibility aliases for older downstream scripts.
    shortlists["top_discordant_join_modules"][output_cols].to_csv(os.path.join(outdir, "top_structural.csv"), index=False)
    shortlists["top_focal_contact_candidates"][output_cols].to_csv(os.path.join(outdir, "top_focal_contact_candidates.csv"), index=False)
    shortlists["top_high_discordance_modules"][output_cols].to_csv(os.path.join(outdir, "top_high_discordance.csv"), index=False)

    params = {"layer5_version": "publication_module_prioritization", "input_dir": input_dir, "input_file": module_summary_csv,
        "purpose": "rank Layer 4 modules", "label_prior": LABEL_PRIOR, "module_type_prior": MODULE_TYPE_PRIOR, "weights_normalized": weights,
        "priority": "0.60 * rank_aggregate_score + 0.40 * weighted_priority_score",
        "weighted_priority_components": weights, "rank_aggregate_components": [ "context_prior", "enrichment", "local_enrichment", "discordance", "module_support", "anchor_recurrence" ]}
    with open(os.path.join(outdir, "priority_parameters.json"), "w") as f:
        json.dump(params, f, indent=2)
        
    plot_priority_bar(scored, os.path.join(plot_dir, "top_priority_modules.png"), os.path.join(plot_dir, "top_priority_modules.pdf"), top_n=args.plot_top_n)
    plot_scatter_by_type(scored, "max_obs_over_exp_raw", "priority", "Maximum raw O/E", "Priority score", "Priority vs raw O/E", os.path.join(plot_dir, "priority_vs_enrichment.png"), os.path.join(plot_dir, "priority_vs_enrichment.pdf"),
        logx=True)
    plot_scatter_by_type(scored, "max_distance_discordance_score_for_scoring", "priority", "Max positive distance-discordance score used for scoring", "Priority score", "Priority vs distance-discordance evidence", os.path.join(plot_dir, "priority_vs_discordance.png"), os.path.join(plot_dir, "priority_vs_discordance.pdf"))
    plot_scatter_by_type( scored, "n_member_blk", "priority", "Number of member blocks in module", "Priority score", "Priority vs module support", os.path.join(plot_dir, "priority_vs_module_support.png"), os.path.join(plot_dir, "priority_vs_module_support.pdf"))
    plot_scatter_by_type(scored, "partner_span_bins", "anchor_recur_supp", "Partner span / module span (bins)", "Anchor-recurrence support score", "Anchor-recurrence support", os.path.join(plot_dir, "anchor_recurrence_support.png"), os.path.join(plot_dir, "anchor_recurrence_support.pdf"))
    plot_score_heatmap(scored, os.path.join(plot_dir, "score_components_heatmap.png"), os.path.join(plot_dir, "score_components_heatmap.pdf"), top_n=args.plot_top_n)
    plot_tier_bar( scored, os.path.join(plot_dir, "priority_tiers_by_module_type.png"), os.path.join(plot_dir, "priority_tiers_by_module_type.pdf"))

    print("Layer 5 module prioritization completed.")
    print(f"Modules scored: {len(scored)}")
    print(f"Outputs written to: {outdir}")
    cols = ["rank_overall","module_id","grouping_type_disp","dominant_label_disp","priority",
    "rank_agg","weighted_priority","priority_tier","recommendation","max_obs_over_exp_raw", "n_member_blk"]
    cols = [c for c in cols if c in scored.columns]
    print("\nTop 10 overall modules:")
    print(scored[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
