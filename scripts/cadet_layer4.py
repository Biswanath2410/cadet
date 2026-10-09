#!/usr/bin/env python3

import os
import json
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Publication plotting style (matches Layers 1-3)
PUB_FONTSIZE = 22
PUB_TICK_FONTSIZE = 22
PUB_CBAR_FONTSIZE = 22
PUB_LEGEND_FONTSIZE = 18
PUB_DPI = 300
PUB_FIGSIZE_RECT = (12, 6)
PUB_FIGSIZE_HEATMAP = (10, 10)

plt.rcParams.update({"font.size": PUB_FONTSIZE, "axes.titlesize": PUB_FONTSIZE, "axes.labelsize": PUB_FONTSIZE, "xtick.labelsize": PUB_TICK_FONTSIZE,
                        "ytick.labelsize": PUB_TICK_FONTSIZE, "legend.fontsize": PUB_LEGEND_FONTSIZE})

# labels / display names

# older labels -> current names
LABEL_ALIASES = {
    "discordant_join_associated_block": "discordant_join_associated_contact",
    "anchor_recurrent_block": "anchor_recurrent_contact",
    "focal_contact_candidate_block": "focal_contact_candidate",
    "junction_neighbor_block": "junction_neighbor_contact",
    "unclassified_significant_block": "unclassified_significant_contact",
    "discordant_join_block": "discordant_join_associated_contact",
    "discordant_join_peak": "discordant_join_associated_contact",
    "stripe_or_anchor_recurrent_block": "anchor_recurrent_contact",
    "stripe_or_anchor_recurrent_peak": "anchor_recurrent_contact",
    "candidate_loop_like_block": "focal_contact_candidate",
    "candidate_loop_like_peak": "focal_contact_candidate",
    "junction_neighbor_peak": "junction_neighbor_contact",
    "ambiguous_block": "unclassified_significant_contact",
    "ambiguous_peak": "unclassified_significant_contact",
}

LABEL_PRIORITY = {"discordant_join_associated_contact": 1, "anchor_recurrent_contact": 2, "focal_contact_candidate": 3, "junction_neighbor_contact": 4,
                "unclassified_significant_contact": 5}

LABEL_DISPLAY = {"discordant_join_associated_contact": "Discordant-join associated", "anchor_recurrent_contact": "Anchor-recurrent",
                "focal_contact_candidate": "Focal contact candidate", "junction_neighbor_contact": "Junction-neighbor",
                "unclassified_significant_contact": "Unclassified significant"}

LABEL_COLOR = {"discordant_join_associated_contact": "red", "anchor_recurrent_contact": "purple", "focal_contact_candidate": "green",
                "junction_neighbor_contact": "deepskyblue", "unclassified_significant_contact": "dimgray"}

MODULE_TYPE_DISPLAY = {"shared_anchor_recurrent_module": "Shared-anchor recurrent module",
                    "discordant_join_associated_module": "Discordant-join associated module",
                    "compact_pairspace_module": "Compact pair-space module",
                    "singleton_contact_module": "Singleton contact module"}

MODULE_TYPE_COLOR = {"shared_anchor_recurrent_module": "purple", "discordant_join_associated_module": "red", "compact_pairspace_module": "green",
                        "singleton_contact_module": "dimgray"}

def norm_label(x) -> str:
    s = str(x) if pd.notna(x) else "unclassified_significant_contact"
    return LABEL_ALIASES.get(s, s if s in LABEL_PRIORITY else "unclassified_significant_contact")

def disp_label(x) -> str:
    return LABEL_DISPLAY.get(norm_label(x), str(x).replace("_", " "))

def safe_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}

def first_existing(df: pd.DataFrame, names: List[str], default=None):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)

# input helpers

def load_matrix(input_dir: str) -> np.ndarray:
    candidates = [
        os.path.join(input_dir, "latent_deconvolved_matrix_symmetric_display.txt"),
        os.path.join(input_dir, "latent_X_symmetric.txt"),
        os.path.join(input_dir, "part1", "latent_X_symmetric.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return np.loadtxt(p)
    raise FileNotFoundError("Could not find latent matrix. Tried: " + "; ".join(candidates))

def load_layer3_blocks(input_dir: str) -> pd.DataFrame:
    p = os.path.join(input_dir, "upgraded_annotation", "merged_peak_blocks_annotated.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing Layer 3 block table: {p}")
    df = pd.read_csv(p)
    if len(df) == 0:
        return df

    required = ["blk_bin1_center", "blk_bin2_center"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Layer 3 block table missing required column: {c}")

    out = df.copy()
    if "peak_blk_id" not in out.columns:
        out["peak_blk_id"] = np.arange(len(out), dtype=int)

    raw_label = first_existing( out, ["primary_blk_label", "upgraded_peak_block_class", "representative_upgraded_peak_class", "representative_peak_class"], "unclassified_significant_contact", )
    out["layer3_blk_label_raw"] = raw_label.astype(str)
    out["layer3_blk_label"] = out["layer3_blk_label_raw"].map(norm_label)
    out["layer3_blk_label_disp"] = out["layer3_blk_label"].map(LABEL_DISPLAY)

    out["bin1"] = pd.to_numeric(out["blk_bin1_center"], errors="coerce")
    out["bin2"] = pd.to_numeric(out["blk_bin2_center"], errors="coerce")
    out["bin1_min"] = pd.to_numeric(first_existing(out, ["blk_bin1_min"], out["bin1"]), errors="coerce")
    out["bin1_max"] = pd.to_numeric(first_existing(out, ["blk_bin1_max"], out["bin1"]), errors="coerce")
    out["bin2_min"] = pd.to_numeric(first_existing(out, ["blk_bin2_min"], out["bin2"]), errors="coerce")
    out["bin2_max"] = pd.to_numeric(first_existing(out, ["blk_bin2_max"], out["bin2"]), errors="coerce")
    out["rep_bin1"] = pd.to_numeric(first_existing(out, ["rep_bin1"], out["bin1"]), errors="coerce")
    out["rep_bin2"] = pd.to_numeric(first_existing(out, ["rep_bin2"], out["bin2"]), errors="coerce")

    # keep geometry in the upper triangle for module calls
    swap = out["bin1"] > out["bin2"]
    if swap.any():
        for a, b in [("bin1", "bin2"), ("bin1_min", "bin2_min"), ("bin1_max", "bin2_max"), ("rep_bin1", "rep_bin2")]:
            tmp = out.loc[swap, a].copy()
            out.loc[swap, a] = out.loc[swap, b].values
            out.loc[swap, b] = tmp.values

    out["obs_over_exp_raw"] = pd.to_numeric(first_existing(out, ["max_obs_over_exp_raw", "obs_over_exp_raw", "mean_obs_over_exp_raw"], 1.0), errors="coerce").fillna(1.0)
    out["local_obs_over_mean"] = pd.to_numeric(first_existing(out, ["max_local_oe", "local_obs_over_mean", "mean_local_obs_over_mean"], np.nan), errors="coerce")
    out["discord"] = pd.to_numeric(first_existing(out, ["max_discord", "discord", "mean_discord"], np.nan), errors="coerce")
    out["n_same_anchor_blk_w5"] = pd.to_numeric(first_existing(out, ["n_same_anchor_blk_w5"], 0), errors="coerce").fillna(0)
    out["n_blk_neighbors_w5"] = pd.to_numeric(first_existing(out, ["n_blk_neighbors_w5"], 0), errors="coerce").fillna(0)
    out["near_discordant_join"] = first_existing(out, ["near_discordant_join"], False).map(safe_bool)
    out["near_junction"] = first_existing(out, ["near_junction"], False).map(safe_bool)
    out["is_anchor_recur_blk"] = first_existing(out, ["is_anchor_recur_blk", "is_anchor_recur_v4"], False).map(safe_bool)

    out = out.dropna(subset=["bin1", "bin2"]).reset_index(drop=True)
    return out

# module building

def close_pairspace(a: pd.Series, b: pd.Series, tol: float) -> bool:
    return abs(float(a["bin1"]) - float(b["bin1"])) <= tol and abs(float(a["bin2"]) - float(b["bin2"])) <= tol

def connected_components(indices: List[int], neighbor_func) -> List[List[int]]:
    remaining: Set[int] = set(indices)
    comps = []
    while remaining:
        start = remaining.pop()
        comp = [start]
        q = deque([start])
        while q:
            i = q.popleft()
            for j in list(remaining):
                if neighbor_func(i, j):
                    remaining.remove(j)
                    comp.append(j)
                    q.append(j)
        comps.append(sorted(comp))
    return comps

def cluster_anchors(anchor_records: List[Tuple[float, int, str]], tol: float) -> List[dict]:
    
    if not anchor_records:
        return []
    recs = sorted(anchor_records, key=lambda x: x[0])
    clusters = []
    cur = [recs[0]]
    for rec in recs[1:]:
        center = np.median([x[0] for x in cur])
        if abs(rec[0] - center) <= tol:
            cur.append(rec)
        else:
            clusters.append(cur)
            cur = [rec]
    clusters.append(cur)

    out = []
    for cl in clusters:
        vals = [x[0] for x in cl]
        blocks = sorted(set(x[1] for x in cl))
        axes = [x[2] for x in cl]
        out.append({"anchor_center": float(np.median(vals)), "anchor_min": float(np.min(vals)), "anchor_max": float(np.max(vals)), "block_indices": blocks,
                       "support_blocks": len(blocks), "support_records": len(cl), "axes": ";".join(sorted(set(axes)))})
    return out

def build_modules(blocks: pd.DataFrame, pairspace_tol: float, anchor_tol: float, min_anchor_support: int) -> pd.DataFrame:
    out = blocks.sort_values(["bin1", "bin2", "peak_blk_id"]).reset_index(drop=True).copy()
    out["module_id"] = -1
    out["module_type"] = ""
    out["module_anchor_bin"] = np.nan
    out["assign_reason"] = ""

    next_mid = 0
    assigned: Set[int] = set()

    # 1) shared-anchor modules. Keep this anchor-based, not one transitive giant box.
    anchor_candidate_mask = (
        (out["layer3_blk_label"] == "anchor_recurrent_contact") |
        (out["is_anchor_recur_blk"]) |
        (out["n_same_anchor_blk_w5"] >= min_anchor_support)
    )
    anchor_records = []
    for idx, row in out[anchor_candidate_mask].iterrows():
        anchor_records.append((float(row["bin1"]), int(idx), "bin1"))
        anchor_records.append((float(row["bin2"]), int(idx), "bin2"))

    anchor_clusters = cluster_anchors(anchor_records, anchor_tol)
    # largest anchors first, since a block can only belong to one module here
    anchor_clusters = sorted(anchor_clusters, key=lambda d: (d["support_blocks"], d["support_records"]), reverse=True)
    for cl in anchor_clusters:
        members = [i for i in cl["block_indices"] if i not in assigned]
        if len(members) < min_anchor_support:
            continue
        for i in members:
            out.loc[i, "module_id"] = next_mid
            out.loc[i, "module_type"] = "shared_anchor_recurrent_module"
            out.loc[i, "module_anchor_bin"] = cl["anchor_center"]
            out.loc[i, "assign_reason"] = f"shared_anchor~{cl['anchor_center']:.2f};support={len(members)}"
            assigned.add(i)
        next_mid += 1

    # 2) discordant-join blocks: compact pair-space grouping only
    remaining_discordant = [
        int(i) for i, r in out.iterrows()
        if i not in assigned and (r["layer3_blk_label"] == "discordant_join_associated_contact" or bool(r["near_discordant_join"]))]
    comps = connected_components(remaining_discordant, lambda i, j: close_pairspace(out.loc[i], out.loc[j], pairspace_tol))
    for comp in comps:
        for i in comp:
            out.loc[i, "module_id"] = next_mid
            out.loc[i, "module_type"] = "discordant_join_associated_module"
            out.loc[i, "assign_reason"] = "discordant_join_pairspace"
            assigned.add(i)
        next_mid += 1

    # 3) everything else: compact modules or singletons
    remaining = [int(i) for i in out.index if i not in assigned]
    comps = connected_components(remaining, lambda i, j: close_pairspace(out.loc[i], out.loc[j], pairspace_tol))
    for comp in comps:
        mtype = "compact_pairspace_module" if len(comp) > 1 else "singleton_contact_module"
        for i in comp:
            out.loc[i, "module_id"] = next_mid
            out.loc[i, "module_type"] = mtype
            out.loc[i, "assign_reason"] = "compact_pairspace" if len(comp) > 1 else "singleton"
            assigned.add(i)
        next_mid += 1

    out["module_id"] = out["module_id"].astype(int)
    return out

def choose_module_label(grp: pd.DataFrame) -> str:
    tmp = grp.copy()
    tmp["priority"] = tmp["layer3_blk_label"].map(LABEL_PRIORITY).fillna(999).astype(int)
    bestp = tmp["priority"].min()
    best = tmp[tmp["priority"] == bestp]
    counts = best["layer3_blk_label"].value_counts()
    tied = counts[counts == counts.max()].index.tolist()
    if len(tied) == 1:
        return tied[0]
    sub = best[best["layer3_blk_label"].isin(tied)]
    return sub.groupby("layer3_blk_label")["obs_over_exp_raw"].max().sort_values(ascending=False).index[0]

def summarize_modules(module_blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, grp in module_blocks.groupby("module_id"):
        label = choose_module_label(grp)
        rep = grp.sort_values(["obs_over_exp_raw", "local_obs_over_mean"], ascending=False).iloc[0]
        module_anchor_bin = np.nan
        if grp["module_anchor_bin"].notna().any():
            module_anchor_bin = float(grp["module_anchor_bin"].dropna().median())
        if pd.notna(module_anchor_bin):
            partners = []
            for _, r in grp.iterrows():
                # partner = the coordinate away from the recurrent anchor
                if abs(float(r["bin1"]) - module_anchor_bin) <= abs(float(r["bin2"]) - module_anchor_bin):
                    partners.append(float(r["bin2"]))
                else:
                    partners.append(float(r["bin1"]))
            partner_span = float(np.max(partners) - np.min(partners) + 1) if partners else 1.0
        else:
            partner_span = float(max(grp["bin1"].max() - grp["bin1"].min() + 1, grp["bin2"].max() - grp["bin2"].min() + 1))

        rows.append({"module_id": int(mid), "module_type": str(grp["module_type"].iloc[0]),
                     "module_type_disp": MODULE_TYPE_DISPLAY.get(str(grp["module_type"].iloc[0]), str(grp["module_type"].iloc[0])),
                    "module_label": label, "module_label_disp": LABEL_DISPLAY.get(label, label),
                    "n_member_blk": int(len(grp)), "member_peak_blk_ids": ";".join(map(str, grp["peak_blk_id"].tolist())),
                    "bin1_min": float(grp["bin1_min"].min()), "bin1_max": float(grp["bin1_max"].max()),
                    "bin2_min": float(grp["bin2_min"].min()), "bin2_max": float(grp["bin2_max"].max()),
                    "module_anchor_bin": module_anchor_bin, "partner_span_bins": partner_span,
                    "max_obs_over_exp_raw": float(grp["obs_over_exp_raw"].max()),
                    "mean_obs_over_exp_raw": float(grp["obs_over_exp_raw"].mean()),
                    "max_local_oe": float(grp["local_obs_over_mean"].max(skipna=True)) if grp["local_obs_over_mean"].notna().any() else np.nan,
                    "max_discord": float(grp["discord"].max(skipna=True)) if grp["discord"].notna().any() else np.nan,
                    "mean_discord": float(grp["discord"].mean(skipna=True)) if grp["discord"].notna().any() else np.nan,
                    "any_near_discordant_join": bool(grp["near_discordant_join"].any()),
                    "any_near_junction": bool(grp["near_junction"].any()), "rep_peak_blk_id": rep["peak_blk_id"],
                    "rep_bin1": float(rep["bin1"]), "rep_bin2": float(rep["bin2"]),
                    "rep_label": str(rep["layer3_blk_label"]),
                    "rep_obs_over_exp_raw": float(rep["obs_over_exp_raw"]),
                    "assign_reason": ";".join(sorted(set(map(str, grp["assign_reason"].tolist()))))})
    df = pd.DataFrame(rows)
    if len(df):
        type_priority = {"discordant_join_associated_module": 1, "shared_anchor_recurrent_module": 2, "compact_pairspace_module": 3,
                            "singleton_contact_module": 4}
        df["module_type_priority"] = df["module_type"].map(type_priority).fillna(99).astype(int)
        df["module_label_priority"] = df["module_label"].map(LABEL_PRIORITY).fillna(999).astype(int)
        df = df.sort_values(["module_type_priority", "module_label_priority", "max_obs_over_exp_raw"], ascending=[True, True, False]).reset_index(drop=True)
    return df

def representative_blocks(module_blocks: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if len(summary) == 0:
        return module_blocks.iloc[[]].copy()
    keep_ids = set(summary["rep_peak_blk_id"].astype(str))
    rep = module_blocks[module_blocks["peak_blk_id"].astype(str).isin(keep_ids)].copy()
    rep = rep.merge(summary[["module_id", "module_type", "module_type_disp", "module_label", "module_label_disp", "n_member_blk", "module_anchor_bin", "partner_span_bins"]], on="module_id", how="left", suffixes=("", "_module"))
    return rep

# plotting helpers

def prepare_log_matrix(matrix: np.ndarray):
    m = np.asarray(matrix, dtype=float)
    z = np.log2(np.maximum(m, 0) + 1.0)
    finite = z[np.isfinite(z)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    return z, 0.0, vmax, r"$\log_{2}(\mathrm{contact}+1)$"

def style_axes(ax):
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    ax.xaxis.label.set_size(PUB_FONTSIZE)
    ax.yaxis.label.set_size(PUB_FONTSIZE)
    ax.title.set_size(PUB_FONTSIZE)

def savefig(fig, png, pdf=None, tight=True):
    if tight:
        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            pass
    fig.savefig(png, dpi=PUB_DPI, bbox_inches="tight")
    if pdf:
        fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

def make_heatmap_figure(fig_width: float, fig_height: float):
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.12, 0.11, 0.56, 0.56])
    cax = fig.add_axes([0.71, 0.11, 0.025, 0.56])
    return fig, ax, cax

def read_path_bins(input_dir: str):
    
    path = os.path.join(input_dir, "path_bins.tsv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    required = ["path_bin", "chrom", "genomic_bp", "seg_idx"]
    if any(c not in df.columns for c in required):
        return None
    return df.sort_values("path_bin").reset_index(drop=True)

def _path_plot_layout(path_df):
    
    if path_df is None or len(path_df) == 0:
        return None, None
    df = path_df.sort_values("path_bin").reset_index(drop=True)
    boundaries = [0]
    intervals = []
    for _, grp in df.groupby("seg_idx", sort=False):
        start_bin = int(grp["path_bin"].min())
        end_bin = int(grp["path_bin"].max()) + 1
        chrom = str(grp.iloc[0]["chrom"])
        bp = grp["genomic_bp"].astype(int).to_numpy()
        diffs = np.abs(np.diff(np.sort(np.unique(bp))))
        diffs = diffs[diffs > 0]
        res = int(np.median(diffs)) if diffs.size else 1
        intervals.append((chrom, int(bp.min()), int(bp.max()) + res))
        if boundaries[-1] != start_bin:
            boundaries.append(start_bin)
        if boundaries[-1] != end_bin:
            boundaries.append(end_bin)
    return boundaries, intervals

def _add_path_annotations(ax, path_df):
    
    boundaries, intervals = _path_plot_layout(path_df)
    if boundaries is None:
        return

    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, linestyle="--", linewidth=0.4, color="black", alpha=0.7)
        ax.axhline(b - 0.5, linestyle="--", linewidth=0.4, color="black", alpha=0.7)

    centers = [
        (boundaries[i] + boundaries[i + 1] - 1) / 2
        for i in range(len(intervals))
    ]
    labels = []
    for chrom, start, end in intervals:
        start_mb = f"{start/1e6:.1f}".rstrip("0").rstrip(".")
        end_mb = f"{end/1e6:.1f}".rstrip("0").rstrip(".")
        labels.append(f"{chrom}:{start_mb}-{end_mb}Mb")

    ax.set_xticks(centers)
    ax.set_xticklabels( labels, rotation=45, ha="left", va="bottom", rotation_mode="anchor", fontsize=PUB_TICK_FONTSIZE )
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=16)

    ax.set_yticks(boundaries)
    ax.set_yticklabels([""] * len(boundaries))
    ax.tick_params(axis="y", labelsize=PUB_TICK_FONTSIZE)

    min_sep_bins = 14
    previous_b = None
    stagger = 0
    trans = ax.get_yaxis_transform()
    for b in boundaries:
        if previous_b is not None and abs(b - previous_b) < min_sep_bins:
            stagger = 1 - stagger
        else:
            stagger = 0
        x_pos = -0.016 - 0.042 * stagger
        ax.text( x_pos, b, str(b), transform=trans, ha="right", va="center", fontsize=PUB_TICK_FONTSIZE, clip_on=False )
        previous_b = b

    n = len(path_df)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

def legend_handles(keys: List[str], color_map: Dict[str, str], display_map: Dict[str, str], marker="s"):
    uniq = []
    for k in keys:
        if k not in uniq:
            uniq.append(k)
    return [Line2D([0], [0], marker=marker, linestyle="None", markerfacecolor="none", markeredgecolor=color_map.get(k, "dimgray"), markeredgewidth=1.8, markersize=7, label=display_map.get(k, k)) for k in uniq]

def plot_member_blocks(matrix: np.ndarray, module_blocks: pd.DataFrame, out_png: str, out_pdf: str, color_by: str, fig_width: float, fig_height: float, min_box_size: float, legend_fontsize: float, title: str, path_df=None):
    if len(module_blocks) == 0:
        return
    z, vmin, vmax, cbar_label = prepare_log_matrix(matrix)
    fig, ax, cax = make_heatmap_figure(fig_width, fig_height)
    im = ax.imshow(z, origin="upper", cmap="Greys", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=PUB_CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)
    labels_seen = []
    for _, r in module_blocks.iterrows():
        if color_by == "module_type":
            key = str(r["module_type"])
            color = MODULE_TYPE_COLOR.get(key, "dimgray")
            display_map = MODULE_TYPE_DISPLAY
            color_map = MODULE_TYPE_COLOR
            legend_title = "Module type"
        else:
            key = str(r["module_label"])
            color = LABEL_COLOR.get(norm_label(key), "dimgray")
            display_map = LABEL_DISPLAY
            color_map = LABEL_COLOR
            legend_title = "Module label"
        labels_seen.append(key)
        x0 = float(r["bin2_min"])
        y0 = float(r["bin1_min"])
        w = max(float(r["bin2_max"] - r["bin2_min"] + 1), float(min_box_size))
        h = max(float(r["bin1_max"] - r["bin1_min"] + 1), float(min_box_size))
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=1.8, alpha=0.95))
        ax.scatter(float(r["rep_bin2"]), float(r["rep_bin1"]), marker="x", s=28, linewidths=1.0, color=color)

    fig.suptitle(title, fontsize=PUB_FONTSIZE, y=0.97)
    ax.set_xlabel("Path bin", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Path bin", fontsize=PUB_FONTSIZE, labelpad=58)
    _add_path_annotations(ax, path_df)
    style_axes(ax)
    handles = legend_handles(labels_seen, color_map, display_map)
    if handles:
        fig.legend(handles=handles, title=legend_title, loc="upper left", bbox_to_anchor=(0.76, 0.66), frameon=False, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    savefig(fig, out_png, out_pdf, tight=False)

def plot_module_reps(matrix: np.ndarray, reps: pd.DataFrame, out_png: str, out_pdf: str, fig_width: float, fig_height: float, legend_fontsize: float, path_df=None):
    if len(reps) == 0:
        return
    z, vmin, vmax, cbar_label = prepare_log_matrix(matrix)
    fig, ax, cax = make_heatmap_figure(fig_width, fig_height)
    im = ax.imshow(z, origin="upper", cmap="Greys", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=PUB_CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)
    seen = []
    for _, r in reps.iterrows():
        key = str(r["module_label"])
        color = LABEL_COLOR.get(norm_label(key), "dimgray")
        seen.append(norm_label(key))
        ax.scatter(float(r["rep_bin2"]), float(r["rep_bin1"]), marker="o", s=70, facecolors="none", edgecolors=color, linewidths=1.8)
    fig.suptitle("Layer 4 nonredundant module representatives", fontsize=PUB_FONTSIZE, y=0.97)
    ax.set_xlabel("Path bin", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Path bin", fontsize=PUB_FONTSIZE, labelpad=58)
    _add_path_annotations(ax, path_df)
    style_axes(ax)
    handles = legend_handles(seen, LABEL_COLOR, LABEL_DISPLAY, marker="o")
    if handles:
        fig.legend(handles=handles, title="Module label", loc="upper left", bbox_to_anchor=(0.76, 0.66), frameon=False, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    savefig(fig, out_png, out_pdf, tight=False)

def plot_bar(summary: pd.DataFrame, col: str, title: str, out_png: str, out_pdf: str, color_map: Dict[str, str], display_map: Dict[str, str]):
    if len(summary) == 0 or col not in summary.columns:
        return
    counts = summary.groupby(col).size().reset_index(name="n").sort_values("n", ascending=False)
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    ys = np.arange(len(counts))
    colors = [color_map.get(str(x), "dimgray") for x in counts[col]]
    labels = [display_map.get(str(x), str(x)) for x in counts[col]]
    ax.barh(ys, counts["n"], color=colors)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=PUB_TICK_FONTSIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Number of modules", fontsize=PUB_FONTSIZE)
    ax.set_title(title, fontsize=PUB_FONTSIZE)
    style_axes(ax)
    savefig(fig, out_png, out_pdf)

def plot_scatter(summary: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str, out_png: str, out_pdf: str):
    if len(summary) == 0 or xcol not in summary.columns or ycol not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    for lab, g in summary.groupby("module_type"):
        ax.scatter(g[xcol], g[ycol], s=70, alpha=0.85, edgecolors="black", linewidths=0.4, color=MODULE_TYPE_COLOR.get(lab, "dimgray"), label=MODULE_TYPE_DISPLAY.get(lab, lab))
    ax.set_xlabel(xlabel, fontsize=PUB_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PUB_FONTSIZE)
    ax.set_title(title, fontsize=PUB_FONTSIZE)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE)
    style_axes(ax)
    savefig(fig, out_png, out_pdf)

def main():
    ap = argparse.ArgumentParser(description="CADET Layer 4: modules")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pairspace_tolerance", type=float, default=10.0, help="Merge compact blocks within this many bins")
    ap.add_argument("--shared_anchor_tolerance", type=float, default=2.0, help="Bins within which anchors count as the same")
    ap.add_argument("--min_anchor_support", type=int, default=2, help="Min blocks in a shared-anchor module")
    ap.add_argument("--fig_width", type=float, default=10.0)
    ap.add_argument("--fig_height", type=float, default=10.0)
    ap.add_argument("--min_box_size", type=float, default=5.0)
    ap.add_argument("--legend_fontsize", type=float, default=18.0)
    ap.add_argument("--plot_member_heatmaps", action="store_true", help="Also plot member blocks colored by module (QC)")
    args = ap.parse_args()

    input_dir = args.input_dir
    outdir = os.path.join(input_dir, "module_consolidation")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("Loading Layer 3 merged contact blocks...")
    blocks = load_layer3_blocks(input_dir)
    print(f"Layer 3 blocks: {len(blocks)}")
    if len(blocks) == 0:
        raise RuntimeError("No Layer 3 blocks found.")

    print("Building Layer 4 modules...")
    module_blocks = build_modules(blocks, pairspace_tol=args.pairspace_tolerance, anchor_tol=args.shared_anchor_tolerance, min_anchor_support=args.min_anchor_support)
    summary = summarize_modules(module_blocks)
    reps = representative_blocks(module_blocks, summary)

    # add summary labels back to each member block
    module_blocks = module_blocks.merge(summary[["module_id", "module_type_disp", "module_label", "module_label_disp", "n_member_blk", "module_anchor_bin", "partner_span_bins"]], on="module_id", how="left", suffixes=("", "_summary"))

    # write tables
    module_blocks.to_csv(os.path.join(outdir, "significant_blocks_with_module_ids.csv"), index=False)
    module_blocks.to_csv(os.path.join(outdir, "significant_peaks_with_module_ids.csv"), index=False)  # compatibility alias
    summary.to_csv(os.path.join(outdir, "module_summary.csv"), index=False)
    reps.to_csv(os.path.join(outdir, "module_representative_blocks.csv"), index=False)
    reps.to_csv(os.path.join(outdir, "module_representative_peaks.csv"), index=False)  # compatibility alias

    params = {"layer4_version": "module_representatives", "input_dir": input_dir, "input_level": "Layer 3 merged contact blocks",
            "pairspace_tolerance": args.pairspace_tolerance, "shared_anchor_tolerance": args.shared_anchor_tolerance,
            "min_anchor_support": args.min_anchor_support, "module_types": MODULE_TYPE_DISPLAY, "label_priority": LABEL_PRIORITY,
            "note": "heatmap shows one representative block per module"}
    with open(os.path.join(outdir, "module_parameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("Module type summary:")
    print(summary.groupby("module_type_disp").size().reset_index(name="n").to_string(index=False))

    matrix = load_matrix(input_dir)
    path_df = read_path_bins(input_dir)
    plot_bar(summary, "module_type", "Layer 4 contact-module types", os.path.join(plot_dir, "module_type_barplot.png"), os.path.join(plot_dir, "module_type_barplot.pdf"), MODULE_TYPE_COLOR, MODULE_TYPE_DISPLAY)
    plot_bar(summary, "module_label", "Layer 4 module labels", os.path.join(plot_dir, "module_label_barplot.png"), os.path.join(plot_dir, "module_label_barplot.pdf"), LABEL_COLOR, LABEL_DISPLAY)
    plot_scatter(summary, "n_member_blk", "max_obs_over_exp_raw", "Number of member blocks", "Maximum raw O/E", "Module support vs enrichment", os.path.join(plot_dir, "module_support_vs_enrichment.png"), os.path.join(plot_dir, "module_support_vs_enrichment.pdf"))
    plot_scatter(summary, "partner_span_bins", "max_obs_over_exp_raw", "Partner span or module span (bins)", "Maximum raw O/E", "Module span vs enrichment", os.path.join(plot_dir, "module_span_vs_enrichment.png"), os.path.join(plot_dir, "module_span_vs_enrichment.pdf"))

    if args.plot_member_heatmaps:
        plot_member_blocks(matrix, module_blocks, os.path.join(plot_dir, "QC_member_blocks_by_module_type_heatmap.png"),
                os.path.join(plot_dir, "QC_member_blocks_by_module_type_heatmap.pdf"), color_by="module_type", fig_width=args.fig_width, fig_height=args.fig_height, min_box_size=args.min_box_size,
                legend_fontsize=args.legend_fontsize, title="QC: Layer 3 member blocks colored by Layer 4 module type", path_df=path_df)
        plot_member_blocks(matrix, module_blocks, os.path.join(plot_dir, "QC_member_blocks_by_module_label_heatmap.png"),
                os.path.join(plot_dir, "QC_member_blocks_by_module_label_heatmap.pdf"), color_by="module_label", fig_width=args.fig_width, fig_height=args.fig_height, min_box_size=args.min_box_size,
                legend_fontsize=args.legend_fontsize, title="QC: Layer 3 member blocks colored by Layer 4 module label", path_df=path_df)
    else:
        print("Skipping member-block heatmaps; use --plot_member_heatmaps for QC.")

    plot_module_reps(matrix, reps, os.path.join(plot_dir, "module_representatives_heatmap.png"),
                os.path.join(plot_dir, "module_representatives_heatmap.pdf"), fig_width=args.fig_width, fig_height=args.fig_height, legend_fontsize=args.legend_fontsize, path_df=path_df)

    print(f"\nOutputs written to: {outdir}")

if __name__ == "__main__":
    main()
