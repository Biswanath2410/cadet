#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

CLASS_COLOR_MAP = {"discordant_join_associated_contact": "#e41a1c",   # red
    "anchor_recurrent_contact": "#984ea3",             # purple
    "focal_contact_candidate": "#33a02c",              # green
    "junction_neighbor_contact": "#00bcd4",            # cyan
    "unclassified_significant_contact": "#666666",     # dark gray

    # old labels kept so older outputs still plot
    "discordant_join_peak": "#e41a1c",
    "discordant_join_block": "#e41a1c",
    "stripe_or_anchor_recurrent_peak": "#984ea3",
    "stripe_or_anchor_recurrent_block": "#984ea3",
    "candidate_loop_like_peak": "#33a02c",
    "candidate_loop_like_block": "#33a02c",
    "junction_neighbor_peak": "#00bcd4",
    "junction_neighbor_block": "#00bcd4",
    "ambiguous_peak": "#666666",
    "ambiguous_block": "#666666"}

LABEL_DISPLAY_MAP = {"discordant_join_associated_contact": "Discordant-join associated",
    "anchor_recurrent_contact": "Anchor-recurrent",
    "focal_contact_candidate": "Focal contact candidate",
    "junction_neighbor_contact": "Junction-neighbor",
    "unclassified_significant_contact": "Unclassified significant",

    # old display labels
    "discordant_join_peak": "Discordant-join associated",
    "discordant_join_block": "Discordant-join associated block",
    "stripe_or_anchor_recurrent_peak": "Anchor-recurrent",
    "stripe_or_anchor_recurrent_block": "Anchor-recurrent block",
    "candidate_loop_like_peak": "Focal contact candidate",
    "candidate_loop_like_block": "Focal contact candidate block",
    "junction_neighbor_peak": "Junction-neighbor",
    "junction_neighbor_block": "Junction-neighbor block",
    "ambiguous_peak": "Unclassified significant",
    "ambiguous_block": "Unclassified significant block"}

BLOCK_DISPLAY_MAP = {"discordant_join_associated_contact": "Discordant-join associated block",
    "anchor_recurrent_contact": "Anchor-recurrent block",
    "focal_contact_candidate": "Focal contact candidate block",
    "junction_neighbor_contact": "Junction-neighbor block",
    "unclassified_significant_contact": "Unclassified significant block"}

def display_label(label):
    return LABEL_DISPLAY_MAP.get(str(label), str(label).replace("_", " "))

def display_block_label(label):
    return BLOCK_DISPLAY_MAP.get(str(label), display_label(label))

# Publication  style:
PUB_FONTSIZE = 22
PUB_TICK_FONTSIZE = 22
PUB_CBAR_FONTSIZE = 22
PUB_LEGEND_FONTSIZE = 18
PUB_DPI = 300
PUB_FIGSIZE_SQUARE = (10, 10)
#PUB_FIGSIZE_RECT = (10, 6)
PUB_FIGSIZE_RECT = (12, 6)
PUB_FIGSIZE_HEATMAP = (10, 10)
PUB_LW = 0.4

plt.rcParams.update({"font.size": PUB_FONTSIZE,
    "axes.titlesize": PUB_FONTSIZE,
    "axes.labelsize": PUB_FONTSIZE,
    "xtick.labelsize": PUB_TICK_FONTSIZE,
    "ytick.labelsize": PUB_TICK_FONTSIZE,
    "legend.fontsize": PUB_LEGEND_FONTSIZE})

def _apply_pub_axes(ax):
    ax.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    ax.xaxis.label.set_size(PUB_FONTSIZE)
    ax.yaxis.label.set_size(PUB_FONTSIZE)
    ax.title.set_size(PUB_FONTSIZE)

def save_pub_figure(fig, outpath):
    fig.savefig(outpath, dpi=PUB_DPI, bbox_inches="tight",facecolor="white")

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
    # Internal segment boundaries.
    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, linestyle="--", linewidth=PUB_LW, color="black", alpha=0.7)
        ax.axhline(b - 0.5, linestyle="--", linewidth=PUB_LW, color="black", alpha=0.7)

    # X-axis: genomic interval labels at segment centers.
    centers = [(boundaries[i] + boundaries[i + 1] - 1) / 2
        for i in range(len(intervals))]
    labels = []
    for chrom, start, end in intervals:
        start_mb = f"{start/1e6:.1f}".rstrip("0").rstrip(".")
        end_mb = f"{end/1e6:.1f}".rstrip("0").rstrip(".")
        labels.append(f"{chrom}:{start_mb}-{end_mb}Mb")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="left", va="bottom",
        rotation_mode="anchor", fontsize=PUB_TICK_FONTSIZE)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=16)

    # Y-axis: show segment boundary bin numbers, but stagger labels that are
    # too close (e.g. 233 and 240) rather than shrinking the font.
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
        ax.text(x_pos,b,str(b),transform=trans,ha="right",va="center",fontsize=PUB_TICK_FONTSIZE,clip_on=False)
        previous_b = b

    n = len(path_df)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

def plot_class_barplot(class_summary, outpath, class_col="upgraded_peak_class"):
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_RECT)
    y = np.arange(len(class_summary))
    colors = [CLASS_COLOR_MAP.get(c, "gray") for c in class_summary[class_col]]
    labels = [display_label(c) for c in class_summary[class_col]]
    ax.barh(y, class_summary["n"].values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PUB_TICK_FONTSIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Number of significant contacts", fontsize=PUB_FONTSIZE)
    fig.suptitle("Primary contact-label summary", fontsize=PUB_FONTSIZE, y=0.97)
    _apply_pub_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=1.2)
    save_pub_figure(fig, outpath)
    plt.close(fig)

def plot_ref_vs_ecdna(df, outpath):
    plt.figure(figsize=PUB_FIGSIZE_RECT)
    for cls, grp in df.groupby("upgraded_peak_class", sort=False):
        x = np.maximum(grp["ecDNA_dist"].values, 1.0)
        y = grp["ref_dist"].values
        mask = np.isfinite(x) & np.isfinite(y)

        if np.any(mask):
            plt.scatter(x[mask],np.maximum(y[mask], 1.0),s=20,alpha=0.75,label=display_label(cls), color=CLASS_COLOR_MAP.get(cls, "gray"))
       # plt.scatter(
       #     np.maximum(grp["ecDNA_dist"].values, 1.0),
       #     np.maximum(grp["ref_dist"].values, 1.0),
       #     s=20,
        #    alpha=0.75,
        #    label=cls,
         #   color=CLASS_COLOR_MAP.get(cls, "gray")
       # )

    mx = max(float(np.nanmax(np.maximum(df["ecDNA_dist"].values, 1.0))), float(np.nanmax(np.maximum(df["ref_dist"].values, 1.0))))
    plt.plot([1, mx], [1, mx], linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("ecDNA path distance (bp)", fontsize=PUB_FONTSIZE)
    plt.ylabel("Reference genomic distance (bp)", fontsize=PUB_FONTSIZE)
    plt.title("Reference vs ecDNA path distance\n(for same-reference-chromosome contacts)", fontsize=PUB_FONTSIZE, pad=14)
    plt.legend(fontsize=PUB_LEGEND_FONTSIZE)
    plt.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    plt.tight_layout(pad=1.2)
    fig = plt.gcf()
    save_pub_figure(fig, outpath)
    plt.close(fig)

def plot_discordance_vs_oe(df, outpath):
    plt.figure(figsize=PUB_FIGSIZE_RECT)
    for cls, grp in df.groupby("upgraded_peak_class", sort=False):
        x = grp["discord"].values
        y = np.maximum(grp["obs_over_exp_raw"].values, 1e-8)
        mask = np.isfinite(x) & np.isfinite(y)

        if np.any(mask):
            plt.scatter(x[mask],y[mask],s=20, alpha=0.75, label=display_label(cls), color=CLASS_COLOR_MAP.get(cls, "gray"))

    plt.yscale("log")
    plt.xlabel("Distance-discordance score", fontsize=PUB_FONTSIZE)
    plt.ylabel(r"Raw O/E", fontsize=PUB_FONTSIZE)
    plt.title("Discordance versus raw O/E", fontsize=PUB_FONTSIZE, pad=14)
    plt.legend(fontsize=PUB_LEGEND_FONTSIZE)
    plt.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    plt.tight_layout(pad=1.2)
    fig = plt.gcf()
    save_pub_figure(fig, outpath)
    plt.close(fig)

def plot_neighbors_vs_oe(df, outpath):
    plt.figure(figsize=PUB_FIGSIZE_RECT)
    for cls, grp in df.groupby("upgraded_peak_class", sort=False):
        plt.scatter( grp["n_sig_neighbors_w5"].values, np.maximum(grp["obs_over_exp_raw"].values, 1e-8),
            s=20, alpha=0.75, label=display_label(cls), color=CLASS_COLOR_MAP.get(cls, "gray"))

    plt.yscale("log")
    plt.xlabel("Nearby significant contacts (5-bin window)", fontsize=PUB_FONTSIZE)
    plt.ylabel(r"Raw O/E", fontsize=PUB_FONTSIZE)
    plt.title("Nearby significant contacts versus raw O/E", fontsize=PUB_FONTSIZE, pad=14)
    plt.legend(fontsize=PUB_LEGEND_FONTSIZE)
    plt.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    plt.tight_layout(pad=1.2)
    fig = plt.gcf()
    save_pub_figure(fig, outpath)
    plt.close(fig)

def plot_class_heatmap(matrix,df,outpath, title="Annotated significant peaks", log_base=2, path_df=None):
    mat = np.array(matrix, dtype=float, copy=True)
    plot_mat = np.ma.masked_where(mat <= 0, mat)
    plot_mat = np.ma.log2(plot_mat) if log_base == 2 else np.ma.log10(plot_mat)

    finite_vals = np.asarray(plot_mat.compressed(), dtype=float)
    if finite_vals.size > 0:
        vmin = np.percentile(finite_vals, 2)
        vmax = np.percentile(finite_vals, 99.8)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

   
    # A square axes prevents any empty strip from appearing inside the heatmap.
    fig, ax = plt.subplots(figsize=PUB_FIGSIZE_HEATMAP)
    ax.set_box_aspect(1)

    im = ax.imshow(plot_mat, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                   origin="upper", aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_2(X_{ij})$" if log_base == 2 else r"$\log_{10}(X_{ij})$",
                   fontsize=PUB_CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)

    for cls, grp in df.groupby("upgraded_peak_class", sort=False):
        ax.scatter(grp["bin2"].values, grp["bin1"].values, s=22, facecolors="none",edgecolors=CLASS_COLOR_MAP.get(cls, "gray"), linewidths=1.0)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=CLASS_COLOR_MAP.get(cls, "gray"), markersize=8, linewidth=0,
               label=display_label(cls))
        for cls in df["upgraded_peak_class"].drop_duplicates().tolist() ]
    ax.legend(handles=handles, loc="lower left", fontsize=PUB_LEGEND_FONTSIZE, frameon=True, facecolor="white", framealpha=0.95)

    ax.set_xlabel("Path bin", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Path bin", fontsize=PUB_FONTSIZE, labelpad=58)
    fig.suptitle(title, fontsize=PUB_FONTSIZE, y=0.97)
    _add_path_annotations(ax, path_df)
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_box_aspect(1)
    _apply_pub_axes(ax)
    # Reserve fixed space for 45-degree genomic labels and the title while
    # keeping the exported heatmap canvas exactly 10 x 10 inches.
    fig.subplots_adjust(left=0.16, right=0.87, bottom=0.11, top=0.68)
    save_pub_figure(fig, outpath)
    plt.close(fig)

def plot_block_heatmap(matrix, pixel_df, block_df, outpath, title="Layer 3 primary labels for merged contact blocks", draw_pixels=False, draw_boxes=True, draw_centers=True, figsize=PUB_FIGSIZE_HEATMAP, legend_fontsize=PUB_LEGEND_FONTSIZE, min_box_size=5, path_df=None):
    mat = np.array(matrix, dtype=float, copy=True)
    plot_mat = np.ma.masked_where(mat <= 0, mat)
    plot_mat = np.ma.log2(plot_mat)

    finite_vals = np.asarray(plot_mat.compressed(), dtype=float)
    if finite_vals.size > 0:
        vmin = np.percentile(finite_vals, 2)
        vmax = np.percentile(finite_vals, 99.8)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_box_aspect(1)

    im = ax.imshow(plot_mat, cmap="Greys", vmin=vmin, vmax=vmax,
                   origin="upper", aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_2(X_{ij})$", fontsize=PUB_CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE)

    label_col = "upgraded_peak_block_class" if "upgraded_peak_block_class" in block_df.columns else "primary_blk_label"

    for _, row in block_df.iterrows():
        cls = str(row.get(label_col, "unclassified_significant_contact"))
        color = CLASS_COLOR_MAP.get(cls, "#666666")
        x0 = float(row["blk_bin2_min"])
        y0 = float(row["blk_bin1_min"])
        width = float(row["blk_bin2_max"] - row["blk_bin2_min"] + 1)
        height = float(row["blk_bin1_max"] - row["blk_bin1_min"] + 1)
        cx = float(row.get("blk_bin2_center", x0 + width / 2.0))
        cy = float(row.get("blk_bin1_center", y0 + height / 2.0))

        if draw_boxes:
            box_w = max(width, float(min_box_size))
            box_h = max(height, float(min_box_size))
            ax.add_patch(Rectangle(
                (cx - box_w / 2.0, cy - box_h / 2.0), box_w, box_h, fill=False,
                edgecolor=color, linewidth=1.8
            ))
        if draw_centers:
            ax.scatter([cx], [cy], s=28, marker="s", facecolors=color,
                       edgecolors="black", linewidths=0.5, zorder=5)

    if draw_pixels and pixel_df is not None and len(pixel_df) > 0:
        pix_label_col = "upgraded_peak_class" if "upgraded_peak_class" in pixel_df.columns else "primary_label"
        for cls, grp in pixel_df.groupby(pix_label_col, sort=False):
            ax.scatter(grp["bin2"].values, grp["bin1"].values, s=12, facecolors="none",
                       edgecolors=CLASS_COLOR_MAP.get(cls, "#666666"), linewidths=0.6, alpha=0.65)

    classes = block_df[label_col].drop_duplicates().tolist() if len(block_df) else []
    handles = []
    for cls in classes:
        n_cls = int((block_df[label_col] == cls).sum())
        handles.append(Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=CLASS_COLOR_MAP.get(cls, "#666666"), markeredgecolor="black",
            markersize=8, linewidth=0, label=f"{display_block_label(cls)} (n={n_cls})"
        ))
    ax.legend(handles=handles, loc="lower left", fontsize=legend_fontsize, frameon=True, facecolor="white", framealpha=0.95)

    ax.set_xlabel("Path bin", fontsize=PUB_FONTSIZE)
    ax.set_ylabel("Path bin", fontsize=PUB_FONTSIZE, labelpad=58)
    fig.suptitle(title, fontsize=PUB_FONTSIZE, y=0.97)
    _add_path_annotations(ax, path_df)
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_box_aspect(1)
    _apply_pub_axes(ax)
    fig.subplots_adjust(left=0.16, right=0.87, bottom=0.11, top=0.68)

    save_pub_figure(fig, outpath)
    plt.close(fig)

def plot_anchor_recurrence(df, outpath):
    if "n_same_anchor_peaks_w5" not in df.columns:
        return
    plt.figure(figsize=PUB_FIGSIZE_RECT)
    for cls, grp in df.groupby("upgraded_peak_class", sort=False):
        plt.scatter(grp["n_same_anchor_peaks_w5"].values, np.maximum(grp["obs_over_exp_raw"].values, 1e-8), s=24,
            alpha=0.75, label=display_label(cls), color=CLASS_COLOR_MAP.get(cls, "gray"))
    plt.yscale("log")
    plt.xlabel("Same-anchor recurrence count (window=5 bins)", fontsize=PUB_FONTSIZE)
    plt.ylabel(r"Raw O/E", fontsize=PUB_FONTSIZE)
    plt.title("Same-anchor recurrence versus raw O/E", fontsize=PUB_FONTSIZE, pad=14)
    plt.legend(fontsize=PUB_LEGEND_FONTSIZE)
    plt.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    plt.tight_layout(pad=1.2)
    fig = plt.gcf()
    save_pub_figure(fig, outpath)
    plt.close(fig)

def plot_block_recurrence(block_df, outpath):
    if "n_same_anchor_blk_w5" not in block_df.columns:
        return
    y_col = "max_obs_over_exp_raw" if "max_obs_over_exp_raw" in block_df.columns else None
    if y_col is None:
        return
    label_col = "upgraded_peak_block_class" if "upgraded_peak_block_class" in block_df.columns else "primary_blk_label"
    plt.figure(figsize=PUB_FIGSIZE_RECT)
    for cls, grp in block_df.groupby(label_col, sort=False):
        plt.scatter(grp["n_same_anchor_blk_w5"].values, np.maximum(grp[y_col].values, 1e-8), s=28, alpha=0.75, label=display_block_label(cls), color=CLASS_COLOR_MAP.get(cls, "gray"))
    plt.yscale("log")
    plt.xlabel("Same-anchor block recurrence (5-bin window)", fontsize=PUB_FONTSIZE)
    plt.ylabel("Maximum raw O/E in block", fontsize=PUB_FONTSIZE)
    plt.title("Block-level same-anchor recurrence versus raw O/E", fontsize=PUB_FONTSIZE, pad=14)
    plt.legend(fontsize=PUB_LEGEND_FONTSIZE)
    plt.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    plt.tight_layout(pad=1.2)
    fig = plt.gcf()
    save_pub_figure(fig, outpath)
    plt.close(fig)

def plot_hub_vs_oe(df, outpath):
    plt.figure(figsize=PUB_FIGSIZE_RECT)

    for hub_state, grp in df.groupby("is_hub_associated", sort=False):
        label = "Neighbor-threshold positive" if hub_state else "Neighbor-threshold negative"
        plt.scatter(grp["n_sig_neighbors_w5"].values,np.maximum(grp["obs_over_exp_raw"].values, 1e-8),s=20,alpha=0.75,label=label)

    plt.yscale("log")
    plt.xlabel("Nearby significant contacts (5-bin window)", fontsize=PUB_FONTSIZE)
    plt.ylabel(r"Raw O/E", fontsize=PUB_FONTSIZE)
    plt.title("Nearby significant-contact count versus raw O/E", fontsize=PUB_FONTSIZE, pad=14)
    plt.legend(fontsize=PUB_LEGEND_FONTSIZE)
    plt.tick_params(axis="both", labelsize=PUB_TICK_FONTSIZE)
    plt.tight_layout(pad=1.2)
    fig = plt.gcf()
    save_pub_figure(fig, outpath)
    plt.close(fig)
