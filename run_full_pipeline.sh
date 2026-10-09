#!/usr/bin/env bash
# Run all CADET layers.
# Usage: BED=sample_ecDNA.bed HIC=sample.hic RUN_NAME=sample bash run_full_pipeline.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
S="$PROJECT_ROOT/scripts"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
BED="${BED:?set BED to the ecDNA path BED file}"
HIC="${HIC:-}"                         # .hic file, or
Y_CACHED="${Y_CACHED:-}"               # observed_Y.txt from an earlier run
RES="${RES:-5000}"
STAGES="${STAGES:-}"                   # e.g. layer6,layer7 (default: all)
SKIP_CIRCOS="${SKIP_CIRCOS:-0}"
SISTER_OFFSET="${SISTER_OFFSET:-}"
SISTER_TOLERANCE="${SISTER_TOLERANCE:-1}"

# Layer 2
LOCAL_FC_THRESHOLD="${LOCAL_FC_THRESHOLD:-1.2}"
DISCOVERY_TOPK="${DISCOVERY_TOPK:-50}"
FDR_ALPHA="${FDR_ALPHA:-0.05}"
FDR_UNIVERSE="${FDR_UNIVERSE:-focal}"
USE_SHARED_ICE="${USE_SHARED_ICE:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

RUN="$RUNS_ROOT/$RUN_NAME"
P1="$RUN/part1"
P2="$RUN/part2_discovery"     # used by Layers 3-8
P2_FDR="$RUN/part2_fdr"       # strict focal-FDR set
OUT="$RUN/cadet_final_results"
mkdir -p "$RUN/logs" "$P1" "$P2" "$P2_FDR"
for ref in oncoKB gencode encode; do ln -sfn "$PROJECT_ROOT/$ref" "$RUNS_ROOT/$ref"; done

run_stage() {
    local name="$1"; shift
    if [[ -n "$STAGES" && ",$STAGES," != *",$name,"* ]]; then return; fi
    echo; echo "[$(date +%H:%M:%S)] STAGE: $name"
    "$@" 2>&1 | tee "$RUN/logs/$name.log"
}

part1() {
    local input=(--hic "$HIC")
    [[ -n "$Y_CACHED" ]] && input=(--y_cached "$Y_CACHED")
    python "$S/cadet_layer1.py" --bed "$BED" --res "$RES" --out "$P1" "${input[@]}"
}

part2() {
    local cmd=(python "$S/cadet_layer2.py" --x "$P1/latent_X_symmetric.txt" --b "$P1/lambda_symmetric.txt"
        --path_bins "$P1/path_bins.tsv" --res "$RES" --fdr_universe "$FDR_UNIVERSE" --local_fc_threshold "$LOCAL_FC_THRESHOLD")
    [[ -n "$SISTER_OFFSET" ]] && cmd+=(--exclude_sister_offset "$SISTER_OFFSET" --sister_tolerance "$SISTER_TOLERANCE")
    [[ "$USE_SHARED_ICE" != "0" ]] && cmd+=(--use_shared_ice)
    echo "Part 2A: discovery candidates for Layers 3-8"
    "${cmd[@]}" --out "$P2" --candidate_topk_total "$DISCOVERY_TOPK" --no_require_global_fdr
    echo "Part 2B: strict focal-FDR set"
    "${cmd[@]}" --out "$P2_FDR" --candidate_topk_total 0 --fdr_alpha "$FDR_ALPHA"
}

prepare_layer3() {
    python "$S/prepare_layer3_inputs.py" --part2_csv "$P2/final_significant_interactions.csv" \
        --path_bins "$P1/path_bins.tsv" --latent_matrix "$P1/latent_X_symmetric.txt" --outdir "$OUT" --res "$RES"
}

layer3() {
    python "$S/cadet_layer3.py" --input_dir "$OUT" --merge_radius_bin1 2 --merge_radius_bin2 2 \
        --loop_anchor_neighbor_max 2 --stripe_span_min_bins 12 --stripe_aspect_min 4 --compact_block_max_bins 6
}

layer4() {
    python "$S/cadet_layer4.py" --input_dir "$OUT" --pairspace_tolerance 10 --shared_anchor_tolerance 2 \
        --min_anchor_support 2 --legend_fontsize 18
}

layer5() { python "$S/cadet_layer5.py" --input_dir "$OUT" --top_n 15 --plot_top_n 25; }

layer6() {
    python "$S/cadet_layer6.py" --input_dir "$OUT" \
        --oncokb_tsv "$PROJECT_ROOT/oncoKB/cancerGeneList.tsv" \
        --gencode_gtf "$PROJECT_ROOT/gencode/gencode.v49.basic.annotation.gtf" \
        --ccres_bed "$PROJECT_ROOT/encode/GRCh38-cCREs.bed" \
        --local_window_bp 25000 --gene_dist_bp 5000 --oncogene_dist_bp 15000 \
        --enhancer_weak_min 1 --enhancer_moderate_min 3 --enhancer_strong_min 5 --plot_top_n 20
}

layer7() { python "$S/cadet_layer7.py" --input_dir "$OUT" --top_n 20; }

layer8() { python "$S/cadet_layer8.py" --input_dir "$OUT" --n_iter 1000 --seed 7 --top_n_targets 20; }

circos() {
    python "$S/cadet_layer8_circos.py" --input_dir "$OUT" --top_n 0 --top_modules 0 --label_top 999 \
        --coord_label_mode range --min_sector_label_bp 0 --color_by module
}

echo "CADET: run $RUN_NAME, resolution $RES, stages ${STAGES:-all}"
start=$(date +%s)
for stage in part1 part2 prepare_layer3 layer3 layer4 layer5 layer6 layer7 layer8; do
    run_stage "$stage" "$stage"
done
[[ "$SKIP_CIRCOS" == "0" ]] && run_stage circos circos
echo; echo "Done in $(( $(date +%s) - start ))s. Outputs: $RUN"
