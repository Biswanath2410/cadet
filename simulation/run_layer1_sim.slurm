#!/bin/bash
#SBATCH --job-name=cadet_L1_sim
#SBATCH --time=04:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --output=cadet_L1_%j.out
#SBATCH --error=cadet_L1_%j.err

set -euo pipefail

python benchmark_layer1.py --out_root bench_layer1 --n_seeds 20 --anchor_window_bins 50
python stress_test_pi_layer1.py --out pi_stress_layer1.tsv --n_seeds 50 --anchor_window_bins 6
python plot_layer1_summary.py \
  --bench bench_layer1/all_runs.tsv \
  --stress pi_stress_layer1.tsv \
  --sim bench_layer1/sims/C_one_dup_asymmetric_seed0 \
  --out layer1_simulation_summary.png
python plot_layer1_heatmaps.py \
  --sim_root bench_layer1/sims \
  --out layer1_heatmaps.png

python plot_alpha_family_recovery.py \
  --bench bench_layer1/all_runs.tsv \
  --out layer1_alpha_family_recovery.png

python plot_layer1_additional_figures.py \
  --bench bench_layer1/all_runs.tsv \
  --sim_root bench_layer1/sims \
  --out_heatmaps layer1_heatmaps_complex.png \
  --out_diagnostics layer1_seedwise_diagnostics.png
