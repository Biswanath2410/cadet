# Layer 1 simulation

Benchmark of CADET Layer 1 (`../scripts/cadet_layer1.py`) on simulated ecDNA Hi-C.

Latent contacts are drawn from `lambda[i,j] = pi[i] * pi[j] * alpha[family(i,j)] * f(d_ij)` and summed over copies to
give the observed matrix Y. Five scenarios (A: no duplication; B/C: one duplication; D/E: two duplications;
B and D with near-equal copies, C and E with unequal copies) are run with 20 seeds in three modes: `cadet_pi_on`,
`cadet_pi_off` and `naive` (backbone only, no EM). `stress_test_pi_layer1.py` checks that pi alone moves allocation to
the correct copy when the backbone gives a 50:50 split.

## Running

```bash
bash run_layer1_sim.slurm        # or sbatch; a few minutes on one CPU
```

## Outputs

Results from our run are in `results/`.

- `bench_layer1/all_runs.tsv`: one row per scenario, seed and mode
- `bench_layer1/summary_by_scenario_mode.tsv`: mean and SD
- `pi_stress_layer1.tsv`, `pi_stress_layer1_summary.tsv`
- `layer1_*.png`: figures

Main metrics: `dup_x_pearson_log` (X on duplicated pairs), `family_l1_mean`, `dominant_pair_acc`,
`recollapse_err` (should be ~0) and the per-family alpha correlations and AUROC (`alpha_r_*`, `alpha_auroc_ambig`).
