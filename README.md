# CADET

Copy-aware deconvolution of Hi-C contacts on ecDNA. Duplicated ecDNA segments map to the same reference region, so
their contacts collapse in reference space. CADET redistributes these contacts along the ecDNA path given in a BED file
(Layer 1), calls enriched contacts (Layer 2) and annotates them (Layers 3–8).

| Layer | Script | Step |
|---|---|---|
| 1 | `cadet_layer1.py` | copy-aware Poisson EM deconvolution |
| 2 | `cadet_layer2.py` | distance-stratified background, local enrichment, FDR |
| 3 | `prepare_layer3_inputs.py`, `cadet_layer3.py` | merge contacts into blocks |
| 4 | `cadet_layer4.py` | group blocks into modules |
| 5 | `cadet_layer5.py` | module priority |
| 6 | `cadet_layer6.py` | genes (GENCODE v49), oncogenes (OncoKB), cCREs (ENCODE) |
| 7 | `cadet_layer7.py` | module interpretation and targets |
| 8 | `cadet_layer8.py`, `cadet_layer8_circos.py` | target recurrence vs shuffled nulls; circos plot |

`requirements.txt` lists the package versions we tested. With pip, install `hic-straw` from bioconda or build it separately. It's only needed to read `.hic` files.


## Installation

```bash
git clone https://github.com/Biswanath2410/cadet.git
cd cadet
conda env create -f environment.yml
conda activate cadet
bash download_references.sh      # OncoKB, GENCODE and ENCODE files for Layer 6
```
## Inputs

**ecDNA BED:** one row per segment, in path order, with a header:

```
chrom	start	end	orientation
chr8	127000000	127500000	+
chr8	127600000	127800000	-
```

**Hi-C:** a `.hic` file for the same sample.

## Running

`run_full_pipeline.sh` runs every layer in order. A typical run is:

```bash
BED=inputs/sample_ecDNA.bed \
HIC=inputs/sample.hic \
RES=5000 \
RUN_NAME=sample \
bash run_full_pipeline.sh
```

Results go to `runs/<RUN_NAME>/` (`part1/` Layer 1, `part2_discovery/` and `part2_fdr/` Layer 2,
`cadet_final_results/` Layers 3–8).

Options: `STAGES=layer6,layer7` runs only some stages, `SKIP_CIRCOS=1` skips the circos plot,
`Y_CACHED=runs/<run>/part1/observed_Y.txt` reuses an earlier contact matrix. Each script can also be run on its own
(`python scripts/<script>.py -h`).

The OncoKB list changes over time; the download date is saved in `reference_download_date.txt`.

## Simulation

`simulation/` has the Layer 1 benchmark (see `simulation/README.md`).

## Citation

Chowdhury, B. (2026). Copy aware deconvolution of Hi-C interactions in extrachromosomal DNA using CADET.
Research Square. https://doi.org/10.21203/rs.3.rs-11142844/v1

## License

MIT. See [LICENSE](LICENSE).
