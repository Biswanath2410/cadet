#!/usr/bin/env bash
# Download the reference files used by Layer 6 into the project folder
set -euo pipefail

mkdir -p oncoKB gencode encode

# OncoKB cancer gene list 
curl -L -o oncoKB/cancerGeneList.tsv \
    https://www.oncokb.org/api/v1/utils/cancerGeneList.txt

# GENCODE release 49
curl -L -o gencode/gencode.v49.basic.annotation.gtf.gz \
    https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.basic.annotation.gtf.gz
gunzip -f gencode/gencode.v49.basic.annotation.gtf.gz

# ENCODE SCREEN V3 cCREs (GRCh38)
curl -L -o encode/GRCh38-cCREs.bed \
    https://downloads.wenglab.org/V3/GRCh38-cCREs.bed
#   https://screen.encodeproject.org
echo "Downloaded on $(date +%Y-%m-%d)" > reference_download_date.txt
echo "Done."
