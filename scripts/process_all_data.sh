#!/bin/bash

set -eu

RAW_DIR="${1:-./data/raw}"
OUT_DIR="${2:-./data/results/tsv}"
PYTHONPATH_PREFIX="${PYTHONPATH:-}"

mkdir -p "$OUT_DIR"

for file in "$RAW_DIR"/*.txt; do
    [ -e "$file" ] || continue
    filename=$(basename "$file")
    stem="${filename%.txt}"
    PYTHONPATH="./src${PYTHONPATH_PREFIX:+:$PYTHONPATH_PREFIX}" python3 -m gnss_txt_parser "$file" -o "$OUT_DIR/${stem}.tsv"
done
