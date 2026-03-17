#!/bin/bash

RAW_DIR="./data/raw"

for file in "$RAW_DIR"/*; do
    filename=$(basename "$file")
    python3 legacy/test.py -n "$filename" -c --debug
done
