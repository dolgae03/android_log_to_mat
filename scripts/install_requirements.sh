#!/bin/bash

set -eu

CONDA_BIN="${CONDA_BIN:-/home/user/anaconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-gnss-parser}"

if [ ! -x "$CONDA_BIN" ]; then
    echo "conda not found at $CONDA_BIN"
    echo "Set CONDA_BIN to your conda executable path and rerun."
    exit 1
fi

"$CONDA_BIN" create -y -n "$ENV_NAME" python=3.10 pandas

cat <<EOF
Conda environment created: $ENV_NAME

Run the parser with:
  PYTHONPATH=./src /home/user/mskim/.conda/envs/$ENV_NAME/bin/python -m gnss_txt_parser
EOF
