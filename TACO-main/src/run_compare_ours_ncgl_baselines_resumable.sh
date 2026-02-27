#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="../results/OURS_compare_ncgl_baselines_${STAMP}"
mkdir -p "$OUT_DIR"

# Run settings
NUM_EPOCHS=${NUM_EPOCHS:-50}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

# Methods: include TACO's method (DYGRA) plus requested NCGL baselines.
METHODS=${METHODS:-"DYGRA BARE GEM ERGNN TWP JOINT"}

# Hyper-params for baselines (can override via env)
GEM_N_MEMORIES=${GEM_N_MEMORIES:-100}
GEM_MARGIN=${GEM_MARGIN:-0.5}
ERGNN_BUDGET=${ERGNN_BUDGET:-20}
TWP_LAMBDA_L=${TWP_LAMBDA_L:-10000.0}
TWP_LAMBDA_T=${TWP_LAMBDA_T:-10000.0}
TWP_BETA=${TWP_BETA:-0.01}

# DYGRA-only MMD alignment opts (opt-in; default off)
DYGRA_MMD_LAMBDA=${DYGRA_MMD_LAMBDA:-0.0}
DYGRA_MMD_SAMPLE=${DYGRA_MMD_SAMPLE:-256}
DYGRA_MMD_BW=${DYGRA_MMD_BW:-1.0}

export OUT_DIR NUM_EPOCHS SEEDS METHODS \
  GEM_N_MEMORIES GEM_MARGIN ERGNN_BUDGET TWP_LAMBDA_L TWP_LAMBDA_T TWP_BETA \
  DYGRA_MMD_LAMBDA DYGRA_MMD_SAMPLE DYGRA_MMD_BW

# Reuse the resume script (it will skip existing seeds and write SUMMARY.md + zip).
./resume_ours_compare_moe_topo_resumable.sh

echo "Done. OUT_DIR=$OUT_DIR"