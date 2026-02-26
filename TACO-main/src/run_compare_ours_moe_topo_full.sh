#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="../.venv_graphcls/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="../results/OURS_compare_moe_topo_${STAMP}"
mkdir -p "$OUT_DIR"

# Follow the ablation script defaults
NUM_RUNS=${NUM_RUNS:-10}
NUM_EPOCHS=${NUM_EPOCHS:-50}

COMMON=(
  --Dataset OURS
  --gnn GCN
  --reduction_rate 0.5
  --buffer_size 20
  --num_runs "$NUM_RUNS"
  --num_epochs "$NUM_EPOCHS"
  --gpu -1

  --use_moe True
  --parallel_fusion True

  --use_topology_feature
  --topo_alpha 0.1
  --topo_trainable_fusion
  --topology_pd_dir ../topology_import_20260220/output
  --topology_group 0
  --topology_window 5
  --topology_bins 32
  --topology_max_end 98
  --topo_seed 42
)

run_one () {
  local METHOD="$1"; shift
  echo "\n==================== ${METHOD} ====================" | tee "$OUT_DIR/${METHOD}.log"
  echo "$PY -W ignore train.py --method ${METHOD} ${COMMON[*]} $*" | tee -a "$OUT_DIR/${METHOD}.log"
  $PY -W ignore train.py --method "$METHOD" "${COMMON[@]}" "$@" 2>&1 | tee -a "$OUT_DIR/${METHOD}.log"

  # Copy artifacts to OUT_DIR
  cp -f "results/${METHOD}_GCN_OURS_reduction_0.5" "$OUT_DIR/" || true
  cp -f "../results/OURS/${METHOD}_GCN_OURS_reduction_0.5.csv" "$OUT_DIR/" || true
}

run_one DYGRA
run_one FINETUNE
run_one SIMPLE_REG --simple_reg_lambda 1e-3
run_one JOINT

# Summarize into markdown
$PY - "$OUT_DIR" <<'PY'
import os, sys, pickle
import numpy as np

out = sys.argv[1]
methods = ["DYGRA","FINETUNE","SIMPLE_REG","JOINT"]

# stats
rows=[]
for m in methods:
    pkl = os.path.join(out, f"{m}_GCN_OURS_reduction_0.5")
    acc=bac=f1=None
    if os.path.exists(pkl):
        with open(pkl,'rb') as f:
            acc,bac,f1=pickle.load(f)
        rows.append((m, float(np.mean(acc))*100, float(np.std(acc))*100,
                    float(np.mean(bac))*100, float(np.std(bac))*100,
                    float(np.mean(f1))*100, float(np.std(f1))*100))
    else:
        rows.append((m, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')))

md=[]
md.append('# OURS (GCN + parallel MoE + topology) compare')
md.append('')
md.append('Settings: reduction_rate=0.5, buffer_size=20, topo_trainable_fusion, topo_alpha=0.1')
md.append('')
md.append('| Method | ACC (mean±std) | BAC (mean±std) | F1 (mean±std) |')
md.append('|---|---:|---:|---:|')
for m,acc_m,acc_s,bac_m,bac_s,f1_m,f1_s in rows:
    md.append(f'| {m} | {acc_m:.2f} ± {acc_s:.2f} | {bac_m:.2f} ± {bac_s:.2f} | {f1_m:.2f} ± {f1_s:.2f} |')
md.append('')
md.append('Note: JOINT is oracle upper bound (uses all tasks).')
open(os.path.join(out,'SUMMARY.md'),'w').write('\n'.join(md))
print(os.path.join(out,'SUMMARY.md'))
PY

# Zip outputs
cd "$OUT_DIR/.."
zip -qr "$(basename "$OUT_DIR").zip" "$(basename "$OUT_DIR")"
echo "Done. OUT_DIR=$OUT_DIR"
