#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="../.venv_graphcls/bin/python"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="../results/DBLP_moe_topo_compare_1run_20eps_${STAMP}"
mkdir -p "$OUT_DIR"

COMMON=(
  --Dataset DBLP
  --gnn GCN
  --gpu -1
  --num_runs 1
  --num_epochs 20
  --lr 1e-2
  --weight-decay 0
  --hidden_dim 48
  --use_moe True
  --parallel_fusion True
  --use_topology_feature
  --topology_pd_dir ../topology_import_20260220/output
  --topology_group 0
  --topology_window 5
  --topology_bins 32
  --topology_max_end 98
  --topo_seed 42
  --reduction_rate 0.5
  --buffer_size 200
)

run_one () {
  local METHOD="$1"; shift
  echo "\n==================== ${METHOD} ====================" | tee "$OUT_DIR/${METHOD}.log"
  echo "$PY -W ignore train.py --method ${METHOD} ${COMMON[*]} $*" | tee -a "$OUT_DIR/${METHOD}.log"
  $PY -W ignore train.py --method "$METHOD" "${COMMON[@]}" "$@" 2>&1 | tee -a "$OUT_DIR/${METHOD}.log"

  # Persist artifacts
  cp -f "../results/DBLP/${METHOD}_GCN_DBLP_reduction_0.5.csv" "$OUT_DIR/" || true
  cp -f "results/${METHOD}_GCN_DBLP_reduction_0.5" "$OUT_DIR/" || true
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

def parse_f1_af(csv_path: str, num_task=10):
    lines = open(csv_path,'r',encoding='utf-8',errors='ignore').read().splitlines()
    # utils.record_results writes header as 'f1' but some environments show it as 'f,1'
    key = 'f1'
    if key not in lines:
        key = 'f,1'
    idx = lines.index(key)
    last = lines[idx + (num_task + 1)]
    last_cell = last.split(',')[-1]
    return float(last_cell.replace('%','').split('±')[0])

rows=[]
for m in methods:
    pkl = os.path.join(out, f"{m}_GCN_DBLP_reduction_0.5")
    csv = os.path.join(out, f"{m}_GCN_DBLP_reduction_0.5.csv")
    acc = bac = f1 = None
    if os.path.exists(pkl):
        with open(pkl,'rb') as f:
            acc, bac, f1 = pickle.load(f)
        f1_ap = float(np.mean(f1))*100.0
    else:
        f1_ap = float('nan')
    f1_af = parse_f1_af(csv) if os.path.exists(csv) else float('nan')
    rows.append((m,f1_ap,f1_af))

md=[]
md.append(f"# DBLP (GCN + parallel MoE + topology) 1run×20eps compare")
md.append("")
md.append("Config: Dataset=DBLP, gnn=GCN, gpu=-1, num_runs=1, num_epochs=20, lr=1e-2")
md.append("Architecture: --use_moe True --parallel_fusion True --use_topology_feature (topology_group=0)")
md.append("Note: JOINT is oracle upper bound (uses all tasks).")
md.append("")
md.append("| Method | F1-AP (%) ↑ | F1-AF (%) ↓ |")
md.append("|---|---:|---:|")
for m,ap,af in rows:
    md.append(f"| {m} | {ap:.2f} | {af:.2f} |")
md.append("")
open(os.path.join(out,'SUMMARY.md'),'w').write("\n".join(md))
print(os.path.join(out,'SUMMARY.md'))
PY

echo "Done. OUT_DIR=$OUT_DIR"