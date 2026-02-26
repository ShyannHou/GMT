#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="../.venv_graphcls/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="../results/OURS_compare_moe_topo_resumable_${STAMP}"
mkdir -p "$OUT_DIR"

NUM_EPOCHS=${NUM_EPOCHS:-50}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

COMMON=(
  --Dataset OURS
  --gnn GCN
  --reduction_rate 0.5
  --buffer_size 20
  --gpu -1

  --num_runs 1
  --num_epochs "$NUM_EPOCHS"

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
  local EXTRA=("$@")

  for seed in $SEEDS; do
    echo "\n==================== ${METHOD} seed=${seed} ====================" | tee -a "$OUT_DIR/RUN.log"
    echo "$PY -W ignore train.py --method ${METHOD} --run_offset ${seed} ${COMMON[*]} ${EXTRA[*]}" | tee -a "$OUT_DIR/RUN.log"

    $PY -W ignore train.py --method "$METHOD" --run_offset "$seed" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$OUT_DIR/${METHOD}_seed${seed}.log"

    # copy artifacts (train.py writes fixed filenames)
    cp -f "results/${METHOD}_GCN_OURS_reduction_0.5" "$OUT_DIR/${METHOD}_seed${seed}.pkl"
    cp -f "../results/OURS/${METHOD}_GCN_OURS_reduction_0.5.csv" "$OUT_DIR/${METHOD}_seed${seed}.csv"
  done
}

run_one DYGRA
run_one FINETUNE
run_one SIMPLE_REG --simple_reg_lambda 1e-3
run_one JOINT

# aggregate summary
$PY - "$OUT_DIR" <<'PY'
import os, sys, pickle
import numpy as np

out=sys.argv[1]
methods=["DYGRA","FINETUNE","SIMPLE_REG","JOINT"]
seeds=list(range(10))

# detect num_task from statistics
import pickle as pkl
with open(os.path.join(os.path.dirname(out), '..', 'data', 'OURS', 'statistics'), 'rb') as f:
    num_task, num_class = pkl.load(f)


def f1_af_from_csv(path):
    lines=open(path,'r',encoding='utf-8',errors='ignore').read().splitlines()
    key='f1'
    if key not in lines:
        key='f,1'
    idx=lines.index(key)
    last=lines[idx+(num_task+1)]
    cell=last.split(',')[-1]
    return float(cell.replace('%','').split('±')[0])

rows=[]
for m in methods:
    accs=[]; bacs=[]; f1s=[]; afs=[]
    for s in seeds:
        pkl_path=os.path.join(out,f"{m}_seed{s}.pkl")
        csv_path=os.path.join(out,f"{m}_seed{s}.csv")
        acc,bac,f1=pickle.load(open(pkl_path,'rb'))
        accs.append(float(np.mean(acc))*100)
        bacs.append(float(np.mean(bac))*100)
        f1s.append(float(np.mean(f1))*100)
        afs.append(f1_af_from_csv(csv_path))
    rows.append((m,
                 np.mean(accs), np.std(accs),
                 np.mean(bacs), np.std(bacs),
                 np.mean(f1s), np.std(f1s),
                 np.mean(afs), np.std(afs)))

md=[]
md.append('# OURS (GCN + parallel MoE + topology) compare (resumable 10 seeds)')
md.append('')
md.append(f'- epochs per task: {os.environ.get("NUM_EPOCHS","50")}, seeds: 0..9 (mask_seed_i via --run_offset)')
md.append('- settings: reduction_rate=0.5, buffer_size=20, topo_trainable_fusion, topo_alpha=0.1')
md.append('')
md.append('| Method | ACC (mean±std) | BAC (mean±std) | F1 (mean±std) | F1-AF (mean±std) |')
md.append('|---|---:|---:|---:|---:|')
for m,acc_m,acc_s,bac_m,bac_s,f1_m,f1_s,af_m,af_s in rows:
    md.append(f'| {m} | {acc_m:.2f} ± {acc_s:.2f} | {bac_m:.2f} ± {bac_s:.2f} | {f1_m:.2f} ± {f1_s:.2f} | {af_m:.2f} ± {af_s:.2f} |')
md.append('')
md.append('Note: JOINT is oracle upper bound (uses all tasks).')
open(os.path.join(out,'SUMMARY.md'),'w').write('\n'.join(md))
print(os.path.join(out,'SUMMARY.md'))
PY

# zip
cd "$OUT_DIR/.."
zip -qr "$(basename "$OUT_DIR").zip" "$(basename "$OUT_DIR")"
echo "Done. OUT_DIR=$OUT_DIR"