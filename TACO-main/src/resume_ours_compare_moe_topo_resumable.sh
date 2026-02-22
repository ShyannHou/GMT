#!/usr/bin/env bash
set -euo pipefail

# Resume a partially completed run produced by run_compare_ours_moe_topo_resumable.sh
# Usage:
#   OUT_DIR=../results/<existing_dir> NUM_EPOCHS=50 ./resume_ours_compare_moe_topo_resumable.sh

cd "$(dirname "$0")"

: "${OUT_DIR:?Set OUT_DIR to existing results dir}"
PY="../.venv_graphcls/bin/python"
NUM_EPOCHS=${NUM_EPOCHS:-50}

# seeds 0..9
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

run_method () {
  local METHOD="$1"; shift
  local EXTRA=("$@")

  for seed in $SEEDS; do
    if [[ -f "$OUT_DIR/${METHOD}_seed${seed}.pkl" && -f "$OUT_DIR/${METHOD}_seed${seed}.csv" ]]; then
      echo "[skip] ${METHOD} seed=${seed} already exists" | tee -a "$OUT_DIR/RUN.log"
      continue
    fi

    echo "\n==================== ${METHOD} seed=${seed} (resume) ====================" | tee -a "$OUT_DIR/RUN.log"
    echo "$PY -W ignore train.py --method ${METHOD} --run_offset ${seed} ${COMMON[*]} ${EXTRA[*]}" | tee -a "$OUT_DIR/RUN.log"

    $PY -W ignore train.py --method "$METHOD" --run_offset "$seed" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$OUT_DIR/${METHOD}_seed${seed}.log"

    cp -f "results/${METHOD}_GCN_OURS_reduction_0.5" "$OUT_DIR/${METHOD}_seed${seed}.pkl"
    cp -f "../results/OURS/${METHOD}_GCN_OURS_reduction_0.5.csv" "$OUT_DIR/${METHOD}_seed${seed}.csv"
  done
}

# resume order (configurable)
METHODS=${METHODS:-"DYGRA FINETUNE SIMPLE_REG JOINT"}
for m in $METHODS; do
  case "$m" in
    DYGRA) run_method DYGRA ;;
    FINETUNE) run_method FINETUNE ;;
    SIMPLE_REG) run_method SIMPLE_REG --simple_reg_lambda 1e-3 ;;
    JOINT) run_method JOINT ;;
    *) echo "Unknown method in METHODS: $m"; exit 2 ;;
  esac
done

echo "Resume complete. Building SUMMARY.md" | tee -a "$OUT_DIR/RUN.log"

$PY - "$OUT_DIR" <<'PY'
import os, sys, pickle
import numpy as np

out=sys.argv[1]
methods=["DYGRA","FINETUNE","SIMPLE_REG","JOINT"]
seeds=list(range(10))

with open(os.path.join(os.path.dirname(out), '..', 'data', 'OURS', 'statistics'), 'rb') as f:
    num_task, num_class = pickle.load(f)


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
        if not (os.path.exists(pkl_path) and os.path.exists(csv_path)):
            continue
        acc,bac,f1=pickle.load(open(pkl_path,'rb'))
        accs.append(float(np.mean(acc))*100)
        bacs.append(float(np.mean(bac))*100)
        f1s.append(float(np.mean(f1))*100)
        afs.append(f1_af_from_csv(csv_path))
    rows.append((m,
                 np.mean(accs), np.std(accs),
                 np.mean(bacs), np.std(bacs),
                 np.mean(f1s), np.std(f1s),
                 np.mean(afs), np.std(afs),
                 len(accs)))

md=[]
md.append('# OURS (GCN + parallel MoE + topology) compare (resumable 10 seeds)')
md.append('')
md.append('- settings: reduction_rate=0.5, buffer_size=20, topo_trainable_fusion, topo_alpha=0.1')
md.append('')
md.append('| Method | seeds_done | ACC (mean±std) | BAC (mean±std) | F1 (mean±std) | F1-AF (mean±std) |')
md.append('|---|---:|---:|---:|---:|---:|')
for m,acc_m,acc_s,bac_m,bac_s,f1_m,f1_s,af_m,af_s,n in rows:
    md.append(f'| {m} | {n} | {acc_m:.2f} ± {acc_s:.2f} | {bac_m:.2f} ± {bac_s:.2f} | {f1_m:.2f} ± {f1_s:.2f} | {af_m:.2f} ± {af_s:.2f} |')
md.append('')
md.append('Note: JOINT is oracle upper bound (uses all tasks).')
open(os.path.join(out,'SUMMARY.md'),'w').write('\n'.join(md))
print(os.path.join(out,'SUMMARY.md'))
PY

# zip
cd "$OUT_DIR/.."
zip -qr "$(basename "$OUT_DIR").zip" "$(basename "$OUT_DIR")"

echo "Done: $OUT_DIR"