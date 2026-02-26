#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="../.venv_graphcls/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="../results/OURS_compare_GCN_baselines_${STAMP}"
mkdir -p "$OUT_DIR"

NUM_EPOCHS=${NUM_EPOCHS:-50}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
METHODS=${METHODS:-"FINETUNE SIMPLE_REG JOINT"}

# Backbone is pure GCN (no MoE, no topology flags)
COMMON=(
  --Dataset OURS
  --gnn GCN
  --reduction_rate 0.5
  --buffer_size 20
  --gpu -1
  --num_runs 1
  --num_epochs "$NUM_EPOCHS"
)

run_one_seed () {
  local METHOD="$1"; local SEED="$2"; shift 2
  local EXTRA=("$@")

  local pkl_out="$OUT_DIR/${METHOD}_seed${SEED}.pkl"
  local csv_out="$OUT_DIR/${METHOD}_seed${SEED}.csv"

  if [[ -f "$pkl_out" && -f "$csv_out" ]]; then
    echo "[skip] ${METHOD} seed=${SEED} exists" | tee -a "$OUT_DIR/RUN.log"
    return
  fi

  echo "\n==================== ${METHOD} seed=${SEED} ====================" | tee -a "$OUT_DIR/RUN.log"
  echo "$PY -W ignore train.py --method ${METHOD} --run_offset ${SEED} ${COMMON[*]} ${EXTRA[*]}" | tee -a "$OUT_DIR/RUN.log"

  $PY -W ignore train.py --method "$METHOD" --run_offset "$SEED" "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee "$OUT_DIR/${METHOD}_seed${SEED}.log"

  cp -f "results/${METHOD}_GCN_OURS_reduction_0.5" "$pkl_out"
  cp -f "../results/OURS/${METHOD}_GCN_OURS_reduction_0.5.csv" "$csv_out"
}

for m in $METHODS; do
  for s in $SEEDS; do
    case "$m" in
      FINETUNE)
        run_one_seed FINETUNE "$s"
        ;;
      SIMPLE_REG)
        run_one_seed SIMPLE_REG "$s" --simple_reg_lambda 1e-3
        ;;
      JOINT)
        run_one_seed JOINT "$s"
        ;;
      *)
        echo "Unknown method: $m"; exit 2
        ;;
    esac
  done
done

# Build SUMMARY.md
$PY - "$OUT_DIR" <<'PY'
import os, sys, glob, pickle
import numpy as np

out=sys.argv[1]

with open(os.path.join('..','data','OURS','statistics'),'rb') as f:
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

def summarize(method):
    pkls=sorted(glob.glob(os.path.join(out,f'{method}_seed*.pkl')))
    if not pkls:
        return None
    accs=[]; bacs=[]; f1s=[]; afs=[]
    for p in pkls:
        acc,bac,f1=pickle.load(open(p,'rb'))
        accs.append(np.mean(acc)*100)
        bacs.append(np.mean(bac)*100)
        f1s.append(np.mean(f1)*100)
        seed=p.split('seed')[-1].split('.pkl')[0]
        c=os.path.join(out,f'{method}_seed{seed}.csv')
        if os.path.exists(c):
            afs.append(f1_af_from_csv(c))
    return len(pkls), float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(afs)), float(np.std(afs))

methods=['FINETUNE','SIMPLE_REG','JOINT']
rows=[]
for m in methods:
    r=summarize(m)
    if r is None:
        continue
    rows.append((m,*r))

md=[]
md.append('# OURS baselines (backbone=GCN)')
md.append('')
md.append('Settings: Dataset=OURS, gnn=GCN, reduction_rate=0.5, buffer_size=20, gpu=-1')
md.append('Runs: seeds 0..9 (mask_seed_i), epochs per task: ' + str(os.environ.get('NUM_EPOCHS','50')))
md.append('')
md.append('| Method | seeds | ACC (mean±std) | F1 (mean±std) | F1-AF (mean±std) |')
md.append('|---|---:|---:|---:|---:|')
for m,n,acc_m,acc_s,f1_m,f1_s,af_m,af_s in rows:
    md.append(f'| {m} | {n} | {acc_m:.2f} ± {acc_s:.2f} | {f1_m:.2f} ± {f1_s:.2f} | {af_m:.2f} ± {af_s:.2f} |')
md.append('')
md.append('Note: JOINT is oracle upper bound (uses all tasks).')
open(os.path.join(out,'SUMMARY.md'),'w').write('\n'.join(md))
print(os.path.join(out,'SUMMARY.md'))
PY

# Zip
dirname=$(basename "$OUT_DIR")
cd "$OUT_DIR/.."
zip -qr "${dirname}.zip" "$dirname"
echo "Done. OUT_DIR=$OUT_DIR"