#!/usr/bin/env bash
set -euo pipefail

cd /root/kunlin/Shyann-Research/TACO-main/src
source ../.venv_graphcls/bin/activate

NUM_RUNS=${NUM_RUNS:-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}
OUT_DIR="../results/OURS_mainline_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

BASE_ARGS=(
  --Dataset OURS
  --method DYGRA
  --gnn GCN
  --reduction_rate 0.5
  --buffer_size 20
  --num_runs "$NUM_RUNS"
  --num_epochs "$NUM_EPOCHS"
  --gpu -1
)

summarize_pkl() {
  local pkl="$1"
  python - "$pkl" <<'PY'
import pickle,sys,numpy as np
p=sys.argv[1]
acc,bac,f1=pickle.load(open(p,'rb'))
print(f"acc_mean={np.mean(acc):.6f} acc_std={np.std(acc):.6f}")
print(f"bac_mean={np.mean(bac):.6f} bac_std={np.std(bac):.6f}")
print(f"f1_mean={np.mean(f1):.6f} f1_std={np.std(f1):.6f}")
PY
}

run_case() {
  local name="$1"; shift
  echo "\n==================== CASE: $name ====================" | tee "$OUT_DIR/$name.log"
  echo "ARGS: ${BASE_ARGS[*]} $*" | tee -a "$OUT_DIR/$name.log"

  python -W ignore train.py "${BASE_ARGS[@]}" "$@" 2>&1 | tee -a "$OUT_DIR/$name.log"

  local pkl_src="results/DYGRA_GCN_OURS_reduction_0.5"
  local csv_src="../results/OURS/DYGRA_GCN_OURS_reduction_0.5.csv"
  local pkl_dst="$OUT_DIR/$name.pkl"
  local csv_dst="$OUT_DIR/$name.csv"

  cp "$pkl_src" "$pkl_dst"
  cp "$csv_src" "$csv_dst"

  {
    echo "---- SUMMARY: $name ----"
    summarize_pkl "$pkl_dst"
  } | tee -a "$OUT_DIR/$name.log"
}

# 1) GNN only baseline
run_case "A_gnn_only"

# 2) Parallel GNN+MoE (no topology)
run_case "B_parallel_moe" \
  --use_moe True \
  --parallel_fusion True

# 3) Parallel GNN+MoE + topology fixed fusion
run_case "C_parallel_moe_topo_fixed" \
  --use_moe True \
  --parallel_fusion True \
  --use_topology_feature \
  --topo_alpha 0.1 \
  --topology_pd_dir ../topology_import_20260220/output \
  --topology_group 0 \
  --topology_window 5 \
  --topology_bins 32 \
  --topology_max_end 98 \
  --topo_seed 42

# 4) Parallel GNN+MoE + topology trainable fusion
run_case "D_parallel_moe_topo_trainable" \
  --use_moe True \
  --parallel_fusion True \
  --use_topology_feature \
  --topo_alpha 0.1 \
  --topo_trainable_fusion \
  --topology_pd_dir ../topology_import_20260220/output \
  --topology_group 0 \
  --topology_window 5 \
  --topology_bins 32 \
  --topology_max_end 98 \
  --topo_seed 42

# Build one markdown summary
python - "$OUT_DIR" <<'PY'
import os,glob,pickle,numpy as np,sys
out=sys.argv[1]
rows=[]
name_map={
  'A_gnn_only':'GNN only',
  'B_parallel_moe':'GNN+MoE parallel (no topology)',
  'C_parallel_moe_topo_fixed':'GNN+MoE+Topology (fixed fusion)',
  'D_parallel_moe_topo_trainable':'GNN+MoE+Topology (trainable fusion)'
}
for key in ['A_gnn_only','B_parallel_moe','C_parallel_moe_topo_fixed','D_parallel_moe_topo_trainable']:
  p=os.path.join(out,f'{key}.pkl')
  acc,bac,f1=pickle.load(open(p,'rb'))
  rows.append((name_map[key],np.mean(acc),np.std(acc),np.mean(bac),np.std(bac),np.mean(f1),np.std(f1)))
md=[]
md.append('# OURS数据主线对比（TACO train.py）')
md.append('')
md.append('> 同配置：Dataset=OURS, method=DYGRA, gnn=GCN, reduction_rate=0.5, buffer_size=20')
md.append('> 运行参数：num_runs=5, num_epochs=3, gpu=-1')
md.append('')
md.append('| 配置 | ACC(mean±std) | BAC(mean±std) | F1(mean±std) |')
md.append('|---|---:|---:|---:|')
for r in rows:
  md.append(f'| {r[0]} | {r[1]:.4f} ± {r[2]:.4f} | {r[3]:.4f} ± {r[4]:.4f} | {r[5]:.4f} ± {r[6]:.4f} |')
md.append('')
md.append('## 文件')
for key in ['A_gnn_only','B_parallel_moe','C_parallel_moe_topo_fixed','D_parallel_moe_topo_trainable']:
  md.append(f'- {key}.log / {key}.pkl / {key}.csv')
open(os.path.join(out,'ABALATION_SUMMARY.md'),'w').write('\n'.join(md))
print(os.path.join(out,'ABALATION_SUMMARY.md'))
PY

echo "Done. Output dir: $OUT_DIR"
