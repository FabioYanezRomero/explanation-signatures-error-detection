#!/usr/bin/env bash
# ==============================================================================
# Revision control experiment: retrain the GCN surrogates with (a) the gold label
# and (b) the teacher label (seed replicate) using the shard-streaming trainer, so
# that the error overlap with BERT can be compared between distillation and
# direct supervision. Same architecture/hyper-parameters as the original models.
# Runs inside the graphsvx image (torch 2.3.1 + PyG). Outputs go to
#   outputs/gnn_models_control/<label_attr>/<backbone>/<dataset>/<graph>/
# Usage: bash scripts/10_train_control_gnns_paired.sh [gold|teacher|both]   (default both)
# ==============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE=${1:-both}
LOG_DIR=logs/control_gnns; mkdir -p "$LOG_DIR"
OUT_ROOT=${OUT_ROOT:-/home/fabio/Documents/GitHub/From-Text-to-Graph-Leveraging-Graph-Neural-Networks-for-Enhanced-Explainability-in-NLP/outputs}  # host path of /app/outputs
SEED=${SEED:-42}; SUBDIR=${SUBDIR:-gnn_models_control}; ONLY_DATASET=${ONLY_DATASET:-}  # e.g. sst2

declare -a CFGS=(
  "stanfordnlp sst2 validation 2 window"
  "stanfordnlp sst2 validation 2 skipgrams"
  "stanfordnlp sst2 validation 2 syntactic"
  "stanfordnlp sst2 validation 2 constituency"
  "SetFit ag_news test 4 constituency"
  "SetFit ag_news test 4 window"
  "SetFit ag_news test 4 skipgrams"
  "SetFit ag_news test 4 syntactic"
)
declare -a LABELS=()
[[ "$MODE" == "gold" || "$MODE" == "both" ]] && LABELS+=("true_label")
[[ "$MODE" == "teacher" || "$MODE" == "both" ]] && LABELS+=("y")

run_one() {  # label backbone dataset test_split ncls graph
  local label=$1 backbone=$2 dataset=$3 test_split=$4 ncls=$5 graph=$6
  local out="/app/outputs/${SUBDIR}/${label}/${backbone}/${dataset}/${graph}"
  local log="${LOG_DIR}/${SUBDIR}_${label}_${dataset}_${graph}.log"
  if [[ -f "${OUT_ROOT}/${out#/app/outputs/}/test_results.json" ]]; then echo "skip ${label} ${dataset} ${graph} (done)"; return; fi
  echo "$(date '+%F %T') start ${label} ${dataset} ${graph}"
  docker compose run --rm -T --no-deps -e CUDA_VISIBLE_DEVICES=0 --entrypoint /bin/bash graphsvx -c \
    "cd /app && python3 -m src.gnn_training.training_streaming \
      --train_data_dir /app/outputs/pyg_graphs/${backbone}/${dataset}/train/${graph} \
      --test_data_dir /app/outputs/pyg_graphs/${backbone}/${dataset}/${test_split}/${graph} \
      --output_dir ${out} --label_attr ${label} --num_classes ${ncls} --seed ${SEED}" > "$log" 2>&1
  echo "$(date '+%F %T') end   ${label} ${dataset} ${graph} exit=$? :: $(tail -1 "$log")"
}

# Both labels of the same configuration run concurrently: they read the same shards,
# so the second run is served from the page cache (the disk read is the bottleneck).
for cfg in "${CFGS[@]}"; do
  read -r backbone dataset test_split ncls graph <<< "$cfg"
  [[ -n "$ONLY_DATASET" && "$dataset" != "$ONLY_DATASET" ]] && continue
  pids=()
  for label in "${LABELS[@]}"; do
    run_one "$label" "$backbone" "$dataset" "$test_split" "$ncls" "$graph" & pids+=($!)
    sleep 20   # stagger so the first run warms the cache
  done
  wait "${pids[@]}"
done
echo "$(date '+%F %T') ALL DONE"
