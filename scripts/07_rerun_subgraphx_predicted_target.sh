#!/usr/bin/env bash
# ==============================================================================
# Re-run SubgraphX on the four hierarchical configurations explaining the
# PREDICTED class (label-free), replacing the original runs that explained the
# dataset label and therefore leaked the error-detection target.
#
# Usage: bash scripts/07_rerun_subgraphx_predicted_target.sh [config ...]
#   configs: sst2_syntactic sst2_constituency ag_syntactic ag_constituency (hierarchical, default)
#            sst2_window sst2_skipgrams ag_window ag_skipgrams (non-hierarchical, full 2x2 design)
# Env:  NUM_SHARDS (default 3), FAIR_BUDGET (default 2000), RUN_TAG (default predtarget),
#       MAX_GRAPHS (default empty = all), OMP_THREADS (default 4), GPU_DEVICE (default cuda:0)
# ==============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NUM_SHARDS=${NUM_SHARDS:-3}
FAIR_BUDGET=${FAIR_BUDGET:-2000}
RUN_TAG=${RUN_TAG:-predtarget}
MAX_GRAPHS=${MAX_GRAPHS:-}
OMP_THREADS=${OMP_THREADS:-4}
GPU_DEVICE=${GPU_DEVICE:-cuda:0}
LOG_DIR="${ROOT_DIR}/logs/subgraphx_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

declare -A CFG_DATASET=([sst2_syntactic]=sst2 [sst2_constituency]=sst2 [ag_syntactic]=ag_news [ag_constituency]=ag_news [sst2_window]=sst2 [sst2_skipgrams]=sst2 [ag_window]=ag_news [ag_skipgrams]=ag_news)
declare -A CFG_BACKBONE=([sst2_syntactic]=stanfordnlp [sst2_constituency]=stanfordnlp [ag_syntactic]=SetFit [ag_constituency]=SetFit [sst2_window]=stanfordnlp [sst2_skipgrams]=stanfordnlp [ag_window]=SetFit [ag_skipgrams]=SetFit)
declare -A CFG_GRAPH=([sst2_syntactic]=syntactic [sst2_constituency]=constituency [ag_syntactic]=syntactic [ag_constituency]=constituency [sst2_window]=window [sst2_skipgrams]=skipgrams [ag_window]=window [ag_skipgrams]=skipgrams)
declare -A CFG_SPLIT=([sst2_syntactic]=validation [sst2_constituency]=validation [ag_syntactic]=test [ag_constituency]=test [sst2_window]=validation [sst2_skipgrams]=validation [ag_window]=test [ag_skipgrams]=test)

CONFIGS=("$@")
if (( ${#CONFIGS[@]} == 0 )); then
    CONFIGS=(sst2_syntactic sst2_constituency ag_syntactic ag_constituency)
fi

max_graphs_flag=""
if [[ -n "${MAX_GRAPHS}" ]]; then
    max_graphs_flag="--max-graphs ${MAX_GRAPHS}"
fi

run_shard() {
    local cfg=$1 shard=$2
    local cmd="cd /app && OMP_NUM_THREADS=${OMP_THREADS} MKL_NUM_THREADS=${OMP_THREADS} python -m src.explain.gnn.subgraphx.main \
        --dataset '${CFG_DATASET[$cfg]}' --graph-type '${CFG_GRAPH[$cfg]}' --backbone '${CFG_BACKBONE[$cfg]}' \
        --split '${CFG_SPLIT[$cfg]}' --device '${GPU_DEVICE}' --fair --target-forward-passes ${FAIR_BUDGET} \
        --target-class predicted --run-tag '${RUN_TAG}' --num-shards ${NUM_SHARDS} --shard-index ${shard} \
        --no-progress ${max_graphs_flag}"
    docker compose run --rm -T -e CUDA_VISIBLE_DEVICES=0 --entrypoint /bin/bash subgraphx -c "${cmd}" \
        > "${LOG_DIR}/${cfg}_shard${shard}of${NUM_SHARDS}.log" 2>&1
}

for cfg in "${CONFIGS[@]}"; do
    echo "[$(date '+%F %T')] START ${cfg} (${NUM_SHARDS} shards, budget ${FAIR_BUDGET}, tag ${RUN_TAG})" | tee -a "${LOG_DIR}/driver.log"
    pids=()
    for (( shard = 0; shard < NUM_SHARDS; ++shard )); do
        run_shard "${cfg}" "${shard}" &
        pids+=($!)
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    if (( status != 0 )); then
        echo "[$(date '+%F %T')] FAILED ${cfg} (see ${LOG_DIR})" | tee -a "${LOG_DIR}/driver.log"
        exit 1
    fi
    echo "[$(date '+%F %T')] DONE ${cfg}" | tee -a "${LOG_DIR}/driver.log"
done
echo "[$(date '+%F %T')] ALL DONE" | tee -a "${LOG_DIR}/driver.log"
