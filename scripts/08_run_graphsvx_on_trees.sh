#!/usr/bin/env bash
# ==============================================================================
# Re-run GraphSVX on the four hierarchical configurations explaining the
# hierarchical (tree) graph types, so that one topology-agnostic explainer is
# shared across all topologies (reviewer request: de-confound topology vs explainer).
#
# Usage: bash scripts/08_run_graphsvx_on_trees.sh [config ...]
#   configs: sst2_syntactic sst2_constituency ag_syntactic ag_constituency (default: all, in that order)
# Env:  NUM_SHARDS (default 3), FAIR_BUDGET (default 2000), RUN_TAG (default predtarget),
#       MAX_GRAPHS (default empty = all), OMP_THREADS (default 4), GPU_DEVICE (default cuda:0)
# ==============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NUM_SHARDS=${NUM_SHARDS:-3}
FAIR_BUDGET=${FAIR_BUDGET:-2000}
RUN_TAG=${RUN_TAG:-trees}
MAX_GRAPHS=${MAX_GRAPHS:-}
OMP_THREADS=${OMP_THREADS:-4}
GPU_DEVICE=${GPU_DEVICE:-cuda:0}
LOG_DIR="${ROOT_DIR}/logs/graphsvx_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

declare -A CFG_DATASET=([sst2_syntactic]=sst2 [sst2_constituency]=sst2 [ag_syntactic]=ag_news [ag_constituency]=ag_news)
declare -A CFG_BACKBONE=([sst2_syntactic]=stanfordnlp [sst2_constituency]=stanfordnlp [ag_syntactic]=SetFit [ag_constituency]=SetFit)
declare -A CFG_GRAPH=([sst2_syntactic]=syntactic [sst2_constituency]=constituency [ag_syntactic]=syntactic [ag_constituency]=constituency)
declare -A CFG_SPLIT=([sst2_syntactic]=validation [sst2_constituency]=validation [ag_syntactic]=test [ag_constituency]=test)

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
    local cmd="cd /app && OMP_NUM_THREADS=${OMP_THREADS} MKL_NUM_THREADS=${OMP_THREADS} python -m src.explain.gnn.graphsvx.main \
        --dataset '${CFG_DATASET[$cfg]}' --graph-type '${CFG_GRAPH[$cfg]}' --backbone '${CFG_BACKBONE[$cfg]}' \
        --split '${CFG_SPLIT[$cfg]}' --device '${GPU_DEVICE}' --fair --target-forward-passes ${FAIR_BUDGET} --seed ${SEED:-42} \
        --num-shards ${NUM_SHARDS} --shard-index ${shard} \
        --no-progress ${max_graphs_flag}"
    docker compose run --rm -T -e CUDA_VISIBLE_DEVICES=0 --entrypoint /bin/bash graphsvx -c "${cmd}" \
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
