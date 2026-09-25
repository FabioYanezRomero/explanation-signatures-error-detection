#!/usr/bin/env bash
# ==============================================================================
# Rebuild the analytics chain after the revision re-runs:
#   1. split every GNN explanation pickle into per-graph records using the
#      corrected shard-index mapping (dataset index = shard + local * num_shards)
#   2. extract the four evaluation dimensions for SubgraphX and GraphSVX
#   3. rebuild the per-module datasets (both benchmarks)
#   4. refit the stratified logistic regressions (Table 5 pipeline)
# Runs inside the subgraphx container (all analytics dependencies are present).
# Env: SKIP_SPLIT=1 to skip step 1, SKIP_LR=1 to skip step 4, SKIP_LLM_AUC=1 to keep the TokenSHAP AUC CSV, BOOTSTRAP_REPS (default 200)
# ==============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
BOOTSTRAP_REPS=${BOOTSTRAP_REPS:-200}
LOG_DIR="${ROOT_DIR}/logs/analytics"
mkdir -p "${LOG_DIR}"

read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
cd /app
if [[ "${SKIP_SPLIT:-0}" != "1" ]]; then
  echo "### 1. splitting explanation pickles (corrected shard mapping)"
  for method in subgraphx graphsvx; do
    while IFS= read -r pkl; do
      echo "  - ${pkl}"
      python -m src.explain.splitters.${method} --input "${pkl}" --format pickle --overwrite 2>&1 | tail -1
    done < <(find /app/outputs/gnn_models -path "*/explanations/${method}/*/results.pkl" | sort)
  done
fi
echo "### 2. extracting dimensions"
for dim in auc consistency progression fidelity; do
  echo "  - ${dim}"
  python -m src.Analytics.${dim}.extract_gnn --methods subgraphx graphsvx 2>&1 | grep "✓\|Completed"
done
if [[ "${SKIP_LLM_AUC:-0}" != "1" ]]; then
  echo "  - auc (TokenSHAP, adds confidence baselines)"
  python -m src.Analytics.auc.extract_llm 2>&1 | grep "✓\|Completed"
fi
echo "### 3. module datasets"
python -m src.use_case.build_module_datasets 2>&1 | grep "•\|Dataset"
if [[ "${SKIP_LR:-0}" != "1" ]]; then
  echo "### 4. logistic regressions (bootstrap reps: ${BOOTSTRAP_REPS})"
  python -m src.use_case.save_logistic_coefficients --bootstrap-reps "${BOOTSTRAP_REPS}" 2>&1 | grep -v "it/s\|s/it" | tail -20
fi
echo "### done"
INNER_EOF

docker compose run --rm -T -e SKIP_SPLIT="${SKIP_SPLIT:-0}" -e SKIP_LR="${SKIP_LR:-0}" -e SKIP_LLM_AUC="${SKIP_LLM_AUC:-0}" -e BOOTSTRAP_REPS="${BOOTSTRAP_REPS}" \
    --entrypoint /bin/bash subgraphx -c "${INNER}" 2>&1 | grep -v "^ Container" | tee "${LOG_DIR}/rebuild_$(date '+%Y%m%d_%H%M%S').log"
