#!/usr/bin/env bash
# ==============================================================================
# Redo the AG News / constituency configuration after regenerating the FULL PyG
# training set (the original GCN had been trained on 12,416 of 120,000 graphs).
# Stages (each idempotent; set START_AT=<n> to resume):
#   1 archive the old model (+ its explanations) and old controls outside gnn_models
#   2 train the paper model (teacher label, seed 42, streaming trainer)
#   3 controls: gold s42, teacher replicate s1, gold s1 (concurrent, shared cache)
#   4 SubgraphX (4 shards, predicted target) and GraphSVX (4 shards) concurrently
#   5 move raw MCTS to subgraphx_v2_rawmcts and recompute compact metrics
#   6 rebuild analytics (split + 4 dimensions + module datasets), LR for the 2 modules
#   7 revision metrics, manuscript tables/figures, token rankings, overlap, cross-model, control
# ==============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD
OLD=/home/fabio/Documents/GitHub/From-Text-to-Graph-Leveraging-Graph-Neural-Networks-for-Enhanced-Explainability-in-NLP
O=$OLD/outputs
START_AT=${START_AT:-1}
LOG=logs/redo_constituency; mkdir -p "$LOG"
stage() { echo "$(date '+%F %T') STAGE $*"; }
fail() { echo "$(date '+%F %T') FAILED $*"; exit 1; }
drun() { docker compose run --rm -T --no-deps -e CUDA_VISIBLE_DEVICES=0 --entrypoint /bin/bash "$@"; }
TRAIN="cd /app && python3 -m src.gnn_training.training_streaming --train_data_dir /app/outputs/pyg_graphs/SetFit/ag_news/train/constituency --test_data_dir /app/outputs/pyg_graphs/SetFit/ag_news/test/constituency --num_classes 4"

n=$(ls "$O/pyg_graphs/SetFit/ag_news/train/constituency" | wc -l); [[ "$n" == 938 ]] || fail "expected 938 shards, found $n"

if (( START_AT <= 1 )); then
  stage 1 archive
  mkdir -p "$O/gnn_models_v1_archive/SetFit/ag_news"
  if [[ -d "$O/gnn_models/SetFit/ag_news/constituency" && ! -d "$O/gnn_models_v1_archive/SetFit/ag_news/constituency_trained_on_12k" ]]; then
    mv "$O/gnn_models/SetFit/ag_news/constituency" "$O/gnn_models_v1_archive/SetFit/ag_news/constituency_trained_on_12k" || fail "archive model"
  fi
  drun graphsvx -c "mkdir -p /app/outputs/gnn_models_control_archive_12k/true_label/SetFit/ag_news /app/outputs/gnn_models_control_archive_12k/y/SetFit/ag_news; \
    [ -d /app/outputs/gnn_models_control/true_label/SetFit/ag_news/constituency ] && mv /app/outputs/gnn_models_control/true_label/SetFit/ag_news/constituency /app/outputs/gnn_models_control_archive_12k/true_label/SetFit/ag_news/; \
    [ -d /app/outputs/gnn_models_control/y/SetFit/ag_news/constituency ] && mv /app/outputs/gnn_models_control/y/SetFit/ag_news/constituency /app/outputs/gnn_models_control_archive_12k/y/SetFit/ag_news/; true" || fail "archive controls"
  mkdir -p revision/results_v1_constituency12k && cp revision/results/*.csv revision/results_v1_constituency12k/ 2>/dev/null; cp -r revision/results/overlap revision/results_v1_constituency12k/ 2>/dev/null
fi

if (( START_AT <= 2 )); then
  stage 2 train paper model
  drun graphsvx -c "$TRAIN --output_dir /app/outputs/gnn_models/SetFit/ag_news/constituency --label_attr y --seed 42" > "$LOG/train_paper_teacher_s42.log" 2>&1 || fail "train paper model"
  tail -1 "$LOG/train_paper_teacher_s42.log"
fi

if (( START_AT <= 3 )); then
  stage 3 controls
  drun graphsvx -c "$TRAIN --output_dir /app/outputs/gnn_models_control/true_label/SetFit/ag_news/constituency --label_attr true_label --seed 42" > "$LOG/control_gold_s42.log" 2>&1 & p1=$!
  sleep 20
  drun graphsvx -c "$TRAIN --output_dir /app/outputs/gnn_models_control/y/SetFit/ag_news/constituency --label_attr y --seed 1" > "$LOG/control_teacher_s1.log" 2>&1 & p2=$!
  sleep 20
  drun graphsvx -c "$TRAIN --output_dir /app/outputs/gnn_models_control_seed1/true_label/SetFit/ag_news/constituency --label_attr true_label --seed 1" > "$LOG/control_gold_s1.log" 2>&1 & p3=$!
  wait $p1 || fail "control gold s42"; wait $p2 || fail "control teacher s1"; wait $p3 || fail "control gold s1"
  for f in control_gold_s42 control_teacher_s1 control_gold_s1; do echo "  $f: $(tail -1 $LOG/$f.log)"; done
fi

if (( START_AT <= 4 )); then
  stage 4 explanations
  NUM_SHARDS=4 RUN_TAG=predtarget bash scripts/07_rerun_subgraphx_predicted_target.sh ag_constituency > "$LOG/subgraphx_driver.log" 2>&1 & p1=$!
  NUM_SHARDS=4 bash scripts/08_run_graphsvx_on_trees.sh ag_constituency > "$LOG/graphsvx_driver.log" 2>&1 & p2=$!
  wait $p1 || fail "subgraphx"; wait $p2 || fail "graphsvx"
  tail -1 "$LOG/subgraphx_driver.log"; tail -1 "$LOG/graphsvx_driver.log"
fi

if (( START_AT <= 5 )); then
  stage 5 recompute subgraphx metrics
  drun subgraphx -c "cd /app && mkdir -p /app/outputs/gnn_models/SetFit/ag_news/constituency/explanations/subgraphx_v2_rawmcts && \
    for d in /app/outputs/gnn_models/SetFit/ag_news/constituency/explanations/subgraphx/SetFit_ag_news_constituency_test_predtarget_shard*; do \
      [ -f \$d/results.pkl ] && [ ! -d /app/outputs/gnn_models/SetFit/ag_news/constituency/explanations/subgraphx_v2_rawmcts/\$(basename \$d) ] && mv \$d /app/outputs/gnn_models/SetFit/ag_news/constituency/explanations/subgraphx_v2_rawmcts/; done; \
    python -m src.explain.gnn.subgraphx.recompute_metrics --only SetFit_ag_news_constituency_test_predtarget 2>&1 | grep -v Warning | tail -6" > "$LOG/recompute.log" 2>&1 || fail "recompute"
  cat "$LOG/recompute.log"
fi

if (( START_AT <= 6 )); then
  stage 6 analytics
  SKIP_LR=1 SKIP_LLM_AUC=1 bash scripts/09_rebuild_analytics.sh > "$LOG/rebuild_analytics.log" 2>&1 || fail "rebuild analytics"
  tail -3 "$LOG/rebuild_analytics.log"
  ( cd "$OLD" && python3 - <<'EOF' ) > "$LOG/lr_constituency.log" 2>&1 || fail "LR"
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, '/home/fabio/Documents/GitHub/Graph-Neural-Networks-Enable-Superior-Error-Detection-in-NLP-Explainability-than-Language-Models/src')
from use_case import save_logistic_coefficients as slc
files = [e for e in slc._discover_module_files() if e[3] == 'constituency' and e[0] == 'setfit_ag_news']
print('modules:', [(e[0], e[2], e[3]) for e in files])
slc._discover_module_files = lambda: files
slc.SUMMARY_PATH = Path('outputs/use_case/module_datasets/error_signal_classification_summary_constituency_agnews.csv')
res, summary = slc.process_modules(Path('outputs/use_case/module_datasets/coefficients'), max_splits=10, bootstrap_reps=200)
for name in ['error_signal_classification_summary.csv', 'error_signal_classification_summary_final_boot200.csv']:
    p = Path('outputs/use_case/module_datasets') / name
    base = pd.read_csv(p)
    keep = ~((base.dataset == 'setfit_ag_news') & (base.graph == 'constituency'))
    full = pd.concat([base[keep], summary.assign(stratify_by='prediction_class')], ignore_index=True)
    full.to_csv(p, index=False)
    print(name, 'modules:', full.groupby(['dataset', 'method', 'graph']).ngroups)
EOF
  tail -3 "$LOG/lr_constituency.log"
fi

if (( START_AT <= 7 )); then
  stage 7 metrics and derived results
  ( cd "$OLD" && python3 "$REPO/src/use_case/revision_metrics.py" ) > "$LOG/revision_metrics.log" 2>&1 || fail "revision_metrics"
  cp "$O"/use_case/revision_metrics/*.csv revision/results/
  ( cd "$OLD" && python3 "$REPO/src/use_case/revision_tables.py" --output-dir "$REPO/revision/results" ) > "$LOG/revision_tables.log" 2>&1 || echo "  (revision_tables failed, see log)"
  ( cd "$OLD" && python3 "$REPO/src/use_case/manuscript_tables.py" --output-dir "$REPO/revision/manuscript_v2/tables" && python3 "$REPO/src/use_case/manuscript_figures.py" --output-dir "$REPO/revision/manuscript_v2/figures" ) > "$LOG/manuscript_tables.log" 2>&1 || fail "manuscript tables"
  drun graphsvx -c "cd /app && python3 -m src.Analytics.tokens.extract_gnn --base-dir /app/outputs/gnn_models --graph-root /app/outputs/graphs --pyg-root /app/outputs/pyg_graphs --output-dir /app/outputs/analytics/tokens_v2 --methods subgraphx graphsvx" > "$LOG/tokens_v2.log" 2>&1 || fail "tokens"
  ( cd "$OLD" && python3 "$REPO/src/use_case/explanation_overlap.py" --output-dir "$REPO/revision/results/overlap" ) > "$LOG/overlap.log" 2>&1 || fail "overlap"
  ( cd "$OLD" && python3 "$REPO/src/use_case/cross_model_detection.py" --output "$REPO/revision/results/cross_model_bert_errors.csv" ) > "$LOG/cross_model.log" 2>&1 || fail "cross-model"
  ( cd "$OLD" && python3 "$REPO/src/use_case/control_error_overlap.py" --control-roots outputs/gnn_models_control outputs/gnn_models_control_seed1 outputs/gnn_models_control_seed2 --output "$REPO/revision/results/control_error_overlap.csv" ) > "$LOG/control.log" 2>&1 || fail "control"
fi
stage 8 ALL DONE
