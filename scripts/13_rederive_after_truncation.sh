#!/usr/bin/env bash
# ==============================================================================
# Re-derive every analysis that depends on the Dimension-1/2 features after
# `src/use_case/truncate_progressions_to_budget.py` restricted the SubgraphX
# progressions to the 20% budget (2026-09-23). Host side, no docker.
#   FT   : fine-tuned-feature run  -> revision/results_ft, tables_ft, figures_ft
#   BASE : pre-trained-feature run -> revision/results (only what the ablation table needs)
# Unaffected and therefore skipped: control_error_overlap (predictions only),
# the 200-bootstrap stratified LR summary (scripts/_lr_module.py, not used by the manuscript).
# ==============================================================================
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OLD=/home/fabio/Documents/GitHub/From-Text-to-Graph-Leveraging-Graph-Neural-Networks-for-Enhanced-Explainability-in-NLP
FT=$OLD/outputs_finetuned_embeddings
RES=$REPO/revision/results_ft
RESB=$REPO/revision/results
LOG=${LOG:-$REPO/logs/rederive}; mkdir -p "$LOG"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/driver.log"; }
run() { local name=$1; shift; log "start $name"; ( "$@" ) > "$LOG/$name.log" 2>&1 && log "done  $name" || log "FAILED $name (see $LOG/$name.log)"; }

# ---------------- fine-tuned run
cd "$FT"
run ft_revision_metrics python3 "$REPO/src/use_case/revision_metrics.py"
cp "$FT"/outputs/use_case/revision_metrics/*.csv "$RES/"
run ft_overlap        python3 "$REPO/src/use_case/explanation_overlap.py" --output-dir "$RES/overlap"
run ft_cross_model    python3 "$REPO/src/use_case/cross_model_detection.py" --output "$RES/cross_model_bert_errors.csv"
run ft_revision_tables python3 "$REPO/src/use_case/revision_tables.py" --output-dir "$RES"
run ft_control_agreement python3 "$REPO/src/use_case/control_agreement_on_bert_errors.py" --output "$RES/control_agreement_on_bert_errors.csv"
run ft_manuscript_tables python3 "$REPO/src/use_case/manuscript_tables.py" --output-dir "$REPO/revision/manuscript_v2/tables_ft" \
    --control-csv "$RES/control_error_overlap.csv" --agreement-csv "$RES/control_agreement_on_bert_errors.csv" --cross-model-csv "$RES/cross_model_bert_errors.csv" --overlap-dir "$RES/overlap"
run ft_ablation_blocks python3 "$REPO/src/use_case/ablation_table_blocks.py" --base "$REPO/revision/results" --ft "$RES" --output "$REPO/revision/manuscript_v2/tables_ft/tab_ablation.tex"
run ft_manuscript_figures python3 "$REPO/src/use_case/manuscript_figures.py" --output-dir "$REPO/revision/manuscript_v2/figures_ft"
run ft_maskout_sign   python3 "$REPO/src/use_case/maskout_drop_sign.py" --module-root outputs/use_case/module_datasets --output "$RES/maskout_drop_sign.csv"
run ft_nested_coefficients python3 "$REPO/src/use_case/nested_coefficients.py" --module-root outputs/use_case/module_datasets --output "$RES/nested_coefficients.csv"
run ft_nested_gbm     python3 "$REPO/src/use_case/nested_gbm.py" --output "$RES/nested_gbm.csv"

# ---------------- pre-trained (base) run: only what the ablation table reads
cd "$OLD"
run base_revision_metrics python3 "$REPO/src/use_case/revision_metrics.py"
cp "$OLD"/outputs/use_case/revision_metrics/*.csv "$RESB/"
run base_cross_model  python3 "$REPO/src/use_case/cross_model_detection.py" --output "$RESB/cross_model_bert_errors.csv"
run base_nested_gbm   python3 "$REPO/src/use_case/nested_gbm.py" --output "$RESB/nested_gbm.csv"
cd "$REPO"
run ablation_table    python3 "$REPO/src/use_case/ablation_table.py" --base "$RESB" --finetuned "$RES" --output "$REPO/revision/manuscript_v2/tables_ft/tab_ablation.tex"
log "ALL DONE"
