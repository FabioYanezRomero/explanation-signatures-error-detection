#!/usr/bin/env bash
# ==============================================================================
# Replicate the GNN side of the pipeline with node features taken from the FINE-TUNED
# teacher checkpoint (the premise of the LLM-as-teacher design) instead of the base
# bert-base-uncased that the original loader silently fell back to (fixed 2026-09-18 in
# src/embeddings/generate.py). Everything is written to a SEPARATE root,
#   <old repo>/outputs_finetuned_embeddings/outputs   (= /app/outputs via OUTPUTS_HOST),
# whose shared inputs (graphs, finetuned_llms, TokenSHAP analytics) are symlinks/copies of
# the original outputs. Nothing under the original outputs/ is modified or deleted.
#
# Passes (PASS=main|controls|derive|all, default all; every step is idempotent):
#   main     per configuration: embed test + train splits -> train the paper model (teacher
#            label, seed 42) and the gold-label control (seed 42) concurrently -> SubgraphX
#            (predicted target, 4 shards) + GraphSVX (4 shards) in the background
#   derive   recompute SubgraphX metrics, rebuild analytics, LR, token rankings, revision
#            metrics/tables/figures, overlap, cross-model and control analyses
#            -> revision/results_ft/, revision/manuscript_v2/{tables,figures}_ft/
#   controls per configuration: (re)embed the train split if evicted -> teacher replicate
#            (seed 1) + gold (seed 1), then derive again (control table)
# Disk: train-split PyG shards (2-43 GB each) are transient. When free space runs low the
# oldest already-consumed train split is evicted (regenerated in ~20 min when needed again).
# Test-split shards, models, explanations and analytics are kept.
# Logs: logs/ft_embeddings/. Resume by re-running (done steps are skipped).
# ==============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD
OLD=/home/fabio/Documents/GitHub/From-Text-to-Graph-Leveraging-Graph-Neural-Networks-for-Enhanced-Explainability-in-NLP
FT=$OLD/outputs_finetuned_embeddings
FTO=$FT/outputs
export OUTPUTS_HOST=$FTO          # docker-compose.override.yml mounts this as /app/outputs
PASS=${PASS:-all}
ONLY=${ONLY:-}                     # optional space-separated subset of configurations
MARGIN_GB=${MARGIN_GB:-12}
OMP=${OMP:-3}
LR_JOBS=${LR_JOBS:-6}
LOG=logs/ft_embeddings; mkdir -p "$LOG"
CONSUMED=$FTO/pyg_graphs/_consumed_train_dirs.txt
OWNER="$(id -u):$(id -g)"
RES=$REPO/revision/results_ft
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG/driver.log"; }
drun() { docker compose run --rm -T --no-deps -e CUDA_VISIBLE_DEVICES=0 --entrypoint /bin/bash "$@"; }
crm() { drun graphsvx -c "rm -rf $*" >/dev/null 2>&1; }           # container-owned paths
free_gb() { df -BG --output=avail "$OLD" | tail -1 | tr -dc '0-9'; }

CFGS=(sst2_window sst2_skipgrams sst2_syntactic sst2_constituency ag_window ag_skipgrams ag_syntactic ag_constituency)
[[ -n "$ONLY" ]] && read -r -a CFGS <<< "$ONLY"

cfg_vars() {  # sets: ds graph hf bb name test ncls epoch tree ckpt pred train_gb slug
  local cfg=$1; graph=${cfg#*_}
  if [[ $cfg == sst2_* ]]; then ds=sst2; hf=stanfordnlp/sst2; bb=stanfordnlp; name=sst2; test=validation; ncls=2; epoch=2
    ckpt=/app/outputs/finetuned_llms/stanfordnlp/sst2/sst2_2025-06-04_14-52-49/model_epoch_2.pt
  else ds=ag_news; hf=SetFit/ag_news; bb=SetFit; name=ag_news; test=test; ncls=4; epoch=4
    ckpt=/app/outputs/finetuned_llms/setfit/ag_news/model_epoch_4.pt; fi
  pred=$(dirname "$ckpt")/predictions.json
  case $graph in window) tree=window.word.k1;; skipgrams) tree=skipgrams.word.k1.n2;; *) tree=$graph;; esac
  case $cfg in ag_constituency) train_gb=43;; ag_*) train_gb=16;; sst2_constituency) train_gb=6;; *) train_gb=2;; esac
  slug=${bb}_${name}_${graph}_${test}
}

ensure_space() {  # GB needed on top of the margin; evicts consumed train splits, else waits
  local need=$1
  while (( $(free_gb) < need + MARGIN_GB )); do
    local victim; victim=$(head -1 "$CONSUMED" 2>/dev/null || true)
    if [[ -n "$victim" && -d "$victim" ]]; then
      log "disk: evicting consumed train split $victim ($(du -sh "$victim" | cut -f1); free $(free_gb) GB, need $((need + MARGIN_GB)))"
      crm "/app/outputs/${victim#$FTO/}"; sed -i 1d "$CONSUMED"
    elif [[ -n "$victim" ]]; then sed -i 1d "$CONSUMED"
    else log "disk: WAITING - need $((need + MARGIN_GB)) GB, free $(free_gb) GB, nothing to evict (free space on / to continue)"; sleep 600; fi
  done
}

embed() {  # cfg split
  local cfg=$1 split=$2; cfg_vars "$cfg"
  local pyg_c=/app/outputs/pyg_graphs/$bb/$name/$split/$graph pyg_h=$FTO/pyg_graphs/$bb/$name/$split/$graph
  [[ -f $pyg_h/_DONE ]] && return 0
  [[ $split == train ]] && ensure_space "$train_gb"
  [[ -d $pyg_h ]] && crm "$pyg_c"
  log "embed $cfg $split (fine-tuned checkpoint epoch $epoch)"
  drun graphsvx -c "cd /app && PYTHONPATH=/app/.pip_ft python3 -m src.embeddings.generate_efficient --graph_type $graph --dataset_name $hf --split $split \
      --tree_dir /app/outputs/graphs/$hf/$split/$tree --output_dir /app/outputs/_env/unused_embeddings --pyg_output_dir $pyg_c \
      --model_name bert-base-uncased --weights_path $ckpt --predictions_file $pred --epoch $epoch --device cuda:0 --batch_size 128 --no_embedding_pickles \
      && touch $pyg_c/_DONE && chown -R $OWNER $pyg_c" > "$LOG/embed_${cfg}_${split}.log" 2>&1
  if grep -q "199/199 parameter tensors" "$LOG/embed_${cfg}_${split}.log" && [[ -f $pyg_h/_DONE ]]; then
    log "  embedded $cfg $split: $(grep -h 'Generated' "$LOG/embed_${cfg}_${split}.log" | tr '\n' ' ')"; return 0
  fi
  log "  FAILED embed $cfg $split (see $LOG/embed_${cfg}_${split}.log)"; return 1
}

train() {  # cfg label seed rel_out   (rel_out relative to /app/outputs)
  local cfg=$1 label=$2 seed=$3 rel=$4; cfg_vars "$cfg"
  [[ -f $FTO/$rel/test_results.json ]] && return 0
  log "train $cfg label=$label seed=$seed -> $rel"
  drun graphsvx -c "cd /app && python3 -m src.gnn_training.training_streaming --train_data_dir /app/outputs/pyg_graphs/$bb/$name/train/$graph \
      --test_data_dir /app/outputs/pyg_graphs/$bb/$name/$test/$graph --output_dir /app/outputs/$rel --label_attr $label --num_classes $ncls --seed $seed \
      && chown -R $OWNER /app/outputs/$rel" > "$LOG/train_${cfg}_${label}_s${seed}.log" 2>&1
  if [[ -f $FTO/$rel/test_results.json ]]; then log "  trained $rel: $(tail -1 "$LOG/train_${cfg}_${label}_s${seed}.log")"; return 0; fi
  log "  FAILED train $rel (see $LOG/train_${cfg}_${label}_s${seed}.log)"; return 1
}

mark_consumed() { cfg_vars "$1"; local d=$FTO/pyg_graphs/$bb/$name/train/$graph; grep -qxF "$d" "$CONSUMED" 2>/dev/null || echo "$d" >> "$CONSUMED"; }

explain_done() {  # cfg method -> 0 if all 4 shards have results
  cfg_vars "$1"; local m=$2 k d
  for k in 1 2 3 4; do
    if [[ $m == subgraphx ]]; then d=$FTO/gnn_models/$bb/$name/$graph/explanations; [[ -f $d/subgraphx/${slug}_predtarget_shard${k}of4/results.pkl || -f $d/subgraphx_v2_rawmcts/${slug}_predtarget_shard${k}of4/results.pkl ]] || return 1
    else [[ -f $FTO/gnn_models/$bb/$name/$graph/explanations/graphsvx/${slug}_shard${k}of4/results.pkl ]] || return 1; fi
  done
}

explain() {  # cfg: SubgraphX + GraphSVX, 4 shards each, all concurrent (blocking)
  local cfg=$1; cfg_vars "$cfg"; local pids=() k
  if ! explain_done "$cfg" subgraphx; then
    log "explain $cfg subgraphx (4 shards)"
    for k in 0 1 2 3; do
      drun subgraphx -c "cd /app && OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP python -m src.explain.gnn.subgraphx.main --dataset $ds --graph-type $graph --backbone $bb \
        --split $test --device cuda:0 --fair --target-forward-passes 2000 --target-class predicted --run-tag predtarget --num-shards 4 --shard-index $k --no-progress" \
        > "$LOG/explain_${cfg}_subgraphx_shard${k}.log" 2>&1 & pids+=($!)
    done
  fi
  if ! explain_done "$cfg" graphsvx; then
    log "explain $cfg graphsvx (4 shards)"
    for k in 0 1 2 3; do
      drun graphsvx -c "cd /app && OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP python -m src.explain.gnn.graphsvx.main --dataset $ds --graph-type $graph --backbone $bb \
        --split $test --device cuda:0 --fair --target-forward-passes 2000 --seed 42 --num-shards 4 --shard-index $k --no-progress" \
        > "$LOG/explain_${cfg}_graphsvx_shard${k}.log" 2>&1 & pids+=($!)
    done
  fi
  wait "${pids[@]:-}" 2>/dev/null
  drun graphsvx -c "chown -R $OWNER /app/outputs/gnn_models/$bb/$name/$graph/explanations" >/dev/null 2>&1
  if explain_done "$cfg" subgraphx && explain_done "$cfg" graphsvx; then log "  explained $cfg"; else log "  FAILED explain $cfg (some shards missing, see $LOG/explain_${cfg}_*.log)"; fi
}

pass_main() {
  local pids=() cfg
  for cfg in "${CFGS[@]}"; do
    cfg_vars "$cfg"
    embed "$cfg" "$test" || continue
    embed "$cfg" train || continue
    train "$cfg" y 42 "gnn_models/$bb/$name/$graph" & local p1=$!
    sleep 30
    train "$cfg" true_label 42 "gnn_models_control/true_label/$bb/$name/$graph" & local p2=$!
    wait $p1; local ok1=$?; wait $p2; local ok2=$?
    (( ok1 == 0 && ok2 == 0 )) && mark_consumed "$cfg"
    (( ok1 == 0 )) && { explain "$cfg" & pids+=($!); }
  done
  log "main: waiting for background explainers"; wait "${pids[@]:-}" 2>/dev/null; log "main: done"
}

pass_controls() {
  local cfg
  for cfg in "${CFGS[@]}"; do
    cfg_vars "$cfg"
    [[ -f $FTO/gnn_models_control/y/$bb/$name/$graph/test_results.json && -f $FTO/gnn_models_control_seed1/true_label/$bb/$name/$graph/test_results.json && -f $FTO/gnn_models_control/true_label/$bb/$name/$graph/test_results.json ]] && continue
    embed "$cfg" "$test" || continue
    embed "$cfg" train || continue
    train "$cfg" true_label 42 "gnn_models_control/true_label/$bb/$name/$graph" || true
    train "$cfg" y 1 "gnn_models_control/y/$bb/$name/$graph" & local p1=$!
    sleep 30
    train "$cfg" true_label 1 "gnn_models_control_seed1/true_label/$bb/$name/$graph" & local p2=$!
    wait $p1; wait $p2
    mark_consumed "$cfg"
  done
  log "controls: done"
}

pass_derive() {
  log "derive: recompute SubgraphX metrics from raw MCTS"
  drun subgraphx -c "cd /app && for d in /app/outputs/gnn_models/*/*/*/explanations/subgraphx/*_predtarget_shard*; do \
      [ -f \$d/results.pkl ] || continue; raw=\$(dirname \$(dirname \$d))/subgraphx_v2_rawmcts; mkdir -p \$raw; [ -d \$raw/\$(basename \$d) ] || mv \$d \$raw/; done; \
      python -m src.explain.gnn.subgraphx.recompute_metrics 2>&1 | grep -v Warning | tail -12; chown -R $OWNER /app/outputs/gnn_models" > "$LOG/recompute.log" 2>&1
  tail -10 "$LOG/recompute.log" | sed 's/^/  /'
  log "derive: rebuild analytics (split, 4 dimensions, module datasets)"
  SKIP_LR=1 SKIP_LLM_AUC=1 bash scripts/09_rebuild_analytics.sh > "$LOG/rebuild_analytics.log" 2>&1 || log "  rebuild_analytics failed (see log)"
  tail -3 "$LOG/rebuild_analytics.log" | sed 's/^/  /'
  drun graphsvx -c "chown -R $OWNER /app/outputs/analytics /app/outputs/use_case" >/dev/null 2>&1
  local summ=$FTO/use_case/module_datasets/error_signal_classification_summary.csv
  if [[ -f $summ ]] && [[ -z $(find "$FTO/use_case/module_datasets" -name 'module_dataset_*.csv' -newer "$summ") ]]; then
    log "derive: LR summary up to date, skipping"
  else
    local n; n=$(cd "$FT" && python3 "$REPO/scripts/_lr_module.py" --count)
    log "derive: logistic regressions for $n module files ($LR_JOBS parallel)"
    rm -rf "$FTO/use_case/module_datasets/lr_parts"
    (cd "$FT" && seq 0 $((n - 1)) | xargs -P "$LR_JOBS" -I{} sh -c "python3 $REPO/scripts/_lr_module.py --index {} > $REPO/$LOG/lr_part_{}.log 2>&1")
    (cd "$FT" && python3 "$REPO/scripts/_lr_module.py" --merge) | tee -a "$LOG/driver.log"
  fi
  log "derive: token rankings (tokens_v2)"
  drun graphsvx -c "cd /app && PYTHONPATH=/app/.pip_ft python3 -m src.Analytics.tokens.extract_gnn --base-dir /app/outputs/gnn_models --graph-root /app/outputs/graphs \
      --pyg-root /app/outputs/pyg_graphs --output-dir /app/outputs/analytics/tokens_v2 --methods subgraphx graphsvx && chown -R $OWNER /app/outputs/analytics" > "$LOG/tokens_v2.log" 2>&1 || log "  tokens_v2 failed"
  log "derive: revision metrics and derived analyses -> $RES"
  mkdir -p "$RES" "$REPO/revision/manuscript_v2/tables_ft" "$REPO/revision/manuscript_v2/figures_ft"
  ( cd "$FT" && python3 "$REPO/src/use_case/revision_metrics.py" ) > "$LOG/revision_metrics.log" 2>&1 || log "  revision_metrics failed"
  cp "$FTO"/use_case/revision_metrics/*.csv "$RES/" 2>/dev/null
  ( cd "$FT" && python3 "$REPO/src/use_case/revision_tables.py" --output-dir "$RES" ) > "$LOG/revision_tables.log" 2>&1 || log "  revision_tables failed"
  ( cd "$FT" && python3 "$REPO/src/use_case/explanation_overlap.py" --output-dir "$RES/overlap" ) > "$LOG/overlap.log" 2>&1 || log "  overlap failed"
  ( cd "$FT" && python3 "$REPO/src/use_case/cross_model_detection.py" --output "$RES/cross_model_bert_errors.csv" ) > "$LOG/cross_model.log" 2>&1 || log "  cross-model failed"
  ( cd "$FT" && python3 "$REPO/src/use_case/control_error_overlap.py" --control-roots outputs/gnn_models_control outputs/gnn_models_control_seed1 --output "$RES/control_error_overlap.csv" ) > "$LOG/control.log" 2>&1 || log "  control failed"
  ( cd "$FT" && python3 "$REPO/src/use_case/manuscript_tables.py" --output-dir "$REPO/revision/manuscript_v2/tables_ft" --control-csv "$RES/control_error_overlap.csv" \
      --cross-model-csv "$RES/cross_model_bert_errors.csv" --overlap-dir "$RES/overlap" && python3 "$REPO/src/use_case/manuscript_figures.py" --output-dir "$REPO/revision/manuscript_v2/figures_ft" ) > "$LOG/manuscript_tables.log" 2>&1 || log "  manuscript tables/figures failed"
  log "derive: done ($(ls "$RES" | wc -l) files in $RES)"
}

log "==== START PASS=$PASS configs: ${CFGS[*]} (free $(free_gb) GB)"
case $PASS in
  main) pass_main ;;
  controls) pass_controls ;;
  derive) pass_derive ;;
  all) pass_main; pass_derive; pass_controls; pass_derive ;;
  *) echo "unknown PASS=$PASS"; exit 1 ;;
esac
log "==== ALL DONE (free $(free_gb) GB)"
