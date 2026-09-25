#!/usr/bin/env bash
# Remaining explainer runs after the disk incident, strictly sequential.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NUM_SHARDS=4 RUN_TAG=trees bash scripts/08_run_graphsvx_on_trees.sh ag_syntactic ag_constituency
NUM_SHARDS=4 RUN_TAG=predtarget bash scripts/07_rerun_subgraphx_predicted_target.sh sst2_window sst2_skipgrams ag_window ag_skipgrams
