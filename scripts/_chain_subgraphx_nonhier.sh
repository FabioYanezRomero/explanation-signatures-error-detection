#!/usr/bin/env bash
# Helper: run the non-hierarchical SubgraphX configurations (2x2 design completion).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NUM_SHARDS=${NUM_SHARDS:-4} RUN_TAG=${RUN_TAG:-predtarget} bash scripts/07_rerun_subgraphx_predicted_target.sh sst2_window sst2_skipgrams ag_window ag_skipgrams
