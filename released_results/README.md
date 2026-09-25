# Result files

CSV outputs of the analysis scripts in `src/use_case/`, from which every table and figure of the paper and of the supplementary material is generated (`src/use_case/manuscript_tables.py`, `manuscript_figures.py`, `supplementary_tables.py`, `ablation_table_blocks.py`, `merge_supplementary_tables.py`).

- `fine_tuned_features/`: surrogates with node features from the fine-tuned teacher (the numbers reported in the paper).
- `pre_trained_features/`: surrogates with node features from the pre-trained encoder (the node-feature ablation, Section 4.4.3).

The driver `scripts/13_rederive_after_truncation.sh` regenerates both folders from the experiment artefacts.
