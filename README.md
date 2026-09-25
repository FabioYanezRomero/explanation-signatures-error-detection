# Do Explanation Signatures Detect Classification Errors? A Controlled Comparison of Graph and Token Explainers

Code, results and manuscript sources for the paper *"Do Explanation Signatures Detect Classification Errors? A Controlled Comparison of Graph and Token Explainers"* (Yáñez-Romero, Montoyo, Suárez, Gutiérrez and Mitkov; under review at *Pattern Recognition*, manuscript PR-D-26-08604).

The repository was previously named *Graph-Neural-Networks-Enable-Superior-Error-Detection-in-NLP-Explainability-than-Language-Models*; the old URL redirects here. The claim in that name did not survive the revision of the paper (see *What changed in the revision* below). The full commit history is kept.

---

## What the paper asks

A fine-tuned language model (BERT) acts as teacher for graph neural network surrogates (GCN) trained on four graph representations of the same texts (constituency and dependency trees, window and skip-gram graphs), so that graph and token explanations refer to the same decision function. Three Shapley-style explainers (SubgraphX, GraphSVX on the graphs; TokenSHAP on the language model) run under a common budget, sparsity and receptive field, and four families of explanation metrics feed error detectors that are always evaluated against majority-class and confidence baselines. Three questions:

1. Does the representation (graph topology versus tokens) or the explainer determine the signature of an explanation?
2. Do explanation-derived features detect classification errors beyond what the model's confidence already provides?
3. What does a graph surrogate trained as the student of a language model inherit from its teacher: its errors, its confidence, or its attributions?

## What the paper finds

- **Signatures do not separate graphs from tokens.** Within the budget every explanation, on every representation, is sufficient and not necessary; degrees of sufficiency differ between explainers, not between discrete and continuous representations.
- **Explanations as error signals add nothing to a language model's confidence.** Explanation features from every explainer and representation carry information about correctness (AUROC 0.79–0.86 used alone), but none beyond the confidence of the language model, for linear and non-linear detectors, for both definitions of error and in the high-confidence regime. No graph configuration detects errors better than the token explainer of the language model itself.
- **What distillation transfers.** Against a gold-trained control with identical architecture and data, distilled surrogates make the same errors on the same instances (the errors reach the surrogate through the node features), but they are confident in the teacher's errors about twice as often. Explanation features help a distilled surrogate only where its confidence was damaged by distillation, and never carry it to the level of the language model. Surrogate attributions overlap with the teacher's at chance level.

## What changed in the revision

The submitted version reported 99.7–100% error-detection accuracy for hierarchical graphs against 88–90% for tokens. Auditing the pipeline in response to the reviewers showed that this result came from defects, all corrected here:

| Defect | Effect | Fix |
|---|---|---|
| SubgraphX explained the label stored with the instance (the teacher's prediction) instead of the class the surrogate predicted | for every disagreement with the teacher the explained class had near-zero confidence, so every confidence-based metric leaked the target | every explainer targets the predicted class; SubgraphX re-run (`scripts/07`) |
| Error detectors trained and selected per *true* class | the true class is unknown at inference time and reveals the error | detectors per *predicted* class and pooled (`src/use_case/revision_metrics.py`) |
| SubgraphX metrics computed on DIG's first result (the near-full graph) instead of the best coalition within the budget | explanations covered 80–94% of the nodes; spurious "necessity" signature | coalition within the 20% budget; metrics recomputed (`src/explain/gnn/subgraphx/recompute_metrics.py`) |
| SubgraphX node ranking spanned the whole graph | Dimension-1 and -2 progressions not comparable with the other explainers | progressions truncated to the budget (`src/use_case/truncate_progressions_to_budget.py`) |
| Node features from the pre-trained encoder, labels from the fine-tuned teacher | surrogate less faithful to its teacher | features and labels from the same fine-tuned checkpoint (`scripts/12`); the pre-trained setting is kept as an ablation |

Both explainers now run on all four graph types (2 explainers × 4 topologies × 2 datasets, plus TokenSHAP), and the protocol adds confidence-based baselines, nested detectors with bootstrap intervals, a gold-trained twin of every surrogate, a node-feature ablation and a cross-model detector.

---

## Repository structure

```
├── scripts/                     # numbered pipeline scripts
│   ├── 01-06_*.sh               #   original pipeline: fine-tune, graphs, embeddings, GNNs, explainers, analytics
│   ├── 07-13_*.sh               #   revision: predicted-class SubgraphX, GraphSVX on trees, control GNNs,
│   │                            #   fine-tuned-feature rerun, re-derivation of every table and figure
├── src/
│   ├── finetuning/              # LLM fine-tuning (BERT)
│   ├── embeddings/              # node features from the fine-tuned (or pre-trained) encoder
│   ├── graph_builders/          # text-to-graph conversion (constituency, dependency, window, skip-gram)
│   ├── convert/                 # NetworkX -> PyTorch Geometric
│   ├── gnn_training/            # GCN surrogates (LLM-as-teacher) and gold-trained controls
│   ├── explain/                 # SubgraphX, GraphSVX, TokenSHAP wrappers under a common budget
│   ├── Analytics/               # the four evaluation dimensions (auc, progression, consistency, fidelity)
│   └── use_case/                # error detectors, controls, and the generators of every table and figure
├── revision/
│   ├── results_ft/              # result CSVs of the fine-tuned-feature run (the numbers in the paper)
│   ├── results/                 # result CSVs of the pre-trained-feature run (ablation)
│   └── manuscript_v2/           # LaTeX sources, generated table bodies and figures of the revised paper
├── tests/                       # pytest suite
├── docker/, docker-compose.yml  # one container per explainer (see below)
└── outputs/                     # generated artefacts (gitignored)
```

## Paper-to-code mapping

| Paper section | Code |
|---|---|
| 3.1 Text-to-graph conversion | `src/graph_builders/` |
| 3.2 Fine-tuning and node features | `src/finetuning/`, `src/embeddings/` |
| 3.3 GNN surrogates and gold-trained controls | `src/gnn_training/`, `scripts/10_train_control_gnns_paired.sh` |
| 3.4 Explainers under a common budget | `src/explain/gnn/`, `src/explain/llm/`, `src/explain/gnn/subgraphx/recompute_metrics.py` |
| 3.5 The four dimensions (Table 1) | `src/Analytics/`, `src/use_case/build_module_datasets.py`, `src/use_case/feature_config.py` |
| 3.6 Error detectors, baselines, bootstrap | `src/use_case/revision_metrics.py`, `nested_coefficients.py`, `nested_gbm.py` |
| 4.2 Dimensions 1–4 | `trajectory_correlations.py`, `maskout_drop_sign.py`, `margin_flips.py`, `coalition_phrase_share.py` |
| 4.3 Detection against baselines, cross-model, second opinion | `cross_model_detection.py`, `surrogate_gap_vs_bert.py`, `explanation_overlap.py` |
| 4.4 Calibration, gold-trained control, ablation | `control_error_overlap.py`, `control_agreement_on_bert_errors.py`, `surrogate_error_overlap_dg.py`, `ablation_table_blocks.py` |
| Tables 2–6, Figures 3–5, Tables S1–S14 | `manuscript_tables.py`, `manuscript_figures.py`, `supplementary_tables.py`, `merge_supplementary_tables.py` |

Every table body in the paper and the supplementary material is a file generated by these scripts from the CSVs in `revision/results_ft/` and `revision/results/`; `scripts/13_rederive_after_truncation.sh` regenerates all of them.

---

## Reproducing the experiments

### Requirements

Docker and Docker Compose v2, an NVIDIA GPU with CUDA and the NVIDIA Container Toolkit.

### Original pipeline

```bash
make build        # build the containers
make up           # start them
make reproduce    # steps 1-6: fine-tune, graphs, embeddings, GNNs, explainers, analytics
```

Or step by step: `make step-1-finetune`, `make step-2-graphs`, `make step-3-embeddings`, `make step-4-train`, `make step-5-explain`, `make step-6-analytics`. Each script supports `--help` and `--dry-run`.

### Revision pipeline

The scripts `scripts/07_*` to `scripts/13_*` reproduce the corrected results in order: SubgraphX with the predicted class as target (07), GraphSVX on the tree graphs (08), analytics rebuild (09), gold-trained and seed-replicate surrogates (10), the fine-tuned-feature rerun of the whole pipeline (12) and the re-derivation of every CSV, table and figure (13). Paths to the experiment artefacts are set in `docker-compose.override.yml` (machine-specific, not versioned).

### Containers

| Container | Purpose |
|---|---|
| `app` | training, analytics, detectors |
| `subgraphx` | SubgraphX explainer (all four graph types) |
| `graphsvx` | GraphSVX explainer (all four graph types) |
| `tokenshap` | TokenSHAP on the language model |

### Tests

```bash
docker compose exec -w /app app pytest tests/ -v
```

---

## Datasets and explainers

- **AG News** (Zhang et al., 2015): four-class topic classification, 120,000 training and 7,600 test instances.
- **SST-2** (Socher et al., 2013): binary sentiment, 67,349 training instances; the 872 labelled validation instances serve as test set.

| Explainer | Model | Representations |
|---|---|---|
| SubgraphX | GCN surrogate | constituency, dependency, window, skip-gram |
| GraphSVX | GCN surrogate | constituency, dependency, window, skip-gram |
| TokenSHAP | BERT | tokens (aggregated to words) |

---

## Citation

```bibtex
@article{yanez2026signatures,
  title   = {Do Explanation Signatures Detect Classification Errors? A Controlled Comparison of Graph and Token Explainers},
  author  = {Y{\'a}{\~n}ez-Romero, Fabio and Montoyo, Andr{\'e}s and Su{\'a}rez, Armando and Guti{\'e}rrez, Yoan and Mitkov, Ruslan},
  year    = {2026},
  note    = {Under review at Pattern Recognition}
}
```

## License

See [LICENSE](LICENSE).
