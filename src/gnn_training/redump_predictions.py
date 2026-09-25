#!/usr/bin/env python3
"""Re-create test_predictions.csv (with the positional index) from saved best checkpoints."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))
from gnn_training.gnn_eval_model import GNNClassifier  # noqa: E402
from gnn_training.training_streaming import _shard_files, dump_predictions  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--roots", nargs="+", required=True); a = ap.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for root in a.roots:
    for cfg_path in sorted(Path(root).rglob("config.json")):
        d = cfg_path.parent
        if not (d / "best_model.pth").exists():
            continue
        c = json.loads(cfg_path.read_text())
        test_files = _shard_files(c["test_data_dir"])
        in_dim = torch.load(test_files[0], map_location="cpu", weights_only=False)[0].x.size(1)
        model = GNNClassifier(in_dim, c["hidden_dim"], c["num_classes"], num_layers=c["num_layers"], dropout=c["dropout"], module=c["module"], pooling=c["pooling"]).to(device)
        model.load_state_dict(torch.load(d / "best_model.pth", map_location=device, weights_only=False)["model_state_dict"])
        df = dump_predictions(model, test_files, device, c["batch_size"], d / "test_predictions.csv")
        print(d, len(df), "acc_gold", round(float((df.prediction_class == df.true_label).mean()), 4), flush=True)
