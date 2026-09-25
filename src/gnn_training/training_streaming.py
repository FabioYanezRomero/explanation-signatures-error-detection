#!/usr/bin/env python3
"""Shard-streaming GCN trainer used for the revision control experiments.

Same model (GNNClassifier), hyper-parameters, optimiser, scheduler and checkpoint
selection rule (lowest test loss, patience 5) as src/gnn_training/training.py, but
the training set is streamed shard by shard with a shuffle buffer instead of
re-loading a whole shard for every single graph, and the training target can be
either the teacher label (``y``) or the gold label (``true_label``).

After training, the best checkpoint is evaluated on the test split and a CSV with
one row per test graph (data_index, teacher label, gold label, prediction,
confidence, probability vector) is written next to the checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch, Data

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from gnn_training.gnn_eval_model import GNNClassifier  # noqa: E402

KEEP = ("x", "edge_index", "y", "true_label", "data_index")


def _shard_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not files:
        raise ValueError(f"No .pt shards in {data_dir}")
    return files


def _slim(graph: Data, label_attr: str) -> Data:
    """Keep only the tensors needed for training; set ``y`` from ``label_attr``."""
    out = Data()
    for key in KEEP:
        if hasattr(graph, key) and getattr(graph, key) is not None:
            val = getattr(graph, key)
            if key in ("y", "true_label"):
                val = torch.as_tensor(val).view(-1).long()
            elif key == "data_index":
                val = torch.as_tensor([int(val)]).long()
            setattr(out, key, val)
    if label_attr != "y":
        out.y = getattr(out, label_attr).clone()
    if not hasattr(out, "true_label"):
        out.true_label = torch.tensor([-1])
    return out


def _load_shard(path: str, label_attr: str) -> List[Data]:
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    graphs = graphs if isinstance(graphs, list) else [graphs]
    return [_slim(g, label_attr) for g in graphs]


def train_batches(files: List[str], *, label_attr: str, batch_size: int, buffer_shards: int, rng: random.Random) -> Iterator[Batch]:
    """Yield shuffled mini-batches: shard order is shuffled, then graphs are shuffled
    inside a buffer of ``buffer_shards`` shards, then batched."""
    order = list(files)
    rng.shuffle(order)
    for start in range(0, len(order), buffer_shards):
        buf: List[Data] = []
        for f in order[start:start + buffer_shards]:
            buf.extend(_load_shard(f, label_attr))
        rng.shuffle(buf)
        for i in range(0, len(buf), batch_size):
            yield Batch.from_data_list(buf[i:i + batch_size])
        del buf


def eval_batches(files: List[str], *, label_attr: str, batch_size: int) -> Iterator[Batch]:
    for f in files:
        graphs = _load_shard(f, label_attr)
        for i in range(0, len(graphs), batch_size):
            yield Batch.from_data_list(graphs[i:i + batch_size])


def run_epoch(model, batches, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in batches:
            batch = batch.to(device)
            out = model(batch)
            loss = F.cross_entropy(out, batch.y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.y.size(0)
            correct += (out.argmax(1) == batch.y).sum().item()
            n += batch.y.size(0)
    return total_loss / max(n, 1), 100.0 * correct / max(n, 1)


def dump_predictions(model, files, device, batch_size, path: Path) -> pd.DataFrame:
    model.eval()
    rows = []
    position = 0  # running index over the sorted shards = the explainers' global_graph_index
    with torch.no_grad():
        for batch in eval_batches(files, label_attr="y", batch_size=batch_size):
            batch = batch.to(device)
            probs = F.softmax(model(batch), dim=1).cpu().numpy()
            for i in range(probs.shape[0]):
                rows.append(dict(position=position, data_index=int(batch.data_index[i]), teacher_label=int(batch.y[i]), true_label=int(batch.true_label[i]),
                                 prediction_class=int(probs[i].argmax()), prediction_confidence=float(probs[i].max()),
                                 **{f"prob_{c}": float(probs[i, c]) for c in range(probs.shape[1])}))
                position += 1
    df = pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data_dir", required=True)
    ap.add_argument("--test_data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--label_attr", choices=["y", "true_label"], default="y", help="training target: teacher label (y) or gold label (true_label)")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--module", default="GCNConv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--buffer_shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train_shards", type=int, default=None, help="debug: use only the first N shards")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(a) | {"device": str(device), "trainer": "training_streaming.py"}, indent=2))

    train_files = _shard_files(a.train_data_dir)
    if a.limit_train_shards:
        train_files = train_files[:a.limit_train_shards]
    test_files = _shard_files(a.test_data_dir)
    in_dim = torch.load(train_files[0], map_location="cpu", weights_only=False)[0].x.size(1)

    model = GNNClassifier(in_dim, a.hidden_dim, a.num_classes, num_layers=a.num_layers, dropout=a.dropout, module=a.module, pooling=a.pooling).to(device)
    optimizer = Adam(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    history = {k: [] for k in ("train_loss", "train_acc", "test_loss", "test_acc", "epoch_seconds")}
    best_loss, best_epoch, bad = float("inf"), -1, 0
    best_path = out_dir / "best_model.pth"
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_batches(train_files, label_attr=a.label_attr, batch_size=a.batch_size, buffer_shards=a.buffer_shards, rng=rng), device, optimizer)
        te_loss, te_acc = run_epoch(model, eval_batches(test_files, label_attr=a.label_attr, batch_size=a.batch_size), device)
        scheduler.step(te_loss)
        for k, v in zip(history, (tr_loss, tr_acc, te_loss, te_acc, time.time() - t0)):
            history[k].append(v)
        improved = te_loss < best_loss
        if improved:
            best_loss, best_epoch, bad = te_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "test_loss": te_loss, "test_acc": te_acc}, best_path)
        else:
            bad += 1
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.2f} | test loss {te_loss:.4f} acc {te_acc:.2f} | {time.time()-t0:.0f}s{' *' if improved else ''}", flush=True)
        (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        if bad >= a.patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])
    df = dump_predictions(model, test_files, device, a.batch_size, out_dir / "test_predictions.csv")
    res = {"best_epoch": best_epoch, "best_test_loss": best_loss, "label_attr": a.label_attr, "n_test": int(len(df)),
           "accuracy_vs_teacher": float((df.prediction_class == df.teacher_label).mean()),
           "accuracy_vs_gold": float((df.prediction_class == df.true_label).mean()) if (df.true_label >= 0).all() else None}
    (out_dir / "test_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
