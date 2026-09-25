#!/usr/bin/env python3

"""
Efficient embedding generation script that works with existing graph files
and loads labels from predictions.json to avoid dataset downloads.
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from ..convert.nx_to_pyg import nx_list_to_pyg
from .generate import (
    iter_tree_batches,
    count_graphs_in_dir,
    collect_constituency_special_labels,
    is_special_label,
    compute_special_embeddings,
    get_word_embeddings,
    clean_graph_whitespace_nodes,
    normalize_special_labels,
    validate_graph_structure,
    _load_finetuned_weights_if_any
)

def load_labels_from_predictions(predictions_file, dataset_name, split, epoch):
    """Load labels from predictions.json file."""
    print(f"Loading labels from: {predictions_file}")

    with open(predictions_file, 'r') as f:
        all_predictions = f.read()

    # Parse JSON efficiently
    import json
    predictions = json.loads(all_predictions)

    # Filter predictions for this epoch and split
    split_predictions = [
        p for p in predictions
        if p['epoch'] == epoch and p['dataset'] == split
    ]

    print(f"Found {len(split_predictions)} predictions for {split} split, epoch {epoch}")

    # Create mapping from data_index to predicted_label
    index_to_label = {int(p['data_index']): int(p['predicted_label']) for p in split_predictions}

    return index_to_label

def _model_hidden_size(model: AutoModel) -> int:
    """Return a best-effort estimate of the model's hidden size for zero initialisation."""
    for attr in ("hidden_size", "d_model", "model_dim", "word_embed_proj_dim"):
        if hasattr(model.config, attr):
            val = getattr(model.config, attr)
            if isinstance(val, int):
                return val
    first_param = next(model.parameters(), None)
    if first_param is None:
        raise ValueError("Unable to determine model hidden size")
    return first_param.shape[-1]


def _list_pickle_batches(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {directory}")
    return sorted(files)


def _iter_knowledge_records(tree_dir):
    for fname in _list_pickle_batches(tree_dir):
        path = os.path.join(tree_dir, fname)
        with open(path, 'rb') as f:
            obj = pkl.load(f)
        if not isinstance(obj, dict) or 'records' not in obj:
            raise ValueError(
                f"Knowledge graph file {fname} must be a dict with a 'records' key"
            )
        dataset = obj.get('dataset')
        split = obj.get('split')
        records = obj['records']
        if not isinstance(records, list):
            raise ValueError(f"Knowledge graph file {fname} has non-list 'records'")
        for record in records:
            if not isinstance(record, dict) or 'graph' not in record:
                raise ValueError(
                    f"Knowledge graph record in {fname} must contain a 'graph' key"
                )
            graph = record['graph']
            if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                raise ValueError(
                    f"Knowledge graph record in {fname} has invalid 'graph' type: {type(graph)}"
                )
            label = record.get('label')
            record_id = record.get('record_id')
            label_text = record.get('label_text')

            # Attach basic metadata so downstream steps can reuse it.
            if dataset is not None:
                graph.graph['dataset'] = dataset
            if split is not None:
                graph.graph['split'] = split
            if record_id is not None:
                graph.graph['record_id'] = record_id
            if label_text is not None:
                graph.graph['label_text'] = label_text

            yield graph, label


def iter_knowledge_batches(tree_dir, batch_size, *, return_labels: bool = False):
    graph_buffer = []
    label_buffer = [] if return_labels else None
    for graph, label in _iter_knowledge_records(tree_dir):
        graph_buffer.append(graph)
        if return_labels:
            label_buffer.append(label)
        if len(graph_buffer) == batch_size:
            if return_labels:
                yield graph_buffer, label_buffer
                graph_buffer, label_buffer = [], []
            else:
                yield graph_buffer
                graph_buffer = []
    if graph_buffer:
        if return_labels:
            yield graph_buffer, label_buffer
        else:
            yield graph_buffer


def count_knowledge_graphs(tree_dir):
    count = 0
    for _graph, _label in _iter_knowledge_records(tree_dir):
        count += 1
    return count


def _extract_knowledge_text(node_id, data):
    if not isinstance(data, dict):
        return str(node_id)
    props = data.get('properties')
    if isinstance(props, dict):
        for key in ('text', 'name', 'label', 'value', 'description', 'surface'):
            value = props.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for key in ('text', 'label', 'name'):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(node_id, str) and node_id.strip():
        return node_id
    return str(node_id)


def _compute_cls_embeddings(texts, model, tokenizer, device, hidden_size):
    embeddings = [np.zeros(hidden_size, dtype=np.float32) for _ in texts]
    indexed_texts = [
        (idx, text.strip())
        for idx, text in enumerate(texts)
        if isinstance(text, str) and text.strip()
    ]
    if not indexed_texts:
        return embeddings
    indices, clean_texts = zip(*indexed_texts)
    enc = tokenizer(
        list(clean_texts),
        return_tensors='pt',
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    for idx, vec in zip(indices, cls_embeddings):
        embeddings[idx] = vec
    return embeddings


def _sorted_constituency_word_nodes(graph):
    nodes = []
    for node_id, data in graph.nodes(data=True):
        label = data.get('label', '')
        if not is_special_label(label):
            nodes.append((node_id, data))
    nodes.sort(key=lambda item: item[1].get('id', 0))
    return nodes


def _sorted_text_nodes(graph):
    nodes = []
    for node_id, data in graph.nodes(data=True):
        text = data.get('text')
        if not isinstance(text, str):
            continue
        node_type = data.get('type')
        if node_type is None or node_type in {'word', 'token'}:
            nodes.append((node_id, data))
    nodes.sort(key=lambda item: (
        item[1].get('token_id'),
        item[1].get('id'),
        item[0]
    ))
    return nodes


def _tokens_from_nodes(nodes, key):
    tokens = []
    for _, data in nodes:
        value = data.get(key)
        if isinstance(value, str):
            tokens.append(value)
    return tokens


def _sentence_from_tokens(tokens):
    return " ".join(tokens)


def process_graph_batch(batch_graphs, model, tokenizer, device, graph_type,
                       special_embeddings=None, batch_idx=0, hidden_size=None):
    """Process a batch of graphs to add contextual embeddings."""
    processed_graphs = []
    special_embeddings = special_embeddings or {}
    for idx_in_batch, graph in enumerate(tqdm(batch_graphs,
                                             desc=f'Processing graphs in batch {batch_idx}',
                                             leave=False, unit='graph')):

        zero_reference = None

        try:
            if graph_type == 'knowledge':
                node_items = list(graph.nodes(data=True))
                node_texts = [_extract_knowledge_text(node_id, data) for node_id, data in node_items]
                graph.graph['node_texts'] = node_texts
                sentence = " ".join(
                    text.strip()
                    for text in node_texts
                    if isinstance(text, str) and text.strip()
                )
                graph.graph['sentence'] = sentence
                node_embs = _compute_cls_embeddings(
                    node_texts,
                    model,
                    tokenizer,
                    device,
                    hidden_size,
                )
                zero_reference = np.zeros(hidden_size, dtype=np.float32)
                for (node_id, data), emb in zip(node_items, node_embs):
                    data['embedding'] = emb if emb is not None else zero_reference
                if not isinstance(graph.graph.get('sentence'), str):
                    graph.graph['sentence'] = ''
            else:
                if graph_type == 'constituency':
                    clean_graph_whitespace_nodes(graph)
                    normalize_special_labels(graph)
                    validate_graph_structure(
                        graph,
                        graph_idx=batch_idx * len(batch_graphs) + idx_in_batch,
                        allow_special_leaves=True,
                    )
                    word_nodes = _sorted_constituency_word_nodes(graph)
                    tokens = _tokens_from_nodes(word_nodes, 'label')
                else:
                    word_nodes = _sorted_text_nodes(graph)
                    tokens = _tokens_from_nodes(word_nodes, 'text')

                sentence = _sentence_from_tokens(tokens)
                graph.graph['sentence'] = sentence

                if not sentence.strip():
                    raise ValueError('Reconstructed sentence is empty')

                word_embs = get_word_embeddings(sentence, model, tokenizer, device)

                if len(word_embs) != len(word_nodes):
                    raise ValueError(
                        f"Word embedding count mismatch (words={len(word_nodes)}, embeddings={len(word_embs)})"
                    )

                for (node_id, data), emb in zip(word_nodes, word_embs):
                    data['embedding'] = emb

                if word_embs:
                    zero_reference = np.zeros_like(word_embs[0])
                elif hidden_size:
                    zero_reference = np.zeros(hidden_size, dtype=np.float32)

                if graph_type == 'constituency':
                    if special_embeddings:
                        special_zero = np.zeros_like(next(iter(special_embeddings.values())))
                    elif zero_reference is not None:
                        special_zero = zero_reference
                    else:
                        special_zero = np.zeros(hidden_size or 0, dtype=np.float32)

                    for _, data in graph.nodes(data=True):
                        label = data.get('label')
                        if is_special_label(label):
                            data['embedding'] = special_embeddings.get(label, special_zero)
                else:
                    fallback = zero_reference if zero_reference is not None else np.zeros(hidden_size or 0, dtype=np.float32)
                    for _, data in graph.nodes(data=True):
                        if 'embedding' not in data:
                            data['embedding'] = fallback

        except Exception as e:
            print(f"Error processing graph {idx_in_batch} in batch {batch_idx}: {e}")
            zero_vec = np.zeros(hidden_size or 0, dtype=np.float32)
            for _, data in graph.nodes(data=True):
                data['embedding'] = zero_vec
            graph.graph['sentence'] = ''

        processed_graphs.append(graph)

    return processed_graphs

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings efficiently from existing graphs.")
    parser.add_argument('--graph_type', type=str, choices=['constituency', 'syntactic', 'window', 'ngrams', 'skipgrams', 'knowledge'],
                       help='Type of graph to process', default='constituency')
    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/sst2')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--tree_dir', type=str, required=True,
                       help='Directory containing graph .pkl files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for embeddings')
    parser.add_argument('--pyg_output_dir', type=str, required=True,
                       help='Output directory for PyG graphs')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--weights_path', type=str, default='',
                       help='Path to finetuned model checkpoint')
    parser.add_argument('--predictions_file', type=str, default='',
                       help='Path to predictions.json file')
    parser.add_argument('--epoch', type=int, default=0,
                       help='Epoch number for predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing graphs')
    parser.add_argument('--max_sentences', type=int, default=None,
                       help='Maximum number of sentences to process (for testing)')
    parser.add_argument('--no_embedding_pickles', action='store_true',
                       help='Do not write the intermediate NetworkX pickles to --output_dir (PyG shards only)')

    args = parser.parse_args()

    print(f"Processing {args.dataset_name}/{args.split}/{args.graph_type}")
    print(f"Input: {args.tree_dir}")
    print(f"Output: {args.output_dir}")
    print(f"PyG Output: {args.pyg_output_dir}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pyg_output_dir, exist_ok=True)

    print("Counting graphs...")
    if args.graph_type == 'knowledge':
        total_graphs = count_knowledge_graphs(args.tree_dir)
    else:
        total_graphs = count_graphs_in_dir(args.tree_dir)
    total_batches = (total_graphs + args.batch_size - 1) // args.batch_size if total_graphs else 0
    print(f"Found {total_graphs} graphs (approx. {total_batches} batches)")
    if args.max_sentences:
        total_graphs = min(total_graphs, args.max_sentences)
        total_batches = (total_graphs + args.batch_size - 1) // args.batch_size if total_graphs else 0
        print(f"Limiting to {total_graphs} samples due to --max_sentences")

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    # Load finetuned weights if provided
    if args.weights_path:
        print(f"Loading finetuned weights from: {args.weights_path}")
        model = _load_finetuned_weights_if_any(model, args.weights_path, strict=True)

    model = model.to(args.device)
    model.eval()

    hidden_size = _model_hidden_size(model)

    # Compute special embeddings for constituency graphs
    special_embeddings = {}
    if args.graph_type == 'constituency':
        special_labels = collect_constituency_special_labels(args.tree_dir)
        print(f"Computing embeddings for {len(special_labels)} special labels...")
        special_embeddings = compute_special_embeddings(
            sorted(special_labels), model, tokenizer, args.device
        )

    # Load labels from predictions if provided
    index_to_label = {}
    if args.predictions_file and os.path.isfile(args.predictions_file):
        index_to_label = load_labels_from_predictions(
            args.predictions_file, args.dataset_name, args.split, args.epoch
        )
        print(f"Loaded {len(index_to_label)} predicted labels from checkpoint")

    # Process batches
    print("Processing batches...")
    processed_count = 0
    embedding_batch_idx = 0
    pyg_batch_idx = 0

    if args.graph_type == 'knowledge':
        batch_iter = iter_knowledge_batches(args.tree_dir, args.batch_size, return_labels=True)
    else:
        batch_iter = iter_tree_batches(args.tree_dir, args.batch_size, return_labels=True)
    for batch_idx, (graph_batch, label_batch) in enumerate(tqdm(batch_iter,
                                                                desc='Processing batches',
                                                                unit='batch',
                                                                total=total_batches if total_batches else None)):
        if args.max_sentences and processed_count >= args.max_sentences:
            break

        remaining = None
        if args.max_sentences:
            remaining = args.max_sentences - processed_count
            if remaining <= 0:
                break

        if remaining is not None and remaining < len(graph_batch):
            graph_batch = graph_batch[:remaining]
            label_batch = label_batch[:remaining]

        processed_graphs = process_graph_batch(
            graph_batch,
            model, tokenizer, args.device, args.graph_type,
            special_embeddings, batch_idx, hidden_size=hidden_size
        )

        for local_idx, graph in enumerate(processed_graphs):
            data_index = processed_count + local_idx
            graph.graph['data_index'] = data_index
            true_label = label_batch[local_idx] if label_batch is not None else None
            if true_label is not None:
                graph.graph['true_label'] = int(true_label)
            pred_label = index_to_label.get(data_index) if index_to_label else None
            if pred_label is None and true_label is not None:
                pred_label = int(true_label)
            graph.graph['predicted_label'] = int(pred_label) if pred_label is not None else None

        if not args.no_embedding_pickles:
            batch_path = os.path.join(
                args.output_dir,
                f'{embedding_batch_idx:05d}.pkl'
            )
            with open(batch_path, 'wb') as f:
                pkl.dump(processed_graphs, f)
            print(f"Saved batch {embedding_batch_idx} to {batch_path}")

        pyg_graphs = nx_list_to_pyg(processed_graphs)

        for graph_meta, pyg_graph in zip(processed_graphs, pyg_graphs):
            pred_label = graph_meta.graph.get('predicted_label')
            if pred_label is None:
                pred_label = 0
            pyg_graph.y = torch.tensor([pred_label], dtype=torch.long)

            data_index = graph_meta.graph.get('data_index')
            if data_index is not None:
                pyg_graph.data_index = int(data_index)

            true_label = graph_meta.graph.get('true_label')
            if true_label is not None:
                pyg_graph.true_label = torch.tensor([int(true_label)], dtype=torch.long)

            sentence = graph_meta.graph.get('sentence')
            if sentence is not None:
                pyg_graph.sentence = sentence

        pyg_batch_path = os.path.join(
            args.pyg_output_dir,
            f'{pyg_batch_idx:05d}.pt'
        )
        torch.save(pyg_graphs, pyg_batch_path)
        print(f"Saved PyG batch {pyg_batch_idx} to {pyg_batch_path}")

        embedding_batch_idx += 1
        pyg_batch_idx += 1
        processed_count += len(processed_graphs)

        del processed_graphs

    print("✅ Embedding generation completed!")
    print(f"Generated {processed_count} embedded graphs")
    print(f"Generated {pyg_batch_idx} PyG batches")

if __name__ == "__main__":
    main()
