#!/usr/bin/env python3
"""IMDb KD smoke test; resume completed generations (not interrupted epochs).

Install: python3 -m pip install torch transformers datasets numpy
Run: caffeinate -i python3 -u run_imdb_recursive_kd.py --device mps
Resume: repeat the same command with --resume.
M1 has the same teacher in both branches; MPS need not give identical outputs.
"""
import argparse
import csv
import gc
import json
import os
import platform
import random
import time
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    p.add_argument('--output', type=Path, default=Path.home() / 'Desktop/imdb_kd_smoke')
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--generations', type=int, default=1)
    p.add_argument('--teacher-epochs', type=int, default=1)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--train-size', type=int, default=200)
    p.add_argument('--eval-size', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--temperature', type=float, default=4.)
    p.add_argument('--model', default='google-bert/bert-base-uncased')
    p.add_argument('--revision', default='main')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    for name in ['generations', 'teacher_epochs', 'epochs', 'train_size',
                 'eval_size', 'batch_size', 'max_length']:
        if getattr(a, name) <= 0:
            p.error(name + ' must be positive')
    if a.temperature <= 0 or a.lr <= 0:
        p.error('temperature and lr must be positive')
    return a


def main():
    a = arguments()
    import numpy as np
    import torch
    import transformers
    import datasets
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS unavailable. Use an Apple Silicon Mac and MPS-enabled PyTorch.')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable.')
    device = torch.device(a.device)
    out = a.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(a).items() if k not in ['output', 'resume']}
    config['versions'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
                          'datasets': datasets.__version__, 'python': platform.python_version()}
    cfg_path = out / 'config.json'
    if cfg_path.exists():
        if not a.resume:
            raise SystemExit('Output already exists: use --resume or a new --output.')
        if json.loads(cfg_path.read_text()) != config:
            raise SystemExit('Resume configuration/version mismatch. Use original settings.')
    elif a.resume:
        raise SystemExit('No saved config to resume.')
    else:
        cfg_path.write_text(json.dumps(config, indent=2))

    def reseed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if a.device == 'mps':
            torch.mps.manual_seed(s)
        elif a.device == 'cuda':
            torch.cuda.manual_seed_all(s)

    def clear():
        gc.collect()
        if a.device == 'mps':
            torch.mps.empty_cache()
        elif a.device == 'cuda':
            torch.cuda.empty_cache()

    # Resolve and persist the immutable model revision before training.
    revision_file = out / 'model_revision.txt'
    if revision_file.exists():
        revision = revision_file.read_text().strip()
    else:
        revision = AutoConfig.from_pretrained(a.model, revision=a.revision)._commit_hash
        if not revision:
            raise RuntimeError('Could not resolve immutable model revision.')
        revision_file.write_text(revision)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=revision)
    ds = load_dataset('stanfordnlp/imdb')

    def subset(split, count, subset_seed):
        if count > len(split):
            raise ValueError('Requested subset exceeds split size.')
        labels = np.asarray(split['label'])
        rng = np.random.default_rng(subset_seed)
        pos = rng.permutation(np.flatnonzero(labels == 1))[:count // 2]
        neg = rng.permutation(np.flatnonzero(labels == 0))[:count - count // 2]
        idx = rng.permutation(np.concatenate([pos, neg]))
        return split.select(idx.tolist()), idx

    train, train_idx = subset(ds['train'], a.train_size, 2026)
    test, test_idx = subset(ds['test'], a.eval_size, 2027)
    selection = out / 'selection.npz'
    if selection.exists():
        with np.load(selection) as old:
            if not (np.array_equal(old['train'], train_idx) and np.array_equal(old['test'], test_idx)):
                raise RuntimeError('Dataset selection changed.')
    else:
        np.savez(selection, train=train_idx, test=test_idx)
    train_text = list(train['text'])
    test_text = list(test['text'])
    train_enc = tokenizer(train_text, truncation=True, padding='max_length',
                          max_length=a.max_length, return_tensors='pt')
    test_enc = tokenizer(test_text, truncation=True, padding='max_length',
                         max_length=a.max_length, return_tensors='pt')
    train_y = torch.tensor(list(train['label']), dtype=torch.long)
    test_y = np.asarray(test['label'])
    lengths = [len(x) for x in tokenizer(train_text, truncation=False)['input_ids']]
    (out / 'length_stats.json').write_text(json.dumps({
        'train_token_length_quantiles': np.quantile(lengths, [0, .5, .9, 1]).tolist(),
        'train_truncation_fraction': float(np.mean(np.asarray(lengths) > a.max_length))}, indent=2))

    def inputs(enc, ix):
        return {k: v[ix].to(device) for k, v in enc.items()}

    def logits(model, enc):
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(enc['input_ids']), a.batch_size):
                chunks.append(model(**inputs(enc, slice(start, start+a.batch_size))).logits.cpu())
        return torch.cat(chunks)

    def new_model(s):
        reseed(s)
        return AutoModelForSequenceClassification.from_pretrained(
            a.model, revision=revision, num_labels=2, attn_implementation='eager').to(device)

    def teacher_targets(path):
        model = AutoModelForSequenceClassification.from_pretrained(
            path, attn_implementation='eager').to(device)
        targets = torch.softmax(logits(model, train_enc) / a.temperature, dim=-1)
        del model
        clear()
        return targets

    def fit(model, targets, epochs, s):
        optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(a.train_size, generator=torch.Generator().manual_seed(s+epoch))
            total = 0.
            started = time.perf_counter()
            for start in range(0, a.train_size, a.batch_size):
                ix = order[start:start+a.batch_size]
                optim.zero_grad(set_to_none=True)
                z = model(**inputs(train_enc, ix)).logits
                if targets is None:
                    loss = torch.nn.functional.cross_entropy(z, train_y[ix].to(device))
                else:
                    loss = a.temperature**2 * torch.nn.functional.kl_div(
                        torch.log_softmax(z/a.temperature, dim=-1), targets[ix].to(device),
                        reduction='batchmean')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optim.step()
                total += loss.item() * len(ix)
            print(f'epoch {epoch+1}/{epochs}: loss={total/a.train_size:.6f}, '
                  f'seconds={time.perf_counter()-started:.1f}', flush=True)

    rows = []
    for seed in a.seeds:
        root = out / f'seed_{seed}'
        stages = [('teacher', 0, root/'M0', None, seed, a.teacher_epochs)]
        for branch in ['recursive', 'fixed']:
            for t in range(1, a.generations+1):
                teacher = root/'M0' if branch == 'fixed' or t == 1 else root/branch/f'M{t-1}'
                stages.append((branch, t, root/branch/f'M{t}', teacher,
                               (seed+100003*t) % (2**32), a.epochs))
        for branch, t, path, teacher, s, epochs in stages:
            marker = path/'complete.json'
            if marker.exists():
                row = json.loads(marker.read_text())
                rows.append(row)
                print(f'SKIP completed {branch} M{t}, seed={seed}', flush=True)
                continue
            print(f'START {branch} M{t}, seed={seed}', flush=True)
            targets = teacher_targets(teacher) if teacher is not None else None
            model = new_model(s)
            fit(model, targets, epochs, s)
            probs = torch.softmax(logits(model, test_enc), dim=-1).numpy()
            path.mkdir(parents=True, exist_ok=True)
            model.to('cpu').save_pretrained(path)
            tokenizer.save_pretrained(path)
            np.savez(path/'predictions.npz', probabilities=probs, labels=test_y, indices=test_idx)
            row = dict(seed=seed, branch=branch, generation=t,
                       accuracy=float(np.mean(probs.argmax(1) == test_y)))
            temporary = path/'complete.tmp'
            temporary.write_text(json.dumps(row, indent=2))
            os.replace(temporary, marker)
            rows.append(row)
            print(f'SAVED {path}: accuracy={row["accuracy"]:.4f}', flush=True)
            del model, targets
            clear()
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'branch', 'generation', 'accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'DONE: {out}/metrics.csv', flush=True)


if __name__ == '__main__':
    main()
