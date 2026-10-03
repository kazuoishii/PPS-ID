"""CIFAR-10 recursive KL distillation, with a paired fixed-teacher control.

Install: python3 -m pip install torch torchvision numpy
Pilot: python3 run_cifar10_recursive_kd.py --seeds 42 --generations 2 --output pilot
Full:  python3 run_cifar10_recursive_kd.py --seeds 42 43 44 45 46 --generations 5 --output full
Logic check (no torch needed): python3 run_cifar10_recursive_kd.py --self-test

All generations including M0 use the same lightweight CNN. M0 learns true
labels; M1..MG learn ONLY teacher soft targets (temperature 4). Each student
starts from fresh random weights. Recursive and fixed branches share M0,
and use identical initialization and minibatch order for each paired slot.
Test predictions are observations for evaluation, NOT training supervision.
Fixed slots are independent distillations from M0, not a recursive chain.
No test-driven checkpoint selection: fixed epoch budget, final checkpoint.
Output directories must be new, preventing accidental result overwrites.
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path


def variation(values):
    loss = sum(max(a - b, 0.0) for a, b in zip(values, values[1:]))
    gain = sum(max(b - a, 0.0) for a, b in zip(values, values[1:]))
    return loss, gain, max(values[0] - values[-1], 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--generations', type=int, default=2)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--teacher-epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temperature', type=float, default=4.0)
    p.add_argument('--data', default='./data')
    p.add_argument('--output', default='./cifar10_kd_pilot')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--train-limit', type=int, default=0, help='Smoke test only; 0 = all 50000')
    p.add_argument('--test-limit', type=int, default=0, help='Smoke test only; 0 = all 10000')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in
                   zip(variation([.8, .7, .75, .65]), (.2, .05, .15)))
        assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in
                   zip(variation([.7, .8]), (0, .1, 0)))
        for q in ([.8, .7, .75, .65], [.7, .8], [.8, .8]):
            g, a, l = variation(q)
            assert math.isclose(q[-1] - q[0], a - g, abs_tol=1e-12)
            assert math.isclose(l, max(g-a, 0), abs_tol=1e-12)
        print('PASS: cumulative loss, recovery, net loss identities')
        return
    if min(args.generations, args.epochs, args.teacher_epochs, args.batch_size) < 1:
        p.error('generations, epochs and batch-size must be positive')
    if args.lr <= 0 or args.temperature <= 0 or min(args.train_limit, args.test_limit, args.workers) < 0:
        p.error('Invalid learning rate, temperature, limit or workers')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error('seeds must be distinct integers in [0, 2**32)')
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    import torchvision
    from torchvision import datasets, transforms

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        p.error('MPS unavailable')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), resolved_device=device, torch_version=torch.__version__,
                  torchvision_version=torchvision.__version__, numpy_version=np.__version__,
                  python_version=sys.version, architecture='CNN32_64_FC256',
                  Q='test_accuracy_fraction', D='mean_JS_natural_log',
                  supervision='M0:CE; students:KL_only', status='running')
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    # Warn rather than silently assuming identical kernels on different devices.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (.4914, .4822, .4465), (.2023, .1994, .2010))])
    train = datasets.CIFAR10(args.data, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(args.data, train=False, download=True, transform=transform)
    # Deterministic subset across all branches/seeds for optional smoke tests.
    if args.train_limit:
        train = Subset(train, np.random.default_rng(2026).permutation(len(train))[:args.train_limit].tolist())
    if args.test_limit:
        test = Subset(test, list(range(min(args.test_limit, len(test)))))
    config.update(train_size=len(train), test_size=len(test))
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def model():
        return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)

    def js(a, b):
        a = np.maximum(a.astype(np.float64), 1e-12)
        b = np.maximum(b.astype(np.float64), 1e-12)
        a /= a.sum(axis=1, keepdims=True)
        b /= b.sum(axis=1, keepdims=True)
        m = (a+b)/2
        return float(np.mean(np.sum((a*np.log(a/m)+b*np.log(b/m))/2, axis=1)))

    def observe(m):
        m.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                probs.append(F.softmax(m(x.to(device)), dim=1).cpu().numpy())
                labels.append(y.numpy())
        probs, labels = np.concatenate(probs), np.concatenate(labels)
        return probs, labels, float(np.mean(probs.argmax(1) == labels))

    def fit(m, teacher, seed, epochs, folder):
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            generator=generator, num_workers=args.workers)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        if teacher is not None:
            teacher.eval()
            teacher.requires_grad_(False)
        with (folder / 'training.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'mean_training_loss'])
            for ep in range(epochs):
                m.train()
                total, count = 0.0, 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits = m(x)
                    if teacher is None:
                        loss = F.cross_entropy(logits, y)
                    else:
                        with torch.no_grad():
                            target = F.softmax(teacher(x)/args.temperature, dim=1)
                        loss = F.kl_div(F.log_softmax(logits/args.temperature, dim=1),
                                       target, reduction='batchmean') * args.temperature**2
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += loss.item()*len(x)
                    count += len(x)
                writer.writerow([ep+1, total/count]); f.flush()
                print(f'{folder.name}: epoch {ep+1}/{epochs}, loss={total/count:.6f}', flush=True)
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, folder / 'model.pt')

    rows = []
    def save_rows():
        with (out/'metrics.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    for seed in args.seeds:
        root = out/f'seed_{seed}'; root.mkdir()
        initial_folder = root/'M0'; initial_folder.mkdir()
        seed_all(seed)
        initial = model()
        fit(initial, None, seed, args.teacher_epochs, initial_folder)
        p0, labels, q0 = observe(initial)
        np.savez_compressed(initial_folder/'observations.npz', probabilities=p0, labels=labels,
                            sample_indices=np.arange(len(labels)))
        rows.append(dict(seed=seed, branch='initial', generation=0, teacher_generation='',
                         Q=q0, D_previous=0., D_initial=0., D_teacher=0., G=0., A=0., L=0.))
        save_rows()
        for branch in ('recursive', 'fixed'):
            teacher, previous_p, qs = initial, p0, [q0]
            for generation in range(1, args.generations+1):
                stage_seed = (seed + 100003*generation) % 2**32
                seed_all(stage_seed)
                student = model()
                folder = root/f'{branch}_M{generation}'; folder.mkdir()
                fit(student, teacher, stage_seed, args.epochs, folder)
                probs, ys, q = observe(student)
                assert np.array_equal(ys, labels)
                np.savez_compressed(folder/'observations.npz', probabilities=probs, labels=ys,
                                    sample_indices=np.arange(len(ys)))
                qs.append(q)
                g, a, l = variation(qs)
                rows.append(dict(seed=seed, branch=branch, generation=generation,
                    teacher_generation=generation-1 if branch=='recursive' else 0,
                    Q=q, D_previous=js(probs, previous_p), D_initial=js(probs, p0),
                    D_teacher=js(probs, previous_p if branch=='recursive' else p0), G=g, A=a, L=l))
                save_rows()
                print(f'{branch} seed={seed} M{generation}: Q={q:.4f} G={g:.4f} L={l:.4f}', flush=True)
                previous_p = probs
                if branch == 'recursive':
                    teacher = student
        del teacher, student, initial
    # Accumulation is done PER seed before aggregating. Fixed G/A are only a
    # slot-ordered noise comparator; fixed slots are not recursive generations.
    fields = ['Q', 'D_previous', 'D_initial', 'D_teacher', 'G', 'A', 'L']
    with (out/'summary.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['branch', 'generation', 'n'] + [f'{k}_{s}' for k in fields for s in ('mean', 'sd')])
        for branch, generation in sorted(set((r['branch'], r['generation']) for r in rows)):
            group = [r for r in rows if r['branch']==branch and r['generation']==generation]
            stats = []
            for k in fields:
                values = [r[k] for r in group]
                stats.extend([float(np.mean(values)), float(np.std(values, ddof=1)) if len(values)>1 else ''])
            w.writerow([branch, generation, len(group)] + stats)
    config['status'] = 'completed'
    (out/'config.json').write_text(json.dumps(config, indent=2))
    print(f'Completed. Results: {out}', flush=True)


if __name__ == '__main__':
    main()
