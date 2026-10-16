"""Plot saved CIFAR-10 metrics; training is not required."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='/content/drive/MyDrive/PPS_ID/kd_full/metrics.csv')
    parser.add_argument('--output', default='/content/drive/MyDrive/PPS_ID/kd_full/cifar10_trajectory.pdf')
    args = parser.parse_args()
    df = pd.read_csv(args.metrics)
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 15,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 12, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.5), constrained_layout=True)
    panels = [('Q', '(a) Task preservation', 'Test accuracy (%)', 100),
              ('D_previous', '(b) Change from preceding model', 'Mean JS divergence (nats)', 1),
              ('D_initial', '(c) Change from initial teacher', 'Mean JS divergence (nats)', 1)]
    seeds = sorted(df.seed.unique())
    for ax, (metric, title, ylabel, scale) in zip(axes, panels):
        for branch, color, label in [('recursive', '#1764a4', 'Recursive'),
                                      ('fixed', '#db7423', 'Fixed teacher')]:
            values = []
            for seed in seeds:
                rows = df[(df.seed == seed) & (df.branch == branch)].sort_values('generation')
                if metric == 'Q':
                    initial = df[(df.seed == seed) & (df.branch == 'initial')]
                    rows = pd.concat([initial, rows]).sort_values('generation')
                x = rows.generation.to_numpy()
                y = rows[metric].to_numpy() * scale
                ax.plot(x, y, color=color, alpha=.22, linewidth=1)
                values.append(y)
            array = np.asarray(values)
            ax.errorbar(x, array.mean(axis=0), yerr=array.std(axis=0, ddof=1),
                        color=color, marker='o', linewidth=2, markersize=5,
                        capsize=4, label=label)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(alpha=.2)
        ax.legend(loc='best', frameon=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

if __name__ == '__main__':
    main()
