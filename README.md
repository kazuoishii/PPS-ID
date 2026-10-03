# PPS-ID

Companion code and results for **Platonic Projection Structures-Based Information Dynamics (PPS-ID): Recursive Transformation of Observable Information** by Kazuo Ishii, Bishnu Prasad Gautam, and Javaid Saher.

PPS-ID evaluates observable-state change separately from changes in a specified task-dependent preservation functional. This repository provides the CIFAR-10 experiment: five generations of recursive KL knowledge distillation and a matched fixed-teacher condition across five paired seeds (42–46).

## Contents

- `code/run_cifar10_recursive_kd.py`: training, evaluation, and metric identity checks.
- `code/plot_cifar10_trajectory_vertical.py`: three-panel trajectory figure.
- `results/metrics.csv`: all 55 result rows.
- `results/summary.csv`: summary statistics.
- `results/config.json`: recorded configuration and runtime versions.
- `figures/cifar10_trajectory.pdf`: manuscript trajectory figure.
- `README.txt`: detailed execution notes.

All models use the same CNN32_64_FC256 architecture. The initial teacher learns true labels; fresh students learn teacher soft targets only (temperature 4). Recursive and fixed-teacher runs use matched student initialization and minibatch order.

## Installation and use

```bash
python -m pip install -r requirements.txt
python code/run_cifar10_recursive_kd.py --self-test
python -u code/run_cifar10_recursive_kd.py --device cuda --seeds 42 43 44 45 46 --generations 5 --teacher-epochs 10 --epochs 10 --output reproduced_results
python code/plot_cifar10_trajectory_vertical.py --metrics results/metrics.csv --output figures/reproduced_trajectory.pdf
```

Training requires a new output directory. The original runtime versions are recorded in `results/config.json`; exact numerical reproduction can vary across hardware and software.

## Results and scope

At generation five, the mean paired fixed-minus-recursive test accuracy difference is **1.542 percentage points** across five seeds. In every recursive run, successive-generation Jensen–Shannon discrepancies decrease while discrepancies from the initial teacher increase. Cumulative decreases record intermediate accuracy losses that endpoint performance can obscure.

These observations are specific to the tested dataset, architecture, training protocol, and finite generation horizon. They do not establish universal degradation or asymptotic convergence.

Saved model checkpoints and per-example predictions are not included. CIFAR-10 images are not redistributed; the training script downloads the dataset through torchvision.

## Publication status

The manuscript is prepared for submission to *Information*. A preprint DOI and arXiv identifier will be added when available.

## License

The MIT license in `LICENSE` applies to the software code and its usage documentation. No separate reuse license is assigned here to the manuscript figure or experimental result files; please contact the corresponding author concerning their reuse.

## Contact

Kazuo Ishii: kishii@rs.sus.ac.jp
