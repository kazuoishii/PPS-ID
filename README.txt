PPS-ID: CIFAR-10 supplementary code and results

Contents
- code/run_cifar10_recursive_kd.py: training and evaluation script.
- code/plot_cifar10_trajectory_vertical.py: trajectory plotting script.
- results/metrics.csv: 55 rows, five seeds (42-46), M0 and five generations
  for each recursive and fixed-teacher condition.
- results/summary.csv: generation-level summary statistics.
- results/config.json: saved configuration of the reported experiment.
- figures/cifar10_trajectory.pdf: vertical three-panel manuscript figure.

Install dependencies
python -m pip install torch torchvision numpy pandas matplotlib

Check metric identities
python code/run_cifar10_recursive_kd.py --self-test

Run the experiment (a new output directory is required)
python -u code/run_cifar10_recursive_kd.py --device cuda --seeds 42 43 44 45 46 --generations 5 --teacher-epochs 10 --epochs 10 --batch-size 128 --lr 0.001 --temperature 4 --output reproduced_results

Plot the supplied results
python code/plot_cifar10_trajectory_vertical.py --metrics results/metrics.csv --output figures/reproduced_trajectory.pdf

The supplied CSVs and configuration are the saved experiment results.
Model checkpoints and per-example prediction arrays are not included in this
archive. Training generates these outputs. Exact numerical reproduction can
vary with hardware and software versions; the original Python, PyTorch, torchvision and NumPy versions are
recorded in results/config.json.
