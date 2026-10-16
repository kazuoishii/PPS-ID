# PPS-ID

Companion code and results for **Platonic Projection Structures-Based Information Dynamics (PPS-ID): Recursive Transformation of Observable Information** by Kazuo Ishii, Bishnu Prasad Gautam, and Saher Javaid.

PPS-ID assesses observable-state change separately from changes in specified task-dependent preservation properties.

## Complete revision experiments and reproducibility materials

The [revision_2026_10](revision_2026_10/) directory contains the reproducibility supplement for the revised Information manuscript **information-4637444**:

- CIFAR-10 soft-only distillation: five seeds, five student generations, recursive and fixed teachers.
- CIFAR-10 mixed-label distillation: the same five seeds and generations, alpha=0.5.
- BERT-base SST-2 and IMDb distillation: two seeds each, five generations, both teacher strategies.
- All **154 saved model-prediction arrays**, **150 CKA records**, recorded configurations, dataset selections and available training logs.
- Training/analysis code, ECE and entropy analyses, correctness transitions, decision-margin checks, uncertainty intervals, derived tables, figures and regeneration commands.

Start with the [revision README](revision_2026_10/README.md) and [training commands](revision_2026_10/TRAINING_COMMANDS.md). To reproduce the saved-prediction analyses on CPU:

```bash
cd revision_2026_10
python -m pip install -r requirements_analysis.txt
python code/reproduce_analysis.py --output regenerated
python code/plot_cifar10_trajectory_vertical.py --metrics protocols/cifar10_soft/metrics.csv --output regenerated/figures/cifar10_trajectory.pdf
```

All 154 saved accuracies, 15 reported numerical fields, and the prediction/margin checks passed the pre-publication analysis run. This is analysis reproduction from saved outputs, not a new training replication.

## Original experiment files

The root `code/`, `results/`, `figures/`, `requirements.txt` and `README.txt` retain the original CIFAR-10 experiment. All models use the shallow CNN32_64_FC256 architecture, not ResNet-18. The initial teacher learns true labels; soft-only students learn teacher targets at temperature 4.

```bash
python -m pip install -r requirements.txt
python code/run_cifar10_recursive_kd.py --self-test
python -u code/run_cifar10_recursive_kd.py --device cuda --seeds 42 43 44 45 46 --generations 5 --teacher-epochs 10 --epochs 10 --output reproduced_results
```

Use new output directories for training. Hardware and software differences can affect results.

## Results and interpretation

Under **soft-only CIFAR-10 supervision**, final fixed-minus-recursive accuracy averages **1.542 percentage points** across five seeds. Mixing ground-truth and soft-target supervision reverses that endpoint ranking. BERT endpoint rankings are dataset- and seed-dependent; first-generation branch differences limit causal attribution to teacher replacement. Cumulative decreases retain intermediate losses even when endpoint accuracy improves. These findings do not establish universal degradation, universal strategy superiority, or asymptotic convergence.

## Availability limits

Trained model checkpoints were not supplied in the supplement and are **not included**. The CKA plot can be regenerated from the 150 records; recomputing CKA from hidden features requires the original 55 soft-only CIFAR-10 checkpoints and CIFAR-10 images. No public checkpoint download location has been supplied. Raw short-prefix MPS repeatability diagnostic logs are unavailable; that diagnostic is not represented as independently reproduced. Public datasets and pretrained BERT must be obtained from their original providers. No raw images or review text are redistributed.

## Publication and license

Revised manuscript submitted to *Information*, manuscript information-4637444. Preprint: [Zenodo](https://zenodo.org/records/23120094).

The MIT license in `LICENSE` applies to software code and usage documentation. Manuscript, figures and derived experimental results are licensed under CC BY 4.0 as specified in the Zenodo record; third-party datasets and models retain their own terms.

Contact: Kazuo Ishii, kishii@rs.sus.ac.jp.
