# Supplementary Materials: Information manuscript information-4637444

This package contains the saved prediction data, recorded protocols, training code, and reproducible analyses supporting the revised PPS-ID manuscript. It uses a shallow CNN (CNN32_64_FC256), not ResNet-18, for CIFAR-10.

## Contents

- `data/cifar10_soft`: 55 original model predictions, five seeds (42-46).
- `data/cifar10_mixed`: 55 mixed-label model predictions, five seeds (42-46).
- `data/sst2` and `data/imdb`: 22 model predictions each, two seeds (42/43).
- `data/cka.csv`: 150 original soft-only CIFAR-10 layer-comparison records.
- `protocols`: original configurations, software versions, selections, metrics, completion records and training logs where supplied.
- `reported_tables`: previously reported derived tables, retained for numeric comparison.
- `code`: training scripts, CKA computation, original trajectory plotting, and a standalone prediction-analysis program.
- `regenerated`: regenerated tables/figures and the validation report from this package.
- `SHA256SUMS.txt`: integrity hashes of all included files other than the hash file itself.

Each dataset has `seed_N/M0` and `seed_N/{recursive,fixed}_M1...M5` folders. Prediction archives store `probabilities`, `labels`, and either `sample_indices` or `indices`. IMDb archives are renamed to `observations.npz` without changing their bytes or key names. CIFAR-10 evaluations contain 10,000 examples, SST-2 872 validation examples, and IMDb a common balanced 1,000-review test subset. No raw images, copyrighted review text, test-selected checkpoints, or new evaluation data are included.

## Reproduce the analyses without training (CPU)

Run from the extracted package root:

```bash
python -m pip install -r requirements_analysis.txt
python code/reproduce_analysis.py --output regenerated
python code/plot_cifar10_trajectory_vertical.py --metrics protocols/cifar10_soft/metrics.csv --output regenerated/figures/cifar10_trajectory.pdf
```

This checks all 154 prediction arrays, labels/indices, original/mixed M0 equality and paired CIFAR-10 M1 equality; computes accuracy, JS, ECE with 10/15/20 bins, entropy, confidence, cumulative downward/upward variation, endpoint loss and individual correctness histories; evaluates margins; computes seed-pair and conditional example-bootstrap intervals; and regenerates the empirical plots. It compares 15 numeric fields with previously reported tables and separately checks the supplied training/evaluation metric records. The current validation report records the outcome and analysis-library versions. Original training environment versions are in `protocols`; they differ from this CPU analysis environment.

### Definitions and uncertainty

- JS uses natural logs, float64, a probability floor of 1e-12, and row renormalization.
- Entropy uses natural logs on the saved temperature-one probabilities.
- ECE is top-label ECE with equal-width confidence bins; the last bin includes confidence one.
- `G` is cumulative decreases of accuracy, `A` cumulative increases, and `L=max(Q0-Qg,0)`; these are fractions in the generated tables. Multiply by 100 for percentage points.
- Recovery counts distinct examples correct at one generation, incorrect at the next, and correct at a later generation. Later correctness need not persist to M5.
- The probability-margin risk set uses `margin <= 2*max_probability_change + 1e-12`. The manuscript's 90 transitions exclude mixed-label runs; the package additionally checks 50 mixed-label transitions, with separate output files.
- Seed-pair Student-t intervals are conditional on the common test set. Example bootstrap intervals resample common examples across fixed trained seed pairs (2,000 repetitions, RNG seed 20261012) and exclude training uncertainty. They are not pooled into one interval.
- Fixed-teacher cumulative variation describes the stated ordering of independently trained students, not a teacher-to-student propagation chain.

## Figure/table mapping

- `regenerated/figures/cifar10_trajectory.pdf`: original CIFAR-10 accuracy and previous/initial JS trajectories.
- `regenerated/figures/cka_trajectories.pdf`: CKA figure, from the included CKA values.
- `regenerated/figures/mixed_comparison.pdf`: soft/mixed accuracy, cumulative decreases, ECE and entropy.
- `regenerated/figures/sst2_validation.pdf`, `imdb_validation.pdf`: individual-seed BERT trajectories.
- `generation_metrics.csv`, `trajectory_summary.csv`, `correctness_transitions.csv`: per-model and per-step numeric results.
- `margin_validation_original90.csv` and `margin_summary_original90.csv`: original margin table; `margin_validation_all.csv` includes the extra mixed-label check.
- `uncertainty.json`: separate seed and example intervals.
- `prediction_audit.csv`, `reported_table_checks.csv`, `source_metric_checks.csv`, `validation_report.json`: audit outputs.
- `cka_summary.csv`: per-layer/reference means and sample SD across five seeds.

Conceptual diagrams and the illustrative algebraic example are generated by the manuscript TeX, not inferred from data. Figure PDF metadata/fonts may vary across plotting-library versions; numerical reproduction is the validation target.

## Recompute CKA values from trained models

The CKA plot can be reproduced entirely from this package. Recomputing CKA itself requires the original 55 `model.pt` checkpoints and CIFAR-10 images, which are not included in the supplied prediction archives. With those checkpoints arranged as `CHECKPOINT_ROOT/seed_N/{M0,recursive_M1,...,fixed_M5}/model.pt`:

```bash
python code/run_cka.py --root CHECKPOINT_ROOT --output cka_recomputed --data cifar10_images --device cuda --samples 1000
```

The subset uses NumPy permutation seed 20261012; features are channel global averages after pool1/pool2 and the FC ReLU. The biased centered linear-Gram CKA uses float64. Spatially pooled features do not retain full feature-map geometry. No mixed-label or BERT CKA is claimed. Hidden features/checkpoints were not independently recomputed in this CPU audit; the supplied 150 values were validated and plotted.

## Retrain the experiments

See `TRAINING_COMMANDS.md`. Training needs external public datasets and, for BERT, the pretrained checkpoint. GPU/MPS/CPU results need not be bitwise identical. Repeated training was not performed as part of this supplementary assembly. Saved prediction-based analyses are reproducible without retraining.

The mixed-label experiment reuses the original M0 teachers and uses alpha=0.5 at T=4. Existing completed students can be resumed; an interrupted student restarts its training rather than resuming an epoch. BERT branches differ already at M1 despite matched nominal settings; retain this limitation when interpreting comparisons. IMDb inputs are truncated at 128 tokens. The supplied short-prefix MPS/CPU repeatability claim is documented in `reported_tables/repeatability_note.txt`; the original 100-update raw logs remain absent. Separate three-update trial records are now supplied and audited in `diagnostics/`; they do not verify the 100-update numerical claims.

This archive does not establish asymptotic convergence, a causal error-propagation mechanism, latent PSD-kernel identification, or universal superiority of a teacher strategy. It is a reproducibility supplement for the reported empirical diagnostics.

## Public repository organization

This directory is published as `revision_2026_10/` in the PPS-ID repository. The repository root retains the original experiment files. All 413 original supplementary files are represented here, together with the additional three-update diagnostic records and audit. Two SST-2 configuration files have their local `/Users/...` output prefixes replaced by `CHECKPOINT_ROOT/`; recorded experimental settings are otherwise retained. `AVAILABILITY_TEXT.txt` has been updated for repository use, and this README and integrity manifest have been updated accordingly. Trained model checkpoints and the original 100-update raw logs were not supplied. Separate three-update diagnostic records are included.

The public third-party datasets and pretrained BERT model must be obtained from their original providers, under their respective terms. No raw image or review text is redistributed.

## Audited short-prefix diagnostics

See [diagnostics/README.md](diagnostics/README.md) for two three-update trials per backend and run `python code/audit_prefix_diagnostics.py` to recompute the recorded comparisons. All comparisons pass. This is distinct from the separately reported 100-update diagnostic, whose raw records remain unavailable.
