# Sharp Error-Correction Certificates in Recursive Information Dynamics

## Submission snapshot — 25 October 2026

Author: Kazuo Ishii, Suwa University of Science.
Manuscript: *Sharp Error-Correction Certificates in Recursive Information Dynamics: A Saturating Switching-Network Model*.
Submitted to Mathematics on 25 October 2026, manuscript ID mathematics-4685224. This is a submission snapshot; journal acceptance and peer review completion are not implied.

The files below reproduce the complete supplementary archive submitted with the manuscript. The numerical files and scripts retain their submitted bytes. This README adds publication context to the submitted reproduction instructions.

This directory is separate from the CIFAR-10/BERT PPS-ID experiments elsewhere in the repository. These are synthetic calculations, not empirical LLM experiments.

# Complete numerical supplement (2026-10-25)

Synthetic model data only. This supplement is the complete current reproduction package. The cited GitHub commit is a historical snapshot and does not include the current additions or interface changes.

## Environment and recommended verification

Use Python 3, NumPy and Matplotlib. ENVIRONMENT.json records the tested environment; requirements.txt pins the tested package versions.

    python -m pip install -r requirements.txt
    python verify_submission.py

The verifier runs all numerical scripts in a temporary copy and compares all archived CSV/JSON results (relative tolerance 1e-10, absolute tolerance 1e-12), without changing the archived files. PDF bytes are not compared because metadata and rendering can vary.

## Default: use the saved inputs

    python reproduction/reproduce.py

Reads reproduction/inputs.npz as authoritative inputs and writes results and Figures 1-3 to reproduction/generated/. A missing input archive raises an error; there is no silent regeneration. To select another destination:

    python reproduction/reproduce.py --output-dir /path/to/new_results

## Optional seeded regeneration

    python reproduction/reproduce.py --regenerate-inputs --output-dir /path/to/regenerated_results

Uses numpy.random.default_rng(20261006): draw h using integers(0,2,1000), then the episode-major 1000-by-6 normal array X, then the 20-by-6 exposure mask with uniform draws <0.65. Source 1 is made available to everyone. Regenerated inputs are saved ONLY in the chosen output directory. Default reproduction never overwrites the archived inputs. Library versions may affect regeneration; archived inputs take precedence.

## File map

- Section 8.1 / Table 2: verify_certificate.py, certificate_results.json, switching.csv, boundary.csv. switching.csv columns: t, network_min, envelope, brier, brier_bound (times 0-300). boundary.csv columns: graph, t, signed_min, deviation_from_negative_fold. Crossings: 5 and 10; certificate: 60.
- Sections 8.2-8.5 / Figures 1-3 / Table 3: reproduction/reproduce.py, inputs.npz, results.json, table.csv, paired.csv, recovery.csv, grid.csv, and all four *_episodes.csv. inputs.npz contains h, X, mask, Complete, Cycle and Modular. recovery.csv stores the two state trajectories; grid.csv stores lambda,e,z,residual,iterations (probability=sigmoid(z)); table.csv stores graph/use, means, episode-level SEs and residuals; paired.csv stores reuse-minus-unique differences and paired SEs. Episode CSVs contain h and unique/reuse accuracy and Brier loss. Cycle_reverse is an additional orientation check, not another row in manuscript Table 3.
- Section 8.6 / Figure 4: verify_heterogeneous.py, heterogeneous.csv, heterogeneous_results.json, heterogeneous_bound.pdf. CSV: t, minimum_signed_state, lower_bound, z_1,...,z_20 (times 0-100). Both crossings: t=2; L1=-1.3; L2=0.04.
- Additional zero-threshold checks: verify_zero_threshold.py, zero_threshold_results.json. Checks positive-input envelopes, zero-input negative sign persistence before numerical underflow, and negative fixed points.
- Root duplication.pdf, recovery.pdf, reference_grid.pdf, heterogeneous_bound.pdf are the four manuscript figures.

Direct execution of the three root experiment scripts writes outputs next to each script. Use verify_submission.py to run safely in a temporary copy.

## Limits and integrity

Numerical tests illustrate the model; proofs establish all-time and classwise results. No LLM, human-participant, calibration or fairness experiment is included. MANIFEST.sha256 covers supplementary files, excluding itself. Editorial history and operating-system metadata are omitted.

## Directed-cycle convention
The saved Cycle matrix has W[i,(i-1) mod 20]=1. For example, zero-based row 0 receives from column 19. It is constructed by np.roll(np.eye(20), -1, axis=1). The Cycle_reverse check uses the opposite direction. Table 3 uses the saved Cycle matrix (accuracy 0.86395 / 0.78800).
