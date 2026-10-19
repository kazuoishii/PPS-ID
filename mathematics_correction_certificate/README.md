# Sharp Error-Correction Certificates in Recursive Information Dynamics

## Saturating switching-network model — synthetic numerical results

Author: Kazuo Ishii. Companion numerical materials for *Sharp Error-Correction Certificates in Recursive Information Dynamics: A Saturating Switching-Network Model*.

These are synthetic calculations of a specified mathematical model, not empirical LLM experiments. This directory is separate from the CIFAR-10/BERT PPS-ID experiments elsewhere in this repository. No manuscript publication or journal acceptance is implied.

## Reproduce

From this directory:

```bash
python -m pip install -r requirements.txt
gzip -dk reproduction/grid.csv.gz  # unpack saved grid for inspection
python verify_certificate.py
python reproduction/reproduce.py
```

Scripts overwrite the numerical outputs in their own directories. The reproduction script is labeled in the supplied source as an independent reconstruction from the manuscript, rather than the original experimental implementation. Generated figure PDFs are written inside reproduction/; the figure PDFs at this directory root are the supplied manuscript figures.

## Contents

- `verify_certificate.py`: switching-network state/loss inequalities and boundary counterexample.
- `certificate_results.json`, `switching.csv`, `boundary.csv`: saved certificate checks.
- `reproduction/reproduce.py`: scalar reversal, deduplication/recovery, reference grid, and paired stochastic episodes.
- `reproduction/inputs.npz`: saved synthetic inputs.
- `reproduction/*_episodes.csv`, `table.csv`, `paired.csv`, `results.json`: episode records and summaries.
- `reproduction/grid.csv.gz` (losslessly compressed), `recovery.csv`: reference sweep and recovery trajectories.
- Root PDF figures: supplied manuscript figures.
- `SHA256SUMS`: integrity hashes of published files (excluding this manifest).

The certificate calculation uses a=0.2, gamma=1.2, kappa=1, D=0.3, and 20 agents. The fold threshold is approximately 0.1660371643. Observed network correction occurs at update 5; the scalar envelope crosses at update 10; the conservative certified update is 60. Boundary checks are numerical illustrations, not substitutes for proof.

The paired experiment uses 1,000 synthetic episodes and seed 20261006. Lower accuracy and lower Brier loss under reuse are parameter-dependent findings, not universal claims about source reuse. No calibration or fairness validation is claimed.
