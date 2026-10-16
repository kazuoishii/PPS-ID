# Three-update SST-2 repeatability records

These records were supplied in m1_diagnostic_results(1).zip (MPS) and m1_diagnostic_cpu_results(1).zip (CPU). Each backend contains two trials of exactly three updates, indexed 0, 1 and 2. They are not the raw records of the separately reported 100-update diagnostic.

Each trial records initial/teacher weight hashes, RNG hashes, batch indices/input hashes, logits, teacher probabilities, losses, classification-head gradients and post-update weight hashes. Comparison source paths were normalized to CHECKPOINT_ROOT; numerical records were retained. These files contain hashes and numerical traces, not trained checkpoints or raw text inputs.

Run from the revision directory:

```bash
python code/audit_prefix_diagnostics.py
```

The audit recomputes all numerical comparisons and checks equality of the recorded hash strings. It does not independently recompute model hashes from checkpoints or rerun training. Both comparisons agree exactly with the supplied comparison.json files.

- Within each backend, initial student/teacher hashes, initial RNG states, and stepwise batches/RNG states match between trials.
- CPU trials match on all recorded quantities at all three updates.
- MPS post-update weight hash strings differ from the first update. Maximum recorded logit difference is 2.682209014892578e-7; maximum loss and classification-head gradient differences are both 9.5367431640625e-7. Teacher probability differences are zero.
- CPU-versus-MPS equality is not asserted. These within-backend short traces do not establish full-training determinism, explain IMDb branch divergence, or quantify final-accuracy effects.
- Original 100-update raw records and trained model checkpoints remain unavailable in the supplied materials.
