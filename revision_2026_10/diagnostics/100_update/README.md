# Recovered 100-update CPU/MPS diagnostic records

Two trials per backend, steps 0--99, recovered from the original CPU/MPS folders supplied by the author. Local source prefixes in comparison.json are normalized to CHECKPOINT_ROOT; numerical values and recorded hash strings are retained.

Run `python code/audit_100_update_diagnostics.py` from the revision directory. The audit recomputes every recorded numerical comparison and checks recorded batch, RNG and weight-hash equality. It does not rerun training or reconstruct weight hashes from model checkpoints.

CPU comparisons are exactly zero. MPS maxima: student logits 1.7136335372924805e-05, loss 1.4781951904296875e-05, classification-head gradient 4.169344902038574e-05; teacher probabilities match exactly. MPS post-update weight hashes differ at all 100 updates. These match the rounded values in the manuscript. These within-backend prefix diagnostics do not quantify full-training accuracy effects or establish causes of IMDb divergence. The exact executed diagnostic script version is not established by these JSON records.

The earlier three-update records remain in ../cpu and ../mps and have a separate audit.
