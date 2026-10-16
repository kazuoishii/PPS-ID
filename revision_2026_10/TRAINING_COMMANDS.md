# Training and checkpoint-based analysis commands

Run from the package root. Use new output directories for independent training runs. Commands reflect the reported hyperparameters; use the recorded configurations/software/model revisions for detailed reproduction. Training dependencies are `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, and for BERT `transformers`, `datasets`, `scikit-learn`. The original version records are retained in `protocols`; this supplement does not assume one common training environment across datasets.

## CIFAR-10, soft-only

```bash
python -u code/run_cifar10_recursive_kd.py --device cuda --seeds 42 43 44 45 46 --generations 5 --teacher-epochs 10 --epochs 10 --batch-size 128 --lr 0.001 --temperature 4 --data cifar10_images --output reproduced_soft
```

## CIFAR-10, mixed labels

`reproduced_soft` must contain the M0 weights. To compare with the reported experiment, use its original M0 checkpoint root instead of newly trained teachers.

```bash
python -u code/run_mixed_kd.py --teacher-root reproduced_soft --output reproduced_mixed --data cifar10_images --device cuda --seeds 42 43 44 45 46 --generations 5 --epochs 10 --batch-size 128 --lr 0.001 --temperature 4 --alpha 0.5 --max-new-students 0
```

## SST-2

```bash
python -u code/run_bert_recursive_kd.py --output reproduced_sst2 --device mps --seeds 42 43 --generations 5 --teacher-epochs 3 --epochs 3 --train-size 5000 --eval-size 0 --batch-size 4 --max-length 128 --lr 0.00002 --temperature 4 --model google-bert/bert-base-uncased --revision 86b5e0934494bd15c9632b12f734a8a67f723594
```

The supplied `protocols/sst2` configurations record the resolved checkpoint revision. Dataset selection and ordering are specified in the script/configurations. MPS nominal seeds did not guarantee equal first-generation outputs.

## IMDb

```bash
python -u code/run_imdb_recursive_kd.py --output reproduced_imdb --device mps --seeds 42 43 --generations 5 --teacher-epochs 3 --epochs 3 --train-size 5000 --eval-size 1000 --batch-size 8 --max-length 128 --lr 0.00002 --temperature 4 --model google-bert/bert-base-uncased
```

Use the resolved checkpoint revision in `protocols/imdb/imdb_kd_5gen/model_revision.txt` via `--revision` for pinned model reproduction. The original balanced dataset selections are retained in `protocols/imdb/selection.npz`; the script uses the recorded selection seeds. Saved dataset indices are the authority for the reported subset.

## Short-prefix BERT diagnostic

`code/diagnose_bert_m1.py` is included as supplied. Its recorded conclusions were reported in a separate draft; no raw diagnostic log is included here, and this supplement does not mark the diagnostic as independently reproduced. Follow its argument parser for any new diagnostic run and save complete logs.
