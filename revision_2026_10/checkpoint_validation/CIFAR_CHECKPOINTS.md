# CIFAR-10 checkpoint availability and inference validation

All 110 model records (55 soft-only, 55 mixed-label; seeds 42--46, M0 plus recursive/fixed M1--M5) are covered by public model.pt files in the repository's `checkpoints/cifar10/` directory. The source contained 105 checkpoint files and 5 mixed-label M0 references to the corresponding soft-only teacher; all five referenced SHA-256 hashes matched. For direct use, each mixed-label M0 public path resolves to the same teacher bytes. There are 95 distinct serialized checkpoint hashes because matched M1 branches also share identical weights.

All 110 accompanying prediction archives matched the previously published files byte for byte; both root configurations matched exactly, and all parameter shapes matched CNN32_64_FC256. `cifar_checkpoint_manifest.json` maps checkpoint hashes to the published prediction hashes and paths. Checkpoints are separate from the supplementary ZIP to keep it small.

## CPU inference check (no retraining)

Each model record was evaluated on the same ordered 10,000 CIFAR-10 test images with the recorded normalization, temperature-one softmax and batch size 128. Repeated hashes reused the same inference result. PyTorch 2.14.1+cpu, NumPy 2.5.2, Python 3.12.14, four CPU threads were used; the original recorded evaluation used CUDA/PyTorch 2.11.0+cu130.

- 109/110 model records reproduce every predicted class and accuracy.
- One image differs for soft-only seed 43 fixed_M3, test index 8578 (zero-based). Saved classes 2 and 5 have exactly equal probabilities, 0.4985714256763458. Saved argmax chooses class 2 (the true label); CPU probabilities choose class 5. Accuracy changes from 72.97% to 72.96%.
- Across all models, maximum absolute probability difference is 6.139278411865234e-06. All arrays satisfy NumPy allclose with atol=1e-5, rtol=1e-5. This is numerical agreement, not bitwise reproduction.
- Every M5 accuracy is reproduced. Original CUDA prediction arrays and reported numerical results are retained.

The test parquet was obtained from https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet (SHA-256 841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de); labels and indices agree with the saved evaluation arrays. Images are not redistributed.

From the repository root:

```bash
pip install torch numpy pyarrow pillow
curl -L https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet -o test.parquet
python revision_2026_10/code/audit_cifar_checkpoint_inference.py --checkpoint-root checkpoints/cifar10 --prediction-root revision_2026_10/data --dataset-parquet test.parquet --output checkpoint_inference_audit
```

Optional `--only cifar10_soft:43:fixed_M3` repeats the tie inspection. Results are `inference_comparison.csv`, `inference_summary.json` and `changed_examples.json`. This validation does not retrain any model, repeat BERT inference, or independently recompute CKA.
