# Improved ERRNet Experiment

This branch keeps the original ERRNet baseline intact and adds an improved
pipeline for course comparison:

1. realistic online synthesis from VOC crops,
2. cascade refinement network,
3. two-stage training: synthetic warmup, then synthetic + Berkeley real
   finetuning.

## Baseline

The original baseline files are unchanged:

```bash
python train_errnet.py --name errnet_baseline --hyper
python test_errnet.py --model errnet_model --dataset ceilnet_table2 -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
```

## Improved Method

Train the improved model:

```bash
python train_errnet_ours.py --name errnet_ours_realistic_cascade --hyper
```

Optional: initialize the cascade coarse branch from a trained baseline ERRNet:

```bash
python train_errnet_ours.py \
  --name errnet_ours_realistic_cascade \
  --coarse_icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --hyper
```

Fast debug run:

```bash
python train_errnet_ours.py --debug --gpu_ids -1 --batchSize 1 --hyper
```

Evaluate all required benchmark datasets:

```bash
bash scripts/eval_ours_all.sh checkpoints/errnet_ours_realistic_cascade/errnet_cascade_latest.pt results/eval_ours
```

Evaluate with TTA if needed:

```bash
python test_errnet.py --model errnet_cascade_model --dataset ceilnet_table2 \
  --result_dir results/eval_ours_tta --save_subdir CEILNet_table2 \
  -r --icnn_path checkpoints/errnet_ours_realistic_cascade/errnet_cascade_latest.pt \
  --hyper --tta
```

## Output Layout

Each evaluation folder contains:

- per-image visualizations: `m_input.png`, `t_label.png`, model output,
- `metrics.csv`,
- `summary.txt` with average PSNR, SSIM, NCC and LMSE.

Use `scripts/collect_metrics.py` if you want one combined markdown/csv table.
