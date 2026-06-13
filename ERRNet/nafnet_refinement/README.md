# ERRNet + Lightweight NAFNet Refiner

This method does not replace the original ERRNet baseline. ERRNet remains the
frozen coarse restoration stage, and a lightweight NAFNet restoration backbone
learns a second-stage refinement.

## Architecture

For a reflection image `I`, the frozen ERRNet produces a coarse transmission
`T0`. The refiner receives:

```text
concat(I, T0, I - T0)
```

The default small model uses:

```text
width=32
middle_blk_num=4
enc_blk_nums=[1,1,2]
dec_blk_nums=[1,1,1]
```

The NAFNet branch predicts a correction that is added to `T0`. Its last layer
is initialized to zero, so training starts from the coarse ERRNet result.

Only the NAFNet branch is optimized. ERRNet and its VGG hypercolumn network are
kept frozen.

## Loss

The training objective contains all four requested terms:

```text
L = lambda_l1 * L1
  + lambda_ssim * SSIMLoss
  + lambda_gradient * GradientLoss
  + lambda_vgg * VGGPerceptualLoss
```

Defaults are `1.0`, `0.2`, `0.1`, and `0.05`.

## Training

Improved Loss ERRNet is the default coarse model:

```bash
python train_errnet_naf_refiner.py \
  --name errnet_naf_refiner_improved \
  --naf_coarse_kind improved \
  --naf_coarse_checkpoint checkpoints/errnet_improved_loss_v1/errnet_060_00463920.pt \
  --gpu_ids 0 \
  --batchSize 1 \
  --nThreads 4 \
  --nEpochs 60 \
  --lr 1e-4 \
  --hyper
```

Use the original baseline as coarse stage:

```bash
python train_errnet_naf_refiner.py \
  --name errnet_naf_refiner_baseline \
  --naf_coarse_kind baseline \
  --naf_coarse_checkpoint checkpoints/errnet/errnet_060_00463920.pt \
  --gpu_ids 0 \
  --batchSize 1 \
  --hyper
```

Small smoke run:

```bash
python train_errnet_naf_refiner.py \
  --name errnet_naf_refiner_smoke \
  --gpu_ids 0 \
  --batchSize 1 \
  --nThreads 0 \
  --nEpochs 1 \
  --max_dataset_size 2 \
  --naf_real_ratio 0 \
  --naf_eval_size 1 \
  --naf_eval_freq 1 \
  --hyper \
  --no-log
```

Checkpoints are written under `checkpoints/<name>/`:

```text
naf_refiner_latest.pt
naf_refiner_best.pt
naf_refiner_<epoch>_<iteration>.pt
```

## Testing

Evaluate all six benchmarks:

```bash
python test_errnet_naf_refiner.py \
  --dataset all \
  --naf_checkpoint checkpoints/errnet_naf_refiner_improved/naf_refiner_best.pt \
  --result_dir results/eval_naf_refiner \
  --gpu_ids 0 \
  --nThreads 0 \
  --no-verbose
```

Evaluate one benchmark:

```bash
python test_errnet_naf_refiner.py \
  --dataset real20 \
  --naf_checkpoint checkpoints/errnet_naf_refiner_improved/naf_refiner_best.pt \
  --gpu_ids 0 \
  --nThreads 0 \
  --no-verbose
```

Each dataset directory contains:

- `per_image_metrics.csv`
- `summary.txt`
- per-image `input.png`, `baseline.png`, `improved.png`,
  `naf_refiner.png`, `gt.png`, and `comparison.png`

PSNR, SSIM, NCC, and LMSE are computed with the existing
`util.index.quality_assess`, matching the original ERRNet evaluator.
