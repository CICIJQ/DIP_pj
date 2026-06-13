# ERRNet Reflection Prior Branch

This branch is additive-only: it adds new entry points and a new model module
without modifying the existing ERRNet files.

## Model

`models/prior_branch.py` wraps ERRNet with:

- a fixed Laplacian/Sobel high-frequency extractor,
- a reflection location/intensity head supervised by a pseudo target from
  `abs(input - target)`,
- a mask-gated residual refiner initialized to zero, so a baseline checkpoint is
  preserved at the start of finetuning.

The generated mask is used in two ways:

- it gates the residual correction: `output = coarse + mask * delta`,
- it adds a background preservation loss outside the pseudo reflection area.

## Train

```bash
python train_errnet_prior.py \
  --name errnet_prior_errnet \
  --prior_init_icnn checkpoints/errnet/errnet_060_00463920.pt \
  --hyper --lambda_gan 0 --lambda_coarse 0.1
```

Useful knobs:

```bash
--prior_lambda_mask 0.1
--prior_lambda_gate 0.05
--prior_lambda_smooth 0.01
--prior_target_low 0.05
--prior_target_high 0.5
```

## Tune

```bash
python scripts/tune_prior_hparams.py \
  --gpu_ids 0 \
  --epochs 3 \
  --max_dataset_size 512 \
  --init_icnn checkpoints/errnet/errnet_060_00463920.pt
```

The tuner writes `results/prior_tune_summary.csv`.

## Evaluate

```bash
bash scripts/eval_prior_all.sh \
  checkpoints/errnet_prior_errnet/errnet_prior_latest.pt \
  results/eval_prior
```

For one dataset:

```bash
python test_errnet_prior.py --dataset real20 \
  --result_dir results/eval_prior --save_subdir real20 \
  -r --icnn_path checkpoints/errnet_prior_errnet/errnet_prior_latest.pt \
  --hyper
```
