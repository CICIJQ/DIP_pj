# Reflection-Aware RDNet Refinement

RA-RDNet is an add-on stage after XReflection-RDNet.  It does not modify the
existing ERRNet, NAFNet Refiner, or XReflection-RDNet code paths.

Pipeline:

1. Input reflection image `I`.
2. Existing XReflection-RDNet output `T_rd`.
3. Estimate a soft reflection confidence map `M` from `I` and `T_rd`.
4. Apply stronger correction in high-`M` regions.
5. Preserve the original RDNet output in low-`M` regions.

## Version A: RCA

Reflection Correction Amplification is training-free.  It extrapolates a little
farther along RDNet's own correction direction:

```text
T_rca = T_rd + strength * M * (T_rd - I)
```

This is intentionally more aggressive than RDNet.  It may remove visible
reflection remnants better, but it can reduce PSNR/SSIM when the mask is too
broad or the strength is too high.

Run one dataset:

```bash
python scripts/run_ra_rdnet_rca.py \
  --input_dir datasets/processed_data/real20 \
  --rdnet_dir results/xreflection_rdnet/real20 \
  --output_dir results/ra_rdnet_rca/real20 \
  --strength 0.75 \
  --mask_gamma 0.65
```

Evaluate against GT:

```bash
python scripts/eval_reflection_methods.py \
  --pred_dir results/ra_rdnet_rca/real20 \
  --gt_dir datasets/processed_data/real20 \
  --pred_filename ra_rdnet_rca.png \
  --output_csv results/ra_rdnet_rca/real20/per_image_metrics.csv \
  --output_json results/ra_rdnet_rca/real20/average_metrics.json \
  --summary_txt results/ra_rdnet_rca/real20/summary.txt
```

Run all course datasets from existing RDNet outputs:

```bash
PYTHON=/SSDHome/home/xumx/anaconda3/envs/errnet/bin/python \
DEVICE=cpu \
STRENGTH=0.75 \
MASK_GAMMA=0.65 \
bash scripts/run_all_ra_rdnet_rca_eval.sh
```

Useful RCA controls:

- `--strength`: larger means more aggressive correction. Try `0.5`, `0.75`,
  `1.0`, `1.25`.
- `--mask_gamma`: lower than `1` expands high-confidence areas. Try `0.6` to
  attack stronger reflections.
- `--mask_sensitivity`: larger increases the map magnitude.
- `--mask_floor`: values such as `0.05` make the correction less local.
- `--max_extra_delta`: caps the extra RGB correction.

Outputs are written as:

```text
results/ra_rdnet_rca/<dataset>/<stem>/ra_rdnet_rca.png
results/ra_rdnet_rca/<dataset>/<stem>/reflection_confidence.png
```

## Version B: RDNet + NAFNet Refiner

The current trainable refiner in this repo is an ungated residual refiner built
on a lightweight NAFNet-style backbone.  It takes `[I, T_rd, I - T_rd]`,
predicts a bounded residual, and adds it back directly:

```text
T_ref = clip(T_rd + Delta_theta(I, T_rd, I - T_rd), 0, 1)
```

The last convolution is zero-initialized, so a fresh checkpoint starts close to
an identity mapping over RDNet.

Training still uses pseudo reflection masks in the loss to focus the residual
updates:

- global L1 to GT;
- reflection-weighted L1 using a pseudo mask from `|I - GT|`;
- background preservation loss to stay close to RDNet in low-reflection areas;
- gradient loss for sharper local correction.

### Prepare RDNet Outputs For Training

The trainer expects paired inputs, GT, and precomputed RDNet outputs.  For
fair benchmark reporting, do not train on the course test sets.  In the current
repo, the clean paired training split is `real_train`, plus any synthetic or
other train-only paired datasets that you create yourself.

The current fixed synthetic export used for fair RDNet-refiner training is:

- `datasets/processed_data/rdnet_refiner_voc_synth_train`
- RDNet outputs under `results/xreflection_rdnet/rdnet_refiner_voc_synth_train`

If you need to refresh one dataset manually, run:

```bash
python scripts/run_xreflection_rdnet.py \
  --xreflection_root external_models/xreflection/XReflection \
  --input_dir datasets/processed_data/real_train \
  --output_dir results/xreflection_rdnet_train/real_train \
  --config external_models/xreflection/XReflection/options/train_rdnet.yml \
  --checkpoint checkpoints/xreflection/rdnet-26.4849.ckpt \
  --device cuda:0 \
  --python /SSDHome/home/xumx/anaconda3/envs/errnet/bin/python \
  --dataset_name real_train
```

The trainer now defaults to `--train_dataset_names fair_train`, which excludes
the benchmark/eval datasets:

- `real20`
- `testdata_CEILNET_table2`
- `sir2_withgt`
- `objects`
- `postcard`
- `wild`
- `reflection_pairs_tight_eval/testdata_CEILNET_table2`

If you intentionally want to run a non-fair ablation that mixes these sets into
training, you must pass `--allow_benchmark_train_mix` explicitly.

With the current repo contents, `fair_train` resolves to:

- `rdnet_refiner_voc_synth_train`
- `real_train`

### Train

```bash
python scripts/train_mg_rdnet_refiner.py \
  --train_data_root datasets/processed_data \
  --train_rdnet_root results/xreflection_rdnet \
  --train_dataset_names fair_train \
  --checkpoint_dir checkpoints/rdnet_naf_refiner_fair \
  --device cuda:0 \
  --epochs 20 \
  --batch_size 4 \
  --patch_size 256 \
  --lr 1e-4 \
  --sample_mode balanced_dataset
```

The script saves:

```text
checkpoints/rdnet_naf_refiner_fair/latest.pth
checkpoints/rdnet_naf_refiner_fair/best_psnr.pth
```

### Inference

```bash
python scripts/infer_mg_rdnet_refiner.py \
  --input_dir datasets/processed_data/real20 \
  --rdnet_dir results/xreflection_rdnet/real20 \
  --checkpoint checkpoints/rdnet_naf_refiner_fair/best_psnr.pth \
  --output_dir results/rdnet_naf_refiner/real20 \
  --device cuda:0
```

Evaluate:

```bash
python scripts/eval_reflection_methods.py \
  --pred_dir results/rdnet_naf_refiner/real20 \
  --gt_dir datasets/processed_data/real20 \
  --pred_filename mg_rdnet_refiner.png \
  --output_csv results/rdnet_naf_refiner/real20/per_image_metrics.csv \
  --output_json results/rdnet_naf_refiner/real20/average_metrics.json \
  --summary_txt results/rdnet_naf_refiner/real20/summary.txt
```

Run all course datasets after training a checkpoint:

```bash
PYTHON=/SSDHome/home/xumx/anaconda3/envs/errnet/bin/python \
DEVICE=cuda:0 \
MG_CHECKPOINT=checkpoints/rdnet_naf_refiner_fair/best_psnr.pth \
bash scripts/run_all_mg_rdnet_eval.sh
```

## Recommended Experiment Order

1. Run RCA on all test sets and inspect `reflection_confidence.png`.
2. Grid-search RCA strength on validation-only images, not on final test images.
3. Precompute RDNet outputs for training-only data.
4. Train RDNet + NAFNet Refiner with a small validation split from training-only data.
5. Compare RDNet, RCA, and the refiner on the fixed course test sets.

RCA is best for a quick visual, aggressive-reflection-removal ablation.
RDNet + NAFNet Refiner is the better route if you need a trainable method with
a clearer technical contribution.
