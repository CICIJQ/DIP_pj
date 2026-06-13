# XReflection / RDNet Evaluation

This adds XReflection RDNet as an external comparison method.  It does not
modify the existing ERRNet model code; all integration lives in `scripts/` and
`external_models/xreflection/`.

## Install XReflection

Clone and install XReflection in its own environment or inside the reserved
external directory:

```bash
git clone https://github.com/hainuo-wang/XReflection.git external_models/xreflection/XReflection
cd external_models/xreflection/XReflection
pip install -r requirements.txt
python setup.py develop
```

The wrapper uses the current XReflection repository entry point:

```bash
python xreflection/tools/train.py --config options/train_rdnet.yml --test_only pretrained.ckpt
```

Some upstream docs may show `tools/train.py` instead.  If your checkout uses a
different path, pass the exact command through `--command_template`.

## RDNet Checkpoint And Config

Download the RDNet checkpoint from the XReflection model zoo and place it here:

```text
checkpoints/xreflection/rdnet-26.4849.ckpt
```

Known model-zoo URL:

```text
https://checkpoints.mingjia.li/rdnet-26.4849.ckpt
```

Default config path:

```text
external_models/xreflection/XReflection/options/train_rdnet.yml
```

The wrapper generates a temporary one-dataset config under the output
directory, so the original XReflection config is not edited.

## Run One Dataset

Example on `real20`:

```bash
python scripts/run_xreflection_rdnet.py \
  --xreflection_root external_models/xreflection/XReflection \
  --input_dir datasets/processed_data/real20 \
  --output_dir results/xreflection_rdnet/real20 \
  --config external_models/xreflection/XReflection/options/train_rdnet.yml \
  --checkpoint checkpoints/xreflection/rdnet-26.4849.ckpt \
  --device cuda:0
```

Evaluate the output against GT:

```bash
python scripts/eval_reflection_methods.py \
  --pred_dir results/xreflection_rdnet/real20 \
  --gt_dir datasets/processed_data/real20 \
  --pred_filename xreflection_rdnet.png \
  --output_csv results/xreflection_rdnet/real20/per_image_metrics.csv \
  --output_json results/xreflection_rdnet/real20/average_metrics.json \
  --summary_txt results/xreflection_rdnet/real20/summary.txt
```

`--gt_dir` may point either to a flat GT image directory or to a dataset root
containing `transmission_layer/`.  If prediction and GT sizes differ, the
prediction is resized to the GT size before PSNR, SSIM, NCC, and LMSE are
computed.

## Run All Course Datasets

Set paths once and run the batch script:

```bash
XREFLECTION_ROOT=external_models/xreflection/XReflection \
RDNET_CONFIG=external_models/xreflection/XReflection/options/train_rdnet.yml \
RDNET_CHECKPOINT=checkpoints/xreflection/rdnet-26.4849.ckpt \
DEVICE=cuda:0 \
bash scripts/run_all_xreflection_eval.sh
```

The script runs:

- `CEILNet_table2`
- `real20`
- `sir2_withgt`
- `SIR2_objects`
- `SIR2_postcard`
- `SIR2_wild`
- `self_collected`

Datasets with GT write `per_image_metrics.csv`, `average_metrics.json`, and
`summary.txt`.  If `SELF_GT_DIR` is empty, self-collected images are inferred
only and no PSNR/SSIM/NCC/LMSE is computed.

## Custom XReflection Command

If upstream XReflection changes its inference interface, keep this repo stable
and pass a command template:

```bash
python scripts/run_xreflection_rdnet.py \
  --xreflection_root external_models/xreflection/XReflection \
  --input_dir datasets/processed_data/real20 \
  --output_dir results/xreflection_rdnet/real20 \
  --config external_models/xreflection/XReflection/options/train_rdnet.yml \
  --checkpoint checkpoints/xreflection/rdnet-26.4849.ckpt \
  --device cuda:0 \
  --command_template "{python} {xreflection_train} --config {generated_config} --test_only {checkpoint}"
```

Supported placeholders include `{xreflection_root}`, `{input_dir}`,
`{prepared_input_dir}`, `{output_dir}`, `{config}`, `{generated_config}`,
`{checkpoint}`, `{device}`, `{run_root}`, `{run_name}`, and `{dataset_name}`.

## Make Paper Comparison Grids

Example:

```bash
python scripts/make_comparison_grid.py \
  --input_dir datasets/processed_data/real20 \
  --errnet_dir results/eval_baseline/real20 \
  --nafnet_dir results/errnet_naf_refiner_improved_best/real20 \
  --xreflection_dir results/xreflection_rdnet/real20 \
  --gt_dir datasets/processed_data/real20 \
  --image_stems 107 89 58 \
  --output paper_visuals/panels/rdnet_real20_grid.png
```

Column order is fixed:

```text
Input | ERRNet | NAFNet Refiner | XReflection-RDNet | GT
```

If no GT directory is provided, or no matching GT is found, the GT column is
omitted automatically.
