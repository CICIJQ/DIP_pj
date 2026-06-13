# DSRNet Local Deployment

This directory is a standalone deployment of <https://github.com/mingcv/DSRNet>.
It is kept separate from the existing `ERRNet` project.

## Environment

Use the local virtual environment:

```bash
cd /home/xumx/dip_pj/DSRNet
source .venv/bin/activate
```

The venv was created with `--system-site-packages` so it reuses the existing
CUDA PyTorch from:

```text
/SSDHome/home/xumx/anaconda3/envs/errnet
```

DSRNet-specific dependencies were installed into:

```text
/home/xumx/dip_pj/DSRNet/.venv
```

## Data Adapter

The original ERRNet datasets were not modified. DSRNet sees them through
symlinks under:

```text
/home/xumx/dip_pj/DSRNet/datasets/errnet_processed_for_dsrnet
```

Mapping:

```text
test/real20_420                 -> ERRNet/datasets/processed_data/real20
test/SIR2/SolidObjectDataset    -> ERRNet/datasets/processed_data/objects
test/SIR2/PostcardDataset       -> ERRNet/datasets/processed_data/postcard
test/SIR2/WildSceneDataset      -> ERRNet/datasets/processed_data/wild
```

## Available Weight

The cloned repository includes:

```text
weights/dsrnet_s_epoch14.pt
```

The README also mentions `dsrnet_l_epoch18.pt` and
`dsrnet_l_4000_epoch33.pt`, but those are not included in the GitHub checkout
and need to be downloaded separately from the links in the upstream README.

## Smoke Test

This command was verified locally on CPU with one selected `real20` image:

```bash
.venv/bin/python eval_sirs.py \
  --inet dsrnet_s \
  --model dsrnet_model_sirs \
  --dataset sirs_dataset \
  --name dsrnet_s_smoke \
  --hyper \
  --if_align \
  --resume \
  --weight_path ./weights/dsrnet_s_epoch14.pt \
  --base_dir datasets/errnet_processed_for_dsrnet \
  --gpu_ids -1 \
  --nThreads 0 \
  --selected 3 \
  --no-verbose \
  --no-log
```

Observed smoke-test metric for the selected image:

```text
PSNR: 22.4286
SSIM: 0.7982
NCC: 0.9228
LMSE: 0.0101
```

Output was written under:

```text
checkpoints/dsrnet_s_smoke/
```

## Full Evaluation Command

For GPU evaluation with the small included checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_sirs.py \
  --inet dsrnet_s \
  --model dsrnet_model_sirs \
  --dataset sirs_dataset \
  --name dsrnet_s_eval \
  --hyper \
  --if_align \
  --resume \
  --weight_path ./weights/dsrnet_s_epoch14.pt \
  --base_dir datasets/errnet_processed_for_dsrnet \
  --gpu_ids 0 \
  --nThreads 4 \
  --no-verbose \
  --no-log
```

## Local Compatibility Patches

Two small compatibility fixes were applied inside this standalone DSRNet copy:

1. `models/arch/__init__.py`: `dsrnet_s` now passes `lrm_blk_nums=[2, 4]`,
   matching the included `dsrnet_s_epoch14.pt` checkpoint.
2. `models/dsrnet_model_sirs.py`: `torch.load(..., map_location=model.device)`
   allows CPU loading of CUDA-saved checkpoints.
