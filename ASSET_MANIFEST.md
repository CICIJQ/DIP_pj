# Asset Manifest

Snapshot date: 2026-06-14

This file records which assets are already tracked in this GitHub repository,
which assets must still be downloaded separately, and where every external
bundle must be placed after download.

## Policy

- Only benchmark-fair RDNet-refiner results are kept in this repository.
- The non-fair mixed benchmark-training run (`all_paired`) is intentionally
  excluded.
- In particular, `ERRNet/results/rdnet_naf_refiner_full_eval` is not part of
  the published GitHub snapshot and should not be reported or reused.

## Already Included In GitHub

- Full source trees for `ERRNet/` and `DSRNet/`
- External XReflection code under `ERRNet/external_models/xreflection/XReflection/`
- Compact released result snapshots under:
  - `ERRNet/released_results/`
  - `DSRNet/released_results/`
- Selected lightweight checkpoints and weights:
  - `DSRNet/weights/dsrnet_s_epoch14.pt` (~39.8 MB)
  - `ERRNet/checkpoints/errnet_naf_refiner_improved/naf_refiner_best.pt` (~32.1 MB)
  - `ERRNet/checkpoints/errnet_naf_refiner_improved/naf_refiner_latest.pt` (~32.1 MB)
  - `ERRNet/checkpoints/rdnet_naf_refiner_fair_voc_realtrain/best_psnr.pth` (~71.3 MB)
  - `ERRNet/checkpoints/rdnet_naf_refiner_fair_voc_realtrain/latest.pth` (~71.3 MB)
  - `ERRNet/checkpoints/rdnet_naf_confgate_fair_voc_realtrain/best_psnr.pth` (~71.3 MB)
  - `ERRNet/checkpoints/rdnet_naf_confgate_fair_voc_realtrain/latest.pth` (~71.3 MB)

## Quick Setup

1. Clone the repository.
2. Read the table below and download the missing external bundles from cloud storage.
3. Copy or extract each bundle into the exact repo-relative target path listed below.
4. If you want to run XReflection/RDNet offline, also pre-populate the auxiliary XReflection cache noted below.

## External Assets Required For Full Reproduction

| Asset group | Local source on maintainer machine | Place after download | Size | Purpose |
| --- | --- | --- | --- | --- |
| Core processed datasets | `/mnt/user4/dip_pj/ERRNet/datasets/processed_data/` | `ERRNet/datasets/processed_data/` | ~3.1 GB | Main training/evaluation inputs used by ERRNet and RDNet-refiner scripts |
| Raw dataset bundle | `/mnt/user4/dip_pj/ERRNet/datasets/raw_data/` | `ERRNet/datasets/raw_data/` | ~3.4 GB | Original/raw data and legacy dataset assets |
| Qualitative extra inputs | `/mnt/user4/dip_pj/reflection_pairs_tight/` | `reflection_pairs_tight/` | ~11 MB | Standalone qualitative comparison inputs |
| ERRNet baseline checkpoints | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet/` | `ERRNet/checkpoints/errnet/` | ~2.2 GB | Original ERRNet aligned model family |
| ERRNet unaligned fine-tune | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_unaligned_ft/` | `ERRNet/checkpoints/errnet_unaligned_ft/` | ~1.1 GB | Stage-2 / unaligned ERRNet model |
| ERRNet attention-rebalanced | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_attn_rebalanced_v1/` | `ERRNet/checkpoints/errnet_attn_rebalanced_v1/` | ~1.8 GB | Attention-rebalanced branch |
| ERRNet improved-loss | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_improved_loss_v1/` | `ERRNet/checkpoints/errnet_improved_loss_v1/` | ~1.6 GB | Improved-loss branch |
| ERRNet realistic cascade | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_ours_realistic_cascade/` | `ERRNet/checkpoints/errnet_ours_realistic_cascade/` | ~1.6 GB | Realistic-cascade branch |
| ERRNet transformer | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_transformer/` | `ERRNet/checkpoints/errnet_transformer/` | ~763 MB | Transformer branch |
| ERRNet transformer cascade | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_transformer_cascade_v2/` | `ERRNet/checkpoints/errnet_transformer_cascade_v2/` | ~2.3 GB | Transformer-cascade branch |
| ERRNet prior-probe archive | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_prior_probe_calib_*/` | `ERRNet/checkpoints/` | several 229 MB dirs | Prior-probe calibration sweeps |
| ERRNet prior-stage2 archive | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_prior_stage2_*/` | `ERRNet/checkpoints/` | several 229 MB dirs | Prior stage-2 sweeps |
| XReflection RDNet checkpoint | `/mnt/user4/dip_pj/ERRNet/checkpoints/xreflection/rdnet-26.4849.ckpt` | `ERRNet/checkpoints/xreflection/rdnet-26.4849.ckpt` | ~3.45 GB | RDNet coarse model used by `run_xreflection_rdnet.py` |

## Optional But Useful External Result Archives

These are not required to run the code, but they are required if teammates
want every original generated image dump rather than the compact snapshots in
`released_results/`.

| Result archive | Local source on maintainer machine | Place after download | Size |
| --- | --- | --- | --- |
| Full RDNet coarse outputs | `/mnt/user4/dip_pj/ERRNet/results/xreflection_rdnet/` | `ERRNet/results/xreflection_rdnet/` | ~3.4 GB |
| RDNet RCA full outputs | `/mnt/user4/dip_pj/ERRNet/results/ra_rdnet_rca/` | `ERRNet/results/ra_rdnet_rca/` | ~406 MB |
| RDNet post-hoc mask full outputs | `/mnt/user4/dip_pj/ERRNet/results/rdnet_naf_refiner_posthoc_mask_eval_gpu/` | `ERRNet/results/rdnet_naf_refiner_posthoc_mask_eval_gpu/` | ~334 MB |
| ERRNet ensemble fixed outputs | `/mnt/user4/dip_pj/ERRNet/results/eval_ensemble_fixed/` | `ERRNet/results/eval_ensemble_fixed/` | ~3.2 GB |
| ERRNet ensemble search outputs | `/mnt/user4/dip_pj/ERRNet/results/eval_ensemble/` | `ERRNet/results/eval_ensemble/` | ~148 MB |

## XReflection Auxiliary Cache Note

`ERRNet/external_models/xreflection/XReflection/options/train_rdnet.yml`
references auxiliary model files named `cls_model.pth` and `focal.pth`.
XReflection downloads them automatically through `torch.hub` on first run.

If your teammates will run RDNet on a machine without outbound internet
access, also pre-populate:

- `$(python -c "import torch; print(torch.hub.get_dir())")/xreflection_aux_checkpoints/`

with the auxiliary files that XReflection expects.

## Included Release Snapshots

The GitHub repo already contains compact, report-ready snapshots for:

- historical ERRNet benchmark runs
- ERRNet TTA and prior-search runs
- ERRNet qualitative panels
- RDNet coarse/fair-refiner/confidence-gate metric summaries
- RDNet RCA and post-hoc mask summary outputs
- ERRNet ensemble summary outputs
- DSRNet summary outputs for both `max512` and `fullres_tiled512_amp`

## Verification Checklist

After placing external assets, the following paths should exist:

- `ERRNet/datasets/processed_data/`
- `ERRNet/checkpoints/errnet/errnet_latest.pt`
- `ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt`
- `ERRNet/checkpoints/xreflection/rdnet-26.4849.ckpt`
- `DSRNet/weights/dsrnet_s_epoch14.pt`

If those exist, the main historical ERRNet runs, DSRNet comparison, and
RDNet-based refinement scripts are all wired to the expected asset layout.
