# Released Results

This directory contains a compact, Git-friendly export of the main
experiment artifacts produced in this project.

The earlier release snapshot was RDNet-heavy. This export now includes
both the historical ERRNet experiments and the later RDNet-based branch.

Included here:

- `errnet_mainline/`: benchmark summaries for the main ERRNet variants
  (`eval_baseline`, `eval_improved_loss`, `eval_attn_rebalanced`,
  `eval_ours`, `eval_transformer`, `eval_transformer_cascade_v2_latest`,
  `eval_prior_stage2_best_full`, `errnet_naf_refiner_improved_best`)
- `errnet_tta/`: TTA benchmark summaries for the main ERRNet variants
- `errnet_prior_search/`: prior-probe and prior-stage2 sweep summaries
- `errnet_ensemble/`: compact summaries from the ensemble-search experiments
- `errnet_qualitative/`: qualitative outputs, ranking tables, panels, and
  self-collected comparison images for the ERRNet line
- `training/errnet_mainline/` and `training/errnet_prior/`: compact loss
  logs and prior evaluation histories
- `training/checkpoint_configs/`: small `opt.txt`, `metrics.txt`, and related
  training metadata copied from local checkpoint folders
- `dataset_manifests/`: small exported dataset manifests used by the
  fair RDNet-refiner training path
- `legacy_stage2_real20/`: small legacy summary for the original
  stage-1/stage-2 ERRNet pipeline on `real20`
- `rdnet_ablation/`, `rdnet_naf_refiner_fair_eval_gpu/`,
  `rdnet_naf_confgate_fair_eval_gpu/`, `rdnet_rca_eval/`,
  `rdnet_naf_refiner_posthoc_mask_eval_gpu/`, and `training/rdnet_*`: the
  later RDNet replacement experiments and refiner runs

Explicitly excluded:

- the non-fair mixed benchmark-training RDNet refiner run
- `ERRNet/results/rdnet_naf_refiner_full_eval`

Not included here:

- full benchmark image dumps for every large evaluation run
- checkpoints and model weights
- raw datasets
- tensorboard logs and other large intermediate artifacts

Refer to the main `ERRNet/` documentation for code details. The files in
this folder are intended to preserve the result tables, summaries, and
qualitative comparisons cited in the project analysis.
