# Dataset split summary

This project keeps the actively used data under `datasets/processed_data/`.
The original downloads live under `datasets/raw_data/`.

## Training Data

| Split | Path | Count | Notes |
| --- | --- | ---: | --- |
| Synthetic source images | `datasets/processed_data/VOCdevkit/VOC2012/PNGImages` | 15,287 | Used by `train_errnet.py` through `CEILDataset`; reflection mixtures are synthesized online. File list: `VOC2012_224_train_png.txt` also has 15,287 entries. |
| Fixed synthetic train-only pairs for RDNet refiner | `datasets/processed_data/rdnet_refiner_voc_synth_train` | 7,643 | Exported from the same VOC training source with `scripts/build_rdnet_refiner_synth_dataset.py`; benchmark-safe because it is derived from training-only VOC images, not from course test sets. |
| Real paired training | `datasets/processed_data/real_train/blended` | 89 | Paired with `real_train/transmission_layer`; copied from `datasets/raw_data/real89`. |
| Real paired training GT | `datasets/processed_data/real_train/transmission_layer` | 89 | Ground truth for the 89 real training images. |
| Unaligned DSLR training, raw only | `datasets/raw_data/Dataset/DSLR/unaligned_train250/blended` | 250 | Used only by `train_errnet_unaligned.py`; not part of the default processed aligned training set. |
| Unaligned DSLR training GT, raw only | `datasets/raw_data/Dataset/DSLR/unaligned_train250/transmission_layer` | 250 | Misaligned/unaligned pair target. |

## Validation Data

There is no separate dedicated validation split in the current processed dataset.
`train_errnet.py` evaluates every 5 epochs on:

| Eval During Training | Path | Count | Notes |
| --- | --- | ---: | --- |
| CEILNet table2 | `datasets/processed_data/testdata_CEILNET_table2` | 100 pairs | Used as eval/test, not a clean training validation split. |
| real20 | `datasets/processed_data/real20` | 20 pairs | Used as eval/test, not a clean training validation split. |

For new training experiments, create a validation split from training-only data
instead of using the course test sets for model selection.

## Course Test Data

| Test Set | Input Path | GT Path | Count |
| --- | --- | --- | ---: |
| CEILNet synthetic table2 | `datasets/processed_data/testdata_CEILNET_table2/blended` | `datasets/processed_data/testdata_CEILNET_table2/transmission_layer` | 100 |
| real20 | `datasets/processed_data/real20/blended` | `datasets/processed_data/real20/transmission_layer` | 20 |
| SIR2 with GT | `datasets/processed_data/sir2_withgt/blended` | `datasets/processed_data/sir2_withgt/transmission_layer` | 480 |
| SIR2 Objects | `datasets/processed_data/objects/blended` | `datasets/processed_data/objects/transmission_layer` | 200 |
| SIR2 Postcard | `datasets/processed_data/postcard/blended` | `datasets/processed_data/postcard/transmission_layer` | 179 |
| SIR2 Wild | `datasets/processed_data/wild/blended` | `datasets/processed_data/wild/transmission_layer` | 101 |
| Self-collected with GT | `datasets/processed_data/reflection_pairs_tight_eval/testdata_CEILNET_table2/blended` | `datasets/processed_data/reflection_pairs_tight_eval/testdata_CEILNET_table2/transmission_layer` | 5 |
| Self-collected no-GT inputs | `datasets/processed_data/reflection_pairs_tight_nogt_inputs` | none | 4 |

## Raw Data Reference

| Raw Data | Path | Count |
| --- | --- | ---: |
| Berkeley real89 inputs | `datasets/raw_data/real89/blended` | 89 |
| Berkeley real89 GT | `datasets/raw_data/real89/transmission_layer` | 89 |
| VOC2012 JPEG source images | `datasets/raw_data/VOCdevkit/VOC2012/JPEGImages` | 17,125 |
| CEILNet synthetic table2 raw inputs | `datasets/raw_data/CEILNet/testdata_reflection_synthetic_table2/*-input.png` | 100 |
| CEILNet synthetic table2 raw label1 | `datasets/raw_data/CEILNet/testdata_reflection_synthetic_table2/*-label1.png` | 100 |
| CEILNet synthetic table2 raw label2 | `datasets/raw_data/CEILNet/testdata_reflection_synthetic_table2/*-label2.png` | 100 |
| CEILNet real raw inputs | `datasets/raw_data/CEILNet/testdata_reflection_real` | 45 |
| SIR2 raw Objects | `datasets/raw_data/robustsirr_test_dataset/SIR2/SolidObjectDataset` | 200 triplets |
| SIR2 raw Postcard | `datasets/raw_data/robustsirr_test_dataset/SIR2/PostcardDataset` | 179 triplets |
| SIR2 raw Wild | `datasets/raw_data/robustsirr_test_dataset/SIR2/WildSceneDataset` | 101 triplets |
| DSLR unaligned test raw | `datasets/raw_data/Dataset/DSLR/unaligned_test50` | 50 pairs |
| Smartphone unaligned raw | `datasets/raw_data/Dataset/Smartphone/unaligned150` | 148 pairs |

## Why `real_train` Has Only 89 Images

`real_train` is only the aligned real paired subset from Berkeley `real89`.
The main ERRNet training set is larger because `train_errnet.py` mixes:

- 70% online synthetic reflection data from VOC images.
- 30% real paired images from `real_train`.

So the 89 real pairs are not the whole training set; they are the real paired
component of the default training mixture.
