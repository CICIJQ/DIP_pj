# No-GT qualitative outputs

This folder contains inference-only outputs for the four images in `/home/xumx/dip_pj/reflection_pairs_tight` that do not have ground-truth transmission images. No quantitative metrics should be reported from this run. NAFNet Refiner and DSRNet-S were executed through loaders that require a target image, using `target=input` only as a placeholder to save outputs.

Main figure:

- `panels/nogt_11_methods_comparison.png`: compact comparison panel.
- `panels/nogt_11_methods_comparison_large.png`: larger inspection version.

Column order follows the previous `reflection_pairs_tight` PSNR ranking, with the input column added first.
