# XReflection-RDNet comparison snapshot

Metrics are computed with `scripts/eval_reflection_methods.py` using stem matching and resizing predictions to the GT size.

## real20

| Method | Images | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| ERRNet baseline | 20 | 21.790616 | 0.748567 | 0.858559 | 0.024337 |
| NAFNet Refiner | 20 | 22.196342 | 0.766813 | 0.880448 | 0.019521 |
| XReflection-RDNet | 20 | 23.248486 | 0.797454 | 0.890643 | 0.014704 |

## self_collected_gt

| Method | Images | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| ERRNet baseline | 5 | 17.183729 | 0.493652 | 0.710729 | 0.105155 |
| NAFNet Refiner | 5 | 17.411903 | 0.523295 | 0.718239 | 0.097318 |
| XReflection-RDNet | 5 | 17.525721 | 0.517736 | 0.752041 | 0.093547 |

## Generated grids

- `results/xreflection_rdnet/real20/comparison_grid.png`
- `results/xreflection_rdnet/self_collected_gt/comparison_grid.png`
- `results/xreflection_rdnet/self_collected_nogt/comparison_grid.png`
