# reflection_pairs_tight Evaluation Ranking

All methods were evaluated on GPU0 on 5 paired images from `/home/xumx/dip_pj/reflection_pairs_tight`. Inputs and GTs were first cropped to their common overlapping size per pair, then evaluated with `max_long_edge=512`. Ranking is by average PSNR.

| Rank | Method | Images | PSNR | SSIM | NCC | LMSE | Note |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Realistic Cascade | 5 | 17.902448 | 0.523148 | 0.749647 | 0.093149 |  |
| 2 | DSRNet-S max512 | 5 | 17.837561 | 0.521176 | 0.705743 | 0.107186 | external model; GPU0; max_long_edge=512 |
| 3 | Attn Rebalanced + TTA | 5 | 17.804679 | 0.528286 | 0.757739 | 0.089025 |  |
| 4 | Improved Loss + TTA | 5 | 17.783222 | 0.524961 | 0.730215 | 0.094298 |  |
| 5 | Attn Rebalanced | 5 | 17.720207 | 0.521157 | 0.753897 | 0.090567 |  |
| 6 | Improved Loss | 5 | 17.645949 | 0.516219 | 0.724199 | 0.098360 |  |
| 7 | NAFNet Refiner | 5 | 17.643307 | 0.515735 | 0.724275 | 0.099067 |  |
| 8 | ERRNet Baseline + TTA | 5 | 17.584607 | 0.502747 | 0.725858 | 0.104530 |  |
| 9 | Transformer Cascade v2 | 5 | 17.506734 | 0.501353 | 0.706161 | 0.111429 |  |
| 10 | Reflection Prior Branch | 5 | 17.424864 | 0.493097 | 0.718332 | 0.108619 |  |
| 11 | ERRNet Baseline | 5 | 17.413774 | 0.493025 | 0.717618 | 0.109448 |  |

Plain Transformer was not rerun in the final ranking because the available checkpoint is incompatible with the current `errnet_transformer` construction: loading fails with coarse-net channel and residual-block mismatches. Transformer Cascade v2 is included.
