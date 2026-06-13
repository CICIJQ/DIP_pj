# 资产清单

快照日期：2026-06-14

这份文件用于说明：

- 哪些资产已经直接放进当前 GitHub 仓库
- 哪些资产仍需额外下载
- 额外下载后的文件应放到仓库中的哪个路径

## 发布原则

- 仓库中只保留 benchmark-fair 的 RDNet refiner 结果。
- 明确排除混入 benchmark/test 数据训练的非公平版本 `all_paired`。
- 特别是 `ERRNet/results/rdnet_naf_refiner_full_eval` 不属于正式发布快照，不应继续引用、汇报或复用。

## GitHub 中已包含的内容

- `ERRNet/` 与 `DSRNet/` 的完整代码树
- `ERRNet/external_models/xreflection/XReflection/` 下的外部 XReflection 代码
- 精简后的 released results：
  - `ERRNet/released_results/`
  - `DSRNet/released_results/`
- 已直接跟踪的小型权重与 checkpoint：
  - `DSRNet/weights/dsrnet_s_epoch14.pt`（约 39.8 MB）
  - `ERRNet/checkpoints/errnet_naf_refiner_improved/naf_refiner_best.pt`（约 32.1 MB）
  - `ERRNet/checkpoints/errnet_naf_refiner_improved/naf_refiner_latest.pt`（约 32.1 MB）
  - `ERRNet/checkpoints/rdnet_naf_refiner_fair_voc_realtrain/best_psnr.pth`（约 71.3 MB）
  - `ERRNet/checkpoints/rdnet_naf_refiner_fair_voc_realtrain/latest.pth`（约 71.3 MB）
  - `ERRNet/checkpoints/rdnet_naf_confgate_fair_voc_realtrain/best_psnr.pth`（约 71.3 MB）
  - `ERRNet/checkpoints/rdnet_naf_confgate_fair_voc_realtrain/latest.pth`（约 71.3 MB）

## 快速使用步骤

1. 先 clone 当前仓库。
2. 按下表从网盘或其他外部存储下载缺失的大文件。
3. 将下载内容解压或复制到表中给出的仓库相对路径。
4. 如果要在离线环境运行 XReflection/RDNet，还需要按下文补齐 XReflection 的辅助缓存文件。

## 完整复现所需的外部资产

| 资产类别 | 维护者本地路径 | 下载后放置路径 | 体积 | 用途 |
| --- | --- | --- | --- | --- |
| 核心处理后数据集 | `/mnt/user4/dip_pj/ERRNet/datasets/processed_data/` | `ERRNet/datasets/processed_data/` | 约 3.1 GB | ERRNet 与 RDNet-refiner 训练/评测的主要输入数据 |
| 原始数据集包 | `/mnt/user4/dip_pj/ERRNet/datasets/raw_data/` | `ERRNet/datasets/raw_data/` | 约 3.4 GB | 原始数据、旧流程依赖和数据来源文件 |
| 额外定性对比输入 | `/mnt/user4/dip_pj/reflection_pairs_tight/` | `reflection_pairs_tight/` | 约 11 MB | 单独保存的定性对比输入图 |
| ERRNet baseline checkpoints | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet/` | `ERRNet/checkpoints/errnet/` | 约 2.2 GB | 原始 ERRNet aligned 模型系列 |
| ERRNet unaligned fine-tune | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_unaligned_ft/` | `ERRNet/checkpoints/errnet_unaligned_ft/` | 约 1.1 GB | ERRNet 第二阶段 / unaligned 微调模型 |
| ERRNet attention-rebalanced | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_attn_rebalanced_v1/` | `ERRNet/checkpoints/errnet_attn_rebalanced_v1/` | 约 1.8 GB | 注意力重平衡分支 |
| ERRNet improved-loss | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_improved_loss_v1/` | `ERRNet/checkpoints/errnet_improved_loss_v1/` | 约 1.6 GB | improved-loss 分支 |
| ERRNet realistic cascade | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_ours_realistic_cascade/` | `ERRNet/checkpoints/errnet_ours_realistic_cascade/` | 约 1.6 GB | realistic-cascade 分支 |
| ERRNet transformer | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_transformer/` | `ERRNet/checkpoints/errnet_transformer/` | 约 763 MB | transformer 分支 |
| ERRNet transformer cascade | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_transformer_cascade_v2/` | `ERRNet/checkpoints/errnet_transformer_cascade_v2/` | 约 2.3 GB | transformer-cascade 分支 |
| ERRNet prior-probe 历史权重 | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_prior_probe_calib_*/` | `ERRNet/checkpoints/` | 多个约 229 MB 目录 | prior-probe 标定/搜索实验 |
| ERRNet prior-stage2 历史权重 | `/mnt/user4/dip_pj/ERRNet/checkpoints/errnet_prior_stage2_*/` | `ERRNet/checkpoints/` | 多个约 229 MB 目录 | prior stage-2 搜索实验 |
| XReflection RDNet 主权重 | `/mnt/user4/dip_pj/ERRNet/checkpoints/xreflection/rdnet-26.4849.ckpt` | `ERRNet/checkpoints/xreflection/rdnet-26.4849.ckpt` | 约 3.45 GB | `run_xreflection_rdnet.py` 使用的 RDNet coarse model |

## 建议另外存网盘的结果档案

这些内容不是“运行代码”所必需，但如果队友希望直接查看所有原始输出图，而不只是 GitHub 里的精简版 `released_results`，则建议同时提供网盘下载。

| 结果档案 | 维护者本地路径 | 下载后放置路径 | 体积 |
| --- | --- | --- | --- |
| RDNet coarse 全量输出 | `/mnt/user4/dip_pj/ERRNet/results/xreflection_rdnet/` | `ERRNet/results/xreflection_rdnet/` | 约 3.4 GB |
| RDNet RCA 全量输出 | `/mnt/user4/dip_pj/ERRNet/results/ra_rdnet_rca/` | `ERRNet/results/ra_rdnet_rca/` | 约 406 MB |
| RDNet post-hoc mask 全量输出 | `/mnt/user4/dip_pj/ERRNet/results/rdnet_naf_refiner_posthoc_mask_eval_gpu/` | `ERRNet/results/rdnet_naf_refiner_posthoc_mask_eval_gpu/` | 约 334 MB |
| ERRNet ensemble fixed 全量输出 | `/mnt/user4/dip_pj/ERRNet/results/eval_ensemble_fixed/` | `ERRNet/results/eval_ensemble_fixed/` | 约 3.2 GB |
| ERRNet ensemble search 全量输出 | `/mnt/user4/dip_pj/ERRNet/results/eval_ensemble/` | `ERRNet/results/eval_ensemble/` | 约 148 MB |

## XReflection 辅助缓存说明

`ERRNet/external_models/xreflection/XReflection/options/train_rdnet.yml`
里还会用到两个辅助模型文件：`cls_model.pth` 和 `focal.pth`。
XReflection 在首次运行时会通过 `torch.hub` 自动下载它们。

如果队友运行环境无法联网，建议额外准备好下面这个缓存目录：

- `$(python -c "import torch; print(torch.hub.get_dir())")/xreflection_aux_checkpoints/`

并将 XReflection 需要的辅助文件提前放进去。

## 仓库里已经提供的精简结果

当前 GitHub 仓库已经包含以下“可直接查看”的精简结果快照：

- 历史 ERRNet benchmark 结果
- ERRNet 的 TTA 与 prior-search 结果
- ERRNet 的定性对比图与排名表
- RDNet coarse / fair refiner / confidence gate 的指标摘要
- RDNet RCA 与 post-hoc mask 的摘要结果
- ERRNet ensemble 的摘要结果
- DSRNet 在 `max512` 和 `fullres_tiled512_amp` 两种模式下的摘要结果

## 放置完成后的检查项

把外部资产放好之后，至少应能看到下面这些关键路径：

- `ERRNet/datasets/processed_data/`
- `ERRNet/checkpoints/errnet/errnet_latest.pt`
- `ERRNet/checkpoints/errnet_unaligned_ft/errnet_latest.pt`
- `ERRNet/checkpoints/xreflection/rdnet-26.4849.ckpt`
- `DSRNet/weights/dsrnet_s_epoch14.pt`

如果这些路径都齐了，那么当前仓库中的主要历史 ERRNet 流程、DSRNet 对比流程，以及基于 RDNet 的 refinement 脚本，就都能按预期资产布局工作。
