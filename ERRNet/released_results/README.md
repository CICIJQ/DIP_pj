# Released Results

这个目录保存的是本项目主要实验结果的精简版、GitHub 友好版导出。

早期发布快照里，RDNet 相关内容偏多；现在这份导出已经同时补齐了：

- 历史 ERRNet 主线实验
- 后续基于 RDNet 的替换与 refinement 实验

这里包含的内容：

- `errnet_mainline/`：ERRNet 主线模型的 benchmark 摘要
  - 包括 `eval_baseline`、`eval_improved_loss`、`eval_attn_rebalanced`、
    `eval_ours`、`eval_transformer`、`eval_transformer_cascade_v2_latest`、
    `eval_prior_stage2_best_full`、`errnet_naf_refiner_improved_best`
- `errnet_tta/`：ERRNet 主线模型的 TTA 结果摘要
- `errnet_prior_search/`：prior-probe 与 prior-stage2 的搜索结果摘要
- `errnet_ensemble/`：ensemble-search 相关实验的精简摘要
- `errnet_qualitative/`：ERRNet 线的定性图、排名表、面板图与自采图对比
- `training/errnet_mainline/` 与 `training/errnet_prior/`：训练日志与 prior 历史记录
- `training/checkpoint_configs/`：从本地 checkpoint 目录提取的小型 `opt.txt`、`metrics.txt`、`loss_log` 等配置痕迹
- `dataset_manifests/`：fair RDNet-refiner 训练用到的小型数据清单
- `legacy_stage2_real20/`：早期 stage-1/stage-2 ERRNet 流程在 `real20` 上的简要摘要
- `rdnet_ablation/`、`rdnet_naf_refiner_fair_eval_gpu/`、
  `rdnet_naf_confgate_fair_eval_gpu/`、`rdnet_rca_eval/`、
  `rdnet_naf_refiner_posthoc_mask_eval_gpu/` 与 `training/rdnet_*`
  - 这些是后续 RDNet 替换与 refinement 相关实验的精简结果

这里明确不保留：

- 非公平的 mixed benchmark-training RDNet refiner 结果
- `ERRNet/results/rdnet_naf_refiner_full_eval`

这里仍未包含：

- 每个大型评测目录下的完整逐图像输出图
- 全部 checkpoint 与大模型权重
- 原始数据集
- tensorboard 日志及其他大型中间文件

更详细的代码背景请查看 `ERRNet/` 主目录说明。这个目录的定位是：

- 保留论文/报告里会引用到的结果表格
- 保留关键摘要与定性对比
- 但不把所有大体积原始输出都直接塞进 GitHub
