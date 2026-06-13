# Released Results

这个目录保存的是 DSRNet 作为外部对比方法时的精简结果导出。

这里包含：

- 在 ERRNet 处理后 benchmark 上运行的 full-resolution tiled 评测摘要
- 同一 benchmark 上 `max_long_edge=512` 模式的评测摘要
- 两种模式下各数据集的 `summary.txt` 与 `per_image_metrics.csv`
- 本地 fine-tuning 过程中保存的训练 loss 日志
- 轻量权重 `DSRNet/weights/dsrnet_s_epoch14.pt`

这里不包含：

- 大体积逐图像输出结果图
- 原始数据集
- 其他大型中间文件
