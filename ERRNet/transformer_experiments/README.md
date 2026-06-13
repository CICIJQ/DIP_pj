# Transformer ERRNet Experiment

This experiment adds a hybrid Transformer cascade branch to the original ERRNet pipeline.

## 新增内容

- `models/arch/transformer.py`
  - `TransformerCascadeNet`: coarse CNN + Transformer refinement cascade
  - `PatchTransformerBlock`: token-based self-attention on squeezed feature maps
  - `TransformerRefineNet`: replaces the final CNN refinement module with Transformer-enhanced global reasoning

- `models/arch/__init__.py`
  - 加入 `errnet_transformer` 架构入口

- `transformer_experiments/train_transformer_errnet.py`
  - 训练入口脚本，默认使用 `errnet_transformer` 结构
  - debug 模式下自动开启小规模实验

## 运行方式

从 `ERRNet` 目录执行：

```bash
cd /home/xumx/dip_pj/ERRNet
python transformer_experiments/train_transformer_errnet.py --debug --gpu_ids 0
```

输出日志与 checkpoints 会放到：

- `checkpoints/errnet_transformer/`
- `results/eval_transformer/`

## 设计思想

- 采用级联结构：先用 `DRNet` 生成粗糙恢复，再用 Transformer 模块做全局细化。
- Transformer 模块引入全局 self-attention，增强长距离依赖和复杂纹理建模能力。
- 结果隔离为独立实验目录，避免影响现有基线和 prior/cascade 分支。
