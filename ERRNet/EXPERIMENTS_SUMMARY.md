# ERRNet 当前尝试方法与结果整理

整理时间：2026-06-08

## 结论摘要

按目前已经保存的评估结果看，如果要求 6 个测试集都完整可比，最新训练完成的 **NAFNet Refiner (Improved Coarse)** 综合效果最好。它在完整 6 数据集上的平均 PSNR / SSIM / LMSE 分别为 **24.857788 / 0.890494 / 0.007127**，超过此前最强的 **Improved Loss**。不过它在 `SIR2_objects`、`SIR2_postcard`、`sir2_withgt` 上不是单集最优，因此报告时仍建议保留逐数据集对比。

新增外部模型 **DSRNet-S** 已完成两组 6 数据集测试。**DSR 全量版**采用不缩放输入的全分辨率滑窗评估（`max_long_edge=-1`、`tile_size=512`、`tile_overlap=64`、AMP、`split_intro_devices=0,1`），其 6 数据集平均 PSNR / SSIM / LMSE 为 **24.382148 / 0.885447 / 0.006664**。此前 `max_long_edge=512` 缩放评估为 **24.786693 / 0.893481 / 0.006961**。直接整图全分辨率推理在 `real20` 上会 OOM；滑窗结果按原图尺寸计算指标，但推理上下文不同于整图前向，因此作为外部模型补充对比，不直接参与下面的整图综合排名。

如果只看单个数据集，最优方法并不完全一致：

| 数据集 | 当前 PSNR 最好方法 | PSNR | SSIM | LMSE | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| CEILNet_table2 | NAFNet Refiner (Improved Coarse) | 30.405712 | 0.960991 | 0.003244 | 当前该集 PSNR/SSIM/LMSE 最好 |
| real20 | Attn Rebalanced | 23.107052 | 0.814886 | 0.021031 | `Ours` 的 SSIM/LMSE 更好：0.818371 / 0.019677 |
| sir2_withgt | Baseline + TTA | 24.079885 | 0.895275 | 0.003906 | TTA 对该集有效 |
| SIR2_objects | Prior Stage2 Best Full | 24.937591 | 0.898826 | 0.002927 | `Transformer Cascade v2 latest/full` 的 SSIM/LMSE 更好：0.901884 / 0.002895 |
| SIR2_postcard | Realistic Cascade (Ours) | 22.822599 | 0.869184 | 0.004861 | `Transformer Cascade v2 latest/full` 的 SSIM/LMSE 更好：0.884959 / 0.004003 |
| SIR2_wild | NAFNet Refiner (Improved Coarse) | 26.024046 | 0.913161 | 0.004794 | 当前该集 PSNR/SSIM/NCC/LMSE 最好 |

`Transformer Cascade v2 latest/full` 已补完 6 数据集评估。结论是：latest checkpoint 没有超过 Improved Loss 的综合 PSNR/LMSE；并且在 `real20`、`CEILNet_table2` 上低于此前 epoch45 的局部结果，说明后续训练没有带来更好的主指标。

`NAFNet Refiner (Improved Coarse)` 已补完 6 数据集评估。结论是：它显著抬高 `CEILNet_table2`，并在 `SIR2_wild` 上取得新最好结果，从而成为当前完整 6 数据集综合第一；主要风险是 `SIR2_objects`、`SIR2_postcard`、`sir2_withgt` 对比原 Baseline/Improved/Prior 有回退。

`DSRNet-S` 的全量版在 `sir2_withgt`、`SIR2_objects`、`SIR2_postcard`、`SIR2_wild` 上 PSNR 很高，但 `real20` 明显低于缩放评估版本；它可以作为外部模型补充实验，但不能替换上表的整图单集最好方法。

## 完整 6 数据集综合排名

这里仅统计已有完整 6 数据集评估的方法，不把只测了 2/3 个数据集的方法混入总排名。PSNR/SSIM 越高越好，LMSE 越低越好。

| 方法 | 平均 PSNR | 平均 SSIM | 平均 LMSE | 判断 |
| --- | ---: | ---: | ---: | --- |
| NAFNet Refiner (Improved Coarse) | 24.857788 | 0.890494 | 0.007127 | 当前完整 6 数据集综合最佳；table2/wild 提升明显，但 objects/postcard/sir2 不是单集最优 |
| Improved Loss | 24.578744 | 0.888263 | 0.007244 | 此前综合最佳；仍是 NAF Refiner 的 coarse model 和强基线 |
| Transformer Cascade v2 latest/full | 24.570704 | 0.889499 | 0.007557 | 均值非常接近 Improved Loss，但已被 NAF Refiner 超过 |
| Prior Stage2 Best Full | 24.432510 | 0.883611 | 0.008146 | 在 objects/sir2/postcard 部分指标好，但 real20/table2 较弱 |
| Baseline | 24.403693 | 0.883980 | 0.007969 | 稳定基线，sir2/objects 表现仍强 |
| Attn Rebalanced | 24.342035 | 0.888091 | 0.007452 | real20 和 table2 强，postcard 拉低均值 |
| Realistic Cascade (Ours) | 21.837659 | 0.853282 | 0.009330 | real20/postcard 有亮点，CEILNet_table2 明显失败 |
| Transformer | 18.183065 | 0.733343 | 0.019371 | 单独 Transformer 分支效果明显不如 CNN 基线 |

### DSRNet 全量版补充评估

下面两组只用于外部模型 DSRNet-S 的补充对比，不和上面的整图主排名直接比较：DSR 全量版保留原图尺寸，但把单张图拆成 tile 前向再拼接，推理上下文不同于整图前向；`max_long_edge=512` 缩放版会改变输入分辨率。

| 方法 | 平均 PSNR | 平均 SSIM | 平均 LMSE | 判断 |
| --- | ---: | ---: | ---: | --- |
| DSRNet-S epoch14, DSR 全量版 full-res tiled512 + AMP | 24.382148 | 0.885447 | 0.006664 | `max_long_edge=-1`，原图尺寸算指标；直接整图全分辨率 OOM，因此使用 512 tile / 64 overlap |
| DSRNet-S epoch14, max_long_edge=512 | 24.786693 | 0.893481 | 0.006961 | 缩放评估下 SSIM/LMSE 很强；CEILNet_table2 明显低于 NAF/Improved，全分辨率 OOM |

## 已尝试方法与结果

### 1. Baseline ERRNet

来源：`train_errnet.py`、`test_errnet.py`、`checkpoints/errnet/`、`results/eval_baseline/`

做法：原始 ERRNet 训练/测试流程，作为所有改进分支的参照。

结果：

- 6 数据集平均 PSNR 24.403693，平均 SSIM 0.883980。
- 在 `sir2_withgt`、`SIR2_objects` 上仍然很稳。
- 加 TTA 后在 `sir2_withgt` 达到 24.079885，是该数据集当前 PSNR/SSIM/LMSE 最好结果。

结论：Baseline 不是综合第一，但稳定性强，是必须保留的参照。

### 2. Improved Loss

来源：`train_errnet_improved_loss.py`、`models/improved_losses.py`、`checkpoints/errnet_improved_loss_v1/`、`results/eval_improved_loss/`

做法：

- 在像素损失中加入 L1、Gradient、SSIM。
- 训练总损失包含 improved pixel loss、VGG loss 和额外 SSIM loss。
- 训练数据使用 synthetic + real 的融合数据，脚本中 `lambda_gan=0`。

结果：

| 数据集 | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| real20 | 22.980068 | 0.815561 | 0.885744 | 0.020396 |
| CEILNet_table2 | 28.797373 | 0.948745 | 0.982771 | 0.003963 |
| sir2_withgt | 23.724507 | 0.889003 | 0.952266 | 0.004721 |
| SIR2_objects | 24.622366 | 0.896152 | 0.982508 | 0.003286 |
| SIR2_postcard | 21.552183 | 0.867915 | 0.921850 | 0.006258 |
| SIR2_wild | 25.795965 | 0.912200 | 0.946271 | 0.004840 |

结论：这是 NAF Refiner 之前的完整 6 数据集综合最强结果，也是 NAF Refiner 使用的 coarse model。主要提升在 `CEILNet_table2` 和 `SIR2_wild`；缺点是 `SIR2_postcard`、`sir2_withgt` 相比 Baseline 有下降。

### 3. Attn Rebalanced

来源：`train_errnet_attn_rebalanced.py`、`checkpoints/errnet_attn_rebalanced_v1/`、`results/eval_attn_rebalanced/`

做法：

- 基于 improved loss 继续调权重。
- 运行时强化 SSIM 项，目标是让结构相似性/感知质量更靠前。

结果：

| 数据集 | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| real20 | 23.107052 | 0.814886 | 0.892834 | 0.021031 |
| CEILNet_table2 | 28.778316 | 0.953623 | 0.982647 | 0.003922 |
| sir2_withgt | 23.321356 | 0.888592 | 0.942197 | 0.004782 |
| SIR2_objects | 24.532808 | 0.898057 | 0.982069 | 0.003231 |
| SIR2_postcard | 20.642405 | 0.869238 | 0.896846 | 0.005940 |
| SIR2_wild | 25.670270 | 0.904150 | 0.943615 | 0.005803 |

结论：`real20` PSNR/NCC 当前最好，`CEILNet_table2` 的 SSIM 很强；但 `postcard` 降得较多，所以综合均值低于 Improved Loss。

### 4. TTA 测试时增强

来源：`test_errnet.py --tta`、`results/eval_*_tta/`

做法：4-way flip test-time augmentation，对 baseline / improved loss / attn rebalanced 做了部分数据集评估。

结果：

| 方法 | 覆盖数据集 | 关键结果 |
| --- | --- | --- |
| Baseline + TTA | real20 / CEILNet_table2 / sir2_withgt | `sir2_withgt` PSNR 24.079885，为该集最好 |
| Improved Loss + TTA | real20 / CEILNet_table2 / sir2_withgt | `CEILNet_table2` PSNR 29.103442 |
| Attn Rebalanced + TTA | real20 / CEILNet_table2 / sir2_withgt | `CEILNet_table2` PSNR 29.135919，为该集最好 |

结论：TTA 对 `CEILNet_table2` 和 `sir2_withgt` 有效，但会让 `real20` PSNR 下降。建议只在对应 benchmark 上按需使用，不作为统一默认开关。

### 5. Realistic Cascade (Ours)

来源：`train_errnet_ours.py`、`models/arch/cascade.py`、`README_OURS.md`、`results/eval_ours/`

做法：

- 使用更真实的在线反射合成：alpha、ghost、noise、JPEG degradation。
- 使用 cascade refinement network。
- 两阶段训练：40 epoch synthetic realistic warmup，20 epoch synthetic + real finetune，默认 real ratio 0.6。

结果：

| 数据集 | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| real20 | 23.046309 | 0.818371 | 0.884563 | 0.019677 |
| CEILNet_table2 | 15.506169 | 0.775861 | 0.845066 | 0.017437 |
| sir2_withgt | 23.085070 | 0.879890 | 0.961664 | 0.004571 |
| SIR2_objects | 23.155985 | 0.882307 | 0.980412 | 0.003748 |
| SIR2_postcard | 22.822599 | 0.869184 | 0.949643 | 0.004861 |
| SIR2_wild | 23.409823 | 0.894076 | 0.945843 | 0.005685 |

结论：`real20` 的 SSIM/LMSE 和 `SIR2_postcard` 的 PSNR/NCC 有亮点，但 `CEILNet_table2` 崩得很明显，导致整体不可作为主方法。

### 6. Reflection Prior Branch（最优版本：Prior Stage2）

来源：`train_errnet_prior.py`、`test_errnet_prior.py`、`models/prior_branch.py`、`README_PRIOR_BRANCH.md`、`results/prior_stage2_summary.csv`、`results/eval_prior_stage2_best_full/`

方法目标：让网络先判断反射主要出现在哪里，再对这些区域做局部残差修正，尽量避免全图 refinement 破坏已经恢复正确的背景内容。

具体流程：

1. **粗恢复**：输入混合图像 \(I\) 先经过预训练 ERRNet，得到粗透射层 \(T_c\)。
2. **构造先验特征**：计算残差 \(R=I-T_c\)，并通过固定的 Laplacian 和 Sobel 算子提取高频响应。Prior Head 的输入为 \(I\)、\(T_c\)、\(R\)、\(|R|\)、Laplacian 和 Sobel 特征。
3. **预测反射 mask**：Prior Head 使用卷积层和 Sigmoid 输出单通道 mask \(M\)，表示各位置需要进一步去反射的强度。训练伪标签主要由 \(\max_c|I-T|\) 经过模糊、归一化和阈值映射得到，其中 \(T\) 是真值透射层。
4. **门控残差细化**：Refiner 根据 \(I\)、\(T_c\)、\(R\)、高频特征和 \(M\) 预测修正量 \(\Delta\)，最终输出为

   \[
   \hat{T}=T_c+G(M)\odot\Delta\cdot s
   \]

   其中 \(G(M)\) 是校准后的门控 mask，\(s\) 控制修正幅度。Refiner 最后一层采用零初始化，因此训练开始时输出与原 ERRNet 粗结果一致。
5. **训练约束**：除最终输出的像素损失和 VGG 损失外，还使用 mask 的 BCE+L1 监督、mask TV 平滑约束、非反射区域保护约束和 identity 约束，限制网络只在必要区域修改结果。

当前最佳配置为 `errnet_prior_stage2_base_s030_id005_g002` 的 epoch 1。该版本冻结基础 ERRNet，使用差分伪标签，残差缩放 \(s=0.3\)，不启用额外 gate 阈值；它在 `CEILNet_table2`、`real20`、`SIR2_wild` 三个调参集上的平均 PSNR 为 **24.838511**。

六数据集完整评估结果：

| 数据集 | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| real20 | 22.549676 | 0.793071 | 0.864021 | 0.026241 |
| CEILNet_table2 | 27.498573 | 0.937138 | 0.978990 | 0.005216 |
| sir2_withgt | 24.032147 | 0.892317 | 0.962563 | 0.004095 |
| SIR2_objects | 24.937591 | 0.898826 | 0.982283 | 0.002927 |
| SIR2_postcard | 22.341397 | 0.881223 | 0.949577 | 0.004231 |
| SIR2_wild | 25.235678 | 0.899092 | 0.946530 | 0.006165 |
| 平均 | 24.432510 | 0.883611 | 0.947327 | 0.008146 |

结论：这是当前 Prior 方法的最佳结果。它在 `SIR2_objects` 上取得当前最高 PSNR 24.937591，并在 `sir2_withgt`、`SIR2_postcard` 上表现较强；但 `real20` 和 `CEILNet_table2` 弱于 Improved Loss，因此六数据集综合结果不是当前第一。

### 7. Transformer

来源：`models/arch/transformer.py`、`transformer_experiments/train_transformer_errnet.py`、`results/eval_transformer/`

做法：在 ERRNet 中加入 Transformer refinement/global self-attention。

结果：6 数据集平均 PSNR 18.183065、平均 SSIM 0.733343，显著低于 Baseline。

结论：单独 Transformer 分支当前失败，不能作为候选主方法。

### 8. Transformer Cascade v2

来源：`transformer_experiments/train_transformer_cascade_v2.py`、`checkpoints/errnet_transformer_cascade_v2/`、`results/eval_transformer_cascade_v2/`、`results/eval_transformer_cascade_v2_latest/`

做法：

- 使用 `errnet_transformer_cascade_model`。
- coarse branch 默认从 `checkpoints/errnet/errnet_latest.pt` 初始化。
- 训练到 epoch 20 开启小 GAN 权重，epoch 30/40 降学习率，epoch 45 调整 synthetic/real ratio 到 0.5/0.5。

epoch45 局部评估结果：

| 数据集 | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| CEILNet_table2 | 29.103454 | 0.951134 | 0.983884 | 0.003825 |
| real20 | 22.913116 | 0.814082 | 0.877050 | 0.021574 |

latest checkpoint 全量评估结果，checkpoint 为 epoch 50 / iteration 386600：

| 数据集 | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: |
| real20 | 22.689688 | 0.812457 | 0.870988 | 0.023058 |
| CEILNet_table2 | 28.314081 | 0.947757 | 0.982776 | 0.004128 |
| sir2_withgt | 23.969842 | 0.894265 | 0.960710 | 0.004187 |
| SIR2_objects | 24.815466 | 0.901884 | 0.981850 | 0.002895 |
| SIR2_postcard | 22.195525 | 0.884959 | 0.949427 | 0.004003 |
| SIR2_wild | 25.439620 | 0.895670 | 0.938842 | 0.007071 |
| 平均 | 24.570704 | 0.889499 | 0.947432 | 0.007557 |

结论：latest/full 在 NAF Refiner 补全前拥有最高平均 SSIM，但平均 PSNR 比 Improved Loss 低 0.008，平均 LMSE 也更高。它可以作为强候选/消融结果保留，但当前不替代 NAF Refiner 作为综合主结果。另一个关键观察是，latest 在 `real20` 和 `CEILNet_table2` 均低于 epoch45 局部结果，说明这个分支需要早停或回退 checkpoint，而不是继续直接采用 latest。

### 9. NAFNet Refiner (Improved Coarse)

来源：`train_errnet_naf_refiner.py`、`test_errnet_naf_refiner.py`、`nafnet_refinement/`、`checkpoints/errnet_naf_refiner_improved/`、`results/errnet_naf_refiner_improved_best/`

做法：

- 使用 `Improved Loss` ERRNet 作为 coarse model，checkpoint 为 `checkpoints/errnet_improved_loss_v1/errnet_060_00463920.pt`。
- NAFNet refiner 读取原图和 coarse 输出，预测更细的透射层结果。
- 当前评估 checkpoint 为 `checkpoints/errnet_naf_refiner_improved/naf_refiner_best.pt`，测试参数为 `naf_coarse_kind=improved`，未启用 TTA。

完整 6 数据集评估结果：

| 数据集 | Images | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| real20 | 20 | 23.060527 | 0.817275 | 0.888185 | 0.020051 |
| CEILNet_table2 | 100 | 30.405712 | 0.960991 | 0.986774 | 0.003244 |
| sir2_withgt | 480 | 23.688231 | 0.888635 | 0.952940 | 0.004831 |
| SIR2_objects | 200 | 24.460216 | 0.893115 | 0.983386 | 0.003437 |
| SIR2_postcard | 179 | 21.507993 | 0.869789 | 0.922206 | 0.006408 |
| SIR2_wild | 101 | 26.024046 | 0.913161 | 0.947121 | 0.004794 |
| 平均 | - | 24.857788 | 0.890494 | 0.946769 | 0.007127 |

结论：NAF Refiner 是当前完整 6 数据集平均 PSNR/SSIM/LMSE 最好的方法，主要收益来自 `CEILNet_table2` 和 `SIR2_wild`。它在 `real20` 上接近 Attn Rebalanced，但在 `SIR2_objects`、`SIR2_postcard`、`sir2_withgt` 上低于对应单集最佳方法，因此更适合作为综合主结果，同时需要保留逐集结果说明其回退点。

### 10. DSRNet-S 外部模型

来源：`/home/xumx/dip_pj/DSRNet/`、`eval_errnet_processed.py`、`weights/dsrnet_s_epoch14.pt`、`results/dsrnet_s_epoch14_errnet_processed_max512/`、`results/dsrnet_s_epoch14_errnet_processed_fullres_tiled512_amp/`

做法：

- 从 <https://github.com/mingcv/DSRNet> 独立部署到 `/home/xumx/dip_pj/DSRNet`，没有修改 ERRNet 原代码。
- 使用仓库自带 `weights/dsrnet_s_epoch14.pt`，架构为 `dsrnet_s`，启用 `--hyper` 和 `--if_align`。
- 直接整图全分辨率推理在 `real20` 上 OOM；两卡拆分 + AMP 也无法稳定跑完整 `real20`。
- 因此补充了 **DSR 全量版**：不缩放输入的全分辨率滑窗评估，`max_long_edge=-1`，`tile_size=512`，`tile_overlap=64`，AMP，`split_intro_devices=0,1`。该结果按原图尺寸计算指标，但推理上下文不同于整图前向。
- 另保留此前 `max_long_edge=512` 缩放评估。这会改变输入分辨率，结果只作为缩放评估补充。

DSR 全量版结果（完整 6 数据集、全分辨率滑窗）：

| 数据集 | Images | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| real20 | 20 | 20.731167 | 0.765325 | 0.853622 | 0.018257 |
| CEILNet_table2 | 100 | 24.431802 | 0.902807 | 0.961047 | 0.007967 |
| sir2_withgt | 480 | 25.307032 | 0.911311 | 0.968997 | 0.003217 |
| SIR2_objects | 200 | 26.076155 | 0.913815 | 0.985038 | 0.002467 |
| SIR2_postcard | 179 | 24.458009 | 0.909026 | 0.959632 | 0.003017 |
| SIR2_wild | 101 | 25.288721 | 0.910400 | 0.953828 | 0.005058 |
| 平均 | - | 24.382148 | 0.885447 | 0.947027 | 0.006664 |

完整 6 数据集缩放评估结果：

| 数据集 | Images | PSNR | SSIM | NCC | LMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| real20 | 20 | 23.133136 | 0.814653 | 0.887775 | 0.019816 |
| CEILNet_table2 | 100 | 24.430280 | 0.902807 | 0.961038 | 0.007967 |
| sir2_withgt | 480 | 25.315682 | 0.911077 | 0.969079 | 0.003269 |
| SIR2_objects | 200 | 26.094166 | 0.914327 | 0.984786 | 0.002501 |
| SIR2_postcard | 179 | 24.464843 | 0.908099 | 0.959986 | 0.003072 |
| SIR2_wild | 101 | 25.282052 | 0.909921 | 0.954092 | 0.005139 |
| 平均 | - | 24.786693 | 0.893481 | 0.952793 | 0.006961 |

结论：DSRNet-S 全量版对 `sir2_withgt`、`SIR2_objects`、`SIR2_postcard`、`SIR2_wild` 很强，平均 LMSE 也低；但 `real20` 相比缩放评估明显下降，平均 PSNR 低于 NAF Refiner / Improved Loss / Baseline。缩放评估的平均 PSNR/SSIM 更高，但改变了输入分辨率。两组结果都适合作为外部模型补充，不直接替代 NAF Refiner 作为主结果。

## 当前推荐

1. 论文/报告主结果建议新增 **NAFNet Refiner (Improved Coarse)** 作为当前完整 6 数据集综合最强方法，同时保留 **Improved Loss** 作为 coarse/baseline 参照。
2. 如果允许按数据集选择策略：`CEILNet_table2` 和 `SIR2_wild` 用 `NAFNet Refiner`；`real20` 用 `Attn Rebalanced`；`sir2_withgt` 用 `Baseline + TTA`；`SIR2_objects` 用 `Prior Stage2`；`SIR2_postcard` 用 `Realistic Cascade (Ours)`，SSIM/LMSE 可参考 `Transformer Cascade v2 latest/full`。
3. DSRNet-S 可作为外部模型补充实验，尤其用于说明 DSR 全量版在 `sir2/objects/postcard/wild` 的强表现；报告中应分别标注 `DSR 全量版 full-res tiled512` 和 `max_long_edge=512` 缩放版，不要和整图前向结果直接混排。
4. 下一步如果继续推进 NAF Refiner，应优先解决 `objects/postcard/sir2` 的回退问题，例如增加按数据集/按反射强度的门控或回退到 coarse 输出的选择策略。

## 数据来源

- 汇总表：`results/metrics_summary_all.md`、`results/metrics_comparison_all.md`
- 全量 summary：`results/**/summary.txt`
- Prior 最优结果：`results/prior_stage2_summary.csv`、`results/eval_prior_stage2_best_full/*/summary.txt`
- Transformer：`results/metrics_transformer.csv`、`results/eval_transformer_cascade_v2/*/summary.txt`、`results/eval_transformer_cascade_v2_latest/*/summary.txt`
- NAF Refiner：`checkpoints/errnet_naf_refiner_improved/naf_refiner_best.pt`、`results/errnet_naf_refiner_improved_best/*/summary.txt`
- DSRNet-S：`/home/xumx/dip_pj/DSRNet/weights/dsrnet_s_epoch14.pt`、`/home/xumx/dip_pj/DSRNet/results/dsrnet_s_epoch14_errnet_processed_max512/*/summary.txt`、`/home/xumx/dip_pj/DSRNet/results/dsrnet_s_epoch14_errnet_processed_fullres_tiled512_amp/*/summary.txt`
- 训练/方法说明：`train_errnet_improved_loss.py`、`train_errnet_attn_rebalanced.py`、`train_errnet_ours.py`、`train_errnet_prior.py`、`transformer_experiments/*.py`
