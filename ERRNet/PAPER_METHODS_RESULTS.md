# 论文写作稿：十种方法的方法设计与结果分析

本文整理当前 ERRNet 项目中已经完成实现和评估的十种方案。为保证结果比较的学术严谨性，主结论采用六个测试集的完整评估均值，并以 PSNR 为主要排序指标，SSIM、NCC 和 LMSE 作为结构质量、相关性与局部误差的辅助指标。其中 PSNR、SSIM、NCC 越高越好，LMSE 越低越好。需要特别说明的是，TTA 仅覆盖 `real20`、`CEILNet_table2` 和 `sir2_withgt` 三个数据集，DSRNet-S 属于外部模型且采用滑窗或缩放推理，因此二者不直接参与 ERRNet 系内整图六数据集主排名，而作为补充实验分析。

## 1. 实验设置与总体排序

所有 ERRNet 系内方法均在相同的处理后数据目录上评估，测试集包括 `real20`、`CEILNet_table2`、`sir2_withgt`、`SIR2_objects`、`SIR2_postcard` 和 `SIR2_wild`。评估脚本统一调用 `util.index.quality_assess` 计算 PSNR、SSIM、NCC 和 LMSE。`real20` 在现有测试脚本中使用 `max_long_edge=512`，其余数据集保持原评估设置。训练数据主要来自 VOC 合成样本与 `real_train` 真实配对数据的混合，除特别说明外，合成/真实采样比例为 0.7/0.3。

完整六数据集可比结果如下：

| 排名 | 方法 | 平均 PSNR | 平均 SSIM | 平均 LMSE | 综合判断 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | NAFNet Refiner (Improved Coarse) | 24.857788 | 0.890494 | 0.007127 | 当前 ERRNet 系内综合最优，尤其提升 `CEILNet_table2` 与 `SIR2_wild` |
| 2 | Improved Loss | 24.578744 | 0.888263 | 0.007244 | 强基线，也是 NAFNet Refiner 的 coarse model |
| 3 | Transformer Cascade v2 latest/full | 24.570704 | 0.889499 | 0.007557 | 平均 SSIM 很强，但 PSNR/LMSE 略低于 Improved Loss |
| 4 | Prior Stage2 Best Full | 24.432510 | 0.883611 | 0.008146 | `SIR2_objects` 单集 PSNR 最优，但 `real20` 与 `CEILNet_table2` 弱 |
| 5 | Baseline ERRNet | 24.403693 | 0.883980 | 0.007969 | 稳定参照，SIR2 系列仍有竞争力 |
| 6 | Attn Rebalanced | 24.342035 | 0.888091 | 0.007452 | `real20` 和 `CEILNet_table2` 强，`postcard` 明显拉低均值 |
| 7 | Realistic Cascade (Ours) | 21.837659 | 0.853282 | 0.009330 | 真实域与 postcard 有亮点，但 table2 明显失败 |
| 8 | Transformer | 18.183065 | 0.733343 | 0.019371 | 单独引入 Transformer refinement 不成功 |

补充评估中，DSRNet-S full-resolution tiled 版本六集平均 PSNR/SSIM/LMSE 为 24.382148/0.885447/0.006664；`max_long_edge=512` 缩放版本为 24.786693/0.893481/0.006961。TTA 的最佳三数据集平均 PSNR 来自 Improved Loss + TTA，为 25.130323；Attn Rebalanced + TTA 则在 `CEILNet_table2` 单集达到最高 PSNR 29.135919。由于 TTA 覆盖不完整，它不能替代六数据集主排名。

十种方案的代表性结果索引如下。A 类表示 ERRNet 系内完整六数据集整图评估，可直接作为主排名，A 类内部严格按平均 PSNR 降序排列；B 类表示外部模型或非整图上下文评估；C 类表示测试时增强的局部三数据集结果。

| 完整可比优先顺序 | 方法 | 口径 | 代表 PSNR | 代表 SSIM | 代表 LMSE | 备注 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | NAFNet Refiner (Improved Coarse) | A: 六集完整 | 24.857788 | 0.890494 | 0.007127 | 当前主结果 |
| 2 | Improved Loss | A: 六集完整 | 24.578744 | 0.888263 | 0.007244 | 最强粗模型 |
| 3 | Transformer Cascade v2 latest/full | A: 六集完整 | 24.570704 | 0.889499 | 0.007557 | SSIM 强，需早停 |
| 4 | Prior Stage2 Best Full | A: 六集完整 | 24.432510 | 0.883611 | 0.008146 | `SIR2_objects` PSNR 最优 |
| 5 | Baseline ERRNet | A: 六集完整 | 24.403693 | 0.883980 | 0.007969 | 稳定参照 |
| 6 | DSRNet-S | B: 六集 full-res tiled | 24.382148 | 0.885447 | 0.006664 | 外部模型；缩放版 PSNR 24.786693 |
| 7 | Attn Rebalanced | A: 六集完整 | 24.342035 | 0.888091 | 0.007452 | `real20` PSNR/NCC 最强 |
| 8 | TTA | C: 三集局部 | 25.130323 | 0.884107 | 0.009877 | Improved Loss + TTA 三集均值；不参与六集排名 |
| 9 | Realistic Cascade (Ours) | A: 六集完整 | 21.837659 | 0.853282 | 0.009330 | `postcard` PSNR 有亮点 |
| 10 | Transformer | A: 六集完整 | 18.183065 | 0.733343 | 0.019371 | 当前负例 |

## 2. 方法设计与实现细节

### 2.1 NAFNet Refiner (Improved Coarse)

该方法是当前完整六数据集综合表现最好的方案，实现位于 `train_errnet_naf_refiner.py`、`test_errnet_naf_refiner.py`、`nafnet_refinement/` 和 `models/arch/nafnet_refiner.py`。其核心思想是将反射去除分解为稳定粗恢复和轻量残差细化两个阶段：首先冻结 Improved Loss ERRNet，得到粗透射层 \(T_0=f_{\mathrm{ERRNet}}(I)\)；随后将输入图像、粗结果和残差拼接为九通道特征：

\[
X_{\mathrm{refine}} = [I, T_0, I-T_0],
\]

再由轻量 NAFNet 预测残差修正量，最终输出为

\[
\hat{T}=T_0 + s \cdot f_{\mathrm{NAF}}(X_{\mathrm{refine}}).
\]

默认结构为 `width=32`、`middle_blk_num=4`、`enc_blk_nums=[1,1,2]`、`dec_blk_nums=[1,1,1]`，`delta_scale=1.0`。NAF 分支最后一层零初始化，因此训练初始状态严格退化为 coarse ERRNet 输出；这一设计显著降低了二阶段训练破坏已有结果的风险。训练时仅优化 NAFNet 分支，ERRNet 和 VGG hypercolumn 网络均冻结，损失函数为

\[
\mathcal{L}=1.0\mathcal{L}_{1}+0.2\mathcal{L}_{SSIM}+0.1\mathcal{L}_{grad}+0.05\mathcal{L}_{VGG}.
\]

这一设计的关键 insight 是：反射去除中的大部分低频透射结构可由 ERRNet 稳定恢复，而残留的反射边缘、局部纹理错配和过平滑问题更适合用低层图像恢复网络进行局部修正。将 \(I-T_0\) 显式输入 refiner，相当于把模型注意力限制在 coarse model 解释失败的位置；零初始化和残差加法则使训练在一个已有可用解附近搜索，而不是重新学习完整映射。NAFBlock 无显式激活函数、包含通道门控与简洁的深度卷积结构，对低层恢复任务具有较好的参数效率，因此适合作为轻量第二阶段。

最佳完整结果为平均 PSNR 24.857788、SSIM 0.890494、LMSE 0.007127。它在 `CEILNet_table2` 上达到 30.405712 PSNR，在 `SIR2_wild` 上达到 26.024046 PSNR，均为当前主实验中的单集最优。其不足也很清晰：在 `sir2_withgt`、`SIR2_objects` 和 `SIR2_postcard` 上低于部分粗模型或门控模型，说明无门控残差细化在反射较弱或背景已恢复良好的样本上可能发生过修正。

### 2.2 Improved Loss

Improved Loss 是当前最重要的强基线，实现位于 `train_errnet_improved_loss.py` 和 `models/improved_losses.py`。该方法不改变 ERRNet 主干，而是修改监督目标：原始 pixel loss 从 MSE/gradient 的组合扩展为 L1、Gradient 和 SSIM 的复合形式，并继续叠加 VGG 感知损失与额外 SSIM 项。训练脚本设置 `lambda_gan=0`，避免 GAN 在小规模数据上引入不稳定纹理。

其核心依据是，反射去除既要求像素精确，又要求保留透射层结构。MSE 倾向于平均化多解空间，容易造成过平滑；L1 对异常误差更稳健，GradientLoss 直接约束边缘和局部梯度，SSIMLoss 则从局部亮度、对比度和结构一致性上约束输出。VGG 感知损失提供中高层语义与纹理约束，但权重不宜过高，否则会牺牲 PSNR。

该方法六集平均 PSNR/SSIM/LMSE 为 24.578744/0.888263/0.007244，是 NAF Refiner 之前的综合最佳方案。其主要收益来自 `CEILNet_table2` 和 `SIR2_wild`：相对 Baseline，`CEILNet_table2` 提升 1.539700 PSNR，`SIR2_wild` 提升 0.346271 PSNR。但在 `sir2_withgt`、`SIR2_objects` 和 `SIR2_postcard` 上略低于 Baseline，表明结构化损失提高了多数场景的清晰度，却可能在某些真实反射分布上削弱原模型的稳健性。

### 2.3 Transformer Cascade v2

Transformer Cascade v2 实现位于 `models/arch/transformer.py` 和 `transformer_experiments/train_transformer_cascade_v2.py`。该方法保留 cascade 思路：先由 DRNet/ERRNet 生成 coarse 输出，再将 \([I,T_c,I-T_c]\) 输入 Transformer refinement 分支预测残差。Transformer 分支先用卷积下采样嵌入特征，然后在 token 序列上执行 multi-head self-attention，并通过 `max_attn_size=112` 限制注意力分辨率，以控制显存和计算量。

设计动机是，玻璃反射常表现为大范围重复纹理、偏移边缘和跨区域相关的 ghosting。纯卷积网络以局部感受野为主，虽然通过金字塔和残差块可以扩大上下文，但对长距离重复结构的显式建模较弱；self-attention 能让 distant patches 直接交互，从理论上更适合处理跨区域反射线索。将 Transformer 放在 cascade 的第二阶段，而不是直接替代粗恢复网络，是为了让全局建模集中在 residual correction 上，避免模型从零开始重建整幅透射层。

latest/full checkpoint 的六集平均 PSNR/SSIM/LMSE 为 24.570704/0.889499/0.007557。它在 `SIR2_objects` 和 `SIR2_postcard` 上取得很强的 SSIM/LMSE，其中 `SIR2_objects` 的 SSIM 为 0.901884，`SIR2_postcard` 的 SSIM 为 0.884959、LMSE 为 0.004003，说明全局 refinement 确实有助于结构一致性。但其平均 PSNR 比 Improved Loss 低 0.008040，且 latest 在 `real20` 和 `CEILNet_table2` 上低于 epoch45 局部结果，说明训练后期存在过拟合或损失权重漂移，实际使用时更适合早停选择，而不是默认采用 latest checkpoint。

### 2.4 Reflection Prior Branch / Prior Stage2

Prior Branch 实现位于 `models/prior_branch.py`、`train_errnet_prior.py` 和 `test_errnet_prior.py`。该方法的核心不是增强全图生成能力，而是先预测反射可能出现的位置，再在这些区域进行门控残差修正。给定输入 \(I\) 和粗输出 \(T_c\)，模型构造残差 \(R=I-T_c\)，同时用固定 Laplacian 与 Sobel 算子提取高频响应。Prior Head 的输入包含 \(I,T_c,R,|R|\)、Laplacian 和 Sobel 特征，共 14 个通道，输出单通道反射 mask \(M\)。Refiner 再以 \([I,T_c,R,\mathrm{lap},\mathrm{grad},M]\) 为输入预测残差 \(\Delta\)，最终输出

\[
\hat{T}=T_c+G(M)\odot \Delta \cdot s.
\]

其中 \(G(M)\) 可通过阈值和 gamma 校准，当前最佳配置使用差分伪标签、冻结基础 ERRNet、残差缩放 \(s=0.3\)，且不启用额外 gate threshold。Refiner 末层同样零初始化，使模型开始训练时等价于粗模型。mask 伪标签由 \(\max_c|I-T|\) 经模糊、归一化、阈值映射得到；训练损失包括输出 pixel/VGG 损失、mask 的 BCE+L1 监督、mask TV 平滑、非反射区域保护约束，以及可选 identity/sparse 约束。

该方法的 insight 是，反射去除并非所有区域都需要同等强度的修改。许多失败案例来自全图 refinement 在无反射区域引入伪影；因此显式预测反射区域并在 mask 内修正，可以提高局部恢复的可控性。固定高频算子进一步利用了反射边缘和真实物体边缘在残差域中的差异，使 prior head 有机会学习“哪里需要改”而不是只学习“怎么改”。

最佳版本 `errnet_prior_stage2_base_s030_id005_g002` 的完整六集平均 PSNR/SSIM/LMSE 为 24.432510/0.883611/0.008146。它在 `SIR2_objects` 上达到 24.937591 PSNR，是当前该数据集的 PSNR 最优；在 `sir2_withgt` 和 `SIR2_postcard` 上也强于 Improved Loss。其主要问题是 `real20` 和 `CEILNet_table2` 明显弱于 Improved Loss，说明当前 mask 伪标签和门控幅度更适合反射区域清晰、残差可分的场景，而对真实域低对比反射或 CEILNet 式合成分布适应不足。

### 2.5 Baseline ERRNet

Baseline ERRNet 是所有改进的参照，实现位于 `train_errnet.py`、`test_errnet.py`、`models/errnet_model.py` 和 `models/arch/default.py`。在 `--hyper` 设置下，输入图像会与 VGG19 多层 hypercolumn 特征拼接，送入 DRNet 主干。DRNet 由卷积、13 个残差块、SE 通道注意力与金字塔池化组成；训练时使用 VOC 合成反射与真实配对数据混合，前期关闭 GAN，epoch 20 后加入小权重 GAN，随后调整学习率与真实数据比例。

Baseline 的设计依据是经典低层恢复范式：VGG hypercolumn 提供跨尺度语义和边缘线索，DRNet 的残差块与金字塔池化负责多尺度透射层重建，SE 模块自适应强调有效通道。它没有显式 mask 或二阶段 refiner，因此优势在于训练稳定、推理简单，缺点是对细粒度反射位置和局部过修正缺乏显式控制。

六集平均 PSNR/SSIM/LMSE 为 24.403693/0.883980/0.007969。虽然整体排名第五，但在 `sir2_withgt` 和 `SIR2_objects` 上仍然稳定，且 Baseline + TTA 在 `sir2_withgt` 达到 24.079885 PSNR，是该数据集当前 PSNR/SSIM/LMSE 最优结果。这说明原始 ERRNet 的归纳偏置仍然强，后续改进必须避免牺牲它在 SIR2 类真实场景上的稳健性。

### 2.6 Attn Rebalanced

Attn Rebalanced 位于 `train_errnet_attn_rebalanced.py`，本质上是 Improved Loss 的损失权重再平衡版本。它继续使用 L1、Gradient、SSIM、VGG 的组合，但将结构相似性项设得更强，并降低 VGG 项默认影响，目标是让优化更偏向结构保真而不是感知纹理匹配。

该设计基于一个观察：反射去除的视觉质量往往不只由平均像素误差决定，局部结构错位、边缘残留和对比度变化会显著影响 SSIM 与主观观感。因此提高 SSIM 权重有望改善真实图像结构一致性，尤其是在 `real20` 这样反射与背景边界较复杂的数据上。

完整六集平均 PSNR/SSIM/LMSE 为 24.342035/0.888091/0.007452。它在 `real20` 上取得 23.107052 PSNR 和 0.892834 NCC，是当前 `real20` PSNR/NCC 最好的 ERRNet 系内结果；在 `CEILNet_table2` 的 SSIM 也达到 0.953623。问题是 `SIR2_postcard` PSNR 下降到 20.642405，显著拉低整体均值。这表明强结构约束会改善某些真实场景，但也可能在纹理复杂、反射与背景相互遮挡的 postcard 场景中造成过度平滑或错误结构保持。

### 2.7 DSRNet-S 外部模型

DSRNet-S 是从外部仓库独立部署的 ICCV 2023 方法，位于 `/localhome/xumx/dip_pj/DSRNet/`。本项目使用 `weights/dsrnet_s_epoch14.pt`，通过 `eval_errnet_processed.py` 适配 ERRNet 的六个测试集。DSRNet 的核心结构是 dual-stream reflection separation：网络同时预测透射层和反射层，并通过 MuGIBlock、VGG feature pyramid、LRM 与重建约束建模两个成分之间的协同关系。相比只预测透射层的 ERRNet，DSRNet 显式建模反射层，因此在理论上更适合分离式任务。

由于直接整图全分辨率推理在 `real20` 上 OOM，当前保留两种评估口径：一是 full-resolution tiled 版本，使用 `tile_size=512`、`tile_overlap=64`、AMP 和双卡拆分，在原图尺寸上计算指标，但 tile 前向改变了整图上下文；二是 `max_long_edge=512` 缩放版本，内存更稳定但改变了输入分辨率。

full-resolution tiled 版本六集平均 PSNR/SSIM/LMSE 为 24.382148/0.885447/0.006664；缩放版本为 24.786693/0.893481/0.006961。DSRNet-S 在 `sir2_withgt`、`SIR2_objects`、`SIR2_postcard` 和 `SIR2_wild` 上非常强，例如 full-resolution tiled 版本在 `SIR2_objects` 达到 26.076155 PSNR，在 `SIR2_postcard` 达到 24.458009 PSNR。但它在 full-resolution tiled 的 `real20` 上只有 20.731167 PSNR，明显低于 ERRNet 系内方法。该结果说明显式双流分离对 SIR2 类数据优势明显，但当前部署口径和内存限制使其不能直接替代 NAF Refiner 作为本项目主结果。

### 2.8 TTA 测试时增强

TTA 实现在 `ERRNetModel.forward_tta`、`ERRNetPriorBranchModel.forward_tta` 和 `ERRNetNAFRefinerModel.forward_tta` 中，当前主要对 Baseline、Improved Loss 和 Attn Rebalanced 做了三数据集评估。具体做法是对输入执行原图、水平翻转、垂直翻转、水平+垂直翻转四种前向，再反变换并平均输出。

其核心依据是，理想的图像恢复模型应对翻转变换保持等变性；但有限数据训练下模型往往对方向存在偏差。TTA 通过在测试阶段显式平均多个等价视角，降低方向性噪声和局部伪影，尤其适合不改变图像语义的 low-level restoration 任务。

实验显示 TTA 对 `CEILNet_table2` 和 `sir2_withgt` 有明显价值。Attn Rebalanced + TTA 在 `CEILNet_table2` 达到 29.135919 PSNR 和 0.957111 SSIM；Baseline + TTA 在 `sir2_withgt` 达到 24.079885 PSNR，是该集当前最佳。但 TTA 会使 `real20` PSNR 下降，例如 Improved Loss 从 22.980068 降到 22.427189。原因可能是真实图像中反射方向和成像畸变并不完全满足翻转等变，平均多个方向会抑制某些有效细节。因此 TTA 更适合作为按 benchmark 选择的测试策略，而不是统一默认设置。

### 2.9 Realistic Cascade (Ours)

Realistic Cascade 位于 `train_errnet_ours.py`、`models/arch/cascade.py` 和 `data/transforms.py`。该方法包含两个改动：第一，使用更真实的在线反射合成，包括空间变化的 transmission/reflection alpha、ghost shift、颜色增益、gamma-domain blending、噪声和 JPEG degradation；第二，使用 cascade refinement network，先由 coarse DRNet 生成初步结果，再将 \([I,T_c,I-T_c]\) 输入 CNN refiner 预测残差。训练流程为 40 epoch realistic synthetic warmup，再用 synthetic + real 按默认 real ratio 0.6 进行 20 epoch finetune。

该方法的 insight 是，传统合成反射往往过于简单，导致训练分布与真实玻璃反射存在 domain gap。通过引入 ghost、噪声、压缩和空间变化强度，模型有机会学习更接近真实摄影的退化过程；cascade 则让 refiner 专注于 coarse model 的局部错误。

实际结果显示该假设只在部分数据上成立。六集平均 PSNR/SSIM/LMSE 为 21.837659/0.853282/0.009330，整体排名较低；但它在 `real20` 的 SSIM/LMSE 为 0.818371/0.019677，是该集结构与局部误差表现最好的 ERRNet 系内方案之一，并在 `SIR2_postcard` 达到 22.822599 PSNR，为该数据集当前 ERRNet 系内 PSNR 最优。最大问题是 `CEILNet_table2` 只有 15.506169 PSNR，说明 realistic synthesis 与 CEILNet 测试分布严重不匹配，模型学到的真实退化先验反而破坏了合成 benchmark 的恢复能力。

### 2.10 Transformer

Transformer 分支实现位于 `models/arch/transformer.py` 和 `transformer_experiments/train_transformer_errnet.py`，目标是在 ERRNet 中引入全局 self-attention refinement。相比 Transformer Cascade v2，该版本缺少更成熟的训练调度和 checkpoint 选择，在当前实验中表现为明显失败。

从方法假设上看，Transformer 的动机是合理的：反射常具有长距离相关性，全局注意力可以补足卷积的局部性。但低层图像恢复不同于高层语义任务，像素级输出对局部连续性、边缘精确度和训练稳定性要求更高。若没有足够数据、合适的归一化与残差约束，self-attention 容易破坏局部纹理一致性，并引入过平滑或结构漂移。

其六集平均 PSNR/SSIM/LMSE 仅为 18.183065/0.733343/0.019371，显著低于 Baseline。这一结果提供了重要负例：直接添加全局注意力并不会自然提升反射去除；更有效的方式是像 Transformer Cascade v2 那样将注意力限制在二阶段残差细化中，并配合更保守的训练策略。

## 3. 结果分析与论文结论

从完整六数据集结果看，当前最有效的方向是“强粗模型 + 保守残差细化”。NAFNet Refiner 的优势来自两个层面：Improved Loss ERRNet 已经提供了比原始 Baseline 更强的初始解，而零初始化的 NAFNet residual branch 又能在不破坏 coarse 输出的前提下补偿局部细节。它在 `CEILNet_table2` 和 `SIR2_wild` 上的大幅提升说明，二阶段低层恢复网络对于细节残留和结构锐化非常有效。

第二个重要观察是，损失设计比单纯更换架构更稳健。Improved Loss 仅修改监督形式，却成为综合第二，并作为 NAFNet Refiner 的基础。相比之下，Transformer 单独分支显著失败，Transformer Cascade v2 虽然接近 Improved Loss，但需要早停和训练策略控制。这说明反射去除任务中的关键瓶颈不是简单的模型容量不足，而是如何在保真、结构、感知质量和防止过修正之间取得平衡。

第三，显式区域先验和显式双流分离都对特定数据集有效。Prior Stage2 在 `SIR2_objects` 上取得单集 PSNR 最优，说明 mask-gated refinement 能有效保护非反射区域；DSRNet-S 在 SIR2 多个子集上明显更强，说明透射/反射双流建模具有任务层面的合理性。然而二者都存在跨数据集泛化或评估口径问题：Prior 在 `real20`、`CEILNet_table2` 较弱，DSRNet-S 的 full-resolution tiled 版本在 `real20` 明显下降。因此论文中应将它们作为“有针对性的先验验证”，而不是替代主方法。

第四，数据合成的真实性并不等同于 benchmark 泛化。Realistic Cascade 在 `real20` 和 `SIR2_postcard` 上有亮点，但在 `CEILNet_table2` 崩溃，说明更复杂的真实退化会改变模型偏置；如果测试集仍包含大量传统合成或特定采集分布，过强的 realistic prior 可能造成负迁移。因此数据合成策略应与目标评估集匹配，或引入分布自适应/门控选择机制。

综合而言，本文建议将 NAFNet Refiner (Improved Coarse) 作为主方法，将 Improved Loss 作为强基线和 coarse model，将 Transformer Cascade v2、Prior Stage2、Realistic Cascade、TTA 和 DSRNet-S 作为消融与补充对比。若未来继续优化，最直接的方向是在 NAFNet Refiner 中引入 Prior Branch 的门控思想：当 refiner 的预测修正可能降低 `objects/postcard/sir2` 表现时，按反射强度、残差置信度或数据集无关的质量估计自动回退到 coarse 输出，从而保留 NAF 在 `table2/wild` 的收益并减少弱反射场景的过修正。
