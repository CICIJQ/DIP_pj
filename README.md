# DIP_pj

这是本地 `dip_pj` 工作区整理后的 GitHub 发布版仓库。

当前仓库中包含：

- `ERRNet/` 与 `DSRNet/` 两个子项目的代码
- 已整理好的实验摘要、定性对比图和 released results
- 一小部分适合直接托管在 GitHub 上的轻量权重与 checkpoint

大体积数据集、重量级 checkpoint，以及完整原始结果输出仍然放在仓库外部。
具体需要下载哪些文件、下载后该放到哪里，请看 [ASSET_MANIFEST.md](./ASSET_MANIFEST.md)。

仓库中的主要子项目：

- `ERRNet/`：基于 ERRNet 的去反射实验、消融、RDNet 接入，以及 RDNet-refiner 代码
- `DSRNet/`：同一项目工作区中使用到的 DSRNet 训练与评测代码

当前仓库遵循的结果发布原则：

- 保留 benchmark-fair 的 RDNet refiner 结果
- 明确排除非公平的 `all_paired` / 混入测试集训练版本

具体的使用方式、结果说明和子项目背景，请分别查看各子目录下的 `README.md` 与 released results 说明文件。
