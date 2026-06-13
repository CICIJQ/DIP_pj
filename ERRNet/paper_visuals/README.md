# Qualitative Figure Selection

This folder contains the selected qualitative examples for the paper.

## Folder layout

- `panels/fig1_main_naf_vs_coarse.png`
  - Main comparison figure.
  - Rows: CEILNet, SIR2 Wild, real20, SIR2 Objects.
  - Columns: Input, ERRNet, Improved Loss, Attn Rebalanced, NAFNet Refiner, GT.
  - Purpose: show that the final NAFNet Refiner is not only the best average method, but also produces visibly stronger restoration than the original ERRNet baseline and the strong coarse-model variants on carefully selected examples.

- `panels/fig2_specialized_methods.png`
  - Dataset-specialized comparison figure.
  - Rows: SIR2 Objects / Prior, SIR2 with GT / TTA, SIR2 Postcard / Realistic Cascade.
  - Columns: Input, Reference, Specialized Method, GT.
  - Purpose: show why some non-best-average methods are still important: they capture dataset-specific strengths.

- `panels/fig3_reflection_pairs_tight_best_vs_errnet.png`
  - Tight-pair comparison figure.
  - Rows: pair1--pair4 from `reflection_pairs_tight`.
  - Columns: Input, original ERRNet, Realistic Cascade, GT.
  - Purpose: show the best average method on the tight-pair set against the original ERRNet output. The selected rows are the cases where Realistic Cascade improves over ERRNet by per-image PSNR.

- `panels/fig3_reflection_pairs_tight_horizontal_nogt.png`
  - Horizontal tight-pair comparison figure.
  - Columns: pair1--pair4 from `reflection_pairs_tight`.
  - Rows: Input, original ERRNet, Realistic Cascade.
  - Purpose: show the best average method on the tight-pair set against the original ERRNet output without displaying ground-truth images. This is the recommended layout for the main paper.

- `panels/reflection_pairs_tight_best_vs_errnet.png`
  - Full tight-pair comparison figure.
  - Rows: all five paired images from `reflection_pairs_tight`.
  - Columns: Input, original ERRNet, Realistic Cascade, GT.
  - Purpose: retain the complete evaluated set for appendix or internal checking.

- `selected_images/`
  - Per-case copied source images and method outputs.
  - Each case folder contains normalized filenames such as `input.png`, `gt.png`, `naf.png`, `improved.png`, `attn.png`, `prior.png`, etc.

- `selection_manifest.csv`
  - Records the dataset, image id, method, copied path, and PSNR used in the generated panels.

## Selected cases

### Main NAFNet Refiner comparison

All main-panel examples satisfy the same quantitative constraint: the original ERRNet output is worse than Improved Loss, Attn Rebalanced, and NAFNet Refiner, while NAFNet Refiner is the best among the four restored outputs.

1. `CEILNet_table2 / 2010_005293`
   - Selected because NAFNet Refiner has a clear per-image PSNR gain over both strong coarse-model variants.

2. `SIR2_wild / 54-m`
   - Selected because NAFNet Refiner strongly improves over the coarse-model variants on a real wild-scene example.

3. `real20 / 107`
   - Selected as a real-image case where NAFNet Refiner is slightly better than Improved Loss and Attn Rebalanced.

4. `SIR2_objects / 18-Focus-19-m`
   - Selected to show a more visible object-scene advantage of NAFNet Refiner over Improved Loss and Attn Rebalanced.

### Dataset-specialized methods

1. `SIR2_objects / 11-Focus-22-m`
   - Specialized method: Reflection Prior Branch.
   - Reference: Baseline ERRNet.
   - Rationale: the prior branch is useful when reflection correction should be spatially localized.

2. `sir2_withgt / hi-5-m-27`
   - Specialized method: Baseline + TTA.
   - Reference: Baseline ERRNet.
   - Rationale: TTA reduces inference variance on a subset where the baseline is already stable.

3. `SIR2_postcard / ea-10-m-3`
   - Specialized method: Realistic Cascade.
   - Reference: NAFNet Refiner.
   - Rationale: realistic synthesis and cascade refinement help on postcard-like scenes where the synthetic-to-real gap is more pronounced.

## Recommended LaTeX layout

Use the two panels as separate figures:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{ERRNet/paper_visuals/panels/fig1_main_naf_vs_coarse.png}
  \caption{Qualitative comparison between the proposed NAFNet Refiner, the original ERRNet baseline, and the strong coarse-model variants. The selected examples cover synthetic, wild, real, and object-scene data; in each row, ERRNet is worse than the three improved variants and NAFNet Refiner gives the best restored output.}
  \label{fig:qual_main}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.92\textwidth]{ERRNet/paper_visuals/panels/fig2_specialized_methods.png}
  \caption{Qualitative examples of dataset-specialized methods. The prior branch, test-time augmentation, and realistic cascade each address a different failure mode.}
  \label{fig:qual_specialized}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{ERRNet/paper_visuals/panels/fig3_reflection_pairs_tight_horizontal_nogt.png}
  \caption{Qualitative comparison on the tight reflection-pair set. Realistic Cascade is the best average method on this set and is shown against the original ERRNet output and the input image. PSNR values are reported below the restored outputs.}
  \label{fig:tight_pairs}
\end{figure*}
```

If the paper has limited space, keep `fig1_main_naf_vs_coarse.png` in the main paper and move `fig2_specialized_methods.png` to the appendix or supplementary section.
