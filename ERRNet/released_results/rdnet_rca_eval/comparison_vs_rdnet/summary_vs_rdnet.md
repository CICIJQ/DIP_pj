# RCA Conservative Parameter Run

Failed aggressive RCA attempt was deleted before this run.

## Parameters

- STRENGTH: 0.18
- MASK_GAMMA: 1.20
- MASK_SENSITIVITY: 0.85
- MASK_BLUR_RADIUS: 5
- MAX_EXTRA_DELTA: 0.08
- MASK_FLOOR: 0.0

## Average Metrics

| Dataset | Images | RDNet PSNR | RCA PSNR | Delta PSNR | RDNet SSIM | RCA SSIM | Delta SSIM | RDNet NCC | RCA NCC | Delta NCC | RDNet LMSE | RCA LMSE | Delta LMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CEILNet synthetic | 100 | 27.132373 | 27.062264 | -0.070109 | 0.937009 | 0.926677 | -0.010332 | 0.978299 | 0.977689 | -0.000610 | 0.004284 | 0.005221 | 0.000937 |
| real20 | 20 | 23.248466 | 23.008472 | -0.239994 | 0.797451 | 0.794964 | -0.002487 | 0.890640 | 0.885109 | -0.005531 | 0.014704 | 0.016188 | 0.001484 |
| SIR2 with GT | 480 | 26.224252 | 25.894747 | -0.329506 | 0.916725 | 0.910215 | -0.006510 | 0.978167 | 0.977008 | -0.001159 | 0.002898 | 0.003396 | 0.000498 |
| SIR2 Objects | 200 | 26.588745 | 26.522394 | -0.066351 | 0.916687 | 0.912499 | -0.004187 | 0.987860 | 0.987395 | -0.000465 | 0.002421 | 0.002565 | 0.000143 |
| SIR2 Postcard | 179 | 25.624858 | 24.825196 | -0.799662 | 0.914443 | 0.903043 | -0.011400 | 0.976413 | 0.973927 | -0.002486 | 0.002968 | 0.004030 | 0.001061 |
| SIR2 Wild | 101 | 26.564778 | 26.547421 | -0.017358 | 0.920846 | 0.918404 | -0.002442 | 0.962081 | 0.961900 | -0.000181 | 0.003717 | 0.003921 | 0.000204 |
| self-collected 5 | 5 | 17.525800 | 17.485618 | -0.040182 | 0.517742 | 0.515504 | -0.002237 | 0.752044 | 0.751365 | -0.000679 | 0.093545 | 0.094881 | 0.001336 |

Average PSNR wins: 0/7. Average LMSE wins: 0/7.
Higher is better for PSNR/SSIM/NCC; lower is better for LMSE.

## Per-Image Win Counts

| Dataset | Images | PSNR wins | SSIM wins | NCC wins | LMSE wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| CEILNet synthetic | 100 | 53 | 13 | 40 | 9 |
| real20 | 20 | 6 | 7 | 1 | 1 |
| SIR2 with GT | 480 | 154 | 43 | 59 | 45 |
| SIR2 Objects | 200 | 99 | 22 | 26 | 34 |
| SIR2 Postcard | 179 | 3 | 0 | 2 | 0 |
| SIR2 Wild | 101 | 52 | 21 | 31 | 11 |
| self-collected 5 | 5 | 1 | 1 | 1 | 0 |
