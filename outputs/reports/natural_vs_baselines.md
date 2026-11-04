### modern-test (757 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| natural | 12.405 [12.104, 12.706] | 11.769 [11.391, 12.165] | 12.564 [12.238, 12.906] | 3.093 [1.341, 5.468] |
| historical | 12.185 [11.879, 12.483] | 11.566 [11.181, 11.977] | 12.423 [12.069, 12.769] | 2.907 [1.297, 5.114] |
| ddcolor | 11.768 [11.441, 12.093] | 11.536 [11.148, 11.934] | 12.909 [12.561, 13.258] | 1.850 [1.089, 3.101] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p (hoán vị theo khối, đã hiệu chỉnh Holm) | p Wilcoxon (tham khảo) | A tốt hơn |
|---|---|---|---|---|---|---|
| natural vs historical | ΔE2000 ↓ | +0.220 | [+0.122, +0.314] | 0.00015 | 3.2e-10 | 39% |
| natural vs historical | AB-MAE ↓ | +0.203 | [+0.125, +0.284] | 0.00015 | 4.5e-10 | 40% |
| natural vs historical | Edge AB-MAE ↓ | +0.142 | [+0.070, +0.212] | 0.0009 | 6.8e-06 | 44% |
| natural vs ddcolor | ΔE2000 ↓ | +0.637 | [+0.398, +0.876] | 0.00015 | 2.9e-07 | 43% |
| natural vs ddcolor | AB-MAE ↓ | +0.233 | [-0.025, +0.487] | 0.16 | 0.027 | 48% |
| natural vs ddcolor | Edge AB-MAE ↓ | -0.345 | [-0.616, -0.072] | 0.012 | 0.0069 | 54% |
| historical vs ddcolor | ΔE2000 ↓ | +0.418 | [+0.171, +0.673] | 0.0013 | 0.0015 | 47% |
| historical vs ddcolor | AB-MAE ↓ | +0.029 | [-0.232, +0.306] | 0.83 | 0.9 | 50% |
| historical vs ddcolor | Edge AB-MAE ↓ | -0.487 | [-0.769, -0.219] | 0.0011 | 2.1e-05 | 57% |

### vicow-test (436 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| natural | 10.504 [10.108, 10.890] | 8.113 [7.818, 8.407] | 9.240 [8.908, 9.588] | 1.548 [1.452, 1.642] |
| historical | 9.563 [9.237, 9.907] | 6.863 [6.621, 7.107] | 7.735 [7.408, 8.068] | 1.222 [1.145, 1.298] |
| ddcolor | 11.432 [10.972, 11.897] | 10.195 [9.588, 10.876] | 12.080 [11.135, 13.101] | 1.846 [1.703, 1.992] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p (hoán vị theo khối, đã hiệu chỉnh Holm) | p Wilcoxon (tham khảo) | A tốt hơn |
|---|---|---|---|---|---|---|
| natural vs historical | ΔE2000 ↓ | +0.941 | [+0.615, +1.215] | 0.00015 | 5.7e-33 | 19% |
| natural vs historical | AB-MAE ↓ | +1.250 | [+1.016, +1.457] | 0.00015 | 2.2e-56 | 11% |
| natural vs historical | Edge AB-MAE ↓ | +1.505 | [+1.258, +1.726] | 0.00015 | 2.9e-60 | 11% |
| natural vs ddcolor | ΔE2000 ↓ | -0.929 | [-1.303, -0.562] | 0.00015 | 7.2e-11 | 62% |
| natural vs ddcolor | AB-MAE ↓ | -2.082 | [-2.784, -1.444] | 0.00015 | 1.3e-21 | 68% |
| natural vs ddcolor | Edge AB-MAE ↓ | -2.840 | [-3.789, -2.048] | 0.00015 | 8.3e-26 | 68% |
| historical vs ddcolor | ΔE2000 ↓ | -1.869 | [-2.358, -1.342] | 0.00015 | 3.8e-28 | 73% |
| historical vs ddcolor | AB-MAE ↓ | -3.332 | [-3.992, -2.713] | 0.00015 | 4.5e-52 | 83% |
| historical vs ddcolor | Edge AB-MAE ↓ | -4.345 | [-5.162, -3.590] | 0.00015 | 4.2e-58 | 88% |
