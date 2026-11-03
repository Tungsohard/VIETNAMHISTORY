### modern-test (757 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| historical | 12.185 [11.886, 12.494] | 11.566 [11.178, 11.970] | 12.423 [12.074, 12.775] | 2.907 [1.306, 5.172] |
| vivid | 12.609 [12.289, 12.921] | 11.906 [11.532, 12.308] | 12.749 [12.397, 13.078] | 3.329 [1.470, 5.891] |
| ddcolor | 11.768 [11.441, 12.093] | 11.536 [11.148, 11.934] | 12.909 [12.561, 13.258] | 1.850 [1.089, 3.101] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |
|---|---|---|---|---|---|
| historical vs vivid | ΔE2000 ↓ | -0.424 | [-0.502, -0.343] | 7.5e-29 | 67% |
| historical vs vivid | AB-MAE ↓ | -0.341 | [-0.423, -0.259] | 4.1e-18 | 63% |
| historical vs vivid | Edge AB-MAE ↓ | -0.326 | [-0.411, -0.245] | 3.2e-16 | 64% |
| historical vs ddcolor | ΔE2000 ↓ | +0.418 | [+0.167, +0.672] | 0.0015 | 47% |
| historical vs ddcolor | AB-MAE ↓ | +0.029 | [-0.246, +0.304] | 0.9 | 50% |
| historical vs ddcolor | Edge AB-MAE ↓ | -0.487 | [-0.767, -0.210] | 2.1e-05 | 57% |
| vivid vs ddcolor | ΔE2000 ↓ | +0.842 | [+0.580, +1.097] | 9.4e-11 | 41% |
| vivid vs ddcolor | AB-MAE ↓ | +0.370 | [+0.098, +0.639] | 0.001 | 45% |
| vivid vs ddcolor | Edge AB-MAE ↓ | -0.160 | [-0.432, +0.108] | 0.25 | 52% |

### vicow-test (436 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| historical | 9.563 [9.230, 9.917] | 6.863 [6.627, 7.118] | 7.735 [7.408, 8.067] | 1.222 [1.141, 1.298] |
| vivid | 11.031 [10.644, 11.419] | 8.936 [8.610, 9.253] | 10.218 [9.856, 10.588] | 1.718 [1.607, 1.822] |
| ddcolor | 11.432 [10.978, 11.907] | 10.195 [9.550, 10.860] | 12.080 [11.156, 13.086] | 1.846 [1.713, 1.994] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |
|---|---|---|---|---|---|
| historical vs vivid | ΔE2000 ↓ | -1.468 | [-1.739, -1.168] | 4.6e-51 | 85% |
| historical vs vivid | AB-MAE ↓ | -2.073 | [-2.312, -1.808] | 1.4e-67 | 93% |
| historical vs vivid | Edge AB-MAE ↓ | -2.483 | [-2.751, -2.192] | 7.7e-70 | 94% |
| historical vs ddcolor | ΔE2000 ↓ | -1.869 | [-2.370, -1.339] | 3.8e-28 | 73% |
| historical vs ddcolor | AB-MAE ↓ | -3.332 | [-3.970, -2.719] | 4.5e-52 | 83% |
| historical vs ddcolor | Edge AB-MAE ↓ | -4.345 | [-5.190, -3.540] | 4.2e-58 | 88% |
| vivid vs ddcolor | ΔE2000 ↓ | -0.402 | [-0.794, +0.003] | 0.0096 | 55% |
| vivid vs ddcolor | AB-MAE ↓ | -1.259 | [-1.958, -0.603] | 4e-06 | 55% |
| vivid vs ddcolor | Edge AB-MAE ↓ | -1.862 | [-2.820, -0.976] | 1.7e-08 | 57% |
