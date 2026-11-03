### modern-test (757 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| historical | 12.164 [11.864, 12.475] | 11.572 [11.183, 11.976] | 12.498 [12.143, 12.857] | 2.683 [1.213, 4.743] |
| vivid | 12.568 [12.246, 12.882] | 11.864 [11.483, 12.266] | 12.770 [12.412, 13.115] | 3.105 [1.375, 5.499] |
| ddcolor | 11.597 [11.268, 11.933] | 11.321 [10.931, 11.726] | 12.735 [12.376, 13.090] | 1.659 [0.986, 2.773] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |
|---|---|---|---|---|---|
| historical vs vivid | ΔE2000 ↓ | -0.404 | [-0.485, -0.321] | 7.3e-26 | 67% |
| historical vs vivid | AB-MAE ↓ | -0.292 | [-0.373, -0.211] | 3.5e-14 | 61% |
| historical vs vivid | Edge AB-MAE ↓ | -0.272 | [-0.352, -0.195] | 5e-13 | 62% |
| historical vs ddcolor | ΔE2000 ↓ | +0.567 | [+0.314, +0.818] | 1e-06 | 43% |
| historical vs ddcolor | AB-MAE ↓ | +0.251 | [-0.017, +0.514] | 0.014 | 47% |
| historical vs ddcolor | Edge AB-MAE ↓ | -0.237 | [-0.501, +0.025] | 0.035 | 53% |
| vivid vs ddcolor | ΔE2000 ↓ | +0.970 | [+0.714, +1.228] | 8.7e-16 | 37% |
| vivid vs ddcolor | AB-MAE ↓ | +0.543 | [+0.283, +0.803] | 1.3e-07 | 41% |
| vivid vs ddcolor | Edge AB-MAE ↓ | +0.035 | [-0.231, +0.293] | 0.47 | 47% |

### vicow-test (436 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| historical | 9.436 [9.082, 9.811] | 6.760 [6.513, 7.033] | 7.721 [7.430, 8.013] | 1.180 [1.101, 1.255] |
| vivid | 10.930 [10.538, 11.325] | 8.793 [8.458, 9.116] | 10.146 [9.826, 10.470] | 1.685 [1.575, 1.789] |
| ddcolor | 11.072 [10.590, 11.558] | 9.739 [9.106, 10.391] | 11.405 [10.577, 12.304] | 1.751 [1.622, 1.892] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |
|---|---|---|---|---|---|
| historical vs vivid | ΔE2000 ↓ | -1.494 | [-1.764, -1.195] | 3.8e-51 | 85% |
| historical vs vivid | AB-MAE ↓ | -2.033 | [-2.276, -1.767] | 1.3e-66 | 93% |
| historical vs vivid | Edge AB-MAE ↓ | -2.425 | [-2.707, -2.125] | 1.3e-68 | 94% |
| historical vs ddcolor | ΔE2000 ↓ | -1.636 | [-2.143, -1.107] | 9.9e-22 | 68% |
| historical vs ddcolor | AB-MAE ↓ | -2.979 | [-3.612, -2.374] | 7.8e-44 | 77% |
| historical vs ddcolor | Edge AB-MAE ↓ | -3.685 | [-4.464, -2.946] | 1.9e-48 | 81% |
| vivid vs ddcolor | ΔE2000 ↓ | -0.142 | [-0.545, +0.263] | 0.55 | 52% |
| vivid vs ddcolor | AB-MAE ↓ | -0.946 | [-1.636, -0.292] | 0.0056 | 51% |
| vivid vs ddcolor | Edge AB-MAE ↓ | -1.259 | [-2.110, -0.461] | 0.00082 | 52% |
