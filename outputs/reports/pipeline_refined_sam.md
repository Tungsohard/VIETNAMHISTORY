### modern-test (757 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| historical | 12.151 [11.850, 12.460] | 11.533 [11.145, 11.936] | 12.413 [12.063, 12.764] | 2.808 [1.271, 4.980] |
| vivid | 12.578 [12.253, 12.894] | 11.860 [11.483, 12.261] | 12.727 [12.376, 13.058] | 3.230 [1.432, 5.711] |
| ddcolor | 11.478 [11.151, 11.805] | 11.219 [10.835, 11.614] | 12.488 [12.143, 12.836] | 1.756 [1.040, 2.939] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |
|---|---|---|---|---|---|
| historical vs vivid | ΔE2000 ↓ | -0.427 | [-0.508, -0.343] | 2.9e-28 | 68% |
| historical vs vivid | AB-MAE ↓ | -0.327 | [-0.411, -0.244] | 9.7e-17 | 62% |
| historical vs vivid | Edge AB-MAE ↓ | -0.314 | [-0.398, -0.232] | 1.1e-15 | 64% |
| historical vs ddcolor | ΔE2000 ↓ | +0.673 | [+0.418, +0.928] | 3.2e-08 | 42% |
| historical vs ddcolor | AB-MAE ↓ | +0.313 | [+0.041, +0.585] | 0.0053 | 46% |
| historical vs ddcolor | Edge AB-MAE ↓ | -0.075 | [-0.354, +0.196] | 0.54 | 51% |
| vivid vs ddcolor | ΔE2000 ↓ | +1.100 | [+0.840, +1.359] | 3.9e-18 | 36% |
| vivid vs ddcolor | AB-MAE ↓ | +0.641 | [+0.373, +0.912] | 7.3e-09 | 40% |
| vivid vs ddcolor | Edge AB-MAE ↓ | +0.239 | [-0.033, +0.503] | 0.016 | 45% |

### vicow-test (436 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| historical | 9.507 [9.164, 9.871] | 6.830 [6.587, 7.092] | 7.829 [7.525, 8.135] | 1.209 [1.129, 1.286] |
| vivid | 11.027 [10.635, 11.419] | 8.926 [8.594, 9.248] | 10.367 [10.028, 10.713] | 1.721 [1.609, 1.828] |
| ddcolor | 11.176 [10.708, 11.650] | 9.919 [9.276, 10.585] | 11.667 [10.779, 12.627] | 1.805 [1.672, 1.952] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |
|---|---|---|---|---|---|
| historical vs vivid | ΔE2000 ↓ | -1.520 | [-1.791, -1.222] | 4.9e-52 | 85% |
| historical vs vivid | AB-MAE ↓ | -2.096 | [-2.340, -1.827] | 2.5e-67 | 93% |
| historical vs vivid | Edge AB-MAE ↓ | -2.537 | [-2.814, -2.240] | 8.1e-70 | 94% |
| historical vs ddcolor | ΔE2000 ↓ | -1.670 | [-2.179, -1.136] | 6.3e-23 | 70% |
| historical vs ddcolor | AB-MAE ↓ | -3.089 | [-3.732, -2.472] | 3.3e-46 | 79% |
| historical vs ddcolor | Edge AB-MAE ↓ | -3.837 | [-4.656, -3.059] | 1.3e-49 | 82% |
| vivid vs ddcolor | ΔE2000 ↓ | -0.150 | [-0.557, +0.260] | 0.49 | 53% |
| vivid vs ddcolor | AB-MAE ↓ | -0.993 | [-1.700, -0.321] | 0.003 | 51% |
| vivid vs ddcolor | Edge AB-MAE ↓ | -1.300 | [-2.226, -0.444] | 0.0018 | 50% |
