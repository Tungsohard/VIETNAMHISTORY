### modern-test (757 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| historical | 12.188 [11.888, 12.497] | 11.568 [11.180, 11.972] | 12.426 [12.077, 12.779] | 2.908 [1.307, 5.173] |
| vivid | 12.615 [12.293, 12.926] | 11.911 [11.537, 12.312] | 12.754 [12.403, 13.083] | 3.331 [1.471, 5.892] |
| ddcolor_official | 11.764 [11.437, 12.091] | 11.539 [11.149, 11.935] | 12.916 [12.566, 13.265] | 1.852 [1.090, 3.103] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |
|---|---|---|---|---|---|
| historical vs vivid | ΔE2000 ↓ | -0.427 | [-0.505, -0.346] | 4.2e-29 | 67% |
| historical vs vivid | AB-MAE ↓ | -0.343 | [-0.426, -0.261] | 2.5e-18 | 63% |
| historical vs vivid | Edge AB-MAE ↓ | -0.328 | [-0.413, -0.246] | 2.6e-16 | 64% |
| historical vs ddcolor_official | ΔE2000 ↓ | +0.423 | [+0.173, +0.680] | 0.0013 | 46% |
| historical vs ddcolor_official | AB-MAE ↓ | +0.029 | [-0.247, +0.307] | 0.93 | 50% |
| historical vs ddcolor_official | Edge AB-MAE ↓ | -0.489 | [-0.771, -0.210] | 1.9e-05 | 57% |
| vivid vs ddcolor_official | ΔE2000 ↓ | +0.850 | [+0.588, +1.109] | 6.2e-11 | 41% |
| vivid vs ddcolor_official | AB-MAE ↓ | +0.372 | [+0.097, +0.640] | 0.001 | 45% |
| vivid vs ddcolor_official | Edge AB-MAE ↓ | -0.161 | [-0.435, +0.110] | 0.25 | 52% |

### vicow-test (436 ảnh)

| Model | ΔE2000 ↓ | AB-MAE ↓ | Edge AB-MAE ↓ | Chroma ratio |
|---|---|---|---|---|
| historical | 9.575 [9.240, 9.929] | 6.872 [6.635, 7.126] | 7.743 [7.416, 8.075] | 1.223 [1.143, 1.300] |
| vivid | 11.033 [10.648, 11.422] | 8.941 [8.614, 9.258] | 10.227 [9.864, 10.596] | 1.720 [1.609, 1.824] |
| ddcolor_official | 11.473 [11.016, 11.948] | 10.266 [9.616, 10.937] | 12.173 [11.236, 13.204] | 1.859 [1.725, 2.008] |

Chênh lệch cặp (A − B), khoảng tin cậy 95%, Wilcoxon p, tỉ lệ ảnh A tốt hơn:

| A vs B | Độ đo | Chênh lệch TB | KTC 95% | p | A tốt hơn |
|---|---|---|---|---|---|
| historical vs vivid | ΔE2000 ↓ | -1.459 | [-1.733, -1.159] | 9.2e-51 | 85% |
| historical vs vivid | AB-MAE ↓ | -2.070 | [-2.309, -1.805] | 1.1e-67 | 93% |
| historical vs vivid | Edge AB-MAE ↓ | -2.484 | [-2.751, -2.194] | 7.3e-70 | 94% |
| historical vs ddcolor_official | ΔE2000 ↓ | -1.899 | [-2.401, -1.365] | 1e-28 | 74% |
| historical vs ddcolor_official | AB-MAE ↓ | -3.394 | [-4.040, -2.781] | 3.4e-53 | 84% |
| historical vs ddcolor_official | Edge AB-MAE ↓ | -4.430 | [-5.295, -3.609] | 6.8e-59 | 87% |
| vivid vs ddcolor_official | ΔE2000 ↓ | -0.440 | [-0.835, -0.032] | 0.0045 | 56% |
| vivid vs ddcolor_official | AB-MAE ↓ | -1.325 | [-2.024, -0.661] | 9.5e-07 | 55% |
| vivid vs ddcolor_official | Edge AB-MAE ↓ | -1.946 | [-2.923, -1.042] | 2.8e-09 | 58% |
