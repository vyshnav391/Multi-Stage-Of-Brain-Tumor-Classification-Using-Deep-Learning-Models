# Feature Engineering Documentation

## Handcrafted Feature Groups

| Group | Features | Count |
|---|---|---|
| Statistical | Mean, Std, Skewness, Kurtosis, Entropy | 5 |
| Gradient/Texture | Gradient Mean, Gradient Std (Sobel) | 2 |
| Edge (Canny) | Edge Density, Mean, Std | 3 |
| Frequency (FFT) | FFT Mean, Std, High-Freq Ratio | 3 |
| Morphological | Contour Area, Perimeter, Circularity | 3 |
| Hu Moments | 7 rotation-invariant shape descriptors | 7 |
| **Total** | | **23** |

## Why Handcrafted Features?

Deep CNN features capture high-level semantic patterns, but handcrafted features provide:
- **Interpretability** — clinicians can reason about texture and edge statistics
- **Complementarity** — low-level features not always preserved in deep layers
- **Robustness** — less sensitive to domain shift / small datasets

## Preprocessing

All features are extracted from grayscale-converted, 128×128 resized images to standardize computation cost.
