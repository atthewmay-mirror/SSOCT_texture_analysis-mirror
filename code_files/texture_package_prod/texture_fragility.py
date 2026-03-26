# Reviewed

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage, stats


def default_fragility_transforms() -> dict[str, callable]:
    return {
        'noise_light': lambda x, rng: x + rng.normal(0, 0.02 * np.nanstd(x), x.shape),
        'blur_sigma1': lambda x, rng: ndimage.gaussian_filter(x, 1.0),
        'gain_shift': lambda x, rng: 1.08 * x + 0.03 * np.nanstd(x),
        'gamma_09': lambda x, rng: np.sign(x) * (np.abs(x) ** 0.9),
        'translate_2px': lambda x, rng: ndimage.shift(x, (2, 2), order=1, mode='nearest'),
    }


def _compare_maps(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return {'pearson_r': np.nan, 'spearman_r': np.nan, 'mae': np.nan, 'r2_like': np.nan}
    av = a[mask].ravel()
    bv = b[mask].ravel()
    pearson = stats.pearsonr(av, bv).statistic
    spearman = stats.spearmanr(av, bv).statistic
    mae = np.mean(np.abs(av - bv))
    sst = np.sum((av - av.mean()) ** 2)
    r2_like = 1 - np.sum((av - bv) ** 2) / max(sst, 1e-8)
    return {'pearson_r': float(pearson), 'spearman_r': float(spearman), 'mae': float(mae), 'r2_like': float(r2_like)}


def evaluate_feature_fragility(
    image: np.ndarray,
    compute_fn: callable,
    transforms: dict[str, callable] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare each feature map against the baseline after a set of small perturbations."""
    transforms = default_fragility_transforms() if transforms is None else transforms
    rng = np.random.default_rng(seed)
    baseline, _ = compute_fn(image)
    rows = []
    for name, tfm in transforms.items():
        changed = tfm(image.astype(np.float32), rng)
        maps, _ = compute_fn(changed)
        for feat in sorted(set(baseline) & set(maps)):
            rows.append({'transform': name, 'feature': feat, **_compare_maps(baseline[feat], maps[feat])})
    return pd.DataFrame(rows)
