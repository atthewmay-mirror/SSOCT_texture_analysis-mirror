from __future__ import annotations

import numpy as np
from scipy import ndimage


def _resample_profile(profile: np.ndarray | None, width: int) -> np.ndarray | None:
    if profile is None:
        return None
    x_src = np.linspace(0, 1, len(profile))
    x_dst = np.linspace(0, 1, width)
    out = np.interp(x_dst, x_src, np.asarray(profile, dtype=np.float32))
    out = ndimage.gaussian_filter1d(out, 2.0)
    out -= out.min()
    out /= max(out.max(), 1e-6)
    return out


def simulate_bscan(
    height: int = 128,
    width: int = 160,
    pathology_strength: float = 0.8,
    seed: int = 0,
    fovea_x: float | None = None,
    onh_x: float | None = None,
    eye: str = 'R',
    pattern: str = 'focal',
    shadow_profile: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic OCT-like B-scan with fovea, ONH, pathology, and optional vessel-shadow profile."""
    rng = np.random.default_rng(seed)
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)[:, None]
    eye = eye.upper()
    fovea_x = width * 0.52 if fovea_x is None else float(fovea_x)
    if onh_x is None:
        onh_x = width * (0.82 if eye == 'R' else 0.18)

    pit = np.exp(-((x - fovea_x) ** 2) / (2 * (0.07 * width) ** 2))
    ilm = 24 + 7 * ((x - width / 2) / (width / 2)) ** 2 - 3.0 * pit + 1.4 * np.sin(x / 17)
    rpe = ilm + 42 + 2.5 * np.sin(x / 31) - 1.2 * pit

    img = 12 + 1.6 * rng.normal(size=(height, width))
    slab = (y > ilm[None]) & (y < rpe[None])
    img += slab * (11 + 2.0 * rng.normal(size=(height, width)))
    img += 22 * np.exp(-0.5 * ((y - ilm[None]) / 1.7) ** 2)
    img += 26 * np.exp(-0.5 * ((y - rpe[None]) / 2.0) ** 2)

    # ONH: widen / drop the layers and darken the slab locally.
    onh_profile = np.exp(-((x - onh_x) ** 2) / (2 * (0.05 * width) ** 2))
    img -= 7.5 * onh_profile[None] * slab
    rpe = rpe + 4.0 * onh_profile

    if pattern == 'focal':
        lesion_x = fovea_x + 0.10 * width * (1 if eye == 'R' else -1)
        lesion = np.exp(-((x - lesion_x) ** 2) / (2 * (0.08 * width) ** 2))
        img -= pathology_strength * 10 * lesion[None] * np.exp(-0.5 * ((y - (rpe[None] - 10)) / 4.5) ** 2)
        img += pathology_strength * 3.5 * lesion[None] * np.sin((y - ilm[None]) / 3.0) * slab
    elif pattern == 'banded':
        band = np.exp(-0.5 * ((y - (0.62 * ilm[None] + 0.38 * rpe[None])) / 4.5) ** 2)
        img += pathology_strength * 4.5 * band * np.sin(x[None] / 7.5)
        img -= pathology_strength * 3.5 * band

    shadow_profile = _resample_profile(shadow_profile, width)
    if shadow_profile is not None:
        img -= (4.5 + 4.5 * shadow_profile[None]) * shadow_profile[None] * (y > ilm[None])

    img = ndimage.gaussian_filter(img, (0.7, 0.7))
    return img.astype(np.float32), ilm.astype(np.float32), rpe.astype(np.float32)


def simulate_oct_volume(
    z: int = 12,
    height: int = 128,
    width: int = 160,
    seed: int = 0,
    eye: str = 'R',
    pattern: str = 'focal',
    shadow_profile: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Stack synthetic B-scans into a small OCT volume with across-slice drift and landmarks."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((z, height, width), dtype=np.float32)
    ilms = np.zeros((z, width), dtype=np.float32)
    rpes = np.zeros((z, width), dtype=np.float32)
    fovea_x = width * 0.52
    onh_x = width * (0.82 if eye.upper() == 'R' else 0.18)
    z_center = (z - 1) / 2

    for i in range(z):
        strength = 0.50 + 0.35 * np.exp(-((i - z_center) ** 2) / (2 * (0.18 * z) ** 2))
        lateral_shift = 2.5 * np.sin(i / 3.0)
        bscan, ilm, rpe = simulate_bscan(
            height=height,
            width=width,
            pathology_strength=float(strength),
            seed=seed + i,
            fovea_x=fovea_x + 0.3 * lateral_shift,
            onh_x=onh_x + 0.8 * lateral_shift,
            eye=eye,
            pattern=pattern,
            shadow_profile=shadow_profile,
        )
        vert_shift = int(np.round(2.0 * np.sin(i / 4.0)))
        vol[i] = ndimage.shift(bscan, (vert_shift, 0), order=1, mode='nearest')
        ilms[i] = ilm + vert_shift
        rpes[i] = rpe + vert_shift

    vol += rng.normal(0, 0.45, size=vol.shape)
    meta = {
        'fovea_xy': (float(fovea_x), float(z_center)),
        'onh_xy': (float(onh_x), float(z_center)),
        'eye': eye.upper(),
        'pattern': pattern,
    }
    return vol, ilms, rpes, meta
