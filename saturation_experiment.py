from pathlib import Path
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def add_saturation_patches(
    img: np.ndarray,
    n_patches_range: tuple = (3, 8),
    intensity_values: list = [240.0, 255.0, 320.0, 500.0],
    patch_size_range: tuple = (40, 80),
    min_mean_intensity: float = 20.0,
    save_path: str | Path | None = None,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.asarray(img, dtype=np.float32)

    squeeze = img.ndim == 3 and img.shape[0] == 1
    if squeeze:
        img = img[0]

    H, W = img.shape
    modified = img.copy()
    occupied = np.zeros((H, W), dtype=bool)
    lo, hi = patch_size_range
    n_patches = rng.integers(n_patches_range[0], n_patches_range[1] + 1)

    placed = 0
    attempts = 0
    while placed < n_patches and attempts < 300:
        attempts += 1
        ph = rng.integers(lo, hi + 1)
        pw = rng.integers(lo, hi + 1)
        if H < ph or W < pw:
            continue
        y0 = rng.integers(0, H - ph + 1)
        x0 = rng.integers(0, W - pw + 1)

        if occupied[y0:y0+ph, x0:x0+pw].any():
            continue

        p_max = modified[y0:y0+ph, x0:x0+pw].max()
        p_mean = modified[y0:y0+ph, x0:x0+pw].mean()
        if p_max < 1.0 or p_mean < min_mean_intensity:
            continue

        target = rng.choice(intensity_values)
        modified[y0:y0+ph, x0:x0+pw] *= target / p_max
        occupied[y0:y0+ph, x0:x0+pw] = True
        placed += 1

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(save_path), modified)

    if squeeze:
        modified = modified[np.newaxis]

    return modified
