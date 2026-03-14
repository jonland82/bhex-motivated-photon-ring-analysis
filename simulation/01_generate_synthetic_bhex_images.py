
"""
01_generate_synthetic_bhex_images.py

Purpose
-------
Generate synthetic black-hole-like image data for:
1) estimator tuning / calibration
2) held-out evaluation

The images are intentionally simplified and pedagogical:
- a thin photon ring
- a broader crescent-like plasma component
- optional mild scattering blur
- additive image noise

The later scripts read these synthetic images, move to Fourier space,
and run a structured estimator there.

No main() wrapper is used on purpose; the script runs top-to-bottom.
"""

from pathlib import Path
import json
import os
import shutil

# Make matplotlib cache writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configurable knobs
# ---------------------------------------------------------------------
OUTPUT_ROOT = Path("bhex_synthetic_dataset")
RANDOM_SEED = 7

IMAGE_SIZE = 256
N_TUNE = 120
N_HOLDOUT = 40

RADIUS_RANGE = (38.0, 78.0)
RING_WIDTH_RANGE = (1.4, 3.8)
RING_AMPLITUDE_RANGE = (0.70, 1.50)

PLASMA_RADIUS_OFFSET_RANGE = (-6.0, 10.0)
PLASMA_WIDTH_RANGE = (10.0, 24.0)
PLASMA_AMPLITUDE_RANGE = (0.75, 2.20)
PLASMA_CRESCENT_STRENGTH_RANGE = (0.20, 0.88)
PLASMA_ANGLE_RANGE = (0.0, 2.0 * np.pi)

SCATTER_SIGMA_RANGE = (0.0, 1.6)
NOISE_STD_RANGE = (0.004, 0.035)

CENTER_JITTER_PIXELS = 3.0
WRITE_COMPOSITE_PNGS = True
N_PREVIEW_PANELS = 12
OVERWRITE_OUTPUT = True


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def make_coordinate_grid(n: int):
    """Return Cartesian and polar coordinates on a centered pixel grid."""
    coords = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords)
    rr = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    return xx, yy, rr, phi


def shift_radius_field(xx, yy, x0, y0):
    """Radial and angular fields around a shifted center."""
    x = xx - x0
    y = yy - y0
    rr = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rr, phi


def gaussian_ring(rr, radius, width):
    """Thin circular ring."""
    return np.exp(-0.5 * ((rr - radius) / max(width, 1e-6)) ** 2)


def crescent_plasma(rr, phi, radius, width, angle, asymmetry):
    """
    Broad annulus modulated by angle, producing a bright crescent and dim side.
    The floor term keeps the emission visible on the faint side.
    """
    radial = np.exp(-0.5 * ((rr - radius) / max(width, 1e-6)) ** 2)
    angular = 0.25 + 0.75 * (1.0 + asymmetry * np.cos(phi - angle)) / (1.0 + asymmetry)
    return radial * angular


def apply_gaussian_blur_fft(image, sigma_pixels):
    """
    Mild scattering-like blur implemented in Fourier space.
    This avoids external dependencies and keeps the script self-contained.
    """
    if sigma_pixels <= 0:
        return image.copy()

    n = image.shape[0]
    freqs = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(freqs, freqs)
    rho2 = fx**2 + fy**2
    transfer = np.exp(-2.0 * (np.pi**2) * sigma_pixels**2 * rho2)

    spectrum = np.fft.fft2(image)
    blurred = np.fft.ifft2(spectrum * transfer).real
    return blurred


def robust_normalize(image):
    """Scale an image to [0, 1] using percentiles so PNG previews look nice."""
    lo = np.percentile(image, 1.0)
    hi = np.percentile(image, 99.5)
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def render_preview_panel(composite, ring, plasma, save_path: Path, title: str):
    """Save a quick three-panel preview for easy visual inspection."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, arr, label in zip(
        axes,
        [composite, ring, plasma],
        ["Composite", "Photon ring", "Plasma"],
    ):
        ax.imshow(robust_normalize(arr), cmap="inferno")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------
if OVERWRITE_OUTPUT and OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

(OUTPUT_ROOT / "tune" / "images_npy").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tune" / "png").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "holdout" / "images_npy").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "holdout" / "png").mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)
xx, yy, _, _ = make_coordinate_grid(IMAGE_SIZE)

rows = []

# ---------------------------------------------------------------------
# Data generation loop
# ---------------------------------------------------------------------
total = N_TUNE + N_HOLDOUT
for idx in range(total):
    split = "tune" if idx < N_TUNE else "holdout"
    sample_id = f"{split}_{idx:04d}"

    ring_radius = rng.uniform(*RADIUS_RANGE)
    ring_width = rng.uniform(*RING_WIDTH_RANGE)
    ring_amplitude = rng.uniform(*RING_AMPLITUDE_RANGE)

    plasma_radius = ring_radius + rng.uniform(*PLASMA_RADIUS_OFFSET_RANGE)
    plasma_width = rng.uniform(*PLASMA_WIDTH_RANGE)
    plasma_amplitude = rng.uniform(*PLASMA_AMPLITUDE_RANGE)
    plasma_angle = rng.uniform(*PLASMA_ANGLE_RANGE)
    plasma_asymmetry = rng.uniform(*PLASMA_CRESCENT_STRENGTH_RANGE)

    scatter_sigma = rng.uniform(*SCATTER_SIGMA_RANGE)
    noise_std = rng.uniform(*NOISE_STD_RANGE)

    x0 = rng.uniform(-CENTER_JITTER_PIXELS, CENTER_JITTER_PIXELS)
    y0 = rng.uniform(-CENTER_JITTER_PIXELS, CENTER_JITTER_PIXELS)

    rr, phi = shift_radius_field(xx, yy, x0, y0)

    ring_clean = ring_amplitude * gaussian_ring(rr, ring_radius, ring_width)
    plasma_clean = plasma_amplitude * crescent_plasma(
        rr=rr,
        phi=phi,
        radius=plasma_radius,
        width=plasma_width,
        angle=plasma_angle,
        asymmetry=plasma_asymmetry,
    )

    composite_clean = ring_clean + plasma_clean
    composite_scattered = apply_gaussian_blur_fft(composite_clean, scatter_sigma)

    ring_scattered = apply_gaussian_blur_fft(ring_clean, scatter_sigma)
    plasma_scattered = np.clip(composite_scattered - ring_scattered, 0.0, None)

    noisy_image = composite_scattered + rng.normal(0.0, noise_std, size=composite_scattered.shape)
    noisy_image = np.clip(noisy_image, 0.0, None)

    np.save(OUTPUT_ROOT / split / "images_npy" / f"{sample_id}_composite.npy", noisy_image.astype(np.float32))
    np.save(OUTPUT_ROOT / split / "images_npy" / f"{sample_id}_ring_true.npy", ring_scattered.astype(np.float32))
    np.save(OUTPUT_ROOT / split / "images_npy" / f"{sample_id}_plasma_true.npy", plasma_scattered.astype(np.float32))

    if WRITE_COMPOSITE_PNGS:
        plt.imsave(OUTPUT_ROOT / split / "png" / f"{sample_id}_composite.png", robust_normalize(noisy_image), cmap="inferno")

    if idx < N_PREVIEW_PANELS:
        render_preview_panel(
            noisy_image,
            ring_scattered,
            plasma_scattered,
            OUTPUT_ROOT / split / "png" / f"{sample_id}_preview.png",
            title=sample_id,
        )

    rows.append(
        {
            "sample_id": sample_id,
            "split": split,
            "ring_radius_true": ring_radius,
            "ring_width_true": ring_width,
            "ring_amplitude_true": ring_amplitude,
            "plasma_radius_true": plasma_radius,
            "plasma_width_true": plasma_width,
            "plasma_amplitude_true": plasma_amplitude,
            "plasma_angle_true": plasma_angle,
            "plasma_asymmetry_true": plasma_asymmetry,
            "scatter_sigma_true": scatter_sigma,
            "noise_std_true": noise_std,
            "x0_true": x0,
            "y0_true": y0,
            "image_size": IMAGE_SIZE,
        }
    )

metadata = pd.DataFrame(rows)
metadata.to_csv(OUTPUT_ROOT / "metadata.csv", index=False)

config_summary = {
    "random_seed": RANDOM_SEED,
    "image_size": IMAGE_SIZE,
    "n_tune": N_TUNE,
    "n_holdout": N_HOLDOUT,
    "radius_range": list(RADIUS_RANGE),
    "ring_width_range": list(RING_WIDTH_RANGE),
    "ring_amplitude_range": list(RING_AMPLITUDE_RANGE),
    "plasma_radius_offset_range": list(PLASMA_RADIUS_OFFSET_RANGE),
    "plasma_width_range": list(PLASMA_WIDTH_RANGE),
    "plasma_amplitude_range": list(PLASMA_AMPLITUDE_RANGE),
    "plasma_crescent_strength_range": list(PLASMA_CRESCENT_STRENGTH_RANGE),
    "scatter_sigma_range": list(SCATTER_SIGMA_RANGE),
    "noise_std_range": list(NOISE_STD_RANGE),
    "center_jitter_pixels": CENTER_JITTER_PIXELS,
}
with open(OUTPUT_ROOT / "dataset_config.json", "w", encoding="utf-8") as f:
    json.dump(config_summary, f, indent=2)

print("\nGenerated synthetic dataset.")
print(f"Output root: {OUTPUT_ROOT.resolve()}")
print(f"Tuning images:  {N_TUNE}")
print(f"Held-out images: {N_HOLDOUT}")
print("\nWhat is in each image?")
print("- A thin photon ring.")
print("- A broader plasma crescent.")
print("- Optional blur and additive noise.")
print("\nWhy this is useful:")
print("- The later tuning script can search over ring templates and regularization.")
print("- The held-out script can test whether the estimator generalizes.")
