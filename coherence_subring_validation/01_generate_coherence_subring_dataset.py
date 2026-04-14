"""
01_generate_coherence_subring_dataset.py

Purpose
-------
Create a synthetic dataset that extends the baseline image-first BHEX prototype in
two explicit ways:

1) provenance / coherence validation
   - a designed criticality-gap control influences ring-background overlap
   - each sample stores empirical coherence and simple bound proxies

2) subring validation
   - the signal is generated as an exponentially weighted tower of subrings
   - each sample stores per-subring truth and finite-truncation envelopes

The baseline `simulation/` workflow is intentionally left unchanged. This suite lives
in its own directory and produces its own dataset and results.
"""

from dataclasses import asdict
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation_common import (
    SUITE_ROOT,
    SuiteConfig,
    apply_gaussian_blur_fft,
    coherence,
    crescent_plasma,
    gaussian_blob,
    gaussian_ring,
    geometric_tail_bound,
    near_critical_shell,
    robust_normalize,
    save_json,
    shifted_fields,
    visibility_from_image,
    make_coordinate_grid,
)


CONFIG = SuiteConfig()
DATA_ROOT = SUITE_ROOT / "coherence_subring_dataset"
PREVIEW_ROOT = DATA_ROOT / "previews"


def render_preview(
    composite: np.ndarray,
    ring_total: np.ndarray,
    background: np.ndarray,
    subrings: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(12.8, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.15, 1.15, 1.65], wspace=0.08, hspace=0.14)

    ax_comp = fig.add_subplot(grid[:, 0])
    ax_ring = fig.add_subplot(grid[:, 1])
    ax_bg = fig.add_subplot(grid[:, 2])

    for ax, image, label in [
        (ax_comp, composite, "Composite"),
        (ax_ring, ring_total, "Aggregate ring"),
        (ax_bg, background, "Background"),
    ]:
        ax.imshow(robust_normalize(image), cmap="inferno")
        ax.set_title(label)
        ax.axis("off")

    subgrid = grid[:, 3].subgridspec(2, 2, wspace=0.06, hspace=0.08)
    for idx in range(subrings.shape[0]):
        ax = fig.add_subplot(subgrid[idx // 2, idx % 2])
        ax.imshow(robust_normalize(subrings[idx]), cmap="magma")
        ax.set_title(f"Subring {idx + 1}")
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if CONFIG.overwrite_output and DATA_ROOT.exists():
    shutil.rmtree(DATA_ROOT)

(DATA_ROOT / "tune" / "images_npy").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "tune" / "png").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "holdout" / "images_npy").mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "holdout" / "png").mkdir(parents=True, exist_ok=True)
PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(CONFIG.dataset_seed)
xx, yy, _, _ = make_coordinate_grid(CONFIG.image_size)

rows = []
total = CONFIG.n_tune + CONFIG.n_holdout

gap_schedule = np.resize(np.array(CONFIG.gap_levels, dtype=float), total)
rng.shuffle(gap_schedule)

for idx in range(total):
    split = "tune" if idx < CONFIG.n_tune else "holdout"
    sample_id = f"{split}_{idx:04d}"

    gap = float(gap_schedule[idx])
    base_radius = float(rng.uniform(*CONFIG.ring_radius_range))
    base_width = float(rng.uniform(*CONFIG.ring_width_range))
    alpha1 = float(rng.uniform(*CONFIG.alpha1_range))
    gamma = float(rng.uniform(*CONFIG.gamma_range))

    plasma_radius = float(base_radius + rng.uniform(*CONFIG.plasma_radius_offset_range))
    plasma_width = float(rng.uniform(*CONFIG.plasma_width_range))
    plasma_amplitude = float(rng.uniform(*CONFIG.plasma_amplitude_range))
    plasma_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    plasma_asymmetry = float(rng.uniform(0.24, 0.88))

    shell_amplitude = float(rng.uniform(*CONFIG.shell_amplitude_range))
    shell_width = float(rng.uniform(*CONFIG.shell_width_range))
    shell_modulation = float(rng.uniform(*CONFIG.shell_modulation_range))
    shell_leakage_weight = float(np.exp(-CONFIG.beta_gap * gap))

    scatter_sigma = float(rng.uniform(*CONFIG.scatter_sigma_range))
    noise_std = float(rng.uniform(*CONFIG.noise_std_range))

    x0 = float(rng.uniform(-CONFIG.center_jitter_pixels, CONFIG.center_jitter_pixels))
    y0 = float(rng.uniform(-CONFIG.center_jitter_pixels, CONFIG.center_jitter_pixels))
    rr, phi = shifted_fields(xx, yy, x0, y0)

    subrings = []
    unweighted_norms = []
    for n in range(CONFIG.n_subrings_true):
        amp_n = alpha1 * np.exp(-gamma * n)
        radius_n = base_radius + CONFIG.subring_spacing * n
        width_n = base_width * (CONFIG.subring_width_growth ** n)
        shape_n = gaussian_ring(rr, radius_n, width_n)
        subrings.append(amp_n * shape_n)
        unweighted_norms.append(float(np.linalg.norm(visibility_from_image(shape_n))))
    subrings = np.stack(subrings, axis=0)
    ring_total_clean = np.sum(subrings, axis=0)

    broad_crescent = plasma_amplitude * crescent_plasma(
        rr=rr,
        phi=phi,
        radius=plasma_radius,
        width=plasma_width,
        angle=plasma_angle,
        asymmetry=plasma_asymmetry,
    )
    near_shell = shell_amplitude * shell_leakage_weight * near_critical_shell(
        rr=rr,
        phi=phi,
        radius=base_radius + 1.5,
        width=shell_width,
        angle=plasma_angle + 0.3,
        modulation=shell_modulation,
    )

    blob_1 = float(rng.uniform(*CONFIG.blob_amplitude_range)) * gaussian_blob(
        xx=xx,
        yy=yy,
        x0=rng.uniform(-18.0, 18.0),
        y0=rng.uniform(-18.0, 18.0),
        sigma_x=rng.uniform(*CONFIG.blob_sigma_range),
        sigma_y=rng.uniform(*CONFIG.blob_sigma_range),
        angle=rng.uniform(0.0, np.pi),
    )
    blob_2 = float(rng.uniform(*CONFIG.blob_amplitude_range)) * gaussian_blob(
        xx=xx,
        yy=yy,
        x0=rng.uniform(-18.0, 18.0),
        y0=rng.uniform(-18.0, 18.0),
        sigma_x=rng.uniform(*CONFIG.blob_sigma_range),
        sigma_y=rng.uniform(*CONFIG.blob_sigma_range),
        angle=rng.uniform(0.0, np.pi),
    )

    background_clean = broad_crescent + near_shell + blob_1 + blob_2

    ring_total = apply_gaussian_blur_fft(ring_total_clean, scatter_sigma)
    subrings_blurred = np.stack([apply_gaussian_blur_fft(arr, scatter_sigma) for arr in subrings], axis=0)
    background = apply_gaussian_blur_fft(background_clean, scatter_sigma)
    composite = np.clip(ring_total + background + rng.normal(0.0, noise_std, size=ring_total.shape), 0.0, None)

    ring_vis = visibility_from_image(ring_total)
    background_vis = visibility_from_image(background)
    subring_vis = np.stack([visibility_from_image(arr) for arr in subrings_blurred], axis=0)

    empirical_coherence = coherence(ring_vis, background_vis)
    subring_coherences = np.array([coherence(vis, background_vis) for vis in subring_vis], dtype=float)
    subring_bound_rhs = float(
        sum(np.linalg.norm(vis.reshape(-1)) * mu for vis, mu in zip(subring_vis, subring_coherences))
        / (np.linalg.norm(ring_vis.reshape(-1)) + 1e-12)
    )

    total_ring_norm = float(np.linalg.norm(ring_vis.reshape(-1)) + 1e-12)
    tail_rows = {}
    c_max = float(max(unweighted_norms))
    for retained in (1, 2, 3):
        tail = np.sum(subring_vis[retained:], axis=0)
        tail_rows[f"tail_rel_n{retained}_true"] = float(np.linalg.norm(tail.reshape(-1)) / total_ring_norm)
        tail_rows[f"tail_rel_n{retained}_bound"] = float(
            geometric_tail_bound(alpha1=alpha1, gamma=gamma, c_max=c_max, retained_subrings=retained) / total_ring_norm
        )

    np.save(DATA_ROOT / split / "images_npy" / f"{sample_id}_composite.npy", composite.astype(np.float32))
    np.save(DATA_ROOT / split / "images_npy" / f"{sample_id}_ring_true.npy", ring_total.astype(np.float32))
    np.save(DATA_ROOT / split / "images_npy" / f"{sample_id}_background_true.npy", background.astype(np.float32))
    np.save(DATA_ROOT / split / "images_npy" / f"{sample_id}_subrings_true.npy", subrings_blurred.astype(np.float32))

    plt.imsave(DATA_ROOT / split / "png" / f"{sample_id}_composite.png", robust_normalize(composite), cmap="inferno")

    if idx < CONFIG.n_preview_examples:
        render_preview(
            composite=composite,
            ring_total=ring_total,
            background=background,
            subrings=subrings_blurred,
            save_path=PREVIEW_ROOT / f"{sample_id}_preview.png",
            title=f"{sample_id} | gap={gap:.2f}, gamma={gamma:.2f}",
        )

    rows.append(
        {
            "sample_id": sample_id,
            "split": split,
            "gap_true": gap,
            "shell_leakage_weight_true": shell_leakage_weight,
            "base_radius_true": base_radius,
            "base_width_true": base_width,
            "alpha1_true": alpha1,
            "gamma_true": gamma,
            "subring_spacing_true": CONFIG.subring_spacing,
            "plasma_radius_true": plasma_radius,
            "plasma_width_true": plasma_width,
            "plasma_amplitude_true": plasma_amplitude,
            "plasma_angle_true": plasma_angle,
            "plasma_asymmetry_true": plasma_asymmetry,
            "shell_amplitude_true": shell_amplitude,
            "shell_width_true": shell_width,
            "shell_modulation_true": shell_modulation,
            "scatter_sigma_true": scatter_sigma,
            "noise_std_true": noise_std,
            "x0_true": x0,
            "y0_true": y0,
            "empirical_coherence_true": empirical_coherence,
            "subring_bound_rhs_true": subring_bound_rhs,
            "subring_1_coherence_true": float(subring_coherences[0]),
            "subring_2_coherence_true": float(subring_coherences[1]),
            "subring_3_coherence_true": float(subring_coherences[2]),
            "subring_4_coherence_true": float(subring_coherences[3]),
            "ring_energy_true": float(np.sum(ring_total**2)),
            "background_energy_true": float(np.sum(background**2)),
            "image_size": CONFIG.image_size,
            **tail_rows,
        }
    )

metadata = pd.DataFrame(rows)
metadata.to_csv(DATA_ROOT / "metadata.csv", index=False)

save_json(asdict(CONFIG), DATA_ROOT / "dataset_config.json")

print("\nGenerated the combined coherence + subring validation dataset.")
print(f"Samples: {len(metadata)} total ({CONFIG.n_tune} tune / {CONFIG.n_holdout} holdout)")
print(f"Image size: {CONFIG.image_size} x {CONFIG.image_size}")
print(f"Gap levels: {', '.join(f'{x:.2f}' for x in CONFIG.gap_levels)}")
print(f"Dataset root: {DATA_ROOT}")
print("\nStored per-sample truth for:")
print("- aggregate ring")
print("- background")
print("- subring stack")
print("- empirical ring-background coherence")
print("- weighted subring coherence bound")
print("- finite truncation envelopes")
