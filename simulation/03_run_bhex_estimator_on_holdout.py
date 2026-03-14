
"""
03_run_bhex_estimator_on_holdout.py

Purpose
-------
Use the tuned estimator on held-out images and generate:
- numerical estimates
- concise stdout analysis
- evaluation plots
- ring-emphasized visualizations where the estimated photon ring is prominent
  and the plasma remains visible but toned down

No main() wrapper is used on purpose; the script runs top-to-bottom.
"""

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configurable knobs
# ---------------------------------------------------------------------
DATA_ROOT = Path("bhex_synthetic_dataset")
TUNING_ROOT = Path("bhex_model_tuning")
OUTPUT_ROOT = Path("bhex_holdout_results")

# Visual emphasis knobs.
PLASMA_BACKGROUND_STRENGTH = 0.58
COMPOSITE_BACKGROUND_STRENGTH = 0.22
MIN_RING_BOOST = 0.30
MAX_RING_BOOST = 1.00
EXPORT_PER_IMAGE_MONTAGES = False


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def robust_normalize(image):
    lo = np.percentile(image, 1.0)
    hi = np.percentile(image, 99.5)
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def downsample_mean(image, factor):
    if factor == 1:
        return image
    h, w = image.shape
    h2 = h // factor
    w2 = w // factor
    trimmed = image[: h2 * factor, : w2 * factor]
    return trimmed.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def upsample_repeat(image, factor):
    if factor == 1:
        return image
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def make_coordinate_grid(n: int):
    coords = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords)
    rr = np.sqrt(xx**2 + yy**2)
    return xx, yy, rr


def gaussian_ring(rr, radius, width):
    return np.exp(-0.5 * ((rr - radius) / max(width, 1e-6)) ** 2)


def visibility_from_image(image):
    return np.fft.fftshift(np.fft.fft2(image))


def image_from_visibility(visibility):
    return np.fft.ifft2(np.fft.ifftshift(visibility)).real


def make_frequency_radius_grid(n: int):
    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(freqs, freqs)
    rho = np.sqrt(fx**2 + fy**2)
    rho /= (rho.max() + 1e-12)
    return rho


def build_template_bank(image_size, radius_grid_work, width_work):
    _, _, rr = make_coordinate_grid(image_size)
    template_stack = []
    for radius in radius_grid_work:
        ring_img = gaussian_ring(rr, radius, width_work)
        template_stack.append(visibility_from_image(ring_img).reshape(-1))
    return np.stack(template_stack, axis=0)


def estimate_one_visibility(y_flat, template_flat, lam, h2_flat):
    weights = (lam * h2_flat) / (1.0 + lam * h2_flat)

    denom = np.sum(np.abs(template_flat) ** 2 * weights[None, :], axis=1) + 1e-12
    numer = np.sum(np.conj(template_flat) * y_flat[None, :] * weights[None, :], axis=1)
    alpha_all = np.maximum(0.0, np.real(numer / denom))

    residual_all = y_flat[None, :] - alpha_all[:, None] * template_flat
    objective_all = np.sum(np.abs(residual_all) ** 2 * weights[None, :], axis=1)
    best_idx = int(np.argmin(objective_all))

    g_best = template_flat[best_idx]
    alpha_best = float(alpha_all[best_idx])
    residual_best = residual_all[best_idx]

    q_hat_flat = residual_best / (1.0 + lam * h2_flat)
    model_vis = alpha_best * g_best + q_hat_flat
    residual_after_q = y_flat - model_vis

    weighted_corr = float(
        np.abs(np.sum(weights * np.conj(g_best) * y_flat))
        / np.sqrt((np.sum(weights * np.abs(g_best) ** 2) + 1e-12) * (np.sum(weights * np.abs(y_flat) ** 2) + 1e-12))
    )
    obs_energy = float(np.sum(np.abs(y_flat) ** 2) + 1e-12)
    residual_fraction = float(np.sum(np.abs(residual_after_q) ** 2) / obs_energy)
    confidence = float(np.clip(weighted_corr * (1.0 - residual_fraction), 0.0, 1.0))

    return {
        "best_idx": best_idx,
        "alpha_hat": alpha_best,
        "confidence": confidence,
        "q_hat_flat": q_hat_flat,
        "ring_hat_flat": alpha_best * g_best,
        "residual_hat_flat": residual_after_q,
        "residual_fraction": residual_fraction,
    }


def make_emphasized_image(composite, ring_hat, plasma_hat, confidence):
    """
    Produce a human-friendly diagnostic image:
    - plasma stays visible as a toned-down grayscale background
    - the estimated ring is overlaid brightly
    - confidence controls how aggressively the ring is emphasized
    """
    base_plasma = robust_normalize(plasma_hat)
    base_composite = robust_normalize(composite)
    ring_norm = robust_normalize(ring_hat)

    ring_boost = MIN_RING_BOOST + (MAX_RING_BOOST - MIN_RING_BOOST) * float(np.clip(confidence, 0.0, 1.0))

    background = PLASMA_BACKGROUND_STRENGTH * base_plasma + COMPOSITE_BACKGROUND_STRENGTH * base_composite
    background = np.clip(background, 0.0, 1.0)

    rgb = np.dstack([background, background, background])

    rgb[..., 0] = np.clip(rgb[..., 0] + 1.00 * ring_boost * ring_norm, 0.0, 1.0)
    rgb[..., 1] = np.clip(rgb[..., 1] + 0.70 * ring_boost * ring_norm, 0.0, 1.0)
    rgb[..., 2] = np.clip(rgb[..., 2] + 0.18 * ring_boost * ring_norm, 0.0, 1.0)

    contour = (ring_norm >= np.percentile(ring_norm, 97.0)).astype(float)
    for c in range(3):
        rgb[..., c] = np.clip(rgb[..., c] + 0.18 * contour, 0.0, 1.0)

    return rgb


def save_montage(input_image, ring_hat, plasma_hat, emphasized, residual, save_path, title):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    panels = [
        (robust_normalize(input_image), "Input"),
        (robust_normalize(ring_hat), "Estimated ring"),
        (robust_normalize(plasma_hat), "Estimated plasma"),
        (emphasized, "Ring emphasized"),
        (robust_normalize(np.abs(residual)), "Residual"),
    ]

    for ax, (arr, label) in zip(axes, panels):
        if arr.ndim == 3:
            ax.imshow(arr)
        else:
            ax.imshow(arr, cmap="inferno")
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scatter(true_values, pred_values, xlabel, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(true_values, pred_values, alpha=0.8)
    lo = min(np.min(true_values), np.min(pred_values))
    hi = max(np.max(true_values), np.max(pred_values))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_hist(values, title, xlabel, save_path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(values, bins=16, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confidence_plot(confidence, radius_abs_error, save_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(confidence, radius_abs_error, alpha=0.82)
    ax.set_xlabel("confidence")
    ax.set_ylabel("absolute radius error")
    ax.set_title("Holdout: confidence vs radius error")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Load tuned model and data
# ---------------------------------------------------------------------
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

with open(TUNING_ROOT / "tuned_model.json", "r", encoding="utf-8") as f:
    tuned_model = json.load(f)

metadata = pd.read_csv(DATA_ROOT / "metadata.csv")
holdout_meta = metadata.loc[metadata["split"] == "holdout"].copy().reset_index(drop=True)

if holdout_meta.empty:
    raise RuntimeError("No holdout images found. Run script 01 first.")

downsample_factor = int(tuned_model["downsample_factor"])
image_size_working = int(tuned_model["image_size_working"])

radius_grid_original = np.arange(
    tuned_model["radius_search_min_original"],
    tuned_model["radius_search_max_original"] + 0.5 * tuned_model["radius_step_original"],
    tuned_model["radius_step_original"],
)
radius_grid_work = radius_grid_original / downsample_factor
rho = make_frequency_radius_grid(image_size_working)
h2_flat = np.power(rho.reshape(-1), 2.0 * float(tuned_model["penalty_power"]))

templates = build_template_bank(
    image_size=image_size_working,
    radius_grid_work=radius_grid_work,
    width_work=float(tuned_model["template_width_original"]) / downsample_factor,
)

lam = float(tuned_model["lambda"])

print("\nLoaded tuned model and held-out images.")
print(f"Held-out images: {len(holdout_meta)}")
print(f"Using lambda = {lam:.4g}")
print(f"Using template width = {float(tuned_model['template_width_original']):.2f} original pixels")
print(f"Downsample factor during inference = {downsample_factor}")
print("\nOperational workflow:")
print("- Start from the image.")
print("- Downsample slightly for speed, then move to Fourier space.")
print("- Recover the ring and plasma components.")
print("- Build an emphasized visualization where the ring brightness reflects signal strength.")


# ---------------------------------------------------------------------
# Run the estimator on held-out images
# ---------------------------------------------------------------------
rows = []

for _, row in holdout_meta.iterrows():
    sample_id = row["sample_id"]
    image_full = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_composite.npy")
    image_work = downsample_mean(image_full, downsample_factor)
    y_flat = visibility_from_image(image_work).reshape(-1)

    result = estimate_one_visibility(
        y_flat=y_flat,
        template_flat=templates,
        lam=lam,
        h2_flat=h2_flat,
    )

    theta_hat_orig = float(radius_grid_work[result["best_idx"]] * downsample_factor)
    ring_hat_work = np.clip(image_from_visibility(result["ring_hat_flat"].reshape(image_size_working, image_size_working)), 0.0, None)
    plasma_hat_work = np.clip(image_from_visibility(result["q_hat_flat"].reshape(image_size_working, image_size_working)), 0.0, None)
    residual_hat_work = image_from_visibility(result["residual_hat_flat"].reshape(image_size_working, image_size_working))

    emphasized_work = make_emphasized_image(
        composite=image_work,
        ring_hat=ring_hat_work,
        plasma_hat=plasma_hat_work,
        confidence=result["confidence"],
    )

    # Upsample back to a friendlier output size for visualization.
    ring_hat_full = upsample_repeat(ring_hat_work, downsample_factor)
    plasma_hat_full = upsample_repeat(plasma_hat_work, downsample_factor)
    residual_hat_full = upsample_repeat(residual_hat_work, downsample_factor)
    emphasized_full = upsample_repeat(emphasized_work, downsample_factor)

    np.save(OUTPUT_ROOT / f"{sample_id}_ring_hat.npy", ring_hat_full.astype(np.float32))
    np.save(OUTPUT_ROOT / f"{sample_id}_plasma_hat.npy", plasma_hat_full.astype(np.float32))
    plt.imsave(OUTPUT_ROOT / f"{sample_id}_ring_emphasized.png", emphasized_full)

    if EXPORT_PER_IMAGE_MONTAGES:
        save_montage(
            input_image=image_full,
            ring_hat=ring_hat_full,
            plasma_hat=plasma_hat_full,
            emphasized=emphasized_full,
            residual=residual_hat_full,
            save_path=OUTPUT_ROOT / f"{sample_id}_montage.png",
            title=f"{sample_id} | radius_hat={theta_hat_orig:.1f} | confidence={result['confidence']:.2f}",
        )

    rows.append(
        {
            "sample_id": sample_id,
            "theta_true": float(row["ring_radius_true"]),
            "theta_hat": theta_hat_orig,
            "alpha_true": float(row["ring_amplitude_true"]),
            "alpha_hat": result["alpha_hat"],
            "radius_error": float(theta_hat_orig - row["ring_radius_true"]),
            "radius_abs_error": float(abs(theta_hat_orig - row["ring_radius_true"])),
            "confidence": float(result["confidence"]),
            "residual_fraction": float(result["residual_fraction"]),
            "noise_std_true": float(row["noise_std_true"]),
            "plasma_amplitude_true": float(row["plasma_amplitude_true"]),
            "scatter_sigma_true": float(row["scatter_sigma_true"]),
        }
    )

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_ROOT / "holdout_predictions.csv", index=False)

# ---------------------------------------------------------------------
# Aggregate plots
# ---------------------------------------------------------------------
save_scatter(
    true_values=results_df["theta_true"].values,
    pred_values=results_df["theta_hat"].values,
    xlabel="true ring radius",
    ylabel="estimated ring radius",
    title="Held-out: true vs estimated radius",
    save_path=OUTPUT_ROOT / "scatter_true_vs_estimated_radius_holdout.png",
)

save_hist(
    values=results_df["radius_error"].values,
    title="Held-out: radius error distribution",
    xlabel="estimated radius - true radius",
    save_path=OUTPUT_ROOT / "hist_radius_error_holdout.png",
)

save_confidence_plot(
    confidence=results_df["confidence"].values,
    radius_abs_error=results_df["radius_abs_error"].values,
    save_path=OUTPUT_ROOT / "confidence_vs_radius_error_holdout.png",
)

ordered = results_df.sort_values("radius_abs_error").reset_index(drop=True)
chosen = [
    ordered.iloc[0]["sample_id"],
    ordered.iloc[len(ordered) // 2]["sample_id"],
    ordered.iloc[-1]["sample_id"],
]
print("\nRepresentative held-out examples:")
for sid in chosen:
    row = results_df.loc[results_df["sample_id"] == sid].iloc[0]
    print(
        f"- {sid}: radius true={row['theta_true']:.1f}, "
        f"estimated={row['theta_hat']:.1f}, "
        f"confidence={row['confidence']:.2f}"
    )

# ---------------------------------------------------------------------
# Concise stdout analysis
# ---------------------------------------------------------------------
mean_radius_mae = float(results_df["radius_abs_error"].mean())
median_radius_mae = float(results_df["radius_abs_error"].median())
mean_confidence = float(results_df["confidence"].mean())

hi_conf_mae = float(
    results_df.loc[results_df["confidence"] >= results_df["confidence"].median(), "radius_abs_error"].mean()
)
lo_conf_mae = float(
    results_df.loc[results_df["confidence"] < results_df["confidence"].median(), "radius_abs_error"].mean()
)

print("\nHold-out summary:")
print(f"- Mean absolute radius error: {mean_radius_mae:.2f} pixels")
print(f"- Median absolute radius error: {median_radius_mae:.2f} pixels")
print(f"- Mean confidence: {mean_confidence:.2f}")
print(f"- Higher-confidence half mean error: {hi_conf_mae:.2f} pixels")
print(f"- Lower-confidence half mean error: {lo_conf_mae:.2f} pixels")
print("- Intuitive read: when the recovered ring remains cleanly separated from the smoother plasma background in Fourier space, the emphasized image tends to look sharper and the numerical radius estimate tends to be better.")
print("\nSaved outputs:")
print(f"- {OUTPUT_ROOT.resolve() / 'holdout_predictions.csv'}")
print("- Aggregate evaluation plots.")
print("- Per-image ring-emphasized PNGs and montages.")
