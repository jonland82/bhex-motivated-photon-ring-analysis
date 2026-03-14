
"""
02_tune_bhex_estimator.py

Purpose
-------
Read the synthetic images produced by script 01, convert them to Fourier space,
tune the structured estimator hyperparameters, print concise analysis to stdout,
and save illustrative plots.

Model sketch
------------
Observed visibility:
    y = alpha * g_theta + q + epsilon

Estimator:
    min_{alpha, theta, q} || y - alpha g_theta - q ||^2 + lambda || H q ||^2

Interpretation:
- g_theta is a photon-ring template in Fourier space.
- q is a nuisance term (plasma / smooth contamination).
- H penalizes high-frequency content in q, encouraging q to stay smooth.

For fixed theta, the nuisance term q has a closed-form solution, which makes a
grid search over theta and lambda practical for this toy prototype.

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
OUTPUT_ROOT = Path("bhex_model_tuning")

# Speed/quality trade-off.
DOWNSAMPLE_FACTOR = 4
MAX_TUNE_IMAGES_FOR_SEARCH = 30  # use None to search on all tuning images

# Search space for hyperparameters.
LAMBDA_GRID = np.logspace(-3, 1.8, 7)
TEMPLATE_WIDTH_GRID = np.array([2.0, 3.0, 4.0])  # original-pixel widths
PENALTY_POWER = 2.0

# Search space for the ring radius parameter theta.
RADIUS_SEARCH_MIN = 34.0  # original pixels
RADIUS_SEARCH_MAX = 82.0  # original pixels
RADIUS_STEP = 1.0         # original pixels


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
    """Cheap anti-aliased downsampling via block averaging."""
    if factor == 1:
        return image
    h, w = image.shape
    h2 = h // factor
    w2 = w // factor
    trimmed = image[: h2 * factor, : w2 * factor]
    return trimmed.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


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
    """
    Precompute g_theta in Fourier space for one template width.

    Returned arrays are flattened to make the estimator vectorized and fast.
    """
    _, _, rr = make_coordinate_grid(image_size)
    template_stack = []
    for radius in radius_grid_work:
        ring_img = gaussian_ring(rr, radius, width_work)
        ring_vis = visibility_from_image(ring_img)
        template_stack.append(ring_vis.reshape(-1))
    return np.stack(template_stack, axis=0)


def estimate_one_visibility(y_flat, template_flat, lam, h2_flat):
    """
    Vectorized estimator for one image against all candidate radii at once.

    Parameters
    ----------
    y_flat : (P,)
        Flattened complex visibility of the observed image.
    template_flat : (T, P)
        Flattened complex template bank for one chosen template width.
    lam : float
        Regularization strength.
    h2_flat : (P,)
        Frequency penalty field raised to the appropriate power.

    Returns
    -------
    dict with the best index, amplitude, confidence, nuisance estimate, etc.
    """
    weights = (lam * h2_flat) / (1.0 + lam * h2_flat)  # low frequencies get soaked into q
    weighted_templates = template_flat * weights[None, :]

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
        "objective": float(objective_all[best_idx]),
        "confidence": confidence,
        "q_hat_flat": q_hat_flat,
        "ring_hat_flat": alpha_best * g_best,
        "residual_hat_flat": residual_after_q,
    }


def save_heatmap(grid_df, x_values, y_values, value_column, title, save_path):
    pivot = (
        grid_df.pivot(index="template_width", columns="lambda", values=value_column)
        .reindex(index=y_values, columns=x_values)
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{x:.3g}" for x in x_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{y:.1f}" for y in y_values])
    ax.set_xlabel("lambda")
    ax.set_ylabel("template width (original pixels)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(value_column.replace("_", " "))
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scatter(true_values, pred_values, xlabel, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(true_values, pred_values, alpha=0.75)
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
    ax.hist(values, bins=18, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_example_panel(input_image, ring_hat, plasma_hat, residual_hat, save_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    panels = [
        (input_image, "Input image"),
        (ring_hat, "Estimated ring"),
        (plasma_hat, "Estimated plasma"),
        (residual_hat, "Residual"),
    ]
    for ax, (arr, label) in zip(axes, panels):
        ax.imshow(robust_normalize(arr), cmap="inferno")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(DATA_ROOT / "metadata.csv")
tune_meta_all = metadata.loc[metadata["split"] == "tune"].copy().reset_index(drop=True)

if tune_meta_all.empty:
    raise RuntimeError("No tuning images found. Run script 01 first.")

if MAX_TUNE_IMAGES_FOR_SEARCH is not None and MAX_TUNE_IMAGES_FOR_SEARCH < len(tune_meta_all):
    tune_meta_search = tune_meta_all.sample(MAX_TUNE_IMAGES_FOR_SEARCH, random_state=7).sort_values("sample_id").reset_index(drop=True)
else:
    tune_meta_search = tune_meta_all.copy()

sample_image_full = np.load(DATA_ROOT / "tune" / "images_npy" / f"{tune_meta_all.iloc[0]['sample_id']}_composite.npy")
sample_image = downsample_mean(sample_image_full, DOWNSAMPLE_FACTOR)
image_size = int(sample_image.shape[0])

radius_grid_original = np.arange(
    RADIUS_SEARCH_MIN,
    RADIUS_SEARCH_MAX + 0.5 * RADIUS_STEP,
    RADIUS_STEP,
)
radius_grid_work = radius_grid_original / DOWNSAMPLE_FACTOR
rho = make_frequency_radius_grid(image_size)
h2_flat = np.power(rho.reshape(-1), 2.0 * PENALTY_POWER)

print("\nLoaded tuning data.")
print(f"Tuning images available: {len(tune_meta_all)}")
print(f"Tuning images used in hyperparameter search: {len(tune_meta_search)}")
print(f"Downsample factor during tuning: {DOWNSAMPLE_FACTOR}")
print(f"Working image size: {image_size} x {image_size}")
print("\nWhat the tuning loop is doing:")
print("- It starts with synthetic images.")
print("- It downsamples them a bit so the search is fast and practical.")
print("- It moves them to Fourier space.")
print("- It searches for the ring radius and amplitude while regularizing plasma as a smoother nuisance term.")

# Preload all search images in working resolution and Fourier space.
search_cache = []
for _, row in tune_meta_search.iterrows():
    sample_id = row["sample_id"]
    image = np.load(DATA_ROOT / "tune" / "images_npy" / f"{sample_id}_composite.npy")
    image_work = downsample_mean(image, DOWNSAMPLE_FACTOR)
    y_flat = visibility_from_image(image_work).reshape(-1)
    search_cache.append((row, image_work, y_flat))

# ---------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------
grid_rows = []

template_banks = {}
for width_orig in TEMPLATE_WIDTH_GRID:
    width_work = float(width_orig) / DOWNSAMPLE_FACTOR
    template_banks[float(width_orig)] = build_template_bank(
        image_size=image_size,
        radius_grid_work=radius_grid_work,
        width_work=width_work,
    )

for width_orig in TEMPLATE_WIDTH_GRID:
    template_flat = template_banks[float(width_orig)]

    for lam in LAMBDA_GRID:
        per_image_rows = []

        for row, image_work, y_flat in search_cache:
            result = estimate_one_visibility(
                y_flat=y_flat,
                template_flat=template_flat,
                lam=float(lam),
                h2_flat=h2_flat,
            )

            theta_hat_orig = float(radius_grid_work[result["best_idx"]] * DOWNSAMPLE_FACTOR)
            alpha_hat = result["alpha_hat"]

            radius_error = float(theta_hat_orig - row["ring_radius_true"])
            alpha_error = float(alpha_hat - row["ring_amplitude_true"])

            per_image_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "lambda": float(lam),
                    "template_width": float(width_orig),
                    "theta_hat": theta_hat_orig,
                    "alpha_hat": alpha_hat,
                    "radius_error": radius_error,
                    "radius_abs_error": abs(radius_error),
                    "alpha_abs_error": abs(alpha_error),
                    "confidence": result["confidence"],
                    "objective": result["objective"],
                }
            )

        per_image_df = pd.DataFrame(per_image_rows)

        score = (
            per_image_df["radius_abs_error"].mean()
            + 0.20 * per_image_df["alpha_abs_error"].mean()
            - 0.10 * per_image_df["confidence"].mean()
        )

        grid_rows.append(
            {
                "lambda": float(lam),
                "template_width": float(width_orig),
                "mean_radius_abs_error": float(per_image_df["radius_abs_error"].mean()),
                "median_radius_abs_error": float(per_image_df["radius_abs_error"].median()),
                "mean_alpha_abs_error": float(per_image_df["alpha_abs_error"].mean()),
                "mean_confidence": float(per_image_df["confidence"].mean()),
                "score": float(score),
            }
        )

grid_df = pd.DataFrame(grid_rows).sort_values("score").reset_index(drop=True)
best_row = grid_df.iloc[0]

best_lambda = float(best_row["lambda"])
best_width_orig = float(best_row["template_width"])
best_template_flat = template_banks[best_width_orig]

print("\nBest hyperparameters found:")
print(f"- lambda = {best_lambda:.4g}")
print(f"- template width = {best_width_orig:.2f} original pixels")
print(f"- mean radius absolute error = {best_row['mean_radius_abs_error']:.3f} pixels")
print(f"- mean alpha absolute error = {best_row['mean_alpha_abs_error']:.3f}")
print(f"- mean confidence = {best_row['mean_confidence']:.3f}")

# ---------------------------------------------------------------------
# Re-run best setting on all tuning images for detailed reporting
# ---------------------------------------------------------------------
prediction_rows = []
example_cache = []

for _, row in tune_meta_all.iterrows():
    sample_id = row["sample_id"]
    image_full = np.load(DATA_ROOT / "tune" / "images_npy" / f"{sample_id}_composite.npy")
    image_work = downsample_mean(image_full, DOWNSAMPLE_FACTOR)
    y_flat = visibility_from_image(image_work).reshape(-1)

    result = estimate_one_visibility(
        y_flat=y_flat,
        template_flat=best_template_flat,
        lam=best_lambda,
        h2_flat=h2_flat,
    )

    theta_hat_orig = float(radius_grid_work[result["best_idx"]] * DOWNSAMPLE_FACTOR)
    alpha_hat = result["alpha_hat"]

    ring_hat = image_from_visibility(result["ring_hat_flat"].reshape(image_size, image_size))
    plasma_hat = image_from_visibility(result["q_hat_flat"].reshape(image_size, image_size))
    residual_hat = image_from_visibility(result["residual_hat_flat"].reshape(image_size, image_size))

    prediction_rows.append(
        {
            "sample_id": sample_id,
            "theta_true": float(row["ring_radius_true"]),
            "theta_hat": theta_hat_orig,
            "alpha_true": float(row["ring_amplitude_true"]),
            "alpha_hat": alpha_hat,
            "radius_error": float(theta_hat_orig - row["ring_radius_true"]),
            "radius_abs_error": float(abs(theta_hat_orig - row["ring_radius_true"])),
            "confidence": result["confidence"],
            "noise_std_true": float(row["noise_std_true"]),
            "plasma_amplitude_true": float(row["plasma_amplitude_true"]),
            "scatter_sigma_true": float(row["scatter_sigma_true"]),
        }
    )

    example_cache.append((sample_id, image_work, ring_hat, plasma_hat, residual_hat, result))

pred_df = pd.DataFrame(prediction_rows)
pred_df.to_csv(OUTPUT_ROOT / "tuning_predictions.csv", index=False)
grid_df.to_csv(OUTPUT_ROOT / "tuning_grid_summary.csv", index=False)

# ---------------------------------------------------------------------
# Save tuned model config
# ---------------------------------------------------------------------
model_config = {
    "data_root": str(DATA_ROOT.resolve()),
    "image_size_original": int(sample_image_full.shape[0]),
    "image_size_working": image_size,
    "downsample_factor": int(DOWNSAMPLE_FACTOR),
    "radius_search_min_original": float(RADIUS_SEARCH_MIN),
    "radius_search_max_original": float(RADIUS_SEARCH_MAX),
    "radius_step_original": float(RADIUS_STEP),
    "lambda": best_lambda,
    "template_width_original": best_width_orig,
    "penalty_power": PENALTY_POWER,
}
with open(OUTPUT_ROOT / "tuned_model.json", "w", encoding="utf-8") as f:
    json.dump(model_config, f, indent=2)

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
save_heatmap(
    grid_df=grid_df,
    x_values=list(LAMBDA_GRID),
    y_values=list(TEMPLATE_WIDTH_GRID),
    value_column="mean_radius_abs_error",
    title="Tuning heatmap: mean absolute radius error",
    save_path=OUTPUT_ROOT / "heatmap_radius_mae.png",
)

save_heatmap(
    grid_df=grid_df,
    x_values=list(LAMBDA_GRID),
    y_values=list(TEMPLATE_WIDTH_GRID),
    value_column="mean_confidence",
    title="Tuning heatmap: mean confidence",
    save_path=OUTPUT_ROOT / "heatmap_confidence.png",
)

save_scatter(
    true_values=pred_df["theta_true"].values,
    pred_values=pred_df["theta_hat"].values,
    xlabel="true ring radius",
    ylabel="estimated ring radius",
    title="Best tuned setting: true vs estimated radius",
    save_path=OUTPUT_ROOT / "scatter_true_vs_estimated_radius.png",
)

save_hist(
    values=pred_df["radius_error"].values,
    title="Best tuned setting: radius error distribution",
    xlabel="estimated radius - true radius",
    save_path=OUTPUT_ROOT / "hist_radius_error.png",
)

ordered = pred_df.sort_values("radius_abs_error").reset_index(drop=True)
example_ids = [
    ordered.iloc[0]["sample_id"],
    ordered.iloc[len(ordered) // 2]["sample_id"],
    ordered.iloc[-1]["sample_id"],
]
example_lookup = {sid: (img, ring, plasma, resid, res) for sid, img, ring, plasma, resid, res in example_cache}

for sid in example_ids:
    image_work, ring_hat, plasma_hat, residual_hat, result = example_lookup[sid]
    save_example_panel(
        input_image=image_work,
        ring_hat=ring_hat,
        plasma_hat=plasma_hat,
        residual_hat=residual_hat,
        save_path=OUTPUT_ROOT / f"{sid}_tuning_example.png",
        title=f"{sid} | radius_hat={radius_grid_work[result['best_idx']] * DOWNSAMPLE_FACTOR:.1f} | confidence={result['confidence']:.2f}",
    )

# ---------------------------------------------------------------------
# Concise stdout analysis
# ---------------------------------------------------------------------
mean_radius_err = pred_df["radius_abs_error"].mean()
median_radius_err = pred_df["radius_abs_error"].median()
hi_conf = pred_df.loc[pred_df["confidence"] >= pred_df["confidence"].median(), "radius_abs_error"].mean()
lo_conf = pred_df.loc[pred_df["confidence"] < pred_df["confidence"].median(), "radius_abs_error"].mean()

print("\nConcise interpretation:")
print(f"- On tuning data, the typical radius error is {mean_radius_err:.2f} pixels (median {median_radius_err:.2f}).")
print(f"- Images with higher confidence have mean error {hi_conf:.2f} pixels.")
print(f"- Images with lower confidence have mean error {lo_conf:.2f} pixels.")
print("- Intuitively: the estimator works best when the ring remains a distinct oscillatory structure in Fourier space and the plasma stays smooth enough to be absorbed by the nuisance model.")
print("\nSaved outputs:")
print(f"- {OUTPUT_ROOT.resolve() / 'tuned_model.json'}")
print(f"- {OUTPUT_ROOT.resolve() / 'tuning_grid_summary.csv'}")
print(f"- {OUTPUT_ROOT.resolve() / 'tuning_predictions.csv'}")
print("- Heatmaps, scatter plots, error histograms, and example reconstruction panels.")
