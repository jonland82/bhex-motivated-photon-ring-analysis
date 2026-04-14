from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import Iterable

import numpy as np


SUITE_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str((SUITE_ROOT / ".mplconfig").resolve()))


def robust_normalize(image: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(image, 1.0))
    hi = float(np.percentile(image, 99.5))
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    h, w = image.shape
    h2 = h // factor
    w2 = w // factor
    trimmed = image[: h2 * factor, : w2 * factor]
    return trimmed.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def upsample_repeat(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def make_coordinate_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords)
    rr = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    return xx, yy, rr, phi


def shifted_fields(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float) -> tuple[np.ndarray, np.ndarray]:
    x = xx - x0
    y = yy - y0
    rr = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rr, phi


def gaussian_ring(rr: np.ndarray, radius: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((rr - radius) / max(width, 1e-6)) ** 2)


def crescent_plasma(
    rr: np.ndarray,
    phi: np.ndarray,
    radius: float,
    width: float,
    angle: float,
    asymmetry: float,
) -> np.ndarray:
    radial = np.exp(-0.5 * ((rr - radius) / max(width, 1e-6)) ** 2)
    angular = 0.20 + 0.80 * (1.0 + asymmetry * np.cos(phi - angle)) / (1.0 + asymmetry)
    return radial * angular


def near_critical_shell(
    rr: np.ndarray,
    phi: np.ndarray,
    radius: float,
    width: float,
    angle: float,
    modulation: float,
) -> np.ndarray:
    radial = np.exp(-0.5 * ((rr - radius) / max(width, 1e-6)) ** 2)
    angular = 0.55 + 0.45 * (1.0 + modulation * np.cos(2.0 * (phi - angle))) / (1.0 + modulation)
    return radial * angular


def gaussian_blob(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    angle: float,
) -> np.ndarray:
    x = xx - x0
    y = yy - y0
    ca = np.cos(angle)
    sa = np.sin(angle)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return np.exp(-0.5 * ((xr / max(sigma_x, 1e-6)) ** 2 + (yr / max(sigma_y, 1e-6)) ** 2))


def apply_gaussian_blur_fft(image: np.ndarray, sigma_pixels: float) -> np.ndarray:
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


def visibility_from_image(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(image))


def image_from_visibility(visibility: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(visibility)).real


def make_frequency_radius_grid(n: int) -> np.ndarray:
    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(freqs, freqs)
    rho = np.sqrt(fx**2 + fy**2)
    rho /= (rho.max() + 1e-12)
    return rho


def radial_profile(image: np.ndarray, n_bins: int) -> np.ndarray:
    _, _, rr, _ = make_coordinate_grid(image.shape[0])
    rr_norm = rr / (rr.max() + 1e-12)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    profile = np.zeros(n_bins, dtype=float)
    for idx in range(n_bins):
        mask = (rr_norm >= bins[idx]) & (rr_norm < bins[idx + 1])
        if np.any(mask):
            profile[idx] = float(np.mean(image[mask]))
    return profile


def coherence(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.abs(np.vdot(a.reshape(-1), b.reshape(-1))))
    den = float(np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1)) + 1e-12)
    return num / den


@dataclass(frozen=True)
class SuiteConfig:
    dataset_seed: int = 23
    image_size: int = 192
    n_tune: int = 120
    n_holdout: int = 60
    downsample_factor: int = 2
    n_subrings_true: int = 4
    gap_levels: tuple[float, ...] = (0.45, 0.85, 1.25, 1.65, 2.05)
    ring_radius_range: tuple[float, float] = (34.0, 68.0)
    ring_width_range: tuple[float, float] = (1.2, 2.0)
    alpha1_range: tuple[float, float] = (0.55, 1.10)
    gamma_range: tuple[float, float] = (0.48, 1.02)
    subring_spacing: float = 1.7
    subring_width_growth: float = 1.10
    plasma_radius_offset_range: tuple[float, float] = (-6.0, 9.0)
    plasma_width_range: tuple[float, float] = (11.0, 24.0)
    plasma_amplitude_range: tuple[float, float] = (0.85, 1.85)
    shell_amplitude_range: tuple[float, float] = (0.40, 0.90)
    shell_width_range: tuple[float, float] = (3.4, 6.2)
    shell_modulation_range: tuple[float, float] = (0.15, 0.85)
    blob_amplitude_range: tuple[float, float] = (0.05, 0.20)
    blob_sigma_range: tuple[float, float] = (7.0, 18.0)
    center_jitter_pixels: float = 2.5
    scatter_sigma_range: tuple[float, float] = (0.0, 1.2)
    noise_std_range: tuple[float, float] = (0.003, 0.022)
    beta_gap: float = 1.15
    n_preview_examples: int = 12
    overwrite_output: bool = True
    tuning_search_images: int = 36
    lambda_grid: tuple[float, ...] = (0.5, 1.5, 4.0, 10.0)
    radius_search_min: float = 30.0
    radius_search_max: float = 72.0
    radius_step: float = 1.0
    single_ring_width: float = 1.7
    gamma_grid: tuple[float, ...] = (0.45, 0.60, 0.75, 0.90, 1.05)
    model_subring_counts: tuple[int, ...] = (2, 4)
    penalty_power: float = 2.0
    representative_gap_targets: tuple[float, ...] = (0.45, 1.25, 2.05)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def geometric_tail_bound(
    alpha1: float,
    gamma: float,
    c_max: float,
    retained_subrings: int,
) -> float:
    if gamma <= 0:
        return float("inf")
    return alpha1 * c_max * np.exp(-gamma * retained_subrings) / (1.0 - np.exp(-gamma))


def build_single_ring_template_bank(
    image_size: int,
    radius_grid_work: np.ndarray,
    width_work: float,
) -> np.ndarray:
    _, _, rr, _ = make_coordinate_grid(image_size)
    templates = []
    for radius in radius_grid_work:
        ring_img = gaussian_ring(rr, float(radius), float(width_work))
        templates.append(visibility_from_image(ring_img).reshape(-1))
    return np.stack(templates, axis=0)


def build_subring_template_bank(
    image_size: int,
    radius_grid_work: np.ndarray,
    gamma_grid: Iterable[float],
    n_subrings: int,
    base_width_work: float,
    spacing_work: float,
    width_growth: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    _, _, rr, _ = make_coordinate_grid(image_size)
    templates = []
    metadata = []
    for gamma in gamma_grid:
        for radius in radius_grid_work:
            image = np.zeros((image_size, image_size), dtype=float)
            for n in range(n_subrings):
                width_n = base_width_work * (width_growth ** n)
                radius_n = float(radius) + spacing_work * n
                image += np.exp(-float(gamma) * n) * gaussian_ring(rr, radius_n, width_n)
            templates.append(visibility_from_image(image).reshape(-1))
            metadata.append(
                {
                    "radius_work": float(radius),
                    "gamma": float(gamma),
                    "n_subrings": int(n_subrings),
                }
            )
    return np.stack(templates, axis=0), metadata


def estimate_structured_model(
    y_flat: np.ndarray,
    template_flat: np.ndarray,
    lam: float,
    h2_flat: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
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

    obs_energy = float(np.sum(np.abs(y_flat) ** 2) + 1e-12)
    residual_fraction = float(np.sum(np.abs(residual_after_q) ** 2) / obs_energy)
    weighted_corr = float(
        np.abs(np.sum(weights * np.conj(g_best) * y_flat))
        / np.sqrt((np.sum(weights * np.abs(g_best) ** 2) + 1e-12) * (np.sum(weights * np.abs(y_flat) ** 2) + 1e-12))
    )
    confidence = float(np.clip(weighted_corr * (1.0 - residual_fraction), 0.0, 1.0))

    return {
        "best_idx": best_idx,
        "alpha_hat": alpha_best,
        "objective": float(objective_all[best_idx]),
        "confidence": confidence,
        "q_hat_flat": q_hat_flat,
        "ring_hat_flat": alpha_best * g_best,
        "residual_hat_flat": residual_after_q,
        "residual_fraction": residual_fraction,
    }


def estimate_amplitude_heuristic(
    y_flat: np.ndarray,
    template_flat: np.ndarray,
    rho_flat: np.ndarray,
) -> dict[str, float | int]:
    amp_obs = np.abs(y_flat)
    weights = np.power(rho_flat, 1.2)
    weights = weights / (np.sum(weights) + 1e-12)

    template_amp = np.abs(template_flat)
    amp_centered = amp_obs - np.sum(weights * amp_obs)
    template_centered = template_amp - np.sum(weights[None, :] * template_amp, axis=1, keepdims=True)

    numer = np.sum(weights[None, :] * template_centered * amp_centered[None, :], axis=1)
    denom = np.sqrt(
        (np.sum(weights[None, :] * template_centered**2, axis=1) + 1e-12)
        * (np.sum(weights * amp_centered**2) + 1e-12)
    )
    score = numer / denom
    best_idx = int(np.argmax(score))
    return {
        "best_idx": best_idx,
        "score": float(score[best_idx]),
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 400) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    lo, hi = np.percentile(means, [5.0, 95.0])
    return float(lo), float(hi)
