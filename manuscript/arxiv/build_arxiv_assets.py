from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ARXIV_ROOT = Path(__file__).resolve().parent
FIG_ROOT = ARXIV_ROOT / "figures"

SUITE_ROOT = REPO_ROOT / "coherence_subring_validation"
DATA_ROOT = SUITE_ROOT / "coherence_subring_dataset"
RESULTS_ROOT = SUITE_ROOT / "coherence_subring_results"


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10.5,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "axes.facecolor": "#fcfcfd",
        "figure.facecolor": "white",
        "axes.edgecolor": "#c8ccd8",
        "grid.color": "#d8dde8",
        "grid.alpha": 0.35,
        "axes.labelcolor": "#1d2433",
        "text.color": "#1d2433",
        "legend.frameon": False,
    }
)


def robust_normalize(image: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(image, 1.0))
    hi = float(np.percentile(image, 99.5))
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_ROOT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


with open(RESULTS_ROOT / "benchmark_summary.json", "r", encoding="utf-8") as f:
    benchmark_summary = json.load(f)

metadata = pd.read_csv(DATA_ROOT / "metadata.csv")
holdout_meta = metadata.loc[metadata["split"] == "holdout"].copy().reset_index(drop=True)
summary_df = pd.read_csv(RESULTS_ROOT / "holdout_method_summary.csv")
pred_df = pd.read_csv(RESULTS_ROOT / "benchmark_predictions_long.csv")
holdout_pred = pred_df.loc[pred_df["split"] == "holdout"].copy().reset_index(drop=True)


def build_setup_figure() -> None:
    sample_id = benchmark_summary["representative_subring_case"]
    row = holdout_meta.loc[holdout_meta["sample_id"] == sample_id].iloc[0]

    composite = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_composite.npy")
    ring_true = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_ring_true.npy")
    subrings = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_subrings_true.npy")

    fig = plt.figure(figsize=(6.8, 2.9))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 0.88, 0.88], wspace=0.05, hspace=0.38)

    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[:, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[1, 3]),
    ]
    images = [composite, ring_true, subrings[0], subrings[1], subrings[2], subrings[3]]
    titles = [
        "Observed image",
        "Aggregate ring",
        "Subring 1",
        "Subring 2",
        "Subring 3",
        "Subring 4",
    ]
    cmaps = ["inferno", "inferno", "magma", "magma", "magma", "magma"]

    for ax, image, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(robust_normalize(image), cmap=cmap)
        ax.set_title(title, pad=4, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.03, top=0.89)
    save_figure(fig, "fig_setup")


def build_gap_and_bound_figure() -> dict[str, float]:
    grouped = (
        holdout_meta.groupby("gap_true", as_index=False)["empirical_coherence_true"]
        .median()
        .sort_values("gap_true")
    )
    gaps = grouped["gap_true"].to_numpy(dtype=float)
    medians = grouped["empirical_coherence_true"].to_numpy(dtype=float)
    coeffs = np.polyfit(gaps, np.log(np.clip(medians, 1e-9, None)), 1)
    beta_hat = -float(coeffs[0])
    scale_hat = float(np.exp(coeffs[1]))
    fit_curve = scale_hat * np.exp(-beta_hat * gaps)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))

    ax = axes[0]
    ax.scatter(
        holdout_meta["gap_true"],
        holdout_meta["empirical_coherence_true"],
        s=20,
        alpha=0.32,
        color="#51607c",
    )
    ax.plot(gaps, medians, marker="o", lw=2.2, color="#d95f02", label="Median by gap")
    ax.plot(gaps, fit_curve, "--", lw=2.0, color="#1b9e77", label=rf"${scale_hat:.3f}e^{{-{beta_hat:.3f}\Delta}}$")
    ax.set_xlabel("Designed coherence gap $\\Delta$")
    ax.set_ylabel("Empirical coherence")
    ax.set_title("Gap-controlled coherence")
    ax.legend(loc="upper right", fontsize=8.8)

    ax = axes[1]
    x = holdout_meta["empirical_coherence_true"].to_numpy(dtype=float)
    y = holdout_meta["subring_bound_rhs_true"].to_numpy(dtype=float)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.scatter(x, y, s=20, alpha=0.62, color="#264653")
    ax.plot([lo, hi], [lo, hi], "--", lw=1.5, color="#c63d2f")
    ax.set_xlabel("Aggregate coherence")
    ax.set_ylabel("Weighted subring bound")
    ax.set_title("Bound validation")

    fig.tight_layout()
    save_figure(fig, "fig_gap_bound")

    return {
        "scale_hat": scale_hat,
        "beta_hat": beta_hat,
        "coverage": float(np.mean(y >= x - 1e-9)),
    }


def build_methods_and_truncation_figure() -> dict[str, float]:
    order = [
        "Amplitude heuristic",
        "Monolithic ring model",
        "Two-subring model",
        "Four-subring model",
    ]
    plot_df = summary_df.set_index("method").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))

    ax = axes[0]
    labels = ["Heuristic", "1-subring", "2-subring", "4-subring"]
    values = plot_df["radius_mae"].to_numpy(dtype=float)
    ci_lo = plot_df["radius_mae_ci_lo"].to_numpy(dtype=float)
    ci_hi = plot_df["radius_mae_ci_hi"].to_numpy(dtype=float)
    colors = ["#8c5a2b", "#355070", "#2a9d8f", "#e76f51"]
    ax.bar(labels, values, color=colors, width=0.72)
    ax.errorbar(
        np.arange(len(labels)),
        values,
        yerr=np.vstack([values - ci_lo, ci_hi - values]),
        fmt="none",
        ecolor="#1d2433",
        capsize=3,
        lw=1.1,
    )
    ax.set_ylabel("Hold-out radius MAE (px)")
    ax.set_title("Geometry recovery")
    ax.tick_params(axis="x", rotation=12)

    retained = np.array([1.0, 2.0, 3.0])
    true_tail = np.array(
        [
            holdout_meta["tail_rel_n1_true"].mean(),
            holdout_meta["tail_rel_n2_true"].mean(),
            holdout_meta["tail_rel_n3_true"].mean(),
        ],
        dtype=float,
    )
    bound_tail = np.array(
        [
            holdout_meta["tail_rel_n1_bound"].mean(),
            holdout_meta["tail_rel_n2_bound"].mean(),
            holdout_meta["tail_rel_n3_bound"].mean(),
        ],
        dtype=float,
    )

    ax = axes[1]
    ax.plot(retained, true_tail, marker="o", lw=2.2, color="#1b9e77", label="True mean tail")
    ax.plot(retained, bound_tail, marker="s", lw=1.9, ls="--", color="#c63d2f", label="Geometric envelope")
    ax.set_xticks(retained)
    ax.set_xlabel("Retained leading subrings $N$")
    ax.set_ylabel("Residual ring norm / total")
    ax.set_title("Finite truncation")
    ax.legend(loc="upper right", fontsize=8.8)

    fig.tight_layout()
    save_figure(fig, "fig_methods_truncation")

    return {
        "tail_rel_n1_true": float(true_tail[0]),
        "tail_rel_n2_true": float(true_tail[1]),
        "tail_rel_n3_true": float(true_tail[2]),
        "tail_rel_n1_bound": float(bound_tail[0]),
        "tail_rel_n2_bound": float(bound_tail[1]),
        "tail_rel_n3_bound": float(bound_tail[2]),
    }


def build_error_coherence_figure() -> None:
    bins = np.linspace(
        float(holdout_pred["empirical_coherence_true"].min()),
        float(holdout_pred["empirical_coherence_true"].max()) + 1e-9,
        7,
    )
    palette = {
        "Amplitude heuristic": "#8c5a2b",
        "Monolithic ring model": "#355070",
        "Two-subring model": "#2a9d8f",
        "Four-subring model": "#e76f51",
    }

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for method, color in palette.items():
        group = holdout_pred.loc[holdout_pred["method"] == method]
        xs = []
        ys = []
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (group["empirical_coherence_true"] >= left) & (group["empirical_coherence_true"] < right)
            if np.any(mask):
                xs.append(0.5 * (left + right))
                ys.append(float(np.median(group.loc[mask, "radius_abs_error"])))
        ax.plot(xs, ys, marker="o", lw=2.0, color=color, label=method)

    ax.set_xlabel("Empirical coherence")
    ax.set_ylabel("Median absolute radius error (px)")
    ax.set_title("Error grows with coherence, but not equally")
    ax.legend(ncol=2, fontsize=8.2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "fig_error_coherence")


asset_summary = {}
build_setup_figure()
asset_summary.update(build_gap_and_bound_figure())
asset_summary.update(build_methods_and_truncation_figure())
build_error_coherence_figure()

with open(ARXIV_ROOT / "asset_summary.json", "w", encoding="utf-8") as f:
    json.dump(asset_summary, f, indent=2)

print("ArXiv figures written to", FIG_ROOT)
