"""
02_run_coherence_subring_benchmark.py

Purpose
-------
Benchmark several inference approaches on the combined provenance/coherence + subring
validation dataset:

- direct amplitude heuristic
- monolithic single-ring structured model
- two-subring structured model
- four-subring structured model

The script tunes the regularization strength on the tune split, evaluates on held-out
images, saves publication-style figures, and exports machine-readable summaries for the
HTML report.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation_common import (
    SUITE_ROOT,
    bootstrap_ci,
    build_single_ring_template_bank,
    build_subring_template_bank,
    downsample_mean,
    estimate_amplitude_heuristic,
    estimate_structured_model,
    image_from_visibility,
    make_frequency_radius_grid,
    robust_normalize,
    save_json,
    visibility_from_image,
)


DATA_ROOT = SUITE_ROOT / "coherence_subring_dataset"
RESULTS_ROOT = SUITE_ROOT / "coherence_subring_results"
FIG_ROOT = RESULTS_ROOT / "figures"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbfd",
        "axes.edgecolor": "#dadbe8",
        "axes.labelcolor": "#182033",
        "text.color": "#182033",
        "axes.titleweight": "bold",
        "font.size": 10.5,
        "axes.titlesize": 13.0,
        "axes.labelsize": 10.5,
        "legend.frameon": False,
    }
)


with open(DATA_ROOT / "dataset_config.json", "r", encoding="utf-8") as f:
    DATASET_CONFIG = json.load(f)

metadata = pd.read_csv(DATA_ROOT / "metadata.csv")
tune_meta = metadata.loc[metadata["split"] == "tune"].copy().reset_index(drop=True)
holdout_meta = metadata.loc[metadata["split"] == "holdout"].copy().reset_index(drop=True)

if tune_meta.empty or holdout_meta.empty:
    raise RuntimeError("Dataset missing tune or holdout split. Run script 01 first.")

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

downsample_factor = int(DATASET_CONFIG["downsample_factor"])
single_ring_width_orig = float(DATASET_CONFIG["single_ring_width"])
radius_grid_original = np.arange(
    float(DATASET_CONFIG["radius_search_min"]),
    float(DATASET_CONFIG["radius_search_max"]) + 0.5 * float(DATASET_CONFIG["radius_step"]),
    float(DATASET_CONFIG["radius_step"]),
)
radius_grid_work = radius_grid_original / downsample_factor
gamma_grid = np.array(DATASET_CONFIG["gamma_grid"], dtype=float)
lambda_grid = np.array(DATASET_CONFIG["lambda_grid"], dtype=float)
model_subring_counts = tuple(int(x) for x in DATASET_CONFIG["model_subring_counts"])
subring_spacing_work = float(DATASET_CONFIG["subring_spacing"]) / downsample_factor
subring_width_growth = float(DATASET_CONFIG["subring_width_growth"])

sample_image = np.load(DATA_ROOT / "tune" / "images_npy" / f"{tune_meta.iloc[0]['sample_id']}_composite.npy")
image_size_work = downsample_mean(sample_image, downsample_factor).shape[0]
rho = make_frequency_radius_grid(image_size_work)
h2_flat = np.power(rho.reshape(-1), 2.0 * float(DATASET_CONFIG.get("penalty_power", 2.0)))

single_ring_templates = build_single_ring_template_bank(
    image_size=image_size_work,
    radius_grid_work=radius_grid_work,
    width_work=single_ring_width_orig / downsample_factor,
)
subring_banks = {}
subring_meta = {}
for n_subrings in model_subring_counts:
    bank, meta_rows = build_subring_template_bank(
        image_size=image_size_work,
        radius_grid_work=radius_grid_work,
        gamma_grid=gamma_grid,
        n_subrings=n_subrings,
        base_width_work=single_ring_width_orig / downsample_factor,
        spacing_work=subring_spacing_work,
        width_growth=subring_width_growth,
    )
    subring_banks[n_subrings] = bank
    subring_meta[n_subrings] = meta_rows


def load_cache(frame: pd.DataFrame) -> list[dict]:
    cache = []
    for _, row in frame.iterrows():
        sample_id = row["sample_id"]
        composite_full = np.load(DATA_ROOT / row["split"] / "images_npy" / f"{sample_id}_composite.npy")
        ring_full = np.load(DATA_ROOT / row["split"] / "images_npy" / f"{sample_id}_ring_true.npy")
        background_full = np.load(DATA_ROOT / row["split"] / "images_npy" / f"{sample_id}_background_true.npy")
        subrings_full = np.load(DATA_ROOT / row["split"] / "images_npy" / f"{sample_id}_subrings_true.npy")

        composite_work = downsample_mean(composite_full, downsample_factor)
        ring_work = downsample_mean(ring_full, downsample_factor)
        background_work = downsample_mean(background_full, downsample_factor)
        subrings_work = np.stack([downsample_mean(arr, downsample_factor) for arr in subrings_full], axis=0)

        cache.append(
            {
                "row": row,
                "sample_id": sample_id,
                "composite_work": composite_work,
                "ring_work": ring_work,
                "background_work": background_work,
                "subrings_work": subrings_work,
                "visibility_flat": visibility_from_image(composite_work).reshape(-1),
            }
        )
    return cache


tune_cache_all = load_cache(tune_meta)
holdout_cache = load_cache(holdout_meta)

rng = np.random.default_rng(41)
tuning_search_images = min(int(DATASET_CONFIG["tuning_search_images"]), len(tune_cache_all))
tune_search_idx = np.sort(rng.choice(len(tune_cache_all), size=tuning_search_images, replace=False))
tune_cache_search = [tune_cache_all[idx] for idx in tune_search_idx]


def evaluate_structured_method(cache: list[dict], template_bank: np.ndarray, template_meta_rows: list[dict] | None, lam: float, method_label: str) -> list[dict]:
    rows = []
    for item in cache:
        result = estimate_structured_model(
            y_flat=item["visibility_flat"],
            template_flat=template_bank,
            lam=lam,
            h2_flat=h2_flat,
        )
        meta_row = template_meta_rows[result["best_idx"]] if template_meta_rows is not None else None
        radius_hat = (
            float(meta_row["radius_work"] * downsample_factor)
            if meta_row is not None
            else float(radius_grid_work[result["best_idx"]] * downsample_factor)
        )
        gamma_hat = float(meta_row["gamma"]) if meta_row is not None else np.nan
        n_subrings_hat = int(meta_row["n_subrings"]) if meta_row is not None else 1

        ring_hat = np.clip(
            image_from_visibility(result["ring_hat_flat"].reshape(image_size_work, image_size_work)),
            0.0,
            None,
        )
        ring_rel_mse = float(
            np.mean((ring_hat - item["ring_work"]) ** 2) / (np.mean(item["ring_work"] ** 2) + 1e-12)
        )

        rows.append(
            {
                "sample_id": item["sample_id"],
                "split": item["row"]["split"],
                "method": method_label,
                "lambda": lam,
                "radius_hat": radius_hat,
                "gamma_hat": gamma_hat,
                "n_subrings_hat": n_subrings_hat,
                "confidence": float(result["confidence"]),
                "residual_fraction": float(result["residual_fraction"]),
                "ring_rel_mse": ring_rel_mse,
                "radius_abs_error": abs(radius_hat - float(item["row"]["base_radius_true"])),
                "gamma_abs_error": abs(gamma_hat - float(item["row"]["gamma_true"])) if not np.isnan(gamma_hat) else np.nan,
                "objective": float(result["objective"]),
            }
        )
    return rows


def evaluate_heuristic_method(cache: list[dict]) -> list[dict]:
    rows = []
    rho_flat = rho.reshape(-1)
    for item in cache:
        result = estimate_amplitude_heuristic(
            y_flat=item["visibility_flat"],
            template_flat=single_ring_templates,
            rho_flat=rho_flat,
        )
        radius_hat = float(radius_grid_work[result["best_idx"]] * downsample_factor)
        rows.append(
            {
                "sample_id": item["sample_id"],
                "split": item["row"]["split"],
                "method": "Amplitude heuristic",
                "lambda": np.nan,
                "radius_hat": radius_hat,
                "gamma_hat": np.nan,
                "n_subrings_hat": 1,
                "confidence": float(np.clip(result["score"], 0.0, 1.0)),
                "residual_fraction": np.nan,
                "ring_rel_mse": np.nan,
                "radius_abs_error": abs(radius_hat - float(item["row"]["base_radius_true"])),
                "gamma_abs_error": np.nan,
                "objective": np.nan,
            }
        )
    return rows


method_specs = [
    {
        "label": "Monolithic ring model",
        "template_bank": single_ring_templates,
        "template_meta": None,
    },
    {
        "label": "Two-subring model",
        "template_bank": subring_banks[2],
        "template_meta": subring_meta[2],
    },
    {
        "label": "Four-subring model",
        "template_bank": subring_banks[4],
        "template_meta": subring_meta[4],
    },
]

tuning_rows = []
tuned_lambdas = {}

for spec in method_specs:
    per_lambda = []
    for lam in lambda_grid:
        eval_rows = evaluate_structured_method(
            cache=tune_cache_search,
            template_bank=spec["template_bank"],
            template_meta_rows=spec["template_meta"],
            lam=float(lam),
            method_label=spec["label"],
        )
        df = pd.DataFrame(eval_rows)
        per_lambda.append(
            {
                "method": spec["label"],
                "lambda": float(lam),
                "radius_mae": float(df["radius_abs_error"].mean()),
                "ring_rel_mse_mean": float(df["ring_rel_mse"].mean()),
                "gamma_mae": float(df["gamma_abs_error"].dropna().mean()) if df["gamma_abs_error"].notna().any() else np.nan,
            }
        )
    tuning_rows.extend(per_lambda)
    best = min(per_lambda, key=lambda row: row["radius_mae"])
    tuned_lambdas[spec["label"]] = float(best["lambda"])

tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv(RESULTS_ROOT / "tuning_search_summary.csv", index=False)

prediction_rows = evaluate_heuristic_method(tune_cache_all) + evaluate_heuristic_method(holdout_cache)
for spec in method_specs:
    lam = tuned_lambdas[spec["label"]]
    prediction_rows.extend(
        evaluate_structured_method(
            cache=tune_cache_all,
            template_bank=spec["template_bank"],
            template_meta_rows=spec["template_meta"],
            lam=lam,
            method_label=spec["label"],
        )
    )
    prediction_rows.extend(
        evaluate_structured_method(
            cache=holdout_cache,
            template_bank=spec["template_bank"],
            template_meta_rows=spec["template_meta"],
            lam=lam,
            method_label=spec["label"],
        )
    )

pred_df = pd.DataFrame(prediction_rows).merge(
    metadata[
        [
            "sample_id",
            "split",
            "gap_true",
            "empirical_coherence_true",
            "subring_bound_rhs_true",
            "base_radius_true",
            "gamma_true",
            "tail_rel_n1_true",
            "tail_rel_n1_bound",
            "tail_rel_n2_true",
            "tail_rel_n2_bound",
            "tail_rel_n3_true",
            "tail_rel_n3_bound",
        ]
    ],
    on=["sample_id", "split"],
    how="left",
)
pred_df.to_csv(RESULTS_ROOT / "benchmark_predictions_long.csv", index=False)

holdout_pred = pred_df.loc[pred_df["split"] == "holdout"].copy()

summary_rows = []
bootstrap_rng = np.random.default_rng(123)
for method, group in holdout_pred.groupby("method"):
    radius_values = group["radius_abs_error"].to_numpy(dtype=float)
    ci_lo, ci_hi = bootstrap_ci(radius_values, rng=bootstrap_rng)
    summary_rows.append(
        {
            "method": method,
            "n_holdout": int(len(group)),
            "radius_mae": float(np.mean(radius_values)),
            "radius_mae_ci_lo": ci_lo,
            "radius_mae_ci_hi": ci_hi,
            "radius_median_abs_error": float(np.median(radius_values)),
            "ring_rel_mse_mean": float(group["ring_rel_mse"].dropna().mean()) if group["ring_rel_mse"].notna().any() else np.nan,
            "gamma_mae": float(group["gamma_abs_error"].dropna().mean()) if group["gamma_abs_error"].notna().any() else np.nan,
        }
    )
summary_df = pd.DataFrame(summary_rows).sort_values("radius_mae")
summary_df.to_csv(RESULTS_ROOT / "holdout_method_summary.csv", index=False)


def binned_curve(frame: pd.DataFrame, method: str, x_col: str, y_col: str, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    group = frame.loc[frame["method"] == method]
    mids = []
    medians = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (group[x_col] >= left) & (group[x_col] < right)
        if np.any(mask):
            mids.append(0.5 * (left + right))
            medians.append(float(np.median(group.loc[mask, y_col])))
    return np.array(mids), np.array(medians)


def save_gap_cases_figure() -> list[str]:
    chosen_ids = []
    fig, axes = plt.subplots(3, 4, figsize=(12.8, 9.2))
    targets = DATASET_CONFIG["gap_levels"][::2]
    for row_idx, gap_target in enumerate(targets):
        candidates = holdout_meta.copy()
        candidates["gap_dist"] = np.abs(candidates["gap_true"] - gap_target)
        chosen = candidates.sort_values(["gap_dist", "noise_std_true"]).iloc[0]
        chosen_ids.append(str(chosen["sample_id"]))
        sample_id = str(chosen["sample_id"])
        composite = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_composite.npy")
        ring_true = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_ring_true.npy")
        background = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_background_true.npy")

        sample_cache = next(item for item in holdout_cache if item["sample_id"] == sample_id)
        result = estimate_structured_model(
            y_flat=sample_cache["visibility_flat"],
            template_flat=subring_banks[4],
            lam=tuned_lambdas["Four-subring model"],
            h2_flat=h2_flat,
        )
        ring_est = np.clip(image_from_visibility(result["ring_hat_flat"].reshape(image_size_work, image_size_work)), 0.0, None)
        ring_est_full = np.repeat(np.repeat(ring_est, downsample_factor, axis=0), downsample_factor, axis=1)

        panels = [
            (composite, f"Composite\nDelta={chosen['gap_true']:.2f}"),
            (ring_true, "True aggregate ring"),
            (background, "True background"),
            (ring_est_full, "Four-subring estimate"),
        ]
        for col_idx, (image, label) in enumerate(panels):
            axes[row_idx, col_idx].imshow(robust_normalize(image), cmap="inferno")
            axes[row_idx, col_idx].set_title(label)
            axes[row_idx, col_idx].axis("off")
    fig.suptitle("Representative held-out cases across the designed coherence gap", fontsize=15, y=0.98)
    fig.tight_layout()
    save_path = FIG_ROOT / "figure_01_gap_cases.png"
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return chosen_ids


def save_subring_tower_figure() -> str:
    candidates = holdout_meta.copy()
    candidates["score"] = (
        np.abs(candidates["gap_true"] - float(np.median(holdout_meta["gap_true"])))
        + np.abs(candidates["gamma_true"] - float(np.quantile(holdout_meta["gamma_true"], 0.35)))
    )
    chosen = candidates.sort_values(["score", "noise_std_true"]).iloc[0]
    sample_id = str(chosen["sample_id"])

    composite = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_composite.npy")
    ring_true = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_ring_true.npy")
    subrings = np.load(DATA_ROOT / "holdout" / "images_npy" / f"{sample_id}_subrings_true.npy")

    fig = plt.figure(figsize=(12.6, 6.2))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.15, 1.15],
        wspace=0.28,
        hspace=0.42,
    )
    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[:, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[1, 3]),
    ]
    images = [composite, ring_true, subrings[0], subrings[1], subrings[2], subrings[3]]
    labels = [
        "Composite",
        "Aggregate ring",
        "Subring 1",
        "Subring 2",
        "Subring 3",
        "Subring 4",
    ]
    cmaps = ["inferno", "inferno", "magma", "magma", "magma", "magma"]
    for ax, image, label, cmap in zip(axes, images, labels, cmaps):
        ax.imshow(robust_normalize(image), cmap=cmap)
        ax.set_title(label, fontsize=13, fontweight="semibold", pad=10)
        ax.axis("off")
    fig.suptitle(
        f"True subring-resolved signal tower for {sample_id} | gamma={chosen['gamma_true']:.2f}, gap={chosen['gap_true']:.2f}",
        fontsize=18,
        y=0.965,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.06, top=0.88)
    fig.savefig(FIG_ROOT / "figure_02_subring_tower.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return sample_id


def save_method_comparison_figure() -> None:
    order = [
        "Amplitude heuristic",
        "Monolithic ring model",
        "Two-subring model",
        "Four-subring model",
    ]
    plot_df = summary_df.set_index("method").loc[order].reset_index()
    colors = ["#8c5a2b", "#264653", "#2a9d8f", "#e76f51"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    ax = axes[0]
    ax.bar(plot_df["method"], plot_df["radius_mae"], color=colors)
    ax.errorbar(
        x=np.arange(len(plot_df)),
        y=plot_df["radius_mae"],
        yerr=np.vstack(
            [
                plot_df["radius_mae"] - plot_df["radius_mae_ci_lo"],
                plot_df["radius_mae_ci_hi"] - plot_df["radius_mae"],
            ]
        ),
        fmt="none",
        ecolor="#202536",
        capsize=4,
        lw=1.2,
    )
    ax.set_ylabel("Held-out radius MAE (pixels)")
    ax.set_title("Geometry recovery comparison")
    ax.tick_params(axis="x", rotation=18)

    ax = axes[1]
    structured = plot_df.loc[plot_df["method"] != "Amplitude heuristic"].copy()
    ax.bar(structured["method"], structured["ring_rel_mse_mean"], color=colors[1:])
    ax.set_ylabel("Mean relative ring MSE")
    ax.set_title("Ring reconstruction quality")
    ax.tick_params(axis="x", rotation=18)

    fig.tight_layout()
    fig.savefig(FIG_ROOT / "figure_03_method_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_error_vs_coherence_figure() -> None:
    bins = np.linspace(
        float(holdout_pred["empirical_coherence_true"].min()),
        float(holdout_pred["empirical_coherence_true"].max()) + 1e-6,
        7,
    )
    fig, ax = plt.subplots(figsize=(7.1, 4.7))
    palette = {
        "Amplitude heuristic": "#8c5a2b",
        "Monolithic ring model": "#264653",
        "Two-subring model": "#2a9d8f",
        "Four-subring model": "#e76f51",
    }
    for method, color in palette.items():
        xs, ys = binned_curve(holdout_pred, method, "empirical_coherence_true", "radius_abs_error", bins)
        ax.plot(xs, ys, marker="o", label=method, color=color, lw=2.0)
    ax.set_xlabel("Empirical ring-background coherence")
    ax.set_ylabel("Median absolute radius error (pixels)")
    ax.set_title("Recovery degrades as coherence rises, but not equally across methods")
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "figure_04_error_vs_coherence.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_gap_and_bound_figure() -> tuple[float, float]:
    group = holdout_meta.groupby("gap_true", as_index=False)["empirical_coherence_true"].median().sort_values("gap_true")
    gaps = group["gap_true"].to_numpy(dtype=float)
    coherence_med = group["empirical_coherence_true"].to_numpy(dtype=float)
    coeffs = np.polyfit(gaps, np.log(np.clip(coherence_med, 1e-6, None)), 1)
    beta_hat = -float(coeffs[0])
    scale_hat = float(np.exp(coeffs[1]))
    fit_curve = scale_hat * np.exp(-beta_hat * gaps)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 8.8),
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )

    ax = axes[0]
    ax.scatter(
        holdout_meta["gap_true"],
        holdout_meta["empirical_coherence_true"],
        alpha=0.28,
        color="#51607c",
        s=22,
        label="Samples",
    )
    ax.plot(gaps, coherence_med, marker="o", color="#e76f51", lw=2.4, label="Median by gap")
    ax.plot(gaps, fit_curve, linestyle="--", color="#2a9d8f", lw=2.0, label=f"exp fit, beta={beta_hat:.2f}")
    ax.set_xlabel("Designed criticality gap")
    ax.set_ylabel("Empirical coherence")
    ax.set_title("Coherence decays as the designed gap widens", pad=10)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.22)

    ax = axes[1]
    ax.scatter(
        holdout_meta["empirical_coherence_true"],
        holdout_meta["subring_bound_rhs_true"],
        alpha=0.68,
        color="#264653",
        s=24,
    )
    lo = float(min(holdout_meta["empirical_coherence_true"].min(), holdout_meta["subring_bound_rhs_true"].min()))
    hi = float(max(holdout_meta["empirical_coherence_true"].max(), holdout_meta["subring_bound_rhs_true"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#c63d2f", lw=1.5)
    ax.set_xlabel("True aggregate coherence")
    ax.set_ylabel("Weighted subring bound RHS")
    ax.set_title("The weighted subring bound upper-bounds aggregate coherence", pad=10)
    ax.grid(alpha=0.22)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.95, hspace=0.34)
    fig.savefig(FIG_ROOT / "figure_05_gap_and_bound.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return scale_hat, beta_hat


def save_truncation_curve_figure() -> None:
    retained = np.array([1, 2, 3], dtype=float)
    true_mean = np.array(
        [
            holdout_meta["tail_rel_n1_true"].mean(),
            holdout_meta["tail_rel_n2_true"].mean(),
            holdout_meta["tail_rel_n3_true"].mean(),
        ]
    )
    bound_mean = np.array(
        [
            holdout_meta["tail_rel_n1_bound"].mean(),
            holdout_meta["tail_rel_n2_bound"].mean(),
            holdout_meta["tail_rel_n3_bound"].mean(),
        ]
    )

    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    ax.plot(retained, true_mean, marker="o", lw=2.4, color="#2a9d8f", label="True average tail fraction")
    ax.plot(retained, bound_mean, marker="s", lw=2.0, linestyle="--", color="#c63d2f", label="Average geometric envelope")
    ax.set_xlabel("Retained leading subrings N")
    ax.set_ylabel("Remaining tail / total ring norm")
    ax.set_title("Only a few leading subrings carry most of the signal")
    ax.set_xticks(retained)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "figure_06_truncation_curve.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


representative_gap_ids = save_gap_cases_figure()
representative_subring_id = save_subring_tower_figure()
save_method_comparison_figure()
save_error_vs_coherence_figure()
scale_hat, beta_hat = save_gap_and_bound_figure()
save_truncation_curve_figure()

bound_coverage = float(np.mean(holdout_meta["subring_bound_rhs_true"] >= holdout_meta["empirical_coherence_true"] - 1e-9))

monolithic_mae = float(summary_df.loc[summary_df["method"] == "Monolithic ring model", "radius_mae"].iloc[0])
four_subring_mae = float(summary_df.loc[summary_df["method"] == "Four-subring model", "radius_mae"].iloc[0])
improvement_pct = 100.0 * (monolithic_mae - four_subring_mae) / max(monolithic_mae, 1e-12)

save_json(
    {
        "tuned_lambdas": tuned_lambdas,
        "holdout_metrics": summary_df.to_dict(orient="records"),
        "coherence_gap_fit": {
            "scale_hat": scale_hat,
            "beta_hat": beta_hat,
        },
        "bound_coverage_fraction": bound_coverage,
        "improvement_monolithic_to_four_subring_pct": improvement_pct,
        "representative_gap_cases": representative_gap_ids,
        "representative_subring_case": representative_subring_id,
        "figure_paths": {
            "gap_cases": str((FIG_ROOT / "figure_01_gap_cases.png").relative_to(SUITE_ROOT)),
            "subring_tower": str((FIG_ROOT / "figure_02_subring_tower.png").relative_to(SUITE_ROOT)),
            "method_comparison": str((FIG_ROOT / "figure_03_method_comparison.png").relative_to(SUITE_ROOT)),
            "error_vs_coherence": str((FIG_ROOT / "figure_04_error_vs_coherence.png").relative_to(SUITE_ROOT)),
            "gap_and_bound": str((FIG_ROOT / "figure_05_gap_and_bound.png").relative_to(SUITE_ROOT)),
            "truncation_curve": str((FIG_ROOT / "figure_06_truncation_curve.png").relative_to(SUITE_ROOT)),
        },
    },
    RESULTS_ROOT / "benchmark_summary.json",
)

print("\nRan the combined coherence + subring benchmark.")
print(f"Tune images used for lambda search: {len(tune_cache_search)}")
print("Tuned lambdas:")
for key, value in tuned_lambdas.items():
    print(f"- {key}: {value:.3g}")
print("\nHeld-out radius MAE:")
for _, row in summary_df.iterrows():
    print(f"- {row['method']}: {row['radius_mae']:.3f} px")
print(f"\nFour-subring improvement over monolithic on held-out radius MAE: {improvement_pct:.1f}%")
print(f"Subring bound coverage on held-out samples: {100.0 * bound_coverage:.1f}%")
print(f"Benchmark outputs written to: {RESULTS_ROOT}")
