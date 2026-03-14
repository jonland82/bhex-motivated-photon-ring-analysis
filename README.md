# BHEX-Motivated Photon Ring Analysis

This repository contains a short paper, a standalone HTML presentation, and a three-stage Python prototype for a Fourier-domain photon-ring inference workflow inspired by the Black Hole Explorer (BHEX) program.

The motivating idea is simple:

- the image contains a broad, messy plasma component and a much thinner photon ring
- a direct long-baseline heuristic can become hard to read once plasma contamination distorts the visibility pattern
- a structured estimator can still recover the ring if the ring remains distinguishable from the nuisance component

The repository makes that story concrete with synthetic images, Fourier-domain fitting, hyperparameter tuning, and held-out reconstruction.

It is a research prototype designed for methodological clarity and communication, not a mission-grade astrophysical pipeline.

For the quickest overview, start with the GitHub Pages site or `index.html`, then read `fourier-domain-analysis-bhex.pdf`, then inspect the scripts in `simulation/`.

At the center of the prototype is a simple decomposition of the observed visibility:

$$
y = \alpha g_{\theta} + q + \varepsilon,
$$

where $g_{\theta}$ is a ring template, $\alpha$ is its strength, $q$ is structured nuisance from the plasma, and $\varepsilon$ is noise.

## Repository contents and scope

This repository combines a paper, a browser presentation, and a runnable prototype around one BHEX-motivated claim: a direct amplitude heuristic based on $|y|$ can become hard to read before a structured estimator loses the ability to recover the ring.

- `fourier-domain-analysis-bhex.pdf`
  - the short paper that states the central mathematical argument
- `index.html`
  - a self-contained presentation of the same story, with figures and summary metrics
- `manuscript/`
  - the LaTeX source for the paper and related manuscript files
- `simulation/01_generate_synthetic_bhex_images.py`
  - generates synthetic images with a thin ring, broader plasma, blur, and noise
- `simulation/02_tune_bhex_estimator.py`
  - fits a Fourier-domain estimator with explicit ring templates $g_{\theta}$ and nuisance term $q$
- `simulation/03_run_bhex_estimator_on_holdout.py`
  - applies the tuned model to held-out images and exports reconstructed ring/plasma diagnostics

Together, these pieces form one deliberately scoped workflow:

1. Generate synthetic black-hole-like images with a thin ring, a broader crescent-like plasma component, blur, and noise.
2. Transform those images into Fourier space and fit a ring-aware estimator that treats the plasma as structured nuisance.
3. Bring the recovered components back into image space and create ring-emphasized visualizations for inspection.

This is not blind source separation. The ring is modeled explicitly as a template family $g_{\theta}$, while the plasma is pushed into a smoother nuisance class $q$.

## Main mathematical contributions

The paper contributes a compact mathematical framing for the prototype:

- it gives a mismatch comparison between a direct amplitude heuristic and a plasma-aware estimator
- it identifies the key controls on that gap: ring-plasma overlap, nuisance strength, and noise
- it argues that heuristic readability can fail before structured recoverability fails
- it reframes the inference problem as physics-structured source separation in Fourier space
- it points toward the next scientific step: estimating coherence or identifiability margins for more realistic astrophysical nuisance classes

## Requirements

The scripts use a small scientific Python stack:

- `numpy`
- `pandas`
- `matplotlib`

Everything else is from the Python standard library. No SciPy dependency is required.

Install the Python packages with:

```bash
python -m pip install numpy pandas matplotlib
```

## How to run the prototype

Run the scripts from the repository root, in this order:

```bash
python simulation/01_generate_synthetic_bhex_images.py
python simulation/02_tune_bhex_estimator.py
python simulation/03_run_bhex_estimator_on_holdout.py
```

These steps depend on one another, so they should be run sequentially.

Practical notes:

- the scripts use relative paths based on your current working directory
- if you run them from the repository root, the output folders will appear at the top level of the repo
- if you run them from another directory, the outputs will be created there instead
- script 1 deletes and recreates `bhex_synthetic_dataset/` by default because `OVERWRITE_OUTPUT = True`

## What each script does

The scripts form a linear pipeline: generate controlled data, tune the estimator, then evaluate on held-out images.

### 1. Generate synthetic images

File: `simulation/01_generate_synthetic_bhex_images.py`

This stage creates a toy dataset with known ground truth. Each sample includes:

- a thin circular photon ring
- a broader crescent-like plasma component
- optional scattering-like blur
- additive noise
- small center jitter

With the current defaults it creates:

- 120 tuning images
- 40 held-out images
- image size `256 x 256`
- a fixed random seed of `7`

It writes:

- `bhex_synthetic_dataset/metadata.csv`
- `bhex_synthetic_dataset/dataset_config.json`
- `bhex_synthetic_dataset/tune/images_npy/*.npy`
- `bhex_synthetic_dataset/tune/png/*.png`
- `bhex_synthetic_dataset/holdout/images_npy/*.npy`
- `bhex_synthetic_dataset/holdout/png/*.png`

Each sample saves:

- the noisy composite image
- the true ring component
- the true plasma component

Only the earliest generated examples get three-panel preview images, so the preview PNGs appear in the tuning split, not the holdout split.

The most useful knobs near the top of the file are:

- dataset size
- image size
- ring radius, width, and amplitude ranges
- plasma offset, width, amplitude, and asymmetry ranges
- blur and noise ranges
- preview count
- overwrite behavior

### 2. Tune the estimator

File: `simulation/02_tune_bhex_estimator.py`

This stage reads the synthetic tuning set, downsamples the images for speed, moves them into Fourier space, and searches for a good estimator configuration.

The fitting objective is intentionally simple:

$$
\min_{\alpha,\theta,q} \; \|y-\alpha g_{\theta}-q\|^2 + \lambda \|Hq\|^2,
$$

so the ring is fit explicitly while the nuisance term is penalized if it carries too much high-frequency structure.

This is calibration rather than machine-learning training: the script is choosing a useful regularization strength and template width for this toy problem.

Conceptually it does four things:

- builds a bank of ring templates
- searches over candidate ring radii $\theta$
- regularizes the nuisance term $q$ so it stays smoother than the ring
- chooses the best hyperparameters by balancing radius error, amplitude error, and confidence

With the current defaults it searches over:

- `lambda` values on a log grid
- template widths of `2.0`, `3.0`, and `4.0` original-image pixels
- a ring-radius search range from `34` to `82` pixels
- a working resolution of `64 x 64` after downsampling by `4`
- a search subset of `30` tuning images before rerunning the best setting on the full tuning split

It writes:

- `bhex_model_tuning/tuned_model.json`
- `bhex_model_tuning/tuning_grid_summary.csv`
- `bhex_model_tuning/tuning_predictions.csv`
- `bhex_model_tuning/heatmap_radius_mae.png`
- `bhex_model_tuning/heatmap_confidence.png`
- `bhex_model_tuning/scatter_true_vs_estimated_radius.png`
- `bhex_model_tuning/hist_radius_error.png`
- `bhex_model_tuning/*_tuning_example.png`

The most important knobs near the top of the file are:

- `DATA_ROOT`
- `OUTPUT_ROOT`
- `DOWNSAMPLE_FACTOR`
- `MAX_TUNE_IMAGES_FOR_SEARCH`
- `LAMBDA_GRID`
- `TEMPLATE_WIDTH_GRID`
- `PENALTY_POWER`
- the radius-search bounds and step size

### 3. Run held-out inference

File: `simulation/03_run_bhex_estimator_on_holdout.py`

This stage loads the tuned model from step 2 and applies it to the held-out images.

For each held-out sample it:

- estimates the ring parameters
- reconstructs the ring and plasma components
- computes residual diagnostics
- exports a ring-emphasized PNG in which the ring $\hat{\alpha} g_{\hat{\theta}}$ stays visually prominent while the plasma remains visible in the background

It writes:

- `bhex_holdout_results/holdout_predictions.csv`
- `bhex_holdout_results/scatter_true_vs_estimated_radius_holdout.png`
- `bhex_holdout_results/hist_radius_error_holdout.png`
- `bhex_holdout_results/confidence_vs_radius_error_holdout.png`
- `bhex_holdout_results/*_ring_hat.npy`
- `bhex_holdout_results/*_plasma_hat.npy`
- `bhex_holdout_results/*_ring_emphasized.png`

Montages are available but off by default. To export them, set:

```python
EXPORT_PER_IMAGE_MONTAGES = True
```

The main visualization knobs in this script control:

- how bright the plasma background remains
- how much of the original composite image remains visible
- the minimum and maximum ring boost as a function of confidence

## Default behavior and example results

The paper, the HTML page, and the default scripts are aligned around the same toy setup. With the current defaults, a full run produces roughly:

- the best tuned `lambda` is about `63.1`
- the best template width is `4.0` pixels in the original image grid
- tuning mean absolute radius error is about `0.24` pixels
- held-out mean absolute radius error is about `0.27` pixels
- held-out mean confidence is about `0.74`

These are not universal scientific conclusions. They are example results for this specific synthetic prototype, random seed, and search grid.

## Suggested reading path

If you want the repository in the order of highest payoff, inspect these in order:

1. `index.html` for the high-level story and figures
2. `fourier-domain-analysis-bhex.pdf` for the compact conceptual argument
3. `bhex_model_tuning/tuned_model.json` after a run
4. `bhex_model_tuning/heatmap_radius_mae.png`
5. `bhex_holdout_results/holdout_predictions.csv`
6. several `bhex_holdout_results/*_ring_emphasized.png` images

The emphasized PNGs are especially useful because they show whether the recovered ring looks like a coherent thin structure rather than a broad halo.

## Scope and limitations

This prototype is intentionally narrow. It currently uses:

- circular Gaussian-like ring templates
- a broad crescent-like plasma model in the synthetic images
- full-image FFTs rather than realistic sparse baseline sampling
- a smoothness penalty for nuisance structure instead of a GRMHD-informed nuisance family
- point estimates and simple confidence scores rather than full uncertainty quantification

That scope is a feature, not a bug. The goal is to isolate the estimation mechanism clearly before adding astrophysical realism.

Natural upgrades, also reflected in the HTML presentation, include:

- elliptical or spin-informed ring families
- explicit baseline masks and more realistic visibility sampling
- GRMHD-informed nuisance classes
- posterior summaries or uncertainty bands
- confidence diagnostics tied more directly to recoverability margins

## Related BHEX papers

The mathematical framing in this repository is motivated by the broader BHEX program. The paper in this repo cites these BHEX collaboration references directly:

- *Black Hole Explorer: Motivation and Vision*, arXiv:2406.12917, https://arxiv.org/abs/2406.12917
- *The Black Hole Explorer: Photon Ring Science, Detection and Shape Measurement*, arXiv:2406.09498, https://arxiv.org/abs/2406.09498
- *Interferometric Inference of Black Hole Spin from Photon Ring Size and Brightness*, arXiv:2509.23628, https://arxiv.org/abs/2509.23628

## Bottom line

This repository is best read as a BHEX-motivated thought experiment made executable.

It starts from the physical picture of a thin photon ring embedded in broader plasma emission, moves that picture into Fourier space, and then asks a practical question:

can a ring-aware estimator still recover the geometry after a simple long-baseline readout has become hard to interpret?

The paper argues yes in principle. The scripts show one concrete toy implementation of that argument.
