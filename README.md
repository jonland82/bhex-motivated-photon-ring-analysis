# BHEX-Motivated Photon Ring Analysis

[Live site](https://jonland82.github.io/bhex-motivated-photon-ring-analysis/)  
[Standalone coherence + subring report](coherence_subring_validation/coherence_subring_results/coherence_subring_report.html)

*Jonathan R. Landers, independent researcher. This project is inspired by published BHEX work and is not affiliated with the BHEX collaboration or mission team.*

Black hole imaging moved from theory to observation with the Event Horizon Telescope's first horizon-scale image of M87* in 2019 and its Sagittarius A* image in 2022, showing that interferometric arrays can recover lensed structure near an event horizon. The proposed Black Hole Explorer (BHEX) mission pushes that program toward longer baselines and cleaner access to the thin photon ring, where understanding ring dynamics and their separation from broader, time-varying plasma emission matters because the ring is expected to carry more direct geometric information about the underlying spacetime.

This repository studies one BHEX-motivated inverse problem in two deliberately separate steps.

The physical picture is a thin photon ring sitting inside brighter, broader plasma emission. A simple long-baseline amplitude heuristic can become hard to read once plasma contamination distorts the visibility pattern. A structured estimator can still recover the ring if the ring remains distinguishable from the nuisance component.

The repository first turns that idea into a compact Fourier-domain prototype. It then adds a second, separate synthetic validation suite designed specifically for the later geodesic-coherence and subring-resolved mathematics. The split is intentional:

- the original baseline experiment stays unchanged
- the later mathematics gets its own dataset with the latent structure it actually needs
- the repo remains reproducible end to end

At a high level, the project contains:

- six short manuscript notes in [manuscript/](manuscript)
- the main GitHub Pages landing page in [index.html](index.html)
- the original runnable prototype in [simulation/](simulation)
- a separate coherence + subring validation suite in [coherence_subring_validation/](coherence_subring_validation)

## Intuition

The narrative arc is:

1. Start with a toy but explicit observation model in Fourier space.
2. Show that structured recoverability can persist after direct heuristic readability starts to degrade.
3. Refine the nuisance side with provenance and coherence language.
4. Refine the signal side with a hierarchy of winding-order subrings.
5. Validate those later refinements in a second synthetic benchmark instead of forcing them into the original prototype.

That gives the repo two complementary experiment tracks:

- `simulation/` validates the original ring-versus-plasma estimation argument.
- `coherence_subring_validation/` validates the later coherence and subring claims in a controlled synthetic setting.

## Mathematical skeleton

The baseline note starts from the visibility decomposition

$$
y = \alpha g_{\theta} + q + \varepsilon,
$$

where $g_{\theta}$ is a ring template, $\alpha$ is its strength, $q$ is structured nuisance, and $\varepsilon$ is noise.

The prototype estimator fits

$$
(\hat{\alpha}, \hat{\theta}, \hat{q})
\in
\arg\min_{\alpha,\theta,q}
\left\| y - \alpha g_{\theta} - q \right\|^2
+
\lambda \left\| Hq \right\|^2,
$$

so the ring is modeled explicitly while the nuisance term is penalized if it carries too much high-frequency structure.

The later provenance/coherence notes refine the nuisance side. In the abstract, the relevant overlap is

$$
\mu(g,q) = \frac{|\langle g, q \rangle|}{\|g\| \, \|q\|},
$$

and the geodesic-provenance framework argues that restricting nuisance to ordinary escaping trajectories should reduce, or at worst preserve, the harmful overlap. The coherence note then bounds that overlap by operator geometry, schematically through quantities of the form

$$
\mu_{\mathrm{prov}} \lesssim \frac{\|A_b^* A_r\|}{\sigma_r \sigma_b}.
$$

The subring note refines the signal side instead of the nuisance side:

$$
y = \sum_{n \ge 1} \alpha_n g_{\theta,n} + q + \varepsilon,
\qquad
\alpha_n = \alpha_1 e^{-\gamma (n-1)}.
$$

That note proves two practical points:

- aggregate ring-background coherence is controlled by a weighted combination of subring-level coherence terms
- the infinite tower is approximable because the neglected tail decays geometrically, with a bound of the form

$$
\left\| \sum_{n > N} \alpha_n g_n \right\|
\le
\frac{\alpha_1 c_{\max} e^{-\gamma N}}{1 - e^{-\gamma}}.
$$

The repository reflects that mathematical arc directly: the first experiment implements the first equation set, while the second experiment instantiates the coherence and subring formulas in a separate synthetic benchmark.

## Repository map

- [index.html](index.html)
  - the main project narrative and landing page for GitHub Pages
- [manuscript/fourier-domain-analysis-bhex.pdf](manuscript/fourier-domain-analysis-bhex.pdf)
  - the original Fourier-domain mismatch and recoverability note
- [manuscript/geodesic_provenance_bhex_note.pdf](manuscript/geodesic_provenance_bhex_note.pdf)
  - provenance-constrained nuisance refinement
- [manuscript/geodesic_coherence_bhex_note.pdf](manuscript/geodesic_coherence_bhex_note.pdf)
  - explicit coherence bounds and criticality-gap framing
- [manuscript/subring_resolved_bhex_note.pdf](manuscript/subring_resolved_bhex_note.pdf)
  - subring-resolved signal model and finite truncation
- [manuscript/subring_refinement_summary_note.pdf](manuscript/subring_refinement_summary_note.pdf)
  - summary of the provenance-and-subring refinement arc
- [manuscript/geodesic_coherence_summary_note_updated.pdf](manuscript/geodesic_coherence_summary_note_updated.pdf)
  - earlier coherence-focused summary note
- [simulation/01_generate_synthetic_bhex_images.py](simulation/01_generate_synthetic_bhex_images.py)
  - baseline dataset generator
- [simulation/02_tune_bhex_estimator.py](simulation/02_tune_bhex_estimator.py)
  - baseline estimator tuning
- [simulation/03_run_bhex_estimator_on_holdout.py](simulation/03_run_bhex_estimator_on_holdout.py)
  - baseline held-out evaluation
- [coherence_subring_validation/run_validation_suite.py](coherence_subring_validation/run_validation_suite.py)
  - one-command runner for the second validation suite
- [coherence_subring_validation/coherence_subring_results/coherence_subring_report.html](coherence_subring_validation/coherence_subring_results/coherence_subring_report.html)
  - generated HTML report for the new experiments

## Requirements

The repo uses a small scientific Python stack:

- `numpy`
- `pandas`
- `matplotlib`

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Reproducible experiment tracks

### 1. Baseline Fourier-domain prototype

This is the original runnable pipeline in [simulation/](simulation). It generates synthetic images with a thin ring and a broader plasma component, tunes the structured Fourier estimator, and evaluates it on held-out data.

Run it from the repository root:

```bash
python simulation/01_generate_synthetic_bhex_images.py
python simulation/02_tune_bhex_estimator.py
python simulation/03_run_bhex_estimator_on_holdout.py
```

With the current defaults, the baseline experiment uses:

- random seed `7`
- `120` tuning images and `40` held-out images
- image size `256 x 256`
- a circular thin-ring signal plus broader crescent-like plasma, blur, noise, and small center jitter

It writes results to:

- [bhex_synthetic_dataset/](bhex_synthetic_dataset)
- [bhex_model_tuning/](bhex_model_tuning)
- [bhex_holdout_results/](bhex_holdout_results)

Current default baseline results:

| Quantity | Value |
| --- | ---: |
| Best tuned `lambda` | `63.1` |
| Best template width | `4.0 px` |
| Held-out radius MAE | `0.269 px` |
| Held-out mean confidence | `0.738` |

What this first experiment establishes:

- a direct heuristic can become visually unreliable before structured recovery fails
- explicit ring templates plus a smooth nuisance class can recover the correct radius accurately on held-out synthetic data
- the first Fourier-domain note is operational, reproducible, and easy to inspect

### 2. Separate coherence + subring validation suite

The second experiment exists because the later notes require structure the baseline dataset does not contain. It therefore lives in its own directory and does not modify the baseline pipeline or its outputs.

Run the full suite with:

```bash
python coherence_subring_validation/run_validation_suite.py
```

Equivalent step-by-step commands:

```bash
python coherence_subring_validation/01_generate_coherence_subring_dataset.py
python coherence_subring_validation/02_run_coherence_subring_benchmark.py
python coherence_subring_validation/03_build_coherence_subring_report.py
```

With the current defaults, the second suite uses:

- random seed `23`
- `120` tuning images and `60` held-out images
- image size `192 x 192`
- `4` true subrings per image
- designed gap levels `0.45, 0.85, 1.25, 1.65, 2.05`
- a nuisance field built from a broad crescent, a gap-controlled near-critical shell, and diffuse blobs

It writes results to:

- [coherence_subring_validation/coherence_subring_dataset/](coherence_subring_validation/coherence_subring_dataset)
- [coherence_subring_validation/coherence_subring_results/](coherence_subring_validation/coherence_subring_results)
- [coherence_subring_validation/coherence_subring_results/coherence_subring_report.html](coherence_subring_validation/coherence_subring_results/coherence_subring_report.html)

Current hold-out validation results:

| Method | Radius MAE (px) | Mean ring rel. MSE | Notes |
| --- | ---: | ---: | --- |
| Amplitude heuristic | `0.883` | `-` | direct readout baseline |
| Monolithic ring model | `0.833` | `0.564` | structured, but one-ring only |
| Two-subring model | `0.424` | `0.400` | best geometry |
| Four-subring model | `0.433` | `0.231` | best full ring reconstruction |

Additional quantitative takeaways:

- subring-aware models cut held-out radius error by about `49%` relative to the monolithic model
- the weighted subring coherence bound upper-bounds aggregate coherence on `100%` of held-out cases
- the first three subrings retain about `91.4%` of the mean total ring norm
- the designed gap and empirical coherence are negatively correlated, as intended by the synthetic construction

What this second experiment establishes:

- the coherence story can be instantiated numerically in a controlled synthetic benchmark
- widening a designed separation gap reduces empirical ring-background overlap
- subring-aware estimators recover geometry and morphology much better than a monolithic one-ring approximation
- the finite-truncation logic is practically meaningful: only a few leading subrings already capture most of the signal

## Why the repo is split into two experiment families

Keeping the original prototype unchanged is part of the point.

The baseline dataset was built to validate the first note:

- one thin ring
- one broad nuisance component
- one structured estimator

That setup is enough for the original mismatch and recoverability story, but not enough to validate the later notes directly. The later notes refer to provenance-constrained nuisance structure and a hierarchy of winding-order subrings. Those objects have to exist in the synthetic data before they can be tested honestly.

So the repository uses a clean division:

- `simulation/` is the original Fourier-domain prototype
- `coherence_subring_validation/` is the later-theory validation suite

That separation keeps the narrative honest and the experiments reproducible.

## Suggested reading path

If you want the shortest path through the repo:

1. Start with [index.html](index.html).
2. Read [manuscript/fourier-domain-analysis-bhex.pdf](manuscript/fourier-domain-analysis-bhex.pdf).
3. Inspect the baseline scripts in [simulation/](simulation).
4. Read [manuscript/geodesic_provenance_bhex_note.pdf](manuscript/geodesic_provenance_bhex_note.pdf) and [manuscript/geodesic_coherence_bhex_note.pdf](manuscript/geodesic_coherence_bhex_note.pdf).
5. Read [manuscript/subring_resolved_bhex_note.pdf](manuscript/subring_resolved_bhex_note.pdf) and [manuscript/subring_refinement_summary_note.pdf](manuscript/subring_refinement_summary_note.pdf).
6. Open the generated [coherence_subring_report.html](coherence_subring_validation/coherence_subring_results/coherence_subring_report.html).

If you want the runnable story in output form after executing the scripts:

1. Inspect [bhex_model_tuning/tuned_model.json](bhex_model_tuning/tuned_model.json).
2. Inspect [bhex_holdout_results/holdout_predictions.csv](bhex_holdout_results/holdout_predictions.csv).
3. Inspect several `bhex_holdout_results/*_ring_emphasized.png` images.
4. Inspect [coherence_subring_validation/coherence_subring_results/holdout_method_summary.csv](coherence_subring_validation/coherence_subring_results/holdout_method_summary.csv).
5. Inspect the figures under [coherence_subring_validation/coherence_subring_results/figures/](coherence_subring_validation/coherence_subring_results/figures).

## Scope and limitations

This is still a research prototype, not a mission-grade astrophysical pipeline.

The baseline code remains intentionally simple:

- circular Gaussian-like ring templates
- full-image FFTs rather than realistic sparse baseline sampling
- a smooth nuisance penalty rather than provenance-constrained transport operators
- point estimates and simple confidence scores rather than full uncertainty quantification

The new validation suite is also intentionally scoped:

- it is synthetic and controlled
- it instantiates the later coherence and subring inequalities directly
- it is not a full ray-traced geodesic forward model

So the repo demonstrates two different things:

- the first-note structured estimator works in a clean toy setting
- the later coherence and subring logic works in a purpose-built synthetic benchmark

What is still not implemented in the main recovery code:

- provenance-constrained nuisance classes inside `simulation/`
- direct estimation of $A_r$, $A_b$, or $A_b^* A_r$
- realistic sparse visibility sampling and baseline masks
- a subring-resolved inference model inside the original prototype itself
- a more realistic criticality-index-based forward model

## Related BHEX papers

The mathematical framing in this repo is motivated by the broader BHEX program. The notes here cite these BHEX collaboration references directly:

- *Black Hole Explorer: Motivation and Vision*, arXiv:2406.12917, https://arxiv.org/abs/2406.12917
- *The Black Hole Explorer: Photon Ring Science, Detection and Shape Measurement*, arXiv:2406.09498, https://arxiv.org/abs/2406.09498
- *Interferometric Inference of Black Hole Spin from Photon Ring Size and Brightness*, arXiv:2509.23628, https://arxiv.org/abs/2509.23628

## Bottom line

This repository is best read as a two-stage executable argument.

The first stage shows, in a deliberately minimal Fourier-domain prototype, that ring-aware structured recovery can succeed after a direct visibility-amplitude heuristic becomes hard to interpret. The second stage shows, in a separate controlled benchmark, that the later geodesic-coherence and subring-resolved mathematics are not merely formal extensions: they produce measurable gains in recoverability, reconstruction quality, and finite-truncation behavior once the synthetic data actually contains the corresponding latent structure.
