"""
03_build_coherence_subring_report.py

Build a standalone HTML report for the combined provenance/coherence + subring
validation suite. The report is intentionally separate from the main repo
landing page so the baseline writeup stays unchanged.
"""

from __future__ import annotations

from pathlib import Path
import html
import json

import numpy as np
import pandas as pd

from validation_common import SUITE_ROOT


DATA_ROOT = SUITE_ROOT / "coherence_subring_dataset"
RESULTS_ROOT = SUITE_ROOT / "coherence_subring_results"
FIG_ROOT = RESULTS_ROOT / "figures"
REPORT_PATH = RESULTS_ROOT / "coherence_subring_report.html"


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def rel_gain(baseline: float, improved: float) -> float:
    return 100.0 * (float(baseline) - float(improved)) / max(float(baseline), 1e-12)


def figure_rel_path(name: str) -> str:
    return Path("figures") / name


def img_card(src: str, title: str, caption: str, large: bool = False) -> str:
    card_class = "figure-card figure-card-large" if large else "figure-card"
    return f"""
    <figure class="{card_class}">
      <img src="{html.escape(src)}" alt="{html.escape(title)}" />
      <figcaption>
        <h3>{html.escape(title)}</h3>
        <p>{html.escape(caption)}</p>
      </figcaption>
    </figure>
    """


def table_html(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{row[key]}</td>" for key, _ in columns)
        rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(rows)
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{head}</tr></thead>
        <tbody>
          {body}
        </tbody>
      </table>
    </div>
    """


with open(RESULTS_ROOT / "benchmark_summary.json", "r", encoding="utf-8") as f:
    benchmark_summary = json.load(f)

with open(DATA_ROOT / "dataset_config.json", "r", encoding="utf-8") as f:
    dataset_config = json.load(f)

summary_df = pd.read_csv(RESULTS_ROOT / "holdout_method_summary.csv")
pred_df = pd.read_csv(RESULTS_ROOT / "benchmark_predictions_long.csv")
metadata = pd.read_csv(DATA_ROOT / "metadata.csv")

holdout_pred = pred_df.loc[pred_df["split"] == "holdout"].copy()
holdout_meta = metadata.loc[metadata["split"] == "holdout"].copy()

method_order = [
    "Amplitude heuristic",
    "Monolithic ring model",
    "Two-subring model",
    "Four-subring model",
]
summary_df["method"] = pd.Categorical(summary_df["method"], categories=method_order, ordered=True)
summary_df = summary_df.sort_values("method").reset_index(drop=True)
summary_lookup = summary_df.set_index("method")

amp_mae = float(summary_lookup.loc["Amplitude heuristic", "radius_mae"])
mono_mae = float(summary_lookup.loc["Monolithic ring model", "radius_mae"])
two_mae = float(summary_lookup.loc["Two-subring model", "radius_mae"])
four_mae = float(summary_lookup.loc["Four-subring model", "radius_mae"])

two_vs_mono = rel_gain(mono_mae, two_mae)
four_vs_mono = rel_gain(mono_mae, four_mae)
two_vs_amp = rel_gain(amp_mae, two_mae)
four_vs_amp = rel_gain(amp_mae, four_mae)

mono_ring_mse = float(summary_lookup.loc["Monolithic ring model", "ring_rel_mse_mean"])
two_ring_mse = float(summary_lookup.loc["Two-subring model", "ring_rel_mse_mean"])
four_ring_mse = float(summary_lookup.loc["Four-subring model", "ring_rel_mse_mean"])
four_recon_gain = rel_gain(mono_ring_mse, four_ring_mse)

bound_coverage = float(benchmark_summary["bound_coverage_fraction"])
beta_hat = float(benchmark_summary["coherence_gap_fit"]["beta_hat"])
scale_hat = float(benchmark_summary["coherence_gap_fit"]["scale_hat"])
gap_corr = float(holdout_meta["gap_true"].corr(holdout_meta["empirical_coherence_true"]))

tail_n1 = float(holdout_meta["tail_rel_n1_true"].mean())
tail_n2 = float(holdout_meta["tail_rel_n2_true"].mean())
tail_n3 = float(holdout_meta["tail_rel_n3_true"].mean())
tail_b1 = float(holdout_meta["tail_rel_n1_bound"].mean())
tail_b2 = float(holdout_meta["tail_rel_n2_bound"].mean())
tail_b3 = float(holdout_meta["tail_rel_n3_bound"].mean())

q1 = float(holdout_pred["empirical_coherence_true"].quantile(1.0 / 3.0))
q2 = float(holdout_pred["empirical_coherence_true"].quantile(2.0 / 3.0))
bucket_rows = []
for label, mask in [
    ("Low coherence", holdout_pred["empirical_coherence_true"] <= q1),
    ("Mid coherence", (holdout_pred["empirical_coherence_true"] > q1) & (holdout_pred["empirical_coherence_true"] <= q2)),
    ("High coherence", holdout_pred["empirical_coherence_true"] > q2),
]:
    group = holdout_pred.loc[mask]
    row = {"bucket": label}
    for method in method_order:
        value = float(group.loc[group["method"] == method, "radius_abs_error"].mean())
        row[method] = fmt(value, 3)
    bucket_rows.append(row)
bucket_df = pd.DataFrame(bucket_rows)

summary_rows = []
for _, row in summary_df.iterrows():
    summary_rows.append(
        {
            "method": html.escape(str(row["method"])),
            "radius_mae": fmt(row["radius_mae"], 3),
            "ci": f"{fmt(row['radius_mae_ci_lo'], 3)} to {fmt(row['radius_mae_ci_hi'], 3)}",
            "median_error": fmt(row["radius_median_abs_error"], 3),
            "ring_mse": fmt(row["ring_rel_mse_mean"], 3),
            "gamma_mae": fmt(row["gamma_mae"], 3),
        }
    )
summary_table = table_html(
    pd.DataFrame(summary_rows),
    [
        ("method", "Method"),
        ("radius_mae", "Radius MAE (px)"),
        ("ci", "90% bootstrap CI"),
        ("median_error", "Median abs. error (px)"),
        ("ring_mse", "Mean ring rel. MSE"),
        ("gamma_mae", "Gamma MAE"),
    ],
)

bucket_table = table_html(
    bucket_df,
    [
        ("bucket", "Empirical coherence band"),
        ("Amplitude heuristic", "Amplitude heuristic"),
        ("Monolithic ring model", "Monolithic ring"),
        ("Two-subring model", "Two-subring"),
        ("Four-subring model", "Four-subring"),
    ],
)

tuned_lambda_lines = [
    f"{method}: {value:g}"
    for method, value in benchmark_summary["tuned_lambdas"].items()
]
tuned_lambda_html = "<br />".join(html.escape(line) for line in tuned_lambda_lines)

html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Coherence + Subring Validation Report</title>
  <style>
    :root {{
      --bg: #f5f3ef;
      --panel: rgba(255, 255, 255, 0.84);
      --panel-strong: #ffffff;
      --ink: #1c2235;
      --muted: #5c667d;
      --line: #d9d3ca;
      --accent: #df6b42;
      --accent-2: #2f9c94;
      --accent-3: #223b53;
      --hero: radial-gradient(circle at top left, rgba(255, 164, 88, 0.32), transparent 38%), radial-gradient(circle at 86% 18%, rgba(51, 92, 122, 0.42), transparent 26%), linear-gradient(140deg, #0b1120, #17253b 52%, #24151d 100%);
      --shadow: 0 24px 60px rgba(20, 24, 40, 0.10);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background:
        radial-gradient(circle at top, rgba(255, 179, 92, 0.12), transparent 24%),
        linear-gradient(180deg, #fbfaf7 0%, var(--bg) 40%, #f2efe8 100%);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", "Trebuchet MS", sans-serif;
      line-height: 1.6;
      overflow-x: hidden;
    }}

    a {{
      color: var(--accent-3);
    }}

    img {{
      max-width: 100%;
      height: auto;
    }}

    code {{
      overflow-wrap: anywhere;
      word-break: break-word;
    }}

    .shell {{
      width: min(1240px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 28px 0 80px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      padding: 42px 42px 36px;
      border-radius: 28px;
      background: var(--hero);
      color: #f8f8fc;
      box-shadow: var(--shadow);
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -8% -18% auto;
      width: 360px;
      height: 360px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(255, 179, 92, 0.34), rgba(255, 179, 92, 0.10) 42%, transparent 70%);
      filter: blur(4px);
      pointer-events: none;
    }}

    .eyebrow {{
      display: inline-block;
      margin-bottom: 14px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.10);
      color: #ffd0ae;
      font-size: 0.80rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    h1, h2, h3 {{
      margin: 0;
      font-family: Georgia, "Palatino Linotype", serif;
      line-height: 1.15;
    }}

    h1 {{
      max-width: 12ch;
      font-size: clamp(2.3rem, 5vw, 4.3rem);
      letter-spacing: -0.035em;
    }}

    h2 {{
      font-size: clamp(1.5rem, 2vw, 2.1rem);
      margin-bottom: 14px;
    }}

    h3 {{
      font-size: 1.08rem;
      margin-bottom: 8px;
    }}

    .hero-grid {{
      display: grid;
      grid-template-columns: 1.18fr 0.82fr;
      gap: 28px;
      align-items: end;
    }}

    .hero-copy p {{
      max-width: 66ch;
      margin: 16px 0 0;
      color: rgba(248, 248, 252, 0.88);
      font-size: 1.02rem;
    }}

    .hero-notes {{
      display: grid;
      gap: 14px;
    }}

    .hero-note {{
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.09);
      border: 1px solid rgba(255, 255, 255, 0.10);
      backdrop-filter: blur(8px);
    }}

    .hero-note strong {{
      display: block;
      color: #ffd0ae;
      font-size: 0.84rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}

    .section {{
      margin-top: 30px;
      padding: 28px 30px 30px;
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid rgba(196, 188, 176, 0.62);
      box-shadow: 0 14px 36px rgba(34, 42, 56, 0.07);
      backdrop-filter: blur(6px);
    }}

    .section p {{
      margin: 0 0 12px;
      color: var(--muted);
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}

    .card {{
      padding: 18px 18px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,248,244,0.90));
      border: 1px solid rgba(213, 206, 196, 0.9);
    }}

    .card .label {{
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .card .value {{
      font-size: 1.7rem;
      line-height: 1.05;
      color: var(--accent-3);
      font-weight: 700;
    }}

    .card p {{
      margin: 8px 0 0;
      font-size: 0.94rem;
    }}

    .split {{
      display: grid;
      grid-template-columns: 0.98fr 1.02fr;
      gap: 26px;
      align-items: start;
    }}

    .equation-box {{
      margin-top: 14px;
      padding: 18px 20px;
      border-radius: 18px;
      background: #fffdf9;
      border: 1px solid #e6ddd0;
      overflow-x: auto;
      max-width: 100%;
      -webkit-overflow-scrolling: touch;
    }}

    .equation {{
      margin: 0 0 12px;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 0.98rem;
      color: #20283e;
      white-space: nowrap;
    }}

    .equation:last-child {{
      margin-bottom: 0;
    }}

    .figure-grid {{
      display: grid;
      grid-template-columns: 1.08fr 0.92fr;
      gap: 18px;
    }}

    .figure-grid-3 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 18px;
    }}

    .figure-card {{
      margin: 0;
      padding: 16px;
      border-radius: 20px;
      background: var(--panel-strong);
      border: 1px solid #e6ddd0;
    }}

    .figure-card-large {{
      padding: 18px;
    }}

    .figure-card img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid #ece6dc;
      background: #f7f2ea;
    }}

    figcaption {{
      margin-top: 12px;
    }}

    figcaption p {{
      margin: 0;
      font-size: 0.95rem;
    }}

    .table-wrap {{
      overflow-x: auto;
      border-radius: 18px;
      border: 1px solid #e6ddd0;
      background: rgba(255, 255, 255, 0.92);
      max-width: 100%;
      -webkit-overflow-scrolling: touch;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}

    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #ede7dc;
      text-align: left;
      font-size: 0.95rem;
    }}

    th {{
      background: #f8f4ee;
      color: #2b3551;
      font-weight: 700;
    }}

    tr:last-child td {{
      border-bottom: 0;
    }}

    .list {{
      margin: 12px 0 0;
      padding-left: 18px;
      color: var(--muted);
    }}

    .list li {{
      margin: 0 0 8px;
    }}

    .code-block {{
      margin-top: 14px;
      padding: 16px 18px;
      border-radius: 18px;
      background: #121a29;
      color: #e8edf8;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 0.92rem;
      overflow-x: auto;
      white-space: pre;
      max-width: 100%;
      -webkit-overflow-scrolling: touch;
    }}

    .kicker {{
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 0.78rem;
      margin-bottom: 8px;
    }}

    .footnote {{
      margin-top: 12px;
      font-size: 0.9rem;
      color: #6a7387;
    }}

    @media (max-width: 1040px) {{
      .hero-grid,
      .cards,
      .split,
      .figure-grid,
      .figure-grid-3 {{
        grid-template-columns: 1fr;
      }}

      .shell {{
        width: min(100vw - 22px, 1000px);
      }}

      .hero {{
        padding: 28px 22px 24px;
      }}

      .section {{
        padding: 22px 18px 22px;
      }}
    }}

    @media (max-width: 720px) {{
      .shell {{
        width: min(100vw - 14px, 1000px);
        padding: 14px 0 48px;
      }}

      .hero {{
        padding: 22px 16px 20px;
        border-radius: 22px;
      }}

      .section {{
        margin-top: 18px;
        padding: 18px 14px 18px;
        border-radius: 20px;
      }}

      .hero-note,
      .card,
      .figure-card,
      .figure-card-large,
      .equation-box,
      .code-block {{
        padding-left: 14px;
        padding-right: 14px;
      }}

      .card .value {{
        font-size: 1.5rem;
      }}

      .equation {{
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
        font-size: 0.9rem;
        line-height: 1.5;
      }}

      .code-block {{
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        word-break: break-word;
        font-size: 0.84rem;
      }}

      table {{
        min-width: 620px;
      }}

      th, td {{
        padding: 10px 10px;
        font-size: 0.88rem;
      }}

      .list {{
        padding-left: 16px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <span class="eyebrow">Separate Validation Suite</span>
          <h1>Coherence + Subring Validation Report</h1>
          <p>
            This report adds a second synthetic benchmark without touching the original
            baseline experiments or the current project landing page. The new suite
            combines a designed coherence dial with a true subring-resolved signal tower
            so the newer provenance/coherence and subring mathematics can be tested in a
            controlled, reproducible setting.
          </p>
        </div>
        <div class="hero-notes">
          <div class="hero-note">
            <strong>Dataset</strong>
            {dataset_config['n_tune'] + dataset_config['n_holdout']} images at {dataset_config['image_size']} x {dataset_config['image_size']},
            split into {dataset_config['n_tune']} tune and {dataset_config['n_holdout']} holdout cases with
            {dataset_config['n_subrings_true']} true subrings and {len(dataset_config['gap_levels'])} designed gap levels.
          </div>
          <div class="hero-note">
            <strong>Main Result</strong>
            Two-subring and four-subring models cut held-out radius error by
            {two_vs_mono:.1f}% and {four_vs_mono:.1f}% versus the monolithic structured model.
          </div>
          <div class="hero-note">
            <strong>Scope</strong>
            The original baseline pipeline and writeup remain unchanged. Everything below lives under
            <code>coherence_subring_validation/</code>.
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="kicker">Executive Summary</div>
      <h2>What this new suite establishes</h2>
      <p>
        The coherence dial works as intended: widening the designed gap reduces empirical
        ring-background overlap, with a hold-out gap/coherence correlation of {fmt(gap_corr, 3)} and an exponential fit
        coefficient of beta = {fmt(beta_hat, 2)}. The subring extension also matters in estimation:
        adding explicit subring structure nearly halves geometry error relative to both the heuristic
        baseline and the monolithic structured model.
      </p>
      <div class="cards">
        <article class="card">
          <span class="label">Geometry</span>
          <div class="value">{fmt(two_mae, 3)} px</div>
          <p>Best hold-out radius MAE from the two-subring model, a {two_vs_mono:.1f}% reduction versus monolithic.</p>
        </article>
        <article class="card">
          <span class="label">Reconstruction</span>
          <div class="value">{fmt(four_ring_mse, 3)}</div>
          <p>Lowest mean relative ring MSE from the four-subring model, improving on monolithic by {four_recon_gain:.1f}%.</p>
        </article>
        <article class="card">
          <span class="label">Coherence Bound</span>
          <div class="value">{pct(bound_coverage)}</div>
          <p>Held-out samples satisfy the weighted subring upper bound on aggregate coherence in every case.</p>
        </article>
        <article class="card">
          <span class="label">Truncation</span>
          <div class="value">{pct(1.0 - tail_n3)}</div>
          <p>On average, the first three subrings retain about {pct(1.0 - tail_n3)} of the total ring norm.</p>
        </article>
      </div>
    </section>

    <section class="section split">
      <div>
        <div class="kicker">Synthetic Design</div>
        <h2>How the validation dataset is built</h2>
        <p>
          Each image contains a black-hole-like ring signal plus structured nuisance emission. The ring is not a single
          template. It is generated as a sum of true subrings with exponentially decaying weights, then blurred and
          corrupted by additive noise. The nuisance class combines a broad crescent, a near-critical shell whose leakage
          weight depends on the designed gap, and diffuse blobs.
        </p>
        <p>
          This lets the suite test two questions separately from the original baseline. First, does ring-background overlap
          fall as the designed gap widens? Second, does a subring-aware estimator recover geometry and morphology better
          than a single-ring approximation?
        </p>
        <ul class="list">
          <li>True subrings per image: {dataset_config['n_subrings_true']}</li>
          <li>Designed gap levels: {", ".join(f"{x:.2f}" for x in dataset_config['gap_levels'])}</li>
          <li>Tuned regularization strengths: {tuned_lambda_html}</li>
        </ul>
      </div>
      <div>
        <div class="kicker">Mathematical Bridge</div>
        <h2>Equations encoded by the suite</h2>
        <div class="equation-box">
          <div class="equation">y = &Sigma;<sub>n=1</sub><sup>N</sup> &alpha;<sub>1</sub> exp(-&gamma;(n-1)) g<sub>&theta;,n</sub> + q + &epsilon;</div>
          <div class="equation">&mu;(g, q) = |&lang;g, q&rang;| / (||g|| ||q||)</div>
          <div class="equation">&mu;(&Sigma;<sub>n</sub> &alpha;<sub>n</sub> g<sub>n</sub>, q) &le; (&Sigma;<sub>n</sub> |&alpha;<sub>n</sub>| ||g<sub>n</sub>|| &mu;(g<sub>n</sub>, q)) / ||&Sigma;<sub>n</sub> &alpha;<sub>n</sub> g<sub>n</sub>||</div>
          <div class="equation">||&Sigma;<sub>n&gt;N</sub> &alpha;<sub>n</sub> g<sub>n</sub>|| &le; &alpha;<sub>1</sub> c<sub>max</sub> exp(-&gamma;N) / (1 - exp(-&gamma;))</div>
        </div>
        <p class="footnote">
          The suite is still operator-lite and synthetic. It is not a full ray-traced geodesic transport benchmark. The value
          here is that the later inequalities are instantiated directly and reproducibly, rather than only argued heuristically.
        </p>
      </div>
    </section>

    <section class="section">
      <div class="kicker">Visual Evidence</div>
      <h2>Representative images and signal structure</h2>
      <div class="figure-grid">
        {img_card(figure_rel_path("figure_01_gap_cases.png").as_posix(), "Representative held-out gap cases", "Across low, medium, and high designed-gap cases, the aggregate ring stays recoverable even as the nuisance field becomes more confounding. The four-subring estimator keeps the ring geometry crisp across the grid.", large=True)}
        {img_card(figure_rel_path("figure_02_subring_tower.png").as_posix(), "True subring-resolved tower", "A single held-out example showing the aggregate image and the four latent subrings that generate it. The later subrings are weaker but still non-negligible, which is why a monolithic one-ring model loses fidelity.")}
      </div>
    </section>

    <section class="section">
      <div class="kicker">Quantitative Comparison</div>
      <h2>Accuracy and reconstruction quality by method</h2>
      <p>
        The heuristic amplitude baseline and the monolithic structured model behave similarly on radius recovery. Adding explicit
        subring structure is the decisive change. The two-subring model gives the lowest radius MAE, while the four-subring model
        is slightly less sharp on radius but substantially better at reconstructing the full ring morphology.
      </p>
      <div class="figure-grid">
        {img_card(figure_rel_path("figure_03_method_comparison.png").as_posix(), "Held-out method comparison", "Subring-aware estimators cut geometry error roughly in half. The four-subring variant produces the cleanest ring reconstruction even though its radius MAE is only marginally above the two-subring model.", large=True)}
        <div>
          {summary_table}
          <p class="footnote">
            Relative to the heuristic baseline, the two-subring and four-subring models improve hold-out radius MAE by
            {two_vs_amp:.1f}% and {four_vs_amp:.1f}%, respectively.
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="kicker">Coherence Validation</div>
      <h2>Where the coherence claims show up in data</h2>
      <div class="figure-grid">
        {img_card(figure_rel_path("figure_04_error_vs_coherence.png").as_posix(), "Error versus empirical coherence", "Every method degrades as ring-background coherence rises, but the subring-aware models degrade far more gracefully. In the highest empirical-coherence third, they still stay near half-pixel mean radius error.", large=True)}
        {img_card(figure_rel_path("figure_05_gap_and_bound.png").as_posix(), "Gap decay and weighted bound", "The left panel shows the intended monotone trend from designed gap to empirical overlap. The right panel shows the weighted subring bound sitting above the aggregate coherence on every held-out case.")}
      </div>
      <p>
        On the hold-out split, the designed gap and empirical coherence correlate at {fmt(gap_corr, 3)}. The fitted decay curve
        has scale {fmt(scale_hat, 2)} and beta {fmt(beta_hat, 2)}. In the highest empirical-coherence third, the mean radius
        errors are still {bucket_df.loc[bucket_df['bucket'] == 'High coherence', 'Two-subring model'].iloc[0]} px for the
        two-subring model and {bucket_df.loc[bucket_df['bucket'] == 'High coherence', 'Four-subring model'].iloc[0]} px for the
        four-subring model, versus {bucket_df.loc[bucket_df['bucket'] == 'High coherence', 'Monolithic ring model'].iloc[0]} px
        for the monolithic model.
      </p>
      {bucket_table}
    </section>

    <section class="section">
      <div class="kicker">Finite Truncation</div>
      <h2>Only a few leading subrings are needed</h2>
      <div class="figure-grid">
        {img_card(figure_rel_path("figure_06_truncation_curve.png").as_posix(), "Average truncation behavior", "The true residual tail falls rapidly as more leading subrings are kept. The geometric envelope stays conservative and decays in the same direction.", large=True)}
        <div>
          <p>
            The mean remaining tail fraction is {pct(tail_n1)} after keeping only the leading subring, {pct(tail_n2)} after keeping
            two, and {pct(tail_n3)} after keeping three. The corresponding geometric envelopes are {pct(tail_b1)}, {pct(tail_b2)},
            and {pct(tail_b3)}.
          </p>
          <p>
            This is the main practical reason the two-subring model already performs so well on geometry: it captures the most
            important correction beyond the monolithic ring. The four-subring model still matters because it recovers the full ring
            field much more faithfully, which is what the reconstruction-MSE panel highlights.
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <div>
        <div class="kicker">Reproducibility</div>
        <h2>How to rerun the suite</h2>
        <p>
          The new validation path is self-contained. It does not modify the existing baseline datasets, baseline tuning outputs,
          or the current main landing page.
        </p>
        <div class="code-block">python coherence_subring_validation\\run_validation_suite.py</div>
        <p>Equivalent step-by-step commands:</p>
        <div class="code-block">python coherence_subring_validation\\01_generate_coherence_subring_dataset.py
python coherence_subring_validation\\02_run_coherence_subring_benchmark.py
python coherence_subring_validation\\03_build_coherence_subring_report.py</div>
      </div>
    </section>
  </main>
</body>
</html>
"""

REPORT_PATH.write_text(html_report, encoding="utf-8")

print("\nBuilt the coherence + subring validation report.")
print(f"Report path: {REPORT_PATH}")
print("The report references the generated figure PNGs with relative paths.")
