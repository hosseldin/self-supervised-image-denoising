
# C17: Self-Supervised Real-World Image Denoising Without Paired Data

## **CSE681 Advanced Digital Image Processing Team Project**

Implementation and evaluation of self-supervised image denoising on real-world sensor noise (SIDD), compared against classical filtering and, if time allows, a supervised baseline.

### **Current status: Milestone 1**

The classical baselines (BM3D, NLM) are implemented,
calibrated and scored. No self-supervised model is trained yet — that is M2.

#### **Objectives**

- Implement a self-supervised denoiser (no clean/noisy pairs required).
- Evaluate on real sensor noise vs. a classical filter and a supervised baseline.

**Compute tier:** GPU-light (M1 ran entirely on CPU).

---

## Results (M1)

Mean over **156 held-out test crops** from 2 scenes never used for calibration.
Reproduced by running [`notebooks/m1_baselines.ipynb`](notebooks/m1_baselines.ipynb).

| Method | Parameter | PSNR (dB) | SSIM | s/crop |
| --- | --- | --- | --- | --- |
| Noisy input | — | 28.166 | 0.5096 | — |
| **BM3D** | `sigma_psd` = 0.20 | **38.440** | **0.9360** | 5.12 |
| NLM | `h` = 0.08 | 36.717 | 0.9248 | 0.18 |

BM3D improves on the noisy input by **+10.27 dB**. Both parameters are interior optima of
the search grid, not boundary selections.

Raw outputs: `results/split.json`, `results/calibration.json`, `results/test_results.json`,
`results/figures/`.

---

## Setup

Requires **Python 3.10+** (developed and run on 3.14.6, Windows 11).

```bash
git clone https://github.com/hosseldin/self-supervised-image-denoising.git  # Clone the repository
cd self-supervised-image-denoising  # Navigate to the project directory
python -m venv .venv  # Create a Python virtual environment
.\.venv\Scripts\activate  # Activate the virtual environment
pip install -r requirements.txt  # Install project dependencies
```

## Dataset

SIDD-Small sRGB (~6.2 GB) is pulled from a Kaggle mirror and cached outside the repo.

1. Run [`notebooks/store_sidd.ipynb`](notebooks/store_sidd.ipynb) — needs a free Kaggle
   account. It downloads and extracts 160 noisy/ground-truth capture pairs.
2. Run [`notebooks/m1_baselines.ipynb`](notebooks/m1_baselines.ipynb) — resolves the
   dataset path through `kagglehub`, so no path needs hardcoding.

On a TLS-inspecting corporate network the `truststore` cell is required; without it Kaggle
requests fail with `CERTIFICATE_VERIFY_FAILED`.

---

## Reproducibility

- **Single seed: `SEED = 681`**, set in the protocol-constants cell. It governs the scene
  shuffle and the calibration subsample — the only stochastic choices in M1.
- **Scene-level split.** SIDD-Small is 160 captures of 10 *physical scenes* (field 2 of
  the directory name). Captures of one scene differ only by phone, ISO and exposure, so
  splitting per capture would leak scene content into the test set. We split 6/2/2 on the
  scene:

  | Split | Scenes | Captures | Crops |
  | --- | --- | --- | --- |
  | train | 001, 003, 005, 008, 009, 010 | 91 | 364 |
  | val | 002, 004 | 30 | 120 |
  | test | 006, 007 | 39 | 156 |

  Train crops are reserved for the M2 self-supervised model and are unused in M1.
- **Cropping.** Four fixed, non-overlapping 256x256 crops per capture, centred on the
  quarter points of each axis. Deterministic — no random crop sampling.
- **Scoring.** PSNR and SSIM on clipped `[0,1]` RGB, 4-pixel border discarded, identical
  for every method. SSIM is Gaussian-weighted (11-px window, sigma 1.5, population
  covariance); SSIM values are not comparable across implementations that differ here.
- **Calibration.** Each filter's single parameter is selected by mean validation PSNR over
  `{0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24}`. Test scenes are untouched during
  calibration. Because BM3D costs ~5–8 s per crop, calibration uses a seeded 24-crop
  subsample of the 120 validation crops (`N_CALIB`); set `N_CALIB = None` to sweep all.

### Known limitations

- Only **2 held-out scenes**, so the generalization estimate is weak. Per-crop standard
  deviations are recorded in `results/test_results.json` (BM3D PSNR std 4.39 dB).
- Calibration uses a 24-crop subsample, so the selected parameters are not guaranteed
  globally optimal.
- This is a **local cropped holdout**, not the official SIDD benchmark. Numbers here are
  not comparable to published SIDD leaderboard entries.

---

## Milestone plan

| M | Wk | Student A — Data & Baseline | Student B — Model/Algorithm | Student C — Evaluation | Deliverable |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | SIDD pipeline; BM3D baseline scoped | Design self-supervised denoising objective | Define PSNR/SSIM evaluation on real noise | Proposal + BM3D baseline numbers |
| 2 | 6 | Full BM3D baseline table | Self-supervised denoiser, first checkpoint | Preliminary comparison on SIDD validation | Progress report, 1 comparison |
| 3 | 9 | Final classical + (if time) supervised baseline | Final self-supervised model | Full SIDD + DND comparison table | Final report, repo, demo |

---

## Layout

```text
notebooks/
  store_sidd.ipynb      # dataset download (run first)
  m1_baselines.ipynb    # M1 deliverable: split, metrics, BM3D/NLM, results
results/                # generated: JSON metrics, figures, LaTeX table
report/m1/              # M1 proposal (LaTeX source + bibliography)
requirements.txt
```

The notebooks are self-contained — there is no importable package to install.

---

## Dataset licence & attribution

SIDD is released under the **MIT License** (dataset and associated code repositories), per
the [official project page](https://abdokamel.github.io/sidd/). Verified 1 Nov 2026.

If you use this data, cite:

> A. Abdelhamed, S. Lin, M. S. Brown. "A High-Quality Denoising Dataset for Smartphone
> Cameras." *CVPR*, 2018.

We obtained SIDD-Small through a
[Kaggle mirror](https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset)
because the official host (`http://130.63.97.225/share/...`) was unreachable from our
network. The mirror's contents match the official `_ReadMe.txt`: 160 scene instances,
322 files, 6.16 GB.

The DND benchmark (M3) is a separate dataset with its own terms — review before use.

## AI-tool usage

Per course policy §9.1: code scaffolding for the baseline notebook was AI-assisted. All
reported numbers were produced by executing `notebooks/m1_baselines.ipynb` in this
repository and are reproducible from it.
