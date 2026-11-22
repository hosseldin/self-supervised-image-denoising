
# C17: Self-Supervised Real-World Image Denoising Without Paired Data

## **CSE681 Advanced Digital Image Processing Team Project**

Implementation and evaluation of self-supervised image denoising on real-world sensor noise (SIDD), compared against classical filtering and, if time allows, a supervised baseline.

### Current status: M1 and M2 classical baselines

This branch contains the M1 baseline experiment and the M2
validation results. In the M2 run, BM3D and NLM were
tested on the same 120 SIDD validation crops, using the
settings selected during M1.

BM3D gave the higher average PSNR: 38.149 dB, compared with
36.527 dB for NLM. NLM was much faster, so keeping both
methods helps show the trade-off between quality and speed.

The M1 test results and M2 validation results are reported
separately because they use different image splits.

The two M2 runs use the same validation crops and filter settings.
The earlier run recorded four CPU threads; the later Colab run requested two
through its runtime and environment settings. BM3D retained its default native
profile in both runs. Timings depend on the machine and library implementation,
so the difference between runs is not evidence of a change in filter quality.

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
| **BM3D** | `sigma_psd` = 0.20 | **38.440** | **0.9360** | 5.05 |
| NLM | `h` = 0.08 | 36.717 | 0.9248 | 0.19 |

BM3D improves on the noisy input by **+10.27 dB**. Both parameters are interior optima of
the search grid, not boundary selections.

Raw outputs: `results/split.json`, `results/calibration.json`, `results/test_results.json`,
`results/figures/`.

---

## Setup for the M1 notebooks

Requires **Python 3.10+** (developed and run on 3.14.6, Windows 11).

```bash
git clone --branch menna-m2-review https://github.com/hosseldin/self-supervised-image-denoising.git  # Clone this review branch
cd self-supervised-image-denoising  # Navigate to the project directory
python -m venv .venv  # Create a Python virtual environment
.\.venv\Scripts\activate  # Activate the virtual environment
pip install -r requirements.txt  # Install project dependencies
```

## Checking the saved M2 results

From the repository folder, run:

```bash
python scripts/review_mennatalla_run.py
```

This checks the later Colab results in `results/mennatalla_m2_2026_11_22/`.
For the earlier M2 run in `results/m2_baselines/`, use:

```bash
python scripts/review_m2_baselines.py
```

Both scripts use Python's standard library. They verify the saved measurements;
they do not run the filters or train a model.
Open `notebooks/m2_baseline_review.ipynb` to review and plot the earlier run
(Matplotlib is needed for its chart).

The review notebook depends on the repository's scripts and result files, so
keep their folder structure intact. To run the filters on images, use the M2
runner below.

## Run the M2 baselines on images

This runner evaluates noisy input, BM3D and NLM on the fixed 120 validation
crops. It needs no CNN checkpoint or PyTorch installation. The filter calls,
float32 image handling and RGB PSNR/SSIM formula match the recorded M2 pipeline.
The metric function is shared with the complete project's evaluation code.

Use Python 3.12 or 3.13. From the repository root on Windows:

```bat
python -m venv .venv-m2
.venv-m2\Scripts\python.exe -m pip install -r requirements-m2.txt
```

On Linux/macOS use `.venv-m2/bin/python` instead. `requirements-m2.txt` defines
the baseline-only environment; the original `requirements.txt` is for M1.
Historical Colab versions are preserved in
`results/mennatalla_m2_2026_11_22/environment_and_protocol.json` (Python 3.13.18,
NumPy 2.4.6). The new installation uses NumPy 2.3.5 with the same BM3D, BM4D,
scikit-image and SciPy versions. New runs record their actual versions and
timings; small numerical differences and larger timing differences are expected.

### Prepare the input images

Choose one input route. Image data are kept outside Git history.

**Existing validation ZIP:** if you have `Mennatalla_M2_Run_Data.zip` from the
shared project package, import it with:

```bat
.venv-m2\Scripts\python.exe scripts\run_m2_baselines.py prepare --validation-zip "PATH\TO\Mennatalla_M2_Run_Data.zip" --output data\m2
```

The importer verifies the ZIP's recorded SHA-256 and copies only its 120 noisy
and clean validation pairs, manifest and provenance. It ignores the code in
that archive. The manifest still describes all splits; this ZIP contains only
the validation images.

**From SIDD-Small:** download and extract the sRGB dataset from the
[Kaggle mirror](https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset).
Choose the `Data` directory containing the 160 capture folders, then run:

```bat
.venv-m2\Scripts\python.exe scripts\run_m2_baselines.py prepare --sidd-dir "PATH\TO\SIDD_Small_sRGB_Only\Data" --output data\m2
```

This recreates the fixed 364/120/156 crop split. Training rows have no clean
targets. Input file hashes identify the downloaded copy; they are not a claim
of verification against an official author-published checksum. Both prepare
commands require a new output folder.

### Run and inspect the results

First check one crop, then run all 120 in a different output folder:

```bat
.venv-m2\Scripts\python.exe scripts\run_m2_baselines.py run --data data\m2 --output runs\m2_smoke_01 --limit 1
.venv-m2\Scripts\python.exe scripts\run_m2_baselines.py run --data data\m2 --output runs\m2_full_01
```

The default requests two threads via OMP/MKL/OpenBLAS before importing native
libraries and retains BM3D's default `np` profile. This is not a measurement of
actual native worker counts. Add `--threads 4` to request a different limit.
BM3D may take many minutes on a CPU. Keep the process running until `COMPLETE`.
Open CSV files in spreadsheet applications after the run completes; some Windows
applications hold an exclusive lock on files they open.
Interrupted runs retain per-crop snapshots, but this runner does not resume;
choose a new output folder to restart.

Open `comparison.csv`, `validation_comparison.png` and `before_after.png` in the
new output folder. Per-crop CSV files, summaries, input hashes, code hashes and
environment metadata are saved alongside them. `--limit` runs are labelled
`smoke_test` and must not be reported as the full validation experiment. The
first crop is selected by manifest order, not by its score. Image scores are
per-crop; the comparison table contains means across the evaluated crops.
Time includes the filter, output conversion and clipping, but excludes image
loading, metric calculation and file writing. Existing result folders are
never overwritten.

For Google Colab, open
[`notebooks/m2_baselines_run.ipynb`](notebooks/m2_baselines_run.ipynb),
or use [Open in Colab](https://colab.research.google.com/github/hosseldin/self-supervised-image-denoising/blob/menna-m2-review/notebooks/m2_baselines_run.ipynb).
It runs the same CLI in a separate process and accepts the validation-data ZIP.
In Colab it connects Drive by default and checks write/read access before saving
new runs under `MyDrive/C17_M2_Baseline_Runs`. Setting `SAVE_TO_DRIVE = False`
uses temporary storage instead. Download the results ZIP and executed notebook
as separate copies; a visible notebook output alone does not preserve its files.

Basic correctness and input checks can be run with:

```bat
.venv-m2\Scripts\python.exe -m unittest discover -s tests -v
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
  store_sidd.ipynb          # download SIDD for the M1 experiment
  m1_baselines.ipynb        # M1 calibration and held-out test evaluation
  m2_baseline_review.ipynb  # review the earlier saved M2 validation run
  m2_baselines_run.ipynb    # execute a new baseline run locally or in Colab
scripts/
  review_m2_baselines.py    # check the earlier M2 measurements
  review_mennatalla_run.py  # check the later Colab measurements
  run_m2_baselines.py      # prepare images and run the fixed CPU baselines
src/c17/data.py             # prepare the fixed scene split and image crops
src/c17/baseline_data.py    # import validation images or crop extracted SIDD
src/c17/baselines.py        # BM3D/NLM evaluation and output figures
src/c17/metrics.py          # unchanged shared RGB PSNR/SSIM formula
results/
  m2_baselines/            # earlier M2 validation results
  mennatalla_m2_2026_11_22/ # later Colab validation results
  verification/            # manifest and source information used by the checks
  figures/                 # M1 figures
report/m1/                 # M1 proposal, source and bibliography
requirements.txt
requirements-m2.txt        # separate baseline-only environment
```

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

Per course policy §9.1: code scaffolding for the baseline notebook was AI-assisted.

The M1 results come from `notebooks/m1_baselines.ipynb`. The M2 validation
results are saved separately in `results/m2_baselines/` and
`results/mennatalla_m2_2026_11_22/`. The two review scripts check those saved
measurements; they do not rerun denoising.

The standalone M2 runner and its setup documentation were also prepared with
AI assistance. New runs are separate from the recorded student Colab run.
