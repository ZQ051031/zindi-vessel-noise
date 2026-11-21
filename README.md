# Underwater Vessel Noise Challenge: reproducible acoustic modelling

An open-source Python solution for the Zindi Underwater Vessel Noise Challenge, with tolerant release loading, physically motivated features, controlled-addition training, grouped validation, inference, uncertainty estimates, explanations and pressure-proxy maps.

> **SYNTHETIC EXAMPLE — NOT REAL RESULTS.** All committed quantitative results and figures use invented backgrounds and parametric synthetic passage references. They are **not competition scores**, evidence of natural vessel detection, or calibrated acoustic-pressure measurements. The research PDF is a synthetic development report with explicitly marked real-data placeholders.

## HARD CONSTRAINTS

- This repository/agent did **not** sign up, enrol, download competition data or submit anything. Those are the user's steps, subject to the competition terms.
- No competition data, real station results, AIS or external acoustic corpora were used. No ONC or HearMyShip recordings were used. References in this version are **parametric synthetic templates**, not recorded vessel passages.
- Test inputs are **inference only**. Never fit, calibrate, normalise using population statistics, pseudo-label or train on `Test.csv`. Fixed independent per-window features are allowed.
- Synthetic labels describe the *controlled addition*, not all naturally occurring vessel energy. Unlabelled real backgrounds are **not verified negatives**.
- Open-source tools only; no AutoML. Seed 42 is the default. Never claim synthetic results as real South African results.

## Current data interface (important)

The live Data tab differs from older descriptions: **there is no `south_africa.csv` or labelled `Train.csv`**. The authoritative release is:

```text
South_African_Real_Acoustics.zip
  acoustics/<site>.parquet
  stations.csv
  recorders.csv
  DATA_DICTIONARY.md
Test.csv
sample_submission.csv
```

Training comprises 1,715,779 unlabelled, generally one-minute frequency-band records from ten South African sites (SAEON/SANBI/SA URN). Parquet files contain identifiers, export time `t`, observed duration `dur`, `TOL_<frequency>` bands and quality flags. Coverage varies by recorder. Read the release dictionary before choosing filters or interpreting timestamps and coordinates.

`Test.csv` has `ID` plus **375 columns**, e.g. `m01_TOL_31.6`: 15 minutes × 25 bands, ordered minute then frequency. Band centres follow `10**(1.5 + k/10)`, k=0…24 (31.6 Hz–7.94 kHz). These are deci-decade centres, not the outer frequency edges. Exports are dB with unverified absolute calibration. **Blank/NaN means missing, never silence.** There are no station IDs, absolute timestamps or AIS in test windows. The parser accepts numeric frequency suffixes with a 3% relative tolerance; malformed, duplicate or unsupported frequencies fail clearly. Suffixes such as `1k` are deliberately rejected rather than guessed.

Target: sum of known added-reference linear energy divided by sum of mixture linear energy over the **same observed cells**, with energy `10**(dB/10)`. The leaderboard is `1 − RMSE` on `share_added_vessel`. `score_added_vessel` is continuous diagnostic evidence evaluated with ROC AUC, not a scored share or a claim about natural vessel presence.

## Architecture

```text
release ZIP/directory ──> duration/QC screening ──> site-minute grid
                                                   │
                                         15-minute backgrounds
                                                   │
                           synthetic CPA references + known energy scaling
                                                   │
                               per-window features + background IDs
                                                   │
                      held-out-site/period CV ──> OOF metrics + cluster CIs
                                                   │
                   baseline / isotonic / Ridge / LightGBM + presence model
                                                   │
                               NNLS ensemble + OOF score calibration
                                                   │
                                        frozen joblib bundle
                                                   │
Test.csv ──> independent per-window features ──> predict ──> submission.csv

release + frozen bundle ──> total-sound / inferred-share maps (not measured shares)
training features + bundle ──> gain / coefficients / TreeSHAP contributions
```

Ridge uses training-only median imputation, missing indicators and standardisation. LightGBM uses one thread, `deterministic=True`, `force_row_wise=True` and seeded randomness. NNLS weights are nonnegative and normalised; optional presence gating is selected on OOF only. Classifier scores use OOF isotonic calibration, with an excess-based fallback. Final predictions are clipped, not rounded or thresholded. Ensemble OOF metrics are *selection estimates*, not nested-CV estimates.

## Repository layout

```text
src/schema.py              Test.csv parsing and canonical column layout
src/data.py                ZIP/directory loader, metadata and regular windows
src/synthetic.py           explicitly synthetic release, backgrounds and additions
src/features.py            NaN-aware per-window acoustic features
src/models.py              deterministic estimators and bundle inference
src/train.py               augmentation, grouped CV, OOF outputs and final fitting
src/predict.py             submission validation, runtime and peak process RSS
src/evaluate.py            RMSE, AUC, IID/cluster bootstrap and share-bin errors
src/pressure_map.py        total-sound proxies, coverage and inferred-share maps
src/explain.py             LightGBM gain, Ridge coefficients, contribution summaries
src/ablation.py            site-held-out spectral/temporal/energy controls
src/run_synthetic_demo.py  one-command local development pipeline
scripts/make_figures.py    committed synthetic figures/tables regeneration
scripts/build_writeup.sh   metadata generation and two-pass pdflatex build
notebooks/solution.ipynb   end-to-end configurable notebook, outputs cleared
tests/test_solution.py     small synthetic-only regression/integration tests
writeup/                   LaTeX, PDF, committed figures and source-number tables
data/, models/, outputs/   user data and generated artifacts (gitignored)
```

## Setup

Python **3.10+**; this version was verified with system Python 3.10.12. On Linux, a working OpenMP runtime is needed for LightGBM. Run commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Exact direct pins are in `requirements.txt`: NumPy 1.26.4, pandas 2.2.3, pyarrow 18.1.0, scikit-learn 1.5.2, LightGBM 4.5.0, SciPy 1.14.1, Matplotlib 3.9.3, joblib 1.4.2, pytest 8.3.4, nbformat 5.10.4, nbconvert 7.16.4, ipykernel 6.29.5 and pyparsing 3.2.0. Transitive dependencies are not fully locked; exact platform/library reproduction should retain the installed environment as well.

## Quick synthetic demo

```bash
python -m src.run_synthetic_demo
# Larger fixed-seed development experiment and all committed write-up figures:
python scripts/make_figures.py
```

The first command writes `outputs/synthetic_demo/`. The second runs four synthetic sites over two days, four augmentation draws per background and 600 held-out synthetic examples, then writes `writeup/figures/` and `writeup/tables/`. Generated release folders carry a `SYNTHETIC` marker. Backgrounds include biology, natural passage-like confounders, missingness and ±3 dB window gains; reference ambient floors vary. This is a development stress test, not a faithful reconstruction of the organisers' recorded references.

The committed larger run took 20.72 s for training, 38.64 s for the full demo and 50.38 s including figure/table generation on an 8-vCPU VM. Inference for 600 examples took 1.811 s (peak process RSS 296.6 MiB); the figure workflow's maximum per-process RSS was 336.77 MiB. The bundle was 644,824 bytes. These are observations, not performance guarantees; consult `writeup/tables/metrics_summary.json` for source values. The synthetic held-out score was 0.96088 and AUC 0.99207; **neither is a competition result**.

## Real data: user downloads first

After obtaining the release under the competition rules, place exactly these files in `data/`:

```text
data/South_African_Real_Acoustics.zip
data/Test.csv
data/sample_submission.csv
```

An extracted directory containing `acoustics/` and the supporting CSVs is also accepted by `--release`. Nothing here downloads or enrols automatically.

```bash
mkdir -p outputs/real
python -m src.train \
  --release data/South_African_Real_Acoustics.zip \
  --out-dir models/real --augment-k 3 --seed 42
python -m src.predict \
  --test data/Test.csv --model models/real/bundle.joblib \
  --sample data/sample_submission.csv --out outputs/real/submission.csv
```

This trains on **real backgrounds plus synthetic additions**. Its validation labels are still synthetic and must not be presented as verified natural vessel labels. Actual full-release runtime and memory have not been measured; feature generation and multiple augmentation draws can be substantial. Inspect coverage/QC and use a user-prepared development subset when appropriate; this implementation is in-memory rather than a streaming trainer.

### Column overrides, quality flags and periods

Automatic detection supports common site/time/duration names and numeric `TOL_*` frequencies. Missing bands remain NaN. The default duration filter retains `60 ± 5` seconds; it does not equate a quality pass with valid calibration. Flags are ignored unless requested:

```bash
python -m src.train --release data/South_African_Real_Acoustics.zip \
  --out-dir models/real_qc --site-col station_id --time-col timestamp \
  --dur-col duration_s --duration-tolerance 5 \
  --drop-bad-flags --flag-cols qc_flag quality_flag --seed 42
```

Those override names are examples, not assertions about the real schema. **Read `DATA_DICTIONARY.md` first.** With flag screening enabled, `0`, `0.0`, `false`, `good`, `ok`, `pass`, `nan` and `none` are treated as good/not marked bad; other values are dropped. Missing flag values are not evidence of quality. Default flag detection uses names containing `flag`, `qc` or `quality`. Explicit flag names avoid accidental screening. Station-coordinate overrides for mapping are `--lon-col`, `--lat-col` and `--station-site-col`.

```bash
# UTC-day-held-out periods rather than sites; requires at least two days:
python -m src.train --release data/South_African_Real_Acoustics.zip \
  --out-dir models/real_period --group-by period --seed 42
```

Default CV holds out sites. The period option groups by export UTC day, not an automatic chronological/purged split. Loader timestamps are parsed as UTC; verify the export clock against the dictionary before physical interpretation or any future AIS join. All augmentation draws from a background stay in one validation group. IID intervals and cluster intervals answer different questions; cluster intervals resample background windows or sites, retaining repeated augmentation observations. Only a few synthetic sites are available, so site-level intervals themselves are uncertain.

### Maps, explanations and ablations

```bash
python -m src.pressure_map \
  --release data/South_African_Real_Acoustics.zip \
  --model models/real/bundle.joblib --out-dir outputs/real/maps
python -m src.explain --model models/real/bundle.joblib \
  --features models/real/SYNTHETIC_addition_features.parquet \
  --out-dir outputs/real/explanations
python -m src.ablation \
  --features models/real/SYNTHETIC_addition_features.parquet \
  --oof models/real/SYNTHETIC_addition_oof.parquet \
  --out-dir outputs/real/ablations --seed 42
```

`SYNTHETIC_addition_*` filenames remain correct with real backgrounds because labels/additions are synthetic. Explanations use training features only. Family attribution sums mean absolute LightGBM TreeSHAP-style contributions; band attribution includes explicitly band-indexed features only, not an arbitrary redistribution of global/group effects. Ridge coefficients are standardised, not original-unit causal effects.

Maps sum linear energy over bands available at every station, requiring each selected common band to be observed in a minute. Relative daily levels subtract each station's own median export level. Coverage is the fraction of grid minutes with a valid broadband proxy; group-band coverage is also exported. Common-band selection is a release-map analysis, **never a test-set fitting step**. Coordinates have uncertainty; no spatial interpolation is performed. A model-inferred added-vessel-like share is not measured vessel energy or natural vessel probability.

## Research write-up and real-results placeholders

`writeup/writeup.pdf` is a six-page-style synthetic development report (the build reports the actual page count). Its **Real South African results — placeholder** section must be completed before claiming real evidence. Commands above generate:

- maps: `station_map.png`, `daily_relative.png`, `daily_inferred_share.png`, `daily_excess_energy.png`, coverage CSV;
- explanations: `contributions_by_family.png`, `contributions_by_band.png`, gain/coefficient plots and CSVs;
- controls: `ablation.csv` and `ablation.json` on controlled additions to released backgrounds;
- validation: model metrics and `SYNTHETIC_addition_oof.parquet` from training.

Keep these outputs in `outputs/real/` for review. Do **not** overwrite the committed synthetic panels and remove their disclaimers to manufacture a real-results report. Verified real-label RMSE/AUC requires an independently justified labelled evaluation, which the release does not supply. There is no automated command to create that missing ground truth. Select and document real acoustic cases, limitations and metadata before adding separate real-data panels and editing the placeholder section.

To regenerate the current synthetic report and rebuild:

```bash
python scripts/make_figures.py
bash scripts/build_writeup.sh
pdfinfo writeup/writeup.pdf
pdftotext writeup/writeup.pdf - | grep -c 'SYNTHETIC EXAMPLE'
```

For the real-data report, run the real train/maps/explain/ablation commands above, review the results, add appropriately labelled real figures/tables to the LaTeX source and then run the same build command. The build intentionally never downloads data, reruns models, or replaces synthetic results with unverified real claims. `scripts/writeup_metadata.py` reads numeric macros from committed JSON and dependency pins from `requirements.txt`.

LaTeX requirements: `pdflatex` with `geometry`, `graphicx`, `booktabs`, `hyperref`, `caption`, `float`, `mathptmx`, `courier` and `cmap` (standard TeX Live packages); `pdfinfo`/`pdftotext` from `poppler-utils` for checks. The build runs twice for references. The source uses A4, one column, 11pt Times serif text and 3 cm margins.

## Notebook and tests

```bash
python -m ipykernel install --prefix .venv --name vessel-noise
jupyter nbconvert --to notebook --execute notebooks/solution.ipynb \
  --output-dir outputs --ExecutePreprocessor.kernel_name=vessel-noise
pytest -q
```

The notebook defaults all of `RELEASE_DIR`, `TEST_CSV`, `SAMPLE_SUB` to `None`, generating explicitly synthetic data. Set all three paths to use real downloaded inputs. It skips evaluation in real mode because labels are unavailable. It trains, validates submission format, maps and explains. The committed notebook has cleared outputs; execution is not part of pytest. The small synthetic tests cover parsing, energy-share exactness, NaNs, ZIP loading, determinism, independent-row inference, mapping, explanations, ablations and bootstrap edge cases.

## Competition submission and compliance notes

- Required columns: `ID,share_added_vessel,score_added_vessel`; exactly one row per sample ID, finite values in `[0,1]`. Inference follows sample ordering. Do not round or threshold predictions.
- Maximum **2 submissions/day**, **200 total**. Choose **2 final submissions** before selection closes **13 December 2026, 12:00 UTC**. Prediction submissions close **13 December 2026, 11:30 UTC**. Approximately 30% of test examples are public and 70% private. Re-check current official rules before acting.
- Top-10 code requests require a response within **48 hours**; the general monitoring clause also permits code requests at any time with a **24-hour** deadline. Preserve the exact environment, trained bundle and run artifacts.
- Research write-up PDF plus runnable code are submitted separately through the **research write-up form**, due 13 December 2026, 11:30 UTC; the supplied guidance specifies separate uploads, each at most 10 MB. No upload is performed by this repository.
- Competition data have challenge-only usage restrictions. Do not commit or redistribute them. Approximate coordinates/AIS proximity do not prove audibility, identity, count or size. No AIS is used here.
- This implementation uses no external data. Permitted future sources include [Ocean Networks Canada](https://data.oceannetworks.ca) and the **HearMyShip dataset**, subject to provider licences, open/equal access and competition declaration requirements. No dataset versions or DOIs are claimed here.
- The competition requires legally assignable rights and assignment of worldwide copyright in winning solution code to Zindi for the top three. `LICENSE` grants MIT permissions for this repository's original code; it does **not** override competition terms, grant rights to third-party data, or guarantee prize eligibility. Review ownership and any public-sharing restrictions before distributing or entering a final solution.

Author: **ZQ051031 (Zindi team name TBD)**.
