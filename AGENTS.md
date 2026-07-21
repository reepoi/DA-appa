# AGENTS

## Fast truths before coding
- `appa/config/__init__.py` hardcodes `PROJECT = Path("/path/to/appa")`; most data/training code reads paths from this constant, so update it (or patch constants at runtime) before running data/experiment scripts.
- Experiment and data scripts use Hydra via `compose("configs/...yaml")` with relative paths; run them from their own directory (for example, `experiments/diffusion/`), not repo root.
- CI is the source of truth for checks: run `pre-commit run --all-files --config pre-commit.yml` and `pytest tests`.
- Never reformat existing code that you did not write unless it is necessary for the functionality you are implementing, and do not perform incidental refactors unless explicitly asked.

## Verified commands
- Full tests: `pytest tests`
- Single test file: `pytest tests/test_diffusion.py`
- Single test function: `pytest tests/test_diffusion.py::test_Denoiser`
- Lint/format checks exactly as CI: `pre-commit run --all-files --config pre-commit.yml`

## Repo layout that matters
- `appa/`: core library (datasets, diffusion, NN blocks, diagnostics, optimization).
- `experiments/`: paper workflows (autoencoder training, latent diffusion training, observations, physics evals).
- `scripts/data/`: ERA5 download and dataset/stat preprocessing.
- `scripts/plots/`: plotting entrypoints driven by YAML configs.
- `docs/paper-experimental-details.md`: paper-pinned experimental setup from `arXiv:2504.18720v3`; use it to distinguish published settings from current config defaults.
- `tests/`: unit tests using synthetic data; no external ERA5 files required for most tests.

## Execution gotchas
- Several `main()` functions assume positional overrides and index `sys.argv[1]`; running without args can crash before help text. Pass at least one override or `-h` where guarded.
- `experiments/*/train.py` schedules jobs through `dawgz` with Slurm-oriented defaults (`backend="slurm"`, partition/account fields in configs); do not assume local single-process training entrypoints.
- Diffusion training requires config values `ae_run` and `latent_dump` (`experiments/diffusion/configs/train.yaml` leaves them as `???`).
- The observations scripts are under `experiments/observations/` (`forecast.py`, `reanalysis.py`, `evaluate.py`), even when older docs mention `experiments/diffusion/forecast.py` or `.../reanalysis.py`.
- Wiki examples sometimes use `latent_file`/`data.h5`; current code expects `latent_path` to point directly to the H5 file used by `LatentBlanketDataset`.

## Paper reproduction quickstart (Wiki-aligned)
- Canonical long-form guide is the Wiki: https://github.com/montefiore-sail/appa/wiki . Use it for parameter values; use this section for execution order and traps.
- Before running pipelines, set `PROJECT` in `appa/config/__init__.py` so `PATH_ERA5`, `PATH_STAT`, `PATH_MASK`, and `PATH_AE` point to real locations.
- Run each script from its own folder (`scripts/data`, `experiments/autoencoder`, etc.) so relative `configs/*.yaml` paths resolve.
- Quote YAML/CLI time strings (for example `"05:00"`), otherwise OmegaConf/PyYAML can reinterpret durations.

- Data prep: `scripts/data/download_era5.py` -> `scripts/data/data_stats.py` -> optional `scripts/data/ground_stations_masks.py`.
- Autoencoder pipeline: `experiments/autoencoder/train.py` -> `experiments/autoencoder/dump.py` -> `scripts/data/latent_stats.py`.
- Denoiser pipeline: `experiments/diffusion/train.py` (with `ae_run=... latent_dump=...`) -> `experiments/diffusion/generate.py` and/or `experiments/diffusion/persistence.py` sanity checks.
- Assimilation/eval pipeline: `experiments/observations/forecast.py` + `experiments/observations/reanalysis.py` -> `experiments/observations/evaluate.py`.
- Physics/plots pipeline: `experiments/physics/power_spectra.py`, `experiments/physics/physical_consistency.py`, then `scripts/plots/*.py` for paper figures.

- For quick paper reproduction with released assets, follow the Wiki page `Using-our-Pre-trained-Weights` and keep directory structure consistent with `PATH_AE` expectations.
