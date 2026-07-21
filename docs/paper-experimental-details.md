# Paper Experimental Details

This document records the experimental setup from `arXiv:2504.18720v3`, "Appa: Bending Weather Dynamics with Latent Diffusion Models for Global Data Assimilation". The paper was published with an earlier version of this repository; treat these values as the paper-pinned setup, not as a guarantee that current default configs still match.

For current execution traps and script ordering, read `AGENTS.md` first. For current code constants, also check `appa/data/const.py` and the relevant `experiments/**/configs/*.yaml` files.

## High-Level Setup

- Task family: global data assimilation with a latent score-based diffusion model.
- Data source: ERA5 via WeatherBench2 / Google Cloud Storage.
- Spatial grid: global 0.25 degree equiangular grid, `721 x 1440` with poles included.
- Temporal resolution: hourly atmospheric states.
- Main model scale: 565M parameters total.
- Components: 340M-parameter convolutional encoder-decoder plus 225M-parameter latent diffusion transformer.
- Latent compression: atmospheric state compressed by about `530x`.
- State channels: 6 surface variables plus 5 atmospheric variables at 13 pressure levels, for 71 predicted channels.
- Context channels used by the diffusion model: local time of day and elapsed year progress.
- Main assimilation tasks: reanalysis, filtering, observational forecasting, and full-state forecasting.

## Data

The paper uses ERA5 from 1993 through 2021. Follow appendix A for the chronological split:

- Training: `1993-01-01` through `2019-12-31`.
- Validation: `2020-01-01` through `2020-12-31`.
- Test: `2021-01-01` through `2021-12-31`.

Pressure levels:

- `50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000` hPa.

Predicted surface variables:

- `2m_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `mean_sea_level_pressure`
- `sea_surface_temperature`
- `total_precipitation` in the paper; current code commonly uses `total_precipitation_6hr`.

Predicted atmospheric variables:

- `temperature`
- `u_component_of_wind`
- `v_component_of_wind`
- `geopotential`
- `specific_humidity`

Preprocessing:

- Standardize each variable separately, and each pressure level separately for atmospheric variables, using training-set statistics.
- Apply the same standardization to validation, test, and generated outputs for metric computation where relevant.
- Sea-surface temperature is undefined over land; after standardization, missing SST values are replaced by zero.

## Autoencoder

Purpose:

- Compress each full ERA5 atmospheric state into a latent field before diffusion modeling.
- Decode sampled latent states back into physical ERA5 variables.

Architecture reported in the paper:

- Fully convolutional encoder-decoder.
- Input state shape: `721 x 1440` pixels with 71 channels.
- Latent shape: `23 x 47` pixels with 128 channels.
- Periodic padding along longitude for global wrap-around.
- Constant zero padding along latitude for polar boundaries.
- Input is padded to the nearest compatible grid size when needed.
- Training loss is latitude- and pressure-level-weighted MSE, following GraphCast / GenCast-style weighting.

Training configuration from Table 2:

- Loss: latitude- and level-weighted MSE.
- Latent noise regularization: `sigma = 0.01`.
- Optimizer: SOAP.
- Initial learning rate: `3e-5` with linear decay.
- Effective batch size: 64 samples per optimizer step.
- Training duration: 95,000 update steps, approximately 2 days.
- Hardware: 64 NVIDIA A100 40GB GPUs.

Reported reconstruction behavior:

- Standardized reconstruction RMSEs are mostly below `0.1`.
- Humidity and wind variables have higher errors than temperature/geopotential fields.
- Surface and low-altitude fields are helped by level weighting.
- Power spectra match ERA5 closely except at small scales where atmospheric energy is low; appendix D.1 notes deviations becoming visible around `1000 km` wavelengths and more pronounced at smaller scales.

Current-code note:

- Current default autoencoder configs may target lower-resolution or updated architectures, for example `experiments/autoencoder/configs/train.yaml` currently defaults to `cae_240x121_f6`. Do not assume that default reproduces the paper autoencoder.

## Latent Denoiser

Purpose:

- Learn a latent trajectory prior over windows of consecutive encoded ERA5 states.
- Provide the prior score used in unconditional generation and posterior-conditioned assimilation.

Architecture reported in the paper:

- Diffusion transformer based on DiT.
- Operates on windows of `W = 24` consecutive latent states.
- Latent window patching uses a temporal patch factor of 2.
- Tokenization in the paper: `23 x 47 x 24 / 2 = 12,972` tokens.
- Token/channel width after patching: 256 channels.

Training objective:

- Variance-exploding score model trained as a denoiser that estimates `E[z | z_t]`.
- Tweedie's formula converts the denoising posterior mean into a score estimate.
- The denoiser is trained on randomly selected windows from ERA5 latent trajectories.

Training configuration from Table 3:

- Loss: denoising score matching with rectified noise schedule.
- Noise range: `sigma_min = 0.001`, `sigma_max = 1000`.
- Optimizer: Adam.
- Initial learning rate: `1e-4`.
- Effective batch size: 256 samples per optimizer step.
- Training duration: 125,000 update steps, approximately 5 days.
- Hardware: 64 NVIDIA A100 40GB GPUs.

Long trajectory generation:

- Local denoiser scores over `W = 24` windows are composed to approximate the score over longer trajectories.
- The paper generalizes SDA-style score composition by using a stride `Delta >= 1` between consecutive windows.
- Larger stride reduces overlap and network evaluations.
- Algorithm 2 in appendix B.2 keeps centered slices from each local denoiser evaluation and stitches them into the full trajectory score estimate.

Current-code note:

- Current `experiments/diffusion/configs/train.yaml` defaults have been changed from the paper setup, including `time_interval: 6`, `blanket_size: 4`, and `update_steps: 500_000`. Check configs and run metadata before interpreting a current run as a paper reproduction.

## Observation Conditioning

The paper conditions the latent diffusion sampler on observations without retraining.

Observation model:

- Observations are defined in physical state space, not latent space.
- The latent-to-observation map is approximated as `measurement_operator(decoder(z))`.
- Observation likelihood is modeled as Gaussian with covariance `Sigma_y`.
- The paper adapts moment matching posterior sampling (MMPS) to a nonlinear decoder-plus-measurement operator by using its Jacobian in the covariance approximation.

Posterior score:

- Sampling uses prior score plus likelihood score: `grad log p(z_t | y) = grad log p(z_t) + grad log p(y | z_t)`.
- Prior score comes from composed latent denoiser windows.
- Likelihood score comes from the decoder and observation operator.

Synthetic observation sources in the paper:

- Ground stations observe all 6 surface variables.
- Satellite-like scans observe the 5 atmospheric variables across 13 pressure levels.
- Ground station network: 11,000 real-world GSOD locations, about 1% of grid points.
- Ground station noise: 1%, represented by covariance `1e-4` in current configs.
- Satellite measurement noise: 10%, represented by covariance `1e-2` in current configs.

Satellite scan setup visible in current observation configs:

- Low Earth orbit mask named `leo`.
- Orbital altitude: 800 km.
- Inclination: 75 degrees.
- Initial phase: 0.
- Observation frequency: 60 minutes.
- Field of view: 5 degrees.

## Assimilation And Forecast Experiments

Evaluated scenarios:

- Reanalysis: infer full trajectories from observations over the same time segment.
- Filtering: report the final state from a reanalysis posterior conditioned on past/current observations.
- Observational forecasting: initialize forecasts from partially observed/reanalyzed states.
- Full-state forecasting: initialize forecasts from full encoded latent states.

Paper observations:

- Reanalysis and filtering improve with longer assimilation windows.
- Improvements saturate beyond about 24 hours.
- Forecast skill decays gradually with lead time but remains better than persistence.
- Observational forecasts initialized from the last 12 hours of a day-long assimilation start near the reanalysis performance plateau.
- Full-state forecasts initialized from two complete states start near the autoencoder reconstruction error and eventually converge toward similar forecast behavior.

Representative observation config values in current code:

- Reanalysis assimilation lengths: `1,2,4,8,16,36,48,72`.
- Forecast config default lead time: `72` hours.
- Forecast autoregressive step default: predict 12 states per step, with the past window set automatically from model blanket size.
- Current observation scripts default to 10 samples per date.
- Current observation configs use four seasonal 2021 start dates: `2021-03-21 0h`, `2021-06-21 0h`, `2021-09-21 0h`, `2021-12-21 0h`.

Forecasting details from appendix B.3:

- The 24-hour latent window is split into a conditioning part and a generated part.
- For observational forecasting, the condition can be the last states from a fully assimilated window.
- For full-state forecasting, the condition is full encoded latent states.
- Autoregressive generation uses a sliding window and advances by the number of newly predicted steps.

Baseline comparison setup:

- IFS and GraphDOP skills are shown as reference baselines using values from the GraphDOP paper.
- The Appa comparison is not a direct fair benchmark because setup details differ.
- To match the GraphDOP January 2023 period, Appa reports a 10-member ensemble of 3-day forecasts.
- Start timestamps for that comparison: first 8 days of January 2023 at midnight.
- Total reported forecast samples: 8 dates times 10 members.

## Metrics

The paper follows WeatherBench2-style evaluation.

Skill:

- RMSE of the ensemble posterior mean against ground truth.
- For assimilation, metrics are averaged over assimilated time steps.

Spread:

- Square root of the ensemble variance.

Spread-skill ratio:

- `sqrt((M + 1) / M) * Spread / Skill` for ensemble size `M`.
- Values below 1 indicate overconfident ensembles.

CRPS:

- Ensemble CRPS using an L1 distance term to ground truth minus the pairwise ensemble distance term.
- In the deterministic case, this reduces to MAE.

Variables shown in current `experiments/observations/configs/evaluate.yaml`:

- `2m_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `mean_sea_level_pressure`
- `sea_surface_temperature`
- `total_precipitation`
- `temperature` at 850 hPa
- `geopotential` at 500 hPa
- `specific_humidity` at 700 hPa

## Physical And Spectral Diagnostics

Power spectra:

- Compare ERA5 ground truth, autoencoder reconstructions, and diffusion prior samples.
- Median and percentile bands are shown.
- Large-scale spectra are close; smaller-scale deviations appear where compression/generation loses fine-scale energy.

Physical consistency checks:

- Altitude consistency from two estimators: geopotential-based altitude and an ideal-gas/hydrostatic pressure-temperature relationship.
- Geostrophic balance: compare wind direction against geopotential gradients and correlate wind magnitude with geopotential-gradient magnitude.
- The paper highlights 500 hPa and 1000 hPa examples.
- Generated samples reproduce the same systematic differences and pressure-level trends observed in ERA5.

Qualitative snapshots:

- Appendix D.4 shows decoded sampled trajectories from 72-hour reanalysis and 3-day observational forecasts.
- Forecast examples are initialized with the last 12 states of a 24-hour assimilation.
- Representative variables include 2m temperature, 10m winds, total precipitation, specific humidity at 700 hPa, and geopotential at 850 hPa.

## Paper-To-Code Pointers

Current repository entrypoints that correspond to the paper workflow:

- Data download: `scripts/data/download_era5.py` with `scripts/data/configs/download_era5.yaml`.
- Data statistics: `scripts/data/data_stats.py`.
- Ground-station masks: `scripts/data/ground_stations_masks.py`.
- Autoencoder training: `experiments/autoencoder/train.py`.
- Latent dump creation: `experiments/autoencoder/dump.py`.
- Latent statistics: `scripts/data/latent_stats.py`.
- Denoiser training: `experiments/diffusion/train.py`.
- Unconditional generation / persistence sanity checks: `experiments/diffusion/generate.py`, `experiments/diffusion/persistence.py`.
- Observation-conditioned reanalysis: `experiments/observations/reanalysis.py`.
- Forecasting: `experiments/observations/forecast.py`.
- Assimilation metrics: `experiments/observations/evaluate.py`.
- Power spectra: `experiments/physics/power_spectra.py` and `scripts/plots/power_spectra.py`.
- Physical consistency: `experiments/physics/physical_consistency.py` and `scripts/plots/physical_consistency.py`.
- Paper figure plotting configs: `scripts/plots/configs/*.yaml`.

## Current-Code Caveats For Agents

- This document is a paper record, not a current config specification.
- Current defaults may be lower resolution, use 6-hour latent dumps, or have different blanket sizes and training durations.
- The original paper setup used hourly, full-resolution ERA5 and a 24-state denoiser window.
- `AGENTS.md` documents current execution gotchas, including `PROJECT` path configuration, Hydra working-directory assumptions, and current script locations.
- Before running or modifying experiments, compare the target run metadata against both this document and the current YAML config.
