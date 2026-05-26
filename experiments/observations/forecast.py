r"""Script for (autoregressive) forecasting experiments."""

import copy
import gc
import h5py
import math
import numpy as np
import os
import shutil
import sys
import torch
import wandb

from dawgz import after, job, schedule
from einops import rearrange
from functools import partial
from omegaconf import OmegaConf
from pathlib import Path

from appa.config import PATH_ERA5, PATH_STAT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TEST,
    ERA5_PRESSURE_LEVELS,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.datasets import ERA5Dataset, LatentBlanketDataset
from appa.data.transforms import StandardizeTransform
from appa.date import add_hours, create_trajectory_timestamps
from appa.diffusion import Denoiser, MMPSDenoiser, create_schedule
from appa.math import str_to_ids
from appa.observations import (
    create_masks,
    create_variable_mask,
    mask_state,
    observator_full,
    observator_partial,
)
from appa.sampling import select_sampler
from appa.save import load_auto_encoder, load_denoiser, safe_load, safe_save


def schedule_jobs(config, unique_id):
    target_dir = Path(config.model_path) / "forecast" / unique_id

    # Save config.yaml in target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_config = copy.deepcopy(config)
    saved_config.unique_id = unique_id
    OmegaConf.save(saved_config, target_dir / "config.yaml")

    # Launch job array of n samples for each window, according to a list of n timestamps
    assimilation_lengths = str(config.pop("assimilation_lengths"))
    lead_times = str(config.pop("lead_times"))

    hardware_cfg = config.pop("hardware")

    num_samples_per_date = config.pop("num_samples_per_date")
    start_dates = config.pop("start_dates")

    date_ids = [i for i in range(len(start_dates)) for _ in range(num_samples_per_date)]
    sample_ids = [j for _ in range(len(start_dates)) for j in range(num_samples_per_date)]

    start_dates = [start_date for start_date in start_dates for _ in range(num_samples_per_date)]
    start_dates = [
        (day, int(hour.split("h")[0])) for date in start_dates for day, hour in [date.split(" ")]
    ]
    start_dates, start_hours = zip(*start_dates)
    num_samples_per_traj_size = len(start_dates)

    jobs = []
    for assimilation_length in str_to_ids(ids_str=assimilation_lengths):
        for lead_time in str_to_ids(lead_times):
            run_name = f"{assimilation_length}_{lead_time}h"

            window_target_dir = target_dir / run_name

            # Unique seed for the trajectory (for splitting masks)
            split_seed = np.random.randint(0, 2**32 - 1)

            @job(
                name=f"appa forecast (gen {run_name})",
                array=num_samples_per_traj_size,
                **hardware_cfg.gen,
            )
            def gen_trajectory(
                i: int,
                assimilation_length: int = assimilation_length,
                lead_time: int = lead_time,
                window_target_dir: Path = window_target_dir,
                split_seed: int = split_seed,
            ):
                target_dir_traj = window_target_dir / f"tmp_{i}"
                target_dir_traj.mkdir(parents=True, exist_ok=True)

                forecast(
                    assimilation_length=assimilation_length,
                    lead_time=lead_time,
                    start_date=start_dates[i],
                    start_hour=start_hours[i],
                    target_dir=target_dir_traj,
                    split_seed=split_seed,
                    date_id=date_ids[i],
                    sample_id=sample_ids[i],
                    **config,
                )

            @after(gen_trajectory)
            @job(
                name=f"appa forecast (agg {run_name})",
                **hardware_cfg.aggregate,
            )
            def aggregate(window_target_dir=window_target_dir):
                if config.initialization in ("observations", "reanalysis"):
                    all_masks = h5py.File(window_target_dir / "masks.h5", "w")
                    os.makedirs(window_target_dir / "masks", exist_ok=True)

                trajectories = []
                timestamps = []
                for i in range(num_samples_per_traj_size):
                    target_dir_traj = window_target_dir / f"tmp_{i}"
                    trajectory = safe_load(target_dir_traj / "trajectory.pt")
                    timestamp = safe_load(target_dir_traj / "timestamps.pt")

                    trajectories.append(trajectory)
                    timestamps.append(timestamp)

                    if config.initialization in ("observations", "reanalysis"):
                        # TODO: Format masks to (d s) ... vs d s ...
                        shutil.move(
                            target_dir_traj / "masks.h5", window_target_dir / "masks" / f"{i}.h5"
                        )
                        all_masks[f"masks_{i}"] = h5py.ExternalLink(f"masks/{i}.h5", "/")
                        start_date_str = [str(e) for e in timestamp[0, 0].tolist()]
                        start_date_str = "-".join(start_date_str[:3]) + f" {start_date_str[3]}h"
                        all_masks[f"masks_{i}"].attrs["start_date"] = start_date_str

                    shutil.rmtree(target_dir_traj)
                    del trajectory, timestamp
                    gc.collect()

                if config.initialization in ("observations", "reanalysis"):
                    all_masks.close()

                trajectories = torch.cat(trajectories)
                timestamps = torch.cat(timestamps)

                trajectories = rearrange(
                    trajectories, "(d s) ... -> d s ...", s=num_samples_per_date
                )
                timestamps = rearrange(timestamps, "(d s) ... -> d s ...", s=num_samples_per_date)

                safe_save(trajectories, window_target_dir / "trajectories.pt")
                safe_save(timestamps, window_target_dir / "timestamps.pt")

                print("Saved to", window_target_dir)

            jobs.append(aggregate)

    schedule(
        *jobs,
        name="appa forecast (autoregressive)",
        account=hardware_cfg.account,
        backend=hardware_cfg.backend,
        env=[
            f"export OMP_NUM_THREADS={hardware_cfg.gen.cpus}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
            "export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor",
        ],
    )


def forecast(
    model_path,
    model_target,
    latent_dump_name,
    diffusion,
    assimilation_length,
    lead_time,
    preds_per_step,
    past_window_size,
    initialization,
    observed_variables,
    masks,
    start_date,
    start_hour,
    precision,
    target_dir,
    split_seed,
    date_id,
    sample_id,
):
    r"""Performs autoregressive forecasting.

    Paper mapping (`arXiv:2504.18720v3`):
        - observational forecasting: Sec. 4.4, Eq. (13)
        - full-state forecasting: Sec. 4.4, Eq. (14)

    Args:
        model_path: Path to the model directory.
        model_target: Target model to load (best or last).
        latent_dump_name: Name of the latent dump h5 file.
        diffusion: Diffusion configuration dictionary:
            - num_steps: Number of diffusion steps.
            - mmps_iters: Number of iterations for MMPS.
            - sampler: Sampler configuration dictionary:
                - type: Type of sampler (pc or lms).
                - config: Configuration for the sampler.
        assimilation_length: Length of the assimilation window.
        lead_time: Total number of states to predict.
        preds_per_step: States to save per autoregressive step.
        past_window_size: Size of the sliding conditioning window.
        initialization: initialization mode.
            - full: use full encoded latent states
            - observations: perform partial reanalysis from observations:
                            [X X X X obs obs | forecast forecast ...]
            - reanalysis: load the last n states from reanalysis performed before.
        masks: Masks configuration if observing partial states.
            Example:
            - name: leo
                type: satellite
                covariance: 1e-2
                config:
                orbital_altitude: 800
                inclination: 75
                initial_phase: 0
                obs_freq: 60
                fov: 5
            - name: weather11k
                type: stations
                covariance: 1e-4
                config:
                num_stations: 11k (or 5k)
                num_valid_stations: 0
        start_date: Date of the first predicted state (YYYY-MM-DD format).
        start_hour: Hour of the first predicted state (0-23).
        precision: Precision for the computation (e.g., float16).
        target_dir: Directory to save the results in.
        split_seed: Seed for splitting masks, if observing partial states.
    """
    device = "cuda"

    # Model and configs
    model_path = Path(model_path)
    latent_dir = model_path.parents[2]
    ae_cfg = compose(model_path.parents[4] / "config.yaml")
    denoiser_cfg = compose(model_path / "config.yaml")
    precision = getattr(torch, precision)
    use_bfloat16 = precision == torch.bfloat16
    trajectory_dt = denoiser_cfg.train.blanket_dt
    blanket_size = denoiser_cfg.train.blanket_size
    if diffusion.num_steps is None:
        diffusion_steps = denoiser_cfg.valid.denoising_steps
    else:
        diffusion_steps = diffusion.num_steps
    pressure_levels = (
        SUB_PRESSURE_LEVELS if ae_cfg.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
    )

    lead_time = lead_time // trajectory_dt

    if preds_per_step == "auto" and past_window_size == "auto":
        past_window_size = blanket_size // 2
        preds_per_step = blanket_size - past_window_size
    elif preds_per_step == "auto":
        preds_per_step = blanket_size - past_window_size
    elif past_window_size == "auto":
        past_window_size = blanket_size - preds_per_step

    if initialization == "reanalysis":
        start_date_initcond, start_hour_initcond = add_hours(
            start_date, start_hour, -blanket_size * trajectory_dt
        )
        start_date_initblanket, start_hour_initblanket = add_hours(
            start_date, start_hour, -blanket_size * trajectory_dt
        )
    else:
        start_date_initcond, start_hour_initcond = add_hours(
            start_date, start_hour, -assimilation_length * trajectory_dt
        )
        start_date_initblanket, start_hour_initblanket = add_hours(
            start_date, start_hour, -past_window_size * trajectory_dt
        )

    stats_path = latent_dir / "stats.pth"
    latent_stats = safe_load(stats_path)
    latent_mean = latent_stats["mean"].cuda()
    latent_std = latent_stats["std"].cuda()
    latent_std = torch.sqrt(latent_std**2 + ae_cfg.ae.noise_level**2)
    cov_z = (ae_cfg.ae.noise_level / latent_std) ** 2

    ae = load_auto_encoder(latent_dir / "ae", "model", device=device, eval_mode=True)
    ae.cuda()

    latent_shape = ae.latent_shape
    if len(latent_shape) == 3:
        latent_shape = latent_shape[0] * latent_shape[1], latent_shape[2]
    state_size = math.prod(latent_shape)

    schedule = create_schedule(denoiser_cfg.train).to(device)
    backbone = load_denoiser(
        model_path, best=model_target == "best", overrides={"checkpointing": True}
    ).backbone.to(device)
    backbone.requires_grad_(False)

    if initialization == "reanalysis":
        cond_start_idx = 0
    else:
        cond_start_idx = past_window_size - assimilation_length

    do_reanalysis = initialization == "reanalysis"

    if initialization == "full":
        latent_ds = LatentBlanketDataset(
            path=latent_dir / latent_dump_name,
            start_date=start_date_initcond,
            start_hour=start_hour_initcond,
            end_date=DATASET_DATES_TEST[-1],
            blanket_size=blanket_size,
            standardize=True,
            stride=trajectory_dt,
        )

        z_obs, init_date = latent_ds[0]
        z_obs, init_date = z_obs[:assimilation_length], init_date[:assimilation_length]
        z_obs_cov = cov_z[None][None].expand(*z_obs.shape[:-1], latent_shape[-1])
        z_obs = z_obs.flatten()
        z_obs_cov = z_obs_cov.flatten()
    elif initialization in ("observations", "reanalysis"):
        init_assim_length = blanket_size if do_reanalysis else assimilation_length

        # Save masks
        #   Steps & delays are computed to match the ones used below.
        #   0 corresponds to the first state of the blanket for
        #   the first assimilation step.
        if do_reanalysis:
            mask_steps = assimilation_length
            mask_delay = blanket_size - assimilation_length
        else:
            mask_steps = assimilation_length
            mask_delay = cond_start_idx
        create_masks(
            masks,
            mask_steps,
            delay=mask_delay,
            split_seed=split_seed,
            trajectory_dt=trajectory_dt,
            save=target_dir,
        )

        (satellite_mask, sat_cov), ((stations_mask, _), stations_cov) = create_masks(
            masks,
            blanket_size,
            delay=0,
            split_seed=split_seed,
            trajectory_dt=trajectory_dt,
        )

        stations_variables = (
            create_variable_mask(
                torch.arange(
                    start=observed_variables.stations.low, end=observed_variables.stations.high + 1
                )
            )
            if observed_variables.stations.enabled
            else None
        )
        satellite_variables = (
            create_variable_mask(
                torch.arange(
                    start=observed_variables.satellites.low,
                    end=observed_variables.satellites.high + 1,
                )
            )
            if observed_variables.satellites.enabled
            else None
        )
        mask_fn = partial(
            mask_state,
            stations_mask=stations_mask,
            stations_cov=stations_cov,
            satellite_mask=satellite_mask,
            satellite_cov=sat_cov,
            stations_variables=stations_variables,
            satellite_variables=satellite_variables,
        )
        variables_and_levels = {
            "state_variables": ERA5_VARIABLES,
            "context_variables": CONTEXT_VARIABLES,
            "levels": pressure_levels,
        }
        st = StandardizeTransform(PATH_STAT, **variables_and_levels)
        era5 = ERA5Dataset(
            path=PATH_ERA5,
            start_date=start_date_initcond,
            end_date=DATASET_DATES_TEST[-1],
            start_hour=start_hour_initcond,
            **variables_and_levels,
            transform=st,
            trajectory_dt=trajectory_dt,
        )

        with torch.no_grad():
            z_obs = []
            z_obs_cov = []
            c_obs = []
            t_obs = []
            for t in range(init_assim_length):
                x_p, c_p, t_p = era5[t]

                x_p = rearrange(x_p.to(device, non_blocking=True), "T Z Lat Lon -> T (Lat Lon) Z")
                observed_values, cov_y_t = mask_fn(
                    x_p, t + cond_start_idx
                )  # Satellites start at the blanket start.
                z_obs.append(observed_values)
                z_obs_cov.append(cov_y_t)

                c_p = rearrange(c_p.to(device, non_blocking=True), "T Z Lat Lon -> T (Lat Lon) Z")

                c_obs.append(c_p)
                t_obs.append(t_p)

                del x_p

            z_obs = torch.cat(z_obs)
            z_obs_cov = torch.cat(z_obs_cov)
            c_obs = torch.cat(c_obs)
            t_obs = torch.cat(t_obs)
        del st, era5
    else:
        raise ValueError(f"Unknown initialization mode {initialization}.")

    saved_states = []
    saved_timestamps = []

    current_traj_size = 0

    num_steps = math.ceil(lead_time / preds_per_step)
    if do_reanalysis:
        num_steps += 1  # First step for initial reanalysis, where only reanalyzed states are kept.
    max_traj_size = assimilation_length + lead_time

    for step in range(num_steps):
        timestamps = create_trajectory_timestamps(
            start_day=start_date_initblanket,
            start_hour=start_hour_initblanket,
            traj_size=blanket_size,
            dt=trajectory_dt,
        )[None]

        if step == 0 and initialization in ("observations", "reanalysis"):
            if do_reanalysis:
                obs_slice_fn = lambda _: slice(0, blanket_size)
            else:  # observations

                def obs_slice_fn(_, cond_start_idx=cond_start_idx):
                    return slice(cond_start_idx, past_window_size)

            A = partial(
                observator_partial,
                context_fn=lambda _: c_obs,
                timestamp_fn=lambda _: t_obs,
                blanket_size=blanket_size,
                blanket_stride=0,  # ignored for 1 blanket only.
                num_latent_channels=latent_shape[-1],
                autoencoder=ae,
                latent_mean=latent_mean,
                latent_std=latent_std,
                slice_fn=obs_slice_fn,
                mask_fn=mask_fn,
            )
        else:

            def obs_slice_fn(_, cond_start_idx=cond_start_idx):
                return slice(cond_start_idx, past_window_size)

            A = partial(
                observator_full,
                blanket_size=blanket_size,
                num_latent_channels=latent_shape[-1],
                slice_fn=obs_slice_fn,
            )

        denoise = Denoiser(backbone).cuda()
        if use_bfloat16:
            denoise = denoise.to(torch.bfloat16)

        denoise = MMPSDenoiser(
            denoise, A, z_obs.cuda(), z_obs_cov.cuda(), iterations=diffusion.mmps_iters
        )
        # arXiv:2306.10574 (SDA): zero-shot conditioning is done at inference by
        # adding likelihood guidance; Appa Sec. 3.3 uses latent operators + MMPS.
        denoise = partial(denoise, date=timestamps.cuda())

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                sampler = select_sampler(diffusion.sampler.type)

                sampler = sampler(
                    denoiser=denoise,
                    steps=diffusion_steps,
                    schedule=schedule,
                    silent=False,
                    **diffusion.sampler.config,
                )
                x1 = torch.randn(1, blanket_size * state_size).cuda()
                samp_start = (x1 * schedule.sigma_tmax().cuda()).flatten(1).cuda()
                sample = sampler(samp_start).reshape((-1, blanket_size, *latent_shape)).cpu()

        if step == 0 and initialization != "reanalysis":
            saved_states.append(sample[:, cond_start_idx:past_window_size])
            saved_timestamps.append(timestamps[:, cond_start_idx:past_window_size])

        if initialization != "reanalysis" or step > 0:
            saved_states.append(sample[:, past_window_size : past_window_size + preds_per_step])
            saved_timestamps.append(
                timestamps[:, past_window_size : past_window_size + preds_per_step]
            )

            z_obs = sample[:, cond_start_idx : past_window_size + preds_per_step][
                :, -past_window_size:
            ].cuda()
            z_obs_cov = cov_z[None][None].expand(*z_obs.shape[:-1], latent_shape[-1]).cuda()
            z_obs = z_obs.flatten()
            z_obs_cov = z_obs_cov.flatten()

            cond_start_idx = max(0, cond_start_idx - preds_per_step)

            start_date_initblanket, start_hour_initblanket = add_hours(
                start_date_initblanket,
                start_hour_initblanket,
                preds_per_step * trajectory_dt,
            )
        else:  # reanalysis, step == 0
            saved_states.append(sample[:, -assimilation_length:])
            saved_timestamps.append(timestamps[:, -assimilation_length:])

            z_obs = sample[:, -assimilation_length:].cuda()
            z_obs_cov = cov_z[None][None].expand(*z_obs.shape[:-1], latent_shape[-1]).cuda()
            z_obs = z_obs.flatten()
            z_obs_cov = z_obs_cov.flatten()

            # After first step of reanalysis, we start conditioning
            # as if it were the first step of the two other modes.
            cond_start_idx = past_window_size - assimilation_length

            start_date_initblanket, start_hour_initblanket = add_hours(
                start_date_initblanket,
                start_hour_initblanket,
                (blanket_size - past_window_size) * trajectory_dt,
            )

        total_state = torch.cat(saved_states, dim=1)
        total_timestamps = torch.cat(saved_timestamps, dim=1)

        if total_state.shape[1] >= max_traj_size:
            total_state = total_state[:, :max_traj_size]
            total_timestamps = total_timestamps[:, :max_traj_size]

        safe_save(total_state, target_dir / "trajectory.pt")
        safe_save(total_timestamps, target_dir / "timestamps.pt")

        current_traj_size = total_state.shape[1]

        if current_traj_size == max_traj_size:
            return


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("python forecast.py [+id=<id>] model_path=... param1=A param2=B ...")
        return

    config = compose("configs/forecast.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" in config:
        unique_id = config.pop("id")
    elif config.hardware.backend == "slurm":
        unique_id = wandb.util.generate_id()
    else:
        unique_id = os.environ["SLURM_JOB_ID"]  # even in async, should be in slurm job

    schedule_jobs(config, unique_id)


if __name__ == "__main__":
    main()
