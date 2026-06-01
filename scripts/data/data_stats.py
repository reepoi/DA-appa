r"""Script to compute pixel-space statistics over a dataset."""

from datetime import timedelta

import numpy as np
import shutil
import sys
import time
import torch
import wandb
import xarray as xr

from dawgz import array, job, schedule
from omegaconf import OmegaConf
from pathlib import Path

from appa.config import PROJECT, PATH_ERA5, PATH_STAT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS, SUB_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    ERA5_VARIABLES,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset
from appa.date import split_interval


def compute_statistics(config):
    num_chunks = config.num_chunks
    output_path = Path(config.output_path)
    tmp_path = output_path.parent / "stats_tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)

    time_intervals = split_interval(num_chunks, config.start_date, config.end_date, dt=timedelta(hours=config.time_interval))

    chunk_job_name = "era5_stats_map"
    @job(
        name=chunk_job_name,
        **config.hardware.chunk,
    )
    def compute_stats_chunk(rank: int):
        dataset = ERA5Dataset(
            config.data_path,
            *time_intervals[rank],
            state_variables=ERA5_VARIABLES,
            context_variables=CONTEXT_VARIABLES,
            # levels=ERA5_PRESSURE_LEVELS,
            levels=SUB_PRESSURE_LEVELS,
            fill_nans=False,
        )

        dataloader = get_dataloader(
            dataset,
            batch_size=config.batch_size,
            num_workers=8,
            prefetch_factor=1,
            shuffle=False,
        )

        len_dl = len(dataloader)

        means = []
        squares_means = []
        start_time = time.time()

        for idx, (data, context, _) in enumerate(dataloader):
            data = torch.cat([data, context], dim=2).cuda()  # [batch, traj_size, channels, ...]
            data_mean = data.nanmean(dim=(-1, -2))  # [batch, traj_size, channels]
            data_squared_mean = (data**2).nanmean(dim=(-1, -2))  # [batch, traj_size, channels]

            means.append(data_mean.cpu())
            squares_means.append(data_squared_mean.cpu())

            if idx % config.log_every == 0:
                elapsed = time.time() - start_time
                print(f"Chunk {rank}, Step {idx}/{len_dl}, Elapsed: {elapsed:.2f}s")
                start_time = time.time()

        means = torch.cat(means, dim=0)
        squares_means = torch.cat(squares_means, dim=0)
        means = means.mean(dim=0)[0]  # Take 0 as traj_size is set to 1 for stats.
        squares_means = squares_means.mean(dim=0)[0]

        torch.save(
            {
                "means": means,
                "squares_means": squares_means,
            },
            tmp_path / f"{rank}.pt",
        )

    @job(
        name="era5_stats_reduce",
        **config.hardware.aggregate,
    )
    def aggregate():
        means = []
        squares_means = []

        for i in range(num_chunks):
            stats = torch.load(tmp_path / f"{i}.pt", weights_only=True)
            means.append(stats["means"])
            squares_means.append(stats["squares_means"])

        means = torch.stack(means, dim=0).mean(dim=0)
        squares_means = torch.stack(squares_means, dim=0).mean(dim=0)
        stds = (squares_means - means**2).sqrt().clip(min=1e-32)

        means = means.numpy()
        stds = stds.numpy()

        num_surface_vars = len(ERA5_SURFACE_VARIABLES)
        num_atmospheric_vars = len(ERA5_ATMOSPHERIC_VARIABLES)
        # num_levels = len(ERA5_PRESSURE_LEVELS)
        num_levels = len(SUB_PRESSURE_LEVELS)
        num_context_vars = len(CONTEXT_VARIABLES)

        ds = xr.Dataset(
            coords={
                "statistic": ["mean", "std"],
                # "level": ERA5_PRESSURE_LEVELS,
                "level": SUB_PRESSURE_LEVELS,
            },
        )

        # Surface variables
        for i in range(num_surface_vars):
            ds[ERA5_SURFACE_VARIABLES[i]] = xr.DataArray(
                data=np.array([means[i], stds[i]]),
                coords={
                    "statistic": ["mean", "std"],
                },
            )

        # Atmospheric variables
        for i in range(num_atmospheric_vars):
            offset_idx = num_surface_vars + i * num_levels
            var_name = ERA5_ATMOSPHERIC_VARIABLES[i]
            ds[var_name] = xr.DataArray(
                data=np.stack(
                    [
                        means[offset_idx : offset_idx + num_levels],
                        stds[offset_idx : offset_idx + num_levels],
                    ],
                ),
                coords={
                    "statistic": ["mean", "std"],
                    # "level": ERA5_PRESSURE_LEVELS,
                    "level": SUB_PRESSURE_LEVELS,
                },
            )

        # Context variables
        for i in range(num_context_vars):
            offset_idx = num_surface_vars + num_atmospheric_vars * num_levels + i
            ds[CONTEXT_VARIABLES[i]] = xr.DataArray(
                data=np.array([means[offset_idx], stds[offset_idx]]),
                coords={
                    "statistic": ["mean", "std"],
                },
            )

        shutil.rmtree(tmp_path, ignore_errors=True)

        ds.to_zarr(output_path, mode="w", consolidated=True)

    compute_stats_chunks = array(
        *(compute_stats_chunk(rank) for rank in range(num_chunks)),
        name=chunk_job_name,
    )

    schedule(
        aggregate().after(compute_stats_chunks),
        name="appa era5 stats",
        backend=config.hardware.backend,
    )


if __name__ == "__main__":
    config = compose(PROJECT/"scripts/data/configs/data_stats.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" not in config:
        config["id"] = wandb.util.generate_id()

    if config.data_path == "era5":
        config.data_path = PATH_ERA5
    if config.output_path == "":
        config.output_path = PATH_STAT

    compute_statistics(config)
