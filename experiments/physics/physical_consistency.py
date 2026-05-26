r"""Script for physical consistency of ground-truth, decoded, or generated samples."""

import dask
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys
import torch
import xarray as xr

from dawgz import after, job, schedule
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from pathlib import Path
from torch.utils.data import TensorDataset

from appa.config import PATH_ERA5, PATH_STAT, compose
from appa.data.const import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_RESOLUTION,
    ERA5_SURFACE_VARIABLES,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset, LatentBlanketDataset
from appa.data.transforms import StandardizeTransform
from appa.diagnostics.const import EARTH_RADIUS
from appa.nn.gae import AutoEncoder
from appa.save import safe_load


def load_denoiser_dataset(path, num_samples_per_date: int = 1):
    r"""Returns a dataset for the denoiser samples."""
    trajectories = safe_load(path)[:, :num_samples_per_date].flatten(0, 2).unsqueeze(1)
    timestamps = (
        safe_load(path.parent / "timestamps.pt")[:, :num_samples_per_date]
        .flatten(0, 2)
        .unsqueeze(1)
    )

    return TensorDataset(trajectories, timestamps)


def load_gt_dataset(data_range, pressure_levels):
    r"""Returns a dataset for the ground-truth samples."""
    return ERA5Dataset(
        path=PATH_ERA5,
        start_date=data_range[0],
        end_date=data_range[1],
        num_samples=None,
        trajectory_size=1,
        trajectory_dt=1,
        state_variables=ERA5_VARIABLES,
        levels=pressure_levels,
    )


def load_ae_dataset(path, data_range):
    r"""Returns a dataset for the decoded latent samples."""
    return LatentBlanketDataset(
        path=path,
        start_date=data_range[0],
        end_date=data_range[1],
        blanket_size=1,
        standardize=False,
    )


def pressure_to_altitude(pressures, temperatures, po, lin=False):
    """Converts pressure to altitude using the barometric formula."""
    zo = 0  # m
    g = 9.80665  # m/s^2
    R = 8.31462
    M = 0.02896
    cp = 1005
    k = R / cp

    while pressures.ndim < temperatures.ndim:
        pressures = pressures[..., None]

    pressures = pressures.repeat(1, *temperatures.shape[1:])

    to = temperatures[-1] * (po / pressures[-1]) ** k
    f_t = 1 / temperatures
    f_o = 1 / to

    p_ratio = po / pressures[-1]
    p_ratio = torch.cat([pressures[1:] / pressures[:-1], p_ratio.unsqueeze(0)]).flip(dims=(0,))

    if lin:
        t_int = (f_o + f_t) / 2
    else:
        t_int = (f_o + f_t[-1]) / 2
        t_int = torch.cat([(f_t[1:] + f_t[:-1]) / 2, t_int.unsqueeze(0)]).flip(dims=(0,))

    def z_est(local_t_integral, p_ratio, z_prev):
        return z_prev + R * p_ratio.log() / (M * g * local_t_integral)

    z = [z_est(t_int[0], p_ratio[0], zo)]

    for i in range(1, t_int.shape[0]):
        z.insert(0, z_est(t_int[i], p_ratio[i], z[0]))

    return torch.stack(z) / 1000


def geopotential_to_altitude(geop):
    """Converts geopotential to altitude."""
    R = EARTH_RADIUS * 1000
    return (geop * R / (9.80665 * R - geop)) / 1000


@torch.no_grad()
def compute_histograms(
    dataloader,
    out_folder,
    target_variables,
    surface_variables,
    chunk_size,
    pressure_levels,
    decoder,
    era5_mean,
    era5_std,
):
    Lon = ERA5_RESOLUTION[0]
    Lat = ERA5_RESOLUTION[1]
    num_levels = len(pressure_levels)

    alt_diff_bins = 200
    alt_diff_range = (-0.3, 0.3)
    geobalance_bins = 200
    geobalance_range = (-1, 1)

    alt_diff_hist_counts = np.zeros((chunk_size, num_levels, alt_diff_bins))
    geobalance_hist_cos_counts = np.zeros((chunk_size, num_levels, geobalance_bins))
    geobalance_hist_sin_counts = np.zeros((chunk_size, num_levels, geobalance_bins))

    alt_diff_hist_edges = None
    geobalance_hist_cos_edges = None
    geobalance_hist_sin_edges = None

    for index, data_sample in enumerate(dataloader):
        if len(data_sample) == 2:  # latent samples
            x, d = data_sample
            x = x.cuda()
            x = decoder(x)
            x = rearrange(x, "... (Lat Lon) Z -> ... Z Lat Lon", Lat=Lat, Lon=Lon).cpu()
            x = x.squeeze(0)
            x = x * era5_std + era5_mean
            x = x[0]  # [Z, Lat, Lon]
        else:  # gt
            x, _, d = data_sample
            x = x.squeeze()

        d = d.squeeze()
        time = np.datetime64(f"{d[0]:04d}-{d[1]:02d}-{d[2]:02d}T{d[3]:02d}:00").astype(
            "datetime64[ns]"
        )

        fields = {}
        for var, idx in target_variables.items():
            if var in surface_variables:
                fields[var] = x[idx]
            else:
                fields[var] = x[idx : idx + num_levels]

        # arXiv:2504.18720v3 Sec. 4.3 (Eq. 10): pressure/temperature altitude estimate.
        # 1. Computing PTA
        PTA = pressure_to_altitude(
            pressures=torch.as_tensor(pressure_levels),
            temperatures=fields["temperature"],
            po=fields["mean_sea_level_pressure"] / 100,
            lin=False,
        )

        # arXiv:2504.18720v3 Sec. 4.3 (Eq. 9): geopotential-based altitude estimate.
        # 2. Computing GTA
        GTA = geopotential_to_altitude(geop=fields["geopotential"])

        # Adding padding
        field_geopotential = torch.nn.functional.pad(
            fields["geopotential"], pad=(1, 1), mode="circular"
        )

        # Determining angle between gradient of geopotential and wind field
        dz_dy, dz_dx = torch.gradient(field_geopotential, dim=(-2, -1))
        dz_dy = -dz_dy

        lat = torch.linspace(torch.pi / 2, -torch.pi / 2, 721)[..., None].repeat(1, 1440)

        dy = torch.pi * 1000 * EARTH_RADIUS / 721
        dx = 2 * torch.pi * 1000 * EARTH_RADIUS / 1440

        dz_dx = dz_dx[..., 1:-1] / dx
        dz_dy = dz_dy[..., 1:-1] / dy

        y_pred = dz_dy
        x_pred = dz_dx

        x_gt = 2 * 7.3e-5 * lat.sin() * fields["u_component_of_wind"]
        y_gt = 2 * 7.3e-5 * lat.sin() * lat.cos() * fields["v_component_of_wind"]

        geop_theta = torch.atan2(y_pred, x_pred)
        wind_theta = torch.atan2(y_gt, x_gt)

        D_THETA = wind_theta - geop_theta

        # Computing wind speeds
        wind_mod = (x_gt**2 + y_gt**2).sqrt()
        grad_mod = (x_pred**2 + y_pred**2).sqrt()

        CORR = []
        for i in range(num_levels):
            cc = torch.stack([wind_mod[i].flatten(), grad_mod[i].flatten()])
            CORR.append(torch.corrcoef(cc)[0, 1])

        # Flatening
        DIFF_PTA_GTA = (PTA - GTA).numpy().reshape(num_levels, -1)

        for level in range(num_levels):
            hist_diff, x_edges_diff, _ = plt.hist(
                DIFF_PTA_GTA[level], bins=alt_diff_bins, range=alt_diff_range, density=True
            )
            alt_diff_hist_counts[index, level] = hist_diff
            alt_diff_hist_edges = x_edges_diff
            plt.clf()

        # Computing sin and cos
        D_THETA_COS = np.cos(D_THETA.numpy().reshape((num_levels, -1)))
        D_THETA_SIN = np.sin(D_THETA.numpy().reshape((num_levels, -1)))

        for level in range(num_levels):
            hist_l_cos, x_edges_l_cos, _ = plt.hist(
                D_THETA_COS[level],
                bins=geobalance_bins,
                range=geobalance_range,
                density=True,
            )
            hist_l_sin, x_edges_l_sin, _ = plt.hist(
                D_THETA_SIN[level],
                bins=geobalance_bins,
                range=geobalance_range,
                density=True,
            )
            geobalance_hist_cos_counts[index, level] = hist_l_cos
            geobalance_hist_sin_counts[index, level] = hist_l_sin
            geobalance_hist_cos_edges = x_edges_l_cos
            geobalance_hist_sin_edges = x_edges_l_sin

            # Clearing plots
            plt.clf()

        # ============================
        #             CORR
        # ============================
        # Defining chunk sizes
        chunk_sizes = {"time": 100, "level": -1}

        # Creating dataset
        dataset_corr = xr.DataArray([CORR], dims=("time", "level"), name="correlation")
        dataset_corr["level"] = pressure_levels
        dataset_corr["time"] = [time]
        dataset_corr["time"].encoding["units"] = "hours since 1900-01-01T00:00:00"
        dataset_corr = dataset_corr.chunk(chunk_sizes)

        zarr_kwargs = {"mode": "w"} if index == 0 else {"append_dim": "time"}
        dataset_corr.to_zarr(out_folder / "correlation.zarr", **zarr_kwargs)

    np.save(out_folder / "altitude_histograms.npy", alt_diff_hist_counts.mean(axis=0))
    np.save(out_folder / "geobalance_histograms_cos.npy", geobalance_hist_cos_counts.mean(axis=0))
    np.save(out_folder / "geobalance_histograms_sin.npy", geobalance_hist_sin_counts.mean(axis=0))
    np.save(out_folder / "altitude_histogram_edges.npy", alt_diff_hist_edges)
    np.save(out_folder / "geobalance_histogram_edges_cos.npy", geobalance_hist_cos_edges)
    np.save(out_folder / "geobalance_histogram_edges_sin.npy", geobalance_hist_sin_edges)


def schedule_jobs(config):
    r"""Schedules computation for physical consistency."""

    path = Path(config.path)
    data_split = config.data_split

    if path.is_file() and path.name == "trajectories.pt":
        data_type = "denoiser"
    elif path.parent.name == "latents" or path.parents[1].name == "latents":
        data_type = "gt" if config.gt else "ae"

        # If .../latents/id/latent_dump.h5 is provided instead of .../latents/id
        if path.parents[1].name == "latents":
            path = path.parent
    else:
        raise ValueError(f"Couldn't infer data type from path {path}.")

    latent_dir = path
    while latent_dir.parent.name != "latents":
        latent_dir = latent_dir.parent

    if data_type == "gt":
        path = PATH_ERA5
    out_folder = (path if data_type == "latents" else path.parent) / "physical_analysis"

    print("Computing physical consistency for", data_type, ", will be stored in", out_folder)

    # Avoids deadlocks.
    dask.config.set(scheduler="synchronous")

    # Autoencoder
    path_ae = latent_dir / "ae"
    cfg_ae = compose(path_ae / "config.yaml")

    surface_variables = ERA5_SURFACE_VARIABLES
    pressure_levels = (
        SUB_PRESSURE_LEVELS if cfg_ae.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
    )
    num_levels = len(pressure_levels)
    num_surface_variables = len(surface_variables)

    # For generated samples analysis only
    num_samples_per_date = config.num_samples_per_date

    target_variables = {
        v: ERA5_SURFACE_VARIABLES.index(v)
        if v in surface_variables
        else num_surface_variables + ERA5_ATMOSPHERIC_VARIABLES.index(v) * num_levels
        for v in config.variables
    }

    st = StandardizeTransform(PATH_STAT, state_variables=ERA5_VARIABLES, levels=pressure_levels)
    era5_mean, era5_std = st.state_mean, st.state_std

    latent_dump_name = config.latent_dump_name

    if data_split == "train":
        data_range = DATASET_DATES_TRAINING
    elif data_split == "valid":
        data_range = DATASET_DATES_VALIDATION
    elif data_split == "test":
        data_range = DATASET_DATES_TEST

    if data_type == "denoiser":
        dataset = load_denoiser_dataset(path, num_samples_per_date)
    elif data_type == "gt":
        dataset = load_gt_dataset(data_range, pressure_levels)
    elif data_type == "ae":
        dataset = load_ae_dataset(path / latent_dump_name, data_range)

    num_chunks = config.num_chunks
    while len(dataset) % num_chunks != 0:
        num_chunks -= 1

    @job(
        name="appa physical consistency (compute)",
        array=num_chunks,
        **config.hardware.chunk,
    )
    def compute_chunk(rank: int):
        r"""Computes the physical consistency for a given chunk."""

        with open_dict(cfg_ae):
            cfg_ae.ae.checkpointing = True
            cfg_ae.ae.noise_level = 0.0
        ae = AutoEncoder(**cfg_ae.ae).cuda()
        state_dict = safe_load(path_ae / "model.pth")
        ae.load_state_dict(state_dict)
        ae.decoder = ae.decoder.cuda()

        if data_type == "denoiser":
            dataset = load_denoiser_dataset(path, num_samples_per_date)
        elif data_type == "gt":
            dataset = load_gt_dataset(data_range, pressure_levels)
        elif data_type == "ae":
            dataset = load_ae_dataset(path / latent_dump_name, data_range)

        dataloader = get_dataloader(
            dataset, rank=rank, world_size=num_chunks, num_workers=4, prefetch_factor=2
        )
        chunk_size = len(dataloader)

        chunk_folder = out_folder / f"tmp_{rank}"
        chunk_folder.mkdir(parents=True, exist_ok=True)

        compute_histograms(
            dataloader,
            chunk_folder,
            target_variables,
            surface_variables,
            chunk_size,
            pressure_levels,
            ae.decoder,
            era5_mean,
            era5_std,
        )

    @after(compute_chunk)
    @job(name="appa physical consistency (aggregate)", **config.hardware.aggregate)
    def aggregate():
        r"""Aggregates the results of the physical consistency."""

        ds = xr.open_mfdataset(
            [out_folder / f"tmp_{rank}" / "correlation.zarr" for rank in range(num_chunks)],
            concat_dim="time",
            combine="nested",
            engine="zarr",
            parallel=True,
        ).sortby("time")
        ds.to_zarr(out_folder / "correlation.zarr", mode="w")

        # Copy file out_folder/tmp_0/
        shutil.copy2(out_folder / "tmp_0" / "altitude_histogram_edges.npy", out_folder)
        shutil.copy2(out_folder / "tmp_0" / "geobalance_histogram_edges_cos.npy", out_folder)
        shutil.copy2(out_folder / "tmp_0" / "geobalance_histogram_edges_sin.npy", out_folder)

        # Average histograms
        for file in [
            "altitude_histograms.npy",
            "geobalance_histograms_cos.npy",
            "geobalance_histograms_sin.npy",
        ]:
            histograms = [np.load(out_folder / f"tmp_{rank}" / file) for rank in range(num_chunks)]
            histograms = np.array(histograms)
            histograms = histograms.mean(axis=0)
            np.save(out_folder / file, histograms)

        OmegaConf.save(config, out_folder / "config.yaml")

        # Remove temporary folders
        for rank in range(num_chunks):
            shutil.rmtree(out_folder / f"tmp_{rank}")

    schedule(
        aggregate,
        name="appa physical consistency",
        account=config.hardware.account,
        backend=config.hardware.backend,
        env=[
            f"export OMP_NUM_THREADS={config.hardware.chunk.cpus}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
            "export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor",
        ],
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("python physical_consistency.py path=... otherparams=...")
        return

    config = compose("configs/physical_consistency.yaml", overrides=sys.argv[1:])
    schedule_jobs(config)


if __name__ == "__main__":
    main()
