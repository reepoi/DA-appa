r"""Script for power spectra analysis of ground-truth, decoded, or generated samples."""

import numpy as np
import shutil
import sys
import torch
import warnings

from dawgz import after, job, schedule
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from pathlib import Path
from torch.utils.data import TensorDataset
from torch_harmonics import RealSHT

from appa.config import PATH_ERA5, PATH_STAT, compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    ERA5_PRESSURE_LEVELS,
    ERA5_RESOLUTION,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset, LatentBlanketDataset
from appa.data.mappings import extract_level_indices
from appa.data.transforms import StandardizeTransform
from appa.save import load_auto_encoder, safe_load

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
import dask  # noqa: E402
import xarray as xr  # noqa: E402


def load_denoiser_dataset(path, num_samples_per_date):
    r"""Returns a dataset for the denoiser samples."""
    trajectories = safe_load(path)[:, :num_samples_per_date].flatten(0, 2).unsqueeze(1)
    timestamps = (
        safe_load(path.parent / "timestamps.pt")[:, :num_samples_per_date]
        .flatten(0, 2)
        .unsqueeze(1)
    )

    return TensorDataset(trajectories, timestamps)


def load_gt_dataset(data_range, pressure_levels, st):
    r"""Returns a dataset for the ground-truth samples."""
    return ERA5Dataset(
        path=PATH_ERA5,
        start_date=data_range[0],
        end_date=data_range[1],
        num_samples=None,
        trajectory_size=1,
        trajectory_dt=1,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=pressure_levels,
        transform=st,
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


@torch.no_grad()
def compute_power_spectra(
    gt_dataloader,
    dataloader,
    out_folder,
    pressure_levels,
    data_type,
    autoencoder,
    era5_mean,
    era5_std,
    standardize,
    # latent std and mean for AE samples
    latent_mean=None,
    latent_std=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if latent_mean is not None and latent_std is not None:
        latent_mean = latent_mean.to(device)
        latent_std = latent_std.to(device)

    Lon = ERA5_RESOLUTION[0]
    Lat = ERA5_RESOLUTION[1]

    if dataloader is not None:
        dataloader_iter = iter(dataloader)

    for index, gt_sample in enumerate(gt_dataloader):
        x_gt, c_gt, t_gt = gt_sample

        x_gt = x_gt.to(device)
        c_gt = c_gt.to(device)

        c_gt = rearrange(c_gt, "b t c h w -> (b t) (h w) c")

        if dataloader is not None:
            try:
                data_sample = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                data_sample = next(dataloader_iter)

            if len(data_sample) == 2:  # latent samples
                z, t = data_sample

                if latent_mean is not None and latent_std is not None:
                    z = z.to(device) * latent_std + latent_mean

                z = z.to(device)[0]
                x = autoencoder.decode(z, t, c_gt)
                x = rearrange(x, "... (Lat Lon) Z -> ... Z Lat Lon", Lat=Lat, Lon=Lon)
                if not standardize:
                    x = x * era5_std.to(device) + era5_mean.to(device)

                x = x[None]
            else:
                raise ValueError(
                    f"Incorrect dataloader: analyzing latents but got {len(data_sample)} items."
                )
        else:
            x = x_gt

        if not standardize:
            sea_mask = np.load("masks/continents.npy")
            x[0, 0, ERA5_VARIABLES.index("sea_surface_temperature")][sea_mask] = 0.0

        t_gt = t_gt.squeeze()
        time = np.datetime64(f"{t_gt[0]:04d}-{t_gt[1]:02d}-{t_gt[2]:02d}T{t_gt[3]:02d}:00").astype(
            "datetime64[ns]"
        )
        # arXiv:2504.18720v3 Sec. 4.2-4.3 (Fig. 4): compare energy distribution
        # across spherical harmonic degrees for ERA5, AE reconstructions, and samples.
        tool_spherical_harmonics = RealSHT(nlat=Lat, nlon=Lon, grid="equiangular").to(device)
        coeffecients = tool_spherical_harmonics(x[:, 0])
        coeffecients = coeffecients.abs()
        coeffecients = coeffecients.pow(2)
        coeffecients_norm_factor = (
            torch.tensor([(2 * l + 1) for l in range(coeffecients.shape[-1])])
            .unsqueeze(0)
            .to(device)
        )

        coeffecients = coeffecients.sum(-2) / coeffecients_norm_factor
        x, coeffecients = x.cpu(), coeffecients.cpu()

        indices = extract_level_indices(ERA5_VARIABLES, sub_levels=pressure_levels)
        data_arrays, stats = (
            [],
            {"spectra": coeffecients},
        )

        for stat_name, stat_data in stats.items():
            stat_slices = {v: stat_data[:, s:e] for v, (s, e) in indices.items()}

            for v, data in stat_slices.items():
                data_array = xr.DataArray(data, dims=("time", "level", "coefficients"), name=v)

                if len(data_array.level) == 1:
                    data_array = data_array.squeeze("level")

                data_array = data_array.expand_dims(statistics=[stat_name])
                data_arrays.append(data_array)

        dataset_spectral = xr.merge(data_arrays)

        chunk_sizes = {"time": 100, "level": -1}
        dataset_spectral = dataset_spectral.chunk(chunk_sizes)
        dataset_spectral["level"] = pressure_levels
        dataset_spectral["time"] = [time]
        dataset_spectral["time"].encoding["units"] = "hours since 1900-01-01T00:00:00"
        dataset_spectral["coefficients"] = range(dataset_spectral.coefficients.size)
        dataset_spectral.level.attrs["Name"] = "Atmospheric Pressure Levels"
        dataset_spectral.level.attrs["Unit"] = "[hPa]"
        dataset_spectral.coefficients.attrs["Name"] = "Spherical Harmonics Degree"
        dataset_spectral.coefficients.attrs["Unit"] = "[-]"
        dataset_spectral.attrs["Description"] = "Power spectra averaged over samples."
        dataset_spectral.attrs["Dataset"] = data_type

        zarr_kwargs = {"mode": "w"} if index == 0 else {"append_dim": "time"}
        dataset_spectral.to_zarr(out_folder / "power_spectra.zarr", **zarr_kwargs)

        if index == 0 or (index + 1) % 10 == 0:
            print(f"Sample {index + 1}/{len(gt_dataloader)} done.", flush=True)


def schedule_jobs(config):
    r"""Schedules computation for power spectra."""

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

    if data_type == "ae":
        out_folder = path / "physical_analysis"
    elif data_type == "gt":
        out_folder = PATH_ERA5.parent / "physical_analysis"
    elif data_type == "denoiser":
        out_folder = path.parent

    print("Computing power spectra for", data_type, ", will be stored in", out_folder)

    # Avoids deadlocks.
    dask.config.set(scheduler="synchronous")

    # Autoencoder
    path_ae = latent_dir / "ae"
    cfg_ae = compose(path_ae / "config.yaml")

    stats_path = latent_dir / "stats.pth"
    latent_stats = safe_load(stats_path)
    latent_mean = latent_stats["mean"]
    latent_std = latent_stats["std"]
    latent_std = torch.sqrt(latent_std**2 + cfg_ae.ae.noise_level**2)

    latent_dump_name = config.latent_dump_name

    pressure_levels = (
        SUB_PRESSURE_LEVELS if cfg_ae.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
    )

    # For generated samples analysis only
    num_samples_per_date = config.num_samples_per_date

    standardize = config.standardize
    st = StandardizeTransform(PATH_STAT, state_variables=ERA5_VARIABLES, levels=pressure_levels)
    era5_mean, era5_std = st.state_mean, st.state_std

    if data_split == "train":
        data_range = DATASET_DATES_TRAINING
    elif data_split == "valid":
        data_range = DATASET_DATES_VALIDATION
    elif data_split == "test":
        data_range = DATASET_DATES_TEST

    if data_type == "denoiser":
        dataset = load_denoiser_dataset(path, num_samples_per_date)
    elif data_type == "gt":
        dataset = load_gt_dataset(data_range, pressure_levels, st if standardize else None)
    elif data_type == "ae":
        dataset = load_ae_dataset(path / latent_dump_name, data_range)

    num_chunks = config.num_chunks
    while len(dataset) % num_chunks != 0:
        num_chunks -= 1

    @job(
        name="appa power spectra (compute)",
        array=num_chunks,
        **config.hardware.chunk,
    )
    def compute_chunk(rank: int):
        r"""Computes the power spectra for a given chunk."""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with open_dict(cfg_ae):
            cfg_ae.ae.checkpointing = True
            cfg_ae.ae.noise_level = 0.0

        ae = load_auto_encoder(path_ae, "model", eval_mode=True)
        ae.decoder = ae.decoder.to(device)

        gt_dataset = load_gt_dataset(data_range, pressure_levels, st if standardize else None)
        gt_dataloader = get_dataloader(
            gt_dataset, rank=rank, world_size=num_chunks, num_workers=4, prefetch_factor=2
        )

        if data_type == "denoiser":
            latent_dataset = load_denoiser_dataset(path, num_samples_per_date)
        elif data_type == "ae":
            latent_dataset = load_ae_dataset(path / latent_dump_name, data_range)
        else:
            latent_dataset = None
            latent_dataloader = None

        if latent_dataset is not None:
            latent_dataloader = get_dataloader(
                latent_dataset, rank=rank, world_size=num_chunks, num_workers=4, prefetch_factor=2
            )

        chunk_folder = out_folder / f"spectral_tmp_{rank}"
        chunk_folder.mkdir(parents=True, exist_ok=True)

        latent_mean_ = latent_mean if data_type == "denoiser" else None
        latent_std_ = latent_std if data_type == "denoiser" else None

        compute_power_spectra(
            gt_dataloader,
            latent_dataloader,
            chunk_folder,
            pressure_levels,
            data_type,
            ae,
            era5_mean,
            era5_std,
            standardize,
            latent_mean=latent_mean_,
            latent_std=latent_std_,
        )

    @after(compute_chunk)
    @job(name="appa power spectra (aggregate)", **config.hardware.aggregate)
    def aggregate():
        r"""Aggregates the results of the power spectra."""

        ds = xr.open_mfdataset(
            [
                out_folder / f"spectral_tmp_{rank}" / "power_spectra.zarr"
                for rank in range(num_chunks)
            ],
            concat_dim="time",
            combine="nested",
            engine="zarr",
            parallel=True,
        ).sortby("time")

        statistics = ["mean", "var", "Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]
        stat_dict = {stat: {} for stat in statistics}

        data_arrays = list()

        for v in ERA5_VARIABLES:
            data = ds[v].sel(statistics="spectra").load().values  # [T, 721]
            dimension = len(data.shape)

            for stat in statistics:
                if stat == "mean":
                    stat_dict[stat][v] = data.mean(axis=0)
                elif stat == "var":
                    stat_dict[stat][v] = data.var(axis=0)
                else:
                    stat_dict[stat][v] = np.quantile(data, float(stat[1:]) / 100, axis=0)

                if dimension == 2:
                    stat_dict[stat][v] = stat_dict[stat][v][np.newaxis, :]

            ds_array = xr.DataArray(
                np.stack([stat_dict[stat][v] for stat in statistics]),
                dims=("statistics", "level", "coefficients"),
                name=v,
            )

            if len(ds_array.level) == 1:
                ds_array = ds_array.squeeze("level")

            data_arrays.append(ds_array)

        stats_ds = xr.merge(data_arrays)
        stats_ds["statistics"] = statistics
        stats_ds["level"] = pressure_levels
        stats_ds["coefficients"] = np.arange(ERA5_RESOLUTION[1])
        stats_ds.to_zarr(out_folder / "statistics.zarr", mode="w")

        for var in ds.variables:
            ds[var].encoding.clear()
        ds.to_zarr(
            out_folder / "power_spectra.zarr",
            mode="w",
            consolidated=True,
        )

        OmegaConf.save(config, out_folder / "config.yaml")

        for rank in range(num_chunks):
            shutil.rmtree(out_folder / f"spectral_tmp_{rank}")

    schedule(
        aggregate,
        name="appa power spectra",
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
        print("python power_spectra.py path=... otherparams=...")
        return

    config = compose("configs/power_spectra.yaml", overrides=sys.argv[1:])
    schedule_jobs(config)


if __name__ == "__main__":
    main()
