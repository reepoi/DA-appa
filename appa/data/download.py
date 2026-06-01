r"""Tools to download ERA5 data from WeatherBench2."""

import os
import xarray as xr

from pathlib import Path
from typing import Optional, Sequence, Tuple

from .const import ERA5_AVAILABLE_CONFIGURATIONS


def display_available_datasets() -> None:
    r"""Displays available ERA5 datasets on WeatherBench2."""
    print("Available ERA5 datasets configurations:")
    for config in ERA5_AVAILABLE_CONFIGURATIONS:
        time_interval, levels, res_long, res_lat, poles = config
        print(f"  {time_interval}h, {levels} levels, {res_long}x{res_lat}, poles={poles}")


def get_bucket_url(
    time_interval: int,
    total_levels: int,
    resolution: Tuple[int, int],
    include_poles: bool,
) -> str:
    r"""Returns URL for the ERA5 dataset based on configuration.

    Arguments:
        time_interval: Time interval between data points (1 or 6 hours).
        total_levels: Number of pressure levels.
        resolution: Spatial resolution (longitude, latitude).
        include_poles: Whether pole data is included.
    """

    config = (time_interval, total_levels, *resolution, include_poles)

    if config not in ERA5_AVAILABLE_CONFIGURATIONS:
        if (*config[:-1], not include_poles) in ERA5_AVAILABLE_CONFIGURATIONS:
            raise ValueError(
                f"Configuration {config} not found. Try with include_poles={not include_poles}."
            )
        display_available_datasets()
        raise ValueError(f"Configuration {config} not found in datasets.")

    return f"gs://weatherbench2/datasets/era5/1959-{ERA5_AVAILABLE_CONFIGURATIONS[config]}"


def download(
    output_path: Path,
    start_date: str,
    end_date: str,
    variables: Optional[Sequence[str]] = None,
    pressure_levels: Optional[Sequence[int]] = None,
    time_interval: int = 1,
    total_levels: int = 37,
    resolution: Tuple[int, int] = (1440, 721),
    include_poles: bool = True,
    chunk_size: int = 1,
) -> None:
    r"""Downloads and saves ERA5 data from WeatherBench2 to a local disk.

    Arguments:
        output_path: Path to save Zarr dataset
        start_date: Start date of the data split (format: 'YYYY-MM-DD').
        end_date: End date of the data split (format: 'YYYY-MM-DD').
        variables: Variable names to retain from the dataset.
        pressure_levels: List of pressure levels to include in the dataset.
        time_interval: Time interval between data points.
        total_levels: Total number of vertical levels in the dataset.
        resolution: Spatial resolution of dataset (longitude, latitude).
        include_poles: Whether to include pole data in the dataset.
        chunk_hours: Size of time chunks (in hours) for processing.
    """

    url = get_bucket_url(time_interval, total_levels, resolution, include_poles)
    output_path = Path(output_path)/Path(url).name
    print(f"Downloading ERA5 data from {url}...", flush=True)

    data = xr.open_zarr(
        url, chunks="auto", storage_options={"token": "anon"}, zarr_format=2, consolidated=False
    )

    if variables:
        data = data[variables]

    if "time" in data.dims:
        time_slice = slice(start_date, end_date)
        if "level" in data.dims and pressure_levels:
            try:
                data = data.sel(time=time_slice, level=pressure_levels)
            except KeyError:
                available_levels = data.level.values
                raise KeyError(
                    f"Invalid pressure levels. Available levels: {available_levels}"
                ) from None
        else:
            data = data.sel(time=time_slice)

    dataset_size_gb = data.nbytes / 1e9
    print(f"Size of the dataset: {dataset_size_gb:.2f} GB", flush=True)

    times = None
    if "time" in data.coords:
        times = data.coords["time"]
    elif os.path.exists(output_path):
        print("Loading existing data to get time coordinate...", flush=True)
        existing_data = xr.open_zarr(output_path, chunks="auto", zarr_format=2, consolidated=False)
        if "time" in existing_data.coords:
            times = existing_data.coords["time"]
    else:
        import pandas as pd

        if "T" not in start_date:
            start_date += "T00"
        if "T" not in end_date:
            end_date += "T23"

        times = pd.date_range(start=start_date, end=end_date, freq=f"{time_interval}H")
        times = xr.DataArray(times, dims=["time"], name="time")

    if times is not None:
        times = times.sel(time=slice(start_date, end_date))

        # Expand variables without time dimension (e.g., orography)
        for var in data:
            if "time" not in data[var].dims:
                print("Expanding variable without time dimension:", var, flush=True)
                data[var] = data[var].expand_dims(time=times)
    else:
        print("No time coordinate found in the dataset.", flush=True)
        print("Variables without time dimension cannot be expanded!", flush=True)

    # Chunk the data to limit memory usage during saving
    data = data.chunk({"time": chunk_size})
    for var in data:
        # Ensure correct Zarr encoding
        del data[var].encoding["chunks"]

    data.to_zarr(output_path, mode="a-", zarr_format=2)

    print(f"Download complete. Zarr store saved to: {output_path}", flush=True)
