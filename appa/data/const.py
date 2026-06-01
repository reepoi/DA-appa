r"""ERA5 constants."""

# ERA5 (GenCast-based)
ERA5_DATE_START = "1993-01-01"
ERA5_DATE_END = "2021-12-31"
ERA5_RESOLUTION = (1440, 721)
ERA5_TIME_INTERVAL = 1
ERA5_NUM_TOTAL_LEVELS = 37
ERA5_NUM_COARSENED_LEVELS = 13

# WeatherBench2
# Buckets are available at https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5
ERA5_AVAILABLE_CONFIGURATIONS = {
    (
        6,
        ERA5_NUM_COARSENED_LEVELS,
        64,
        32,
        False,
    ): "2023_01_10-6h-64x32_equiangular_conservative.zarr",
    (
        6,
        ERA5_NUM_COARSENED_LEVELS,
        64,
        32,
        True,
    ): "2022-6h-64x32_equiangular_with_poles_conservative.zarr",
    (6, ERA5_NUM_COARSENED_LEVELS, 64, 33, False): "2022-6h-64x33.zarr",
    (6, ERA5_NUM_COARSENED_LEVELS, 128, 64, False): "2022-6h-128x64_equiangular_conservative.zarr",
    (
        6,
        ERA5_NUM_COARSENED_LEVELS,
        128,
        64,
        True,
    ): "2022-6h-128x64_equiangular_with_poles_conservative.zarr",
    (  # Target
        6,
        ERA5_NUM_COARSENED_LEVELS,
        240,
        121,
        True,
    ): "2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
    (
        6,
        ERA5_NUM_COARSENED_LEVELS,
        512,
        256,
        False,
    ): "2022-6h-512x256_equiangular_conservative.zarr",
    (6, ERA5_NUM_COARSENED_LEVELS, 1440, 721, True): "2023_01_10-wb13-6h-1440x721.zarr",
    (6, ERA5_NUM_TOTAL_LEVELS, 1440, 721, True): "2022-full_37-6h-0p25deg-chunk-1.zarr-v2/",
    (
        1,
        ERA5_NUM_COARSENED_LEVELS,
        240,
        121,
        True,
    ): "2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr",
    (
        1,
        ERA5_NUM_COARSENED_LEVELS,
        360,
        181,
        True,
    ): "2022-1h-360x181_equiangular_with_poles_conservative.zarr",
    (
        1,
        ERA5_NUM_TOTAL_LEVELS,
        512,
        256,
        False,
    ): "2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr",
    (1, ERA5_NUM_TOTAL_LEVELS, 1440, 721, True): "2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
}

# fmt: off
ERA5_PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000
]

SUB_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

BASE_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# fmt: on
ERA5_SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    # "total_precipitation",
    "total_precipitation_6hr",
]

ERA5_ATMOSPHERIC_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
]

ERA5_VARIABLES = ERA5_SURFACE_VARIABLES + ERA5_ATMOSPHERIC_VARIABLES

CONTEXT_VARIABLES = [
    # "toa_incident_solar_radiation",
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "slope_of_sub_gridscale_orography",
    "standard_deviation_of_orography",
    "land_sea_mask",
]

# Corresponding atmospheric variables for surface variables
ATM_SURF_VARIABLE_MAPPINGS = {
    "2m_temperature": "temperature",
    "10m_u_component_of_wind": "u_component_of_wind",
    "10m_v_component_of_wind": "v_component_of_wind",
}

DATA_NAN_VALUE = 0.0
DATASET_DATES_TRAINING = ("1993-01-01", "2019-12-31")
DATASET_DATES_VALIDATION = ("2020-01-01", "2020-12-31")
DATASET_DATES_TEST = ("2021-01-01", "2021-12-31")

# Toy dataset
TOY_DATASET_VARIABLES = ["2m_temperature", "temperature"]
TOY_DATASET_DATES_TRAINING = ("2017-07-01", "2017-09-30")
TOY_DATASET_DATES_VALIDATION = ("2018-07-01", "2018-09-30")
TOY_DATASET_DATES_TEST = ("2019-07-01", "2019-09-30")


def display_available_datasets() -> None:
    r"""Displays available ERA5 datasets on WeatherBench2."""
    print("Available ERA5 datasets configurations:")
    for config in ERA5_AVAILABLE_CONFIGURATIONS:
        time_interval, levels, res_long, res_lat, poles = config
        print(f"  {time_interval}h, {levels} levels, {res_long}x{res_lat}, poles={poles}")


if __name__ == '__main__':
    display_available_datasets()
