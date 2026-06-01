r"""Script to download ERA5 data from WeatherBench2."""

import sys

from dawgz import job, schedule
from functools import partial
from omegaconf import OmegaConf

from appa.config import PROJECT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    ERA5_NUM_COARSENED_LEVELS,
    ERA5_NUM_TOTAL_LEVELS,
    ERA5_PRESSURE_LEVELS,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.download import download
from appa.date import assert_date_format

if __name__ == "__main__":
    config = compose(PROJECT/"scripts/data/configs/download_era5_240x121.yaml", overrides=sys.argv[1:])

    assert_date_format(config.start_date)
    assert_date_format(config.end_date)

    OmegaConf.update(
        config,
        "resolution",
        [int(res) for res in config.resolution.split("x")],
    )

    list_levels = config.pressure_levels
    if isinstance(list_levels, str):
        if list_levels == "era5":
            list_levels = ERA5_PRESSURE_LEVELS
        elif list_levels == "era5_sub":
            list_levels = SUB_PRESSURE_LEVELS
        else:
            raise ValueError(f"Invalid pressure levels: {list_levels}")

    list_vars = config.variables
    if isinstance(list_vars, str):
        if list_vars == "era5":
            list_vars = ERA5_VARIABLES + CONTEXT_VARIABLES
        else:
            raise ValueError(f"Invalid variable list: {list_vars}")
    if config.split_vars:
        list_vars = [[var] for var in list_vars]
    else:
        list_vars = [list(list_vars)]

    hardware = config.hardware
    use_coarsened_levels = config.use_coarsened_levels
    split_vars = config.split_vars

    del config.variables
    del config.pressure_levels
    del config.use_coarsened_levels
    del config.split_vars
    del config.hardware

    jobs = []

    for job_vars in list_vars:
        dawgz_download = partial(
            download,
            **config,
            variables=job_vars,
            pressure_levels=list_levels,
            total_levels=ERA5_NUM_COARSENED_LEVELS if use_coarsened_levels else ERA5_NUM_TOTAL_LEVELS,
        )
        dawgz_download()
    #
    #     jobs.append(
    #         job(
    #             dawgz_download,
    #             name="ERA5" if not split_vars else f"ERA5-{job_vars[0]}",
    #             **hardware.job,
    #         )()
    #     )
    #
    # schedule(
    #     *jobs,
    #     name="ERA5",
    #     # export="ALL",
    #     backend=hardware.backend,
    #     # account=hardware.account,
    # )
