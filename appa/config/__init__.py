r"""Global paths and configuration helpers."""

__all__ = [
    "compose",
]

from pathlib import Path

from .hydra import compose

PROJECT = Path(__file__).parent.parent.parent

PATH_AE = PROJECT / "autoencoders"

PATH_ERA5 = PROJECT / "data" / "era5_1993-2021-1h-1440x721.zarr"
PATH_STAT = PROJECT / "data" / "stats_era5_1993-2021-1h-1440x721.zarr"
PATH_MASK = PROJECT / "data" / "masks_era5_1993-2021-1h-1440x721.zarr"
