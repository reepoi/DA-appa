r"""Global paths and configuration helpers."""

__all__ = [
    "compose",
]

from pathlib import Path

from .hydra import compose

PROJECT = Path(__file__).parent.parent.parent

PATH_AE = PROJECT/"autoencoders"

# PATH_ERA5 = PROJECT / "data" / "era5_1993-2021-1h-1440x721.zarr"
PATH_ERA5 = Path("/vast/users/ac.ttransue/appa/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
PATH_STAT = PATH_ERA5.parent/f'stats_{PATH_ERA5.name}'
PATH_MASK = PATH_ERA5.parent/f'masks_{PATH_ERA5.name}'
