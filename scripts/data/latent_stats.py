r"""Script to perform statistical analysis of a dataset."""

import shutil
import sys
import torch

from dawgz import array, job, schedule
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from appa.config import PROJECT
from appa.config.hydra import compose
from appa.data.datasets import LatentBlanketDataset
from appa.date import assert_date_format
from appa.save import safe_load, safe_save


def compute_latent_statistics(
    latent_path: str,
    start_date: str,
    end_date: str,
    time_interval: int,
    chunk_size: int,
    subchunk_size: int,
    hardware: DictConfig,
):
    latent_path = Path(latent_path)
    latent_dir = latent_path.parent

    stats_path = latent_dir / "stats.pth"
    tmp_stats_folder = latent_dir / "tmp_stats"
    tmp_stats_folder.mkdir(exist_ok=True)

    assert latent_path.exists(), f"Latent data not found at {latent_path}."

    end_hour = 24 - time_interval
    # arXiv:2504.18720v3 Sec. 3.1-3.2: latent mean/std are used to standardize
    # encoded blankets before denoiser training and downstream sampling.
    dataset = LatentBlanketDataset(
        latent_path,
        start_date,
        end_date,
        1,
        standardize=False,
        end_hour=end_hour,
    )
    num_samples = len(dataset)
    num_channels = dataset.get_dim()

    chunk_indices = torch.arange(num_samples).split(chunk_size)
    num_chunks = len(chunk_indices)
    chunk_sizes = [len(chunk) for chunk in chunk_indices]

    def job_settings(section):
        settings = OmegaConf.to_container(section, resolve=True)
        if "account" in hardware and "account" not in settings:
            settings["account"] = hardware.account
        return settings

    chunk_settings = job_settings(hardware.chunk)
    chunk_throttle = chunk_settings.pop("throttle", None)
    aggregate_settings = job_settings(hardware.aggregate)

    @job(
        name="appa_latent_stats_chunk",
        **chunk_settings,
    )
    def chunk_stats(chunk_id: int):
        # h5 objects cannot be pickled :-)
        dataset = LatentBlanketDataset(
            latent_path,
            start_date,
            end_date,
            1,
            standardize=False,
            end_hour=end_hour,
        )

        mean_x = torch.zeros(num_channels)
        mean_x2 = torch.zeros(num_channels)
        num_accumulated = 0

        indices = chunk_indices[chunk_id].split(subchunk_size)

        print(f"Processing chunk {chunk_id} with {len(indices)} subchunks.")

        for index in indices:
            chunk = torch.tensor(dataset.latents[index])
            num_new = torch.prod(torch.tensor(chunk.shape[:-1]))
            mean_x = (num_accumulated / (num_new + num_accumulated)) * mean_x + (num_new / (num_accumulated + num_new)) * chunk.mean(
                dim=(0, 1)
            )
            mean_x2 = (num_accumulated / (num_new + num_accumulated)) * mean_x2 + (
                num_new / (num_accumulated + num_new)
            ) * chunk.square().mean(dim=(0, 1))
            num_accumulated += num_new

        safe_save(
            {
                "mean_x": mean_x,
                "mean_x2": mean_x2,
            },
            tmp_stats_folder / f"stats_{chunk_id}.pth",
        )

    @job(
        name="appa_latent_stats_reduce",
        **aggregate_settings,
    )
    def aggregate():
        mean_x = torch.zeros(num_channels)
        mean_x2 = torch.zeros(num_channels)

        for chunk_id in range(num_chunks):
            chunk_stats = safe_load(tmp_stats_folder / f"stats_{chunk_id}.pth")
            mean_x += chunk_stats["mean_x"] * chunk_sizes[chunk_id] / num_samples
            mean_x2 += chunk_stats["mean_x2"] * chunk_sizes[chunk_id] / num_samples

        stats = {
            "mean": mean_x,
            "std": (mean_x2 - mean_x**2).sqrt(),
        }

        safe_save(stats, stats_path)
        shutil.rmtree(tmp_stats_folder)

        print(
            f"Successfully saved statistics (mean={stats['mean'].mean()}, std={stats['std'].mean()})."
        )

    chunk_array = array(
        *(chunk_stats(chunk_id) for chunk_id in range(num_chunks)),
        name="appa_latent_stats_map",
        throttle=chunk_throttle,
    )

    schedule(
        aggregate().after(chunk_array),
        name="latent stats",
        backend=hardware.backend,
    )


if __name__ == "__main__":
    config = compose(PROJECT / "scripts/data/configs/latent_stats.yaml", overrides=sys.argv[1:])

    assert_date_format(config.start_date)
    assert_date_format(config.end_date)

    compute_latent_statistics(**config)
