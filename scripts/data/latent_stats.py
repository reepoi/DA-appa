r"""Script to perform statistical analysis of a dataset."""

import shutil
import sys
import torch

from dawgz import after, job, schedule
from omegaconf import DictConfig
from pathlib import Path

from appa.config.hydra import compose
from appa.data.datasets import LatentBlanketDataset
from appa.date import assert_date_format
from appa.save import safe_load, safe_save


def compute_latent_statistics(
    latent_path: str,
    start_date: str,
    end_date: str,
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

    # arXiv:2504.18720v3 Sec. 3.1-3.2: latent mean/std are used to standardize
    # encoded blankets before denoiser training and downstream sampling.
    dataset = LatentBlanketDataset(
        latent_path,
        start_date,
        end_date,
        1,
        standardize=False,
    )
    num_samples = len(dataset)
    num_channels = dataset.get_dim()

    chunk_indices = torch.arange(num_samples).split(chunk_size)
    num_chunks = len(chunk_indices)
    chunk_sizes = [len(chunk) for chunk in chunk_indices]

    @job(
        name="appa latent stats (chunk)",
        array=num_chunks,
        **hardware.chunk_stats,
    )
    def chunk_stats(chunk_id: int):
        # h5 objects cannot be pickled :-)
        dataset = LatentBlanketDataset(
            latent_path,
            start_date,
            end_date,
            1,
            standardize=False,
        )

        mean_x = torch.zeros(num_channels)
        mean_x2 = torch.zeros(num_channels)
        N_el = 0

        indices = chunk_indices[chunk_id].split(subchunk_size)

        print(f"Processing chunk {chunk_id} with {len(indices)} subchunks.")

        for index in indices:
            chunk = torch.tensor(dataset.latents[index])
            new_el = torch.prod(torch.tensor(chunk.shape[:-1]))
            mean_x = (N_el / (new_el + N_el)) * mean_x + (new_el / (N_el + new_el)) * chunk.mean(
                dim=(0, 1)
            )
            mean_x2 = (N_el / (new_el + N_el)) * mean_x2 + (
                new_el / (N_el + new_el)
            ) * chunk.square().mean(dim=(0, 1))
            N_el += new_el

        safe_save(
            {
                "mean_x": mean_x,
                "mean_x2": mean_x2,
            },
            tmp_stats_folder / f"stats_{chunk_id}.pth",
        )

    @after(chunk_stats)
    @job(
        name="appa latent stats (agg)",
        **hardware.aggregate,
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

    schedule(
        aggregate,
        name="latent stats",
        export="ALL",
        backend="slurm",
        account=hardware.account,
    )


if __name__ == "__main__":
    config = compose("configs/latent_stats.yaml", overrides=sys.argv[1:])

    assert_date_format(config.start_date)
    assert_date_format(config.end_date)

    compute_latent_statistics(**config)
