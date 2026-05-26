r"""Encode ERA5 into latent space with an AE."""

import dask
import gc
import h5py
import numpy as np
import shutil
import sys
import time
import torch
import wandb

from dawgz import after, job, schedule
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path

from appa.config import PATH_ERA5, PATH_STAT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset
from appa.data.transforms import StandardizeTransform
from appa.date import split_interval
from appa.save import load_auto_encoder


def dump_to_latent(config):
    num_chunks = config.num_chunks
    start_date = config.start_date
    end_date = config.end_date
    data_path = config.data_path
    model_path = Path(config.model_path)
    use_best = config.checkpoint == "best"

    save_every = config.save_every
    log_every = config.log_every

    output_path = model_path / "latents" / config.id
    (output_path / "tmp").mkdir(parents=True, exist_ok=True)

    # arXiv:2504.18720v3 Sec. 3.1: latent-diffusion training and inference run on
    # encoded ERA5 states, so this script materializes the latent dataset.
    # Copy AE
    (output_path / "ae").mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path / "config.yaml", output_path / "ae" / "config.yaml")
    shutil.copy2(
        model_path / ("model_best.pth" if use_best else "model_last.pth"),
        output_path / "ae" / "model.pth",
    )

    hardware_cfg = config.hardware

    time_intervals = split_interval(num_chunks, start_date, end_date)

    @job(
        name="appa dump (chunk)",
        array=num_chunks,
        **hardware_cfg.latent_chunk,
    )
    @torch.no_grad()
    def dump_chunk(rank: int):
        dask.config.set(scheduler="synchronous")

        time_interval = time_intervals[rank]

        st = StandardizeTransform(
            path=PATH_STAT,
            state_variables=ERA5_VARIABLES,
            context_variables=CONTEXT_VARIABLES,
            levels=SUB_PRESSURE_LEVELS,
        )

        dataset = ERA5Dataset(
            data_path,
            *time_interval,
            state_variables=ERA5_VARIABLES,
            context_variables=CONTEXT_VARIABLES,
            levels=SUB_PRESSURE_LEVELS,
            fill_nans=True,
            transform=st,
        )

        dataloader = get_dataloader(
            dataset,
            batch_size=config.batch_size,
            num_workers=8,
            prefetch_factor=1,
            shuffle=False,
        )

        autoencoder = load_auto_encoder(output_path / "ae", model_name="model")
        autoencoder.cuda()
        autoencoder.eval()

        latent_state_shape = autoencoder.latent_shape
        if len(latent_state_shape) == 3:
            latent_state_shape = (
                latent_state_shape[0] * latent_state_shape[1],
                latent_state_shape[2],
            )

        len_dl = len(dataloader)

        print("Total dataset size:", len(dataset), flush=True)
        print("Local chunk size:", len_dl, flush=True)

        dump_tmp_file = h5py.File(output_path / f"tmp/{rank}.h5", "w")
        dump_tmp_file.create_dataset("latents", (len_dl, *latent_state_shape), "float32")
        dump_tmp_file.create_dataset("dates", (len_dl, 4), "int32")

        dumped_states = []
        timestamps = []
        curr_idx = 0
        start_time = time.time()

        def save_progress():
            nonlocal curr_idx, dumped_states, timestamps, start_time

            dumped_states = np.concatenate(dumped_states, axis=0)
            timestamps = np.concatenate(timestamps, axis=0)
            dump_tmp_file["latents"][curr_idx : curr_idx + len(dumped_states)] = dumped_states
            dump_tmp_file["dates"][curr_idx : curr_idx + len(dumped_states)] = timestamps
            curr_idx += len(dumped_states)

            dumped_states = []
            timestamps = []
            gc.collect()

        for idx, (state, context, date) in enumerate(dataloader):
            state = rearrange(
                state.to("cuda", non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
            )
            context = rearrange(
                context.to("cuda", non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
            )
            date = rearrange(date.to("cuda", non_blocking=True), "B T D -> (B T) D")

            z = autoencoder.encode(state, date, context)

            dumped_states.append(z.cpu().numpy())
            timestamps.append(date.cpu().numpy())

            if (idx + 1) % log_every == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"{idx + 1}/{len_dl} - Avg. time per sample: {(elapsed_time / log_every):.2f}s",
                    flush=True,
                )
                start_time = time.time()

            if len(dumped_states) >= save_every:
                save_progress()

        save_progress()
        dump_tmp_file.close()

    @after(dump_chunk)
    @job(
        name="appa dump (merge)",
        **hardware_cfg.aggregate,
    )
    def aggregate():
        dataset = ERA5Dataset(
            path=data_path,
            start_date=start_date,
            end_date=end_date,
            state_variables=ERA5_VARIABLES,
            context_variables=CONTEXT_VARIABLES,
            levels=[],
            fill_nans=True,
        )
        total_num_samples = len(dataset)

        # TODO: Function to compute shape from AE config alone.
        autoencoder = load_auto_encoder(output_path / "ae", model_name="model")
        latent_state_shape = autoencoder.latent_shape
        if len(latent_state_shape) == 3:
            latent_state_shape = (
                latent_state_shape[0] * latent_state_shape[1],
                latent_state_shape[2],
            )
        del autoencoder

        dump_tmp_file = h5py.File(output_path / config.dump_name, "w")
        dump_tmp_file.create_dataset(
            "latents", (total_num_samples, *latent_state_shape), "float32"
        )
        dump_tmp_file.create_dataset("dates", (total_num_samples, 4), "int32")

        curr_idx = 0

        for rank in range(num_chunks):
            dump_tmp_file_chunk = h5py.File(output_path / f"tmp/{rank}.h5", "r")

            latents = dump_tmp_file_chunk["latents"][:]
            dates = dump_tmp_file_chunk["dates"][:]

            print(f"Chunk {rank} shape: {latents.shape}, {dates.shape}", flush=True)

            num_samples_chunk = latents.shape[0]

            # Truncate last chunk, if necessary.
            num_samples_chunk = min(num_samples_chunk, total_num_samples - curr_idx)
            latents = latents[:num_samples_chunk]
            dates = dates[:num_samples_chunk]

            dump_tmp_file["latents"][curr_idx : curr_idx + num_samples_chunk] = latents
            dump_tmp_file["dates"][curr_idx : curr_idx + num_samples_chunk] = dates
            curr_idx += num_samples_chunk

        print(dump_tmp_file["latents"].shape, dump_tmp_file["dates"].shape, flush=True)

        dump_tmp_file.close()

        shutil.rmtree(output_path / "tmp")

    schedule(
        aggregate,
        name="appa dump",
        export="ALL",
        account=config.hardware.account,
        backend=config.hardware.backend,
    )


if __name__ == "__main__":
    config = compose("configs/dump.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" not in config:
        config["id"] = wandb.util.generate_id()

    if config.data_path == "era5":
        config.data_path = PATH_ERA5

    dump_to_latent(config)
