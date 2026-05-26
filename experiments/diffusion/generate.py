r"""Script for unconditional generation experiments."""

import copy
import gc
import math
import os
import shutil
import sys
import torch
import torch.distributed as dist
import wandb
import warnings

from dawgz import after, job, schedule
from einops import rearrange
from functools import partial
from math import ceil, floor
from omegaconf import OmegaConf
from pathlib import Path

from appa.config.hydra import compose
from appa.date import create_trajectory_timestamps
from appa.diffusion import Denoiser, TrajectoryDenoiser, create_schedule
from appa.math import str_to_ids
from appa.sampling import select_sampler
from appa.save import load_auto_encoder, load_denoiser, safe_load, safe_save


def schedule_jobs(config, unique_id):
    target_dir = Path(config.model_path) / "prior" / unique_id

    # Save config.yaml in target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_config = copy.deepcopy(config)
    saved_config.unique_id = unique_id
    OmegaConf.save(saved_config, target_dir / "config.yaml")

    # Launch job array of n samples for each window, according to a list of n timestamps
    trajectory_sizes = str(config.pop("trajectory_sizes"))
    blanket_size = compose(f"{config.model_path}/config.yaml").train.blanket_size
    overlap = config.blanket_overlap
    blanket_stride = blanket_size - overlap

    hardware_cfg = config.pop("hardware")
    gen_cfg = hardware_cfg.gen
    gen_gpus = gen_cfg.pop("gpus")

    num_samples_per_date = config.pop("num_samples_per_date")
    start_dates = config.pop("start_dates")
    start_dates = [start_date for start_date in start_dates for _ in range(num_samples_per_date)]
    start_dates = [
        (day, int(hour.split("h")[0])) for date in start_dates for day, hour in [date.split(" ")]
    ]
    start_dates, start_hours = zip(*start_dates)
    num_samples_per_traj_size = len(start_dates)

    jobs = []
    for unpadded_traj_size in str_to_ids(ids_str=trajectory_sizes):
        # TODO: External function to compute padded traj size
        #       given desired (ie unpadded) traj size, blanket size, num of gpus or "auto", etc.
        #       - auto mode when given maxblankets_per_gpu: compute gpus needed.
        #       - if given num of gpus, then determine padded traj size (& blankets per gpu)

        trajectory_size = max(blanket_size, unpadded_traj_size)

        # Pad to get a valid number of blankets.
        while (trajectory_size - blanket_size) % blanket_stride != 0:
            trajectory_size += 1

        num_gpus = gen_gpus
        gpus_per_node = hardware_cfg.gpus_per_node

        num_blankets = (trajectory_size - blanket_size) // blanket_stride + 1
        if num_gpus > num_blankets:
            warnings.warn(
                f"Lowering number of GPUs from {num_gpus} to {num_blankets}.",
                stacklevel=2,
            )
            num_gpus = num_blankets

        if num_gpus > gpus_per_node:
            if ceil(num_gpus / gpus_per_node) * gpus_per_node <= num_blankets:
                num_nodes = ceil(num_gpus / gpus_per_node)
            else:
                num_nodes = floor(num_blankets / gpus_per_node)
            num_gpus = gpus_per_node
        else:
            num_nodes = 1

        if num_nodes > 1:
            interpreter = f"torchrun --nnodes {num_nodes} --nproc-per-node {num_gpus} --rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:12345 --rdzv_id=$SLURM_JOB_ID"
        else:
            interpreter = f"torchrun --nnodes 1 --nproc-per-node {num_gpus} --standalone"

        window_target_dir = target_dir / f"{unpadded_traj_size}h"

        @job(
            name=f"appa prior (gen {unpadded_traj_size}h)",
            nodes=num_nodes,
            gpus=num_gpus,
            array=num_samples_per_traj_size,
            interpreter=interpreter,
            **gen_cfg,
        )
        def gen_trajectory(
            i: int,
            padded_traj_size: int = trajectory_size,
            unpadded_traj_size: int = unpadded_traj_size,
            window_target_dir: Path = window_target_dir,
        ):
            target_dir_traj = window_target_dir / f"tmp_{i}"
            target_dir_traj.mkdir(parents=True, exist_ok=True)

            generate_prior_trajectory(
                unpadded_trajectory_size=unpadded_traj_size,
                padded_trajectory_size=padded_traj_size,
                start_date=start_dates[i],
                start_hour=start_hours[i],
                target_dir=target_dir_traj,
                **config,
            )

        @after(gen_trajectory)
        @job(
            name=f"appa prior (agg {unpadded_traj_size}h)",
            **hardware_cfg.aggregate,
        )
        def aggregate(window_target_dir=window_target_dir):
            trajectories = []
            timestamps = []
            for i in range(num_samples_per_traj_size):
                target_dir_traj = window_target_dir / f"tmp_{i}"
                trajectory = safe_load(target_dir_traj / "trajectory.pt")
                timestamp = safe_load(target_dir_traj / "timestamps.pt")

                trajectories.append(trajectory)
                timestamps.append(timestamp)

                shutil.rmtree(target_dir_traj)
                del trajectory, timestamp
                gc.collect()

            trajectories = torch.cat(trajectories)
            timestamps = torch.cat(timestamps)

            trajectories = rearrange(trajectories, "(d s) ... -> d s ...", s=num_samples_per_date)
            timestamps = rearrange(timestamps, "(d s) ... -> d s ...", s=num_samples_per_date)

            safe_save(trajectories, window_target_dir / "trajectories.pt")
            safe_save(timestamps, window_target_dir / "timestamps.pt")

            print("Saved to", window_target_dir)

        jobs.append(aggregate)

    schedule(
        *jobs,
        name="appa prior",
        account=hardware_cfg.account,
        backend=hardware_cfg.backend,
        env=[
            f"export OMP_NUM_THREADS={gen_cfg.cpus}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
            "export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor",
        ],
    )


def generate_prior_trajectory(
    model_path,
    model_target,
    diffusion,
    unpadded_trajectory_size,
    padded_trajectory_size,
    blanket_overlap,
    start_date,
    start_hour,
    precision,
    target_dir,
):
    r"""Generates an unconditional trajectory given a trained model.

    Args:
        model_path (str): Path to the trained model (lap folder).
        model_target (str): Target of the model (best or last).
        diffusion (dict): Diffusion parameters (num_steps and sampler).
        unpadded_trajectory_size (int): Size of the trajectory to generate and save.
        padded_trajectory_size (int): Size of the padded trajectory to fit blankets.
        blanket_overlap (int): Number of states overlapping between blankets.
        start_date (str): Start date for the trajectory generation ("yyyy-mm-dd" format).
        start_hour (int): Start hour for the trajectory generation (0-23).
        precision (str): Precision for the model (e.g., float16).
        target_dir (Path): Directory to save the generated trajectory.
    """

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device_id = os.environ.get("LOCAL_RANK", rank)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Model and configs
    model_path = Path(model_path)
    latent_dir = model_path.parents[2]
    denoiser_cfg = compose(model_path / "config.yaml")
    precision = getattr(torch, precision)
    use_bfloat16 = precision == torch.bfloat16
    trajectory_dt = denoiser_cfg.train.blanket_dt
    blanket_size = denoiser_cfg.train.blanket_size
    blanket_stride = blanket_size - blanket_overlap
    if diffusion.num_steps is None:
        diffusion_steps = denoiser_cfg.valid.denoising_steps
    else:
        diffusion_steps = diffusion.num_steps

    ae = load_auto_encoder(latent_dir / "ae", "model", device=device, eval_mode=True)
    ae.encoder.cpu()
    ae.requires_grad_(False)

    latent_shape = ae.latent_shape
    if len(latent_shape) == 3:
        latent_shape = latent_shape[0] * latent_shape[1], latent_shape[2]
    state_size = math.prod(latent_shape)

    if use_bfloat16:
        torch.set_default_dtype(torch.bfloat16)

    schedule = create_schedule(denoiser_cfg.train).to(device)
    backbone = load_denoiser(
        model_path, best=model_target == "best", overrides={"checkpointing": True}
    ).backbone.to(device)
    backbone.requires_grad_(False)

    if rank == 0:
        timestamps = create_trajectory_timestamps(
            start_date, start_hour, unpadded_trajectory_size, trajectory_dt
        )
        safe_save(
            timestamps[None],
            target_dir / "timestamps.pt",
        )
        del timestamps
        gc.collect()
        torch.cuda.empty_cache()

    timestamps = create_trajectory_timestamps(
        start_date, start_hour, padded_trajectory_size, trajectory_dt
    )[None]

    denoise = Denoiser(backbone).cuda()
    if use_bfloat16:
        denoise = denoise.to(torch.bfloat16)

    denoise = TrajectoryDenoiser(
        denoise,
        blanket_size=blanket_size,
        blanket_stride=blanket_stride,
        state_size=state_size,
        distributed=True,
        pass_blanket_ids=False,
    )

    denoise = partial(denoise, date=timestamps.cuda())

    @torch.no_grad()
    def sample():
        sampler = select_sampler(diffusion.sampler.type)

        sampler = sampler(
            denoiser=denoise,
            steps=diffusion_steps,
            schedule=schedule,
            silent=rank > 0,
            **diffusion.sampler.config,
        )
        x1 = torch.randn(len(timestamps), padded_trajectory_size * state_size).cuda()
        # arXiv:2306.10574 (SDA): non-autoregressive trajectory generation starts from
        # high noise and integrates reverse diffusion to sample the prior jointly.
        samp_start = (x1 * schedule.sigma_tmax().cuda()).flatten(1).cuda()
        return sampler(samp_start).reshape((-1, padded_trajectory_size, *latent_shape)).cpu()

    if rank == 0:
        print("Starting generation for", unpadded_trajectory_size, "hours")

    if precision != torch.float16:
        sampled_traj = sample()
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            sampled_traj = sample()

    if rank == 0:
        # Trim to assimilation window.
        sampled_traj = sampled_traj[:, :unpadded_trajectory_size]
        safe_save(sampled_traj, target_dir / "trajectory.pt")

    dist.destroy_process_group()


def main():
    if sys.argv[1] in ("--help", "-h"):
        print("python generate.py [+id=<id>] model_path=... param1=A param2=B ...")
        return

    config = compose("configs/generate.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" in config:
        unique_id = config.pop("id")
    elif config.hardware.backend == "slurm":
        unique_id = wandb.util.generate_id()
    else:
        unique_id = os.environ["SLURM_JOB_ID"]  # even in async, should be in slurm job

    schedule_jobs(config, unique_id)


if __name__ == "__main__":
    main()
