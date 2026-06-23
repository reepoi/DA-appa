r"""Autoencoder training script."""

import argparse
import cloudpickle
import dask
import dawgz
import itertools
import os
import re
import wandb

from functools import partial
from omegaconf import DictConfig, OmegaConf, open_dict

import appa

from appa.config import PROJECT, PATH_AE, PATH_ERA5, PATH_STAT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset
from appa.data.transforms import StandardizeTransform
from appa.diagnostics.plots import plot_image_grid
from appa.losses import AELoss
from appa.optim import get_optimizer, safe_gd_step
from appa.save import safe_load, safe_save, select_ae_architecture

cloudpickle.register_pickle_by_value(appa)


def set_model_channels(cfg: DictConfig, pressure_levels: list) -> None:
    r"""Apply channel-count fields derived from the dataset configuration."""

    with open_dict(cfg):
        num_surface_variables = len(ERA5_SURFACE_VARIABLES) if ERA5_SURFACE_VARIABLES else 0
        num_atmospheric_variables = (
            (len(ERA5_ATMOSPHERIC_VARIABLES) * len(pressure_levels))
            if ERA5_ATMOSPHERIC_VARIABLES and pressure_levels
            else 0
        )
        num_context_variables = len(CONTEXT_VARIABLES) if CONTEXT_VARIABLES else 0
        cfg.ae["in_channels"] = num_surface_variables + num_atmospheric_variables
        cfg.ae["context_channels"] = num_context_variables


def build_standard_ae(cfg: DictConfig, device):
    r"""Build the standard autoencoder used by this training script."""

    ae_arch = select_ae_architecture(cfg.ae.name)
    return ae_arch(**cfg.ae).to(device)


def standard_loss_shape(cfg: DictConfig) -> tuple[int, int]:
    r"""Return the latitude/longitude shape used by the standard AE loss."""

    return tuple(cfg.ae.shape)


def standard_resume_state(
    ae,
    optimizer,
    scheduler,
    cfg: DictConfig,
    prev_runpath,
    device,
    forked_run: bool,
):
    r"""Load the previous lap model, optimizer, scheduler, and metadata."""

    for obj, name in zip(
        (ae.module, optimizer, scheduler), ("model", "optimizer", "scheduler")
    ):
        ckpt_path = prev_runpath / f"{name}_last.pth"
        ckpt = safe_load(ckpt_path, map_location=device)

        if name == "model":
            obj.load_state_dict(ckpt, strict=False)
        else:
            obj.load_state_dict(ckpt)

    with open(prev_runpath / "metadata.yaml", "r") as f:
        metadata = OmegaConf.load(f)

    return metadata["last_step_done"] + 1, float(metadata["best_val_loss"])


def standard_metadata(cfg: DictConfig) -> dict:
    r"""Return additional metadata to persist for a standard AE run."""

    return {}


def train(
    runid: str,
    cfg: DictConfig,
    fork_lap: int = 0,
    lap: int = 0,
    build_model_fn=build_standard_ae,
    loss_shape_fn=standard_loss_shape,
    resume_state_fn=standard_resume_state,
    metadata_fn=standard_metadata,
    run_label: str = "ae",
    description_prefix: str = "AE",
):
    r"""Train for one lap an autoencoder.

    Args:
        runid: The run id.
        cfg: The training configuration as a DictConfig.
        fork_lap: If starting from a checkpoint, the lap to fork from.
        lap: The current lap number.
    """
    import matplotlib.pyplot as plt
    import os
    import time
    import torch
    import torch.distributed as dist

    from einops import rearrange
    from math import ceil
    from torch.amp.grad_scaler import GradScaler
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import trange

    # DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_nodes = ceil(world_size / torch.cuda.device_count())
    num_cpu = len(os.sched_getaffinity(0)) * num_nodes
    device_id = os.environ.get("LOCAL_RANK", rank)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    dask.config.set(scheduler="synchronous")

    torch.set_float32_matmul_precision("high")

    assert (
        cfg.train.batch_size_per_step % (cfg.train.batch_element_per_gpu * world_size) == 0
    ), "Batch size must be divisible by number of GPU * samples per GPU"

    # Data
    n_valid_samples = cfg.valid.n_valid_samples
    n_snapshots = cfg.valid.n_valid_snapshots
    lnum_samples = [
        None,
        # Pad the number of samples to be divisible by the world size
        n_valid_samples + (-n_valid_samples) % world_size,
        n_snapshots + (-n_snapshots) % world_size,
    ]

    splits = [
        DATASET_DATES_TRAINING,
        DATASET_DATES_VALIDATION,
        DATASET_DATES_VALIDATION,
    ]

    pressure_levels = (
        SUB_PRESSURE_LEVELS if cfg.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
    )
    st = StandardizeTransform(
        PATH_STAT,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=pressure_levels,
    )
    datasets = [
        ERA5Dataset(
            path=PATH_ERA5,
            start_date=start_date,
            end_date=end_date,
            num_samples=num_samples,
            transform=st,
            trajectory_size=cfg.train.trajectory_size,
            state_variables=ERA5_VARIABLES,
            context_variables=CONTEXT_VARIABLES,
            levels=pressure_levels,
        )
        for (start_date, end_date), num_samples in zip(splits, lnum_samples)
    ]

    # Validation is always non-shuffled
    shuffles = [cfg.train.shuffle_training, False, False]

    train_loader, valid_quant_loader, valid_quali_loader = (
        get_dataloader(
            dataset=dataset,
            batch_size=cfg.train.batch_element_per_gpu,
            num_workers=num_cpu // world_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
        )
        for dataset, shuffle in zip(datasets, shuffles)
    )

    # Model
    set_model_channels(cfg, pressure_levels)
    ae = build_model_fn(cfg, device)

    latent_shape = ae.latent_shape

    ae = DistributedDataParallel(
        ae,
        device_ids=[device],
    )

    trainable_params = [p for p in ae.parameters() if p.requires_grad]
    assert trainable_params, "No trainable parameters found."

    optimizer, scheduler = get_optimizer(
        params=trainable_params,
        update_steps=cfg.train.update_steps,
        **cfg.optim,
    )

    PATH = PATH_AE / f"{runid}"
    runpath = PATH / f"{lap}"
    runpath.mkdir(parents=True, exist_ok=True)

    if lap > 0:
        prev_runpath = PATH / f"{lap - 1}"

        if not prev_runpath.exists() or not (prev_runpath / "metadata.yaml").exists():
            print("Previous lap does not exist or ran for too short. Exiting.")

            dist.destroy_process_group()
            return

        start_step, best_val_loss = resume_state_fn(
            ae=ae,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            prev_runpath=prev_runpath,
            device=device,
            forked_run=("forked_from" in cfg and lap == fork_lap),
        )
    else:
        start_step = 0
        best_val_loss = float("inf")

    if start_step >= cfg.train.update_steps:
        dist.destroy_process_group()
        return

    # Logging
    if rank == 0:
        wandb_config = OmegaConf.to_container(cfg).copy()

        # Additional config fields
        wandb_config["num_model_params"] = sum(p.numel() for p in ae.parameters())
        wandb_config["num_trainable_params"] = sum(p.numel() for p in trainable_params)
        wandb_config["path"] = runpath
        wandb_config["hardware"] = {
            "num_nodes": num_nodes,
            "num_gpus": world_size,
            "num_cpus": num_cpu,
        }
        wandb_config["ae"]["latent_shape"] = latent_shape

        # Run name format
        if run_label == "ae":
            run_name = f"{runid} {cfg.ae.name}"
            group = f"train_{runid}"
        else:
            run_name = f"{runid} {run_label} {cfg.ae.name}"
            group = f"train_{run_label}_{runid}"

        description = f"{description_prefix} {cfg.ae.name} " + ae.module.description()

        if "forked_from" in cfg:
            description = f"Forked from {cfg.forked_from}.\n" + description

        wandb_args = dict(
            name=run_name,
            project="taost-da-appa",
            entity=cfg.wandb_entity,
            config=wandb_config,
            group=group,
            resume="allow",
            notes=description,
        )

        # WAITING FOR DEPLOYEMENT OF FORK_FROM IN WANDB
        # if cfg.fork_from and lap == fork_lap:
        #     run = wandb.init(fork_from=f"{runid}?step={start_step}", **wandb_args)
        #     runid = run.id

        run = wandb.init(id=runid, **wandb_args)

        OmegaConf.save(wandb_config, runpath / "config.yaml")

        # Kill switch
        open(runpath / ".running", "a").close()

    scaler = GradScaler("cuda")
    precision = getattr(torch, cfg.train.precision)

    loss_shape = loss_shape_fn(cfg)
    model_loss = AELoss(
        criterion=cfg.loss.error,
        N_lat=loss_shape[0],
        N_lon=loss_shape[1],
        latitude_weighting=cfg.loss.latitude_weighting,
        level_weighting=cfg.loss.level_weighting,
        levels=pressure_levels,
    )

    if rank == 0:
        update_steps = trange(start_step, cfg.train.update_steps, ncols=88, ascii=True)
    else:
        update_steps = range(start_step, cfg.train.update_steps)

    grad_acc_steps = cfg.train.batch_size_per_step // (
        cfg.train.batch_element_per_gpu * world_size
    )

    count_log_train = 0
    count_log_val = 0
    count_log_save = 0
    losses = []
    grads = []
    latent_stats = []

    # Train
    ae.train()

    def cycle(loader):
        for epoch in itertools.count():
            if loader.sampler is not None:
                loader.sampler.set_epoch(epoch)
            for batch in loader:
                yield batch

    train_iter = cycle(train_loader)

    dist.barrier()

    for step in update_steps:
        start = time.time()

        if rank == 0 and not os.path.isfile(runpath / ".running"):
            print("Run aborted by removing the .running file. Stopping...")
            wandb.finish()
            # job_id = os.environ["SLURM_ARRAY_JOB_ID"]
            # os.system(f"scancel {job_id}")
            dist.destroy_process_group()
            return

        for acc_step in range(grad_acc_steps):
            state, context, timestamp = next(train_iter)
            state = rearrange(
                state.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
            )
            context = rearrange(
                context.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
            )
            timestamp = rearrange(timestamp.to(device, non_blocking=True), "B T D -> (B T) D")

            # Forward
            with torch.autocast(device_type="cuda", dtype=precision):
                if acc_step + 1 == grad_acc_steps:
                    # Only synchronize the last step
                    z, x = ae(state, timestamp, context)
                else:
                    with ae.no_sync():
                        z, x = ae(state, timestamp, context)

                loss = model_loss(x, state) + cfg.loss.l2_weight * (z**2).mean()

                z_detached = z.detach()
                z_abs = z_detached.abs()
                saturation = ae.module.saturation
                saturation_bound = ae.module.saturation_bound
                if saturation in ("softclip", "softclip2", "tanh"):
                    z_sat_frac = (z_abs > 0.9 * saturation_bound).float().mean()
                else:
                    z_sat_frac = torch.as_tensor(float("nan"), device=device)
                latent_stats.append(
                    torch.stack(
                        (
                            z_detached.square().mean().sqrt(),
                            z_abs.max(),
                            z_sat_frac,
                        )
                    )
                )

            # Backward
            if acc_step + 1 == grad_acc_steps:
                # Only synchronize the last step
                scaler.scale(loss / grad_acc_steps).backward()
                grad = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip, scaler=scaler)
                grads.append(grad)
            else:
                with ae.no_sync():
                    scaler.scale(loss / grad_acc_steps).backward()

            losses.append(loss.detach())

        # LR scheduler
        scheduler.step()

        count_log_train += cfg.train.batch_size_per_step
        count_log_val += cfg.train.batch_size_per_step
        count_log_save += cfg.train.batch_size_per_step

        if rank == 0:
            logs = {}
            logs["train/update_step_time"] = time.time() - start
            logs["train/samples_seen"] = cfg.train.batch_size_per_step * (step + 1)
            logs["train/update_steps_done"] = step + 1

        # Logging
        if cfg.train.log_interval <= count_log_train:
            losses = torch.stack(losses)
            grads = torch.stack(grads)
            latent_stats = torch.stack(latent_stats)

            if rank == 0:
                losses_list = [torch.empty_like(losses) for _ in range(world_size)]
                grads_list = [torch.empty_like(grads) for _ in range(world_size)]
                latent_stats_list = [torch.empty_like(latent_stats) for _ in range(world_size)]
            else:
                losses_list = None
                grads_list = None
                latent_stats_list = None

            dist.gather(losses, losses_list, dst=0)
            dist.gather(grads, grads_list, dst=0)
            dist.gather(latent_stats, latent_stats_list, dst=0)

            if rank == 0:
                losses = torch.cat(losses_list).cpu()
                grads = torch.cat(grads_list).cpu()
                latent_stats = torch.cat(latent_stats_list).cpu()
                logs["train/losses/mean"] = losses.mean().item()
                logs["train/losses/std"] = losses.std(unbiased=False).item()
                logs["train/grad_norm/mean"] = grads.mean().item()
                logs["train/grad_norm/std"] = grads.std(unbiased=False).item()
                logs["train/lr"] = optimizer.param_groups[0]["lr"]
                logs["train/latent/rms"] = latent_stats[:, 0].mean().item()
                logs["train/latent/abs_max"] = latent_stats[:, 1].max().item()
                z_sat_frac = latent_stats[:, 2]
                if z_sat_frac.isfinite().any():
                    logs["train/latent/sat_frac_90"] = z_sat_frac[z_sat_frac.isfinite()].mean().item()

        # Validation
        if cfg.valid.log_interval <= count_log_val:
            with torch.no_grad():
                ae.eval()

                # Set all model's gradients to None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                # Quantitative validation
                valid_losses = []
                for state, context, timestamp in valid_quant_loader:
                    state = rearrange(state, "B T Z Lat Lon -> (B T) (Lat Lon) Z").to(
                        device, non_blocking=True
                    )
                    context = rearrange(context, "B T Z Lat Lon -> (B T) (Lat Lon) Z").to(
                        device, non_blocking=True
                    )
                    timestamp = rearrange(
                        timestamp.to(device, non_blocking=True), "B T D -> (B T) D"
                    )

                    with torch.autocast(
                        device_type="cuda",
                        dtype=precision,
                    ):
                        _, x = ae(state, timestamp, context)
                        loss = model_loss(state, x)
                        rmse = torch.sqrt((x - state).square().mean(dim=1)).mean(dim=0)
                        surf_error = rmse[: len(ERA5_SURFACE_VARIABLES)]
                        atm_error = rmse[len(ERA5_SURFACE_VARIABLES) :]
                    valid_losses.append(loss)

                    del x, state, context

                valid_losses = torch.stack(valid_losses)
                torch.cuda.empty_cache()

                # Qualitative validation
                snapshots = []
                dates = []
                for data, context, timestamp in valid_quali_loader:
                    state = rearrange(
                        data.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
                    ).nan_to_num()
                    context = rearrange(
                        context.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
                    ).nan_to_num()
                    timestamp = rearrange(
                        timestamp.to(device, non_blocking=True), "B T D -> (B T) D"
                    )

                    with torch.autocast(device_type="cuda", dtype=precision):
                        _, x = ae(state, timestamp, context)
                    del context

                    x = rearrange(
                        x,
                        "(B T) (Lat Lon) Z -> B T Z Lat Lon",
                        Lon=data.shape[-1],
                        B=data.shape[0],
                    ).cpu()
                    state = rearrange(
                        state,
                        "(B T) (Lat Lon) Z -> B T Z Lat Lon",
                        Lon=data.shape[-1],
                        B=data.shape[0],
                    ).cpu()
                    snapshot = torch.cat([state[:, 0, 0, :, :], x[:, 0, 0, :, :]]).to(device)
                    snapshots.append(snapshot)
                    dates.append(timestamp.to(device))
                    del x, data, state

                snapshots = torch.stack(snapshots)
                dates = torch.stack(dates)
                if rank == 0:
                    losses_list = [torch.empty_like(valid_losses) for _ in range(world_size)]
                    surface_rmse = [torch.empty_like(surf_error) for _ in range(world_size)]
                    atm_rmse = [torch.empty_like(atm_error) for _ in range(world_size)]
                    snapshots_list = [torch.empty_like(snapshots) for _ in range(world_size)]
                    dates_list = [torch.empty_like(dates) for _ in range(world_size)]
                else:
                    losses_list = None
                    surface_rmse = None
                    atm_rmse = None
                    snapshots_list = None
                    dates_list = None

                dist.gather(valid_losses, losses_list, dst=0)
                dist.gather(surf_error, surface_rmse, dst=0)
                dist.gather(atm_error, atm_rmse, dst=0)
                dist.gather(snapshots, snapshots_list, dst=0)
                dist.gather(dates, dates_list, dst=0)

                if rank == 0:
                    valid_losses = torch.cat(losses_list).cpu()
                    logs["valid/losses/mean"] = valid_losses.mean().item()
                    logs["valid/losses/std"] = valid_losses.std(unbiased=False).item()

                    surf_error = torch.stack(surface_rmse).cpu().mean(0)
                    atm_error = torch.stack(atm_rmse).cpu().mean(0)
                    # Surface
                    for channel_id, var_surf in enumerate(ERA5_SURFACE_VARIABLES):
                        logs[f"valid/surface_rmse/{var_surf}"] = surf_error[channel_id].item()
                    # Atmosphere
                    for channel_id, var_atm in enumerate(ERA5_ATMOSPHERIC_VARIABLES):
                        for level_id, level in enumerate(pressure_levels):
                            idx = channel_id * len(pressure_levels) + level_id
                            logs[f"valid/atmosphere_rmse/{var_atm}_{level}hPa"] = atm_error[
                                idx
                            ].item()

                    # Order snapshot gt/pred pairs by date and take n_snapshots first ones.
                    snapshots = torch.cat(snapshots_list).cpu()
                    dates = torch.cat(dates_list).cpu().squeeze(1)
                    dates = [d.tolist() for d in dates]
                    dates = [
                        f"{date[0]:04d}/{date[1]:02d}/{date[2]:02d} {date[3]:02d}h"
                        for date in dates
                    ]
                    dates, snapshots = zip(*sorted(zip(dates, snapshots)))
                    dates = dates[:n_snapshots]
                    snapshots = torch.stack(snapshots)
                    snapshots = snapshots[:n_snapshots]
                    # Set the same vmap for all snapshots
                    vmin = snapshots[:, 0].min().item()
                    vmax = snapshots[:, 0].max().item()
                    snapshots = rearrange(snapshots, "B T ... -> (T B) ...")

                    fig = plot_image_grid(
                        snapshots,
                        shape=(2, n_snapshots),
                        vmin=vmin,
                        vmax=vmax,
                        x_titles=dates,
                        fontsize=24,
                        tex_font=False,
                    )
                    logs["valid/snapshots"] = wandb.Image(fig)
                    plt.close()

                    if logs["valid/losses/mean"] < best_val_loss:
                        best_val_loss = logs["valid/losses/mean"]
                        safe_save(
                            ae.module.state_dict(),
                            runpath / "model_best.pth",
                        )

                del snapshots, snapshots_list, valid_losses, losses_list
                torch.cuda.empty_cache()

                # Reset to train after validation
                ae.train()

                count_log_val = 0

        if rank == 0 and count_log_save >= cfg.train.save_interval:
            for obj, name in zip(
                (ae.module, optimizer, scheduler), ("model", "optimizer", "scheduler")
            ):
                safe_save(
                    obj.state_dict(),
                    runpath / f"{name}_last.pth",
                )

            with open(runpath / "metadata.yaml", "w") as f:
                metadata = {
                    "last_step_done": step,
                    "best_val_loss": best_val_loss,
                    "lap": lap,
                    "runid": runid,
                }
                metadata.update(metadata_fn(cfg))
                OmegaConf.save(metadata, f)

            count_log_save = 0

        if cfg.valid.log_interval <= count_log_val or cfg.train.log_interval <= count_log_train:
            if cfg.train.log_interval <= count_log_train:
                count_log_train = 0
                losses = []
                grads = []
                latent_stats = []
                if rank == 0:
                    update_steps.set_postfix(loss=logs["train/losses/mean"])
            if rank == 0:
                run.log(logs, step=step)

    if rank == 0:
        run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str, help="Hydra overrides.")
    parser.add_argument("--cpus-per-gpu", type=int, default=8, help="Number of CPU cores per GPU.")
    parser.add_argument("--gpus", type=int, default=4, help="Total number of GPUs.")
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Enforces a number of nodes. Useful to avoid needing full nodes.",
    )
    parser.add_argument("--ram", type=str, default="60GB", help="Amount of RAM per GPU.")
    parser.add_argument("--time", type=str, default="0", help="Time limit (default max Cobalt time)")
    parser.add_argument("--queue", type=str, default="gpu_h100", help="Cobalt JLSE queue")
    # parser.add_argument(
    #     "--partition",
    #     type=str,
    #     default="gpu",
    #     help="Slurm partition.",
    # )
    parser.add_argument("--lap-start", type=int, default=0, help="Lap number to start from.")
    parser.add_argument("--laps", type=int, default=1, help="Maximum number of laps to perform.")
    parser.add_argument(
        "--fork", type=str, default=None, help="Restart the a given runid/lap with a new id."
    )
    parser.add_argument(
        "--continue",
        dest="continue_",
        type=str,
        default=None,
        help="Restart the a given runid/lap with the same id.",
    )

    args = parser.parse_args()

    lap_start = args.lap_start
    laps = args.laps

    assert not (args.continue_ and args.fork), "Cannot continue and fork at the same time."

    if args.continue_:
        runid, lap_start = args.continue_.split("/")
        lap_start = int(lap_start) + 1
        config_path = PATH_AE / args.continue_ / "config.yaml"
    elif args.fork:
        lap_start = int(args.fork.split("/")[1]) + 1
        config_path = PATH_AE / args.fork / "config.yaml"

        runid = wandb.util.generate_id()
        forked_lap_path = PATH_AE / args.fork
        new_lap_path = PATH_AE / runid / f"{lap_start - 1}"

        # Create symlink to the run we fork from
        os.makedirs(PATH_AE / runid, exist_ok=True)
        new_lap_path.symlink_to(forked_lap_path)

        args.overrides = [f"++forked_from={args.fork}"] + args.overrides
    else:
        config_path = PROJECT/"experiments/autoencoder/configs/train.yaml"
        runid = wandb.util.generate_id()

    cfg = compose(
        config_file=config_path,
        overrides=args.overrides,
    )

    if args.nodes is not None:
        assert args.gpus % args.nodes == 0, "Number of GPUs must be divisible by number of nodes."

        num_nodes = args.nodes
        num_gpus = args.gpus // num_nodes
    else:
        # gpu_per_nodes = 8 if args.partition == "ia" else 4  # gpu
        # num_nodes = args.gpus // gpu_per_nodes if args.gpus >= gpu_per_nodes else 1
        # num_gpus = min(args.gpus, gpu_per_nodes)  # Local number of GPUs.
        num_nodes = 1
        num_gpus = args.gpus

    match = re.match(r"(\d+)([A-Za-z]+)", args.ram)
    ram_amount, unit = int(match.group(1)) * num_gpus, match.group(2)
    ram = f"{ram_amount}{unit}"

    if num_nodes > 1:
        interpreter = f"uv run torchrun --nnodes {num_nodes} --nproc-per-node {num_gpus} --rdzv_backend=c10d --rdzv_endpoint=${{SLURMD_NODENAME:-$(head -n 1 $COBALT_NODEFILE)}}:12345 --rdzv_id=${{SLURM_JOB_ID:-$COBALT_JOBID}}"
    else:
        interpreter = f"uv run torchrun --nnodes 1 --nproc-per-node {num_gpus} --standalone"

    env = [
        "export OMP_NUM_THREADS=" + f"{args.cpus_per_gpu}",
        "export WANDB_SILENT=true",
        "export XDG_CACHE_HOME=$HOME/.cache",
        "export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor",
    ]
    job_name = "appa_ae_" + re.sub(r"\W", "_", runid)
    job = dawgz.job(
        partial(train, runid, cfg, lap_start),
        name=job_name,
        interpreter=interpreter,
        env=env,
        nodes=num_nodes,
        cpus=args.cpus_per_gpu * num_gpus,
        gpus=num_gpus,
        ram=ram,
        time=args.time,
        queue=args.queue,
        # account=cfg.slurm_account,
    )

    previous_job = None
    for lap in range(lap_start, lap_start + args.laps):
        current_job = job(lap)

        if previous_job is not None:
            current_job.after(previous_job)

        previous_job = current_job

    dawgz.schedule(
        current_job,
        name=f"appa ae {runid}",
        backend="cobalt",
    )
