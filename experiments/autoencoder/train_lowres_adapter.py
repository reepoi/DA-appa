r"""Train the low-resolution adapter autoencoder."""

import argparse
import dawgz
import os
import re
import wandb

from functools import partial
from omegaconf import DictConfig, OmegaConf

from appa.config import PROJECT, PATH_AE
from appa.config.hydra import compose
from appa.save import safe_load, select_ae_architecture

from experiments.autoencoder.lowres_adapter import LowResAdaptedConvAE
from experiments.autoencoder.train import train as train_autoencoder


def build_lowres_adapter(cfg: DictConfig, device):
    r"""Load the pretrained CAE and wrap it in the low-resolution adapter."""

    ae_arch = select_ae_architecture(cfg.ae.name)
    ae = ae_arch(**cfg.ae).to(device)
    checkpoint = safe_load(cfg.adapter.pretrained_checkpoint, map_location=device)
    incompatible = ae.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Pretrained checkpoint did not match cae_f32 exactly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    return LowResAdaptedConvAE(
        ae,
        lowres_shape=tuple(cfg.adapter.lowres_shape),
        injection_shape=tuple(cfg.adapter.injection_shape),
        freeze_reused=not cfg.adapter.train_reused,
        input_channels=cfg.ae.in_channels,
        block_type=cfg.adapter.block_type,
        blocks_per_stage=cfg.adapter.blocks_per_stage,
    ).to(device)


def lowres_loss_shape(cfg: DictConfig) -> tuple[int, int]:
    r"""Return the latitude/longitude shape used by the low-resolution loss."""

    return tuple(cfg.adapter.lowres_shape)


def lowres_metadata(cfg: DictConfig) -> dict:
    r"""Return additional metadata for low-resolution adapter checkpoints."""

    return {
        "adapter_train_reused": bool(cfg.adapter.train_reused),
        "adapter_block_type": str(cfg.adapter.block_type),
        "adapter_blocks_per_stage": int(cfg.adapter.blocks_per_stage),
    }


def lowres_resume_state(
    ae,
    optimizer,
    scheduler,
    cfg: DictConfig,
    prev_runpath,
    device,
    forked_run: bool,
):
    r"""Resume low-res adapter runs, resetting optimizer/scheduler on fork by default."""

    ckpt = safe_load(prev_runpath / "model_last.pth", map_location=device)
    ae.module.load_state_dict(ckpt, strict=False)

    with open(prev_runpath / "metadata.yaml", "r") as f:
        metadata = OmegaConf.load(f)

    if not forked_run:
        prev_train_reused = bool(metadata.get("adapter_train_reused", cfg.adapter.train_reused))
        if prev_train_reused != bool(cfg.adapter.train_reused):
            raise ValueError(
                "adapter.train_reused changed during --continue/non-fork resume. "
                "Use --fork to change the trainable parameter set."
            )

    if forked_run and cfg.adapter.reset_optimizer_on_fork:
        return 0, float("inf")

    optimizer.load_state_dict(safe_load(prev_runpath / "optimizer_last.pth", map_location=device))
    scheduler.load_state_dict(safe_load(prev_runpath / "scheduler_last.pth", map_location=device))

    return metadata["last_step_done"] + 1, float(metadata["best_val_loss"])


def train(
    runid: str,
    cfg: DictConfig,
    fork_lap: int = 0,
    lap: int = 0,
):
    r"""Train one lap of the low-resolution adapted autoencoder."""

    return train_autoencoder(
        runid=runid,
        cfg=cfg,
        fork_lap=fork_lap,
        lap=lap,
        build_model_fn=build_lowres_adapter,
        loss_shape_fn=lowres_loss_shape,
        resume_state_fn=lowres_resume_state,
        metadata_fn=lowres_metadata,
        run_label="lowres_adapter",
        description_prefix="Low-res adapter AE",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str, help="Hydra overrides.")
    parser.add_argument("--cpus-per-gpu", type=int, default=8, help="Number of CPU cores per GPU.")
    parser.add_argument("--gpus", type=int, default=4, help="Total number of GPUs.")
    parser.add_argument("--nodes", type=int, default=None, help="Enforces a number of nodes.")
    parser.add_argument("--ram", type=str, default="60GB", help="Amount of RAM per GPU.")
    parser.add_argument("--time", type=str, default="0", help="Time limit (default max Cobalt time)")
    parser.add_argument("--queue", type=str, default="gpu_h100", help="Cobalt JLSE queue")
    parser.add_argument("--lap-start", type=int, default=0, help="Lap number to start from.")
    parser.add_argument("--laps", type=int, default=1, help="Maximum number of laps to perform.")
    parser.add_argument("--fork", type=str, default=None, help="Fork a given runid/lap with a new id.")
    parser.add_argument(
        "--continue",
        dest="continue_",
        type=str,
        default=None,
        help="Restart a given runid/lap with the same id.",
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
        os.makedirs(PATH_AE / runid, exist_ok=True)
        new_lap_path.symlink_to(forked_lap_path)
        args.overrides = [f"++forked_from={args.fork}"] + args.overrides
    else:
        config_path = PROJECT / "experiments/autoencoder/configs/train_lowres_adapter.yaml"
        runid = wandb.util.generate_id()

    cfg = compose(config_file=config_path, overrides=args.overrides)

    if args.nodes is not None:
        assert args.gpus % args.nodes == 0, "Number of GPUs must be divisible by number of nodes."
        num_nodes = args.nodes
        num_gpus = args.gpus // num_nodes
    else:
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
    job_name = "appa_lowres_ae_" + re.sub(r"\W", "_", runid)
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
    )

    previous_job = None
    for lap in range(lap_start, lap_start + laps):
        current_job = job(lap)
        if previous_job is not None:
            current_job.after(previous_job)
        previous_job = current_job

    dawgz.schedule(current_job, name=f"appa lowres ae {runid}", backend="cobalt")
