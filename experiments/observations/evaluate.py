r"""Script to evaluate trajectories against ground truth."""

import numpy as np
import os
import shutil
import sys
import torch
import wandb
import xarray as xr

from dawgz import after, job, schedule
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path
from weatherbench2 import config as wb2_config
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam
from weatherbench2.metrics import CRPS, EnsembleMeanMSE, EnsembleVariance
from weatherbench2.regions import LandRegion

from appa.config import PATH_ERA5, PATH_MASK, PATH_STAT, compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TEST,
    ERA5_PRESSURE_LEVELS,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.datasets import ERA5Dataset
from appa.data.mappings import tensor_to_xarray
from appa.data.transforms import StandardizeTransform
from appa.save import load_auto_encoder, safe_load

METRICS = {
    "crps": {
        "estimator": CRPS,
        "distributed": True,  # To be computed with WeatherBench
        "root": False,  # Square root applied after ensemble average
    },
    "skill": {
        "estimator": EnsembleMeanMSE,
        "distributed": True,
        "root": True,
    },
    "spread": {
        "estimator": EnsembleVariance,
        "distributed": True,
        "root": True,
    },
    "spread_skill_ratio": {
        "estimator": None,
        "distributed": False,
        "root": False,
    },
}
# arXiv:2504.18720v3 Sec. 4.4 (Fig. 8): this script reports CRPS, skill (RMSE of
# ensemble mean), and spread statistics for reanalysis/filtering/forecast runs.


def tensor_to_datetime(tensor):
    return np.array([
        np.datetime64(f"{yy}-{mm:02d}-{dd:02d}T{hh:02d}", "ns") for (yy, mm, dd, hh) in tensor
    ])


def ensemble_to_xarray(
    ensemble,
    timestamps,
    ensemble_dim: str = "ensemble",
):
    lead = np.array([np.timedelta64(ti - timestamps[0], "ns") for ti in timestamps])
    xr_pred = tensor_to_xarray(
        ensemble, ERA5_VARIABLES, sub_levels=SUB_PRESSURE_LEVELS, roll=False
    )
    xr_pred = xr_pred.expand_dims({"time": np.array([timestamps[0]])})
    xr_pred = xr_pred.rename({"trajectory": "prediction_timedelta", "batch": ensemble_dim})
    xr_pred = xr_pred.assign_coords(level=SUB_PRESSURE_LEVELS)
    lat = torch.linspace(90, -90, 721)
    lon = torch.linspace(0, 360 - 360 / 1440, 1440)
    xr_pred = xr_pred.assign_coords({"latitude": lat, "longitude": lon})
    xr_pred = xr_pred.assign_coords(prediction_timedelta=lead)

    return xr_pred


def ensemble_statistics(
    timestamps,
    eval_configs,
    output_path,
    variables,
    pressure_levels,
    ensemble_id: int,
    ensemble_size: int,
    trajectory_dt: int = 1,  # TODO infer from timestamps, avoid using timedeltas.
    use_beam: bool = False,
    beam_chunk_size: int = 75,
    ensemble_dim: str = "ensemble",
):
    r"""Computes statistics for a single ensemble."""

    out_folder = output_path.parent

    tmp_folder = out_folder / f"tmp_{output_path.stem}"
    tmp_folder.mkdir(parents=True, exist_ok=True)

    timestamps = tensor_to_datetime(timestamps[0])
    selection = wb2_config.Selection(
        variables=variables,
        levels=pressure_levels,
        time_slice=slice(
            timestamps[0],
            timestamps[0] + np.timedelta64((len(timestamps) - 1) * trajectory_dt, "h"),
            trajectory_dt,
        ),
    )

    all_preds_path = out_folder / f"tmp_ensemble_{ensemble_id}" / "predictions_all.zarr"

    paths = wb2_config.Paths(
        obs=PATH_ERA5,
        forecast=all_preds_path,
        output_dir=tmp_folder,
    )

    data_config = wb2_config.Data(selection=selection, paths=paths)

    if use_beam:
        evaluate_with_beam(
            data_config,
            eval_configs,
            runner="DirectRunner",
            input_chunks={"lead_time": beam_chunk_size},
        )
    else:
        evaluate_in_memory(data_config, eval_configs)

    # Open ensemble.nc
    nc_results = xr.open_dataset(tmp_folder / "ensemble.nc")
    # Save to zarr
    nc_results.to_zarr(output_path, mode="w")

    shutil.rmtree(tmp_folder)

    print(f"Saved to .../{Path(*output_path.parts[-6:])}.")


def main():
    config = compose("configs/evaluate.yaml", overrides=sys.argv[1:])

    runs_path = Path(config.runs_path)

    if "denoisers" in str(runs_path):  # Forecast, ...
        denoiser_dir = runs_path.parents[1]
        latent_dir = runs_path.parents[4]

        denoiser_cfg = compose(denoiser_dir / "config.yaml")
        trajectory_dt = denoiser_cfg.train.blanket_dt
    else:  # Latent trajectories
        latent_dir = runs_path.parents[1]
        gen_metadata = compose(runs_path / "metadata.yaml")
        trajectory_dt = gen_metadata.trajectory_dt

    if "id" in config:
        unique_id = config["id"]
    elif config.hardware.backend == "slurm":
        unique_id = wandb.util.generate_id()
    else:
        unique_id = os.environ["SLURM_JOB_ID"]  # even in async, should be in slurm job

    metrics = config.metrics

    mask = xr.open_zarr(PATH_MASK)
    mask = mask.sea_surface_temperature_mask
    region = LandRegion(mask)

    ensemble_dim = "ensemble"
    eval_configs = {
        "ensemble": wb2_config.Eval(
            metrics={
                metric: METRICS[metric]["estimator"](ensemble_dim=ensemble_dim)
                for metric in metrics
                if METRICS[metric]["distributed"]
            },
            regions={"ocean": region, "global": None},
        )
    }

    ae_cfg = compose(latent_dir / "ae" / "config.yaml")
    stats_path = latent_dir / "stats.pth"
    latent_stats = safe_load(stats_path)
    latent_mean = latent_stats["mean"]
    latent_std = latent_stats["std"]
    latent_std = torch.sqrt(latent_std**2 + ae_cfg.ae.noise_level**2)
    ae_pressure_levels = (
        SUB_PRESSURE_LEVELS if ae_cfg.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
    )

    st = StandardizeTransform(PATH_STAT, state_variables=ERA5_VARIABLES, levels=ae_pressure_levels)
    era5mean, era5std = st.state_mean.squeeze(), st.state_std.squeeze()

    variables = list(config.variables)

    variables_names = [v.name for v in variables]
    variables_levels = list(set(l for v in variables if "levels" in v for l in v["levels"]))

    jobs = []
    ensemble_size = None

    # Iterate over folders in runs_path
    for i, pred_folder in enumerate(runs_path.iterdir()):
        if not pred_folder.is_dir() or not pred_folder.name.endswith("h"):
            continue

        if config.runs != "all":
            if isinstance(config.runs, str):
                if config.runs != pred_folder.name:
                    continue
            elif pred_folder.name not in config.runs:
                continue

        if not (pred_folder / "trajectories.pt").exists():
            continue

        out_folder = pred_folder / "evaluation" / unique_id
        out_folder.mkdir(parents=True, exist_ok=True)

        # if (Path(out_folder / "metrics.zarr")).exists():
        #     if not config.recompute:
        #         continue
        #     else:
        #         os.remove(out_folder / "metrics.zarr")

        OmegaConf.save(config, out_folder / "config.yaml")

        times = safe_load(pred_folder / "timestamps.pt")

        num_ensembles = times.shape[0]
        ensemble_size = times.shape[1]

        @job(
            name=f"appa eval (decode {pred_folder.name})",
            array=num_ensembles * ensemble_size,
            **config.hardware.decode,
        )
        def decode_to_zarr(
            i: int,
            pred_folder=pred_folder,
            out_folder=out_folder,
            era5mean=era5mean,
            era5std=era5std,
            ensemble_size=ensemble_size,
        ):
            r"""Decode a run of an ensemble into a Zarr dataset."""

            import warnings

            warnings.filterwarnings(action="ignore")

            ensemble_id = i // ensemble_size
            ensemble_index = i % ensemble_size

            ensemble_folder = out_folder / f"ensemble_{ensemble_id}.zarr"

            trajectories = safe_load(pred_folder / "trajectories.pt")
            times = safe_load(pred_folder / "timestamps.pt")

            for ti in times:
                assert (
                    ti.unique(dim=0).shape[0] == 1
                ), "Timestamp mismatch in ensemble forecast. Found different timestamps between ensemble members."
            trajectory = trajectories[ensemble_id, ensemble_index] * latent_std + latent_mean
            timestamps = times[ensemble_id, ensemble_index]

            ae = load_auto_encoder(latent_dir / "ae", "model", device="cpu", eval_mode=True)
            ae.requires_grad_(False)
            ae.cuda()

            out_folder = ensemble_folder.parent

            tmp_folder = out_folder / f"tmp_{ensemble_folder.stem}"
            tmp_folder.mkdir(parents=True, exist_ok=True)

            timestamps = tensor_to_datetime(timestamps)

            all_preds_path = tmp_folder / "predictions_all.zarr"

            # Retrieve context for decoding
            variables_and_levels = {
                "state_variables": ERA5_VARIABLES,
                "context_variables": CONTEXT_VARIABLES,
                "levels": ae_pressure_levels,
            }
            st = StandardizeTransform(PATH_STAT, **variables_and_levels)
            era5 = ERA5Dataset(
                path=PATH_ERA5,
                start_date=np.datetime_as_string(timestamps[0], unit="D"),
                start_hour=int(np.datetime_as_string(timestamps[0], unit="h").split("T")[1][:2]),
                end_date=DATASET_DATES_TEST[-1],  # Could just compute the end date
                **variables_and_levels,
                transform=st,
                trajectory_dt=trajectory_dt,
            )

            if not all_preds_path.exists():
                particle_traj = []
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    with torch.no_grad():
                        for i, pj in enumerate(trajectory):
                            _, context, timestamp = era5[i]

                            context = rearrange(context, "T Z Lat Lon -> T (Lat Lon) Z")

                            particle_traj.append(
                                ae.decode(
                                    pj.unsqueeze(0).cuda(), timestamp.cuda(), context.cuda()
                                ).cpu()
                            )

                            del context, timestamp
                ensemble_pred = torch.cat(particle_traj, dim=0).unsqueeze(0)
                del particle_traj

                tmp_file = tmp_folder / f"predictions_{ensemble_index}.zarr"

                ensemble_pred = ensemble_pred * era5std + era5mean
                ensemble_pred = rearrange(
                    ensemble_pred, "B T (Lat Lon) Z -> B T Z Lat Lon", Lat=721
                )

                xr_pred = ensemble_to_xarray(ensemble_pred, timestamps)
                xr_pred = xr_pred[variables_names]
                if "level" in xr_pred.dims:
                    xr_pred = xr_pred.sel(level=variables_levels)
                xr_pred.to_zarr(tmp_file, mode="w")

        @after(decode_to_zarr)
        @job(
            name=f"appa eval (merge decode {pred_folder.name})",
            array=num_ensembles,
            **config.hardware.aggregate,
        )
        def merge_decode(
            i: int,
            out_folder=out_folder,
            ensemble_size=ensemble_size,
        ):
            import warnings

            warnings.filterwarnings(action="ignore")

            tmp_folder = out_folder / f"tmp_ensemble_{i}"
            all_preds_path = tmp_folder / "predictions_all.zarr"

            # Merge Zarr datasets.
            if not all_preds_path.exists() and (tmp_folder / "predictions_0.zarr").exists():
                ensemble_files = [
                    tmp_folder / f"predictions_{i}.zarr" for i in range(ensemble_size)
                ]
                datasets = [xr.open_zarr(p) for p in ensemble_files]
                combined = xr.concat(datasets, dim=ensemble_dim)
                combined.to_zarr(all_preds_path, mode="w")

                for i in range(ensemble_size):
                    shutil.rmtree(tmp_folder / f"predictions_{i}.zarr")

        @after(merge_decode)
        @job(
            name=f"appa eval (metrics {pred_folder.name})",
            array=num_ensembles * len(variables),
            **config.hardware.metrics,
        )
        def compute_ensemble_metrics(
            i: int,
            out_folder=out_folder,
            pred_folder=pred_folder,
            ensemble_size=ensemble_size,
        ):
            import warnings

            warnings.filterwarnings(action="ignore")

            ensemble_id = i // len(variables)
            ensemble_var = i % len(variables)

            times = safe_load(pred_folder / "timestamps.pt")
            timestamps = times[ensemble_id]

            variable = [variables[ensemble_var].name]
            var_levels = (
                variables[ensemble_var]["levels"] if "levels" in variables[ensemble_var] else []
            )

            print("Evaluating", variable, var_levels, flush=True)

            ensemble_statistics(
                timestamps=timestamps,
                eval_configs=eval_configs,
                output_path=out_folder / f"ensemble_{ensemble_id}_{ensemble_var}.zarr",
                variables=variable,
                pressure_levels=var_levels,
                ensemble_id=ensemble_id,
                ensemble_size=ensemble_size,
                trajectory_dt=trajectory_dt,
                use_beam=config.split_chunks,
                beam_chunk_size=config.chunk_size,
                ensemble_dim=ensemble_dim,
            )

        @after(compute_ensemble_metrics)
        @job(
            name=f"appa eval (agg {pred_folder.name})",
            **config.hardware.aggregate,
        )
        def aggregate(
            pred_folder=pred_folder,
            out_folder=out_folder,
            num_ensembles=num_ensembles,
            ensemble_size=ensemble_size,
        ):
            import warnings

            warnings.filterwarnings(action="ignore")

            times = safe_load(pred_folder / "timestamps.pt")

            # Merge each ensemble
            for ensemble_id in range(num_ensembles):
                datasets = []
                for var_id in range(len(variables)):
                    ds = xr.open_zarr(out_folder / f"ensemble_{ensemble_id}_{var_id}.zarr")
                    datasets.append(ds)
                merged = xr.merge(datasets)

                # Zarr cannot store variable-length strings natively
                # Therefore, when writing to disk, objects are converted to
                # fixed-length strings. It infers the max over known strings.
                # This could prevent appending longer strings.
                for v in merged.variables:
                    if merged[v].dtype.kind in {"O", "S", "U"}:
                        merged[v] = merged[v].astype("<U255")

                merged.to_zarr(
                    out_folder / f"ensemble_{ensemble_id}.zarr", mode="w", zarr_format=3
                )

                for var_id in range(len(variables)):
                    shutil.rmtree(out_folder / f"ensemble_{ensemble_id}_{var_id}.zarr")

            datasets = [
                xr.open_zarr(out_folder / f"ensemble_{i}.zarr") for i in range(num_ensembles)
            ]
            for i, ds in enumerate(datasets):
                start_time = tensor_to_datetime(times[i][:1, 0])[0]
                datasets[i] = ds.expand_dims(ensemble=[start_time])
            metrics_ds = xr.concat(datasets, dim="ensemble")

            target_file = out_folder / "metrics.zarr"
            metrics_ds.to_zarr(target_file, mode="w", group="all")

            # Average across ensembles (e.g., forecast, filtering)
            avg_ensembles = metrics_ds.mean(dim="ensemble")
            rooted_metrics = [metric for metric in metrics if METRICS[metric]["root"]]
            avg_ensembles.loc[dict(metric=rooted_metrics)] = np.sqrt(
                avg_ensembles.sel(metric=rooted_metrics)
            )
            if "spread_skill_ratio" in metrics:
                spread = avg_ensembles.sel(metric="spread")
                skill = avg_ensembles.sel(metric="skill")
                ssr = spread / skill * np.sqrt((ensemble_size + 1) / ensemble_size)

                avg_ensembles = xr.concat(
                    [avg_ensembles, ssr.expand_dims(metric=["spread_skill_ratio"])], dim="metric"
                )
            avg_ensembles.to_zarr(target_file, mode="a", group="avg_ensembles")

            # Average across ensembles and over time (e.g., reanalysis)
            avg_time = avg_ensembles.mean(dim="lead_time")
            avg_time.to_zarr(target_file, mode="a", group="avg_time")

            # Display summary.
            tot_space = max([len(var) for var in variables])
            for metric in metrics:
                print(f"--- {metric} ---")
                for var in variables:
                    region = "ocean" if var.name == "sea_surface_temperature" else "global"
                    print(f"  {var.name} ({region})")

                    var_metrics = avg_time.sel(metric=metric, region=region)[var.name]
                    if "levels" in var:
                        values = {
                            str(level): var_metrics.sel(level=level).mean().values.tolist()
                            for level in var["levels"]
                        }
                    else:
                        values = {"surface": var_metrics.mean().values.tolist()}

                    for lvl, val in values.items():
                        print(f"    - {lvl}{' ' * (tot_space - len(lvl))}  {val:.4f}")

            for i in range(num_ensembles):
                shutil.rmtree(out_folder / f"tmp_ensemble_{i}")  # Decoded predictions
                shutil.rmtree(out_folder / f"ensemble_{i}.zarr")  # Computed metrics

            print(f"Saved evaluation to .../{Path(*target_file.parts[-6:])}.")

        jobs.append(aggregate)

    hardware_cfg = config.hardware

    schedule(
        *jobs,
        name="appa evaluation",
        account=hardware_cfg.account,
        backend=hardware_cfg.backend,
        env=[
            f"export OMP_NUM_THREADS={hardware_cfg.metrics.cpus}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
            "export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor",
        ],
    )


if __name__ == "__main__":
    main()
