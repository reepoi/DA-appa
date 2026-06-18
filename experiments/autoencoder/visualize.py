# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.8",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="wide")


@app.cell
def _():
    import h5py
    import marimo as mo
    import numpy as np
    import torch

    from einops import rearrange
    from matplotlib import pyplot as plt
    from omegaconf import OmegaConf
    from pathlib import Path

    return OmegaConf, Path, h5py, mo, np, plt, rearrange, torch


@app.cell
def _(Path):
    DEFAULT_RUN_PATH = Path("/vast/users/ac.ttransue/appa/autoencoders/xla0mo5m/1")
    DEFAULT_DAWGZ_PATH = Path("/home/ac.ttransue/GitHub/DA-appa/.dawgz/parched_present_5eb61d25")
    return DEFAULT_DAWGZ_PATH, DEFAULT_RUN_PATH


@app.cell
def _(mo, DEFAULT_RUN_PATH):
    intro = mo.md(
        "# Autoencoder Run Viewer\n\n"
        "Interactive inspection for the autoencoder trained by dawgz workflow "
        "`parched_present_5eb61d25`, second lap (`xla0mo5m/1`).\n\n"
        "Use the controls below to inspect run metadata, load a checkpoint, and "
        "run a small on-demand reconstruction against ERA5 validation/test data."
    )
    run_path_box = mo.ui.text(
        value=str(DEFAULT_RUN_PATH),
        label="Autoencoder lap path",
        full_width=True,
    )
    checkpoint_choice = mo.ui.dropdown(
        options=["model_best", "model_last"],
        value="model_best",
        label="Checkpoint",
    )
    device_choice = mo.ui.dropdown(
        options=["auto", "cuda", "cpu"],
        value="auto",
        label="Device",
    )
    mo.vstack([intro, run_path_box, mo.hstack([checkpoint_choice, device_choice])])
    return checkpoint_choice, device_choice, run_path_box


@app.cell
def _(OmegaConf, Path, run_path_box):
    run_path = Path(run_path_box.value).expanduser()
    config_path = run_path / "config.yaml"
    metadata_path = run_path / "metadata.yaml"
    config_exists = config_path.exists()
    metadata_exists = metadata_path.exists()
    ae_config = OmegaConf.load(config_path) if config_exists else None
    ae_metadata = OmegaConf.load(metadata_path) if metadata_exists else None
    return ae_config, ae_metadata, config_exists, metadata_exists, run_path


@app.cell
def _(mo, run_path, config_exists, metadata_exists):
    status_items = [
        {"item": "run path", "value": str(run_path), "available": run_path.exists()},
        {"item": "config.yaml", "value": str(run_path / "config.yaml"), "available": config_exists},
        {"item": "metadata.yaml", "value": str(run_path / "metadata.yaml"), "available": metadata_exists},
        {"item": "model_best.pth", "value": str(run_path / "model_best.pth"), "available": (run_path / "model_best.pth").exists()},
        {"item": "model_last.pth", "value": str(run_path / "model_last.pth"), "available": (run_path / "model_last.pth").exists()},
    ]
    mo.ui.table(status_items, label="Run artifacts")
    return (status_items,)


@app.cell
def _(OmegaConf, ae_config, ae_metadata, mo):
    if ae_config is None:
        run_summary = mo.md("No `config.yaml` found for this run path.")
    else:
        ae_section = ae_config.get("ae", {})
        train_section = ae_config.get("train", {})
        config_summary = {
            "architecture": ae_section.get("name"),
            "latent_shape": list(ae_section.get("latent_shape", [])),
            "input_channels": ae_section.get("in_channels"),
            "context_channels": ae_section.get("context_channels"),
            "sub_pressure_levels": train_section.get("sub_pressure_levels"),
            "update_steps_target": train_section.get("update_steps"),
            "forked_from": ae_config.get("forked_from", None),
        }
        metadata_summary = (
            OmegaConf.to_container(ae_metadata, resolve=True) if ae_metadata is not None else {}
        )
        run_summary = mo.vstack(
            [
                mo.md("## Run Summary"),
                mo.ui.table(
                    [{"field": k, "value": str(v)} for k, v in metadata_summary.items()],
                    label="metadata.yaml",
                ),
                mo.ui.table(
                    [{"field": k, "value": str(v)} for k, v in config_summary.items()],
                    label="config.yaml highlights",
                ),
            ]
        )
    run_summary
    return


@app.cell
def _(ae_config):
    from appa.data.const import (
        ERA5_ATMOSPHERIC_VARIABLES as ATMOSPHERIC_VARIABLES,
        ERA5_PRESSURE_LEVELS as ALL_PRESSURE_LEVELS,
        ERA5_SURFACE_VARIABLES as SURFACE_VARIABLES,
        SUB_PRESSURE_LEVELS as TRAINING_PRESSURE_LEVELS,
    )

    sub_pressure_levels = True
    if ae_config is not None:
        sub_pressure_levels = bool(ae_config.get("train", {}).get("sub_pressure_levels", True))

    pressure_levels = TRAINING_PRESSURE_LEVELS if sub_pressure_levels else ALL_PRESSURE_LEVELS
    feature_options = []
    feature_names = []
    for feature_index, surface_variable in enumerate(SURFACE_VARIABLES):
        label = f"{feature_index:02d} | {surface_variable}"
        feature_options.append(label)
        feature_names.append(surface_variable)
    offset = len(SURFACE_VARIABLES)
    for atm_index, atm_variable in enumerate(ATMOSPHERIC_VARIABLES):
        for level_index, level in enumerate(pressure_levels):
            feature_index = offset + atm_index * len(pressure_levels) + level_index
            label = f"{feature_index:02d} | {atm_variable} @ {level} hPa"
            feature_options.append(label)
            feature_names.append(atm_variable)
    return feature_names, feature_options, pressure_levels, sub_pressure_levels


@app.cell
def _(feature_options, mo):
    start_date_box = mo.ui.text(value="2020-01-01", label="Start date (YYYY-MM-DD)")
    start_hour_slider = mo.ui.slider(start=0, stop=23, step=1, value=0, label="Start hour")
    sample_index_slider = mo.ui.slider(start=0, stop=63, step=1, value=0, label="Sample index")
    feature_choice = mo.ui.dropdown(
        options=feature_options,
        value=feature_options[0],
        label="Feature to plot",
        full_width=True,
    )
    run_reconstruction_button = mo.ui.run_button(label="Run reconstruction")
    mo.vstack(
        [
            mo.md("## Reconstruction Controls"),
            mo.hstack([start_date_box, start_hour_slider, sample_index_slider]),
            feature_choice,
            run_reconstruction_button,
        ]
    )
    return (
        feature_choice,
        run_reconstruction_button,
        sample_index_slider,
        start_date_box,
        start_hour_slider,
    )


@app.cell
def _(torch, device_choice):
    if device_choice.value == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = device_choice.value
    return (device_name,)


@app.cell
def _(checkpoint_choice, mo, run_path):
    selected_checkpoint_path = run_path / f"{checkpoint_choice.value}.pth"
    mo.md(f"Selected checkpoint: `{selected_checkpoint_path}`")
    return (selected_checkpoint_path,)


@app.cell
def _(
    checkpoint_choice,
    device_name,
    mo,
    rearrange,
    run_path,
    run_reconstruction_button,
    sample_index_slider,
    start_date_box,
    start_hour_slider,
    torch,
):
    mo.stop(
        not run_reconstruction_button.value,
        mo.md("Click **Run reconstruction** to load the model and evaluate one ERA5 sample."),
    )

    from appa.config import PATH_ERA5, PATH_STAT
    from appa.data.const import (
        CONTEXT_VARIABLES,
        DATASET_DATES_VALIDATION,
        ERA5_VARIABLES,
        SUB_PRESSURE_LEVELS,
    )
    from appa.data.datasets import ERA5Dataset
    from appa.data.transforms import StandardizeTransform
    from appa.save import load_auto_encoder

    transform = StandardizeTransform(
        PATH_STAT,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=SUB_PRESSURE_LEVELS,
    )
    dataset = ERA5Dataset(
        path=PATH_ERA5,
        start_date=start_date_box.value,
        start_hour=start_hour_slider.value,
        end_date=DATASET_DATES_VALIDATION[-1],
        num_samples=sample_index_slider.value + 1,
        transform=transform,
        trajectory_size=1,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=SUB_PRESSURE_LEVELS,
    )

    state, context, timestamp = dataset[sample_index_slider.value]
    state_batch = rearrange(state[None], "B T Z Lat Lon -> (B T) (Lat Lon) Z").to(device_name)
    context_batch = rearrange(context[None], "B T Z Lat Lon -> (B T) (Lat Lon) Z").to(device_name)
    timestamp_batch = rearrange(timestamp[None], "B T D -> (B T) D").to(device_name)

    autoencoder = load_auto_encoder(
        run_path,
        model_name=checkpoint_choice.value,
        device=device_name,
        eval_mode=True,
    ).to(device_name)
    autoencoder.requires_grad_(False)

    with torch.no_grad():
        latent, reconstruction = autoencoder(state_batch, timestamp_batch, context_batch)

    lon = state.shape[-1]
    state_grid = rearrange(state_batch.cpu(), "T (Lat Lon) Z -> T Z Lat Lon", Lon=lon)
    reconstruction_grid = rearrange(
        reconstruction.cpu(), "T (Lat Lon) Z -> T Z Lat Lon", Lon=lon
    )
    state_unstd, _ = transform.unstandardize(state_grid)
    reconstruction_unstd, _ = transform.unstandardize(reconstruction_grid)
    error_unstd = reconstruction_unstd - state_unstd
    mse_by_feature = (error_unstd).square().mean(dim=(-1, -2)).squeeze(0)
    standardized_mse_by_feature = (reconstruction_grid - state_grid).square().mean(dim=(-1, -2)).squeeze(0)
    latent_cpu = latent.detach().cpu()
    timestamp_list = timestamp.tolist()

    return (
        error_unstd,
        latent_cpu,
        mse_by_feature,
        standardized_mse_by_feature,
        reconstruction_unstd,
        state_unstd,
        timestamp_list,
    )


@app.cell
def _(feature_choice):
    selected_feature_index = int(feature_choice.value.split(" | ")[0])
    selected_feature_label = feature_choice.value.split(" | ", 1)[1]
    return selected_feature_index, selected_feature_label


@app.cell
def _(
    error_unstd,
    feature_names,
    mo,
    mse_by_feature,
    standardized_mse_by_feature,
    np,
    selected_feature_index,
    selected_feature_label,
):
    selected_error = error_unstd[0, selected_feature_index]
    selected_abs_error = selected_error.abs()
    base_variable_name = feature_names[selected_feature_index]
    mse = float(mse_by_feature[selected_feature_index])
    standardized_mse = float(standardized_mse_by_feature[selected_feature_index])
    metrics_rows = [
        {"metric": "MSE", "value": mse},
        {"metric": "RMSE", "value": float(np.sqrt(mse))},
        {"metric": "standardized MSE", "value": standardized_mse},
        {"metric": "standardized RMSE", "value": float(np.sqrt(standardized_mse))},
        {"metric": "mean signed error", "value": float(selected_error.mean())},
        {"metric": "mean absolute error", "value": float(selected_abs_error.mean())},
        {"metric": "p95 absolute error", "value": float(np.quantile(selected_abs_error.numpy(), 0.95))},
        {"metric": "max absolute error", "value": float(selected_abs_error.max())},
    ]
    mo.vstack(
        [
            mo.md(f"## Metrics: `{selected_feature_label}` (`{base_variable_name}`)"),
            mo.ui.table(metrics_rows),
        ]
    )
    return base_variable_name, metrics_rows


@app.cell
def _(
    error_unstd,
    np,
    plt,
    reconstruction_unstd,
    selected_feature_index,
    selected_feature_label,
    state_unstd,
    timestamp_list,
):
    ground_truth_image = state_unstd[0, selected_feature_index].numpy()
    reconstruction_image = reconstruction_unstd[0, selected_feature_index].numpy()
    error_image = error_unstd[0, selected_feature_index].numpy()

    value_min = min(float(ground_truth_image.min()), float(reconstruction_image.min()))
    value_max = max(float(ground_truth_image.max()), float(reconstruction_image.max()))
    error_abs_max = max(abs(float(error_image.min())), abs(float(error_image.max())))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    images = [ground_truth_image, reconstruction_image, error_image]
    titles = ["Ground truth", "Reconstruction", "Signed error"]
    cmaps = ["viridis", "viridis", "seismic"]
    limits = [(value_min, value_max), (value_min, value_max), (-error_abs_max, error_abs_max)]
    for axis, image, title, cmap, (vmin, vmax) in zip(axes, images, titles, cmaps, limits):
        # ERA5Dataset returns spatial tensors as (longitude, latitude). Matplotlib
        # expects image rows, columns, so transpose to (latitude, longitude).
        plot_image = np.roll(image, image.shape[0] // 2, axis=0).T
        plotted = axis.imshow(
            plot_image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=(-180, 180, -90, 90),
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("West to East")
        axis.set_ylabel("South to North")
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.set_yticks([-90, -45, 0, 45, 90])
        fig.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"{selected_feature_label} at {timestamp_list[0]}")
    fig
    return


@app.cell
def _(latent_cpu, mo):
    latent_rows = [
        {"metric": "shape", "value": str(tuple(latent_cpu.shape))},
        {"metric": "mean", "value": float(latent_cpu.mean())},
        {"metric": "std", "value": float(latent_cpu.std(unbiased=False))},
        {"metric": "min", "value": float(latent_cpu.min())},
        {"metric": "max", "value": float(latent_cpu.max())},
    ]
    mo.vstack([mo.md("## Latent Summary"), mo.ui.table(latent_rows)])
    return (latent_rows,)


@app.cell
def _(h5py, mo, run_path):
    reconstruction_dirs = sorted((run_path / "reconstruction").glob("*/errors.h5")) if (run_path / "reconstruction").exists() else []
    reconstruction_rows = []
    for errors_path in reconstruction_dirs:
        with h5py.File(errors_path, "r") as errors_file:
            reconstruction_rows.append(
                {
                    "id": errors_path.parent.name,
                    "path": str(errors_path),
                    "std_mse_shape": str(errors_file["std_mse"].shape)
                    if "std_mse" in errors_file
                    else "missing",
                    "hist_shape": str(errors_file["signed_errors_histograms"].shape)
                    if "signed_errors_histograms" in errors_file
                    else "missing",
                }
            )
    precomputed_summary = (
        mo.ui.table(reconstruction_rows, label="Precomputed reconstruction outputs")
        if reconstruction_rows
        else mo.md("No precomputed `reconstruction/*/errors.h5` outputs found for this lap.")
    )
    precomputed_summary
    return (reconstruction_rows,)


if __name__ == "__main__":
    app.run()
