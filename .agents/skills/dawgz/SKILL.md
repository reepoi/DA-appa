---
name: dawgz
description: Use when users ask about dawgz Python workflow scheduling, @dawgz.job, dawgz.schedule, Job.after, Job.waitfor, dawgz.array, Slurm/Cobalt backends, or DA-appa experiment scripts that submit jobs through dawgz.
---

# dawgz Workflow Scheduling Skill

Use this skill when working with `dawgz`, a lightweight Python library for declaring directed acyclic workflow graphs and scheduling them locally or through HPC backends.

## Core Model

- A dawgz job is a Python function decorated with `@dawgz.job` or `@dawgz.job(...)`.
- Calling a decorated function does not run it immediately. It captures the call as a `dawgz.Job` containing the function, arguments, source, resources, shell, interpreter, and environment.
- Dependencies are attached to captured jobs with `.after(...)`.
- A workflow is submitted with `dawgz.schedule(*jobs, backend=...)`.
- The scheduler walks backward from the submitted leaf jobs, prunes already-marked jobs, rejects cycles, submits dependencies first, records workflow metadata, and returns the scheduler instance.
- Job names must contain only alphanumeric characters and underscores. If no `name=` is provided, the function name is used.

## Minimal Pattern

```python
import dawgz


@dawgz.job(cpus=4, ram="16GB", time="01:00:00", partition="gpu")
def train(seed: int) -> None:
    ...


@dawgz.job(cpus=2, ram="8GB", time="00:15:00")
def evaluate(seed: int) -> None:
    ...


if __name__ == "__main__":
    train_job = train(0)
    eval_job = evaluate(0).after(train_job)

    dawgz.schedule(eval_job, backend="slurm")
```

## Job Decorator Arguments

- `name`: override the job name. Keep it alphanumeric plus underscores.
- `shell`: shell used in generated backend scripts, default `/bin/bash`.
- `interpreter`: command used to execute generated Python wrappers on scheduled workers, default `python`. Use this for runtime parity, for example `uv run --with numpy --with cloudpickle python3`, `python3`, or `torchrun ...`.
- `env`: list of shell lines inserted before the interpreter command, commonly `export ...` or `module load ...`.
- `settings`: backend-specific settings dictionary.
- Additional keyword arguments on `@dawgz.job(...)` are merged into `settings`.

Prefer explicit resource settings at the decorator site so generated scheduler scripts are inspectable through the dawgz CLI.

## Dependencies

- Use `child.after(parent)` for normal success dependencies.
- Use `child.after(parent, status="failure")` when the child should run only if the dependency fails.
- Use `child.after(parent, status="any")` when the child should run regardless of dependency success or failure.
- Use `.waitfor("all")` to require all dependencies. This is the default.
- Use `.waitfor("any")` to run after any dependency satisfies its requested status.
- Use `.mark("success")`, `.mark("failure")`, `.mark("cancelled")`, or `.mark("pending")` to tell dawgz whether an already-created job should be pruned or still scheduled.

Example:

```python
a_job = a()
b_job = b()
c_job = c().after(a_job, b_job).waitfor("all")
fallback_job = fallback().after(a_job, status="failure")
first_done_job = summarize().after(a_job, b_job, status="any").waitfor("any")
```

## Job Arrays

- Use `dawgz.array(*jobs, name=None, throttle=None)` for many independent jobs.
- Jobs inside an array must not already have dependencies or dependents.
- All jobs in an array must share the same `shell`, `interpreter`, `env`, and `settings`.
- `throttle` limits concurrency for Slurm arrays and is emulated for Cobalt by chaining submissions in windows.

```python
jobs = [process(i) for i in range(100)]
array = dawgz.array(*jobs, name="process", throttle=8)
final = merge().after(array)

dawgz.schedule(final, backend="slurm")
```

## Backends

- `async`: runs jobs asynchronously with a local `ProcessPoolExecutor`. Use `max_workers=N` or `max_workers=None`. Ignores resource settings, `interpreter`, and `env`.
- `dummy`: debug backend based on `async`; prints `START ...` and `END ...` instead of running job code.
- `slurm`: writes `.sh`, `.py`, and `.pkl` files, submits with `sbatch --parsable`, checks state with `sacct`, and cancels with `scancel`.
- `cobalt`: writes wrapper files, submits with `qsub`, checks state with `qstat`, and cancels with `qdel`.

When developing a workflow, run it first with `backend="dummy"` or a small `backend="async"` case before submitting to Slurm or Cobalt.

## Slurm Notes

- Most settings are passed directly to `sbatch` after replacing underscores with hyphens.
- Translated settings include `tasks -> ntasks`, `tasks_per_node -> ntasks-per-node`, `cpus -> cpus-per-task`, `gpus -> gpus-per-task`, `ram -> mem`, `memory -> mem`, `timelimit -> time`, and `timeout -> time`.
- If `ntasks` is not set, dawgz defaults to `nodes=1` and `ntasks-per-node=1`.
- Dependencies map to `afterok`, `afternotok`, and `afterany`.
- `.waitfor("any")` uses `?` between dependency clauses; `.waitfor("all")` uses `,`.
- Arrays are submitted with `#SBATCH --array=0-N` or `#SBATCH --array=0-N%throttle`.
- Generated scripts run `srun {interpreter} {pyfile}`.

Common Slurm job settings:

```python
@dawgz.job(
    account="<account>",
    partition="<partition>",
    cpus=8,
    gpus=1,
    ram="32GB",
    time="02:00:00",
)
def work() -> None:
    ...
```

## Cobalt Notes

- Cobalt settings are translated only for `nodes`/`nodecount -> -n`, `time`/`timelimit`/`timeout -> -t`, `queue -> -q`, and `project`/`account -> -A`.
- Unknown settings are ignored by the Cobalt command builder unless handled specially.
- Use `attrs` for placement constraints: `settings={"attrs": {"location": "nid00001,nid00002"}}` or `attrs={...}`.
- Do not use `nodelist` for Cobalt. dawgz warns that it is ignored and recommends `attrs` instead.
- `env` lines are parsed into `qsub --env` only when they start with `export KEY=VALUE`. Other `env` lines still run inside the wrapper script.
- The wrapper script runs `{interpreter} $1` from the dawgz workflow directory.
- Cobalt arrays are implemented as separate `qsub` calls, not native array jobs. With `throttle`, each job after the initial window depends on the job submitted `throttle` positions earlier.

Example Cobalt settings:

```python
@dawgz.job(
    nodes=1,
    queue="<queue>",
    project="<project>",
    time="30:00",
    attrs={"location": "nid00001,nid00002"},
    interpreter="uv run --with numpy --with cloudpickle python3",
)
def work() -> None:
    ...
```

## CLI And Workflow Records

- dawgz records workflows under its dawgz directory and appends `workflows.csv`.
- Use `dawgz` to list workflows.
- Use `dawgz <workflow-index>` to list jobs for one workflow.
- Use `dawgz <workflow-index> <job-index>` to show logs.
- Use `dawgz <workflow-index> <job-index> --input` to show captured call arguments.
- Use `dawgz <workflow-index> <job-index> --source` to show captured source.
- Use `dawgz <workflow-index> <job-index> --settings` to show generated scheduler script when available.
- Use `dawgz --cancel`, `dawgz <workflow-index> --cancel`, or `dawgz <workflow-index> <job-index> --cancel` carefully; backend cancellation calls `scancel` or `qdel`.

For tests or isolated runs, redirect the dawgz metadata directory with:

```python
dawgz.set_dawgz_dir(tmp_path / ".dawgz")
```

## Serialization And Runtime Pitfalls

- dawgz pickles job calls with `cloudpickle`; captured arguments and function context must be serializable.
- Modifying globals after a `Job` has been created does not affect that job's captured function context.
- Imported modules are normally referenced, not copied. If the worker must use the exact current module code, register it with `cloudpickle.register_pickle_by_value(my_module)` before creating jobs.
- For Slurm and Cobalt, make sure the worker environment can import project code and dependencies. Prefer setting `interpreter` and `env` rather than relying on the submit-host shell state.
- Keep workflow construction inside `if __name__ == "__main__":` to avoid accidental scheduling or multiprocessing import issues.
- For local `async`, choose `max_workers` deliberately. The default in this source is `1`, not all cores.

## DA-appa-Specific Guidance

- DA-appa experiment training scripts use dawgz with Cobalt-oriented defaults. Do not assume they are plain local training entrypoints.
- Before changing DA-appa workflow code, inspect the relevant Hydra config and job decorator settings together.
- When converting a DA-appa experiment to local debugging, prefer a minimal `backend="dummy"` or `backend="async"` path and small synthetic inputs rather than deleting resource settings.
- When adding new scheduled steps, submit the final leaf jobs to `dawgz.schedule`; dawgz will include dependencies automatically.

## Agent Checklist

- Identify whether the user wants workflow construction, backend submission, CLI inspection, or debugging.
- Preserve the `@dawgz.job` capture model: create jobs by calling decorated functions, then wire dependencies, then call `dawgz.schedule` under `if __name__ == "__main__":`.
- Use `dummy` or `async` for safe verification unless the user explicitly wants HPC submission.
- For Slurm or Cobalt changes, inspect generated settings with the dawgz CLI or scheduler path when possible.
- Avoid broad cancellation or queue mutation unless the user explicitly confirms target workflow/job indices or backend job IDs.
