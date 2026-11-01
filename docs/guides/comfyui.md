# KohakuMoss Training for ComfyUI

Two workflow nodes: **KohakuMoss Training** and **KohakuMoss uPlot Monitor**.
Kohaku runs as a Python library in ComfyUI's existing prompt worker. Training
occupies the queue until it finishes; the browser receives live metrics while
the monitor's backend is still waiting for the training output.

## Installation

In ComfyUI-Manager, choose **Install via Git URL** and paste this repository's
GitHub URL. The repository root is the custom-node package: it contains
`__init__.py`, `WEB_DIRECTORY`, `install.py` and the native trainer sources.
There is no nested integration directory to install separately.

Use **ComfyUI's own Python environment**. Do not create a separate venv.
The extension provides `install.py` for ComfyUI-Manager's installation hook.
For a manual installation, run this with the Python that launches ComfyUI:

```sh
python custom_nodes/<repository-directory>/install.py
```

The installer installs this checkout in editable mode with its CUDA extras,
so Manager's Git updates also update the imported trainer source. It does not
clone or pin a second upstream trainer repository. It requires an already installed,
stable `torch>=2.13,<2.15`, constrains installed Torch companion packages, and
fails instead of replacing an incompatible Torch stack. Other required Python
dependencies (including native CUDA extras) are resolved in the shared environment. Pip must be
available to that interpreter. No packages are installed during node import,
workflow load, or training execution. Restart ComfyUI after installation.

`requirements.txt` deliberately contains no unconstrained Kohaku dependency:
the guarded installer is responsible for dependency installation. Installation
can run pip subprocesses; training never launches another training process.

## Workflow

Load [examples/two-node-workflow.json](../../examples/two-node-workflow.json).
Set `WEIGHTS` to your ComfyUI models folder or a supported HF weights root,
choose the transformer/text encoder/VAE, and set `dataset_path` and `source`.
The model dropdowns respect ComfyUI's additional model search paths. Absolute
component paths and other native knobs can be supplied through `advanced_json`.

For a flat folder, choose `source=folder`; for Kohya-style repeat directories,
choose `kohya`. An optional nonempty `DATASETS` JSON list replaces the dataset
widget with native multi-source entries. Empty `DATASETS` uses the widgets.
Native cache storage is the only dataset cache used by this extension.

Connect `training_job` to the monitor and Queue Prompt. Training alone is also
an output node and can be queued. Every explicit queue runs again, even with
unchanged settings; `run_token` is an optional manual workflow marker. Loading
or refreshing the page does not queue a run.

The monitor shows actual metric steps rather than interpolated progress.
Defaults are `METRICS_EVERY=5`, `LOG_EVERY=10`, `METRICS_LEVEL=bench`; set the
first to 1 for per-step metrics at additional synchronization/message cost.
Loss, LR and available native VRAM metrics are drawn with local uPlot 1.6.31,
at most twice per second. Missing/non-finite metric values appear as gaps.
Smoothing affects only the displayed loss, not the native history.

JSON widgets map to native dictionaries. `advanced_json` overrides widget
knobs, then host constraints override both: one selected CUDA GPU,
`PARALLEL=none`, `ENCODE_DEVICES=()`, `NUM_WORKERS=0`, `DATA_SCAN_WORKERS=1`,
native driver and the live logger. A separate text-encoder GPU is rejected.
Native validation still decides which training/model/cache combinations work.

Offline cache passes run sequentially in the prompt worker and report written
caption/latent counts. They use the native encoder identity/compute-spec checks.
Cache enumeration uses estimated token lengths so it can build an initially
empty exact-token cache; the training token-budget setting is unchanged.
Online/off cache modes continue through the native training path.

Use native `SAVE_EVERY`, `SAVE_MINUTES`, `KEEP_LAST` and `FINAL_CHECKPOINT` for
checkpoints, and `SAVE_FORMAT=comfy` for ComfyUI exports. `RESUME=latest` applies
when you queue a new task using the same output directory/run name. It is not
an in-memory continuation. Increasing steps on resume may change the native
scheduler curve. Existing native run-file rotation rules remain in effect.

There are no Pause, Continue, Stop or Extend controls. ComfyUI Interrupt is
not a reliable safe stop for Kohaku training. Refreshing the browser leaves
training running. Other ordinary queued GPU workflows wait for training.

## Events and recovery

The frontend associates the monitor with the training node through its graph
link, not through an upstream runtime output. A persistent workflow identifier,
the graph node ID and a per-execution `job_id` distinguish runs. Put the two
nodes in the main workflow graph; subgraph execution IDs have not been validated.

Events: `kohakumoss.job_started`, `kohakumoss.metrics`, `kohakumoss.status`.
Every event includes task identity, monotonic `event_seq` and a JSON-safe
snapshot; metric events also include `step` and `metrics`.

Read-only routes, under ComfyUI's existing server and access policy:

- `GET /kohakumoss/jobs?origin_node_id=...&workflow_id=...`
- `GET /kohakumoss/jobs/{job_id}`
- `GET /kohakumoss/jobs/{job_id}/history`

The server retains 20 recent tasks and at most 10,000 rows per task. History
recovery reads the recorded native `metrics.jsonl`, never an HTTP-supplied
path, and returns at most 10,000 merged rows. Truncated/unavailable history
is marked. Old tasks can read a verified native rotated metrics file; when a
resume/reused directory makes old file ownership ambiguous, only that old
task's retained memory rows are returned. Realtime metrics come from the logger,
not file polling. Task discovery does not survive a ComfyUI server restart;
the native files/checkpoints do survive. HTTP routes are read-only, not controls.

## Lightweight checks

With the shared environment's existing dependencies, from the repository root:

```sh
python -m unittest discover -s tests/comfyui -v
node --test tests/comfyui/monitor.test.js
```

These checks cover configuration boundaries, real ConsoleLogger persistence,
session ownership/locking/history, read-only HTTP behavior and frontend data
binding/recovery. The runtime boundary test substitutes the GPU training call;
it does not train a synthetic model or claim GPU/ComfyUI integration coverage.

Actual-deployment acceptance remains manual: queue a short real training run,
observe loss before it returns, refresh/reconnect the monitor, verify native
cache/checkpoints/export/resume, queue a second run, then run ordinary ComfyUI
inference. Observe the process tree during kernel prewarm/compilation: the fixed
training/data worker settings do not guarantee that compilers spawn no helpers.

The extension temporarily exits ComfyUI's inference mode and restores selected
device/RNG, compiler tag/pool settings and third-party logging levels. Native
kernel caches and allocator initialization can persist for the server lifetime.
Failure recovery and training-then-inference still require real-device validation.

See [validation](comfyui-validation.md) for the checks actually run. uPlot's MIT
license is retained in `web/vendor/LICENSE`; extension code follows the included
KohakuMoss license. No trainer or ComfyUI core source changes are required.
