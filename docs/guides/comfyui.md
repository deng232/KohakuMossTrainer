# KohakuMoss Training for ComfyUI

Three workflow nodes: **KohakuMoss Folder Loader**, **KohakuMoss Training**
and **KohakuMoss uPlot Monitor**.
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
and resolves LyCORIS from the patched `deng232/LyCORIS` GitHub fork, pinned to
commit `72f9d62972490b9636e88a75e12ea614e77f26cb`. No local `LyCORIS/` directory or
runtime patch hook is required. Git must be available to install this dependency.
Manager's Git updates also update the imported trainer source. The installer does not
clone or pin a second upstream trainer repository. It requires an already installed,
stable `torch>=2.13,<2.15`, constrains installed Torch companion packages, and
fails instead of replacing an incompatible Torch stack. Other required Python
dependencies (including native CUDA extras) are resolved in the shared environment.
The installer prefers uv (Python module or executable), explicitly targeting the
ComfyUI interpreter for installation and dependency checks, and falls back to pip.
uv-only environments do not need pip or a separate venv.
No packages are installed during node import,
workflow load, or training execution. Restart ComfyUI after installation.

`requirements.txt` deliberately contains no unconstrained Kohaku dependency:
the guarded installer is responsible for dependency installation. Installation
can run pip subprocesses; training never launches another training process.

## Workflow

Manager's "Used In Workflow" is a workflow/package usage view, not a backend
node-registration check. Load the example workflow or add the KohakuMoss
nodes before checking it. If the nodes are searchable but the usage view still
shows zero, Manager may lack this Git repository's node-to-package mapping or
have stale cached metadata. The root `node_list.json` supplies explicit node
names for Manager's scanner; it does not itself update the central Manager
database. If the nodes are not searchable, inspect ComfyUI startup output for
an import/installation failure instead.

Load [examples/folder-training-workflow.json](../../examples/folder-training-workflow.json).
Set `WEIGHTS` to your ComfyUI models folder or a supported HF weights root,
choose the transformer/text encoder/VAE, and connect Folder Loader → Training
→ Monitor. Set `dataset_path` on Folder Loader.
Use **Choose dataset folder (upload)** directly below `dataset_path` on the
Folder Loader node to open the browser's local
folder picker, like Load Image's upload action. Images and `.txt`, `.caption`,
`.json`, `.jsonl` sidecars are uploaded with their subdirectory structure to
`ComfyUI/input/kohakumoss_datasets/<unique-id>/`. Other file types are skipped.
The server path is filled in only after every supported file succeeds; this
does not queue training. Hand-entered server paths remain supported. Files are
limited to 100 MiB each. Failed/cancelled uploads can leave earlier completed
files in their isolated folder; no existing dataset is overwritten.

Target area uses **MP**: `1.0` means 1,000,000 pixels. The native `TARGET_AREA`
knob still receives integer pixel area. Older graph workflows with a pixel-valued widget convert to MP on
load; older API prompts should use `target_megapixels`.

The model dropdowns list explicit files from ComfyUI's model search paths.
Select transformer, text encoder and VAE files before queueing; automatic model
lookup and the VARIANT selector have been removed. Base/sft is determined by
the transformer file you select. Old automatic selections must be replaced.

Folder Loader recursively reads images and same-name caption sidecars (for
example `1.png` and `1.txt`). The complete text is used as the caption:
tags, natural language and mixed text all work, without tag shuffling or
dropout. No `source`, `CAPTION` or `DATASETS` widget is needed. Dataset and
caption settings come from the loader and take precedence over advanced JSON.
Numeric directory prefixes such as `10_character` are ordinary folder names;
this loader does not apply Kohya repeat weights or add folder names to captions.
Native cache storage is the only dataset cache used by this extension.
Cache files use the native default `<dataset_path>/cache`; no cache-directory
widget is needed.

Older two-node workflows require adding Folder Loader and connecting its dataset
output. Old Training widget values are migrated, but dataset paths must be set
on the loader. The updated example includes all three nodes.

Connect `training_job` to the monitor and Queue Prompt. Training alone is also
an output node and can be queued. Every explicit queue runs again, even with
unchanged settings; `run_token` is a legacy workflow marker with no effect on
training or caching. Loading
or refreshing the page does not queue a run.

The monitor shows actual metric steps rather than interpolated progress.
Defaults are `METRICS_EVERY=5`, `LOG_EVERY=10`, `METRICS_LEVEL=bench`; set the
first to 1 for per-step metrics at additional synchronization/message cost.
Loss, LR and available native VRAM metrics are drawn with local uPlot 1.6.31,
at most twice per second. Missing/non-finite metric values appear as gaps.
Smoothing affects only the displayed loss, not the native history.

## Advanced JSON

JSON widgets such as `SCHEDULE` and encoder settings map to native dictionaries.
The `advanced_json` control is hidden and contributes no overrides (`{}`).
Old workflow override values are discarded on load. Training uses the exposed
widgets and native defaults, followed by Folder Loader settings and enforced
single-GPU host constraints.

The native parameter list and defaults are in [the training script](../../scripts/train/moss.py);
behavior is described in [the training documentation](../concepts/training.md).
Native validation still decides which training/model/cache combinations work.

Offline cache passes run sequentially in the prompt worker and report written
caption/latent counts. They use the native encoder identity/compute-spec checks.
Cache enumeration uses estimated token lengths so it can build an initially
empty exact-token cache; the training token-budget setting is unchanged.
Online/off cache modes continue through the native training path.

Use native `SAVE_EVERY`, `SAVE_MINUTES`, `KEEP_LAST` and `FINAL_CHECKPOINT` for
checkpoints, and `SAVE_FORMAT=comfy` for ComfyUI exports. `RESUME=latest` applies
when you queue a new task using the same output directory/run name. It is not
an in-memory continuation.

**Browse server checkpoints**, below RESUME, opens an in-app folder dialog
listing directories on the **ComfyUI server**, starting at ComfyUI's output
directory. Navigate into a run's `checkpoints/step_XXXXXXXX` folder and choose
**Use this checkpoint**. The folder must contain `state.json` and
`trainable.safetensors`; partial directories are hidden. The action fills in
RESUME without queueing training. No local files are selected or uploaded.
Browsing is limited to ComfyUI's output directory, including symlink targets.
For checkpoints elsewhere, enter their server path manually. Empty RESUME
starts fresh; `latest` uses this run's newest native checkpoint.

Increasing steps on resume may change the native
scheduler curve. Existing native run-file rotation rules remain in effect.

There are no Pause, Continue, Stop or Extend controls. ComfyUI Interrupt is
not a reliable safe stop for Kohaku training. Refreshing the browser leaves
training running. Other ordinary queued GPU workflows wait for training.

## Events and recovery

The frontend associates the monitor with the training node through its graph
link, not through an upstream runtime output. A persistent workflow identifier,
the graph node ID and a per-execution `job_id` distinguish runs. Put the training and monitor
nodes in the main workflow graph; subgraph execution IDs have not been validated.

Events: `kohakumoss.job_started`, `kohakumoss.metrics`, `kohakumoss.status`.
Every event includes task identity, monotonic `event_seq` and a JSON-safe
snapshot; metric events also include `step` and `metrics`.

Read-only routes, under ComfyUI's existing server and access policy:

- `GET /kohakumoss/checkpoints?path=...` (server output folder browser)
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
the native files/checkpoints do survive. Job HTTP routes are read-only, not controls.
Dataset upload uses separate `POST /kohakumoss/datasets` and
`POST /kohakumoss/datasets/{upload_id}/files` routes under the same server policy.

## Lightweight checks

With the shared environment's existing dependencies, from the repository root:

```sh
python -m unittest discover -s tests/comfyui -v
node --test tests/comfyui/*.test.js
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
