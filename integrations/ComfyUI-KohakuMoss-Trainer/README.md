# ComfyUI-KohakuMoss-Trainer

One **KohakuMoss Trainer** output node submits background training and returns a
STRING `job_id`. The **KohakuMoss** sidebar shows job history, native metrics, a loss
chart, recent console output, checkpoints, exports, Stop and Resume. Training runs
in its own Python environment; this plugin never imports the trainer into ComfyUI.

Initial target: Linux, one NVIDIA GPU, with the native 12 GiB memory preset as the
default. This is a source-checkout integration, not a replacement training loop.

## Installation

### ComfyUI-Manager (complete fork)

Install `https://github.com/deng232/KohakuMossTrainer` with Manager's
**Install via Git URL**, wait for dependency setup and restart ComfyUI. The fork's
root `__init__.py` loads this integration and serves its web widgets; its root
`install.py` runs the same installer described below. Do not add a second symlink
to this nested directory when using the root installation. Disable an existing
manual node link before restarting. Use Manager's install/repair operation to
retry a failed dependency installation.

### Manual source-checkout installation

Use a checkout containing this integration. ComfyUI-Manager's install/repair
operation runs this plugin's `install.py`, which calls the
[official installer](../../docs/guides/install.md) to create the trainer's
**separate** environment, install dependencies and run the doctor. Downloads happen
during installation, never when importing the node or submitting training.
`KOHAKUMOSS_REPO` and `KOHAKUMOSS_VENV` are respected during installation too.

For an existing manually linked checkout that Manager does not list, run the same
hook inside the ComfyUI container:

```bash
cd /workdir/qwen21/custom_nodes/ComfyUI-KohakuMoss-Trainer
/runtime/venv/bin/python install.py
```

This hook does not register a local checkout in Manager's public catalog. The
plugin still requires its ancestor trainer checkout (or `KOHAKUMOSS_REPO`).

Alternatively, in the trainer checkout, install manually:

```bash
bash scripts/install/install.sh
.venv/bin/kohakumoss-doctor
```

Or install manually with `uv` (example for a driver supporting CUDA 13.0; choose the
matching PyTorch index as described in the install guide):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'torch>=2.13,<2.15' torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -e '.[cuda]' --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
.venv/bin/kohakumoss-doctor
```

Link the plugin into ComfyUI's active custom-node directory, using absolute paths:

```bash
ln -s /path/to/KohakuMossTrainer/integrations/ComfyUI-KohakuMoss-Trainer /path/to/ComfyUI/custom_nodes/ComfyUI-KohakuMoss-Trainer
```

Restart ComfyUI. ComfyUI V3 node API and the public extension sidebar API are
required (development reference: ComfyUI 0.42.0, commit
`d07dff4565ea9804277fff8cdfa0c817e08c3d44`, frontend 1.57.1).
The ComfyUI side uses only the standard library and ComfyUI's existing `aiohttp`.
Do **not** install the trainer's requirements into ComfyUI's environment.

Server-side environment variables, set before starting ComfyUI:

| Variable | Default | Purpose |
| --- | --- | --- |
| `KOHAKUMOSS_REPO` | This plugin's ancestor trainer checkout | Native scripts and presets; install this checkout editable into the training venv |
| `KOHAKUMOSS_VENV` | `<trainer>/.venv` | Separate environment with `python`, `kogine`, `kohakumoss-config` |
| `KOHAKUMOSS_WORKSPACE` | `<ComfyUI output>/kohakumoss` | Job metadata, configs, logs, runs, shared native caches and kernel caches |
| `KOHAKUMOSS_DATA_ROOT` | ComfyUI input directory | Allowed root for resolved dataset directories |

Use the **same workspace** across ComfyUI instances sharing this GPU. The workspace
lock limits the entire inspect/cache/train/export pipeline to one active job;
another submission is rejected with a busy error. This does not reserve memory
against other training tools or ordinary ComfyUI inference.

No dependencies or weights are installed or downloaded during node execution. Worker
environment variables are restricted; hub access is offline. Weight files must
already exist in ComfyUI's registered model directories, including
`extra_model_paths.yaml` directories.

## Dataset and workflow

Put images with matching `.txt` sidecars under the dataset root. `folder` accepts
an image tree; `kohya` follows the native `<repeats>_<name>/` folder convention
(the kohya preset excludes root images). See [native data documentation](../../docs/concepts/data.md).

1. Import [the example workflow](examples/kohakumoss_single_node_workflow.json), or
   add **Kohaku/Training -> KohakuMoss Trainer**.
2. Click **choose folder** to open the browser's native folder picker. The selected
   images and caption sidecars are uploaded together, preserving their relative
   directory structure. Wait for **Ready to queue training**. Select the MOSS diffusion model, Qwen3.5 text encoder and
   Flux2 VAE using the normal model lists.
3. Set rank, steps and learning rate; leave `vram_12g`, `offline_both`, periodic
   saves initially. Training-time sampling is disabled; test exported LoRAs in your inference workflow.
4. Queue Prompt. The node returns a job id without waiting for training. Open the
   **KohakuMoss** sidebar to inspect the job and control it.

An unchanged workflow can be queued again to start another unique run. Retries
of the same prompt/node submission reuse their job id. Merely opening the sidebar
or editing a widget does not start training. Widget changes do not modify an
already submitted job.

Resize and batch controls (`resolution_mode`, `target_mp`, `max_tokens`,
`batching`, `batch_size`, `grad_accum`) are visible without opening Advanced.
**Max target area (MP)** uses decimal megapixels: 1 MP = 1,000,000 pixels. The default
is 1 MP; 1.048576 MP corresponds to 1024 x 1024 and 0.589824 MP to 768 x 768.
The native resizer aligns output dimensions to multiples of 16 and retains the
original aspect ratio in `resize` mode. Existing job metadata using pixel areas
remains resumable. Dataset selection uses a native `<input type="file"
webkitdirectory>` folder picker, like an image upload picker. Browsers supply
relative paths, not usable local absolute paths, so the dataset is copied to
`<KOHAKUMOSS_DATA_ROOT>/.kohaku_uploads/<unique id>/<selected folder>/`. The server
path is saved in the workflow only after every file succeeds. Existing files are
never replaced; selecting again creates another upload. Original client files are
not modified. Supported image extensions match the native folder source: JPG,
JPEG, PNG, WebP, BMP, JXL and AVIF; TXT, CAPTION and JSON sidecars are included.
Other files are ignored. Native folder selection requires a browser supporting
`webkitdirectory`. Each file uses ComfyUI's configured upload-size limit; an error
keeps the previously selected dataset. Files from interrupted uploads remain on
disk and can be removed manually when no longer needed.
The training seed stays fixed between submissions; there is no automatic seed
control widget to shift serialized workflow values.

Successful native `comfy` exports are copied atomically to `output_path`, which
defaults to the first registered ComfyUI `loras` directory. Existing LoRAs are never replaced; conflicts gain a
numeric suffix. Refresh the model lists if necessary and use the normal **Load
LoRA** node with a compatible native MOSS model.

## Configuration and cache behavior

The worker runs the native commands, with argument arrays and no shell:

```text
kohakumoss-config new train ... --out <job>/train.py
kohakumoss-config check <job>/train.py <job>/train_data.py
kogine run scripts/data/inspect_data.py --config <job>/train_data.py
kogine run scripts/data/cache_text.py --config <job>/train_data.py
kogine run scripts/data/cache_latents.py --config <job>/train_data.py
kogine run scripts/train/moss.py --config <job>/train.py
```

Validated literals specialize newly generated configs. The generated data config
inherits the train config and retains the native encoder factories, with the
selected encoder/VAE paths and training placement plans. The text cache factory
uses the same compute settings as training. `max_tokens=0` preserves the native
profile limit; the 12 GiB preset currently supplies 6144 tokens. `batch_size`
applies only with `batching=count`; `grad_accum` maps to `GRAD_ACC`, separately
from `MICROBATCHES`. `caption_template=native` preserves the preset's caption dict.

`offline_both` prepares both native caches before training. `auto` preserves the
generator's cache policy (currently offline for both) and performs both incremental
cache passes. `off` skips those passes and disables both training caches. Native
cache keys decide valid reuse. The workspace cache namespace includes the dataset
path and encoder/VAE file path, size and modification time, to separate different
weight files. Keep those attributes accurate when replacing weights. Data presets
and caption/resolution settings remain covered by the native catalog/cache keys.

Images and captions are not rewritten. `CACHE_DIR` is explicitly set under the
workspace rather than the native dataset-local default. Job configs, actual argv,
trainer commit, package versions, model paths/stat information and host resource
warnings are stored in `<workspace>/jobs/<job_id>/job.json`.

## Stop, resume and restart

Stop sends SIGTERM to the process group started for this job. The state remains
`stopping` until the command and its live descendants have exited. After 30 seconds,
the owned group is forcibly stopped. If exit still cannot be confirmed, the job
becomes `unknown` and retains the GPU lock until ComfyUI is restarted, rather than
allowing another training job. **Stopping does not guarantee a newly saved checkpoint.**

Use **choose checkpoint** in the node to select a complete server-side checkpoint
from a previous job, including a completed job. This selector is hidden in `new`
mode and shown in `resume_latest` mode. Selection sets the resume mode and
job automatically. In resume mode, `steps` means **additional steps**: checkpoint
2000 plus 1000 steps trains to 3000. The sidebar also offers a checkpoint selector
and Additional steps. Other training settings come from the original job.

`output_path` provides **choose folder** to browse server-side LoRA directories.
Unlike the input upload picker, it selects a server directory without uploading
files. It defaults to ComfyUI's registered LoRA directory and accepts that
absolute path or a relative subdirectory inside it. Only the exported, loadable
LoRA `.safetensors` is published there; configs, optimizer state, complete
checkpoints and logs stay in the original workspace. Exports receive unique names
so earlier results remain available. New runs save a final complete checkpoint.
After training completes, refresh your LoRA loader and test the published file,
then select the checkpoint and queue additional training. Training submission is
asynchronous; wait for completion before running inference with the new LoRA.

Checkpoint discovery matches the native complete-directory test: a final step
name plus `state.json` and `trainable.safetensors`; `.partial` directories are
ignored. The native loader validates the remaining checkpoint contents.

Metadata and native logs survive a ComfyUI restart. A possibly surviving worker is
marked `unknown` and blocks new submissions. The restarted server does not adopt
or signal inherited PIDs. Stop that worker externally, then restart ComfyUI again;
it will mark the job `interrupted` and offer Resume when a checkpoint exists.
The recorded Linux boot/process identity helps identify the previous worker safely.

Console logs are capped at 64 MiB per job; the oldest console output is discarded
at that limit. Native metrics/logs remain in the run directory. The sidebar keeps
the latest 1000 metric rows from at most the last 2 MiB; the complete native
`metrics.jsonl` stays on disk. Missing files, malformed/incomplete rows, rotation
and resume truncation are tolerated.

## Memory, security and verification

For a 3080 Ti, start with existing int8 ConvRot MOSS and Qwen3.5 weights and the
small Flux2 VAE, native `vram_12g`, and sampling disabled. Rank 32 and the preset
are starting settings, **not a guarantee** of fitting every dataset in 12 GiB.
Streamed weights, optimizer states and caches also require substantial RAM and
disk. Avoid inference while training. OOM and nonzero exits are visible as failed
jobs; they are not retried automatically. Native export is performed at the end
of the training command before publication.

The routes inherit ComfyUI's access controls and reject browser cross-origin
requests. They add no authentication system. Keep the server local or behind
authenticated access for trusted operators; an unauthenticated public `--listen`
server is unsafe for training jobs and exposes local job/log paths. Repo/venv
paths are server configuration, not remotely editable workflow fields. No
arbitrary Python config/command input is exposed.

Basic CPU verification, without the training dependencies:

```bash
python -m unittest discover -s integrations/ComfyUI-KohakuMoss-Trainer/tests -v
node --input-type=module --check < integrations/ComfyUI-KohakuMoss-Trainer/web/trainer.js
node --test integrations/ComfyUI-KohakuMoss-Trainer/tests/test_panel.mjs
node --test integrations/ComfyUI-KohakuMoss-Trainer/tests/test_dataset_browser.mjs
```

Tests use CLI process fixtures and small ComfyUI boundary doubles; they do not
emulate a model or prove training correctness. Native CLI `check`/inspect, real
cache construction, GPU training, browser rendering and loading the exported LoRA
require the separately installed environment and actual weights. These have not
been validated by the CPU checks.

The integration is covered by the repository's [KohakuMoss License](../../LICENSE).
Upstream training reference: `9102868723a119887734353363d0732b60229aef`.
