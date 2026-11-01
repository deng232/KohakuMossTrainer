# Validation record

This is an implementation with lightweight checks, not a real-training acceptance report.

| Component | Observed version / status |
| --- | --- |
| Python | 3.14.8 |
| Node.js | 24.21.0 |
| KohakuMoss | Source APIs at `1a40815022906fd9d2a19f9b2da823b91d1c1b01` |
| uPlot | Vendored 1.6.31, upstream ESM/CSS and MIT license |
| Torch / CUDA | Not installed / not available for validation here |
| ComfyUI / frontend | No runnable local installation |

Passed: 19 Python unittest checks and 6 Node built-in tests; Ruff, Black and
JavaScript syntax checks. Python tests use the real native ConsoleLogger and
registry, temporary metric files and an actual local aiohttp test server.
Runtime orchestration tests replace GPU training at its library boundary.
No fake model/encoder was trained. No ComfyUI dependency environment was modified.
Test utilities were supplied through Nix, without creating a venv.

A headless-browser attempt could not start correctly in this isolated environment.
Browser/GPU test harnesses were removed and further environment troubleshooting
stopped at the user's request. No rendered-chart or real-ComfyUI success is claimed.
The full trainer test suite was not run: this change is confined to the extension,
and the isolated environment has no Torch, real model, dataset or training GPU.

Pending on the deployment machine:

1. Manager installation in the existing ComfyUI interpreter; confirm the Torch
   stack is unchanged and Kohaku imports from this checkout's `src/` directory.
2. A short real training run: see actual step/loss in uPlot before Training
   returns and before Monitor's backend executes; confirm the queue waits.
3. Refresh/reconnect/relink the monitor, then repeat Queue with unchanged inputs.
4. Build the native text/latent caches, save/export, resume a checkpoint and
   verify restored step/optimizer/scheduler/loader behavior.
5. Run ordinary ComfyUI inference after training, and inspect GPU memory,
   repeated-registration/resource cleanup and the compiler's helper processes.
6. Repeat single-GPU acceptance on Windows. Native compile/allocator caches and
   unselected-device RNG behavior require particular attention in a shared process.

There are no core patches, training subprocesses, stop/pause controls, or an
independent venv in this extension. The no-subprocess claim is limited to its
training launcher and the explicitly disabled native train/data process branches;
kernel/compiler helper processes have not been measured.
