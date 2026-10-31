# KohakuMoss Trainer for ComfyUI

The output node calls the native trainer **as a library inside ComfyUI**. It holds
ComfyUI's execution queue through cache preparation, training, exports and cleanup.
There is no background trainer process or separate Python environment. Native CLI
commands still work independently.

Install the repository root with **Manager -> Install via Git URL**:
`https://github.com/deng232/KohakuMossTrainer`. Manager runs root `install.py` using
its current Python, installing editable `kohakumoss[cuda]` into that environment.
The installer constrains installed torch, torchvision, torchaudio, Triton and CUDA
packages. It rejects incompatible torch rather than replacing ComfyUI's GPU stack.
Requires Linux, NVIDIA GPU and torch >=2.13,<2.15. Restart ComfyUI after repair.

Manual repair, **inside the running ComfyUI container**, uses:

```bash
cd /workdir/qwen21/custom_nodes/KohakuMossTrainer
/runtime/venv/bin/python install.py
```

Do not also install/link the nested integration: duplicate nodes and routes can
conflict. Old `.venv`, caches, checkpoints and history are left intact. Keep the
same workspace path when migrating. `KOHAKUMOSS_VENV` is no longer used.

Choose local transformer, Qwen text encoder and Flux2 VAE files using the node's
model selectors. Choose a dataset folder; the browser uploads images and caption
sidecars under ComfyUI's input directory. Queue Prompt starts training. The example
[workflow](examples/kohakumoss_single_node_workflow.json) retains the same inputs
and `job_id` output. Learning rates such as `0.0001` retain their precision.

The KohakuMoss sidebar receives WebSocket phase/metric events and periodically
retrieves persisted snapshots, recent logs, loss history, results and complete
checkpoints. Reopening the sidebar or reconnecting restores the job state. Native
ComfyUI progress advances by optimizer step. Preparation phases report their
activity without pretending to know a percentage; cache batches report missing
items and cache hits. Training reports loss, LR, step duration, ETA and GPU memory.
Deferred metrics are marked with `metrics_step`; an unsynced loss is unavailable.

Both ComfyUI Interrupt and sidebar Stop finish the current optimizer step, save a
complete native checkpoint, then clean up. Stop during preparation cancels at the
next safe boundary; there is no training state to save yet. Loading weights or
compiling a kernel can delay this boundary. There is no forced process kill.
Continue training returns a standard ComfyUI prompt which the frontend adds to
ComfyUI's queue. It uses original job settings, selected complete checkpoint and
additional steps; it cannot start training outside the queue. Failed jobs show
errors both in ComfyUI and the sidebar.

Only one embedded GPU device with `PARALLEL='none'` is supported. Dataset scanning
and loading do not spawn worker processes. ComfyUI inference models are unloaded
before training; queued inference waits until trainer cleanup finishes. The
library avoids CLI signal/atexit hooks and restores host RNG/loggers/tuning state.

Server settings:

| Variable | Default |
| --- | --- |
| `KOHAKUMOSS_REPO` | Ancestor trainer checkout, installed into the same Python |
| `KOHAKUMOSS_WORKSPACE` | `<ComfyUI output>/kohakumoss` |
| `KOHAKUMOSS_DATA_ROOT` | ComfyUI input directory |

Dataset/output paths are confined to configured roots. LoRA publication never
overwrites an existing model. Job metadata/logs/configs are under workspace/jobs;
encoder caches are under workspace/cache. Kernel/model download caches use the
host process's normal `XDG_CACHE_HOME`, `HF_HOME`, `TRITON_CACHE_DIR`, etc. An old
trainer environment does not control these caches. Container cache persistence
therefore depends on your existing environment variables and mounted directories.

CPU checks (with trainer dependencies in the current environment):

```bash
python -m unittest discover -s integrations/ComfyUI-KohakuMoss-Trainer/tests -v
python -m pytest --noconftest tests/test_embedded_events.py tests/test_embedded_training.py
node --test integrations/ComfyUI-KohakuMoss-Trainer/tests/*.mjs
```

CPU tests cover blocking queue ownership, safe cancellation boundaries, failed
initialization cleanup, host state restoration, config compatibility, upload/path
security, publication and frontend behavior. Actual model/GPU training, exported
LoRA inference and repeated-run memory behavior require hardware smoke tests.

Routes inherit ComfyUI access controls and reject cross-origin browser requests.
They expose local paths and logs; use authenticated access with trusted operators.
The integration uses the repository's [license](../../LICENSE).
