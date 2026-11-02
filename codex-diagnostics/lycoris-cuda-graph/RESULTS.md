# LyCORIS CUDA Graph probe results

Interpreter requested: `/runtime/venv/bin/python`

BLOCKED: the requested ComfyUI interpreter is unavailable. No GPU experiment was started; no dependency or environment changes were made. Set PY to the existing ComfyUI interpreter and rerun.

| Experiment | Exit code |
| --- | --- |
| pointer-async | not run |
| pointer-native | not run |
| A-async-graph | not run |
| B-async-no-graph | not run |
| C-native-graph | not run |

## Evidence

See `environment.log` for versions, device availability, repository commit and working tree.

## Root-cause confidence

**Undetermined.** All five GPU experiments are unrun, not passed or failed.
The runner exited 2 because /runtime/venv/bin/python is absent. NVIDIA device nodes
are absent, host Torch/Triton/LyCORIS are missing, nvidia-smi exited 139, and
podman cannot access /home/deng/.config. No environment repair was attempted.

The implementation passed Python syntax/CLI-help and Bash syntax checks only.
Next GPU investigation targets remain LyCORIS _Batch.__init__ and
_bypass_fwd_run; no fix or successful preservation of runtime performance is claimed.

## Run in the existing container

```sh
podman exec -w /workdir/qwen21/custom_nodes/KohakuMossTrainer qwen21 \
  bash .codex-diagnostics/lycoris-cuda-graph/run.sh
```

The default target interpreter is /runtime/venv/bin/python. Existing logs are
preserved; subsequent runs write into a timestamped subdirectory. Each A/B/C run
uses seed 0, the same shapes/dtype and a new disposable LyCORIS cache. Only child
probe processes run: no training, package patching or dependency installation.
The probe may abort its own process; full stdout/stderr and its exit code are saved.

## Interpretation and limits

Observed: only exit codes and logs above. A zero exit alone does not prove every graph was captured: LyCORIS can fall back. Check FIRST EXCEPTION and IN GRAPH CAPTURE records.

Inferred: compare A/B/C before attributing failure to allocator, graph capture or integration. No automatic upstream blame or repair claim is made.

Unverified: real training, offload/ring behavior, backward kernels, actual training shapes, BF16 and a root-cause fix. The probe uses FP16 (512,256), up (192,8), down (8,256). Pointer queries and tracing may perturb capture; pointer mode is separate from A/B/C.

No fix applied: Triton, Graph timing and cudaMallocAsync remain enabled in A; B only disables LyCORIS graph timing; C only changes the allocator.

Changes: root lycoris_cuda_graph_probe.py and diagnostic files only. Package patches: none. Runtime patches: none.


```text
?? .codex-diagnostics/lycoris-cuda-graph/RESULTS.md
?? .codex-diagnostics/lycoris-cuda-graph/environment.log
?? .codex-diagnostics/lycoris-cuda-graph/run.sh
?? KohakuMoss_LyCORIS_CUDA_Graph_Codex_Test.md
?? lycoris_cuda_graph_probe.py
```
