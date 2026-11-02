#!/usr/bin/env bash
# Run only disposable probe processes, never the ComfyUI server.
set -uo pipefail
ulimit -c 0
cd "$(dirname "$0")/../.."
diagnostic_dir="$PWD/.codex-diagnostics/lycoris-cuda-graph"
probe_python="${PY:-/runtime/venv/bin/python}"
if [[ -e "$diagnostic_dir/environment.log" ]]; then
  diagnostic_dir="$diagnostic_dir/$(date -u +%Y%m%dT%H%M%S)-$$"
fi
mkdir -p "$diagnostic_dir"

report() {
  {
    printf '# LyCORIS CUDA Graph probe results\n\n'
    printf 'Interpreter requested: `%s`\n\n' "$probe_python"
    printf '%s\n\n' "$1"
    printf '| Experiment | Exit code |\n| --- | --- |\n'
    for experiment in pointer-async pointer-native A-async-graph B-async-no-graph C-native-graph; do
      code='not run'
      [[ ! -f "$diagnostic_dir/$experiment.exit" ]] || code="$(cat "$diagnostic_dir/$experiment.exit")"
      printf '| %s | %s |\n' "$experiment" "$code"
    done
    printf '\n## Evidence\n\nSee `environment.log` for versions, device availability, repository commit and working tree.\n'
    for experiment in pointer-async pointer-native A-async-graph B-async-no-graph C-native-graph; do
      if [[ -f "$diagnostic_dir/$experiment.log" ]]; then
        printf '\n### %s.log\n\n```text\n' "$experiment"
        awk '/allocator:|Graph timing|Graph Timing|IN GRAPH CAPTURE|driver_result|FIRST EXCEPTION|Traceback|Error:|completed/ {print NR ": " $0; if (++n == 35) exit}' "$diagnostic_dir/$experiment.log"
        printf '```\n'
      fi
    done
    printf '\n## Interpretation and limits\n\n'
    printf 'Observed: only exit codes and logs above. A zero exit alone does not prove every graph was captured: LyCORIS can fall back. Check FIRST EXCEPTION and IN GRAPH CAPTURE records.\n\n'
    printf 'Inferred: compare A/B/C before attributing failure to allocator, graph capture or integration. No automatic upstream blame or repair claim is made.\n\n'
    printf 'Root-cause confidence: undetermined until experiments run. Next investigation targets: LyCORIS _Batch.__init__ and _bypass_fwd_run; no performance feature is disabled by default.\n\n'
    printf 'Unverified: real training, offload/ring behavior, backward kernels, actual training shapes, BF16 and a root-cause fix. The probe uses FP16 (512,256), up (192,8), down (8,256). Pointer queries and tracing may perturb capture; pointer mode is separate from A/B/C.\n\n'
    printf 'No fix applied: Triton, Graph timing and cudaMallocAsync remain enabled in A; B only disables LyCORIS graph timing; C only changes the allocator.\n\n'
    printf 'Changes: root lycoris_cuda_graph_probe.py and diagnostic files only. Package patches: none. Runtime patches: none.\n\n'
    git diff --stat
    printf '\n```text\n'
    git status --short --untracked-files=all
    printf '```\n'
  } > "$diagnostic_dir/RESULTS.md"
  printf 'Report: %s/RESULTS.md\n' "$diagnostic_dir"
}

{
  date -Is
  printf 'Requested Python: %s\n' "$probe_python"
  for device_path in /runtime/venv/bin/python /dev/nvidia0 /dev/nvidiactl; do
    if [[ -e "$device_path" ]]; then
      printf '%s: present\n' "$device_path"
    else
      printf '%s: missing\n' "$device_path"
    fi
  done
  git remote -v
  git rev-parse HEAD
  git status --short --untracked-files=all
  nvidia-smi
  printf 'nvidia-smi exit: %s\n' "$?"
  printf 'Workaround references (empty means none):\n'
  rg -n 'lycoris_timing_compat|check_allocator' kohaku_comfy || true
} > "$diagnostic_dir/environment.log" 2>&1

if ! command -v "$probe_python" >> "$diagnostic_dir/environment.log" 2>&1; then
  report 'BLOCKED: the requested ComfyUI interpreter is unavailable. No GPU experiment was started; no dependency or environment changes were made. Set PY to the existing ComfyUI interpreter and rerun.'
  exit 2
fi

env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_ALLOC_CONF -u CUDA_LAUNCH_BLOCKING \
  "$probe_python" -u - <<'PY' >> "$diagnostic_dir/environment.log" 2>&1
import hashlib
import importlib.metadata as metadata
from pathlib import Path
import sys
import torch
import triton
import lycoris
from lycoris.kernels.tuner import measure, timing
from lycoris.kernels.triton.lora import stream
print("Python:", sys.executable, sys.version, flush=True)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
print("Triton:", triton.__version__, "LyCORIS:", metadata.version("lycoris-lora"), flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), "No usable CUDA GPU"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
for module in (timing, measure):
    source = Path(module.__file__)
    print("Source:", source, "sha256:", hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
assert callable(measure.time_candidates) and callable(timing._Batch.__init__)
assert callable(stream.lora_bypass_fwd) and callable(stream._bypass_fwd_run)
PY
if [[ "$?" != 0 ]]; then
  report 'BLOCKED: environment validation failed; see environment.log. No GPU experiment was started, and dependencies were not changed.'
  exit 2
fi

run_case() {
  local name="$1" backend="$2"
  shift 2
  env -u PYTORCH_CUDA_ALLOC_CONF -u CUDA_LAUNCH_BLOCKING \
    PYTORCH_ALLOC_CONF="backend:$backend" \
    "$probe_python" -u lycoris_cuda_graph_probe.py "$@" \
    > "$diagnostic_dir/$name.log" 2>&1
  printf '%s\n' "$?" > "$diagnostic_dir/$name.exit"
}

run_case pointer-async cudaMallocAsync pointer
run_case pointer-native native pointer
run_case A-async-graph cudaMallocAsync lycoris
run_case B-async-no-graph cudaMallocAsync lycoris --no-graph-timing
run_case C-native-graph native lycoris
report 'Five independent probe processes completed. Inspect first exceptions and actual allocator values in each full log before drawing conclusions.'
