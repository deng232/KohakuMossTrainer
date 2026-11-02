# FIX_RESULTS

Status: diagnostic implementation ready; GPU repair NOT verified.

## Evidence already measured

Original GPU logs: `20261101T063636-143/`.

| Case | Result |
| --- | --- |
| A: cudaMallocAsync + Graph | Exit 134, Triton pointer error followed by fatal cuMemFreeAsync cleanup |
| B: cudaMallocAsync, no Graph | Exit 0; original probe did not measure numerical error |
| C: native + Graph | Exit 0; original probe did not measure numerical error |
| Async allocation inside capture | Driver pointer query returns 1 before replay, 0 after first replay |

These observations implicate allocation/launcher validation during capture. They do
not yet identify the faulty LyCORIS argument or prove a preallocation repair.
Driver version remains unknown (`nvidia-smi` failed in the original environment).

## Changes and scope

Only `fix_probe.py` and diagnostic reports were added. The probe reads the installed
stream module and executes a separate on-disk copy, preserving relative imports
and Triton source inspection. Installed package files and KohakuMossTrainer's
training path are never written. Source replacements require exactly one match;
source drift fails explicitly. Each GPU case has a fresh process and tuning cache.

The test-copy forward function optionally accepts `out`. Its benchmark factory
allocates an output per candidate/buffer-set closure before capture and retains
that tensor throughout graph benchmarking. The direct backward experiment
optionally accepts independent `gx/h/q` scratch tensors. Its returned GEMM outputs
remain graph-managed Torch tensors. This is not a complete backward-tuner repair.

`locate` wraps the real Triton launch, records all five pointer arguments in launch
order, and uses the original tuner graph configuration. Original A/B/C are rerun.
The independent copy experiment uses the same warm stream, shared pool and
thread-local capture mode as LyCORIS. Preallocation experiments are gated on
observing only `out` fail its pointer query and a passing preallocated copy test.

The fused experiment exercises FP16/BF16, three independent buffer sets, forward
and backward, five replays with changing inputs, input preservation, disjoint
scratch/output addresses, numerical errors and CUDA-event microbenchmark timing.
The autotuning experiment retains all candidates, including TORCH and ATOMIC;
unsupported branches remain visible as failures. Every tuner Graph fallback is
recorded and causes failure even if the tuner catches its exception.

## Run inside the existing container

From the host (no shell quoting or heredoc needed):

```sh
podman exec qwen21 /runtime/venv/bin/python -u /workdir/qwen21/custom_nodes/KohakuMossTrainer/codex-diagnostics/lycoris-cuda-graph/fix_probe.py suite
```

Use the actual installed repository path if different. The suite creates a unique
`fix-TIMESTAMP-PID/` directory next to this report containing logs, exit codes,
source copies, an experimental source diff and its own `FIX_RESULTS.md`. A failing
child cannot terminate ComfyUI. Each child is limited to 900 seconds. Expected
reproduction failures still make the suite return nonzero: inspect individual
cases, not just the suite exit status. Negative child exits denote Unix signals.

## Verification performed here

- Python compilation, CLI help, Ruff lint and formatting: passed.
- Strict source-replacement rejection of missing/ambiguous matches: passed.
- Transformation against the fetched upstream stream source, compilation of the
  transformed module, optional forward/backward buffer signatures and unified
  diff generation: passed. This is not validation against the installed wheel.
- Local suite preflight: exit 2, explicitly BLOCKED because this Python has no
  Torch; `/dev/nvidia0` and `/runtime/venv/bin/python` are absent here. No GPU child
  was started. See `fix-20261101T064900-5482/FIX_RESULTS.md`.
- New GPU numeric errors, Graph passes and performance impact: NOT MEASURED.
- Adapter prewarm and minimal real training: NOT RUN. Those depend on successful
  GPU regression and review of remaining candidate/backward-tuner coverage.

No repair is claimed and no production patch is proposed as validated. Generated
`experimental-stream.diff` is a hypothesis-testing patch only. Inspect the new
script with `git diff --no-index /dev/null codex-diagnostics/lycoris-cuda-graph/fix_probe.py`.
Disabling Graph, Triton or cudaMallocAsync is not the proposed repair.

## GPU run 20261101T065131-387 — reviewed

The hypothesis is confirmed for the streaming forward launch: flat/down/up/base
all have Driver result 0; out alone has result 1 during capture. The minimal copy
case aborts with capture-time allocation and passes with external output, zero
absolute/relative error over five replays.

Preallocated fused FP16 and BF16 forward/backward passed: three sets, five replays,
input preservation and independent buffers. These are direct fixed-config tests,
not proof that every autotuning candidate is compatible.

| Dtype | Graph ms / three sets | Eager ms / three sets |
| --- | --- | --- |
| FP16 | 0.0686192 | 0.3412480 |
| BF16 | 0.0738144 | 0.3131392 |

This compares launch paths in this microbenchmark, not before/after training speed.
Original A aborts (-6/SIGABRT), B and C pass (0). Preallocated full autotuning still
aborts (-6): five CAPTURE_OK markers precede an uncaptured-free warning and
cuMemFreeAsync failure. No first exception or candidate identity was recorded, so
neither the remaining faulty branch nor cleanup itself is proven responsible.

The diagnostic script now logs CANDIDATE_BEGIN and CAPTURE_FIRST_EXCEPTION before
LyCORIS cleanup, so another isolated tuner run can identify that missing evidence.
No additional speculative allocator/kernel change was made. Graph/Triton/async
remain enabled; no production code was changed. Training remains blocked on full
autotuner compatibility. Graph-destruction warnings in the passing direct test
also remain a lifecycle issue to inspect before production use.

## GPU run 20261101T065549-703 — atomic branch identified

FP16 full forward autotuning passed, including the atomic candidate. BF16 passed
four streaming candidates and TORCH, then failed in ATOMIC. The recorded first
exception is bypass.py:337 -> launch(best, out) -> Triton pointer validation.
This variability means the first run's failure after five captures did not by
itself identify its dtype or candidate; this run supplies that missing evidence.

The test copy now lets atomic bypass.lora_bypass_fwd accept optional out, and the
streaming benchmark factory allocates a separate compute-dtype atomic output for
each atomic candidate/buffer-set closure before capture. Streaming and atomic
buffers remain distinct, including when output dtype differs from compute dtype.
The original candidate list and nested atomic tuning are retained. The atomic
module is copied into an independent namespace; installed module attributes and
package files remain unchanged. Both experimental source diffs are emitted.

Static compilation, Ruff, and forward/atomic source transformation checks passed.
This extension has NOT been GPU tested here. Rerun the same suite command to
validate it; no production repair or backward autotuner compatibility is claimed.

## Reviewed GPU results — 20261101T065848-1002

Forward autotuning repair PASSES in the tested environment and shape. This is not
full training compatibility certification.

- Original A still aborts; B and C pass. Reproduction controls remain effective.
- Preallocated copy passes, exact output over five replays.
- Preallocated fused FP16/BF16 forward/backward passes three independent sets and
  five replays with changed inputs and no input overwrites.
- Full forward autotuning exits 0 in FP16 and BF16, with 18 CAPTURE_OK records
  (nine per dtype, 39 buffer sets), including streaming, TORCH and ATOMIC.
  No CAPTURE_FIRST_EXCEPTION or rejected Graph fallback is present.
- Final autotuned FP16 output: max abs 0, max relative 0, relative Frobenius 0.
- Final autotuned BF16 output: max abs 0.015625, max relative 97.0, relative
  Frobenius 0.0011479532 (below the existing BF16 tolerance 0.04).
  Large pointwise relative error near zero does not imply that every element
  passed a strict pointwise tolerance; acceptance uses relative Frobenius.

| Direct fused dtype | Max absolute error across fwd/bwd | Max relative error | Max relative Frobenius |
| --- | --- | --- | --- |
| torch.float16 | 5.96046448e-08 | 0.0596046448 | 2.90544611e-09 |
| torch.bfloat16 | 0.25 | 757.857117 | 0.00366375223 |

Direct fused graph/eager timings for three sets: FP16 0.0687328/0.3281920 ms;
BF16 0.0740400/0.3386880 ms. This is not a pre/post patch performance comparison;
no training performance claim is supported.

Root cause evidence now covers two capture-time outputs passed to Triton:
stream._bypass_fwd_run and the atomic bypass.lora_bypass_fwd branch. Moving each
Triton destination outside capture preserves allocator, kernels, Graph timing
and candidate autotuning in this forward test.

Reviewable source diffs are in this run's experimental-stream.diff and
experimental-bypass.diff. They remain diagnostic patches: backward scratch
preallocation was tested with a fixed configuration, but is not threaded through
the backward tuner or its atomic GradPack branch. Full backward autotuning,
adapter prewarm, real training, actual model shapes and broader lifecycle coverage
remain untested. Do not apply as a complete production repair yet.

## Backward autotuning diagnostic added

Run the existing ComfyUI interpreter with `fix_probe.py backward-suite`. This
runs eight fresh processes: original async Graph, async without Graph, native
Graph, and async Graph with the validated forward patch, separately for FP16 and
BF16. The fourth case measures whether the forward repair suffices; no backward
benchmark-factory or atomic GradPack fix is silently assumed or introduced.

The real backward tuner retains all candidates, logs capture-time scratch
pointers and first exceptions, rejects Graph fallback, and compares all three
returned gradients to the TORCH branch. Five public calls change x/grad and check
input preservation; Graph benchmarking/replay is performed by the original tuner.
This is not a standalone five-replay correctness test of the chosen backward
Graph. The earlier fixed-config fused experiment provides that separate coverage.

Syntax, Ruff, CLI checks and a mocked orchestration self-check passed (eight
children, dtype/control separation, logs and exits). GPU execution remains pending
because this environment has no Torch or GPU. No production/package changes.

## Backward run 20261101T070550-1302 — reviewed and next experiment

Both FP16 and BF16: original async Graph and the forward-only patch abort with
SIGABRT (-6); async without Graph and native Graph pass. Streaming backward
capture-time Driver queries: flat/down/up/grad return 0, while gx/h/q each return
1. The first exception is the streaming Triton backward launch. This confirms
that forward output preallocation alone does not repair backward autotuning.

A new backward-fixed case threads per-candidate/per-set gx/h/q into streaming
backward benchmarking. It also preallocates atomic gx/GradPack through the atomic
API and zeros the reused pack on each call, so the reset is captured on every
Graph replay. The suite retains the original four controls and adds this fifth
case for each dtype (ten fresh processes total). Atomic correctness is separately
checked over five Graph replays with changed upstream gradients. Nested atomic
autotuning is retained and may select its own eager candidate; this extra check
covers the atomic API's selected path, not a forced nested kernel configuration.

Installed module/package files and production training remain unchanged. Source
patches now have mode/dtype-specific diff filenames. Compilation, Ruff, source
transformation checks, simulated independent factory buffers, atomic reset
injection and ten-case orchestration passed locally. The backward-fixed GPU
experiment is pending; rerun `fix_probe.py backward-suite` in the container.

## Reviewed backward GPU results — 20261101T071018-1659

Backward-fixed passes in FP16 and BF16 (exit 0). Original async Graph and the forward-only repair still abort; no-Graph async and native Graph controls pass.

| Dtype | Tuner captures | Max absolute error | Max pointwise relative error | Max relative Frobenius |
| --- | --- | --- | --- | --- |
| fp16 | 9 | 0.015625 | 0.0596046448 | 4.53608154e-05 |
| bf16 | 9 | 0.5 | 361.442566 | 0.00376440375 |

Each streaming/torch/atomic shortlist candidate is retained, with 28 input buffer
sets in this shape. No Graph fallback or first exception is recorded; all logged
streaming scratch pointer queries succeed. All three gradients were compared on
five changed-input calls; the separate atomic API Graph passed five replays with
changed upstream gradients. Pointwise relative errors near zero can be large;
acceptance uses existing relative-Frobenius thresholds, not pointwise allclose.

Together with forward run 20261101T065848-1002 this validates the scoped
forward/backward autotuning repair for the tested shapes/dtypes/device/software.
It does not establish full training compatibility. Atomic nested tuning can
choose its eager candidate; no claim is made that every nested candidate was
forced or that gradients are bitwise identical.

Reviewable combined forward/backward source diffs are this run's
experimental-stream-backward-fixed-fp16.diff and
experimental-bypass-backward-fixed-fp16.diff (BF16 copies are also retained).
No production/package modification was performed. Fixed-case backward performance
was not timed separately; adapter prewarm, model shapes, and minimal real training
remain NOT RUN. Passing logs retain uncaptured-free warnings during Graph cleanup;
they did not abort these processes but broader lifetime validation is still needed.

## Installable minimal patch — normal training test requested

Per the updated instruction, no dataset preparation, additional prewarm runner,
or real-training runner was added. The exact two-file GPU-tested diff is now
`scripts/install/lycoris-4.1.3-cuda-graph.patch`; the one-shot application script is
`scripts/install/patch_lycoris_cuda_graph.py`. It requires version 4.1.3 and the
original source hashes, stages and compiles both changes, retains original source
backups, atomically replaces files and rolls back an earlier replacement if the
second replacement fails. Repeated application is a no-op. It does not change
Torch/allocator settings, candidates, or ComfyUI runtime code.

Source-only checks passed: application, idempotency, rejection of modified source,
and application against the PyPI 4.1.3 files. The resulting files were compared
byte-for-byte with the successfully GPU-tested backward-fixed source copies and
matched. Ruff and git diff whitespace checks passed.

The installed container package has NOT been patched by this agent: the local
interpreter has no LyCORIS, /runtime/venv and GPU are absent, and Podman cannot
access /home/deng/.config. Run the application script with the container's existing
ComfyUI interpreter, then restart ComfyUI before normal training. No complete
training success, nested-candidate coverage expansion, or uncaptured-free safety
certification is claimed. Package reinstall/upgrade can replace these patched
files; version/source guards deliberately refuse unknown versions.
