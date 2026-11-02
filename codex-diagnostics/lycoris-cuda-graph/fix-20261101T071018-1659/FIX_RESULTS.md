# FIX_RESULTS

Experimental diagnostics only. Exit 0 requires explicit assertions; nonzero or
negative exits are failures (negative values are Unix signals). No package or
training-path files were modified. Each case uses a fresh process/cache.

| Experiment | Allocator | Exit | Evidence |
| --- | --- | --- | --- |
| backward-A-original-fp16 | cudaMallocAsync | -6 | [backward-A-original-fp16.log](backward-A-original-fp16.log) |
| backward-B-no-graph-fp16 | cudaMallocAsync | 0 | [backward-B-no-graph-fp16.log](backward-B-no-graph-fp16.log) |
| backward-C-native-fp16 | native | 0 | [backward-C-native-fp16.log](backward-C-native-fp16.log) |
| backward-forward-patch-fp16 | cudaMallocAsync | -6 | [backward-forward-patch-fp16.log](backward-forward-patch-fp16.log) |
| backward-fixed-fp16 | cudaMallocAsync | 0 | [backward-fixed-fp16.log](backward-fixed-fp16.log) |
| backward-A-original-bf16 | cudaMallocAsync | -6 | [backward-A-original-bf16.log](backward-A-original-bf16.log) |
| backward-B-no-graph-bf16 | cudaMallocAsync | 0 | [backward-B-no-graph-bf16.log](backward-B-no-graph-bf16.log) |
| backward-C-native-bf16 | native | 0 | [backward-C-native-bf16.log](backward-C-native-bf16.log) |
| backward-forward-patch-bf16 | cudaMallocAsync | -6 | [backward-forward-patch-bf16.log](backward-forward-patch-bf16.log) |
| backward-fixed-bf16 | cudaMallocAsync | 0 | [backward-fixed-bf16.log](backward-fixed-bf16.log) |

In backward-suite each dtype/control is a separate process. The forward-patch
case tests only the existing forward repair. The additional backward-fixed case
threads preallocated buffers into both backward benchmark branches, and separately
checks five Graph replays of the atomic API to detect accumulation without reset. Five calls with changed inputs validate
selected backward outputs; benchmark Graph replay occurs inside the real tuner.

Read POINTER records to identify the first failing argument in launch order.
Copy-inside versus copy-preallocated isolates allocation placement, without LoRA
math. A/B/C use the real tuner; Graph cases reject silent eager fallback.
The fused experiment checks FP16/BF16 forward and backward, three independent
buffer sets, repeated replay with changed inputs, input preservation, output
storage independence, numerical errors, and CUDA-event timing.

The tuner patch covers streaming and atomic forward. backward-fixed also threads
streaming scratch and atomic GradPack through backward benchmark factories. TORCH/ATOMIC candidates are kept;
any failure remains a failed test, never filtered out to manufacture success.
Backward-fixed is an experimental backward tuner repair; other backward controls
retain the original benchmark factory. See per-case logs for exact tested configurations and errors.
Maximum relative error uses abs(reference).clamp_min(1e-6); acceptance follows
LyCORIS relative Frobenius tolerance (FP16 1e-2, BF16 4e-2), not max-relative alone.

Generated *.diff files are experimental source-copy diffs, NOT validated fixes.
Performance numbers are local microbenchmarks; no training speedup is established.
Adapter prewarm and real training are NOT RUN: these diagnostics deliberately do
not enter the production training path. Run those only after reviewing GPU passes
and completing coverage of any remaining allocator-sensitive branches.

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
