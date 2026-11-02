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
| backward-A-original-bf16 | cudaMallocAsync | -6 | [backward-A-original-bf16.log](backward-A-original-bf16.log) |
| backward-B-no-graph-bf16 | cudaMallocAsync | 0 | [backward-B-no-graph-bf16.log](backward-B-no-graph-bf16.log) |
| backward-C-native-bf16 | native | 0 | [backward-C-native-bf16.log](backward-C-native-bf16.log) |
| backward-forward-patch-bf16 | cudaMallocAsync | -6 | [backward-forward-patch-bf16.log](backward-forward-patch-bf16.log) |

In backward-suite each dtype/control is a separate process. The forward-patch
case tests the existing forward repair; it deliberately does not thread scratch
buffers into the backward benchmark factory. A failure here is evidence for the
next repair, not a claimed backward fix. Five calls with changed inputs validate
selected backward outputs; benchmark Graph replay occurs inside the real tuner.

Read POINTER records to identify the first failing argument in launch order.
Copy-inside versus copy-preallocated isolates allocation placement, without LoRA
math. A/B/C use the real tuner; Graph cases reject silent eager fallback.
The fused experiment checks FP16/BF16 forward and backward, three independent
buffer sets, repeated replay with changed inputs, input preservation, output
storage independence, numerical errors, and CUDA-event timing.

The tuner patch covers streaming and atomic forward. TORCH/ATOMIC candidates are kept;
any failure remains a failed test, never filtered out to manufacture success.
Backward fused scratch preallocation is an independent experiment, not a complete
backward tuner patch. See per-case logs for exact tested configurations and errors.
Maximum relative error uses abs(reference).clamp_min(1e-6); acceptance follows
LyCORIS relative Frobenius tolerance (FP16 1e-2, BF16 4e-2), not max-relative alone.

Generated *.diff files are experimental source-copy diffs, NOT validated fixes.
Performance numbers are local microbenchmarks; no training speedup is established.
Adapter prewarm and real training are NOT RUN: these diagnostics deliberately do
not enter the production training path. Run those only after reviewing GPU passes
and completing coverage of any remaining allocator-sensitive branches.
