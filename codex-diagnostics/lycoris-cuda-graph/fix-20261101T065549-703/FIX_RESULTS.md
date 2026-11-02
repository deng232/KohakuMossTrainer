# FIX_RESULTS

Experimental diagnostics only. Exit 0 requires explicit assertions; nonzero or
negative exits are failures (negative values are Unix signals). No package or
training-path files were modified. Each case uses a fresh process/cache.

| Experiment | Allocator | Exit | Evidence |
| --- | --- | --- | --- |
| locate | cudaMallocAsync | -6 | [locate.log](locate.log) |
| copy-inside | cudaMallocAsync | -6 | [copy-inside.log](copy-inside.log) |
| copy-preallocated | cudaMallocAsync | 0 | [copy-preallocated.log](copy-preallocated.log) |
| A-original | cudaMallocAsync | -6 | [A-original.log](A-original.log) |
| B-no-graph | cudaMallocAsync | 0 | [B-no-graph.log](B-no-graph.log) |
| C-native | native | 0 | [C-native.log](C-native.log) |
| fused-preallocated | cudaMallocAsync | 0 | [fused-preallocated.log](fused-preallocated.log) |
| tuner-preallocated | cudaMallocAsync | -6 | [tuner-preallocated.log](tuner-preallocated.log) |

Read POINTER records to identify the first failing argument in launch order.
Copy-inside versus copy-preallocated isolates allocation placement, without LoRA
math. A/B/C use the real tuner; Graph cases reject silent eager fallback.
The fused experiment checks FP16/BF16 forward and backward, three independent
buffer sets, repeated replay with changed inputs, input preservation, output
storage independence, numerical errors, and CUDA-event timing.

The tuner patch covers streaming forward only. TORCH/ATOMIC candidates are kept;
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
