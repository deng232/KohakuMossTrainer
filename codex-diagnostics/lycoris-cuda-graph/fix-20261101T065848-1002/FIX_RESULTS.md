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
| tuner-preallocated | cudaMallocAsync | 0 | [tuner-preallocated.log](tuner-preallocated.log) |

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
