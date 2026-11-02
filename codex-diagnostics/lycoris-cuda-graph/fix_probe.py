#!/usr/bin/env python3
"""Disposable GPU experiments; installed LyCORIS files are never written."""

import argparse
import ctypes
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Unsupported LyCORIS source: expected once: {old!r}")
    return source.replace(old, new, 1)


def run_suite(backward=False):
    root = Path(__file__).resolve().parent
    dest = root / (
        time.strftime("fix-%Y%m%dT%H%M%S", time.gmtime()) + f"-{os.getpid()}"
    )
    dest.mkdir()
    cases = [
        ("locate", "cudaMallocAsync", "locate", False),
        ("copy-inside", "cudaMallocAsync", "copy", False),
        ("copy-preallocated", "cudaMallocAsync", "copy", True),
        ("A-original", "cudaMallocAsync", "tuner", False),
        ("B-no-graph", "cudaMallocAsync", "tuner-eager", False),
        ("C-native", "native", "tuner", False),
        ("fused-preallocated", "cudaMallocAsync", "fused", True),
        ("tuner-preallocated", "cudaMallocAsync", "tuner", True),
    ]
    cases = [
        (name, backend, mode, patch, "both") for name, backend, mode, patch in cases
    ]
    if backward:
        cases = [
            (f"{name}-{dtype}", backend, mode, patch, dtype)
            for dtype in ("fp16", "bf16")
            for name, backend, mode, patch in (
                ("backward-A-original", "cudaMallocAsync", "backward", False),
                ("backward-B-no-graph", "cudaMallocAsync", "backward-eager", False),
                ("backward-C-native", "native", "backward", False),
                ("backward-forward-patch", "cudaMallocAsync", "backward", True),
                ("backward-fixed", "cudaMallocAsync", "backward-fixed", True),
            )
        ]
    rows = []
    if importlib.util.find_spec("torch") is None:
        (dest / "FIX_RESULTS.md").write_text(
            "# FIX_RESULTS\n\nBLOCKED: this Python has no Torch; no GPU child was started.\n"
            f"Interpreter: `{sys.executable}`. Run suite with the existing ComfyUI Python.\n"
            "No packages or production files were modified; numerical errors and performance are unmeasured.\n"
        )
        print(dest / "FIX_RESULTS.md", flush=True)
        return 2
    # Each failure is isolated, including C++ aborts; preserve every log and exit.
    for name, backend, mode, patch, dtype in cases:
        env = dict(os.environ)
        for key in ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_LAUNCH_BLOCKING"):
            env.pop(key, None)
        env["PYTORCH_ALLOC_CONF"] = f"backend:{backend}"
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            mode,
            "--artifacts",
            str(dest),
            "--dtype",
            dtype,
        ]
        if patch:
            command.append("--preallocate")
        if patch and mode != "copy" and not mode.startswith("backward"):
            locate = (dest / "locate.log").read_text()
            records = [
                json.loads(line.split("POINTER ", 1)[1])
                for line in locate.splitlines()
                if line.startswith("POINTER ")
            ]
            isolated_out = (
                len(records) >= 5
                and all(r["driver_result"] == 0 for r in records[:4])
                and records[4]["name"] == "out"
                and records[4]["driver_result"] == 1
            )
            copy_pass = (dest / "copy-preallocated.exit").read_text().strip() == "0"
            if not isolated_out or not copy_pass:
                (dest / f"{name}.log").write_text(
                    "SKIPPED: output-pointer hypothesis not confirmed by prerequisites.\n"
                )
                (dest / f"{name}.exit").write_text("125\n")
                rows.append(f"| {name} | {backend} | 125 | [{name}.log]({name}.log) |")
                continue
        print("Running", name, flush=True)
        with (dest / f"{name}.log").open("w") as log:
            try:
                result = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                log.write("TIMEOUT after 900 seconds\n")
        (dest / f"{name}.exit").write_text(f"{code}\n")
        rows.append(f"| {name} | {backend} | {code} | [{name}.log]({name}.log) |")
    report = (
        """# FIX_RESULTS

Experimental diagnostics only. Exit 0 requires explicit assertions; nonzero or
negative exits are failures (negative values are Unix signals). No package or
training-path files were modified. Each case uses a fresh process/cache.

| Experiment | Allocator | Exit | Evidence |
| --- | --- | --- | --- |
"""
        + "\n".join(rows)
        + """

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
"""
    )
    (dest / "FIX_RESULTS.md").write_text(report)
    print(dest / "FIX_RESULTS.md", flush=True)
    return int(any("| 0 |" not in row for row in rows))


def child(args):
    import importlib.metadata as metadata

    import torch
    import triton
    import triton.language as tl

    print(
        "ENV",
        sys.version,
        torch.__version__,
        torch.version.cuda,
        triton.__version__,
        metadata.version("lycoris-lora"),
        flush=True,
    )
    print("ALLOCATOR", torch.cuda.memory.get_allocator_backend(), flush=True)
    print("GPU", torch.cuda.get_device_name(), flush=True)
    torch.manual_seed(42)
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    assert driver.cuInit(0) == 0
    query = driver.cuPointerGetAttribute
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    query.restype = ctypes.c_int

    def pointers(labels, tensors):
        for index, (label, tensor) in enumerate(zip(labels, tensors)):
            if not isinstance(tensor, torch.Tensor):
                continue
            ptr = ctypes.c_uint64()
            rc = query(ctypes.byref(ptr), 3, tensor.data_ptr())
            print(
                "POINTER",
                json.dumps(
                    dict(
                        index=index,
                        name=label,
                        device=str(tensor.device),
                        address=hex(tensor.data_ptr()),
                        driver_result=rc,
                        device_pointer=hex(ptr.value),
                        capturing=torch.cuda.is_current_stream_capturing(),
                        stream=torch.cuda.current_stream().cuda_stream,
                    )
                ),
                flush=True,
            )

    def check(got, ref, label):
        got, ref = got.float(), ref.float()
        delta = got - ref
        relative = delta.norm() / ref.norm().clamp_min(1e-12)
        print(
            "ERROR",
            label,
            json.dumps(
                dict(
                    max_abs=delta.abs().max().item(),
                    max_rel=(delta.abs() / ref.abs().clamp_min(1e-6)).max().item(),
                    relative_frobenius=relative.item(),
                )
            ),
            flush=True,
        )
        assert torch.isfinite(got).all(), label
        assert relative <= tolerance, label

    def milliseconds(call):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 20

    @triton.jit
    def copy_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Y + offsets, tl.load(X + offsets, offsets < N, other=0), offsets < N)

    if args.mode == "copy":
        x = torch.arange(1024, device="cuda", dtype=torch.float32)
        outside = torch.empty_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            copy_kernel[(4,)](x, outside, x.numel(), BLOCK=256)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=torch.cuda.graph_pool_handle(),
            stream=stream,
            capture_error_mode="thread_local",
        ):
            out = outside if args.preallocate else torch.empty_like(x)
            pointers(("x", "out"), (x, out))
            copy_kernel[(4,)](x, out, x.numel(), BLOCK=256)
        for iteration in range(5):
            x.add_(1)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, x)
        tolerance = 0.0
        check(out, x, "copy")
        print("PASS copy capture + 5 replays", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix="lycoris-fix-") as cache:
        os.environ["LYCORIS_KERNEL_CACHE_DIR"] = cache
        from lycoris.kernels import set_tuning
        from lycoris.kernels.triton.lora import stream
        from lycoris.kernels.tuner import measure, timing

        source = inspect.getsource(stream)
        print(
            "SOURCE",
            stream.__file__,
            hashlib.sha256(source.encode()).hexdigest(),
            flush=True,
        )
        namespace = dict(vars(stream))
        patched = source
        if args.preallocate:
            patched = replace_once(
                patched,
                "def _bypass_fwd_run(cfg, flat, u, d, scale, base2, out_dtype):",
                "def _bypass_fwd_run(cfg, flat, u, d, scale, base2, out_dtype, out=None, atomic_out=None):",
            )
            patched = replace_once(
                patched,
                "        y = bypass.lora_bypass_fwd(flat, u, d, scale)",
                "        y = bypass.lora_bypass_fwd(flat, u, d, scale, out=atomic_out)",
            )
            patched = replace_once(
                patched,
                "    out = torch.empty(t, o, device=flat.device, dtype=out_dtype)",
                "    if out is None:\n        out = torch.empty(t, o, device=flat.device, dtype=out_dtype)",
            )
            patched = replace_once(
                patched,
                "def _bypass_bwd_run(cfg, g2, flat, u, d, scale):",
                "def _bypass_bwd_run(cfg, g2, flat, u, d, scale, buffers=None):",
            )
            patched = replace_once(
                patched,
                "    gx = torch.empty_like(flat)\n    h = torch.empty(t, r, device=flat.device, dtype=flat.dtype)\n    q = torch.empty_like(h)",
                "    if buffers is None:\n        gx = torch.empty_like(flat)\n        h = torch.empty(t, r, device=flat.device, dtype=flat.dtype)\n        q = torch.empty_like(h)\n    else:\n        gx, h, q = buffers",
            )
            patched = replace_once(
                patched,
                "            lambda c, s: lambda: _bypass_fwd_run(\n                c, s[0], up, down, gamma, s[1], out_dtype\n            ),",
                "            make_call,",
            )
            marker = "    cfg = select(\n        dev,\n        key,\n        lambda: _plan_bypass_fwd"
            replacement = """    def make_call(c, s):
        # One retained output per candidate and buffer set, allocated before capture.
        out = torch.empty(t, o, device=x.device, dtype=out_dtype)
        atomic_out = torch.empty(t, o, device=x.device, dtype=x.dtype) if c == ATOMIC else None
        return lambda: _bypass_fwd_run(c, s[0], up, down, gamma, s[1], out_dtype, out, atomic_out)

    cfg = select(
        dev,
        key,
        lambda: _plan_bypass_fwd"""
            patched = replace_once(patched, marker, replacement)
            if args.mode == "backward-fixed":
                patched = replace_once(
                    patched,
                    "        return bypass.lora_bypass_bwd(flat, u, d, g2, scale)",
                    "        return bypass.lora_bypass_bwd(flat, u, d, g2, scale, buffers=buffers)",
                )
                patched = replace_once(
                    patched,
                    "            lambda c, s: lambda: _bypass_bwd_run(c, s[0], s[1], up, down, gamma),",
                    "            make_backward_call,",
                )
                marker_bwd = "    cfg = select(\n        dev,\n        key,\n        lambda: _plan_bypass_bwd"
                replacement_bwd = """    def make_backward_call(c, s):
        buffers = None
        if c == ATOMIC:
            buffers = (torch.empty_like(s[1]), bypass.GradPack(x.device, (o, r), (r, i)))
        elif c != TORCH:
            buffers = (torch.empty_like(s[1]),
                       torch.empty(t, r, device=x.device, dtype=x.dtype),
                       torch.empty(t, r, device=x.device, dtype=x.dtype))
        return lambda: _bypass_bwd_run(c, s[0], s[1], up, down, gamma, buffers)

    cfg = select(
        dev,
        key,
        lambda: _plan_bypass_bwd"""
                patched = replace_once(patched, marker_bwd, replacement_bwd)
            args.artifacts.mkdir(parents=True, exist_ok=True)
            (
                args.artifacts / f"experimental-stream-{args.mode}-{args.dtype}.diff"
            ).write_text(
                "".join(
                    difflib.unified_diff(
                        source.splitlines(True),
                        patched.splitlines(True),
                        fromfile="a/lycoris/kernels/triton/lora/stream.py",
                        tofile="b/lycoris/kernels/triton/lora/stream.py",
                    )
                )
            )
        args.artifacts.mkdir(parents=True, exist_ok=True)
        copy_path = (
            args.artifacts / f"stream_test_copy_{args.mode}_{args.preallocate}.py"
        )
        copy_path.write_text(patched)
        # Triton JIT uses inspect.getsource; the diagnostic copy must exist on disk.
        exec(compile(patched, str(copy_path), "exec"), namespace)
        if args.preallocate:
            original_bypass = stream.bypass
            atomic_source = inspect.getsource(original_bypass)
            atomic_patched = replace_once(
                atomic_source,
                "def lora_bypass_fwd(x, up, down, gamma: float = 1.0) -> torch.Tensor:",
                "def lora_bypass_fwd(x, up, down, gamma: float = 1.0, out=None) -> torch.Tensor:",
            )
            atomic_patched = replace_once(
                atomic_patched,
                "    out = torch.empty(t, o, device=x.device, dtype=x.dtype)",
                "    if out is None:\n        out = torch.empty(t, o, device=x.device, dtype=x.dtype)",
            )
            if args.mode == "backward-fixed":
                atomic_patched = replace_once(
                    atomic_patched,
                    "def lora_bypass_bwd(x, up, down, grad, gamma: float = 1.0):",
                    "def lora_bypass_bwd(x, up, down, grad, gamma: float = 1.0, buffers=None):",
                )
                atomic_patched = replace_once(
                    atomic_patched,
                    "    gx = torch.empty_like(x)\n    # One fp32 allocation for both atomic targets: one zero-fill, one cast.\n    pack = GradPack(x.device, (o, r), (r, i))",
                    "    if buffers is None:\n        gx = torch.empty_like(x)\n        pack = GradPack(x.device, (o, r), (r, i))\n    else:\n        gx, pack = buffers\n        # Reused atomic destinations must start at zero on every replay.\n        pack.flat.zero_()",
                )
            atomic_path = args.artifacts / f"bypass_test_copy_{args.mode}.py"
            atomic_path.write_text(atomic_patched)
            atomic_copy = types.ModuleType(original_bypass.__name__)
            vars(atomic_copy).update(vars(original_bypass))
            exec(compile(atomic_patched, str(atomic_path), "exec"), vars(atomic_copy))
            namespace["bypass"] = atomic_copy
            (
                args.artifacts / f"experimental-bypass-{args.mode}-{args.dtype}.diff"
            ).write_text(
                "".join(
                    difflib.unified_diff(
                        atomic_source.splitlines(True),
                        atomic_patched.splitlines(True),
                        fromfile="a/lycoris/kernels/triton/lora/bypass.py",
                        tofile="b/lycoris/kernels/triton/lora/bypass.py",
                    )
                )
            )

        class LaunchProbe:
            def __init__(self, kernel, labels):
                self.kernel, self.labels = kernel, labels

            def __getitem__(self, grid):
                launch = self.kernel[grid]

                def call(*values, **kwargs):
                    if torch.cuda.is_current_stream_capturing():
                        pointers(self.labels, values)
                    return launch(*values, **kwargs)

                return call

        if args.mode == "locate":
            namespace["_bypass_fwd_kernel"] = LaunchProbe(
                namespace["_bypass_fwd_kernel"],
                ("flat", "down", "up", "base2_or_out", "out"),
            )
        if args.mode.startswith("backward"):
            namespace["_bypass_bwd_kernel"] = LaunchProbe(
                namespace["_bypass_bwd_kernel"],
                ("flat", "down", "up", "grad", "gx", "h", "q"),
            )
        original_init = timing._Batch.__init__
        captures = []
        fallback_failures = []

        def strict_init(batch, calls, dev, pool):
            # Print before capture: cleanup after a swallowed exception can abort C++.
            call = calls[0]
            closed = dict(
                zip(
                    getattr(call.__code__, "co_freevars", ()),
                    (cell.cell_contents for cell in (call.__closure__ or ())),
                )
            )
            print(
                "CANDIDATE_BEGIN",
                repr(closed.get("c", "unknown")),
                "sets=",
                len(calls),
                "graph=",
                pool is not None,
                flush=True,
            )
            previous_trace = sys.gettrace()

            def trace(frame, event, value):
                if frame.f_code is not original_init.__code__:
                    return None
                if event == "exception":
                    print("CAPTURE_FIRST_EXCEPTION", file=sys.stderr, flush=True)
                    traceback.print_exception(*value, file=sys.stderr)
                    sys.stderr.flush()
                return trace

            sys.settrace(trace)
            try:
                original_init(batch, calls, dev, pool)
            finally:
                sys.settrace(previous_trace)
            if pool is not None:
                if batch.graph is None:
                    fallback_failures.append(True)
                    raise AssertionError("SILENT GRAPH FALLBACK REJECTED")
                captures.append(True)
                print("CAPTURE_OK", len(captures), flush=True)

        timing._Batch.__init__ = strict_init
        if args.mode.endswith("eager"):
            original_time = measure.time_candidates

            def eager(*a, **kw):
                kw["use_graph"] = False
                return original_time(*a, **kw)

            measure.time_candidates = eager
        set_tuning(True)
        cfg = dict(
            BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, SPLIT_O=2, num_warps=2, num_stages=4
        )
        dtypes = {
            "both": (torch.float16, torch.bfloat16),
            "fp16": (torch.float16,),
            "bf16": (torch.bfloat16,),
        }
        for dtype in dtypes[args.dtype]:
            tolerance = 1e-2 if dtype == torch.float16 else 4e-2
            up = torch.randn(192, 8, device="cuda", dtype=dtype) * 0.1
            down = torch.randn(8, 256, device="cuda", dtype=dtype) * 0.1
            sets = [
                (
                    torch.randn(512, 256, device="cuda", dtype=dtype),
                    torch.randn(512, 192, device="cuda", dtype=dtype),
                    torch.randn(512, 192, device="cuda", dtype=dtype),
                )
                for _ in range(3)
            ]
            x, base, grad = sets[0]
            if args.mode.startswith("backward"):
                before = len(captures)
                saved_up, saved_down = up.clone(), down.clone()
                for iteration in range(5):
                    x.add_(0.01)
                    grad.mul_(0.99)
                    saved_x, saved_grad = x.clone(), grad.clone()
                    got = namespace["lora_bypass_bwd"](x, up, down, grad, gamma=0.5)
                    refs = namespace["_bypass_bwd_run"](
                        namespace["TORCH"], grad, x, up, down, 0.5
                    )
                    assert len(got) == 3
                    for index, (g, ref) in enumerate(zip(got, refs)):
                        check(
                            g,
                            ref,
                            f"backward {dtype} replay-call={iteration} gradient={index}",
                        )
                    for g, operand in (
                        (x, saved_x),
                        (grad, saved_grad),
                        (up, saved_up),
                        (down, saved_down),
                    ):
                        assert torch.equal(g, operand), "Backward input overwritten"
                assert not fallback_failures, (
                    "Backward tuner caught a rejected Graph fallback"
                )
                if not args.mode.endswith("eager"):
                    assert len(captures) > before, (
                        "Backward autotuning captured no Graph"
                    )
                print("PASS backward autotuning + 5 calls", dtype, flush=True)
                if args.mode == "backward-fixed":
                    atomic = namespace["bypass"]
                    atomic_buffers = (
                        torch.empty_like(x),
                        atomic.GradPack(x.device, (192, 8), (8, 256)),
                    )
                    # Warm nested tuning before capture; retain outputs for every replay.
                    atomic.lora_bypass_bwd(
                        x, up, down, grad, 0.5, buffers=atomic_buffers
                    )
                    warm = timing._warm_stream()
                    warm.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warm):
                        atomic.lora_bypass_bwd(
                            x, up, down, grad, 0.5, buffers=atomic_buffers
                        )
                    torch.cuda.current_stream().wait_stream(warm)
                    atomic_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        atomic_graph,
                        pool=torch.cuda.graph_pool_handle(),
                        stream=warm,
                        capture_error_mode="thread_local",
                    ):
                        atomic_result = atomic.lora_bypass_bwd(
                            x, up, down, grad, 0.5, buffers=atomic_buffers
                        )
                    for replay in range(5):
                        grad.mul_(0.99)
                        atomic_graph.replay()
                        torch.cuda.synchronize()
                        refs = namespace["_bypass_bwd_run"](
                            namespace["TORCH"], grad, x, up, down, 0.5
                        )
                        for index, (g, ref) in enumerate(zip(atomic_result, refs)):
                            check(
                                g,
                                ref,
                                f"atomic {dtype} replay={replay} gradient={index}",
                            )
                    del atomic_result, atomic_graph
                    torch.cuda.synchronize()
                    print("PASS atomic backward Graph + 5 replays", dtype, flush=True)
                continue
            if args.mode.startswith("tuner") or args.mode == "locate":
                before = len(captures)
                y = namespace["lora_bypass_fwd"](x, up, down, gamma=0.5, base=base)
                check(
                    y,
                    namespace["_torch_bypass"](x, up, down, 0.5, base, dtype),
                    str(dtype),
                )
                assert not fallback_failures, "Tuner caught a rejected Graph fallback"
                if args.mode != "tuner-eager":
                    assert len(captures) > before, "No Graph captured during autotuning"
                print("PASS autotuning", dtype, flush=True)
                continue

            calls, outputs, scratch = [], [], []
            for x, base, grad in sets:
                out = torch.empty_like(base)
                buffers = (
                    torch.empty_like(x),
                    torch.empty(512, 8, device="cuda", dtype=dtype),
                    torch.empty(512, 8, device="cuda", dtype=dtype),
                )
                outputs.append(out)
                scratch.extend(buffers)

                def call(x=x, base=base, grad=grad, out=out, buffers=buffers):
                    if args.preallocate:
                        y = namespace["_bypass_fwd_run"](
                            cfg, x, up, down, 0.5, base, dtype, out
                        )
                        gradients = namespace["_bypass_bwd_run"](
                            cfg, grad, x, up, down, 0.5, buffers
                        )
                    else:
                        y = namespace["_bypass_fwd_run"](
                            cfg, x, up, down, 0.5, base, dtype
                        )
                        gradients = ()
                    return y, gradients

                calls.append(call)
            all_buffers = outputs + scratch
            assert len({t.data_ptr() for t in all_buffers}) == len(all_buffers)
            warm = timing._warm_stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                for call in calls:
                    call()
            torch.cuda.current_stream().wait_stream(warm)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=torch.cuda.graph_pool_handle(),
                stream=warm,
                capture_error_mode="thread_local",
            ):
                results = [call() for call in calls]
            print("CAPTURE_OK direct fused", dtype, flush=True)
            up_before, down_before = up.clone(), down.clone()
            for iteration in range(5):
                for x, base, grad in sets:
                    x.add_(0.01)
                    base.add_(0.02)
                    grad.mul_(0.99)
                saved = [(x.clone(), b.clone(), g.clone()) for x, b, g in sets]
                graph.replay()
                torch.cuda.synchronize()
                for j, ((x, base, grad), (y, gradients)) in enumerate(
                    zip(sets, results)
                ):
                    label = f"{dtype} replay={iteration} set={j}"
                    check(
                        y,
                        namespace["_torch_bypass"](x, up, down, 0.5, base, dtype),
                        label,
                    )
                    if args.preallocate:
                        refs = namespace["_bypass_bwd_run"](
                            namespace["TORCH"], grad, x, up, down, 0.5
                        )
                        for k, (got, ref) in enumerate(zip(gradients, refs)):
                            check(got, ref, f"{label} gradient={k}")
                    for actual, expected in zip((x, base, grad), saved[j]):
                        assert torch.equal(actual, expected), "Input overwritten"
                assert torch.equal(up, up_before) and torch.equal(down, down_before)
            print(
                "TIMING",
                dtype,
                "graph_ms_3sets=",
                milliseconds(graph.replay),
                "eager_ms_3sets=",
                milliseconds(lambda: [call() for call in calls]),
                flush=True,
            )
            del results, graph
            torch.cuda.synchronize()
            print("PASS fused forward/backward + 5 replays", dtype, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "suite",
            "backward-suite",
            "locate",
            "copy",
            "fused",
            "tuner",
            "tuner-eager",
            "backward",
            "backward-eager",
            "backward-fixed",
        ),
    )
    parser.add_argument("--dtype", choices=("both", "fp16", "bf16"), default="both")
    parser.add_argument("--preallocate", action="store_true")
    parser.add_argument(
        "--artifacts", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    if args.mode in ("suite", "backward-suite"):
        return run_suite(backward=args.mode == "backward-suite")
    try:
        child(args)
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
