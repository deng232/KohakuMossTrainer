#!/usr/bin/env python3
"""Reproduce/diagnose LyCORIS Triton vs PyTorch cudaMallocAsync graph capture.

Run in fresh processes with PYTORCH_ALLOC_CONF=backend:{native,cudaMallocAsync}.
The LyCORIS case intentionally uses a private disposable tuning cache.
"""

import argparse
import ctypes
import ctypes.util
import importlib.metadata
import os
import sys
import tempfile
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("pointer", "lycoris"))
parser.add_argument("--no-graph-timing", action="store_true")
args = parser.parse_args()

import torch


def status():
    print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda, flush=True)
    try:
        print("triton:", importlib.metadata.version("triton"), flush=True)
    except importlib.metadata.PackageNotFoundError:
        print("triton: not installed", flush=True)
    try:
        print("lycoris-lora:", importlib.metadata.version("lycoris-lora"), flush=True)
    except importlib.metadata.PackageNotFoundError:
        pass
    print("allocator:", torch.cuda.memory.get_allocator_backend(), flush=True)
    print(
        "gpu:",
        torch.cuda.get_device_name(),
        "device:",
        torch.cuda.current_device(),
        flush=True,
    )
    print("CUDA_LAUNCH_BLOCKING:", os.getenv("CUDA_LAUNCH_BLOCKING"), flush=True)


def pointer_probe():
    """Tests whether driver considers a graph-captured fresh allocation addressable."""
    name = ctypes.util.find_library("cuda") or "libcuda.so.1"
    driver = ctypes.CDLL(name)
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    assert driver.cuInit(0) == 0
    # CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3 from cuda.h
    get = driver.cuPointerGetAttribute
    get.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    get.restype = ctypes.c_int

    def check(label, t):
        got = ctypes.c_uint64()
        rc = get(ctypes.byref(got), 3, t.data_ptr())
        print(
            f"{label:>19s}: device={t.device}, ptr={hex(t.data_ptr())}, "
            f"driver_result={rc}, device_ptr={hex(got.value)}",
            flush=True,
        )
        return rc

    x = torch.empty((1024,), device="cuda")
    check("outside x", x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        check("captured external x", x)
        y = torch.empty_like(x)
        check("captured new y", y)
        y.copy_(x)
    check("after capture y", y)
    graph.replay()
    torch.cuda.synchronize()
    check("after replay y", y)
    print("graph capture/replay completed", flush=True)


def lycoris_probe():
    # Force a new tuning decision regardless of the existing user cache.
    # Keep the temporary directory for the full call because the tuner may write to it.
    with tempfile.TemporaryDirectory(prefix="lycoris-probe-") as tempdir:
        os.environ["LYCORIS_KERNEL_CACHE_DIR"] = tempdir
        from lycoris.kernels import set_tuning
        from lycoris.kernels.triton.lora import stream
        from lycoris.kernels.tuner import measure, timing

        set_tuning(True)

        if args.no_graph_timing:
            original_timing = measure.time_candidates

            def without_graph(*a, **kw):
                kw["use_graph"] = False
                return original_timing(*a, **kw)

            measure.time_candidates = without_graph
            print("LyCORIS tuner: CUDA Graph timing DISABLED", flush=True)
        else:
            print("LyCORIS tuner: CUDA Graph timing ENABLED", flush=True)

        # Report the exception LyCORIS normally swallows, before fatal cleanup.
        original_init = timing._Batch.__init__

        def traced_init(self, *a, **kw):
            previous = sys.gettrace()

            def trace(frame, event, value):
                if frame.f_code is not original_init.__code__:
                    return None
                if event == "exception":
                    print("[LYCORIS FIRST EXCEPTION]", file=sys.stderr, flush=True)
                    traceback.print_exception(*value, file=sys.stderr)
                    sys.stderr.flush()
                return trace

            sys.settrace(trace)
            try:
                return original_init(self, *a, **kw)
            finally:
                sys.settrace(previous)

        timing._Batch.__init__ = traced_init

        original_kernel = stream._bypass_fwd_run

        def traced_kernel(cfg, flat, up, down, scale, base, out_dtype):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                print(
                    "IN GRAPH CAPTURE; stream=",
                    torch.cuda.current_stream().cuda_stream,
                    "cfg=",
                    cfg,
                    flush=True,
                )
                for label, t in (
                    ("flat", flat),
                    ("up", up),
                    ("down", down),
                    ("base", base),
                ):
                    if t is not None:
                        print(
                            " ",
                            label,
                            "device=",
                            t.device,
                            "ptr=",
                            hex(t.data_ptr()),
                            "shape=",
                            tuple(t.shape),
                            flush=True,
                        )
            return original_kernel(cfg, flat, up, down, scale, base, out_dtype)

        stream._bypass_fwd_run = traced_kernel

        torch.manual_seed(0)
        x = torch.randn((512, 256), dtype=torch.float16, device="cuda")
        up = torch.randn((192, 8), dtype=torch.float16, device="cuda")
        down = torch.randn((8, 256), dtype=torch.float16, device="cuda")
        base = torch.randn((512, 192), dtype=torch.float16, device="cuda")
        print("testing lora_bypass_fwd; inputs all on", x.device, flush=True)
        y = stream.lora_bypass_fwd(x, up, down, gamma=0.5, base=base)
        torch.cuda.synchronize()
        assert y.shape == (512, 192)
        print("LyCORIS kernel + tuner completed; shape=", tuple(y.shape), flush=True)


if __name__ == "__main__":
    try:
        status()
        if args.mode == "pointer":
            pointer_probe()
        else:
            lycoris_probe()
    except BaseException:
        traceback.print_exc()
        raise
