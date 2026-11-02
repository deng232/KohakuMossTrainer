# Codex 测试任务：定位 LyCORIS CUDA Graph + cudaMallocAsync 的 Triton 指针异常

> 目标仓库：`https://github.com/deng232/KohakuMossTrainer`  
> 对照仓库：`https://github.com/KohakuBlueleaf/LyCORIS`  
> 优先运行环境：当前 NixOS + ComfyUI Python venv + NVIDIA RTX 3080 Ti（12 GB）。  
> 本文档是**测试任务**，不是“关闭 CUDA Graph”的正式修复请求。

## 0. 任务要求 / 不允许的捷径

你（Codex）应当**实际执行**测试、保存日志、确认首次异常位置，并提交证据与结论。不要仅阅读代码后推测结果。

1. **先做独立最小复现**：不用 MOSS 权重、不加载数据集、不启动完整训练。使用下方诊断脚本，按 A/B/C 测试同一 LyCORIS 调用。
2. **必须保留原始失败路径作为对照**：A 使用 `cudaMallocAsync + LyCORIS CUDA Graph timing`；B 只关闭 LyCORIS Graph timing；C 使用 `native + Graph timing`。
3. 不要仅用 `KERNELS={"lycoris":"torch"}`、`TUNE=off`、升级 PyTorch、关闭 Triton、关闭整个 ComfyUI 的 `cudaMallocAsync` 作为“修复成功”依据。这些可以作为**附加排除实验**，但不可替代 A/B/C。
4. 不要修改上游 KohakuMoss 核心代码、LyCORIS site-packages 或现有训练数据/检查点。仅允许在仓库临时目录创建诊断脚本与日志。如确实需要插桩，优先用独立探针和临时 Python 包装；任何源码修改必须独立保存 diff、可还原。
5. 若 `kohaku_comfy/compat.py` 或 `runtime.py` 中存在前面添加的 `lycoris_timing_compat` workaround，**不得让它污染基线**。本文件的 standalone 探针直接调用 LyCORIS，不经过 `kohaku_comfy.runtime`；如最后做真实训练复现，先确认 workaround 是否启用，并在报告中注明。
6. 每次 A/B/C 都必须是**全新 Python 进程**，打印 `torch.cuda.memory.get_allocator_backend()` 确认实际生效的 allocator。不得在 PyTorch CUDA 已初始化后尝试热切换 allocator。
7. 发生 `Fatal Python error: Aborted` / exit code 134 时，保留完整 stdout/stderr；禁止把整个 ComfyUI 容器当成复现脚本进程直接崩溃退出。
8. 当前用户最关注保留 **Triton fused kernels、CUDA Graph benchmark、cudaMallocAsync**，因此不要把绕过其中某项等同于根治。

## 1. 背景与待验证假设

此前真实训练日志：LyCORIS adapter prewarm 进入 Triton LoRA bypass autotuning 后，出现类似：

```text
ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
...
CUDA_ERROR_INVALID_VALUE from cuMemFreeAsync
CUDAMallocAsyncAllocator.cpp
Fatal Python error: Aborted
```

需要明确区分**先发生的 Triton pointer 检查异常**与**后续 allocator 释放时的 abort**。后者可能是 CUDA 上下文异常后的二次错误，不应先入为主视为根因。

已核对的真实源码路径：

- LyCORIS `lycoris/kernels/tuner/timing.py`: `time_candidates(..., use_graph=True)`；`_Batch.__init__` 先在独立 CUDA stream 预热，然后调用 `torch.cuda.graph(...)` 捕获候选 kernel，后续 `graph.replay()` 和 CUDA Event 计时。
- LyCORIS `lycoris/kernels/triton/lora/stream.py`: `_bypass_fwd_run()` 在 fused path 中 `torch.empty(...)` 创建 `out`，随后启动 Triton kernel。
- LyCORIS `lycoris/kernels/tuner/measure.py`: `measurer()` 通过模块级 `time_candidates` 调用计时器。
- KohakuMoss `src/kohakumoss/adapters/bind/autotuned.py`: LoRA adapter 的上层自动调优与 prewarm。
- Fork `kohaku_comfy/runtime.py`: Custom Node 在 ComfyUI 的 prompt worker **同进程**调用 `native.run(knobs)`；主进程 allocator 通常早已确定。

**假设 H1**：Graph capture 中分配的 Tensor 地址，在 Triton 的 CUDA Driver 指针属性校验时不可访问（需验证；绝不能凭 `(cpu tensor?)` 就认定 Tensor 位于 CPU）。  
**假设 H2**：CUDA stream 同步或捕获前后的内存生命周期导致地址不可用。  
**假设 H3**：只在 `cudaMallocAsync` 与 Graph 联用时触发，native allocator 不复现。  
**假设 H4**：独立复现正常，只有 ComfyUI / MOSS 的 offload、缓存或预热环境会触发。

以上都是假设，不是已经确认的修复结论。

## 2. 环境与版本记录（先执行）

在仓库根目录执行，创建 `.codex-diagnostics/lycoris-cuda-graph/`。不要清理模型、训练输出或用户现有缓存。

```bash
mkdir -p .codex-diagnostics/lycoris-cuda-graph
OUT="$PWD/.codex-diagnostics/lycoris-cuda-graph"
PY=/runtime/venv/bin/python
if [ ! -x "$PY" ]; then PY="$(command -v python)"; fi
printf 'PY=%s\nOUT=%s\n' "$PY" "$OUT"
{
  date -Is
  git remote -v
  git rev-parse HEAD
  git status --short
  nvidia-smi
  "$PY" -c 'import torch,triton,importlib.metadata as m; print("torch",torch.__version__,"torch cuda",torch.version.cuda,"triton",triton.__version__); print("lycoris",m.version("lycoris-lora")); print("CUDA available",torch.cuda.is_available()); print("gpu",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
} > "$OUT/environment.log" 2>&1
cat "$OUT/environment.log"
```

- 记录显卡、驱动、CUDA runtime、PyTorch、Triton、`lycoris-lora` 版本，以及 fork 的 commit SHA 和已有未提交修改。
- 如果找不到当前 venv，先定位真正运行 ComfyUI 的 Python 解释器。**不要默认 `python` 就是 ComfyUI 的环境。**
- 如果没 GPU、LyCORIS 依赖不匹配、或测试环境无法执行，**报告阻碍并停止**，不要通过未经授权的依赖升级改变现有环境。

## 3. 创建独立复现脚本

在仓库根目录创建 `lycoris_cuda_graph_probe.py`。若已有同名文件，请比较内容、保留现有文件并使用临时文件名。脚本不修改 ComfyUI/KohakuMoss 源码；它只在自身进程临时 monkey-patch LyCORIS 用于定位。

```python
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
    print("gpu:", torch.cuda.get_device_name(), "device:", torch.cuda.current_device(), flush=True)
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
        print(f"{label:>19s}: device={t.device}, ptr={hex(t.data_ptr())}, "
              f"driver_result={rc}, device_ptr={hex(got.value)}", flush=True)
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
        from lycoris.kernels.tuner import measure
        from lycoris.kernels.triton.lora import stream
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

        original_kernel = stream._bypass_fwd_run
        def traced_kernel(cfg, flat, up, down, scale, base, out_dtype):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                print("IN GRAPH CAPTURE; stream=", torch.cuda.current_stream().cuda_stream,
                      "cfg=", cfg, flush=True)
                for label, t in (("flat", flat), ("up", up), ("down", down), ("base", base)):
                    if t is not None:
                        print(" ", label, "device=", t.device, "ptr=", hex(t.data_ptr()),
                              "shape=", tuple(t.shape), flush=True)
            return original_kernel(cfg, flat, up, down, scale, base, out_dtype)
        stream._bypass_fwd_run = traced_kernel

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
```

这段探针包含两个子命令：

- `pointer`：对 graph capture 前、capture 中新分配、capture 后的 Tensor 执行 `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_POINTER)`；输出 `driver_result`，其中 0 = CUDA_SUCCESS，1 = CUDA_ERROR_INVALID_VALUE。
- `lycoris`：真实运行 `lycoris.kernels.triton.lora.stream.lora_bypass_fwd`；在 Graph capture 期间打印 operands 的 `device / data_ptr / shape`；`--no-graph-timing` 仅把 `measure.time_candidates(..., use_graph=False)` 传下去，保留 Triton 本身。

注意：**capture 期间 cuPointerGetAttribute 返回错误，只证明该时点的 Driver 指针属性查询失败，不能单独证明底层内存非法或 PyTorch bug。** 若查询动作本身导致 capture 失败，必须单独报告“探针干扰”，不能误判为同一个 LyCORIS 问题。

## 4. 第一轮：指针可见性实验（独立进程）

执行以下两个不同 allocator 的 `pointer` 实验：

```bash
mkdir -p .codex-diagnostics/lycoris-cuda-graph
OUT="$PWD/.codex-diagnostics/lycoris-cuda-graph"
PY=/runtime/venv/bin/python
if [ ! -x "$PY" ]; then PY="$(command -v python)"; fi

set +e
env -u PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF=backend:cudaMallocAsync \
  "$PY" -u lycoris_cuda_graph_probe.py pointer \
  > "$OUT/pointer-async.log" 2>&1
printf '%s\n' "$?" > "$OUT/pointer-async.exit"

env -u PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF=backend:native \
  "$PY" -u lycoris_cuda_graph_probe.py pointer \
  > "$OUT/pointer-native.log" 2>&1
printf '%s\n' "$?" > "$OUT/pointer-native.exit"
set -e
```

核对每个日志里打印的 `allocator:`，并保存所有 `driver_result`：

| 情况 | 解释方向 |
| --- | --- |
| async capture 中新 Tensor 的指针查询失败；native 成功 | 支持 allocator × Graph capture 的相互作用，需要更小的独立 Triton kernel 验证 |
| 两者 capture 中都失败 | 更像 CUDA Graph 内指针属性查询本身不适用，或探针干扰 |
| 两者都成功 | H1 未被当前最小指针实验支持；继续查 LyCORIS 的 stream、具体 kernel/shape |
| 进程 abort | 保留**最早**错误与 exit code；不要只记录最后一行 `cuMemFreeAsync` |

## 5. 第二轮：LyCORIS A/B/C 核心复现（必须执行）

每次都用新的 Python 子进程。`LYCORIS_KERNEL_CACHE_DIR` 由诊断脚本设置为临时目录，避免已有缓存令 autotuner 跳过测试。

```bash
OUT="$PWD/.codex-diagnostics/lycoris-cuda-graph"
PY=/runtime/venv/bin/python
if [ ! -x "$PY" ]; then PY="$(command -v python)"; fi
set +e

# A: cudaMallocAsync + Graph Timing ON — 基线
 env -u PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF=backend:cudaMallocAsync \
  "$PY" -u lycoris_cuda_graph_probe.py lycoris \
  > "$OUT/A-async-graph.log" 2>&1
printf '%s\n' "$?" > "$OUT/A-async-graph.exit"

# B: cudaMallocAsync + Graph Timing OFF — 只改 Graph 计时
 env -u PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF=backend:cudaMallocAsync \
  "$PY" -u lycoris_cuda_graph_probe.py lycoris --no-graph-timing \
  > "$OUT/B-async-no-graph.log" 2>&1
printf '%s\n' "$?" > "$OUT/B-async-no-graph.exit"

# C: native + Graph Timing ON — 只改 allocator
 env -u PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF=backend:native \
  "$PY" -u lycoris_cuda_graph_probe.py lycoris \
  > "$OUT/C-native-graph.log" 2>&1
printf '%s\n' "$?" > "$OUT/C-native-graph.exit"
set -e
```

**重要：不允许添加 `CUDA_LAUNCH_BLOCKING=1` 到第一轮 Graph capture 基线中。** 这个环境变量可能改变甚至破坏 Graph capture 语义；如需进一步定位，作为额外单独实验并注明。

结果解释：

| A | B | C | 优先下一步 |
| --- | --- | --- | --- |
| 失败 | 成功 | 成功 | 调查 async allocator × graph capture / 特定 pointer 校验 |
| 失败 | 成功 | 失败 | Graph capture 或 stream 使用本身的问题，不局限于 async |
| 失败 | 失败 | 成功 | allocator 相关问题，但 Graph timing 并非唯一触发条件 |
| 成功 | 成功 | 成功 | 最小复现没有触发；回到 fork 的训练集成、offload、具体 shape/dtype |
| 其他 | 其他 | 其他 | 保存原始报错逐项解释；不要强行套入上表 |

每条实验至少记录：实际 allocator，`Graph Timing ENABLED/DISABLED`，出现 `IN GRAPH CAPTURE` 与否，首次 traceback，最终 exit code，是否输出 `LyCORIS kernel + tuner completed`。

## 6. 若 A 失败：逐层定位，不要直接删掉 Graph

优先级：

1. 根据 `A-async-graph.log` 确认是 **Graph capture 内**还是 `_Batch` capture 前 side-stream warmup 出错；不要仅根据 `time_candidates` 出现在栈中就认定发生在 capture 内。
2. 在 `lycoris/kernels/triton/lora/stream.py::_bypass_fwd_run` 的独立探针 wrapper 中，比对 `flat, up, down, base, out` 的 `device`、`dtype`、`shape`、`stride`、`data_ptr`；尤其关注在 Graph 内由 `torch.empty` 新建的 `out`。
3. 如果外部输入指针都有效，且只在 Graph 内新建 `out` 时失败，做额外**受控对照**：capture 外预分配输出 vs capture 内分配，并比较同一 Triton kernel 的行为。不要仅凭改变内存地址就断言修复。
4. 对照 ComfyUI `comfy/model_prefetch.py` 的 capture stream、`wait_stream`、cuBLAS workspace 提前初始化、Graph 释放与 pool 的生命周期管理。只迁移有证据必要的机制，避免全盘照抄。
5. 如果怀疑 PyTorch/Triton 版本问题，使用**全新隔离 venv**验证升级组合；不要修改当前 ComfyUI venv，也不要未经验证移除 `torch>=2.13,<2.15` 的项目约束。
6. 如果所有 standalone 测试通过，才使用原训练配置进行**受控训练复现**，先确认 Custom Node 的 `lycoris_timing_compat` workaround 状态，再补充 offload/ring streaming、模型参数迁移、CUDA stream 事件等日志。

## 7. 如果发现根因，才提交修复提案

优先提出保留以下三者的方案：

- `cudaMallocAsync`
- LyCORIS CUDA Graph benchmark
- Triton fused LoRA kernel

提出的修复应提供：精确文件/函数/行号、机制说明、最小 diff 或 patch、A/B/C 重新测试以及功能回归结果。如果只能通过 `use_graph=False` 规避，明确标注 **workaround，不是根治**。

## 8. Codex 最终必须交付的报告

把结果写入 `.codex-diagnostics/lycoris-cuda-graph/RESULTS.md`，不要只回复“试过了/可以运行”。报告至少包含：

1. **环境版本**：GPU、Driver、Python、PyTorch、PyTorch CUDA、Triton、LyCORIS、实际 allocator、fork commit。
2. **实验表**：pointer-async/native 及 A/B/C，各自退出码、是否复现、最早错误位置。
3. **证据**：关键日志原文的文件名与行号（最多引用支持结论的相关行）；区分 observed / inferred / unverified。
4. **根因可信度**：明确是“已证实”“支持但未证实”或“尚不清楚”。
5. **修复建议**：说明能否保留 Graph、Triton、cudaMallocAsync；推荐的最小修改范围。
6. **尚未测试内容**：如果硬件、权限、依赖或模型文件缺失，明确指出，不编造成功结果。
7. **改动清单**：`git diff --stat`、`git status --short`，所有额外诊断文件/补丁的位置。

### 完成条件

- 最低完成：环境记录 + pointer-async/native + A/B/C + `RESULTS.md`。
- 若复现：定位到首次失败的具体调用，或清楚说明无法继续定位的证据缺口。
- 若未复现：写明本探针的 shape/dtype 与真实训练差别，并给出下一步的**最小**复现需求。不得声称原训练问题已经修复。

**最后向用户简要汇报：哪个组合失败、首个异常在哪里、下一步应该修改哪一个函数，以及有没有保持原有性能特性。**
