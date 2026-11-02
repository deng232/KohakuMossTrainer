# LyCORIS CUDA Graph probe results

Interpreter requested: `/runtime/venv/bin/python`

Five independent probe processes completed. Inspect first exceptions and actual allocator values in each full log before drawing conclusions.

| Experiment | Exit code |
| --- | --- |
| pointer-async | 0 |
| pointer-native | 0 |
| A-async-graph | 134 |
| B-async-no-graph | 0 |
| C-native-graph | 0 |

## Evidence

See `environment.log` for versions, device availability, repository commit and working tree.

### pointer-async.log

```text
4: allocator: cudaMallocAsync
7:           outside x: device=cuda:0, ptr=0x302000000, driver_result=0, device_ptr=0x302000000
8: captured external x: device=cuda:0, ptr=0x302000000, driver_result=0, device_ptr=0x302000000
9:      captured new y: device=cuda:0, ptr=0x8d4000000, driver_result=1, device_ptr=0x0
10:     after capture y: device=cuda:0, ptr=0x8d4000000, driver_result=1, device_ptr=0x0
11:      after replay y: device=cuda:0, ptr=0x8d4000000, driver_result=0, device_ptr=0x8d4000000
12: graph capture/replay completed
```

### pointer-native.log

```text
4: allocator: native
7:           outside x: device=cuda:0, ptr=0x7070c1c00000, driver_result=0, device_ptr=0x7070c1c00000
8: captured external x: device=cuda:0, ptr=0x7070c1c00000, driver_result=0, device_ptr=0x7070c1c00000
9:      captured new y: device=cuda:0, ptr=0x7070c5600000, driver_result=0, device_ptr=0x7070c5600000
10:     after capture y: device=cuda:0, ptr=0x7070c5600000, driver_result=0, device_ptr=0x7070c5600000
11:      after replay y: device=cuda:0, ptr=0x7070c5600000, driver_result=0, device_ptr=0x7070c5600000
12: graph capture/replay completed
```

### A-async-graph.log

```text
4: allocator: cudaMallocAsync
7: LyCORIS tuner: CUDA Graph timing ENABLED
9: IN GRAPH CAPTURE; stream= 918776576 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
14: [LYCORIS FIRST EXCEPTION]
15: Traceback (most recent call last):
53: ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
64: frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x9d (0x74c1b635e60d in /runtime/venv/lib/python3.13/site-packages/torch/lib/libc10.so)
```

### B-async-no-graph.log

```text
4: allocator: cudaMallocAsync
7: LyCORIS tuner: CUDA Graph timing DISABLED
9: LyCORIS kernel + tuner completed; shape= (512, 192)
```

### C-native-graph.log

```text
4: allocator: native
7: LyCORIS tuner: CUDA Graph timing ENABLED
9: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
14: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
19: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
24: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
29: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
34: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
39: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
44: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
49: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
54: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
59: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
64: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
69: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
74: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
79: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
84: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
89: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
94: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
99: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
104: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
109: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
114: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
119: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
124: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
129: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
134: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
139: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
144: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
149: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
154: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
159: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
164: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
169: IN GRAPH CAPTURE; stream= 414094112 cfg= {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_O': 2, 'num_warps': 2, 'num_stages': 4}
```

## Interpretation and limits

Observed: only exit codes and logs above. A zero exit alone does not prove every graph was captured: LyCORIS can fall back. Check FIRST EXCEPTION and IN GRAPH CAPTURE records.

Inferred: compare A/B/C before attributing failure to allocator, graph capture or integration. No automatic upstream blame or repair claim is made.

Root-cause confidence: undetermined until experiments run. Next investigation targets: LyCORIS _Batch.__init__ and _bypass_fwd_run; no performance feature is disabled by default.

Unverified: real training, offload/ring behavior, backward kernels, actual training shapes, BF16 and a root-cause fix. The probe uses FP16 (512,256), up (192,8), down (8,256). Pointer queries and tracing may perturb capture; pointer mode is separate from A/B/C.

No fix applied: Triton, Graph timing and cudaMallocAsync remain enabled in A; B only disables LyCORIS graph timing; C only changes the allocator.

Changes: root lycoris_cuda_graph_probe.py and diagnostic files only. Package patches: none. Runtime patches: none.


```text
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/A-async-graph.exit
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/A-async-graph.log
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/B-async-no-graph.exit
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/B-async-no-graph.log
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/C-native-graph.exit
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/C-native-graph.log
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/RESULTS.md
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/environment.log
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/pointer-async.exit
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/pointer-async.log
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/pointer-native.exit
?? .codex-diagnostics/lycoris-cuda-graph/20261101T063636-143/pointer-native.log
?? .codex-diagnostics/lycoris-cuda-graph/RESULTS.md
?? .codex-diagnostics/lycoris-cuda-graph/environment.log
?? .codex-diagnostics/lycoris-cuda-graph/run.sh
?? KohakuMoss_LyCORIS_CUDA_Graph_Codex_Test.md
?? lycoris_cuda_graph_probe.py
```
