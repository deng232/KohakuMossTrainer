# KohakuMossTrainer × ComfyUI：两节点、无训练核心改动集成设计（v2）

**状态：供 Codex 实施。此文档取代之前包含 `TrainingHooks`、`StepObserver`、修改 `NativeDriver` 和暂停控制的设计。**

参考：
- https://github.com/KohakuBlueleaf/KohakuMossTrainer/
- https://github.com/Comfy-Org/ComfyUI/
- https://github.com/Comfy-Org/ComfyUI_frontend/
- https://github.com/leeoniya/uPlot

## 0. 严格边界（不要自行扩大）

- **不得修改 KohakuMossTrainer 的训练计算、训练循环、callback、checkpoint、调度器、数据缓存或停止机制**。只使用现存 Python library API；如当前发行包的公共导入组织不便，允许向 KohakuMoss package `__init__` 添加**仅重导出（re-export）**的稳定入口，不改业务逻辑；最好连此项也省略。
- 不启动 `kogine`、`subprocess.Popen` 或另一个训练 Python 进程。KohakuMoss 的 `training.entry.run()` 在 ComfyUI 的同一 Python 进程中执行。
- 训练相关内部进程分支首版全部禁用：`PARALLEL='none'`, 单 GPU，`ENCODE_DEVICES=()`；设置 `NUM_WORKERS=0`、`DATA_SCAN_WORKERS=1` 以避免 Dataset/scan 路径的 worker process。若 kernel/compile 自身启动辅助进程，需要另外查明并报告，不能仅据这四项宣称整个进程树绝无子进程。
- 暂时**不实现**内存 Pause/Continue、运行中 Extend Steps、异步强杀训练线程或“任意步立即保存 checkpoint 再停止”。原生 `SAVE_EVERY`、`SAVE_MINUTES`、`FINAL_CHECKPOINT` 和重启时 `RESUME='latest'` 可以复用。不得把 ComfyUI 通用 Interrupt 描述成 Kohaku 的可靠安全停止。
- 需要两个 ComfyUI **工作流节点**：`KohakuMoss Training` 和 `KohakuMoss uPlot Monitor`。前端 uPlot 图表不是后端训练任务。
- 以“**训练占用 ComfyUI 原有的 prompt-worker；浏览器 WebSocket/UI 独立更新**”为 v1 默认策略，避免返回后其他 ComfyUI workflow 在同一 GPU 上不受控地并发推理。不要为了让下游节点能执行而擅自改为后台训练线程。
- 不使用 ComfyUI 内置 `Make Training Dataset` 缓存。KohakuMoss 原生 DataConfig/cache 为唯一真源；不新增 Dataset/Export 自定义节点。

## 1. 已验证的现成 API 与关键发现

### 1.1 训练入口本身就是 Python library

`src/kohakumoss/training/entry.py`：

```python
from kohakumoss.training.entry import run
state = run(knobs)
```

`run(g: dict)` 现有行为：校验 knobs，`ENCODE_DEVICES` 非空时走 encode-feed 独立训练进程；某些 `PARALLEL` layout 会 spawn；单 GPU / native / 无 encode-feed 时调用 `worker()` 并在训练完成时返回 `TrainState`。**不能从 `run()` 返回值获取训练中的实时状态**。

### 1.2 不改 Trainer 就能实时订阅指标：LOGGER Registry

`src/kohakumoss/logging/__init__.py`：

```python
LOGGERS = Registry('logger')
LOGGERS.add('console', ...)
LOGGERS.add('wandb', ...)

def build_logger(kind, output_dir, name, project, config, is_main, resume=False):
    return LOGGERS.get(kind)(output_dir=output_dir, name=name,
                             project=project, config=config, resume=resume)
```

`src/kohakumoss_common/registry.py`：`LOGGERS.add(name, factory)`；`Registry.get` 也支持 dotted Python import path。`knobs.py` 的 LOGGER 校验使用该 Registry。

`src/kohakumoss/training/callbacks/monitor.py` 的 `LogMetrics.on_step_end` 已调用：

```python
self.logger.log(self.keep(metrics), state.step)
```

这意味着只在 **ComfyUI custom node package** 内注册自定义 Logger，即可获得真正的训练指标和实际 step，不需要修改 Trainer 的核心源码！

### 1.3 Logger 的重要限制

- `LogMetrics` 仅按 `METRICS_EVERY`（未设置时按 `LOG_EVERY`）发出指标；**不等于每一步必发**。需要每步曲线可设 `METRICS_EVERY=1`，但宿主值同步与消息会带来额外成本。建议默认 `METRICS_EVERY=5`、`LOG_EVERY=10`，允许用户调整。
- `METRICS_LEVEL='train'` 会调用 `train_row` 过滤字段，**VRAM 指标可能被滤掉**。需要 VRAM 时设 `METRICS_LEVEL='bench'`（或 `debug`），然后根据实际收到字段绘图。绝不虚构缺失的指标。
- Logger 只在 Trainer 构造完成后开始工作。因此**模型构建 / prewarm 期间无法从它获知精确进度**。v1 的任务状态只标为 `INITIALIZING`；到第一条指标/日志后标为 `TRAINING`。若通过宿主单独执行 cache library，可在宿主层记录 `CACHE_TEXT`、`CACHE_LATENTS` 阶段和进度。
- 事件实时，历史与重连可以利用原生 `<run>/metrics.jsonl`，但禁止**把轮询该文件当作实时指标来源**。

## 2. 推荐工作流：两个节点，后端执行与前端更新分离

```text
[ KohakuMoss Training ] --[TRAINING_JOB]--> [ KohakuMoss uPlot Monitor ]
          |                                        |
          | Python 训练（prompt-worker，阻塞）    | Frontend JS on WebSocket
          v                                        ^
 kohakumoss.training.entry.run(knobs)               |
          |                                        |
      LogMetrics → LOGGER='comfyui_live' → SessionStore → PromptServer.send_sync
```

**核心陷阱**：ComfyUI 在上游训练节点 `execute()` 结束之前，不会把 `TRAINING_JOB` 值交给下游 node 的 Python `execute()`。因此监控**不能**依赖下游后端执行，也不能等待这个 output 的运行时值。

**解决方式**：`KohakuMoss uPlot Monitor` 是**前端主动呈现的 UI 节点**。它的 JS 在 `nodeCreated` / workflow load 时，通过 LiteGraph 链接拓扑找出输入 socket 的 `origin_id`（连接的 Training node），记录该训练节点的稳定图内 node ID。每次 Training 开始时，后端事件带 `origin_node_id` 和新生成的 `job_id`；JS 将它与 uPlot Monitor 绑定。不必等待后端运行时 output 传递。

`TRAINING_JOB` 类型的连线用于 UI 的**静态关联**与工作流可读性；真正实时路由用事件字段。训练完成后 Training node 可以输出 JSON-safe 的 `job_id`、`run_dir` 等结果，下游 Python monitor 返回空结果即可。不要通过会变动的 `run_name` 单独区分两个同时存在的 workflow 执行；至少使用 `job_id`。

Training node 应标记为可执行输出节点或明确提供可排队执行的接线方式，保证只运行训练节点也能 Queue Prompt。必须验证是否因 ComfyUI 的节点执行缓存而被跳过；建议有明确的 `run_token`/手动重跑机制和唯一 `job_id`，而非页面刷新时重启训练。

## 3. Custom Node #1：KohakuMoss Training

### 3.1 输入分组（普通 ComfyUI widget / editable input）

| 分组 | 例子 | 数据来源 |
|---|---|---|
| Model | WEIGHTS root、模型选择、VAE、TE | Kohaku config + Comfy `folder_paths` 文件发现 |
| Data | DATASETS 路径、source、resolution、caption | Kohaku DataConfig |
| Cache | LATENT_CACHE、TEXT_CACHE、CACHE_DIR、编码精度 | Kohaku 原生 cache |
| Train | ALGO、DIM、ALPHA、LR、STEPS、MAX_TOKENS、GRAD_ACC | Kohaku 原生 knobs |
| Schedule | SCHEDULE、warmup、end | Kohaku 原生 AnySchedule 配置 |
| Memory | MEMORY_PROFILE、PRECISION、device | Kohaku 原生 presets |
| Save | SAVE_EVERY、SAVE_MINUTES、KEEP_LAST、FINAL_CHECKPOINT、SAVE_FORMAT | Kohaku CheckpointManager |
| Monitor | METRICS_EVERY、METRICS_LEVEL、LOG_EVERY | Kohaku LogMetrics |
| Run | RUN_NAME、OUTPUT_DIR、RESUME | Kohaku 原生恢复约定 |

首版默认单 GPU，无多 rank。Dataset、模型路径绝不硬编码到作者开发机。`Save LoRA Weights` 节点无须参与：`SAVE_FORMAT='comfy'` 由 Kohaku 原生导出。

### 3.2 执行时序

```text
node.execute()
  ├── 生成唯一 job_id；保存 origin_node_id
  ├── 原子申请 TrainerSession 单例锁
  ├── 合成配置（train_starter/compose，或加载有效 Kohaku 配置）
  ├── 设置 LOGGER='comfyui_live' / METRICS_LEVEL='bench'
  ├── 校验 kohakumoss.training.knobs.check(knobs)
  ├── 强制 no-spawn 单 GPU 约束
  ├── 更新状态 INITIALIZING / 推送 job_started
  ├── 清理不再需要的 Comfy 模型（prompt-worker 内）
  ├── 需要时调用 Kohaku 原生数据缓存 API（同进程，可发送 cache progress）
  ├── run(knobs)  # 同线程、同步、训练直到结束
  ├── 更新 COMPLETED、最终 checkpoint / export 文件
  └── finally: 释放 Session 锁、任务资源及可释放缓存
```

Kohaku 的 `run()` 在开始时会进行 signal/logging 等 process-global 初始化；需要检查它们在 ComfyUI 下的兼容性，尤其不能让训练停止动作发送 SIGTERM 到 ComfyUI PID。首版无任意步 Stop 按钮。

### 3.3 Runtime Session

- `job_id`: 每次执行产生新 ID；同一个 workflow node 可以有历史多个 job。
- `origin_node_id`: Comfy workflow 中训练节点的 unique ID。
- `run_name`, `run_dir`, `status`, `phase`, `last_metric_step`, `last_metrics`, `total_steps`，`started_at`, `error`。
- 有界 ring buffer，例如最多 10k 指标行（历史落盘由 Kohaku 负责）。
- 一个 `threading.Lock` 保证不同时执行两个 Train node。
- `JobSnapshot` 仅持有数字和字符串，绝不把 GPU tensor / optimizer / TrainState 可变对象交给 HTTP 线程。

## 4. 实时 Logger：不修改 Kohaku 代码

在 custom node 包里定义 `LiveLogger`，在加载时注册：

```python
from kohakumoss.logging import LOGGERS
from kohakumoss.logging.sinks import ConsoleLogger

class LiveLogger(ConsoleLogger):
    def __init__(self, output_dir, name, project, config, resume=False):
        super().__init__(output_dir)
        self.job_id = session_store.find_by_run_dir(output_dir)

    def log(self, metrics: dict, step: int) -> None:
        super().log(metrics, step)  # 原生 metrics.jsonl 保留
        session_store.publish_metrics(self.job_id, step, dict(metrics))

LOGGERS.add("comfyui_live", LiveLogger)
```

这段为**示例接口**：Codex 需在真实代码中实现 `session_store`、校验 run_dir 映射、防止 `job_id=None`、处理没有网络订阅者的情况，并复制/转换数值为 JSON-safe 类型。保留 `ConsoleLogger` 的 `resume_at`、`close` 和历史功能。`Registry.add` 是现有、可验证的 API。

指标事件样例：

```json
{
  "job_id": "...",
  "origin_node_id": "17",
  "event_seq": 22,
  "step": 150,
  "total_steps": 2000,
  "timestamp": 1793450000,
  "metrics": {"loss": 0.42, "lr": 0.0001, "grad_norm": 0.8, "step_time": 2.4}
}
```

以上数字是结构示例，非真实训练数据。对于 NaN、Inf、None 要用 JSON 合法形式表示，例如 `null`，不要让 WebSocket 发送含非法 JSON 的内容。

`SessionStore.publish_metrics` 只做 CPU 标量入队/快照更新。调用 `PromptServer.instance.send_sync('kohakumoss.metrics', payload)`（ComfyUI 已有线程安全队列桥），不得在训练线程阻塞等待浏览器帧渲染。浏览器不在线也不影响训练。

## 5. Custom Node #2：KohakuMoss uPlot Monitor

这是 **Backend no-op + Frontend JS 自绘** 的监控节点。

### 5.1 Node 输入与 UI

- 1 个 `TRAINING_JOB` 输入（连 Training 的输出，用图连线确定 upstream node ID）。
- 可选视图设置：显示 Loss/LR/VRAM、平滑窗口、图表宽高、历史点数。
- 单节点内显示：`status`、当前收到的 `step`、已知 `total_steps`、loss/lr、主 uPlot 曲线。
- 高级详情可以作为 UI 面板弹出或独立侧边栏，但不新增第三个 workflow node。

### 5.2 前端生命周期

1. 在 `WEB_DIRECTORY` 内用 `app.registerExtension()` 注册对 Monitor node 的扩展。
2. `nodeCreated` / workflow load 时初始化 DOM/canvas；从 `inputs[0].link -> graph.links[link].origin_id` 找 upstream Training node。
3. `api.addCustomEventListener('kohakumoss.job_started', ...)`、`api.addCustomEventListener('kohakumoss.metrics', ...)`，按 upstream node ID + job_id 过滤。
4. 显示时用 uPlot `setData()` 更新；收消息频率可以每 5 steps，但 UI 最多约每秒刷新 2 次。注意 uPlot 数据是 `[xSteps, yLoss, yLR, ...]` 的列式数组，所有列长度相同，无数据记为 `null`。
5. on node removed / graph reload 时注销监听器并销毁 uPlot 实例，避免重复订阅、内存泄露。
6. 页面刷新或 WebSocket 重连后通过 HTTP 恢复 `GET /kohakumoss/jobs?origin_node_id=...` + `GET /kohakumoss/jobs/{job_id}/history`，再继续消费实时事件。
7. 前端拿不到 Job 事件时显示 `Waiting for training`，不得凭图连接假定已有正在运行的训练。

建议打包 **uPlot 的本地 JS 和 CSS**，不要加载 CDN，不依赖 WebGPU。可用 `<canvas>`/DOM widget，具体采用当前 ComfyUI frontend 推荐 API；避免覆盖 LiteGraph 全局 `onDrawForeground` 引起其它节点冲突。uPlot 是 MIT 许可，分发时保留声明。

## 6. HTTP API 与状态恢复

注册 `PromptServer.instance.routes` 下的 API：

| Endpoint | 行为 |
|---|---|
| `GET /kohakumoss/jobs` | 返回近期任务和当前任务，支持通过 node_id 查找 |
| `GET /kohakumoss/jobs/{job_id}` | 返回最后一次结构化 snapshot |
| `GET /kohakumoss/jobs/{job_id}/history` | 返回内存 ring buffer + 可信 run_dir 的原生 `metrics.jsonl` 历史（不是实时 polling） |

**没有** `POST /pause`、`POST /continue`、`POST /stop`、`POST /extend_steps`：这些不在 v1 的功能边界中，不能做一个实际上无效的按钮。`RESUME` 是下次重新 Queue Train node 时的配置，不是同一个在内存中的任务继续。

HTTP API 只能读取服务端已记录的 job，不接受任意文件路径来读历史。沿用 ComfyUI 的同源与身份策略；不要开额外 HTTP/WebSocket 服务。

## 7. GPU/队列与训练流程语义（不能掩盖）

**v1 同步 Training 方案**：

- `KohakuMoss Training.execute()` 占用 ComfyUI 现有 prompt-worker 直到训练完毕。
- uPlot Monitor 的 **前端 UI** 在此期间保持活跃（由 WebSocket 更新）；它的 **Python execute** 尚不能拿到 upstream 的运行时 `TRAINING_JOB` 输出，属预期。
- 好处：ComfyUI 正常队列不会在该训练后面启动新的 GPU workflow，减少 GPU 争用；无需再创建训练线程或修改 ComfyUI 调度核心。
- 代价：训练期间不能同时执行其他 queued workflows；用户前端刷新不停止训练。

**不要默认改成 `threading.Thread` 启动训练并让 Training node 立即返回**：虽然同进程且能立刻得到 Job ID，但 ComfyUI 此后可以继续执行其它 GPU workflows，`comfy.model_management.in_training` 不应被假定为完整 GPU 互斥调度系统。若未来确实需要异步 job，必须先设计并验证对其他 ComfyUI nodes 的 GPU 排它控制，而不是简单加一把自定义节点内部锁。

## 8. 训练期间可以做到 / 无法做到

| 需求 | 现有库能否做到 | v1 处理 |
|---|---|---|
| 真正的 Loss/LR 曲线 | 可以，原生 LogMetrics + Logger Registry | 实现 |
| 训练总进度 | 有真实 step 和 STEPS，但仅按 metrics cadence 收到 step | 实现；不虚构中间步 |
| 显存监控 | 按 METRICS_LEVEL=bench/debug 和原生指标可用性 | 实现可选显示 |
| Cache progress | Kohaku 原生 cache builder 支持 progress callback | 实现（当使用此库路径时） |
| 模型 build/prewarm 内部细分 | 当前未有公开同类 phase observer | v1 仅显示 INITIALIZING |
| 自动 checkpoint | 原生 SAVE_EVERY/SAVE_MINUTES | 实现配置及展示 |
| 训练结束 LoRA 导出 | 原生 SAVE_FORMAT=comfy | 实现 |
| 从完整 checkpoint 继续 | 原生 RESUME=latest | 新一次任务支持 |
| 内存 Pause/Continue | 非现成 API | 不实现 |
| 任意步保存并安全停止 | 不具备可直接外部调用的受控 step hook | 不实现 |
| 训练运行中动态追加 STEPS | 无公开安全动态控制入口 | 不实现 |
| 训练后增加 STEPS 并从 checkpoint 恢复 | 可以配合已有 RESUME，但必须测调度器曲线变化 | 允许新任务配置 |

## 9. 文件建议（均在 custom node repo 内，不改 Kohaku core）

```text
custom_nodes/ComfyUI-KohakuMoss/
  __init__.py
  nodes.py             # 两个 ComfyUI nodes
  config.py            # 配置层组合 + 原生校验
  cache.py             # 调用原生 Kohaku cache library
  live_logger.py       # LOGGERS.add('comfyui_live', LiveLogger)
  session.py           # job_id / node_id / snapshot / metrics ring buffer
  api.py               # REST endpoints + WS sends
  web/
    index.js           # app.registerExtension
    monitor.js         # graph linkage + event store
    uplot.iife.min.js  # 本地固定版本
    uPlot.min.css
  tests/
    test_logger_registry.py
    test_realtime_during_execute.py
    test_job_binding.py
    test_no_train_subprocess.py
    test_model_compatibility.py
```

如果 custom node 包目录名包含连字符，注意 **Python dotted import** 路径不可直接用此目录名；这里采用 `LOGGERS.add` 注册对象而非 dotted path，避免模块路径别名问题。

## 10. 实施顺序及硬性验收

### P0 — 证实 Logger 插件，绕开所有 Trainer core patch

- 在当前 ComfyUI 环境安装/导入 KohakuMoss 与完整 Python 依赖；检测 PyTorch 版本冲突，不自动重装 Torch。
- 在 custom node init 中调用 `LOGGERS.add('comfyui_live', LiveLogger)`。
- 合成有效 knobs，`LOGGER='comfyui_live'`，直接 `run(knobs)`。
- 用 mock Logger 测试每条指标能进入 SessionStore；用真实短训练在执行**尚未返回时**观察 WebSocket 指标。

**通过判定：无需修改任何 KohakuMoss 源文件即可实时收到训练中实际 metrics。**

### P1 — 两节点图绑定

- 实现 Training 和 Monitor 两个 node；明确 `TRAINING_JOB` 是静态连接，实时绑定走 JS `origin_id` + WS job_started。
- 训练中监控节点即使没有后端 execute，uPlot 也能持续更新。
- 断开/重连后读取 HTTP snapshot/history，曲线不丢失。

### P2 — cache、保存、兼容性

- 使用 Kohaku 原生离线 cache API 和原生 encoder spec 校验；不得用 Comfy encoded `.pkl` 代替。
- 以原生 `SAVE_EVERY` 保存 checkpoint，`SAVE_FORMAT='comfy'` 导出 LoRA。
- 以原生 `RESUME='latest'` 恢复，验证 loss/step/optimizer/scheduler/loader 状态。
- 训练后跑 ComfyUI 普通推理；检查 Logger 重复注册、显存占用、worker process、重复 queue 等。

### 最终交付

1. 只修改 ComfyUI extension repo；若确需 Kohaku package re-export，给出逐行依据并保持训练逻辑零改动。
2. 交付两个可用节点和完整 ComfyUI workflow JSON。
3. 用 pytest / GPU 实际测试支撑“训练尚在进行中，uPlot 已显示真实 step/loss”的声明。
4. 报告真实验证过的 PyTorch/CUDA/ComfyUI/Kohaku 版本；不可编造训练结果。
5. 删除旧方案中的 `TrainingHooks`、`StepObserver`、`NativeDriver` 补丁、Pause/Stop endpoints、subprocess launcher 和 stdout parser。

**核心原则：KohakuMoss = Python library + 原生 metrics sink；ComfyUI = 两个工作流节点 + 前端图表与状态展示；Trainer core = 零修改。**
