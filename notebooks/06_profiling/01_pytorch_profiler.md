# PyTorch Profiler 集成指南

> 关联 contracts: `00_contracts/profiling_contracts.md`
> 关联 phases: E (perf test) 用本接口产出 trace；F (perf plan) 消费 trace 写瓶颈分析。

## 什么时候用、什么时候不用

| 场景 | 用 profiler | 说明 |
|---|---|---|
| E 阶段 perf oracle 跑稳态吞吐 | ✅ | 一次 benchmark 窗口抓一份 trace，供 F 分析 |
| F 阶段 perf planner 找瓶颈 | ✅（读，不写）| 直接看 `.json.gz` 的 kernel 分布 |
| 第一次启动 server、还在 debug crash | ❌ | crash 与 perf 无关，开 profiler 反而拖慢排查 |
| 单元测试、correctness oracle | ❌ | profiler 有 5–15% 开销，会影响 latency 测量 |

简单判定：**只在「代码已能稳定跑完一个 benchmark」之后开 profiler**。

---

## 输出格式

`torch.profiler` 的 `export_chrome_trace(path)` 产生标准 Chrome tracing JSON。

- 用 `gzip` 压缩后扩展名 `.json.gz`
- 打开方式：
  - Chrome → 地址栏输入 `chrome://tracing` → Load file
  - Perfetto UI：`https://ui.perfetto.dev/`（推荐，支持大文件）
  - 命令行：`gunzip -c file.json.gz | jq '.traceEvents | length'`

trace 里能看到的事件类别：

- `cpu` —— Python / C++ 函数调用，按 thread
- `cuda` —— kernel launch + cudaDeviceSynchronize
- `cuda_runtime` —— cuLaunchKernel / cudaMemcpy / cudaMalloc 等 runtime 调用

---

## 关键 schedule 参数解读

```python
schedule = tp.schedule(wait=1, warmup=1, active=3, repeat=1)
```

含义：profiler 跑 5 个 step：

1. **wait (1)** —— 不采样，让系统进入稳态
2. **warmup (1)** —— 开始采样但丢弃（warmup 期间的统计不稳定）
3. **active (3)** —— 真正记录的 3 个 step
4. 整个 wait→warmup→active 序列重复 `repeat` 次

「step」由你的代码显式调用 `profiler.step()` 推进。在 inference server 里推荐的做法：在每次 `forward()` 后 step 一次。这样 1 step ≈ 1 个 batch 的 forward。

如果 server 是 continuous batching，每秒可能跑几十个 step —— `active=3` 就是几十毫秒的窗口，足够看到 kernel 分布。

---

## 完整集成示例

`profile.py`（参考 contracts 里的骨架，这里是 server 集成版）：

```python
# profile.py
import os
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import torch.profiler as tp


class ProfileContext:
    def __init__(self) -> None:
        self.enabled = os.environ.get("METAINFER_PROFILE", "0") == "1"
        self.profiler: Optional[tp.profiler.profile] = None
        self._timer: Optional[threading.Timer] = None
        self.rank = int(os.environ.get("LOCAL_RANK", "0"))

    def start(self) -> None:
        if not self.enabled:
            print("[metainfer-profile] enabled=0", flush=True)
            return
        outdir = Path(os.environ.get("METAINFER_PROFILE_OUTDIR", "."))
        outdir.mkdir(parents=True, exist_ok=True)
        activities_str = os.environ.get("METAINFER_PROFILE_ACTIVITIES", "CPU,CUDA").upper()
        activities = []
        if "CPU" in activities_str:
            activities.append(tp.ProfilerActivity.CPU)
        if "CUDA" in activities_str:
            activities.append(tp.ProfilerActivity.CUDA)

        wait = int(os.environ.get("METAINFER_PROFILE_SCHEDULE_WAIT", "1"))
        warmup = int(os.environ.get("METAINFER_PROFILE_SCHEDULE_WARMUP", "1"))
        active = int(os.environ.get("METAINFER_PROFILE_SCHEDULE_ACTIVE", "3"))
        repeat = int(os.environ.get("METAINFER_PROFILE_SCHEDULE_REPEAT", "1"))
        duration_s = int(os.environ.get("METAINFER_PROFILE_DURATION_S", "60"))

        ctx_self = self

        def _handler(p: "tp.profiler.profile") -> None:
            out = outdir / f"metainfer-profile-rank{ctx_self.rank}-{int(time.time())}.json.gz"
            p.export_chrome_trace(str(out))
            print(f"[metainfer-profile] wrote {out}", flush=True)

        self.profiler = tp.profile(
            activities=activities,
            schedule=tp.schedule(wait=wait, warmup=warmup,
                                 active=active, repeat=repeat),
            on_trace_ready=_handler,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.profiler.start()
        print(f"[metainfer-profile] enabled=1 outdir={outdir} "
              f"activities={activities_str} duration_s={duration_s}", flush=True)

        self._timer = threading.Timer(duration_s, self._safe_stop)
        self._timer.daemon = True
        self._timer.start()

        try:
            signal.signal(signal.SIGUSR1, self._on_sigusr1)
        except (ValueError, OSError):
            pass  # not in main thread

    def _on_sigusr1(self, signum, frame) -> None:
        self._safe_stop()

    def _safe_stop(self) -> None:
        if self.profiler is None:
            return
        try:
            self.profiler.stop()
        except Exception:
            pass
        self.profiler = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def step(self) -> None:
        if self.profiler is not None:
            try:
                self.profiler.step()
            except Exception:
                pass

    def stop(self) -> None:
        self._safe_stop()
```

`server.py`：

```python
# server.py
from profile import ProfileContext

def serve(port: int, model_dir: str) -> None:
    # ... model loading ...
    pctx = ProfileContext()
    pctx.start()

    def on_request_done():
        pctx.step()  # advance profiler schedule after each forward

    try:
        # ... main serve loop, call on_request_done() after each batch ...
        server.serve_forever()
    finally:
        pctx.stop()
```

---

## 常见坑

### 1. `.json.gz` 文件是空的（0 字节或只有 header）

**原因**：profiler 还在 active 状态被 `os._exit()` 强杀，没机会 export。

**修复**：在 server 的 SIGTERM handler / `finally:` 块里 **必须** 调 `ctx.stop()`，然后才退出。不要用 `os._exit()`；用 `sys.exit()` 或正常 raise。

### 2. trace 里看不到 CUDA kernel

**原因**：`METAINFER_PROFILE_ACTIVITIES` 没包含 `CUDA`，或者 server 在 profiler 还没采到 active step 时就退出了。

**修复**：
- 检查环境变量
- 确认 `METAINFER_PROFILE_DURATION_S` ≥ benchmark wall time + 5s
- 确认 `step()` 在每次 forward 后被调用（否则 schedule 不会推进）

### 3. profiler 启用后吞吐掉了 30%

**原因**：`record_shapes=True` 或 `with_stack=True` 被默认打开。

**修复**：保持 `record_shapes=False, profile_memory=False, with_stack=False`。只在调试某个具体 kernel 形状问题时短时启用。

### 4. 多 rank 时文件互相覆盖

**原因**：trace handler 输出文件名没带 rank/pid。

**修复**：用上面的自定义 handler，文件名 `metainfer-profile-rank{LOCAL_RANK}-{ts}.json.gz`。

### 5. trace 文件巨大（>1GB）

**原因**：active step 太多或 schedule repeat 无限。

**修复**：`active=3, repeat=1` 是合理的稳态采样。如果你只是想看「稳定状态下都跑了什么 kernel」，3 个 step 已经足够；不要 active=1000。

---

## Perf oracle 怎么用这些 trace

Perf oracle 跑完 benchmark 后：

1. 不解析 trace 内容（避免依赖 `torch.profiler` 的 Python API）
2. 只在 `perf-report.json` 的 `profile_artifacts` 字段记录：
   ```json
   "profile_artifacts": [
     {
       "path": ".../profile/metainfer-profile-rank0-1720000000.json.gz",
       "size_bytes": 1234567,
       "host": "k100-01",
       "rank": 0
     }
   ]
   ```
3. F-step perf planner 拿到路径后，可以：
   - gunzip 后 jq 看顶 10 个最耗时的 kernel
   - 或者 `python -c "import gzip,json; d=json.load(gzip.open('...')); ..."`
   - 写到 `perf_plan.md` 的「Bottleneck analysis」节作为引用

---

## 替代方案：py-spy / nsys

如果 PyTorch profiler 不够用（例如想看 Python 端 GIL / sys call），可以并行用：

- **py-spy**：进程外采样，零侵入
  ```bash
  py-spy record -o flame.svg --pid <server_pid> --duration 60
  ```
- **nsys (Nsight Systems)**：系统级 trace，含 CPU + CUDA + NCCL
  ```bash
  nsys profile -o report -t cuda,nvtx,osrt -d 60 <command>
  ```

但这些都 **不在** 推理框架代码内集成 —— 是 perf oracle 或开发者手工跑的进程外工具。框架代码只保证 `torch.profiler` 的 in-process 接口可用。
