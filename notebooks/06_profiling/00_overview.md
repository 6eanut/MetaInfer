# Profiling 章节

这一章定义推理框架必须预留的 profiling 接口，以及 E-step perf oracle / F-step perf planner 如何使用它。

## 阅读顺序

1. [`01_pytorch_profiler.md`](01_pytorch_profiler.md) —— 集成指南、完整示例、常见坑
2. [`../00_contracts/profiling_contracts.md`](../00_contracts/profiling_contracts.md) —— 接口契约（环境变量、命名、生命周期）

## TL;DR

每次迭代产出的推理框架 **必须** 在 server 启动路径中：

1. 读取 `METAINFER_PROFILE` 环境变量
2. 若为 `1`，启动 `torch.profiler`（schedule + on_trace_ready），并启动一个后台定时器在 `METAINFER_PROFILE_DURATION_S` 秒后自动 stop+export
3. 在 SIGTERM / atexit 处理里确保 `profiler.stop()` 被调用
4. 输出 `.json.gz` 到 `METAINFER_PROFILE_OUTDIR`，文件名带 rank + 时间戳

环境变量未设时，profiler 不启动，server 性能与未集成 profiler 时一致。
