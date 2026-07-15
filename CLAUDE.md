## 设计哲学

- 纯 Python，无编译依赖
- 文件系统即数据库；server 与 orchestrator 解耦，通过文件系统传递状态
- 多节点通过共享文件系统协同，每个节点只写自己的 `nodes/<node_id>/`

## 运行时目录结构

每个 task 占用 **两个并列子树**，挂在 `$METAINFER_ROOT/nodes/<node_id>/` 下：

```
$METAINFER_ROOT/                    (默认 <cwd>)
└── nodes/
    └── <node_id>/                  (默认 hostname；$METAINFER_NODE_ID 可覆盖)
        ├── workspaces/             ← 迭代生成产物（结构由 task 包定义）
        │   └── <task_id>/
        └── .metainfer/             ← 元数据 + 日志
            ├── registry.json       (含 workspace_dir)
            ├── registry.lock
            ├── runtime.json
            ├── runtime.lock
            └── tasks/<task_id>/
                ├── requirements.json   {"task_id", "task_type", "created_at", "form": {...}, ...}
                ├── run.json            RunStatus (current_phase, current_iteration, …)
                ├── timeline.jsonl      每行: {"ts": float, "type": str, "payload": dict}
                ├── orchestrator.{pid,log}
                ├── agents.json
                ├── token_budget.json
                ├── iterations/<NNN>.json
                └── logs/<NNN>/
```

关键不变性：
- workspace_dir 由 orchestrator 写、用户读；.metainfer 由 orchestrator + WebUI 协同写。
- WebUI 重置时同时清两个目录：`reset_state_dir(state_dir, workspace_dir, task_id, task_type)`。
- Orchestrator CLI 必须接受 `run <req.json> --state-dir … --workspace-dir …`。

## 项目测试

- 每个修改都需要测试用例。Agent 操作使用 Mock 进行测试。

## 添加新任务类型

所有 task 类型是对等 plugin（包括 shell 自身 `sys-shell`）。新增一个类型 **只改 `metainfer/tasks/<your_task>/` 下的文件**。

**完整骨架和详细注释见：** `metainfer/tasks/example/`

核心步骤：
1. 复制 `metainfer/tasks/example/` → `metainfer/tasks/<your_task>/`
2. 全局替换 `X-type-id` / `X` / `example` 为你自己的名字
3. 取消 `register()` 调用的注释
4. 实现 `orchestrator/pipeline.py` 的迭代逻辑
5. 写测试

### 公共层契约摘要

| 层 | 文件 | 关键约束 |
|---|---|---|
| 文件 | `timeline.jsonl` | `{"ts": float, "type": str, "payload": dict}`；shell 不解释 `type` |
| 文件 | `requirements.json` | `task_type` 必须与 `TaskPlugin.task_type` 和 `WebPlugin.type` 一致 |
| 文件 | `run.json` | shell 读取 11 个字段（见 `state_reader.read_run` defaults）；task 字符串是 opaque token |
| CLI | orchestrator argv | 必须接受 `run <req.json> --state-dir … --workspace-dir …` |
| Web | `build_router(plugin)` | 返回相对路径 `APIRouter`；shell 挂载到 `/api/{type}/{task_id}` |
| Web | `_state_readers.py` | task 专属读取；**不往公共 `state_reader.py` 加 task 逻辑** |
| QA | `qa_config.resolve_target` | `(state_dir, payload) -> {events_file, target_workdir, target_label}` |

### URL 架构

- `sys-shell` → `/api/sys-shell`（无 `{task_id}`）
- task plugin → `/api/{type}/{task_id}`
- 前端静态资源 → `/static/plugins/{type}/`

### 验证

```bash
python -c "from metainfer.server.registry import all_plugins; import metainfer.tasks; print([p.type for p in all_plugins()])"
python -c "from metainfer.orchestrator.tasks import all_tasks; import metainfer.tasks; print([p.task_type for p in all_tasks()])"
python -m pytest
```
