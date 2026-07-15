## 设计哲学

- 简洁，无第三方依赖
- 文件系统即数据库
- web server是项目入口，各个功能是独立的编排器进程，每个编排器各自编排自己的agent，agent执行后将数据存储到文件系统，web server读取落盘的文件并在前端展示。web server和编排器解耦，分别可以各自独立重启，任意一方重启后都可以继续恢复原来的工作。
  - 该设计为日后一个控制平台管理多机器、多节点任务打下基础，多结点之间通过共享文件系统来传递状态信息。 

## 运行时目录结构

每个 task 在磁盘上占**两个并列子树**：workspace（用户关心的迭代生成产物）和 state_dir（元数据 + 日志 + prompt 等中间产物），都挂在当前节点 `nodes/<hostname>/` 下。最外层是 `METAINFER_ROOT`（默认 `<cwd>`，多节点共享文件系统时指向共享挂载点）。

```
$METAINFER_ROOT/                                (默认 <cwd>；多节点用共享挂载点)
└── nodes/
    └── <hostname>/                             (或 $METAINFER_NODE_ID)
        ├── workspaces/                         ← 迭代生成产物（用户视角的"成果"）
        │   └── <task_id>/
        │       ├── (gen-infer-framework)
        │       │   ├── 001/                    ← 迭代 N 的框架代码
        │       │   └── 002/
        │       └── (calc-theoretical-value)
        │           ├── step0/ … step4/
        │
        └── .metainfer/                         ← 元数据 + 日志 + prompt 中间产物
            ├── registry.json                   ← 全局任务清单（带 workspace_dir 字段）
            ├── runtime.json
            └── tasks/
                └── <task_id>/
                    ├── requirements.json
                    ├── run.json
                    ├── timeline.jsonl
                    ├── orchestrator.{pid,log}
                    ├── agents.json
                    ├── token_budget.json
                    ├── iterations/*.json       (gf only)
                    └── logs/<NNN>/             (gf only — prompt / oracle / server 日志)
```

关键不变性：
- workspace_dir 由 orchestrator 写、用户读；.metainfer 由 orchestrator + WebUI 协同写。
- WebUI 重置任务时**同时**清两个目录（`reset_state_dir(state_dir, workspace_dir, ...)`）。
- Orchestrator CLI 必须显式接受 `--state-dir` **和** `--workspace-dir`；launcher 同时传两个。
- 多节点场景：每个节点只写自己的 `nodes/<hostname>/`，中央控制器扫 `nodes/*/` 看全局。
- **不支持向后兼容**：老的单层 `state_dir/code/` 布局作废；in-flight 任务 reset 重建。

## 关于项目测试

- 每一个修改都要编写对应的测试用例。模块编写独立的测试用例，重要函数编写内联的unit test或者doc test。涉及到Agent操作的地方，使用Mock的Agent进行测试。

## 添加新任务类型 (task type)

每个 task 都是**完全自包含**的功能包，所有属于 X 的 orchestrator、web handler、前端、测试、表单 schema 都跟着 X 走，放在同一个目录下。**没有「基础 task」和「扩展 task」的区分** —— 每个任务类型都是对等的 plugin。新增一个任务类型 X 的步骤（**完全不需要修改** `metainfer/web/app.py`、`metainfer/static/index.html`，也不需要碰公共的 `metainfer/static/components/`、`metainfer/static/views/`）：

### 0. 目录结构（一个 task 包 = 一个完整功能包）

```
metainfer/tasks/X/                         # X 是 package 名（snake_case）
├── __init__.py                            # 触发 plugin 注册：from .orchestrator import plugin; from .web_server_handler import plugin
├── form.yaml                              # 问题表单 schema（X 专属）
├── orchestrator/                          # X 的编排逻辑（pipeline / phases / cli / ...）
│   ├── __init__.py                        # from metainfer.orchestrator.tasks import register; register(PLUGIN)
│   ├── plugin.py                          # TaskPlugin(task_type="X-type-id", cli_module=..., phases_module=...)
│   ├── cli.py / pipeline.py / phases.py / ...
├── web_server_handler/                    # X 的 web plugin（HTTP 路由 / QA / detail view 注册）
│   ├── __init__.py                        # 空 / docstring
│   ├── plugin.py                          # WebPlugin(type="X-type-id", detail_view_module=..., frontend_dir=..., importmap_entries=..., qa_config=...)
│   ├── routes.py / _readers.py / _qa.py   # 按需
├── static/                                # X 专属前端 ES 模块（X-detail.js 等），自动 mount 到 /static/plugins/<type>/
└── tests/                                 # X 的所有测试（orchestrator + web plugin）
```

参考实现：`metainfer/tasks/calc_value/` 和 `metainfer/tasks/gen_infer_framework/`。

### 1. Orchestrator 侧（在 task 包内）
- `metainfer/tasks/X/orchestrator/`：至少包含 `cli.py`（入口）、`pipeline.py`（迭代循环）、`orchestrator.py`（启动器）、`plugin.py`（`TaskPlugin` 注册）、`phases.py`（状态机节点/边）。
- 在 `metainfer/tasks/X/__init__.py` 里写 `from .orchestrator import plugin` 触发 `TaskPlugin` 注册（参考现有 task 包）。

### 2. 前端表单（在 task 包内）
- `metainfer/tasks/X/form.yaml`：问题表 schema，WebUI 新建任务表单从这里读。参考 `metainfer/tasks/calc_value/form.yaml`。
- `metainfer/web/forms.py`：在 `TASK_TYPE_META` 里登记 `{<X-type-id>: {label, description}}`（这是公共的 task-type picker metadata，所以还在 web 层）。

### 3. WebUI 插件（在 task 包内 —— 每个 task 都要有）
**没有「按需」**：每个 task type 都必须有自己的 web_server_handler 包，至少声明 `detail_view_module` + `frontend_dir` + `importmap_entries` + `qa_config`。即使是「全用通用端点」的 task，也要建一个最小 plugin 包来表达它是一个一等任务。

- `metainfer/tasks/X/web_server_handler/plugin.py`：构造 `WebPlugin(...)` 并调用 `register(plugin)`，必填字段：
  - `type="<X-type-id>"`：必须与 orchestrator 的 `TaskPlugin.task_type` 一致，也必须与 `form.yaml` 里的 type 字段一致
  - `detail_view_module="app/X-detail"`：detail body 视图的 importmap key（shell 会 `import(detailViewModule)` 动态加载）
  - `frontend_dir=Path(__file__).resolve().parent.parent / "static"`：指向 task 包里的 `static/` 目录，会被自动 mount 到 `/static/plugins/<type>/`
  - `importmap_entries={...}`：要把哪些 importmap key 指向 `/static/plugins/<type>/...?v=CACHE_BUST`（`CACHE_BUST` 由后端替换）。**至少要包含 `detail_view_module` 这一条**。
  - `qa_config=...`：QA pathsolver（即使是 frontend-driven 的简单实现也要有，参考 `gen_infer_framework/web_server_handler/_qa.py`）
- `metainfer/tasks/X/static/X-detail.js`：X 的详情 body 组件，`export default function XDetailView({ taskId, run, status, data, onOpenRetro })`。共享数据（iterations/timeline/charts/graph/agents）由 shell 通过 `data` prop 传入。
- `routes.py`（可选）：X 专属的 FastAPI 路由（用 `from metainfer.web._helpers import task_or_404, state_dir_for, require_task_type`）。没有专属路由就不建。
- **无需修改** `metainfer/static/index.html` 或 `metainfer/web/app.py` —— `metainfer/tasks/__init__.py` 通过 `pkgutil.iter_modules` 自动发现并 import 每个 task 包；每个 task 包的 `__init__.py` 再 import 自己的 orchestrator/web_server_handler plugin，触发 `register()`。`app.py._serve_index()` 会把所有 plugin 的 `importmap_entries` 注入到 index.html，`app.mount` 会把每个 plugin 的 `frontend_dir` 挂到 `/static/plugins/<type>/`。
- **`metainfer/static/views/task-detail.js` 是 shell**：负责 header / 控制按钮 / Reset/Retrospective modal / BudgetBar / 共享数据拉取，然后根据 `detail_view_module` 动态 import plugin 的 body 组件。加新 task 不需要碰这个文件。

### 4. 测试（跟着 X 走）
- `metainfer/tasks/X/tests/`：X 的所有测试（orchestrator 逻辑 + web plugin 路由/QA）。包内 `__init__.py` 空文件，conftest 共享 fixture 从 `metainfer.testing` 引入。
- 共享 mock 工具统一从 `metainfer.testing` 引入：`from metainfer.testing import MockAgentManager, FakeStore, FakeLauncher, isolated_env, write_calc_script`

### 5. 注册到 pytest.ini（必改 —— 这是唯一需要改公共文件的地方）
- `pytest.ini` 的 `testpaths` 加一行 `metainfer/tasks/X/tests`，让 pytest 自动扫到。

### 验证
- `python -c "from metainfer.web.registry import all_plugins; import metainfer.tasks; print([p.type for p in all_plugins()])"` 应包含 `'<X-type-id>'`
- `python -c "from metainfer.orchestrator.tasks import all_tasks; import metainfer.tasks; print([p.task_type for p in all_tasks()])"` 也应包含 `'<X-type-id>'`（orchestrator 侧也注册了）
- 跑 `python -m pytest` 全绿（pytest.ini 里的 testpaths 自动扫到 X 的 tests/）
- 启动 WebUI，新建任务下拉里应出现 X，详情页能渲染（专属前端模块从 `/static/plugins/<X-type-id>/...` 加载）
