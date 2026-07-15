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
        │       ├── <task_pkg>/                 ← task 包各自的 workspace 子树（如迭代目录、step 输出）
        │       │   └── ...                     ← 结构由 task 包自己定义；公共代码不感知
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
                    ├── iterations/*.json       (per-iter records, when applicable)
                    └── logs/<NNN>/             (prompt / oracle / server logs)
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
│   ├── plugin.py                          # WebPlugin(type="X-type-id", label=..., description=..., detail_view_module=..., frontend_dir=..., importmap_entries=..., qa_config=...)
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
- task-type 的 label + description 直接写在下面 §3 的 `WebPlugin` 里，**没有**公共登记表。

### 3. WebUI 插件（在 task 包内 —— 每个 task 都要有）
**没有「按需」**：每个 task type 都必须有自己的 web_server_handler 包，至少声明 `type` + `label` + `description` + `detail_view_module` + `frontend_dir` + `qa_config`。即使是「全用通用端点」的 task，也要建一个最小 plugin 包来表达它是一个一等任务。

- `metainfer/tasks/X/web_server_handler/plugin.py`：构造 `WebPlugin(...)` 并调用 `register(plugin)`，必填字段：
  - `type="<X-type-id>"`：必须与 orchestrator 的 `TaskPlugin.task_type` 一致
  - `label="..."` + `description="..."`：task-type picker 用，**直接写在这里**（不再有 `TASK_TYPE_META`）
  - `detail_view_module="app/X-detail"`：detail body 视图的 importmap key（shell 会 `import(detailViewModule)` 动态加载）。对应的 `X-detail.js` 必须存在于 `static/` 目录下（见下面 importmap 自动发现）。
  - `frontend_dir=Path(__file__).resolve().parent.parent / "static"`：指向 task 包里的 `static/` 目录，会被自动 mount 到 `/static/plugins/<type>/`
  - `importmap_entries={...}`（**可选，绝大多数情况留空**）：`create_app` 会**自动**遍历 `frontend_dir/*.js` 并按 `app/<文件名去 .js>` 注册 importmap，指向 `/static/plugins/<type>/<file>?v=CACHE_BUST`。所以只要文件名跟 importmap key 一致（`X-detail.js` ↔ `app/X-detail`），就不用列。**只有在以下两种情况才需要手动填 `importmap_entries`**：(a) 想用非 `app/<stem>` 形式的 key；(b) 想 override shell 的同名 key（例如用自己的 `state-graph.js` 替换公共 `app/state-graph`），服务端 merge 让 plugin 优先。
  - `extra_stylesheets=["X.css"]`（可选）：CSS 文件名列表（相对于 `frontend_dir`）。`create_app` 会在 shell 的 `styles.css` 后面注入对应的 `<link rel="stylesheet" href="/static/plugins/<type>/<file>?v=TOKEN">`。**X 专属样式只能走这条路**，不允许往公共 `metainfer/static/styles.css` 里加 task-specific CSS。路径校验会拒绝任何 `..` 逃逸 `frontend_dir` 的尝试。
  - `qa_config=...`：QA pathsolver（即使是 frontend-driven 的简单实现也要有，参考 `gen_infer_framework/web_server_handler/_qa.py`）
  - `register_routes=...`（可选）：X 专属路由注册回调。签名 `(app, deps, plugin) -> None`，第三个参数是 WebPlugin 本身，避免循环 import。**需要 QA 的 task 直接调** `from metainfer.web.qa_routes import register_qa_routes; register_qa_routes(app, plugin, prefix="/qa")`，不要复制 QA 路由壳。
  - `extra_watch_paths=...`（可选）：告诉 SSE watcher 额外要 watch mtime 的文件（比如增量刷新用的中间产物），签名 `(entry) -> List[Path]`。公共 `sse.py` 不再硬编码任何 task 路径。
- `metainfer/tasks/X/static/X-detail.js`：X 的详情 body 组件，`export default function XDetailView({ taskId, run, status, data, onOpenRetro })`。共享数据（iterations/timeline/charts/graph/agents）由 shell 通过 `data` prop 传入。共享 widget（charts/state-graph/iterations-table）直接 `import "app/charts"` 等，shell 自带 importmap 条目指向 `metainfer/static/components/` 下的默认实现；如需分歧，在 plugin 的 `importmap_entries` 里 override 同名 key 即可。
- `metainfer/tasks/X/static/X-api.js`（按需）：X 专属的 fetch helper（命中 `/api/tasks/<id>/<X>/...`）放这里，不要往公共 `metainfer/static/components/api.js` 里加。参考 `calc_value/static/calc-api.js`。它会随 importmap 自动发现挂到 `app/X-api`。
- `metainfer/tasks/X/static/X.css`（按需）：X 专属样式，配合 `extra_stylesheets=["X.css"]` 注入。参考 `calc_value/static/calc.css`。
- `routes.py`（可选）：X 专属的 FastAPI 路由（用 `from metainfer.web._helpers import task_or_404, state_dir_for, require_task_type`）。没有专属路由就不建。
- **无需修改** `metainfer/static/index.html`、`metainfer/static/styles.css`、`metainfer/static/components/`、`metainfer/static/views/`、`metainfer/web/app.py`、`metainfer/web/forms.py`、`metainfer/web/sse.py`、`metainfer/web/state_reader.py` —— 新加 task 包**全程只改自己目录内的文件**。
- **`metainfer/static/views/task-detail.js` 是 shell**：负责 header / 控制按钮 / Reset/Retrospective modal（懒加载）/ BudgetBar / 共享数据拉取，然后根据 `detail_view_module` 动态 import plugin 的 body 组件。加新 task 不需要碰这个文件。
- **`phases_module` 协议**：如果 task 想要 state-graph 渲染，task 包的 `phases.py` 必须导出 `graph_payload(current, last_outcome, last_label) -> dict`，返回 `{current, nodes, edges, active_edge, last_outcome, terminal_nodes, outcome_legend}`。公共 `state_reader.py` 不认识 task 类型，只调这个函数。

### 4. 测试（跟着 X 走）
- `metainfer/tasks/X/tests/`：X 的所有测试（orchestrator 逻辑 + web plugin 路由/QA）。包内 `__init__.py` 空文件，conftest 共享 fixture 从 `metainfer.testing` 引入。
- 共享 mock 工具统一从 `metainfer.testing` 引入：`from metainfer.testing import MockAgentManager, FakeStore, FakeLauncher, isolated_env`。**`metainfer/testing/` 只放跨任务通用的工具——禁止往里加任何只服务单个 task 的 helper**（曾经的 `calc_helpers.py` 已经搬回 `metainfer/tasks/calc_value/tests/_helpers.py`）。X 专属的测试 helper 放在 `metainfer/tasks/X/tests/_helpers.py`。

### 5. pytest 自动发现 —— 无需改 pytest.ini
`pytest.ini` 的 `testpaths = metainfer` 已经覆盖整个包，pytest 会自动递归扫到 `metainfer/tasks/X/tests/`。

### 验证
- `python -c "from metainfer.web.registry import all_plugins; import metainfer.tasks; print([p.type for p in all_plugins()])"` 应包含 `'<X-type-id>'`
- `python -c "from metainfer.orchestrator.tasks import all_tasks; import metainfer.tasks; print([p.task_type for p in all_tasks()])"` 也应包含 `'<X-type-id>'`（orchestrator 侧也注册了）
- 跑 `python -m pytest` 全绿（pytest.ini 自动扫到 X 的 tests/）
- 启动 WebUI，新建任务下拉里应出现 X（label/description 来自 plugin），详情页能渲染（专属前端模块从 `/static/plugins/<X-type-id>/...` 加载）
