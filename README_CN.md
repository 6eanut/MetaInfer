<p align="center">
  <img src="https://raw.githubusercontent.com/MetaInfer/MetaInfer/main/docs/logo.png" alt="MetaInfer" width="200" onerror="this.style.display='none'">
</p>

<h1 align="center">MetaInfer</h1>

<p align="center">
  <em>LLM 驱动的推理工程 —— 确定性编排，不可变 Oracle。</em>
</p>

<p align="center">
  <a href="https://github.com/MetaInfer/MetaInfer/actions/workflows/ci.yml">
    <img src="https://github.com/MetaInfer/MetaInfer/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/metainfer">
    <img src="https://img.shields.io/pypi/v/metainfer?color=blue" alt="PyPI">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-%3E%3D3.9-blue" alt="Python">
  </a>
  <a href="https://github.com/MetaInfer/MetaInfer/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
</p>

<p align="center">
  <a href="https://star-history.com/#MetaInfer/MetaInfer&Date">
    <img src="https://api.star-history.com/svg?repos=MetaInfer/MetaInfer&type=Date" alt="Star History" width="600">
  </a>
</p>

---

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 这是什么？

MetaInfer 是一个轻量级 Web 应用，运行 LLM 驱动的推理工程任务。
你描述想要什么（模型专属推理服务器、理论 FLOPs 分析），MetaInfer
就启动一个确定性 Python 编排器，驱策 Claude Code 子 agent
按结构化流水线完成工作。

- **WebUI** — FastAPI 长驻主进程 + Preact SPA 前端
- **编排器** — 每任务一个短生命周期子进程；单个任务崩溃不会拖垮面板
- **文件系统即状态** — 无数据库、无消息队列；所有状态都是磁盘上的 JSON

### 内置任务类型

| 类型 | 说明 |
|---|---|
| **`gen-infer-framework`** | 构建模型专属推理服务器，提供 OpenAI 兼容 HTTP API。不可变 oracle 启动 `serve.sh`，发送固定 prompt，派发 LLM 裁判判定正确性。 |
| **`calc-theoretical-value`** | 计算 LLM 单次前向传播的理论 FLOPs 和显存带宽。完全只读的确定性流水线：模型检查 → 显存建模 → 计算图 → 可视化。 |
| **`example`** | 构建新任务类型的规范骨架。复制、重命名、取消 `register()` 注释、实现流水线——不改动任何共享代码。 |

## 快速开始

```bash
git clone https://github.com/MetaInfer/MetaInfer.git
cd MetaInfer
pip install -r requirements.txt
./serve.py
```

打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)，点击 **+ New Task**，
选择类型，填写表单，面板随即实时更新。

```bash
# 其他启动方式
./serve.py --host 0.0.0.0 --port 9000
METAINFER_PORT=9000 ./serve.py
python -m metainfer.server.app
```

> **无需安装。** `serve.py` 自动将仓库根目录加入 `sys.path`。
> 仅当需要 `metainfer-web` 和 `metainfer-orchestrator` 命令时才执行
> `pip install -e .`。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│  浏览器 (Preact + HTM, 无构建步骤)                        │
│  任务标签页 · 状态图 · 迭代记录 · Agent 面板               │
└────────────────────┬────────────────────────────────────┘
                     │  HTTP + SSE
┌────────────────────┴────────────────────────────────────┐
│  WebUI 服务端 (FastAPI, 主进程)                           │
│  表单 Schema · 任务注册表 · 进程启停                        │
│  SSE: 文件监控 → 浏览器推送                               │
└───┬──────────────────────────────────────┬──────────────┘
    │ 子进程 (LocalLauncher)               │ (未来) HTTP
┌───┴───────────────┐              ┌──────┴──────────────┐
│ 编排器 #1          │   ...        │ 远程节点               │
│ 每任务子进程        │              │ RemoteLauncher        │
│ 任务流水线          │              │ over HTTPS            │
│ Sub-agent 管理器    │              └─────────────────────┘
└───────────────────┘
```

> **Launcher Protocol** 是多机协同的扩展点。将 `LocalLauncher` 替换为
> `RemoteLauncher` 即可通过 HTTPS 启停远程编排器——无需修改其他代码。

## 工作原理

每个任务在当前节点下占用**两个并列目录**：

```text
$METAINFER_ROOT/                  (默认 cwd)
└── nodes/
    └── <node_id>/                (主机名；$METAINFER_NODE_ID 可覆盖)
        ├── workspaces/
        │   └── <task_id>/        ← 迭代产物
        └── .metainfer/
            ├── registry.json     ← 全局任务列表
            └── tasks/<task_id>/  ← 元数据、日志、时间线、迭代记录
```

WebUI 写入 `requirements.json`、启动编排器，然后**监控文件变更**——
所有面板数据均来源于磁盘 JSON。无内存状态，无 IPC。重启 WebUI
即可无缝恢复。

## 扩展

### 新建任务类型

复制骨架即可。骨架包含完整注释——表单 Schema、流水线、Web 路由、
QA 端点、静态资源、测试。

```bash
cp -r metainfer/tasks/example metainfer/tasks/<your_task>
# 1. 将 X-type-id / X / example 替换为你的标识
# 2. 取消 __init__.py、orchestrator/plugin.py、server/plugin.py 中 register() 的注释
# 3. 实现 orchestrator/pipeline.py
# 4. 编写测试
```

**仓库中其他文件不需要任何改动。** 完整注释骨架见
`metainfer/tasks/example/`，契约约束详见 `CLAUDE.md`。

### 新增表单字段

编辑你任务的 `form.yaml`：

```yaml
- key: model_path
  question: "模型权重路径？"
  header: "模型"
  required: true
  form: text              # text | textarea | select | multiselect | file | number
  options:                # 仅 select / multiselect 使用
    - label: "选项 A"
      description: "含义说明"
```

### 知识库

将 markdown 文档放入 `metainfer/tasks/<pkg>/notebooks/`。
Prompt 模板已引用该目录——无需代码改动。

## 仓库结构

```text
MetaInfer/
├── pyproject.toml
├── serve.py
├── README.md / README_CN.md
│
├── metainfer/
│   ├── server/                WebUI 后端 (FastAPI)
│   │   ├── app.py             create_app、插件路由、静态挂载
│   │   ├── launcher.py        LocalLauncher (Protocol → RemoteLauncher 扩展)
│   │   ├── registry.py        registry.json CRUD (fcntl.flock)
│   │   ├── forms.py           form.yaml → 前端 Schema
│   │   ├── state_reader.py    文件 → JSON (仅 shell 层字段)
│   │   ├── sse.py             mtime 监控 → SSE 广播
│   │   ├── reconcile.py       启动时清理孤儿编排器进程
│   │   └── qa.py / qa_routes.py  QA 端点框架
│   │
│   ├── orchestrator/          每任务子进程框架
│   │   ├── state.py           跨进程安全 JSON StateStore
│   │   ├── subagent_manager.py  Claude Code 子进程生命周期
│   │   ├── agent_pool.py      多 agent 并发池
│   │   ├── token_budget.py    token / 费用追踪
│   │   ├── gpu_preflight.py   oracle 运行前 GPU 显存清理
│   │   └── tasks/             TaskPlugin 自动发现
│   │
│   └── tasks/                 每种类型一个 task 包
│       ├── example/           规范骨架（注释状态，可直接复制）
│       ├── sys_shell/         shell UI + 任务生命周期 API
│       ├── gen_infer_framework/
│       └── calc_value/
```

## License

MIT

架构细节、设计理念和新任务类型添加方法见 [CONTRIBUTING.md](CONTRIBUTING.md)。
