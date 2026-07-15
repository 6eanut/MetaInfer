<p align="center">
  <img src="https://raw.githubusercontent.com/MetaInfer/MetaInfer/main/docs/logo.png" alt="MetaInfer" width="200" onerror="this.style.display='none'">
</p>

<h1 align="center">MetaInfer</h1>

<p align="center">
  <em>LLM-driven inference engineering — deterministic orchestration, immutable oracles.</em>
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
  <a href="README_CN.md">中文文档</a>
</p>

---

## What is MetaInfer?

MetaInfer is a lightweight web application that runs LLM-driven inference
engineering tasks. You describe what you want (a model-specific inference
server, a theoretical FLOPs analysis), and MetaInfer spawns a
deterministic Python orchestrator that drives Claude Code sub-agents
through a structured pipeline.

- **WebUI** — long-lived FastAPI main process with a Preact SPA frontend
- **Orchestrator** — one short-lived subprocess per task; crashes never
  take down the dashboard
- **File-system state** — no database, no message queue; everything is
  observable JSON on disk

### Built-in task types

| Type | What it does |
|---|---|
| **`gen-infer-framework`** | Build a model-specific inference server with an OpenAI-compatible HTTP API. An immutable oracle boots `serve.sh`, sends fixed prompts, and dispatches an LLM judge to verdict correctness. |
| **`calc-theoretical-value`** | Compute theoretical FLOPs and memory-traffic for an LLM forward pass. Fully read-only deterministic pipeline: model inspection → memory modeling → compute graph → visualization. |
| **`example`** | Canonical skeleton for building new task types. Copy, rename, uncomment `register()`, implement your pipeline — no shared code touched. |

## Quick start

```bash
git clone https://github.com/MetaInfer/MetaInfer.git
cd MetaInfer
pip install -r requirements.txt
./serve.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), click **+ New Task**,
pick a type, fill the form, and watch the dashboard come alive.

```bash
# Other ways to start
./serve.py --host 0.0.0.0 --port 9000
METAINFER_PORT=9000 ./serve.py
python -m metainfer.server.app
```

> **No install required.** `serve.py` adds the repo root to `sys.path`.
> Install `pip install -e .` only if you want the `metainfer-web` and
> `metainfer-orchestrator` console scripts.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser (Preact + HTM, no build step)                   │
│  Task tabs · live state graph · iterations · agents     │
└────────────────────┬────────────────────────────────────┘
                     │  HTTP + SSE
┌────────────────────┴────────────────────────────────────┐
│  WebUI Server (FastAPI, main process)                    │
│  Form schemas · task registry · process lifecycle        │
│  SSE: file watcher → browser push                        │
└───┬──────────────────────────────────────┬──────────────┘
    │ subprocess (LocalLauncher)           │ (future) HTTP
┌───┴───────────────┐              ┌──────┴──────────────┐
│ Orchestrator #1   │   ...        │ Remote Node          │
│ Per-task child    │              │ RemoteLauncher       │
│ Task pipeline     │              │ over HTTPS           │
│ Sub-agent manager │              └─────────────────────┘
└───────────────────┘
```

> **The Launcher Protocol** is the extension point for multi-machine
> collaboration. Swap `LocalLauncher` for a `RemoteLauncher` that
> dispatches spawn/kill over HTTPS — no other code changes.

## How it works

Each task gets **two parallel directories** under the current node:

```text
$METAINFER_ROOT/                  (defaults to cwd)
└── nodes/
    └── <node_id>/                (hostname; override with $METAINFER_NODE_ID)
        ├── workspaces/
        │   └── <task_id>/        ← generated artifacts
        └── .metainfer/
            ├── registry.json     ← global task list
            └── tasks/<task_id>/  ← metadata, logs, timeline, iter records
```

The WebUI writes `requirements.json`, spawns the orchestrator, then
**watches files** for changes — all panels derive from JSON on disk.
No in-memory state, no IPC. Restart the WebUI and it picks up exactly
where it left off.

## Extending

### New task type

Copy the skeleton. It's fully annotated — form schema, pipeline,
web routes, QA endpoints, static assets, tests.

```bash
cp -r metainfer/tasks/example metainfer/tasks/<your_task>
# 1. Replace X-type-id / X / example with your own ids
# 2. Uncomment register() in __init__.py, orchestrator/plugin.py, server/plugin.py
# 3. Implement orchestrator/pipeline.py
# 4. Add tests
```

**Nothing else in the repo needs to change.** See `metainfer/tasks/example/`
for the complete annotated skeleton and `CLAUDE.md` for contract details.

### New form fields

Edit your task's `form.yaml`:

```yaml
- key: model_path
  question: "Path to model weights?"
  header: "Model"
  required: true
  form: text              # text | textarea | select | multiselect | file | number
  options:                # only for select / multiselect
    - label: "Option A"
      description: "What it means"
```

### Knowledge base

Drop markdown files into `metainfer/tasks/<pkg>/notebooks/`. Prompt
templates already reference the notebooks directory — no code changes
needed.

## Repository

```text
MetaInfer/
├── pyproject.toml
├── serve.py
├── README.md / README_CN.md
│
├── metainfer/
│   ├── server/                WebUI backend (FastAPI)
│   │   ├── app.py             create_app, plugin routing, static mount
│   │   ├── launcher.py        LocalLauncher (Protocol → RemoteLauncher extension)
│   │   ├── registry.py        registry.json CRUD with fcntl.flock
│   │   ├── forms.py           form.yaml → frontend schema
│   │   ├── state_reader.py    file → JSON (shell-level fields only)
│   │   ├── sse.py             mtime watcher → SSE broadcast
│   │   ├── reconcile.py       orphan orchestrator cleanup on startup
│   │   └── qa.py / qa_routes.py  QA endpoint framework
│   │
│   ├── orchestrator/          per-task subprocess framework
│   │   ├── state.py           cross-process-safe JSON StateStore
│   │   ├── subagent_manager.py  Claude Code subprocess lifecycle
│   │   ├── agent_pool.py      multi-agent concurrency
│   │   ├── token_budget.py    token / cost tracking
│   │   ├── gpu_preflight.py   GPU VRAM cleanup before oracle runs
│   │   └── tasks/             TaskPlugin auto-discovery
│   │
│   └── tasks/                 one package per task type
│       ├── example/           canonical skeleton (commented-out, ready to copy)
│       ├── sys_shell/         shell UI + task lifecycle API
│       ├── gen_infer_framework/
│       └── calc_value/
```

## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture details, design
principles, and how to add new task types.
