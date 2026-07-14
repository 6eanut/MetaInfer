# MetaInfer

MetaInfer is a lightweight **Web application** that orchestrates
LLM-driven inference engineering tasks. Each task spawns a per-task
deterministic Python orchestrator subprocess that drives Claude Code
sub-agents through an ABCDEF loop (Plan → Implement → Test → Review →
Perf Test → Perf Plan). The WebUI is the long-lived main process;
orchestrators come and go per task.

## What it does

Three task types ship out of the box. Each runs the same ABCDEF state
machine; only the correctness check differs:

| Task type | When to use | Correctness check |
|---|---|---|
| **`gen-infer-framework`** | Build a model-specific inference server with an OpenAI-compatible HTTP API | Immutable oracle boots `serve.sh`, sends fixed prompts, dispatches an LLM-judge sub-agent to verdict each response |
| **`port-model`** | Port a new model into vLLM / SGLang / TensorRT-LLM | Agent-authored `test.sh` (JSON contract) |
| **`opt-kernel`** | Optimize an existing GPU kernel for a target shape & GPU | Agent-authored `test.sh` with perf metrics |

```
┌──────────────────────────────────────────────────────────┐
│  Browser (Preact SPA, no build step)                     │
│   • task tabs + new-task overlay                          │
│   • live: state graph · iterations · charts · agents     │
└─────────────────────┬────────────────────────────────────┘
                      │ HTTP + SSE
┌─────────────────────┴────────────────────────────────────┐
│  WebUI Server (FastAPI, 主进程)                           │
│   • form schema (tasks/*.yaml)                           │
│   • task registry (~/.metainfer/registry.json)           │
│   • LocalLauncher: spawn/kill orchestrator subprocesses  │
│   • SSE: file mtime watch → browser refresh              │
└──────┬───────────────────────────────────┬───────────────┘
       │ subprocess                        │ (future) HTTP
┌──────┴──────────────┐               ┌────┴──────────────┐
│ Orchestrator #1     │   ...         │ Remote Node       │
│ (per-task child)    │               │  HTTP API         │
│  ABCDEF pipeline    │               └───────────────────┘
│  subagent manager   │
│  ccb -p sub-agents  │
└─────────────────────┘
```

The Launcher Protocol is the seam for future multi-machine
collaboration — a `RemoteLauncher` will implement the same interface
over HTTPS, so the WebUI manages remote orchestrators without code
changes.

## Quick start

```bash
git clone https://github.com/<your-org>/MetaInfer.git
cd MetaInfer
pip install -r requirements.txt   # fastapi, uvicorn, pyyaml（纯 Python）
./serve.py                        # 启动 WebUI，前台运行
```

打开浏览器访问 <http://127.0.0.1:8765>。点 **+ New Task**，选任务类型，
填表单提交，看面板实时更新。

**其他启动方式**（等价）：

```bash
./serve.py --host 0.0.0.0 --port 9000   # 自定义监听
METAINFER_PORT=9000 ./serve.py           # 用环境变量
python -m metainfer.web.app              # 不用 serve.py 包装
metainfer-web                            # 需要先 pip install -e .
```

> 不需要 `pip install`：`serve.py` 自己会把仓库根目录加到 `sys.path`，
> 直接 `import metainfer`。装包只是为了拿到 `metainfer-web` 命令。

## Install（可选，推荐用于生产/常用）

```bash
pip install -e .
```

That installs two entry points:

- `metainfer-web` — the WebUI server (the long-lived main process)
- `metainfer-orchestrator` — per-task orchestrator (spawned by the
  WebUI; you don't usually invoke this directly)

Runtime deps are pure Python (`fastapi`, `uvicorn`, `pyyaml`); no
compiled extensions. The only external binary is `ccb` (Claude Code),
which the orchestrator shells out to for sub-agents.

## Usage

Start the WebUI（任选一种，详见 Quick start）：

```bash
./serve.py                        # 推荐：仓库根目录直接启动
metainfer-web                     # 装包后可用
```

Open the URL, click **+ New Task**, pick a task type, fill the form.
On submit the WebUI:

1. Writes `requirements.json` to `~/.metainfer/tasks/<id>/`
2. Adds the task to the registry (`~/.metainfer/registry.json`)
3. Spawns `python -m metainfer.orchestrator.cli run <req.json>
   --state-dir ~/.metainfer/tasks/<id>/` as a subprocess
4. Watches the task's state files; SSE pushes diffs to every open tab

Switch between tasks via the tab strip. Click an iteration row to read
its retrospective (the orchestrator's postmortem markdown). Kill or
restart a task from the header buttons.

### Configuring the Claude Code binary

Sub-agents shell out to `ccb` by default. Override per-task with the
**Extra orchestrator args** field in the new-task form, or set
`METAINFER_CLAUDE_BIN` in the environment:

```bash
METAINFER_CLAUDE_BIN=/usr/local/bin/claude metainfer-web
```

Precedence: `--claude-bin` flag (passed via extra args) >
`METAINFER_CLAUDE_BIN` env var > `ccb` default.

Sub-agents run in `bypassPermissions` mode with `IS_SANDBOX=1` so they
don't hang on permission prompts. Override with
`METAINFER_PERMISSION_MODE` if you need stricter control.

### Driving the orchestrator directly

The WebUI is the intended entry point, but for debugging you can spawn
an orchestrator by hand:

```bash
metainfer-orchestrator run requirements.json --state-dir /tmp/task-debug
```

This skips the registry — the WebUI won't show the task — but writes
the same observable files (`run.json`, `iterations/`, `agents.json`,
`timeline.jsonl`).

## Repository layout

```
MetaInfer/
├── pyproject.toml                  # package + console scripts
├── README.md
│
├── metainfer/                      # top-level Python package
│   ├── orchestrator/               # per-task subprocess
│   │   ├── cli.py                  # entry: metainfer-orchestrator run <req>
│   │   ├── orchestrator.py         # run_with_requirements(): spawn → run → exit
│   │   ├── pipeline.py             # ABCDEF main loop
│   │   ├── phases.py               # TRANSITIONS table (data) + graph helpers
│   │   ├── state.py                # cross-process-safe JSON state store
│   │   ├── subagent_manager.py     # Claude Code subprocess lifecycle
│   │   ├── iteration.py            # numbered iteration folder management
│   │   ├── prompts/                # prompt templates for A/B/C/D/E/F agents
│   │   └── oracles/                # immutable correctness oracles
│   │
│   ├── web/                        # WebUI backend
│   │   ├── app.py                  # FastAPI: routes + SSE + static mount
│   │   ├── launcher.py             # LocalLauncher (Protocol = seam for remote)
│   │   ├── tasks.py                # registry.json CRUD (fcntl.flock-guarded)
│   │   ├── forms.py                # tasks/*.yaml → frontend form schema
│   │   ├── state_reader.py         # read-only file → JSON for every panel
│   │   ├── paths.py                # ~/.metainfer/{registry,tasks}/...
│   │   └── sse.py                  # polling file watcher → SSE broadcast
│   │
│   └── static/                     # frontend (Preact + HTM, no build step)
│       ├── index.html              # importmap for preact/htm/chart.js/marked
│       ├── main.js                 # SPA entry: shell + tabstrip + SSE
│       ├── components/             # state-graph, charts, agents, timeline, form…
│       ├── views/                  # task-detail, new-task
│       ├── vendor/                 # preact/htm/chart.js/marked (local, no CDN)
│       └── styles.css              # design tokens + components
│
├── tasks/                          # LEGACY stub form schemas (opt-kernel.yaml, port-model.yaml)
│
└── legacy/                         # archived pre-refactor skill bundle
```

Note: each task package's knowledge base lives inside its package —
e.g. `metainfer/tasks/gen_infer_framework/notebooks/` — not at the
top level. Form schemas for full task types live at
`metainfer/tasks/<pkg>/form.yaml`; legacy stub types still ship
`tasks/<type>.yaml`.
└── legacy/                         # archived pre-refactor skill bundle
```

### Per-task state directory

Every task gets its own self-contained directory under
`~/.metainfer/tasks/<id>/`:

```
~/.metainfer/tasks/<id>/
├── requirements.json     # frozen inputs from the form
├── orchestrator.pid      # lifecycle markers (pid, started_at, finished_at)
├── orchestrator.log      # stdout+stderr of the subprocess
├── run.json              # RunStatus: phase, iteration, final_status
├── timeline.jsonl        # append-only event stream
├── agents.json           # latest SubAgentManager snapshot
├── iterations/           # one record per iteration
│   ├── 001.json          # status, goal, perf, retrospective path, …
│   └── …
├── code/                 # iteration N's working tree (visible source)
└── logs/                 # per-iteration sub-agent output
```

All WebUI panels derive from these files. There's no in-memory state
shared between the WebUI and the orchestrator — they communicate
solely through the filesystem. This is what lets a WebUI restart pick
up exactly where the old one left off.

## Extending

### New form fields

Edit `tasks/<task_type>.yaml`. Each entry is one field:

```yaml
- key: my_new_field
  question: "What value for X?"   # shown as help text
  header: "X"                     # short label
  required: true
  form: text                      # text|textarea|select|multiselect|file|number
  # OR use multi/options to infer the widget:
  multi: false
  options:
    - label: "A"
      description: "implies ..."
```

Add `override_component: <name>` to delegate to a task-specific widget
(future task-specific widgets register in
`metainfer/static/components/form-renderer.js`).

### New task types

1. Add `tasks/<new-task>.yaml` (form schema).
2. Register the task type in `metainfer/orchestrator/paths.py:TASK_TYPES`
   and add metadata to `metainfer/web/forms.py:TASK_TYPE_META`.
3. If correctness needs an objective check, add an oracle under
   `metainfer/orchestrator/oracles/<new-task>/` and register it in
   `metainfer/orchestrator/oracles/__init__.py`.
4. The transition table in `phases.py` automatically routes
   `gen-infer-framework`-style tasks to the oracle path; otherwise the
   agent writes its own `test.sh`.

### Knowledge base

Drop markdown files into `metainfer/tasks/<task_pkg>/notebooks/<topic>/`.
Prompt templates already tell sub-agents to consult the `notebooks/`
path passed to them — no code change required. Each task package owns
its own knowledge base.
Keep each file short: one concept, one example, one gotcha list.

## Design notes

- **WebUI as main process.** The server is the only long-lived thing.
  Per-task orchestrators run as subprocesses; a crash takes down only
  that task, never the dashboard. Future multi-machine work adds a
  `RemoteLauncher` that dispatches spawn/kill over HTTPS to a remote
  node running its own LocalLauncher.
- **Determinism over agency.** Long-running, multi-day tasks diverge
  if the LLM drives control flow. The orchestrator is plain Python;
  sub-agents only do work, never decide what runs next.
- **Data-driven state machine.** `phases.py:TRANSITIONS` is the single
  source of truth for the ABCDEF graph; the frontend auto-derives its
  flow diagram from this table, so adding an edge or phase needs no UI
  code.
- **File-based state.** Everything goes through `StateStore` JSON
  files so the WebUI can observe state from a separate process without
  IPC. The same files survive WebUI restarts.
- **No-build frontend.** Preact + HTM via an importmap. The browser
  loads ES modules directly; no transpile step, no bundler. Vendor
  files are local (not CDN) so the app works offline / behind a proxy.
- **Iteration folders.** Each iteration gets a fresh copy of the
  previous iteration's directory, so a bad iteration never poisons a
  good one.
- **Immutable oracles.** For tasks where the agent would otherwise
  grade its own homework (e.g. inference-framework correctness), the
  oracle ships with the package, lives outside the iteration directory,
  and is the source of truth for pass/fail.
