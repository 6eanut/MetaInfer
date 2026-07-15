# Contributing to MetaInfer

Thanks for contributing. This guide covers setup, architecture, conventions,
and the pull request workflow.

## Development setup

```bash
git clone https://github.com/MetaInfer/MetaInfer.git
cd MetaInfer
pip install -e ".[dev]"
```

Installs two console scripts:

- `metainfer-web` — the WebUI server (long-lived main process)
- `metainfer-orchestrator` — per-task orchestrator (spawned by the WebUI)

> Alternatively, use `./serve.py` which adds the repo root to `sys.path`
> without installing. But `pip install -e .` is recommended for development
> — it gives you the console scripts and keeps the package editable.

## Running tests

```bash
python -m pytest                  # full suite
python -m pytest -q               # quiet
python -m pytest metainfer/tasks/<pkg>/tests/   # single package
```

All agent operations must use mocks. CI runs the full suite on every push.

## Verifying plugin registration

```bash
python -c "from metainfer.server.registry import all_plugins; import metainfer.tasks; print([p.type for p in all_plugins()])"
python -c "from metainfer.orchestrator.tasks import all_tasks; import metainfer.tasks; print([p.task_type for p in all_tasks()])"
```

## Architecture

### Process model

- **WebUI** is the only long-lived process. It owns the FastAPI server,
  SSE broadcast loop, and `LocalLauncher` for spawning orchestrators.
- **Orchestrator** is a short-lived child process. One per task. It runs
  the task pipeline, shells out to `ccb` for sub-agents, writes state
  files, then exits. Crashes take down only that task.
- **No IPC.** The two processes communicate entirely through the
  filesystem. The WebUI watches mtimes and pushes diffs to the browser.

### Two-layer framework

```
metainfer/server/          ← WebUI framework (FastAPI routes, SSE, registry)
metainfer/orchestrator/    ← orchestrator framework (StateStore, agent pool, …)
metainfer/tasks/<pkg>/     ← task packages (each: pipeline + web plugin + static)
```

Every task package is a peer plugin. Neither `server/` nor `orchestrator/`
contains task-specific code. The `sys_shell` package itself follows the
same plugin conventions as task packages — it's just mounted at
`/api/sys-shell` without a `{task_id}` segment.

### Runtime directory layout

```text
$METAINFER_ROOT/                    (defaults to cwd)
└── nodes/
    └── <node_id>/                  ($METAINFER_NODE_ID or hostname)
        ├── workspaces/
        │   └── <task_id>/          ← generated artifacts (structure owned by task)
        └── .metainfer/             ← metadata + logs
            ├── registry.json       global task list + workspace_dir refs
            ├── registry.lock       flock for atomic updates
            ├── runtime.json        live WebUI + orchestrator PIDs
            ├── runtime.lock
            └── tasks/<task_id>/
                ├── requirements.json   {"task_id", "task_type", "created_at", "form": {...}}
                ├── run.json            RunStatus: phase, iteration, final_status
                ├── timeline.jsonl      append-only: {"ts", "type", "payload"}
                ├── orchestrator.{pid,log}
                ├── agents.json         latest SubAgentManager snapshot
                ├── token_budget.json
                ├── iterations/<NNN>.json
                └── logs/<NNN>/         per-iteration prompt/oracle/server logs
```

Key invariants:

- `workspace_dir` is written by the orchestrator, read by the user.
- `.metainfer/` is co-written by the orchestrator and the WebUI.
- Task reset clears both directories: `reset_state_dir(state_dir, workspace_dir, task_id, task_type)`.
- The orchestrator CLI must accept `run <req.json> --state-dir … --workspace-dir …`.

`METAINFER_ROOT` overrides the root (for shared NFS mounts);
`METAINFER_NODE_ID` overrides the node id (for multi-node deployments).

## Adding a new task type

The canonical starting point is `metainfer/tasks/example/` — a fully
annotated skeleton with all `register()` calls commented out.

```bash
cp -r metainfer/tasks/example metainfer/tasks/<your_task>
```

Then:

1. Globally replace `X-type-id` / `X` / `example` with your identifiers
2. Uncomment `register()` calls in:
   - `__init__.py` (triggers import of both plugins)
   - `orchestrator/plugin.py` (`TaskPlugin`)
   - `server/plugin.py` (`WebPlugin`)
3. Implement `orchestrator/pipeline.py` iteration logic
4. Write tests

**No other files in the repo need modification.** Auto-discovery in
`metainfer/tasks/__init__.py` uses `pkgutil.iter_modules` and picks
up new packages automatically (skips underscore-prefixed names).

### Contract checklist

| Layer | File | Constraint |
|---|---|---|
| File | `timeline.jsonl` | `{"ts": float, "type": str, "payload": dict}` — shell never interprets `type` |
| File | `requirements.json` | `task_type` must match `TaskPlugin.task_type` and `WebPlugin.type` |
| File | `run.json` | Shell reads 11 fields (see `state_reader.read_run` defaults); task string is opaque |
| CLI | orchestrator argv | Must accept `run <req.json> --state-dir … --workspace-dir …` |
| Web | `build_router(plugin)` | Returns `APIRouter` with relative paths; shell mounts at `/api/{type}/{task_id}` |
| Web | `_state_readers.py` | Task-specific readers only — never add task logic to `server/state_reader.py` |
| QA | `qa_config.resolve_target` | `(state_dir, payload) -> {events_file, target_workdir, target_label}` |

### URL architecture

- `sys-shell` → `/api/sys-shell` (no `{task_id}`)
- Task plugin → `/api/{type}/{task_id}`
- Static assets → `/static/plugins/{type}/`

### Form schema (`form.yaml`)

Each entry describes one form field:

```yaml
- key: my_field
  question: "What value?"    # shown as help text
  header: "Field"            # short label, ≤12 chars
  required: true
  form: text                 # text | textarea | select | multiselect | file | number
  multi: false               # for selects: allow multiple values
  options:                   # for select / multiselect
    - label: "A"
      description: "What choosing A means"
```

Add `override_component: <name>` to delegate rendering to a task-specific
widget in the frontend form renderer.

## Configuring the orchestrator

### Claude Code binary

Sub-agents shell out to `ccb` by default. Override via:

```bash
METAINFER_CLAUDE_BIN=/usr/local/bin/claude metainfer-web
```

Or per-task through the **Extra orchestrator args** field in the new-task form.

Precedence: `--claude-bin` flag (extra args) > `METAINFER_CLAUDE_BIN` env > `ccb` default.

### Permission mode

Sub-agents run in `bypassPermissions` mode with `IS_SANDBOX=1`.
Override with `METAINFER_PERMISSION_MODE` for stricter control.

### Driving the orchestrator directly (debugging)

```bash
metainfer-orchestrator run requirements.json \
  --state-dir /tmp/task-debug \
  --workspace-dir /tmp/task-workspace
```

This bypasses the registry (the WebUI won't show the task) but writes
the same observable files — useful for debugging pipeline logic.

## Design principles

- **Determinism over agency.** The orchestrator is plain Python with
  a fixed control flow. Sub-agents only do work (generating code,
  analyzing output, writing tests), never decide what runs next.
- **File-based state.** JSON on disk under `nodes/<node>/.metainfer/`.
  No in-memory state shared between WebUI and orchestrator. WebUI
  restarts pick up exactly where the previous one left off.
- **Self-contained packages.** Each task package owns its pipeline,
  routes, static assets, and tests. Shared layers (`server/`,
  `orchestrator/`) contain no task-specific code.
- **No-build frontend.** Preact + HTM via importmap. The browser loads
  ES modules directly — no transpiler, no bundler. Vendor files are
  local for offline / air-gapped use.
- **Immutable oracles.** Correctness checks live inside the task
  package, outside iteration directories. The agent cannot edit its
  own grading rubric.
- **Iteration isolation.** Each iteration copies the previous one's
  directory, so a bad iteration never poisons a good one.
- **Multi-node by design.** `nodes/<node_id>/` layout isolates writes
  per node. Multiple machines on a shared filesystem never collide.
  Central controllers can scan `nodes/*/` for global visibility.

## Submitting changes

1. Open an issue describing the bug or feature first
2. Branch from `master`:
   ```bash
   git checkout -b feat/my-feature master
   ```
3. Write tests — every change needs coverage. Mock agent operations.
4. Run `python -m pytest` and ensure everything passes
5. Commit with a meaningful message following the existing convention
   (see `git log --oneline` for examples)
6. Push and open a PR against `master`

PRs that add or modify task packages should include the verification
commands from the [Verifying plugin registration](#verifying-plugin-registration)
section in the PR description.
