---
name: gen-infer-framework
description: Build a model-specific LLM inference framework that exposes an OpenAI-compatible HTTP API. Use when the user wants to create a new inference server from scratch (e.g. "build an inference framework for Llama-3 8B on A100").
---

# gen-infer-framework

Build a minimal, model-specific inference framework that serves an
OpenAI-compatible HTTP API. Driven by the deterministic MetaInfer
orchestrator running a Plan → Implement → Test → Optimize loop.

This is a **pure Claude Code skill** — no plugin, no wrapper, no PATH
install. The skill directory carries its own scripts (`run.py`,
`metainfer/`) and knowledge base (`notebooks/`) alongside this SKILL.md.

## When to invoke

The user wants to build a new inference server (not port into an
existing framework, not optimize an existing kernel). Trigger phrases:

- "build an inference framework for <model>"
- "implement a serving stack for <model>"
- "write an inference server for <model> on <gpu>"

If the user wants to add a model to vLLM/SGLang/TensorRT-LLM, use
`port-model` instead. If they want to speed up an existing kernel, use
`opt-kernel`.

## What it does

1. **Check for an existing task** in `<cwd>/.metainfer/state/*/requirements.json`
   (see "Resume detection" below).
2. **Interview** the user from `questions.yaml` next to this SKILL.md
   (target model, size, hardware, dtype, batch, KV cache strategy, etc.)
   — skipped if resuming.
3. **Freeze** answers into `requirements.json` under
   `<cwd>/.metainfer/state/<task_id>/`.
4. **Hand off** to the orchestrator via the skill's launcher:
   `python <skill_dir>/run.py run <cwd>/.metainfer/state/<task_id>/requirements.json`
   (see "Invocation" below for how to resolve `<skill_dir>`).

The orchestrator then runs the 6-phase ABCDEF loop:

```
A: plan → B: implement (self-tested) → C: correctness test → D: review + retro
                                                            ├─ C ok  → E: perf test → F: perf plan → A (new iter)
                                                            └─ C fail → B (new iter, redo)
```

- D runs after every C, pass or fail (advisory — does not gate).
- E runs an independent, heavier perf benchmark (writes `perf_report.json`).
- F writes `perf_plan.md` (no code changes); next iteration's A executes the plan.
- C fail short-circuits E/F and routes back to B for a redo, after D captures lessons.

The WebUI on port 8765 shows live progress through all six phases.

### On-disk layout

Per-iteration **code** is written directly under the user's CWD so it is
visible and easy to browse. Tracking metadata + debug logs stay hidden
under `.metainfer/`:

```
<cwd>/
├── <task_id>/                       ← iteration code (visible)
│   ├── 001/
│   │   ├── plan.md
│   │   ├── serve.sh
│   │   └── *.py
│   ├── 002/                         ← next iter copies prev code forward
│   └── ...
└── .metainfer/
    ├── state/<task_id>/             ← run.json, iteration records, timeline
    └── logs/<task_id>/              ← per-iteration agent/oracle logs
        ├── 001/
        │   ├── *.prompt.txt
        │   ├── *.log
        │   ├── oracle-report.json
        │   ├── server.stderr.log
        │   └── server.stdout.log
        ├── 002/
        │   └── prev-iter/           ← snapshot of iter 001's diagnostics
        └── ...
```

The reviewer's `review.md` is also written under `.metainfer/logs/<task_id>/<NNN>/`
(not inside the code dir) so it stays with the rest of the tracking data.

## Resume detection

Before running the interview, scan `<cwd>/.metainfer/state/*/requirements.json`.

- **No files found** → fresh task. Run the interview normally; the
  orchestrator starts at A_plan, iteration 1.
- **One or more files found** → a previous run was interrupted. Tell the
  user which task_ids exist and ask whether to resume or start fresh:
  - **Resume `<task_id>`** → skip the interview entirely and run
    `python <skill_dir>/run.py run <cwd>/.metainfer/state/<task_id>/requirements.json`.
    The orchestrator will reconstruct progress from `state/` and
    `iterations/`, discard any iteration that was killed mid-flight, and
    pick up from there.
  - **Start fresh** → run the interview; the new `task_id` must differ
    from any existing one (otherwise you would overwrite it).

The orchestrator's crash-recovery rule is: an iteration is "complete"
only when its folder contains a `.metainfer-completed` sentinel, which
is written as the final step of the close path. On resume, if the
highest-numbered iteration folder lacks that sentinel, the orchestrator
deletes the folder (and its state record) and restarts that iteration
from its original `start_phase`. The first iteration's crash is handled
the same way — its folder is wiped and iteration 1 begins fresh.

## Correctness contract

The agent does NOT write its own test. The skill ships an **immutable
oracle** at `metainfer/oracles/infer_framework/` (next to this SKILL.md)
that:

1. starts `serve.sh <port>` on a free localhost port,
2. waits for `/v1/models` (or probes `/v1/chat/completions`),
3. sends fixed prompts via stdlib `urllib` (no third-party HTTP client),
4. dispatches a judge sub-agent to verdict each response,
5. kills the server and writes `oracle-report.json`.

The agent's only deliverable for the test phase is `serve.sh` — a bash
script that starts an OpenAI-compatible HTTP server in the foreground
on `$1` port. See `metainfer/prompts/__init__.py` for the full contract.

## Invocation

### Resolving `<skill_dir>` (IMPORTANT)

This SKILL.md lives at `<skill_dir>/SKILL.md`. Resolve `<skill_dir>`
from this file's absolute path:

- If you found this file via Claude Code's skill discovery
  (`~/.claude/skills/gen-infer-framework/SKILL.md` is a symlink to here),
  read the symlink target to get the real path. From Python:
  `Path("<path_to_this_SKILL.md>").resolve().parent`.
- If you found it by browsing, use that directory directly.

Then invoke the orchestrator via the launcher:

```bash
python <skill_dir>/run.py run <cwd>/.metainfer/state/<task_id>/requirements.json
```

`run.py` is self-contained: it inserts `<skill_dir>` at the front of
`sys.path` so `import metainfer` resolves to the package next to it,
then delegates to `metainfer.cli.main`. No `PYTHONPATH`, no wrapper, no
global install, no `find /`.

DO NOT:
- Search the user's CWD or filesystem for `run.py` or `metainfer/` —
  they live in this skill's directory, never co-located with the project
  being worked on.
- Try to "create" `run.py` if it's missing — that means the install is
  broken; tell the user instead.
- Call `python …/metainfer/cli.py` directly — `cli.py` uses relative
  imports (`from .orchestrator import …`) that only work in module mode
  via `run.py`'s sys.path bootstrap.

### Long-running process notes

- The orchestrator is a long-running foreground Python process. Always
  launch with `run_in_background: true` — you'll get a completion
  notification. Meanwhile read the output file or poll the WebUI on
  port 8765.
- DO NOT use `sleep N && head …` to wait for output — sleeps ≥ 2s are
  blocked on the host. The background launch + completion notification
  pattern replaces that entirely.

### Sub-agent permissions (important)

Sub-agents are launched non-interactively (`ccb -p ...` with the prompt
piped via stdin). They **cannot answer permission prompts**, so the
permission mode must accept tool uses silently. The orchestrator's
default is `--permission-mode bypassPermissions`, which skips ALL
permission checks AND the LLM-based Bash safety classifier. The latter
matters: under `auto`, that classifier has been observed to deny trusted
scripts like `bash perf.sh` 60+ times in a single perf-test run,
wasting 7 minutes before the agent gave up and falsely reported success
(see the qwen3-8b iter3 incident). The orchestrator's sub-agents run
trusted code in a controlled working dir, so the classifier adds false
positives without real safety value.

Available modes:

| Mode | Behavior | Works as root? |
|---|---|---|
| `bypassPermissions` (default) | Skip ALL permission checks AND the Bash safety classifier | Yes — `_build_env` sets `IS_SANDBOX=1` to bypass ccb's root/EUID=0 hard-exit |
| `auto` | Accept all tool uses, but the Bash classifier still runs and can deny commands it deems risky | Yes |
| `acceptEdits` | Auto-accept Edit/Write only; Bash still prompts → hangs sub-agents | Yes, but limited |
| `plan` / `default` | Read-only / prompts for everything | Hangs sub-agents |

Override (rarely needed):

```bash
python <skill_dir>/run.py run requirements.json --permission-mode auto
# or:
METAINFER_PERMISSION_MODE=auto python <skill_dir>/run.py run requirements.json
```

Precedence: `--permission-mode` flag > `METAINFER_PERMISSION_MODE` env >
`bypassPermissions` default.

## Knowledge base

Reference designs, model specifics, and postmortems live in
`notebooks/` next to this SKILL.md. Sub-agents are told to consult it.
