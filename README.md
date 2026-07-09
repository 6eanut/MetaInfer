# MetaInfer

MetaInfer is a bundle of **pure Claude Code skills** for LLM inference
engineering. Each skill is a self-contained directory carrying its own
SKILL.md, scripts, and knowledge base. No plugin, no wrapper, no global
install — the skill IS the unit.

## What it does

Three task types, each an autonomous ABCDEF loop (Plan → Implement →
Test → Review → Perf Test → Perf Plan) driven by a deterministic Python
orchestrator that spawns Claude Code sub-agents:

| Skill | When to use | Correctness check |
|---|---|---|
| **`gen-infer-framework`** | Build a model-specific inference server with an OpenAI-compatible HTTP API | Immutable oracle boots `serve.sh`, sends fixed prompts, dispatches an LLM-judge sub-agent to verdict each response |
| **`port-model`** | Port a new model into vLLM / SGLang / TensorRT-LLM | Agent-authored `test.sh` (JSON contract) |
| **`opt-kernel`** | Optimize an existing GPU kernel for a target shape & GPU | Agent-authored `test.sh` with perf metrics |

```
   ┌─────────────────────────────────────────────────────────────┐
   │  skill invoked  ──►  requirements.json                      │
   │                            │                                │
   │                            ▼                                │
   │              deterministic Python orchestrator              │
   │   ┌──────────────────────────────────────────────────────────┐   │
   │   │  A: Plan ─► B: Implement ─► C: Test ─► D: Review         │   │
   │   │                                            │             │   │
   │   │   ◄── fail ──────────────────────────────  │  ── pass ─► │   │
   │   │                                            ▼             │   │
   │   │                          E: Perf Test ─► F: Perf Plan ─► A   │
   │   └──────────────────────────────────────────────────────────┘   │
   │              live WebUI dashboard (port 8765)                     │
   └──────────────────────────────────────────────────────────────────┘
```

## Install

One-time setup: symlink the skills into Claude Code's per-user skill
directory so they're discoverable in any session.

```bash
git clone https://github.com/<your-org>/MetaInfer.git
cd MetaInfer
bash scripts/install.sh
```

That's it. `install.sh` is idempotent — safe to re-run after pulling
updates. It only creates `~/.claude/skills/<name>` symlinks pointing
into this checkout; no `pip install`, no PATH modifications, no
`/usr/local/bin` writes.

## Usage

From inside a Claude Code session, the skills are auto-discovered and
show up by name. Trigger any workflow:

```
gen-infer-framework
port-model
opt-kernel
```

Each skill interviews you from its `questions.yaml`, writes
`requirements.json` into `.metainfer/state/<task_id>/` in your CWD, and
hands control to the orchestrator. The orchestrator prints the WebUI
URL — open it to watch the ABCDEF state machine, iteration history,
perf charts, and live sub-agent status.

**Resume detection.** Before interviewing, the skill scans
`.metainfer/state/*/requirements.json`. If it finds one, the previous
run was interrupted — it asks whether to resume (skip the interview,
re-launch the orchestrator on the existing `requirements.json`) or
start fresh under a new `task_id`.

You can also drive the orchestrator directly (without going through
Claude Code):

```bash
python <repo>/skills/gen-infer-framework/run.py run .metainfer/state/<task_id>/requirements.json
python <repo>/skills/gen-infer-framework/run.py run requirements.json --port 9000 --no-web
python <repo>/skills/gen-infer-framework/run.py web  .metainfer/state/<task_id>/   # restart dashboard only
```

`run.py` is self-contained: it inserts the skill dir at the front of
`sys.path` so `import metainfer` resolves, then delegates to
`metainfer.cli.main`. No `PYTHONPATH`, no wrapper, no global install.

### Resume & crash recovery

The orchestrator never silently overwrites prior work. On launch it
checks for an existing `run.json`:

- **Not present** → fresh run; starts at A_plan, iteration 1.
- **Present and `finished=true`** → the task already completed; the
  orchestrator refuses to overwrite (use a new `task_id` to start over).
- **Present and not finished** → resume. The orchestrator calls
  `IterationWorkspace.discard_latest_incomplete()`: if the
  highest-numbered iteration folder lacks a `.metainfer-completed`
  sentinel (written as the **last** step of every clean close), that
  folder and its state record are deleted, and the iteration restarts
  from its recorded `start_phase`.

### Configuring the Claude Code binary

Sub-agents shell out to `ccb` by default. Override per-invocation with
`--claude-bin`, or set `METAINFER_CLAUDE_BIN` in the environment:

```bash
python run.py run requirements.json                              # uses ccb
python run.py run requirements.json --claude-bin claude           # one-shot
METAINFER_CLAUDE_BIN=/usr/local/bin/claude python run.py run requirements.json
```

Precedence: `--claude-bin` flag > `METAINFER_CLAUDE_BIN` env var >
`ccb` default.

## Repository layout

```
MetaInfer/                              ← repo / checkout root
├── skills/                             ← one subdir per skill
│   ├── gen-infer-framework/            ← the "heavy" skill: carries the orchestrator
│   │   ├── SKILL.md                    ← frontmatter + task contract
│   │   ├── questions.yaml              ← interview questions
│   │   ├── run.py                      ← self-contained Python launcher
│   │   ├── metainfer/                  ← orchestrator package (lives here, with this skill)
│   │   │   ├── paths.py                ← single source of truth for paths
│   │   │   ├── cli.py                  ← `run.py` entry → cli.main
│   │   │   ├── orchestrator.py         ← wiring: req → store → manager → pipeline → web
│   │   │   ├── pipeline.py             ← ABCDEF state machine, transition-table-driven
│   │   │   ├── phases.py               ← TRANSITIONS table (data), graph helpers
│   │   │   ├── iteration.py            ← numbered iteration folder management
│   │   │   ├── state.py                ← JSON state store
│   │   │   ├── subagent_manager.py     ← Claude Code subprocess lifecycle
│   │   │   ├── prompts/                ← prompt templates for A/B/C/D/E/F agents
│   │   │   ├── oracles/                ← immutable correctness oracles
│   │   │   │   ├── base.py
│   │   │   │   ├── judge.py            ← LLM-judge sub-agent dispatch
│   │   │   │   └── infer_framework/    ← boots serve.sh, sends HTTP, judges replies
│   │   │   └── web/                    ← FastAPI dashboard + static frontend
│   │   └── notebooks/                  ← knowledge base (lives here, with this skill)
│   │       ├── 00_overview/
│   │       ├── 01_framework_design/
│   │       ├── 02_model_specifics/
│   │       ├── 03_operators/
│   │       ├── 04_parallel_strategies/
│   │       ├── 05_inference_service/
│   │       ├── 06_experience/
│   │       ├── 07_improvementPlan/
│   │       └── 08_issues/
│   ├── opt-kernel/
│   │   ├── SKILL.md
│   │   └── questions.yaml
│   └── port-model/
│       ├── SKILL.md
│       └── questions.yaml
└── scripts/
    └── install.sh                      ← one-shot skill symlink installer
```

### Per-task working directories (gitignored)

```
.metainfer/                         ← created in the user's CWD
├── state/<task_id>/                ← requirements.json + run state
└── logs/<task_id>/                 ← numbered iteration workspaces
<task_id>/                          ← created in the user's CWD
└── <NNN>/                          ← iteration CODE (visible, top-level)
```

Iteration code never touches the skill bundle — the bundle is read-only
knowledge + scripts, so one checkout serves many tasks.

## Path resolution

`skills/gen-infer-framework/metainfer/paths.py` is the single source of
truth. Because `paths.py` lives at
`<skill_root>/metainfer/paths.py`, the skill root is
`Path(__file__).resolve().parent.parent`. Sibling skills sit under
`<skill_root>/../`. Helpers:

- `skill_root()` — absolute path to the gen-infer-framework skill dir
  (where `run.py`, `metainfer/`, and `notebooks/` live)
- `skills_dir()` / `skill_dir(task_type)` — sibling skills under the
  same `skills/` parent
- `skill_md(task_type)` / `question_file(task_type)` — per-task assets
- `notebooks_dir()` — knowledge base (lives with this skill)
- `launcher()` — absolute path to `run.py`

All other modules import these helpers; no hardcoded absolute paths.

## Extending

### Question banks

Edit `skills/<task_type>/questions.yaml`:

```yaml
- key: my_new_field
  question: "What value for X?"
  header: "X"            # <= 12 chars
  required: true
  multi: false
  options:
    - label: "A"
      description: "implies ..."
    - label: "B"
      description: "implies ..."
```

### Knowledge base

Drop markdown files into `skills/gen-infer-framework/notebooks/<topic>/`.
Prompt templates already tell sub-agents to consult `notebooks/` — no
code change required. Keep each file short: one concept, one example,
one gotcha list.

### New task types

1. Create `skills/<new-task>/SKILL.md` (frontmatter `name` + `description`).
2. Add `skills/<new-task>/questions.yaml`.
3. Add the task type to `TASK_TYPES` in
   `skills/gen-infer-framework/metainfer/paths.py`.
4. If correctness needs an objective check, add an oracle under
   `skills/gen-infer-framework/metainfer/oracles/<new-task>/` and
   register it in `metainfer/oracles/__init__.py`.
5. The transition table in `phases.py` automatically routes
   `gen-infer-framework`-style tasks to the oracle path; otherwise the
   agent writes its own `test.sh`.

## Design notes

- **Skills over plugins.** Each skill is a self-contained directory
  with its SKILL.md, scripts, and knowledge base attached. No plugin
  wrapper, no `package.json`, no SessionStart hook, no PATH binary. The
  skill IS the unit. Discovery is just `~/.claude/skills/<name>/`
  symlinks.
- **Determinism over agency.** Long-running, multi-day tasks diverge
  if the LLM drives control flow. The orchestrator is plain Python;
  sub-agents only do work, never decide what runs next.
- **Data-driven state machine.** `phases.py:TRANSITIONS` is the single
  source of truth for the ABCDEF graph; the WebUI auto-derives its flow
  diagram from this table, so adding an edge or phase needs no UI code.
- **Outcome enum.** `Outcome.{ok,logic_fail,infra_fail,
  perf_regression,aborted}` is the only contract between the test
  runner and the transition table — easy to extend.
- **File-based state.** Everything goes through `StateStore` JSON files
  so the WebUI can observe state from a separate process without IPC.
- **Iteration folders.** Each iteration gets a fresh copy of the
  previous iteration's directory, so a bad iteration never poisons a
  good one.
- **Immutable oracles.** For tasks where the agent would otherwise
  grade its own homework (e.g. inference-framework correctness), the
  oracle ships with the skill, lives outside the iteration directory,
  and is the source of truth for pass/fail. An LLM-judge sub-agent
  rules on free-form responses to avoid string-match brittleness.
- **Zero third-party deps for the oracle.** The HTTP probe uses stdlib
  `urllib`; the judge dispatches through the existing
  `SubAgentManager` (a `claude -p` subprocess), so no SDK install is
  needed at runtime.
