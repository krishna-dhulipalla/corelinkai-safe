# AGENTS.md

## Purpose

This repository uses long-horizon, documentation-driven development for agentic and multi-step coding work.
Before making major changes, read the project docs in `.agents/`.

## Source of truth

- Product goal and intended stack: `.agents/PRD.md`
- Active implementation plan: `.agents/execution-plans/`
- Major work log: `.agents/progress.md`
- Current architecture: `.agents/architecture/current.md`
- Version history and rationale: `.agents/architecture/versions.md`
- Major health checks: `.agents/checkpoints/`
- Scoped feature validation: `.agents/reviews/`
- Compacted context and handoff summaries: `.agents/memory/`

## Working mode

- Do not silently introduce major runtime-flow changes.
- When changing architecture or control flow, update the relevant docs.
- Keep project instructions practical and lightweight; do not overfit them to one tool.

## Commands

Local setup:

```powershell
uv sync --extra test
```

Run the agent locally:

```powershell
$env:NEBIUS_API_KEY = "<token>"
$env:NEBIUS_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
$env:POLICY_GRAPH_RECURSION_LIMIT = "20"
uv run python src/server.py --host 127.0.0.1 --port 9010 --card-url http://127.0.0.1:9010
```

Optional LangSmith node tracing:

```powershell
$env:LANGSMITH_API_KEY = "<token>"
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_PROJECT = "corelinkai-safe"
```

`LANGSMITH_API_KEY` alone does not create traces. Set `LANGSMITH_TRACING=true`; the LangGraph runtime wraps graph nodes with LangSmith spans when tracing is enabled.

Run tests against a running local agent:

```powershell
uv run python -m pytest tests --agent-url http://127.0.0.1:9010
```

Run with Pi-Bench for benchmark smoke testing:

1. Clone Pi-Bench separately, preferably as a sibling repo, not inside this repository.

```powershell
git clone https://github.com/Jyoti-Ranjan-Das845/pi-bench ..\pi-bench
cd ..\pi-bench
python -m pip install -e .
```

2. Start this agent from `corelinkai-safe` in another terminal.

```powershell
uv run python src/server.py --host 127.0.0.1 --port 9010 --card-url http://127.0.0.1:9010
```

3. From the Pi-Bench repo, run an A2A smoke pass against the external agent.

```powershell
python examples/a2a_demo/run_a2a.py --external --host 127.0.0.1 --port 9010 --serve-user --user-kind scripted --scenarios-dir scenarios/retail --concurrency 1 --max-steps 20 --save-to runs/corelinkai-safe-retail-smoke.json
```

Use the first Pi-Bench run only as a diagnostic smoke test. Classify failures by protocol, parsing, invalid tool call, wrong decision, tool argument, ordering/state, under-refusal, or over-refusal before changing architecture.

If the evaluator runs from Docker and calls a host-run agent, bind with `--host 0.0.0.0` but advertise a reachable URL such as `--card-url http://host.docker.internal:9010`; do not advertise `0.0.0.0` as the agent URL.

## Constraints

- Do not deploy, modify secrets, or perform destructive operations unless explicitly requested.
- Do not invent architecture rationale; check `.agents/architecture/` first.
- Do not treat benchmark-specific shortcuts as general solutions unless documented.
- Preserve determinism and traceability when working on runtime, evaluation, or retrieval flows.
- Avoid large speculative refactors unless the execution plan or checkpoint justifies them.
- If context is unclear, prefer reading the relevant doc rather than guessing.

## Progress logging

Add entries to `.agents/progress.md` only for meaningful milestones, not tiny edits.

Each entry should include:

- date
- role: planner / coder / reviewer
- what changed
- validation or evidence
- next step/ handoff notes

Keep the file focused on recent major updates.
Move older condensed history into `.agents/memory/summaries.md`(this is triggered manually by user).

## Execution-plan rules

- Keep one execution plan file per version.
- A version plan may contain multiple phases.
- Each phase should include objective, tasks, implementation notes(this if after phase done), and exit criteria.
- If a later stabilization track is needed, append it to the same version file instead of creating many tiny phase files.

## Checkpoints and reviews (this triggered manually)

Create a checkpoint under `.agents/checkpoints/` when:

- architecture shifts materially
- repeated regressions appear
- the system may be drifting from project goals
- complexity rises enough that a direction review is needed

Create a scoped review under `.agents/reviews/` for smaller feature-level validation.

## Self-improvement rule

This file may be updated when a genuinely repeating workflow appears across multiple sessions or versions.
Keep changes small and general:

- add reusable commands/ replace outdated commands
- add stable doc pointers
- add recurring constraints
- add short workflow rules that reduce repeated prompting

Do not turn this file into a locked policy wall.
Do not add temporary task-specific instructions that belong in `.agents/execution-plans/` or `.agents/reviews/`.

## Tool-specific notes

- `AGENTS.md` is the main shared instruction file.
- `CLAUDE.md` should remain a thin pointer to this file plus only Claude-specific notes.
- Tool-specific rules belong in `.agents/rules/` only when they clearly add value and are not worth duplicating here.
