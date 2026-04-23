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

- Prefer small, verifiable changes over broad rewrites.
- Respect the current version execution plan before changing architecture.
- Do not silently introduce major runtime-flow changes.
- When changing architecture or control flow, update the relevant docs.
- Keep project instructions practical and lightweight; do not overfit them to one tool.

## Commands

Update these for the repo if they differ.

### Environment / install

- Install dependencies: `pip install -r requirements.txt`
- Create venv if needed: `python -m venv .venv`
- Activate venv (Unix): `source .venv/bin/activate`
- Activate venv (Windows PowerShell): `.venv\\Scripts\\Activate.ps1`

### Run

- Main app: `python app.py`

### Tests

- Run all tests: `pytest`
- Run one test file: `pytest tests/test_file.py`
- Run one test: `pytest tests/test_file.py -k test_name`

### Quality

- Format: `black .`
- Lint: `ruff check .`
- Type check if configured: `mypy src`

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
- why it changed
- validation or evidence
- next step

Keep the file focused on recent major updates.
Move older condensed history into `.agents/memory/summaries.md`.

## Execution-plan rules

- Keep one execution plan file per version.
- A version plan may contain multiple phases.
- Each phase should include objective, tasks, implementation notes, and exit criteria.
- If a later stabilization track is needed, append it to the same version file instead of creating many tiny phase files.

## Checkpoints and reviews

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
