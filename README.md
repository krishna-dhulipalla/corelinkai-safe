# CoreLink AI Safe

CoreLink AI Safe is an early policy/safety agent for A2A-compatible agent benchmarks. The current runtime is a LangGraph Policy Graph Runtime: it converts benchmark context into a normalized policy case, plans evidence needs, proposes one action at a time, validates that action through a hybrid runtime gate, and emits canonical policy decisions with traceable evidence.

The first test target is Pi-Bench. Pi-Bench support is treated as benchmark testing and protocol integration, not as the full product boundary.

## Current Shape

```text
src/
  server.py                         # stable entrypoint wrapper
  a2a_bridge/                       # A2A server, executor, agent bridge
  adapters/                         # benchmark/protocol adapters
  runtime/                          # LangGraph runtime, policy case models, gate, decision emitter
  llm/                              # model-provider clients
tests/
  test_agent.py                     # A2A conformance checks
  test_policy_runtime.py            # runtime and Pi-Bench contract checks
.agents/                            # planning, architecture, and progress docs
```

The top-level `src/server.py`, `src/agent.py`, and `src/executor.py` files remain as thin compatibility wrappers so Docker and template test commands keep working while the internal folders evolve.

`a2a_bridge` is intentionally not named `a2a` because this project depends on the external `a2a-sdk` package, imported as `a2a`.

## Model Provider

The runtime currently uses Nebius Token Factory through an OpenAI-compatible chat endpoint.

Required for model-backed runs:

```powershell
$env:NEBIUS_API_KEY = "<token>"
```

Optional:

```powershell
$env:NEBIUS_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
$env:POLICY_GRAPH_RECURSION_LIMIT = "20"
```

If no model key is configured, the agent remains protocol-valid and falls back to safe escalation, but it is not competitive.

Optional LangSmith tracing:

```powershell
$env:LANGSMITH_API_KEY = "<token>"
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_PROJECT = "corelinkai-safe"
```

## Run Locally

```powershell
uv sync --extra test
uv run python src/server.py --host 127.0.0.1 --port 9009
```

For Docker:

```powershell
docker build -t corelinkai-safe .
docker run -p 9009:9009 --env-file .env corelinkai-safe --host 0.0.0.0 --port 9009
```

## Test

Start the server, then run:

```powershell
uv run python -m pytest tests --agent-url http://127.0.0.1:9009
```

Runtime-only tests:

```powershell
uv run python -m pytest tests/test_policy_runtime.py
```

## CI Status

GitHub Actions currently builds the Docker image and runs tests only. Package/image publishing is intentionally paused during early development.
