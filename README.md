# CoreLink AI Safe

CoreLink AI Safe is an early policy/safety agent for A2A-compatible agent benchmarks. The current runtime is a LangGraph Flow-Control Policy Graph Runtime: it converts benchmark context into a normalized policy case, builds a trusted flow plan, labels memory, maps tool capabilities, proposes one action at a time, validates that action through flow-control and runtime gates, and emits canonical policy decisions with traceable evidence.

The first test target is Pi-Bench. Pi-Bench support is treated as benchmark testing and protocol integration, not as the full product boundary.

## Current Shape

```text
src/
  server.py                         # stable entrypoint wrapper
  a2a_bridge/                       # A2A server, executor, agent bridge
  adapters/                         # benchmark/protocol adapters
  runtime/                          # LangGraph flow-control runtime, policy models, gates, decision emitter
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
$env:NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
$env:NEBIUS_PRIMARY_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
$env:NEBIUS_MEDIUM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
$env:POLICY_GRAPH_RECURSION_LIMIT = "20"
```

`NEBIUS_MODEL` is still accepted as a backward-compatible primary-model fallback. The local `.env` file is loaded automatically when values are not already present in the process environment.

Check model access:

```powershell
uv run python scripts/nebius_preflight.py --list
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
uv run python src/server.py --host 127.0.0.1 --port 9010 --card-url http://127.0.0.1:9010
```

For Docker:

```powershell
docker build -t corelinkai-safe .
docker run -p 9010:9010 --env-file .env corelinkai-safe --host 0.0.0.0 --port 9010 --card-url http://host.docker.internal:9010
```

## Test

Start the server, then run:

```powershell
$env:LANGSMITH_TRACING = "false"
uv run python -m pytest tests --agent-url http://127.0.0.1:9010
```

Runtime-only tests:

```powershell
uv run python -m pytest tests/test_policy_runtime.py
```

## Business Case Demo

Start the agent locally, then run selected realistic Pi-Bench business cases from a sibling Pi-Bench checkout:

```powershell
uv run python scripts/run_business_cases.py --pi-bench-root ..\pi-bench --host 127.0.0.1 --port 9010
```

The default demo set covers retail refunds, retail fraud-disclosure privacy, helpdesk cross-employee access, and FINRA trust-wire escalation. The script writes a Markdown report and JSON summary under `generated/business_cases/`.

If Pi-Bench dependencies are installed in a different Python environment, pass that executable with `--python`.

Full Pi-Bench smoke remains a diagnostic benchmark step:

```powershell
cd ..\pi-bench
python examples/a2a_demo/run_a2a.py --external --host 127.0.0.1 --port 9010 --serve-user --user-kind scripted --scenarios-dir scenarios/retail --concurrency 1 --max-steps 20 --save-to runs/corelinkai-safe-retail-smoke.json
```

## CI Status

GitHub Actions currently builds the Docker image and runs tests only. Package/image publishing is intentionally paused during early development.
