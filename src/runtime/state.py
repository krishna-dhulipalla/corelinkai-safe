from __future__ import annotations

import os
from dataclasses import dataclass
from operator import add
from typing import Annotated, Any, TypedDict

from llm.nebius import NebiusChatClient
from runtime.models import (
    ActionProposal,
    AuditEvent,
    GateDecision,
    InboundRequest,
    PolicyCase,
    PolicySession,
    ToolSchema,
)


class PolicyGraphState(TypedDict, total=False):
    request: InboundRequest
    context_id: str
    session: PolicySession
    messages: list[dict[str, Any]]
    policy_context: list[str]
    tool_catalog: list[ToolSchema]
    facts: list[str]
    evidence: list[str]
    provenance_labels: dict[str, str]
    plan: list[str]
    policy_obligations: list[str]
    case: PolicyCase
    candidate_payload: dict[str, Any]
    raw_model_output: str
    proposal: ActionProposal
    gate_decision: GateDecision
    final_response: dict[str, Any]
    graph_errors: Annotated[list[str], add]
    audit_events: Annotated[list[AuditEvent], add]
    iteration_count: int


@dataclass(frozen=True)
class PolicyGraphSettings:
    recursion_limit: int = 20
    langsmith_project: str = ""
    langsmith_tracing: bool = False

    @classmethod
    def from_env(cls) -> "PolicyGraphSettings":
        return cls(
            recursion_limit=_positive_int(
                os.environ.get("POLICY_GRAPH_RECURSION_LIMIT"), 20
            ),
            langsmith_project=os.environ.get("LANGSMITH_PROJECT", "").strip(),
            langsmith_tracing=_env_flag("LANGSMITH_TRACING")
            or _env_flag("LANGCHAIN_TRACING_V2"),
        )


@dataclass
class PolicyGraphRuntimeContext:
    model_client: NebiusChatClient
    settings: PolicyGraphSettings


def _positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
