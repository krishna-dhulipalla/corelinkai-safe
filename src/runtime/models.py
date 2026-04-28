from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4


CANONICAL_DECISIONS = {"ALLOW", "ALLOW-CONDITIONAL", "DENY", "ESCALATE"}
POLICY_BOOTSTRAP_EXTENSION = "urn:pi-bench:policy-bootstrap:v1"

ActionKind = Literal[
    "tool_call",
    "message",
    "record_decision",
    "clarify",
    "refuse",
    "escalate",
]
GateDecisionName = Literal[
    "allow",
    "block",
    "revise",
    "clarify",
    "verify",
    "refuse",
    "escalate",
]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass
class ToolSchema:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def required_args(self) -> list[str]:
        required = self.parameters.get("required", [])
        return [str(item) for item in required] if isinstance(required, list) else []


@dataclass
class PolicyCase:
    case_id: str
    benchmark: str
    domain: str
    task_summary: str
    actor: str
    requester: str
    resource_refs: list[str]
    requested_action: str
    available_tools: list[ToolSchema]
    policy_clauses: list[str]
    known_facts: list[str]
    required_evidence: list[str]
    sensitive_fields: list[str]
    history_summary: str
    risk_flags: list[str]

    @property
    def tool_names(self) -> set[str]:
        return {tool.name for tool in self.available_tools}

    def get_tool(self, name: str) -> ToolSchema | None:
        for tool in self.available_tools:
            if tool.name == name:
                return tool
        return None


@dataclass
class ActionProposal:
    proposal_id: str
    kind: ActionKind
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    decision_label: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    policy_refs: list[str] = field(default_factory=list)
    rationale_summary: str = ""


@dataclass
class GateDecision:
    decision: GateDecisionName
    reason_code: str
    feedback: str = ""
    policy_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    safe_replacement: ActionProposal | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


@dataclass
class AuditEvent:
    event_type: str
    payload: dict[str, Any]


@dataclass
class PolicySession:
    context_id: str
    benchmark_context: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    domain: str = ""
    run_id: str = ""
    audit_events: list[AuditEvent] = field(default_factory=list)

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        self.audit_events.append(AuditEvent(event_type=event_type, payload=payload))


@dataclass
class InboundRequest:
    is_bootstrap: bool
    context_id: str | None = None
    benchmark_context: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    domain: str = ""
    run_id: str = ""
    seed: int | None = None
    text: str = ""


@dataclass
class RuntimeResponse:
    data: dict[str, Any]
    audit_events: list[AuditEvent] = field(default_factory=list)
