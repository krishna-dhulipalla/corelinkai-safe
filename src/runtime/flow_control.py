from __future__ import annotations

import json
import re
from typing import Any

from runtime.models import ActionProposal, GateDecision, PolicyCase, ToolSchema
from runtime.policies import safe_message


READ_TERMS = (
    "read",
    "get",
    "lookup",
    "search",
    "check",
    "list",
    "fetch",
    "retrieve",
    "verify",
)

MUTATE_TERMS = (
    "approve",
    "deny",
    "refund",
    "transfer",
    "wire",
    "grant",
    "reset",
    "delete",
    "update",
    "create",
    "process",
    "submit",
)

EGRESS_TERMS = (
    "send",
    "email",
    "notify",
    "message",
    "disclose",
    "share",
    "export",
)

METADATA_ATTACK_TERMS = (
    "ignore previous",
    "ignore instructions",
    "system prompt",
    "hidden evaluator",
    "developer message",
    "exfiltrate",
)

SENSITIVE_EGRESS_TERMS = (
    "fraud_score",
    "risk score",
    "internal",
    "confidential",
    "hidden evaluator",
    "system prompt",
    "investigation",
)


def label_memory(
    *,
    policy_context: list[str],
    messages: list[dict[str, Any]],
    evidence: list[str],
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {
        "policy_context": {
            "integrity": "trusted_policy",
            "confidentiality": "policy",
            "allowed_sinks": ["solver", "policy_gate", "record_decision"],
        },
        "user_task": {
            "integrity": "trusted_task",
            "confidentiality": "public",
            "allowed_sinks": ["solver", "policy_gate", "tool_args", "record_decision", "user_message"],
        },
        "assistant_trajectory": {
            "integrity": "agent_control",
            "confidentiality": "internal",
            "allowed_sinks": ["solver", "policy_gate", "record_decision"],
        },
    }
    for index, item in enumerate(policy_context, start=1):
        labels[f"policy_context:{index}"] = {
            "integrity": "trusted_policy",
            "confidentiality": "policy",
            "summary": item[:160],
            "allowed_sinks": ["solver", "policy_gate", "record_decision"],
        }
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "message"))
        if role in {"tool", "multi_tool"}:
            labels[f"message:{index}"] = {
                "integrity": "untrusted_observation",
                "confidentiality": "private_observation",
                "allowed_sinks": ["solver", "policy_gate", "record_decision"],
                "blocked_sinks": ["user_message", "external_egress"],
            }
        elif role == "user":
            labels[f"message:{index}"] = {
                "integrity": "trusted_task",
                "confidentiality": "public",
                "allowed_sinks": ["solver", "policy_gate", "tool_args", "record_decision", "user_message"],
            }
        else:
            labels[f"message:{index}"] = {
                "integrity": "agent_control",
                "confidentiality": "internal",
                "allowed_sinks": ["solver", "policy_gate", "record_decision"],
            }
    for index, item in enumerate(evidence, start=1):
        labels[f"evidence:{index}"] = {
            "integrity": "untrusted_observation",
            "confidentiality": "private_observation",
            "summary": item[:160],
            "allowed_sinks": ["solver", "policy_gate", "record_decision"],
            "blocked_sinks": ["user_message", "external_egress"],
        }
    return labels


def build_tool_capabilities(case: PolicyCase) -> dict[str, dict[str, Any]]:
    return {tool.name: _capability_for_tool(tool) for tool in case.available_tools}


def build_trusted_flow_plan(
    case: PolicyCase,
    tool_capabilities: dict[str, dict[str, Any]],
    *,
    has_tool_evidence: bool,
) -> list[dict[str, Any]]:
    read_tools = [
        name
        for name, capability in tool_capabilities.items()
        if capability["category"] == "observe"
    ]
    mutating_tools = [
        name
        for name, capability in tool_capabilities.items()
        if capability["category"] == "mutate"
    ]
    egress_tools = [
        name
        for name, capability in tool_capabilities.items()
        if capability["category"] == "egress"
    ]
    plan: list[dict[str, Any]] = []
    if read_tools:
        plan.append(
            {
                "step_id": "observe",
                "purpose": "Gather environment evidence needed for policy decisioning.",
                "allowed_tools": read_tools,
                "requires_prior_tool_evidence": False,
                "input_integrity": ["trusted_task", "trusted_policy"],
                "output_label": "untrusted_observation",
            }
        )
    if mutating_tools:
        plan.append(
            {
                "step_id": "act",
                "purpose": "Perform consequential environment actions only after verification.",
                "allowed_tools": mutating_tools,
                "requires_prior_tool_evidence": True,
                "input_integrity": ["trusted_task", "trusted_policy", "untrusted_observation"],
                "output_label": "private_observation",
            }
        )
    if "record_decision" in tool_capabilities:
        plan.append(
            {
                "step_id": "final_decision",
                "purpose": "Record the canonical benchmark decision with policy and evidence references.",
                "allowed_tools": ["record_decision"],
                "requires_prior_tool_evidence": False,
                "input_integrity": ["trusted_task", "trusted_policy", "untrusted_observation"],
                "output_label": "benchmark_decision",
            }
        )
    if egress_tools:
        plan.append(
            {
                "step_id": "egress",
                "purpose": "Send only public or explicitly permitted information to external recipients.",
                "allowed_tools": egress_tools,
                "requires_prior_tool_evidence": False,
                "input_integrity": ["trusted_task", "trusted_policy"],
                "blocked_input_integrity": ["untrusted_observation"],
                "output_label": "external_egress",
            }
        )
    if not plan:
        plan.append(
            {
                "step_id": "respond",
                "purpose": "Return a safe message when no external tools are available.",
                "allowed_tools": [],
                "requires_prior_tool_evidence": False,
                "input_integrity": ["trusted_task", "trusted_policy"],
                "output_label": "user_message",
            }
        )
    if has_tool_evidence:
        for step in plan:
            if step["step_id"] in {"act", "final_decision"}:
                step["reachable"] = True
    return plan


def verify_flow_plan(
    case: PolicyCase,
    trusted_plan: list[dict[str, Any]],
    tool_capabilities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    available = set(case.tool_names)
    planned_tools = set()
    for step in trusted_plan:
        for tool_name in step.get("allowed_tools", []):
            planned_tools.add(str(tool_name))
            if tool_name not in available:
                errors.append(f"planned tool is unavailable: {tool_name}")
            capability = tool_capabilities.get(str(tool_name), {})
            if capability.get("metadata_trust") == "suspicious":
                warnings.append(f"tool metadata is suspicious: {tool_name}")
    for tool_name, capability in tool_capabilities.items():
        if capability["category"] in {"mutate", "egress", "decision"} and tool_name not in planned_tools:
            warnings.append(f"consequential tool omitted from trusted plan: {tool_name}")
    if "record_decision" in available and "record_decision" not in planned_tools:
        errors.append("record_decision is available but not reachable in the trusted plan")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "planned_tools": sorted(planned_tools),
        "policy_refs": case.policy_clauses[:8],
    }


def constrained_extract_facts(
    evidence: list[str],
    resource_refs: list[str],
) -> list[str]:
    extracted: list[str] = []
    resource_tokens = {ref.lower() for ref in resource_refs}
    for item in evidence:
        lowered = item.lower()
        if resource_tokens and not any(token in lowered for token in resource_tokens):
            continue
        for key in (
            "eligible",
            "status",
            "reason",
            "delivery_date",
            "return_window_days",
            "restocking_fee_applicable",
            "decision",
            "error",
        ):
            match = re.search(rf'"?{re.escape(key)}"?\s*[:=]\s*"?([^",}}]+)', item, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                extracted.append(f"constrained:{key}={value}")
    deduped: list[str] = []
    for item in extracted:
        if item not in deduped:
            deduped.append(item)
    return deduped[:12]


def validate_flow_control(
    *,
    case: PolicyCase,
    proposal: ActionProposal,
    trusted_plan: list[dict[str, Any]],
    plan_verification: dict[str, Any],
    tool_capabilities: dict[str, dict[str, Any]],
    evidence: list[str],
) -> GateDecision:
    if not plan_verification.get("valid", False):
        return GateDecision(
            decision="escalate",
            reason_code="flow_plan_invalid",
            feedback="The trusted flow plan failed verification, so the action cannot proceed.",
            policy_refs=case.policy_clauses[:3],
            safe_replacement=safe_message("I need to escalate because the policy execution plan is not valid."),
        )

    if proposal.kind != "tool_call":
        return _validate_message_sink(case, proposal)

    capability = tool_capabilities.get(proposal.tool_name)
    if capability is None:
        return GateDecision(
            decision="block",
            reason_code="flow_tool_without_capability",
            feedback=f"Tool {proposal.tool_name!r} has no flow-control capability mapping.",
        )

    step = _step_for_tool(trusted_plan, proposal.tool_name)
    if step is None:
        return GateDecision(
            decision="block",
            reason_code="flow_out_of_plan_tool",
            feedback=f"Tool {proposal.tool_name!r} is not reachable in the trusted flow plan.",
        )

    if capability.get("metadata_trust") == "suspicious":
        return GateDecision(
            decision="block",
            reason_code="flow_suspicious_tool_metadata",
            feedback=f"Tool {proposal.tool_name!r} has suspicious metadata and cannot be used without review.",
        )

    if capability["category"] == "mutate" and not evidence:
        return GateDecision(
            decision="verify",
            reason_code="flow_mutation_requires_observation",
            feedback="The trusted plan requires observation evidence before consequential actions.",
            policy_refs=case.policy_clauses[:3],
        )

    if capability["category"] == "egress" and _proposal_contains_sensitive_data(proposal):
        return GateDecision(
            decision="block",
            reason_code="flow_sensitive_egress",
            feedback="The proposed egress action includes private or internal data.",
            safe_replacement=safe_message("I cannot disclose internal or private information."),
        )

    if capability["category"] == "decision":
        decision = str(proposal.arguments.get("decision") or proposal.decision_label or "").upper()
        if decision in {"ALLOW", "ALLOW-CONDITIONAL", "DENY"} and not (
            evidence or proposal.evidence_refs or case.known_facts
        ):
            return GateDecision(
                decision="verify",
                reason_code="flow_final_decision_requires_evidence",
                feedback="Non-escalation final decisions need policy-grounded evidence.",
                policy_refs=case.policy_clauses[:3],
            )

    return GateDecision(
        decision="allow",
        reason_code="flow_control_allow",
        policy_refs=case.policy_clauses[:3],
        evidence_refs=proposal.evidence_refs,
    )


def _capability_for_tool(tool: ToolSchema) -> dict[str, Any]:
    lowered = f"{tool.name} {tool.description} {json.dumps(tool.parameters, default=str)}".lower()
    if tool.name == "record_decision":
        category = "decision"
        sink = "record_decision"
    elif any(term in lowered for term in EGRESS_TERMS):
        category = "egress"
        sink = "external_egress"
    elif any(term in lowered for term in READ_TERMS):
        category = "observe"
        sink = "runtime_memory"
    elif any(term in lowered for term in MUTATE_TERMS):
        category = "mutate"
        sink = "environment_state"
    else:
        category = "observe"
        sink = "runtime_memory"
    return {
        "tool_name": tool.name,
        "category": category,
        "sink": sink,
        "metadata_trust": "suspicious"
        if any(term in lowered for term in METADATA_ATTACK_TERMS)
        else "declared_tool_metadata",
        "requires_prior_tool_evidence": category == "mutate",
    }


def _validate_message_sink(case: PolicyCase, proposal: ActionProposal) -> GateDecision:
    if _content_contains_sensitive_data(proposal.content):
        return GateDecision(
            decision="block",
            reason_code="flow_sensitive_user_message",
            feedback="The proposed user-facing message includes private or internal data.",
            safe_replacement=safe_message("I cannot disclose internal or confidential details."),
        )
    return GateDecision(decision="allow", reason_code="flow_message_sink_allow")


def _step_for_tool(
    trusted_plan: list[dict[str, Any]], tool_name: str
) -> dict[str, Any] | None:
    for step in trusted_plan:
        if tool_name in step.get("allowed_tools", []):
            return step
    return None


def _proposal_contains_sensitive_data(proposal: ActionProposal) -> bool:
    payload = f"{proposal.content} {json.dumps(proposal.arguments, default=str)}"
    return _content_contains_sensitive_data(payload)


def _content_contains_sensitive_data(content: str) -> bool:
    lowered = content.lower()
    return any(term in lowered for term in SENSITIVE_EGRESS_TERMS)
