from __future__ import annotations

import json

from runtime.models import (
    CANONICAL_DECISIONS,
    ActionProposal,
    GateDecision,
    PolicyCase,
    new_id,
)


SENSITIVE_DISCLOSURE_TERMS = (
    "system prompt",
    "hidden evaluator",
    "confidential internal",
    "fraud_score",
    "risk score",
)

MUTATING_TOOL_TERMS = (
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
    "record",
)

READ_TOOL_TERMS = (
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


def validate_proposal(case: PolicyCase, proposal: ActionProposal) -> GateDecision:
    if proposal.kind != "tool_call":
        if _contains_sensitive_disclosure(proposal.content):
            return GateDecision(
                decision="block",
                reason_code="sensitive_disclosure",
                feedback="The proposed message appears to disclose internal or confidential information.",
                safe_replacement=safe_message(
                    "I cannot disclose internal or confidential details. I can continue using the applicable policy process."
                ),
            )
        return GateDecision(decision="allow", reason_code="message_safe")

    if not proposal.tool_name:
        return GateDecision(
            decision="block",
            reason_code="missing_tool_name",
            feedback="The proposed tool call did not include a tool name.",
            safe_replacement=safe_message("I need a valid benchmark tool before taking action."),
        )

    tool = case.get_tool(proposal.tool_name)
    if tool is None:
        return GateDecision(
            decision="block",
            reason_code="unavailable_tool",
            feedback=f"Tool {proposal.tool_name!r} is not available for this task.",
            safe_replacement=safe_message(
                "I cannot use a tool that is not available in this policy task."
            ),
        )

    if proposal.tool_name == "record_decision":
        _hydrate_record_decision_arguments(case, proposal)

    missing = [arg for arg in tool.required_args if arg not in proposal.arguments]
    if missing:
        return GateDecision(
            decision="block",
            reason_code="missing_required_arguments",
            feedback=f"Missing required tool arguments: {', '.join(missing)}.",
            safe_replacement=safe_message(
                "I need the required verified details before taking that action."
            ),
        )

    if _arguments_disclose_sensitive_data(proposal):
        return GateDecision(
            decision="block",
            reason_code="sensitive_argument_disclosure",
            feedback="The proposed tool arguments appear to include internal or confidential data.",
            safe_replacement=safe_message(
                "I cannot pass internal or confidential details into that action."
            ),
        )

    if proposal.tool_name == "record_decision":
        return _validate_record_decision(case, proposal)

    if _is_mutating_tool(proposal.tool_name) and not (
        proposal.evidence_refs or _has_prior_tool_evidence(case)
    ):
        return GateDecision(
            decision="verify",
            reason_code="high_impact_action_requires_evidence",
            feedback=(
                "High-impact actions need policy-grounded evidence. Use an available "
                "read/check/lookup tool first when one can verify the required facts."
            ),
            policy_refs=proposal.policy_refs or case.policy_clauses[:3],
        )

    return GateDecision(
        decision="allow",
        reason_code="tool_call_valid",
        policy_refs=proposal.policy_refs,
        evidence_refs=proposal.evidence_refs,
    )


def emit_response_data(case: PolicyCase, proposal: ActionProposal) -> dict[str, object]:
    if proposal.kind == "tool_call":
        return {
            "tool_calls": [
                {
                    "id": proposal.proposal_id,
                    "type": "function",
                    "function": {
                        "name": proposal.tool_name,
                        "arguments": json.dumps(proposal.arguments, sort_keys=True),
                    },
                }
            ],
            **({"content": proposal.content} if proposal.content else {}),
        }

    content = proposal.content.strip() or _fallback_decision_json(case, proposal)
    return {"content": content}


def fallback_proposal(case: PolicyCase) -> ActionProposal:
    if "record_decision" in case.tool_names:
        arguments = {
            "decision": "ESCALATE",
            "reason": "Model client is unavailable, so the safe policy-compliance fallback is escalation.",
            "notes": "Model client is unavailable, so the safe policy-compliance fallback is escalation.",
            "policy_sections_cited": case.policy_clauses[:3],
        }
        proposal = ActionProposal(
            proposal_id=new_id("proposal"),
            kind="tool_call",
            tool_name="record_decision",
            arguments=arguments,
            decision_label="ESCALATE",
            policy_refs=case.policy_clauses[:3],
            rationale_summary="Safe fallback without model access.",
        )
        _hydrate_record_decision_arguments(case, proposal)
        return proposal
    return safe_message(_fallback_decision_json(case, None))


def safe_message(content: str) -> ActionProposal:
    return ActionProposal(
        proposal_id=new_id("proposal"),
        kind="message",
        content=content,
    )


def safe_decision_message(proposal: ActionProposal) -> ActionProposal:
    decision = proposal.decision_label if proposal.decision_label in CANONICAL_DECISIONS else "ESCALATE"
    return safe_message(
        "```json\n"
        + json.dumps(
            {
                "decision": decision,
                "reason": "The canonical decision tool call could not be validated.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```"
    )


def _hydrate_record_decision_arguments(
    case: PolicyCase, proposal: ActionProposal
) -> None:
    tool = case.get_tool("record_decision")
    if tool is None:
        return
    required = set(tool.required_args)
    if "order_id" in required:
        order_id = _first_resource(case, ("ord", "order"))
        if order_id:
            proposal.arguments.setdefault("order_id", order_id)
    if "customer_id" in required:
        customer_id = _first_resource(case, ("cust", "customer"))
        if customer_id:
            proposal.arguments.setdefault("customer_id", customer_id)
    if "policy_sections_cited" in required and case.policy_clauses:
        proposal.arguments.setdefault("policy_sections_cited", case.policy_clauses[:3])
    if "notes" in required and proposal.rationale_summary:
        proposal.arguments.setdefault("notes", proposal.rationale_summary)


def _first_resource(case: PolicyCase, prefixes: tuple[str, ...]) -> str:
    for ref in case.resource_refs:
        lowered = ref.lower()
        if any(lowered.startswith(prefix) for prefix in prefixes):
            return ref
    return ""


def _validate_record_decision(
    case: PolicyCase, proposal: ActionProposal
) -> GateDecision:
    decision = str(
        proposal.arguments.get("decision") or proposal.decision_label or ""
    ).upper()
    if decision not in CANONICAL_DECISIONS:
        return GateDecision(
            decision="block",
            reason_code="invalid_canonical_decision",
            feedback="record_decision requires ALLOW, ALLOW-CONDITIONAL, DENY, or ESCALATE.",
            safe_replacement=safe_decision_message(proposal),
        )

    proposal.arguments["decision"] = decision
    if proposal.policy_refs:
        proposal.arguments.setdefault("policy_sections_cited", proposal.policy_refs)
    elif case.policy_clauses:
        proposal.policy_refs = case.policy_clauses[:3]
        proposal.arguments.setdefault("policy_sections_cited", case.policy_clauses[:3])
    if proposal.evidence_refs:
        proposal.arguments.setdefault("evidence_cited", proposal.evidence_refs[:5])
    elif case.known_facts:
        proposal.evidence_refs = case.known_facts[:3]
        proposal.arguments.setdefault("evidence_cited", case.known_facts[:3])
    if proposal.rationale_summary:
        proposal.arguments.setdefault("reason", proposal.rationale_summary)

    return GateDecision(
        decision="allow",
        reason_code="record_decision_valid",
        policy_refs=proposal.policy_refs,
        evidence_refs=proposal.evidence_refs,
    )


def _fallback_decision_json(
    case: PolicyCase, proposal: ActionProposal | None
) -> str:
    decision = (
        proposal.decision_label
        if proposal and proposal.decision_label in CANONICAL_DECISIONS
        else "ESCALATE"
    )
    return (
        "```json\n"
        + json.dumps(
            {
                "decision": decision,
                "reason": "A safe fallback decision was produced because no validated tool call was available.",
                "policy_sections_cited": case.policy_clauses[:3],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```"
    )


def _contains_sensitive_disclosure(content: str) -> bool:
    lower = content.lower()
    return any(term in lower for term in SENSITIVE_DISCLOSURE_TERMS)


def _arguments_disclose_sensitive_data(proposal: ActionProposal) -> bool:
    payload = json.dumps(proposal.arguments, default=str).lower()
    return any(term in payload for term in ("system prompt", "hidden evaluator"))


def _is_mutating_tool(tool_name: str) -> bool:
    lowered = tool_name.lower()
    if any(term in lowered for term in READ_TOOL_TERMS):
        return False
    return any(term in lowered for term in MUTATING_TOOL_TERMS)


def _has_prior_tool_evidence(case: PolicyCase) -> bool:
    return any(fact.startswith("tool_result:") for fact in case.known_facts)
