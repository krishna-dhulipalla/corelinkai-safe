from __future__ import annotations

import json
import re
from typing import Any

from runtime.models import PolicyCase, PolicySession, ToolSchema, new_id


SENSITIVE_TERMS = (
    "internal",
    "confidential",
    "fraud_score",
    "risk score",
    "investigation",
    "hidden",
    "system prompt",
    "evaluator",
)

HIGH_IMPACT_TOOL_TERMS = (
    "refund",
    "reset",
    "grant",
    "transfer",
    "wire",
    "approve",
    "deny",
    "escalate",
    "record_decision",
    "access",
    "disclose",
    "delete",
    "update",
    "create",
)

VERIFY_TOOL_TERMS = (
    "read",
    "get",
    "lookup",
    "search",
    "check",
    "list",
    "fetch",
    "retrieve",
    "verify",
    "find",
)


def build_context_bundle(session: PolicySession) -> dict[str, Any]:
    policy_context = _context_texts(session.benchmark_context)
    tool_catalog = [_tool_schema(tool) for tool in session.tools]
    facts = _known_facts(policy_context, session.messages)
    tool_evidence = _tool_result_facts(session.messages)
    labels = {
        "benchmark_context": "trusted_policy_context",
        "user_messages": "untrusted_user_input",
        "tool_results": "untrusted_observation_data",
        "assistant_messages": "agent_trajectory",
    }
    return {
        "policy_context": policy_context,
        "tool_catalog": tool_catalog,
        "facts": [*facts, *tool_evidence],
        "evidence": tool_evidence,
        "provenance_labels": labels,
    }


def build_policy_case(session: PolicySession) -> PolicyCase:
    context_texts = _context_texts(session.benchmark_context)
    tools = [_tool_schema(tool) for tool in session.tools]
    last_user = _last_message_content(session.messages, "user")
    domain = session.domain or _domain_from_context(session.benchmark_context)
    policy_clauses = _extract_policy_clauses(context_texts)
    known_facts = _known_facts(context_texts, session.messages)
    tool_evidence = _tool_result_facts(session.messages)

    return PolicyCase(
        case_id=new_id("case"),
        benchmark="pi-bench",
        domain=domain,
        task_summary=_trim(" ".join(context_texts[:3]), 1600),
        actor="assistant",
        requester="user",
        resource_refs=_extract_resource_refs(last_user + "\n" + "\n".join(context_texts)),
        requested_action=last_user,
        available_tools=tools,
        policy_clauses=policy_clauses,
        known_facts=[*known_facts, *tool_evidence],
        required_evidence=policy_clauses[:10],
        sensitive_fields=_sensitive_fields(context_texts),
        history_summary=_history_summary(session.messages),
        risk_flags=_risk_flags(tools, context_texts),
    )


def _context_texts(context: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for node in context:
        kind = str(node.get("kind", "context") or "context")
        metadata = node.get("metadata", {})
        metadata_text = (
            " ".join(f"{key}={value}" for key, value in metadata.items())
            if isinstance(metadata, dict)
            else ""
        )
        content = str(node.get("content", "") or "")
        combined = "\n".join(part for part in (kind, metadata_text, content) if part)
        if combined:
            texts.append(combined)
    return texts


def _tool_schema(raw: dict[str, Any]) -> ToolSchema:
    function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = str(function.get("name", "") or "").strip()
    return ToolSchema(
        name=name,
        description=str(function.get("description", "") or ""),
        parameters=function.get("parameters", {})
        if isinstance(function.get("parameters"), dict)
        else {},
        raw=raw,
    )


def _domain_from_context(context: list[dict[str, Any]]) -> str:
    for node in context:
        metadata = node.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("domain", "domain_name", "policy_pack"):
                if metadata.get(key):
                    return str(metadata[key])
    return ""


def _last_message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role and message.get("content"):
            return _message_content(message)
    return ""


def _history_summary(messages: list[dict[str, Any]]) -> str:
    summary = []
    for message in messages[-10:]:
        role = str(message.get("role", "message"))
        content = _message_content(message)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            content += f" tool_calls={tool_calls}"
        if role == "multi_tool":
            content = json.dumps(message.get("tool_messages", []), ensure_ascii=False)
        summary.append(f"{role}: {_trim(content, 600)}")
    return "\n".join(summary)


def _known_facts(
    context_texts: list[str], messages: list[dict[str, Any]]
) -> list[str]:
    facts = []
    text_inputs = [
        *context_texts[:10],
        *[_message_content(message) for message in messages[-8:]],
    ]
    for text in text_inputs:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            clean = sentence.strip()
            if 20 <= len(clean) <= 300:
                facts.append(clean)
            if len(facts) >= 35:
                return facts
    return facts


def _tool_result_facts(messages: list[dict[str, Any]]) -> list[str]:
    facts: list[str] = []
    for message in messages:
        if message.get("role") == "tool":
            facts.append(_tool_fact(message))
        elif message.get("role") == "multi_tool":
            for submessage in message.get("tool_messages", []):
                if isinstance(submessage, dict):
                    facts.append(_tool_fact(submessage))
    return [fact for fact in facts if fact]


def _tool_fact(message: dict[str, Any]) -> str:
    name = str(message.get("name", "tool") or "tool")
    call_id = str(message.get("id", "") or "")
    content = _message_content(message)
    if not content:
        return ""
    prefix = f"tool_result:{name}"
    if call_id:
        prefix += f":{call_id}"
    return f"{prefix} => {_trim(content, 600)}"


def _extract_policy_clauses(texts: list[str]) -> list[str]:
    found: list[str] = []
    patterns = (
        r"\b[A-Z]{2,}(?:-[A-Z0-9]+){1,6}-\d{1,4}\b",
        r"\b[A-Z]{2,}_[A-Z0-9_]{2,}_\d{1,4}\b",
    )
    for text in texts:
        for pattern in patterns:
            for match in re.findall(pattern, text):
                if match not in found:
                    found.append(match)
    if found:
        return found[:40]
    return _fallback_policy_refs(texts)


def _fallback_policy_refs(texts: list[str]) -> list[str]:
    refs: list[str] = []
    for index, text in enumerate(texts[:8], start=1):
        lowered = text.lower()
        if any(term in lowered for term in ("policy", "rule", "sop", "must", "may")):
            refs.append(f"context_policy_{index}")
    return refs[:12]


def _extract_resource_refs(text: str) -> list[str]:
    found = []
    for pattern in (
        r"\b[A-Z]{2,}_[A-Z0-9_]{3,}\b",
        r"\b[A-Z]{2,}_\d{4,}_[A-Z0-9]+\b",
        r"\b[A-Z]{2,}\d{4,}\b",
        r"\b(?:order|ticket|account|case|customer|request)[-_ ]?[A-Z0-9]{3,}\b",
    ):
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            value = str(match)
            if value not in found:
                found.append(value)
    return found[:25]


def _sensitive_fields(texts: list[str]) -> list[str]:
    fields: list[str] = []
    joined = "\n".join(texts).lower()
    for term in SENSITIVE_TERMS:
        if term in joined:
            fields.append(term)
    return fields


def _risk_flags(tools: list[ToolSchema], context_texts: list[str]) -> list[str]:
    flags: list[str] = []
    tool_text = " ".join(tool.name for tool in tools).lower()
    for term in HIGH_IMPACT_TOOL_TERMS:
        if term in tool_text:
            flags.append(f"high_impact_tool:{term}")
    joined = "\n".join(context_texts).lower()
    for term in (
        "pressure",
        "urgency",
        "conflicting",
        "ambiguity",
        "private",
        "internal",
        "unauthorized",
        "exception",
        "approval",
    ):
        if term in joined:
            flags.append(f"context:{term}")
    return flags


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text", "") or item.get("content", "")))
            else:
                chunks.append(str(item))
        return " ".join(chunk for chunk in chunks if chunk)
    if content is None:
        return ""
    return str(content)


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n...[truncated]"
