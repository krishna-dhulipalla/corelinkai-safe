import pytest

from a2a.types import DataPart, Message, Part, Role

from adapters.pibench import extract_request
from runtime.core import (
    PolicyGraphSettings,
    PolicyCaseRuntime,
    build_policy_case,
    emit_response_data,
    fallback_proposal,
    proposal_from_model,
    validate_proposal,
)
from runtime.models import InboundRequest, PolicySession


class UnconfiguredModel:
    configured = False


class FakeModel:
    configured = True

    def __init__(self, content: str):
        self.content = content

    async def complete(self, messages):
        return self.content


class SequenceModel:
    configured = True

    def __init__(self, contents: list[str]):
        self.contents = list(contents)

    async def complete(self, messages):
        if len(self.contents) > 1:
            return self.contents.pop(0)
        return self.contents[0]


def _record_decision_tool():
    return {
        "type": "function",
        "function": {
            "name": "record_decision",
            "description": "Record the canonical policy decision.",
            "parameters": {
                "type": "object",
                "required": ["decision"],
                "properties": {
                    "decision": {"type": "string"},
                    "policy_sections_cited": {"type": "array"},
                },
            },
        },
    }


def _read_order_tool():
    return {
        "type": "function",
        "function": {
            "name": "read_order",
            "description": "Read order state before taking a policy decision.",
            "parameters": {
                "type": "object",
                "required": ["order_id"],
                "properties": {"order_id": {"type": "string"}},
            },
        },
    }


def _refund_tool():
    return {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Issue a refund to the customer.",
            "parameters": {
                "type": "object",
                "required": ["order_id"],
                "properties": {"order_id": {"type": "string"}},
            },
        },
    }


def _pibench_record_decision_tool():
    return {
        "name": "record_decision",
        "description": "Record the canonical policy decision.",
        "parameters": {
            "order_id": {"type": "string", "required": True},
            "decision": {"type": "string", "required": True},
            "policy_sections_cited": {"type": "array", "required": True},
        },
    }


def test_extracts_pibench_bootstrap_datapart():
    message = Message(
        kind="message",
        role=Role.user,
        parts=[
            Part(
                root=DataPart(
                    kind="data",
                    data={
                        "bootstrap": True,
                        "benchmark_context": [{"kind": "policy", "content": "Rule"}],
                        "tools": [_record_decision_tool()],
                        "run_id": "run-1",
                        "domain": "retail",
                    },
                )
            )
        ],
        message_id="m1",
    )

    request = extract_request(message)

    assert request.is_bootstrap is True
    assert request.domain == "retail"
    assert request.run_id == "run-1"
    assert request.tools[0]["function"]["name"] == "record_decision"


@pytest.mark.asyncio
async def test_runtime_bootstrap_stores_context_and_returns_context_id():
    runtime = PolicyCaseRuntime(model_client=UnconfiguredModel())

    response = await runtime.handle(
        InboundRequest(
            is_bootstrap=True,
            benchmark_context=[{"kind": "policy", "content": "BM-RET-GEN-01"}],
            tools=[_record_decision_tool()],
            domain="retail",
        )
    )

    assert response.data["bootstrapped"] is True
    context_id = response.data["context_id"]
    assert context_id in runtime.sessions
    assert runtime.sessions[context_id].benchmark_context[0]["kind"] == "policy"


@pytest.mark.asyncio
async def test_runtime_emits_record_decision_tool_call_from_model_json():
    runtime = PolicyCaseRuntime(
        model_client=FakeModel(
            """
            {
              "kind": "record_decision",
              "decision": "DENY",
              "arguments": {"decision": "DENY"},
              "policy_refs": ["BM-RET-GEN-01"],
              "evidence_refs": ["order outside policy"],
              "rationale": "Policy blocks the request."
            }
            """
        )
    )
    boot = await runtime.handle(
        InboundRequest(
            is_bootstrap=True,
            benchmark_context=[
                {
                    "kind": "policy",
                    "content": "BM-RET-GEN-01: Deny requests that do not meet requirements.",
                }
            ],
            tools=[_record_decision_tool()],
        )
    )

    response = await runtime.handle(
        InboundRequest(
            is_bootstrap=False,
            context_id=boot.data["context_id"],
            messages=[{"role": "user", "content": "Please process this request."}],
        )
    )

    tool_call = response.data["tool_calls"][0]
    assert tool_call["function"]["name"] == "record_decision"
    assert '"decision": "DENY"' in tool_call["function"]["arguments"]
    assert "policy_sections_cited" in tool_call["function"]["arguments"]


def test_gate_blocks_unavailable_tool_without_domain_specific_rules():
    session = PolicySession(
        context_id="ctx",
        benchmark_context=[{"kind": "policy", "content": "BM-RET-GEN-01"}],
        tools=[_record_decision_tool()],
        messages=[{"role": "user", "content": "Issue a refund."}],
    )
    case = build_policy_case(session)
    proposal = proposal_from_model(
        {
            "kind": "tool_call",
            "tool_name": "process_refund",
            "arguments": {"order_id": "ORD_1"},
        },
        case,
    )

    decision = validate_proposal(case, proposal)

    assert decision.decision == "block"
    assert decision.reason_code == "unavailable_tool"


def test_response_data_uses_pibench_tool_call_shape():
    session = PolicySession(
        context_id="ctx",
        benchmark_context=[{"kind": "policy", "content": "BM-RET-GEN-01"}],
        tools=[_record_decision_tool()],
    )
    case = build_policy_case(session)
    proposal = proposal_from_model(
        {
            "kind": "record_decision",
            "decision": "ALLOW",
            "arguments": {"decision": "ALLOW"},
        },
        case,
    )

    data = emit_response_data(case, proposal)

    assert set(data) == {"tool_calls"}
    assert data["tool_calls"][0]["type"] == "function"
    assert data["tool_calls"][0]["function"]["name"] == "record_decision"


@pytest.mark.asyncio
async def test_graph_emits_normal_tool_call_before_final_decision():
    runtime = PolicyCaseRuntime(
        model_client=FakeModel(
            """
            {
              "kind": "tool_call",
              "tool_name": "read_order",
              "arguments": {"order_id": "ORD_123"},
              "policy_refs": ["BM-RET-GEN-01"],
              "rationale": "Need order state before deciding."
            }
            """
        )
    )
    boot = await runtime.handle(
        InboundRequest(
            is_bootstrap=True,
            benchmark_context=[
                {"kind": "policy", "content": "BM-RET-GEN-01: Verify order status before refunds."}
            ],
            tools=[_read_order_tool(), _record_decision_tool()],
        )
    )

    response = await runtime.handle(
        InboundRequest(
            is_bootstrap=False,
            context_id=boot.data["context_id"],
            messages=[{"role": "user", "content": "Can I refund order ORD_123?"}],
        )
    )

    tool_call = response.data["tool_calls"][0]
    assert tool_call["function"]["name"] == "read_order"
    assert '"order_id": "ORD_123"' in tool_call["function"]["arguments"]


def test_gate_blocks_invalid_canonical_decision():
    session = PolicySession(
        context_id="ctx",
        benchmark_context=[{"kind": "policy", "content": "BM-RET-GEN-01"}],
        tools=[_record_decision_tool()],
        messages=[{"role": "user", "content": "Please decide."}],
    )
    case = build_policy_case(session)
    proposal = proposal_from_model(
        {
            "kind": "record_decision",
            "arguments": {"decision": "MAYBE"},
            "policy_refs": ["BM-RET-GEN-01"],
        },
        case,
    )

    decision = validate_proposal(case, proposal)

    assert decision.decision == "block"
    assert decision.reason_code == "invalid_canonical_decision"


def test_fallback_record_decision_supports_pibench_flat_required_schema():
    session = PolicySession(
        context_id="ctx",
        benchmark_context=[
            {
                "kind": "policy",
                "content": "BM-RET-GEN-01: Escalate when model evidence is unavailable.",
            }
        ],
        tools=[_pibench_record_decision_tool()],
        messages=[{"role": "user", "content": "Please review order ORD_20260216_4821."}],
    )
    case = build_policy_case(session)
    proposal = fallback_proposal(case)

    decision = validate_proposal(case, proposal)

    assert proposal.arguments["order_id"] == "ORD_20260216_4821"
    assert proposal.arguments["decision"] == "ESCALATE"
    assert proposal.arguments["policy_sections_cited"] == ["BM-RET-GEN-01"]
    assert decision.allowed


@pytest.mark.asyncio
async def test_graph_revisits_evidence_plan_after_unsafe_mutating_action():
    runtime = PolicyCaseRuntime(
        model_client=SequenceModel(
            [
                """
                {
                  "kind": "tool_call",
                  "tool_name": "process_refund",
                  "arguments": {"order_id": "ORD_123"},
                  "policy_refs": ["BM-RET-GEN-01"],
                  "rationale": "Refund the order."
                }
                """,
                """
                {
                  "kind": "record_decision",
                  "decision": "ESCALATE",
                  "arguments": {"decision": "ESCALATE"},
                  "policy_refs": ["BM-RET-GEN-01"],
                  "rationale": "Escalate until order evidence is verified."
                }
                """,
            ]
        ),
        settings=PolicyGraphSettings(recursion_limit=30),
    )
    boot = await runtime.handle(
        InboundRequest(
            is_bootstrap=True,
            benchmark_context=[
                {"kind": "policy", "content": "BM-RET-GEN-01: Refunds require verified order evidence."}
            ],
            tools=[_refund_tool(), _record_decision_tool()],
        )
    )

    response = await runtime.handle(
        InboundRequest(
            is_bootstrap=False,
            context_id=boot.data["context_id"],
            messages=[{"role": "user", "content": "Refund ORD_123 now."}],
        )
    )

    tool_call = response.data["tool_calls"][0]
    assert tool_call["function"]["name"] == "record_decision"
    assert '"decision": "ESCALATE"' in tool_call["function"]["arguments"]
