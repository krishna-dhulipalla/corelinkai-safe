import pytest

from a2a.types import DataPart, Message, Part, Role

from adapters.pibench import extract_request
from runtime.core import (
    PolicyCaseRuntime,
    build_policy_case,
    emit_response_data,
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
