import argparse
from uuid import uuid4

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from a2a_bridge.executor import Executor
from adapters.pibench import extract_request_from_json_message
from llm.nebius import DEFAULT_NEBIUS_MODEL
from runtime.core import PolicyCaseRuntime
from runtime.models import POLICY_BOOTSTRAP_EXTENSION


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="policy-case-runtime",
        name="Policy Case Runtime",
        description=(
            "Policy-compliance agent for Pi-Bench tasks with runtime action "
            "validation, canonical decisions, and audit-ready traces."
        ),
        tags=["policy", "safety", "pi-bench", "agentbeats"],
        examples=[
            "Evaluate a policy request, use the provided benchmark tools, and record ALLOW, ALLOW-CONDITIONAL, DENY, or ESCALATE."
        ],
    )

    agent_card = AgentCard(
        name="CoreLink Policy Case Runtime",
        description=(
            "A benchmark-neutral policy/safety participant agent optimized first "
            "for Pi-Bench A2A evaluation."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    runtime = PolicyCaseRuntime()
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(runtime=runtime),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = server.build()

    async def agent_json(_request):
        """Pi-Bench bootstrap capability card.

        Pi-Bench checks this non-standard A2A card path for the bootstrap
        extension before falling back to stateless requests.
        """
        return JSONResponse(
            {
                "name": "corelink-policy-case-runtime",
                "description": agent_card.description,
                "url": agent_card.url,
                "extensions": [POLICY_BOOTSTRAP_EXTENSION],
                "capabilities": {"message": True},
                "model": DEFAULT_NEBIUS_MODEL,
            }
        )

    async def health(_request):
        return JSONResponse({"status": "ok", "agent": agent_card.name})

    async def pibench_message_send(request: Request):
        """Accept Pi-Bench's lean JSON-RPC request and basic A2A JSON-RPC."""
        try:
            body = await request.json()
        except Exception:
            return _jsonrpc_error(None, -32700, "Invalid JSON")

        request_id = body.get("id")
        method = body.get("method")
        if method not in {"message/send", "message/stream"}:
            return _jsonrpc_error(request_id, -32601, "Unknown method")

        message = body.get("params", {}).get("message", {})
        if not isinstance(message, dict):
            return _jsonrpc_error(request_id, -32602, "Missing message")

        try:
            inbound = extract_request_from_json_message(message)
            response = await runtime.handle(inbound)
        except Exception as exc:
            return _jsonrpc_error(request_id, -32000, str(exc))

        payload = _jsonrpc_success(request_id, message, response.data)
        if method == "message/stream":
            async def events():
                import json

                yield f"data: {json.dumps(payload)}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        return JSONResponse(payload)

    def _jsonrpc_success(request_id, inbound_message: dict, data: dict):
        # Pi-Bench bootstrap parser expects result.status.message.parts. Standard
        # A2A SDK clients expect result to be a Message with messageId/role/parts.
        standard_a2a = bool(
            inbound_message.get("messageId") or inbound_message.get("message_id")
        )
        part = {"kind": "data", "data": data}
        if standard_a2a:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "kind": "message",
                    "messageId": str(uuid4()),
                    "contextId": inbound_message.get("contextId")
                    or inbound_message.get("context_id"),
                    "role": "agent",
                    "parts": [part],
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "status": {
                    "message": {
                        "role": "agent",
                        "parts": [part],
                    }
                }
            },
        }

    def _jsonrpc_error(request_id, code: int, message: str):
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

    app.routes.insert(0, Route("/", pibench_message_send, methods=["POST"]))
    app.routes.insert(0, Route("/.well-known/agent.json", agent_json, methods=["GET"]))
    app.add_route("/health", health, methods=["GET"])
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
