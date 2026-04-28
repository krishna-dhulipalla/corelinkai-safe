from __future__ import annotations

import json
from typing import Any

from a2a.types import DataPart, Message, Part, TextPart

from runtime.models import InboundRequest


def extract_request(message: Message) -> InboundRequest:
    """Extract the Pi-Bench DataPart contract, with text fallback for normal A2A tests."""
    data_payload: dict[str, Any] | None = None
    text_chunks: list[str] = []

    for part in message.parts or []:
        root = getattr(part, "root", part)
        kind = getattr(root, "kind", None)
        if isinstance(root, DataPart) or kind == "data":
            data = getattr(root, "data", None)
            if isinstance(data, dict):
                data_payload = data
                break
        if isinstance(root, TextPart) or kind == "text":
            text = getattr(root, "text", "")
            if text:
                text_chunks.append(str(text))

    if data_payload is not None:
        return _request_from_data(data_payload)

    text = "\n".join(text_chunks).strip()
    messages = [{"role": "user", "content": text}] if text else []
    return InboundRequest(is_bootstrap=False, messages=messages, text=text)


def extract_request_from_json_message(message: dict[str, Any]) -> InboundRequest:
    """Extract a Pi-Bench request from raw JSON-RPC message params.

    Pi-Bench's adapter sends a lean message object that may omit standard A2A
    fields such as messageId. This parser intentionally accepts that benchmark
    shape before the strict SDK validator sees it.
    """
    text_chunks: list[str] = []
    for part in message.get("parts", []):
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "data" and isinstance(part.get("data"), dict):
            return _request_from_data(part["data"])
        if part.get("kind", "text") == "text" and part.get("text"):
            text_chunks.append(str(part["text"]))

    text = "\n".join(text_chunks).strip()
    messages = [{"role": "user", "content": text}] if text else []
    return InboundRequest(is_bootstrap=False, messages=messages, text=text)


def data_part(data: dict[str, Any]) -> Part:
    return Part(root=DataPart(kind="data", data=data))


def _request_from_data(data: dict[str, Any]) -> InboundRequest:
    return InboundRequest(
        is_bootstrap=bool(data.get("bootstrap")),
        context_id=_optional_str(data.get("context_id")),
        benchmark_context=_list_of_dicts(data.get("benchmark_context")),
        tools=_list_of_dicts(data.get("tools")),
        messages=_list_of_dicts(data.get("messages")),
        domain=_optional_str(data.get("domain")) or "",
        run_id=_optional_str(data.get("run_id")) or "",
        seed=_optional_int(data.get("seed")),
        text=json.dumps(data, sort_keys=True),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
