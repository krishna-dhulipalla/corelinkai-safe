from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm.nebius import (  # noqa: E402
    DEFAULT_NEBIUS_MEDIUM_MODEL,
    DEFAULT_NEBIUS_PRIMARY_MODEL,
    ModelClientError,
    NebiusChatClient,
    NebiusConfig,
    load_env_file,
)


def main() -> int:
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Check Nebius Token Factory model access.")
    parser.add_argument("--list", action="store_true", help="Print accessible model IDs")
    parser.add_argument("--primary-model", default="", help="Primary model to probe")
    parser.add_argument("--medium-model", default="", help="Medium model to probe")
    args = parser.parse_args()

    primary_model = args.primary_model or NebiusConfig.from_env(role="primary").model
    medium_model = args.medium_model or NebiusConfig.from_env(role="medium").model

    if args.list:
        for model_id in list_models():
            print(model_id)

    print(f"primary={primary_model}")
    print(f"medium={medium_model}")
    ok = asyncio.run(_probe_models(primary_model, medium_model))
    return 0 if ok else 2


def list_models() -> list[str]:
    config = NebiusConfig.from_env(role="primary")
    if not config.configured:
        raise SystemExit("NEBIUS_API_KEY is not configured")
    url = config.base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Model listing failed: {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Model listing failed: {exc}") from exc

    models = data.get("data", [])
    return sorted(str(item.get("id", "")) for item in models if item.get("id"))


async def _probe_models(primary_model: str, medium_model: str) -> bool:
    primary = _client_for_model(primary_model, role="primary")
    medium = _client_for_model(medium_model, role="medium")
    primary_ok = await _probe(primary, "primary")
    medium_ok = await _probe(medium, "medium")
    return primary_ok and medium_ok


def _client_for_model(model: str, *, role: str) -> NebiusChatClient:
    config = NebiusConfig.from_env(role=role)
    config.model = model or (
        DEFAULT_NEBIUS_MEDIUM_MODEL if role == "medium" else DEFAULT_NEBIUS_PRIMARY_MODEL
    )
    return NebiusChatClient(config)


async def _probe(client: NebiusChatClient, role: str) -> bool:
    messages = [
        {"role": "system", "content": "Return only compact JSON."},
        {"role": "user", "content": f'Return {{"ok":true,"role":"{role}"}} exactly.'},
    ]
    try:
        raw = await client.complete(messages)
    except ModelClientError as exc:
        print(f"{role}_probe=ERROR {exc}")
        return False
    print(f"{role}_probe={raw.strip() or 'EMPTY_RESPONSE'}")
    return bool(raw.strip())


if __name__ == "__main__":
    raise SystemExit(main())
