from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_NEBIUS_PRIMARY_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
DEFAULT_NEBIUS_MEDIUM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_NEBIUS_MODEL = DEFAULT_NEBIUS_PRIMARY_MODEL
DEFAULT_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"


class ModelClientError(RuntimeError):
    pass


@dataclass
class NebiusConfig:
    api_key: str
    model: str = DEFAULT_NEBIUS_MODEL
    base_url: str = DEFAULT_NEBIUS_BASE_URL
    timeout_seconds: int = 90
    temperature: float = 0.1

    @classmethod
    def from_env(cls, *, role: str = "primary") -> "NebiusConfig":
        load_env_file()
        role = role.strip().lower()
        if role == "medium":
            model = (
                os.environ.get("NEBIUS_MEDIUM_MODEL", "").strip()
                or DEFAULT_NEBIUS_MEDIUM_MODEL
            )
        else:
            model = (
                os.environ.get("NEBIUS_PRIMARY_MODEL", "").strip()
                or os.environ.get("NEBIUS_MODEL", "").strip()
                or DEFAULT_NEBIUS_PRIMARY_MODEL
            )
        return cls(
            api_key=os.environ.get("NEBIUS_API_KEY", "").strip(),
            model=model,
            base_url=os.environ.get("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL).strip()
            or DEFAULT_NEBIUS_BASE_URL,
            timeout_seconds=int(os.environ.get("NEBIUS_TIMEOUT_SECONDS", "90")),
            temperature=float(os.environ.get("NEBIUS_TEMPERATURE", "0.1")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


class NebiusChatClient:
    """Small OpenAI-compatible client for Nebius Token Factory chat completions."""

    def __init__(self, config: NebiusConfig | None = None):
        self.config = config or NebiusConfig.from_env()

    @property
    def configured(self) -> bool:
        return self.config.configured

    async def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.config.configured:
            raise ModelClientError("NEBIUS_API_KEY is not configured")
        return await asyncio.to_thread(self._complete_sync, messages)

    def _complete_sync(self, messages: list[dict[str, str]]) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelClientError(f"Nebius request failed: {exc.code} {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"Nebius request failed: {exc}") from exc

        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelClientError(f"Unexpected Nebius response shape: {data}") from exc


@dataclass
class NebiusModelRouter:
    """Role-based model access for graph nodes.

    `auxiliary_enabled` is false when tests inject one fake client through the
    old `model_client` seam, so auxiliary nodes do not consume the fake solver
    response before `candidate_action`.
    """

    primary: Any
    medium: Any
    auxiliary_enabled: bool = True

    @classmethod
    def from_env(cls) -> "NebiusModelRouter":
        return cls(
            primary=NebiusChatClient(NebiusConfig.from_env(role="primary")),
            medium=NebiusChatClient(NebiusConfig.from_env(role="medium")),
            auxiliary_enabled=True,
        )

    @classmethod
    def from_single_client(cls, client: Any) -> "NebiusModelRouter":
        return cls(primary=client, medium=client, auxiliary_enabled=False)

    def client_for(self, role: str) -> Any:
        if role.strip().lower() == "medium":
            return self.medium
        return self.primary

    def role_configured(self, role: str) -> bool:
        if role.strip().lower() == "medium" and not self.auxiliary_enabled:
            return False
        client = self.client_for(role)
        return bool(getattr(client, "configured", False))


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Load .env values without overriding variables already in the process."""
    env_path = Path(path) if path else _find_env_file()
    if env_path is None or not env_path.exists():
        return []

    loaded: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = _strip_env_value(value.strip())
        if name and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def _find_env_file() -> Path | None:
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / ".env"
        if candidate.exists():
            return candidate
    return None


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response that may include fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Expected model output to be a JSON object")
    return parsed
